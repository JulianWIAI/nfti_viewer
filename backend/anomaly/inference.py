"""
anomaly/inference.py — ONNX session management and binary mask extraction.
══════════════════════════════════════════════════════════════════════════════

Separating inference from pre/post-processing provides three benefits:
  1. The InferenceSession can be cached at module level and reused across
     requests — ONNX warm-up (~0.5–2 s) is amortised over the server lifetime.
  2. The model call can be unit-tested independently of NIfTI I/O.
  3. Swapping the ONNX model only requires updating AnomalyConfig.model_path.

SESSION LIFECYCLE
──────────────────
get_session(cfg) returns a cached InferenceSession keyed by
(model_path, providers).  The cache is module-level and thread-safe for reads
(dict lookup).  Writes happen only once per (model_path, providers) combination.
Multiple concurrent FastAPI requests share the same session; InferenceSession.run()
is documented as thread-safe for CPUExecutionProvider.

OUTPUT SHAPE CONTRACT
──────────────────────
The module handles two common output formats from anomaly / segmentation models:

  SIGMOID (single-class binary detection):
    Raw shape:  [1, X, Y, Z, 1]  or  [1, X, Y, Z]
    Semantics:  output[..., 0] = P(anomaly voxel) ∈ [0, 1]
    Binarise:   mask = (P >= cfg.threshold)

  SOFTMAX (multi-class segmentation, one class = anomaly):
    Raw shape:  [1, X, Y, Z, K]  (K ≥ 2 classes)
    Semantics:  output[..., k] = P(class k) for each voxel
    Binarise:   mask = (argmax(output, axis=-1) == cfg.anomaly_class_idx)

The `sigmoid_output` flag in AnomalyConfig selects the interpretation.
"""
from __future__ import annotations

import numpy as np

# ── Optional onnxruntime with graceful absence handling ───────────────────────

try:
    import onnxruntime as ort    # type: ignore
    _HAS_ONNX = True
except ImportError:
    ort = None
    _HAS_ONNX = False

from .config import AnomalyConfig, DEFAULT_CONFIG


# ── Module-level session cache ────────────────────────────────────────────────
# Key: (str(model_path), tuple(providers))
# Value: ort.InferenceSession
_SESSION_CACHE: dict[tuple[str, tuple[str, ...]], object] = {}


# ── Dependency check ──────────────────────────────────────────────────────────

def check_onnx() -> None:
    """Raise RuntimeError if onnxruntime is not installed."""
    if not _HAS_ONNX:
        raise RuntimeError(
            "onnxruntime is not installed.  Run: pip install onnxruntime"
        )


# ── Session factory (with caching) ───────────────────────────────────────────

def get_session(cfg: AnomalyConfig = DEFAULT_CONFIG):  # → ort.InferenceSession
    """
    Return a (possibly cached) ONNX InferenceSession for cfg.model_path.

    CACHE KEY
    ──────────
    (str(model_path), tuple(providers)) — changing either field invalidates
    the cache entry and forces re-initialisation on the next call.

    PROVIDER FALLBACK
    ──────────────────
    onnxruntime silently skips unavailable providers; it will fall back to
    CPUExecutionProvider even if only CUDAExecutionProvider is requested.
    The list in cfg.providers expresses a preference, not a requirement.

    Raises
    ------
    RuntimeError  if onnxruntime is not installed.
    RuntimeError  if the model file does not exist at cfg.model_path.
    """
    check_onnx()

    if not cfg.model_path.exists():
        raise RuntimeError(
            f"AnomalySeg ONNX model not found at '{cfg.model_path}'.  "
            "Place AnomalySeg.onnx in backend/models/ or run "
            "python backend/download_models.py to fetch it."
        )

    cache_key: tuple[str, tuple[str, ...]] = (str(cfg.model_path), cfg.providers)

    if cache_key not in _SESSION_CACHE:
        # First call: initialise and cache.  Subsequent calls return immediately.
        _SESSION_CACHE[cache_key] = ort.InferenceSession(   # type: ignore[union-attr]
            str(cfg.model_path),
            providers=list(cfg.providers),
        )

    return _SESSION_CACHE[cache_key]


# ── ONNX inference ────────────────────────────────────────────────────────────

def run_inference(
    tensor: np.ndarray,
    cfg:    AnomalyConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Run the AnomalySeg ONNX model on the preprocessed input tensor.

    Parameters
    ----------
    tensor : float32 C-contiguous ndarray of shape [1, X, Y, Z, 1].
             Built by preprocess.build_tensor().  Must be contiguous —
             onnxruntime raises if the array is not writable and contiguous.

    Returns
    -------
    probs  : float32 ndarray of shape (X, Y, Z, K).
             The batch dimension is stripped; K = number of output channels.
             For sigmoid models K=1; for softmax models K≥2.

    Notes
    ─────
    Input/output node names are discovered at runtime via the session metadata
    API (sess.get_inputs()[0].name, sess.get_outputs()[0].name) so the router
    does not need to hard-code model-specific names.
    """
    sess     = get_session(cfg)
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # sess.run() returns a list; we always consume the first (and only) output.
    raw: np.ndarray = sess.run([out_name], {in_name: tensor})[0]  # [1, X, Y, Z, K]

    # Strip the batch dimension → (X, Y, Z, K).
    # Use [0] indexing rather than squeeze() to preserve the channel axis when K=1.
    probs: np.ndarray = raw[0]
    return probs.astype(np.float32)


# ── Binary mask extraction ────────────────────────────────────────────────────

def extract_binary_mask(
    probs: np.ndarray,
    cfg:   AnomalyConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Convert raw model probabilities to a strict binary anomaly mask.

    SIGMOID MODE  (cfg.sigmoid_output = True)
    ──────────────────────────────────────────
    probs[..., cfg.anomaly_class_idx] gives the per-voxel probability of being
    an anomaly.  Voxels at or above cfg.threshold are classified as anomalous.

    If the model output has no channel axis (shape (X, Y, Z) rather than
    (X, Y, Z, 1)), the indexing step is skipped automatically.

    SOFTMAX MODE  (cfg.sigmoid_output = False)
    ───────────────────────────────────────────
    np.argmax across the last axis yields the most probable class per voxel.
    The mask is 1 where argmax equals cfg.anomaly_class_idx.

    Returns
    -------
    mask : uint8 ndarray of shape (X, Y, Z), values 0 or 1.
           Ready to pass to postprocess.uncrop_unpad().
    """
    if cfg.sigmoid_output:
        # ── Sigmoid: threshold the anomaly-class probability channel ──────────
        if probs.ndim == 3:
            # Model output has no channel axis — treat the whole map as P(anomaly)
            prob_anomaly = probs
        else:
            # Extract the anomaly class channel (usually index 0 for binary models)
            prob_anomaly = probs[..., cfg.anomaly_class_idx]

        # Hard threshold: 1 = anomaly, 0 = healthy tissue
        mask = (prob_anomaly >= cfg.threshold).astype(np.uint8)

    else:
        # ── Softmax: argmax gives the winning class index per voxel ───────────
        class_map = np.argmax(probs, axis=-1).astype(np.int32)
        mask = (class_map == cfg.anomaly_class_idx).astype(np.uint8)

    return mask
