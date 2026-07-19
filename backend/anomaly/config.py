"""
anomaly/config.py — Immutable configuration for the AnomalySeg inference pipeline.
═══════════════════════════════════════════════════════════════════════════════════

All tunable parameters live here so preprocess.py, inference.py, and
postprocess.py share a single source of truth without circular imports.

Changing the model (e.g. different BraTS variant, custom input shape) only
requires updating DEFAULT_CONFIG or constructing a new AnomalyConfig — the
pipeline code itself stays untouched.

BRATS CONVENTION
─────────────────
The standard BraTS challenge uses volumes of shape (240, 240, 155) at 1mm
isotropic resolution.  Channels-last inference tensors are shaped:

    [batch, X, Y, Z, channels]  →  [1, 240, 240, 155, 1]

INFERENCE SESSION CACHING
──────────────────────────
The (model_path, providers) tuple is used as a cache key in inference.py so
that changing these fields forces a re-initialisation of the ONNX session.
Sessions are expensive to initialise (~0.5–2 s) so reuse across requests is
important for server throughput.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AnomalyConfig:
    """
    Frozen configuration dataclass for the AnomalySeg pipeline.

    All fields have production-ready defaults.  Override individual fields by
    constructing a new instance:

        cfg = AnomalyConfig(threshold=0.3, normalisation="minmax")
    """

    # ── Model location ────────────────────────────────────────────────────────

    # Absolute path to the AnomalySeg ONNX model file.
    # Place AnomalySeg.onnx in backend/models/ or set ANOMALY_MODEL_PATH env var.
    model_path: Path = Path(__file__).parent.parent / "models" / "AnomalySeg.onnx"

    # ── Spatial preprocessing ─────────────────────────────────────────────────

    # Target voxel spacing (mm) to resample inputs to before crop/pad.
    # BraTS-style models expect 1mm isotropic resolution.
    target_vox_mm: float = 1.0

    # Model input shape (X, Y, Z) — the crop/pad target dimensions.
    # Standard BraTS: 240 × 240 × 155.  Override for non-BraTS models.
    target_shape: tuple[int, int, int] = (240, 240, 155)

    # ── Normalisation ─────────────────────────────────────────────────────────

    # Strategy applied to the resampled, cropped volume before inference.
    #   "zscore"  → zero-mean / unit-variance over non-zero (brain) voxels.
    #               Preferred for anomaly / tumour detection models trained on
    #               BraTS data, where brain-tissue intensity varies across sites.
    #   "minmax"  → linear rescale to [0, 1] using global min/max.
    #               Matches the SynthSeg pipeline convention.
    normalisation: str = "zscore"

    # ── Output interpretation ─────────────────────────────────────────────────

    # Probability threshold for converting model output to a binary mask.
    # Applied to P(anomaly) after sigmoid (or to the anomaly-class column after
    # softmax).  Lower threshold → more sensitive (higher recall) detection.
    threshold: float = 0.5

    # Output channel index that corresponds to the anomaly class.
    #   Sigmoid single-class head: output[..., 0] = P(anomaly)  → idx 0
    #   Softmax two-class head:    output[..., 1] = P(anomaly)  → idx 1
    anomaly_class_idx: int = 0

    # True  → treat model output as sigmoid probabilities; apply `threshold`.
    # False → treat model output as softmax logits; use argmax and compare
    #         the winning class index against `anomaly_class_idx`.
    sigmoid_output: bool = True

    # ── ONNX runtime ─────────────────────────────────────────────────────────

    # Execution providers in preference order.  onnxruntime tries each in turn
    # and silently falls back to the next if the provider is unavailable.
    # CUDAExecutionProvider is attempted first but ignored gracefully if CUDA
    # is not installed or the device is absent.
    providers: tuple[str, ...] = (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )

    # ── Memory safety ─────────────────────────────────────────────────────────

    # Maximum allowed voxel count after resampling to 1mm isotropic.
    # Guards against OOM crashes on pathologically large uploads.
    # 300^3 = 27 000 000 voxels ≈ 108 MB for float32.
    max_resampled_voxels: int = 300 ** 3


# Singleton used as the default argument throughout the pipeline.
# Import this instead of constructing AnomalyConfig() everywhere to avoid
# the (tiny) overhead of repeated dataclass instantiation.
DEFAULT_CONFIG = AnomalyConfig()
