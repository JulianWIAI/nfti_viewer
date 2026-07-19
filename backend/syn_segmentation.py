"""
syn_segmentation.py — Reusable SynthSeg inference for the SyN registration pipeline
═════════════════════════════════════════════════════════════════════════════════════

Extracts the SynthSeg ONNX inference logic from main.py into a standalone
module so the SyN registration router can:

  1. Run SynthSeg on BOTH the raw (un-warped) NIfTI volumes to get integer
     label maps (FreeSurfer IDs 0–60).
  2. Compute hippocampal volumes from the raw label arrays using the real
     voxel geometry (NOT the warped geometry) so reported cm³ figures reflect
     true anatomy.

DESIGN NOTES
─────────────
• Model is loaded lazily on first call and cached in a module-level singleton
  (_ort_session) so the ~200 MB ONNX graph is parsed only once per process.
• All operations are synchronous.  Callers must dispatch to asyncio.to_thread()
  before calling from an async FastAPI endpoint.
• Duplicate crop/pad helpers are kept here (instead of importing from main.py)
  to avoid a circular import between main.py and the registration router.
  If the project ever extracts them into a shared utility module, remove these.

PREREQUISITES
─────────────
  pip install scipy onnxruntime
  Place SynthSeg.onnx in backend/models/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ── Paths & constants ─────────────────────────────────────────────────────────

# Absolute path to the ONNX model, resolved relative to this source file so it
# works regardless of the working directory from which uvicorn is launched.
_MODEL_PATH = Path(__file__).parent / "models" / "SynthSeg.onnx"

# SynthSeg v2 requires exactly this voxel count per axis.
_SYNTHSEG_SIZE = 256

# Mapping from SynthSeg class index (0–32) → FreeSurfer label ID.
# Matches the _SYNTHSEG_LABEL_MAP in main.py exactly.
_LABEL_MAP: list[int] = [
     0,   #  0 → background
     2,   #  1 → left cerebral white matter
     3,   #  2 → left cerebral cortex
     4,   #  3 → left lateral ventricle
     5,   #  4 → left inferior lateral ventricle
     7,   #  5 → left cerebellum white matter
     8,   #  6 → left cerebellum cortex
    10,   #  7 → left thalamus
    11,   #  8 → left caudate
    12,   #  9 → left putamen
    13,   # 10 → left pallidum
    14,   # 11 → 3rd ventricle
    15,   # 12 → 4th ventricle
    16,   # 13 → brain stem
    17,   # 14 → left hippocampus  ← used for volumetrics
    18,   # 15 → left amygdala
    24,   # 16 → CSF
    26,   # 17 → left accumbens area
    28,   # 18 → left ventral DC
    41,   # 19 → right cerebral white matter
    42,   # 20 → right cerebral cortex
    43,   # 21 → right lateral ventricle
    44,   # 22 → right inferior lateral ventricle
    46,   # 23 → right cerebellum white matter
    47,   # 24 → right cerebellum cortex
    49,   # 25 → right thalamus
    50,   # 26 → right caudate
    51,   # 27 → right putamen
    52,   # 28 → right pallidum
    53,   # 29 → right hippocampus  ← used for volumetrics
    54,   # 30 → right amygdala
    58,   # 31 → right accumbens area
    60,   # 32 → right ventral DC
]
_LABEL_MAP_ARRAY = np.array(_LABEL_MAP, dtype=np.uint8)

# FreeSurfer label IDs for the hippocampus — these are what volumetrics reports.
LABEL_HIPPOCAMPUS_LH: int = 17  # left hippocampus
LABEL_HIPPOCAMPUS_RH: int = 53  # right hippocampus

# ── Lazy ONNX session singleton ───────────────────────────────────────────────

_ort_session = None  # type: ignore[var-annotated]


def _get_session():
    """Load and cache the SynthSeg ONNX InferenceSession (one-time cost)."""
    global _ort_session
    if _ort_session is None:
        import onnxruntime as ort
        _ort_session = ort.InferenceSession(
            str(_MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
    return _ort_session


# ── Dependency guard ──────────────────────────────────────────────────────────

def check_synthseg_available() -> None:
    """
    Raise RuntimeError if SynthSeg dependencies or the ONNX model are missing.

    Called once at the start of each registration request so the router can
    return a clear 500 rather than an opaque ImportError later in the pipeline.
    """
    missing_pkgs: list[str] = []
    try:
        import scipy  # noqa: F401
    except ImportError:
        missing_pkgs.append("scipy")
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        missing_pkgs.append("onnxruntime")
    if missing_pkgs:
        raise RuntimeError(
            f"Missing required packages: {', '.join(missing_pkgs)}. "
            f"Run: pip install {' '.join(missing_pkgs)}"
        )
    if not _MODEL_PATH.exists():
        raise RuntimeError(
            f"SynthSeg model not found at {_MODEL_PATH}. "
            "Run backend/download_models.py to fetch it, or place SynthSeg.onnx "
            "in backend/models/."
        )


# ── Internal crop / pad helpers ───────────────────────────────────────────────
# Duplicated from main.py to avoid circular imports.

def _crop_pad_256(arr: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Centre-crop or zero-pad each spatial axis to exactly 256 voxels."""
    N = _SYNTHSEG_SIZE
    offsets: list[tuple[int, int]] = []
    result = arr
    for axis, s in enumerate(arr.shape):
        diff = N - s
        if diff >= 0:
            before = diff // 2
            after  = diff - before
            pad_width = [(0, 0)] * result.ndim
            pad_width[axis] = (before, after)
            result = np.pad(result, pad_width, mode="constant", constant_values=0)
            offsets.append((before, after))
        else:
            start = (-diff) // 2
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(start, start + N)
            result = result[tuple(slices)]
            offsets.append((-start, -(s - (start + N))))
    return result, offsets


def _uncrop_unpad(
    arr: np.ndarray,
    offsets: list[tuple[int, int]],
    orig_shape: tuple,
) -> np.ndarray:
    """Reverse _crop_pad_256: strip padding / restore cropped voxels."""
    result = arr
    for axis, (before, _after) in enumerate(offsets):
        orig_s = orig_shape[axis]
        if before >= 0:
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(before, before + orig_s)
            result = result[tuple(slices)]
        else:
            start = -before
            out = np.zeros(
                [result.shape[i] if i != axis else orig_s for i in range(result.ndim)],
                dtype=result.dtype,
            )
            dst = [slice(None)] * result.ndim
            dst[axis] = slice(start, start + result.shape[axis])
            out[tuple(dst)] = result
            result = out
    return result


# ── Public inference function ─────────────────────────────────────────────────

def run_synthseg(data: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Run SynthSeg ONNX v2 inference on a 3-D float32 brain volume.

    Mirrors the /api/segment pipeline in main.py step-for-step:
      1. Derive voxel spacing from the affine diagonal.
      2. Resample to 1 mm isotropic (scipy.ndimage.zoom, order=1).
      3. Normalise to [0, 1].
      4. Centre-crop or zero-pad to 256³.
      5. ONNX inference  →  [1, 256, 256, 256, 33] logits.
      6. argmax(axis=-1)  →  33 class indices  →  FreeSurfer label IDs.
      7. Reverse crop/pad.
      8. Resample label map back to original voxel space (nearest-neighbour
         to preserve integer label values).

    Parameters
    ----------
    data   : float32 3-D ndarray in the original voxel space.
    affine : float64 (4, 4) RAS-mm affine for `data`.

    Returns
    -------
    uint8 ndarray with the same shape as `data`.
    Values are FreeSurfer label IDs (0 = background, 17 = LH hippocampus, etc.).
    """
    from scipy import ndimage as ndi  # lazy import — only needed at runtime

    orig_shape = data.shape

    # ── 1. Voxel spacing from affine column norms ─────────────────────────
    vox_mm = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    if not np.all((vox_mm > 0) & (vox_mm < 50)):
        raise ValueError(f"Implausible voxel sizes {vox_mm.tolist()} mm; is this a structural MRI?")

    zoom_factors = tuple(float(v) for v in vox_mm)

    # ── 2. Resample to 1 mm isotropic ────────────────────────────────────
    if not np.allclose(vox_mm, 1.0, atol=0.05):
        resampled: np.ndarray = ndi.zoom(data, zoom_factors, order=1)
    else:
        resampled = data.copy()
    resampled_shape = resampled.shape

    # ── 3. Normalise to [0, 1] ────────────────────────────────────────────
    lo, hi = float(resampled.min()), float(resampled.max())
    norm: np.ndarray = (resampled - lo) / (hi - lo + 1e-9)
    del resampled

    # ── 4. Crop / pad to 256³ ────────────────────────────────────────────
    vol256, offsets = _crop_pad_256(norm)
    del norm

    # ── 5. ONNX inference (channels-last: [1, 256, 256, 256, 1]) ─────────
    inp = vol256[np.newaxis, ..., np.newaxis].astype(np.float32)
    del vol256
    sess     = _get_session()
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    logits   = sess.run([out_name], {in_name: inp})[0]   # [1,256,256,256,33]
    del inp

    # ── 6. argmax → FreeSurfer label IDs ─────────────────────────────────
    class_idx = np.argmax(logits[0], axis=-1).astype(np.int32)   # [256,256,256]
    class_idx = np.clip(class_idx, 0, len(_LABEL_MAP_ARRAY) - 1)
    labels_256: np.ndarray = _LABEL_MAP_ARRAY[class_idx]          # uint8
    del logits, class_idx

    # ── 7. Reverse crop / pad ────────────────────────────────────────────
    labels_1mm = _uncrop_unpad(labels_256, offsets, resampled_shape)
    del labels_256

    # ── 8. Resample back to original space (nearest to preserve IDs) ─────
    if not np.allclose(vox_mm, 1.0, atol=0.05):
        inv_zoom = tuple(s / r for s, r in zip(orig_shape, labels_1mm.shape))
        labels_orig: np.ndarray = ndi.zoom(
            labels_1mm.astype(np.float32), inv_zoom, order=0,
        ).astype(np.uint8)
        # scipy.zoom can return shape ±1 per axis due to floating-point rounding;
        # clamp to the expected shape with a zero-padded canvas.
        if labels_orig.shape != orig_shape:
            canvas = np.zeros(orig_shape, dtype=np.uint8)
            clip = tuple(slice(0, min(s, o)) for s, o in zip(labels_orig.shape, orig_shape))
            canvas[clip] = labels_orig[clip]
            labels_orig = canvas
    else:
        labels_orig = labels_1mm.astype(np.uint8)
    del labels_1mm

    return labels_orig


# ── Public volumetrics function ───────────────────────────────────────────────

def compute_hippocampal_volumes(
    labels: np.ndarray,
    affine: np.ndarray,
) -> dict[str, float]:
    """
    Compute left- and right-hemisphere hippocampal volumes in cm³.

    IMPORTANT: Always pass the RAW (un-warped) label array and its associated
    affine.  Volumes computed from a warped array reflect the registration
    target's voxel geometry, not the subject's true anatomy.

    Parameters
    ----------
    labels : uint8 ndarray of FreeSurfer label IDs, same shape as the raw volume.
    affine : float64 (4, 4) RAS-mm affine for `labels` (provides voxel spacing).

    Returns
    -------
    {"lh": float_cm3, "rh": float_cm3}  — rounded to 3 decimal places.
    """
    # Voxel volume in mm³ = product of column L2 norms of the 3×3 rotation block.
    vox_mm     = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    vox_vol_mm3 = float(np.prod(vox_mm))
    mm3_to_cm3  = 1.0 / 1000.0

    lh_voxels = int(np.sum(labels == LABEL_HIPPOCAMPUS_LH))
    rh_voxels = int(np.sum(labels == LABEL_HIPPOCAMPUS_RH))

    return {
        "lh": round(lh_voxels * vox_vol_mm3 * mm3_to_cm3, 3),
        "rh": round(rh_voxels * vox_vol_mm3 * mm3_to_cm3, 3),
    }
