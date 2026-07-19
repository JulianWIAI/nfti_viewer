"""
anomaly/preprocess.py — NIfTI → ONNX-ready float32 tensor for AnomalySeg inference.
═══════════════════════════════════════════════════════════════════════════════════════

This module mirrors the SynthSeg preprocessing pipeline from main.py, adapted
for anomaly detection models that operate in BraTS (240×240×155) input space.

PUBLIC API
───────────
    check_dependencies()         — raise if nibabel / scipy are absent
    load_nifti(bytes)            → nib.Nifti1Image
    extract_volume(nii)          → (data, affine, orig_shape, vox_mm)
    resample_to_isotropic(...)   → (resampled, resampled_shape)
    normalise(data, cfg)         → float32 ndarray
    crop_pad(arr, target_shape)  → (volume, offsets)
    build_tensor(vol)            → float32 ndarray [1, X, Y, Z, 1]

PIPELINE CALL ORDER (orchestrated by the router):
    1. check_dependencies()
    2. nii = load_nifti(raw_bytes)
    3. data, affine, orig_shape, vox_mm = extract_volume(nii)
    4. resampled, resampled_shape = resample_to_isotropic(data, vox_mm, cfg)
    5. normed = normalise(resampled, cfg)
    6. cropped, offsets = crop_pad(normed, cfg.target_shape)
    7. tensor = build_tensor(cropped)   → fed to inference.run_inference()

Z-SCORE NORMALISATION
──────────────────────
Anomaly and tumour segmentation models trained on BraTS data expect z-score
normalised input, not [0,1] min-max.  Crucially, the mean and standard
deviation are computed only over non-zero voxels (brain tissue), ignoring the
large air background — matching the pre-processing used during BraTS training.

MEMORY SAFETY
──────────────
Resampling to 1mm isotropic can dramatically increase volume size for
low-resolution scans (e.g. 3mm × 3mm × 3mm → 27× more voxels).
`resample_to_isotropic` estimates the output size and raises MemoryError
before calling scipy.ndimage.zoom if the result would exceed `cfg.max_resampled_voxels`.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

# ── Optional nibabel and scipy with graceful absence handling ─────────────────

try:
    import nibabel as nib          # type: ignore
    _HAS_NIBABEL = True
except ImportError:
    nib = None
    _HAS_NIBABEL = False

try:
    import scipy.ndimage as _ndi   # type: ignore
    _HAS_SCIPY = True
except ImportError:
    _ndi = None
    _HAS_SCIPY = False

from .config import AnomalyConfig, DEFAULT_CONFIG


# ── Dependency check ──────────────────────────────────────────────────────────

def check_dependencies() -> None:
    """
    Raise RuntimeError if nibabel or scipy are not installed.

    Called at the top of the router endpoint so the server returns a clear
    500 error before attempting any NIfTI I/O or array manipulation.
    """
    if not _HAS_NIBABEL:
        raise RuntimeError(
            "nibabel is not installed.  Run: pip install nibabel"
        )
    if not _HAS_SCIPY:
        raise RuntimeError(
            "scipy is not installed.  Run: pip install scipy"
        )


# ── NIfTI loading ─────────────────────────────────────────────────────────────

def load_nifti(content: bytes):  # → nib.Nifti1Image
    """
    Deserialise raw NIfTI bytes into a nibabel image object.

    STRATEGY
    ─────────
    Attempt the in-memory path first (nibabel >= 4.0 exposes from_bytes()).
    Fall back to a temporary file for older nibabel versions and for .nii.gz
    payloads, which require seekable file I/O for gzip decompression.

    The temp file is always cleaned up in the finally block even if nibabel
    raises an exception during parsing.

    Parameters
    ----------
    content : raw bytes from UploadFile.read() — may be .nii or .nii.gz

    Returns
    -------
    nib.Nifti1Image (or compatible subclass)
    """
    check_dependencies()

    # ── Fast path: in-memory deserialization ──────────────────────────────────
    try:
        return nib.Nifti1Image.from_bytes(content)   # type: ignore[union-attr]
    except (AttributeError, Exception):
        pass   # nibabel < 4.0 or .nii.gz — fall through to temp file

    # ── Fallback: materialise to a named temp file ────────────────────────────
    # nibabel uses the file extension to detect .gz compression, so we keep
    # the .nii suffix even for potentially compressed data.  nibabel's gzip
    # detection is header-based, not extension-based, so this is safe.
    tmp = tempfile.NamedTemporaryFile(suffix=".nii", delete=False)
    try:
        tmp.write(content)
        tmp.flush()
        tmp.close()
        return nib.load(tmp.name)   # type: ignore[union-attr]
    finally:
        # Always clean up the temp file regardless of success or failure.
        Path(tmp.name).unlink(missing_ok=True)


# ── Volume extraction ─────────────────────────────────────────────────────────

def extract_volume(
    nii,   # nib.Nifti1Image
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], np.ndarray]:
    """
    Extract a 3-D float32 data array, the 4×4 affine, the original shape,
    and the voxel dimensions (mm) from a nibabel image object.

    4-D inputs (fMRI, multi-echo) are handled by selecting the first volume
    on the last axis, matching the SynthSeg pipeline convention.

    Returns
    -------
    data        : float32 ndarray of shape (X, Y, Z)
    affine      : float64 ndarray of shape (4, 4) — RAS mm world transform
    orig_shape  : (X, Y, Z) — needed to resample the mask back at the end
    vox_mm      : float64 ndarray of shape (3,) — voxel sizes in mm

    Raises
    ------
    ValueError  if volume dimensionality is not 3 (or 4 reducible to 3)
    ValueError  if computed voxel sizes are physically implausible
    """
    # get_fdata(dtype=float32) avoids an extra copy vs default float64
    data: np.ndarray = np.asarray(nii.get_fdata(dtype=np.float32))
    affine: np.ndarray = nii.affine.astype(np.float64)

    # ── Dimensionality normalization ──────────────────────────────────────────
    if data.ndim == 4:
        # Take the first volume (b=0 for DWI, t=0 for fMRI, echo=0 for GRE)
        data = data[..., 0]
    elif data.ndim != 3:
        raise ValueError(
            f"Expected a 3-D NIfTI volume, got shape {list(data.shape)}.  "
            "Reduce to 3-D before uploading."
        )

    orig_shape: tuple[int, int, int] = data.shape  # type: ignore[assignment]

    # ── Voxel size extraction ─────────────────────────────────────────────────
    # Column norms of the 3×3 sub-matrix give voxel sizes along each axis.
    # This matches nibabel's pixdim convention for RAS affines.
    vox_mm: np.ndarray = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))

    if not np.all((vox_mm > 0) & (vox_mm < 50)):
        raise ValueError(
            f"Implausible voxel sizes: {vox_mm.tolist()} mm.  "
            "Check that the NIfTI header is valid."
        )

    return data, affine, orig_shape, vox_mm


# ── Isotropic resampling ──────────────────────────────────────────────────────

def resample_to_isotropic(
    data:    np.ndarray,
    vox_mm:  np.ndarray,
    cfg:     AnomalyConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """
    Zoom `data` so each voxel represents cfg.target_vox_mm mm (typically 1mm).

    SKIP CONDITION
    ───────────────
    If voxels are already within 5% of the target size, the zoom is skipped
    and the original array is returned as-is.  This avoids a costly identity
    resampling.

    MEMORY GUARD
    ─────────────
    The estimated output size is computed before calling scipy.ndimage.zoom.
    If the output would exceed cfg.max_resampled_voxels, MemoryError is raised
    before allocating anything — the server returns a clean 413 error rather
    than crashing the process.

    INTERPOLATION ORDER
    ────────────────────
    order=1 (trilinear) balances speed and quality for MRI intensities.
    The input is a floating-point intensity image so linear interpolation is
    appropriate.  Nearest-neighbour (order=0) is reserved for the label/mask
    inverse-resample in postprocess.py to preserve binary values.

    Returns
    -------
    resampled       : float32 ndarray in isotropic voxel space
    resampled_shape : (X, Y, Z) shape of the resampled volume
    """
    target = float(cfg.target_vox_mm)
    # Zoom factor per axis: current vox size / target vox size.
    # For a 3mm voxel with target 1mm: zoom = 3.0 → output is 3x larger.
    zoom_factors = tuple(float(v) / target for v in vox_mm)

    # ── Memory safety check ────────────────────────────────────────────────────
    estimated_nvox = int(np.prod([s * z for s, z in zip(data.shape, zoom_factors)]))
    if estimated_nvox > cfg.max_resampled_voxels:
        raise MemoryError(
            f"Resampled volume would be ~{estimated_nvox / 1e6:.1f}M voxels "
            f"(server limit: {cfg.max_resampled_voxels / 1e6:.0f}M).  "
            "Please downsample the NIfTI to ≤ 3mm voxels before uploading."
        )

    # ── Skip near-isotropic volumes ────────────────────────────────────────────
    if np.allclose(vox_mm, target, atol=target * 0.05):
        return data.astype(np.float32), data.shape  # type: ignore[return-value]

    # ── Trilinear zoom ────────────────────────────────────────────────────────
    resampled = _ndi.zoom(data, zoom_factors, order=1)   # type: ignore[union-attr]
    return resampled.astype(np.float32), resampled.shape  # type: ignore[return-value]


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalise(
    data: np.ndarray,
    cfg:  AnomalyConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Normalise a float32 volume according to cfg.normalisation.

    "zscore"
    ─────────
    Zero-mean / unit-variance normalisation computed only over non-zero
    (brain tissue) voxels.  Air background voxels are excluded from the
    statistics to avoid deflating the mean and inflating the std, which
    would produce poorly-scaled inputs to the model.

    This is the standard preprocessing for BraTS-trained anomaly models
    (Menze et al. 2015; Isensee et al. 2021).

    "minmax"
    ─────────
    Linear rescale to [0, 1] using global min/max with a tiny epsilon
    to prevent division by zero for flat (empty) volumes.  Matches the
    SynthSeg pipeline convention in main.py.

    Returns a new float32 array; the input is not modified.
    """
    if cfg.normalisation == "zscore":
        # Brain mask: all voxels with intensity > 0.
        # For T2/FLAIR, background air is reliably 0 after nibabel loading.
        mask = data > 0
        if mask.any():
            mu  = float(data[mask].mean())
            sig = float(data[mask].std()) + 1e-9   # epsilon prevents /0
        else:
            # Completely blank volume — skip normalisation (model handles 0s)
            mu, sig = 0.0, 1.0
        return ((data - mu) / sig).astype(np.float32)

    # min-max fallback (matches SynthSeg main.py)
    lo = float(data.min())
    hi = float(data.max())
    return ((data - lo) / (hi - lo + 1e-9)).astype(np.float32)


# ── Crop / pad to model input size ────────────────────────────────────────────

def crop_pad(
    arr:          np.ndarray,
    target_shape: tuple[int, int, int],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Centre-crop OR zero-pad each axis of `arr` to exactly target_shape[axis].

    DECISION PER AXIS
    ──────────────────
    current_size < target_size → zero-pad symmetrically (centre-align content)
    current_size > target_size → centre-crop (discard outer voxels equally)
    current_size = target_size → pass-through (no copy)

    The returned `offsets` encode the exact transform so postprocess.uncrop_unpad
    can reverse it without storing the intermediate array.

    OFFSET SIGN CONVENTION
    ───────────────────────
    Positive before/after → padding was added; uncrop_unpad slices them off.
    Negative before       → volume was cropped; uncrop_unpad re-embeds it.
    (The sign convention mirrors _crop_pad_256 in main.py for consistency.)

    Parameters
    ----------
    arr          : float32 ndarray — resampled, normalised volume
    target_shape : (X, Y, Z) — the model's required spatial dimensions

    Returns
    -------
    result  : ndarray with shape == target_shape, dtype preserved from arr
    offsets : list of (before, after) per axis — consumed by uncrop_unpad
    """
    result  = arr
    offsets: list[tuple[int, int]] = []

    for axis, (current_s, target_s) in enumerate(zip(arr.shape, target_shape)):
        diff = target_s - current_s

        if diff > 0:
            # ── Zero-pad: content is smaller than the model expects ───────────
            before = diff // 2
            after  = diff - before
            pad_w  = [(0, 0)] * result.ndim
            pad_w[axis] = (before, after)
            result = np.pad(result, pad_w, mode="constant", constant_values=0.0)
            offsets.append((before, after))

        elif diff < 0:
            # ── Centre-crop: content is larger than the model expects ─────────
            start  = (-diff) // 2
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(start, start + target_s)
            result = result[tuple(slices)]
            # Negative `before` signals to uncrop_unpad that this axis was cropped.
            offsets.append((-start, -(current_s - (start + target_s))))

        else:
            # ── Exact match — no copy needed ──────────────────────────────────
            offsets.append((0, 0))

    return result, offsets


# ── Tensor construction ───────────────────────────────────────────────────────

def build_tensor(vol: np.ndarray) -> np.ndarray:
    """
    Wrap a (X, Y, Z) volume into a channels-last 5-D ONNX input tensor.

    Output shape: [1, X, Y, Z, 1] = [batch, X, Y, Z, channels].

    This channels-last layout is used by SynthSeg, most BraTS-trained models,
    and the expected default for TensorFlow-derived ONNX exports.  Pytorch-
    derived models may expect channels-first [1, 1, X, Y, Z] — in that case,
    override this function or transpose in the router before passing to the
    inference module.

    The result is contiguous float32, which is what onnxruntime requires for
    CPUExecutionProvider (avoids a hidden copy inside the ONNX runtime).
    """
    # np.newaxis at position 0 (batch) and -1 (channel)
    tensor = vol[np.newaxis, ..., np.newaxis]
    # np.ascontiguousarray ensures C-contiguous layout for onnxruntime
    return np.ascontiguousarray(tensor, dtype=np.float32)
