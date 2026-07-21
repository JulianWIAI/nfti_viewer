"""
longitudinal/preprocess.py — NIfTI loading for the longitudinal delta pipeline.
═════════════════════════════════════════════════════════════════════════════════

WHY WE DO NOT RESAMPLE TO ISOTROPIC HERE
──────────────────────────────────────────
Unlike anomaly/preprocess.py and the SynthSeg pipeline, longitudinal
co-registration must work in the original voxel space of each scan so that:

  1. The dipy affine registration metric (mutual information) is computed
     in physical world coordinates — the affines carry all spacing
     information, so both images are treated correctly regardless of their
     individual voxel sizes.

  2. After registration, the followup is resampled into the BASELINE'S
     voxel grid (same shape, spacing, and affine as baseline).  The subtraction
     map therefore inherits the baseline's spatial resolution exactly.

  3. Resampling to an intermediate isotropic space would introduce two
     interpolation steps (baseline → 1mm, followup → 1mm) accumulating
     partial-volume artifacts in the delta, whereas keeping original spaces
     means only the followup is interpolated once.

PUBLIC API
───────────
    check_dependencies()              → raises if nibabel is absent
    load_nifti(bytes)                 → nib.Nifti1Image
    extract_volume(nii, cfg)          → (data, affine, orig_shape)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

# ── Optional nibabel with graceful absence handling ───────────────────────────

try:
    import nibabel as nib          # type: ignore
    _HAS_NIBABEL = True
except ImportError:
    nib = None
    _HAS_NIBABEL = False

from .config import LongitudinalConfig, DEFAULT_CONFIG


# ── Dependency check ──────────────────────────────────────────────────────────

def check_dependencies() -> None:
    """Raise RuntimeError if nibabel is not installed."""
    if not _HAS_NIBABEL:
        raise RuntimeError(
            "nibabel is not installed.  Run: pip install nibabel"
        )


# ── NIfTI loading ─────────────────────────────────────────────────────────────

def load_nifti(content: bytes):  # → nib.Nifti1Image
    """
    Deserialise raw NIfTI bytes into a nibabel image object.

    STRATEGY (mirrors anomaly/preprocess.py)
    ──────────────────────────────────────────
    Attempt the in-memory path first (nibabel >= 4.0 exposes from_bytes()).
    Fall back to a named temporary file for older nibabel versions and for
    .nii.gz payloads, which require seekable file I/O for gzip decompression.
    The temp file is always cleaned up in the finally block.

    Parameters
    ----------
    content : raw bytes from UploadFile.read() — may be .nii or .nii.gz
    """
    check_dependencies()

    def _relaxed_load(loader, *args, **kwargs):
        """Retry with a relaxed quaternion threshold on w2 precision errors."""
        try:
            return loader(*args, **kwargs)
        except ValueError as exc:
            if "w2 should be positive" not in str(exc):
                raise
            old = nib.Nifti1Header.quaternion_threshold
            try:
                nib.Nifti1Header.quaternion_threshold = -1e-6
                return loader(*args, **kwargs)
            finally:
                nib.Nifti1Header.quaternion_threshold = old

    # ── Fast path: in-memory deserialisation ──────────────────────────────────
    try:
        return _relaxed_load(nib.Nifti1Image.from_bytes, content)   # type: ignore[union-attr]
    except (AttributeError, Exception):
        pass   # nibabel < 4.0 or .nii.gz — fall through to temp file

    # ── Fallback: materialise to a named temp file ────────────────────────────
    tmp = tempfile.NamedTemporaryFile(suffix=".nii", delete=False)
    try:
        tmp.write(content)
        tmp.flush()
        tmp.close()
        return _relaxed_load(nib.load, tmp.name)   # type: ignore[union-attr]
    finally:
        Path(tmp.name).unlink(missing_ok=True)


# ── Volume extraction ─────────────────────────────────────────────────────────

def extract_volume(
    nii,                              # nib.Nifti1Image
    cfg: LongitudinalConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """
    Extract a 3-D float32 array, the 4×4 affine, and the original shape from
    a nibabel image.

    IMPORTANT — NO RESAMPLING
    ──────────────────────────
    Both baseline and followup are returned in their original voxel space.
    The dipy registration works with the affines to compute a physically
    meaningful world-space transform, so no upfront resampling is needed.

    4-D inputs (fMRI, DWI) are handled by selecting the first volume on the
    last axis, matching the convention used across all other pipeline modules.

    Parameters
    ----------
    nii : nibabel NIfTI image
    cfg : LongitudinalConfig — used only for max_voxels memory guard

    Returns
    -------
    data        : float32 ndarray of shape (X, Y, Z)
    affine      : float64 ndarray of shape (4, 4) — RAS-mm world transform
    orig_shape  : (X, Y, Z) voxel dimensions

    Raises
    ------
    ValueError   if dimensionality is not 3 (or 4 reducible to 3)
    MemoryError  if the volume exceeds cfg.max_voxels
    """
    # get_fdata(dtype=float32) avoids an extra float64→float32 copy.
    data: np.ndarray = np.asarray(nii.get_fdata(dtype=np.float32))
    affine: np.ndarray = nii.affine.astype(np.float64)

    # ── Dimensionality normalisation ──────────────────────────────────────────
    if data.ndim == 4:
        # Take the first volume (b=0 for DWI, t=0 for fMRI, echo=0 for GRE).
        data = data[..., 0]
    elif data.ndim != 3:
        raise ValueError(
            f"Expected a 3-D NIfTI volume, got shape {list(data.shape)}.  "
            "Reduce to 3-D before uploading."
        )

    orig_shape: tuple[int, int, int] = data.shape  # type: ignore[assignment]

    # ── Memory guard ─────────────────────────────────────────────────────────
    # Check BEFORE any dipy allocation — two volumes + registered + delta
    # can peak at 4 × n_voxels × 4 bytes.
    nvox = int(np.prod(orig_shape))
    if nvox > cfg.max_voxels:
        raise MemoryError(
            f"Volume has {nvox / 1e6:.1f}M voxels "
            f"(server limit per volume: {cfg.max_voxels / 1e6:.0f}M).  "
            "Please downsample to ≤ 1mm isotropic before uploading."
        )

    return data, affine, orig_shape
