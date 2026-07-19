"""
anomaly/postprocess.py — Binary mask post-processing and Fortran-order serialisation.
═══════════════════════════════════════════════════════════════════════════════════════

This module is the mirror image of preprocess.py.  It takes the binary mask
produced by inference.extract_binary_mask() (in model space: 240×240×155) and
transforms it back to the original NIfTI voxel space before encoding for JSON.

PIPELINE CALL ORDER (orchestrated by the router):
    mask_1mm      = uncrop_unpad(mask_model, offsets, resampled_shape)
    mask_orig     = resample_to_original(mask_1mm, orig_shape, vox_mm, cfg)
    payload       = serialise_mask(mask_orig, affine)

FORTRAN-ORDER REQUIREMENT ── THE CRITICAL STEP
────────────────────────────────────────────────
vtk.js vtkImageData stores voxels with the X axis varying fastest in memory,
which is Fortran / column-major order — identical to NIfTI's on-disk byte
layout.  If the mask is serialised in C (row-major) order, the volume appears
incorrectly transposed in the 3D viewer.

The exact pattern required (mirrored from the SynthSeg pipeline in main.py):

    np.asfortranarray(binary_mask.astype(np.uint8)).tobytes(order='F')

Step 1 — `np.asfortranarray()`:
    Allocates a new array (if necessary) with Fortran-contiguous memory layout.
    For a C-contiguous input this involves a transpose copy; the result is a
    physically column-major buffer.

Step 2 — `.tobytes(order='F')`:
    Serialises the array in column-major order.  Combined with step 1, this
    guarantees X-fastest byte order regardless of the input array's strides.

    NOTE: calling only `.tobytes(order='F')` on a non-contiguous view (e.g. a
    sliced array) can produce incorrect output because numpy may serialise the
    underlying buffer in its native stride order rather than the requested 'F'
    order.  Always call `np.asfortranarray()` first to force contiguity.

NEAREST-NEIGHBOUR RESAMPLING
─────────────────────────────
`resample_to_original` uses scipy.ndimage.zoom with order=0 (nearest-neighbour).
This is mandatory for binary masks — trilinear (order=1) or cubic (order=3)
interpolation would create fractional voxel values at mask boundaries, destroying
the binary nature of the output.
"""
from __future__ import annotations

import base64

import numpy as np

try:
    import scipy.ndimage as _ndi   # type: ignore
    _HAS_SCIPY = True
except ImportError:
    _ndi = None
    _HAS_SCIPY = False


# ── Reverse crop / pad ────────────────────────────────────────────────────────

def uncrop_unpad(
    arr:          np.ndarray,
    offsets:      list[tuple[int, int]],
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """
    Reverse the preprocess.crop_pad() transform.

    For each axis the offset tuple encodes whether the axis was padded or cropped:

      Positive (before, after) → padding was added during preprocessing.
        Action: slice off `before` voxels from the start and `after` from the
                end, yielding the original pre-padding extent.

      Negative before          → volume was cropped during preprocessing.
        Action: re-embed the cropped array into a zero-filled canvas of
                size target_shape[axis], placing it at position abs(before).

    Parameters
    ----------
    arr          : binary mask in model space (e.g. (240, 240, 155))
    offsets      : list of (before, after) per axis from preprocess.crop_pad()
    target_shape : the shape to restore (typically `resampled_shape` from
                   preprocess.resample_to_isotropic())

    Returns
    -------
    result : ndarray with shape == target_shape, dtype preserved from arr
    """
    result = arr

    for axis, (before, _after) in enumerate(offsets):
        orig_s = target_shape[axis]

        if before >= 0:
            # ── Axis was zero-padded — slice off the padding ──────────────────
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(before, before + orig_s)
            result = result[tuple(slices)]

        else:
            # ── Axis was cropped — re-embed in a zero canvas ──────────────────
            # `before` is stored as a negative offset; abs(before) = start index.
            start = -before
            canvas_shape         = list(result.shape)
            canvas_shape[axis]   = orig_s
            canvas               = np.zeros(canvas_shape, dtype=result.dtype)
            dst                  = [slice(None)] * result.ndim
            dst[axis]            = slice(start, start + result.shape[axis])
            canvas[tuple(dst)]   = result
            result               = canvas

    return result


# ── Resample mask back to original voxel space ────────────────────────────────

def resample_to_original(
    mask:       np.ndarray,
    orig_shape: tuple[int, ...],
    vox_mm:     np.ndarray,
    target_vox: float = 1.0,
) -> np.ndarray:
    """
    Resample a binary mask from 1mm isotropic space back to `orig_shape`.

    INTERPOLATION ORDER
    ────────────────────
    order=0 (nearest-neighbour) is mandatory for binary masks.  Higher-order
    interpolation produces non-integer values at voxel boundaries, which breaks
    the binary guarantee that the frontend and vtk.js overlay pipeline rely on.

    SHAPE ROUNDING CORRECTION
    ──────────────────────────
    scipy.ndimage.zoom applies floating-point zoom factors and rounds the output
    shape independently per axis.  This can produce a shape that differs from
    orig_shape by ±1 voxel.  The correction block clips the zoomed result into
    a zero-filled canvas of exactly orig_shape — the same pattern as the
    SynthSeg inverse-resample in main.py.

    Parameters
    ----------
    mask       : uint8 binary ndarray in 1mm isotropic space
    orig_shape : the original (X, Y, Z) shape before any preprocessing
    vox_mm     : original voxel sizes in mm (used only for the skip-condition check)
    target_vox : the target voxel size used during preprocessing (default 1.0 mm)

    Returns
    -------
    uint8 ndarray with shape == orig_shape, values 0 or 1
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is not installed.  Run: pip install scipy")

    # ── Skip the zoom if no resampling was done during preprocessing ──────────
    if mask.shape == orig_shape:
        return mask.astype(np.uint8)

    # Inverse zoom factors: how much larger/smaller the original space is
    # relative to the current (1mm) space.
    inv_zoom = tuple(
        float(orig_s) / float(current_s)
        for orig_s, current_s in zip(orig_shape, mask.shape)
    )

    # order=0 preserves binary values; prefilter=False avoids a spline prefilter
    # that would be meaningless for a binary array and wastes CPU time.
    zoomed = _ndi.zoom(  # type: ignore[union-attr]
        mask.astype(np.float32),
        inv_zoom,
        order=0,
        prefilter=False,
    ).astype(np.uint8)

    # ── Correct ±1 voxel rounding errors from scipy ────────────────────────────
    if zoomed.shape != orig_shape:
        canvas = np.zeros(orig_shape, dtype=np.uint8)
        # Use element-wise min so we never index out of bounds on either array.
        clip = tuple(slice(0, min(s, o)) for s, o in zip(zoomed.shape, orig_shape))
        canvas[clip] = zoomed[clip]
        return canvas

    return zoomed


# ── Serialisation ─────────────────────────────────────────────────────────────

def serialise_mask(
    mask:   np.ndarray,
    affine: np.ndarray,
) -> dict:
    """
    Encode the final binary mask for JSON transport to the vtk.js frontend.

    ══ FORTRAN-ORDER SERIALISATION — DO NOT MODIFY THIS SECTION ══════════════

    vtk.js vtkImageData stores voxels with X varying fastest (Fortran /
    column-major order, matching the NIfTI on-disk byte layout).  Serialising
    in C (row-major) order would transpose the mask in the 3D viewer.

    TWO-STEP PROCEDURE (both steps are required):

      Step 1:  np.asfortranarray(binary_mask.astype(np.uint8))
               Forces physical column-major contiguity in memory.  For a
               C-contiguous input this allocates a new transposed copy;
               without this step, non-contiguous views can produce incorrect
               tobytes() output.

      Step 2:  .tobytes(order='F')
               Serialises the buffer in X-fastest (column-major) byte order.
               Combined with step 1, this is equivalent to the NIfTI Fortran
               convention used by nibabel's to_bytes() method.

    ══════════════════════════════════════════════════════════════════════════

    AFFINE SERIALISATION
    ─────────────────────
    The 4×4 RAS-mm affine is flattened row-major (C order) into a 16-element
    list of float64 values.  This matches the format expected by the vtk.js
    frontend's NIfTI parser and the SynthSeg SegmentResult schema.

    Parameters
    ----------
    mask   : uint8 ndarray of shape (X, Y, Z), values 0 or 1
             Output of resample_to_original().
    affine : float64 ndarray of shape (4, 4) — RAS-mm world transform
             Passed through unmodified from extract_volume().

    Returns
    -------
    dict with keys:
      mask        : str  — base64-encoded uint8 bytes in Fortran (X-fastest) order
      dims        : list[int]   — [X, Y, Z] original voxel dimensions
      affine      : list[float] — 16-element row-major flattened 4×4 RAS affine
      n_anomaly   : int         — number of anomaly voxels (1-valued voxels)
    """
    # ── Ensure correct dtype before any Fortran conversion ────────────────────
    binary_u8: np.ndarray = mask.astype(np.uint8)

    # ── Step 1: Force physical Fortran-contiguous memory layout ───────────────
    # np.asfortranarray() guarantees that the array is column-major contiguous.
    # For an already Fortran-contiguous array this is a no-op (no copy).
    fortran_mask: np.ndarray = np.asfortranarray(binary_u8)

    # ── Step 2: Serialise in Fortran (X-fastest) byte order ───────────────────
    # tobytes(order='F') on a Fortran-contiguous array emits bytes in X-fastest
    # order, matching vtk.js vtkImageData's memory layout exactly.
    mask_b64: str = base64.b64encode(
        fortran_mask.tobytes(order='F')
    ).decode("ascii")

    return {
        # Primary payload: base64 Fortran-order binary mask
        "mask":      mask_b64,

        # Spatial metadata required by the vtk.js frontend to reconstruct the volume
        "dims":      list(mask.shape),           # [X, Y, Z]
        "affine":    affine.flatten().tolist(),   # 16 float64 values, row-major

        # Diagnostic: voxel count of detected anomaly region
        "n_anomaly": int(np.sum(binary_u8 > 0)),
    }
