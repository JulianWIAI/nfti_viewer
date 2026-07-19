"""
longitudinal/delta.py — Subtraction map computation and Fortran-order serialisation.
══════════════════════════════════════════════════════════════════════════════════════

This module is the mirror image of registration.py.  It takes the registered
follow-up array (in baseline voxel space) and computes the voxel-wise
difference, then serialises the result for JSON transport to the vtk.js frontend.

SIGN CONVENTION FOR THE DELTA
───────────────────────────────
    delta  =  registered_followup  −  baseline

    Positive (+) : follow-up has MORE signal than baseline.
                   Interpretation: tissue growth, gadolinium-enhancing lesion
                   progression, oedema expansion, or CSF accumulation.

    Negative (−) : follow-up has LESS signal than baseline.
                   Interpretation: grey-matter atrophy, white-matter lesion
                   shrinkage, or tissue replacement by hypointense gliosis.

    Near zero    : no detectable longitudinal change (within noise floor).

WHY float32 (NOT uint8)?
─────────────────────────
The delta contains continuous positive and negative intensity values spanning
a large dynamic range (typically ±500 to ±2000 for raw 16-bit MRI intensities
after registration).  Unlike the binary anomaly mask or the uint8 FreeSurfer
label map, the delta cannot be quantised without destroying clinically relevant
information.  The frontend uses a symmetric diverging colormap (blue→white→red)
applied over [min_val, 0, max_val] to visualise both atrophy and growth.

FORTRAN-ORDER SERIALISATION — CRITICAL STEP
─────────────────────────────────────────────
vtk.js vtkImageData stores voxels with X varying fastest in memory (column-
major / Fortran order), identical to NIfTI on-disk layout.  Serialising in C
(row-major) order transposes the delta volume in the 3-D viewer.

The exact two-step pattern (mirrored from anomaly/postprocess.py):

    np.asfortranarray(delta.astype(np.float32)).tobytes(order='F')

Step 1 — np.asfortranarray():
    Forces physical Fortran-contiguous layout.  For a C-contiguous input this
    allocates a new transposed copy; without this step, non-contiguous views
    can silently produce incorrect tobytes() output.

Step 2 — .tobytes(order='F'):
    Serialises in X-fastest byte order.  Combined with Step 1, this guarantees
    column-major byte layout regardless of the input array's strides.

PUBLIC API
───────────
    compute_delta(baseline, registered)    → float32 delta ndarray
    serialise_delta(delta, affine)         → JSON-ready dict
"""
from __future__ import annotations

import base64

import numpy as np


# ── Subtraction ───────────────────────────────────────────────────────────────

def compute_delta(
    baseline:   np.ndarray,
    registered: np.ndarray,
) -> np.ndarray:
    """
    Compute voxel-wise: delta = registered_followup − baseline.

    Both arrays must be in the same voxel space (i.e., `registered` must
    already have been resampled via registration.run_registration so that
    registered.shape == baseline.shape).

    Returns
    -------
    delta : float32 ndarray of shape (X, Y, Z)
            Positive values → growth/expansion; negative → atrophy/loss.
    """
    if registered.shape != baseline.shape:
        raise ValueError(
            f"Shape mismatch: registered {registered.shape} ≠ "
            f"baseline {baseline.shape}.  "
            "run_registration() must be called before compute_delta()."
        )

    # Subtraction produces float32 when both inputs are float32.
    # Explicit cast ensures dtype contract regardless of input dtypes.
    return (registered - baseline).astype(np.float32)


# ── Serialisation ─────────────────────────────────────────────────────────────

def serialise_delta(
    delta:  np.ndarray,
    affine: np.ndarray,
) -> dict:
    """
    Encode the float32 delta volume for JSON transport to the vtk.js frontend.

    ══ FORTRAN-ORDER SERIALISATION — DO NOT MODIFY THIS SECTION ══════════

    vtk.js vtkImageData stores voxels with X varying fastest (Fortran /
    column-major order, matching the NIfTI on-disk byte layout).
    Serialising in C (row-major) order would produce a transposed delta
    volume in the 3-D viewer.

    TWO-STEP PROCEDURE (both steps required):

      Step 1:  np.asfortranarray(delta.astype(np.float32))
               Forces physical Fortran-contiguous memory layout.

      Step 2:  .tobytes(order='F')
               Serialises in X-fastest byte order.

    ══════════════════════════════════════════════════════════════════════

    SUMMARY STATISTICS
    ───────────────────
    min_val / max_val are returned so the frontend can normalise the
    colormap symmetrically around zero without reprocessing the raw bytes.

    n_positive / n_negative count the voxels showing growth vs. atrophy,
    providing a quick clinical summary without full voxel-based morphometry.

    Parameters
    ----------
    delta  : float32 ndarray of shape (X, Y, Z)
             Output of compute_delta().
    affine : float64 ndarray of shape (4, 4) — baseline RAS-mm world transform.
             Passed through to the frontend so it can align the delta volume
             with the original structural MRI in world space.

    Returns
    -------
    dict with keys:
      delta       : str         — base64-encoded float32 bytes in Fortran order
      dims        : list[int]   — [X, Y, Z] voxel dimensions
      affine      : list[float] — 16 float64 values, row-major flattened
      min_val     : float       — minimum delta value (for colormap lower bound)
      max_val     : float       — maximum delta value (for colormap upper bound)
      n_positive  : int         — voxels with delta > 0 (growth / fluid expansion)
      n_negative  : int         — voxels with delta < 0 (atrophy / tissue loss)
    """
    # ── Ensure correct float32 dtype ──────────────────────────────────────────
    delta_f32: np.ndarray = delta.astype(np.float32)

    # ── Step 1: Force Fortran-contiguous physical layout ──────────────────────
    # np.asfortranarray() returns the array with column-major memory strides.
    # For a C-contiguous input this allocates a new physically transposed copy.
    # For an already F-contiguous array this is a no-op (no allocation).
    fortran_delta: np.ndarray = np.asfortranarray(delta_f32)

    # ── Step 2: Serialise in X-fastest byte order ─────────────────────────────
    # tobytes(order='F') on a Fortran-contiguous array emits bytes in X-fastest
    # order, matching vtkImageData's internal memory layout exactly.
    delta_b64: str = base64.b64encode(
        fortran_delta.tobytes(order='F')
    ).decode('ascii')

    # ── Summary statistics ────────────────────────────────────────────────────
    # Computed on the float32 array (identical numerical values to delta_f32).
    min_val    = float(np.min(delta_f32))
    max_val    = float(np.max(delta_f32))
    n_positive = int(np.sum(delta_f32 > 0))
    n_negative = int(np.sum(delta_f32 < 0))

    return {
        # Primary payload: base64 Fortran-order float32 delta
        'delta':      delta_b64,

        # Spatial metadata required by vtk.js to reconstruct the volume
        'dims':       list(delta.shape),          # [X, Y, Z]
        'affine':     affine.flatten().tolist(),   # 16 float64 values, row-major

        # Colormap anchor points for the diverging blue→white→red scale
        'min_val':    min_val,
        'max_val':    max_val,

        # Clinical summary counts
        'n_positive': n_positive,
        'n_negative': n_negative,
    }
