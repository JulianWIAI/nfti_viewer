"""
nifti_layout.py — NIfTI axis-order & serialisation utilities
═════════════════════════════════════════════════════════════

WHY THIS MODULE EXISTS
──────────────────────
vtk.js vtkImageData expects scalar data in Fortran order: the first voxel
dimension (X / i-axis) varies fastest in the flat byte buffer.  Raw NIfTI
files store data the same way (dim[1] varies fastest), which is why
nifti-reader-js can hand the raw ArrayBuffer directly to vtk.js.

On the Python side, nibabel's get_fdata() returns a numpy array that has
the same logical voxel ordering (arr[i,j,k] = NIfTI voxel (i,j,k)), but
numpy's C-order default means the last axis varies fastest in memory.

The trap: np.ndarray.tobytes() traverses in the order controlled by its
own `order` kwarg, NOT the array's memory layout.  Calling

    np.asfortranarray(arr).tobytes()          # ← WRONG

creates an F-contiguous copy in RAM but then serialises it in C order
(the default), producing last-axis-fastest bytes — the opposite of what
vtk.js needs.  The correct call is

    arr.tobytes(order='F')                    # ← RIGHT

This file provides:
  • serialize_labels_f()  — guaranteed F-order serialisation
  • deserialize_labels()  — inverse (for unit tests / round-trips)
  • as_canonical_context()— context manager to run inference in RAS space
                           and re-orient labels back automatically
  • verify_layout()       — diagnostic print to confirm byte order at runtime
"""

from __future__ import annotations

import base64
import contextlib
from typing import Generator

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Core serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def serialize_labels_f(labels: np.ndarray) -> str:
    """
    Serialise a 3-D uint8 label volume to a base64 string in Fortran order.

    Fortran order means the first numpy axis (axis 0 = X/i) varies fastest
    in the output bytes.  This matches:
      • The NIfTI file format (dim[1] varies fastest on disk)
      • vtk.js vtkImageData point scalars (X varies fastest)

    Parameters
    ----------
    labels : np.ndarray
        Shape (nx, ny, nz), dtype uint8.  The labels must already be in
        the original voxel space (not canonical-reoriented).

    Returns
    -------
    str
        Base64-encoded bytes ready to send to the frontend.
    """
    assert labels.ndim == 3, f"Expected 3-D array, got shape {labels.shape}"
    arr = labels.astype(np.uint8, copy=False)
    # tobytes(order='F') = first-axis-fastest traversal, regardless of arr's
    # own memory layout (C or F contiguous).
    return base64.b64encode(arr.tobytes(order='F')).decode("ascii")


def deserialize_labels(b64_str: str, dims: tuple[int, int, int]) -> np.ndarray:
    """
    Round-trip inverse of serialize_labels_f.

    Decodes base64 → bytes → uint8 numpy array of shape (nx, ny, nz) using
    Fortran order (first axis varies fastest in the raw bytes).

    Parameters
    ----------
    b64_str : str
        Base64 string as returned by serialize_labels_f.
    dims : (nx, ny, nz)
        Spatial dimensions — must equal the dims sent in the API response.

    Returns
    -------
    np.ndarray
        uint8 array of shape (nx, ny, nz).
    """
    raw = base64.b64decode(b64_str)
    flat = np.frombuffer(raw, dtype=np.uint8)
    # np.reshape with order='F' interprets the flat bytes in Fortran order:
    # first index (axis 0 = X) varies fastest.
    return flat.reshape(dims, order='F')


# ─────────────────────────────────────────────────────────────────────────────
# 2.  as_closest_canonical context manager
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def as_canonical_context(
    nii_img,
) -> Generator[tuple[np.ndarray, np.ndarray, tuple[int, ...]], None, None]:
    """
    Context manager that yields a (data, affine, orig_shape) triple in RAS
    canonical orientation and automatically supplies the inverse-reorientation
    function for labels after the `with` block.

    USE CASE
    ────────
    SynthSeg was trained on volumes in LIA (Left-Inferior-Anterior) space
    re-coded from standard MNI/RAS.  Feeding a file that has a very different
    orientation can degrade segmentation accuracy.  Running in the canonical
    orientation first, then reverting to the original axis order before
    serialisation, gives the best of both worlds.

    Note: the FRONTEND does NOT apply any canonical reorientation.  The labels
    that reach the browser MUST be in the original voxel space so that they
    align with the raw NIfTI bytes that nifti-reader-js passes to vtk.js.
    We therefore always apply the inverse reorientation before serialising.

    Usage
    -----
    import nibabel as nib
    from nifti_layout import as_canonical_context, serialize_labels_f

    nii = nib.load("subject.nii.gz")
    with as_canonical_context(nii) as (data_ras, affine_ras, orig_shape, invert):
        # run SynthSeg on data_ras ...
        labels_ras = ...                    # shape matching data_ras
        labels_orig = invert(labels_ras)    # back to original voxel order
    b64 = serialize_labels_f(labels_orig)
    dims = list(orig_shape)

    Parameters
    ----------
    nii_img : nibabel image
        Any nibabel spatial image (Nifti1Image, Nifti2Image, …).

    Yields
    ------
    data_ras    : float32 ndarray in closest-to-RAS axis order
    affine_ras  : 4×4 float64 affine of the canonical image
    orig_shape  : shape of the original (non-canonical) volume
    invert      : callable (labels_ras → labels_orig) that reverses the axis
                  permutation + flip applied by as_closest_canonical
    """
    import nibabel as nib  # type: ignore

    orig_shape: tuple[int, ...] = tuple(nii_img.shape[:3])

    # nib.as_closest_canonical reorders axes to standard RAS and may flip signs
    canon_img = nib.as_closest_canonical(nii_img)
    data_ras   = np.asarray(canon_img.get_fdata(dtype=np.float32))
    if data_ras.ndim == 4:
        data_ras = data_ras[..., 0]
    affine_ras = canon_img.affine.astype(np.float64)

    # Compute the axis permutation + flip by comparing the two affines.
    # nib.io_orientation gives a table of (axis, flip) pairs:
    #   ornt[i] = (source_axis_for_output_i, flip_sign)
    orig_ornt  = nib.io_orientation(nii_img.affine)
    canon_ornt = nib.io_orientation(canon_img.affine)
    # The transform from canonical back to original orientation:
    revert_ornt = nib.ornt_transform(canon_ornt, orig_ornt)

    def invert(labels_ras: np.ndarray) -> np.ndarray:
        """Apply the inverse orientation transform to a label volume."""
        assert labels_ras.ndim == 3
        reverted = nib.apply_orientation(labels_ras.astype(np.uint8), revert_ornt)
        # Clamp to orig_shape in case floating-point rounding gives ±1 voxel
        if reverted.shape != orig_shape:
            canvas = np.zeros(orig_shape, dtype=np.uint8)
            clip   = tuple(
                slice(0, min(s, o)) for s, o in zip(reverted.shape, orig_shape)
            )
            canvas[clip] = reverted[clip]
            reverted = canvas
        return reverted

    yield data_ras, affine_ras, orig_shape, invert


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Runtime diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def verify_layout(labels: np.ndarray, dims: tuple[int, int, int]) -> None:
    """
    Print a concise diagnostic confirming byte-order and shape are correct.

    Call this in the /api/segment handler (temporarily) to verify the pipeline.
    Remove before production.

    Example output:
        [nifti_layout] shape   OK  (256, 300, 180) == (256, 300, 180)
        [nifti_layout] F-bytes OK  first 8 bytes differ from C-bytes (expected)
        [nifti_layout] label@[0,0,0] = 2  (should be a valid FreeSurfer ID)
    """
    nx, ny, nz = dims
    assert labels.shape == (nx, ny, nz), (
        f"Shape mismatch: labels {labels.shape} != dims {dims}"
    )
    print(f"[nifti_layout] shape   OK  {labels.shape} == {dims}")

    f_bytes = labels.tobytes(order='F')
    c_bytes = labels.tobytes(order='C')
    if f_bytes == c_bytes:
        print("[nifti_layout] WARNING: F-bytes == C-bytes "
              "(volume is trivially 1-D or all-zero; no way to confirm order)")
    else:
        print("[nifti_layout] F-bytes OK  (differ from C-bytes as expected for 3-D volume)")

    # Show a non-zero label if one exists within the first slice
    first_slice = labels[0, :, :]
    nz_idx = np.argwhere(first_slice != 0)
    if len(nz_idx):
        j, k = nz_idx[0]
        val = int(labels[0, j, k])
        print(f"[nifti_layout] non-zero label at [0,{j},{k}] = {val}")
    else:
        print("[nifti_layout] first X-slice is all background (label 0)")
