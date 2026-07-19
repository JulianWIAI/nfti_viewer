"""
dti/masking.py — Brain extraction for 4-D DWI data using median_otsu
══════════════════════════════════════════════════════════════════════

WHY BRAIN EXTRACTION?
──────────────────────
Raw diffusion-weighted images include the skull, scalp, background noise, and
air outside the head.  Fitting the tensor model to these non-brain voxels:

  1. Is computationally wasteful (brain is typically < 30% of the volume box).
  2. Produces numerically meaningless tensor estimates in the skull (which has
     very low water diffusivity with no coherent direction).
  3. Can cause the tracker to wander outside the brain, generating spurious
     streamlines that fly through air.

median_otsu
────────────
The algorithm applies a 3-D median filter to the mean b=0 image to remove
salt-and-pepper noise, then uses Otsu's threshold to binarise the result.
Otsu's method automatically finds the intensity that minimises intra-class
variance between background and foreground — no manual threshold needed.

  1. Mean over the requested b=0 volumes → single 3-D mean b0 image.
  2. Apply 3-D median filter (radius=median_radius) `numpass` times.
  3. Otsu threshold → binary mask.
  4. Optional dilation: expands the mask by a few voxels to include
     partial-volume voxels at the brain surface.

The mask is returned as a boolean (True = brain, False = background).
The masked data has all background voxels set to zero.
"""

from __future__ import annotations

import numpy as np


def extract_brain_mask(
    data:          np.ndarray,
    b0s_mask:      np.ndarray,
    median_radius: int = 4,
    numpass:       int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply skull-stripping to a 4-D DWI dataset and return a binary brain mask.

    Parameters
    ----------
    data          : (x, y, z, N) float32 array — full 4-D DWI volume
    b0s_mask      : (N,) bool array — True for b=0 volumes in the gradient table
                    (from gtab.b0s_mask); used to select which volumes to average
                    for the Otsu threshold step
    median_radius : radius of the 3-D median filter in voxels
    numpass       : number of median-filter passes (more = smoother mask edge)

    Returns
    -------
    masked_data : (x, y, z, N) float32 — copy of `data` with background zeroed
    mask        : (x, y, z) bool — True inside the brain

    Notes
    -----
    autocrop=False keeps the output the same shape as the input so the affine
    remains valid and all downstream arrays are aligned without extra transforms.

    dilate=2 expands the mask by 2 voxels in each direction to capture voxels
    at the brain surface that median_otsu may clip due to partial-volume effects.
    """
    from dipy.segment.mask import median_otsu  # type: ignore[import]

    # vol_idx: indices of the b=0 volumes (those with no diffusion weighting).
    # Otsu thresholding on the mean b0 image gives a cleaner mask than using
    # diffusion-weighted volumes, which have signal attenuation in all directions.
    vol_idx = list(np.where(b0s_mask)[0])

    if not vol_idx:
        # Fallback: use all volumes if no b0s are identified.
        # This is unusual but can happen if b0_threshold is set too low.
        vol_idx = None  # median_otsu uses the first volume by default

    masked_data, mask = median_otsu(
        data,
        vol_idx=vol_idx,
        median_radius=median_radius,
        numpass=numpass,
        autocrop=False,   # preserve the original array shape
        dilate=2,         # include partial-volume voxels at the brain boundary
    )

    return masked_data.astype(np.float32), mask.astype(bool)
