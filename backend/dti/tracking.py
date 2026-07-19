"""
dti/tracking.py — Deterministic fiber tractography via LocalTracking
══════════════════════════════════════════════════════════════════════

ALGORITHMIC OVERVIEW
──────────────────────

Deterministic tractography treats white-matter mapping as an ODE integration
problem.  A fibre is represented as a curve r(t) through 3-D space that
satisfies:

    dr/dt = d(r(t))

where d(r) is the local principal diffusion direction at position r
(the eigenvector e₁ corresponding to the largest eigenvalue λ₁).

The tracker uses simple Euler integration:

    r(t + Δt) = r(t) + Δt · d(r(t))

with step size Δt = step_size (mm).  At each step, d is obtained by
trilinearly interpolating the pre-computed principal eigenvector field.

HOW LOCALTRACKING WORKS
─────────────────────────
dipy's LocalTracking uses the following loop:

  1. Place a seed point inside the white-matter mask.
  2. Propagate forward and backward along the local eigenvector.
  3. At each step, query the StoppingCriterion:
       FA < fa_stop  →  stop (entered grey matter or CSF)
       outside mask  →  stop
  4. Collect the visited positions as a polyline (list of 3-D points).
  5. Repeat for all seeds; collect all polylines as the tractogram.

DIRECTION GETTER — FROM_PMF APPROACH
──────────────────────────────────────
Instead of calling peaks_from_model (which re-fits the tensor), we re-use
the TensorFit object stored in TensorFitResult._raw_fit and call

    tenfit.odf(sphere)   →   (x, y, z, n_vertices) array

to evaluate the orientation distribution function (ODF) at each sphere
vertex for each voxel.  For the tensor model, the ODF is:

    ODF(v; D) ∝ exp(−b · vᵀ D v)

evaluated at a representative high-b value (typically 1 s/mm² for the
ODF shape, independent of the actual b-values used for fitting).

After evaluating the ODF, any negative values (numerical artefact of the
tensor model when diffusivities are small) are clipped to zero.

DeterministicMaximumDirectionGetter.from_pmf then wraps the ODF values,
so that at each step it returns the sphere vertex direction closest to the
current propagation heading, enforcing the max_angle constraint.

This is the officially recommended single-tensor deterministic tractography
approach in dipy and avoids a second tensor model fitting pass.
"""

from __future__ import annotations

import numpy as np


def run_tractography(
    data:   np.ndarray,
    affine: np.ndarray,
    gtab:   object,          # dipy GradientTable — kept for potential future use
    mask:   np.ndarray,
    tensor: 'TensorFitResult',
    config: 'TractographyConfig',
) -> object:  # returns dipy Streamlines
    """
    Run deterministic DTI tractography and return raw (unfiltered) streamlines.

    Pipeline stages (in order):
      1. Load the pre-computed sphere for ODF evaluation.
      2. Evaluate the tensor ODF on the sphere for every voxel — O(n_voxels).
      3. Build a DeterministicMaximumDirectionGetter wrapping the ODF PMF.
      4. Build a ThresholdStoppingCriterion from the FA map.
      5. Generate seed points from white-matter voxels with FA > fa_seed.
      6. Run LocalTracking (Euler integration, both directions from each seed).
      7. Materialise the lazy generator into a Streamlines object.

    Parameters
    ----------
    data   : (x, y, z, N) float32 — brain-masked DWI data
    affine : (4, 4) float64 — NIfTI affine (voxel index → mm world coordinates)
    gtab   : dipy GradientTable (carried forward; not used in current pipeline)
    mask   : (x, y, z) bool — brain mask (True = brain voxel)
    tensor : TensorFitResult from tensor_fit.fit_tensor_model()
             must contain a valid _raw_fit (dipy TensorFit) object
    config : TractographyConfig

    Returns
    -------
    Streamlines — lazy-materialised set of raw fibre polylines.
    Coordinates are in world (mm) space defined by `affine`.

    Raises
    ------
    ValueError : if no seed points are found inside the white matter mask
    """
    from dipy.data import get_sphere                   # type: ignore[import]
    from dipy.direction import (                       # type: ignore[import]
        DeterministicMaximumDirectionGetter,
    )
    from dipy.tracking.stopping_criterion import (     # type: ignore[import]
        ThresholdStoppingCriterion,
    )
    from dipy.tracking.utils import seeds_from_mask    # type: ignore[import]
    from dipy.tracking.local_tracking import (         # type: ignore[import]
        LocalTracking,
    )
    from dipy.tracking.streamline import Streamlines   # type: ignore[import]

    fa       = tensor.fa
    raw_fit  = tensor._raw_fit   # dipy TensorFit (avoids a second model.fit())

    # ── Step 1: discrete sphere ────────────────────────────────────────────
    # The sphere vertices define the angular resolution of ODF evaluation.
    # 'symmetric362' provides 362 evenly distributed unit vectors, giving
    # ~7° angular resolution — sufficient for single-tensor DTI where the
    # principal eigenvector has no angular ambiguity.
    sphere = get_sphere(config.sphere_name)

    # ── Step 2: evaluate tensor ODF on sphere vertices ─────────────────────
    # tenfit.odf(sphere) computes, for each voxel, the normalised diffusion
    # ODF evaluated at each of the sphere's n_vertices directions.
    # Output shape: (x, y, z, n_vertices) float64.
    #
    # The tensor ODF  f(v) = (vᵀ D v)^(-3/2)  is proportional to the
    # probability of displacement along direction v.  It has a maximum along
    # the principal eigenvector e₁ and falls off with the ratio λ₁/λ₂,λ₃.
    #
    # Negative values can appear in noisy voxels where the WLS solution
    # produces slightly negative eigenvalues; clipping them to 0 prevents
    # the direction getter from inverting the direction.
    odf = raw_fit.odf(sphere)                  # (x, y, z, n_vertices)
    odf = np.clip(odf, 0, None)               # remove numerical negatives

    # ── Step 3: direction getter ───────────────────────────────────────────
    # DeterministicMaximumDirectionGetter.from_pmf interprets the ODF values
    # as a probability mass function (PMF) over the sphere vertices.
    # At each Euler step it picks the sphere vertex closest to the current
    # propagation direction that is also within max_angle degrees of that
    # direction, enforcing anatomical plausibility.
    #
    # pmf_threshold=0.1 ignores sphere vertices whose ODF value is below 10%
    # of the maximum — this suppresses noise-driven direction changes in
    # low-FA voxels just above the stopping threshold.
    direction_getter = DeterministicMaximumDirectionGetter.from_pmf(
        pmf=odf,
        max_angle=config.max_angle,
        sphere=sphere,
        pmf_threshold=0.1,
    )

    # ── Step 4: stopping criterion ─────────────────────────────────────────
    # ThresholdStoppingCriterion(metric_map, threshold) stops the tracker
    # when the interpolated FA at the current position drops below `threshold`.
    #
    # Bilinear interpolation of FA: the tracker does NOT snap to voxel centres;
    # it evaluates FA at sub-voxel positions by trilinear interpolation of the
    # fa array.  This produces smoother stopping behaviour near the WM/GM
    # boundary than hard per-voxel thresholding would.
    stopping_criterion = ThresholdStoppingCriterion(fa, config.fa_stop)

    # ── Step 5: seed points ────────────────────────────────────────────────
    # Seeds are placed at the centre of every voxel where FA > fa_seed AND
    # the voxel is inside the brain mask.
    #
    # seeds_from_mask places `density` seeds per voxel.  With density=1 the
    # seeds form a regular grid over the white-matter skeleton.  Higher density
    # increases tractogram density but multiplies runtime proportionally.
    #
    # Coordinates returned by seeds_from_mask are in world (mm) space,
    # transformed by `affine` from voxel indices.
    seed_mask = (fa > config.fa_seed) & mask.astype(bool)
    n_wm_voxels = int(np.sum(seed_mask))

    if n_wm_voxels == 0:
        raise ValueError(
            f"No seed points found.  FA > {config.fa_seed} within the brain "
            "mask returned zero voxels.  Possible causes:\n"
            "  • FA threshold too high — try lowering fa_seed to 0.25\n"
            "  • Brain mask too aggressive — check median_radius / numpass\n"
            "  • Data not properly preprocessed (eddy / motion correction)\n"
            "  • B-value too low to generate adequate FA contrast"
        )

    seeds = seeds_from_mask(seed_mask, affine=affine, density=config.seeds_per_voxel)

    # ── Step 6: local tracking ─────────────────────────────────────────────
    # LocalTracking propagates each seed in both directions (bi-directional
    # Euler integration) until the stopping criterion fires.
    #
    # max_cross=1 limits to one fibre direction per voxel — appropriate for
    # the single-tensor model which cannot represent crossing fibres.
    # For crossing-fibre regions, a CSD (Constrained Spherical Deconvolution)
    # model with max_cross=2–3 would be required.
    #
    # return_all=False discards streamlines that never entered valid tissue
    # (e.g. seeds placed in voxels where the stopping criterion fires
    # immediately before any integration step).
    streamlines_gen = LocalTracking(
        direction_getter=direction_getter,
        stopping_criterion=stopping_criterion,
        seeds=seeds,
        affine=affine,
        step_size=config.step_size,
        max_cross=config.max_cross,
        return_all=False,
    )

    # ── Step 7: materialise ────────────────────────────────────────────────
    # LocalTracking returns a lazy generator.  Calling Streamlines() forces
    # full evaluation — all seeds are integrated to completion before this
    # call returns.  This is where most of the computation happens.
    return Streamlines(streamlines_gen)
