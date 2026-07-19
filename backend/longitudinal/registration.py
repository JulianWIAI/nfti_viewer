"""
longitudinal/registration.py — dipy affine co-registration pipeline.
══════════════════════════════════════════════════════════════════════

This module is the mathematical core of the longitudinal pipeline.  It maps
the follow-up volume into the exact voxel space of the baseline so that a
simple array subtraction yields a physically meaningful change map.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MUTUAL INFORMATION (MI) — THE SIMILARITY METRIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mutual information measures the statistical dependence between the joint
intensity distribution of two images A (baseline) and B (followup):

    MI(A, B)  =  H(A) + H(B) − H(A, B)

where:
    H(X)     = Shannon entropy of the marginal distribution of X
             = −Σ p(x) log p(x)
    H(A, B)  = Shannon entropy of the JOINT distribution
             = −Σ p(a, b) log p(a, b)

INTUITION FOR ALIGNMENT
─────────────────────────
When A and B are perfectly aligned, each intensity value in A predicts the
intensity in B at the same location (e.g., cortex is bright in both scans).
The joint histogram concentrates along a narrow diagonal ridge → H(A, B) is
small → MI is large.

When misaligned, brain tissue from scan A overlaps with skull or background
from scan B, spreading the joint histogram across many intensity combinations
→ H(A, B) is large → MI is small.

The optimiser therefore MAXIMISES MI (equivalently, minimises −MI) with
respect to the transformation parameters.

WHY MUTUAL INFORMATION (NOT MEAN SQUARED ERROR)?
──────────────────────────────────────────────────
MSE assumes the two images have the same absolute intensities.  This breaks
between longitudinal scans where MRI signal intensity is not standardised
across sessions (different coil loading, scanner drift, different protocols).
MI is modality-independent: it only requires that corresponding tissues
produce CORRELATED (not identical) intensities, making it robust to the
inter-session intensity drifts typical in clinical longitudinal MRI.

PARZEN-WINDOW DENSITY ESTIMATION
──────────────────────────────────
dipy's MutualInformationMetric does not use hard histogram bins.  Instead it
approximates the joint probability distribution p(a, b) using a B-spline
(Parzen window) kernel:

    p̂(a, b)  ≈  (1/N) Σ_i  K(a − a_i) K(b − b_i)

where K is a cubic B-spline and i ranges over sampled voxels.  This makes
the MI surface smooth and continuously differentiable, enabling the L-BFGS-B
gradient-based optimiser to descend to the optimum efficiently.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MULTI-SCALE PYRAMID STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The registration is performed at three resolution levels (coarse → fine):

    Level  factors  sigmas  level_iters    Purpose
    ─────  ───────  ──────  ───────────    ──────────────────────────────────
      0      4×      3.0      10 000       Rough global alignment; large basin
      1      2×      1.0       1 000       Refine at half resolution
      2      1×      0.0         100       Fine-tune at full resolution

Level 0 uses aggressive Gaussian smoothing (σ = 3 voxels) so that small-scale
anatomical noise does not trap the optimiser in a local minimum.  The solution
from each level initialises the next, warm-starting the next level's optimiser.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE TRANSFORMATION MATRIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After optimisation, the result object (an AffineMap) carries:

    final_map.affine  : 4×4 matrix T in WORLD (RAS-mm) space

    T maps a physical point p_moving (in the follow-up's world coordinates)
    to the corresponding physical point p_static (in the baseline's world
    coordinates):

        p_static  =  T  @  p_moving

    In homogeneous coordinates:

        [ x_s ]   [ r00  r01  r02  tx ] [ x_m ]
        [ y_s ] = [ r10  r11  r12  ty ] [ y_m ]
        [ z_s ]   [ r20  r21  r22  tz ] [ z_m ]
        [  1  ]   [  0    0    0    1 ] [  1  ]

    For a rigid transform:  the 3×3 submatrix is a proper rotation matrix
    (det = +1, R^T R = I); tx, ty, tz encode the translation.
    For an affine transform: the 3×3 submatrix can include scaling and shear.

BACKWARD (PULL) RESAMPLING IN final_map.transform()
──────────────────────────────────────────────────────
AffineMap.transform(moving) implements pull resampling:

  For every voxel index (i, j, k) in the OUTPUT grid (baseline shape):
    1. Convert (i, j, k) → world point p_s  using the static (baseline) affine.
    2. Apply T⁻¹ to get p_m = T⁻¹ @ p_s  (the corresponding point in followup).
    3. Convert p_m → fractional voxel coordinates in the followup volume.
    4. Tri-linearly interpolate the followup intensity there.

Every output voxel is guaranteed to be filled — no holes — unlike push
(forward) resampling which can leave empty voxels when the transform is not
surjective.  The output array has the same shape as the static (baseline).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import numpy as np

# ── Optional dipy with graceful absence handling ──────────────────────────────

try:
    from dipy.align.imaffine import (    # type: ignore
        MutualInformationMetric,
        AffineRegistration,
        transform_centers_of_mass,
    )
    from dipy.align.transforms import (  # type: ignore
        TranslationTransform3D,
        RigidTransform3D,
        AffineTransform3D,
    )
    _HAS_DIPY = True
except ImportError:
    _HAS_DIPY = False

from .config import LongitudinalConfig, DEFAULT_CONFIG


# ── Dependency check ──────────────────────────────────────────────────────────

def check_dipy() -> None:
    """Raise RuntimeError if dipy (or its scipy dependency) is not installed."""
    if not _HAS_DIPY:
        raise RuntimeError(
            "dipy is not installed.  Run: pip install dipy\n"
            "dipy requires scipy; both will be installed together."
        )


# ── Registration pipeline ─────────────────────────────────────────────────────

def run_registration(
    static:        np.ndarray,
    static_affine: np.ndarray,
    moving:        np.ndarray,
    moving_affine: np.ndarray,
    cfg:           LongitudinalConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Co-register the moving (follow-up) volume into the static (baseline) space.

    Returns the follow-up array resampled onto the baseline's voxel grid
    (shape == static.shape, spacing == baseline spacing).  The caller can
    then compute delta = result − static in a single numpy subtraction.

    TRANSFORMATION CHAIN
    ─────────────────────
    Step 0: Centers-of-mass translation
      A closed-form rough alignment: no MI optimisation.  Shifts the moving
      image so both brains are centred at the same world point.  Prevents the
      MI optimiser at Level 0 from starting far from the global optimum.

    Step 1: Translation optimisation (3 DOF)
      Starting from Step 0, optimises a pure-translation transform using the
      MI metric and the multi-scale pyramid.  Isolating translation first
      stabilises the largest-magnitude component of the transform before
      introducing rotational degrees of freedom.

    Step 2: Rigid body optimisation (6 DOF)
      Starting from Step 1's refined translation, adds rotation (3 Euler
      angles).  For same-subject longitudinal brain MRI, this is sufficient:
      only head tilt between sessions differs; the brain itself does not
      scale or shear between scans from the same scanner.

    Step 3 [optional]: Full affine optimisation (12 DOF)
      Starting from Step 2.  Adds 3 scaling + 3 shear DOF.  Use only with
      cfg.transform_type='affine' and cross-scanner longitudinal data.

    Parameters
    ----------
    static        : float32 (X, Y, Z) baseline array
    static_affine : (4, 4) float64 RAS-mm affine of the baseline
    moving        : float32 (X, Y, Z) follow-up array (may differ in shape)
    moving_affine : (4, 4) float64 RAS-mm affine of the follow-up
    cfg           : hyper-parameters controlling the metric and pyramid

    Returns
    -------
    registered : float32 ndarray with shape == static.shape
    """
    check_dipy()

    # ── Mutual information metric ─────────────────────────────────────────────
    # nbins : joint histogram resolution (see module docstring on MI).
    # sampling_proportion : None = all voxels; 0.3 = 30% random subsample.
    metric = MutualInformationMetric(
        cfg.nbins,
        cfg.sampling_proportion,
    )

    # ── Multi-scale pyramid ───────────────────────────────────────────────────
    # verbosity=0 suppresses the per-iteration console output that dipy
    # normally prints.  This prevents log flooding in the uvicorn server.
    #
    # AffineRegistration does not mutate cfg — it is safe to share the
    # instance across concurrent requests because .optimize() is stateless
    # (each call returns a new AffineMap).
    affreg = AffineRegistration(
        metric      = metric,
        level_iters = list(cfg.level_iters),
        sigmas      = list(cfg.sigmas),
        factors     = [int(f) for f in cfg.factors],  # dipy 1.x requires int factors
        verbosity   = 0,                              # VerbosityLevels.NONE
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 0: Centers-of-mass alignment
    # ─────────────────────────────────
    # transform_centers_of_mass() computes a pure-translation AffineMap that
    # moves the centroid of the moving image to the centroid of the static
    # image in world (RAS-mm) space.  No MI is evaluated here — this is a
    # fast closed-form computation.
    #
    # c_of_mass.affine : 4×4 world-space matrix encoding the rough translation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    c_of_mass = transform_centers_of_mass(
        static, static_affine,
        moving, moving_affine,
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 1: Translation optimisation (3 DOF)
    # ──────────────────────────────────────────
    # Starting from the CoM alignment (starting_affine = c_of_mass.affine),
    # the optimiser adjusts the three translation parameters tx, ty, tz to
    # maximise MI at each pyramid level before moving to the next.
    #
    # TranslationTransform3D parameterises the transform as:
    #
    #     T(p)  =  p + [tx, ty, tz]^T
    #
    # translation_map.affine : 4×4 world-space pure-translation matrix
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    translation_map = affreg.optimize(
        static, moving,
        TranslationTransform3D(),
        params0           = None,          # optimiser chooses initial params
        static_grid2world = static_affine,
        moving_grid2world = moving_affine,
        starting_affine   = c_of_mass.affine,
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 2: Rigid body optimisation (6 DOF)
    # ──────────────────────────────────────────
    # Starting from the refined translation, adds 3 rotational DOF.
    # RigidTransform3D parameterises rotation via Euler angles (α, β, γ) so
    # the parameter space is smooth and compact, enabling efficient L-BFGS-B
    # optimisation.
    #
    # The resulting 4×4 rigid matrix T is:
    #
    #     T  =  [ R | t ]
    #           [ 0 | 1 ]
    #
    # where R is a 3×3 proper rotation matrix (det R = +1, R^T R = I) and t
    # is the translation vector.  det R = +1 guarantees that the
    # transformation preserves brain topology (no reflections).
    #
    # rigid_map.affine : 4×4 world-space rigid matrix (the TRANSFORMATION MATRIX
    #                    described in the module docstring)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    rigid_map = affreg.optimize(
        static, moving,
        RigidTransform3D(),
        params0           = None,
        static_grid2world = static_affine,
        moving_grid2world = moving_affine,
        starting_affine   = translation_map.affine,
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 3 [optional]: Full affine optimisation (12 DOF)
    # ────────────────────────────────────────────────────
    # AffineTransform3D extends RigidTransform3D with 3 anisotropic scaling
    # parameters and 3 shear parameters:
    #
    #     A  =  R · diag(sx, sy, sz) · S_shear
    #
    # Starting from the rigid solution so the rotation and translation are
    # already close to optimal — the extra DOF only correct for scaling /
    # shear differences between scanners or protocols.
    #
    # WARNING: full affine can produce anatomically unrealistic deformations
    # (shrinking / stretching individual brain lobes) if the optimiser
    # over-fits to noise.  Only enable with cfg.transform_type='affine'.
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if cfg.transform_type == 'affine':
        final_map = affreg.optimize(
            static, moving,
            AffineTransform3D(),
            params0           = None,
            static_grid2world = static_affine,
            moving_grid2world = moving_affine,
            starting_affine   = rigid_map.affine,
        )
    else:
        # 'rigid' — sufficient for same-subject same-scanner longitudinal MRI.
        final_map = rigid_map

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # APPLY TRANSFORM: resample follow-up → baseline voxel grid
    # ──────────────────────────────────────────────────────────
    # AffineMap.transform() uses backward (pull) resampling (see module
    # docstring).  The output array has exactly static.shape — the same grid
    # as the baseline — so the subsequent delta subtraction is a trivial
    # element-wise operation with no shape mismatch.
    #
    # interpolation='linear' : tri-linear interpolation.  Nearest-neighbour
    # (order=0) would preserve voxel intensity values but produce blocky
    # boundaries, inflating the delta near registration boundaries.
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    registered: np.ndarray = final_map.transform(moving, interpolation='linear')

    return registered.astype(np.float32)
