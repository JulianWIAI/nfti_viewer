"""
syn_registration/pipeline.py — Full Affine + SyN diffeomorphic registration.
══════════════════════════════════════════════════════════════════════════════

Implements the two-stage inter-subject registration pipeline:

  STAGE 1 — Affine pre-registration (Mutual Information)
  ─────────────────────────────────────────────────────────
  Brings the two brains into rough alignment using a 12-DOF affine transform.
  Pipeline: centers-of-mass → Translation (3 DOF) → Rigid (6 DOF) → Affine
  (12 DOF). Identical algorithm to the longitudinal router, but run on two
  different subjects rather than two scans of the same subject.

  After this stage, `prealigned` has shape == static.shape and occupies the
  same voxel-space coordinate system as `static`.

  STAGE 2 — SyN diffeomorphic non-linear registration
  ────────────────────────────────────────────────────
  Corrects for the residual non-linear shape differences (sulcal pattern,
  individual anatomy) that the affine transform cannot capture.

  Uses dipy's SymmetricDiffeomorphicRegistration (SyN) with a
  CrossCorrelationMetric (local normalised CC with radius 3 → 7³ window).

  SyN optimises a symmetric diffeomorphic velocity field v(x,t) that
  minimises:
      E(φ) = CC(I∘φ⁻¹, J∘φ) + ‖v‖²_L²

  The symmetric formulation ensures the registration is path-independent and
  the resulting transform is invertible.

  ITERATION SCHEDULE
  ───────────────────
  By default, syn_level_iters = [10, 10, 5] — a deliberately aggressive
  schedule tuned for web-response times (~2–5 min on a 1 mm isotropic brain
  volume at 256³).  Clinical overnight pipelines use [200, 100, 50] or
  higher.  The web schedule produces a usable warp adequate for visual
  comparison in a dual-viewer, but is NOT appropriate for VBM or
  atlas-based analyses requiring high registration accuracy.

PUBLIC API
──────────
  check_syn_dependencies()     → raises RuntimeError if dipy is missing
  run_syn_pipeline(...)        → float32 warped array in static voxel space
"""
from __future__ import annotations

import numpy as np

from .config import SynConfig


# ── Dependency guard ──────────────────────────────────────────────────────────

def check_syn_dependencies() -> None:
    """
    Verify that all required packages are importable.

    Raises RuntimeError with a pip install hint if dipy is missing.
    dipy bundles scipy as a dependency; importing dipy.align is sufficient.
    """
    try:
        import dipy.align.imaffine  # noqa: F401
        import dipy.align.imwarp   # noqa: F401
        import dipy.align.metrics  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "dipy is required for SyN registration.  "
            "Run: pip install dipy"
        ) from exc


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_syn_pipeline(
    static:        np.ndarray,
    static_affine: np.ndarray,
    moving:        np.ndarray,
    moving_affine: np.ndarray,
    cfg:           SynConfig,
) -> np.ndarray:
    """
    Full affine + SyN registration: warp `moving` into `static`'s voxel space.

    Parameters
    ----------
    static        : float32 ndarray of shape (X, Y, Z) — reference (Subject A).
    static_affine : float64 ndarray of shape (4, 4) — RAS-mm world transform of static.
    moving        : float32 ndarray of shape (X', Y', Z') — moving (Subject B).
                    May have different dimensions than static.
    moving_affine : float64 ndarray of shape (4, 4) — RAS-mm world transform of moving.
    cfg           : SynConfig — registration hyperparameters.

    Returns
    -------
    warped : float32 ndarray of shape (X, Y, Z) — Subject B warped to Subject A's
             voxel space.  shape == static.shape.

    Algorithm
    ---------
    1. Centers-of-mass translation (closed-form, no optimisation).
    2. AffineRegistration (MI metric): Translation → Rigid → Affine.
    3. affine_map.transform(moving) → prealigned  (shape == static.shape).
    4. SymmetricDiffeomorphicRegistration (CC metric) on static + prealigned.
    5. mapping.transform(prealigned) → warped     (shape == static.shape).
    """
    # ── Import dipy lazily (not all deployments have it) ─────────────────────
    from dipy.align.imaffine import (
        AffineRegistration,
        MutualInformationMetric,
        transform_centers_of_mass,
    )
    from dipy.align.transforms import (
        TranslationTransform3D,
        RigidTransform3D,
        AffineTransform3D,
    )
    from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from dipy.align.metrics import CrossCorrelationMetric

    # ── Stage 1a: Centers-of-mass pre-alignment (closed-form) ────────────────
    # Provides a good initial translation before the MI optimisation begins.
    c_of_mass = transform_centers_of_mass(
        static,       static_affine,
        moving,       moving_affine,
    )

    # ── Stage 1b: Affine pre-registration (MI metric) ─────────────────────────
    # Three nested transform levels; each feeds its output into the next as the
    # starting_affine — this is the standard coarse-to-fine initialisation chain.
    metric = MutualInformationMetric(
        cfg.affine_nbins,
        cfg.affine_sampling_proportion,
    )
    affreg = AffineRegistration(
        metric      = metric,
        level_iters = list(cfg.affine_level_iters),
        sigmas      = list(cfg.affine_sigmas),
        factors     = [int(f) for f in cfg.affine_factors],  # dipy 1.x requires int factors
        verbosity   = 0,     # suppress console progress bars
    )

    # Step 1: Translation (3 DOF) — warm-started from CoM
    translation_map = affreg.optimize(
        static, moving,
        TranslationTransform3D(), None,
        static_affine, moving_affine,
        starting_affine = c_of_mass.affine,
    )
    del c_of_mass

    # Step 2: Rigid body (6 DOF) — warm-started from translation
    rigid_map = affreg.optimize(
        static, moving,
        RigidTransform3D(), None,
        static_affine, moving_affine,
        starting_affine = translation_map.affine,
    )
    del translation_map

    # Step 3: Full affine (12 DOF) — warm-started from rigid
    affine_map = affreg.optimize(
        static, moving,
        AffineTransform3D(), None,
        static_affine, moving_affine,
        starting_affine = rigid_map.affine,
    )
    del rigid_map, metric, affreg

    # ── Stage 1c: Resample moving → static voxel space ────────────────────────
    # After transform(), prealigned has:
    #   shape  == static.shape
    #   affine == static_affine (same voxel coordinate system)
    # Releasing moving here frees its RAM before SyN allocates its own buffers.
    prealigned = affine_map.transform(moving, interpolation='linear')
    del moving, affine_map

    # ── Stage 2: SyN diffeomorphic registration ───────────────────────────────
    # Both static and prealigned are now in the same voxel-space, so we pass
    # static_affine for both grid2world arguments.
    #
    # CrossCorrelationMetric(radius=r) uses a (2r+1)³ neighbourhood; r=3 gives
    # a 7³-voxel window — the standard choice for structural MRI.
    cc_metric = CrossCorrelationMetric(radius=cfg.cc_radius)
    sdr = SymmetricDiffeomorphicRegistration(
        cc_metric,
        list(cfg.syn_level_iters),
    )

    mapping = sdr.optimize(
        static,
        prealigned,
        static_grid2world = static_affine,
        moving_grid2world = static_affine,  # prealigned is already in static space
    )

    # Apply the diffeomorphic warp to produce the final registered volume.
    warped = mapping.transform(prealigned, interpolation='linear')
    del prealigned, mapping, sdr, cc_metric

    return warped.astype(np.float32)


def run_syn_pipeline_with_seg(
    static:        np.ndarray,
    static_affine: np.ndarray,
    moving:        np.ndarray,
    moving_affine: np.ndarray,
    moving_seg:    np.ndarray,
    cfg:           SynConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full affine + SyN registration that also warps a segmentation label map.

    Identical to run_syn_pipeline() for the intensity volume, but additionally
    applies every transform to `moving_seg` using nearest-neighbour interpolation
    so that integer label values (e.g. 17 = LH hippocampus) are not destroyed
    by linear blending.

    Parameters
    ----------
    static        : float32 (X, Y, Z) reference volume (Subject A).
    static_affine : float64 (4, 4) RAS-mm affine for static.
    moving        : float32 (X', Y', Z') moving volume (Subject B).
    moving_affine : float64 (4, 4) RAS-mm affine for moving.
    moving_seg    : uint8  (X', Y', Z') label map for moving — FreeSurfer IDs.
    cfg           : SynConfig — registration hyperparameters.

    Returns
    -------
    warped_vol : float32 (X, Y, Z) — intensity volume warped to static space.
    warped_seg : uint8  (X, Y, Z) — label map warped to static space via
                 nearest-neighbour interpolation (label values preserved).
    """
    from dipy.align.imaffine import (
        AffineRegistration,
        MutualInformationMetric,
        transform_centers_of_mass,
    )
    from dipy.align.transforms import (
        TranslationTransform3D,
        RigidTransform3D,
        AffineTransform3D,
    )
    from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from dipy.align.metrics import CrossCorrelationMetric

    # ── Stage 1a: Centers-of-mass pre-alignment ───────────────────────────────
    c_of_mass = transform_centers_of_mass(
        static,       static_affine,
        moving,       moving_affine,
    )

    # ── Stage 1b: Affine pre-registration (MI metric) ─────────────────────────
    metric = MutualInformationMetric(
        cfg.affine_nbins,
        cfg.affine_sampling_proportion,
    )
    affreg = AffineRegistration(
        metric      = metric,
        level_iters = list(cfg.affine_level_iters),
        sigmas      = list(cfg.affine_sigmas),
        factors     = [int(f) for f in cfg.affine_factors],  # dipy 1.x requires int factors
        verbosity   = 0,
    )

    translation_map = affreg.optimize(
        static, moving,
        TranslationTransform3D(), None,
        static_affine, moving_affine,
        starting_affine = c_of_mass.affine,
    )
    del c_of_mass

    rigid_map = affreg.optimize(
        static, moving,
        RigidTransform3D(), None,
        static_affine, moving_affine,
        starting_affine = translation_map.affine,
    )
    del translation_map

    affine_map = affreg.optimize(
        static, moving,
        AffineTransform3D(), None,
        static_affine, moving_affine,
        starting_affine = rigid_map.affine,
    )
    del rigid_map, metric, affreg

    # ── Stage 1c: Resample both volume and segmentation ───────────────────────
    # Intensity volume uses linear interpolation; segmentation uses nearest so
    # FreeSurfer label IDs (integers) are not blurred by floating-point mixing.
    prealigned     = affine_map.transform(moving,                      interpolation='linear')
    prealigned_seg = affine_map.transform(moving_seg.astype(np.float32), interpolation='nearest')
    del moving, moving_affine, affine_map

    # ── Stage 2: SyN diffeomorphic registration ───────────────────────────────
    cc_metric = CrossCorrelationMetric(radius=cfg.cc_radius)
    sdr = SymmetricDiffeomorphicRegistration(
        cc_metric,
        list(cfg.syn_level_iters),
    )

    mapping = sdr.optimize(
        static,
        prealigned,
        static_grid2world = static_affine,
        moving_grid2world = static_affine,
    )

    # Apply the same diffeomorphic warp to both arrays.
    warped_vol = mapping.transform(prealigned,     interpolation='linear')
    warped_seg = mapping.transform(prealigned_seg, interpolation='nearest')
    del prealigned, prealigned_seg, mapping, sdr, cc_metric

    return warped_vol.astype(np.float32), warped_seg.astype(np.uint8)
