"""
longitudinal/config.py — Configuration dataclass for the longitudinal delta pipeline.
══════════════════════════════════════════════════════════════════════════════════════

All registration hyper-parameters live here so the router and the registration
module stay free of magic constants.  Every field has a comment explaining the
range of sensible values and the effect of changing it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LongitudinalConfig:
    """
    Hyper-parameters for the dipy affine registration + delta pipeline.

    REGISTRATION STRATEGY
    ──────────────────────
    'rigid'  → 6 DOF  (3 rotation + 3 translation).
               Best choice for same-subject, same-scanner longitudinal scans
               where only head position differs between sessions.  Preserves
               all anatomical proportions — no scaling or shearing applied.

    'affine' → 12 DOF (rotation + translation + scaling + shear).
               Needed when scans come from different scanners or field-of-view
               settings with different voxel sizes or slice orientations.
               WARNING: can distort anatomy if the optimiser is under-constrained
               (very short sessions, low brain contrast).  Use with caution.

    MULTI-SCALE PYRAMID
    ────────────────────
    level_iters, sigmas, factors together define the coarse-to-fine schedule.
    All three tuples must have the same length.  The schedule runs from index 0
    (coarsest) to index -1 (finest):

      level_iters  : max gradient-descent iterations per level.
                     Higher → more accurate but slower.  10 000 / 1 000 / 100
                     is the standard dipy tutorial value for brain MRI.

      sigmas        : Gaussian smoothing σ in *voxels* applied to both images
                     before computing MI.  Larger σ → broader basin of
                     attraction (good for the coarse level to escape local
                     optima); 0.0 at the finest level preserves all detail.

      factors       : integer downsampling factor per level.  4 → quarter
                     resolution; 2 → half; 1 → full resolution.  Downsampling
                     speeds up MI computation by reducing the number of voxels.

    MUTUAL INFORMATION BINS
    ────────────────────────
    nbins controls the resolution of the joint intensity histogram used to
    estimate MI.  The standard value in the brain-MRI registration literature
    (Maes et al. 1997; Viola & Wells 1997) is 32.
      • Fewer bins   → faster, but coarser intensity discretisation.
      • More bins    → slower with diminishing accuracy gains above 64.

    SAMPLING PROPORTION
    ────────────────────
    sampling_proportion: fraction of voxels randomly sampled per pyramid level
    when estimating the MI metric.  None uses all voxels (most accurate).
    Setting 0.3 uses a 30% random subsample — 3× faster MI estimation with
    minimal accuracy cost for large volumes (> 256³).

    MEMORY GUARD
    ─────────────
    max_voxels: maximum number of voxels in a single input volume before the
    pipeline raises MemoryError.  Two float32 volumes + the registered output
    + the delta ≈ 4 × voxels × 4 bytes.  50M voxels → ~800 MB peak.
    """

    # ── Mutual information metric ─────────────────────────────────────────────
    # Joint histogram resolution (bins per axis of the 2-D intensity histogram).
    nbins: int = 32

    # Fraction of voxels used for MI estimation (None = all voxels).
    sampling_proportion: float | None = None

    # ── Multi-scale pyramid schedule (coarsest → finest) ─────────────────────
    level_iters: tuple[int, ...] = (10_000, 1_000, 100)
    sigmas:      tuple[float, ...] = (3.0, 1.0, 0.0)
    factors:     tuple[float, ...] = (4.0, 2.0, 1.0)

    # ── Registration strategy ─────────────────────────────────────────────────
    # 'rigid' (6 DOF) or 'affine' (12 DOF).
    transform_type: str = 'rigid'

    # ── Memory safety ─────────────────────────────────────────────────────────
    # Maximum voxels per input volume.  Raised before any scipy/dipy call.
    # Peak RAM ≈ 4 × max_voxels × 4 bytes (baseline + followup + registered + delta).
    # 50 M voxels → ~800 MB peak, safe for typical workstations.
    max_voxels: int = 50_000_000


# Singleton used as the default argument throughout the package.
DEFAULT_CONFIG = LongitudinalConfig()
