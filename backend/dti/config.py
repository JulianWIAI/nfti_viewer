"""
dti/config.py — TractographyConfig: all tuning knobs in one place
══════════════════════════════════════════════════════════════════════

Having a single config dataclass lets the FastAPI endpoint accept optional
Form fields that override specific parameters while all the pipeline functions
receive a single typed object instead of a long keyword-argument list.

Parameter guidance
──────────────────
  fa_stop         0.2 is the standard clinical threshold.  Below this FA the
                  diffusion tensor is too isotropic to reliably indicate a
                  single fibre direction.  Raising to 0.25 increases
                  specificity but may cut tracts short; lowering to 0.15
                  recovers more tract but increases spurious paths in grey
                  matter / CSF.

  fa_seed         Must be > fa_stop.  0.3 restricts seeds to coherent white
                  matter where the principal eigenvector is reliable.
                  Lowering generates more seeds (longer pipeline runtime).

  step_size       0.5 mm is typical for DTI; smaller values increase point
                  count and runtime; larger values can overshoot sharp bends.

  max_angle       30° per step enforces anatomical plausibility.  White matter
                  fibres are smooth — sharp turns indicate noise or crossing
                  fibres that the single-tensor model cannot resolve.

  min_length_mm   Streamlines shorter than ~30 mm are typically false positives
                  arising from noisy seed voxels near the FA threshold.

  max_streamlines Hard cap for the JSON payload.  10 000 streamlines × ~30
                  points per streamline × ~24 chars per coordinate ≈ 7 MB.

  tol_error       Douglas–Peucker tolerance in mm.  1.0 mm is perceptually
                  lossless at standard rendering zoom levels and removes ~75–90%
                  of intermediate points.

  decimal_places  JSON float precision.  0.01 mm granularity is far beyond any
                  clinical DTI resolution (≥ 1.5 mm voxels), so 2 dp costs
                  nothing diagnostically while saving ~30 % of character count
                  compared to full float64 precision.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TractographyConfig:
    """
    Unified configuration for the full DTI tractography pipeline.

    All parameters have sensible defaults so the caller can omit any field
    that does not need customisation.
    """

    # ── Gradient table ────────────────────────────────────────────────────
    # Volumes with b-value ≤ b0_threshold are treated as b=0 (no diffusion
    # weighting).  FSL typically uses b0_threshold=0; some scanners write
    # small non-zero values (e.g. 5 s/mm²) for their b0 acquisitions.
    b0_threshold: float = 50.0

    # ── Brain mask (median_otsu) ──────────────────────────────────────────
    # median_radius controls the radius of the 3-D median filter used to
    # remove salt-and-pepper noise before Otsu thresholding.
    # numpass = number of filter iterations (more = smoother mask).
    median_radius: int = 4
    numpass: int       = 4

    # ── Tensor model ──────────────────────────────────────────────────────
    # WLS (weighted least squares) is more robust to noise than OLS
    # (ordinary least squares) because it down-weights measurements that
    # deviate most from the model prediction.
    fit_method: str = 'WLS'

    # ── FA thresholds ─────────────────────────────────────────────────────
    fa_stop: float = 0.20  # stop propagating below this (grey matter / CSF)
    fa_seed: float = 0.30  # seed only from above this (coherent white matter)

    # ── Sphere for ODF evaluation ─────────────────────────────────────────
    # 'symmetric362' has 362 vertices — a good balance between angular
    # resolution and computation time.  'repulsion724' doubles the vertices
    # for higher angular precision at ~2× cost.
    sphere_name: str = 'symmetric362'

    # ── Tracking ─────────────────────────────────────────────────────────
    step_size:       float = 0.50    # integration step in mm
    seeds_per_voxel: int   = 1       # seeds placed per seed-mask voxel
    max_angle:       float = 30.0    # maximum turn per step in degrees
    max_cross:       int   = 1       # max fibre directions per voxel
                                     # (1 = single-tensor assumption)

    # ── Compression / serialisation ───────────────────────────────────────
    min_length_mm:   float = 30.0    # discard shorter streamlines (likely FP)
    max_streamlines: int   = 10_000  # hard cap; random subsample if exceeded
    tol_error:       float = 1.0     # Douglas–Peucker tolerance in mm
    decimal_places:  int   = 2       # JSON coordinate precision
