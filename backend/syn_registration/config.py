"""
syn_registration/config.py — Frozen configuration dataclass for the SyN pipeline.
══════════════════════════════════════════════════════════════════════════════════

Keeps all tunable hyperparameters in one place so the router, the pipeline, and
any future callers all draw from the same source of truth.

DESIGN NOTES
────────────
Two separate schedules are embedded here:

  1. Affine pre-registration (MutualInformation + CoM → Trans → Rigid → Affine)
     Uses the same settings as the longitudinal pipeline:
       level_iters = [1000, 100, 10] — coarse-to-fine MI optimisation
       sigmas      = [3.0, 1.0, 0.0] — Gaussian pre-smoothing per level
       factors     = [4.0, 2.0, 1.0] — downsampling per level

  2. SyN diffeomorphic step (CrossCorrelation metric)
     Deliberately aggressive for web-response times:
       syn_level_iters = [10, 10, 5]
     This is ≈20× fewer iterations than a clinical overnight setting
     ([200, 100, 50]) but produces a usable non-linear warp in ~2–5 minutes
     on a 1 mm isotropic brain volume.

     cc_radius = 3 sets a 7×7×7-voxel window for local normalised CC — the
     standard choice for inter-subject structural MRI.

Frozen dataclass: all fields are immutable after construction so a single
DEFAULT_SYN_CONFIG instance can be safely shared across concurrent requests.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SynConfig:
    """Immutable configuration for the full SyN registration pipeline."""

    # ── Affine pre-registration ────────────────────────────────────────────────
    # Number of histogram bins for the Mutual Information metric.
    affine_nbins: int = 32

    # Proportion of voxels sampled for MI estimation (None = all voxels).
    # Setting to e.g. 0.3 gives a 3× speedup at the cost of slightly noisier
    # gradients; None is safer for cross-subject data.
    affine_sampling_proportion: float | None = None

    # Iterations per pyramid level: [coarse, medium, fine].
    affine_level_iters: tuple[int, ...] = (1_000, 100, 10)

    # Gaussian pre-smoothing sigma per pyramid level (mm).
    affine_sigmas: tuple[float, ...] = (3.0, 1.0, 0.0)

    # Downsampling factor per pyramid level.
    affine_factors: tuple[float, ...] = (4.0, 2.0, 1.0)

    # ── SyN diffeomorphic step ─────────────────────────────────────────────────
    # Iterations per SyN pyramid level: [coarse, medium, fine].
    # Kept deliberately small so the API responds within a few minutes.
    syn_level_iters: tuple[int, ...] = (10, 10, 5)

    # Radius of the local normalised cross-correlation window (in voxels).
    # Total window = (2*radius+1)³.  radius=3 → 7³=343-voxel window.
    cc_radius: int = 3

    # ── Safety cap ─────────────────────────────────────────────────────────────
    # Maximum number of voxels per volume.  300³ ≈ 27M voxels → ~1 GB peak RAM
    # for the pair of float32 arrays + registered output.  Reject oversized
    # uploads before allocating to give a clean 413 error.
    max_voxels: int = 300 ** 3


DEFAULT_SYN_CONFIG = SynConfig()
