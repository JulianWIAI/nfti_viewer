"""
dti/compression.py — Streamline decimation, filtering, and JSON serialisation
══════════════════════════════════════════════════════════════════════════════

WHY COMPRESSION IS NON-NEGOTIABLE FOR WEB DELIVERY
────────────────────────────────────────────────────
A typical whole-brain DTI tractogram at standard clinical resolution contains:

  • 50 000–200 000 streamlines
  • 100–500 points per streamline (at step_size=0.5 mm for 50–250 mm fibres)
  • 3 float64 values per point

Raw JSON with full precision:
  100 000 streamlines × 200 pts × 3 coords × ~18 chars = ~10.8 GB
  ─── completely unusable for a web browser ───

Target: < 10–15 MB  (fits in a single HTTP response without streaming)

THREE-STAGE REDUCTION
──────────────────────

Stage 1 — LENGTH FILTER  (discard streamlines shorter than min_length_mm)
──────────────────────────
Short streamlines (< 30 mm) arise from:
  • Noise in voxels at the FA boundary (many seeds, few steps before stopping)
  • Partial-volume effects at the WM/GM interface
  • Artefactual tracking through crossing-fibre regions

Removing them eliminates ~30–60% of streamlines without affecting any
anatomically meaningful tract.  They are also visually uninformative — too
short to reveal tract morphology.

Arc-length computation (manual, version-safe):
  L = Σ_i ||r_{i+1} − r_i||₂    (sum of Euclidean step lengths in mm)

Stage 2 — RANDOM SUBSAMPLING  (cap at max_streamlines)
──────────────────────────────────────────────────────
After length filtering, if the count still exceeds max_streamlines, we
randomly select without replacement.  Random subsampling is statistically
unbiased: the resulting subset is geometrically representative of the full
tractogram.  Deterministic subsampling (e.g. every N-th streamline) would
introduce spatial bias because the tracking order follows a raster-scan seed
ordering.

Stage 3 — POINT DECIMATION  (Douglas–Peucker algorithm)
─────────────────────────────────────────────────────────
dipy's compress_streamlines implements a variant of the Ramer–Douglas–Peucker
(RDP) polyline simplification algorithm:

  Given a polyline P = [p₀, p₁, …, pₙ], the algorithm recursively removes
  intermediate points whose perpendicular distance to the chord connecting
  their neighbours is less than tol_error mm.  The end-points are always kept.

  At tol_error=1.0 mm:
    • The path deviates from the original by at most 1 mm anywhere.
    • Typical reduction: 70–90% of intermediate points removed.
    • Straight white-matter tracts (e.g. corpus callosum) lose > 95% of points.
    • Curved tracts (e.g. cingulum, arcuate fasciculus) retain more points.

  A 1 mm error is well below the voxel resolution of any clinical DTI dataset
  (≥ 1.5 mm isotropic) and imperceptible at standard rendering zoom levels.

COMBINED EFFECT (typical numbers)
──────────────────────────────────
  Before:  100 000 streamlines × 200 points × 3 coords
  After stage 1:  40 000 streamlines (length filter removes 60%)
  After stage 2:  10 000 streamlines (random subsample)
  After stage 3:  10 000 streamlines × ~25 points × 3 coords
  JSON size: 10 000 × 25 × 3 × ~8 chars ≈ 6 MB  ✓

SERIALISATION FORMAT
──────────────────────
Each streamline is a JSON array of [x, y, z] triples:

    [
        [[x₀, y₀, z₀], [x₁, y₁, z₁], ...],   ← streamline 0
        [[x₀, y₀, z₀], ...],                   ← streamline 1
        ...
    ]

Coordinates are in world (RAS mm) space and rounded to `decimal_places`
decimal places (default 2, i.e. 0.01 mm precision).

Vectorised rounding via integer arithmetic:
  rounded = round(x × 10^d) / 10^d
  (cheaper than Python's round() in a nested Python loop)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class TractographyResult:
    """
    Final output of the full tractography + compression pipeline.

    Fields are JSON-serialisable and map directly onto the FastAPI response model.

    Attributes
    ----------
    streamlines_json    : nested list [ [ [x,y,z], ... ], ... ] — one list
                          per streamline, each a list of [x, y, z] triples
    n_streamlines       : number of streamlines in the output
    n_before_filter     : number of streamlines before length filter + subsample
    mean_length_mm      : mean arc-length of the output streamlines in mm
    peak_fa             : maximum FA in the FA map (from TensorFitResult)
    duration_ms         : total wall-clock time for the pipeline in milliseconds
    """
    streamlines_json:  list[list[list[float]]]
    n_streamlines:     int
    n_before_filter:   int
    mean_length_mm:    float
    peak_fa:           float
    duration_ms:       float


# ── Internal helpers ─────────────────────────────────────────────────────────

def _arc_lengths(streamlines: object) -> np.ndarray:
    """
    Compute the arc-length of each streamline in mm.

    Arc-length is defined as the sum of Euclidean distances between consecutive
    points:  L = Σ_i ||r_{i+1} − r_i||₂

    Implemented manually rather than importing from dipy.tracking.utils.length
    so the function is robust to dipy version changes.

    Parameters
    ----------
    streamlines : dipy Streamlines or any iterable of (P_i, 3) arrays

    Returns
    -------
    lengths : (N,) float64 array of arc-lengths in mm
    """
    lengths = []
    for sl in streamlines:
        if len(sl) < 2:
            lengths.append(0.0)
        else:
            # np.diff(sl, axis=0) gives (P-1, 3) step vectors
            # linalg.norm with axis=1 gives (P-1,) step lengths
            lengths.append(float(np.sum(np.linalg.norm(np.diff(sl, axis=0), axis=1))))
    return np.array(lengths, dtype=np.float64)


# ── Public API ────────────────────────────────────────────────────────────────

def compress_and_filter(
    streamlines: object,              # dipy Streamlines
    config:      'TractographyConfig',
    rng:         np.random.Generator | None = None,
) -> object:                          # returns dipy Streamlines
    """
    Apply length filtering, random subsampling, and point decimation.

    This function is the bottleneck between tractography (many streamlines,
    many points) and serialisation (few streamlines, few points per streamline).

    Parameters
    ----------
    streamlines : raw Streamlines from run_tractography()
    config      : TractographyConfig — provides min_length_mm, max_streamlines,
                  tol_error
    rng         : optional numpy Generator for reproducible subsampling
                  (defaults to seed=42 if None)

    Returns
    -------
    Filtered and decimated Streamlines (dipy object, still lazy-iterable)
    """
    from dipy.tracking.streamline import (    # type: ignore[import]
        Streamlines,
        compress_streamlines,
    )

    if rng is None:
        rng = np.random.default_rng(seed=42)

    # Convert to Streamlines if not already (ensures array() indexing works)
    streamlines = Streamlines(streamlines)

    # ── Stage 1: length filter ─────────────────────────────────────────────
    # Compute arc-lengths before any decimation so we filter on the true
    # path length, not the chord length.
    lengths  = _arc_lengths(streamlines)
    long_idx = np.where(lengths >= config.min_length_mm)[0]
    filtered = Streamlines(streamlines[i] for i in long_idx)

    # ── Stage 2: random subsampling ────────────────────────────────────────
    n_filtered = len(filtered)
    if n_filtered > config.max_streamlines:
        # Choose max_streamlines indices uniformly at random (no replacement).
        chosen = rng.choice(n_filtered, size=config.max_streamlines, replace=False)
        chosen.sort()                              # preserve spatial scan order
        filtered = Streamlines(filtered[i] for i in chosen)

    # ── Stage 3: point decimation (Douglas–Peucker) ────────────────────────
    # compress_streamlines removes interior points whose perpendicular distance
    # to the chord connecting their neighbours is < tol_error mm.
    # End-points are always preserved.
    decimated = compress_streamlines(filtered, tol_error=config.tol_error)

    return Streamlines(decimated)


def serialize_streamlines(
    streamlines:    object,     # dipy Streamlines
    decimal_places: int = 2,
) -> list[list[list[float]]]:
    """
    Convert a Streamlines object to a JSON-serialisable nested Python list.

    The output format mirrors what vtk.js expects for line rendering:
        [
            [[x0, y0, z0], [x1, y1, z1], ...],   # streamline 0
            [[x0, y0, z0], ...],                   # streamline 1
            ...
        ]

    Coordinates are in world (RAS mm) space.
    Rounded to `decimal_places` decimal places using vectorised integer
    arithmetic for performance: round(x × 10^d) / 10^d.

    Parameters
    ----------
    streamlines    : decimated/filtered Streamlines
    decimal_places : precision of coordinate floats in the JSON output
                     (2 = 0.01 mm, adequate for any clinical DTI resolution)

    Returns
    -------
    Nested Python list suitable for json.dumps() / orjson.dumps().
    """
    factor = 10 ** decimal_places
    result: list[list[list[float]]] = []

    for sl in streamlines:
        # sl is a (P, 3) float64 numpy array, P = number of points
        # Vectorised rounding:
        #   1. Multiply by factor           → shift decimal place right
        #   2. np.round() to nearest int   → snap to grid
        #   3. Divide by factor            → shift decimal place back
        #   4. .tolist()                   → convert to nested Python lists
        rounded = (np.round(sl * factor) / factor).tolist()
        result.append(rounded)

    return result
