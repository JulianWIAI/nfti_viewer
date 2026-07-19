"""
routers/localization.py — EEG cortical source localisation endpoint
════════════════════════════════════════════════════════════════════════════════

POST /api/eeg/localize

Accepts a raw BrainVision EEG file set (uploaded as multipart form-data or
referenced by a local path) and solves the electromagnetic inverse problem to
recover the 3-D cortical origins of the measured scalp potentials.

Returns a JSON list of peak activation coordinates in MNI space:
    [{"x": float, "y": float, "z": float, "amplitude": float}, ...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE ELECTROMAGNETIC FORWARD MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A set of current dipoles J ∈ ℝ^{n_src × 3} (one per cortical source, oriented
in the surface normal direction) produces scalp potentials M ∈ ℝ^{n_ch × n_t}
according to:

    M  =  L · J  +  ε                                                  (1)

    L  ∈ ℝ^{n_ch × n_src}   — lead-field matrix ("forward operator")
    J  ∈ ℝ^{n_src × n_t}    — unknown source currents  [A·m]
    ε  ∈ ℝ^{n_ch × n_t}    — sensor noise  (Gaussian, covariance C_n)

The lead-field entry L[i, j] is the potential measured at electrode i
when dipole j has unit amplitude and all other dipoles are silent.
It is computed analytically by integrating the quasi-static Maxwell
equations over a Boundary Element Model (BEM) of the head.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE INVERSE PROBLEM AND REGULARISATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Recovering J from M is ill-posed (n_src ≫ n_ch, so the system is severely
underdetermined).  The minimum-norm estimate (MNE) adds an L2 penalty on ‖J‖:

    Ĵ_MNE  =  argmin‖M − L·J‖²_{C_n⁻¹}  +  λ²·‖J‖²_K⁻¹

The closed-form solution is a Wiener filter:

    W  =  K · Lᵀ · (L · K · Lᵀ + λ² · C_n)⁻¹                        (2)
    Ĵ  =  W · M                                                        (3)

    K  ∈ ℝ^{n_src × n_src}  — source covariance prior
                               (diagonal, depth-weighted to compensate for
                               the systematic bias toward superficial sources)
    λ² =  1 / SNR²           — Tikhonov regularisation parameter
                               (larger → stronger smoothing → deeper sources)

The MNE estimate is spatially blurred.  Two post-hoc noise-normalisation
schemes yield better-localised z-score maps:

  dSPM (Dale et al., 2000)
  ─────────────────────────
  The noise floor at each source i is the expected std of Ĵ[i] under the null:

      σ_dSPM[i]  =  sqrt( [W · C_n · Wᵀ][i,i] )                      (4a)
      z_dSPM[i]  =  Ĵ[i] / σ_dSPM[i]            (dimensionless z-score)

  sLORETA (Pascual-Marqui, 2002)
  ────────────────────────────────
  Normalise by the full signal-plus-noise covariance, which yields exact zero
  localisation error for a point source in the noise-free limit:

      C_data       =  L · K · Lᵀ + λ² · C_n                           (4b)
      σ_sLORETA[i] =  sqrt( [W · C_data · Wᵀ][i,i] )
      z_sLORETA[i] =  Ĵ[i] / σ_sLORETA[i]

  Raw MNE
  ────────
  Return Ĵ directly in physical units (A·m) without normalisation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COORDINATE FRAME CHAIN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Four coordinate frames are involved; each transition is a 4×4 rigid-body
(or affine) transformation matrix:

  ┌──────────────────────────┐
  │  Head frame (HEAD)       │   Origin: midpoint of LPA and RPA landmarks
  │  Unit: metres            │   X-axis: points toward right pre-auricular
  │                          │   Z-axis: points toward vertex (up)
  │  Channel positions and   │
  │  electrode montage live  │
  │  in this frame.          │
  └────────────┬─────────────┘
               │  T_head_to_mri  (4×4 rigid body, the "trans" file)
               │  For a standard 10-20 montage + fsaverage:
               │  T_head_to_mri ≈ identity (both frames aligned by convention)
               │  MNE shorthand: trans = 'fsaverage'
               ▼
  ┌──────────────────────────┐
  │  MRI / surface RAS frame │   Origin: centre of mass of cortical surface
  │  Unit: metres            │   Axes: Right / Anterior / Superior
  │                          │
  │  src[h]['rr']  — vertex  │   h=0: left hemisphere, h=1: right
  │  positions in this frame │
  └────────────┬─────────────┘
               │  T_mri_to_mni  (12-parameter affine stored in
               │               fsaverage/mri/transforms/talairach.xfm)
               │  Applied internally by mne.vertex_to_mni()
               ▼
  ┌──────────────────────────┐
  │  MNI 152 space           │   Standard space for cross-subject comparison
  │  Unit: millimetres       │   Coordinates returned to the vtk.js frontend
  └──────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PEAK EXTRACTION — GREEDY NON-MAXIMUM SUPPRESSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Adjacent cortical vertices often have correlated amplitudes, so the top-N
sources by amplitude alone would cluster around a single focus.  We apply
greedy non-maximum suppression (NMS) with a minimum spatial separation:

  1. Sort all sources by amplitude A[i] = mean(|Ĵ[i, t_min:t_max]|), descending.
  2. Accept the highest source unconditionally.
  3. Accept source i only if dist(r_i, r_j) > min_dist_mm for every already-
     accepted source j  (Euclidean distance in MNI mm space).
  4. Stop when n_peaks sources have been accepted.

This gives spatially spread peaks that represent distinct activation foci
rather than a single hot-spot repeated n_peaks times.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import tempfile
import time
from collections import OrderedDict
from typing import Literal

import mne
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Module-level geometry caches ──────────────────────────────────────────────
#
# src and BEM only depend on the fsaverage template and the source-space
# density parameter (spacing).  They are identical for every uploaded EEG file,
# so caching them avoids re-running ~5 s of setup per request.
#
# The forward solution additionally depends on channel positions; it is cached
# separately by a hash of the channel layout.

# subjects_dir returned by fetch_fsaverage — set on first call.
_SUBJECTS_DIR: str | None = None

# Source space cache: spacing → mne.SourceSpaces object.
_SRC_CACHE: dict[str, object] = {}

# BEM cache: single object (fsaverage BEM never changes).
_BEM_CACHE: object | None = None

# Forward solution cache: (channel_hash, spacing) → forward dict.
# OrderedDict enables simple FIFO eviction when the cache is full.
_FWD_CACHE: OrderedDict[tuple[str, str], object] = OrderedDict()
_FWD_CACHE_MAX = 8   # limit RAM: each forward solution is ~10-50 MB


# ── Pydantic response models ──────────────────────────────────────────────────

class SourcePeak(BaseModel):
    """One peak activation focus in MNI space."""

    x:         float   # MNI X coordinate (mm) — positive = right hemisphere
    y:         float   # MNI Y coordinate (mm) — positive = anterior
    z:         float   # MNI Z coordinate (mm) — positive = superior
    amplitude: float   # time-averaged absolute activation at this source
    hemisphere: str    # "lh" or "rh"


class LocalizeResponse(BaseModel):
    """Complete response from POST /api/eeg/localize."""

    method:        str              # dSPM | sLORETA | MNE
    n_peaks:       int              # number of peaks actually returned
    n_sources:     int              # total sources in the full source space
    peaks:         list[SourcePeak] # MNI coordinates + amplitudes (NMS-filtered)
    tmin:          float            # analysis window start used (s)
    tmax:          float            # analysis window end used (s)
    duration_ms:   float            # wall-clock compute time


# ── Geometry setup helpers (called once per process) ──────────────────────────

def _init_subjects_dir() -> str:
    """
    Return the MNE fsaverage subjects directory, downloading once per process.

    mne.datasets.fetch_fsaverage() checks for an existing local copy before
    downloading, so repeated calls are cheap after the first (~50 MB download).
    Thread-safe under Python's GIL: the global assignment is atomic.
    """
    global _SUBJECTS_DIR
    if _SUBJECTS_DIR is None:
        logger.info("Fetching MNE fsaverage template (~50 MB, one-time download)…")
        _SUBJECTS_DIR = mne.datasets.fetch_fsaverage(verbose=False)
    return _SUBJECTS_DIR


def _get_cached_bem(subjects_dir: str) -> object:
    """
    Return the precomputed 3-shell BEM solution for fsaverage, loading once.

    The BEM solution file is shipped with the MNE fsaverage dataset:
        fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif

    The three shells are:
      1. Inner skull (brain surface)       — conductivity ≈ 0.3300 S/m
      2. Outer skull (cranium surface)     — conductivity ≈ 0.0041 S/m
      3. Scalp                             — conductivity ≈ 0.3300 S/m

    The skull's low conductivity causes the smearing of scalp potentials
    that makes EEG source localisation inherently less accurate than MEG.
    The BEM models this effect analytically.

    If the precomputed file is absent (e.g. custom MNE install), the BEM is
    built from the fsaverage surfaces — this takes ~2–5 minutes and is only
    done once.
    """
    global _BEM_CACHE
    if _BEM_CACHE is None:
        bem_path = os.path.join(
            subjects_dir, 'fsaverage', 'bem',
            'fsaverage-5120-5120-5120-bem-sol.fif',
        )
        if os.path.exists(bem_path):
            logger.info("Loading precomputed fsaverage BEM solution…")
            _BEM_CACHE = mne.read_bem_solution(bem_path, verbose=False)
        else:
            logger.warning(
                "BEM file not found at %s — rebuilding from surfaces (~2 min)…",
                bem_path,
            )
            model     = mne.make_bem_model(
                'fsaverage', ico=4, subjects_dir=subjects_dir, verbose=False,
            )
            _BEM_CACHE = mne.make_bem_solution(model, verbose=False)
    return _BEM_CACHE


def _get_cached_source_space(subjects_dir: str, spacing: str) -> object:
    """
    Return a cortical source space for fsaverage at the requested density.

    mne.setup_source_space() places one candidate dipole at the centre of each
    face of a recursively subdivided icosahedron (ico) or octahedron (oct)
    projected onto the cortical surface.  The `spacing` parameter controls
    the mesh resolution:

        spacing  n_sources (total)  approx. spacing
        ────────────────────────────────────────────
        'oct5'   2,048              ~6 mm
        'oct6'   8,192              ~4 mm   ← default
        'ico4'   5,120              ~5 mm
        'ico5'   20,484             ~2 mm   (high-res, slow fwd solution)

    Source positions (src[h]['rr']) are in metres, in the MRI surface RAS
    frame.  The 'used' array (src[h]['inuse']) indicates which vertices
    survived the mindist exclusion in make_forward_solution.
    """
    if spacing not in _SRC_CACHE:
        logger.info("Setting up fsaverage source space (spacing=%s)…", spacing)
        _SRC_CACHE[spacing] = mne.setup_source_space(
            'fsaverage',
            spacing=spacing,
            subjects_dir=subjects_dir,
            verbose=False,
        )
    return _SRC_CACHE[spacing]


# ── Channel-layout fingerprint (for forward-solution caching) ─────────────────

def _channel_layout_hash(info: mne.Info) -> str:
    """
    Return a short hex digest that uniquely identifies the EEG channel layout.

    The forward solution depends only on:
      • Which channels are present (by name)
      • Their positions in head coordinates (loc[:3])

    It does NOT depend on the actual recorded data values, so any two files
    with identical 64-channel 10-20 cap configurations will share one cached
    forward solution.

    Positions are rounded to 0.01 mm before hashing to tolerate floating-point
    noise from different file-writing software.
    """
    # Collect (name, rounded_position) tuples for all EEG channels.
    eeg_idx = mne.pick_types(info, eeg=True)
    rows: list[str] = []
    for i in eeg_idx:
        ch   = info['chs'][i]
        pos  = tuple(round(float(v) * 100_000) for v in ch['loc'][:3])   # 0.01 mm
        rows.append(f"{ch['ch_name']}|{pos}")

    digest = hashlib.sha256("\n".join(rows).encode()).hexdigest()
    return digest[:16]    # 16 hex chars = 64 bits — collision probability negligible


# ── EEG montage assignment ────────────────────────────────────────────────────

def _ensure_eeg_positions(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Return a copy of `raw` containing only EEG channels, with positions set.

    HEAD COORDINATE FRAME ASSIGNMENT
    ──────────────────────────────────
    The forward solution requires electrode positions in the head coordinate
    frame (defined by LPA, RPA, nasion landmarks).  BrainVision files exported
    without digitisation will have all loc[:3] == 0.

    When positions are absent, we assign the standard 10-20 montage:
      • mne.channels.make_standard_montage('standard_1020')
      • Positions are averages across subjects — appropriate for group analysis
        but less accurate than subject-specific digitisation.
      • Channels not found in the 10-20 layout are kept but get no position
        and are excluded from the forward solution by make_forward_solution.

    The montage sets both the electrode positions (in head coordinates, metres)
    AND the head shape (digitisation points representing the scalp surface),
    which the BEM solver uses to orient the current dipoles.
    """
    # Work on a copy — never mutate the object stored in the session cache.
    raw = raw.copy().pick_types(eeg=True, meg=False, stim=False, exclude='bads')

    eeg_idx     = mne.pick_types(raw.info, eeg=True)
    has_pos     = any(
        np.any(raw.info['chs'][i]['loc'][:3] != 0.0) for i in eeg_idx
    )

    if not has_pos:
        logger.info(
            "No electrode positions in file — assigning standard_1020 montage."
        )
        montage = mne.channels.make_standard_montage('standard_1020')
        # match_case=False: tolerate Fp1 vs FP1 etc.
        # on_missing='warn': skip channels not in the 10-20 layout.
        raw.set_montage(montage, match_case=False, on_missing='warn')

    return raw


# ── File loading ──────────────────────────────────────────────────────────────

def _load_brainvision(
    vhdr_bytes: bytes,
    eeg_bytes:  bytes,
    vmrk_bytes: bytes | None,
    vhdr_name:  str,
    eeg_name:   str,
    vmrk_name:  str | None,
) -> mne.io.BaseRaw:
    """
    Write BrainVision bytes to a temporary directory and load with MNE.

    WHY A TEMP DIRECTORY?
    ──────────────────────
    BrainVision is a multi-file format.  The .vhdr header contains relative
    path references to the .eeg data file and the .vmrk marker file:

        DataFile=sub01.eeg
        MarkerFile=sub01.vmrk

    MNE resolves these paths relative to the .vhdr location.  Placing all
    three files in the same temporary directory satisfies this requirement
    without modifying the file content.

    The caller is responsible for deleting the temp directory after use.
    We therefore return (raw, tmpdir) and let _run_pipeline() clean up.
    """
    tmpdir = tempfile.mkdtemp(prefix="eeg_localize_")
    try:
        vhdr_path = os.path.join(tmpdir, vhdr_name)
        eeg_path  = os.path.join(tmpdir, eeg_name)

        with open(vhdr_path, 'wb') as f:
            f.write(vhdr_bytes)
        with open(eeg_path, 'wb') as f:
            f.write(eeg_bytes)

        if vmrk_bytes is not None and vmrk_name is not None:
            with open(os.path.join(tmpdir, vmrk_name), 'wb') as f:
                f.write(vmrk_bytes)

        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    return raw, tmpdir


# ── Forward solution (cached) ─────────────────────────────────────────────────

def _get_or_compute_forward(
    info:         mne.Info,
    subjects_dir: str,
    spacing:      str,
) -> object:
    """
    Return the forward solution for the given channel layout, computing if needed.

    WHAT THE FORWARD SOLUTION CONTAINS
    ─────────────────────────────────────
    fwd['sol']['data']  ∈ ℝ^{n_ch × (n_src × 3)}
        The lead-field matrix L (before orientation collapsing).
        Each triplet of columns [j, j+1, j+2] represents the three Cartesian
        components (normal, tangential-1, tangential-2 in surface RAS) of one
        source dipole.

    After convert_forward_solution(surf_ori=True, force_fixed=False):
        Columns are reordered to surface-oriented frame — the first component
        is along the cortical surface normal, others are tangential.

    After convert_forward_solution(surf_ori=True, force_fixed=True):
        Only the normal component survives → L ∈ ℝ^{n_ch × n_src}.
        MNE make_inverse_operator will further handle orientation internally
        based on the `loose` parameter.

    CACHING STRATEGY
    ─────────────────
    Key = (channel_hash, spacing).  The forward solution is purely a function
    of channel geometry and source space — not of the recorded signal values.
    Stale entries are evicted FIFO when the cache exceeds _FWD_CACHE_MAX.
    """
    layout_hash = _channel_layout_hash(info)
    cache_key   = (layout_hash, spacing)

    if cache_key in _FWD_CACHE:
        logger.debug("Forward solution cache hit (hash=%s, spacing=%s).", layout_hash, spacing)
        # Move to end to record recent use (FIFO eviction favours older entries).
        _FWD_CACHE.move_to_end(cache_key)
        return _FWD_CACHE[cache_key]

    logger.info(
        "Computing forward solution (hash=%s, spacing=%s) — "
        "this may take 30–90 s on the first call…",
        layout_hash, spacing,
    )
    src = _get_cached_source_space(subjects_dir, spacing)
    bem = _get_cached_bem(subjects_dir)

    # trans='fsaverage' uses an identity head→MRI transform.
    # Valid when the electrode montage is standard 10-20 and the template brain
    # is fsaverage — both are defined in the same head coordinate convention.
    #
    # mindist=5 mm: exclude dipoles within 5 mm of the inner skull surface.
    # Dipoles very close to the BEM boundary have poorly conditioned lead fields
    # that create artefactual high-amplitude sources.
    fwd = mne.make_forward_solution(
        info,
        trans='fsaverage',
        src=src,
        bem=bem,
        meg=False,
        eeg=True,
        mindist=5.0,
        verbose=False,
    )

    # Convert to surface-oriented free orientation.
    # The matrix is now reordered so component 0 at each vertex is the cortical
    # surface normal direction — the dominant component for EEG.
    fwd = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=False, verbose=False,
    )

    # FIFO eviction when the cache is full.
    if len(_FWD_CACHE) >= _FWD_CACHE_MAX:
        evicted = next(iter(_FWD_CACHE))
        _FWD_CACHE.pop(evicted)
        logger.debug("Evicted forward cache entry: %s", evicted)

    _FWD_CACHE[cache_key] = fwd
    return fwd


# ── Noise covariance and inverse operator ─────────────────────────────────────

def _build_inverse_operator(
    raw:          mne.io.BaseRaw,
    fwd:          object,
    analysis_win: tuple[float, float],
    snr:          float,
) -> object:
    """
    Estimate the noise covariance matrix and assemble the inverse operator.

    NOISE COVARIANCE  C_n  ∈ ℝ^{n_ch × n_ch}
    ─────────────────────────────────────────
    C_n characterises the background sensor noise (amplifier thermal noise,
    muscle artefacts, environmental interference).  It appears in the Wiener
    filter W = K·Lᵀ·(L·K·Lᵀ + λ²·C_n)⁻¹ (equation 2 in module docstring).

    Ideal C_n comes from an eyes-open/eyes-closed baseline or an empty-room
    recording.  In the absence of a dedicated baseline we use the pre-analysis
    window:
        • tmin=0 to the start of the analysis window → "pre-stimulus" window.
        • If the analysis window starts at t=0, fall back to an ad-hoc
          diagonal covariance (equal noise assumed on all channels).

    Ad-hoc cov uses MNE's per-channel-type noise floors (stored in defaults).
    It is less accurate than empirical cov but avoids spurious artefacts from
    estimating a high-dimensional covariance matrix from few samples.

    INVERSE OPERATOR PARAMETERS
    ──────────────────────────────
    loose = 0.2
        Partial orientation constraint: dipoles are allowed 20% tangential
        component relative to the surface normal.  loose=0 fixes orientations
        to the normal (fewer parameters, better-conditioned but less flexible);
        loose=1 gives fully free orientation (over-parameterised for EEG).

    depth = 0.8
        Depth-weighting exponent applied to K (the source prior).
        The lead-field entries L[i,j] decay as 1/r² with source depth r.
        Without depth-weighting, superficial sources dominate the minimum-norm
        estimate even when the true generator is deep.  depth=0.8 partially
        compensates by up-weighting deep sources in the prior K.
    """
    tmin_win, tmax_win = analysis_win

    # Use everything before the analysis window as the noise baseline.
    baseline_end = min(tmin_win, raw.times[-1])
    if baseline_end > 1.0:
        # Empirical covariance from the baseline segment.
        # method='empirical' is the fastest (sample covariance, no shrinkage).
        # For small n_ch or few samples, 'shrunk' or 'oas' are better estimators.
        logger.info(
            "Computing empirical noise covariance from t=[0, %.2f] s…", baseline_end,
        )
        noise_cov = mne.compute_raw_covariance(
            raw,
            tmin=0.0,
            tmax=baseline_end,
            method='empirical',
            verbose=False,
        )
    else:
        # No baseline available — use the ad-hoc diagonal approximation.
        logger.info("No pre-analysis baseline — using ad-hoc diagonal noise covariance.")
        noise_cov = mne.make_ad_hoc_cov(raw.info, verbose=False)

    # Assemble the regularised inverse operator.
    # MNE pre-computes and stores the Wiener filter matrices W (eq. 2) along
    # with the dSPM/sLORETA normalisation diagonal vectors (eq. 4a/4b) for
    # all three methods simultaneously — the choice of method is made later
    # at apply_inverse() time without re-solving the linear system.
    inv = mne.minimum_norm.make_inverse_operator(
        raw.info,
        fwd,
        noise_cov,
        loose=0.2,
        depth=0.8,
        verbose=False,
    )

    return inv


# ── Apply inverse and extract peak sources ────────────────────────────────────

def _vertex_to_mni_safe(
    vertices:     np.ndarray,
    hemi:         int,
    subjects_dir: str,
) -> np.ndarray:
    """
    Convert vertex indices to MNI coordinates (mm) with a NumPy fallback.

    mne.vertex_to_mni() applies the talairach.xfm affine (the 12-parameter
    transform stored in fsaverage/mri/transforms/talairach.xfm) to map from
    the MRI surface RAS frame (metres) to MNI 152 space (mm).

    If the transform file is unavailable (rare), we fall back to multiplying
    the raw surface positions by 1000 m→mm.  For fsaverage this is already
    approximately MNI space, with errors < 3 mm.
    """
    try:
        return mne.vertex_to_mni(
            vertices,
            hemis=hemi,
            subject='fsaverage',
            subjects_dir=subjects_dir,
            verbose=False,
        )
    except Exception as exc:
        logger.warning(
            "vertex_to_mni failed (%s) — using raw surface positions × 1000 as fallback.",
            exc,
        )
        # _SRC_CACHE must already exist because the forward solution was computed.
        spacing = next(iter(_SRC_CACHE))
        src     = _SRC_CACHE[spacing]
        pos_m   = src[hemi]['rr'][vertices]   # (n, 3) metres
        return pos_m * 1000.0                  # → mm (approximately MNI)


def _greedy_nms(
    mni_coords:  np.ndarray,    # (n_sources, 3) MNI mm
    amplitudes:  np.ndarray,    # (n_sources,)
    hemisphere:  list[str],     # "lh" or "rh" per source
    n_peaks:     int,
    min_dist_mm: float,
) -> list[SourcePeak]:
    """
    Select up to `n_peaks` spatially separated activation peaks.

    Algorithm: greedy non-maximum suppression (NMS)
    ─────────────────────────────────────────────────
    1. Sort sources by amplitude A[i] = mean_t(|Ĵ[i,t]|) in descending order.
    2. Maintain a list S of accepted peak coordinates.
    3. For each candidate source i (in sorted order):
         Accept if min_{j ∈ S} ‖r_i − r_j‖₂ > min_dist_mm
    4. Stop when |S| = n_peaks.

    Euclidean distance is computed in MNI mm space, which is isotropic and
    metrically meaningful for inter-source spacing comparisons.

    Parameters
    ──────────
    min_dist_mm : minimum separation between accepted peaks [mm].
                  ~20 mm corresponds to ~5 cortical source vertices at oct6
                  spacing and roughly matches the EEG spatial resolution.
    """
    sorted_idx = np.argsort(amplitudes)[::-1]   # highest amplitude first

    accepted_coords: list[np.ndarray] = []
    peaks:           list[SourcePeak] = []

    for idx in sorted_idx:
        if len(peaks) >= n_peaks:
            break

        coord = mni_coords[idx]    # (3,) MNI mm

        # Reject if too close to any previously accepted peak.
        if accepted_coords:
            existing = np.array(accepted_coords)                    # (k, 3)
            dists    = np.linalg.norm(existing - coord[np.newaxis], axis=1)
            if dists.min() < min_dist_mm:
                continue

        accepted_coords.append(coord)
        peaks.append(SourcePeak(
            x=float(coord[0]),
            y=float(coord[1]),
            z=float(coord[2]),
            amplitude=float(amplitudes[idx]),
            hemisphere=hemisphere[int(idx)],
        ))

    return peaks


def _apply_inverse_and_get_peaks(
    raw:          mne.io.BaseRaw,
    inv:          object,
    subjects_dir: str,
    tmin:         float,
    tmax:         float,
    method:       str,
    snr:          float,
    n_peaks:      int,
    min_dist_mm:  float,
) -> tuple[list[SourcePeak], int]:
    """
    Apply the inverse operator to the EEG segment and extract peak sources.

    APPLYING THE INVERSE  (equation 3: Ĵ = W · M)
    ─────────────────────────────────────────────────
    MNE builds an EvokedArray from the raw segment and calls the
    pre-assembled Wiener filter W internally.  The output is a
    SourceTimeCourse object:

        stc.data  ∈ ℝ^{n_src × n_t}    — source amplitudes over time
        stc.vertices  — [lh_vertex_indices, rh_vertex_indices]
        stc.times — time axis in seconds

    The amplitude at each source is summarised by its time-mean absolute
    value over the requested window:

        A[i]  =  (1/T) ∑_t |Ĵ[i, t]|                                  (5)

    Using |·| rather than the raw signed value is appropriate because the
    polarity of a dipole's reconstructed amplitude depends on its arbitrary
    normal-vector sign convention, which can differ between hemispheres.

    Parameters
    ──────────
    snr         : signal-to-noise ratio used for regularisation λ² = 1/SNR²
    method      : "dSPM", "sLORETA", or "MNE" — selects the normalisation
    """
    # Clamp the time window to the recording duration.
    t0 = float(np.clip(tmin, raw.times[0], raw.times[-1]))
    t1 = float(np.clip(tmax, raw.times[0], raw.times[-1]))
    if t0 >= t1:
        raise ValueError(f"tmin={t0:.3f} s must be less than tmax={t1:.3f} s.")

    # Extract the segment of interest as an EvokedArray.
    # EvokedArray wraps any (n_ch, n_t) numpy array as an MNE Evoked object.
    # We pick only channels present in the forward solution so the dimensions
    # of M (eq. 1) match those of L (eq. 2).
    inv_ch_names = [inv['info']['ch_names'][i] for i in range(len(inv['info']['ch_names']))]
    raw_seg      = raw.copy().crop(tmin=t0, tmax=t1)
    raw_seg      = raw_seg.pick_channels(inv_ch_names, ordered=True)

    data    = raw_seg.get_data()                        # (n_ch, n_t)
    evoked  = mne.EvokedArray(data, raw_seg.info, tmin=t0, verbose=False)

    # λ² = 1/SNR² — the regularisation parameter in the Wiener filter (eq. 2).
    # Smaller λ² (higher SNR) → less regularisation → sharper but noisier maps.
    # Larger λ² (lower SNR)   → more regularisation → smoother, deeper maps.
    lambda2 = 1.0 / snr ** 2

    # apply_inverse computes Ĵ = W·M and applies the chosen normalisation:
    #   dSPM    → divide by √([W·C_n·Wᵀ][i,i])           eq. 4a
    #   sLORETA → divide by √([W·C_data·Wᵀ][i,i])        eq. 4b
    #   MNE     → no normalisation, return Ĵ in A·m
    # pick_ori=None collapses the 3 dipole components to a scalar amplitude
    # by taking the vector norm — appropriate for visualisation.
    stc = mne.minimum_norm.apply_inverse(
        evoked, inv,
        lambda2=lambda2,
        method=method,
        pick_ori=None,
        verbose=False,
    )
    # stc.data : (n_src_total, n_t)   n_src_total = n_lh_used + n_rh_used

    # ── Time-average the absolute amplitude (equation 5) ──────────────────────
    amplitudes = np.mean(np.abs(stc.data), axis=1)   # (n_src_total,)
    n_total    = int(len(amplitudes))

    # ── Convert vertex indices to MNI coordinates (mm) ────────────────────────
    # stc.vertices[0] — indices into the LH source space
    # stc.vertices[1] — indices into the RH source space
    # vertex_to_mni() applies the talairach.xfm affine stored in fsaverage.
    n_lh     = len(stc.vertices[0])
    lh_mni   = _vertex_to_mni_safe(stc.vertices[0], hemi=0, subjects_dir=subjects_dir)
    rh_mni   = _vertex_to_mni_safe(stc.vertices[1], hemi=1, subjects_dir=subjects_dir)
    all_mni  = np.vstack([lh_mni, rh_mni])           # (n_total, 3) mm

    # Build hemisphere label list matching the stc source ordering.
    hemi_labels = ['lh'] * n_lh + ['rh'] * len(stc.vertices[1])

    # ── Greedy non-maximum suppression → top N peaks ─────────────────────────
    peaks = _greedy_nms(all_mni, amplitudes, hemi_labels, n_peaks, min_dist_mm)

    return peaks, n_total


# ── Orchestrating pipeline (runs in thread pool) ──────────────────────────────

def _run_pipeline(
    vhdr_bytes:  bytes,
    eeg_bytes:   bytes,
    vmrk_bytes:  bytes | None,
    vhdr_name:   str,
    eeg_name:    str,
    vmrk_name:   str | None,
    tmin:        float,
    tmax:        float,
    method:      str,
    snr:         float,
    n_peaks:     int,
    min_dist_mm: float,
    spacing:     str,
) -> LocalizeResponse:
    """
    Full synchronous pipeline: file loading → geometry → inverse → peaks.

    Called inside asyncio.to_thread() so the FastAPI event loop is not blocked.
    All heavy numpy/MNE operations execute here in a worker thread.
    """
    t_wall = time.perf_counter()
    tmpdir = None

    try:
        # ── 1. Load raw EEG ─────────────────────────────────────────────────
        raw, tmpdir = _load_brainvision(
            vhdr_bytes, eeg_bytes, vmrk_bytes, vhdr_name, eeg_name, vmrk_name,
        )
        raw = _ensure_eeg_positions(raw)

        n_eeg = len(mne.pick_types(raw.info, eeg=True))
        if n_eeg < 5:
            raise ValueError(
                f"Only {n_eeg} EEG channel(s) with known positions. "
                "Need ≥ 5 to compute a meaningful forward solution."
            )

        # ── 2. fsaverage geometry ────────────────────────────────────────────
        subjects_dir = _init_subjects_dir()

        # ── 3. Forward solution (cached by channel layout) ───────────────────
        fwd = _get_or_compute_forward(raw.info, subjects_dir, spacing)

        # ── 4. Noise covariance + inverse operator ───────────────────────────
        inv = _build_inverse_operator(raw, fwd, (tmin, tmax), snr)

        # ── 5. Apply inverse + extract peaks ─────────────────────────────────
        peaks, n_sources = _apply_inverse_and_get_peaks(
            raw, inv, subjects_dir, tmin, tmax, method, snr, n_peaks, min_dist_mm,
        )

    finally:
        # Always clean up the temp directory that holds the uploaded files.
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    elapsed_ms = (time.perf_counter() - t_wall) * 1000.0
    logger.info(
        "Localisation done: method=%s, %d peaks from %d sources, %.1f ms",
        method, len(peaks), n_sources, elapsed_ms,
    )

    return LocalizeResponse(
        method=method,
        n_peaks=len(peaks),
        n_sources=n_sources,
        peaks=peaks,
        tmin=tmin,
        tmax=tmax,
        duration_ms=round(elapsed_ms, 1),
    )


# ── FastAPI router ─────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/eeg", tags=["Source Localisation"])


@router.post("/localize", response_model=LocalizeResponse)
async def localize_sources(
    # ── Required: BrainVision file triplet ───────────────────────────────────
    vhdr_file: UploadFile = File(
        ...,
        description="BrainVision header file (.vhdr). Contains channel names, "
                    "sampling rate, and references to the data and marker files.",
    ),
    eeg_file: UploadFile = File(
        ...,
        description="BrainVision binary data file (.eeg). Raw int16/float32 samples.",
    ),
    vmrk_file: UploadFile = File(
        None,
        description="BrainVision marker file (.vmrk, optional). Needed only if "
                    "the .vhdr references markers for baseline computation.",
    ),
    # ── Analysis parameters ───────────────────────────────────────────────────
    tmin: float = Form(
        default=0.0,
        description="Analysis window start in seconds from recording onset.",
    ),
    tmax: float = Form(
        default=1.0,
        description="Analysis window end in seconds from recording onset.",
    ),
    method: Literal["dSPM", "sLORETA", "MNE"] = Form(
        default="dSPM",
        description=(
            "Inverse method.\n"
            "dSPM    — z-score relative to noise floor; best for detecting active regions.\n"
            "sLORETA — zero-error localisation for point sources (noise-free).\n"
            "MNE     — raw amplitude in A·m; best for comparing magnitudes."
        ),
    ),
    snr: float = Form(
        default=3.0,
        description=(
            "Assumed signal-to-noise ratio. Regularisation λ² = 1/SNR².\n"
            "Use 3.0 for averaged evoked data, 1.0 for single-trial raw data."
        ),
    ),
    n_peaks: int = Form(
        default=10,
        ge=1, le=100,
        description="Number of spatially separated peak sources to return.",
    ),
    min_dist_mm: float = Form(
        default=20.0,
        ge=0.0,
        description=(
            "Minimum MNI-space distance (mm) between returned peaks. "
            "Prevents adjacent vertices from all appearing as separate peaks. "
            "~20 mm ≈ EEG spatial resolution with standard 10-20 montage."
        ),
    ),
    spacing: Literal["oct5", "oct6", "ico4", "ico5"] = Form(
        default="oct6",
        description=(
            "Cortical source-space density.\n"
            "oct6 (~8192 sources, ~4 mm) is the recommended default.\n"
            "oct5 (~2048 sources, ~6 mm) is 4× faster but lower spatial resolution."
        ),
    ),
) -> LocalizeResponse:
    """
    Solve the EEG electromagnetic inverse problem and return peak activation
    coordinates in MNI space.

    **Upload the three BrainVision files** (.vhdr, .eeg, .vmrk) as multipart
    form-data.  The header and data files are mandatory; the marker file is
    needed only if a pre-stimulus baseline is present for noise covariance
    estimation.

    **Pipeline (runs off-thread to avoid blocking the server):**
    1. Write files to a temporary directory and load with MNE.
    2. Assign standard 10-20 electrode positions if absent.
    3. Set up the cortical source space on the fsaverage template MRI.
    4. Load the precomputed 3-shell BEM solution.
    5. Compute (or retrieve from cache) the lead-field matrix **L**.
    6. Estimate the noise covariance matrix **C_n** from the pre-analysis window.
    7. Build the regularised inverse operator **W** = K·Lᵀ·(L·K·Lᵀ + λ²C_n)⁻¹.
    8. Apply **W** to the requested time segment → source time-course **Ĵ**.
    9. Greedy NMS selects `n_peaks` spatially separated peak sources.
    10. Convert peak vertex indices → MNI coordinates via `mne.vertex_to_mni`.

    **Typical compute times (first call per process):**
    - fsaverage download : ~30 s  (one-time, network-dependent)
    - Forward solution   : ~30–90 s  (cached for repeated calls with same cap)
    - Inverse + peaks    : < 5 s

    **Returns** `peaks`: a list of `{"x", "y", "z", "amplitude", "hemisphere"}`
    dicts in MNI 152 mm space, ready for `vtkPoints` / point-cloud rendering.
    """
    # Read all file bytes upfront so the UploadFile handles are released before
    # we enter the thread pool (UploadFile is not thread-safe).
    vhdr_bytes = await vhdr_file.read()
    eeg_bytes  = await eeg_file.read()
    vmrk_bytes = (await vmrk_file.read()) if vmrk_file else None

    # Validate that the uploaded files have the correct extensions.
    def _ext(filename: str | None) -> str:
        return os.path.splitext(filename or '')[-1].lower()

    if _ext(vhdr_file.filename) != '.vhdr':
        raise HTTPException(
            status_code=422,
            detail=f"vhdr_file must be a .vhdr file, got '{vhdr_file.filename}'.",
        )
    if _ext(eeg_file.filename) not in ('.eeg', '.dat'):
        raise HTTPException(
            status_code=422,
            detail=f"eeg_file must be a .eeg/.dat file, got '{eeg_file.filename}'.",
        )

    try:
        result = await asyncio.to_thread(
            _run_pipeline,
            vhdr_bytes,
            eeg_bytes,
            vmrk_bytes,
            vhdr_file.filename or 'upload.vhdr',
            eeg_file.filename  or 'upload.eeg',
            vmrk_file.filename if vmrk_file else None,
            tmin,
            tmax,
            method,
            snr,
            n_peaks,
            min_dist_mm,
            spacing,
        )
        return result
    except ValueError as exc:
        # Client errors: bad time range, too few channels, etc.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Source localisation pipeline failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
