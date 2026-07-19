"""
source_localization.py — Cortical source localisation via MNE-Python
═════════════════════════════════════════════════════════════════════════════

Solves the electromagnetic inverse problem for an already-loaded EEG session,
mapping scalp-sensor potentials back to their 3-D cortical origins.

MATHEMATICAL OVERVIEW
──────────────────────
The forward model (lead-field matrix L) relates unknown source currents J to
the measured sensor signals M:

    M  =  L · J  +  ε                              (1)
    M  : (n_channels, n_times)   — observed EEG potentials
    L  : (n_channels, n_sources) — lead-field (computed analytically via BEM)
    J  : (n_sources,  n_times)   — unknown dipole currents  [A·m]
    ε  : sensor noise (assumed Gaussian, characterised by noise covariance C_n)

Recovering J from M is ill-posed because n_sources ≫ n_channels.
Tikhonov-regularised minimum-norm families add a penalty on ||J||:

    Ĵ = argmin  ||M − L·J||²_{C_n^{-1}}  +  λ²·||J||²

The three supported inverse methods differ only in how they post-normalise Ĵ:

  MNE    — raw minimum-norm estimate (unit: A·m).
           Ĵ_MNE = (L^T C_n^{-1} L + λ² I)^{-1} L^T C_n^{-1} M

  dSPM   — dynamic Statistical Parametric Mapping (Dale et al., 2000).
           Divides each source estimate by the square-root of its expected
           noise variance → dimensionless z-score.
           z_i = Ĵ_i / sqrt([L·C_n·L^T]_{ii})
           Best for detecting which regions are significantly activated.

  sLORETA — standardised Low-Resolution Electromagnetic Tomography
           (Pascual-Marqui, 2002).  Normalises by the estimated source
           covariance → zero localisation error for point sources (noise-free).

REGULARISATION
───────────────
λ² = 1 / SNR²

SNR is the assumed signal-to-noise ratio of the averaged evoked response.
Typical values: 3.0 (averaged data, high SNR), 1.0 (single-trial, low SNR).
Lower SNR → stronger regularisation → smoother, more conservative maps.

VOLUME CONDUCTOR MODEL (BEM)
──────────────────────────────
A three-shell Boundary Element Model (BEM) separates the volume into:
  1. Brain (inner skull surface)
  2. Skull (outer skull surface)
  3. Scalp
Each shell has homogeneous conductivity.  The lead-field is integrated over
the shell surfaces using the linear collocation BEM method.  The fsaverage
template ships with a precomputed 5120-triangle BEM solution.

COORDINATE SYSTEM
──────────────────
All returned positions are in the MRI/surface RAS frame, in metres.
This matches the vtk.js world-space convention already used by the MRI plugin
(NIfTI affine is in mm; divide by 1000 to get metres).

OUTPUT (consumed by the vtk.js frontend)
──────────────────────────────────────────
{
  "method":       "dSPM",
  "n_sources":    8192,
  "positions":    [[x, y, z], ...],   // metres, MRI RAS — for vtkPoints
  "amplitudes":   [float, ...],        // time-averaged |activation|
  "amplitude_max": float,              // scalar max — for LUT colormap range
  "hemisphere":   ["lh",...,"rh",...], // which hemisphere each source is in
  "times":        [float, ...],        // seconds — full STC time axis
  "duration_ms":  float
}

The vtk.js point-cloud renderer ingests `positions` as vtkPoints and
`amplitudes` as a vtkDataArray for colour-mapping (jet / viridis LUT).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Literal

import mne
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Shared session store — eeg_sessions dict and require_eeg() helper.
# require_eeg() raises HTTP 404 automatically when a session is missing.
from session_store import require_eeg

logger = logging.getLogger(__name__)

# ── Module-level fsaverage cache ──────────────────────────────────────────────
#
# mne.datasets.fetch_fsaverage() downloads ~50 MB of template MRI data once
# and returns the local subjects_dir.  Subsequent calls are instant (disk
# check only).  We cache the path so _get_subjects_dir() is cheap after the
# first call even if MNE itself doesn't short-circuit internally.

_SUBJECTS_DIR: str | None = None


def _get_subjects_dir() -> str:
    """
    Return the fsaverage subjects_dir, downloading once per process.

    The download is idempotent — MNE checks the target directory before
    fetching.  Thread-safe: Python's GIL protects the module-level assignment.
    """
    global _SUBJECTS_DIR
    if _SUBJECTS_DIR is None:
        logger.info("Fetching fsaverage template MRI (one-time ~50 MB download)…")
        _SUBJECTS_DIR = mne.datasets.fetch_fsaverage(verbose=False)
    return _SUBJECTS_DIR


# ── Pydantic request / response models ────────────────────────────────────────

class LocalizeRequest(BaseModel):
    """Parameters for one source-localisation job."""

    session_id: str
    """ID of a loaded EEG session (returned by POST /api/load-eeg)."""

    method: Literal["dSPM", "sLORETA", "MNE"] = "dSPM"
    """
    Inverse method.
    dSPM / sLORETA → dimensionless z-score maps, good for localising activity.
    MNE             → physical unit (A·m), good for comparing amplitudes.
    """

    snr: float = Field(default=3.0, gt=0.0, description="Assumed signal-to-noise ratio.  λ² = 1/SNR².")
    """
    Higher SNR → less regularisation → sharper but noisier maps.
    Use 3.0 for clean averaged evokeds, 1.0 for single-trial raw data.
    """

    tmin: float = Field(default=0.0, description="Analysis window start (s from recording onset).")
    tmax: float = Field(default=1.0, description="Analysis window end (s from recording onset).")

    spacing: Literal["oct5", "oct6", "ico4", "ico5"] = "oct6"
    """
    Source-space density:
      oct5 → ~2048 sources (~6 mm spacing)  — faster
      oct6 → ~8192 sources (~4 mm spacing)  — default, good trade-off
      ico5 → ~20484 sources (~2 mm spacing) — high-res, slow
    """


class LocalizeResponse(BaseModel):
    """Serialised source-localisation result for the vtk.js point cloud."""

    method:        str
    n_sources:     int
    positions:     list[list[float]]   # [[x, y, z], ...] in metres
    amplitudes:    list[float]         # time-averaged |activation| per source
    amplitude_max: float               # max amplitude (LUT upper bound)
    hemisphere:    list[str]           # "lh" or "rh" for each source
    times:         list[float]         # seconds — full STC time axis
    duration_ms:   float


# ── Private helpers ────────────────────────────────────────────────────────────

def _pick_eeg_and_set_montage(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Return a copy of `raw` containing only EEG channels, with positions set.

    If the recording already contains digitisation points (e.g. from a
    BrainVision file exported with electrode coordinates), they are preserved.
    Otherwise the standard 10-20 montage is applied by channel-name matching.
    Channels not in the 10-20 layout are kept but will have no position and
    are therefore excluded from the forward solution by make_forward_solution.

    A copy is taken so we never mutate the session's stored Raw object.
    """
    raw = raw.copy().pick_types(eeg=True, meg=False, stim=False, exclude='bads')

    # Check whether any EEG channel already has a non-zero Cartesian position.
    eeg_idxs    = mne.pick_types(raw.info, eeg=True)
    has_loc     = any(
        np.any(raw.info['chs'][i]['loc'][:3] != 0) for i in eeg_idxs
    )

    if not has_loc:
        logger.info("EEG channel positions absent — applying standard_1020 montage.")
        montage = mne.channels.make_standard_montage('standard_1020')
        # match_case=False: tolerates mixed-case names ("Fp1" vs "FP1").
        # on_missing='warn': skip channels not in the 10-20 layout gracefully.
        raw.set_montage(montage, match_case=False, on_missing='warn')

    return raw


def _load_or_build_bem(subjects_dir: str, subject: str) -> object:
    """
    Return the fsaverage BEM solution, loading the precomputed FIF or building
    it from surfaces if the file does not exist on disk.

    The three-shell BEM solution ships with mne.datasets.fetch_fsaverage().
    Building from scratch (make_bem_model + make_bem_solution) takes ~2 min
    and is only needed on installations where the file was manually removed.
    """
    bem_path = os.path.join(
        subjects_dir, subject, 'bem',
        'fsaverage-5120-5120-5120-bem-sol.fif',
    )
    if os.path.exists(bem_path):
        return mne.read_bem_solution(bem_path, verbose=False)

    logger.warning(
        "Precomputed BEM solution not found at %s — rebuilding (may take ~2 min).",
        bem_path,
    )
    model = mne.make_bem_model(subject, ico=4, subjects_dir=subjects_dir, verbose=False)
    return mne.make_bem_solution(model, verbose=False)


# ── Core pipeline (runs in a thread pool) ─────────────────────────────────────

def _run_localization(req: LocalizeRequest) -> LocalizeResponse:
    """
    Execute the full MNE-Python source-localisation pipeline synchronously.

    This function is called inside asyncio.to_thread() so it never blocks the
    FastAPI event loop.  All heavy numpy and MNE-Python work happens here.

    Steps
    ─────
    1.  Load Raw from session store and isolate EEG with positions.
    2.  Set up cortical source space on fsaverage (oct6 → ~8192 sources).
    3.  Load the precomputed 3-shell BEM solution.
    4.  Compute the forward solution (lead-field matrix L).
    5.  Estimate noise covariance (ad-hoc diagonal — no separate baseline needed).
    6.  Build the Tikhonov-regularised inverse operator from L and C_n.
    7.  Create an EvokedArray from the requested time window of the raw recording.
    8.  Apply the chosen inverse method → SourceTimeCourse (STC).
    9.  Extract source positions (metres, MRI RAS) and time-averaged amplitudes.
    """
    t_wall = time.perf_counter()

    # ── 1. Session & EEG channels ────────────────────────────────────────────
    raw = require_eeg(req.session_id)
    raw = _pick_eeg_and_set_montage(raw)

    n_eeg = len(mne.pick_types(raw.info, eeg=True))
    if n_eeg < 5:
        raise ValueError(
            f"Only {n_eeg} EEG channel(s) with positions — need at least 5 "
            "to compute a meaningful forward solution."
        )

    # ── 2. fsaverage source space ────────────────────────────────────────────
    subjects_dir = _get_subjects_dir()
    subject      = 'fsaverage'

    # setup_source_space returns two hemispheres: src[0]=lh, src[1]=rh.
    # spacing='oct6' places one source at the centre of each face of a
    # recursively subdivided octahedron projected onto the cortical surface.
    src = mne.setup_source_space(
        subject, spacing=req.spacing, subjects_dir=subjects_dir, verbose=False,
    )

    # ── 3. BEM solution ──────────────────────────────────────────────────────
    bem = _load_or_build_bem(subjects_dir, subject)

    # ── 4. Forward solution ──────────────────────────────────────────────────
    # trans='fsaverage' applies the identity head→MRI transform.  This is
    # valid because a standard 10-20 montage defines electrode positions in
    # the same head coordinate frame that fsaverage uses.
    # mindist=5 mm: exclude sources very close to the inner skull to avoid
    # numerically unstable dipoles near the BEM boundary.
    fwd = mne.make_forward_solution(
        raw.info,
        trans='fsaverage',
        src=src,
        bem=bem,
        meg=False,
        eeg=True,
        mindist=5.0,
        verbose=False,
    )

    # Convert to surface orientation (free-orientation dipoles collapsed to
    # surface normal) for better-conditioned inversion.
    fwd = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=False, verbose=False,
    )

    # ── 5. Noise covariance (ad-hoc diagonal) ───────────────────────────────
    # make_ad_hoc_cov assumes equal, independent noise on every channel based
    # on the channel-type noise floor defined in the MNE defaults.  It avoids
    # the need for a separate empty-room or baseline recording.
    # For better accuracy, a real noise covariance from a pre-stimulus baseline
    # can be passed instead: mne.compute_covariance(epochs, tmax=0.0).
    noise_cov = mne.make_ad_hoc_cov(raw.info, verbose=False)

    # ── 6. Inverse operator ──────────────────────────────────────────────────
    # loose=0.2  → allow 20 % tangential component (partial orientation).
    # depth=0.8  → depth-weighting to correct the systematic bias toward
    #              superficial sources in minimum-norm estimates.
    inv = mne.minimum_norm.make_inverse_operator(
        raw.info, fwd, noise_cov,
        loose=0.2, depth=0.8, verbose=False,
    )

    # ── 7. Evoked from the analysis window ──────────────────────────────────
    # Clamp the requested window to the actual recording duration.
    t_min = float(np.clip(req.tmin, raw.times[0], raw.times[-1]))
    t_max = float(np.clip(req.tmax, raw.times[0], raw.times[-1]))
    if t_min >= t_max:
        raise ValueError(f"tmin ({t_min:.3f}s) must be less than tmax ({t_max:.3f}s).")

    # Crop and pick channels that are also present in the forward solution.
    # EvokedArray wraps any ndarray as an Evoked-compatible object — no
    # averaging across trials is required.
    raw_win  = raw.copy().crop(tmin=t_min, tmax=t_max)
    fwd_chs  = fwd['info']['ch_names']
    raw_win  = raw_win.pick_channels(fwd_chs, ordered=True)
    data     = raw_win.get_data()                         # (n_ch, n_times)
    evoked   = mne.EvokedArray(data, raw_win.info, tmin=t_min, verbose=False)

    # ── 8. Apply inverse ────────────────────────────────────────────────────
    lambda2 = 1.0 / req.snr ** 2   # regularisation: smaller λ² = sharper maps
    stc = mne.minimum_norm.apply_inverse(
        evoked, inv,
        lambda2=lambda2,
        method=req.method,
        pick_ori=None,   # collapse x/y/z dipole components → scalar amplitude
        verbose=False,
    )
    # stc.data : (n_sources, n_times)  where n_sources = n_lh + n_rh used vertices
    # stc.vertices : [lh_vertex_indices, rh_vertex_indices]

    # ── 9. Extract positions and amplitudes ─────────────────────────────────
    # src[0]['rr'] / src[1]['rr'] : all candidate vertex positions in metres.
    # stc.vertices[0/1]           : integer indices of the *used* vertices.
    lh_pos  = src[0]['rr'][stc.vertices[0]]          # (n_lh, 3) metres
    rh_pos  = src[1]['rr'][stc.vertices[1]]          # (n_rh, 3) metres
    all_pos = np.vstack([lh_pos, rh_pos])            # (n_total, 3)

    # Time-average the absolute amplitude for a single scalar per source.
    # np.abs handles signed MNE estimates; dSPM/sLORETA are already positive.
    amplitudes = np.mean(np.abs(stc.data), axis=1)  # (n_total,)

    # Hemisphere label for each source — lets the frontend colour by hemisphere.
    hemisphere = (
        ['lh'] * len(stc.vertices[0]) +
        ['rh'] * len(stc.vertices[1])
    )

    elapsed_ms = (time.perf_counter() - t_wall) * 1000.0
    logger.info(
        "Source localisation done: %d sources, method=%s, %.1f ms",
        len(amplitudes), req.method, elapsed_ms,
    )

    return LocalizeResponse(
        method=req.method,
        n_sources=int(len(amplitudes)),
        positions=all_pos.tolist(),
        amplitudes=amplitudes.tolist(),
        amplitude_max=float(amplitudes.max()),
        hemisphere=hemisphere,
        times=stc.times.tolist(),
        duration_ms=round(elapsed_ms, 1),
    )


# ── FastAPI router ─────────────────────────────────────────────────────────────

router = APIRouter(tags=["Source Localisation"])


@router.post("/api/eeg/localize-session", response_model=LocalizeResponse)
async def localize_sources(req: LocalizeRequest) -> LocalizeResponse:
    """
    Solve the EEG inverse problem for a previously-loaded EEG session.

    Runs the full MNE-Python pipeline (fsaverage source space + 3-shell BEM
    forward solution + dSPM / sLORETA / MNE inverse) inside a thread-pool
    executor so the FastAPI event loop is never blocked.

    **Typical wall-clock times (first call per process):**
    - fsaverage download  : ~30 s (network-dependent, one-time only)
    - setup_source_space  : ~5 s
    - make_forward_solution : ~20–60 s (depends on n_channels, n_sources)
    - apply_inverse       : < 1 s

    Subsequent calls (same process, different time window / method) skip
    the download and reuse the cached subjects_dir path, but recompute the
    forward solution each time.  A production system would cache fwd + inv
    per session_id.

    **Returns:**
    Source positions in MRI RAS metres and time-averaged activation amplitudes,
    ready for vtk.js point-cloud rendering or surface heatmap interpolation.
    """
    try:
        return await asyncio.to_thread(_run_localization, req)
    except ValueError as exc:
        # Client-side errors: bad session_id, too few channels, bad tmin/tmax.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Source localisation failed unexpectedly.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
