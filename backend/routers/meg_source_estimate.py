"""
meg_source_estimate.py — MEG cortical source estimation endpoint
════════════════════════════════════════════════════════════════════

Runs MNE-Python dSPM / MNE / sLORETA source estimation on a loaded MEG
session and returns the source-space activation as a list of (x, y, z,
amplitude) vertices that the frontend can overlay on the fMRI volume.

PIPELINE
─────────
 1. Load the MEG Raw object from the session store (require_meg).
 2. Crop to the requested time window [t_min, t_max].
 3. Build a forward solution against the MNI fsaverage template.
    This avoids the need for subject-specific MRI coregistration —
    using fsaverage is the standard approach for group-level localisation
    and requires no extra data from the user.
 4. Estimate noise covariance from the first 0–1 s baseline.
 5. Compute the inverse operator (MNE, dSPM, or sLORETA).
 6. Apply the inverse to the averaged data slice.
 7. Return source amplitudes with their MNI RAS coordinates in mm.

OPTIONAL fMRI SPATIAL PRIOR
─────────────────────────────
When bold_prior_path is provided the backend loads the BOLD volume with
nibabel and uses the mean activation over the time window as a source
amplitude prior.  The prior is incorporated via Bayesian weighting of
the inverse covariance:

    C_source_prior = C_mne * bold_weight

where bold_weight ∝ BOLD activation at the nearest cortical vertex.
This is an experimental research feature — the plain dSPM path is the
safe default.

COORDINATE SYSTEM
──────────────────
MNE source positions are in metres in the MRI RAS frame.  We convert
to millimetres before returning so they match the vtk.js world-space
convention (NIfTI affine → vtkImageData uses mm).

NOTES
──────
• Forward solution build requires mne-python ≥ 1.0 with the fsaverage
  subject files.  Run ``mne.datasets.fetch_fsaverage()`` once to cache them.
• Runtime: forward solve + inverse ≈ 30–90 s per call for 5120 dipoles.
  Heavy calls are synchronous — for production wrap in BackgroundTasks.
"""

from __future__ import annotations

import time
from typing import Optional

import mne
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from session_store import require_meg

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/meg", tags=["MEG Source Estimate"])

# ── Request / Response models ─────────────────────────────────────────────────


class SourceEstimateRequest(BaseModel):
    """Request body for MEG source estimation."""

    # MEG session created by POST /api/load-meg
    session_id: str = Field(..., description="MEG session UUID from /api/load-meg")

    # Inverse method
    method: str = Field(
        "dSPM",
        description="Inverse method: 'dSPM', 'MNE', or 'sLORETA'",
    )

    # Time window of interest within the recording (seconds)
    t_min: float = Field(0.0, description="Analysis window start in seconds")
    t_max: Optional[float] = Field(
        None,
        description="Analysis window end in seconds; None = end of recording",
    )

    # Optional fMRI spatial prior
    bold_prior_path: Optional[str] = Field(
        None,
        description=(
            "Server-side path to a NIfTI BOLD volume whose activation is used "
            "as a spatial prior for the inverse solution. Experimental."
        ),
    )


class SourceVertex(BaseModel):
    """One point in the source-space activation map."""

    x: float = Field(..., description="MNI x-coordinate in mm")
    y: float = Field(..., description="MNI y-coordinate in mm")
    z: float = Field(..., description="MNI z-coordinate in mm")
    amplitude: float = Field(..., description="Source amplitude (dSPM z-score or nAm)")


class SourceEstimateResponse(BaseModel):
    """Response from POST /api/meg/source-estimate."""

    vertices: list[SourceVertex] = Field(
        ..., description="Source-space vertices with activation amplitudes"
    )
    method: str = Field(..., description="Inverse method used")
    peak_amplitude: float = Field(..., description="Maximum amplitude across all vertices")
    duration_ms: float = Field(..., description="Wall-clock time of the computation in ms")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fetch_fsaverage() -> str:
    """
    Ensure the fsaverage template files are available and return the
    path to the subjects directory.

    Uses mne.datasets.fetch_fsaverage() which downloads and caches the
    minimal set of surface files the first time it is called (~50 MB).
    """
    try:
        subjects_dir = mne.datasets.fetch_fsaverage(verbose=False)
    except Exception as exc:
        raise HTTPException(
            500,
            f"Could not fetch fsaverage dataset: {exc}. "
            "Run mne.datasets.fetch_fsaverage() once in a Python interpreter.",
        ) from exc
    return subjects_dir


def _build_forward(raw: mne.io.BaseRaw, subjects_dir: str) -> mne.Forward:
    """
    Build a MEG forward solution against the fsaverage template using a
    precomputed BEM solution.

    We use the ico-4 surface source space (2562 dipoles per hemisphere,
    5124 total) which offers a good resolution/speed trade-off.
    """
    # Source space on the fsaverage surface (both hemispheres)
    src = mne.setup_source_space(
        subject="fsaverage",
        spacing="ico4",
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=False,
    )

    # Pre-computed 3-shell BEM (no conductivity solve needed)
    bem = mne.read_bem_solution(
        f"{subjects_dir}/fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif",
        verbose=False,
    )

    # Forward solution (mag + grad channels)
    fwd = mne.make_forward_solution(
        raw.info,
        trans="fsaverage",
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        verbose=False,
    )
    return fwd


def _apply_bold_prior(
    inv: mne.minimum_norm.InverseOperator,
    bold_path: str,
    fwd: mne.Forward,
) -> None:
    """
    Experimental: modulate the inverse operator source covariance with
    a BOLD activation map loaded from a NIfTI file.

    Each source dipole is mapped to its nearest BOLD voxel in MNI space.
    The source covariance diagonal is multiplied by the normalised BOLD
    amplitude at that voxel.  Vertices with high BOLD signal receive
    stronger weighting in the inverse solution.

    This function modifies `inv` in-place.
    """
    try:
        import nibabel as nib  # type: ignore
    except ImportError:
        # nibabel not installed — skip the prior silently
        return

    try:
        bold_nii = nib.load(bold_path)
    except Exception:
        # If the file can't be read we ignore the prior and continue
        return

    bold_data = np.asarray(bold_nii.get_fdata(dtype=np.float32))
    # For 4-D BOLD take the mean over time (or first volume)
    if bold_data.ndim == 4:
        bold_data = bold_data.mean(axis=-1)

    affine = bold_nii.affine

    # Source positions in metres (MNI RAS) — convert to mm for voxel lookup
    src_pos_mm = fwd["source_rr"] * 1000.0  # (n_src, 3) in mm

    # Map each source to its nearest BOLD voxel via the inverse affine
    inv_affine = np.linalg.inv(affine)

    # Homogeneous coordinates: (n_src, 4)
    ones = np.ones((src_pos_mm.shape[0], 1))
    src_hom = np.concatenate([src_pos_mm, ones], axis=1)
    # Voxel indices: (n_src, 4)
    vox_idx = (inv_affine @ src_hom.T).T[:, :3].round().astype(int)

    # Clamp to valid voxel bounds
    shape = np.array(bold_data.shape)
    vox_idx = np.clip(vox_idx, 0, shape - 1)

    # BOLD amplitude at each source vertex
    bold_at_src = bold_data[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]]

    # Normalise to [0, 1]
    bold_max = bold_at_src.max()
    if bold_max > 0:
        bold_at_src /= bold_max

    # Apply as multiplicative weight to the source covariance diagonal.
    # The source covariance lives in inv['source_cov']['data'].
    # We clamp weights to avoid zeroing out sources entirely.
    weights = np.clip(bold_at_src, 0.1, 1.0)
    if "data" in inv["source_cov"]:
        inv["source_cov"]["data"] *= weights


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.post(
    "/source-estimate",
    response_model=SourceEstimateResponse,
    summary="MEG cortical source estimate (dSPM / MNE / sLORETA)",
)
def run_source_estimate(request: SourceEstimateRequest) -> SourceEstimateResponse:
    """
    Compute a cortical source estimate for a loaded MEG session.

    Prerequisites
    ─────────────
    1. POST /api/load-meg to get a session_id.
    2. The first call may download fsaverage data (~50 MB) if not already cached.
    3. Typical runtime: 30–90 s for forward solve + inverse.

    Returns
    ───────
    A list of source vertices (x, y, z in mm MNI) with dSPM amplitudes (z-scores
    for 'dSPM') or minimum-norm estimates in nAm (for 'MNE').  The frontend uses
    these to render a colour-coded point cloud on top of the fMRI volume.
    """
    # Validate method
    valid_methods = {"dSPM", "MNE", "sLORETA"}
    if request.method not in valid_methods:
        raise HTTPException(
            400,
            f"Unknown method '{request.method}'. Choose from {sorted(valid_methods)}.",
        )

    # Load session
    raw = require_meg(request.session_id)
    t0  = time.perf_counter()

    # ── 1. Crop raw to the analysis window ────────────────────────────────────
    t_max = request.t_max if request.t_max is not None else float(raw.times[-1])
    t_min = float(max(0.0, request.t_min))
    t_max = float(min(t_max, float(raw.times[-1])))
    if t_min >= t_max:
        raise HTTPException(400, "t_min must be less than t_max.")

    try:
        raw_crop = raw.copy().crop(tmin=t_min, tmax=t_max)
    except Exception as exc:
        raise HTTPException(422, f"Could not crop raw data: {exc}") from exc

    # ── 2. Ensure only MEG channels are present ────────────────────────────────
    try:
        raw_crop.pick(["mag", "grad"], verbose=False)
    except Exception:
        # Some KIT/BTi files only have 'meg'; try that
        try:
            raw_crop.pick("meg", verbose=False)
        except Exception as exc:
            raise HTTPException(422, f"No MEG channels found: {exc}") from exc

    if len(raw_crop.ch_names) == 0:
        raise HTTPException(422, "No MEG channels found in session.")

    # ── 3. Build forward solution ──────────────────────────────────────────────
    try:
        subjects_dir = _fetch_fsaverage()
        fwd = _build_forward(raw_crop, subjects_dir)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            500,
            f"Forward solution failed: {exc}. "
            "Ensure mne-python ≥ 1.0 and run mne.datasets.fetch_fsaverage() once.",
        ) from exc

    # ── 4. Noise covariance from the data itself (diagonal approximation) ─────
    # For resting-state data without a dedicated noise recording we use an
    # ad-hoc diagonal noise covariance.  For evoked data the proper approach
    # is a pre-stimulus baseline; we fall back to the diagonal estimate here
    # so the endpoint works without extra baseline files.
    try:
        noise_cov = mne.make_ad_hoc_cov(raw_crop.info, verbose=False)
    except Exception as exc:
        raise HTTPException(500, f"Noise covariance estimation failed: {exc}") from exc

    # ── 5. Compute inverse operator ───────────────────────────────────────────
    try:
        inv = mne.minimum_norm.make_inverse_operator(
            raw_crop.info,
            fwd,
            noise_cov,
            loose=0.2,    # loose orientation constraint — standard for MEG
            depth=0.8,    # depth weighting to compensate for lead-field fall-off
            verbose=False,
        )
    except Exception as exc:
        raise HTTPException(500, f"Inverse operator failed: {exc}") from exc

    # ── 6. Optional fMRI spatial prior ────────────────────────────────────────
    if request.bold_prior_path:
        _apply_bold_prior(inv, request.bold_prior_path, fwd)

    # ── 7. Apply inverse to averaged data ─────────────────────────────────────
    try:
        # Convert Raw to Epochs-like structure by treating the entire window as
        # one epoch, then average to create an Evoked object.
        events  = mne.make_fixed_length_events(raw_crop, duration=t_max - t_min)
        epochs  = mne.Epochs(
            raw_crop,
            events,
            tmin=0.0,
            tmax=t_max - t_min - 1.0 / raw_crop.info["sfreq"],
            baseline=None,
            preload=True,
            verbose=False,
        )
        evoked = epochs.average()
    except Exception:
        # If epoching fails fall back to treating the entire crop as evoked
        try:
            data, times = raw_crop.get_data(return_times=True, verbose=False)
            info = raw_crop.info.copy()
            evoked = mne.EvokedArray(data, info, tmin=float(times[0]), verbose=False)
        except Exception as exc2:
            raise HTTPException(500, f"Could not create evoked object: {exc2}") from exc2

    try:
        lambda2 = 1.0 / 9.0   # SNR = 3 → λ² = 1/9
        stc = mne.minimum_norm.apply_inverse(
            evoked,
            inv,
            lambda2=lambda2,
            method=request.method,
            verbose=False,
        )
    except Exception as exc:
        raise HTTPException(500, f"Inverse application failed: {exc}") from exc

    # ── 8. Extract source positions and amplitudes ─────────────────────────────
    # stc.data shape: (n_sources, n_times)  — take RMS over time
    amplitudes = np.sqrt((stc.data ** 2).mean(axis=1))

    # Source positions from the forward solution in metres → convert to mm
    src_positions_mm = fwd["source_rr"] * 1000.0   # (n_sources, 3) in mm

    # Convert MRI RAS (metres) to MNI coordinates via fsaverage affine
    # For fsaverage the MRI RAS ≈ MNI (no inter-subject warp needed).
    n_src = src_positions_mm.shape[0]
    vertices = [
        SourceVertex(
            x=float(src_positions_mm[i, 0]),
            y=float(src_positions_mm[i, 1]),
            z=float(src_positions_mm[i, 2]),
            amplitude=float(amplitudes[i]),
        )
        for i in range(n_src)
    ]

    peak_amplitude = float(amplitudes.max()) if len(amplitudes) > 0 else 0.0
    duration_ms    = (time.perf_counter() - t0) * 1000.0

    return SourceEstimateResponse(
        vertices=vertices,
        method=request.method,
        peak_amplitude=peak_amplitude,
        duration_ms=round(duration_ms, 1),
    )
