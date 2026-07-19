"""
meg_analysis.py — Automated MEG signal-quality analysis
=========================================================
FastAPI router providing three analysis endpoints that operate on recordings
already loaded via ``POST /api/load-meg``.  All endpoints require a
``session_id`` query parameter returned by that upload endpoint.

Endpoints
─────────
POST /api/meg/detect-artifacts
    Identifies eye-blink (EOG) and muscle / motion artefacts using MNE's
    built-in detectors.  Returns time-stamped annotation objects suitable for
    rendering as coloured background spans in the waveform canvas.

POST /api/meg/detect-spikes
    Applies a 20 Hz high-pass filter and a per-channel robust MAD-based
    threshold (default 5 σ) to flag sharp transient events that may indicate
    epileptiform activity.  Returns peak timestamps and the channels involved.

POST /api/meg/frequency-bands
    Computes a Welch PSD across 1–50 Hz, then partitions it into the five
    standard neurological frequency bands:
      δ Delta   1–4 Hz
      θ Theta   4–8 Hz
      α Alpha   8–12 Hz
      β Beta   12–30 Hz
      γ Gamma  30–50 Hz
    Returns the **relative** band power (band / total over [1,50] Hz) as a
    proxy for the spectral character of the recording.

Design notes
────────────
• MNE is imported lazily (inside each function body) so this module can be
  loaded quickly even if MNE is not yet on the path.
• All analysis is synchronous (heavy numpy / scipy work) but FastAPI's default
  thread-pool executor runs sync route handlers off the event loop, so the
  async interface of the application is not blocked.
• For a production deployment, move the heavy computation to a Celery / RQ
  worker and return a task ID for polling.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from session_store import require_meg

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/meg", tags=["MEG Analysis"])


# ── Shared response models ────────────────────────────────────────────────────

class ArtifactAnnotation(BaseModel):
    """One artefact window identified in the recording."""

    type:     str   = Field(description="'blink' (EOG) or 'muscle' (high-freq power)")
    onset:    float = Field(description="Window start time in seconds from recording start")
    duration: float = Field(description="Window duration in seconds")
    channel:  str   = Field(description="Source channel name or 'all' for sensor-space artefacts")


class ArtifactResponse(BaseModel):
    session_id:  str
    n_blinks:    int                     = Field(description="Number of detected eye blinks")
    n_muscle:    int                     = Field(description="Number of muscle/motion segments")
    annotations: list[ArtifactAnnotation] = Field(description="Chronologically sorted artefact list")


class SpikeMarker(BaseModel):
    """One detected epileptiform-like transient event."""

    time:      float     = Field(description="Peak crossing time in seconds")
    channels:  list[str] = Field(description="Channels exceeding the MAD threshold at this moment")
    amplitude: float     = Field(description="Maximum absolute amplitude across active channels (SI)")


class SpikeResponse(BaseModel):
    session_id:    str
    n_spikes:      int   = Field(description="Total number of detected spike events")
    threshold_mad: float = Field(description="MAD multiplier that was used (default 5)")
    spikes:        list[SpikeMarker]


class BandPower(BaseModel):
    """Relative spectral power for the five standard neurological bands."""

    delta: float = Field(description="1–4 Hz  relative power (deep sleep, large lesions)")
    theta: float = Field(description="4–8 Hz  relative power (drowsiness, memory)")
    alpha: float = Field(description="8–12 Hz relative power (relaxed wakefulness)")
    beta:  float = Field(description="12–30 Hz relative power (active cognition, motor)")
    gamma: float = Field(description="30–50 Hz relative power (high-level cognition)")


class FrequencyBandsResponse(BaseModel):
    session_id:    str
    n_channels:    int       = Field(description="Number of channels averaged for the PSD")
    channel_types: list[str] = Field(description="MNE channel type(s) included")
    bands:         BandPower


# ── Endpoint 1: Artefact detection ───────────────────────────────────────────

@router.post(
    "/detect-artifacts",
    response_model=ArtifactResponse,
    summary="Detect EOG blinks and muscle artefacts via MNE preprocessing",
)
def detect_artifacts(session_id: str = Query(...)) -> ArtifactResponse:
    """
    Run two complementary MNE artefact detectors:

    **EOG blink detection** (``mne.preprocessing.find_eog_events``)
        Band-passes an EOG channel at 1–10 Hz and picks amplitude peaks.
        Each detected event is padded to ±150 ms around the peak to capture
        the full blink waveform.  If no dedicated EOG channel exists, MNE
        tries frontal EEG channels (Fp1/Fp2) as fallback; if that also fails
        the detection is silently skipped.

    **Muscle / motion detection** (``mne.preprocessing.annotate_muscle_zscore``)
        Z-scores broadband power in the 110–140 Hz range (the MEG muscle band)
        across gradiometer or magnetometer channels.  Segments where the z-score
        exceeds ``threshold=4`` are annotated as muscle contamination and merged
        if shorter than 100 ms.  Useful for detecting swallowing, jaw clenching,
        and postural motion.

    Returns a flat, chronologically sorted list of annotation objects.  The
    ``type`` field is ``'blink'`` or ``'muscle'`` — use this to choose colours
    when rendering background spans (e.g. yellow for blinks, grey for muscle).
    """
    import mne  # lazy import — avoids loading MNE on every module import

    raw = require_meg(session_id)
    annotations: list[ArtifactAnnotation] = []
    n_blinks = 0
    n_muscle = 0

    # ── EOG blink detection ───────────────────────────────────────────────────
    try:
        # find_eog_events returns [n_events, 3]: [sample_idx, 0, event_id].
        eog_events = mne.preprocessing.find_eog_events(raw, verbose=False)
        blink_half = 0.15   # seconds either side of the blink peak

        for evt in eog_events:
            # Convert sample index to time in seconds using the raw time axis.
            peak_time = float(raw.times[int(evt[0])])
            onset     = max(0.0, peak_time - blink_half)
            annotations.append(ArtifactAnnotation(
                type="blink",
                onset=round(onset, 4),
                duration=round(blink_half * 2, 4),
                channel="EOG",
            ))

        n_blinks = len(eog_events)

    except Exception:
        # No EOG channel present or peak-finding failed — continue without blinks.
        pass

    # ── Muscle / motion detection ─────────────────────────────────────────────
    try:
        # Channel-type priority: gradiometers → magnetometers → EEG.
        # Gradiometers are less sensitive to head motion and more sensitive
        # to local muscle currents, making them the ideal choice.
        if len(mne.pick_types(raw.info, meg="grad")) > 0:
            ch_type = "grad"
        elif len(mne.pick_types(raw.info, meg="mag")) > 0:
            ch_type = "mag"
        else:
            ch_type = "eeg"

        muscle_annots, _scores = mne.preprocessing.annotate_muscle_zscore(
            raw,
            ch_type=ch_type,
            threshold=4.0,          # z-score threshold; 4 is conservative
            min_length_good=0.1,    # merge gaps shorter than 100 ms
            filter_freq=(110, 140), # MEG muscle frequency band
            verbose=False,
        )

        for annot in muscle_annots:
            annotations.append(ArtifactAnnotation(
                type="muscle",
                onset=round(float(annot["onset"]), 4),
                duration=round(float(annot["duration"]), 4),
                channel="all",
            ))

        n_muscle = len(muscle_annots)

    except Exception:
        # Muscle detection failed (e.g. no appropriate channel type) — skip.
        pass

    # Sort the combined list chronologically before returning.
    annotations.sort(key=lambda a: a.onset)

    return ArtifactResponse(
        session_id=session_id,
        n_blinks=n_blinks,
        n_muscle=n_muscle,
        annotations=annotations,
    )


# ── Endpoint 2: Epileptiform spike detection ──────────────────────────────────

@router.post(
    "/detect-spikes",
    response_model=SpikeResponse,
    summary="Flag epileptiform-like transient spikes using a MAD threshold",
)
def detect_spikes(
    session_id:     str   = Query(...),
    mad_multiplier: float = Query(5.0,  description="MAD threshold multiplier (default 5 σ)"),
    min_gap_sec:    float = Query(0.05, description="Minimum inter-spike gap to merge events (s)"),
) -> SpikeResponse:
    """
    Apply a 20 Hz high-pass filter and a per-channel robust statistical
    threshold to identify sharp transient events:

        threshold_ch = |median_ch| + mad_multiplier × MAD_ch × 1.4826

    The 1.4826 factor converts the median absolute deviation (MAD) to an
    equivalent Gaussian standard deviation (σ).  Using the MAD rather than the
    sample standard deviation makes the threshold robust to the spikes
    themselves — a classical "outlier-aware" estimator used in EEG/MEG QC.

    Events that occur within ``min_gap_sec`` of each other are merged into
    a single spike marker to avoid reporting the same discharge multiple times.
    The returned ``channels`` list names which sensors exceeded the threshold at
    the spike peak, useful for localising the irritative zone.

    ⚠️  Clinical disclaimer: This detector is intended as a data-exploration
    aid, not a certified medical device.  False positives are common in
    high-amplitude artefacts; false negatives occur for low-amplitude or
    polymorphic discharges.  Epileptiform activity requires expert review.
    """
    import mne  # lazy import

    raw = require_meg(session_id)

    # ── High-pass filter at 20 Hz ─────────────────────────────────────────────
    # A 20 Hz HPF removes slow baseline drift and alpha/beta oscillations,
    # accentuating the sharp morphology of epileptiform spikes.
    # load_data() is called explicitly before filter() because MNE raises
    # "Raw data needs to be preloaded" when filter() is applied to a
    # preload=False raw object (which KIT files always use on first load).
    try:
        raw_hp = raw.copy().load_data(verbose=False).filter(
            l_freq=20.0, h_freq=None, method="fir", verbose=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to preload/filter data for spike detection: {exc}",
        ) from exc

    # ── Pick signal channels ──────────────────────────────────────────────────
    picks = mne.pick_types(raw_hp.info, meg=True, eeg=True, exclude="bads")
    if len(picks) == 0:
        raise HTTPException(
            status_code=400,
            detail="No signal channels remain after excluding bad channels.",
        )

    ch_names = [raw_hp.ch_names[p] for p in picks]

    # Load the picked data slice from disk (honours preload=False).
    data:  np.ndarray = raw_hp.get_data(picks=picks)   # [n_ch, n_samples]
    times: np.ndarray = raw_hp.times                   # [n_samples]

    # ── Compute per-channel MAD thresholds ────────────────────────────────────
    medians    = np.median(data, axis=1, keepdims=True)            # [n_ch, 1]
    mads       = np.median(np.abs(data - medians), axis=1, keepdims=True)
    # Robust threshold: use the absolute value of the median as a DC offset
    # so that channels with a non-zero mean are not trivially crossed.
    thresholds = np.abs(medians) + mad_multiplier * mads * 1.4826  # [n_ch, 1]

    # ── Locate threshold crossings ────────────────────────────────────────────
    # True wherever any channel's absolute amplitude exceeds its threshold.
    any_above: np.ndarray = (np.abs(data) > thresholds).any(axis=0)  # [n_samples]

    # Rising edge detection: find samples where the boolean flag flips 0→1.
    crossings: np.ndarray = (
        np.where(np.diff(any_above.astype(np.int8)) == 1)[0] + 1
    )

    # ── Merge crossings closer than min_gap_sec ───────────────────────────────
    min_gap_samples = int(min_gap_sec * raw_hp.info["sfreq"])
    merged: list[int] = []

    if len(crossings) > 0:
        merged.append(int(crossings[0]))
        for c in crossings[1:]:
            if c - merged[-1] > min_gap_samples:
                merged.append(int(c))

    # ── Build SpikeMarker objects ─────────────────────────────────────────────
    spikes: list[SpikeMarker] = []
    above_mask: np.ndarray = np.abs(data) > thresholds   # [n_ch, n_samples]

    for sample_idx in merged:
        t = float(times[sample_idx])

        # Identify which channels fired at or near this sample.
        # Check a ±3-sample neighbourhood to handle slight timing offsets.
        lo  = max(0,                   sample_idx - 3)
        hi  = min(data.shape[1] - 1,  sample_idx + 3)
        active: np.ndarray = above_mask[:, lo : hi + 1].any(axis=1)

        if not active.any():
            continue   # safety: skip if neighbourhood is also clean

        active_chs = [ch_names[i] for i in np.where(active)[0]]
        peak_amp   = float(np.abs(data[:, sample_idx]).max())

        spikes.append(SpikeMarker(
            time=round(t, 5),
            channels=active_chs[:10],   # cap at 10 names to keep response slim
            amplitude=round(peak_amp, 15),
        ))

    return SpikeResponse(
        session_id=session_id,
        n_spikes=len(spikes),
        threshold_mad=float(mad_multiplier),
        spikes=spikes,
    )


# ── Endpoint 3: Frequency band power ─────────────────────────────────────────

@router.post(
    "/frequency-bands",
    response_model=FrequencyBandsResponse,
    summary="Compute relative PSD power per standard neurological frequency band",
)
def frequency_bands(
    session_id: str        = Query(...),
    t_start:    float | None = Query(None, description="Analysis window start (s); None = full recording"),
    t_end:      float | None = Query(None, description="Analysis window end (s);   None = full recording"),
) -> FrequencyBandsResponse:
    """
    Compute a Welch power spectral density (PSD) over [1, 50] Hz and
    partition it into the five canonical neurological frequency bands.

    The PSD is averaged across all good channels of the preferred sensor type
    (magnetometers → gradiometers → EEG) to obtain a single "global" spectral
    fingerprint for the recording or the requested time window.

    **Relative power** for each band is computed as::

        band_power / sum_power_over_[1,50]_Hz

    This normalisation removes the effect of absolute signal amplitude (which
    varies with sensor distance and subject anatomy) and makes the band-power
    profile comparable across subjects and sessions.

    Clinical interpretation hints:
      • Elevated delta:  large structural lesion, coma, or deep anaesthesia.
      • Elevated theta:  temporal lobe pathology, reduced consciousness.
      • Dominant alpha:  normal eyes-closed resting state.
      • Low alpha:       arousal, cognitive load, or seizure onset.
      • Elevated gamma:  cognitively demanding tasks; can also be muscle.
    """
    import mne  # lazy import

    raw = require_meg(session_id)

    # ── Select channel type ───────────────────────────────────────────────────
    # Magnetometers give the best global field estimate; fall back as needed.
    if len(mne.pick_types(raw.info, meg="mag")) > 0:
        ch_type = "mag"
        picks = mne.pick_types(raw.info, meg="mag", exclude="bads")
    elif len(mne.pick_types(raw.info, meg="grad")) > 0:
        ch_type = "grad"
        picks = mne.pick_types(raw.info, meg="grad", exclude="bads")
    else:
        ch_type = "eeg"
        picks = mne.pick_types(raw.info, eeg=True, exclude="bads")

    if len(picks) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No usable '{ch_type}' channels found after bad-channel exclusion.",
        )

    # ── Compute Welch PSD ─────────────────────────────────────────────────────
    psd_kwargs: dict = dict(
        method="welch",
        fmin=1.0,
        fmax=50.0,
        n_fft=2048,   # ~4 s segments at 512 Hz; balances frequency resolution and variance
        picks=picks,
        verbose=False,
    )
    if t_start is not None:
        psd_kwargs["tmin"] = float(t_start)
    if t_end is not None:
        psd_kwargs["tmax"] = float(t_end)

    try:
        spectrum = raw.compute_psd(**psd_kwargs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PSD computation failed: {exc}") from exc

    # psd: [n_channels, n_freqs] in unit²/Hz
    psd:   np.ndarray = spectrum.get_data()
    freqs: np.ndarray = spectrum.freqs

    # Average across channels → [n_freqs] global power spectrum.
    mean_psd = psd.mean(axis=0)

    # ── Partition into bands ──────────────────────────────────────────────────
    BANDS: dict[str, tuple[float, float]] = {
        "delta": ( 1.0,  4.0),
        "theta": ( 4.0,  8.0),
        "alpha": ( 8.0, 12.0),
        "beta":  (12.0, 30.0),
        "gamma": (30.0, 50.0),
    }

    total_power = float(mean_psd.sum()) or 1.0   # guard against all-zero PSD

    band_powers: dict[str, float] = {}
    for band_name, (flo, fhi) in BANDS.items():
        mask = (freqs >= flo) & (freqs < fhi)
        band_powers[band_name] = round(
            float(mean_psd[mask].sum()) / total_power, 5
        )

    return FrequencyBandsResponse(
        session_id=session_id,
        n_channels=int(len(picks)),
        channel_types=[ch_type],
        bands=BandPower(
            delta=band_powers["delta"],
            theta=band_powers["theta"],
            alpha=band_powers["alpha"],
            beta=band_powers["beta"],
            gamma=band_powers["gamma"],
        ),
    )
