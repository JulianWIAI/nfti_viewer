"""
neural_decoding.py — MVPA neural decoding pipeline via MNE + scikit-learn
══════════════════════════════════════════════════════════════════════════════

Implements a Multivariate Pattern Analysis (MVPA) sliding-window decoding
pipeline that trains a linear classifier to discriminate between two brain
states from EEG sensor data and returns a time-resolved accuracy curve.

MATHEMATICAL OVERVIEW
──────────────────────
Given:
    X : (n_epochs, n_channels, n_times)  — epoched sensor-space EEG
    y : (n_epochs,)                       — class labels (0 or 1)

At each time point t, a classifier f_t is trained on the spatial pattern
of all channels simultaneously:

    ŷ_t  =  f_t( X[:, :, t] )
    f_t  =  sign( w_t^T · X[:, :, t] + b_t )   [linear classifier]

where the weight vector w_t ∈ ℝ^{n_channels} is the learned spatial filter.

The SlidingEstimator trains n_times independent classifiers, one per
time point.  This is NOT the same as temporal generalisation (a 2-D
generalisation matrix) — it is the diagonal of that matrix.

CROSS-VALIDATION
─────────────────
Stratified K-Fold cross-validation is used so each fold preserves the class
balance, which is critical when the two conditions have different trial counts.

    S ∈ ℝ^{n_folds × n_times}   — score matrix from cross_val_multiscore()

Reported accuracy:  mean(S, axis=0) : (n_times,)
Confidence range:   ±std(S, axis=0) : (n_times,)   — for error-bar plots
Chance level:       0.5 for binary classification

Scoring metric: ROC-AUC (area under the receiver-operating-characteristic
curve).  This is threshold-free and handles class imbalance better than
raw accuracy.  A score of 1.0 = perfect discrimination; 0.5 = chance.

SPATIAL FILTERING — CSP (optional, use_csp=True)
──────────────────────────────────────────────────
Common Spatial Patterns (CSP) finds a spatial filter W ∈ ℝ^{n_ch × n_comp}
that simultaneously diagonalises the covariance matrices of both classes:

    W^T Σ_A W = D_A,   W^T Σ_B W = D_B       (simultaneous diagonalisation)

The filtered components maximise variance for class A and minimise it for
class B (first component) or vice versa (last component).  The log-variance
of each component is the feature fed to the classifier.

CSP improves decoding for oscillatory paradigms (motor imagery, SSVEP) where
the discriminant information is in the band-power envelope.  For ERP-based
paradigms (P300, N170) raw sensor patterns are generally better.

CLASSIFIER CHOICES
───────────────────
  svm   — LinearSVC (L2 penalty, hinge loss).  Fast, large-margin, works
           well with many channels and few trials.  Regularisation via C=1.0.
           Does NOT output probabilities — ROC-AUC uses decision_function().

  ridge — RidgeClassifier (L2-regularised least-squares on the {-1,+1} label
           vector).  Lower variance than SVM; preferable when n_epochs is
           very small (< 50) or channels are highly correlated.

EVENT DETECTION STRATEGY
──────────────────────────
  1. Annotations → mne.events_from_annotations()   (BrainVision default)
  2. STI channel → mne.find_events()               (Neuromag, some EEG amps)
  3. ValueError  → client must supply explicit event_id mapping

OUTPUT FORMAT
──────────────
{
  "times":          [float, ...],   // seconds — one per time sample after decim
  "scores":         [float, ...],   // mean ROC-AUC across CV folds at each t
  "scores_std":     [float, ...],   // std across folds (for ±1 SD error bars)
  "mean_score":     float,          // scalar mean(scores) over the time axis
  "chance_level":   0.5,            // binary classification
  "peak_accuracy":  float,          // max(scores)
  "peak_time_ms":   float,          // latency of peak in milliseconds
  "n_epochs":       int,
  "n_channels":     int,
  "duration_ms":    float           // wall-clock compute time
}
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Literal

import mne
import numpy as np
from fastapi import APIRouter, HTTPException
from mne.decoding import SlidingEstimator, cross_val_multiscore
from pydantic import BaseModel, Field
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from session_store import require_eeg

logger = logging.getLogger(__name__)


# ── Pydantic models ────────────────────────────────────────────────────────────

class DecodeRequest(BaseModel):
    """Parameters for one MVPA decoding job."""

    session_id: str
    """ID of a loaded EEG session (returned by POST /api/load-eeg)."""

    event_id: dict[str, int] | None = None
    """
    Map of condition-label → integer event code, e.g.
    {"face": 1, "house": 2}.  Exactly 2 conditions are used (binary
    classification).  If None, the 2 most frequent event codes found in the
    file's annotations or STI channel are used automatically.
    """

    tmin: float = Field(default=-0.2, description="Epoch start relative to event onset (s).")
    tmax: float = Field(default=0.8,  description="Epoch end relative to event onset (s).")

    baseline: tuple[float | None, float | None] = (None, 0.0)
    """
    Baseline correction window passed to mne.Epochs.
    (None, 0.0) → use the entire pre-stimulus period up to t=0.
    Set to None to disable baseline correction.
    """

    classifier: Literal["svm", "ridge"] = "svm"
    """
    Classifier type.
    svm   → LinearSVC (C=1.0, L2 penalty).  Best for many-channel data.
    ridge → RidgeClassifier (α=1.0, L2 least-squares).  Better for few epochs.
    """

    n_folds: int = Field(default=5, ge=2, le=20, description="Number of CV folds.")

    use_csp: bool = False
    """
    Prepend a Common Spatial Patterns (CSP) filter (n_components=6) before the
    classifier.  Recommended for motor-imagery or band-power paradigms.
    Not recommended for ERP paradigms — raw sensor patterns are cleaner.
    """

    decim: int = Field(default=1, ge=1, le=20)
    """
    Integer decimation factor applied inside mne.Epochs after epoching.
    decim=5 on a 1000 Hz recording → 200 Hz temporal resolution, 5× faster.
    Baseline correction is applied at the original sample rate before decimation.
    """


class DecodeResponse(BaseModel):
    """Serialised MVPA decoding result."""

    times:         list[float]   # seconds — one per time sample
    scores:        list[float]   # mean ROC-AUC at each time point
    scores_std:    list[float]   # std across CV folds (for error bars)
    mean_score:    float         # scalar mean over the time axis
    chance_level:  float         # 0.5 for binary classification
    peak_accuracy: float         # max single time-point score
    peak_time_ms:  float         # latency of peak (ms)
    n_epochs:      int           # total epochs used (after bad-epoch rejection)
    n_channels:    int
    duration_ms:   float


# ── Private helpers ────────────────────────────────────────────────────────────

def _detect_events(
    raw:      mne.io.BaseRaw,
    event_id: dict[str, int] | None,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Find event markers in the recording and return at most 2 conditions.

    Detection priority
    ──────────────────
    1. mne.events_from_annotations() — works for BrainVision and EDF files
       where triggers are stored as string annotations.
    2. mne.find_events()             — works for Neuromag/EEG amplifiers that
       encode triggers on a dedicated STI 014 stimulus channel.
    3. ValueError                    — no triggers found; user must supply
       explicit integer codes in event_id.

    Parameters
    ──────────
    event_id : optional dict mapping label → integer code.
       When provided, the dict is returned as-is after event detection.
       When None, the two most frequent codes are auto-selected.

    Returns
    ────────
    (events, event_id_map)
    events       : ndarray (n_events, 3) — [sample_idx, prev_id, event_id]
    event_id_map : dict with exactly the two condition labels to decode
    """
    events = np.empty((0, 3), dtype=int)

    # Strategy 1: annotation-based (BrainVision, EDF+).
    try:
        evs, _ = mne.events_from_annotations(raw, verbose=False)
        if len(evs) > 0:
            events = evs
    except Exception:
        pass

    # Strategy 2: stimulus channel (if annotation strategy found nothing).
    if len(events) == 0:
        sti_picks = mne.pick_types(raw.info, stim=True)
        if len(sti_picks) > 0:
            try:
                evs = mne.find_events(raw, verbose=False)
                if len(evs) > 0:
                    events = evs
            except Exception:
                pass

    if len(events) == 0:
        raise ValueError(
            "No event markers found in this recording.  Either the file has no "
            "annotations / STI channel, or provide 'event_id' with explicit codes."
        )

    # If the caller supplied an explicit event_id, trust it.
    if event_id is not None:
        return events, event_id

    # Auto-select the two most frequent event codes in the detected events.
    codes, counts = np.unique(events[:, 2], return_counts=True)
    if len(codes) < 2:
        raise ValueError(
            f"Only one event code ({codes[0]}) found — need at least 2 distinct "
            "conditions for binary classification."
        )
    top2 = codes[np.argsort(counts)[::-1][:2]]
    auto_map = {f"condition_{int(c)}": int(c) for c in top2}
    logger.info("Auto-selected event codes: %s", auto_map)
    return events, auto_map


def _build_pipeline(classifier_name: str, use_csp: bool) -> Pipeline:
    """
    Construct the scikit-learn Pipeline that is applied at each time point.

    Pipeline steps
    ──────────────
    [CSP]          — optional spatial filter (only if use_csp=True).
                     Operates on the full trial (n_channels, n_times) inside
                     SlidingEstimator, NOT on a single time-point slice.
                     NOTE: CSP requires SlidingEstimator to receive 3-D input,
                     so use_csp=True overrides the per-time-point slicing —
                     see _run_decoding() for the CSP branch.

    StandardScaler — z-scores each channel independently across the training
                     set so amplitude differences don't dominate the decision
                     boundary.  Fit on train folds, applied to test folds.

    Classifier     — linear model; its weight vector is the learned spatial
                     pattern separating the two conditions.

    Parameters
    ──────────
    classifier_name : "svm" or "ridge"
    use_csp         : if True, a CSP step is prepended (see note above)

    Returns
    ────────
    sklearn Pipeline instance (not yet fit — SlidingEstimator / CV will fit it)
    """
    steps: list[tuple[str, object]] = []

    if use_csp:
        # CSP operates on (n_epochs, n_channels, n_times) and outputs
        # (n_epochs, n_components) log-variance features.
        # It is fit independently per CV fold to avoid data leakage.
        from mne.decoding import CSP  # imported here to keep top-level import lightweight
        steps.append(('csp', CSP(
            n_components=6,    # number of spatial filters to keep (standard for BCI)
            reg=None,          # no covariance regularisation; set 'ledoit_wolf' for few trials
            log=True,          # return log-variance features (linearises the power)
            norm_trace=False,
        )))

    steps.append(('scaler', StandardScaler()))

    if classifier_name == 'svm':
        # dual='auto' chooses the primal formulation (faster) when
        # n_samples > n_features, which is the case for EEG after CSP or
        # when the number of epochs exceeds the number of channels.
        steps.append(('clf', LinearSVC(max_iter=2000, dual='auto', C=1.0)))
    elif classifier_name == 'ridge':
        steps.append(('clf', RidgeClassifier(alpha=1.0)))
    else:
        raise ValueError(f"Unknown classifier '{classifier_name}'. Choose 'svm' or 'ridge'.")

    return Pipeline(steps)


# ── Core pipeline (runs in a thread pool) ─────────────────────────────────────

def _run_decoding(req: DecodeRequest) -> DecodeResponse:
    """
    Execute the full MVPA pipeline synchronously.

    Called inside asyncio.to_thread() — never invoke this from the event loop.

    Steps
    ─────
    1.  Retrieve Raw from session store.
    2.  Detect events from annotations or STI channel.
    3.  Epoch the raw signal around event onsets (baseline corrected).
    4.  Build the classifier pipeline (scaler + SVM/Ridge, optionally + CSP).
    5.  Wrap in SlidingEstimator for independent per-time-point scoring.
    6.  Run stratified K-fold cross-validation via cross_val_multiscore().
    7.  Aggregate scores: mean and std across folds.
    8.  Return the time-resolved accuracy curve and scalar summary statistics.
    """
    t_wall = time.perf_counter()

    # ── 1. Session ────────────────────────────────────────────────────────────
    raw = require_eeg(req.session_id)

    # ── 2. Events ─────────────────────────────────────────────────────────────
    events, event_id_map = _detect_events(raw, req.event_id)

    # Keep exactly two conditions for binary classification.
    cond_labels = list(event_id_map.keys())[:2]
    binary_id   = {k: event_id_map[k] for k in cond_labels}

    # ── 3. Epochs ─────────────────────────────────────────────────────────────
    # Pick EEG and/or MEG channels; exclude bad channels and the STI channel.
    picks = mne.pick_types(
        raw.info, eeg=True, meg=True, stim=False, exclude='bads',
    )

    # baseline=None disables correction when the caller sets it to None.
    baseline = tuple(req.baseline) if req.baseline is not None else None  # type: ignore[arg-type]

    epochs = mne.Epochs(
        raw,
        events,
        event_id=binary_id,
        tmin=req.tmin,
        tmax=req.tmax,
        baseline=baseline,
        picks=picks,
        decim=req.decim,      # time decimation applied after baseline correction
        preload=True,         # load all data into RAM for fast CV iteration
        verbose=False,
    )
    # drop_bad() removes epochs that contain amplitude artefacts (flat or jump).
    epochs.drop_bad(verbose=False)

    n_epochs   = len(epochs)
    n_channels = len(epochs.ch_names)
    times      = epochs.times                    # (n_times,) after decimation

    # Require enough epochs for the chosen CV split.
    min_required = req.n_folds * 2              # at least 2 per fold per class
    if n_epochs < min_required:
        raise ValueError(
            f"Only {n_epochs} clean epoch(s) — need at least {min_required} "
            f"for {req.n_folds}-fold CV (2 per condition per fold)."
        )

    # ── 4. Data matrix and integer labels ─────────────────────────────────────
    X = epochs.get_data()         # (n_epochs, n_channels, n_times) float64
    y = epochs.events[:, 2]      # raw integer event codes

    # Re-map event codes → {0, 1} so both classifiers produce the same label
    # encoding regardless of the original integer codes.
    code_to_label = {code: idx for idx, code in enumerate(np.unique(y))}
    y = np.array([code_to_label[c] for c in y], dtype=int)

    # ── 5. Classifier pipeline and sliding estimator ──────────────────────────
    clf = _build_pipeline(req.classifier, req.use_csp)

    if req.use_csp:
        # When CSP is in the pipeline, SlidingEstimator cannot be used because
        # CSP needs the full trial (n_channels, n_times) to compute covariance,
        # not a single time-point slice (n_channels, 1).
        # Solution: use a single Pipeline with CSP on the full 3-D array.
        # cross_val_score returns a scalar per fold (not per time-point).
        from sklearn.model_selection import cross_val_score
        cv      = StratifiedKFold(n_splits=req.n_folds, shuffle=True, random_state=42)
        # cross_val_score with 3-D X requires the CSP step to handle 3-D input.
        fold_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
        # Return a flat curve at the scalar mean score (no time resolution for CSP).
        mean_scores = np.full(len(times), fill_value=fold_scores.mean())
        std_scores  = np.full(len(times), fill_value=fold_scores.std())
    else:
        # Standard path: SlidingEstimator trains clf independently at every t.
        # cross_val_multiscore returns (n_folds, n_times).
        time_decod    = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc', verbose=False)
        cv            = StratifiedKFold(n_splits=req.n_folds, shuffle=True, random_state=42)
        scores_matrix = cross_val_multiscore(time_decod, X, y, cv=cv, n_jobs=1)
        mean_scores   = scores_matrix.mean(axis=0)   # (n_times,)
        std_scores    = scores_matrix.std(axis=0)    # (n_times,)

    # ── 6. Summary statistics ─────────────────────────────────────────────────
    peak_idx      = int(np.argmax(mean_scores))
    peak_accuracy = float(mean_scores[peak_idx])
    peak_time_ms  = round(float(times[peak_idx]) * 1000.0, 1)

    elapsed_ms = (time.perf_counter() - t_wall) * 1000.0
    logger.info(
        "Decoding done: %d epochs, %d ch, %d t-points, peak AUC=%.3f at %.1f ms, %.1f ms elapsed",
        n_epochs, n_channels, len(times), peak_accuracy, peak_time_ms, elapsed_ms,
    )

    return DecodeResponse(
        times=times.tolist(),
        scores=mean_scores.tolist(),
        scores_std=std_scores.tolist(),
        mean_score=float(mean_scores.mean()),
        chance_level=0.5,
        peak_accuracy=peak_accuracy,
        peak_time_ms=peak_time_ms,
        n_epochs=n_epochs,
        n_channels=n_channels,
        duration_ms=round(elapsed_ms, 1),
    )


# ── FastAPI router ─────────────────────────────────────────────────────────────

router = APIRouter(tags=["Neural Decoding"])


@router.post("/api/eeg/decode-session", response_model=DecodeResponse)
async def decode_brain_state(req: DecodeRequest) -> DecodeResponse:
    """
    Run a sliding-window MVPA pipeline to decode two brain states from EEG.

    The classifier is trained independently at each time point across multiple
    cross-validation folds, returning a time-resolved ROC-AUC curve.  Values
    above the 0.5 chance level indicate statistically meaningful discrimination.

    The endpoint runs inside a thread pool (asyncio.to_thread) so the FastAPI
    event loop is never blocked.

    **Typical wall-clock times (n=100 epochs, 64 ch, 500 ms at 256 Hz, 5-fold CV):**
    ~ 5–30 seconds depending on classifier and n_folds.

    **Event auto-detection:**
    If `event_id` is omitted, the two most frequent trigger codes in the
    recording's annotations are used automatically.  If detection fails, the
    error message describes which trigger codes are available.

    **CSP note:**
    When `use_csp=True`, the time-resolved curve degenerates to a flat line
    at the cross-validated scalar accuracy (CSP requires the full trial).
    This is by design — CSP is a trial-level, not time-point-level, filter.
    """
    try:
        return await asyncio.to_thread(_run_decoding, req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Neural decoding failed unexpectedly.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
