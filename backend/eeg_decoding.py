"""
eeg_decoding.py — Time-resolved MVPA pipeline (MNE-Python + scikit-learn)
════════════════════════════════════════════════════════════════════════════

Pure computation module: no FastAPI dependency.
The FastAPI route handler lives in routers/decoding.py and calls
`run_time_resolved_mvpa` via asyncio.to_thread().

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS MVPA / NEURAL DECODING?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Multivariate Pattern Analysis (MVPA) asks: can a linear classifier distinguish
between two cognitive conditions (e.g., "Stimulus A" vs "Stimulus B") from the
multichannel neural signal recorded at a single time point?

Classical ERP analysis averages trials and tests each electrode independently.
MVPA uses *all* channels simultaneously, exploiting the distributed spatial
pattern of neural activation — which is far more sensitive to condition effects.

The output is a "decoding time-course": for each time point in the epoch, we
report the Area Under the ROC Curve (AUC) of the best-fitting linear classifier.
AUC = 0.5 → chance (classifier is guessing); AUC = 1.0 → perfect decoding.

Peaks in the time-course reveal WHEN the brain encodes the cognitive distinction,
and their latency/duration can be compared across conditions, groups, or regions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ML ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Array shapes at each stage of the pipeline:

  X_raw : (n_epochs, n_channels, n_times)   ← epochs.get_data()
  y     : (n_epochs,)                        ← binary {0, 1}

  ┌─────────────────────────────────────────────────────────────────────┐
  │ SlidingEstimator (outer wrapper)                                    │
  │                                                                     │
  │   For t in 0 … n_times - 1:                                        │
  │     X_t = X_raw[:, :, t]   shape (n_epochs, n_channels)            │
  │                                                                     │
  │     ┌─────────────────────────────────────────────────────────┐    │
  │     │ Inner scikit-learn Pipeline                              │    │
  │     │                                                          │    │
  │     │  1. Vectorizer                                          │    │
  │     │     Input : (n_epochs, n_channels)    [2-D]             │    │
  │     │     Output: (n_epochs, n_channels)    [no-op for 2-D]   │    │
  │     │     WHY: SlidingEstimator already reduces 3-D → 2-D     │    │
  │     │     by slicing at each t.  Vectorizer is REQUIRED for   │    │
  │     │     full spatiotemporal decoding where X is still 3-D:  │    │
  │     │       (n_epochs, n_ch, n_t) → (n_epochs, n_ch * n_t)   │    │
  │     │     Including it here keeps the pipeline portable.      │    │
  │     │                                                          │    │
  │     │  2. StandardScaler                                      │    │
  │     │     Input : (n_epochs, n_channels)                      │    │
  │     │     Output: (n_epochs, n_channels)   z-scored per feat  │    │
  │     │     WHY: LR is sensitive to feature scale.  Channels    │    │
  │     │     record in different physical units (µV, fT, fT/cm); │    │
  │     │     z-scoring normalises them to comparable variance.   │    │
  │     │     Fit on train fold; applied to test fold (no leak).  │    │
  │     │                                                          │    │
  │     │  3. LogisticRegression (liblinear, L2)                  │    │
  │     │     Input : (n_epochs, n_channels)  [scaled]            │    │
  │     │     Output: binary prediction / decision score           │    │
  │     │     WHY: Linear classifier is the standard choice for   │    │
  │     │     neuroimaging decoding: it trains fast (important for │    │
  │     │     n_times fits per fold), is interpretable (the weight │    │
  │     │     vector is a spatial filter), and avoids overfitting  │    │
  │     │     when n_channels ≪ n_epochs.  SVM is equivalent but  │    │
  │     │     LR gives calibrated probability outputs.             │    │
  │     └─────────────────────────────────────────────────────────┘    │
  │                                                                     │
  │   Returns: scores[fold, t] = AUC at time t for this fold           │
  └─────────────────────────────────────────────────────────────────────┘

  cross_val_multiscore (k-fold, stratified):
    scores : (n_folds, n_times)   AUC per fold per time point
    mean   : (n_times,)           mean across folds   → the time-course
    std    : (n_times,)           std across folds    → shading band

WHY roc_auc AND NOT ACCURACY?
  Accuracy is biased by class imbalance (if class A has 80 % of trials,
  predicting A always gives 80 % accuracy).  AUC is insensitive to imbalance
  because it measures the ranking quality of the decision boundary, not its
  absolute threshold.  It is the recommended metric for neuroimaging decoding
  (Combrisson & Jerbi, 2015).

WHY StratifiedKFold?
  Stratified splitting preserves the class ratio in each fold.  Without it,
  some folds may contain only one class, causing AUC to be undefined.  MNE's
  cross_val_multiscore uses StratifiedKFold internally when y is binary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mne
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import SlidingEstimator, cross_val_multiscore, Vectorizer


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class MVPAConfig:
    """
    All tunable parameters for the time-resolved MVPA pipeline.

    Separating config from code makes unit testing straightforward:
      config = MVPAConfig(class_a=1, class_b=2, n_folds=3)
      result = run_time_resolved_mvpa(raw, config)
    """

    # ── Experimental design ──────────────────────────────────────────────────

    # Integer event code for the first condition (label 0 in y).
    class_a: int
    # Integer event code for the second condition (label 1 in y).
    class_b: int

    # ── Epoching ─────────────────────────────────────────────────────────────

    # Epoch window relative to event onset (seconds).
    # tmin = -0.2 captures 200 ms of pre-stimulus baseline.
    tmin: float = -0.2
    tmax: float = 0.8

    # Baseline correction window.  (None, 0.0) = mean-subtract the
    # [-tmin, 0] interval (standard pre-stimulus baseline).
    # Set apply_baseline = False to skip correction entirely.
    baseline_end: float = 0.0
    apply_baseline: bool = True

    # ── Cross-validation ─────────────────────────────────────────────────────

    # Number of cross-validation folds.  5 is standard; use 3 for small
    # datasets (< 30 epochs per class) or 10 for large ones.
    n_folds: int = 5

    # ── Classifier hyperparameters ───────────────────────────────────────────

    # Inverse regularisation strength for LogisticRegression.
    # Larger C = weaker regularisation.  C = 1.0 is a conservative default
    # that works well across most neuroimaging datasets.
    C: float = 1.0

    # Maximum solver iterations.  200 is usually sufficient after z-scoring.
    max_iter: int = 200


# ── Result ─────────────────────────────────────────────────────────────────────

@dataclass
class MVPAResult:
    """
    All outputs of run_time_resolved_mvpa.

    JSON-serialisable via Python's dataclasses.asdict() or direct attribute
    access.  The FastAPI route handler converts this to a Pydantic response.
    """

    # Time axis matching the epoch window, in seconds.  Length = n_times.
    times: list[float]

    # Mean AUC across k folds at every time point.  Shape (n_times,).
    # The primary output: plot this against times to see the decoding curve.
    scores: list[float]

    # Std of AUC across folds.  Shape (n_times,).
    # Use to draw a shading band (±1 std) around the decoding curve.
    scores_std: list[float]

    # Theoretical chance level for binary AUC (always 0.5).
    chance_level: float

    # Trial counts.
    n_epochs: int
    n_epochs_class_a: int
    n_epochs_class_b: int

    # Data dimensions.
    n_channels: int
    n_times: int

    # Peak decoding statistics.
    peak_score: float   # highest mean AUC across the time axis
    peak_time_s: float  # time (s) at which peak AUC occurs

    # Wall-clock time for the entire pipeline (ms).
    duration_ms: float


# ── Public entry point ─────────────────────────────────────────────────────────

def run_time_resolved_mvpa(raw: mne.io.BaseRaw, config: MVPAConfig) -> MVPAResult:
    """
    Run the time-resolved MVPA pipeline on a loaded MNE Raw object.

    This function is synchronous and CPU-bound.  The FastAPI route handler
    runs it in a thread pool via  await asyncio.to_thread(run_time_resolved_mvpa, ...)
    so the event loop is never blocked.

    Parameters
    ----------
    raw :
        Loaded MNE Raw object (BrainVision, FIF, or any format MNE supports).
        Must contain at least EEG or MEG channels and stimulus annotations.
    config :
        All pipeline parameters (epoch window, classifier settings, etc.).

    Returns
    -------
    MVPAResult
        Decoding time-course and summary statistics.

    Raises
    ------
    ValueError
        If fewer than n_folds * 2 epochs are found (not enough to cross-validate).
    """
    t_start = time.perf_counter()

    # ── 1. Epoch the raw data ────────────────────────────────────────────────
    epochs, y = _build_epochs(raw, config)

    # epochs.get_data() returns a float64 array of shape
    # (n_epochs, n_channels, n_times).  This is the canonical input shape for
    # mne.decoding.SlidingEstimator and mne.decoding.cross_val_multiscore.
    X: np.ndarray = epochs.get_data()   # (n_epochs, n_channels, n_times)

    # ── 2. Build the SlidingEstimator pipeline ───────────────────────────────
    decoder = _build_sliding_decoder(config)

    # ── 3. Cross-validate — the main computational cost ─────────────────────
    # scores : (n_folds, n_times)  AUC at each time point for each fold
    scores: np.ndarray = _cross_validate(decoder, X, y, config)

    # ── 4. Aggregate over folds ──────────────────────────────────────────────
    mean_scores: np.ndarray = scores.mean(axis=0)   # (n_times,)
    std_scores:  np.ndarray = scores.std(axis=0)    # (n_times,)

    peak_idx  = int(np.argmax(mean_scores))
    n_a       = int((y == 0).sum())
    n_b       = int((y == 1).sum())
    duration  = (time.perf_counter() - t_start) * 1_000.0

    return MVPAResult(
        times            = epochs.times.tolist(),
        scores           = mean_scores.tolist(),
        scores_std       = std_scores.tolist(),
        chance_level     = 0.5,
        n_epochs         = len(epochs),
        n_epochs_class_a = n_a,
        n_epochs_class_b = n_b,
        n_channels       = X.shape[1],
        n_times          = X.shape[2],
        peak_score       = float(mean_scores[peak_idx]),
        peak_time_s      = float(epochs.times[peak_idx]),
        duration_ms      = round(duration, 1),
    )


# ── Private helpers ────────────────────────────────────────────────────────────

def _build_epochs(
    raw: mne.io.BaseRaw,
    config: MVPAConfig,
) -> tuple[mne.Epochs, np.ndarray]:
    """
    Extract stimulus-locked epochs and build binary class labels.

    Steps
    ─────
    1. Call mne.events_from_annotations() to parse the stimulus trigger log
       embedded in the raw file's annotation track.  For BrainVision files
       this reads the .vmrk marker file that accompanies the .vhdr/.eeg pair.

    2. Filter the events array to rows whose trigger value is class_a or class_b.

    3. Create mne.Epochs with baseline correction (if configured).

    4. Return (epochs, y) where y is a binary integer array
       aligned to epochs.events (class_a → 0, class_b → 1).

    Data-shape note
    ───────────────
    epochs.get_data() → (n_epochs, n_channels, n_times), float64.
    This 3-D array is the direct input to SlidingEstimator.

    Parameters
    ----------
    raw    : Preloaded or lazily-loaded MNE Raw object.
    config : Pipeline configuration.

    Returns
    -------
    epochs : mne.Epochs — preloaded, baseline-corrected.
    y      : np.ndarray of shape (n_epochs,), dtype int, values {0, 1}.
    """
    # Parse the stimulus annotation log embedded in the file.
    # event_id_map: dict[str, int] maps annotation description → integer code.
    # events[:, 2] contains these integer codes; we use them to filter below.
    events, _event_id_map = mne.events_from_annotations(raw, verbose=False)

    if len(events) == 0:
        raise ValueError(
            "No stimulus annotations found in the raw file. "
            "Check that the .vmrk marker file was uploaded alongside the .vhdr."
        )

    # Filter to the two target trigger codes.  events[:, 2] is the trigger value.
    target_codes = np.array([config.class_a, config.class_b])
    mask = np.isin(events[:, 2], target_codes)

    n_found = mask.sum()
    min_required = config.n_folds * 2   # need at least 1 trial per class per fold
    if n_found < min_required:
        raise ValueError(
            f"Only {n_found} epochs found for event codes "
            f"{config.class_a} and {config.class_b}. "
            f"Need at least {min_required} ({config.n_folds} folds × 2 classes). "
            "Check that the event codes match the triggers in your .vmrk file."
        )

    events_filtered = events[mask]

    # Construct the event_id dict with string keys — required by mne.Epochs.
    event_id = {
        str(config.class_a): config.class_a,
        str(config.class_b): config.class_b,
    }

    # Baseline tuple: (start, end) where None = epoch boundary.
    # (None, 0.0) is the standard pre-stimulus interval.
    baseline = (None, config.baseline_end) if config.apply_baseline else None

    # Epoch the data.  preload=True loads all trial data into RAM immediately
    # so that get_data() returns the full 3-D array without disk reads.
    epochs = mne.Epochs(
        raw,
        events_filtered,
        event_id  = event_id,
        tmin      = config.tmin,
        tmax      = config.tmax,
        baseline  = baseline,
        preload   = True,
        verbose   = False,
    )

    # Drop flat epochs (all-zero channels or saturated amplifier channels).
    # This avoids NaN / ±Inf in the scaled features.
    epochs.drop_bad(verbose=False)

    if len(epochs) == 0:
        raise ValueError(
            "All epochs were dropped after artifact rejection. "
            "The data may contain saturated channels or severe baseline drift."
        )

    # Binary labels aligned to epochs.events (post-drop order).
    # class_a → 0 (negative class), class_b → 1 (positive class).
    y: np.ndarray = (epochs.events[:, 2] == config.class_b).astype(int)

    # Validate that both classes are still present after dropping.
    if len(np.unique(y)) < 2:
        present = int(np.unique(y)[0])
        missing = config.class_a if present == 1 else config.class_b
        raise ValueError(
            f"After artifact rejection, only event code {present} remains. "
            f"Event code {missing} has no surviving epochs. "
            "Try widening the epoch window or relaxing artifact thresholds."
        )

    return epochs, y


def _build_sliding_decoder(config: MVPAConfig) -> SlidingEstimator:
    """
    Construct the SlidingEstimator that runs an inner sklearn pipeline at
    each time point independently.

    Inner pipeline architecture (receives 2-D slices from SlidingEstimator):
    ─────────────────────────────────────────────────────────────────────────
    Step 1 — Vectorizer
      In the SlidingEstimator context SlidingEstimator already reduces the
      3-D input from (n_epochs, n_channels, n_times) to 2-D slices of shape
      (n_epochs, n_channels), one per time point.  For a 2-D array Vectorizer
      is a pass-through (flattens to the same shape it received).

      It is included here because the SAME inner pipeline can be used for
      full spatiotemporal decoding (without SlidingEstimator) where the input
      remains 3-D: (n_epochs, n_channels, n_times) → Vectorizer →
      (n_epochs, n_channels × n_times).  Keeping Vectorizer in makes the
      pipeline portable between both decoding modes.

    Step 2 — StandardScaler
      z-scores each feature (channel) independently: subtracts the per-feature
      mean and divides by the per-feature std computed on the *training* fold.
      The same scaling parameters are applied to the test fold (no leakage).
      Why: EEG/MEG channels record in different units (µV, fT, fT/cm) and at
      different absolute amplitudes; z-scoring puts them on the same scale so
      the L2 penalty in LogisticRegression penalises all features equally.

    Step 3 — LogisticRegression (solver='liblinear', penalty='l2')
      Linear binary classifier with L2 regularisation.
      Why linear: (a) n_channels is small compared to n_epochs, so the
      decision boundary is likely low-dimensional; (b) the weight vector is
      a *spatial filter* and can be back-projected to channel space for
      neuroimaging interpretation; (c) fast training enables running one
      classifier per time point per cross-validation fold.
      Why liblinear: best solver for small datasets; no mini-batches or
      convergence issues that lbfgs/saga can show on <100 samples.

    SlidingEstimator wrapper
    ────────────────────────
    SlidingEstimator(inner_clf, scoring='roc_auc') iterates the last axis of
    its 3-D input X, fitting/scoring inner_clf independently at each t.
    Output of transform() or predict_proba() has shape (n_epochs, n_times).
    cross_val_multiscore() calls it n_folds times and returns (n_folds, n_times).
    """
    inner_clf = make_pipeline(
        Vectorizer(),                                       # 3-D→2-D safety step (see above)
        StandardScaler(),                                   # z-score per channel per fold
        LogisticRegression(
            solver   = "liblinear",   # best for small datasets (< 10 000 samples)
            penalty  = "l2",          # L2 is standard for spatial filter decoding
            C        = config.C,      # inverse regularisation: 1.0 is a safe default
            max_iter = config.max_iter,
        ),
    )

    return SlidingEstimator(
        inner_clf,
        n_jobs  = 1,          # parallelise at the cross-validation level instead
        scoring = "roc_auc",  # threshold-free, imbalance-robust AUC metric
        verbose = False,
    )


def _cross_validate(
    decoder: SlidingEstimator,
    X: np.ndarray,
    y: np.ndarray,
    config: MVPAConfig,
) -> np.ndarray:
    """
    Run k-fold cross-validation and return the per-fold, per-timepoint AUC matrix.

    Uses mne.decoding.cross_val_multiscore instead of sklearn's cross_val_score
    because:
      • cross_val_score wraps sklearn's _fit_and_score which cannot handle 3-D
        arrays.  It would require custom splitter logic to extract folds from X.
      • cross_val_multiscore is designed for MNE's multi-output estimators
        (SlidingEstimator, GeneralizingEstimator) and returns an (n_folds, n_times)
        array directly.
      • Internally it uses StratifiedKFold, which preserves the class ratio in
        each fold (essential for binary decoding to avoid undefined AUC folds).

    Parameters
    ----------
    decoder : SlidingEstimator wrapping the inner pipeline.
    X       : (n_epochs, n_channels, n_times) — the full epoch array.
    y       : (n_epochs,) — binary class labels {0, 1}.
    config  : Pipeline configuration (n_folds).

    Returns
    -------
    scores : np.ndarray of shape (n_folds, n_times), dtype float64.
             Each entry is the AUC of the classifier at a specific (fold, time).
    """
    scores: np.ndarray = cross_val_multiscore(
        decoder,
        X,
        y,
        cv     = config.n_folds,   # int → StratifiedKFold(n_splits=n_folds)
        n_jobs = 1,                # no parallelism: avoids pickling errors on Windows
    )
    # scores.shape == (n_folds, n_times)
    return scores
