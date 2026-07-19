"""
routers/decoding.py — File-upload MVPA decoding endpoint
════════════════════════════════════════════════════════════════════════════════

POST /api/eeg/decode

Accepts a raw BrainVision EEG file set (multipart form-data) and two integer
event codes identifying the two cognitive conditions to compare.  Runs the
full time-resolved MVPA pipeline (epoching → SlidingEstimator → k-fold
cross-validation) and returns the decoding time-course as JSON.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGISTRATION IN main.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Add these two lines to backend/main.py:

    from routers.decoding import router as _upload_decoding_router
    app.include_router(_upload_decoding_router)   # POST /api/eeg/decode

NOTE: The existing neural_decoding.py also registers POST /api/eeg/decode
(session-based variant).  To avoid a route collision, rename that endpoint
to /api/eeg/decode-session by changing the @router.post decorator in
neural_decoding.py before mounting this router.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUEST FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Content-Type: multipart/form-data

  Files:
    vhdr_file   (.vhdr)  BrainVision header — required
    eeg_file    (.eeg or .dat)  binary data — required
    vmrk_file   (.vmrk)  marker / event log — required for decoding

  Form fields:
    class_a         int    event code for condition A (e.g. 1)
    class_b         int    event code for condition B (e.g. 2)
    tmin            float  epoch start in seconds, default -0.2
    tmax            float  epoch end in seconds, default 0.8
    baseline_end    float  pre-stimulus baseline end in seconds, default 0.0
    apply_baseline  bool   whether to apply baseline correction, default true
    n_folds         int    k-fold cross-validation splits, default 5
    C               float  LR regularisation (inverse strength), default 1.0

  Example (curl):
    curl -X POST http://localhost:8000/api/eeg/decode \
      -F "vhdr_file=@sub01.vhdr" \
      -F "eeg_file=@sub01.eeg"   \
      -F "vmrk_file=@sub01.vmrk" \
      -F "class_a=1" -F "class_b=2"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
  "times":             [−0.2, −0.19, …, 0.8],   // epoch time axis (seconds)
  "scores":            [0.51, 0.53, …, 0.82],   // mean AUC per time point
  "scores_std":        [0.04, 0.03, …, 0.07],   // ±1 std across folds
  "chance_level":      0.5,
  "n_epochs":          120,
  "n_epochs_class_a":  60,
  "n_epochs_class_b":  60,
  "n_channels":        64,
  "n_times":           101,
  "peak_score":        0.82,                    // highest mean AUC
  "peak_time_s":       0.17,                    // time of peak AUC
  "duration_ms":       4823.0
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BRAINVISION FILE FORMAT — WHY ALL THREE FILES?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BrainVision uses a three-file format:
  .vhdr  — plain-text header (amplifier settings, channel names, file paths)
  .eeg   — binary waveform data (big-endian int16 or float32 interleaved)
  .vmrk  — plain-text marker log (stimulus triggers and response codes)

The .vhdr contains relative paths to the other two files.  MNE's
read_raw_brainvision() resolves those paths against the .vhdr's parent
directory.  All three files must therefore be written to the same temporary
directory before MNE is called.

The marker (.vmrk) file is particularly important for decoding: it provides
the trigger timestamps that define the epoch onsets.  Without it, MNE cannot
extract events and the pipeline will fail at the epoching step.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NON-BLOCKING EXECUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The MNE + scikit-learn pipeline is CPU-bound and may take 30–120 s for a
typical EEG dataset (64 channels, 200 epochs, 5 folds).  Running it on the
FastAPI event loop would block ALL other requests during that time.

asyncio.to_thread() offloads the synchronous _run_pipeline() call to Python's
default ThreadPoolExecutor, allowing the event loop to remain responsive.
Thread safety: MNE and numpy/sklearn are safe to call from threads; only
global caches (if any) would need locking.

UploadFile read caveat: FastAPI's UploadFile is NOT thread-safe — its async
read() must be called on the event loop, not inside the thread.  All file
bytes are therefore read BEFORE asyncio.to_thread() is entered.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import mne
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from eeg_decoding import MVPAConfig, MVPAResult, run_time_resolved_mvpa

# ── FastAPI router ─────────────────────────────────────────────────────────────

router = APIRouter()

# ── Response model ─────────────────────────────────────────────────────────────

class DecodeResponse(BaseModel):
    """
    JSON body returned by POST /api/eeg/decode.

    All list fields are aligned: times[i], scores[i], and scores_std[i]
    correspond to the same time point in the epoch.

    Plotting recipe (frontend):
      x = times
      y = scores              ← plot as a line
      y_lo = scores - scores_std  }
      y_hi = scores + scores_std  } ← fill_between for ±1 std band
      Draw a horizontal dashed line at chance_level = 0.5.
    """

    # ── Decoding time-course ─────────────────────────────────────────────────

    times: list[float] = Field(
        description="Epoch time axis in seconds (length = n_times).  "
                    "Aligns with the scores and scores_std arrays.",
    )
    scores: list[float] = Field(
        description="Mean AUC across cross-validation folds at each time point.  "
                    "Range [0, 1]; values > 0.5 indicate above-chance decoding.",
    )
    scores_std: list[float] = Field(
        description="Standard deviation of AUC across folds at each time point.  "
                    "Use for the ±1 std shading band around the decoding curve.",
    )
    chance_level: float = Field(
        default=0.5,
        description="Theoretical chance AUC for binary classification (always 0.5).",
    )

    # ── Dataset summary ──────────────────────────────────────────────────────

    n_epochs: int = Field(description="Total number of accepted trials.")
    n_epochs_class_a: int = Field(description="Trials assigned to class A (event code class_a).")
    n_epochs_class_b: int = Field(description="Trials assigned to class B (event code class_b).")
    n_channels: int = Field(description="Number of EEG channels used.")
    n_times: int = Field(description="Number of time points in the epoch window.")

    # ── Peak statistics ──────────────────────────────────────────────────────

    peak_score: float = Field(
        description="Highest mean AUC value across the decoding time-course.",
    )
    peak_time_s: float = Field(
        description="Time (seconds) at which the peak AUC occurs.  "
                    "Interpretable as the latency of maximum discriminability.",
    )

    # ── Diagnostics ──────────────────────────────────────────────────────────

    duration_ms: float = Field(
        description="Total wall-clock time for the pipeline in milliseconds.",
    )


# ── Route handler ──────────────────────────────────────────────────────────────

@router.post(
    "/api/eeg/decode",
    response_model=DecodeResponse,
    summary="Time-resolved MVPA (SlidingEstimator) on uploaded BrainVision EEG",
    tags=["EEG Decoding"],
)
async def decode_mvpa(
    # ── File uploads ────────────────────────────────────────────────────────
    vhdr_file: UploadFile = File(
        ...,
        description="BrainVision header file (.vhdr).  Required.",
    ),
    eeg_file: UploadFile = File(
        ...,
        description="BrainVision binary data file (.eeg or .dat).  Required.",
    ),
    vmrk_file: UploadFile = File(
        ...,
        description="BrainVision marker file (.vmrk).  "
                    "Required: contains the event triggers for epoching.",
    ),

    # ── Experimental design ──────────────────────────────────────────────────
    class_a: int = Form(
        ...,
        description="Integer event code for condition A (e.g. 1).  "
                    "Must match a trigger value in the .vmrk file.",
    ),
    class_b: int = Form(
        ...,
        description="Integer event code for condition B (e.g. 2).  "
                    "Must be different from class_a.",
    ),

    # ── Epoching parameters ──────────────────────────────────────────────────
    tmin: float = Form(
        default=-0.2,
        description="Epoch start in seconds relative to event onset.  "
                    "Negative values capture pre-stimulus baseline.",
    ),
    tmax: float = Form(
        default=0.8,
        description="Epoch end in seconds relative to event onset.",
    ),
    baseline_end: float = Form(
        default=0.0,
        description="End of the pre-stimulus baseline window (seconds).  "
                    "The baseline is (tmin, baseline_end).  Ignored if apply_baseline=false.",
    ),
    apply_baseline: bool = Form(
        default=True,
        description="Apply mean-baseline correction using the interval [tmin, baseline_end].  "
                    "Set to false for data already baseline-corrected offline.",
    ),

    # ── Cross-validation ─────────────────────────────────────────────────────
    n_folds: int = Form(
        default=5,
        ge=2,
        le=20,
        description="Number of cross-validation folds (k in k-fold).  "
                    "5 is standard; use 3 if n_epochs_per_class < 15.",
    ),

    # ── Classifier ───────────────────────────────────────────────────────────
    C: float = Form(
        default=1.0,
        gt=0.0,
        description="Inverse regularisation strength for LogisticRegression.  "
                    "Larger = weaker regularisation.  1.0 is a safe default.",
    ),
) -> DecodeResponse:
    """
    Run a time-resolved MVPA decoding analysis on uploaded BrainVision EEG data.

    The endpoint:
    1. Reads all uploaded file bytes from the async stream (before threading).
    2. Validates file extensions.
    3. Writes the three BrainVision files to a temporary directory so MNE
       can resolve the relative paths in the .vhdr header.
    4. Offloads the synchronous MNE + scikit-learn pipeline to a thread pool
       via asyncio.to_thread() so the event loop is not blocked.
    5. Cleans up the temporary directory in a finally block.
    6. Returns the decoding time-course as a DecodeResponse.
    """
    # Validation: event codes must be distinct.
    if class_a == class_b:
        raise HTTPException(422, "class_a and class_b must be different event codes.")

    # Validation: epoch window must be valid.
    if tmin >= tmax:
        raise HTTPException(422, "tmin must be strictly less than tmax.")

    # Validation: file extensions.
    _require_extension(vhdr_file, ".vhdr", "vhdr_file")
    _require_extension(eeg_file,  (".eeg", ".dat"), "eeg_file")
    _require_extension(vmrk_file, ".vmrk", "vmrk_file")

    # ── Read all bytes on the event loop BEFORE entering the thread ──────────
    # UploadFile's async read() must be called on the event loop.  It is NOT
    # safe to call inside a thread pool.  Reading all bytes here avoids any
    # thread-safety issues with the file stream.
    try:
        vhdr_bytes = await vhdr_file.read()
        eeg_bytes  = await eeg_file.read()
        vmrk_bytes = await vmrk_file.read()
    except Exception as exc:
        raise HTTPException(500, f"Failed to read uploaded files: {exc}") from exc

    if not vhdr_bytes:
        raise HTTPException(422, "vhdr_file is empty.")
    if not eeg_bytes:
        raise HTTPException(422, "eeg_file is empty.")
    if not vmrk_bytes:
        raise HTTPException(422, "vmrk_file is empty.")

    # ── Build pipeline config from the validated form parameters ─────────────
    config = MVPAConfig(
        class_a        = class_a,
        class_b        = class_b,
        tmin           = tmin,
        tmax           = tmax,
        baseline_end   = baseline_end,
        apply_baseline = apply_baseline,
        n_folds        = n_folds,
        C              = C,
    )

    # ── Run the synchronous pipeline in a thread pool ────────────────────────
    # asyncio.to_thread() submits _run_pipeline() to the default
    # ThreadPoolExecutor and suspends this coroutine until the thread completes.
    # The event loop remains free to handle other HTTP requests during this time.
    try:
        result: MVPAResult = await asyncio.to_thread(
            _run_pipeline,
            vhdr_bytes,
            eeg_bytes,
            vmrk_bytes,
            vhdr_file.filename or "data.vhdr",
            eeg_file.filename  or "data.eeg",
            vmrk_file.filename or "data.vmrk",
            config,
        )
    except ValueError as exc:
        # Pipeline validation errors (insufficient epochs, etc.) → 422
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Decoding pipeline failed: {exc}") from exc

    return DecodeResponse(
        times            = result.times,
        scores           = result.scores,
        scores_std       = result.scores_std,
        chance_level     = result.chance_level,
        n_epochs         = result.n_epochs,
        n_epochs_class_a = result.n_epochs_class_a,
        n_epochs_class_b = result.n_epochs_class_b,
        n_channels       = result.n_channels,
        n_times          = result.n_times,
        peak_score       = result.peak_score,
        peak_time_s      = result.peak_time_s,
        duration_ms      = result.duration_ms,
    )


# ── Private helpers ────────────────────────────────────────────────────────────

def _run_pipeline(
    vhdr_bytes:  bytes,
    eeg_bytes:   bytes,
    vmrk_bytes:  bytes,
    vhdr_name:   str,
    eeg_name:    str,
    vmrk_name:   str,
    config:      MVPAConfig,
) -> MVPAResult:
    """
    Synchronous pipeline entry point — runs in a thread pool.

    1. Write the three BrainVision files to a temp directory.
    2. Load the raw data with MNE.
    3. Call run_time_resolved_mvpa() from eeg_decoding.py.
    4. Delete the temp directory in a finally block.

    The temp directory is created with tempfile.mkdtemp() and is unique per
    request.  Concurrent calls to this function are safe.
    """
    tmpdir = tempfile.mkdtemp(prefix="eeg_decode_")
    try:
        # Write all three files to the same directory so MNE can find them
        # via the relative paths stored inside the .vhdr header.
        vhdr_path = _write_bytes(tmpdir, vhdr_name, vhdr_bytes)
        _write_bytes(tmpdir, eeg_name,  eeg_bytes)
        _write_bytes(tmpdir, vmrk_name, vmrk_bytes)

        # Load the raw data with MNE.
        # preload=True is required here because the epoch extraction in
        # run_time_resolved_mvpa calls epochs.get_data(), which needs the
        # full raw waveform in memory.  The temp files can be deleted after
        # loading because all data is copied into the Raw object's buffer.
        try:
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
        except Exception as exc:
            raise ValueError(
                f"MNE could not parse the BrainVision files: {exc}.  "
                "Make sure the .vhdr, .eeg, and .vmrk files all belong to the same recording."
            ) from exc

        # Suppress MNE's verbose output for the compute-intensive steps.
        mne.set_log_level("ERROR")

        return run_time_resolved_mvpa(raw, config)

    finally:
        # Always remove the temp directory, even if an exception was raised.
        shutil.rmtree(tmpdir, ignore_errors=True)


def _write_bytes(directory: str, filename: str, content: bytes) -> str:
    """
    Write `content` to `directory / filename` and return the full path.

    Uses the basename of `filename` only (strips any path components the
    client may have included) to prevent directory traversal.
    """
    safe_name = Path(filename).name           # e.g. "sub01.vhdr" (no parent dir)
    full_path = os.path.join(directory, safe_name)
    with open(full_path, "wb") as fh:
        fh.write(content)
    return full_path


def _require_extension(
    upload: UploadFile,
    allowed: str | tuple[str, ...],
    field_name: str,
) -> None:
    """
    Raise HTTP 422 if the uploaded file's extension is not in `allowed`.

    `allowed` may be a single string (".vhdr") or a tuple (".eeg", ".dat").
    Comparison is case-insensitive.
    """
    fname = upload.filename or ""
    ext   = Path(fname).suffix.lower()
    if isinstance(allowed, str):
        allowed = (allowed,)
    if ext not in allowed:
        allowed_str = " or ".join(allowed)
        raise HTTPException(
            422,
            f"'{field_name}' must be a {allowed_str} file (got '{fname}').",
        )
