"""
main.py — Neuroimaging Data Bridge
====================================
FastAPI application that wraps MNE-Python to serve MEG (.fif) data to the
NeuroViewer TypeScript frontend over HTTP.

Architecture overview
─────────────────────
  Browser                      FastAPI                MNE / filesystem
  ───────                      ───────                ────────────────
  POST /api/load-meg   ──────► store Raw in memory ─► mne.read_raw_fif()
  GET  /api/meg/metadata ────► return channel list ──► raw.info
  GET  /api/meg/channels ────► slice + decimate ──────► raw.get_data(tmin, tmax)
  GET  /api/meg/psd ─────────► Welch PSD ─────────────► raw.compute_psd()
  DELETE /api/sessions/{id} ► evict + delete tempfile

Session model
─────────────
A loaded mne.io.BaseRaw object is stored in a module-level dict keyed by a
random UUID (session_id). The session_id is returned to the frontend after
each successful load and is required for all data endpoints.

preload=False is used so multi-hundred-MB FIF files do not exhaust RAM on
load. MNE reads sample data lazily from the temp file when get_data() is
called.  The temp file is deleted when the session is destroyed.

Run with:
  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import base64
import io
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Annotated, Optional

import mne
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Shared session store (imported by name so all modules mutate the same dicts)
from session_store import (
    meg_sessions  as _sessions,
    meg_filenames as _filenames,
    meg_tempfiles as _tempfiles,
    meg_tempdirs  as _meg_tempdirs,
    eeg_sessions  as _eeg_sessions,
    eeg_filenames as _eeg_filenames,
    eeg_tempdirs  as _eeg_tempdirs,
)
# ── Analysis sub-routers
from mri_analysis              import router as _mri_router
from meg_analysis              import router as _meg_analysis_router
from source_localization       import router as _source_loc_router
from neural_decoding           import router as _decoding_router
from routers.localization      import router as _upload_loc_router
from routers.decoding          import router as _upload_decoding_router
from routers.tractography      import router as _tractography_router
from routers.anomaly_detection import router as _anomaly_router
from routers.longitudinal      import router as _longitudinal_router
from routers.registration      import router as _registration_router

# ── Optional heavy deps (segmentation only) ───────────────────────────────────
try:
    import nibabel as nib  # type: ignore
    _HAS_NIBABEL = True
except ImportError:
    _HAS_NIBABEL = False

try:
    import scipy.ndimage as _ndi  # type: ignore
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import onnxruntime as _ort  # type: ignore
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

# Path to the SynthSeg ONNX model (download via backend/download_models.py)
_SYNTHSEG_MODEL = Path(__file__).parent / "models" / "SynthSeg.onnx"

# SynthSeg class-index → FreeSurfer label ID mapping (33 output classes)
_SYNTHSEG_LABEL_MAP: list[int] = [
     0,   # 0  → background
     2,   # 1  → left cerebral white matter
     3,   # 2  → left cerebral cortex
     4,   # 3  → left lateral ventricle
     5,   # 4  → left inferior lateral ventricle
     7,   # 5  → left cerebellum white matter
     8,   # 6  → left cerebellum cortex
    10,   # 7  → left thalamus
    11,   # 8  → left caudate
    12,   # 9  → left putamen
    13,   # 10 → left pallidum
    14,   # 11 → 3rd ventricle
    15,   # 12 → 4th ventricle
    16,   # 13 → brain stem
    17,   # 14 → left hippocampus
    18,   # 15 → left amygdala
    24,   # 16 → CSF
    26,   # 17 → left accumbens area
    28,   # 18 → left ventral DC
    41,   # 19 → right cerebral white matter
    42,   # 20 → right cerebral cortex
    43,   # 21 → right lateral ventricle
    44,   # 22 → right inferior lateral ventricle
    46,   # 23 → right cerebellum white matter
    47,   # 24 → right cerebellum cortex
    49,   # 25 → right thalamus
    50,   # 26 → right caudate
    51,   # 27 → right putamen
    52,   # 28 → right pallidum
    53,   # 29 → right hippocampus
    54,   # 30 → right amygdala
    58,   # 31 → right accumbens area
    60,   # 32 → right ventral DC
]
_LABEL_MAP_ARRAY = np.array(_SYNTHSEG_LABEL_MAP, dtype=np.uint8)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MEG Data Bridge",
    description="MNE-Python preprocessing engine for the NeuroViewer frontend",
    version="0.2.0",
)

# Allow the Vite dev server (and production build preview) to call this API.
# Adjust origins for your deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",   # vite preview
    ],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# ── Mount analysis sub-routers ────────────────────────────────────────────────
# These routers share the session dicts via session_store.py and add:
#   POST /api/mri/volumetrics    — tissue volumes from a SynthSeg label map
#   POST /api/meg/detect-artifacts — EOG + muscle artefact detection
#   POST /api/meg/detect-spikes    — epileptiform spike detection
#   POST /api/meg/frequency-bands  — Welch PSD band-power analysis
app.include_router(_mri_router)
app.include_router(_meg_analysis_router)
app.include_router(_source_loc_router)        # POST /api/eeg/localize-session  (session-based)
app.include_router(_decoding_router)          # POST /api/eeg/decode-session    (session-based)
app.include_router(_upload_loc_router)        # POST /api/eeg/localize          (file-upload, full pipeline)
app.include_router(_upload_decoding_router)   # POST /api/eeg/decode            (file-upload, full pipeline)
app.include_router(_tractography_router)      # POST /api/dti/tractography      (file-upload, full pipeline)
app.include_router(_anomaly_router)           # POST /api/anomalies/detect      (file-upload, full pipeline)
app.include_router(_longitudinal_router)      # POST /api/longitudinal/delta    (two-file upload, registration + subtraction)
app.include_router(_registration_router)      # POST /api/registration/syn      (two-file upload, affine + SyN warp)

# Suppress verbose MNE console output
mne.set_log_level("WARNING")

# ── In-memory session store ───────────────────────────────────────────────────
# Maps session_id → (Raw object, temp-file path | None)
# For production: replace with Redis + shared worker pool.

# Session dicts are now imported from session_store (see imports above).
# This keeps meg_analysis.py and future router modules in sync without
# duplication.  The aliases (_sessions, _filenames, _tempfiles) bind to the
# same dict objects, so mutations here are immediately visible everywhere.


def _require(session_id: str) -> mne.io.BaseRaw:
    """Return the Raw object for session_id, or 404."""
    raw = _sessions.get(session_id)
    if raw is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. "
                   "POST /api/load-meg first.",
        )
    return raw


# ── Channel type / unit helpers ───────────────────────────────────────────────

# MNE channel type strings (from mne.channel_type) → physical unit
_TYPE_TO_UNIT: dict[str, str] = {
    "mag":       "T",
    "grad":      "T/m",
    "eeg":       "V",
    "eog":       "V",
    "ecg":       "V",
    "emg":       "V",
    "ref_meg":   "T",
    "misc":      "a.u.",
    "stim":      "a.u.",
    "bio":       "a.u.",
    "fnirs_cw_amplitude": "V",
    "fnirs_od":  "a.u.",
    "hbo":       "M",
    "hbr":       "M",
    "temperature": "°C",
    "gsr":       "S",
    "resp":      "%",
}


def _ch_type(raw: mne.io.BaseRaw, idx: int) -> str:
    """Return a short MNE channel-type string for channel index idx."""
    try:
        return mne.channel_type(raw.info, idx)
    except Exception:
        return "misc"


def _ch_unit(raw: mne.io.BaseRaw, idx: int) -> str:
    return _TYPE_TO_UNIT.get(_ch_type(raw, idx), "a.u.")


# ── Server-side min/max decimation ────────────────────────────────────────────

def decimate_minmax(
    data: np.ndarray,   # [n_channels, n_samples]
    n_out: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample data to n_out output buckets using a min/max envelope.

    For each bucket the minimum and maximum values across all raw samples
    inside that bucket are computed.  This guarantees that no transient
    spike is lost regardless of the compression ratio.

    Returns
    -------
    mins, maxs  each of shape [n_channels, n_out]
    """
    n_ch, n_samp = data.shape

    if n_samp <= n_out:
        return data.copy(), data.copy()

    # Build bucket boundary indices with uniform spacing
    bounds = np.linspace(0, n_samp, n_out + 1).astype(int)
    # Ensure monotonically increasing (can collapse for tiny windows)
    bounds = np.clip(bounds, 0, n_samp)

    actual_out = len(bounds) - 1
    mins = np.empty((n_ch, actual_out), dtype=np.float32)
    maxs = np.empty((n_ch, actual_out), dtype=np.float32)

    for i in range(actual_out):
        start, end = int(bounds[i]), int(bounds[i + 1])
        end = max(end, start + 1)               # guarantee at least 1 sample
        end = min(end, n_samp)
        bucket = data[:, start:end]
        mins[:, i] = bucket.min(axis=1)
        maxs[:, i] = bucket.max(axis=1)

    return mins, maxs


# ── Pydantic response models ──────────────────────────────────────────────────

class ChannelInfo(BaseModel):
    """Metadata for one recording channel."""
    name:  str = Field(description="Channel label, e.g. 'MEG 0111'")
    type:  str = Field(description="MNE type: 'mag', 'grad', 'eeg', 'eog', …")
    unit:  str = Field(description="Physical unit: 'T', 'T/m', 'V', …")
    index: int = Field(description="Zero-based channel index in the Raw object")


class MegMetadata(BaseModel):
    session_id:     str
    filename:       str
    sampling_rate:  float   = Field(description="Sampling frequency in Hz")
    total_duration: float   = Field(description="Recording length in seconds")
    n_samples:      int
    n_channels:     int
    channels:       list[ChannelInfo]


class ChannelTrace(BaseModel):
    """Decimated signal for one channel over a requested time window."""
    name:   str
    times:  list[float] = Field(description="Time axis in seconds (length = n_points)")
    values: list[float] = Field(description="Centre value ((min+max)/2) per bucket")
    mins:   list[float] = Field(description="Min envelope per bucket")
    maxs:   list[float] = Field(description="Max envelope per bucket")
    unit:   str


class ChannelDataResponse(BaseModel):
    session_id: str
    t_start:    float
    t_end:      float
    n_points:   int   = Field(description="Actual output points (≤ requested)")
    channels:   list[ChannelTrace]


class PsdChannel(BaseModel):
    name:   str
    psd_db: list[float] = Field(description="Power in dB (10·log10(unit²/Hz))")


class PsdResponse(BaseModel):
    session_id: str
    freqs:      list[float] = Field(description="Frequency axis in Hz")
    channels:   list[PsdChannel]
    n_fft:      int
    method:     str


class LoadResult(BaseModel):
    session_id:     str
    filename:       str
    n_channels:     int
    sampling_rate:  float
    total_duration: float


# ── Helper: pick the primary .fif from a split set ───────────────────────────

def _find_primary_fif(filenames: list[str]) -> str:
    """
    Return the filename that should be passed to read_raw_fif() from a list
    of .fif filenames.  The primary file has no split suffix, or the lowest
    split number when all files are continuation parts.

    Handles both BIDS-style  (_split-02_meg.fif)  and MNE-legacy-style  (-1.fif).
    """
    def _split_num(name: str) -> int:
        stem = Path(name).name
        m = re.search(r"[_-]split[_-](\d+)", stem, re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r"-(\d+)\.fif$", stem, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return 0  # no split indicator → this is the primary file

    return min(filenames, key=lambda n: _split_num(n))


# ── Endpoint: upload one or more .fif files ───────────────────────────────────

@app.post("/api/load-meg", response_model=LoadResult, summary="Upload one or more .fif MEG files")
async def load_meg(files: list[UploadFile] = File(...)) -> LoadResult:
    """
    Accept one or more .fif file uploads.  For split recordings, upload all
    parts together — they are written to a temp directory with their original
    names so MNE can locate continuation files via the paths stored in the
    FIF header.

    Returns a session_id for subsequent /api/meg/* calls.
    """
    if not files:
        raise HTTPException(400, "No files uploaded.")

    fif_uploads = [f for f in files if f.filename and f.filename.lower().endswith(".fif")]
    if not fif_uploads:
        raise HTTPException(400, "No .fif files found in the upload.")

    tmpdir = tempfile.mkdtemp(prefix="meg_")
    try:
        for upload in fif_uploads:
            content = await upload.read()
            if len(content) == 0:
                raise HTTPException(422, f"Uploaded file '{upload.filename}' is empty.")

            # Detect DataLad / git-annex pointer files.
            if len(content) < 1000:
                try:
                    snippet = content[:256].decode("utf-8", errors="ignore")
                    if "/annex/objects/" in snippet or snippet.startswith("../") or snippet.startswith("/annex/"):
                        raise HTTPException(
                            422,
                            f"'{upload.filename}' is a DataLad/git-annex pointer, not actual MEG data. "
                            "Run 'datalad get <filename>' (or 'git annex get <filename>') "
                            "to download the file content first.",
                        )
                except HTTPException:
                    raise
                except Exception:
                    pass

            out_path = os.path.join(tmpdir, Path(upload.filename).name)
            with open(out_path, "wb") as fh:
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())

        # Pick the primary file (no split suffix, or lowest split number)
        primary_name = _find_primary_fif([f.filename for f in fif_uploads])  # type: ignore[arg-type]
        primary_path = os.path.join(tmpdir, Path(primary_name).name)

        raw = None
        last_exc: Exception | None = None
        for kwargs in [
            {"preload": False, "verbose": False},
            {"preload": False, "verbose": False, "allow_maxshield": True},
        ]:
            try:
                raw = mne.io.read_raw_fif(primary_path, **kwargs)
                break
            except Exception as exc:
                last_exc = exc

        if raw is None:
            msg = str(last_exc)
            hint = ""
            if "does not start with a file id tag" in msg:
                hint = (
                    " — make sure you are uploading a raw MEG acquisition file "
                    "(*_meg.fif), not an epochs, evoked, or forward-solution file."
                )
            raise HTTPException(422, f"MNE could not parse the file: {msg}{hint}") from last_exc

        sid = str(uuid.uuid4())
        _sessions[sid]     = raw
        _filenames[sid]    = primary_name
        _meg_tempdirs[sid] = tmpdir  # cleaned up by DELETE /api/sessions/{id}

        return LoadResult(
            session_id=sid,
            filename=primary_name,
            n_channels=len(raw.ch_names),
            sampling_rate=float(raw.info["sfreq"]),
            total_duration=float(raw.times[-1]),
        )

    except HTTPException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(500, f"Unexpected error loading MEG: {exc}") from exc


# ── Endpoint: load by filesystem path (local dev shortcut) ───────────────────

@app.post(
    "/api/load-meg-path",
    response_model=LoadResult,
    summary="Load a .fif file by absolute server-side path",
)
async def load_meg_path(path: Annotated[str, Form()]) -> LoadResult:
    """
    Load a .fif file that is already on the server's filesystem.
    Useful during local development when the OpenNeuro dataset is
    mounted at a known path.

    Unlike /api/load-meg, this endpoint does NOT copy the file; MNE
    reads directly from the provided path, so split .fif continuations
    (raw-1.fif, raw-2.fif, …) are found automatically.
    """
    fif_path = Path(path)
    if not fif_path.exists():
        raise HTTPException(404, f"File not found: {path}")
    if fif_path.suffix.lower() != ".fif":
        raise HTTPException(400, "Only .fif files are accepted.")

    try:
        raw = mne.io.read_raw_fif(str(fif_path), preload=False, verbose=False)
    except Exception as exc:
        raise HTTPException(422, f"MNE could not parse the file: {exc}") from exc

    sid = str(uuid.uuid4())
    _sessions[sid]  = raw
    _filenames[sid]  = fif_path.name
    _tempfiles[sid]  = None   # not a temp copy; don't delete on session close

    return LoadResult(
        session_id=sid,
        filename=fif_path.name,
        n_channels=len(raw.ch_names),
        sampling_rate=float(raw.info["sfreq"]),
        total_duration=float(raw.times[-1]),
    )


# ── KIT coregistration helpers ───────────────────────────────────────────────

def _parse_kit_pos(pos_path: str) -> list[tuple[float, float, float]]:
    """
    Parse a KIT/Polhemus .pos digitisation file.

    Each data line contains 3 or more whitespace-separated numbers; any leading
    index column and trailing comments are tolerated by always taking the *last*
    three numeric tokens on a line.  Comment lines beginning with %, //, #, or !
    are skipped.

    Units are auto-detected and converted to metres:
      values > 10  → assumed millimetres  (×0.001)
      values > 1   → assumed centimetres  (×0.01)
      else         → assumed metres       (×1)
    """
    points: list[tuple[float, float, float]] = []
    with open(pos_path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped[0] in ("%", "/", "#", "!"):
                continue
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[-3]), float(parts[-2]), float(parts[-1])
                    points.append((x, y, z))
                except ValueError:
                    continue

    if not points:
        return points

    max_val = max(abs(v) for p in points for v in p)
    scale = 0.001 if max_val > 10.0 else (0.01 if max_val > 1.0 else 1.0)
    return [(x * scale, y * scale, z * scale) for x, y, z in points]


def _apply_kit_coregistration(
    raw: "mne.io.BaseRaw",
    pos_path: str,
    n_hpi: int = 5,
) -> None:
    """
    Parse a KIT .pos file and inject head digitisation into raw.info['dig'].

    Uses MNE's make_dig_montage API so we never touch the ELP/HSP file format
    (which varies between MNE versions and causes parser failures).

    KIT digitisation convention:
      • first 3 points  — nasion, LPA, RPA
      • next n_hpi       — HPI coils
      • remainder        — head-surface point cloud
    """
    import numpy as np

    points = _parse_kit_pos(pos_path)
    if len(points) < 3:
        raise ValueError(
            f"Only {len(points)} numeric points in {Path(pos_path).name}; "
            "need ≥ 3 (nasion, LPA, RPA)."
        )

    nasion = np.array(points[0])
    lpa    = np.array(points[1])
    rpa    = np.array(points[2])

    actual_hpi = min(n_hpi, len(points) - 3)
    hpi_arr    = np.array(points[3 : 3 + actual_hpi]) if actual_hpi > 0 else None
    surf_pts   = points[3 + actual_hpi :]
    hsp_arr    = np.array(surf_pts) if surf_pts else None

    montage = mne.channels.make_dig_montage(
        nasion=nasion, lpa=lpa, rpa=rpa,
        hsp=hsp_arr, hpi=hpi_arr,
        coord_frame="head",
    )

    # raw.info may be locked in MNE ≥ 1.0 — unlock if the method exists
    try:
        with raw.info._unlock():
            raw.info["dig"] = montage.dig
    except AttributeError:
        raw.info["dig"] = montage.dig


# ── Endpoint: upload KIT/Yokogawa file set (.con + optional .mrk / .pos) ────

@app.post(
    "/api/load-meg-kit",
    response_model=LoadResult,
    summary="Upload a KIT/Yokogawa MEG file set (.con + .mrk + .pos)",
)
async def load_meg_kit(
    con_file: UploadFile            = File(...,  description="Raw KIT data file (.con)"),
    mrk_file: UploadFile | None     = File(None, description="HPI marker file (.mrk) — optional but recommended"),
    pos_file: UploadFile | None     = File(None, description="Sensor position file (.pos) — optional"),
) -> LoadResult:
    """
    Load a KIT/Yokogawa MEG recording from its native file set:

      .con  — raw continuous MEG data (required)
      .mrk  — head-position indicator (HPI) coil positions used for
              head-to-device coregistration (strongly recommended)
      .pos  — digitised electrode / sensor positions

    All files are written to a temporary directory so MNE's read_raw_kit()
    can find them by relative path.  The directory is cleaned up when the
    session is deleted via DELETE /api/sessions/{session_id}.
    """
    if not con_file.filename or not con_file.filename.lower().endswith(".con"):
        raise HTTPException(400, "con_file must be a .con file.")

    tmpdir = tempfile.mkdtemp(prefix="kit_")
    try:
        # Write the .con file
        con_content = await con_file.read()
        if not con_content:
            raise HTTPException(422, "The .con file is empty.")
        con_path = os.path.join(tmpdir, Path(con_file.filename).name)
        with open(con_path, "wb") as fh:
            fh.write(con_content)

        # Write optional .mrk file
        mrk_path: str | None = None
        if mrk_file and mrk_file.filename:
            mrk_content = await mrk_file.read()
            if mrk_content:
                mrk_path = os.path.join(tmpdir, Path(mrk_file.filename).name)
                with open(mrk_path, "wb") as fh:
                    fh.write(mrk_content)

        # Write optional .pos file (previously discarded; now used for coregistration)
        pos_path: str | None = None
        if pos_file and pos_file.filename:
            pos_content = await pos_file.read()
            if pos_content:
                pos_path = os.path.join(tmpdir, Path(pos_file.filename).name)
                with open(pos_path, "wb") as fh:
                    fh.write(pos_content)

        # Load raw, falling back through less-informative options:
        #   1. mrk only  — partial coregistration (rejected by some MNE versions)
        #   2. nothing   — waveforms only
        # Digitisation from .pos is applied AFTER loading via make_dig_montage so
        # we never touch the ELP/HSP file format (which varies across MNE versions).
        attempts: list[dict] = []
        if mrk_path:
            attempts.append({"preload": False, "verbose": False, "mrk": mrk_path})
        attempts.append({"preload": False, "verbose": False})

        raw: mne.io.BaseRaw | None = None
        last_kit_exc: Exception | None = None
        for kwargs in attempts:
            try:
                raw = mne.io.read_raw_kit(con_path, **kwargs)
                break
            except Exception as exc:
                last_kit_exc = exc
                if "need to be provided as a group" not in str(exc):
                    break

        if raw is None:
            raise HTTPException(422, f"MNE could not parse the KIT files: {last_kit_exc}") from last_kit_exc

        # Inject head digitisation from .pos into raw.info (non-fatal)
        if pos_path:
            try:
                _apply_kit_coregistration(raw, pos_path)
            except Exception:
                pass

        sid = str(uuid.uuid4())
        _sessions[sid]      = raw
        _filenames[sid]     = con_file.filename
        _tempfiles[sid]     = None          # temp dir tracked separately below
        _meg_tempdirs[sid]  = tmpdir

        return LoadResult(
            session_id=sid,
            filename=con_file.filename,
            n_channels=len(raw.ch_names),
            sampling_rate=float(raw.info["sfreq"]),
            total_duration=float(raw.times[-1]),
        )

    except HTTPException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(500, f"Unexpected error loading KIT data: {exc}") from exc


# ── Endpoint: channel metadata ────────────────────────────────────────────────

@app.get(
    "/api/meg/metadata",
    response_model=MegMetadata,
    summary="Full channel list and recording metadata",
)
def get_metadata(session_id: str = Query(...)) -> MegMetadata:
    """
    Returns the channel inventory (name, type, unit, index) together with
    the global recording parameters (fs, duration, n_samples).

    The frontend calls this once after a successful load to populate:
      • The channel-list sidebar (grouped by type: mag / grad / eog / ecg)
      • The time-window slider maximum
      • The amplitude-scale auto-initialiser
    """
    raw = _require(session_id)

    channels = [
        ChannelInfo(
            name=raw.ch_names[i],
            type=_ch_type(raw, i),
            unit=_ch_unit(raw, i),
            index=i,
        )
        for i in range(len(raw.ch_names))
    ]

    return MegMetadata(
        session_id=session_id,
        filename=_filenames.get(session_id, "unknown"),
        sampling_rate=float(raw.info["sfreq"]),
        total_duration=float(raw.times[-1]),
        n_samples=len(raw.times),
        n_channels=len(raw.ch_names),
        channels=channels,
    )


# ── Endpoint: chunked + decimated channel data ────────────────────────────────

@app.get(
    "/api/meg/channels",
    response_model=ChannelDataResponse,
    summary="Downsampled channel timeseries for a time window",
)
def get_channels(
    session_id: str       = Query(...),
    channels:   list[str] = Query(..., description="Channel names to fetch"),
    t_start:    float     = Query(0.0,  description="Window start in seconds"),
    t_end:      float     = Query(30.0, description="Window end in seconds"),
    n_points:   int       = Query(800,  description="Target output points (≈ canvas px width)"),
) -> ChannelDataResponse:
    """
    Returns a min/max envelope for the requested channels within the time
    window [t_start, t_end], decimated to n_points output buckets.

    How to choose n_points
    ──────────────────────
    Set n_points ≈ Math.floor(canvasElement.clientWidth * devicePixelRatio).
    The server will never send more samples than the canvas can display,
    keeping each response under ~100 KB for 10 channels × 800 points.

    Response fields per channel
    ───────────────────────────
    times  : uniform time axis [t_start … t_end], length n_points
    values : (mins + maxs) / 2  — use for single-line rendering
    mins   : minimum per bucket — use for ribbon lower edge
    maxs   : maximum per bucket — use for ribbon upper edge
    unit   : physical unit string

    All arrays are plain JSON lists of 32-bit floats.
    """
    raw = _require(session_id)

    # Validate
    unknown = [ch for ch in channels if ch not in raw.ch_names]
    if unknown:
        raise HTTPException(400, f"Unknown channels: {unknown}")

    t_end_max = float(raw.times[-1])
    t_start   = float(max(0.0,         t_start))
    t_end     = float(min(t_end_max,   t_end))
    if t_start >= t_end:
        raise HTTPException(400, "t_start must be strictly less than t_end.")

    n_points = int(max(2, min(n_points, 8192)))   # safety clamp

    # Resolve channel indices (ordered=True preserves request order)
    picks = mne.pick_channels(raw.ch_names, include=channels, ordered=True)

    # Read the data slice from disk (fast even for large files)
    data, _ = raw.get_data(
        picks=picks,
        tmin=t_start,
        tmax=t_end,
        return_times=True,
        verbose=False,
    )
    # data: float64 [n_channels, n_samples]

    # Server-side min/max decimation → two float32 arrays [n_ch, n_out]
    mins, maxs = decimate_minmax(data.astype(np.float32), n_points)
    n_out = mins.shape[1]

    # Uniform time axis for the output buckets
    out_times: list[float] = np.linspace(t_start, t_end, n_out).tolist()

    # Build per-channel traces
    ch_unit = {raw.ch_names[p]: _ch_unit(raw, p) for p in picks}
    traces  = [
        ChannelTrace(
            name=channels[i],
            times=out_times,
            values=((mins[i] + maxs[i]) / 2).tolist(),
            mins=mins[i].tolist(),
            maxs=maxs[i].tolist(),
            unit=ch_unit.get(channels[i], "a.u."),
        )
        for i in range(len(channels))
    ]

    return ChannelDataResponse(
        session_id=session_id,
        t_start=t_start,
        t_end=t_end,
        n_points=n_out,
        channels=traces,
    )


# ── Endpoint: Welch Power Spectral Density ────────────────────────────────────

@app.get(
    "/api/meg/psd",
    response_model=PsdResponse,
    summary="Welch PSD for selected channels",
)
def get_psd(
    session_id: str            = Query(...),
    channels:   list[str]      = Query(...),
    fmin:       float          = Query(1.0,   description="Minimum frequency in Hz"),
    fmax:       float          = Query(100.0, description="Maximum frequency in Hz"),
    n_fft:      int            = Query(2048,  description="FFT window length (samples)"),
    t_start:    Optional[float] = Query(None, description="Analysis segment start (None = full)"),
    t_end:      Optional[float] = Query(None, description="Analysis segment end   (None = full)"),
) -> PsdResponse:
    """
    Computes Welch PSD using mne.io.BaseRaw.compute_psd() and returns
    power in dB (10 · log10(unit²/Hz)) on a linear frequency axis.

    Brain-rhythm reference bands
    ────────────────────────────
    δ   0.5–4 Hz      θ   4–8 Hz
    α   8–13 Hz       β  13–30 Hz
    γ  30–100 Hz      HFO 100–600 Hz (MEG-specific)

    Use t_start/t_end to restrict PSD to the currently visible time window,
    which avoids re-processing the entire recording on every pan event.
    """
    raw = _require(session_id)

    unknown = [ch for ch in channels if ch not in raw.ch_names]
    if unknown:
        raise HTTPException(400, f"Unknown channels: {unknown}")

    picks = mne.pick_channels(raw.ch_names, include=channels, ordered=True)

    psd_kwargs: dict = dict(
        method="welch",
        fmin=float(fmin),
        fmax=float(fmax),
        n_fft=int(n_fft),
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
        raise HTTPException(500, f"PSD computation failed: {exc}") from exc

    psd_data: np.ndarray = spectrum.get_data()   # [n_channels, n_freqs], unit²/Hz
    freqs:    np.ndarray = spectrum.freqs         # [n_freqs], Hz

    # Convert to decibels; add tiny floor to avoid log(0)
    psd_db = 10.0 * np.log10(np.maximum(psd_data, 1e-30))

    return PsdResponse(
        session_id=session_id,
        freqs=freqs.tolist(),
        channels=[
            PsdChannel(name=channels[i], psd_db=psd_db[i].tolist())
            for i in range(len(channels))
        ],
        n_fft=int(n_fft),
        method="welch",
    )


# ── Endpoint: 2-D topomap ─────────────────────────────────────────────────────

@app.get("/api/meg/topomap", summary="Render a 2-D sensor topomap for the current time window")
def get_topomap(
    session_id: str   = Query(...),
    t_start:    float = Query(..., description="Window start in seconds"),
    t_end:      float = Query(..., description="Window end in seconds"),
) -> dict:
    """
    Average the raw signal over [t_start, t_end], then render a 2-D topomap
    with MNE + Matplotlib (Agg backend) and return it as a base-64 PNG data URL.

    Magnetometers are preferred; falls back to all MEG channels if none exist.
    The plot uses the dark theme of the viewer.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = _require(session_id)

    # Pick magnetometers; fall back to all MEG
    raw_w = raw.copy()
    for pick_type in ("mag", "meg"):
        try:
            raw_w.pick(pick_type)
            break
        except Exception:
            continue
    else:
        raise HTTPException(422, "No MEG channels available for topomap.")

    try:
        start_i = raw_w.time_as_index(t_start)[0]
        stop_i  = raw_w.time_as_index(t_end)[0]
        if stop_i <= start_i:
            stop_i = start_i + 1
        data   = raw_w.get_data(start=start_i, stop=stop_i)  # (n_ch, n_t)
        values = data.mean(axis=1)                            # average over window
    except Exception as exc:
        raise HTTPException(500, f"Could not read MEG data: {exc}") from exc

    BG = "#12121f"
    try:
        fig, ax = plt.subplots(figsize=(3.6, 3.6), facecolor=BG)
        ax.set_facecolor(BG)
        mne.viz.plot_topomap(
            values, raw_w.info,
            axes=ax, show=False,
            cmap="RdBu_r",
            res=128,
            contours=4,
            sensors=True,
            extrapolate="head",
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110,
                    bbox_inches="tight", facecolor=BG, edgecolor="none")
        plt.close(fig)
        buf.seek(0)
    except Exception as exc:
        plt.close("all")
        raise HTTPException(500, f"Topomap render failed: {exc}") from exc

    return {"image": "data:image/png;base64," + base64.b64encode(buf.read()).decode()}


# ── Session management endpoints ──────────────────────────────────────────────

@app.get("/api/sessions", summary="List active sessions (debug)")
def list_sessions() -> dict:
    return {
        "sessions": [
            {
                "session_id": sid,
                "filename":   _filenames.get(sid, "?"),
                "n_channels": len(_sessions[sid].ch_names),
            }
            for sid in _sessions
        ]
    }


@app.delete(
    "/api/sessions/{session_id}",
    summary="Delete session and free resources",
)
def delete_session(session_id: str) -> dict:
    """
    Remove the Raw object from memory and delete the associated temp file
    (if the session was created via file upload rather than /api/load-meg-path).
    Call this when the user closes the viewer or loads a new file.
    """
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found.")

    del _sessions[session_id]
    _filenames.pop(session_id, None)

    # Clean up single temp file (.fif upload)
    tmp = _tempfiles.pop(session_id, None)
    if tmp:
        Path(tmp).unlink(missing_ok=True)

    # Clean up temp directory (KIT file set)
    tmpdir = _meg_tempdirs.pop(session_id, None)
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return {"deleted": session_id}


# ── EEG (BrainVision) session store ──────────────────────────────────────────
# Separate from MEG sessions so their lifecycles don't interfere.

# EEG session dicts are also imported from session_store (see imports above).


def _require_eeg(session_id: str) -> mne.io.BaseRaw:
    raw = _eeg_sessions.get(session_id)
    if raw is None:
        raise HTTPException(
            404,
            f"EEG session '{session_id}' not found. POST /api/load-eeg first.",
        )
    return raw


# ── Endpoint: upload BrainVision EEG file set ─────────────────────────────────

@app.post("/api/load-eeg", response_model=LoadResult, summary="Upload BrainVision EEG files")
async def load_eeg(files: list[UploadFile] = File(...)) -> LoadResult:
    """
    Accept a BrainVision EEG file set (.vhdr + .eeg + .vmrk) and load with MNE.
    All three files should be uploaded together; MNE's BrainVision reader
    discovers the .eeg and .vmrk automatically from the .vhdr's references.

    Returns a session_id for subsequent /api/eeg/* calls.
    """
    if not files:
        raise HTTPException(400, "No files uploaded.")

    vhdr_upload = next(
        (f for f in files if f.filename and f.filename.lower().endswith(".vhdr")),
        None,
    )
    if vhdr_upload is None:
        raise HTTPException(
            400,
            "No .vhdr header file found. Upload the .vhdr, .eeg, and .vmrk files together.",
        )

    # Write all uploaded files to a persistent temp directory so MNE can find
    # the .eeg data file and .vmrk markers via the paths in the .vhdr header.
    tmpdir = tempfile.mkdtemp(prefix="eeg_")
    try:
        for upload in files:
            if not upload.filename:
                continue
            content = await upload.read()
            if len(content) == 0:
                raise HTTPException(422, f"Uploaded file '{upload.filename}' is empty.")
            out_path = os.path.join(tmpdir, Path(upload.filename).name)
            with open(out_path, "wb") as fh:
                fh.write(content)

        vhdr_path = os.path.join(tmpdir, Path(vhdr_upload.filename).name)

        try:
            # preload=False: MNE reads lazily from tmpdir for large .eeg files.
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose=False)
        except Exception as exc:
            raise HTTPException(422, f"MNE could not parse the EEG files: {exc}") from exc

        sid = str(uuid.uuid4())
        _eeg_sessions[sid]  = raw
        _eeg_filenames[sid] = vhdr_upload.filename
        _eeg_tempdirs[sid]  = tmpdir

        return LoadResult(
            session_id=sid,
            filename=vhdr_upload.filename,
            n_channels=len(raw.ch_names),
            sampling_rate=float(raw.info["sfreq"]),
            total_duration=float(raw.times[-1]),
        )

    except HTTPException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(500, f"Unexpected error loading EEG: {exc}") from exc


# ── Endpoint: EEG channel metadata ───────────────────────────────────────────

@app.get("/api/eeg/metadata", response_model=MegMetadata, summary="EEG channel list and metadata")
def get_eeg_metadata(session_id: str = Query(...)) -> MegMetadata:
    raw = _require_eeg(session_id)
    channels = [
        ChannelInfo(name=raw.ch_names[i], type=_ch_type(raw, i), unit=_ch_unit(raw, i), index=i)
        for i in range(len(raw.ch_names))
    ]
    return MegMetadata(
        session_id=session_id,
        filename=_eeg_filenames.get(session_id, "unknown"),
        sampling_rate=float(raw.info["sfreq"]),
        total_duration=float(raw.times[-1]),
        n_samples=len(raw.times),
        n_channels=len(raw.ch_names),
        channels=channels,
    )


# ── Endpoint: EEG decimated channel data ─────────────────────────────────────

@app.get("/api/eeg/channels", response_model=ChannelDataResponse, summary="EEG channel timeseries")
def get_eeg_channels(
    session_id: str       = Query(...),
    channels:   list[str] = Query(...),
    t_start:    float     = Query(0.0),
    t_end:      float     = Query(30.0),
    n_points:   int       = Query(800),
) -> ChannelDataResponse:
    raw = _require_eeg(session_id)

    unknown = [ch for ch in channels if ch not in raw.ch_names]
    if unknown:
        raise HTTPException(400, f"Unknown channels: {unknown}")

    t_end_max = float(raw.times[-1])
    t_start   = float(max(0.0,       t_start))
    t_end     = float(min(t_end_max, t_end))
    if t_start >= t_end:
        raise HTTPException(400, "t_start must be strictly less than t_end.")

    n_points = int(max(2, min(n_points, 8192)))

    picks = mne.pick_channels(raw.ch_names, include=channels, ordered=True)
    data, _ = raw.get_data(
        picks=picks, tmin=t_start, tmax=t_end, return_times=True, verbose=False,
    )

    mins, maxs = decimate_minmax(data.astype(np.float32), n_points)
    n_out = mins.shape[1]
    out_times: list[float] = np.linspace(t_start, t_end, n_out).tolist()

    ch_unit = {raw.ch_names[p]: _ch_unit(raw, p) for p in picks}
    traces = [
        ChannelTrace(
            name=channels[i],
            times=out_times,
            values=((mins[i] + maxs[i]) / 2).tolist(),
            mins=mins[i].tolist(),
            maxs=maxs[i].tolist(),
            unit=ch_unit.get(channels[i], "a.u."),
        )
        for i in range(len(channels))
    ]

    return ChannelDataResponse(
        session_id=session_id,
        t_start=t_start,
        t_end=t_end,
        n_points=n_out,
        channels=traces,
    )


# ── Endpoint: delete EEG session ──────────────────────────────────────────────

@app.delete("/api/eeg/{session_id}", summary="Delete EEG session and free resources")
def delete_eeg_session(session_id: str) -> dict:
    if session_id not in _eeg_sessions:
        raise HTTPException(404, "EEG session not found.")
    del _eeg_sessions[session_id]
    _eeg_filenames.pop(session_id, None)
    tmpdir = _eeg_tempdirs.pop(session_id, None)
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return {"deleted": session_id}


# ── Segmentation endpoint ─────────────────────────────────────────────────────

class SegmentResult(BaseModel):
    """Label map returned by SynthSeg."""
    labels:      str        = Field(description="Base64-encoded uint8 flat array of FreeSurfer label IDs")
    dims:        list[int]  = Field(description="[x, y, z] dimensions of the label volume")
    affine:      list[float] = Field(description="4×4 RAS affine, row-major flattened (16 values)")
    n_labels:    int        = Field(description="Number of unique labels including background")
    duration_ms: float      = Field(description="Server-side wall-clock time in ms")


_SYNTHSEG_SIZE = 256  # SynthSeg v2 expects exactly 256×256×256 voxels


def _load_nifti(content: bytes):  # type: ignore[return]
    """Load a NIfTI image from raw bytes, falling back to a temp file."""
    try:
        return nib.Nifti1Image.from_bytes(content)  # type: ignore[attr-defined]
    except AttributeError:
        pass
    tmp = tempfile.NamedTemporaryFile(suffix=".nii", delete=False)
    try:
        tmp.write(content); tmp.flush(); tmp.close()
        return nib.load(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _crop_pad_256(arr: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Centre-crop or zero-pad each axis to exactly 256 voxels.
    Returns the adjusted array and the pad amounts applied (negative = crop).
    """
    N = _SYNTHSEG_SIZE
    offsets: list[tuple[int, int]] = []
    result = arr
    for axis, s in enumerate(arr.shape):
        diff = N - s
        if diff >= 0:
            # pad
            before = diff // 2
            after  = diff - before
            pad_width = [(0, 0)] * result.ndim
            pad_width[axis] = (before, after)
            result = np.pad(result, pad_width, mode="constant", constant_values=0)
            offsets.append((before, after))
        else:
            # crop
            start = (-diff) // 2
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(start, start + N)
            result = result[tuple(slices)]
            offsets.append((-start, -(s - (start + N))))  # negative = was cropped
    return result, offsets


def _uncrop_unpad(arr: np.ndarray, offsets: list[tuple[int, int]], orig_shape: tuple) -> np.ndarray:
    """Reverse _crop_pad_256: remove padding / restore cropped regions."""
    result = arr
    for axis, (before, _after) in enumerate(offsets):
        orig_s = orig_shape[axis]
        if before >= 0:
            # was padded → slice off the padding
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(before, before + orig_s)
            result = result[tuple(slices)]
        else:
            # was cropped → re-embed into a zero array
            start = -before
            out = np.zeros(
                [result.shape[i] if i != axis else orig_s for i in range(result.ndim)],
                dtype=result.dtype,
            )
            dst = [slice(None)] * result.ndim
            dst[axis] = slice(start, start + result.shape[axis])
            out[tuple(dst)] = result
            result = out
    return result


@app.post(
    "/api/segment",
    response_model=SegmentResult,
    summary="SynthSeg brain anatomy segmentation",
)
async def segment_volume(file: UploadFile = File(...)) -> SegmentResult:
    """
    Run SynthSeg v2 brain anatomy segmentation on a NIfTI volume.

    Prerequisites
    ─────────────
    pip install nibabel scipy onnxruntime
    Place SynthSeg.onnx in backend/models/

    Pipeline (matches SynthSeg v2 channels-last ONNX export)
    ─────────
    1. Load NIfTI with nibabel.
    2. Resample to 1 mm isotropic (scipy.ndimage.zoom, order=1).
    3. Normalise voxels to [0, 1].
    4. Crop / pad to exactly 256×256×256.
    5. Run ONNX inference:
         input  → [1, 256, 256, 256, 1]   float32  (channels-last)
         output → [1, 256, 256, 256, 33]  float32  (channels-last)
    6. argmax(axis=-1) → 33 class indices → FreeSurfer label IDs.
    7. Reverse crop/pad → resample back to original space (nearest).
    8. Return as base64 uint8 + dims + affine.
    """
    if not _HAS_NIBABEL:
        raise HTTPException(500, "nibabel not installed. Run: pip install nibabel")
    if not _HAS_SCIPY:
        raise HTTPException(500, "scipy not installed. Run: pip install scipy")
    if not _HAS_ONNX:
        raise HTTPException(500, "onnxruntime not installed. Run: pip install onnxruntime")
    if not _SYNTHSEG_MODEL.exists():
        raise HTTPException(
            500,
            f"SynthSeg model not found at {_SYNTHSEG_MODEL}. "
            "Place SynthSeg.onnx in backend/models/",
        )

    content = await file.read()
    if not content:
        raise HTTPException(422, "Uploaded file is empty.")

    t0 = time.perf_counter()

    # ── 1. Load NIfTI ─────────────────────────────────────────────────────
    try:
        nii = _load_nifti(content)
    except Exception as exc:
        raise HTTPException(422, f"Could not parse NIfTI: {exc}") from exc

    try:
        data: np.ndarray = np.asarray(nii.get_fdata(dtype=np.float32))
    except MemoryError:
        raise HTTPException(
            500,
            "Not enough memory to load this volume. Try a lower-resolution file.",
        )
    except Exception as exc:
        raise HTTPException(422, f"Could not read NIfTI data: {exc}") from exc

    affine: np.ndarray = nii.affine.astype(np.float64)

    # SynthSeg requires a 3-D volume.  For 4-D files (fMRI, multi-echo, DWI)
    # take the first volume rather than refusing outright.
    if data.ndim == 4:
        data = data[..., 0]
    elif data.ndim != 3:
        raise HTTPException(
            422,
            f"Expected a 3-D NIfTI volume; got shape {list(data.shape)}. "
            "SynthSeg only supports structural (T1/T2/FLAIR) scans.",
        )

    orig_shape = data.shape

    # ── 2. Resample to 1 mm isotropic ─────────────────────────────────────
    vox_mm = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))   # [dx, dy, dz] in mm

    # Sanity-check voxel sizes — nonsensical values (0, negative, > 50 mm)
    # indicate a corrupt or non-anatomical file.
    if not np.all((vox_mm > 0) & (vox_mm < 50)):
        raise HTTPException(
            422,
            f"Implausible voxel sizes: {vox_mm.tolist()} mm. "
            "Check that this is a structural MRI file.",
        )

    zoom_factors = tuple(float(v) for v in vox_mm)

    # Guard: estimate resampled size to avoid allocating gigabytes.
    estimated_nvox = int(np.prod([s * z for s, z in zip(data.shape, zoom_factors)]))
    if estimated_nvox > 512 ** 3:
        raise HTTPException(
            422,
            f"Resampled volume would be ~{estimated_nvox / 1e6:.0f} M voxels — too large. "
            "Re-export the scan at ≥ 1 mm isotropic resolution.",
        )

    try:
        if not np.allclose(vox_mm, 1.0, atol=0.05):
            resampled: np.ndarray = _ndi.zoom(data, zoom_factors, order=1)
        else:
            resampled = data
    except MemoryError:
        raise HTTPException(
            500,
            f"Out of memory while resampling {list(data.shape)} volume. "
            "Try a lower-resolution file.",
        )

    resampled_shape = resampled.shape

    # ── 3. Normalise to [0, 1] ────────────────────────────────────────────
    lo, hi = float(resampled.min()), float(resampled.max())
    norm: np.ndarray = (resampled - lo) / (hi - lo + 1e-9)

    # ── 4. Crop / pad to 256×256×256 ─────────────────────────────────────
    vol256, offsets = _crop_pad_256(norm)   # (256, 256, 256)

    # ── 5. ONNX inference (channels-last: [1, 256, 256, 256, 1]) ─────────
    inp = vol256[np.newaxis, ..., np.newaxis].astype(np.float32)   # [1,256,256,256,1]

    try:
        sess = _ort.InferenceSession(
            str(_SYNTHSEG_MODEL),
            providers=["CPUExecutionProvider"],
        )
        in_name  = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        out = sess.run([out_name], {in_name: inp})[0]  # [1, 256, 256, 256, 33]
    except MemoryError:
        raise HTTPException(
            500,
            "Out of memory during ONNX inference. "
            "The model output requires ~2.3 GB RAM — close other applications and retry.",
        )
    except Exception as exc:
        raise HTTPException(500, f"ONNX inference failed: {exc}") from exc

    # ── 6. argmax (last axis) → FreeSurfer label IDs ─────────────────────
    class_idx: np.ndarray = np.argmax(out[0], axis=-1).astype(np.int32)   # [256,256,256]
    class_idx = np.clip(class_idx, 0, len(_LABEL_MAP_ARRAY) - 1)
    labels_256: np.ndarray = _LABEL_MAP_ARRAY[class_idx]                  # uint8

    # ── 7. Reverse crop/pad → resample back to original space ─────────────
    labels_1mm = _uncrop_unpad(labels_256, offsets, resampled_shape)

    if not np.allclose(vox_mm, 1.0, atol=0.05):
        inv_zoom = tuple(s / r for s, r in zip(orig_shape, labels_1mm.shape))
        labels_orig: np.ndarray = _ndi.zoom(
            labels_1mm.astype(np.float32), inv_zoom, order=0,
        ).astype(np.uint8)
        # scipy.ndimage.zoom can return shape off by ±1 per axis due to
        # floating-point rounding (e.g. round(230 * 1.113) → 257 instead of 256).
        # A mismatch causes a different row stride between the overlay and the MRI
        # vtkImageData in vtk.js, producing diagonal stripe artefacts.
        if labels_orig.shape != orig_shape:
            canvas = np.zeros(orig_shape, dtype=np.uint8)
            clip = tuple(slice(0, min(s, o)) for s, o in zip(labels_orig.shape, orig_shape))
            canvas[clip] = labels_orig[clip]
            labels_orig = canvas
    else:
        labels_orig = labels_1mm.astype(np.uint8) if labels_1mm.dtype != np.uint8 else labels_1mm

    duration_ms = (time.perf_counter() - t0) * 1000.0

    # ── 8. Encode and return ──────────────────────────────────────────────
    # vtk.js vtkImageData stores voxels in Fortran order (X varies fastest,
    # i.e. the first numpy axis).  np.ndarray.tobytes() ignores the array's
    # own memory layout and uses the *order* argument to determine traversal;
    # it defaults to 'C' (last-axis-fastest) even for F-contiguous arrays.
    # np.asfortranarray(arr).tobytes() therefore still produces C-order bytes.
    # We must explicitly pass order='F' to get first-axis-fastest serialisation.
    labels_b64 = base64.b64encode(
        labels_orig.tobytes(order='F')
    ).decode("ascii")

    return SegmentResult(
        labels=labels_b64,
        dims=list(labels_orig.shape),
        affine=affine.flatten().tolist(),
        n_labels=int(np.unique(labels_orig).shape[0]),
        duration_ms=round(duration_ms, 1),
    )


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/api/health", summary="Liveness probe")
def health() -> dict:
    return {"status": "ok", "mne_version": mne.__version__}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,        # auto-restart on file changes during development
        log_level="info",
    )
