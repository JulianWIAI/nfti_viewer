"""
bids_events.py — BIDS _events.tsv parsing endpoint
════════════════════════════════════════════════════════

Parses a BIDS-compliant *_events.tsv file and returns a colour-annotated
event list that the frontend can use to overlay stimulus markers on the
MEG waveform canvas and drive the EventTimelineRibbon.

BIDS events.tsv schema (REQUIRED columns)
──────────────────────────────────────────
  onset        — stimulus onset in seconds from the start of the run.
  duration     — event duration in seconds (may be 0 for instantaneous events).

OPTIONAL columns used here:
  trial_type   — categorical label for the event class (e.g. "deviant_tone").
                 When absent every row is tagged "event".
  response_time — latency from onset to the subject's behavioural response
                 in seconds (NaN/n/a when the subject did not respond).

COLOUR ASSIGNMENT
──────────────────
Colours are assigned to trial_type values by sorting the unique labels
alphabetically and cycling through a fixed 12-colour palette.  Deterministic
ordering means the same .tsv always produces the same colour map, regardless
of the order in which trial types appear in the file.

ENDPOINTS
──────────
  POST /api/bids/events/upload
    Accepts a .tsv file via multipart form upload (browser drag-and-drop).

  GET /api/bids/events?path=<absolute_server_path>
    Reads a .tsv file that is already on the server's filesystem.
    Useful during local development when the BIDS dataset is mounted at a
    known path (e.g. /data/ds004482/sub-01/func/*_events.tsv).

Both endpoints return the same BidsEventsResponse JSON structure.
"""

from __future__ import annotations

import io
import os
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/bids", tags=["BIDS Events"])

# ── Colour palette ────────────────────────────────────────────────────────────

# 12 visually distinct colours chosen for good contrast on the dark (#12121f)
# radiological background.  Sorted psychophysically: warm hues first, then
# cool, then neutral — making adjacent trial types easy to distinguish.
_EVENT_PALETTE: list[str] = [
    "#ef5350",   #  0 — vivid red          (deviant / rare stimuli)
    "#42a5f5",   #  1 — azure blue         (standard / frequent stimuli)
    "#66bb6a",   #  2 — medium green       (target / go trials)
    "#ffca28",   #  3 — amber              (cue / warning)
    "#ab47bc",   #  4 — purple             (oddball / violation)
    "#26c6da",   #  5 — cyan               (probe / catch)
    "#ff7043",   #  6 — deep orange        (response / button press)
    "#26a69a",   #  7 — teal               (feedback)
    "#d4e157",   #  8 — lime               (rest / baseline)
    "#ec407a",   #  9 — pink               (neutral filler)
    "#7e57c2",   # 10 — medium violet      (noise / artefact epoch)
    "#8d6e63",   # 11 — brown              (miscellaneous)
]


def _assign_colors(trial_types: list[str]) -> dict[str, str]:
    """
    Map each unique trial_type to a hex colour deterministically.

    Sorts trial_type labels alphabetically before assigning palette indices
    so the colour map is stable regardless of the order events appear in the file.
    """
    sorted_types = sorted(set(trial_types))
    return {
        tt: _EVENT_PALETTE[i % len(_EVENT_PALETTE)]
        for i, tt in enumerate(sorted_types)
    }


# ── Pydantic models ───────────────────────────────────────────────────────────


class ExperimentEventOut(BaseModel):
    """One stimulus event from the BIDS events.tsv, ready for the frontend."""

    # Unique row identifier — used by jumpToEvent() in SyncContext
    id: str = Field(..., description="Unique event identifier, e.g. 'evt_0042'")

    # Core BIDS fields
    onset: float = Field(..., description="Event onset in seconds from run start")
    duration: float = Field(..., description="Event duration in seconds (0 = instantaneous)")
    trial_type: str = Field(..., description="Categorical event label (e.g. 'deviant_tone')")

    # Optional BIDS field
    response_time: Optional[float] = Field(
        None,
        description="Seconds from onset to behavioural response; null when absent or n/a",
    )

    # Backend-assigned display colour
    color: str = Field(..., description="Hex colour string for this trial_type (e.g. '#ef5350')")


class BidsEventsResponse(BaseModel):
    """Full response from both events endpoints."""

    events: list[ExperimentEventOut] = Field(..., description="Parsed event list")
    trial_types: list[str] = Field(..., description="Sorted unique trial_type labels")
    color_map: dict[str, str] = Field(..., description="trial_type → hex colour mapping")
    n_events: int = Field(..., description="Total number of rows in the .tsv")
    total_duration: float = Field(
        ...,
        description="Max(onset + duration) across all events in seconds",
    )


# ── Core parser ───────────────────────────────────────────────────────────────


def _parse_tsv(content: bytes | str, filename: str = "events.tsv") -> BidsEventsResponse:
    """
    Parse a BIDS events.tsv byte payload and return a BidsEventsResponse.

    Steps
    ─────
    1. Read the TSV with pandas (tab-separated, recognises n/a and NaN as NA).
    2. Validate that the required columns (onset, duration) are present.
    3. Coerce numeric columns; fill missing trial_type with "event".
    4. Assign deterministic colours per trial_type.
    5. Build and return the response model.
    """
    # ── 1. Read ───────────────────────────────────────────────────────────────
    try:
        if isinstance(content, bytes):
            df = pd.read_csv(io.BytesIO(content), sep="\t", na_values=["n/a", "N/A", "nan", "NaN"])
        else:
            df = pd.read_csv(io.StringIO(content), sep="\t", na_values=["n/a", "N/A", "nan", "NaN"])
    except Exception as exc:
        raise HTTPException(422, f"Could not parse TSV: {exc}") from exc

    # ── 2. Validate required columns ──────────────────────────────────────────
    missing = [c for c in ("onset", "duration") if c not in df.columns]
    if missing:
        raise HTTPException(
            422,
            f"The TSV file is missing required BIDS columns: {missing}. "
            "A BIDS events.tsv must contain at least 'onset' and 'duration'.",
        )

    # ── 3. Coerce and fill ────────────────────────────────────────────────────
    df["onset"]    = pd.to_numeric(df["onset"],    errors="coerce").fillna(0.0)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)

    # trial_type: string, fall back to "event" for rows without a label
    if "trial_type" in df.columns:
        df["trial_type"] = df["trial_type"].fillna("event").astype(str)
    else:
        df["trial_type"] = "event"

    # response_time: optional float (seconds), NaN → None
    if "response_time" in df.columns:
        df["response_time"] = pd.to_numeric(df["response_time"], errors="coerce")
    else:
        df["response_time"] = np.nan

    # ── 4. Colour assignment ──────────────────────────────────────────────────
    trial_types = sorted(df["trial_type"].unique().tolist())
    color_map   = _assign_colors(trial_types)

    # ── 5. Build event list ───────────────────────────────────────────────────
    events: list[ExperimentEventOut] = []
    for row_idx, row in df.iterrows():
        rt_raw = row["response_time"]
        rt_val: Optional[float] = None if (pd.isna(rt_raw)) else float(rt_raw)

        events.append(
            ExperimentEventOut(
                id=f"evt_{row_idx:04d}",
                onset=float(row["onset"]),
                duration=float(row["duration"]),
                trial_type=str(row["trial_type"]),
                response_time=rt_val,
                color=color_map[str(row["trial_type"])],
            )
        )

    # Total duration: latest offset across all events
    offsets = df["onset"] + df["duration"]
    total_duration = float(offsets.max()) if len(offsets) > 0 else 0.0

    return BidsEventsResponse(
        events=events,
        trial_types=trial_types,
        color_map=color_map,
        n_events=len(events),
        total_duration=total_duration,
    )


# ── Endpoint: browser file upload ─────────────────────────────────────────────


@router.post(
    "/events/upload",
    response_model=BidsEventsResponse,
    summary="Upload a BIDS events.tsv file from the browser",
)
async def upload_events_tsv(
    file: UploadFile = File(..., description="BIDS-compliant *_events.tsv file"),
) -> BidsEventsResponse:
    """
    Accept a BIDS events.tsv file via multipart upload and return parsed events.

    Call this endpoint when the user drags a .tsv file into the UI.
    The file is parsed in-memory and discarded immediately — no session is
    created and no disk writes occur.

    Example response
    ─────────────────
    {
      "events": [
        {"id": "evt_0000", "onset": 0.0, "duration": 0.5,
         "trial_type": "deviant_tone", "response_time": null, "color": "#ef5350"},
        ...
      ],
      "trial_types": ["deviant_tone", "regular_square"],
      "color_map":   {"deviant_tone": "#ef5350", "regular_square": "#42a5f5"},
      "n_events": 120,
      "total_duration": 450.0
    }
    """
    if not file.filename or not file.filename.lower().endswith(".tsv"):
        raise HTTPException(400, "Only .tsv files are accepted.")

    content = await file.read()
    if not content:
        raise HTTPException(422, "Uploaded file is empty.")

    return _parse_tsv(content, filename=file.filename)


# ── Endpoint: server-side path ────────────────────────────────────────────────


@router.get(
    "/events",
    response_model=BidsEventsResponse,
    summary="Parse a BIDS events.tsv from a server-side path",
)
def get_events_from_path(
    path: str = Query(
        ...,
        description="Absolute path to the *_events.tsv file on the server",
    ),
) -> BidsEventsResponse:
    """
    Read a BIDS events.tsv from the server filesystem and return parsed events.

    Use this endpoint during local development when the BIDS dataset is mounted
    at a known location, e.g.:

        GET /api/bids/events?path=/data/ds004482/sub-01/func/sub-01_task-audiovis_events.tsv

    The path must be absolute and the file must be readable by the server process.
    """
    tsv_path = Path(path)

    if not tsv_path.exists():
        raise HTTPException(404, f"File not found: {path}")
    if not tsv_path.is_file():
        raise HTTPException(400, f"Path is not a regular file: {path}")
    if tsv_path.suffix.lower() != ".tsv":
        raise HTTPException(400, "Only .tsv files are accepted.")

    try:
        content = tsv_path.read_bytes()
    except PermissionError as exc:
        raise HTTPException(403, f"Permission denied: {path}") from exc
    except Exception as exc:
        raise HTTPException(500, f"Could not read file: {exc}") from exc

    return _parse_tsv(content, filename=tsv_path.name)
