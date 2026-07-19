"""
session_store.py — Shared in-memory session store
==================================================
Centralises the module-level dictionaries that track active MEG and EEG
sessions so that both ``main.py`` and the analysis routers (meg_analysis.py,
etc.) can share a single source of truth without circular imports or a
database dependency.

How Python dict sharing works
─────────────────────────────
``from session_store import meg_sessions`` binds the local name
``meg_sessions`` to the **same dict object** that lives in this module.
Mutations like ``meg_sessions[sid] = raw`` modify the dict in-place, which
means every module that imported it will immediately see the new key — no
message passing required.  The only footgun is reassignment (``meg_sessions =
{}`` would rebind only the local name and break sharing), which we never do.

Usage
─────
In main.py (loading)::

    from session_store import meg_sessions as _sessions
    _sessions[sid] = raw          # visible everywhere

In meg_analysis.py (reading)::

    from session_store import require_meg
    raw = require_meg(session_id) # raises 404 if not found
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Avoid importing MNE at module level so the store can be imported cheaply
# even before MNE is fully initialised.  The type hint is still correct for
# static analysis.
if TYPE_CHECKING:
    import mne  # noqa: F401 — used only for type annotations


# ── MEG sessions ──────────────────────────────────────────────────────────────

# Maps session_id (UUID string) → mne.io.BaseRaw object.
# MNE opens the file with preload=False, so the raw object holds a file
# handle but does not read sample data until get_data() is called.
meg_sessions:  dict[str, "mne.io.BaseRaw"] = {}

# Maps session_id → original filename uploaded by the client (display only).
meg_filenames: dict[str, str] = {}

# Maps session_id → absolute path to the temporary .fif copy on disk, or None
# when the file was loaded via /api/load-meg-path (no copy was made).
# Used by the DELETE endpoint to clean up disk space.
meg_tempfiles: dict[str, str | None] = {}

# Maps session_id → temporary directory that holds a KIT file set (.con + .mrk + .pos).
# Only populated for sessions created via /api/load-meg-kit; the entire directory is
# removed when the session is deleted.
meg_tempdirs: dict[str, str] = {}


# ── EEG sessions ──────────────────────────────────────────────────────────────

# Maps session_id → mne.io.BaseRaw (loaded from BrainVision .vhdr).
eeg_sessions:  dict[str, "mne.io.BaseRaw"] = {}

# Maps session_id → the .vhdr filename (displayed in the UI metadata panel).
eeg_filenames: dict[str, str] = {}

# Maps session_id → the temporary directory that holds the full BrainVision
# triplet (.vhdr + .eeg + .vmrk).  The entire dir is removed on session delete.
eeg_tempdirs:  dict[str, str] = {}


# ── Lookup helpers ────────────────────────────────────────────────────────────

def require_meg(session_id: str) -> "mne.io.BaseRaw":
    """
    Return the MEG Raw object for *session_id*.

    Raises
    ------
    fastapi.HTTPException
        HTTP 404 when no session with this ID exists — the client should POST
        /api/load-meg first to create a session.
    """
    # HTTPException is imported lazily to keep the module importable without
    # FastAPI being on the Python path in standalone / test contexts.
    from fastapi import HTTPException  # noqa: PLC0415

    raw = meg_sessions.get(session_id)
    if raw is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"MEG session '{session_id}' not found. "
                "POST /api/load-meg first."
            ),
        )
    return raw


def require_eeg(session_id: str) -> "mne.io.BaseRaw":
    """
    Return the EEG Raw object for *session_id*.

    Raises
    ------
    fastapi.HTTPException
        HTTP 404 when no session with this ID exists.
    """
    from fastapi import HTTPException  # noqa: PLC0415

    raw = eeg_sessions.get(session_id)
    if raw is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"EEG session '{session_id}' not found. "
                "POST /api/load-eeg first."
            ),
        )
    return raw
