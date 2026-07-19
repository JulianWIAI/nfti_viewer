"""
routers/longitudinal.py — POST /api/longitudinal/delta
══════════════════════════════════════════════════════════════════════════════

Accepts two NIfTI file uploads (baseline + follow-up), co-registers the
follow-up into the baseline's voxel space using dipy affine registration
(mutual information metric, multi-scale coarse-to-fine pyramid), subtracts
the arrays to produce a float32 delta map, and serialises the result in
Fortran order for direct consumption by the vtk.js frontend.

PIPELINE STAGES (all CPU-bound work dispatched to asyncio.to_thread)
─────────────────────────────────────────────────────────────────────
  1. Load both NIfTIs via nibabel (in-memory or temp-file fallback).
  2. Extract float32 arrays + float64 affines; validate dimensions.
  3. dipy co-registration:
       Step 0 — centers-of-mass translation (closed-form, no MI)
       Step 1 — Translation (3 DOF, MI-optimised)
       Step 2 — Rigid body  (6 DOF, MI-optimised)
       Step 3 — Full affine (12 DOF, optional, transform_type='affine')
  4. Compute delta = registered_followup − baseline.
  5. Serialise delta as base64 Fortran-order float32.
  6. Return JSON with delta, dims, affine, min/max, voxel counts.

NON-BLOCKING GUARANTEE
───────────────────────
UploadFile.read() is awaited on the event loop (thread-unsafe — must stay
on the async path).  All CPU-bound work (nibabel I/O, dipy registration,
numpy arithmetic) is delegated to a ThreadPoolExecutor worker thread via
asyncio.to_thread(), mirroring the anomaly and tractography routers.
The event loop remains free to handle other requests during the ~30–120 s
registration.  Registration time depends on volume size and transform_type:
  rigid  ≈  30–60 s  for a 256³ volume
  affine ≈  60–120 s for a 256³ volume

INTEGRATION — add these two lines to backend/main.py:
    from routers.longitudinal import router as _longitudinal_router
    app.include_router(_longitudinal_router)   # POST /api/longitudinal/delta

PREREQUISITES
─────────────
    pip install nibabel dipy
"""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from longitudinal import (
    DEFAULT_CONFIG,
    LongitudinalConfig,
    check_dependencies,
    check_dipy,
    load_nifti,
    extract_volume,
    run_registration,
    compute_delta,
    serialise_delta,
)

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix='/api/longitudinal', tags=['Longitudinal'])

# ── Response model ────────────────────────────────────────────────────────────

class DeltaResult(BaseModel):
    """Longitudinal delta map returned to the vtk.js frontend."""

    # Primary payload ─────────────────────────────────────────────────────────
    delta: str = Field(
        description=(
            "Base64-encoded float32 array in Fortran order (X varies fastest). "
            "Contains signed delta intensities: positive = growth/expansion, "
            "negative = atrophy/loss.  Decode with atob() → Float32Array."
        )
    )

    # Spatial metadata ────────────────────────────────────────────────────────
    dims: list[int] = Field(
        description="[X, Y, Z] voxel dimensions matching the baseline scan."
    )
    affine: list[float] = Field(
        description=(
            "4×4 RAS-mm affine of the baseline scan, row-major flattened into "
            "16 float64 values.  Used by vtk.js to align the delta volume with "
            "the structural MRI in world space."
        )
    )

    # Colormap anchor points ──────────────────────────────────────────────────
    min_val: float = Field(
        description=(
            "Minimum delta intensity.  Use as the lower bound of the diverging "
            "colormap (typically blue for maximum atrophy)."
        )
    )
    max_val: float = Field(
        description=(
            "Maximum delta intensity.  Use as the upper bound of the diverging "
            "colormap (typically red for maximum growth/expansion)."
        )
    )

    # Clinical summary ────────────────────────────────────────────────────────
    n_positive: int = Field(
        description="Number of voxels with positive delta (growth / fluid expansion)."
    )
    n_negative: int = Field(
        description="Number of voxels with negative delta (atrophy / tissue loss)."
    )

    # Diagnostics ─────────────────────────────────────────────────────────────
    duration_ms:    float = Field(description="Total server-side wall-clock time in ms.")
    transform_type: str   = Field(description="Registration strategy used: 'rigid' or 'affine'.")


# ── Synchronous pipeline (runs in worker thread) ──────────────────────────────

def _run_pipeline(
    baseline_bytes: bytes,
    followup_bytes: bytes,
    cfg:            LongitudinalConfig,
) -> dict:
    """
    End-to-end synchronous pipeline: load → extract → register → delta → serialise.

    Designed to be dispatched to a ThreadPoolExecutor via asyncio.to_thread()
    so the uvicorn event loop is not blocked.

    MEMORY MANAGEMENT
    ──────────────────
    Intermediate arrays are explicitly deleted after use so Python's reference
    counter releases them promptly (dipy allocates several internal copies
    during pyramid-level optimisation).

    At peak: two float32 volumes + the registered output + the delta array
    ≈ 4 × n_voxels × 4 bytes.  For 256³ = 67M voxels: ~1 GB peak RAM.
    The memory guard in extract_volume() raises MemoryError before this point
    if the per-volume voxel count exceeds cfg.max_voxels.
    """
    t0 = time.perf_counter()

    # ── Stage 1: Load NIfTI images ────────────────────────────────────────────
    # load_nifti() tries the in-memory path first (nibabel >= 4.0), falls back
    # to a named temp file.  The temp file is deleted inside load_nifti.
    nii_base = load_nifti(baseline_bytes)
    del baseline_bytes          # release the raw upload bytes early
    nii_fup  = load_nifti(followup_bytes)
    del followup_bytes

    # ── Stage 2: Extract float32 arrays + affines ─────────────────────────────
    # extract_volume() does NOT resample — both volumes stay in their original
    # voxel space so the dipy affines carry all spatial information needed for
    # the registration.
    base_data, base_affine, _base_shape = extract_volume(nii_base, cfg)
    del nii_base
    fup_data,  fup_affine,  _fup_shape  = extract_volume(nii_fup,  cfg)
    del nii_fup

    # ── Stage 3: Affine co-registration ──────────────────────────────────────
    # run_registration() implements:
    #   CoM alignment → Translation (3 DOF) → Rigid (6 DOF) → [Affine (12 DOF)]
    # Returns float32 array with shape == base_data.shape.
    # The follow-up is now in the exact voxel space of the baseline.
    registered = run_registration(
        base_data,  base_affine,
        fup_data,   fup_affine,
        cfg,
    )
    # Release the original follow-up arrays — registered is all we need.
    del fup_data, fup_affine

    # ── Stage 4: Compute delta ────────────────────────────────────────────────
    # delta = registered_followup − baseline
    # Positive: growth/expansion; Negative: atrophy/loss
    delta = compute_delta(base_data, registered)
    del base_data, registered

    # ── Stage 5: Fortran-order serialisation ──────────────────────────────────
    # serialise_delta() applies:
    #   np.asfortranarray(delta.astype(np.float32)).tobytes(order='F')
    # before base64 encoding, guaranteeing X-fastest byte order for vtk.js.
    payload = serialise_delta(delta, base_affine)
    del delta, base_affine

    duration_ms = (time.perf_counter() - t0) * 1000.0
    payload['duration_ms']    = round(duration_ms, 1)
    payload['transform_type'] = cfg.transform_type

    return payload


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    '/delta',
    summary=(
        "Longitudinal delta: co-register a follow-up NIfTI to the baseline "
        "and return a Fortran-order float32 subtraction map."
    ),
    response_model=DeltaResult,
)
async def compute_longitudinal_delta(
    baseline: UploadFile = File(
        ...,
        description=(
            "Baseline (earlier) NIfTI scan (.nii / .nii.gz).  "
            "The delta volume is produced in this scan's voxel space."
        ),
    ),
    followup: UploadFile = File(
        ...,
        description=(
            "Follow-up (later) NIfTI scan (.nii / .nii.gz).  "
            "Will be co-registered to the baseline before subtraction."
        ),
    ),
    transform_type: str = Form(
        default='rigid',
        description=(
            "'rigid'  — 6 DOF (rotation + translation). "
            "Recommended for same-subject same-scanner longitudinal MRI.  "
            "'affine' — 12 DOF (+ scaling + shear). "
            "Use only for cross-scanner data with different voxel sizes."
        ),
    ),
) -> JSONResponse:
    """
    Co-register the follow-up NIfTI to the baseline using dipy affine
    registration (mutual information, multi-scale pyramid), compute the
    voxel-wise delta (follow-up − baseline), and return a base64 float32
    Fortran-order delta map.

    **Positive delta** → tissue growth, fluid expansion, lesion progression.
    **Negative delta** → atrophy, tissue loss, lesion shrinkage.

    The returned affine, dims, min_val, and max_val are the values needed
    by the vtk.js frontend to reconstruct and display the overlay with
    a symmetric diverging colormap centred on zero.

    Prerequisites
    ─────────────
    pip install nibabel dipy
    """
    # ── Validate file extensions ──────────────────────────────────────────────
    for field_name, upload in [('baseline', baseline), ('followup', followup)]:
        fn = (upload.filename or '').lower()
        if not (fn.endswith('.nii') or fn.endswith('.nii.gz')):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"'{field_name}' must be a .nii or .nii.gz file, "
                    f"got '{upload.filename}'."
                ),
            )

    # ── Validate transform_type ───────────────────────────────────────────────
    if transform_type not in ('rigid', 'affine'):
        raise HTTPException(
            status_code=422,
            detail=(
                f"transform_type must be 'rigid' or 'affine', "
                f"got '{transform_type}'."
            ),
        )

    # ── Dependency checks ─────────────────────────────────────────────────────
    # These are O(1) module-level attribute reads — safe to call on every request.
    try:
        check_dependencies()   # nibabel
        check_dipy()           # dipy + scipy
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ── Read file bytes on the event loop ─────────────────────────────────────
    # UploadFile.read() is not thread-safe — must be awaited here, not inside
    # asyncio.to_thread().  This mirrors the pattern used by the anomaly and
    # tractography routers.
    baseline_bytes = await baseline.read()
    followup_bytes = await followup.read()

    if not baseline_bytes:
        raise HTTPException(status_code=422, detail="baseline file is empty.")
    if not followup_bytes:
        raise HTTPException(status_code=422, detail="followup file is empty.")

    # ── Build per-request config ──────────────────────────────────────────────
    # Create a new frozen config for this request so that the transform_type
    # query parameter is honoured without mutating the module-level DEFAULT_CONFIG.
    cfg = LongitudinalConfig(transform_type=transform_type)

    # ── Dispatch CPU-bound pipeline to a worker thread ────────────────────────
    # asyncio.to_thread() sends _run_pipeline to the default ThreadPoolExecutor.
    # The event loop is freed immediately to handle other requests during the
    # ~30–120 s registration.
    try:
        payload = await asyncio.to_thread(
            _run_pipeline,
            baseline_bytes,
            followup_bytes,
            cfg,
        )
    except MemoryError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Longitudinal delta computation failed: {exc}",
        ) from exc

    # JSONResponse avoids Pydantic's double-serialisation of the large base64
    # delta string (which would add ~33% overhead from unnecessary escaping).
    return JSONResponse(content=payload)
