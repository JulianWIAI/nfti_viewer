"""
routers/registration.py — POST /api/registration/syn
══════════════════════════════════════════════════════════════════════════════

Accepts two NIfTI file uploads (subject_a_ref + subject_b_moving), runs the
full affine + SyN diffeomorphic registration pipeline, and returns:

  • Warped Subject B intensity volume (float32, Fortran-order base64)
  • SynthSeg label maps for both subjects (uint8, Fortran-order base64)
  • Comparative hippocampal volumetrics (cm³)

PIPELINE STAGES (CPU-bound work dispatched to asyncio.to_thread)
──────────────────────────────────────────────────────────────────
  1. Load both NIfTIs via nibabel (in-memory or temp-file fallback).
  2. Extract float32 arrays + float64 affines; validate dimensions.
  3. (Optional, graceful-degradation) SynthSeg ONNX inference on both raw
     volumes → FreeSurfer label maps in original voxel space.
  4. Compute hippocampal volumes (labels 17 & 53) from raw label maps using
     each subject's own voxel geometry so reported cm³ reflect true anatomy.
  5. Affine pre-registration (CoM → Translation → Rigid → Full Affine).
  6. SyN diffeomorphic registration (CrossCorrelation metric, level_iters=[10,10,5]).
     If SynthSeg succeeded: simultaneously warp Subject B's label map with
     nearest-neighbour interpolation so integer FreeSurfer IDs are preserved.
  7. Serialise all arrays as Fortran-order base64.
  8. Return JSON with all payloads, metadata, and volumetrics.

NON-BLOCKING GUARANTEE
───────────────────────
UploadFile.read() is awaited on the event loop (thread-unsafe).  All CPU-bound
work (~2–5 minutes for a 256³ brain volume) is dispatched to a ThreadPoolExecutor
worker via asyncio.to_thread(), mirroring the longitudinal and anomaly routers.

GRACEFUL DEGRADATION
─────────────────────
If SynthSeg or its ONNX model is unavailable, the pipeline still returns the
warped intensity volume.  seg_a and seg_b_warped are empty strings and
volumetrics is null in that case.

PREREQUISITES
─────────────
    pip install nibabel dipy scipy onnxruntime
"""
from __future__ import annotations

import asyncio
import base64
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from syn_registration import (
    DEFAULT_SYN_CONFIG,
    SynConfig,
    check_syn_dependencies,
    run_syn_pipeline,
    run_syn_pipeline_with_seg,
)
from longitudinal.preprocess import load_nifti, extract_volume, check_dependencies
from syn_segmentation import (
    check_synthseg_available,
    run_synthseg,
    compute_hippocampal_volumes,
)

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix='/api/registration', tags=['Registration'])

# ── Response model ────────────────────────────────────────────────────────────

class SynResult(BaseModel):
    """Warped Subject B volume + segmentation + volumetrics returned to the frontend."""

    # Primary payload ─────────────────────────────────────────────────────────
    warped: str = Field(
        description=(
            "Base64-encoded float32 array in Fortran order (X varies fastest). "
            "Subject B warped into Subject A's voxel space. "
            "Decode with atob() → Uint8Array → Float32Array.buffer."
        )
    )

    # Spatial metadata ────────────────────────────────────────────────────────
    dims: list[int] = Field(
        description="[X, Y, Z] voxel dimensions — matches Subject A's voxel space."
    )
    affine: list[float] = Field(
        description=(
            "4×4 RAS-mm affine of Subject A (the reference scan), row-major "
            "flattened into 16 float64 values. The warped volume is in this space."
        )
    )

    # Colormap anchor points ──────────────────────────────────────────────────
    min_val: float = Field(description="Minimum intensity of the warped volume.")
    max_val: float = Field(description="Maximum intensity of the warped volume.")

    # Diagnostics ─────────────────────────────────────────────────────────────
    duration_ms: float = Field(description="Total server-side wall-clock time in ms.")

    # Segmentation payloads (empty string when SynthSeg unavailable) ──────────
    seg_a: str = Field(
        default="",
        description=(
            "Base64-encoded uint8 Fortran-order array of FreeSurfer label IDs "
            "for Subject A (reference), in Subject A's original voxel space. "
            "Empty string if SynthSeg is unavailable."
        ),
    )
    seg_b_warped: str = Field(
        default="",
        description=(
            "Base64-encoded uint8 Fortran-order array of FreeSurfer label IDs "
            "for Subject B, warped into Subject A's voxel space using "
            "nearest-neighbour interpolation so integer label values are preserved. "
            "Empty string if SynthSeg is unavailable."
        ),
    )

    # Comparative volumetrics (null when SynthSeg unavailable) ────────────────
    volumetrics: Any = Field(
        default=None,
        description=(
            "Hippocampal volumes computed from each subject's RAW (un-warped) "
            "label map using the subject's own voxel geometry. "
            "Shape: {subjectA: {lh: float, rh: float}, subjectB: {lh: float, rh: float}} "
            "All values are in cm³, rounded to 3 decimal places. "
            "Null if SynthSeg is unavailable."
        ),
    )


# ── Synchronous pipeline (dispatched to worker thread) ───────────────────────

def _run_pipeline(
    ref_bytes:    bytes,
    moving_bytes: bytes,
    cfg:          SynConfig,
) -> dict:
    """
    End-to-end synchronous pipeline: load → segment → register → serialise.

    Designed to be dispatched to a ThreadPoolExecutor via asyncio.to_thread().
    All dipy, nibabel, and ONNX calls are CPU-bound and must NOT run on the
    event loop.

    MEMORY MANAGEMENT
    ──────────────────
    Intermediate arrays are explicitly deleted after use.  At peak:
      - static + moving + prealigned + warped (float32) + dipy internal buffers
      ≈ ~2–4 GB for 256³.  seg arrays (uint8) add ~64 MB each.
    The max_voxels guard in extract_volume() raises MemoryError before the
    largest allocations if the per-volume voxel count is too high.

    GRACEFUL DEGRADATION
    ─────────────────────
    SynthSeg inference is attempted but not required.  If it fails (missing
    ONNX model, missing packages, ONNX runtime error), the pipeline falls back
    to run_syn_pipeline() for intensity only and returns empty seg fields.
    """
    t0 = time.perf_counter()

    # ── Stage 1: Load NIfTI images ────────────────────────────────────────────
    nii_ref    = load_nifti(ref_bytes)
    del ref_bytes
    nii_moving = load_nifti(moving_bytes)
    del moving_bytes

    # ── Stage 2: Extract float32 arrays + affines ─────────────────────────────
    ref_data,    ref_affine,    _ref_shape    = extract_volume(nii_ref,    cfg)
    del nii_ref
    moving_data, moving_affine, _moving_shape = extract_volume(nii_moving, cfg)
    del nii_moving

    # ── Stage 3: SynthSeg inference on both raw volumes ───────────────────────
    # Attempted with graceful degradation — a missing model or package only
    # suppresses the segmentation output; registration still proceeds.
    seg_a_raw:    np.ndarray | None = None
    seg_b_raw:    np.ndarray | None = None
    synthseg_ok:  bool              = False
    volumetrics:  dict | None       = None

    try:
        check_synthseg_available()  # raises RuntimeError if missing

        # Run SynthSeg on Subject A (reference) in its own voxel space.
        seg_a_raw = run_synthseg(ref_data, ref_affine)

        # Run SynthSeg on Subject B (moving) in its own voxel space.
        # NOTE: this is intentionally on moving_data BEFORE warping so that
        # hippocampal volumes reflect the subject's true anatomy.
        seg_b_raw = run_synthseg(moving_data, moving_affine)

        synthseg_ok = True

    except Exception:
        # Graceful degradation: log suppressed; seg fields will be empty strings.
        seg_a_raw   = None
        seg_b_raw   = None
        synthseg_ok = False

    # ── Stage 4: Hippocampal volumetrics from raw (un-warped) label maps ──────
    # Volumes MUST be computed before warping — warping changes voxel geometry.
    if synthseg_ok and seg_a_raw is not None and seg_b_raw is not None:
        vols_a = compute_hippocampal_volumes(seg_a_raw, ref_affine)
        vols_b = compute_hippocampal_volumes(seg_b_raw, moving_affine)
        volumetrics = {
            'subjectA': vols_a,
            'subjectB': vols_b,
        }

    # ── Stage 5: Affine + SyN registration ───────────────────────────────────
    # If SynthSeg succeeded, simultaneously warp the Subject B label map
    # through the same displacement field using nearest-neighbour interpolation
    # so that integer FreeSurfer label IDs (e.g. 17 = LH hippocampus) are not
    # blurred by linear mixing.
    warped_seg: np.ndarray | None = None

    if synthseg_ok and seg_b_raw is not None:
        warped, warped_seg = run_syn_pipeline_with_seg(
            ref_data,    ref_affine,
            moving_data, moving_affine,
            seg_b_raw,
            cfg,
        )
        del seg_b_raw
    else:
        warped = run_syn_pipeline(
            ref_data,    ref_affine,
            moving_data, moving_affine,
            cfg,
        )
    del moving_data, moving_affine

    # ── Stage 6: Intensity statistics for the warped volume ───────────────────
    warped_f32 = warped.astype(np.float32)
    min_val    = float(np.min(warped_f32))
    max_val    = float(np.max(warped_f32))
    del warped

    # ── Stage 7: Fortran-order serialisation ──────────────────────────────────
    # np.asfortranarray() guarantees column-major physical layout.
    # .tobytes(order='F') emits bytes in X-fastest (Fortran) order.
    # Both steps are required — see longitudinal/delta.py for full explanation.
    warped_b64 = base64.b64encode(
        np.asfortranarray(warped_f32).tobytes(order='F')
    ).decode('ascii')
    del warped_f32

    # Serialise Subject A label map (original space, un-warped).
    seg_a_b64: str = ""
    if seg_a_raw is not None:
        seg_a_b64 = base64.b64encode(
            np.asfortranarray(seg_a_raw).tobytes(order='F')
        ).decode('ascii')
        del seg_a_raw

    # Serialise warped Subject B label map (Subject A voxel space, NN-interpolated).
    seg_b_warped_b64: str = ""
    if warped_seg is not None:
        seg_b_warped_b64 = base64.b64encode(
            np.asfortranarray(warped_seg).tobytes(order='F')
        ).decode('ascii')
        del warped_seg

    duration_ms = (time.perf_counter() - t0) * 1000.0

    return {
        'warped':       warped_b64,
        'dims':         list(ref_data.shape),           # [X, Y, Z]
        'affine':       ref_affine.flatten().tolist(),  # 16 float64, row-major
        'min_val':      min_val,
        'max_val':      max_val,
        'duration_ms':  round(duration_ms, 1),
        'seg_a':        seg_a_b64,
        'seg_b_warped': seg_b_warped_b64,
        'volumetrics':  volumetrics,
    }


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    '/syn',
    summary=(
        "SyN inter-subject registration: warp Subject B into Subject A's voxel space "
        "using dipy Affine (MI) + SymmetricDiffeomorphicRegistration (CC)."
    ),
    response_model=SynResult,
)
async def compute_syn_registration(
    subject_a_ref: UploadFile = File(
        ...,
        description=(
            "Reference scan — Subject A (.nii / .nii.gz). "
            "The warped volume is returned in this scan's voxel space."
        ),
    ),
    subject_b_moving: UploadFile = File(
        ...,
        description=(
            "Moving scan — Subject B (.nii / .nii.gz). "
            "Will be warped to match Subject A's brain morphology."
        ),
    ),
) -> JSONResponse:
    """
    Co-register Subject B to Subject A using a two-stage pipeline:

    1. **Affine pre-registration** (12 DOF, Mutual Information metric) to
       handle gross differences in position, orientation, and scale.
    2. **SyN diffeomorphic warp** (CrossCorrelation metric, `level_iters=[10,10,5]`)
       to correct non-linear shape differences such as sulcal patterns.

    Additionally (when SynthSeg + ONNX model are available):

    3. **SynthSeg ONNX** segmentation on both raw subjects → FreeSurfer label maps.
    4. **Hippocampal volumetrics** (labels 17 & 53) from un-warped label maps.
    5. **Label map warp** for Subject B via nearest-neighbour interpolation.

    ⚠ **Performance**: 256³ volumes take approximately 2–5 minutes on a CPU.
    The event loop is kept free via asyncio.to_thread().

    Prerequisites: `pip install nibabel dipy scipy onnxruntime`
    """
    # ── Validate file extensions ──────────────────────────────────────────────
    for field_name, upload in [
        ('subject_a_ref',    subject_a_ref),
        ('subject_b_moving', subject_b_moving),
    ]:
        fn = (upload.filename or '').lower()
        if not (fn.endswith('.nii') or fn.endswith('.nii.gz')):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"'{field_name}' must be a .nii or .nii.gz file, "
                    f"got '{upload.filename}'."
                ),
            )

    # ── Dependency checks ─────────────────────────────────────────────────────
    try:
        check_dependencies()     # nibabel
        check_syn_dependencies() # dipy
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ── Read file bytes on the event loop ─────────────────────────────────────
    # UploadFile.read() is not thread-safe — must be awaited here.
    ref_bytes    = await subject_a_ref.read()
    moving_bytes = await subject_b_moving.read()

    if not ref_bytes:
        raise HTTPException(status_code=422, detail="subject_a_ref file is empty.")
    if not moving_bytes:
        raise HTTPException(status_code=422, detail="subject_b_moving file is empty.")

    # ── Dispatch CPU-bound pipeline to a worker thread ────────────────────────
    # The ~2–5 minute registration runs entirely in the thread pool.
    # The event loop is free to handle other requests during this time.
    cfg = DEFAULT_SYN_CONFIG
    try:
        payload = await asyncio.to_thread(
            _run_pipeline,
            ref_bytes,
            moving_bytes,
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
            detail=f"SyN registration failed: {exc}",
        ) from exc

    return JSONResponse(content=payload)
