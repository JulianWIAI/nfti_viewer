"""
routers/anomaly_detection.py — POST /api/anomalies/detect
══════════════════════════════════════════════════════════════

Accepts a single NIfTI file upload (T2 or FLAIR scan) and returns a binary
anomaly mask (tumours, lesions, hyperintensities) as a base64-encoded
Fortran-order uint8 array, ready for direct ingestion by the vtk.js frontend.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGISTRATION IN main.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Add these two lines to backend/main.py after the existing include_router calls:

    from routers.anomaly_detection import router as _anomaly_router
    app.include_router(_anomaly_router)   # POST /api/anomalies/detect

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUEST FORMAT — multipart/form-data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Required files:
    nifti_file        — 3-D (or 4-D, first volume used) NIfTI scan
                        (.nii or .nii.gz, typically T2 or FLAIR)

  Optional form fields:
    threshold         float [0,1]   Probability threshold for anomaly mask.
                                    Default: 0.5 (natural sigmoid boundary).
                                    Lower = more sensitive, higher = more specific.
    normalisation     str           "zscore" (default) or "minmax"
    sigmoid_output    bool          True (default) = sigmoid output head;
                                    False = softmax argmax mode.
    anomaly_class_idx int           Output channel index for anomaly class.
                                    Default: 0 (binary sigmoid models).

Example (curl):
    curl -X POST http://localhost:8000/api/anomalies/detect \\
      -F "nifti_file=@patient01_FLAIR.nii.gz" \\
      -F "threshold=0.4"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT — application/json
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
  "mask":        "<base64 string>",
  "dims":        [182, 218, 182],
  "affine":      [1.0, 0.0, ..., 1.0],
  "n_anomaly":   4823,
  "duration_ms": 3421.7
}

  mask        — base64-encoded uint8 Fortran-order binary mask.
                Decode with: np.frombuffer(base64.b64decode(mask), dtype=np.uint8)
                             .reshape(dims, order='F')
  dims        — [X, Y, Z] original voxel dimensions of the input NIfTI.
  affine      — 16-element row-major-flattened 4×4 RAS-mm world transform.
  n_anomaly   — voxel count of detected anomaly region (diagnostic).
  duration_ms — server-side wall-clock time in ms (NIfTI I/O + inference + post-processing).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THREADING MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ONNX inference and scipy resampling are CPU-bound and may run for 10–120 s on
whole-brain 1mm volumes.  Running them directly in an `async def` handler
would block the FastAPI event loop, preventing the server from responding to
other requests (health checks, MEG sessions, etc.) during the computation.

Pattern (mirrors routers/tractography.py):
  1. Bytes are read from the UploadFile on the event loop (async I/O, safe).
  2. The synchronous pipeline is wrapped in _run_pipeline() and dispatched to
     the default ThreadPoolExecutor via asyncio.to_thread().
  3. The event loop remains responsive while the thread runs inference.

FILE BYTE SAFETY:
  UploadFile.read() is async and NOT thread-safe.  All bytes are consumed on
  the event loop (await file.read()) before entering the thread, consistent
  with the tractography and decoding router patterns in this codebase.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORTRAN-ORDER REQUIREMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The response mask is encoded in Fortran (column-major / X-fastest) order.
This matches the vtk.js vtkImageData memory layout and the NIfTI on-disk
convention.  Full explanation in postprocess.py → serialise_mask().
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Anomaly pipeline sub-modules ──────────────────────────────────────────────
from anomaly.config      import AnomalyConfig
from anomaly.preprocess  import (
    check_dependencies,
    load_nifti,
    extract_volume,
    resample_to_isotropic,
    normalise,
    crop_pad,
    build_tensor,
)
from anomaly.inference   import (
    check_onnx,
    run_inference,
    extract_binary_mask,
)
from anomaly.postprocess import (
    uncrop_unpad,
    resample_to_original,
    serialise_mask,
)

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(tags=["Anomaly Detection"])

# ── Response model ────────────────────────────────────────────────────────────

class AnomalyResult(BaseModel):
    """
    JSON body returned by POST /api/anomalies/detect.

    The mask field is the primary payload.  dims and affine allow the vtk.js
    frontend to reconstruct the volume geometry without re-reading the NIfTI.
    """

    mask: str = Field(
        description=(
            "Base64-encoded uint8 binary mask in Fortran (X-fastest) byte order.  "
            "1 = anomaly voxel, 0 = healthy tissue.  "
            "Decode: np.frombuffer(base64.b64decode(mask), dtype=np.uint8)"
            ".reshape(dims, order='F')"
        ),
    )

    dims: list[int] = Field(
        description="[X, Y, Z] voxel dimensions of the mask (= original NIfTI shape).",
    )

    affine: list[float] = Field(
        description=(
            "4×4 RAS-mm world transform, row-major flattened to 16 float64 values.  "
            "Identical to the input NIfTI affine — not modified by the pipeline."
        ),
    )

    n_anomaly: int = Field(
        description=(
            "Total number of voxels classified as anomalous (1-valued voxels).  "
            "Useful as a quick diagnostic: very large values may indicate a false-"
            "positive flood-fill; zero means no anomaly was detected."
        ),
    )

    duration_ms: float = Field(
        description="Server-side wall-clock time in ms, from NIfTI load to response.",
    )


# ── Synchronous pipeline (runs in a thread pool via asyncio.to_thread) ────────

def _run_pipeline(
    content: bytes,
    cfg:     AnomalyConfig,
) -> dict:
    """
    Full anomaly detection pipeline — synchronous, CPU-bound.

    Dispatched to a thread pool by the async endpoint handler so the FastAPI
    event loop is never blocked.  All I/O (file bytes) is received before
    this function is called, so no async operations are needed here.

    PIPELINE STAGES
    ────────────────
      1. Deserialise raw bytes → nibabel NIfTI image.
      2. Extract float32 volume, affine, original shape, voxel sizes.
      3. Resample to 1mm isotropic (if needed) → record resampled_shape.
      4. Z-score normalise over brain-tissue voxels.
      5. Centre-crop / zero-pad to model input shape → record offsets.
      6. Build channels-last 5-D tensor: [1, X, Y, Z, 1].
      7. ONNX inference → raw probability map: [X, Y, Z, K].
      8. Extract binary mask (threshold or argmax) → uint8 [X, Y, Z].
      9. Reverse crop/pad → 1mm isotropic space.
      10. Resample back to original NIfTI space (nearest-neighbour).
      11. Serialise as Fortran-order base64 uint8 + metadata dict.

    Returns
    -------
    dict matching the AnomalyResult schema (used by JSONResponse in the endpoint).
    """
    t0 = time.perf_counter()

    # ── Stage 1: NIfTI deserialisation ───────────────────────────────────────
    try:
        nii = load_nifti(content)
    except Exception as exc:
        raise ValueError(f"Could not parse NIfTI file: {exc}") from exc

    # ── Stage 2: Volume extraction ────────────────────────────────────────────
    try:
        data, affine, orig_shape, vox_mm = extract_volume(nii)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    # ── Stage 3: Isotropic resampling (skip if already ~1mm) ─────────────────
    try:
        resampled, resampled_shape = resample_to_isotropic(data, vox_mm, cfg)
    except MemoryError as exc:
        raise MemoryError(str(exc)) from exc

    # Free the original data array — it is no longer needed and may be large.
    # The resampled volume is the working buffer for the rest of the pipeline.
    del data

    # ── Stage 4: Normalisation ────────────────────────────────────────────────
    # In-place replacement: `normed` has the same shape as `resampled` but is
    # guaranteed float32.  We delete `resampled` immediately to halve peak RAM.
    normed = normalise(resampled, cfg)
    del resampled

    # ── Stage 5: Crop / pad to model input shape ──────────────────────────────
    # `offsets` encodes per-axis pad/crop amounts needed to reverse this step.
    cropped, offsets = crop_pad(normed, cfg.target_shape)
    del normed

    # ── Stage 6: Build 5-D channels-last tensor ───────────────────────────────
    # Shape: [1, X, Y, Z, 1]  (batch × spatial × channel)
    # np.ascontiguousarray inside build_tensor() prevents a hidden onnxruntime copy.
    tensor = build_tensor(cropped)
    del cropped

    # ── Stage 7: ONNX inference ───────────────────────────────────────────────
    # Returns float32 probability map of shape (X, Y, Z, K) — batch dim stripped.
    try:
        probs = run_inference(tensor, cfg)
    except Exception as exc:
        raise RuntimeError(f"ONNX inference failed: {exc}") from exc
    finally:
        del tensor   # release the large input tensor regardless of outcome

    # ── Stage 8: Binary mask extraction ──────────────────────────────────────
    # sigmoid: threshold P(anomaly) at cfg.threshold → uint8 {0, 1}
    # softmax: argmax → class index map → binary anomaly indicator
    mask_model: np.ndarray = extract_binary_mask(probs, cfg)
    del probs

    # ── Stage 9: Reverse crop / pad → 1mm isotropic space ────────────────────
    # Strips the zero-padding or re-embeds cropped data using the recorded offsets.
    mask_1mm: np.ndarray = uncrop_unpad(mask_model, offsets, resampled_shape)
    del mask_model

    # ── Stage 10: Resample binary mask back to original voxel space ───────────
    # Uses nearest-neighbour (order=0) to preserve the binary {0,1} values.
    # Handles ±1 voxel rounding errors from scipy.ndimage.zoom internally.
    mask_orig: np.ndarray = resample_to_original(
        mask_1mm, orig_shape, vox_mm, target_vox=cfg.target_vox_mm,
    )
    del mask_1mm

    # ── Stage 11: Fortran-order serialisation ─────────────────────────────────
    # np.asfortranarray() + tobytes(order='F') — see postprocess.py for the
    # detailed explanation of why both steps are required.
    duration_ms = (time.perf_counter() - t0) * 1000.0
    payload = serialise_mask(mask_orig, affine)
    payload["duration_ms"] = round(duration_ms, 1)

    return payload


# ── Async endpoint ────────────────────────────────────────────────────────────

@router.post(
    "/api/anomalies/detect",
    response_model=AnomalyResult,
    summary="Anomaly detection (tumour / lesion / hyperintensity segmentation)",
    description=(
        "Runs the AnomalySeg ONNX model on an uploaded 3-D NIfTI scan and "
        "returns a binary anomaly mask in the original voxel space.  "
        "Preprocessing: 1mm isotropic resampling → z-score normalisation → "
        "BraTS-standard crop/pad (240×240×155).  "
        "Postprocessing: reverse crop/pad → nearest-neighbour resample → "
        "Fortran-order base64 serialisation.  "
        "Typical runtime: 10–120 s depending on volume size and hardware."
    ),
    tags=["Anomaly Detection"],
)
async def detect_anomalies(
    nifti_file: UploadFile = File(
        ...,
        description=(
            "3-D (or 4-D, first volume used) NIfTI scan in T2 or FLAIR contrast.  "
            "Accepts .nii and .nii.gz.  "
            "The model was trained on BraTS-style 1mm isotropic T2/FLAIR volumes."
        ),
    ),
    threshold: float = Form(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Probability threshold for classifying a voxel as anomalous.  "
            "Applied to the sigmoid output (P(anomaly) >= threshold → mask=1).  "
            "Lower values increase sensitivity (more detections, more false positives); "
            "higher values increase specificity."
        ),
    ),
    normalisation: str = Form(
        default="zscore",
        description=(
            "Intensity normalisation strategy applied before inference.  "
            "  'zscore' — zero-mean / unit-variance over non-zero (brain) voxels.  "
            "            Preferred for BraTS-trained models (default).  "
            "  'minmax' — rescale to [0, 1] using global min/max."
        ),
    ),
    sigmoid_output: bool = Form(
        default=True,
        description=(
            "True (default): treat model output as sigmoid probabilities and apply threshold.  "
            "False: treat model output as softmax logits and use argmax to identify anomaly class."
        ),
    ),
    anomaly_class_idx: int = Form(
        default=0,
        ge=0,
        description=(
            "Output channel index corresponding to the anomaly class.  "
            "For binary sigmoid models the single output channel (index 0) encodes P(anomaly).  "
            "For multi-class softmax models, set this to the anomaly class index."
        ),
    ),
) -> JSONResponse:
    """
    Run the AnomalySeg anomaly detection pipeline on the uploaded NIfTI file.

    NON-BLOCKING DESIGN
    ────────────────────
    1. Validate inputs (fast, on the event loop).
    2. Read file bytes asynchronously — UploadFile.read() is not thread-safe
       and must remain on the event loop.
    3. Dispatch the synchronous CPU pipeline (_run_pipeline) to the default
       ThreadPoolExecutor via asyncio.to_thread().  The event loop is free
       to handle other requests while inference runs in the thread.
    4. Return a JSONResponse with the AnomalyResult payload.
    """

    # ── Input validation ──────────────────────────────────────────────────────

    # Validate NIfTI file extension.
    fname = nifti_file.filename or ""
    if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file type: '{fname}'.  "
                "Upload a NIfTI file (.nii or .nii.gz)."
            ),
        )

    # Validate normalisation string.
    if normalisation not in ("zscore", "minmax"):
        raise HTTPException(
            status_code=422,
            detail=f"Unknown normalisation '{normalisation}'.  Choose 'zscore' or 'minmax'.",
        )

    # ── Dependency checks ─────────────────────────────────────────────────────
    # Raise 500 with a clear message before any heavy work if deps are absent.
    try:
        check_dependencies()   # nibabel + scipy
        check_onnx()           # onnxruntime
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ── Model existence check ─────────────────────────────────────────────────
    # Inline here so the error reaches the client before file bytes are read.
    cfg = AnomalyConfig(
        threshold         = threshold,
        normalisation     = normalisation,
        sigmoid_output    = sigmoid_output,
        anomaly_class_idx = anomaly_class_idx,
    )
    if not cfg.model_path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"AnomalySeg model not found at '{cfg.model_path}'.  "
                "Place AnomalySeg.onnx in backend/models/ or run "
                "python backend/download_models.py."
            ),
        )

    # ── Read file bytes on the event loop ─────────────────────────────────────
    # UploadFile.read() is an async coroutine and is NOT thread-safe.
    # We must await it here (on the event loop) before passing bytes to the thread.
    content: bytes = await nifti_file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    # ── Dispatch synchronous pipeline to a thread pool ─────────────────────────
    # asyncio.to_thread() submits _run_pipeline to the default ThreadPoolExecutor
    # and suspends this coroutine until the thread completes, keeping the event
    # loop responsive throughout the (potentially long) inference.
    try:
        payload = await asyncio.to_thread(_run_pipeline, content, cfg)

    except ValueError as exc:
        # NIfTI parsing or shape errors (client mistake — 422)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except MemoryError as exc:
        # Volume too large to resample (client mistake — 413)
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    except RuntimeError as exc:
        # ONNX runtime failure (server error — 500)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ── Return serialised result ───────────────────────────────────────────────
    # JSONResponse is used instead of returning AnomalyResult directly so that
    # FastAPI does not re-serialise the already-built dict (avoids a double-copy
    # of the potentially large base64 mask string).
    return JSONResponse(content=payload)
