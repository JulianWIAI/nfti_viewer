"""
routers/tractography.py — POST /api/dti/tractography
══════════════════════════════════════════════════════

Accepts a 4-D diffusion NIfTI file plus .bval and .bvec gradient files.
Runs the full DTI tractography pipeline in a thread pool (non-blocking) and
returns compressed streamline JSON.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGISTRATION IN main.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Add these two lines to backend/main.py (after the existing include_router calls):

    from routers.tractography import router as _tractography_router
    app.include_router(_tractography_router)   # POST /api/dti/tractography

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUEST FORMAT — multipart/form-data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Files (all required):
    nifti_file   — 4-D diffusion NIfTI (.nii or .nii.gz)
    bval_file    — b-values (.bval)  — one per volume, space-separated
    bvec_file    — gradient vectors (.bvec)  — FSL 3×N or BIDS N×3 format

  Form fields (all optional — defaults match TractographyConfig):
    fa_stop          float   FA stopping threshold        default 0.20
    fa_seed          float   FA seeding threshold         default 0.30
    step_size        float   Euler integration step (mm)  default 0.50
    max_angle        float   Max turn per step (degrees)  default 30.0
    seeds_per_voxel  int     Seeds per WM voxel           default 1
    min_length_mm    float   Minimum tract length (mm)    default 30.0
    max_streamlines  int     Hard output cap              default 10 000
    tol_error        float   DP decimation tolerance (mm) default 1.0

  Example (curl):
    curl -X POST http://localhost:8000/api/dti/tractography \\
      -F "nifti_file=@sub01_dwi.nii.gz" \\
      -F "bval_file=@sub01_dwi.bval"    \\
      -F "bvec_file=@sub01_dwi.bvec"    \\
      -F "fa_stop=0.2" -F "max_streamlines=5000"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT — application/json
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
  "streamlines":      [[[x,y,z], [x,y,z], ...], ...],
  "n_streamlines":    8432,
  "n_before_filter":  51230,
  "mean_length_mm":   67.4,
  "peak_fa":          0.87,
  "duration_ms":      48123.0
}

  streamlines     — nested list: one entry per tract, each a list of
                    [x, y, z] float triples in RAS mm world space.
  n_before_filter — streamline count before length-filter + subsampling
                    (useful to gauge how aggressive the compression was).
  peak_fa         — maximum FA observed in the FA map (quality indicator;
                    low peak FA may suggest motion artefacts or a non-DWI
                    file was uploaded).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THREADING MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The DTI pipeline (NIfTI loading + tensor fitting + LocalTracking) is
CPU-bound and may run for 2–10 minutes on a whole-brain dataset.

asyncio.to_thread() offloads _run_pipeline() to the default
ThreadPoolExecutor, keeping the FastAPI event loop free to handle other
requests (health checks, MEG sessions, etc.) during the computation.

File byte safety: UploadFile.read() is async and NOT thread-safe; all bytes
are read on the event loop before entering the thread (same pattern as the
EEG decoding endpoint in routers/decoding.py).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NIfTI TEMP FILE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

nibabel can load uncompressed .nii from BytesIO but requires a real file path
for .nii.gz (gzip format requires seekable random access).  Rather than
detecting the format, we always write to a named temp file — it is the
simplest approach that handles both cases uniformly.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from dti.config import TractographyConfig
from dti.gradients import build_gradient_table
from dti.masking import extract_brain_mask
from dti.tensor_fit import fit_tensor_model
from dti.tracking import run_tractography
from dti.compression import compress_and_filter, serialize_streamlines

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(tags=["DTI Tractography"])

# ── Response model ─────────────────────────────────────────────────────────────

class TractographyResponse(BaseModel):
    """
    JSON body returned by POST /api/dti/tractography.

    The streamlines field is the primary payload.  All other fields are
    diagnostic metadata to help the frontend display quality indicators.
    """

    # ── Primary payload ──────────────────────────────────────────────────
    streamlines: list[list[list[float]]] = Field(
        description=(
            "Array of fibre tracts.  Each element is a polyline represented "
            "as an array of [x, y, z] coordinate triples in RAS mm space.  "
            "Pass directly to a vtk.js vtkPolyData + vtkCellArray line renderer."
        ),
    )

    # ── Compression statistics ───────────────────────────────────────────
    n_streamlines: int = Field(
        description="Number of streamlines in the output (after filtering).",
    )
    n_before_filter: int = Field(
        description=(
            "Number of streamlines before length filtering and subsampling.  "
            "n_streamlines / n_before_filter gives the compression ratio."
        ),
    )
    mean_length_mm: float = Field(
        description="Mean arc-length of the output streamlines in mm.",
    )

    # ── Quality indicators ───────────────────────────────────────────────
    peak_fa: float = Field(
        description=(
            "Maximum FA value observed in the FA map.  "
            "Values < 0.5 may indicate motion artefacts or a non-DWI upload."
        ),
    )

    # ── Diagnostics ──────────────────────────────────────────────────────
    duration_ms: float = Field(
        description="Total wall-clock time for the pipeline in milliseconds.",
    )


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/api/dti/tractography",
    summary="DTI white-matter tractography (deterministic, LocalTracking)",
    description=(
        "Runs the full dipy DTI tractography pipeline on uploaded NIfTI + "
        "gradient files and returns compressed streamline JSON.  "
        "Typical runtime: 2–10 minutes depending on dataset size and parameters."
    ),
    tags=["DTI Tractography"],
)
async def run_tractography_endpoint(
    # ── File uploads ────────────────────────────────────────────────────
    nifti_file: UploadFile = File(
        ...,
        description=(
            "4-D diffusion-weighted NIfTI file (.nii or .nii.gz).  "
            "Must contain ≥ 1 b=0 volume and ≥ 6 diffusion-weighted volumes."
        ),
    ),
    bval_file: UploadFile = File(
        ...,
        description="FSL .bval file: one b-value per volume, space-separated.",
    ),
    bvec_file: UploadFile = File(
        ...,
        description=(
            "FSL .bvec file: gradient directions.  "
            "Accepts both 3×N (row-major) and N×3 (column-major) formats."
        ),
    ),

    # ── Tractography parameters (all optional) ──────────────────────────
    fa_stop: float = Form(
        default=0.20,
        ge=0.0, le=1.0,
        description="FA stopping threshold.  Tracking halts when FA drops below this.",
    ),
    fa_seed: float = Form(
        default=0.30,
        ge=0.0, le=1.0,
        description="FA seeding threshold.  Seeds placed only where FA exceeds this.",
    ),
    step_size: float = Form(
        default=0.50,
        gt=0.0,
        description="Euler integration step in mm.",
    ),
    max_angle: float = Form(
        default=30.0,
        gt=0.0, le=90.0,
        description="Maximum propagation angle change per step in degrees.",
    ),
    seeds_per_voxel: int = Form(
        default=1,
        ge=1, le=4,
        description="Number of seeds placed per seed-mask voxel.",
    ),
    min_length_mm: float = Form(
        default=30.0,
        ge=0.0,
        description="Minimum tract arc-length in mm.  Shorter tracts are discarded.",
    ),
    max_streamlines: int = Form(
        default=10_000,
        ge=100, le=100_000,
        description="Maximum number of streamlines in the response.  Randomly subsampled if exceeded.",
    ),
    tol_error: float = Form(
        default=1.0,
        gt=0.0,
        description=(
            "Douglas–Peucker point-decimation tolerance in mm.  "
            "Higher = fewer points per streamline = smaller response."
        ),
    ),
) -> JSONResponse:
    """
    Run DTI tractography on the uploaded files.

    Pipeline stages:
      1. Validate file extensions.
      2. Read all bytes from the async UploadFile streams (event-loop safe).
      3. Offload the synchronous pipeline to a thread pool via asyncio.to_thread.
      4. Inside the thread:
           a. Build GradientTable from .bval / .bvec bytes.
           b. Write NIfTI bytes to a temp file; load with nibabel.
           c. Extract brain mask (median_otsu).
           d. Fit tensor model (WLS) → FA, eigenvectors.
           e. Run deterministic LocalTracking → raw streamlines.
           f. Filter (length), subsample, decimate (Douglas–Peucker).
           g. Serialise to nested Python lists.
      5. Return as JSONResponse.
    """

    # ── Validate file extensions ──────────────────────────────────────────
    _require_extension(nifti_file, ('.nii', '.gz'), 'nifti_file')
    _require_extension(bval_file,  ('.bval',),       'bval_file')
    _require_extension(bvec_file,  ('.bvec',),       'bvec_file')

    # ── Validate parameter consistency ────────────────────────────────────
    if fa_seed <= fa_stop:
        raise HTTPException(
            422,
            f"fa_seed ({fa_seed}) must be strictly greater than fa_stop ({fa_stop}).  "
            "Seeds placed in low-FA voxels would be stopped immediately."
        )

    # ── Read all file bytes on the event loop ─────────────────────────────
    # UploadFile.read() is async and must NOT be called from a thread.
    # Materialising all bytes here is safe and avoids passing the async
    # stream object into the thread pool.
    try:
        nifti_bytes = await nifti_file.read()
        bval_bytes  = await bval_file.read()
        bvec_bytes  = await bvec_file.read()
    except Exception as exc:
        raise HTTPException(500, f"Failed to read uploaded files: {exc}") from exc

    _require_nonempty(nifti_bytes, 'nifti_file')
    _require_nonempty(bval_bytes,  'bval_file')
    _require_nonempty(bvec_bytes,  'bvec_file')

    # ── Build config ──────────────────────────────────────────────────────
    config = TractographyConfig(
        fa_stop         = fa_stop,
        fa_seed         = fa_seed,
        step_size       = step_size,
        max_angle       = max_angle,
        seeds_per_voxel = seeds_per_voxel,
        min_length_mm   = min_length_mm,
        max_streamlines = max_streamlines,
        tol_error       = tol_error,
    )

    # ── Run pipeline in thread pool ────────────────────────────────────────
    nifti_name = nifti_file.filename or 'data.nii.gz'
    try:
        result = await asyncio.to_thread(
            _run_pipeline,
            nifti_bytes, bval_bytes, bvec_bytes,
            nifti_name,
            config,
        )
    except ValueError as exc:
        # Pipeline validation errors (no seeds, bad gradient table, etc.) → 422
        raise HTTPException(422, str(exc)) from exc
    except MemoryError:
        raise HTTPException(
            500,
            "Out of memory during tractography.  Try a lower-resolution dataset "
            "or reduce seeds_per_voxel."
        )
    except Exception as exc:
        raise HTTPException(500, f"Tractography pipeline failed: {exc}") from exc

    # ── Return JSON ───────────────────────────────────────────────────────
    # Use JSONResponse directly rather than a Pydantic response_model to avoid
    # Pydantic's validation overhead on a large nested list (10MB+).
    # orjson (if installed) serialises nested float lists ~5× faster than stdlib;
    # swap the import and response class for production use:
    #   from fastapi.responses import ORJSONResponse
    #   return ORJSONResponse({...})
    payload = {
        "streamlines":    result.streamlines_json,
        "n_streamlines":  result.n_streamlines,
        "n_before_filter": result.n_before_filter,
        "mean_length_mm": result.mean_length_mm,
        "peak_fa":        result.peak_fa,
        "duration_ms":    result.duration_ms,
    }
    return JSONResponse(content=payload)


# ── Synchronous pipeline (runs in thread pool) ────────────────────────────────

def _run_pipeline(
    nifti_bytes: bytes,
    bval_bytes:  bytes,
    bvec_bytes:  bytes,
    nifti_name:  str,
    config:      TractographyConfig,
) -> 'TractographyResult':
    """
    Synchronous entry point for the full DTI tractography pipeline.

    Runs inside asyncio.to_thread() so the event loop is not blocked.
    All dipy and nibabel operations are GIL-releasing or numpy-based,
    so multiple concurrent requests would proceed in parallel (subject to
    the ThreadPoolExecutor worker limit).

    Steps:
      1. Parse .bval / .bvec bytes → GradientTable.
      2. Write NIfTI to a temp file; load with nibabel.
      3. Validate the loaded volume is 4-D.
      4. Extract brain mask (median_otsu).
      5. Fit TensorModel (WLS) → FA, eigenvectors, raw TensorFit.
      6. Run deterministic LocalTracking with tensor ODF direction getter.
      7. Filter + decimate streamlines.
      8. Serialise and return TractographyResult.

    The temp directory is always removed in the finally block, even if
    the pipeline fails partway through.
    """
    from dti.compression import TractographyResult  # local import to avoid circularity

    t0 = time.perf_counter()

    # ── Step 1: gradient table ─────────────────────────────────────────────
    try:
        gtab = build_gradient_table(bval_bytes, bvec_bytes, config.b0_threshold)
    except ValueError as exc:
        raise ValueError(f"Gradient table error: {exc}") from exc

    # ── Step 2: load NIfTI from temp file ─────────────────────────────────
    # nibabel requires a real filesystem path for gzip-compressed .nii.gz
    # files (gzip needs seekable random access that BytesIO doesn't provide).
    # Writing to a temp file handles both .nii and .nii.gz uniformly.
    tmpdir = tempfile.mkdtemp(prefix='dti_tract_')
    try:
        nifti_path = _write_bytes(tmpdir, nifti_name, nifti_bytes)

        try:
            import nibabel as nib                  # type: ignore[import]
            img = nib.load(nifti_path)
        except Exception as exc:
            raise ValueError(f"nibabel could not parse the NIfTI file: {exc}") from exc

        # Load as float32 to halve memory usage vs float64.
        # get_fdata(dtype=np.float32) returns float32 directly from nibabel's
        # data scaler, avoiding an extra copy.
        try:
            data: np.ndarray = img.get_fdata(dtype=np.float32)
        except MemoryError:
            raise MemoryError(
                "Insufficient RAM to load the DWI volume.  "
                "Try a lower-resolution dataset."
            )

        affine: np.ndarray = img.affine.astype(np.float64)

        # ── Step 3: validate dimensionality ───────────────────────────────
        if data.ndim != 4:
            raise ValueError(
                f"Expected a 4-D DWI NIfTI volume (x, y, z, N_volumes); "
                f"got shape {list(data.shape)}.  "
                "SynthSeg / structural NIfTI files are not accepted here — "
                "upload the raw DWI acquisition."
            )

        n_vols = data.shape[3]
        n_gradients = len(gtab.bvals)
        if n_vols != n_gradients:
            raise ValueError(
                f"Volume count mismatch: the NIfTI has {n_vols} volumes but "
                f"the gradient table has {n_gradients} entries.  "
                "Ensure the .bval and .bvec files match this exact DWI acquisition."
            )

        # ── Step 4: brain extraction ───────────────────────────────────────
        masked_data, mask = extract_brain_mask(
            data,
            b0s_mask=gtab.b0s_mask,
            median_radius=config.median_radius,
            numpass=config.numpass,
        )

        n_brain_voxels = int(np.sum(mask))
        if n_brain_voxels < 1000:
            raise ValueError(
                f"Brain mask contains only {n_brain_voxels} voxels — this is "
                "abnormally small.  Check that the uploaded file is a valid 4-D "
                "DWI acquisition and not a structural or functional scan."
            )

        # ── Step 5: tensor model fitting ──────────────────────────────────
        # After this call, tensor.fa is the FA map and tensor._raw_fit
        # is the dipy TensorFit object (needed for ODF evaluation in tracking).
        tensor = fit_tensor_model(
            masked_data,
            gtab,
            mask,
            method=config.fit_method,
        )

        # ── Step 6: deterministic tractography ────────────────────────────
        raw_streamlines = run_tractography(
            data=masked_data,
            affine=affine,
            gtab=gtab,
            mask=mask,
            tensor=tensor,
            config=config,
        )
        n_before_filter = len(raw_streamlines)

        # ── Step 7: filter + decimate ─────────────────────────────────────
        decimated = compress_and_filter(raw_streamlines, config)

        # ── Step 8: serialise ─────────────────────────────────────────────
        streamlines_json = serialize_streamlines(decimated, config.decimal_places)

        # Compute mean length of the OUTPUT (post-filter) streamlines.
        if len(decimated) > 0:
            lens = [
                float(np.sum(np.linalg.norm(np.diff(sl, axis=0), axis=1)))
                for sl in decimated
                if len(sl) >= 2
            ]
            mean_length_mm = float(np.mean(lens)) if lens else 0.0
        else:
            mean_length_mm = 0.0

        duration_ms = (time.perf_counter() - t0) * 1000.0

        return TractographyResult(
            streamlines_json=streamlines_json,
            n_streamlines=len(streamlines_json),
            n_before_filter=n_before_filter,
            mean_length_mm=round(mean_length_mm, 1),
            peak_fa=round(tensor.peak_fa, 4),
            duration_ms=round(duration_ms, 1),
        )

    finally:
        # Always clean up the temp directory, even on exception.
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Private helpers ───────────────────────────────────────────────────────────

def _write_bytes(directory: str, filename: str, content: bytes) -> str:
    """
    Write `content` to `directory / basename(filename)` and return the path.

    Using Path.name strips any directory component from the client filename
    (prevents path traversal: e.g. filename='../../etc/passwd' → 'passwd').
    """
    safe_name = Path(filename).name
    full_path = os.path.join(directory, safe_name)
    with open(full_path, 'wb') as fh:
        fh.write(content)
    return full_path


def _require_extension(
    upload:   UploadFile,
    allowed:  tuple[str, ...],
    field_name: str,
) -> None:
    """
    Raise HTTP 422 if the uploaded file's extension is not in `allowed`.
    Comparison is case-insensitive.
    """
    fname = upload.filename or ''
    ext   = Path(fname).suffix.lower()
    # .nii.gz has a double extension; check for .gz and also the stem
    if ext == '.gz' and fname.lower().endswith('.nii.gz'):
        ext = '.gz'   # accept .nii.gz
    if ext not in allowed:
        raise HTTPException(
            422,
            f"'{field_name}' must be one of {allowed} (got '{fname}').",
        )


def _require_nonempty(content: bytes, field_name: str) -> None:
    """Raise HTTP 422 if the byte string is empty."""
    if not content:
        raise HTTPException(422, f"'{field_name}' file is empty.")
