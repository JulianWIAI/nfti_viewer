"""
mri_analysis.py — MRI tissue classification and volumetrics
============================================================
FastAPI router that operates on the label map returned by POST /api/segment.

Endpoint
--------
POST /api/mri/volumetrics
    Accepts the base64-encoded uint8 label array, the voxel grid dimensions,
    and the NIfTI voxel spacing.  Returns:
      • Hippocampal volume (left and right, in mm³) with a normative reference.
      • Gray matter / white matter / CSF total volumes and their brain fractions.
      • A hippocampal asymmetry index (a clinically relevant metric for temporal
        lobe epilepsy and Alzheimer's disease screening).

FreeSurfer label taxonomy
─────────────────────────
SynthSeg v2 outputs 33 classes (including background) mapped to FreeSurfer
label IDs.  This module groups them into three tissue macrostructures:

  Gray Matter  (gm)  — cerebral cortex, subcortical nuclei, cerebellar cortex.
  White Matter (wm)  — cerebral WM, cerebellar WM, brain stem.
  CSF                — lateral / 3rd / 4th ventricles and extra-ventricular CSF.

Normative reference
───────────────────
Hippocampal volume norms are derived from published meta-analyses of healthy
adult cohorts aged 18–65 scanned at 1.5 T with FreeSurfer v6 parcellation
(Wenger et al. 2014; Jack et al. 2010).  Reported as mean ± 1 SD per
hemisphere.  These values are approximate and should not replace clinical
assessment.

Dependencies
────────────
numpy  — already required by main.py; no additional installation needed.
"""

from __future__ import annotations

import base64

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/mri", tags=["MRI Analysis"])


# ── FreeSurfer label → tissue class mapping ───────────────────────────────────
# Keys are FreeSurfer integer label IDs as used by SynthSeg v2.
# Values: "bg" = background/skip, "gm" = gray matter, "wm" = white matter,
#         "csf" = cerebrospinal fluid.

LABEL_TISSUE_MAP: dict[int, str] = {
    0:  "bg",    # background (non-brain voxels)

    # ── Left hemisphere ───────────────────────────────────────────────────────
    2:  "wm",    # left cerebral white matter
    3:  "gm",    # left cerebral cortex
    4:  "csf",   # left lateral ventricle
    5:  "csf",   # left inferior lateral ventricle
    7:  "wm",    # left cerebellum white matter
    8:  "gm",    # left cerebellum cortex
    10: "gm",    # left thalamus    (subcortical gray matter)
    11: "gm",    # left caudate     (subcortical GM)
    12: "gm",    # left putamen     (subcortical GM)
    13: "gm",    # left pallidum    (subcortical GM)
    14: "csf",   # 3rd ventricle
    15: "csf",   # 4th ventricle
    16: "wm",    # brain stem       (predominantly myelinated tracts)
    17: "gm",    # left hippocampus (medial temporal GM)
    18: "gm",    # left amygdala    (medial temporal GM)
    24: "csf",   # CSF              (extra-ventricular, e.g. sulcal CSF)
    26: "gm",    # left accumbens area (subcortical GM)
    28: "gm",    # left ventral diencephalon (subcortical GM)

    # ── Right hemisphere (mirror labels in FreeSurfer convention) ─────────────
    41: "wm",    # right cerebral white matter
    42: "gm",    # right cerebral cortex
    43: "csf",   # right lateral ventricle
    44: "csf",   # right inferior lateral ventricle
    46: "wm",    # right cerebellum white matter
    47: "gm",    # right cerebellum cortex
    49: "gm",    # right thalamus
    50: "gm",    # right caudate
    51: "gm",    # right putamen
    52: "gm",    # right pallidum
    53: "gm",    # right hippocampus
    54: "gm",    # right amygdala
    58: "gm",    # right accumbens area
    60: "gm",    # right ventral diencephalon
}

# Pre-built uint8 arrays of label IDs per tissue class.
# np.isin() with these arrays is O(n_voxels) and fully vectorised — far
# faster than iterating over the label map in a Python loop.
_GM_LABELS  = np.array([k for k, v in LABEL_TISSUE_MAP.items() if v == "gm"],  dtype=np.uint8)
_WM_LABELS  = np.array([k for k, v in LABEL_TISSUE_MAP.items() if v == "wm"],  dtype=np.uint8)
_CSF_LABELS = np.array([k for k, v in LABEL_TISSUE_MAP.items() if v == "csf"], dtype=np.uint8)


# ── Normative hippocampal volumes ─────────────────────────────────────────────
# Source: pooled estimates from Wenger et al. (2014) Neurobiology of Aging
# and Jack et al. (2010) NeuroImage; 1.5 T / FreeSurfer v6; adults 18–65 yrs.
# Values represent mean ± 1 SD per hemisphere in mm³.
NORMATIVE_HIPPO_MEAN_MM3: float = 3900.0   # mm³ per hemisphere
NORMATIVE_HIPPO_SD_MM3:   float =  400.0   # mm³ (1 standard deviation)


# ── Pydantic request / response models ────────────────────────────────────────

class VolumetricsRequest(BaseModel):
    """
    Payload sent by the frontend after a successful /api/segment call.

    labels_b64
        The ``labels`` field from the SegmentResult — base64-encoded flat
        uint8 array of FreeSurfer label IDs in **Fortran order** (X varies
        fastest, matching the vtk.js vtkImageData layout).
    dims
        [x, y, z] voxel grid dimensions, taken from ``SegmentResult.dims``.
    voxel_mm
        [dx, dy, dz] voxel spacing in millimetres.  For a NIfTI file these
        are ``header.pixDims[1..3]``.
    """

    labels_b64: str         = Field(description="Base64 uint8 Fortran-order label array")
    dims:       list[int]   = Field(description="[x, y, z] voxel dimensions")
    voxel_mm:   list[float] = Field(description="[dx, dy, dz] voxel spacing in mm")


class HippocampalVolumes(BaseModel):
    """Per-hemisphere hippocampal volume with a normative reference band."""

    left_mm3:           float = Field(description="Left hippocampal volume (mm³)")
    right_mm3:          float = Field(description="Right hippocampal volume (mm³)")
    # Asymmetry index: positive = left larger; negative = right larger.
    # Formula: (L − R) / ((L + R) / 2) × 100.
    # Values > ±15 % warrant clinical attention in an epilepsy work-up.
    asymmetry_index:    float = Field(description="(L − R) / mean × 100  [%]")
    normative_mean_mm3: float = Field(description="Normative mean per hemisphere (mm³)")
    normative_sd_mm3:   float = Field(description="Normative 1-SD range (mm³)")


class TissueVolumes(BaseModel):
    """Absolute and fractional tissue-class volumes for the whole brain."""

    gm_mm3:       float = Field(description="Total gray matter volume (mm³)")
    wm_mm3:       float = Field(description="Total white matter volume (mm³)")
    csf_mm3:      float = Field(description="Total CSF volume (mm³)")
    # Fractions are relative to the sum of GM + WM + CSF (excludes background).
    gm_fraction:  float = Field(description="GM / (GM + WM + CSF)")
    wm_fraction:  float = Field(description="WM / (GM + WM + CSF)")
    csf_fraction: float = Field(description="CSF / (GM + WM + CSF)")


class VolumetricsResponse(BaseModel):
    hippocampus:      HippocampalVolumes
    tissue_volumes:   TissueVolumes
    voxel_volume_mm3: float = Field(description="Volume of one voxel (mm³)")
    total_brain_mm3:  float = Field(description="Total brain parenchyma = GM + WM (mm³)")


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/volumetrics",
    response_model=VolumetricsResponse,
    summary="Compute tissue volumes from a SynthSeg label map",
)
async def compute_volumetrics(req: VolumetricsRequest) -> VolumetricsResponse:
    """
    Decode the base64 label array returned by ``POST /api/segment`` and
    compute voxel-count-based volumes for the hippocampus and the three
    tissue macrostructures.

    The label array is expected in **Fortran order** (X index varies fastest),
    which is the byte layout used by both nibabel and vtk.js ``vtkImageData``.
    Reshaping with ``order='F'`` ensures that index ``[i, j, k]`` maps to the
    same voxel as in the original NIfTI file.
    """
    # ── 1. Decode and validate ────────────────────────────────────────────────
    try:
        raw_bytes = base64.b64decode(req.labels_b64)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid base64: {exc}") from exc

    if len(req.dims) != 3:
        raise HTTPException(status_code=422, detail="dims must have exactly 3 elements [x, y, z]")
    if len(req.voxel_mm) != 3:
        raise HTTPException(status_code=422, detail="voxel_mm must have exactly 3 elements [dx, dy, dz]")

    expected_voxels = req.dims[0] * req.dims[1] * req.dims[2]
    if len(raw_bytes) != expected_voxels:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Label byte count ({len(raw_bytes)}) does not match "
                f"dims {req.dims} (expected {expected_voxels} bytes)."
            ),
        )

    # ── 2. Reshape to 3-D voxel grid (Fortran order = X fastest) ─────────────
    labels: np.ndarray = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
        req.dims, order="F"
    )

    # ── 3. Voxel volume ───────────────────────────────────────────────────────
    voxel_vol: float = float(
        req.voxel_mm[0] * req.voxel_mm[1] * req.voxel_mm[2]
    )

    # ── 4. Hippocampal voxel counts ───────────────────────────────────────────
    # FreeSurfer label IDs: 17 = left hippocampus, 53 = right hippocampus.
    left_vox:  int = int(np.sum(labels == 17))
    right_vox: int = int(np.sum(labels == 53))
    left_mm3:  float = round(left_vox  * voxel_vol, 2)
    right_mm3: float = round(right_vox * voxel_vol, 2)

    # Asymmetry index: (L − R) / mean × 100 %.
    # Guard division by zero when both hippocampi are absent (e.g. phantom data).
    denom = (left_mm3 + right_mm3) / 2.0 if (left_mm3 + right_mm3) > 0 else 1.0
    asym_index = round((left_mm3 - right_mm3) / denom * 100.0, 2)

    # ── 5. Tissue-class voxel counts (vectorised with np.isin) ────────────────
    gm_vox:  int = int(np.isin(labels, _GM_LABELS).sum())
    wm_vox:  int = int(np.isin(labels, _WM_LABELS).sum())
    csf_vox: int = int(np.isin(labels, _CSF_LABELS).sum())

    # Total classified tissue (excludes background label 0).
    total_tissue: int = gm_vox + wm_vox + csf_vox or 1   # avoid /0 on empty mask

    gm_mm3:  float = round(gm_vox  * voxel_vol, 2)
    wm_mm3:  float = round(wm_vox  * voxel_vol, 2)
    csf_mm3: float = round(csf_vox * voxel_vol, 2)

    # ── 6. Assemble response ──────────────────────────────────────────────────
    return VolumetricsResponse(
        hippocampus=HippocampalVolumes(
            left_mm3=left_mm3,
            right_mm3=right_mm3,
            asymmetry_index=asym_index,
            normative_mean_mm3=NORMATIVE_HIPPO_MEAN_MM3,
            normative_sd_mm3=NORMATIVE_HIPPO_SD_MM3,
        ),
        tissue_volumes=TissueVolumes(
            gm_mm3=gm_mm3,
            wm_mm3=wm_mm3,
            csf_mm3=csf_mm3,
            gm_fraction=round(gm_vox  / total_tissue, 4),
            wm_fraction=round(wm_vox  / total_tissue, 4),
            csf_fraction=round(csf_vox / total_tissue, 4),
        ),
        voxel_volume_mm3=round(voxel_vol, 6),
        total_brain_mm3=round(gm_mm3 + wm_mm3, 2),
    )
