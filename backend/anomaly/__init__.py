"""
anomaly — AnomalySeg preprocessing, inference, and postprocessing package.
══════════════════════════════════════════════════════════════════════════════

Package structure:

    config.py      AnomalyConfig dataclass + DEFAULT_CONFIG singleton
    preprocess.py  NIfTI loading, resampling, normalisation, crop/pad, tensor build
    inference.py   ONNX session caching, model execution, binary mask extraction
    postprocess.py Uncrop/unpad, inverse resample, Fortran-order serialisation

Public re-exports allow the router to import everything from one place:

    from anomaly import (
        DEFAULT_CONFIG,
        check_dependencies, check_onnx,
        load_nifti, extract_volume,
        resample_to_isotropic, normalise, crop_pad, build_tensor,
        get_session, run_inference, extract_binary_mask,
        uncrop_unpad, resample_to_original, serialise_mask,
    )
"""

from .config import AnomalyConfig, DEFAULT_CONFIG

from .preprocess import (
    check_dependencies,
    load_nifti,
    extract_volume,
    resample_to_isotropic,
    normalise,
    crop_pad,
    build_tensor,
)

from .inference import (
    check_onnx,
    get_session,
    run_inference,
    extract_binary_mask,
)

from .postprocess import (
    uncrop_unpad,
    resample_to_original,
    serialise_mask,
)

__all__ = [
    # Configuration
    "AnomalyConfig",
    "DEFAULT_CONFIG",
    # Preprocessing
    "check_dependencies",
    "load_nifti",
    "extract_volume",
    "resample_to_isotropic",
    "normalise",
    "crop_pad",
    "build_tensor",
    # Inference
    "check_onnx",
    "get_session",
    "run_inference",
    "extract_binary_mask",
    # Postprocessing
    "uncrop_unpad",
    "resample_to_original",
    "serialise_mask",
]
