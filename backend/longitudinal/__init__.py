"""
longitudinal — NIfTI longitudinal delta pipeline.
══════════════════════════════════════════════════

Tracks brain change between two structural MRI sessions by co-registering
the follow-up volume to the baseline using dipy affine registration (mutual
information metric + multi-scale pyramid), then computing a voxel-wise
subtraction map (the "delta").

Package structure:

    config.py        LongitudinalConfig dataclass + DEFAULT_CONFIG singleton
    preprocess.py    NIfTI loading, volume extraction, dependency checks
    registration.py  dipy MI affine registration (CoM → Translation → Rigid → Affine)
    delta.py         Subtraction map, summary statistics, Fortran-order serialisation

Public re-exports allow the router to import everything from one place:

    from longitudinal import (
        DEFAULT_CONFIG,
        check_dependencies, check_dipy,
        load_nifti, extract_volume,
        run_registration,
        compute_delta, serialise_delta,
    )
"""

from .config import LongitudinalConfig, DEFAULT_CONFIG

from .preprocess import (
    check_dependencies,
    load_nifti,
    extract_volume,
)

from .registration import (
    check_dipy,
    run_registration,
)

from .delta import (
    compute_delta,
    serialise_delta,
)

__all__ = [
    # Configuration
    'LongitudinalConfig',
    'DEFAULT_CONFIG',
    # Preprocessing
    'check_dependencies',
    'load_nifti',
    'extract_volume',
    # Registration
    'check_dipy',
    'run_registration',
    # Delta computation
    'compute_delta',
    'serialise_delta',
]
