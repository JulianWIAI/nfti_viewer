"""
syn_registration — Affine + SyN diffeomorphic inter-subject registration.
══════════════════════════════════════════════════════════════════════════

Tracks structural differences between two different brains (Subject A
reference vs. Subject B moving) by first aligning them with an affine
transform, then refining with a non-linear SyN diffeomorphic warp.

Package structure:

    config.py    SynConfig dataclass + DEFAULT_SYN_CONFIG singleton
    pipeline.py  Full affine + SyN pipeline; check_syn_dependencies()

Public re-exports allow the router to import everything from one place:

    from syn_registration import (
        DEFAULT_SYN_CONFIG,
        SynConfig,
        check_syn_dependencies,
        run_syn_pipeline,
    )
"""

from .config   import SynConfig, DEFAULT_SYN_CONFIG
from .pipeline import check_syn_dependencies, run_syn_pipeline, run_syn_pipeline_with_seg

__all__ = [
    'SynConfig',
    'DEFAULT_SYN_CONFIG',
    'check_syn_dependencies',
    'run_syn_pipeline',
    'run_syn_pipeline_with_seg',
]
