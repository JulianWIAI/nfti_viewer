#!/usr/bin/env python3
"""
download_models.py — Fetch and install the SynthSeg ONNX model
═══════════════════════════════════════════════════════════════

Usage
─────
  python backend/download_models.py

What it does
────────────
1. Checks whether backend/models/SynthSeg.onnx already exists.
2. Tries each acquisition strategy in order:
   a) FreeSurfer local installation  ($FREESURFER_HOME/models/)
   b) GitHub release asset (.h5) + tf2onnx conversion
   c) Prints manual instructions and exits with code 1.

Requirements for strategy (b):
  pip install tensorflow tf2onnx requests

The final segmentation endpoint only needs:
  pip install nibabel scipy onnxruntime
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

MODELS_DIR  = Path(__file__).parent / "models"
ONNX_PATH   = MODELS_DIR / "SynthSeg.onnx"

# SynthSeg GitHub repository
_GITHUB_API  = "https://api.github.com/repos/BBillot/SynthSeg/releases/latest"
_GITHUB_RAW  = "https://raw.githubusercontent.com/BBillot/SynthSeg/master/models"

# FreeSurfer bundled model locations (checked in order)
_FS_CANDIDATES = [
    Path(os.environ.get("FREESURFER_HOME", "")) / "models" / "synthseg_weights_c.h5",
    Path(os.environ.get("FREESURFER_HOME", "")) / "python" / "packages" / "freesurfer" / "samseg" / "SynthSeg_weights_c.h5",
]


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if ONNX_PATH.exists():
        print(f"✓ Model already present: {ONNX_PATH}")
        return

    print("SynthSeg.onnx not found — trying acquisition strategies…\n")

    # Strategy A: local FreeSurfer installation
    if _try_freesurfer():
        return

    # Strategy B: GitHub release H5 → ONNX conversion
    if _try_github_release():
        return

    _print_manual_instructions()
    sys.exit(1)


# ── Strategy A: FreeSurfer ────────────────────────────────────────────────────

def _try_freesurfer() -> bool:
    print("Strategy A: checking for local FreeSurfer installation…")
    for h5_path in _FS_CANDIDATES:
        if h5_path.exists():
            print(f"  Found: {h5_path}")
            return _convert_h5(h5_path)
    print("  FreeSurfer not found (or $FREESURFER_HOME not set).\n")
    return False


# ── Strategy B: GitHub release ────────────────────────────────────────────────

def _try_github_release() -> bool:
    print("Strategy B: fetching latest SynthSeg release from GitHub…")
    try:
        req = urllib.request.Request(
            _GITHUB_API,
            headers={"User-Agent": "nfti_viewer/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            release = json.load(resp)
    except Exception as exc:
        print(f"  GitHub API error: {exc}\n")
        return False

    # Find an .h5 asset in the release
    asset_url: str | None = None
    for asset in release.get("assets", []):
        name: str = asset.get("name", "")
        if name.lower().endswith(".h5") and "synthseg" in name.lower():
            asset_url = asset["browser_download_url"]
            asset_name = name
            break

    if asset_url is None:
        print("  No SynthSeg .h5 asset found in the latest release.\n")
        return False

    print(f"  Downloading {asset_name} …")
    h5_tmp = MODELS_DIR / asset_name
    try:
        urllib.request.urlretrieve(asset_url, h5_tmp)
    except Exception as exc:
        print(f"  Download failed: {exc}\n")
        h5_tmp.unlink(missing_ok=True)
        return False

    ok = _convert_h5(h5_tmp)
    h5_tmp.unlink(missing_ok=True)
    return ok


# ── H5 → ONNX conversion ─────────────────────────────────────────────────────

def _convert_h5(h5_path: Path) -> bool:
    """Convert a Keras .h5 model to ONNX using tf2onnx."""
    print(f"  Converting {h5_path.name} → SynthSeg.onnx (requires tensorflow + tf2onnx) …")
    try:
        import tensorflow as tf          # type: ignore  # noqa: F401
        import tf2onnx                   # type: ignore  # noqa: F401
    except ImportError:
        print(
            "  tensorflow or tf2onnx not installed.\n"
            "  Run:  pip install tensorflow tf2onnx\n"
        )
        return False

    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--keras", str(h5_path),
        "--output", str(ONNX_PATH),
        "--opset", "17",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not ONNX_PATH.exists():
        print(f"  Conversion failed:\n{result.stderr}\n")
        ONNX_PATH.unlink(missing_ok=True)
        return False

    size_mb = ONNX_PATH.stat().st_size / 1_048_576
    print(f"  ✓ Saved: {ONNX_PATH} ({size_mb:.1f} MB)\n")
    return True


# ── Manual instructions ───────────────────────────────────────────────────────

def _print_manual_instructions() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Automatic download failed — manual steps to install SynthSeg.onnx         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Option 1 — from FreeSurfer 7.4+                                             ║
║    Set FREESURFER_HOME and re-run this script, or copy manually:             ║
║      cp $FREESURFER_HOME/models/synthseg_weights_c.h5 /tmp/sg.h5             ║
║      pip install tensorflow tf2onnx                                          ║
║      python -m tf2onnx.convert --keras /tmp/sg.h5 \\                          ║
║             --output backend/models/SynthSeg.onnx --opset 17                ║
║                                                                              ║
║  Option 2 — from the SynthSeg GitHub releases                                ║
║    1. Download an .h5 file from:                                             ║
║       https://github.com/BBillot/SynthSeg/releases                          ║
║    2. Convert:                                                               ║
║       pip install tensorflow tf2onnx                                         ║
║       python -m tf2onnx.convert --keras SynthSeg_weights.h5 \\               ║
║              --output backend/models/SynthSeg.onnx --opset 17               ║
║                                                                              ║
║  After placing SynthSeg.onnx in backend/models/, also install:               ║
║    pip install nibabel scipy onnxruntime                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
