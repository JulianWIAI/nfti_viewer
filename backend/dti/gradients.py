"""
dti/gradients.py — Parse .bval / .bvec bytes → dipy GradientTable
════════════════════════════════════════════════════════════════════

WHY THREE FILES?
─────────────────
A diffusion-weighted MRI acquisition is described by its gradient table:

  .bval  — one b-value per volume (scalar; s/mm²)
           b=0 volumes have no diffusion weighting (reference signal S₀)
           DW volumes typically have b = 700–3000 s/mm²

  .bvec  — one unit gradient direction per volume (3-component vector)
           by convention: 3 rows × N columns  (FSL format)
           alternatively:  N rows × 3 columns  (BIDS / row-per-volume format)

  Together they form a GradientTable: N pairs of (b_i, g_i) used to design
  and then interpret the diffusion-sensitisation applied to each volume.

DIFFUSION SIGNAL MODEL
───────────────────────
The Stejskal-Tanner equation gives the expected signal attenuation:

    S_i / S_0 = exp(−b_i · g_i^T · D · g_i)

  where D is the 3×3 diffusion tensor we want to estimate.

Providing the wrong b-values or mis-oriented b-vectors gives a completely
wrong tensor (and thus wrong FA map and wrong tractography).  Validating
shape agreement between .bval and .bvec is therefore essential.
"""

from __future__ import annotations

import numpy as np


def _parse_bvals(content: bytes) -> np.ndarray:
    """
    Parse the text content of a .bval file into a 1-D float64 array.

    .bval files contain N whitespace-separated floating-point numbers
    (one per DWI volume), possibly spread across multiple lines.
    The FSL convention stores everything on a single line; some tools
    use one value per line.  We handle both by splitting on any whitespace.

    Parameters
    ----------
    content : raw bytes of the .bval file

    Returns
    -------
    bvals : (N,) float64 array of b-values in s/mm²
    """
    text = content.decode('utf-8', errors='replace')
    tokens = text.split()           # splits on any run of whitespace / newlines
    if not tokens:
        raise ValueError("bval file is empty or contains no numeric values.")
    return np.array(tokens, dtype=np.float64)


def _parse_bvecs(content: bytes) -> np.ndarray:
    """
    Parse the text content of a .bvec file into an (N, 3) float64 array.

    FSL standard format: 3 rows × N columns.
      Row 0 → x-components of all N gradient directions
      Row 1 → y-components
      Row 2 → z-components

    Some tools write N rows × 3 columns (one gradient per line).  We detect
    this by checking whether the parsed array has exactly 3 rows with more
    than 3 columns (FSL format) or more than 3 rows with exactly 3 columns
    (column format), and transpose accordingly.

    Parameters
    ----------
    content : raw bytes of the .bvec file

    Returns
    -------
    bvecs : (N, 3) float64 array of unit gradient direction vectors
    """
    text   = content.decode('utf-8', errors='replace')
    rows   = [
        [float(v) for v in line.split()]
        for line in text.strip().splitlines()
        if line.strip()
    ]

    if not rows:
        raise ValueError("bvec file is empty or contains no numeric values.")

    arr = np.array(rows, dtype=np.float64)   # shape varies by format

    # ── Auto-detect and normalise to (N, 3) ──────────────────────────────
    if arr.ndim != 2:
        raise ValueError(f"Unexpected bvec array shape: {arr.shape}")

    if arr.shape[0] == 3 and arr.shape[1] != 3:
        # FSL row-major format → (3, N): transpose to (N, 3)
        arr = arr.T
    elif arr.shape[1] != 3:
        raise ValueError(
            f"Cannot interpret bvec array of shape {arr.shape}: "
            "expected either (3, N) or (N, 3)."
        )

    # Result is (N, 3)
    return arr


def build_gradient_table(
    bvals_bytes:  bytes,
    bvecs_bytes:  bytes,
    b0_threshold: float = 50.0,
) -> object:  # returns dipy.core.gradients.GradientTable
    """
    Build a dipy GradientTable from the raw bytes of .bval and .bvec files.

    The GradientTable encodes the full diffusion encoding scheme and exposes:
      gtab.bvals      — (N,) b-values
      gtab.bvecs      — (N, 3) unit gradient vectors
      gtab.b0s_mask   — boolean mask of b0 volumes (b ≤ b0_threshold)
      gtab.gradients  — (N, 3) b_i * g_i  (scaled gradient vectors)

    Parameters
    ----------
    bvals_bytes  : raw .bval file bytes
    bvecs_bytes  : raw .bvec file bytes
    b0_threshold : volumes with b ≤ this value are classified as b=0

    Returns
    -------
    gtab : dipy GradientTable

    Raises
    ------
    ValueError : if bval/bvec lengths disagree or parsing fails
    """
    # Import inside function so the module can be imported without dipy
    # (useful for type-checking passes where dipy is not installed).
    from dipy.core.gradients import gradient_table  # type: ignore[import]

    bvals = _parse_bvals(bvals_bytes)
    bvecs = _parse_bvecs(bvecs_bytes)

    # Sanity check: number of volumes must match between bvals and bvecs.
    if bvals.shape[0] != bvecs.shape[0]:
        raise ValueError(
            f"bval / bvec length mismatch: "
            f"{bvals.shape[0]} b-values vs {bvecs.shape[0]} gradient directions."
        )

    n_b0 = int(np.sum(bvals <= b0_threshold))
    if n_b0 == 0:
        raise ValueError(
            f"No b=0 volumes found with b0_threshold={b0_threshold}. "
            "Check that the .bval file contains at least one b≈0 volume "
            "(the undiffusion-weighted reference image)."
        )

    return gradient_table(bvals, bvecs, b0_threshold=b0_threshold)
