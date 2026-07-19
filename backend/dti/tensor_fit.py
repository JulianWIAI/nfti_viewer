"""
dti/tensor_fit.py — DTI Tensor Model fitting and scalar-map computation
════════════════════════════════════════════════════════════════════════

DIFFUSION TENSOR MODEL — MATHEMATICAL BACKGROUND
─────────────────────────────────────────────────

The Gaussian diffusion model assumes that in each voxel the displacement
probability of a water molecule follows a 3-D Gaussian whose covariance
is the symmetric 3×3 diffusion tensor D.

Bloch-Torrey signal equation (Stejskal-Tanner):

    S(b, g) = S₀ · exp(−b · gᵀ D g)

  where
    S₀ = signal without diffusion weighting (b = 0 s/mm²)
    b  = diffusion-weighting strength  (s/mm²)
    g  = unit gradient direction vector  (3 × 1)
    D  = symmetric 3×3 diffusion tensor  (6 unique elements)

Log-linearisation:

    ln(S_i / S₀) = −b_i · g_iᵀ D g_i

This reduces the problem to a linear system: for N DWI volumes we have N
equations and 6 unknowns (the unique tensor elements).  WLS (weighted
least-squares) weights each equation by its predicted signal magnitude to
downweight high-attenuation (noisy) measurements.

EIGENDECOMPOSITION
──────────────────
Diagonalising D gives:

    D = V Λ Vᵀ

  where
    Λ = diag(λ₁, λ₂, λ₃)  with λ₁ ≥ λ₂ ≥ λ₃ (eigenvalues, mm²/s)
    V = [e₁ | e₂ | e₃]     (eigenvectors = principal diffusion directions)

The principal eigenvector e₁ (column 0 of evecs) points along the axis of
fastest diffusion — i.e. along the local fibre orientation.  This is the
direction used by the tractography algorithm.

FRACTIONAL ANISOTROPY (FA)
──────────────────────────
                     ┌───────────────────────────────────────────┐
                     │           (λ₁-λ̄)² + (λ₂-λ̄)² + (λ₃-λ̄)²  │
    FA = √(3/2) · √ │  ───────────────────────────────────────  │
                     │           λ₁² + λ₂² + λ₃²               │
                     └───────────────────────────────────────────┘
    λ̄ = (λ₁ + λ₂ + λ₃) / 3   (mean diffusivity)

  FA = 0  →  perfectly isotropic (CSF, random diffusion in all directions)
  FA = 1  →  perfectly anisotropic (hypothetical 1-D diffusion along one axis)

Clinical white matter:   FA ≈ 0.3–0.9
Grey matter:             FA ≈ 0.1–0.2
CSF:                     FA ≈ 0.0–0.1

MEAN DIFFUSIVITY (MD)
─────────────────────
    MD = (λ₁ + λ₂ + λ₃) / 3     (mm²/s)

Elevated MD indicates cell loss / oedema; reduced MD indicates cytotoxic
oedema.  We compute it here for completeness but use FA for tractography.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class TensorFitResult:
    """
    Output of a TensorModel.fit() call, with derived scalar maps.

    The `_raw_fit` field stores the dipy TensorFit object so that the
    tracking module can call `_raw_fit.odf(sphere)` to evaluate the
    orientation distribution function on a discrete sphere — avoiding
    the need to re-fit the tensor model a second time.

    Attributes
    ----------
    fa        : (x, y, z) float32 — Fractional Anisotropy, range [0, 1],
                NaN-cleaned (background/noise voxels are set to 0)
    md        : (x, y, z) float32 — Mean Diffusivity in mm²/s
    evecs     : (x, y, z, 3, 3) float32 — eigenvector matrix;
                column i is eigenvector i in descending eigenvalue order
    evals     : (x, y, z, 3) float32 — eigenvalues λ₁ ≥ λ₂ ≥ λ₃ in mm²/s
    peak_fa   : float — maximum FA value across the brain mask
    _raw_fit  : dipy TensorFit — kept private; consumed by tracking.py
    """
    fa:       np.ndarray
    md:       np.ndarray
    evecs:    np.ndarray
    evals:    np.ndarray
    peak_fa:  float
    _raw_fit: object  # dipy TensorFit — not serialisable, used only internally


# ── Fitting function ─────────────────────────────────────────────────────────

def fit_tensor_model(
    data:   np.ndarray,
    gtab:   object,     # dipy GradientTable
    mask:   np.ndarray,
    method: str = 'WLS',
) -> TensorFitResult:
    """
    Fit the Diffusion Tensor model to masked 4-D DWI data.

    Parameters
    ----------
    data   : (x, y, z, N) float32 — full 4-D DWI volume (may be brain-masked)
    gtab   : dipy GradientTable (from gradients.build_gradient_table)
    mask   : (x, y, z) bool — True inside brain (outside voxels are skipped)
    method : WLS (recommended) or OLS (faster, noisier)

    Returns
    -------
    TensorFitResult with FA, MD, eigenvectors, eigenvalues, and the raw
    dipy TensorFit object for ODF evaluation in tracking.py.

    Notes
    -----
    Fitting is performed only on voxels where mask=True; background voxels
    get eigenvalues = 0 and FA = 0, producing a zero-padded output.

    np.nan_to_num is applied to FA and MD to replace any NaN that arises
    from division-by-zero in degenerate voxels (e.g. zero signal in all DWI
    directions, which yields rank-deficient normal equations).
    """
    from dipy.reconst.dti import TensorModel  # type: ignore[import]

    # Instantiate and fit the tensor model.
    # `fit_method` selects the numerical method:
    #   'WLS' — weighted least squares: more robust but slightly slower
    #   'OLS' — ordinary least squares: faster but sensitive to outliers
    model  = TensorModel(gtab, fit_method=method)
    tenfit = model.fit(data, mask=mask)

    # ── FA (Fractional Anisotropy) ─────────────────────────────────────────
    # Clip to [0, 1] to handle tiny numerical overshoots (> 1 can occur in
    # voxels with very low SNR where the WLS solution is ill-conditioned).
    fa = np.clip(tenfit.fa, 0.0, 1.0).astype(np.float32)
    fa = np.nan_to_num(fa, nan=0.0, posinf=0.0, neginf=0.0)

    # ── MD (Mean Diffusivity) — diagnostic scalar map ─────────────────────
    md = np.nan_to_num(tenfit.md, nan=0.0).astype(np.float32)

    # ── Eigenvectors and eigenvalues ──────────────────────────────────────
    # evecs[:,:,:,:,i] = eigenvector i  (i=0 is the principal direction)
    # evals[:,:,:,i]   = eigenvalue  i  in mm²/s
    # Dipy guarantees descending eigenvalue order (λ₁ ≥ λ₂ ≥ λ₃).
    evecs = np.nan_to_num(tenfit.evecs, nan=0.0).astype(np.float32)
    evals = np.nan_to_num(tenfit.evals, nan=0.0).astype(np.float32)

    peak_fa = float(np.max(fa[mask.astype(bool)]))

    return TensorFitResult(
        fa=fa,
        md=md,
        evecs=evecs,
        evals=evals,
        peak_fa=peak_fa,
        _raw_fit=tenfit,
    )
