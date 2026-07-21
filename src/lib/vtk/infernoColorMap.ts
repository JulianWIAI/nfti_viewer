/**
 * infernoColorMap.ts — Inferno perceptual colour transfer function for BOLD fMRI
 * ────────────────────────────────────────────────────────────────────────────────
 *
 * Builds a vtkColorTransferFunction that approximates the "inferno" perceptual
 * colormap (Matplotlib / D3) with 7 control points:
 *
 *   0.000 → (0.000, 0.000, 0.016)  near-black
 *   0.142 → (0.176, 0.016, 0.329)  deep purple
 *   0.285 → (0.459, 0.031, 0.529)  violet
 *   0.428 → (0.706, 0.122, 0.373)  red-magenta
 *   0.571 → (0.898, 0.353, 0.114)  orange
 *   0.714 → (0.988, 0.647, 0.039)  amber-yellow
 *   1.000 → (0.988, 1.000, 0.643)  bright yellow-white
 *
 * WHY INFERNO FOR BOLD?
 * ──────────────────────
 * • Monotonically increasing luminance  →  perceptually uniform, no false
 *   features at iso-luminance steps (unlike rainbow/jet).
 * • Black at 0 signal  →  voxels with no BOLD modulation are invisible against
 *   the dark radiological background; no masking threshold required.
 * • High-luminance yellow peak  →  peak activation pops clearly on dark screens.
 * • Colour-blind safe across the most common deficiencies (deuteranopia, protanopia).
 *
 * This matches the inferno palette used in nilearn, AFNI, and FSLeyes.
 *
 * USAGE
 * ──────
 *   import { buildInfernoColorTransferFunction } from './infernoColorMap';
 *
 *   const ctf = buildInfernoColorTransferFunction({ minValue: 0, maxValue: 6.5 });
 *   glyphMapper.setLookupTable(ctf);
 *   glyphMapper.setUseLookupTableScalarRange(true);
 */

import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';

// ── Options ───────────────────────────────────────────────────────────────────

export interface InfernoColorMapOptions {
  /**
   * Scalar value mapped to the darkest colour (near-black).
   * Typically 0 or the statistical threshold (e.g. z > 2.3).
   */
  minValue: number;
  /**
   * Scalar value mapped to the brightest colour (yellow-white).
   * Typically the 99th-percentile amplitude or a fixed statistical ceiling.
   */
  maxValue: number;
}

// ── Factory ───────────────────────────────────────────────────────────────────

/**
 * Build an inferno vtkColorTransferFunction scaled to [minValue, maxValue].
 *
 * @param options - Scalar range for the colour mapping.
 */
export function buildInfernoColorTransferFunction(
  options: InfernoColorMapOptions,
): ReturnType<typeof vtkColorTransferFunction.newInstance> {
  const { minValue, maxValue } = options;

  // Protect against a degenerate range (all values identical).
  const span = maxValue - minValue || 1.0;
  const lo   = minValue;

  const ctf = vtkColorTransferFunction.newInstance();

  // addRGBPoint(scalar, R, G, B)  — all channels in linear [0, 1].
  // The 7 control points below are sampled from the 256-colour inferno LUT
  // at evenly spaced positions, then rounded to 3 decimal places.

  ctf.addRGBPoint(lo + 0.000 * span,  0.000, 0.000, 0.016); // near-black
  ctf.addRGBPoint(lo + 0.142 * span,  0.176, 0.016, 0.329); // deep purple
  ctf.addRGBPoint(lo + 0.285 * span,  0.459, 0.031, 0.529); // violet
  ctf.addRGBPoint(lo + 0.428 * span,  0.706, 0.122, 0.373); // red-magenta
  ctf.addRGBPoint(lo + 0.571 * span,  0.898, 0.353, 0.114); // orange
  ctf.addRGBPoint(lo + 0.714 * span,  0.988, 0.647, 0.039); // amber-yellow
  ctf.addRGBPoint(lo + 1.000 * span,  0.988, 1.000, 0.643); // bright yellow-white

  // Clamp extrapolation so scalars below min stay black and above max stay white.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (ctf as any).setClampingOn();

  return ctf;
}
