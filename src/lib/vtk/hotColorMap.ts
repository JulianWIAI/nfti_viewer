/**
 * hotColorMap.ts — "Hot" colour transfer function for scalar activation maps
 * ───────────────────────────────────────────────────────────────────────────
 *
 * Builds a vtkColorTransferFunction that maps a scalar range [min, max] to a
 * perceptually-ordered sequence of colours used in brain-activation heatmaps:
 *
 *   dark red → red → orange → yellow → near-white
 *
 * WHY THIS COLORMAP?
 * ───────────────────
 * The "hot" family of colormaps is standard in neuroimaging (SPM, FSL, MNE)
 * because:
 *   1. The luminance monotonically increases from min to max, so the map is
 *      approximately perceptually uniform — brighter always means stronger.
 *   2. The red channel encodes mid-range values well (retinal sensitivity to
 *      red is high), giving good contrast in typical fMRI/EEG activation maps.
 *   3. "White = peak" is intuitive for clinical audiences.
 *
 * COLORMAP CONSTRUCTION
 * ──────────────────────
 * A vtkColorTransferFunction is a piecewise-linear interpolant over the scalar
 * axis.  Each control point is (scalar_value, R, G, B) with R/G/B ∈ [0, 1].
 * Between control points the RGB channels are linearly interpolated in linear
 * light space (not gamma-encoded — vtk.js works in linear space internally).
 *
 * Control points (relative normalised position → colour):
 *
 *   0.000 → (0.30, 0.00, 0.00)   very dark red    — lowest visible amplitude
 *   0.250 → (0.80, 0.00, 0.00)   vivid red
 *   0.500 → (1.00, 0.35, 0.00)   red-orange
 *   0.700 → (1.00, 0.80, 0.00)   amber / warm yellow
 *   0.850 → (1.00, 1.00, 0.20)   yellow-white
 *   1.000 → (1.00, 1.00, 0.85)   warm near-white  — peak amplitude
 *
 * The minimum is NOT mapped to black — that would make low-amplitude sources
 * invisible against the dark MRI background.  Starting at dark red ensures
 * every returned source is visible while still maintaining a clear hierarchy.
 *
 * USAGE
 * ──────
 *   import { buildHotColorTransferFunction } from './hotColorMap';
 *
 *   const ctf = buildHotColorTransferFunction({ minValue: 0.0, maxValue: 45.2 });
 *   glyphMapper.setLookupTable(ctf);
 *   glyphMapper.setUseLookupTableScalarRange(true);
 */

import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';

// ── Options ───────────────────────────────────────────────────────────────────

export interface HotColorMapOptions {
  /**
   * Scalar value that maps to the darkest colour (dark red).
   * Set to the minimum amplitude across all source points.
   */
  minValue: number;
  /**
   * Scalar value that maps to the brightest colour (near-white).
   * Set to the maximum amplitude.  All values above this are clamped to white.
   */
  maxValue: number;
}

// ── Factory ───────────────────────────────────────────────────────────────────

/**
 * Build and return a "hot" vtkColorTransferFunction scaled to [minValue, maxValue].
 *
 * The returned CTF can be attached to any vtk.js mapper via:
 *   mapper.setLookupTable(ctf)
 *   mapper.setUseLookupTableScalarRange(true)   // use ctf's own range, not mapper's
 *
 * @param options - Scalar range for the colour mapping.
 */
export function buildHotColorTransferFunction(
  options: HotColorMapOptions,
): ReturnType<typeof vtkColorTransferFunction.newInstance> {
  const { minValue, maxValue } = options;

  // Guard against degenerate range (all amplitudes identical).
  // Expanding by ε prevents the CTF from having zero-width bins,
  // which would cause vtk.js to produce undefined colour lookups.
  const span    = maxValue - minValue || 1.0;
  const lo      = minValue;
  const hi      = minValue + span;

  // ── Build the piecewise-linear colour function ────────────────────────────
  const ctf = vtkColorTransferFunction.newInstance();

  // addRGBPoint(scalar, R, G, B) — all channels in linear [0, 1] space.
  // Scalar values are in the original amplitude units (not normalised to 0–1).
  // Linear interpolation between control points is done by vtk.js internally.

  // 0.000 — very dark red: visible against black backgrounds, signals low activity
  ctf.addRGBPoint(lo + 0.000 * span,  0.30,  0.00,  0.00);

  // 0.250 — vivid red: clearly distinguishable from dark red
  ctf.addRGBPoint(lo + 0.250 * span,  0.80,  0.00,  0.00);

  // 0.500 — red-orange: midpoint; the green channel starts rising here
  ctf.addRGBPoint(lo + 0.500 * span,  1.00,  0.35,  0.00);

  // 0.700 — amber: approaching peak (dominant red + growing green)
  ctf.addRGBPoint(lo + 0.700 * span,  1.00,  0.80,  0.00);

  // 0.850 — yellow: high-amplitude sources stand out clearly
  ctf.addRGBPoint(lo + 0.850 * span,  1.00,  1.00,  0.20);

  // 1.000 — warm near-white: maximum activation; slightly warm tint (not pure
  //         white) to remain distinguishable from the white background in
  //         screenshot contexts.
  ctf.addRGBPoint(hi,                  1.00,  1.00,  0.85);

  // Clamp extrapolation: scalars outside [lo, hi] are pinned to the nearest
  // colour rather than producing undefined / wrapping behaviour.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (ctf as any).setClampingOn();

  return ctf;
}
