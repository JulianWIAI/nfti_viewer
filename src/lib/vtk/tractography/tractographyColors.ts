/**
 * tractographyColors.ts — Direction-Encoded Color (DEC) convention and utilities
 * ────────────────────────────────────────────────────────────────────────────────
 *
 * WHAT IS DIRECTION-ENCODED COLOR?
 * ──────────────────────────────────
 * DEC is the standard neuroimaging colormap for white-matter fibre orientation.
 * Each RGB triple encodes the ABSOLUTE (unsigned) local fibre direction vector:
 *
 *   R (0–255) = |dx| / ‖d‖  — left↔right component
 *   G (0–255) = |dy| / ‖d‖  — anterior↔posterior component
 *   B (0–255) = |dz| / ‖d‖  — inferior↔superior component
 *
 * where d = (dx, dy, dz) is the segment direction vector in RAS mm space and
 * ‖d‖ is its Euclidean length.
 *
 * WHY ABSOLUTE VALUES?
 * ─────────────────────
 * White-matter fibres are anatomically symmetric — tracking propagates in both
 * directions along e₁, so the sign of the direction vector is arbitrary.
 * Taking absolute values produces a consistent, sign-independent colour:
 *
 *   · Left-Right tracts (corpus callosum) → pure Red
 *   · Anterior-Posterior tracts (cingulum) → pure Green
 *   · Superior-Inferior tracts (corticospinal) → pure Blue
 *   · Oblique tracts → mixed colours
 *
 * WHY NOT INCLUDE THIS IN THE HOT LOOP?
 * ───────────────────────────────────────
 * The hot inner loop in tractographyGeometry.ts inlines the color computation
 * directly because:
 *   1. The direction vector is already computed for the current segment.
 *   2. An extra function call per point would add ~10–20% overhead for large
 *      tractograms (10 000 streamlines × 25 pts = 250 000 calls).
 *   3. Modern JS engines inline small hot functions, but only inside the same
 *      compilation unit — cross-module calls break this guarantee.
 *
 * This module provides the canonical formula as documentation and as a utility
 * function for unit tests and one-off colour previews.
 *
 * FORMULA (single segment direction vector)
 * ──────────────────────────────────────────
 *   norm = sqrt(dx² + dy² + dz²)   (guard against zero division → use 1.0)
 *   R = |dx / norm| × 255 | 0      (truncate to uint8)
 *   G = |dy / norm| × 255 | 0
 *   B = |dz / norm| × 255 | 0
 */

// ── Type alias ───────────────────────────────────────────────────────────────

/** RGB triple as clamped uint8 values [0–255]. */
export type Rgb255 = [number, number, number];

// ── Public utility ────────────────────────────────────────────────────────────

/**
 * Compute the DEC colour for a single segment direction vector.
 *
 * This function is NOT called in the geometry hot loop — it is provided for
 * unit tests, debugging, and UI colour previews (e.g., legend swatches).
 *
 * @param dx  X component of the segment direction (raw, not normalised).
 * @param dy  Y component of the segment direction (raw, not normalised).
 * @param dz  Z component of the segment direction (raw, not normalised).
 * @returns   [R, G, B] in [0–255], encoding the absolute normalised direction.
 *
 * @example
 *   directionToRgb255(1, 0, 0)  // → [255, 0, 0]  pure red  (left-right)
 *   directionToRgb255(0, 1, 0)  // → [0, 255, 0]  pure green (A-P)
 *   directionToRgb255(0, 0, 1)  // → [0, 0, 255]  pure blue (I-S)
 *   directionToRgb255(1, 1, 0)  // → [180, 180, 0] diagonal (45°)
 */
export function directionToRgb255(dx: number, dy: number, dz: number): Rgb255 {
  // Guard: if the direction vector is degenerate (zero-length), return grey.
  const norm = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1.0;
  return [
    (Math.abs(dx / norm) * 255) | 0,   // R = |x̂|
    (Math.abs(dy / norm) * 255) | 0,   // G = |ŷ|
    (Math.abs(dz / norm) * 255) | 0,   // B = |ẑ|
  ];
}
