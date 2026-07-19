/**
 * heatmapColorScale.ts — Pre-computed color LUT for the N×N connectivity heatmap
 * ──────────────────────────────────────────────────────────────────────────────────
 *
 * COLOR SCALE DESIGN
 * ───────────────────
 * The scale is dark-mode optimised and spans void → deep blue → accent-green:
 *
 *   t = 0.000  →  rgb(10,  10,  10)   near-void (zero connections ≡ background)
 *   t = 0.001  →  rgb(10,  28,  60)   first visible dark navy
 *   t = 0.050  →  rgb(15,  45, 120)   deep blue
 *   t = 0.200  →  rgb(20,  80, 180)   medium blue
 *   t = 0.500  →  rgb( 0, 160, 130)   teal transition
 *   t = 0.800  →  rgb( 0, 210,  80)   bright lime
 *   t = 1.000  →  rgb( 0, 230, 118)   accent-green (#00e676)
 *
 * Special values (not on the scale above):
 *   zero cells  →  rgb(30,  30,  30)  = #1e1e1e (var(--bg-section))
 *   diagonal    →  rgb(38,  38,  38)  = #262626  (no self-connections)
 *
 * IMPLEMENTATION
 * ───────────────
 * A 512-entry lookup table (LUT) is pre-computed once at module load.
 * Each entry is [R, G, B] (Uint8 values).  `fiberCountToRgb(count, max)`
 * maps `count` to a LUT index in O(1) via `Math.round((count / max) * 511)`.
 *
 * 512 entries gives sub-0.2% quantisation error across the full range — more
 * than enough for a screen display (human eyes resolve ~100 shades per channel).
 */

// ── Piecewise color stops ─────────────────────────────────────────────────────
// Each stop is [t, r, g, b] where t ∈ [0, 1].
// Must be sorted ascending by t.

type ColorStop = [number, number, number, number]; // [t, r, g, b]

const STOPS: ColorStop[] = [
  [0.000,  10,  10,  10],
  [0.001,  10,  28,  60],
  [0.050,  15,  45, 120],
  [0.200,  20,  80, 180],
  [0.500,   0, 160, 130],
  [0.800,   0, 210,  80],
  [1.000,   0, 230, 118],
];

// ── Special-case constants ────────────────────────────────────────────────────

/** Background fill for cells with exactly zero fibers. */
export const ZERO_RGB:     [number, number, number] = [30, 30, 30];   // #1e1e1e
/** Background fill for diagonal cells (self-connections don't exist). */
export const DIAGONAL_RGB: [number, number, number] = [38, 38, 38];   // #262626

// ── LUT build ─────────────────────────────────────────────────────────────────

const LUT_SIZE = 512;

/**
 * Pre-computed color table mapping [0, LUT_SIZE-1] → [R, G, B].
 * Index 0 corresponds to t=0 (minimum non-zero fiber count).
 * Index LUT_SIZE-1 corresponds to t=1 (maximum fiber count).
 *
 * NOTE: The LUT does NOT include the zero or diagonal special cases —
 * those are handled separately in fiberCountToRgb().
 */
const COLOR_LUT: Uint8Array = buildLut();

function buildLut(): Uint8Array {
  const lut = new Uint8Array(LUT_SIZE * 3);

  for (let idx = 0; idx < LUT_SIZE; idx++) {
    const t = idx / (LUT_SIZE - 1);

    // Find the two surrounding color stops.
    let lo = STOPS[0]!;
    let hi = STOPS[STOPS.length - 1]!;
    for (let s = 0; s < STOPS.length - 1; s++) {
      if (t >= STOPS[s]![0] && t <= STOPS[s + 1]![0]) {
        lo = STOPS[s]!;
        hi = STOPS[s + 1]!;
        break;
      }
    }

    // Interpolate linearly within the stop interval.
    const span = hi[0] - lo[0];
    const f    = span > 0 ? (t - lo[0]) / span : 0;

    lut[idx * 3]     = Math.round(lo[1] + f * (hi[1] - lo[1]));
    lut[idx * 3 + 1] = Math.round(lo[2] + f * (hi[2] - lo[2]));
    lut[idx * 3 + 2] = Math.round(lo[3] + f * (hi[3] - lo[3]));
  }

  return lut;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Map a fibre count to an [R, G, B] triple using the pre-computed LUT.
 *
 * @param count   The fiber count for this cell (0 = no connection).
 * @param maxVal  The maximum fiber count in the entire matrix (used to normalise).
 * @param isDiag  True when i === j (diagonal cell — no self-connections in DTI).
 * @returns       [R, G, B] as integer values in [0, 255].
 */
export function fiberCountToRgb(
  count:  number,
  maxVal: number,
  isDiag: boolean,
): [number, number, number] {
  if (isDiag)    return DIAGONAL_RGB;
  if (count <= 0) return ZERO_RGB;
  if (maxVal <= 0) return ZERO_RGB;

  // Map count to [0, LUT_SIZE-1].  clamp to guard against floating-point overshoot.
  const t   = Math.min(count / maxVal, 1);
  const idx = Math.round(t * (LUT_SIZE - 1));

  return [
    COLOR_LUT[idx * 3]!,
    COLOR_LUT[idx * 3 + 1]!,
    COLOR_LUT[idx * 3 + 2]!,
  ];
}

/**
 * Build the full 512-entry color LUT as a Float64Array of [r, g, b] triples
 * for drawing the legend bar.  Each entry is one row of the bar (from top = max
 * to bottom = min).
 *
 * Used exclusively by drawColorLegend() in heatmapRenderer.ts.
 */
export function buildLegendGradient(): Uint8Array {
  // Return a copy — the caller shouldn't mutate the module-level buffer.
  return new Uint8Array(COLOR_LUT);
}
