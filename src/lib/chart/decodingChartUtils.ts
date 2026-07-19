/**
 * decodingChartUtils.ts — Pure utility functions for the DecodingTimeline SVG chart
 * ──────────────────────────────────────────────────────────────────────────────────
 *
 * All functions are stateless and side-effect free — they take numbers in,
 * return numbers or strings out.  This keeps the heavy lifting (scale math,
 * SVG path construction) out of the React component and testable in isolation.
 *
 * Linear scale design
 * ─────────────────────
 * A LinearScale is a callable function (value → pixel) that also carries an
 * `invert` method (pixel → value) for click-to-seek.  We attach `domain` and
 * `range` directly to the function object so callers can inspect them without
 * storing them separately.
 *
 * SVG path conventions
 * ──────────────────────
 * All path builders return a plain SVG `d` attribute string using absolute M/L
 * commands.  Coordinates are already in pixel space (post-scale), so callers
 * set the path's `d` prop directly with no additional transforms.
 */

// ── Linear scale ─────────────────────────────────────────────────────────────

/** A callable linear mapping with an inverse and inspectable domain/range. */
export interface LinearScale {
  /** Map a domain value to a range (pixel) value. */
  (value: number): number;
  /** Map a range (pixel) value back to the domain. */
  invert(pixel: number): number;
  domain: readonly [number, number];
  range:  readonly [number, number];
}

/**
 * Build a linear scale that maps [d0, d1] → [r0, r1].
 *
 * Guards against a zero-length domain (returns r0 for all inputs rather than
 * dividing by zero).
 */
export function createLinearScale(
  domain: readonly [number, number],
  range:  readonly [number, number],
): LinearScale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const dSpan = d1 - d0;
  const rSpan = r1 - r0;

  const forward = (v: number): number =>
    dSpan === 0 ? r0 : r0 + ((v - d0) / dSpan) * rSpan;

  const invert = (px: number): number =>
    rSpan === 0 ? d0 : d0 + ((px - r0) / rSpan) * dSpan;

  (forward as LinearScale).invert = invert;
  (forward as LinearScale).domain = domain;
  (forward as LinearScale).range  = range;

  return forward as LinearScale;
}

// ── SVG path builders ─────────────────────────────────────────────────────────

/**
 * Build an open SVG polyline path for `times` vs `scores`.
 * Returns an empty string when the arrays are empty.
 */
export function buildSvgLinePath(
  times:   readonly number[],
  scores:  readonly number[],
  xScale:  LinearScale,
  yScale:  LinearScale,
): string {
  if (times.length === 0) return '';
  const parts: string[] = [];
  for (let i = 0; i < times.length; i++) {
    const x = xScale(times[i]).toFixed(2);
    const y = yScale(scores[i]).toFixed(2);
    parts.push(i === 0 ? `M${x},${y}` : `L${x},${y}`);
  }
  return parts.join(' ');
}

/**
 * Build a closed SVG area path (suitable for `fill`) beneath the score line.
 *
 * The path traces the score line forward, then drops vertically to `baselineY`
 * (the pixel Y at the bottom of the plot area) and closes back to the start.
 *
 * @param baselineY  Pixel Y of the area's lower baseline (typically plotH).
 */
export function buildSvgAreaPath(
  times:     readonly number[],
  scores:    readonly number[],
  xScale:    LinearScale,
  yScale:    LinearScale,
  baselineY: number,
): string {
  if (times.length === 0) return '';
  const line    = buildSvgLinePath(times, scores, xScale, yScale);
  const lastX   = xScale(times[times.length - 1]).toFixed(2);
  const firstX  = xScale(times[0]).toFixed(2);
  const baseY   = baselineY.toFixed(2);
  return `${line} L${lastX},${baseY} L${firstX},${baseY} Z`;
}

/**
 * Build a closed SVG band path (suitable for `fill`) spanning [lower, upper].
 *
 * Traces `upper` forward, then `lower` backward to create a closed band
 * matching the standard shaded-error-interval visual.
 */
export function buildSvgStdBandPath(
  times:   readonly number[],
  upper:   readonly number[],
  lower:   readonly number[],
  xScale:  LinearScale,
  yScale:  LinearScale,
): string {
  if (times.length === 0) return '';
  const parts: string[] = [];

  // Forward sweep along the upper bound.
  for (let i = 0; i < times.length; i++) {
    const x = xScale(times[i]).toFixed(2);
    const y = yScale(upper[i]).toFixed(2);
    parts.push(i === 0 ? `M${x},${y}` : `L${x},${y}`);
  }

  // Backward sweep along the lower bound to close the shape.
  for (let i = times.length - 1; i >= 0; i--) {
    const x = xScale(times[i]).toFixed(2);
    const y = yScale(lower[i]).toFixed(2);
    parts.push(`L${x},${y}`);
  }

  parts.push('Z');
  return parts.join(' ');
}

// ── Time index lookup ─────────────────────────────────────────────────────────

/**
 * Return the index of the element in `times` whose value is closest to
 * `targetTime`.  Used to convert a clicked pixel back to a frame index.
 *
 * O(n) linear scan — acceptable for typical EEG epoch sizes (≤ 2 000 points).
 */
export function findClosestTimeIndex(
  times:      readonly number[],
  targetTime: number,
): number {
  if (times.length === 0) return 0;
  let best     = 0;
  let bestDist = Math.abs(times[0] - targetTime);
  for (let i = 1; i < times.length; i++) {
    const d = Math.abs(times[i] - targetTime);
    if (d < bestDist) { bestDist = d; best = i; }
  }
  return best;
}

// ── Tick generators ───────────────────────────────────────────────────────────

/**
 * Generate Y-axis tick values spaced `step` apart within [min, max].
 *
 * The first tick is rounded UP to the nearest multiple of `step` so that
 * e.g. `generateYTicks(0.3, 1.0, 0.1)` yields [0.3, 0.4, ..., 1.0] not
 * an off-by-epsilon mess.
 */
export function generateYTicks(
  min:  number,
  max:  number,
  step: number,
): number[] {
  const ticks: number[] = [];
  const start = Math.ceil((min - 1e-9) / step) * step;
  for (let v = start; v <= max + 1e-9; v += step) {
    ticks.push(Math.round(v * 10_000) / 10_000);
  }
  return ticks;
}

/**
 * Generate `count + 1` X-axis tick values evenly spaced between `tmin` and
 * `tmax` (inclusive of both endpoints).
 */
export function generateXTicks(
  tmin:  number,
  tmax:  number,
  count: number,
): number[] {
  const ticks: number[] = [];
  for (let i = 0; i <= count; i++) {
    ticks.push(tmin + (i / count) * (tmax - tmin));
  }
  return ticks;
}

// ── Formatters ────────────────────────────────────────────────────────────────

/**
 * Format a time value in seconds as a short millisecond string.
 * Examples: 0 → "0ms", -0.2 → "-200ms", 0.8 → "800ms"
 */
export function formatTimeMs(seconds: number): string {
  return `${Math.round(seconds * 1000)}ms`;
}
