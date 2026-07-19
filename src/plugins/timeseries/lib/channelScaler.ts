/**
 * channelScaler.ts — Signal downsampling and amplitude scaling utilities
 * ────────────────────────────────────────────────────────────────────────
 *
 * Two decimation strategies are provided:
 *
 *   decimateLTTB  — Largest Triangle Three Buckets (Steinarsson 2013).
 *                   Perceptually optimal for single-line charts; preserves
 *                   visually significant extrema. O(n).
 *
 *   decimateMinMax — Min/max envelope approach.
 *                    For each output bucket, stores both the minimum and
 *                    maximum sample found in the input range.  Produces
 *                    2 × maxPoints output values and is preferred when the
 *                    rendered width is known (pixel-perfect signal envelope).
 *
 * Both functions operate on arbitrary Float32Arrays and return new arrays —
 * the input data is never mutated.
 *
 * computeChannelRange  — Fast single-pass min/max scan used to initialise
 *                        the per-channel amplitude scale slider.
 *
 * normaliseAmplitude   — Scales a channel so its peak-to-peak amplitude
 *                        maps to a fixed pixel height; used by the canvas
 *                        renderer to place channels in their lanes.
 */

// ── LTTB (Largest Triangle Three Buckets) ────────────────────────────────────

/**
 * Downsample `data` to at most `maxPoints` points using the LTTB algorithm.
 *
 * LTTB works in three stages for each bucket:
 *   1. Fix the left point (previously selected point).
 *   2. Compute the average of the next bucket (the "right" reference point).
 *   3. Select the point in the current bucket that maximises the triangle
 *      area formed by left → candidate → right average.
 *
 * This strategy preserves peaks and troughs that are visually prominent
 * even after aggressive downsampling (e.g. 100k → 1k samples).
 *
 * @param data      Source signal values (one sample per element).
 * @param time      Corresponding time values; must be same length as data.
 * @param maxPoints Target output length (≥ 3 to produce meaningful output).
 * @returns Object with decimated { values, times } Float32Arrays.
 */
export function decimateLTTB(
  data: Float32Array,
  time: Float32Array,
  maxPoints: number,
): { values: Float32Array; times: Float32Array } {
  const n = data.length;

  // No decimation needed
  if (maxPoints <= 0 || n <= maxPoints) {
    return { values: new Float32Array(data), times: new Float32Array(time) };
  }

  const outVals  = new Float32Array(maxPoints);
  const outTimes = new Float32Array(maxPoints);

  // Always include first and last sample
  outVals[0]  = data[0]!;
  outTimes[0] = time[0]!;
  outVals[maxPoints - 1]  = data[n - 1]!;
  outTimes[maxPoints - 1] = time[n - 1]!;

  // Number of internal (non-boundary) buckets
  const bucketCount  = maxPoints - 2;
  const bucketSize   = (n - 2) / bucketCount;

  let prevIdx = 0; // index of the previously selected point

  for (let b = 0; b < bucketCount; b++) {
    // Boundaries of current bucket in input space (exclusive of first/last samples)
    const bucketStart = Math.floor((b    ) * bucketSize) + 1;
    const bucketEnd   = Math.floor((b + 1) * bucketSize) + 1;

    // Boundaries of next bucket (used as the right anchor)
    const nextStart = bucketEnd;
    const nextEnd   = Math.min(Math.floor((b + 2) * bucketSize) + 1, n - 1);

    // Compute average of next bucket (right reference)
    let avgT = 0;
    let avgV = 0;
    const nextLen = nextEnd - nextStart;
    if (nextLen > 0) {
      for (let k = nextStart; k < nextEnd; k++) {
        avgT += time[k]!;
        avgV += data[k]!;
      }
      avgT /= nextLen;
      avgV /= nextLen;
    } else {
      avgT = time[n - 1]!;
      avgV = data[n - 1]!;
    }

    const prevT = time[prevIdx]!;
    const prevV = data[prevIdx]!;

    // Select the point in the current bucket with the largest triangle area
    let maxArea = -1;
    let maxIdx  = bucketStart;

    for (let k = bucketStart; k < bucketEnd; k++) {
      // Twice the triangle area (no ÷2 needed — we only need relative comparison)
      const area = Math.abs(
        (prevT - avgT) * (data[k]! - prevV) -
        (prevT - time[k]!) * (avgV - prevV),
      );
      if (area > maxArea) {
        maxArea = area;
        maxIdx  = k;
      }
    }

    outVals[b + 1]  = data[maxIdx]!;
    outTimes[b + 1] = time[maxIdx]!;
    prevIdx = maxIdx;
  }

  return { values: outVals, times: outTimes };
}

// ── Min/Max envelope decimation ───────────────────────────────────────────────

/**
 * Downsample `data` into a min/max envelope with at most `maxPoints` buckets.
 *
 * For each bucket the function stores both the minimum and maximum value found
 * in the input range.  The returned array has up to 2 × maxPoints elements,
 * interleaved as [min₀, max₀, min₁, max₁, …].  This produces a visual
 * "ribbon" that exactly covers all sample values inside each pixel column.
 *
 * The companion `timesForMinMax()` helper generates a matching time axis.
 *
 * Preferred over LTTB when:
 *   • Rendering a fixed-width canvas where each bucket = 1 screen pixel.
 *   • The signal is noisy and LTTB might miss brief transients.
 *
 * @param data      Source signal (Float32Array).
 * @param time      Time axis, same length as data.
 * @param maxPoints Number of output buckets (≈ canvas pixel width).
 * @returns Object with { mins, maxs, times } — each Float32Array of length maxPoints.
 */
export function decimateMinMax(
  data: Float32Array,
  time: Float32Array,
  maxPoints: number,
): { mins: Float32Array; maxs: Float32Array; times: Float32Array } {
  const n = data.length;

  if (maxPoints <= 0 || n === 0) {
    return {
      mins:  new Float32Array(0),
      maxs:  new Float32Array(0),
      times: new Float32Array(0),
    };
  }

  if (n <= maxPoints) {
    // No decimation — treat each sample as its own bucket
    const mins  = new Float32Array(n);
    const maxs  = new Float32Array(n);
    const times = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      mins[i]  = data[i]!;
      maxs[i]  = data[i]!;
      times[i] = time[i]!;
    }
    return { mins, maxs, times };
  }

  const mins  = new Float32Array(maxPoints);
  const maxs  = new Float32Array(maxPoints);
  const times = new Float32Array(maxPoints);
  const step  = n / maxPoints;

  for (let b = 0; b < maxPoints; b++) {
    const start = Math.floor(b       * step);
    const end   = Math.floor((b + 1) * step);

    let minV =  Infinity;
    let maxV = -Infinity;

    for (let k = start; k < end; k++) {
      const v = data[k]!;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }

    mins[b]  = minV === Infinity ? 0 : minV;
    maxs[b]  = maxV === -Infinity ? 0 : maxV;
    times[b] = time[Math.floor((start + end) / 2)]!;
  }

  return { mins, maxs, times };
}

// ── Channel range ─────────────────────────────────────────────────────────────

/**
 * Returns the [min, max] physical value range of a channel.
 * Used to initialise the amplitude scale slider and auto-scale the display.
 */
export function computeChannelRange(data: Float32Array): [min: number, max: number] {
  let min =  Infinity;
  let max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i]!;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return [min === Infinity ? 0 : min, max === -Infinity ? 0 : max];
}

/**
 * Computes a reasonable default amplitude scale for a set of channels.
 * Returns the median peak-to-peak range across all channels, divided by
 * the target lane height in pixels.  This gives µV/px (or equivalent).
 */
export function computeAutoScale(
  channels: Float32Array[],
  laneHeightPx: number,
): number {
  if (channels.length === 0 || laneHeightPx <= 0) return 1;

  const ranges = channels.map((ch) => {
    const [min, max] = computeChannelRange(ch);
    return max - min;
  }).filter((r) => r > 0);

  if (ranges.length === 0) return 1;

  ranges.sort((a, b) => a - b);
  const median = ranges[Math.floor(ranges.length / 2)]!;

  // Use 80 % of the lane height to leave a small margin between channels
  return median / (laneHeightPx * 0.8);
}

// ── Windowed slice ────────────────────────────────────────────────────────────

/**
 * Returns the start and end sample indices for the time window [tStart, tEnd]
 * given a uniform time axis.  Uses binary search for O(log n) lookup.
 *
 * @returns [startIdx, endIdx] — both clamped to [0, data.length).
 */
export function timeWindowToSampleRange(
  time: Float32Array,
  tStart: number,
  tEnd: number,
): [startIdx: number, endIdx: number] {
  const n = time.length;
  if (n === 0) return [0, 0];

  // Binary search for tStart
  let lo = 0;
  let hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (time[mid]! < tStart) lo = mid + 1;
    else hi = mid;
  }
  const startIdx = lo;

  // Binary search for tEnd
  lo = startIdx;
  hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (time[mid]! > tEnd) hi = mid - 1;
    else lo = mid;
  }
  const endIdx = Math.min(lo + 1, n);

  return [startIdx, endIdx];
}
