/**
 * laneSpectrogram.ts — Sliding-window dominant frequency band analysis
 * ──────────────────────────────────────────────────────────────────────
 *
 * PURPOSE
 * ────────
 * Decomposes a single MEG channel's time-domain signal into a sequence of
 * contiguous time segments, each labelled with the frequency band (delta,
 * theta, alpha, beta, gamma) that carries the most power at that moment.
 * The output is used to paint coloured "spectral tint" regions on each
 * lane background in the MEG viewer canvas, giving the researcher an
 * at-a-glance picture of where one oscillation ends and the next begins.
 *
 * SCIENTIFIC BASIS
 * ─────────────────
 * This is a simplified Short-Time Fourier Transform (STFT) analysis:
 *   1. A short window (typically 64 samples ≈ 0.5 s at decimated SR) is
 *      slid across the signal with 75 % overlap (hop = WIN/4).
 *   2. For each window position, `computeBandPowers` (from bandPower.ts)
 *      runs a Hann-windowed DFT and returns relative power in 5 bands.
 *   3. The band with the highest power at that position is the "dominant"
 *      band for that time slice.
 *   4. Adjacent slices with the same dominant band are merged into a single
 *      contiguous segment to reduce rendering overhead.
 *
 * LIMITATIONS
 * ─────────────
 * The Heisenberg uncertainty principle of time-frequency analysis applies:
 * the shorter the window, the better the time resolution but the worse
 * the frequency resolution.  For the default window (64 samples at SR ≈
 * 120 Hz), the frequency resolution is ≈ 1.9 Hz per bin — sufficient to
 * separate delta from theta but with some blurring near band boundaries.
 *
 * Rapidly changing oscillations (e.g. a single gamma burst lasting < 50 ms)
 * may not be captured if the decimated signal's Nyquist is too low.
 * This is an inherent trade-off of working with pre-decimated data.
 */

import {
  computeBandPowers,
  BAND_ORDER,
  type BandPowers,
} from './bandPower';

// ── Types ─────────────────────────────────────────────────────────────────────

/**
 * A contiguous time segment over which one frequency band is dominant.
 * Both fractions are relative to the full signal array [0, 1]:
 *   0 corresponds to timeStart, 1 corresponds to timeEnd.
 */
export interface BandSegment {
  /** Fraction of the total signal range at which this segment begins. */
  startFraction: number;
  /** Fraction at which this segment ends. */
  endFraction:   number;
  /** The dominant frequency band across this segment. */
  band:          keyof BandPowers;
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/**
 * Largest power-of-2 ≤ n, clamped to [8, ∞).
 * Used here independently of bandPower.ts to keep this module self-contained.
 */
function prevPow2(n: number): number {
  let p = 8;
  while (p * 2 <= n) p *= 2;
  return p;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Computes per-time-position dominant frequency band for a single channel trace.
 *
 * @param signal      Decimated signal values (the `values` array from ChannelTrace).
 * @param effectiveSR Effective sample rate of the decimated signal in Hz.
 *                    Derive as:  signal.length / windowDurationInSeconds
 *
 * @returns Array of BandSegments ready to be painted as lane-background tints.
 *          Empty array when the signal is too short to analyse reliably.
 */
export function computeLaneSpectrogram(
  signal:      number[],
  effectiveSR: number,
): BandSegment[] {
  // Minimum 16 samples and 2 Hz effective SR for a meaningful analysis
  if (signal.length < 16 || effectiveSR < 2) return [];

  const N = signal.length;

  // Window size: ~1/8 of the full signal, power-of-2, clamped to [16, 128].
  // This targets roughly 0.4–0.6 s of signal — enough to resolve alpha and
  // beta while still giving ~(N / (WIN/4)) ≈ 30+ time positions per render.
  const WIN = prevPow2(Math.min(128, Math.max(16, Math.floor(N / 8))));

  // Hop of WIN/4 → 75 % overlap → smooth temporal transitions without
  // requiring individual-sample resolution (which would be too noisy).
  const HOP = Math.max(1, Math.floor(WIN / 4));

  // ── Compute dominant band for each analysis frame ──────────────────────────

  interface Frame {
    startFrac: number;  // fraction of signal where this frame starts
    endFrac:   number;  // fraction where it ends (= startFrac + HOP/N)
    band:      keyof BandPowers;
  }

  const frames: Frame[] = [];

  for (let pos = 0; pos + WIN <= N; pos += HOP) {
    // Extract the windowed slice for this frame position
    const windowSlice = signal.slice(pos, pos + WIN);
    const powers      = computeBandPowers(windowSlice, effectiveSR);

    // Identify the band with the highest relative power
    let dominant: keyof BandPowers = 'alpha';   // reasonable default
    let maxPow   = -Infinity;
    for (const band of BAND_ORDER) {
      if (powers[band] > maxPow) {
        maxPow   = powers[band];
        dominant = band;
      }
    }

    frames.push({
      startFrac: pos / N,
      endFrac:   Math.min(1, (pos + HOP) / N),
      band:      dominant,
    });
  }

  if (frames.length === 0) return [];

  // Extend the last frame to cover the tail of the signal (avoids a gap at
  // the right edge when N is not exactly divisible by HOP)
  frames[frames.length - 1]!.endFrac = 1;

  // ── Merge adjacent frames with the same dominant band ──────────────────────
  // This collapses e.g. [alpha][alpha][alpha][beta] into [alpha][beta],
  // reducing the number of fillRect calls during canvas rendering.
  const segments: BandSegment[] = [];

  for (const frame of frames) {
    const last = segments[segments.length - 1];
    if (last && last.band === frame.band) {
      // Extend the current segment rather than creating a new one
      last.endFraction = frame.endFrac;
    } else {
      segments.push({
        startFraction: frame.startFrac,
        endFraction:   frame.endFrac,
        band:          frame.band,
      });
    }
  }

  return segments;
}
