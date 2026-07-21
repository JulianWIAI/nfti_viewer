/**
 * bandPower.ts — Client-side MEG/EEG frequency band power estimation
 * ─────────────────────────────────────────────────────────────────────
 *
 * HOW BAND POWER COMPUTATION WORKS
 * ─────────────────────────────────
 * Any MEG signal is a mixture of all frequency components simultaneously.
 * To know how much power (energy) resides in a specific band (e.g. beta),
 * we apply a Fourier transform which decomposes the signal into its
 * constituent frequencies.  The squared magnitude at each frequency bin
 * is the "power" at that frequency (Parseval's theorem).  Summing the
 * squares within a band's frequency range gives the band's total power.
 * Dividing by the total across all bands gives the relative contribution.
 *
 * WHY CLIENT-SIDE IS SUFFICIENT
 * ──────────────────────────────
 * The backend already decimates the raw signal to fit the canvas width
 * (~400–800 points per window).  The resulting effective Nyquist is:
 *   f_Nyq = n_points / (2 × window_duration_s)
 * For a 5 s window at 600 points → f_Nyq ≈ 60 Hz — covers delta through
 * low-gamma.  No extra server round-trip is needed.
 *
 * METHOD USED HERE
 * ─────────────────
 * 1. Cap the signal at 256 samples (sufficient resolution for 5 coarse
 *    bands; keeps the DFT cost below 0.5 ms for up to 20 channels).
 * 2. Apply a Hann window to suppress spectral leakage at the segment edges.
 * 3. Compute the one-sided power spectrum via a plain DFT (O(N²)).
 *    For N=256 this is ~65 k multiply-adds — negligible per render frame.
 * 4. Integrate (sum) squared magnitudes in each band's frequency range.
 * 5. Normalise so all five band powers sum to 1 (relative fractions).
 *
 * KNOWN LIMITATIONS
 * ──────────────────
 * • Very short windows (< 0.5 s) have coarse frequency resolution; delta
 *   and theta bins may bleed into each other.  This is an inherent
 *   time-frequency trade-off, not a bug.
 * • When f_Nyq < 30 Hz (very narrow time window), the gamma band will show
 *   near-zero power because no gamma-range bins exist.  The bar is still
 *   rendered for visual consistency.
 */

// ── Band definitions ──────────────────────────────────────────────────────────

/** Relative power in each of the five standard MEG/EEG frequency bands. */
export interface BandPowers {
  delta: number;   // 1–4 Hz    slow oscillations; high in deep sleep / pathological waking
  theta: number;   // 4–8 Hz    memory encoding, REM sleep, cognitive load
  alpha: number;   // 8–13 Hz   relaxed wakefulness, visual idling / inhibition
  beta:  number;   // 13–30 Hz  active cognition, sensorimotor processing
  gamma: number;   // 30–100 Hz high-frequency binding, sensory processing
}

/**
 * Display colour for each band.
 * Follows the de-facto convention used in FieldTrip / EEGLAB visualisations:
 *   slow (delta) = violet → fast (gamma) = red.
 */
export const BAND_COLORS: Readonly<Record<keyof BandPowers, string>> = {
  delta: '#7c4dff',   // deep violet
  theta: '#2196f3',   // blue
  alpha: '#4caf50',   // green
  beta:  '#ff9800',   // orange
  gamma: '#f44336',   // red
};

/** Short human-readable labels shown in tooltips and canvas legends. */
export const BAND_LABELS: Readonly<Record<keyof BandPowers, string>> = {
  delta: 'δ Delta   1–4 Hz',
  theta: 'θ Theta   4–8 Hz',
  alpha: 'α Alpha   8–13 Hz',
  beta:  'β Beta    13–30 Hz',
  gamma: 'γ Gamma   30–100 Hz',
};

/**
 * Ordered array of band keys, low-to-high frequency.
 * Use this to iterate bands in a consistent, predictable order.
 */
export const BAND_ORDER: ReadonlyArray<keyof BandPowers> = [
  'delta', 'theta', 'alpha', 'beta', 'gamma',
];

// ── Internal helpers ──────────────────────────────────────────────────────────

/**
 * Returns the largest power-of-2 that is ≤ n, clamped to a minimum of 8.
 * Used to pick a DFT size that exactly fits a power-of-two block of the signal.
 */
function prevPow2(n: number): number {
  let p = 8;
  while (p * 2 <= n) p *= 2;
  return p;
}

/**
 * Multiplies each sample by a Hann window coefficient.
 *
 * Without windowing, the abrupt start/end of the time segment is treated by
 * the DFT as a discontinuity, causing energy to "leak" from every frequency
 * into all others.  The Hann window tapers the signal smoothly to zero at
 * both ends, dramatically reducing this leakage at the cost of slightly
 * widening each spectral peak.
 *
 *   w[n] = 0.5 · (1 − cos(2π n / (N − 1)))
 */
function applyHann(signal: number[]): number[] {
  const N = signal.length;
  if (N <= 1) return signal;
  return signal.map(
    (v, n) => v * 0.5 * (1 - Math.cos((2 * Math.PI * n) / (N - 1))),
  );
}

/**
 * Computes the one-sided power spectrum |X[k]|² via a direct DFT.
 *
 * Returns an array of length ⌊N/2⌋ where index k corresponds to frequency
 *   f_k = k · (sampleRate / N)   Hz
 *
 * The direct (non-FFT) DFT is O(N²).  For N ≤ 256 this takes < 1 ms and
 * avoids the complexity of importing an FFT library.
 */
function dftPower(signal: number[]): number[] {
  const N   = signal.length;
  const out = new Array<number>(Math.floor(N / 2)).fill(0);

  for (let k = 0; k < out.length; k++) {
    let re = 0, im = 0;
    const omega = (2 * Math.PI * k) / N;

    for (let n = 0; n < N; n++) {
      // X[k] = Σ x[n] · e^{−j·ω·n}  (forward DFT convention)
      re += signal[n]! * Math.cos(omega * n);
      im -= signal[n]! * Math.sin(omega * n);
    }
    out[k] = re * re + im * im;   // power = |X[k]|²
  }

  return out;
}

/**
 * Sums power spectrum bins that fall within [fLo, fHi) Hz.
 * @param spectrum — output of dftPower()
 * @param df       — frequency resolution in Hz per bin  (= sampleRate / N)
 * @param fLo      — lower bound of the band in Hz (inclusive)
 * @param fHi      — upper bound of the band in Hz (exclusive)
 */
function sumBandPower(
  spectrum: number[],
  df:       number,
  fLo:      number,
  fHi:      number,
): number {
  if (df <= 0) return 0;
  const lo = Math.max(0, Math.floor(fLo / df));
  const hi = Math.min(spectrum.length - 1, Math.ceil(fHi / df));
  let total = 0;
  for (let k = lo; k <= hi; k++) total += spectrum[k]!;
  return total;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Computes normalised relative band powers from a decimated MEG/EEG segment.
 *
 * @param signal      Time-domain signal values (the 'values' array from the
 *                    backend ChannelTrace, i.e. the per-point mean amplitude).
 * @param sampleRate  Effective sample rate of the decimated signal in Hz.
 *                    Derive as:  signal.length / windowDurationInSeconds
 *
 * @returns BandPowers where all five values sum to ≈ 1 (relative fractions).
 *          Returns equal weights (0.2 each) on degenerate / unanalysable input.
 */
export function computeBandPowers(signal: number[], sampleRate: number): BandPowers {
  // Degenerate / unusable inputs → flat prior (no band dominates)
  const fallback: BandPowers = {
    delta: 0.2, theta: 0.2, alpha: 0.2, beta: 0.2, gamma: 0.2,
  };
  if (signal.length < 8 || sampleRate < 2) return fallback;

  // Cap at 256 points: sufficient for 5-band resolution, fast DFT
  const N        = prevPow2(Math.min(signal.length, 256));
  const windowed = applyHann(signal.slice(0, N));
  const spectrum = dftPower(windowed);

  const df      = sampleRate / N;          // Hz per frequency bin
  const nyquist = sampleRate / 2;          // effective Nyquist frequency of the decimated signal

  const raw: BandPowers = {
    delta: sumBandPower(spectrum, df, 1,  4),
    theta: sumBandPower(spectrum, df, 4,  8),
    alpha: sumBandPower(spectrum, df, 8,  13),
    beta:  sumBandPower(spectrum, df, 13, 30),
    // Gamma capped at Nyquist to avoid summing empty bins
    gamma: sumBandPower(spectrum, df, 30, Math.min(nyquist, 100)),
  };

  // Normalise: convert absolute power to relative fractions (sum → 1)
  const total = (raw.delta + raw.theta + raw.alpha + raw.beta + raw.gamma) || 1;

  return {
    delta: raw.delta / total,
    theta: raw.theta / total,
    alpha: raw.alpha / total,
    beta:  raw.beta  / total,
    gamma: raw.gamma / total,
  };
}
