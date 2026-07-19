/**
 * megAnalysisApi.ts — HTTP client for MEG signal analysis endpoints
 * ──────────────────────────────────────────────────────────────────
 * Typed wrappers around the three analysis endpoints added by meg_analysis.py.
 * All methods accept a ``session_id`` string returned by ``/api/load-meg`` and
 * an optional ``AbortSignal`` for cancellation (e.g. on component unmount).
 */

import type {
  ArtifactResult,
  SpikeResult,
  FrequencyBandsResult,
} from '../types/analysis.types';

// Re-export so component files have a single import point.
export type { ArtifactResult, SpikeResult, FrequencyBandsResult };

const BASE = 'http://localhost:8000';

// ── Shared helper ─────────────────────────────────────────────────────────────

/** POST to a URL with no request body (all parameters are query-string). */
async function postQuery<T>(
  url: string,
  signal?: AbortSignal,
): Promise<T> {
  const resp = await fetch(url, { method: 'POST', signal });
  if (!resp.ok) {
    const text = await resp.text().catch(() => resp.statusText);
    throw new Error(`${url} failed (${resp.status}): ${text}`);
  }
  return resp.json() as Promise<T>;
}

// ── API object ────────────────────────────────────────────────────────────────

export const megAnalysisApi = {
  /**
   * Detect EOG eye-blinks and muscle/motion artefacts in a loaded MEG session.
   *
   * Returns an array of ``ArtifactAnnotation`` objects (onset, duration, type)
   * sorted chronologically.  The frontend renders these as coloured background
   * spans: yellow for blinks, grey for muscle artefacts.
   *
   * Internally uses:
   *   • ``mne.preprocessing.find_eog_events``  — blink detection
   *   • ``mne.preprocessing.annotate_muscle_zscore``  — muscle detection
   */
  async detectArtifacts(
    sessionId: string,
    signal?:   AbortSignal,
  ): Promise<ArtifactResult> {
    return postQuery<ArtifactResult>(
      `${BASE}/api/meg/detect-artifacts?session_id=${encodeURIComponent(sessionId)}`,
      signal,
    );
  },

  /**
   * Detect epileptiform-like transient spikes using a 20 Hz HPF + MAD threshold.
   *
   * Returns an array of ``SpikeMarker`` objects with peak time, the channels
   * involved, and the absolute peak amplitude.  The frontend renders these as
   * vertical red dashed lines overlaid on the waveform canvas.
   *
   * @param sessionId    MEG session ID from /api/load-meg.
   * @param madMultiplier  Number of MAD-derived σ above which a crossing is a spike.
   *                       Default 5; lower = more sensitive, higher = more specific.
   * @param minGapSec    Minimum gap (seconds) between distinct spike events.
   */
  async detectSpikes(
    sessionId:     string,
    madMultiplier: number  = 5.0,
    minGapSec:     number  = 0.05,
    signal?:       AbortSignal,
  ): Promise<SpikeResult> {
    const params = new URLSearchParams({
      session_id:     sessionId,
      mad_multiplier: String(madMultiplier),
      min_gap_sec:    String(minGapSec),
    });
    return postQuery<SpikeResult>(
      `${BASE}/api/meg/detect-spikes?${params}`,
      signal,
    );
  },

  /**
   * Compute relative Welch PSD power per standard neurological frequency band
   * (δ, θ, α, β, γ) for an already-loaded MEG session.
   *
   * The result is averaged across all good channels of the preferred type
   * (magnetometers → gradiometers → EEG) and normalised so all bands sum to 1.
   *
   * @param sessionId  MEG session ID.
   * @param tStart     Analysis window start in seconds (null = start of recording).
   * @param tEnd       Analysis window end in seconds (null = end of recording).
   */
  async getFrequencyBands(
    sessionId: string,
    tStart?:   number,
    tEnd?:     number,
    signal?:   AbortSignal,
  ): Promise<FrequencyBandsResult> {
    const params = new URLSearchParams({ session_id: sessionId });
    if (tStart != null) params.set('t_start', String(tStart));
    if (tEnd   != null) params.set('t_end',   String(tEnd));
    return postQuery<FrequencyBandsResult>(
      `${BASE}/api/meg/frequency-bands?${params}`,
      signal,
    );
  },
};
