/**
 * decodingApi.ts — Typed HTTP client for POST /api/eeg/decode
 * ──────────────────────────────────────────────────────────────
 *
 * Sends the three BrainVision files and analysis parameters to the FastAPI
 * backend's time-resolved MVPA endpoint.  The backend runs a SlidingEstimator
 * pipeline (StandardScaler → LogisticRegression per time point, k-fold AUC)
 * and returns the decoding time-course as JSON.
 *
 * Request format: multipart/form-data (three file uploads + form fields)
 * Response format: DecodeResponse JSON (see routers/decoding.py)
 *
 * API base URL:  Uses VITE_API_URL env var when set, falls back to
 *                http://localhost:8000 to match the FastAPI dev server.
 */

const API_BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';

// ── Request shape ─────────────────────────────────────────────────────────────

/**
 * Parameters for a single MVPA decoding run.
 *
 * The three File objects are the raw BrainVision file-set:
 *   vhdrFile — plain-text header (.vhdr)
 *   eegFile  — binary waveform data (.eeg or .dat)
 *   vmrkFile — event marker log (.vmrk)
 *
 * All three must come from the same recording (the .vhdr references the other
 * two via relative paths; the backend writes them to the same temp directory
 * before calling MNE).
 *
 * Optional parameters match FastAPI form defaults:
 *   tmin / tmax      — epoch window in seconds (default -0.2 / 0.8)
 *   baselineEnd      — pre-stim baseline end (default 0.0)
 *   applyBaseline    — whether to correct baseline (default true)
 *   nFolds           — k-fold CV splits (default 5)
 *   C                — logistic regression regularisation (default 1.0)
 */
export interface DecodingRequest {
  vhdrFile:       File;
  eegFile:        File;
  vmrkFile:       File;
  classA:         number;
  classB:         number;
  tmin?:          number;
  tmax?:          number;
  baselineEnd?:   number;
  applyBaseline?: boolean;
  nFolds?:        number;
  C?:             number;
}

// ── Response shape ────────────────────────────────────────────────────────────

/**
 * Decoded result returned by POST /api/eeg/decode.
 *
 * All three arrays (times, scores, scoresStd) are co-indexed:
 *   times[i]     — time in seconds relative to stimulus onset
 *   scores[i]    — mean AUC across k folds at that time point
 *   scoresStd[i] — std of AUC across folds (for ±1 SD band)
 *
 * Field names are camelCased here; the service layer translates the
 * snake_case JSON from the backend.
 */
export interface DecodingResult {
  times:          number[];
  scores:         number[];
  scoresStd:      number[];
  chanceLevel:    number;
  nEpochs:        number;
  nEpochsClassA:  number;
  nEpochsClassB:  number;
  nChannels:      number;
  nTimes:         number;
  peakScore:      number;
  peakTimeS:      number;
  durationMs:     number;
}

// ── API client ────────────────────────────────────────────────────────────────

export const decodingApi = {
  /**
   * Run a time-resolved MVPA analysis on the supplied BrainVision EEG files.
   *
   * Builds a FormData with the three files and the form parameters, POSTs
   * to /api/eeg/decode, and returns a camelCase-typed DecodingResult.
   *
   * Throws an Error (with the backend `detail` string when available) on
   * any non-2xx response.
   *
   * @param req — files + analysis parameters
   * @param signal — optional AbortSignal for cancellation
   */
  async runMvpaDecoding(req: DecodingRequest, signal?: AbortSignal): Promise<DecodingResult> {
    const form = new FormData();

    // File uploads — use the original filename so the backend can validate the
    // extension and write it to the temp directory with the right name.
    form.append('vhdr_file', req.vhdrFile, req.vhdrFile.name);
    form.append('eeg_file',  req.eegFile,  req.eegFile.name);
    form.append('vmrk_file', req.vmrkFile, req.vmrkFile.name);

    // Required form fields.
    form.append('class_a', String(req.classA));
    form.append('class_b', String(req.classB));

    // Optional fields — omit when undefined so the backend uses its defaults.
    if (req.tmin          !== undefined) form.append('tmin',           String(req.tmin));
    if (req.tmax          !== undefined) form.append('tmax',           String(req.tmax));
    if (req.baselineEnd   !== undefined) form.append('baseline_end',   String(req.baselineEnd));
    if (req.applyBaseline !== undefined) form.append('apply_baseline', String(req.applyBaseline));
    if (req.nFolds        !== undefined) form.append('n_folds',        String(req.nFolds));
    if (req.C             !== undefined) form.append('C',              String(req.C));

    const resp = await fetch(`${API_BASE}/api/eeg/decode`, {
      method: 'POST',
      body:   form,
      signal,
    });

    if (!resp.ok) {
      // Extract the FastAPI `detail` field when the backend returns JSON.
      let detail = `HTTP ${resp.status} ${resp.statusText}`;
      try {
        const err = await resp.json() as { detail?: unknown };
        if (err.detail) detail = String(err.detail);
      } catch { /* ignore JSON parse failures on non-JSON error bodies */ }
      throw new Error(detail);
    }

    // Translate snake_case JSON to camelCase TypeScript.
    const json = await resp.json() as {
      times: number[];
      scores: number[];
      scores_std: number[];
      chance_level: number;
      n_epochs: number;
      n_epochs_class_a: number;
      n_epochs_class_b: number;
      n_channels: number;
      n_times: number;
      peak_score: number;
      peak_time_s: number;
      duration_ms: number;
    };

    return {
      times:         json.times,
      scores:        json.scores,
      scoresStd:     json.scores_std,
      chanceLevel:   json.chance_level,
      nEpochs:       json.n_epochs,
      nEpochsClassA: json.n_epochs_class_a,
      nEpochsClassB: json.n_epochs_class_b,
      nChannels:     json.n_channels,
      nTimes:        json.n_times,
      peakScore:     json.peak_score,
      peakTimeS:     json.peak_time_s,
      durationMs:    json.duration_ms,
    };
  },
};
