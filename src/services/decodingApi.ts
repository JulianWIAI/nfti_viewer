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
 *
 * Upload progress is tracked via XMLHttpRequest (xhrPost) so callers can
 * display a byte-level progress bar during the file transfer phase.
 */

import { xhrPost } from '../lib/xhrUpload';

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

// ── Optional progress callbacks ────────────────────────────────────────────────

/**
 * Options for decodingApi.runMvpaDecoding enabling upload-progress tracking.
 *
 * onUploadProgress — called with a percentage 0–100 as bytes are sent
 * onUploadComplete — called once all bytes have been transmitted
 * signal           — optional AbortSignal for cancellation
 */
export interface DecodingOptions {
  onUploadProgress?: (pct: number) => void;
  onUploadComplete?: () => void;
  signal?:           AbortSignal;
}

// ── Raw snake_case response from the backend ──────────────────────────────────

/** Shape of the raw JSON the FastAPI endpoint returns (snake_case). */
interface RawDecodeResponse {
  times:            number[];
  scores:           number[];
  scores_std:       number[];
  chance_level:     number;
  n_epochs:         number;
  n_epochs_class_a: number;
  n_epochs_class_b: number;
  n_channels:       number;
  n_times:          number;
  peak_score:       number;
  peak_time_s:      number;
  duration_ms:      number;
}

// ── API client ────────────────────────────────────────────────────────────────

export const decodingApi = {
  /**
   * Run a time-resolved MVPA analysis on the supplied BrainVision EEG files.
   *
   * Builds a FormData with the three files and the form parameters, POSTs
   * to /api/eeg/decode via xhrPost (for upload progress), and returns a
   * camelCase-typed DecodingResult.
   *
   * Throws an Error (with the backend `detail` string when available) on
   * any non-2xx response.
   *
   * @param req     Files + analysis parameters.
   * @param options Optional progress callbacks and abort signal.
   */
  async runMvpaDecoding(req: DecodingRequest, options?: DecodingOptions): Promise<DecodingResult> {
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

    // Use xhrPost for real byte-level upload progress tracking.
    // The raw response is snake_case JSON; we translate to camelCase below.
    const json = await xhrPost<RawDecodeResponse>({
      url:              `${API_BASE}/api/eeg/decode`,
      form,
      onUploadProgress: options?.onUploadProgress,
      onUploadComplete: options?.onUploadComplete,
      signal:           options?.signal,
    });

    // Translate snake_case JSON to camelCase TypeScript.
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
