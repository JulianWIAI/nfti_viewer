/**
 * megApi.ts — FastAPI MEG backend client
 * ────────────────────────────────────────
 *
 * Typed wrapper around the Python FastAPI server defined in backend/main.py.
 * All interfaces mirror the Pydantic models in main.py exactly — if you add
 * a field to a Python model, add it here too (and vice versa).
 *
 * Configuration
 * ─────────────
 * The base URL is read from the VITE_MEG_API_URL env variable, which you set
 * in a .env.local file at the project root:
 *
 *   VITE_MEG_API_URL=http://localhost:8000
 *
 * AbortController / cancellation
 * ──────────────────────────────
 * Every data-fetching function accepts an optional AbortSignal so in-flight
 * HTTP requests can be cancelled when the user pans the timeline before the
 * previous request completes.  Use createChunkFetcher() for a ready-made
 * cancel-on-next-call wrapper.
 *
 * Debounce
 * ────────
 * Use debounce() (exported below) to delay chunk requests until the user
 * has stopped dragging the time-window scrubber for `wait` ms.
 *
 * Usage example (inside a React effect):
 *
 *   const fetcher = createChunkFetcher(sessionId);
 *   const debouncedFetch = debounce(fetcher, 80);
 *
 *   // On every scroll/pan event:
 *   debouncedFetch(visibleChannels, timeStart, timeEnd, canvasWidth);
 */

// ── Types matching backend/main.py Pydantic models ───────────────────────────

/** Metadata for a single recording channel. */
export interface ChannelInfo {
  /** MNE channel label, e.g. 'MEG 0111'. */
  name:  string;
  /** MNE type string: 'mag' | 'grad' | 'eeg' | 'eog' | 'ecg' | 'misc' | … */
  type:  string;
  /** Physical unit: 'T' | 'T/m' | 'V' | 'a.u.' | … */
  unit:  string;
  /** Zero-based index in the Raw object's channel list. */
  index: number;
}

/** Recording-level metadata returned by GET /api/meg/metadata. */
export interface MegMetadata {
  session_id:     string;
  filename:       string;
  /** Sampling frequency in Hz (e.g. 1000 for Elekta Neuromag). */
  sampling_rate:  number;
  /** Total recording duration in seconds. */
  total_duration: number;
  n_samples:      number;
  n_channels:     number;
  /** Full channel list in recording order. */
  channels:       ChannelInfo[];
}

/**
 * Decimated timeseries for one channel over a requested time window.
 * All arrays have the same length (n_points).
 */
export interface ChannelTrace {
  name:   string;
  /** Uniform time axis in seconds. Length = n_points. */
  times:  number[];
  /** Centre value per bucket — (mins[i] + maxs[i]) / 2. */
  values: number[];
  /** Minimum signal value in each bucket (ribbon lower edge). */
  mins:   number[];
  /** Maximum signal value in each bucket (ribbon upper edge). */
  maxs:   number[];
  /** Physical unit of the values, e.g. 'T' or 'T/m'. */
  unit:   string;
}

/** Response from GET /api/meg/channels. */
export interface ChannelDataResponse {
  session_id: string;
  t_start:    number;
  t_end:      number;
  /** Actual number of output points per channel (≤ the requested n_points). */
  n_points:   number;
  channels:   ChannelTrace[];
}

/** PSD in dB for one channel. */
export interface PsdChannel {
  name:   string;
  /** Power in dB (10·log10(unit²/Hz)), length matches PsdResponse.freqs. */
  psd_db: number[];
}

/** Response from GET /api/meg/psd. */
export interface PsdResponse {
  session_id: string;
  /** Frequency axis in Hz. */
  freqs:    number[];
  channels: PsdChannel[];
  n_fft:    number;
  method:   string;
}

/** Minimal result returned by both load endpoints. */
export interface LoadResult {
  session_id:     string;
  filename:       string;
  n_channels:     number;
  sampling_rate:  number;
  total_duration: number;
}

// ── Configuration ─────────────────────────────────────────────────────────────

const API_BASE =
  (import.meta.env.VITE_MEG_API_URL as string | undefined) ??
  'http://localhost:8000';

// ── Internal fetch helper ─────────────────────────────────────────────────────

class MegApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: string,
  ) {
    super(`MEG API ${status}: ${detail}`);
    this.name = 'MegApiError';
  }
}

async function apiFetch<T>(
  path:  string,
  init?: RequestInit,
): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(`${API_BASE}${path}`, init);
  } catch (err) {
    // Network-level failure (server not running, CORS preflight blocked, etc.)
    throw new Error(
      `MEG backend is not reachable at ${API_BASE}. ` +
      'Start it with: uvicorn main:app --reload --port 8000',
    );
  }

  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json() as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // ignore parse errors; use statusText
    }
    throw new MegApiError(resp.status, detail);
  }

  return resp.json() as Promise<T>;
}

// ── Public API ────────────────────────────────────────────────────────────────

export const megApi = {
  // ── Loading ────────────────────────────────────────────────────────────────

  /**
   * Upload one or more .fif files to the backend and open a session.
   *
   * For split recordings, pass all parts together so the backend can write
   * them to the same temp directory and MNE can locate continuation files.
   * Call deleteSession() when done.
   */
  async loadFifFiles(files: File[], signal?: AbortSignal): Promise<LoadResult> {
    const form = new FormData();
    for (const f of files) form.append('files', f);
    return apiFetch<LoadResult>('/api/load-meg', {
      method: 'POST',
      body: form,
      signal,
    });
  },

  /** Convenience wrapper for a single .fif file. */
  async loadFile(file: File, signal?: AbortSignal): Promise<LoadResult> {
    return this.loadFifFiles([file], signal);
  },

  /**
   * Upload a KIT/Yokogawa MEG file set to the backend and open a session.
   *
   * The .con file is required; .mrk (HPI marker positions) and .pos (sensor
   * positions) are optional but strongly recommended — MNE uses them for
   * head-to-device coregistration.
   *
   * @param conFile  Raw continuous MEG data (.con) — required
   * @param mrkFile  HPI marker file (.mrk)          — optional
   * @param posFile  Sensor position file (.pos)      — optional
   */
  async loadKitFiles(
    conFile:  File,
    mrkFile?: File,
    posFile?: File,
    signal?:  AbortSignal,
  ): Promise<LoadResult> {
    const form = new FormData();
    form.append('con_file', conFile);
    if (mrkFile) form.append('mrk_file', mrkFile);
    if (posFile) form.append('pos_file', posFile);
    return apiFetch<LoadResult>('/api/load-meg-kit', {
      method: 'POST',
      body:   form,
      signal,
    });
  },

  /**
   * Load a .fif file by its absolute path on the server filesystem.
   *
   * Useful during local development when the OpenNeuro dataset is mounted
   * at a known location (e.g. /data/ds007640/sub-01/meg/…).
   */
  async loadPath(path: string, signal?: AbortSignal): Promise<LoadResult> {
    const form = new FormData();
    form.append('path', path);
    return apiFetch<LoadResult>('/api/load-meg-path', {
      method: 'POST',
      body: form,
      signal,
    });
  },

  // ── Metadata ───────────────────────────────────────────────────────────────

  /**
   * Fetch the full channel inventory and recording parameters.
   *
   * Call once after loadFile() / loadPath().  The returned MegMetadata
   * drives the channel-list sidebar, slider maxima, and auto-scale.
   */
  async getMetadata(sessionId: string, signal?: AbortSignal): Promise<MegMetadata> {
    return apiFetch<MegMetadata>(
      `/api/meg/metadata?session_id=${enc(sessionId)}`,
      { signal },
    );
  },

  // ── Time-series chunks ─────────────────────────────────────────────────────

  /**
   * Fetch decimated channel data for a viewport time window.
   *
   * n_points should equal Math.floor(canvas.clientWidth * devicePixelRatio)
   * so the server sends exactly as many points as the canvas can display.
   *
   * Pass an AbortSignal from createChunkFetcher() to cancel superseded
   * requests when the user pans the timeline.
   *
   * @param sessionId - Session from loadFile()
   * @param channels  - Channel names to fetch (keep ≤ 30 for fast JSON)
   * @param tStart    - Window start in seconds
   * @param tEnd      - Window end in seconds
   * @param nPoints   - Target output length per channel (≈ canvas width)
   * @param signal    - AbortSignal for cancellation
   */
  async getChannelData(
    sessionId: string,
    channels:  string[],
    tStart:    number,
    tEnd:      number,
    nPoints:   number,
    signal?:   AbortSignal,
  ): Promise<ChannelDataResponse> {
    const params = new URLSearchParams({
      session_id: sessionId,
      t_start:    tStart.toFixed(6),
      t_end:      tEnd.toFixed(6),
      n_points:   String(nPoints),
    });
    for (const ch of channels) params.append('channels', ch);

    return apiFetch<ChannelDataResponse>(`/api/meg/channels?${params}`, { signal });
  },

  // ── PSD ────────────────────────────────────────────────────────────────────

  /**
   * Compute Welch PSD for a set of channels.
   *
   * Pass tStart/tEnd to restrict the analysis to the current viewport
   * window rather than processing the entire recording.
   *
   * @param fmin   Minimum frequency in Hz (default 1)
   * @param fmax   Maximum frequency in Hz (default 100)
   * @param nFft   Welch FFT window length in samples (default 2048)
   * @param tStart Analysis window start in seconds (undefined = full recording)
   * @param tEnd   Analysis window end in seconds   (undefined = full recording)
   */
  async getPsd(
    sessionId: string,
    channels:  string[],
    fmin   = 1,
    fmax   = 100,
    nFft   = 2048,
    tStart?: number,
    tEnd?:   number,
    signal?: AbortSignal,
  ): Promise<PsdResponse> {
    const params = new URLSearchParams({
      session_id: sessionId,
      fmin:       String(fmin),
      fmax:       String(fmax),
      n_fft:      String(nFft),
    });
    for (const ch of channels) params.append('channels', ch);
    if (tStart !== undefined) params.set('t_start', tStart.toFixed(6));
    if (tEnd   !== undefined) params.set('t_end',   tEnd.toFixed(6));

    return apiFetch<PsdResponse>(`/api/meg/psd?${params}`, { signal });
  },

  // ── Topomap ────────────────────────────────────────────────────────────────

  /**
   * Render a 2-D sensor topomap averaged over the given time window.
   * Returns a base-64 PNG data URL ready to drop into <img src=...>.
   */
  async getTopomap(
    sessionId: string,
    tStart:    number,
    tEnd:      number,
    signal?:   AbortSignal,
  ): Promise<{ image: string }> {
    const params = new URLSearchParams({
      session_id: sessionId,
      t_start:    tStart.toFixed(6),
      t_end:      tEnd.toFixed(6),
    });
    return apiFetch<{ image: string }>(`/api/meg/topomap?${params}`, { signal });
  },

  // ── Session cleanup ────────────────────────────────────────────────────────

  /**
   * Delete the session and free the temp file on the server.
   *
   * Call inside a React useEffect cleanup:
   *
   *   useEffect(() => {
   *     return () => { megApi.deleteSession(sessionId); };
   *   }, [sessionId]);
   */
  async deleteSession(sessionId: string): Promise<void> {
    await apiFetch<unknown>(
      `/api/sessions/${enc(sessionId)}`,
      { method: 'DELETE' },
    );
  },

  // ── Health check ───────────────────────────────────────────────────────────

  /** Returns true if the backend is reachable. */
  async isAlive(): Promise<boolean> {
    try {
      const r = await apiFetch<{ status: string }>('/api/health');
      return r.status === 'ok';
    } catch {
      return false;
    }
  },
} as const;

// ── Utilities ─────────────────────────────────────────────────────────────────

/** Shorthand for encodeURIComponent. */
function enc(s: string): string {
  return encodeURIComponent(s);
}

/**
 * createChunkFetcher — stateful fetch wrapper with automatic cancellation.
 *
 * Each call to the returned function cancels any pending request from the
 * previous call before issuing a new one.  This prevents stale responses
 * from overwriting newer data when the user scrolls quickly.
 *
 * Example:
 *   const fetch = createChunkFetcher(sessionId);
 *   const debouncedFetch = debounce(fetch, 80);
 *
 *   // in scroll handler:
 *   debouncedFetch(channels, t0, t1, canvasWidth);
 */
export function createChunkFetcher(sessionId: string) {
  let controller: AbortController | null = null;

  return async (
    channels: string[],
    tStart:   number,
    tEnd:     number,
    nPoints:  number,
  ): Promise<ChannelDataResponse> => {
    // Cancel the previous in-flight request (if any)
    controller?.abort();
    controller = new AbortController();

    return megApi.getChannelData(
      sessionId,
      channels,
      tStart,
      tEnd,
      nPoints,
      controller.signal,
    );
  };
}

/**
 * debounce — delays invoking fn until wait ms after the last call.
 *
 * Returns a debounced version of fn together with a cancel() method.
 *
 * @param fn    Function to debounce
 * @param wait  Quiet period in milliseconds (80–150 ms is good for scroll)
 */
export function debounce<T extends (...args: never[]) => unknown>(
  fn:   T,
  wait: number,
): ((...args: Parameters<T>) => void) & { cancel(): void } {
  let timer: ReturnType<typeof setTimeout> | undefined;

  const debounced = (...args: Parameters<T>) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };

  debounced.cancel = () => clearTimeout(timer);

  return debounced;
}

/**
 * groupByType — convenience helper for the channel-list sidebar.
 *
 * Groups a channel array by their `type` string.
 *
 * Example output:
 *   { mag: [...102 channels], grad: [...204 channels], eog: [...2 channels] }
 */
export function groupByType(
  channels: ChannelInfo[],
): Record<string, ChannelInfo[]> {
  const result: Record<string, ChannelInfo[]> = {};
  for (const ch of channels) {
    (result[ch.type] ??= []).push(ch);
  }
  return result;
}

// Re-export the error class so callers can instanceof-check it
export { MegApiError };
