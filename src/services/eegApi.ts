/**
 * eegApi.ts — FastAPI EEG (BrainVision) backend client
 * ──────────────────────────────────────────────────────
 *
 * Mirrors the MEG API pattern exactly; the only differences are:
 *   • /api/load-eeg   accepts multiple files (form field "files[]")
 *   • /api/eeg/*      for metadata and channel data
 *   • /api/eeg/{id}   DELETE for session cleanup
 *
 * Types (ChannelInfo, ChannelTrace, ChannelDataResponse, LoadResult)
 * are shared with megApi.ts since both use the same Pydantic models.
 */

export type {
  ChannelInfo,
  ChannelTrace,
  ChannelDataResponse,
  LoadResult,
} from './megApi';

import type { ChannelInfo, ChannelDataResponse, LoadResult } from './megApi';

// ── Metadata response ─────────────────────────────────────────────────────────

/** EEG recording-level metadata (mirrors MegMetadata). */
export interface EegMetadata {
  session_id:     string;
  filename:       string;
  sampling_rate:  number;
  total_duration: number;
  n_samples:      number;
  n_channels:     number;
  channels:       ChannelInfo[];
}

// ── Configuration ─────────────────────────────────────────────────────────────

const API_BASE =
  (import.meta.env.VITE_MEG_API_URL as string | undefined) ??
  'http://localhost:8000';

// ── Internal fetch helper ─────────────────────────────────────────────────────

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new Error(
      `EEG backend is not reachable at ${API_BASE}. ` +
      'Start it with: uvicorn main:app --reload --port 8000',
    );
  }
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json() as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch { /* ignore */ }
    throw new Error(`EEG API ${resp.status}: ${detail}`);
  }
  return resp.json() as Promise<T>;
}

// ── Public API ────────────────────────────────────────────────────────────────

export const eegApi = {
  /**
   * Upload the BrainVision file set (.vhdr + .eeg + .vmrk) to the backend.
   * All three files must be passed; MNE discovers the .eeg and .vmrk via
   * the references inside the .vhdr header.
   */
  async loadFiles(files: File[], signal?: AbortSignal): Promise<LoadResult> {
    const form = new FormData();
    for (const f of files) form.append('files', f);
    return apiFetch<LoadResult>('/api/load-eeg', { method: 'POST', body: form, signal });
  },

  /** Fetch the full channel list and recording parameters. */
  async getMetadata(sessionId: string, signal?: AbortSignal): Promise<EegMetadata> {
    return apiFetch<EegMetadata>(
      `/api/eeg/metadata?session_id=${encodeURIComponent(sessionId)}`,
      { signal },
    );
  },

  /**
   * Fetch decimated channel data for a viewport time window.
   * n_points should equal Math.floor(canvas.clientWidth * devicePixelRatio).
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
    return apiFetch<ChannelDataResponse>(`/api/eeg/channels?${params}`, { signal });
  },

  /** Delete the session and free the temp directory on the server. */
  async deleteSession(sessionId: string): Promise<void> {
    await apiFetch<unknown>(
      `/api/eeg/${encodeURIComponent(sessionId)}`,
      { method: 'DELETE' },
    );
  },
} as const;
