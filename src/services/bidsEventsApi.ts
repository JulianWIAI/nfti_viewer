/**
 * bidsEventsApi.ts — API client for the BIDS events endpoints
 * ─────────────────────────────────────────────────────────────
 *
 * Wraps the two backend endpoints defined in backend/routers/bids_events.py:
 *
 *   uploadEventsFile(file)      → POST /api/bids/events/upload
 *   fetchEventsFromPath(path)   → GET  /api/bids/events?path=<server_path>
 *
 * Both return a BidsEventsResult, mapping the backend snake_case keys
 * to the frontend camelCase ExperimentEvent fields.
 *
 * USAGE
 * ──────
 *   // Browser drag-and-drop (recommended for most users):
 *   const result = await bidsEventsApi.uploadEventsFile(tsvFile);
 *   setEvents(result.events);
 *
 *   // Server-mounted BIDS dataset (local dev):
 *   const result = await bidsEventsApi.fetchEventsFromPath(
 *     '/data/ds004482/sub-01/func/sub-01_task-audiovis_events.tsv'
 *   );
 */

import type { BidsEventsResult, ExperimentEvent } from '../types/bids_events.types';

// ── Internal fetch helper ─────────────────────────────────────────────────────

const API_BASE =
  (import.meta.env.VITE_MEG_API_URL as string | undefined) ??
  'http://localhost:8000';

/**
 * Shared fetch wrapper: throws a descriptive Error on non-OK responses.
 */
async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new Error(
      `Backend is not reachable at ${API_BASE}. ` +
      'Start it with: uvicorn main:app --reload --port 8000',
    );
  }

  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = await resp.json() as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch { /* ignore JSON parse errors */ }
    throw new Error(`BIDS events API error: ${detail}`);
  }

  return resp.json() as Promise<T>;
}

// ── Response normaliser ───────────────────────────────────────────────────────

/**
 * Convert the backend snake_case event object into the frontend camelCase
 * ExperimentEvent.  The rest of the BidsEventsResponse fields already match.
 */
function normaliseEvent(raw: {
  id: string;
  onset: number;
  duration: number;
  trial_type: string;
  response_time: number | null;
  color: string;
}): ExperimentEvent {
  return {
    id:           raw.id,
    onset:        raw.onset,
    duration:     raw.duration,
    trialType:    raw.trial_type,
    responseTime: raw.response_time,
    color:        raw.color,
  };
}

/**
 * Convert the raw backend response to a frontend BidsEventsResult.
 */
function normaliseResponse(raw: {
  events:         ReturnType<typeof normaliseEvent> extends ExperimentEvent ? any[] : any[];
  trial_types:    string[];
  color_map:      Record<string, string>;
  n_events:       number;
  total_duration: number;
}): BidsEventsResult {
  return {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    events:        (raw.events as any[]).map(normaliseEvent),
    trialTypes:    raw.trial_types,
    colorMap:      raw.color_map,
    nEvents:       raw.n_events,
    totalDuration: raw.total_duration,
  };
}

// ── Public API object ─────────────────────────────────────────────────────────

export const bidsEventsApi = {
  /**
   * Upload a .tsv file from the browser and receive parsed events.
   *
   * The file is sent as multipart/form-data and parsed in-memory on the
   * server — no persistent session is created.
   *
   * @param file - The _events.tsv File object from a drag-drop or file input.
   */
  async uploadEventsFile(file: File): Promise<BidsEventsResult> {
    const form = new FormData();
    form.append('file', file);

    const raw = await apiFetch<Record<string, unknown>>('/api/bids/events/upload', {
      method: 'POST',
      body:   form,
    });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return normaliseResponse(raw as any);
  },

  /**
   * Fetch events from a .tsv file that is already on the server filesystem.
   *
   * Useful during local development when the BIDS dataset is mounted at a
   * known path (e.g. when running uvicorn from the same machine that hosts
   * the OpenNeuro dataset).
   *
   * @param serverPath - Absolute server-side path to the *_events.tsv file.
   */
  async fetchEventsFromPath(serverPath: string): Promise<BidsEventsResult> {
    const encoded = encodeURIComponent(serverPath);
    const raw = await apiFetch<Record<string, unknown>>(
      `/api/bids/events?path=${encoded}`,
    );

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return normaliseResponse(raw as any);
  },
} as const;
