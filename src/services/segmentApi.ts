/**
 * segmentApi.ts — Client for the /api/segment SynthSeg endpoint
 *
 * Sends the original NIfTI file to the FastAPI backend for server-side
 * SynthSeg brain anatomy segmentation.  Returns a label map in the
 * original voxel space, encoded as a base64 uint8 flat array.
 *
 * Setup: run `python backend/download_models.py` once to fetch the model.
 */

const BASE = '';

// ── Response type ─────────────────────────────────────────────────────────────

export interface SegmentResult {
  /** Base64-encoded Uint8Array of FreeSurfer label IDs, flat, row-major. */
  labels: string;
  /** Volume dimensions [x, y, z] matching the original NIfTI. */
  dims: [number, number, number];
  /** 4×4 RAS affine, flattened row-major (16 values). */
  affine: number[];
  /** Number of unique labels found (including background). */
  n_labels: number;
  /** Server-side wall-clock duration in milliseconds. */
  duration_ms: number;
}

// ── API ───────────────────────────────────────────────────────────────────────

export const segmentApi = {
  /**
   * POST the NIfTI File to /api/segment.
   * The File object is a zero-copy Blob reference — no extra memory is used.
   */
  async segment(file: File, signal?: AbortSignal): Promise<SegmentResult> {
    const form = new FormData();
    form.append('file', file, file.name);

    const resp = await fetch(`${BASE}/api/segment`, {
      method: 'POST',
      body: form,
      signal,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error((err as { detail?: string }).detail ?? `Segmentation failed (${resp.status})`);
    }

    return resp.json() as Promise<SegmentResult>;
  },
};
