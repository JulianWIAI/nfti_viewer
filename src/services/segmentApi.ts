/**
 * segmentApi.ts — Client for the /api/segment SynthSeg endpoint
 *
 * Sends the original NIfTI file to the FastAPI backend for server-side
 * SynthSeg brain anatomy segmentation.  Returns a label map in the
 * original voxel space, encoded as a base64 uint8 flat array.
 *
 * Setup: run `python backend/download_models.py` once to fetch the model.
 *
 * Upload progress is tracked via XMLHttpRequest (xhrPost) so callers can
 * display a byte-level progress bar during the file transfer phase.
 */

import { xhrPost } from '../lib/xhrUpload';

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

// ── Optional progress callbacks ────────────────────────────────────────────────

/**
 * Options passed to segmentApi.segment for upload-progress tracking.
 *
 * onUploadProgress — called with a percentage 0–100 as bytes are sent
 * onUploadComplete — called once all bytes have been transmitted to the server
 * signal           — optional AbortSignal for cancellation
 */
export interface SegmentOptions {
  signal?:           AbortSignal;
  onUploadProgress?: (pct: number) => void;
  onUploadComplete?: () => void;
}

// ── API ───────────────────────────────────────────────────────────────────────

export const segmentApi = {
  /**
   * POST the NIfTI File to /api/segment.
   *
   * Always uses xhrPost so upload progress callbacks are always available.
   * The File object is a zero-copy Blob reference — no extra memory is used.
   *
   * @param file    The NIfTI File to upload.
   * @param options Optional progress callbacks and abort signal.
   */
  async segment(file: File, options?: SegmentOptions): Promise<SegmentResult> {
    // Build the multipart form payload.
    const form = new FormData();
    form.append('file', file, file.name);

    // Use xhrPost for real byte-level upload progress tracking.
    // The URL, form, and callbacks are forwarded to the XHR wrapper.
    return xhrPost<SegmentResult>({
      url:               `${BASE}/api/segment`,
      form,
      onUploadProgress:  options?.onUploadProgress,
      onUploadComplete:  options?.onUploadComplete,
      signal:            options?.signal,
    });
  },
};
