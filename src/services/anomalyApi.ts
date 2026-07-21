/**
 * anomalyApi.ts — HTTP client for POST /api/anomalies/detect
 * ─────────────────────────────────────────────────────────────
 *
 * Mirrors the pattern used by segmentApi.ts: a thin wrapper that
 * returns a typed result object or throws a descriptive Error.
 *
 * Upload progress is tracked via XMLHttpRequest (xhrPost) so callers can
 * display a byte-level progress bar during the file transfer phase.
 */

import { xhrPost } from '../lib/xhrUpload';

/** Typed response from the FastAPI anomaly detection endpoint. */
export interface AnomalyApiResult {
  /** Base64-encoded uint8 binary mask in Fortran order (X varies fastest). */
  mask:        string;
  /** [X, Y, Z] voxel dimensions of the mask — matches the original NIfTI space. */
  dims:        [number, number, number];
  /** 4×4 RAS affine, row-major flattened into 16 float64 values. */
  affine:      number[];
  /** Total number of anomalous voxels (mask value = 1). */
  n_anomaly:   number;
  /** Server-side wall-clock processing time in milliseconds. */
  duration_ms: number;
}

// ── Optional progress callbacks ────────────────────────────────────────────────

/**
 * Options for anomalyApi.detect enabling upload-progress tracking.
 *
 * onUploadProgress — called with a percentage 0–100 as bytes are sent
 * onUploadComplete — called once all bytes have been transmitted
 * signal           — optional AbortSignal for cancellation
 */
export interface AnomalyOptions {
  onUploadProgress?: (pct: number) => void;
  onUploadComplete?: () => void;
  signal?:           AbortSignal;
}

export const anomalyApi = {
  /**
   * POST /api/anomalies/detect
   *
   * Upload a NIfTI file and receive a binary anomaly mask.  The response is
   * already in the original voxel space — the backend handles resampling from
   * 1mm isotropic BraTS space (240×240×155) back to the source dimensions.
   *
   * Always uses xhrPost so upload-progress callbacks are available.
   * Field name must match the FastAPI parameter name 'nifti_file'.
   *
   * @param file    The same NIfTI File object uploaded by the user.
   * @param options Optional progress callbacks and abort signal.
   * @throws Error with a human-readable message on network failure or non-OK status.
   */
  async detect(file: File, options?: AnomalyOptions): Promise<AnomalyApiResult> {
    // Build the multipart form payload.
    // Field name must match the FastAPI parameter name 'nifti_file'.
    const form = new FormData();
    form.append('nifti_file', file);

    // Use xhrPost for real byte-level upload progress tracking.
    return xhrPost<AnomalyApiResult>({
      url:              '/api/anomalies/detect',
      form,
      onUploadProgress: options?.onUploadProgress,
      onUploadComplete: options?.onUploadComplete,
      signal:           options?.signal,
    });
  },
};
