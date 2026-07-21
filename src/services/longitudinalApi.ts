/**
 * longitudinalApi.ts — HTTP client for POST /api/longitudinal/delta
 * ──────────────────────────────────────────────────────────────────
 *
 * Mirrors anomalyApi.ts: a thin, typed wrapper that returns a strongly-
 * typed result object or throws a descriptive Error on failure.
 *
 * The endpoint accepts two NIfTI uploads (baseline + follow-up), co-registers
 * the follow-up to the baseline voxel space using dipy affine registration
 * (mutual information metric, CoM → Translation → Rigid → optional Affine),
 * subtracts the arrays, and returns the float32 delta volume serialised in
 * Fortran order (X-fastest, matching vtk.js vtkImageData layout).
 *
 * Upload progress is tracked via XMLHttpRequest (xhrPost) so callers can
 * display a byte-level progress bar during the file transfer phase.
 */

import { xhrPost } from '../lib/xhrUpload';

/** Typed response from the FastAPI longitudinal delta endpoint. */
export interface LongitudinalApiResult {
  /**
   * Base64-encoded float32 delta volume in Fortran order (X varies fastest).
   * Decode with atob() → Uint8Array → Float32Array buffer reinterpretation.
   * Positive values = tissue growth; negative values = atrophy.
   */
  delta: string;

  /** [X, Y, Z] voxel dimensions matching the baseline scan's voxel space. */
  dims: [number, number, number];

  /**
   * 4×4 RAS-mm affine of the baseline scan, row-major flattened into 16 float64
   * values.  Passed to the vtk.js frontend so the delta volume aligns with the
   * structural MRI in world space.
   */
  affine: number[];

  /** Minimum delta intensity — lower bound for the diverging colormap (blue). */
  min_val: number;

  /** Maximum delta intensity — upper bound for the diverging colormap (red). */
  max_val: number;

  /** Number of voxels with positive delta (growth / fluid expansion). */
  n_positive: number;

  /** Number of voxels with negative delta (atrophy / tissue loss). */
  n_negative: number;

  /** Server-side wall-clock processing time in milliseconds. */
  duration_ms: number;

  /** Registration strategy used: 'rigid' (6 DOF) or 'affine' (12 DOF). */
  transform_type: string;
}

// ── Optional progress callbacks ────────────────────────────────────────────────

/**
 * Options for longitudinalApi.computeDelta enabling upload-progress tracking.
 *
 * onUploadProgress — called with a percentage 0–100 as bytes are sent
 * onUploadComplete — called once all bytes have been transmitted
 * signal           — optional AbortSignal for cancellation
 */
export interface LongitudinalOptions {
  onUploadProgress?: (pct: number) => void;
  onUploadComplete?: () => void;
  signal?:           AbortSignal;
}

export const longitudinalApi = {
  /**
   * POST /api/longitudinal/delta
   *
   * Upload two NIfTI files (baseline + follow-up) and receive a float32 delta
   * volume co-registered to the baseline voxel space.
   *
   * Always uses xhrPost so upload-progress callbacks are available.
   * Both files are appended to the same FormData; the backend separates them
   * by the field names 'baseline', 'followup', and 'transform_type'.
   *
   * @param baseline      Earlier scan — the delta is returned in this scan's space.
   * @param followup      Later scan — co-registered to baseline before subtraction.
   * @param transformType 'rigid' (recommended for same-scanner data, ~30–60 s) or
   *                      'affine' (cross-scanner with different voxel sizes, ~60–120 s).
   * @param options       Optional progress callbacks and abort signal.
   * @throws Error with a human-readable message on network failure or non-OK status.
   */
  async computeDelta(
    baseline:      File,
    followup:      File,
    transformType: 'rigid' | 'affine' = 'rigid',
    options?:      LongitudinalOptions,
  ): Promise<LongitudinalApiResult> {
    // Build the multipart form payload with both NIfTI files and the
    // transform type parameter.  Field names match the FastAPI endpoint.
    const form = new FormData();
    form.append('baseline',       baseline);
    form.append('followup',       followup);
    form.append('transform_type', transformType);

    // Use xhrPost for real byte-level upload progress tracking.
    return xhrPost<LongitudinalApiResult>({
      url:              '/api/longitudinal/delta',
      form,
      onUploadProgress: options?.onUploadProgress,
      onUploadComplete: options?.onUploadComplete,
      signal:           options?.signal,
    });
  },
};
