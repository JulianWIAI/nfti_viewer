/**
 * longitudinalApi.ts — HTTP client for POST /api/longitudinal/delta
 * ──────────────────────────────────────────────────────────────────
 *
 * Mirrors anomalyApi.ts: a thin, typed fetch wrapper that returns a strongly-
 * typed result object or throws a descriptive Error on failure.
 *
 * The endpoint accepts two NIfTI uploads (baseline + follow-up), co-registers
 * the follow-up to the baseline voxel space using dipy affine registration
 * (mutual information metric, CoM → Translation → Rigid → optional Affine),
 * subtracts the arrays, and returns the float32 delta volume serialised in
 * Fortran order (X-fastest, matching vtk.js vtkImageData layout).
 */

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

export const longitudinalApi = {
  /**
   * POST /api/longitudinal/delta
   *
   * Upload two NIfTI files (baseline + follow-up) and receive a float32 delta
   * volume co-registered to the baseline voxel space.
   *
   * @param baseline      Earlier scan — the delta is returned in this scan's space.
   * @param followup      Later scan — co-registered to baseline before subtraction.
   * @param transformType 'rigid' (recommended for same-scanner data, ~30–60 s) or
   *                      'affine' (cross-scanner with different voxel sizes, ~60–120 s).
   * @throws Error with a human-readable message on network failure or non-OK status.
   */
  async computeDelta(
    baseline:      File,
    followup:      File,
    transformType: 'rigid' | 'affine' = 'rigid',
  ): Promise<LongitudinalApiResult> {
    const form = new FormData();
    form.append('baseline',       baseline);
    form.append('followup',       followup);
    form.append('transform_type', transformType);

    const res = await fetch('/api/longitudinal/delta', {
      method: 'POST',
      body:   form,
    });

    if (!res.ok) {
      // Surface the FastAPI detail message when available.
      const text = await res.text().catch(() => res.statusText);
      let detail = text;
      try {
        const json = JSON.parse(text) as { detail?: string };
        if (json.detail) detail = json.detail;
      } catch { /* not JSON — use raw text */ }
      throw new Error(`Longitudinal delta failed (${res.status}): ${detail}`);
    }

    return res.json() as Promise<LongitudinalApiResult>;
  },
};
