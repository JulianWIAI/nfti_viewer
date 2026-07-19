/**
 * anomalyApi.ts — HTTP client for POST /api/anomalies/detect
 * ─────────────────────────────────────────────────────────────
 *
 * Mirrors the pattern used by segmentApi.ts: a thin fetch wrapper that
 * returns a typed result object or throws a descriptive Error.
 */

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

export const anomalyApi = {
  /**
   * POST /api/anomalies/detect
   *
   * Upload a NIfTI file and receive a binary anomaly mask.  The response is
   * already in the original voxel space — the backend handles resampling from
   * 1mm isotropic BraTS space (240×240×155) back to the source dimensions.
   *
   * @param file  The same NIfTI File object uploaded by the user.
   * @throws Error with a human-readable message on network failure or non-OK status.
   */
  async detect(file: File): Promise<AnomalyApiResult> {
    const form = new FormData();
    // Field name must match the FastAPI parameter name 'nifti_file'.
    form.append('nifti_file', file);

    const res = await fetch('/api/anomalies/detect', {
      method: 'POST',
      body:   form,
    });

    if (!res.ok) {
      // Surface the FastAPI detail message when available.
      // FastAPI validation errors return detail as an array, not a string,
      // so coerce to string explicitly to avoid "[object Object]" display.
      const text = await res.text().catch(() => res.statusText);
      let detail = text;
      try {
        const json = JSON.parse(text) as { detail?: unknown };
        if (json.detail !== undefined) {
          detail = typeof json.detail === 'string'
            ? json.detail
            : JSON.stringify(json.detail);
        }
      } catch { /* not JSON — use raw text */ }
      throw new Error(`Anomaly detection failed (${res.status}): ${detail}`);
    }

    return res.json() as Promise<AnomalyApiResult>;
  },
};
