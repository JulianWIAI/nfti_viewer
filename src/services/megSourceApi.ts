/**
 * megSourceApi.ts — API client for the MEG source-estimate endpoint
 * ──────────────────────────────────────────────────────────────────
 *
 * Sends a POST to /api/meg/source-estimate with the MEG session ID and
 * optional fMRI activation baseline, then returns parsed SourceEstimateResult.
 *
 * The backend runs MNE-Python dSPM / MNE / sLORETA source localisation on the
 * MEG data from the session and returns a source-space point cloud of
 * (x, y, z, amplitude) vertices.  If an fMRI BOLD volume is provided as a
 * spatial prior the backend uses it to regularise the inverse solution.
 *
 * This client is stateless — each call corresponds to one backend job.
 * Long-running jobs (typically 2–4 min) should be triggered with user
 * confirmation and should show progress feedback in the calling component.
 */

import type { SourceEstimateResult } from '../types/fmri.types';

// ── Request body ──────────────────────────────────────────────────────────────

export interface SourceEstimateRequest {
  /**
   * MEG session ID returned by POST /api/load-meg.
   * The backend looks up the loaded Raw object from the session store.
   */
  sessionId: string;

  /**
   * Source estimation method passed to MNE-Python.
   * 'dSPM' (dynamic statistical parametric mapping) is the default and is
   * the most robust for evoked / resting-state MEG.
   */
  method?: 'dSPM' | 'MNE' | 'sLORETA';

  /**
   * Analysis window start in seconds (default: 0).
   * The backend averages the MEG signal over [tMin, tMax] before estimating.
   */
  tMin?: number;

  /**
   * Analysis window end in seconds (default: full recording).
   */
  tMax?: number;

  /**
   * Optional: path or identifier of an fMRI BOLD volume on the backend.
   * When provided the backend uses the BOLD activation map as a spatial
   * prior to weight the MEG inverse solution (Bayesian regularisation).
   * This is an experimental feature — omit for standard dSPM.
   */
  boldPriorPath?: string;
}

// ── API client function ───────────────────────────────────────────────────────

/**
 * Request a MEG source estimate from the backend.
 *
 * @param request - Session ID and optional analysis parameters.
 * @returns Resolved SourceEstimateResult once the backend job completes.
 * @throws Error if the HTTP response is not OK.
 */
export async function requestSourceEstimate(
  request: SourceEstimateRequest,
): Promise<SourceEstimateResult> {
  const response = await fetch('/api/meg/source-estimate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id:      request.sessionId,
      method:          request.method     ?? 'dSPM',
      t_min:           request.tMin       ?? 0,
      t_max:           request.tMax       ?? null,
      bold_prior_path: request.boldPriorPath ?? null,
    }),
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const json = await response.json();
      detail = json.detail ?? detail;
    } catch { /* non-JSON body */ }
    throw new Error(`Source estimate failed: ${detail}`);
  }

  const json = await response.json();

  // Map snake_case backend keys to camelCase frontend types
  return {
    vertices:      json.vertices,
    method:        json.method,
    peakAmplitude: json.peak_amplitude,
    durationMs:    json.duration_ms,
  };
}
