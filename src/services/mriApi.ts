/**
 * mriApi.ts — HTTP client for the MRI analysis endpoints
 * ────────────────────────────────────────────────────────
 * Provides a typed wrapper around POST /api/mri/volumetrics so the frontend
 * never constructs raw fetch() calls inline.
 *
 * The base URL is resolved from the same origin as the other API calls.
 */

import type { VolumetricsResult } from '../types/analysis.types';

// Re-export so component files only need one import.
export type { VolumetricsResult };

const BASE = '';

/**
 * Encode a Uint8Array to a base64 string without using the spread operator
 * (which blows the call stack for large volumes — a typical 256³ brain has
 * 16.7M bytes, far above the maximum spread size on V8).
 */
function uint8ToBase64(arr: Uint8Array): string {
  let binary = '';
  // Process in 8 kB chunks to keep string concatenation within budget.
  const CHUNK = 8192;
  for (let offset = 0; offset < arr.length; offset += CHUNK) {
    binary += String.fromCharCode(...arr.subarray(offset, offset + CHUNK));
  }
  return btoa(binary);
}

// ── API object ────────────────────────────────────────────────────────────────

export const mriApi = {
  /**
   * Send the segmentation label map to the server and receive tissue volumes.
   *
   * @param labelFlat  Flat uint8 array of FreeSurfer label IDs in Fortran order
   *                   — the same array stored in ``SegmentationBundle.labelFlat``.
   * @param dims       [x, y, z] voxel grid dimensions.
   * @param pixDims    [dx, dy, dz] voxel spacing in mm from the NIfTI header
   *                   (``header.pixDims[1..3]``).
   * @param signal     Optional AbortController signal for cancellation.
   */
  async computeVolumetrics(
    labelFlat: Uint8Array,
    dims:      [number, number, number],
    pixDims:   [number, number, number],
    signal?:   AbortSignal,
  ): Promise<VolumetricsResult> {
    // Encode the label array to base64 — the same format returned by /api/segment.
    const labelsB64 = uint8ToBase64(labelFlat);

    const resp = await fetch(`${BASE}/api/mri/volumetrics`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        labels_b64: labelsB64,
        dims:       dims,
        voxel_mm:   pixDims,
      }),
      signal,
    });

    if (!resp.ok) {
      const text = await resp.text().catch(() => resp.statusText);
      throw new Error(`Volumetrics request failed (${resp.status}): ${text}`);
    }

    return resp.json() as Promise<VolumetricsResult>;
  },
};
