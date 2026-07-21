/**
 * dualSegmentApi.ts — Concurrent SynthSeg segmentation for two subjects.
 *
 * Fires both /api/segment requests simultaneously via Promise.all so that
 * the combined wait time equals the longer of the two jobs rather than
 * their sum.  No backend changes are required — the existing endpoint is
 * stateless and handles concurrent requests independently.
 *
 * Upload progress callbacks are forwarded independently per subject so the
 * GlobalTaskBar can show separate progress bars for Subject A and Subject B.
 */

import { segmentApi } from './segmentApi';
import type { SegmentResult } from './segmentApi';

// ── Per-subject callback options ──────────────────────────────────────────────

/**
 * Optional upload-progress callbacks for each of the two subjects.
 *
 * A callbacks:
 *   onProgressA   — called with pct 0–100 as Subject A bytes are sent
 *   onCompleteA   — called once Subject A upload is fully done
 *
 * B callbacks:
 *   onProgressB   — called with pct 0–100 as Subject B bytes are sent
 *   onCompleteB   — called once Subject B upload is fully done
 */
export interface DualSegmentOptions {
  onProgressA?: (pct: number) => void;
  onCompleteA?: () => void;
  onProgressB?: (pct: number) => void;
  onCompleteB?: () => void;
}

export const dualSegmentApi = {
  /**
   * Run SynthSeg on two NIfTI files concurrently.
   *
   * Fires both segmentApi.segment() calls simultaneously via Promise.all.
   * Each call receives its own independent progress callbacks so the UI
   * can show separate progress bars for Subject A and Subject B.
   *
   * @param fileA   Subject A's original NIfTI File object.
   * @param fileB   Subject B's original NIfTI File object.
   * @param options Optional per-subject upload-progress callbacks.
   * @returns Promise resolving to [resultA, resultB] in native voxel spaces.
   * @throws If either request fails the whole Promise rejects.
   */
  async segmentBoth(
    fileA:    File,
    fileB:    File,
    options?: DualSegmentOptions,
  ): Promise<[SegmentResult, SegmentResult]> {
    // Run both requests concurrently; each has its own independent callbacks.
    // Promise.all rejects immediately if either segmentation fails.
    return Promise.all([
      segmentApi.segment(fileA, {
        // Forward Subject A progress callbacks if provided.
        onUploadProgress: options?.onProgressA,
        onUploadComplete: options?.onCompleteA,
      }),
      segmentApi.segment(fileB, {
        // Forward Subject B progress callbacks if provided.
        onUploadProgress: options?.onProgressB,
        onUploadComplete: options?.onCompleteB,
      }),
    ]);
  },
};
