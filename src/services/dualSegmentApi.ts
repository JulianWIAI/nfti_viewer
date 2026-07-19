/**
 * dualSegmentApi.ts — Concurrent SynthSeg segmentation for two subjects.
 *
 * Fires both /api/segment requests simultaneously via Promise.all so that
 * the combined wait time equals the longer of the two jobs rather than
 * their sum.  No backend changes are required — the existing endpoint is
 * stateless and handles concurrent requests independently.
 */

import { segmentApi } from './segmentApi';
import type { SegmentResult } from './segmentApi';

export const dualSegmentApi = {
  /**
   * Run SynthSeg on two NIfTI files concurrently.
   *
   * @param fileA  Subject A's original NIfTI File object.
   * @param fileB  Subject B's original NIfTI File object.
   * @returns Promise resolving to [resultA, resultB] in native voxel spaces.
   * @throws If either request fails the whole Promise rejects.
   */
  async segmentBoth(
    fileA: File,
    fileB: File,
  ): Promise<[SegmentResult, SegmentResult]> {
    return Promise.all([
      segmentApi.segment(fileA),
      segmentApi.segment(fileB),
    ]);
  },
};
