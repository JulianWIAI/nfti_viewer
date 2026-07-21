/**
 * fmri.types.ts — TypeScript types for 4-D fMRI data payloads.
 *
 * A 4-D NIfTI file stores one 3-D brain volume per TR (repetition time).
 * FmriPayload extends VolumePayload with the extra metadata needed by the
 * fMRI renderer and the multimodal sync context.
 *
 * Data flow:
 *   nifti.worker.ts  →  FmriPayload  →  FmriPanel  →  fmriVolumeRenderer
 *                                    ↘  SyncContext  (TR, currentVolumeIdx)
 */

import type { VolumePayload } from './nifti.types';

// ── 4-D fMRI payload ─────────────────────────────────────────────────────────

/**
 * Extends VolumePayload with 4-D specific fields derived from the NIfTI header.
 *
 * The underlying volumeData buffer is the complete 4-D array (all timepoints
 * concatenated).  fmriVolumeRenderer slices out one 3-D frame on demand.
 */
export interface FmriPayload extends VolumePayload {
  /**
   * Number of temporal volumes in the 4-D NIfTI.
   * Equal to dims[4] in the NIfTI header (at least 1).
   */
  nTimepoints: number;

  /**
   * Repetition time in seconds — the interval between consecutive volumes.
   * Stored in pixDims[4] of the NIfTI header (may be 0 for single-volume files).
   */
  tr: number;
}

// ── Source-estimate overlay (returned by POST /api/meg/source-estimate) ───────

/**
 * One vertex of the source-space mesh with an associated activation amplitude.
 * Returned from the backend as a flat arrays and mapped to a BOLD/MEG overlay.
 */
export interface SourceVertex {
  /** MNI x-coordinate in mm. */
  x: number;
  /** MNI y-coordinate in mm. */
  y: number;
  /** MNI z-coordinate in mm. */
  z: number;
  /** Amplitude of the source estimate (e.g., dSPM statistic or nAm). */
  amplitude: number;
}

/**
 * Response from POST /api/meg/source-estimate.
 * Contains the source-space point cloud used by boldOverlay.ts to build
 * a vtk.js glyph mapper in the fMRI pane.
 */
export interface SourceEstimateResult {
  /** Source-space vertices with activation amplitudes. */
  vertices: SourceVertex[];
  /** Reconstruction method used on the backend (e.g. 'dSPM', 'MNE', 'sLORETA'). */
  method: string;
  /** Peak amplitude across all vertices — used to scale the colour map. */
  peakAmplitude: number;
  /** Wall-clock duration of the backend computation in ms. */
  durationMs: number;
}
