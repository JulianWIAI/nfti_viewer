/**
 * tractographyTypes.ts — Shared TypeScript type definitions for the DTI
 * tractography vtk.js rendering pipeline.
 *
 * COORDINATE CONVENTION
 * ──────────────────────
 * All coordinates are in world (RAS mm) space as returned by the FastAPI
 * endpoint POST /api/dti/tractography.  The backend serialises streamlines
 * in the NIfTI world frame defined by the 4×4 sform affine — identical to
 * the frame used by the existing volumetric MRI renderer, so no additional
 * coordinate transform is needed before passing to buildTractographyOverlay().
 *
 * JSON PAYLOAD SHAPE (from backend)
 * ───────────────────────────────────
 * {
 *   streamlines: [               // outer array = N streamlines
 *     [ [x,y,z], [x,y,z], … ],  // inner array = ordered 3-D points along fibre
 *     …
 *   ],
 *   n_streamlines:   number,
 *   n_before_filter: number,
 *   mean_length_mm:  number,
 *   peak_fa:         number,
 *   duration_ms:     number,
 * }
 *
 * The TypeScript types below map directly onto this payload and onto the
 * three GPU buffers constructed in tractographyGeometry.ts.
 */

import type vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';

// ── Streamline primitives ────────────────────────────────────────────────────

/**
 * A single 3-D coordinate in world (RAS mm) space.
 * Index 0 = X (left→right), 1 = Y (posterior→anterior), 2 = inferior→superior.
 */
export type StreamlinePoint = [number, number, number];

/**
 * An ordered sequence of points tracing one white-matter fibre from one
 * cortical insertion to another.  Must contain at least 2 points to be
 * rendered as a line (single-point streamlines are silently skipped).
 */
export type Streamline = StreamlinePoint[];

/**
 * The full tractogram — the outer array returned by POST /api/dti/tractography
 * under the key `streamlines`.
 */
export type StreamlineArray = Streamline[];

// ── GPU buffer output from tractographyGeometry.ts ──────────────────────────

/**
 * Pre-allocated flat GPU buffers produced by buildTractographyBuffers().
 *
 * These are passed directly to vtk.js vtkDataArray / vtkCellArray without
 * copying.  Each buffer is allocated exactly once to avoid GC pressure.
 *
 * pointsFlat   : Float32Array of length totalPoints × 3
 *                Layout: [x₀,y₀,z₀, x₁,y₁,z₁, …]
 *
 * cellData     : Uint32Array describing VTK line cells
 *                Legacy flat format: [n₀, id₀₀, id₀₁, …, n₁, id₁₀, …]
 *                where nᵢ = number of points in streamline i and idᵢⱼ is the
 *                global (0-based) index of the j-th point of streamline i.
 *
 * colorsFlat   : Uint8Array of length totalPoints × 3
 *                Direction-Encoded Color (DEC): each RGB triple encodes the
 *                local fibre orientation as |dx|, |dy|, |dz| scaled to 0–255.
 *                X=Red (L-R), Y=Green (A-P), Z=Blue (I-S).
 *
 * stats        : Scalar summary returned in TractographyBundle for UI display.
 */
export interface TractographyBuffers {
  pointsFlat:  Float32Array;
  cellData:    Uint32Array;
  colorsFlat:  Uint8Array;
  stats:       TractographyGeometryStats;
}

// ── Runtime statistics ───────────────────────────────────────────────────────

/**
 * Scalar summary computed alongside buffer construction in
 * buildTractographyBuffers().  All values describe the rendered (post-filter,
 * post-decimation) tractogram, not the raw tractography output.
 */
export interface TractographyGeometryStats {
  /** Number of streamlines actually rendered (skips single-point streamlines). */
  renderedStreamlines: number;
  /** Total number of 3-D points across all rendered streamlines. */
  totalPoints:         number;
  /** Size of pointsFlat in bytes (= totalPoints × 3 × 4). */
  pointsByteLength:    number;
  /** Size of colorsFlat in bytes (= totalPoints × 3 × 1). */
  colorsByteLength:    number;
}

// ── Live overlay handle ──────────────────────────────────────────────────────

/**
 * Object returned by buildTractographyOverlay().
 *
 * USAGE PATTERN
 * ──────────────
 *   const bundle = buildTractographyOverlay(apiResponse.streamlines);
 *   ctx.volumeRenderer.addActor(bundle.actor);
 *   ctx.renderWindow.render();
 *
 *   // Hide/show without rebuilding:
 *   bundle.setVisible(false);
 *   ctx.renderWindow.render();
 *
 *   // On new upload or unmount — ALWAYS dispose before removing actor:
 *   bundle.dispose();
 *   ctx.volumeRenderer.removeActor(bundle.actor);
 *   ctx.renderWindow.render();
 */
export interface TractographyBundle {
  /** vtk.js actor — add to ctx.volumeRenderer.addActor(bundle.actor). */
  actor:      ReturnType<typeof vtkActor.newInstance>;
  /** Show or hide the overlay. Caller must call renderWindow.render(). */
  setVisible: (visible: boolean) => void;
  /** Release all vtk.js pipeline objects. Call before removing actor. */
  dispose:    () => void;
  /** Geometry statistics for display in the UI (point count, memory usage). */
  stats:      TractographyGeometryStats;
}
