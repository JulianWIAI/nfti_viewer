/**
 * tractographyGeometry.ts — Two-pass GPU buffer builder for DTI streamlines
 * ────────────────────────────────────────────────────────────────────────────
 *
 * PURPOSE
 * ────────
 * Converts the nested JSON streamline array  [ [ [x,y,z], … ], … ]  returned
 * by POST /api/dti/tractography into three flat typed arrays that are passed
 * directly to vtk.js without any intermediate copies:
 *
 *   pointsFlat  Float32Array   (x,y,z) coordinates of every point
 *   cellData    Uint32Array    VTK line-cell connectivity in legacy flat format
 *   colorsFlat  Uint8Array     Direction-Encoded Color (DEC) per point
 *
 * WHY TWO PASSES?
 * ─────────────────
 * Dynamic array growth (push() / concat()) causes:
 *   · Multiple V8 heap allocations as the internal buffer doubles in size.
 *   · GC pressure on the main thread right before we need peak GPU bandwidth.
 *   · Up to 2× peak memory usage (old + new buffer coexist during realloc).
 *
 * Two passes eliminate all of this:
 *   Pass 1 — iterate over all streamlines, count totals, allocate exactly once.
 *   Pass 2 — iterate again, writing into the pre-allocated buffers.
 *
 * The cost of iterating the JS array twice is negligible compared to the GPU
 * buffer upload that follows — both passes are pure pointer arithmetic over a
 * contiguous array and stay in L1/L2 cache.
 *
 * VTK CELL ARRAY FORMAT
 * ──────────────────────
 * vtk.js uses the legacy flat (VTK-5) format for line cells:
 *
 *   [ n₀, id₀₀, id₀₁, …, id₀_{n₀-1},
 *     n₁, id₁₀, id₁₁, …, id₁_{n₁-1},
 *     … ]
 *
 * where nᵢ is the number of points in streamline i, and idᵢⱼ is the
 * 0-based global point index.  The total array length is:
 *
 *   Σᵢ (1 + nᵢ)    (one count field + nᵢ point IDs per streamline)
 *
 * DIRECTION-ENCODED COLOR (DEC) — HOT PATH
 * ──────────────────────────────────────────
 * DEC is computed inline (not via directionToRgb255 from tractographyColors.ts)
 * to avoid 250 000 cross-module function calls per typical tractogram.
 *
 * For each point i > 0:
 *   d = sl[i] - sl[i-1]          (backward difference → local segment direction)
 *   norm = ‖d‖₂  (guard: use 1.0 if zero)
 *   R = |dx/norm| × 255 | 0
 *   G = |dy/norm| × 255 | 0
 *   B = |dz/norm| × 255 | 0
 *
 * For point i = 0 (first point of each streamline):
 *   Use the forward difference d = sl[1] - sl[0] so the first point gets the
 *   same colour as the opening segment direction, not an arbitrary zero.
 *
 * MEMORY USAGE (typical 10 000 streamlines × 25 points)
 * ──────────────────────────────────────────────────────
 *   pointsFlat:  250 000 pts × 3 × 4 B  =  3.0 MB
 *   cellData:    250 000 IDs + 10 000 counts × 4 B  ≈  1.04 MB
 *   colorsFlat:  250 000 pts × 3 × 1 B  =  0.75 MB
 *   ─────────────────────────────────────────────────
 *   Total GPU upload: ≈ 4.8 MB  — well within the 10–15 MB target
 */

import type {
  StreamlineArray,
  TractographyBuffers,
  TractographyGeometryStats,
} from './tractographyTypes';

/**
 * Convert a nested streamline array into three pre-allocated GPU buffers.
 *
 * Streamlines with fewer than 2 points are silently skipped — they cannot
 * form a VTK line cell (which requires at least 2 endpoints).
 *
 * @param streamlines  Raw streamlines from POST /api/dti/tractography.
 * @returns            Three flat buffers + geometry statistics.
 */
export function buildTractographyBuffers(
  streamlines: StreamlineArray,
): TractographyBuffers {

  // ── Pass 1: count totals ──────────────────────────────────────────────────
  //
  // totalPoints      = Σᵢ nᵢ          (number of 3-D points across all lines)
  // totalCellEntries = Σᵢ (1 + nᵢ)   (count field + point IDs per cell)
  //
  let totalPoints      = 0;
  let totalCellEntries = 0;

  for (const sl of streamlines) {
    if (sl.length < 2) continue;    // skip degenerate single-point streamlines
    totalPoints      += sl.length;
    totalCellEntries += 1 + sl.length;  // 1 count slot + N point-ID slots
  }

  // ── Allocate once ──────────────────────────────────────────────────────────
  //
  // Float32Array:  4 B per component — 32-bit precision is sufficient for mm
  //                coordinates; vtk.js uploads Float32 to the GPU anyway.
  // Uint32Array:   4 B per entry — supports up to 4 billion unique point IDs,
  //                far more than any clinical tractogram.
  // Uint8Array:    1 B per channel — GPU expects 0–255 per RGB channel.
  //
  const pointsFlat = new Float32Array(totalPoints * 3);
  const cellData   = new Uint32Array(totalCellEntries);
  const colorsFlat = new Uint8Array(totalPoints * 3);

  // ── Pass 2: fill buffers ───────────────────────────────────────────────────
  //
  // ptIdx    : global point index (advances by 1 per point written)
  // cellIdx  : position in cellData  (advances by 1 + nᵢ per streamline)
  //
  let ptIdx              = 0;
  let cellIdx            = 0;
  let renderedStreamlines = 0;   // count of non-degenerate streamlines actually written

  for (const sl of streamlines) {
    if (sl.length < 2) continue;
    renderedStreamlines++;

    const nPts          = sl.length;
    const firstGlobalPt = ptIdx;    // global index of this streamline's first point

    // ── Cell connectivity ──────────────────────────────────────────────────
    // Write: [nPts, globalId₀, globalId₁, …, globalId_{nPts-1}]
    cellData[cellIdx++] = nPts;
    for (let i = 0; i < nPts; i++) {
      cellData[cellIdx++] = firstGlobalPt + i;
    }

    // ── Points + DEC colors ───────────────────────────────────────────────
    for (let i = 0; i < nPts; i++) {
      const pt    = sl[i];
      const flat3 = ptIdx * 3;   // byte offset into pointsFlat / colorsFlat

      // Write XYZ coordinates.
      pointsFlat[flat3]     = pt[0];
      pointsFlat[flat3 + 1] = pt[1];
      pointsFlat[flat3 + 2] = pt[2];

      // Compute DEC color — INLINED for hot-path performance.
      // Use forward difference for the first point (i = 0) so it matches the
      // opening segment colour; backward difference for all subsequent points.
      let dx: number;
      let dy: number;
      let dz: number;

      if (i === 0) {
        // Forward difference: direction from point 0 → point 1.
        dx = sl[1][0] - sl[0][0];
        dy = sl[1][1] - sl[0][1];
        dz = sl[1][2] - sl[0][2];
      } else {
        // Backward difference: direction from point i-1 → point i.
        const prev = sl[i - 1];
        dx = pt[0] - prev[0];
        dy = pt[1] - prev[1];
        dz = pt[2] - prev[2];
      }

      // Normalise.  Guard against zero-length segments (coincident points).
      const norm = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1.0;

      // |·| × 255, truncated to uint8 via bitwise-OR 0.
      colorsFlat[flat3]     = (Math.abs(dx / norm) * 255) | 0;  // R = |x̂|
      colorsFlat[flat3 + 1] = (Math.abs(dy / norm) * 255) | 0;  // G = |ŷ|
      colorsFlat[flat3 + 2] = (Math.abs(dz / norm) * 255) | 0;  // B = |ẑ|

      ptIdx++;
    }
  }

  // ── Statistics ────────────────────────────────────────────────────────────
  const stats: TractographyGeometryStats = {
    renderedStreamlines,
    totalPoints:      ptIdx,
    pointsByteLength: pointsFlat.byteLength,
    colorsByteLength: colorsFlat.byteLength,
  };

  return { pointsFlat, cellData, colorsFlat, stats };
}
