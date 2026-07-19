/**
 * connectomeEdgeHighlight.ts — Per-edge RGBA color array for the connectome highlight mode
 * ──────────────────────────────────────────────────────────────────────────────────────────
 *
 * PURPOSE
 * ────────
 * When the user clicks a cell (i, j) in the 2D ConnectomeHeatmap, the 3D viewer
 * must instantly isolate that specific fiber tract by:
 *
 *   • Selected edge   — accent-green (#00e676), full opacity (alpha = 255)
 *   • All other edges — dim grey, near-transparent (alpha = GHOST_ALPHA ≈ 5%)
 *
 * This file is responsible only for building the per-cell RGBA Uint8Array that
 * the vtk.js mapper ingests when switched to `setColorModeToDirectScalars()`.
 * It is a pure function with no side effects — the vtk.js plumbing lives in
 * connectomeEdges.ts.
 *
 * DATA MODEL
 * ───────────
 * `edgePairs` is an ordered list of `[pointIndexA, pointIndexB]` pairs.
 * The index corresponds to the global point index used by the vtkCellArray
 * (i.e., the row/column index in `nodeIds`).  Entry `e` in `edgePairs`
 * matches cell `e` in the vtkCellArray (cells stored as [2, idxA, idxB]).
 *
 * `sourceIdx` and `targetIdx` are the same global point indices — passed
 * directly from `selectedEdge.source` / `selectedEdge.target` in React state.
 *
 * MODE SWITCHING IN connectomeEdges.ts
 * ──────────────────────────────────────
 *
 *   NORMAL MODE (no selection):
 *     • polyData cell scalars = Float32 FibreCount array
 *     • mapper: setColorModeToMapScalars() + CTF
 *     • actor opacity = edgeOpacity (default 0.70)
 *
 *   HIGHLIGHT MODE (cell selected):
 *     • polyData cell scalars = Uint8 RGBA array (built here)
 *     • mapper: setColorModeToDirectScalars()
 *     • actor opacity = 1.0   (per-cell alpha drives transparency)
 *
 * The Float32 FibreCount array is ALWAYS kept up-to-date by filterByWeight()
 * so that reverting to normal mode is a simple setScalars() swap.
 *
 * CSS PALETTE REFERENCE
 * ──────────────────────
 *   #00e676 = var(--accent-green)  → selected edge
 *   #4d4d4d = dim grey             → ghosted edges
 *   GHOST_ALPHA / 255 ≈ 5%         → near-transparent ghost
 */

// ── Visual constants ──────────────────────────────────────────────────────────

/** RGBA for the single selected edge — accent-green, fully opaque. */
export const HIGHLIGHT_R =   0;
export const HIGHLIGHT_G = 230;
export const HIGHLIGHT_B = 118;
export const HIGHLIGHT_A = 255;   // fully opaque

/** RGBA for all ghosted (non-selected) edges — dim grey, near-transparent. */
export const GHOST_R =  60;
export const GHOST_G =  60;
export const GHOST_B =  60;
export const GHOST_A =  13;   // ~5% opacity — visible as a faint skeleton

// ── buildHighlightRgba ────────────────────────────────────────────────────────

/**
 * Build a per-cell RGBA Uint8Array for vtk.js direct-scalar coloring.
 *
 * For the edge matching [sourceIdx, targetIdx] (or [targetIdx, sourceIdx]):
 *   → accent-green, fully opaque
 * For all other edges:
 *   → dim grey, near-transparent (ghosted)
 *
 * @param edgePairs  Ordered list of [pointIndexA, pointIndexB] pairs, one per
 *                   cell in the vtkCellArray.  Built and maintained by
 *                   buildConnectomeEdges / filterByWeight in connectomeEdges.ts.
 * @param sourceIdx  Global point index of the selected source region (= matrix row).
 * @param targetIdx  Global point index of the selected target region (= matrix col).
 * @returns          Uint8Array of length edgePairs.length × 4.
 *                   Layout: [R,G,B,A, R,G,B,A, …], one quad per edge.
 */
export function buildHighlightRgba(
  edgePairs: ReadonlyArray<readonly [number, number]>,
  sourceIdx: number,
  targetIdx: number,
): Uint8Array {
  const n    = edgePairs.length;
  const rgba = new Uint8Array(n * 4);

  for (let e = 0; e < n; e++) {
    const [idxA, idxB] = edgePairs[e]!;

    // Edge is undirected — match in either direction.
    const isSelected =
      (idxA === sourceIdx && idxB === targetIdx) ||
      (idxA === targetIdx && idxB === sourceIdx);

    const base = e * 4;
    if (isSelected) {
      rgba[base]     = HIGHLIGHT_R;
      rgba[base + 1] = HIGHLIGHT_G;
      rgba[base + 2] = HIGHLIGHT_B;
      rgba[base + 3] = HIGHLIGHT_A;
    } else {
      rgba[base]     = GHOST_R;
      rgba[base + 1] = GHOST_G;
      rgba[base + 2] = GHOST_B;
      rgba[base + 3] = GHOST_A;
    }
  }

  return rgba;
}
