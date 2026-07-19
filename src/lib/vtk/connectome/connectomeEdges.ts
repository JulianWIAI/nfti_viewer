/**
 * connectomeEdges.ts — 3-D line overlay for connectome edges
 * ────────────────────────────────────────────────────────────
 *
 * PIPELINE OVERVIEW
 * ──────────────────
 *
 *   matrix (N×N) + nodeIds + nodes
 *         │
 *         ▼  Pass 1: count suprathreshold edges
 *   nEdges (scalar), maxWeight (scalar)
 *         │
 *         ▼  Allocate exactly once:
 *   coordsFlat  Float32Array(N × 3)          — one position per region node
 *   cellData    Uint32Array(nEdges × 3)       — [2, idxA, idxB] per line
 *   weightArr   Float32Array(nEdges)          — fibre count per line cell
 *   edgePairs   [number,number][]             — [idxA, idxB] per line (used by highlight)
 *         │
 *         ▼  Pass 2: fill all four buffers
 *         │
 *         ▼
 *   vtkDataArray (pts)          3-component Float32, node positions
 *   vtkCellArray (lines)        linesArray.setData(cellData)
 *   vtkDataArray (cellScalars)  1-component Float32, fibre counts per cell
 *         │
 *         ▼
 *   vtkPolyData
 *     ├─ setPoints(pts)
 *     ├─ setLines(lines)
 *     └─ getCellData().setScalars(cellScalars)   per-cell fibre count
 *         │
 *         ▼
 *   vtkMapper
 *     ├─ setInputData(polyData)
 *     ├─ setColorModeToMapScalars()         drive color through the CTF (NORMAL mode)
 *     ├─ setScalarModeToUseCellData()       one color per line (not per point)
 *     ├─ setScalarVisibility(true)
 *     └─ setLookupTable(edgeCTF)            pale-blue → cobalt-blue ramp
 *         │
 *         ▼
 *   vtkActor  (opacity = edgeOpacity, lineWidth = edgeLineWidth, lighting OFF)
 *
 * HIGHLIGHT MODE
 * ───────────────
 * When highlightEdge(sourceIdx, targetIdx) is called with non-null values,
 * the mapper switches from CTF-mapped float scalars to per-cell RGBA Uint8
 * scalars (direct scalar mode).  This isolates one tract visually:
 *
 *   Selected edge   → accent-green (#00e676), alpha 255 (fully opaque)
 *   All other edges → dim grey,    alpha 13   (~5% opacity, ghosted)
 *
 * The Float32 cellScalars array is always kept current by filterByWeight()
 * even while in highlight mode, so reverting to normal is a single swap.
 *
 * TWO-PASS BUFFER BUILDING
 * ─────────────────────────
 * Pass 1 counts active edges → allocates typed arrays exactly sized.
 * Pass 2 fills arrays in a single forward sweep.
 */

import vtkActor                  from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper                 from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkPolyData               from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkCellArray              from '@kitware/vtk.js/Common/Core/CellArray';
import vtkDataArray              from '@kitware/vtk.js/Common/Core/DataArray';
import vtkColorTransferFunction  from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';

import type {
  ConnectomeNode,
  ConnectomeOptions,
  ConnectomeEdgeBundle,
} from './connectomeTypes';

import { buildHighlightRgba } from './connectomeEdgeHighlight';

/**
 * Build a vtk.js vtkPolyData → vtkMapper → vtkActor pipeline that renders
 * suprathreshold connectome edges as weight-colored lines in RAS mm space.
 *
 * @param matrix   N×N fibre-count connectivity matrix (A[i][j] = fibre count).
 * @param nodeIds  Ordered label strings — nodeIds[i] indexes matrix row/col i.
 * @param nodes    Node properties dictionary (center_of_mass for positions).
 * @param options  Optional visual tuning (threshold, opacity, line width).
 * @returns        ConnectomeEdgeBundle with actor + lifecycle helpers.
 */
export function buildConnectomeEdges(
  matrix:  number[][],
  nodeIds: string[],
  nodes:   Record<string, ConnectomeNode>,
  options: ConnectomeOptions = {},
): ConnectomeEdgeBundle {

  // ── Configuration defaults ────────────────────────────────────────────────
  const threshold = options.edgeThreshold ?? 10;
  const opacity   = options.edgeOpacity   ?? 0.70;
  const lineWidth = options.edgeLineWidth ?? 1.5;

  const n = nodeIds.length;

  // ── Pass 1: count active edges and find maximum weight ────────────────────
  let nEdges    = 0;
  let maxWeight = threshold;

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const w = Math.max(matrix[i]?.[j] ?? 0, matrix[j]?.[i] ?? 0);
      if (w > threshold) {
        nEdges++;
        if (w > maxWeight) maxWeight = w;
      }
    }
  }

  // ── Allocate buffers exactly once ─────────────────────────────────────────
  const coordsFlat = new Float32Array(n * 3);
  const cellData   = new Uint32Array(nEdges * 3);
  const weightArr  = new Float32Array(nEdges);

  // ── Mutable closure state for highlight mode ─────────────────────────────
  //
  // edgePairs[e] = [pointIndexA, pointIndexB] for edge cell e.
  // Rebuilt alongside cellData/weightArr whenever filterByWeight() is called.
  // Used by _applyHighlight() to build the per-cell RGBA array.
  //
  let edgePairs: Array<[number, number]> = [];

  // The currently active selection, or null in normal mode.
  // Stored so that filterByWeight() can re-apply the highlight after rebuilding.
  let currentHighlight: { sourceIdx: number; targetIdx: number } | null = null;

  // ── Fill node position buffer ─────────────────────────────────────────────
  for (let i = 0; i < n; i++) {
    const node = nodes[nodeIds[i]!];
    if (!node) {
      coordsFlat[i * 3] = coordsFlat[i * 3 + 1] = coordsFlat[i * 3 + 2] = 0;
      continue;
    }
    coordsFlat[i * 3    ] = node.center_of_mass[0];
    coordsFlat[i * 3 + 1] = node.center_of_mass[1];
    coordsFlat[i * 3 + 2] = node.center_of_mass[2];
  }

  // ── Pass 2: fill cell connectivity, weight, and edgePairs arrays ──────────
  let edgeIdx = 0;
  let cellIdx = 0;

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const w = Math.max(matrix[i]?.[j] ?? 0, matrix[j]?.[i] ?? 0);
      if (w <= threshold) continue;

      cellData[cellIdx    ] = 2;   // VTK: 2 points per line
      cellData[cellIdx + 1] = i;
      cellData[cellIdx + 2] = j;
      cellIdx += 3;

      weightArr[edgeIdx] = w;
      edgePairs[edgeIdx] = [i, j];
      edgeIdx++;
    }
  }

  // ── vtkDataArray: node positions ──────────────────────────────────────────
  const pts = vtkDataArray.newInstance({
    numberOfComponents: 3,
    values: coordsFlat,
  });

  // ── vtkCellArray: line connectivity ──────────────────────────────────────
  const linesArray = vtkCellArray.newInstance();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (linesArray as any).setData(cellData);

  // ── vtkDataArray: per-cell fibre count (Float32) — the "normal mode" scalars
  //
  // This array is ALWAYS kept current by filterByWeight(), even while the mapper
  // is in highlight mode (direct-scalar RGBA).  When highlight mode exits, this
  // array is re-installed as the active scalar without any extra rebuild.
  //
  const cellScalars = vtkDataArray.newInstance({
    name:               'FibreCount',
    numberOfComponents: 1,
    values:             weightArr,
  });

  // ── vtkPolyData ───────────────────────────────────────────────────────────
  const polyData = vtkPolyData.newInstance();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (polyData as any).setPoints(pts);
  polyData.setLines(linesArray);
  polyData.getCellData().setScalars(cellScalars);

  // ── Edge colour transfer function (cool-blue sequential ramp) ─────────────
  const edgeCTF = vtkColorTransferFunction.newInstance();
  const mid     = threshold + (maxWeight - threshold) * 0.5;
  edgeCTF.addRGBPoint(threshold, 0.80, 0.88, 0.96);   // pale sky-blue
  edgeCTF.addRGBPoint(mid,       0.35, 0.55, 0.85);   // medium blue
  edgeCTF.addRGBPoint(maxWeight, 0.10, 0.28, 0.70);   // deep cobalt

  // ── vtkMapper (starts in normal mode) ────────────────────────────────────
  const mapper = vtkMapper.newInstance();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (mapper as any).setInputData(polyData);

  mapper.setColorModeToMapScalars();
  mapper.setScalarModeToUseCellData();
  mapper.setScalarVisibility(true);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (mapper as any).setLookupTable(edgeCTF);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (mapper as any).setUseLookupTableScalarRange(true);

  // ── vtkActor ──────────────────────────────────────────────────────────────
  const actor = vtkActor.newInstance();
  actor.setMapper(mapper as unknown as Parameters<typeof actor.setMapper>[0]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const prop = (actor as any).getProperty();
  prop.setOpacity(opacity);
  prop.setLineWidth(lineWidth);
  prop.setLighting(false);

  // ── Internal mode helpers ──────────────────────────────────────────────────
  //
  // _applyHighlight: builds per-cell RGBA array and switches mapper to
  //   direct-scalar mode so vtk.js uses the raw RGBA values as colors.
  //   Actor opacity is set to 1.0 so per-cell alpha is the sole transparency
  //   driver (if actor opacity were 0.7, the selected edge would only render
  //   at 70% even though its cell alpha is 255).
  //
  // _restoreNormal: reinstalls the Float32 FibreCount scalar and returns the
  //   mapper to map-scalars (CTF) mode with the original actor opacity.
  //
  // Both are called only by highlightEdge() and filterByWeight().
  //

  function _applyHighlight(sourceIdx: number, targetIdx: number): void {
    // Build per-cell RGBA array from edgePairs and the selected indices.
    const rgba = buildHighlightRgba(edgePairs, sourceIdx, targetIdx);

    // Wrap the Uint8Array in a vtkDataArray (4 components = RGBA per cell).
    const rgbaArray = vtkDataArray.newInstance({
      name:               'EdgeHighlight',
      numberOfComponents: 4,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dataType:           'Uint8Array' as any,
      values:             rgba,
    });

    // Swap the active cell scalar to the RGBA array.
    polyData.getCellData().setScalars(rgbaArray);

    // Switch mapper to direct-scalar mode: uses R,G,B,A values directly
    // without routing through the color transfer function.
    mapper.setColorModeToDirectScalars();

    // Actor opacity must be 1.0 — the per-cell alpha drives transparency.
    // If actor opacity were < 1, the selected edge would not be fully opaque.
    prop.setOpacity(1.0);

    // Notify vtk.js that the data changed so the mapper re-uploads to GPU.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (polyData as any).modified();
  }

  function _restoreNormal(): void {
    // Reinstall the Float32 FibreCount array (always current; kept up-to-date
    // by filterByWeight even while in highlight mode).
    polyData.getCellData().setScalars(cellScalars);

    // Return the mapper to CTF-mapped mode.
    mapper.setColorModeToMapScalars();

    // Reconnect the CTF (direct-scalar mode ignores it; reattach to be safe).
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (mapper as any).setLookupTable(edgeCTF);

    // Restore the original actor opacity (not per-cell alpha).
    prop.setOpacity(opacity);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (polyData as any).modified();
  }

  // ── Lifecycle helpers ──────────────────────────────────────────────────────

  function setVisible(visible: boolean): void {
    actor.setVisibility(visible);
  }

  function dispose(): void {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (actor    as any).delete();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (mapper   as any).delete();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (polyData as any).delete();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (edgeCTF  as any).delete();
  }

  // ── In-place edge weight filter ────────────────────────────────────────────
  //
  // Rebuilds the cell connectivity and per-cell scalars without recreating any
  // vtk.js objects.  Also rebuilds edgePairs so highlightEdge() stays correct
  // after the threshold changes.
  //
  // If a highlight was active when this is called, it is re-applied after the
  // rebuild so the user sees the same tract highlighted with the new edge set.
  //
  function filterByWeight(newThreshold: number): void {
    // ── Pass 1: count survivors ──────────────────────────────────────────────
    let nE  = 0;
    let maxW = newThreshold;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const w = Math.max(matrix[i]?.[j] ?? 0, matrix[j]?.[i] ?? 0);
        if (w > newThreshold) {
          nE++;
          if (w > maxW) maxW = w;
        }
      }
    }

    // ── Allocate new buffers ─────────────────────────────────────────────────
    const newCellData  = new Uint32Array(nE * 3);
    const newWeightArr = new Float32Array(nE);
    const newEdgePairs: Array<[number, number]> = [];

    // ── Pass 2: fill ─────────────────────────────────────────────────────────
    let eIdx = 0;
    let cIdx = 0;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const w = Math.max(matrix[i]?.[j] ?? 0, matrix[j]?.[i] ?? 0);
        if (w <= newThreshold) continue;
        newCellData[cIdx++] = 2;
        newCellData[cIdx++] = i;
        newCellData[cIdx++] = j;
        newWeightArr[eIdx]   = w;
        newEdgePairs[eIdx]   = [i, j];
        eIdx++;
      }
    }

    // ── Swap backing data (no vtk object rebuild) ────────────────────────────
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (linesArray  as any).setData(newCellData);
    // Always update cellScalars (the Float32 weight data) — this keeps the
    // shadow scalar current for when _restoreNormal() swaps it back in.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (cellScalars as any).setData(newWeightArr);

    // ── Update edgePairs (used by _applyHighlight) ───────────────────────────
    edgePairs = newEdgePairs;

    // ── Rebuild CTF for new [newThreshold, maxW] range ───────────────────────
    edgeCTF.removeAllPoints();
    const midW = newThreshold + (maxW - newThreshold) * 0.5;
    edgeCTF.addRGBPoint(newThreshold, 0.80, 0.88, 0.96);
    edgeCTF.addRGBPoint(midW,         0.35, 0.55, 0.85);
    edgeCTF.addRGBPoint(maxW,         0.10, 0.28, 0.70);

    // ── Re-apply highlight or restore normal ──────────────────────────────────
    // If a highlight was active before this call, re-apply it with the fresh
    // edgePairs so the selected edge remains highlighted in the new edge set.
    // (If the selected edge no longer passes the new threshold, it simply won't
    //  appear in edgePairs and all edges will be ghosted — correct behavior.)
    if (currentHighlight !== null) {
      _applyHighlight(currentHighlight.sourceIdx, currentHighlight.targetIdx);
    } else {
      // Normal mode: reinstall FibreCount scalar so CTF drives colors.
      _restoreNormal();
    }
  }

  // ── Highlight / de-highlight a specific edge ──────────────────────────────
  //
  // ENTER HIGHLIGHT MODE: sourceIdx and targetIdx are both non-null.
  //   • Switch mapper to direct-scalar RGBA.
  //   • Selected edge → accent-green at full opacity.
  //   • All other edges → dim grey at ~5% opacity.
  //
  // EXIT HIGHLIGHT MODE: either argument is null.
  //   • Reinstall Float32 FibreCount scalar.
  //   • Switch mapper back to CTF-mapped colors.
  //   • Restore original actor opacity.
  //
  function highlightEdge(sourceIdx: number | null, targetIdx: number | null): void {
    if (sourceIdx !== null && targetIdx !== null) {
      // Enter (or update) highlight mode.
      currentHighlight = { sourceIdx, targetIdx };
      _applyHighlight(sourceIdx, targetIdx);
    } else {
      // Exit highlight mode — return to normal CTF rendering.
      currentHighlight = null;
      _restoreNormal();
    }
  }

  return { actor, setVisible, dispose, filterByWeight, highlightEdge };
}
