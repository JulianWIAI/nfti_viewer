/**
 * ConnectomeHeatmap.tsx — High-performance Canvas 2D heatmap of the N×N adjacency matrix
 * ──────────────────────────────────────────────────────────────────────────────────────────
 *
 * CROSS-FILTER INTEGRATION
 * ─────────────────────────
 * When the user clicks a cell (i, j):
 *   1. `onCellSelect(i, j, fiberCount)` is called — the parent stores
 *      `selectedEdge = { source: i, target: j }` in VolumetricContext.
 *   2. ConnectomePanel reads `selectedEdge` and calls
 *      `bundle.edgeBundle.highlightEdge(i, j)` — the 3D VTK viewer isolates
 *      that fiber tract (accent-green) and ghosts all others.
 *   3. The heatmap draws a persistent white border on cells (i,j) and (j,i)
 *      via `drawSelection()`, reflecting the undirected edge.
 *
 * Clicking the same cell again OR clicking outside the matrix calls
 * `onCellSelect(null, null, null)` — both the heatmap selection and the
 * 3D highlight are cleared simultaneously.
 *
 * STATE / PROP CONTRACT
 * ──────────────────────
 *   Controlled selection: the parent passes `activeRow` / `activeCol` (the
 *   same i, j it stores in selectedEdge).  The heatmap uses these only for
 *   drawing — it does NOT own the selection state.  Toggle detection is done
 *   by comparing the clicked cell against `activeRow`/`activeCol`.
 *
 * DRAW LIFECYCLE
 * ───────────────
 * All canvas-influencing state (hover, activeRow, activeCol) is mirrored into
 * refs so that:
 *   • draw() has a stable identity (no recreations on every hover tick).
 *   • scheduleRedraw() is stable, avoiding effect dependency cascades.
 *   • Refs are always updated synchronously before any RAF fires.
 *
 * CSS CLASSES  (defined in App.css under "── ConnectomeHeatmap ──")
 * ──────────────────────────────────────────────────────────────────
 *   .connectome-heatmap            outer wrapper (position: relative)
 *   .connectome-heatmap__canvas    the <canvas> element
 *   .heatmap-tooltip               floating tooltip div
 *   .heatmap-tooltip__pair         source ↔ target name line
 *   .heatmap-tooltip__value        fiber count line
 */

import {
  useRef,
  useState,
  useEffect,
  useCallback,
  useMemo,
  type JSX,
} from 'react';
import { computeLayout }       from '../lib/heatmap/heatmapLayout';
import { buildLabelInfoArray } from '../lib/heatmap/heatmapLabels';
import {
  drawBackground,
  drawCells,
  drawGridLines,
  drawGroupBlocks,
  drawAxisLabels,
  drawColorLegend,
  drawSelection,
  drawHighlight,
  hitTest,
} from '../lib/heatmap/heatmapRenderer';
import type { HeatmapLayout }    from '../lib/heatmap/heatmapTypes';
import type { HeatmapLabelInfo } from '../lib/heatmap/heatmapTypes';
import { BRAIN_LABELS }          from '../lib/vtk/labelVisibility';
import HeatmapTooltip            from './HeatmapTooltip';

// ── Module-level BRAIN_LABELS lookup ─────────────────────────────────────────
const ID_TO_NAME = new Map(BRAIN_LABELS.map((l) => [String(l.id), l.name]));

// ── Props ─────────────────────────────────────────────────────────────────────

export interface ConnectomeHeatmapProps {
  /** N×N symmetric fiber-count matrix from POST /api/connectomics/matrix. */
  matrix:       number[][];

  /**
   * Ordered list of FreeSurfer label ID strings matching the matrix rows/cols.
   * nodeIds[i] is the label for row i and column i.
   */
  nodeIds:      string[];

  /**
   * Called when the user clicks a non-diagonal cell — or when the same cell
   * is clicked again (toggle off) or when empty/diagonal space is clicked.
   *
   * @param sourceIdx  Row index of the clicked cell, or null to deselect.
   * @param targetIdx  Column index of the clicked cell, or null to deselect.
   * @param fiberCount max(A[i][j], A[j][i]) for the clicked cell, or null.
   */
  onCellSelect?: (
    sourceIdx:  number | null,
    targetIdx:  number | null,
    fiberCount: number | null,
  ) => void;

  /**
   * The currently active row selection (= selectedEdge.source from context).
   * When non-null, a persistent white border is drawn on this row index.
   * The heatmap does NOT own this state — it is controlled by the parent.
   */
  activeRow?: number | null;

  /**
   * The currently active column selection (= selectedEdge.target from context).
   * Paired with activeRow; together they identify the selected cell.
   */
  activeCol?: number | null;
}

// ── Hover state (internal to the canvas) ─────────────────────────────────────

interface HoverState {
  row:  number;
  col:  number;
  cssX: number;
  cssY: number;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function ConnectomeHeatmap({
  matrix,
  nodeIds,
  onCellSelect,
  activeRow,
  activeCol,
}: ConnectomeHeatmapProps): JSX.Element {

  // ── DOM refs ─────────────────────────────────────────────────────────────────
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // ── Imperative canvas state ───────────────────────────────────────────────────
  // Stored in refs so draw() doesn't need to re-close over changing prop/state
  // values — it always reads the LATEST value without recreating the closure.
  const layoutRef    = useRef<HeatmapLayout | null>(null);
  const rafRef       = useRef<number | null>(null);
  const hoverRef     = useRef<HoverState | null>(null);   // mirrors 'hover' state
  const activeRowRef = useRef<number>(-1);                // mirrors 'activeRow' prop
  const activeColRef = useRef<number>(-1);                // mirrors 'activeCol' prop

  // Sync refs on every render (before any RAF fires).
  hoverRef.current    = null;   // reset; updated by mouse handlers below
  activeRowRef.current = activeRow ?? -1;
  activeColRef.current = activeCol ?? -1;

  // ── Derived data (memoised) ───────────────────────────────────────────────────
  const labels: HeatmapLabelInfo[] = useMemo(
    () => buildLabelInfoArray(nodeIds),
    [nodeIds],
  );

  const maxVal: number = useMemo(() => {
    let max = 0;
    for (const row of matrix) for (const v of row) if (v > max) max = v;
    return Math.max(max, 1);
  }, [matrix]);

  // ── React hover state — drives tooltip DOM position + content only ────────────
  // Canvas highlight rendering uses hoverRef (synced synchronously in handlers).
  const [hover, setHover] = useState<HoverState | null>(null);

  // Keep hoverRef in sync with the React state on every render.
  // Note: hoverRef is also set synchronously in mouse handlers BEFORE setHover
  // so that a scheduled RAF always reads the latest value even if React hasn't
  // re-rendered yet.
  hoverRef.current = hover;

  // ── Core draw function ────────────────────────────────────────────────────────
  //
  // draw() reads hover and selection from refs so it has a stable identity
  // (no closure over changing props/state that would force recreation and
  // cascade through scheduleRedraw → effect dependencies).
  //
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const layout = layoutRef.current;
    if (!canvas || !layout) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // ── Layer 1: dark background ─────────────────────────────────────────────
    drawBackground(ctx, layout);

    // ── Layer 2: matrix cells (single ImageData GPU upload) ──────────────────
    drawCells(ctx, layout, matrix, maxVal);

    // ── Layer 3: subtle grid lines ───────────────────────────────────────────
    drawGridLines(ctx, layout);

    // ── Layer 4: tissue-class color strips on both axes ──────────────────────
    drawGroupBlocks(ctx, layout, labels);

    // ── Layer 5: rotated X and horizontal Y axis labels ───────────────────────
    drawAxisLabels(ctx, layout, labels);

    // ── Layer 6: gradient color legend (right side) ───────────────────────────
    drawColorLegend(ctx, layout, maxVal);

    // ── Layer 7: persistent selection borders ─────────────────────────────────
    //
    // Draw a white border on both (activeRow, activeCol) AND (activeCol, activeRow)
    // to reflect the undirected nature of the fiber tract.
    // Drawn BEFORE the hover highlight so hover (green) renders on top.
    //
    const ar = activeRowRef.current;
    const ac = activeColRef.current;
    if (ar >= 0 && ac >= 0) {
      drawSelection(ctx, layout, ar, ac);
      if (ar !== ac) drawSelection(ctx, layout, ac, ar);   // symmetric cell
    }

    // ── Layer 8: hover highlight (accent-green border) ────────────────────────
    const h = hoverRef.current;
    if (h !== null) {
      drawHighlight(ctx, layout, h.row, h.col);
    }
  }, [matrix, maxVal, labels]);   // refs are stable — not in deps

  // ── RAF-scheduled redraw ──────────────────────────────────────────────────────

  const scheduleRedraw = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      draw();
      rafRef.current = null;
    });
  }, [draw]);

  // ── Layout computation + canvas sizing ───────────────────────────────────────

  const recomputeLayout = useCallback(() => {
    const canvas    = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const { width: cssW, height: cssH } = container.getBoundingClientRect();
    const dpr    = window.devicePixelRatio ?? 1;
    const n      = nodeIds.length;
    const layout = computeLayout(cssW, cssH, n, dpr);

    canvas.width  = layout.canvasW;
    canvas.height = layout.canvasH;

    layoutRef.current = layout;
  }, [nodeIds.length]);

  // ── ResizeObserver — initial draw + resize redraws ────────────────────────────

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    recomputeLayout();
    scheduleRedraw();

    const observer = new ResizeObserver(() => {
      recomputeLayout();
      hoverRef.current = null;
      setHover(null);
      scheduleRedraw();
    });
    observer.observe(container);

    return () => {
      observer.disconnect();
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [recomputeLayout, scheduleRedraw]);

  // ── Redraw when matrix / nodeIds change ──────────────────────────────────────

  useEffect(() => {
    recomputeLayout();
    hoverRef.current = null;
    setHover(null);
    scheduleRedraw();
  }, [matrix, nodeIds, recomputeLayout, scheduleRedraw]);

  // ── Redraw when the active selection changes (cross-filter update) ────────────
  //
  // When the parent updates selectedEdge (e.g., user clicked a different cell
  // or de-selected), we need to redraw the canvas so the persistent white border
  // appears/disappears/moves immediately.
  //
  // activeRowRef / activeColRef are already synced at the top of this render,
  // so the RAF callback reads the correct updated values.
  //
  useEffect(() => {
    scheduleRedraw();
    // Intentionally omit 'scheduleRedraw' from deps — it changes when draw
    // changes (matrix/labels/maxVal), which already has its own effect above.
    // This effect should only fire on selection changes to avoid double redraws.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRow, activeCol]);

  // ── Mouse handlers ────────────────────────────────────────────────────────────

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const layout = layoutRef.current;
    if (!layout) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const cssX  = e.clientX - rect.left;
    const cssY  = e.clientY - rect.top;

    const hit = hitTest(cssX, cssY, layout);

    if (!hit) {
      if (hoverRef.current !== null) {
        hoverRef.current = null;
        setHover(null);
        scheduleRedraw();
      }
      return;
    }

    const newHover: HoverState = { row: hit.row, col: hit.col, cssX, cssY };

    if (!hoverRef.current || hoverRef.current.row !== hit.row || hoverRef.current.col !== hit.col) {
      // Cell changed → full redraw for new highlight position.
      hoverRef.current = newHover;
      setHover(newHover);
      scheduleRedraw();
    } else if (hoverRef.current.cssX !== cssX || hoverRef.current.cssY !== cssY) {
      // Same cell, cursor moved → only tooltip position needs updating (no redraw).
      hoverRef.current = newHover;
      setHover(newHover);
    }
  }, [scheduleRedraw]);

  const handleMouseLeave = useCallback(() => {
    if (hoverRef.current !== null) {
      hoverRef.current = null;
      setHover(null);
      scheduleRedraw();
    }
  }, [scheduleRedraw]);

  // ── Click handler — select / deselect edge ────────────────────────────────────
  //
  // TOGGLE LOGIC
  // ─────────────
  // Click (i, j) where (i, j) ≠ current selection  → select: call onCellSelect(i, j, count)
  // Click (i, j) where (i, j) = current selection  → deselect: call onCellSelect(null, null, null)
  // Click diagonal (i, i)                           → deselect (no self-connections)
  // Click outside plot area                         → deselect
  //
  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const layout = layoutRef.current;
    if (!layout) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const hit  = hitTest(e.clientX - rect.left, e.clientY - rect.top, layout);

    // Clicking outside the plot or on the diagonal → deselect.
    if (!hit || hit.row === hit.col) {
      onCellSelect?.(null, null, null);
      return;
    }

    // Toggle: clicking the already-selected cell deselects it.
    if (activeRowRef.current === hit.row && activeColRef.current === hit.col) {
      onCellSelect?.(null, null, null);
      return;
    }

    // New cell selected — compute fiber count (symmetric matrix: take max).
    const fiberCount = Math.max(
      matrix[hit.row]?.[hit.col] ?? 0,
      matrix[hit.col]?.[hit.row] ?? 0,
    );
    onCellSelect?.(hit.row, hit.col, fiberCount);
  }, [matrix, onCellSelect]);

  // ── Tooltip data ──────────────────────────────────────────────────────────────

  const tooltipData = hover !== null && layoutRef.current
    ? {
        sourceName: ID_TO_NAME.get(nodeIds[hover.row] ?? '') ?? `Label ${nodeIds[hover.row] ?? '?'}`,
        targetName: ID_TO_NAME.get(nodeIds[hover.col] ?? '') ?? `Label ${nodeIds[hover.col] ?? '?'}`,
        fiberCount: hover.row === hover.col
          ? null
          : Math.max(
              matrix[hover.row]?.[hover.col] ?? 0,
              matrix[hover.col]?.[hover.row] ?? 0,
            ),
      }
    : null;

  const containerCssW = containerRef.current?.getBoundingClientRect().width ?? 600;

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div
      ref={containerRef}
      className="connectome-heatmap"
      role="img"
      aria-label="Structural connectivity matrix heatmap"
    >
      <canvas
        ref={canvasRef}
        className="connectome-heatmap__canvas"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        style={{ cursor: 'crosshair' }}
      />

      {hover !== null && tooltipData !== null && (
        <HeatmapTooltip
          cssX={hover.cssX}
          cssY={hover.cssY}
          containerW={containerCssW}
          sourceName={tooltipData.sourceName}
          targetName={tooltipData.targetName}
          fiberCount={tooltipData.fiberCount}
        />
      )}
    </div>
  );
}
