/**
 * heatmapTypes.ts — Shared type definitions for the ConnectomeHeatmap Canvas 2D renderer
 * ─────────────────────────────────────────────────────────────────────────────────────────
 *
 * Three data structures cover the complete rendering contract:
 *
 *   HeatmapLayout   — all pixel measurements computed from container + DPR once per resize
 *   HeatmapCellHit  — the result of converting a mouse event to a matrix cell (i, j)
 *   HeatmapLabelInfo — per-region metadata consumed by the axis label / group-block drawers
 */

// ── Layout ────────────────────────────────────────────────────────────────────

/**
 * All measurements (in physical canvas pixels = CSS pixels × devicePixelRatio)
 * needed by every draw function.  Computed once by computeLayout() and passed
 * to all sub-drawers — never recomputed inside a draw call.
 *
 * COORDINATE SYSTEM
 * ─────────────────
 * Canvas 2D uses (0,0) at the top-left corner.
 * X increases rightward, Y increases downward.
 *
 * The heatmap plot area begins at (plotX, plotY) and spans plotSide × plotSide
 * physical pixels.  Row i → y = plotY + i * cellH, column j → x = plotX + j * cellW.
 */
export interface HeatmapLayout {
  /** Canvas width in physical pixels (CSS width × dpr). */
  canvasW:    number;
  /** Canvas height in physical pixels (CSS height × dpr). */
  canvasH:    number;
  /** Device pixel ratio at render time. */
  dpr:        number;

  /** Physical-pixel X of the left edge of the plot area. */
  plotX:      number;
  /** Physical-pixel Y of the top edge of the plot area. */
  plotY:      number;
  /** Side length of the square plot area in physical pixels. */
  plotSide:   number;

  /** Physical-pixel width of one matrix cell (= plotSide / n). */
  cellW:      number;
  /** Physical-pixel height of one matrix cell (= plotSide / n). */
  cellH:      number;

  /** Number of matrix rows/columns (= node_ids.length). */
  n:          number;

  // ── Margin measurements (physical pixels) ─────────────────────────────────

  /** Left margin: rotated label area + group-block strip + gap. */
  marginLeft:   number;
  /** Top margin: rotated label area + group-block strip + gap. */
  marginTop:    number;
  /** Right margin: legend gap + bar + value text + padding. */
  marginRight:  number;
  /** Bottom margin: small breathing room below the matrix. */
  marginBottom: number;

  // ── Color-legend bar (right side) ─────────────────────────────────────────

  /** Physical-pixel X of the left edge of the color bar. */
  legendX:    number;
  /** Physical-pixel Y of the top edge of the color bar (aligned with plotY). */
  legendY:    number;
  /** Physical-pixel height of the color bar (= plotSide). */
  legendH:    number;
  /** Physical-pixel width of the color bar. */
  legendW:    number;

  // ── Tissue group indicator blocks ─────────────────────────────────────────

  /** Physical-pixel width of the group-color strip on the Y axis. */
  groupBlockW: number;
  /** Physical-pixel height of the group-color strip on the X axis. */
  groupBlockH: number;
}

// ── Cell hit ─────────────────────────────────────────────────────────────────

/**
 * Result of hit-testing a mouse position against the matrix plot area.
 * `null` is returned when the mouse is outside the cell area.
 *
 * COORDINATE CONVENTION
 * ──────────────────────
 * `row` is the Y index (source region), `col` is the X index (target region).
 * Both are 0-based and in [0, n-1].
 */
export interface HeatmapCellHit {
  /** Row index (Y): source region index in node_ids. */
  row: number;
  /** Column index (X): target region index in node_ids. */
  col: number;
}

// ── Per-label metadata for axis drawing ──────────────────────────────────────

/**
 * Pre-computed per-label display metadata.  Built once by buildLabelInfoArray()
 * and consumed by drawGroupBlocks() and drawAxisLabels().
 */
export interface HeatmapLabelInfo {
  /** FreeSurfer label ID as a decimal string, e.g. "17". */
  labelId:  string;
  /** Abbreviated name for axis label, e.g. "Hipp-L". */
  abbrev:   string;
  /** Tissue macro-class: "gm" | "wm" | "csf" | "unknown". */
  group:    string;
  /** CSS-compatible colour string for the group indicator block. */
  groupRgb: string;
}
