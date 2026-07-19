/**
 * heatmapLayout.ts — DPR-aware layout computation for the heatmap canvas
 * ─────────────────────────────────────────────────────────────────────────
 *
 * LAYOUT DIAGRAM (CSS pixels, left-to-right, top-to-bottom)
 * ───────────────────────────────────────────────────────────
 *
 *   ┌──────────────────────────────────────────────────────────────────┐
 *   │  marginTop (rotated label height + group block H + gap)          │
 *   │  ┌──────┬──┬──────────────────────────────────┬────────────────┐│
 *   │  │ axis │gb│     plot area (plotSide²)         │  legend bar   ││
 *   │  │ lbls │  │                                   │               ││
 *   │  │(72px)│8p│  row labels on Y axis             │  legendW(12px)││
 *   │  │      │x │                                   │               ││
 *   │  └──────┴──┴──────────────────────────────────┴────────────────┘│
 *   │  marginBottom                                                    │
 *   └──────────────────────────────────────────────────────────────────┘
 *
 *   marginLeft  = labelW(72) + groupBlockW(8) + gap(8)       = 88 px
 *   marginTop   = labelH_rot(64) + groupBlockH(8) + gap(8)   = 80 px
 *   marginRight = legendGap(12) + legendW(12) + numW(30) + pad(6) = 60 px
 *   marginBottom = 20 px
 *
 * ALL MEASUREMENTS ARE IN PHYSICAL PIXELS (CSS × devicePixelRatio).
 * Every constant below is defined in CSS px and multiplied by `dpr`.
 */

import type { HeatmapLayout } from './heatmapTypes';

// ── CSS pixel constants ───────────────────────────────────────────────────────
// These represent the intended *CSS* pixel dimensions before DPR scaling.

const LABEL_W_CSS      = 72;   // width of rotated Y-axis text area
const LABEL_H_ROT_CSS  = 64;   // height consumed by 45° rotated X-axis labels
const GROUP_BLOCK_CSS  = 8;    // width/height of tissue-group colour strip
const GAP_CSS          = 8;    // gap between strip and axis labels / plot edge

const LEGEND_GAP_CSS   = 12;   // gap between right edge of plot and legend bar
const LEGEND_W_CSS     = 12;   // width of the gradient bar
const LEGEND_NUM_W_CSS = 30;   // width reserved for tick labels right of bar
const LEGEND_PAD_CSS   = 6;    // right padding

const MARGIN_BOTTOM_CSS = 20;

// ── Derived margin totals (CSS px) ───────────────────────────────────────────

const MARGIN_LEFT_CSS   = LABEL_W_CSS + GROUP_BLOCK_CSS + GAP_CSS;          // 88
const MARGIN_TOP_CSS    = LABEL_H_ROT_CSS + GROUP_BLOCK_CSS + GAP_CSS;       // 80
const MARGIN_RIGHT_CSS  = LEGEND_GAP_CSS + LEGEND_W_CSS + LEGEND_NUM_W_CSS + LEGEND_PAD_CSS; // 60

// ── computeLayout ─────────────────────────────────────────────────────────────

/**
 * Compute the full pixel layout for the heatmap canvas.
 *
 * @param cssW   CSS pixel width of the container element.
 * @param cssH   CSS pixel height of the container element.
 * @param n      Number of matrix rows/columns.
 * @param dpr    window.devicePixelRatio (defaults to 1 if not provided).
 * @returns      HeatmapLayout — all measurements in physical pixels.
 */
export function computeLayout(
  cssW: number,
  cssH: number,
  n:    number,
  dpr  = window.devicePixelRatio ?? 1,
): HeatmapLayout {
  // Physical pixel canvas dimensions.
  const canvasW = Math.round(cssW * dpr);
  const canvasH = Math.round(cssH * dpr);

  // Margins in physical pixels.
  const marginLeft   = Math.round(MARGIN_LEFT_CSS   * dpr);
  const marginTop    = Math.round(MARGIN_TOP_CSS    * dpr);
  const marginRight  = Math.round(MARGIN_RIGHT_CSS  * dpr);
  const marginBottom = Math.round(MARGIN_BOTTOM_CSS * dpr);

  // The plot is always a square — take the smaller of the two available spans.
  const availW   = canvasW - marginLeft - marginRight;
  const availH   = canvasH - marginTop  - marginBottom;
  const plotSide = Math.max(0, Math.min(availW, availH));

  // Plot origin (physical pixels).
  const plotX = marginLeft;
  const plotY = marginTop;

  // Cell dimensions (floating point — rounding is applied per-cell in the renderer).
  const cellW = n > 0 ? plotSide / n : 0;
  const cellH = n > 0 ? plotSide / n : 0;

  // Legend bar geometry (physical pixels).
  const legendGap = Math.round(LEGEND_GAP_CSS * dpr);
  const legendW   = Math.round(LEGEND_W_CSS   * dpr);
  const legendX   = plotX + plotSide + legendGap;
  const legendY   = plotY;
  const legendH   = plotSide;

  // Group block dimensions (physical pixels).
  const groupBlockW = Math.round(GROUP_BLOCK_CSS * dpr);
  const groupBlockH = Math.round(GROUP_BLOCK_CSS * dpr);

  return {
    canvasW, canvasH, dpr,
    plotX, plotY, plotSide,
    cellW, cellH, n,
    marginLeft, marginTop, marginRight, marginBottom,
    legendX, legendY, legendH, legendW,
    groupBlockW, groupBlockH,
  };
}
