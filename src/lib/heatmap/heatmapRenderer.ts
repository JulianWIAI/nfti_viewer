/**
 * heatmapRenderer.ts — Canvas 2D draw functions for the connectivity heatmap
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * ALL FUNCTIONS ARE PURE (no React state, no DOM queries).
 * They take a CanvasRenderingContext2D and the precomputed layout/data, draw,
 * and return nothing.  The React component is responsible for calling them
 * in the correct order and clearing the canvas between frames.
 *
 * DRAW ORDER (called by the React component):
 *   1. drawBackground     — dark fill for the whole canvas
 *   2. drawCells          — N×N matrix cells using ImageData (one GPU upload)
 *   3. drawGridLines      — subtle 1-physical-pixel lines between cells
 *   4. drawGroupBlocks    — tissue-class colour strips on both axes
 *   5. drawAxisLabels     — rotated X-axis labels, horizontal Y-axis labels
 *   6. drawColorLegend    — gradient bar + min/max tick labels on the right
 *   7. drawHighlight      — hover highlight border (called separately on mousemove)
 *
 * PERFORMANCE NOTES
 * ──────────────────
 * drawCells() uses ctx.createImageData(plotSide, plotSide) → writes all
 * N×N cell pixels in a single typed-array pass → ctx.putImageData().
 * This is a single GPU texture upload rather than N² fillRect calls.
 * For a 33×33 matrix at 2× DPR with ~20px/cell = 660×660px = 435k pixels — trivial.
 */

import type { HeatmapLayout }    from './heatmapTypes';
import type { HeatmapLabelInfo } from './heatmapTypes';
import { fiberCountToRgb, buildLegendGradient } from './heatmapColorScale';

// ── 1. Background ─────────────────────────────────────────────────────────────

/**
 * Fill the entire canvas with the dark background colour.
 * Called first so subsequent draws composite on top of a clean surface.
 */
export function drawBackground(
  ctx:    CanvasRenderingContext2D,
  layout: HeatmapLayout,
): void {
  ctx.fillStyle = '#141414';   // slightly darker than var(--bg-section)
  ctx.fillRect(0, 0, layout.canvasW, layout.canvasH);
}

// ── 2. Cells — ImageData path ─────────────────────────────────────────────────

/**
 * Render the N×N matrix cells into a single ImageData and upload via putImageData.
 *
 * Each cell occupies Math.round(cellW) × Math.round(cellH) physical pixels.
 * Cell (i, j) corresponds to matrix[i][j] (row i = Y, column j = X).
 *
 * @param matrix   The N×N fiber count matrix from the API.
 * @param maxVal   Maximum fiber count across the entire matrix (for normalisation).
 */
export function drawCells(
  ctx:    CanvasRenderingContext2D,
  layout: HeatmapLayout,
  matrix: number[][],
  maxVal: number,
): void {
  const { plotX, plotY, plotSide, cellW, cellH, n } = layout;

  if (plotSide <= 0 || n <= 0) return;

  const imgData = ctx.createImageData(plotSide, plotSide);
  const data    = imgData.data;   // Uint8ClampedArray: [R,G,B,A, R,G,B,A, …]

  for (let i = 0; i < n; i++) {
    // Pixel Y bounds for row i.
    const y0 = Math.round(i * cellH);
    const y1 = Math.round((i + 1) * cellH);

    for (let j = 0; j < n; j++) {
      // Pixel X bounds for column j.
      const x0 = Math.round(j * cellW);
      const x1 = Math.round((j + 1) * cellW);

      const count  = matrix[i]?.[j] ?? 0;
      const isDiag = i === j;
      const [r, g, b] = fiberCountToRgb(count, maxVal, isDiag);

      // Fill every pixel in the [x0,x1) × [y0,y1) cell rectangle.
      for (let py = y0; py < y1 && py < plotSide; py++) {
        for (let px = x0; px < x1 && px < plotSide; px++) {
          const base = (py * plotSide + px) * 4;
          data[base]     = r;
          data[base + 1] = g;
          data[base + 2] = b;
          data[base + 3] = 255;
        }
      }
    }
  }

  ctx.putImageData(imgData, plotX, plotY);
}

// ── 3. Grid Lines ─────────────────────────────────────────────────────────────

/**
 * Draw 1-physical-pixel grid lines between cells.
 * Drawn after putImageData so they composite cleanly on top.
 */
export function drawGridLines(
  ctx:    CanvasRenderingContext2D,
  layout: HeatmapLayout,
): void {
  const { plotX, plotY, plotSide, cellW, cellH, n } = layout;
  if (n <= 1 || plotSide <= 0) return;

  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth   = 1;
  ctx.beginPath();

  // Vertical lines (between columns).
  for (let j = 1; j < n; j++) {
    const x = plotX + Math.round(j * cellW);
    ctx.moveTo(x + 0.5, plotY);
    ctx.lineTo(x + 0.5, plotY + plotSide);
  }

  // Horizontal lines (between rows).
  for (let i = 1; i < n; i++) {
    const y = plotY + Math.round(i * cellH);
    ctx.moveTo(plotX,             y + 0.5);
    ctx.lineTo(plotX + plotSide,  y + 0.5);
  }

  ctx.stroke();
  ctx.restore();
}

// ── 4. Group blocks ───────────────────────────────────────────────────────────

/**
 * Draw tissue-class colour strips adjacent to both axes.
 *
 * Y-axis strip:  x ∈ [plotX - groupBlockW - GAP, plotX - GAP),  height = cellH per label
 * X-axis strip:  y ∈ [plotY - groupBlockH - GAP, plotY - GAP),  width  = cellW per label
 *
 * The gap between the strip and the plot edge is 4 physical pixels.
 */
export function drawGroupBlocks(
  ctx:    CanvasRenderingContext2D,
  layout: HeatmapLayout,
  labels: HeatmapLabelInfo[],
): void {
  const { plotX, plotY, cellW, cellH, groupBlockW, groupBlockH, dpr, n } = layout;
  const GAP = Math.round(4 * dpr);

  for (let i = 0; i < n; i++) {
    const info = labels[i];
    if (!info) continue;
    ctx.fillStyle = info.groupRgb;

    // Y-axis strip (left of plot).
    const yMid = plotY + Math.round(i * cellH);
    const yEnd = plotY + Math.round((i + 1) * cellH);
    ctx.fillRect(
      plotX - groupBlockW - GAP,
      yMid,
      groupBlockW,
      yEnd - yMid,
    );

    // X-axis strip (above plot).
    const xMid = plotX + Math.round(i * cellW);
    const xEnd = plotX + Math.round((i + 1) * cellW);
    ctx.fillRect(
      xMid,
      plotY - groupBlockH - GAP,
      xEnd - xMid,
      groupBlockH,
    );
  }
}

// ── 5. Axis labels ────────────────────────────────────────────────────────────

/**
 * Draw abbreviated anatomy labels on both axes.
 *
 * X-axis (top): labels are rotated –45° so they fit in the marginTop band.
 *               Text anchor is 'left', baseline 'middle'.
 *
 * Y-axis (left): labels are drawn horizontally, right-aligned.
 *                Text anchor is 'right', baseline 'middle'.
 *
 * Font size scales with cellH to avoid overflow in small matrices.
 */
export function drawAxisLabels(
  ctx:    CanvasRenderingContext2D,
  layout: HeatmapLayout,
  labels: HeatmapLabelInfo[],
): void {
  const { plotX, plotY, cellW, cellH, groupBlockW, groupBlockH, dpr, n } = layout;
  const GAP       = Math.round(4 * dpr);
  const fontSize  = Math.max(8, Math.min(11, Math.floor(cellH * 0.55))) * dpr;
  const fontStyle = `${fontSize}px -apple-system, system-ui, sans-serif`;

  ctx.save();
  ctx.fillStyle = 'rgba(200,200,200,0.82)';
  ctx.font      = fontStyle;

  for (let i = 0; i < n; i++) {
    const info = labels[i];
    if (!info) continue;

    const cellCentreY = plotY + (i + 0.5) * cellH;
    const cellCentreX = plotX + (i + 0.5) * cellW;

    // ── Y-axis label (horizontal, right-aligned) ─────────────────────────────
    ctx.textAlign    = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(
      info.abbrev,
      plotX - groupBlockW - GAP * 2,   // right edge = 2 gaps left of group strip
      cellCentreY,
    );

    // ── X-axis label (rotated –45°) ──────────────────────────────────────────
    ctx.save();
    ctx.translate(cellCentreX, plotY - groupBlockH - GAP * 2);
    ctx.rotate(-Math.PI / 4);
    ctx.textAlign    = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(info.abbrev, 0, 0);
    ctx.restore();
  }

  ctx.restore();
}

// ── 6. Color legend ───────────────────────────────────────────────────────────

/**
 * Draw the vertical gradient bar to the right of the matrix, plus min/max labels.
 *
 * The bar maps max fiber count (top) → min fiber count (bottom) using the same
 * piecewise color scale as the cell fill.  Two tick marks label the endpoints.
 */
export function drawColorLegend(
  ctx:    CanvasRenderingContext2D,
  layout: HeatmapLayout,
  maxVal: number,
): void {
  const { legendX, legendY, legendH, legendW, dpr } = layout;
  if (legendH <= 0) return;

  const lut = buildLegendGradient();   // 512 [R,G,B] entries

  // Render the gradient bar as a 1-column ImageData, stretched to legendW.
  const barData = ctx.createImageData(1, legendH);
  const bArr    = barData.data;

  for (let py = 0; py < legendH; py++) {
    // py=0 = top = max fibers, py=legendH-1 = bottom = min fibers.
    const t   = 1 - py / (legendH - 1);
    const idx = Math.round(t * 511);
    const base = py * 4;
    bArr[base]     = lut[idx * 3]!;
    bArr[base + 1] = lut[idx * 3 + 1]!;
    bArr[base + 2] = lut[idx * 3 + 2]!;
    bArr[base + 3] = 255;
  }

  // Draw the 1-pixel-wide bar then scale to legendW via drawImage.
  const tmpCanvas = new OffscreenCanvas(1, legendH);
  const tmpCtx    = tmpCanvas.getContext('2d')!;
  tmpCtx.putImageData(barData, 0, 0);
  ctx.drawImage(tmpCanvas, legendX, legendY, legendW, legendH);

  // Border around the bar.
  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth   = 1;
  ctx.strokeRect(legendX, legendY, legendW, legendH);

  // Tick labels.
  const fontSize  = Math.max(8, Math.round(9 * dpr));
  ctx.fillStyle   = 'rgba(180,180,180,0.8)';
  ctx.font        = `${fontSize}px -apple-system, system-ui, sans-serif`;
  ctx.textAlign   = 'left';

  ctx.textBaseline = 'top';
  ctx.fillText(String(maxVal), legendX + legendW + Math.round(4 * dpr), legendY);

  ctx.textBaseline = 'bottom';
  ctx.fillText('0', legendX + legendW + Math.round(4 * dpr), legendY + legendH);
}

// ── 7. Hover highlight ────────────────────────────────────────────────────────

/**
 * Draw a highlight border around the hovered cell (row, col).
 * Draws a bright accent-green 1.5px rect — does NOT repaint cells.
 *
 * Call this after drawCells/drawGridLines, or redraw the full frame and call last.
 */
export function drawHighlight(
  ctx:    CanvasRenderingContext2D,
  layout: HeatmapLayout,
  row:    number,
  col:    number,
): void {
  const { plotX, plotY, cellW, cellH } = layout;

  const x = plotX + Math.round(col * cellW);
  const y = plotY + Math.round(row * cellH);
  const w = Math.round((col + 1) * cellW) - Math.round(col * cellW);
  const h = Math.round((row + 1) * cellH) - Math.round(row * cellH);

  ctx.save();
  ctx.strokeStyle = '#00e676';   // --accent-green
  ctx.lineWidth   = 1.5;
  ctx.strokeRect(x + 0.75, y + 0.75, w - 1.5, h - 1.5);
  ctx.restore();
}

// ── 7b. Persistent selection border ──────────────────────────────────────────

/**
 * Draw a persistent (non-hover) selection border around cell (row, col).
 * Visually distinct from the hover highlight:
 *   • Colour: white  rgba(255,255,255,0.90)  (vs accent-green for hover)
 *   • Width:  2 physical pixels              (vs 1.5 for hover)
 *
 * Call this BEFORE drawHighlight so the hover overlay renders on top.
 * Typically called twice — for (row, col) AND (col, row) — to reflect the
 * undirected nature of the selected fiber tract.
 */
export function drawSelection(
  ctx:    CanvasRenderingContext2D,
  layout: HeatmapLayout,
  row:    number,
  col:    number,
): void {
  const { plotX, plotY, cellW, cellH } = layout;

  const x = plotX + Math.round(col * cellW);
  const y = plotY + Math.round(row * cellH);
  const w = Math.round((col + 1) * cellW) - Math.round(col * cellW);
  const h = Math.round((row + 1) * cellH) - Math.round(row * cellH);

  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.90)';
  ctx.lineWidth   = 2;
  ctx.strokeRect(x + 1, y + 1, w - 2, h - 2);
  ctx.restore();
}

// ── Hit test ─────────────────────────────────────────────────────────────────

/**
 * Convert a mouse event position (CSS pixels relative to the canvas element)
 * to a matrix cell (row, col).
 *
 * @param cssX   offsetX from the mouse event (CSS pixels).
 * @param cssY   offsetY from the mouse event (CSS pixels).
 * @param layout Precomputed layout (physical pixels).
 * @returns      { row, col } or null if outside the plot area.
 */
export function hitTest(
  cssX:   number,
  cssY:   number,
  layout: HeatmapLayout,
): { row: number; col: number } | null {
  const { plotX, plotY, plotSide, cellW, cellH, n, dpr } = layout;

  const px = cssX * dpr;
  const py = cssY * dpr;

  if (px < plotX || px >= plotX + plotSide) return null;
  if (py < plotY || py >= plotY + plotSide) return null;

  const col = Math.floor((px - plotX) / cellW);
  const row = Math.floor((py - plotY) / cellH);

  if (col < 0 || col >= n || row < 0 || row >= n) return null;

  return { row, col };
}
