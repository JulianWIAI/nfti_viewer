/**
 * HeatmapTooltip.tsx — Floating tooltip for ConnectomeHeatmap cell hover
 * ─────────────────────────────────────────────────────────────────────────
 *
 * A simple absolutely-positioned DOM element that follows the mouse cursor.
 * Rendered inside the heatmap container (which must have position: relative).
 *
 * POSITIONING
 * ────────────
 * The tooltip appears 14 CSS pixels right of and below the cursor.
 * When the cursor is close to the right edge of the container, the tooltip
 * is flipped to the left (leftOf mode) to prevent overflow.
 *
 * PROPS
 * ──────
 *   cssX, cssY     — cursor position relative to the container element
 *   containerW     — container width in CSS pixels (for flip detection)
 *   sourceName     — anatomy name of the source (row) region
 *   targetName     — anatomy name of the target (column) region
 *   fiberCount     — fibre count for this cell, or null for the diagonal
 */

import type { JSX } from 'react';

interface HeatmapTooltipProps {
  /** Cursor X in CSS pixels, relative to the heatmap container. */
  cssX:       number;
  /** Cursor Y in CSS pixels, relative to the heatmap container. */
  cssY:       number;
  /** Container width in CSS pixels — used to decide left/right flip. */
  containerW: number;
  /** Human-readable name of the source (row) region. */
  sourceName: string;
  /** Human-readable name of the target (column) region. */
  targetName: string;
  /** Fiber count to display; null means diagonal (no self-connection). */
  fiberCount: number | null;
}

const OFFSET_PX  = 14;
const TOOLTIP_W  = 200;  // approximate tooltip width for flip detection

export default function HeatmapTooltip({
  cssX,
  cssY,
  containerW,
  sourceName,
  targetName,
  fiberCount,
}: HeatmapTooltipProps): JSX.Element {

  // Flip to left side of cursor when near right edge.
  const flipLeft = cssX + OFFSET_PX + TOOLTIP_W > containerW;
  const left     = flipLeft
    ? cssX - TOOLTIP_W - OFFSET_PX
    : cssX + OFFSET_PX;
  const top      = cssY + OFFSET_PX;

  const bodyText = fiberCount === null
    ? '(diagonal — no self-connection)'
    : `${fiberCount.toLocaleString()} fibres`;

  return (
    <div
      className="heatmap-tooltip"
      style={{ left, top }}
      aria-hidden   // screen readers get data via the canvas aria-label
    >
      <span className="heatmap-tooltip__pair">
        {sourceName} ↔ {targetName}
      </span>
      <span className="heatmap-tooltip__value">{bodyText}</span>
    </div>
  );
}
