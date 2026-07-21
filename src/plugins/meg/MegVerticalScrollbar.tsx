/**
 * MegVerticalScrollbar.tsx — Vertical scroll thumb for the MEG waveform canvas
 * ──────────────────────────────────────────────────────────────────────────────
 *
 * Renders a narrow 8 px strip positioned absolutely on the right edge of the
 * canvas container.  Mirrors the existing horizontal MegScrollbar pattern but
 * runs top-to-bottom.
 *
 * Behaviour:
 *   • Hidden when all channel lanes already fit within the visible height.
 *   • Thumb size is proportional to the visible fraction of the total content.
 *   • Click anywhere on the track to jump scroll to that row.
 *   • Drag the thumb to pan continuously; pointer capture keeps the drag live
 *     even when the pointer moves outside the element.
 *   • Mouse-wheel scrolling is handled by the parent canvas via onWheel;
 *     this component only provides the visual indicator and click/drag input.
 *
 * CSS classes (defined in App.css):
 *   .meg-vscroll          — the scrollbar track
 *   .meg-vscroll__thumb   — the draggable thumb
 */

import { useRef, type FC } from 'react';

export interface MegVerticalScrollbarProps {
  /** Total scrollable content height in px — the sum of all channel lane heights. */
  totalH:   number;
  /** Visible area height in px — canvas height minus the time-axis strip (AXIS_H). */
  viewH:    number;
  /** Current scroll offset in px; 0 = scrolled to the top. */
  offset:   number;
  /** Called with the new offset (clamped) whenever the user drags or clicks. */
  onChange: (offset: number) => void;
}

const MegVerticalScrollbar: FC<MegVerticalScrollbarProps> = ({
  totalH,
  viewH,
  offset,
  onChange,
}) => {
  // ── Early exit — no scrollbar needed when all content fits ───────────────
  if (totalH <= viewH || viewH <= 0) return null;

  const maxScroll = totalH - viewH;

  // Thumb geometry: thumb height is proportional to the visible fraction
  const thumbRatio = Math.min(1, viewH / totalH);
  const thumbH     = Math.max(24, thumbRatio * viewH);                   // ≥ 24 px for usability
  const thumbTop   = (offset / maxScroll) * (viewH - thumbH);            // px from top of track

  // Drag state — stored in refs to avoid re-renders on every mousemove
  const draggingRef  = useRef(false);
  const startYRef    = useRef(0);    // clientY at drag start
  const startOffRef  = useRef(0);    // scrollOffset at drag start

  /** Begin drag: capture the pointer so move events fire outside the element. */
  const onThumbPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    draggingRef.current  = true;
    startYRef.current    = e.clientY;
    startOffRef.current  = offset;
    e.currentTarget.setPointerCapture(e.pointerId);
    e.preventDefault();   // prevent text selection while dragging
    e.stopPropagation();  // don't bubble to the track click handler
  };

  /** While dragging: translate the pixel delta into a scroll offset delta. */
  const onThumbPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return;
    const dy      = e.clientY - startYRef.current;
    // Scale the pixel drag distance by the ratio of content height to track height
    const dOffset = (dy / (viewH - thumbH)) * maxScroll;
    onChange(Math.max(0, Math.min(maxScroll, startOffRef.current + dOffset)));
  };

  const onThumbPointerUp     = () => { draggingRef.current = false; };
  const onThumbPointerCancel = () => { draggingRef.current = false; };

  /** Track click (outside the thumb): jump scroll so the clicked row is centred. */
  const onTrackClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const track = e.currentTarget.getBoundingClientRect();
    // Centre the thumb on the click position
    const relY  = e.clientY - track.top - thumbH / 2;
    const ratio = Math.max(0, Math.min(1, relY / (viewH - thumbH)));
    onChange(ratio * maxScroll);
  };

  return (
    <div
      className="meg-vscroll"
      style={{ height: viewH }}
      onClick={onTrackClick}
      role="scrollbar"
      aria-orientation="vertical"
      aria-valuenow={Math.round(offset)}
      aria-valuemin={0}
      aria-valuemax={Math.round(maxScroll)}
      aria-label="Vertical scroll — channel lanes"
    >
      <div
        className="meg-vscroll__thumb"
        style={{ height: thumbH, transform: `translateY(${thumbTop.toFixed(1)}px)` }}
        onPointerDown={onThumbPointerDown}
        onPointerMove={onThumbPointerMove}
        onPointerUp={onThumbPointerUp}
        onPointerCancel={onThumbPointerCancel}
      />
    </div>
  );
};

export default MegVerticalScrollbar;
