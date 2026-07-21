/**
 * EventTimelineRibbon.tsx — Proportionally laid-out BIDS event marker strip
 * ───────────────────────────────────────────────────────────────────────────
 *
 * Renders a horizontal ruler with one colour-coded chip per experimental
 * event.  Chip positions are proportional to onset time so the strip is a
 * faithful time-domain overview of the entire run.
 *
 * Clicking any chip calls jumpToEvent() — the shared SyncContext time cursor
 * snaps to that event's onset, moving both the MEG waveform cursor and the
 * fMRI volume index simultaneously.
 *
 * The chip that corresponds to activeEvent (the event whose onset is ≤
 * currentTimeSec) is highlighted with a white ring so the user always knows
 * "where they are" in the experiment.
 *
 * Consumer:
 *   MultimodalWorkspace mounts this directly below the split-panel row.
 *
 * Props:
 *   totalDuration — max(onset + duration) from the events.tsv;
 *                   passed from the bidsEventsApi response so the ruler
 *                   scales to the full run, not just the last event onset.
 */

import { type FC } from 'react';
import { useSyncContext } from '../contexts/SyncContext';

// ── Component ──────────────────────────────────────────────────────────────────

interface EventTimelineRibbonProps {
  /** Full experiment duration in seconds (from BidsEventsResult.totalDuration). */
  totalDuration: number;
}

const EventTimelineRibbon: FC<EventTimelineRibbonProps> = ({ totalDuration }) => {
  const { events, activeEvent, jumpToEvent } = useSyncContext();

  // Render nothing if no events are loaded
  if (events.length === 0) return null;

  // Guard against zero-duration edge-case (prevents division by zero)
  const dur = totalDuration > 0 ? totalDuration : 1;

  // Collect unique (trialType → colour) pairs for the legend
  const legendEntries = [...new Map(events.map((e) => [e.trialType, e.color])).entries()];

  return (
    <div className="event-timeline-ribbon" aria-label="Event timeline">
      {/* ── Proportional chip strip ──────────────────────────────────────────── */}
      <div className="event-timeline-ribbon__track">
        {events.map((evt) => {
          const leftPct  = (evt.onset / dur) * 100;
          const isActive = activeEvent?.id === evt.id;

          return (
            <button
              key={evt.id}
              className={`event-timeline-chip${isActive ? ' event-timeline-chip--active' : ''}`}
              style={{ left: `${leftPct}%`, background: evt.color }}
              onClick={() => jumpToEvent(evt.id)}
              title={`${evt.trialType}  @  ${evt.onset.toFixed(3)} s`}
              aria-pressed={isActive}
            />
          );
        })}
      </div>

      {/* ── Trial type colour legend ─────────────────────────────────────────── */}
      <div className="event-timeline-ribbon__legend">
        {legendEntries.map(([trialType, color]) => (
          <span key={trialType} className="event-timeline-legend-item">
            <span
              className="event-timeline-legend-swatch"
              style={{ background: color }}
            />
            {trialType}
          </span>
        ))}
      </div>
    </div>
  );
};

export default EventTimelineRibbon;
