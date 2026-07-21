/**
 * bids_events.types.ts — TypeScript interfaces for BIDS events.tsv data
 * ────────────────────────────────────────────────────────────────────────
 *
 * These types mirror the Pydantic models in backend/routers/bids_events.py.
 * A BIDS events.tsv records experimental stimulus onsets, durations, and
 * categorical trial types — the ground truth for what happened during a run.
 *
 * Usage in this platform:
 *   • SyncContext holds the loaded event list and exposes jumpToEvent().
 *   • MegPanel draws event onset lines and duration shading on the canvas.
 *   • EventTimelineRibbon renders clickable chips for each event, laid out
 *     proportionally along the total run duration.
 *
 * BIDS events.tsv column → TypeScript field mapping:
 *   onset         → onset         (required)
 *   duration      → duration      (required; 0 for point events)
 *   trial_type    → trialType     (optional in spec; defaults to "event")
 *   response_time → responseTime  (optional; null when absent or n/a)
 */

// ── Single event ──────────────────────────────────────────────────────────────

/**
 * One parsed row from a BIDS *_events.tsv file.
 * Returned by POST /api/bids/events/upload and GET /api/bids/events?path=...
 */
export interface ExperimentEvent {
  /**
   * Unique row identifier assigned by the backend (e.g. "evt_0042").
   * Used by jumpToEvent() in SyncContext to locate events by ID.
   */
  id: string;

  /**
   * Event onset time in seconds, measured from the start of the neuroimaging
   * run (i.e. from the first volume of the fMRI acquisition or the first
   * sample of the MEG recording).
   */
  onset: number;

  /**
   * Duration of the event in seconds.
   * 0 for point (instantaneous) events.
   * Used to draw the shaded background rectangle on the MEG canvas.
   */
  duration: number;

  /**
   * Categorical label for this trial type (e.g. "deviant_tone", "target").
   * All events with the same trialType share a colour.
   */
  trialType: string;

  /**
   * Time in seconds from stimulus onset to the subject's behavioural
   * response.  null when the subject did not respond or the column is absent.
   */
  responseTime: number | null;

  /**
   * Hex colour string assigned by the backend based on trialType.
   * Colour assignment is deterministic — same trial_type always gets the
   * same colour within a session regardless of event order.
   * Example: "#ef5350"
   */
  color: string;
}

// ── Full API response ─────────────────────────────────────────────────────────

/**
 * Complete response from the BIDS events endpoints.
 * Includes the event list plus derived metadata useful for rendering.
 */
export interface BidsEventsResult {
  /** All parsed events in file order (ascending onset). */
  events: ExperimentEvent[];

  /** Alphabetically sorted list of unique trial_type labels. */
  trialTypes: string[];

  /**
   * Mapping from trial_type label → hex colour.
   * Useful for rendering a legend in EventTimelineRibbon.
   */
  colorMap: Record<string, string>;

  /** Total number of rows parsed from the TSV. */
  nEvents: number;

  /**
   * max(onset + duration) across all events in seconds.
   * Use as the total duration of the experiment for timeline scaling.
   */
  totalDuration: number;
}

// ── Null / empty state ────────────────────────────────────────────────────────

/** Convenience constant for the "no events loaded" state. */
export const EMPTY_BIDS_EVENTS: BidsEventsResult = {
  events:        [],
  trialTypes:    [],
  colorMap:      {},
  nEvents:       0,
  totalDuration: 0,
};
