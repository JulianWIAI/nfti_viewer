/**
 * SyncContext.tsx — Shared time-domain synchronisation for the multimodal workspace
 * ──────────────────────────────────────────────────────────────────────────────────
 *
 * Central state hub that links:
 *   • MEG waveform time axis (continuous seconds)
 *   • fMRI volume index (discrete TR steps)
 *   • BIDS experimental events (stimulus onsets from *_events.tsv)
 *
 * Core relationships:
 *   currentVolumeIdx  = Math.floor(currentTimeSec / tr)
 *   jumpToEvent(id)   → setCurrentTimeSec(event.onset)
 *                      → cascades to MegPanel cursor + fMRI volume jump
 *
 * Consumer hierarchy:
 *   MultimodalWorkspace  ← SyncProvider
 *     ├── FmriPanel        reads: currentVolumeIdx, tr
 *     ├── MegPanel         reads: currentTimeSec, events
 *     │                    writes: setCurrentTimeSec
 *     └── EventTimelineRibbon  reads: events, currentTimeSec
 *                              writes: jumpToEvent
 *
 * DESIGN NOTE — events live here, not in a plugin:
 *   The event list is decoupled from both the MEG session and the fMRI file.
 *   A single events.tsv can annotate any combination of modalities.  Keeping
 *   events in SyncContext means any descendant (ribbon, canvas, 3D overlay)
 *   can consume them without coupling to a specific plugin.
 */

import {
  createContext,
  useContext,
  useState,
  useMemo,
  useCallback,
  type FC,
  type ReactNode,
} from 'react';
import type { ExperimentEvent } from '../types/bids_events.types';

// ── Context value shape ────────────────────────────────────────────────────────

export interface SyncContextValue {
  // ── Time state ────────────────────────────────────────────────────────────

  /**
   * Current scrub / playback position in seconds (MEG time axis).
   * Initialised to 0; updated by MegPanel clicks, TimeSlider, or jumpToEvent.
   */
  currentTimeSec: number;

  /** Set the shared time position (MEG writes, fMRI and ribbon read). */
  setCurrentTimeSec: (t: number) => void;

  // ── fMRI TR ───────────────────────────────────────────────────────────────

  /**
   * Repetition time of the loaded fMRI volume in seconds.
   * 0 until FmriPanel calls setTr() after parsing the header.
   */
  tr: number;

  /** Register the TR once the fMRI payload is known (call from FmriPanel). */
  setTr: (tr: number) => void;

  /**
   * Derived fMRI volume index: Math.floor(currentTimeSec / tr).
   * 0 when tr === 0 (fMRI not loaded).
   */
  currentVolumeIdx: number;

  // ── BIDS experimental events ──────────────────────────────────────────────

  /**
   * All events parsed from the loaded *_events.tsv file.
   * Empty array until the user loads a TSV via the MultimodalWorkspace sidebar.
   */
  events: ExperimentEvent[];

  /**
   * Replace the event list after parsing a new events.tsv.
   * Resets any previously loaded events.
   */
  setEvents: (events: ExperimentEvent[]) => void;

  /**
   * Instantly snap currentTimeSec to the onset of the event identified by id.
   * Cascades to both the MEG cursor and the fMRI volume index.
   * No-ops silently when no event with that id exists.
   */
  jumpToEvent: (eventId: string) => void;

  /**
   * The event whose onset is closest to (and ≤) currentTimeSec.
   * null when no events are loaded or currentTimeSec precedes all events.
   * Highlighted by EventTimelineRibbon to show "you are here".
   */
  activeEvent: ExperimentEvent | null;
}

// ── Context creation ──────────────────────────────────────────────────────────

const SyncContext = createContext<SyncContextValue | null>(null);

// ── Provider ──────────────────────────────────────────────────────────────────

interface SyncProviderProps {
  children: ReactNode;
}

/**
 * Provides the shared multimodal time state to the entire workspace subtree.
 * Mount this exactly once, above FmriPanel, MegPanel, and EventTimelineRibbon.
 */
export const SyncProvider: FC<SyncProviderProps> = ({ children }) => {
  const [currentTimeSec, setCurrentTimeSec] = useState<number>(0);
  const [tr,             setTr]             = useState<number>(0);
  const [events,         setEvents]         = useState<ExperimentEvent[]>([]);

  // ── Derived: fMRI volume index ────────────────────────────────────────────
  // When TR is 0 (no fMRI loaded) clamp to 0 — always a valid volume index.
  const currentVolumeIdx = useMemo<number>(
    () => (tr > 0 ? Math.floor(currentTimeSec / tr) : 0),
    [currentTimeSec, tr],
  );

  // ── Derived: active event ─────────────────────────────────────────────────
  // The event whose onset is closest-to-or-before currentTimeSec.
  // We scan in reverse order (events are in onset order) so we can stop at
  // the first event whose onset is ≤ current time — O(n) worst case but
  // events lists are small (typically < 500 rows).
  const activeEvent = useMemo<ExperimentEvent | null>(() => {
    if (events.length === 0) return null;
    // Walk backwards through onset-sorted events to find last onset ≤ now
    for (let i = events.length - 1; i >= 0; i--) {
      if (events[i].onset <= currentTimeSec) return events[i];
    }
    return null;
  }, [events, currentTimeSec]);

  // ── jumpToEvent ───────────────────────────────────────────────────────────
  // Stable reference (no recreate on every render) so consumers that depend
  // on it in useCallback do not get stale closures.
  const jumpToEvent = useCallback((eventId: string) => {
    const target = events.find((e) => e.id === eventId);
    if (target) setCurrentTimeSec(target.onset);
  }, [events]);

  const value: SyncContextValue = {
    currentTimeSec,
    setCurrentTimeSec,
    tr,
    setTr,
    currentVolumeIdx,
    events,
    setEvents,
    jumpToEvent,
    activeEvent,
  };

  return <SyncContext.Provider value={value}>{children}</SyncContext.Provider>;
};

// ── Hook ──────────────────────────────────────────────────────────────────────

/**
 * Consume the shared multimodal sync state.
 * Must be called from a descendant of SyncProvider — throws otherwise.
 */
export function useSyncContext(): SyncContextValue {
  const ctx = useContext(SyncContext);
  if (!ctx) throw new Error('useSyncContext must be called inside a SyncProvider');
  return ctx;
}
