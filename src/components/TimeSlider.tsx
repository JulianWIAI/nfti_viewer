/**
 * TimeSlider.tsx — Scrubber bar for navigating a 4-D fMRI timeseries
 * ────────────────────────────────────────────────────────────────────
 *
 * Renders a horizontal slider that displays the current playback position
 * in seconds and frames.  An optional Play/Pause button steps through
 * volumes automatically at a configurable frame rate.
 *
 * Integration with SyncContext:
 *   The slider reads currentTimeSec and writes setCurrentTimeSec from the
 *   SyncContext — no props needed for the time value itself.  Only the
 *   structural parameters (totalDuration, tr, nTimepoints) need to be
 *   passed explicitly because they come from the parsed fMRI payload.
 *
 * Play behaviour:
 *   The component uses setInterval with a ~100 ms tick.  Each tick advances
 *   currentTimeSec by one TR.  At the last volume the playback wraps to 0.
 *   Pausing freezes at the current position.
 */

import { useState, useEffect, useRef, type FC } from 'react';
import { useSyncContext } from '../contexts/SyncContext';

// ── Props ─────────────────────────────────────────────────────────────────────

interface TimeSliderProps {
  /** Total duration of the timeseries in seconds (= nTimepoints × tr). */
  totalDuration: number;
  /** TR in seconds — one step per TR during playback. */
  tr: number;
  /** Total number of volumes (used for display). */
  nTimepoints: number;
}

// ── Component ─────────────────────────────────────────────────────────────────

const TimeSlider: FC<TimeSliderProps> = ({ totalDuration, tr, nTimepoints }) => {
  const { currentTimeSec, setCurrentTimeSec, currentVolumeIdx } = useSyncContext();
  const [playing, setPlaying] = useState(false);

  // Keep a stable ref to setCurrentTimeSec so the interval callback does not
  // become stale while the component re-renders.
  const setTimeRef = useRef(setCurrentTimeSec);
  setTimeRef.current = setCurrentTimeSec;

  const trRef = useRef(tr);
  trRef.current = tr;

  const totalRef = useRef(totalDuration);
  totalRef.current = totalDuration;

  // Track current time in a ref so the interval always has the latest value
  const currentTimeRef = useRef(currentTimeSec);
  currentTimeRef.current = currentTimeSec;

  // Auto-advance by one TR per tick
  useEffect(() => {
    if (!playing || tr <= 0) return;

    const id = setInterval(() => {
      const next = currentTimeRef.current + trRef.current;
      // Wrap at end of timeseries
      setTimeRef.current(next >= totalRef.current ? 0 : next);
    }, Math.max(50, tr * 1000)); // advance in real-time; floor at 50 ms for fast TRs

    return () => clearInterval(id);
  }, [playing, tr]);

  // Stop playback when there is nothing to play
  const canPlay = nTimepoints > 1 && tr > 0;

  // Formatted time string: "12.3 s  (vol 7 / 150)"
  const label = tr > 0
    ? `${currentTimeSec.toFixed(1)} s  (vol ${currentVolumeIdx + 1} / ${nTimepoints})`
    : `${currentTimeSec.toFixed(1)} s`;

  return (
    <div className="time-slider">
      {/* Play / Pause button */}
      <button
        className={`time-slider__play-btn${playing ? ' time-slider__play-btn--active' : ''}`}
        disabled={!canPlay}
        onClick={() => setPlaying((p) => !p)}
        title={playing ? 'Pause' : 'Play'}
        aria-label={playing ? 'Pause playback' : 'Play timeseries'}
      >
        {playing ? '⏸' : '▶'}
      </button>

      {/* Scrubber input */}
      <input
        className="time-slider__range"
        type="range"
        min={0}
        max={totalDuration}
        step={tr > 0 ? tr : 0.1}
        value={currentTimeSec}
        disabled={totalDuration === 0}
        onChange={(e) => {
          setPlaying(false); // stop playback on manual scrub
          setCurrentTimeSec(Number(e.target.value));
        }}
      />

      {/* Time label */}
      <span className="time-slider__label">{label}</span>
    </div>
  );
};

export default TimeSlider;
