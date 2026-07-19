/**
 * TimeScrubber.tsx — Temporal playback and frame-scrub UI for EEG source overlays
 * ─────────────────────────────────────────────────────────────────────────────────
 *
 * Provides:
 *   • A range slider for frame-by-frame scrubbing
 *   • Play / Pause button with configurable FPS
 *   • Frame counter display
 *
 * PERFORMANCE CONTRACT
 * ─────────────────────
 * The browser fires `input` events faster than the display refresh rate (up to
 * 1 000+ events/s on some platforms when the slider is dragged quickly).  Calling
 * `renderWindow.render()` on each raw event would stall the browser thread and
 * produce visible stuttering.
 *
 * Coalescing strategy:
 *   1. Each `input` event stores the desired frame in `pendingFrameRef`.
 *   2. If no requestAnimationFrame is pending, schedule one.
 *   3. The RAF callback fires at most once per display refresh (~16 ms at 60 Hz),
 *      reads the latest `pendingFrameRef`, and calls `onScrub` exactly once.
 *
 * This ensures renderWindow.render() is called at most once per display frame,
 * regardless of how fast the slider fires events.
 *
 * PLAYBACK LOOP
 * ──────────────
 * Playback uses a requestAnimationFrame loop that advances the frame counter
 * at the configured FPS using timestamp-based pacing:
 *
 *   • Timestamp comparison (performance.now()) prevents drift that would occur
 *     with a fixed-delay setInterval approach.
 *   • `lastTime += mspf` (not `= now`) prevents catch-up frames after a
 *     temporary pause (e.g., tab hidden then re-focused).
 *   • The loop self-cancels via `playingRef` when playing becomes false,
 *     avoiding the stale-closure problem with useEffect cleanup.
 *
 * When playback reaches the last frame it wraps back to 0 (loop mode).
 *
 * USAGE
 * ──────
 *   // In VolumetricViewer (or any component that owns the VTK bundle):
 *   const handleScrub = useCallback((frameIndex: number) => {
 *     temporalBundle.updateTimeFrame(frameIndex);
 *     ctx.renderWindow.render();
 *   }, [temporalBundle, ctx]);
 *
 *   <TimeScrubber
 *     frameCount={temporalBundle.frameCount}
 *     onScrub={handleScrub}
 *     fps={24}
 *   />
 *
 * CSS classes (defined in App.css under "── TimeScrubber ──"):
 *   .time-scrubber         Outer container
 *   .time-scrubber__row    Flex row (play btn + slider + counter)
 *   .time-scrubber__play   Play/Pause icon button
 *   .time-scrubber__track  Slider + progress bar wrapper
 *   .time-scrubber__slider Range input
 *   .time-scrubber__fill   Coloured fill overlay (decorative)
 *   .time-scrubber__label  Title label
 *   .time-scrubber__counter  Frame counter (e.g. "12 / 99")
 */

import {
  useState,
  useEffect,
  useRef,
  useCallback,
  type FC,
  type ChangeEvent,
} from 'react';

// ── Props ─────────────────────────────────────────────────────────────────────

export interface TimeScrubberProps {
  /** Total number of time frames in the temporal data. Must be ≥ 1. */
  frameCount: number;
  /**
   * Called with the frame index to display next.
   *
   * The caller is responsible for calling renderWindow.render() after
   * updating the VTK pipeline state (see temporalSourceOverlay.updateTimeFrame).
   * This separation keeps the component decoupled from vtk.js.
   *
   * This callback fires at most once per display frame (RAF-throttled during
   * scrubbing; once per logical frame during playback).
   */
  onScrub: (frameIndex: number) => void;
  /**
   * Playback speed in frames per second.
   * The RAF loop targets this rate; actual rate is capped by the display Hz.
   * @default 30
   */
  fps?: number;
  /** Optional additional CSS class on the outer container. */
  className?: string;
}

// ── Component ─────────────────────────────────────────────────────────────────

/**
 * TimeScrubber — playback / scrub control for a temporal EEG source overlay.
 *
 * Intentionally decoupled from vtk.js: it calls `onScrub(frameIndex)` and the
 * parent is responsible for the actual VTK + render call.
 */
const TimeScrubber: FC<TimeScrubberProps> = ({
  frameCount,
  onScrub,
  fps = 30,
  className,
}) => {
  // ── State ──────────────────────────────────────────────────────────────────
  const [frame,   setFrame]   = useState(0);
  const [playing, setPlaying] = useState(false);

  // ── Refs (mirror state for RAF closures — avoids stale captures) ───────────
  // React setState is async; refs give RAF closures synchronous access to the
  // latest values without capturing stale state from closure creation time.
  const playingRef      = useRef(false);
  const frameRef        = useRef(0);

  // RAF handle for scrub-event coalescing.
  const scrubRafRef     = useRef<number | null>(null);
  // Stores the frame index of the latest slider event, ready for the RAF tick.
  const pendingFrameRef = useRef(0);

  // Keep refs in sync with React state.
  useEffect(() => { playingRef.current = playing; }, [playing]);
  useEffect(() => { frameRef.current   = frame;   }, [frame]);

  // ── Scrub handler (RAF-throttled) ──────────────────────────────────────────
  const handleSliderChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const idx = Number(e.target.value);

    // Update React state for the controlled slider value immediately so the
    // thumb moves in real time — this is purely visual and cheap.
    setFrame(idx);
    frameRef.current = idx;

    // Store the latest desired frame for the next RAF flush.
    pendingFrameRef.current = idx;

    // Only schedule one RAF per display frame.  If one is already pending,
    // we've already "claimed" this frame; the latest pendingFrameRef value
    // will be read when the RAF fires.
    if (scrubRafRef.current !== null) return;

    scrubRafRef.current = requestAnimationFrame(() => {
      scrubRafRef.current = null;
      // Call the VTK update + render exactly once per display frame.
      onScrub(pendingFrameRef.current);
    });
  }, [onScrub]);

  // ── Playback loop ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!playing) return;

    const mspf = 1000 / fps; // milliseconds per frame
    let lastTime = performance.now();
    let rafId: number;

    const tick = (now: number) => {
      // Self-cancel if playing was set to false since this RAF was scheduled.
      if (!playingRef.current) return;

      if (now - lastTime >= mspf) {
        // Use += mspf (not = now) to keep pacing regular even if a frame was
        // slightly late — this prevents burst catch-up after lag spikes.
        lastTime += mspf;

        const next = (frameRef.current + 1) % frameCount;
        frameRef.current = next;
        setFrame(next);
        onScrub(next);
      }

      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);

    // Cleanup: cancel the loop if playing flips to false or deps change.
    return () => cancelAnimationFrame(rafId);
  }, [playing, fps, frameCount, onScrub]);

  // ── Pointer-down on slider: pause so scrubbing doesn't fight the loop ──────
  const handleSliderPointerDown = useCallback(() => {
    setPlaying(false);
  }, []);

  // ── Play / Pause toggle ────────────────────────────────────────────────────
  const togglePlay = useCallback(() => {
    setPlaying((prev) => !prev);
  }, []);

  // ── Derived display values ─────────────────────────────────────────────────
  const maxFrame = Math.max(0, frameCount - 1);
  // Progress fill width as a percentage (for the decorative fill bar).
  const fillPct  = frameCount > 1 ? (frame / maxFrame) * 100 : 0;

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className={`time-scrubber${className ? ` ${className}` : ''}`}>

      {/* Section label */}
      <span className="time-scrubber__label">Time Frame</span>

      <div className="time-scrubber__row">

        {/* Play / Pause */}
        <button
          className="time-scrubber__play"
          onClick={togglePlay}
          aria-label={playing ? 'Pause playback' : 'Start playback'}
          title={playing ? 'Pause' : 'Play'}
          type="button"
        >
          {playing
            ? /* pause icon */
              <svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor" aria-hidden>
                <rect x="3"  y="2" width="4" height="12" rx="1"/>
                <rect x="9" y="2" width="4" height="12" rx="1"/>
              </svg>
            : /* play icon */
              <svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor" aria-hidden>
                <path d="M3 2.5l10 5.5-10 5.5V2.5z"/>
              </svg>
          }
        </button>

        {/* Slider track */}
        <div className="time-scrubber__track">
          {/* Decorative fill bar behind the native thumb */}
          <div
            className="time-scrubber__fill"
            style={{ width: `${fillPct}%` }}
            aria-hidden
          />
          <input
            type="range"
            className="time-scrubber__slider"
            min={0}
            max={maxFrame}
            step={1}
            value={frame}
            onPointerDown={handleSliderPointerDown}
            onChange={handleSliderChange}
            aria-label="Time frame scrubber"
            aria-valuemin={0}
            aria-valuemax={maxFrame}
            aria-valuenow={frame}
            aria-valuetext={`Frame ${frame + 1} of ${frameCount}`}
          />
        </div>

        {/* Frame counter */}
        <span
          className="time-scrubber__counter"
          aria-live="polite"
          aria-atomic="true"
        >
          {frame + 1}&thinsp;/&thinsp;{frameCount}
        </span>

      </div>
    </div>
  );
};

export default TimeScrubber;
