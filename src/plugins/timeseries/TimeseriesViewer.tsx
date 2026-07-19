/**
 * TimeseriesViewer.tsx — EEG / iEEG / MEG multi-channel waveform viewer
 * ────────────────────────────────────────────────────────────────────────
 *
 * Renders the parsed TimeseriesPayload onto an HTML <canvas> element using the
 * useCanvasRenderer hook (min/max ribbon decimation for pixel-perfect fidelity).
 *
 * The viewer owns the TimeseriesContext that is consumed by TimeseriesControls,
 * following the same pattern as VolumetricViewer / VolumetricControls.
 *
 * Layout:
 *   ┌──────────────────────────────────────────┐
 *   │  <canvas>  — full parent height/width     │
 *   │    lane 0: channel Fp1  ~~~~~~~~~~~~~~~~  │
 *   │    lane 1: channel Fp2  ~~~~~~~~~~~~~~~~  │
 *   │    …                                      │
 *   │    time axis                              │
 *   └──────────────────────────────────────────┘
 *
 * When no data is loaded the canvas is replaced by an empty-state message.
 *
 * Keyboard shortcuts (when the viewer container is focused):
 *   ← / →   pan the time window by half a window width
 *   + / -   zoom in / out (halve or double window width)
 *   Home    jump to t = 0
 *   End     jump to the last window
 */

import {
  type FC,
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useRef,
  type RefObject,
} from 'react';
import type { PluginViewerProps } from '../../types/plugin.types';
import type { TimeseriesPayload, TimeseriesViewerControls } from '../../types/timeseries.types';
import { computeAutoScale } from './lib/channelScaler';
import { useCanvasRenderer } from './hooks/useCanvasRenderer';

// ── Context ───────────────────────────────────────────────────────────────────

export interface TimeseriesContextValue {
  payload:    TimeseriesPayload | null;
  controls:   TimeseriesViewerControls;
  setControls: (partial: Partial<TimeseriesViewerControls>) => void;
  totalDuration: number;
}

export const TimeseriesContext = createContext<TimeseriesContextValue | null>(null);

export function useTimeseriesContext(): TimeseriesContextValue {
  const ctx = useContext(TimeseriesContext);
  if (!ctx) throw new Error('useTimeseriesContext must be used inside TimeseriesViewer');
  return ctx;
}

// ── Default controls ──────────────────────────────────────────────────────────

function defaultControls(payload: TimeseriesPayload | null): TimeseriesViewerControls {
  const duration  = payload ? payload.time[payload.time.length - 1]! : 30;
  const timeEnd   = Math.min(30, duration);
  const allIdxs   = payload ? payload.channels.map((_, i) => i) : [];
  const scale     = payload
    ? computeAutoScale(payload.data, 60)
    : 1;

  return {
    timeStart:           0,
    timeEnd,
    amplitudeScale:      scale,
    laneHeightPx:        60,
    firstVisibleChannel: 0,
    visibleChannelCount: 20,
    selectedChannels:    allIdxs.slice(0, 20),
    filterLowHz:         0,
    filterHighHz:        0,
  };
}

// ── Component ─────────────────────────────────────────────────────────────────

const TimeseriesViewer: FC<PluginViewerProps> = ({ data, controlsSlot }) => {
  const payload = data?.kind === 'timeseries' ? data.payload : null;

  const [controls, setControlsState] = useState<TimeseriesViewerControls>(
    () => defaultControls(payload),
  );

  const setControls = useCallback((partial: Partial<TimeseriesViewerControls>) => {
    setControlsState((prev) => ({ ...prev, ...partial }));
  }, []);

  // Re-initialise controls when a new file is loaded
  useEffect(() => {
    setControlsState(defaultControls(payload));
  }, [payload]);

  const totalDuration = payload
    ? (payload.time[payload.time.length - 1]! - payload.time[0]!)
    : 0;

  // Canvas renderer
  const canvasRef = useCanvasRenderer(payload, controls);

  // Keyboard navigation
  const containerRef = useRef<HTMLDivElement>(null);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (!payload) return;
      const windowWidth = controls.timeEnd - controls.timeStart;
      const step        = windowWidth * 0.5;

      switch (e.key) {
        case 'ArrowLeft': {
          const newStart = Math.max(0, controls.timeStart - step);
          setControls({ timeStart: newStart, timeEnd: newStart + windowWidth });
          e.preventDefault();
          break;
        }
        case 'ArrowRight': {
          const newEnd = Math.min(totalDuration, controls.timeEnd + step);
          setControls({ timeStart: newEnd - windowWidth, timeEnd: newEnd });
          e.preventDefault();
          break;
        }
        case '+':
        case '=': {
          const newWidth = Math.max(1, windowWidth / 2);
          const centre   = (controls.timeStart + controls.timeEnd) / 2;
          setControls({
            timeStart: Math.max(0, centre - newWidth / 2),
            timeEnd:   Math.min(totalDuration, centre + newWidth / 2),
          });
          e.preventDefault();
          break;
        }
        case '-': {
          const newWidth = Math.min(totalDuration, windowWidth * 2);
          const centre   = (controls.timeStart + controls.timeEnd) / 2;
          setControls({
            timeStart: Math.max(0, centre - newWidth / 2),
            timeEnd:   Math.min(totalDuration, centre + newWidth / 2),
          });
          e.preventDefault();
          break;
        }
        case 'Home':
          setControls({ timeStart: 0, timeEnd: windowWidth });
          e.preventDefault();
          break;
        case 'End': {
          const newEnd = totalDuration;
          setControls({ timeStart: Math.max(0, newEnd - windowWidth), timeEnd: newEnd });
          e.preventDefault();
          break;
        }
      }
    },
    [payload, controls, totalDuration, setControls],
  );

  const contextValue: TimeseriesContextValue = {
    payload,
    controls,
    setControls,
    totalDuration,
  };

  return (
    <TimeseriesContext.Provider value={contextValue}>
      <div className="plugin-workspace">
        <div
          ref={containerRef}
          className="ts-container"
          tabIndex={0}
          onKeyDown={handleKeyDown}
          aria-label="EEG / MEG timeseries viewer — use arrow keys to pan, +/- to zoom"
          style={{ outline: 'none' }}
        >
          {payload ? (
            <>
              <canvas
                ref={canvasRef as RefObject<HTMLCanvasElement>}
                className="ts-canvas"
                style={{ width: '100%', height: '100%', display: 'block' }}
              />
              <TimeseriesOverlay payload={payload} controls={controls} />
            </>
          ) : (
            <div className="empty-state">
              <p>Upload an EDF, BDF, or EEG file to begin</p>
            </div>
          )}
        </div>
        {controlsSlot}
      </div>
    </TimeseriesContext.Provider>
  );
};

// ── Overlay: scrollbar + zoom indicator ──────────────────────────────────────

interface OverlayProps {
  payload:  TimeseriesPayload;
  controls: TimeseriesViewerControls;
}

function TimeseriesOverlay({ payload, controls }: OverlayProps) {
  const { setControls } = useTimeseriesContext();
  const duration = payload.time[payload.time.length - 1]! - payload.time[0]!;
  if (duration <= 0) return null;

  const thumbLeft  = controls.timeStart / duration;
  const thumbWidth = (controls.timeEnd - controls.timeStart) / duration;

  // Drag state
  const draggingRef  = useRef(false);
  const dragStartRef = useRef(0);
  const dragTSRef    = useRef(0);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const bar = e.currentTarget.getBoundingClientRect();
    const relX = (e.clientX - bar.left) / bar.width;
    const newCentre = relX * duration;
    const halfWin   = (controls.timeEnd - controls.timeStart) / 2;
    const newStart  = Math.max(0, newCentre - halfWin);
    const newEnd    = Math.min(duration, newCentre + halfWin);
    setControls({ timeStart: newStart, timeEnd: newEnd });

    draggingRef.current  = true;
    dragStartRef.current = e.clientX;
    dragTSRef.current    = newStart;
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return;
    const bar     = e.currentTarget.getBoundingClientRect();
    const dx      = (e.clientX - dragStartRef.current) / bar.width;
    const dtSecs  = dx * duration;
    const win     = controls.timeEnd - controls.timeStart;
    const newStart = Math.max(0, Math.min(duration - win, dragTSRef.current + dtSecs));
    setControls({ timeStart: newStart, timeEnd: newStart + win });
  };

  const onPointerUp = () => { draggingRef.current = false; };

  return (
    <div className="ts-scrollbar"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      role="scrollbar"
      aria-orientation="horizontal"
      aria-valuenow={Math.round(controls.timeStart)}
      aria-valuemin={0}
      aria-valuemax={Math.round(duration)}
    >
      <div
        className="ts-scrollbar__thumb"
        style={{
          left:  `${(thumbLeft  * 100).toFixed(2)}%`,
          width: `${(thumbWidth * 100).toFixed(2)}%`,
        }}
      />
    </div>
  );
}

export default TimeseriesViewer;
