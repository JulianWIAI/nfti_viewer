/**
 * EegViewer.tsx — Backend-powered BrainVision EEG waveform viewer
 * ─────────────────────────────────────────────────────────────────
 *
 * Streams decimated EEG channel data from the FastAPI backend on demand,
 * exactly like MegViewer does for MEG data.
 *
 * Keyboard shortcuts (focus the viewer first):
 *   ← / →   pan half a window width
 *   + / -   zoom in / out
 *   Home    jump to t = 0
 *   End     jump to end of recording
 */

import {
  type FC,
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useRef,
} from 'react';
import type { PluginViewerProps } from '../../types/plugin.types';
import type { EegSessionPayload } from '../../types/eeg.types';
import type { ChannelTrace } from '../../services/eegApi';
import { eegApi } from '../../services/eegApi';

// ── Controls & context types ──────────────────────────────────────────────────

export interface EegViewerControls {
  timeStart:        number;
  timeEnd:          number;
  selectedChannels: string[];
  amplitudeScale:   number;
  laneHeightPx:     number;
}

export interface EegContextValue {
  payload:      EegSessionPayload | null;
  controls:     EegViewerControls;
  setControls:  (partial: Partial<EegViewerControls>) => void;
  traces:       ChannelTrace[];
  loadingChunk: boolean;
}

export const EegContext = createContext<EegContextValue | null>(null);

export function useEegContext(): EegContextValue {
  const ctx = useContext(EegContext);
  if (!ctx) throw new Error('useEegContext must be used inside EegViewer');
  return ctx;
}

// ── Defaults ──────────────────────────────────────────────────────────────────

function defaultControls(payload: EegSessionPayload | null): EegViewerControls {
  const timeEnd = payload ? Math.min(10, payload.totalDuration) : 10;

  // Default: first 16 EEG-type channels; fall back to first 16 channels overall
  const eegNames = payload?.channels
    .filter((ch) => ch.type === 'eeg')
    .slice(0, 16)
    .map((ch) => ch.name) ?? [];

  const selectedChannels =
    eegNames.length > 0
      ? eegNames
      : (payload?.channels.slice(0, 16).map((ch) => ch.name) ?? []);

  return { timeStart: 0, timeEnd, selectedChannels, amplitudeScale: 1.0, laneHeightPx: 50 };
}

// ── Canvas colour palette ─────────────────────────────────────────────────────

const PALETTE = [
  '#4fc3f7', '#81c784', '#ffb74d', '#f48fb1',
  '#ce93d8', '#80cbc4', '#fff176', '#ff8a65',
  '#90caf9', '#a5d6a7', '#ffcc02', '#ef9a9a',
  '#b39ddb', '#80deea', '#e6ee9c', '#ffab91',
];

const LABEL_W = 90;

// ── Component ─────────────────────────────────────────────────────────────────

const EegViewer: FC<PluginViewerProps> = ({ data, controlsSlot }) => {
  const payload = data?.kind === 'eeg' ? data.payload : null;

  const [controls, setControlsState] = useState<EegViewerControls>(
    () => defaultControls(payload),
  );
  const [traces,       setTraces]       = useState<ChannelTrace[]>([]);
  const [loadingChunk, setLoadingChunk] = useState(false);

  const setControls = useCallback((partial: Partial<EegViewerControls>) => {
    setControlsState((prev) => ({ ...prev, ...partial }));
  }, []);

  useEffect(() => {
    setControlsState(defaultControls(payload));
    setTraces([]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [payload?.sessionId]);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const abortRef  = useRef<AbortController | null>(null);
  const timerRef  = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Chunk fetch (debounced, cancellable) ────────────────────────────────────
  useEffect(() => {
    if (!payload) return;

    if (timerRef.current) clearTimeout(timerRef.current);

    const sessionId = payload.sessionId;
    const ch = controls.selectedChannels;
    const t0 = controls.timeStart;
    const t1 = controls.timeEnd;

    timerRef.current = setTimeout(async () => {
      if (ch.length === 0) { setTraces([]); return; }

      const canvas  = canvasRef.current;
      const nPoints = canvas
        ? Math.max(200, Math.floor(canvas.clientWidth * (window.devicePixelRatio || 1)))
        : 600;

      abortRef.current?.abort();
      abortRef.current = new AbortController();

      setLoadingChunk(true);
      try {
        const resp = await eegApi.getChannelData(
          sessionId, ch, t0, t1, nPoints, abortRef.current.signal,
        );
        setTraces(resp.channels);
      } catch (err) {
        if (err instanceof Error && err.name !== 'AbortError') {
          console.error('EEG chunk fetch failed:', err);
        }
      } finally {
        setLoadingChunk(false);
      }
    }, 80);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [payload?.sessionId, controls.selectedChannels, controls.timeStart, controls.timeEnd]);

  // ── Canvas render ───────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const w   = canvas.clientWidth;
    const h   = canvas.clientHeight;
    if (w === 0 || h === 0) return;

    canvas.width  = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = '#0d0d1a';
    ctx.fillRect(0, 0, w, h);

    if (traces.length === 0) {
      ctx.fillStyle = '#555';
      ctx.font = '14px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(loadingChunk ? 'Loading…' : 'No channels selected', w / 2, h / 2);
      return;
    }

    const plotW = w - LABEL_W;
    const laneH = controls.laneHeightPx;

    traces.forEach((trace, idx) => {
      const color = PALETTE[idx % PALETTE.length]!;
      const midY  = idx * laneH + laneH / 2;
      const n     = trace.times.length;
      if (n === 0) return;

      // Lane separator
      ctx.strokeStyle = '#1e1e3a';
      ctx.lineWidth   = 0.5;
      ctx.beginPath();
      ctx.moveTo(LABEL_W, idx * laneH);
      ctx.lineTo(w,       idx * laneH);
      ctx.stroke();

      // Per-channel auto-scale
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < n; i++) {
        if ((trace.mins[i] ?? 0) < lo) lo = trace.mins[i] ?? 0;
        if ((trace.maxs[i] ?? 0) > hi) hi = trace.maxs[i] ?? 0;
      }
      const range = hi - lo || 1e-30;
      const mid   = (hi + lo) / 2;
      const scale = (laneH * 0.8 / range) * controls.amplitudeScale;

      const xOf = n <= 1
        ? (_i: number) => LABEL_W + plotW / 2
        : (i: number)  => LABEL_W + (i / (n - 1)) * plotW;
      const yOf = (v: number) => midY - (v - mid) * scale;

      // Ribbon fill
      ctx.fillStyle = color + '28';
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = xOf(i); const y = yOf(trace.maxs[i] ?? 0);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      for (let i = n - 1; i >= 0; i--) {
        ctx.lineTo(xOf(i), yOf(trace.mins[i] ?? 0));
      }
      ctx.closePath();
      ctx.fill();

      // Centre line
      ctx.strokeStyle = color;
      ctx.lineWidth   = 1;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = xOf(i); const y = yOf(trace.values[i] ?? 0);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Channel label
      ctx.fillStyle    = color;
      ctx.font         = '10px monospace';
      ctx.textAlign    = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(trace.name, 4, midY);
    });
  }, [traces, controls.amplitudeScale, controls.laneHeightPx, loadingChunk]);

  // ── Keyboard navigation ─────────────────────────────────────────────────────
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (!payload) return;
      const win = controls.timeEnd - controls.timeStart;
      const dur = payload.totalDuration;

      switch (e.key) {
        case 'ArrowLeft': {
          const s = Math.max(0, controls.timeStart - win * 0.5);
          setControls({ timeStart: s, timeEnd: s + win });
          e.preventDefault(); break;
        }
        case 'ArrowRight': {
          const end = Math.min(dur, controls.timeEnd + win * 0.5);
          setControls({ timeStart: end - win, timeEnd: end });
          e.preventDefault(); break;
        }
        case '+': case '=': {
          const nw = Math.max(0.1, win / 2);
          const c  = (controls.timeStart + controls.timeEnd) / 2;
          setControls({ timeStart: Math.max(0, c - nw / 2), timeEnd: Math.min(dur, c + nw / 2) });
          e.preventDefault(); break;
        }
        case '-': {
          const nw = Math.min(dur, win * 2);
          const c  = (controls.timeStart + controls.timeEnd) / 2;
          setControls({ timeStart: Math.max(0, c - nw / 2), timeEnd: Math.min(dur, c + nw / 2) });
          e.preventDefault(); break;
        }
        case 'Home':
          setControls({ timeStart: 0, timeEnd: win });
          e.preventDefault(); break;
        case 'End':
          setControls({ timeStart: Math.max(0, dur - win), timeEnd: dur });
          e.preventDefault(); break;
      }
    },
    [payload, controls, setControls],
  );

  const totalDuration = payload?.totalDuration ?? 0;
  const thumbLeft     = totalDuration > 0 ? controls.timeStart / totalDuration : 0;
  const thumbWidth    = totalDuration > 0
    ? (controls.timeEnd - controls.timeStart) / totalDuration
    : 1;

  const contextValue: EegContextValue = { payload, controls, setControls, traces, loadingChunk };

  return (
    <EegContext.Provider value={contextValue}>
      <div className="plugin-workspace">
        <div
          className="ts-container"
          tabIndex={0}
          onKeyDown={handleKeyDown}
          aria-label="EEG viewer — arrow keys to pan, +/- to zoom"
          style={{ outline: 'none' }}
        >
          {payload ? (
            <>
              <canvas
                ref={canvasRef}
                className="ts-canvas"
                style={{ width: '100%', height: '100%', display: 'block' }}
              />
              <EegScrollbar
                totalDuration={totalDuration}
                controls={controls}
                setControls={setControls}
                thumbLeft={thumbLeft}
                thumbWidth={thumbWidth}
              />
            </>
          ) : (
            <div className="empty-state">
              <p>Drop the .vhdr, .eeg and .vmrk files together to load EEG data</p>
            </div>
          )}
        </div>
        {controlsSlot}
      </div>
    </EegContext.Provider>
  );
};

// ── Scrollbar ─────────────────────────────────────────────────────────────────

interface ScrollbarProps {
  totalDuration: number;
  controls:      EegViewerControls;
  setControls:   (p: Partial<EegViewerControls>) => void;
  thumbLeft:     number;
  thumbWidth:    number;
}

function EegScrollbar({ totalDuration, controls, setControls, thumbLeft, thumbWidth }: ScrollbarProps) {
  const draggingRef  = useRef(false);
  const dragStartRef = useRef(0);
  const dragTSRef    = useRef(0);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const bar    = e.currentTarget.getBoundingClientRect();
    const relX   = (e.clientX - bar.left) / bar.width;
    const centre = relX * totalDuration;
    const half   = (controls.timeEnd - controls.timeStart) / 2;
    const s      = Math.max(0, centre - half);
    const end    = Math.min(totalDuration, centre + half);
    setControls({ timeStart: s, timeEnd: end });
    draggingRef.current  = true;
    dragStartRef.current = e.clientX;
    dragTSRef.current    = s;
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return;
    const bar = e.currentTarget.getBoundingClientRect();
    const dx  = (e.clientX - dragStartRef.current) / bar.width;
    const dt  = dx * totalDuration;
    const win = controls.timeEnd - controls.timeStart;
    const s   = Math.max(0, Math.min(totalDuration - win, dragTSRef.current + dt));
    setControls({ timeStart: s, timeEnd: s + win });
  };

  const onPointerUp = () => { draggingRef.current = false; };

  return (
    <div
      className="ts-scrollbar"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      role="scrollbar"
      aria-orientation="horizontal"
      aria-valuenow={Math.round(controls.timeStart)}
      aria-valuemin={0}
      aria-valuemax={Math.round(totalDuration)}
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

export default EegViewer;
