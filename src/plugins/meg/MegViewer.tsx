/**
 * MegViewer.tsx — Backend-powered MEG waveform viewer
 * ──────────────────────────────────────────────────────
 *
 * Unlike the EEG viewer (which holds all data in memory), MegViewer streams
 * decimated chunks from the FastAPI backend on demand. The backend returns
 * min/max envelopes so the canvas can draw a ribbon at any zoom level without
 * losing transients.
 *
 * Data flow:
 *   controls change (pan / zoom / channel select)
 *     → debounced HTTP GET /api/meg/channels (AbortController cancels previous)
 *     → setTraces(resp.channels)
 *     → canvas renders ribbons from ChannelTrace[]
 *
 * Layout (same as TimeseriesViewer — reuses .ts-container / .ts-canvas CSS):
 *   ┌──────────────────────────────────────┐
 *   │  <canvas>  — fills parent            │
 *   │    90px label | waveform ribbons     │
 *   └──────────────────────────────────────┘
 *   [ timeline scrollbar                   ]
 *
 * Keyboard shortcuts (focus the viewer first):
 *   ← / →   pan half a window width
 *   + / -   zoom in / out (halve / double window)
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
import type { MegSessionPayload } from '../../types/meg.types';
import type { ChannelTrace } from '../../services/megApi';
import { megApi } from '../../services/megApi';
import type { ArtifactAnnotation, SpikeMarker } from '../../types/analysis.types';
import { megAnalysisApi } from '../../services/megAnalysisApi';

// ── Controls & context types ──────────────────────────────────────────────────

export interface MegViewerControls {
  timeStart:        number;
  timeEnd:          number;
  selectedChannels: string[];
  amplitudeScale:   number;
  laneHeightPx:     number;
}

export interface MegContextValue {
  payload:          MegSessionPayload | null;
  controls:         MegViewerControls;
  setControls:      (partial: Partial<MegViewerControls>) => void;
  traces:           ChannelTrace[];
  loadingChunk:     boolean;
  /** Artefact annotations returned by /api/meg/detect-artifacts. */
  artifacts:        ArtifactAnnotation[];
  /** Spike markers returned by /api/meg/detect-spikes. */
  spikes:           SpikeMarker[];
  artifactsLoading: boolean;
  spikesLoading:    boolean;
  /** True once detect-artifacts has completed at least once (success or error). */
  artifactsDone:    boolean;
  /** True once detect-spikes has completed at least once (success or error). */
  spikesDone:       boolean;
  artifactsError:   string | null;
  spikesError:      string | null;
  detectArtifacts:  () => Promise<void>;
  detectSpikes:     () => Promise<void>;
}

export const MegContext = createContext<MegContextValue | null>(null);

export function useMegContext(): MegContextValue {
  const ctx = useContext(MegContext);
  if (!ctx) throw new Error('useMegContext must be used inside MegViewer');
  return ctx;
}

// ── Defaults ──────────────────────────────────────────────────────────────────

function defaultControls(payload: MegSessionPayload | null): MegViewerControls {
  const timeEnd = payload ? Math.min(5, payload.totalDuration) : 5;

  // Prefer first 10 magnetometers; fall back to first 10 channels
  const magNames = payload?.channels
    .filter((ch) => ch.type === 'mag')
    .slice(0, 10)
    .map((ch) => ch.name) ?? [];

  const selectedChannels =
    magNames.length > 0
      ? magNames
      : (payload?.channels.slice(0, 10).map((ch) => ch.name) ?? []);

  return { timeStart: 0, timeEnd, selectedChannels, amplitudeScale: 1.0, laneHeightPx: 60 };
}

// ── Canvas colour palette ─────────────────────────────────────────────────────

const PALETTE = [
  '#4fc3f7', '#81c784', '#ffb74d', '#f48fb1',
  '#ce93d8', '#80cbc4', '#fff176', '#ff8a65',
  '#90caf9', '#a5d6a7', '#ffcc02', '#ef9a9a',
  '#b39ddb', '#80deea', '#e6ee9c', '#ffab91',
];

const LABEL_W = 90; // px reserved on the left for channel names

// ── Component ─────────────────────────────────────────────────────────────────

const MegViewer: FC<PluginViewerProps> = ({ data, controlsSlot }) => {
  const payload = data?.kind === 'meg' ? data.payload : null;

  const [controls, setControlsState] = useState<MegViewerControls>(
    () => defaultControls(payload),
  );
  const [traces,           setTraces]           = useState<ChannelTrace[]>([]);
  const [loadingChunk,     setLoadingChunk]     = useState(false);
  const [artifacts,        setArtifacts]        = useState<ArtifactAnnotation[]>([]);
  const [spikes,           setSpikes]           = useState<SpikeMarker[]>([]);
  const [artifactsLoading, setArtifactsLoading] = useState(false);
  const [spikesLoading,    setSpikesLoading]    = useState(false);
  const [artifactsDone,    setArtifactsDone]    = useState(false);
  const [spikesDone,       setSpikesDone]       = useState(false);
  const [artifactsError,   setArtifactsError]   = useState<string | null>(null);
  const [spikesError,      setSpikesError]      = useState<string | null>(null);

  const setControls = useCallback((partial: Partial<MegViewerControls>) => {
    setControlsState((prev) => ({ ...prev, ...partial }));
  }, []);

  // Reset state when a new file is loaded
  useEffect(() => {
    setControlsState(defaultControls(payload));
    setTraces([]);
    setArtifacts([]);
    setSpikes([]);
    setArtifactsDone(false);
    setSpikesDone(false);
    setArtifactsError(null);
    setSpikesError(null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [payload?.sessionId]);

  const detectArtifacts = useCallback(async () => {
    if (!payload) return;
    setArtifactsLoading(true);
    setArtifactsError(null);
    try {
      const res = await megAnalysisApi.detectArtifacts(payload.sessionId);
      setArtifacts(res.annotations);
      setArtifactsDone(true);
    } catch (err) {
      setArtifactsError(err instanceof Error ? err.message : String(err));
      setArtifactsDone(true);
    } finally {
      setArtifactsLoading(false);
    }
  }, [payload]);

  const detectSpikes = useCallback(async () => {
    if (!payload) return;
    setSpikesLoading(true);
    setSpikesError(null);
    try {
      const res = await megAnalysisApi.detectSpikes(payload.sessionId);
      setSpikes(res.spikes);
      setSpikesDone(true);
    } catch (err) {
      setSpikesError(err instanceof Error ? err.message : String(err));
      setSpikesDone(true);
    } finally {
      setSpikesLoading(false);
    }
  }, [payload]);

  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const abortRef   = useRef<AbortController | null>(null);
  const timerRef   = useRef<ReturnType<typeof setTimeout> | null>(null);

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

      const canvas = canvasRef.current;
      const nPoints = canvas
        ? Math.max(200, Math.floor(canvas.clientWidth * (window.devicePixelRatio || 1)))
        : 600;

      abortRef.current?.abort();
      abortRef.current = new AbortController();

      setLoadingChunk(true);
      try {
        const resp = await megApi.getChannelData(
          sessionId, ch, t0, t1, nPoints, abortRef.current.signal,
        );
        setTraces(resp.channels);
      } catch (err) {
        if (err instanceof Error && err.name !== 'AbortError') {
          console.error('MEG chunk fetch failed:', err);
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

    // Background
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

    const plotW   = w - LABEL_W;
    const laneH   = controls.laneHeightPx;

    // ── Artifact spans and spike markers (drawn before waveforms) ─────────
    const timeRange = controls.timeEnd - controls.timeStart;
    if ((artifacts.length > 0 || spikes.length > 0) && timeRange > 0) {
      const toX = (t: number) =>
        LABEL_W + ((t - controls.timeStart) / timeRange) * plotW;

      // Blink → translucent yellow  |  muscle → translucent grey
      for (const a of artifacts) {
        const x1 = toX(a.onset);
        const x2 = toX(a.onset + a.duration);
        if (x2 < LABEL_W || x1 > w) continue;
        ctx.fillStyle = a.type === 'blink'
          ? 'rgba(255, 220, 50, 0.12)'
          : 'rgba(160, 160, 160, 0.09)';
        ctx.fillRect(
          Math.max(LABEL_W, x1), 0,
          Math.min(w, x2) - Math.max(LABEL_W, x1),
          h,
        );
        if (x2 - x1 > 16) {
          ctx.fillStyle = a.type === 'blink'
            ? 'rgba(255, 220, 50, 0.50)'
            : 'rgba(160, 160, 160, 0.38)';
          ctx.font = '8px monospace';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'top';
          ctx.fillText(
            a.type === 'blink' ? 'blink' : 'muscle',
            Math.max(LABEL_W, x1) + 2, 2,
          );
        }
      }

      // Spike lines — red dashed verticals + downward triangle marker
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 60, 60, 0.60)';
      ctx.lineWidth   = 1.2;
      ctx.setLineDash([3, 3]);
      for (const s of spikes) {
        const x = Math.round(toX(s.time));
        if (x < LABEL_W || x > w) continue;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(255, 60, 60, 0.78)';
      for (const s of spikes) {
        const x = Math.round(toX(s.time));
        if (x < LABEL_W || x > w) continue;
        ctx.beginPath();
        ctx.moveTo(x - 4, 0);
        ctx.lineTo(x + 4, 0);
        ctx.lineTo(x,     7);
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();
    }

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

      // Per-channel auto-scale: fit full amplitude into 80 % of lane height
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < n; i++) {
        const mn = trace.mins[i] ?? 0;
        const mx = trace.maxs[i] ?? 0;
        if (mn < lo) lo = mn;
        if (mx > hi) hi = mx;
      }
      const range = hi - lo || 1e-12;
      const mid   = (hi + lo) / 2;
      const scale = (laneH * 0.8 / range) * controls.amplitudeScale;

      const xOf = n <= 1
        ? (_i: number) => LABEL_W + plotW / 2
        : (i: number)  => LABEL_W + (i / (n - 1)) * plotW;
      const yOf = (v: number) => midY - (v - mid) * scale;

      // Ribbon fill (min–max envelope)
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
  }, [traces, controls.amplitudeScale, controls.laneHeightPx, controls.timeStart, controls.timeEnd, loadingChunk, artifacts, spikes]);

  // ── Keyboard navigation ─────────────────────────────────────────────────────
  const containerRef = useRef<HTMLDivElement>(null);

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

  const contextValue: MegContextValue = {
    payload, controls, setControls, traces, loadingChunk,
    artifacts, spikes, artifactsLoading, spikesLoading,
    artifactsDone, spikesDone, artifactsError, spikesError,
    detectArtifacts, detectSpikes,
  };

  return (
    <MegContext.Provider value={contextValue}>
      <div className="plugin-workspace">
        <div
          ref={containerRef}
          className="ts-container"
          tabIndex={0}
          onKeyDown={handleKeyDown}
          aria-label="MEG viewer — arrow keys to pan, +/- to zoom"
          style={{ outline: 'none' }}
        >
          {payload ? (
            <>
              <canvas
                ref={canvasRef}
                className="ts-canvas"
                style={{ width: '100%', height: '100%', display: 'block' }}
              />
              <MegScrollbar
                totalDuration={totalDuration}
                controls={controls}
                setControls={setControls}
                thumbLeft={thumbLeft}
                thumbWidth={thumbWidth}
              />
            </>
          ) : (
            <div className="empty-state">
              <p>Upload a .fif MEG file to begin</p>
            </div>
          )}
        </div>
        {controlsSlot}
      </div>
    </MegContext.Provider>
  );
};

// ── Scrollbar ─────────────────────────────────────────────────────────────────

interface ScrollbarProps {
  totalDuration: number;
  controls:      MegViewerControls;
  setControls:   (p: Partial<MegViewerControls>) => void;
  thumbLeft:     number;
  thumbWidth:    number;
}

function MegScrollbar({ totalDuration, controls, setControls, thumbLeft, thumbWidth }: ScrollbarProps) {
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

export default MegViewer;
