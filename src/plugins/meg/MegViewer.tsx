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
import type { ExperimentEvent } from '../../types/bids_events.types';
import type { ChannelTrace } from '../../services/megApi';
import { megApi } from '../../services/megApi';
import type { ArtifactAnnotation, SpikeMarker } from '../../types/analysis.types';
import { megAnalysisApi } from '../../services/megAnalysisApi';
import { typeColor, sensorPod } from '../../lib/meg/channelColors';
import {
  computeBandPowers,
  BAND_COLORS,
  BAND_ORDER,
  type BandPowers,
} from '../../lib/meg/bandPower';
import {
  computeLaneSpectrogram,
  type BandSegment,
} from '../../lib/meg/laneSpectrogram';
import MegVerticalScrollbar from './MegVerticalScrollbar';

// ── Time-axis helpers ─────────────────────────────────────────────────────────

const AXIS_H  = 32;  // px: height of the time-axis strip at the bottom
const PROX_PX = 12;  // px: event-marker hover detection radius

function niceInterval(span: number): number {
  const raw = span / 8;
  const candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1800];
  return candidates.find((c) => c >= raw) ?? candidates[candidates.length - 1]!;
}

function formatAxisTime(seconds: number, interval: number): string {
  return `${seconds.toFixed(interval < 1 ? 1 : 0)}s`;
}

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
  /** BIDS events loaded from a *_events.tsv file (drag-drop in MegControls). */
  events:           ExperimentEvent[];
  setEvents:        (events: ExperimentEvent[]) => void;
  /** When true, each channel lane background is tinted with its dominant frequency band colour. */
  bandTintEnabled:    boolean;
  setBandTintEnabled: (enabled: boolean) => void;
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

// typeColor and sensorPod are imported from ../../lib/meg/channelColors

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
  const [events,        setEvents]        = useState<ExperimentEvent[]>([]);
  /** Toggles the per-lane spectral band background tint. */
  const [bandTintEnabled,    setBandTintEnabled]    = useState(false);
  /** Pre-computed dominant-band segments per channel name — rebuilt whenever traces or tint flag change. */
  const [laneSegments,       setLaneSegments]       = useState<Map<string, BandSegment[]>>(new Map());
  const [crosshairTime,  setCrosshairTime]  = useState<number | null>(null);
  const [tooltip,        setTooltip]        = useState<{ x: number; y: number; evt: ExperimentEvent } | null>(null);
  const [channelTooltip, setChannelTooltip] = useState<{
    x: number; y: number;
    name: string; chType: string; unit: string;
    pod: string | null; siblings: string[];
  } | null>(null);
  const [jumpInput,     setJumpInput]     = useState('');
  /** Vertical scroll offset in CSS px — 0 = top of channel list. */
  const [scrollOffset,  setScrollOffset]  = useState(0);
  /** Visible waveform height (canvas height − AXIS_H), kept in sync via ResizeObserver. */
  const [viewH,         setViewH]         = useState(0);

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
    setScrollOffset(0);   // scroll back to the top when a new file is loaded
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

  // ── Track canvas height for the vertical scrollbar ─────────────────────────
  // ResizeObserver keeps viewH in sync whenever the panel is resized so the
  // scrollbar thumb proportions always match the actual visible area.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      setViewH(Math.max(0, canvas.clientHeight - AXIS_H));
    });
    ro.observe(canvas);
    // Set initial value immediately (before the first ResizeObserver callback)
    setViewH(Math.max(0, canvas.clientHeight - AXIS_H));
    return () => ro.disconnect();
  }, []);

  // ── Spectral band tint: compute dominant-band segments per channel ─────────
  // Run only when the tint is enabled or traces change; kept separate from the
  // canvas effect so scrolling and crosshair updates don't re-run the DFT.
  useEffect(() => {
    if (!bandTintEnabled || traces.length === 0) {
      setLaneSegments(new Map());
      return;
    }
    const windowDur   = controls.timeEnd - controls.timeStart;
    const effectiveSR = windowDur > 0 ? (traces[0]?.values.length ?? 0) / windowDur : 0;
    const map = new Map<string, BandSegment[]>();
    for (const trace of traces) {
      map.set(trace.name, computeLaneSpectrogram(trace.values, effectiveSR));
    }
    setLaneSegments(map);
  }, [traces, bandTintEnabled, controls.timeStart, controls.timeEnd]);

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

    const plotH    = h - AXIS_H;              // waveform area height (CSS px)
    const plotW    = w - LABEL_W;             // waveform area width
    const timeRange = controls.timeEnd - controls.timeStart;

    // Waveform background
    ctx.fillStyle = '#0d0d1a';
    ctx.fillRect(0, 0, w, plotH);

    // Time-axis strip background
    ctx.fillStyle = '#080812';
    ctx.fillRect(0, plotH, w, AXIS_H);

    if (traces.length === 0) {
      ctx.fillStyle    = '#555';
      ctx.font         = '14px system-ui, sans-serif';
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(loadingChunk ? 'Loading…' : 'No channels selected', w / 2, plotH / 2);
      // fall through — still draw the time axis
    } else {
      const laneH         = controls.laneHeightPx;
      const totalContentH = traces.length * laneH;                        // full height of all lanes
      const maxScroll     = Math.max(0, totalContentH - plotH);
      const clampedOffset = Math.min(scrollOffset, maxScroll);             // never scroll past the end

      // Effective sample rate of the decimated signal (same for all traces)
      const windowDur   = controls.timeEnd - controls.timeStart;
      const effectiveSR = windowDur > 0 ? (traces[0]?.values.length ?? 0) / windowDur : 0;

      // ── Enter scrolled coordinate space ────────────────────────────────────
      // Clip to the waveform area then translate so lane 0 moves up by the
      // scroll offset.  Everything drawn inside this save/restore block is in
      // content (lane) coordinates; the clip prevents drawing into the axis strip.
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, w, plotH);
      ctx.clip();
      ctx.translate(0, -clampedOffset);

      // ── Artifact spans ──────────────────────────────────────────────────────
      if ((artifacts.length > 0 || spikes.length > 0) && timeRange > 0) {
        const toX = (t: number) => LABEL_W + ((t - controls.timeStart) / timeRange) * plotW;

        for (const a of artifacts) {
          const x1 = toX(a.onset);
          const x2 = toX(a.onset + a.duration);
          if (x2 < LABEL_W || x1 > w) continue;
          ctx.fillStyle = a.type === 'blink'
            ? 'rgba(255, 220, 50, 0.12)'
            : 'rgba(160, 160, 160, 0.09)';
          // Span the full content height so the shade covers all lanes when scrolled
          ctx.fillRect(Math.max(LABEL_W, x1), 0, Math.min(w, x2) - Math.max(LABEL_W, x1), totalContentH);
          if (x2 - x1 > 16) {
            ctx.fillStyle = a.type === 'blink' ? 'rgba(255, 220, 50, 0.50)' : 'rgba(160, 160, 160, 0.38)';
            ctx.font = '8px monospace';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText(a.type === 'blink' ? 'blink' : 'muscle', Math.max(LABEL_W, x1) + 2, 2);
          }
        }

        // Spike lines — extend through all lanes
        ctx.save();
        ctx.strokeStyle = 'rgba(255, 60, 60, 0.60)';
        ctx.lineWidth   = 1.2;
        ctx.setLineDash([3, 3]);
        for (const s of spikes) {
          const x = Math.round(toX(s.time));
          if (x < LABEL_W || x > w) continue;
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, totalContentH); ctx.stroke();
        }
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(255, 60, 60, 0.78)';
        for (const s of spikes) {
          const x = Math.round(toX(s.time));
          if (x < LABEL_W || x > w) continue;
          ctx.beginPath();
          ctx.moveTo(x - 4, 0); ctx.lineTo(x + 4, 0); ctx.lineTo(x, 7);
          ctx.closePath(); ctx.fill();
        }
        ctx.restore();
      }

      // ── BIDS event duration shading ─────────────────────────────────────────
      if (events.length > 0 && timeRange > 0) {
        const toX = (t: number) => LABEL_W + ((t - controls.timeStart) / timeRange) * plotW;
        for (const evt of events) {
          const evtEnd = evt.onset + Math.max(evt.duration, 0.04);
          if (evtEnd < controls.timeStart || evt.onset > controls.timeEnd) continue;
          const x1 = toX(evt.onset);
          const x2 = toX(evtEnd);
          ctx.fillStyle = evt.color + '1a';
          ctx.fillRect(x1, 0, x2 - x1, totalContentH);
        }
      }

      // ── Waveform lanes ──────────────────────────────────────────────────────
      const channels    = payload?.channels ?? [];
      const legendItems: Array<{ color: string; label: string }> = [];
      const seenTypes   = new Set<string>();

      traces.forEach((trace, idx) => {
        const ch    = channels.find((c) => c.name === trace.name);
        const color = typeColor(ch?.type ?? '');
        const midY  = idx * laneH + laneH / 2;

        // Accumulate one legend entry per unique channel type (for the axis strip legend)
        const t = ch?.type ?? '';
        if (!seenTypes.has(t)) {
          seenTypes.add(t);
          legendItems.push({
            color,
            label: t === 'mag' ? 'Mag' : t === 'grad' ? 'Grad' : t || 'Ch',
          });
        }
        const n = trace.times.length;
        if (n === 0) return;

        // ── Band tint — lane background coloured by dominant frequency band ──────
        // Painted first so the waveform ribbon always sits on top.
        if (bandTintEnabled) {
          const segs = laneSegments.get(trace.name);
          if (segs && segs.length > 0) {
            ctx.save();
            ctx.globalAlpha = 0.22;
            for (const seg of segs) {
              ctx.fillStyle = BAND_COLORS[seg.band];
              ctx.fillRect(
                LABEL_W + seg.startFraction * plotW,
                idx * laneH,
                Math.max(1, (seg.endFraction - seg.startFraction) * plotW),
                laneH,
              );
            }
            ctx.restore();
          }
        }

        ctx.strokeStyle = '#1e1e3a';
        ctx.lineWidth   = 0.5;
        ctx.beginPath();
        ctx.moveTo(LABEL_W, idx * laneH);
        ctx.lineTo(w, idx * laneH);
        ctx.stroke();

        let lo = Infinity, hi = -Infinity;
        for (let i = 0; i < n; i++) {
          const mn = trace.mins[i] ?? 0; const mx = trace.maxs[i] ?? 0;
          if (mn < lo) lo = mn; if (mx > hi) hi = mx;
        }
        const range = hi - lo || 1e-12;
        const mid   = (hi + lo) / 2;
        const scale = (laneH * 0.8 / range) * controls.amplitudeScale;

        const xOf = n <= 1
          ? (_i: number) => LABEL_W + plotW / 2
          : (i: number)  => LABEL_W + (i / (n - 1)) * plotW;
        const yOf = (v: number) => midY - (v - mid) * scale;

        ctx.fillStyle = color + '28';
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const x = xOf(i); const y = yOf(trace.maxs[i] ?? 0);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        for (let i = n - 1; i >= 0; i--) ctx.lineTo(xOf(i), yOf(trace.mins[i] ?? 0));
        ctx.closePath(); ctx.fill();

        ctx.strokeStyle = color; ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const x = xOf(i); const y = yOf(trace.values[i] ?? 0);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();

        ctx.fillStyle = color; ctx.font = '10px monospace';
        ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        ctx.fillText(trace.name, 4, midY);

        // ── Band power mini-bars in the label column ──────────────────────────
        // Five 4 px-wide bars (δ θ α β γ) stacked at the right edge of the
        // 90 px label column.  Height is proportional to relative band power.
        // Computed client-side from the decimated trace.values via a 256-pt DFT.
        if (effectiveSR >= 2 && trace.values.length >= 8) {
          const powers  = computeBandPowers(trace.values, effectiveSR);
          const maxBarH = Math.max(4, laneH - 8);
          const barBase = idx * laneH + laneH - 4;   // bottom of bar area in content coords
          BAND_ORDER.forEach((band, bi) => {
            const bx = 62 + bi * 5;                             // x within the label column
            const bh = Math.max(1, (powers as BandPowers)[band] * maxBarH);
            ctx.fillStyle = BAND_COLORS[band] + 'cc';           // 80 % opacity
            ctx.fillRect(bx, barBase - bh, 4, bh);
          });
        }
      });

      // ── BIDS event onset lines + trial-type labels ──────────────────────────
      if (events.length > 0 && timeRange > 0) {
        const toX = (t: number) => LABEL_W + ((t - controls.timeStart) / timeRange) * plotW;
        ctx.save();
        for (const evt of events) {
          if (evt.onset < controls.timeStart || evt.onset > controls.timeEnd) continue;
          const ex = toX(evt.onset);
          // Line extends through all lanes (totalContentH) so it's visible when scrolled
          ctx.beginPath(); ctx.moveTo(ex, 0); ctx.lineTo(ex, totalContentH);
          ctx.strokeStyle = evt.color + 'aa'; ctx.lineWidth = 1;
          ctx.setLineDash([]); ctx.stroke();
          ctx.fillStyle = evt.color; ctx.font = '9px sans-serif';
          ctx.textAlign = 'left'; ctx.textBaseline = 'top';
          ctx.fillText(evt.trialType, ex + 2, 3);
        }
        ctx.restore();
      }

      // ── Exit scrolled coordinate space ──────────────────────────────────────
      ctx.restore();

      // ── Type legend in the label-column of the axis strip (fixed, unscrolled) ─
      // Drawn after ctx.restore() so it stays at the bottom even when scrolled.
      if (legendItems.length > 0) {
        ctx.save();
        ctx.font         = '8px monospace';
        ctx.textBaseline = 'middle';
        const totalH = legendItems.length * 11;
        let ly = plotH + (AXIS_H - totalH) / 2 + 5;
        for (const item of legendItems) {
          ctx.fillStyle = item.color;
          ctx.beginPath(); ctx.arc(6, ly, 3, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = 'rgba(180,180,200,0.65)';
          ctx.textAlign = 'left';
          ctx.fillText(item.label, 12, ly);
          ly += 11;
        }
        ctx.restore();
      }
    }

    // ── Time axis ─────────────────────────────────────────────────────────────
    ctx.strokeStyle = 'rgba(200,200,220,0.35)';
    ctx.lineWidth   = 1;
    ctx.beginPath(); ctx.moveTo(LABEL_W, plotH); ctx.lineTo(w, plotH); ctx.stroke();

    if (timeRange > 0 && plotW > 0) {
      const interval  = niceInterval(timeRange);
      const firstTick = Math.ceil(controls.timeStart / interval) * interval;

      ctx.fillStyle    = 'rgba(180,180,200,0.80)';
      ctx.strokeStyle  = 'rgba(180,180,200,0.50)';
      ctx.lineWidth    = 1;
      ctx.font         = '10px monospace';
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'top';

      for (let t = firstTick; t <= controls.timeEnd + 1e-9; t += interval) {
        const x = LABEL_W + ((t - controls.timeStart) / timeRange) * plotW;
        if (x < LABEL_W || x > w + 1) continue;
        ctx.beginPath(); ctx.moveTo(x, plotH); ctx.lineTo(x, plotH + 5); ctx.stroke();
        ctx.fillText(formatAxisTime(t, interval), x, plotH + 7);
      }

      // Event markers — colored upward-pointing triangles on the axis strip
      if (events.length > 0) {
        const TRI_H = 7;
        for (const evt of events) {
          if (evt.onset < controls.timeStart || evt.onset > controls.timeEnd) continue;
          const ex = LABEL_W + ((evt.onset - controls.timeStart) / timeRange) * plotW;
          ctx.beginPath();
          ctx.moveTo(ex,              plotH + 1);
          ctx.lineTo(ex - TRI_H / 2,  plotH + TRI_H + 1);
          ctx.lineTo(ex + TRI_H / 2,  plotH + TRI_H + 1);
          ctx.closePath();
          ctx.fillStyle = evt.color;
          ctx.fill();
        }
      }
    }

    // ── Crosshair (placed by click) ───────────────────────────────────────────
    if (crosshairTime !== null && timeRange > 0) {
      const t = crosshairTime;
      if (t >= controls.timeStart && t <= controls.timeEnd) {
        const cx = LABEL_W + ((t - controls.timeStart) / timeRange) * plotW;
        ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
        ctx.strokeStyle = 'rgba(255,255,255,0.65)';
        ctx.lineWidth   = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
        // Time label on the axis strip
        ctx.fillStyle    = 'rgba(255,255,255,0.80)';
        ctx.font         = '10px monospace';
        ctx.textAlign    = cx > w - 60 ? 'right' : 'left';
        ctx.textBaseline = 'bottom';
        ctx.fillText(`${t.toFixed(2)}s`, cx + (cx > w - 60 ? -3 : 3), h - 2);
      }
    }
  }, [traces, controls.amplitudeScale, controls.laneHeightPx, controls.timeStart, controls.timeEnd,
      loadingChunk, artifacts, spikes, events, crosshairTime, payload, scrollOffset,
      laneSegments, bandTintEnabled]);

  // ── Click handler — places crosshair ────────────────────────────────────────
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect  = canvas.getBoundingClientRect();
    const xRel  = e.clientX - rect.left - LABEL_W;
    const drawW = rect.width - LABEL_W;
    if (xRel < 0 || drawW <= 0) return;
    const t = controls.timeStart + (xRel / drawW) * (controls.timeEnd - controls.timeStart);
    setCrosshairTime(Math.max(controls.timeStart, Math.min(t, controls.timeEnd)));
  }, [controls.timeStart, controls.timeEnd]);

  // ── Hover handler — channel labels (left strip) + event-marker tooltip ────────
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) { setTooltip(null); setChannelTooltip(null); return; }
    const rect      = canvas.getBoundingClientRect();
    const domX      = e.clientX - rect.left;
    const domY      = e.clientY - rect.top;
    const plotH     = canvas.clientHeight - AXIS_H;
    const drawW     = rect.width - LABEL_W;
    const timeRange = controls.timeEnd - controls.timeStart;

    // ── Channel label hover (left strip, above axis) ──────────────────────────
    if (domX < LABEL_W && domY < plotH && traces.length > 0) {
      const laneIdx = Math.floor(domY / controls.laneHeightPx);
      const trace   = traces[laneIdx];
      if (trace) {
        const ch = payload?.channels.find((c) => c.name === trace.name);
        if (ch) {
          const pod      = sensorPod(ch.name);
          const siblings = (payload?.channels ?? [])
            .filter((c) => pod !== null && sensorPod(c.name) === pod && c.name !== ch.name)
            .map((c) => c.name);
          setChannelTooltip({ x: domX, y: domY, name: ch.name, chType: ch.type, unit: ch.unit, pod, siblings });
          setTooltip(null);
          return;
        }
      }
    }
    setChannelTooltip(null);

    // ── Event marker hover (axis strip) ──────────────────────────────────────
    if (events.length === 0 || domY < plotH || drawW <= 0 || timeRange <= 0) {
      setTooltip(null);
      return;
    }
    for (const evt of events) {
      if (evt.onset < controls.timeStart || evt.onset > controls.timeEnd) continue;
      const evtX = LABEL_W + ((evt.onset - controls.timeStart) / timeRange) * drawW;
      if (Math.abs(domX - evtX) <= PROX_PX) { setTooltip({ x: domX, y: domY, evt }); return; }
    }
    setTooltip(null);
  }, [controls.timeStart, controls.timeEnd, controls.laneHeightPx, events, traces, payload]);

  // ── Mouse-wheel vertical scroll ─────────────────────────────────────────────
  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    // Only intercept vertical wheel when there are more lanes than fit on screen
    const totalH  = traces.length * controls.laneHeightPx;
    const maxOff  = Math.max(0, totalH - viewH);
    if (maxOff <= 0) return;           // nothing to scroll — let the event bubble
    e.preventDefault();
    setScrollOffset((prev) => Math.max(0, Math.min(maxOff, prev + e.deltaY)));
  }, [traces.length, controls.laneHeightPx, viewH]);

  // ── Pan + jump callbacks ────────────────────────────────────────────────────
  const panLeft = useCallback(() => {
    const win = controls.timeEnd - controls.timeStart;
    const s   = Math.max(0, controls.timeStart - win * 0.5);
    setControls({ timeStart: s, timeEnd: s + win });
  }, [controls.timeStart, controls.timeEnd, setControls]);

  const panRight = useCallback(() => {
    const dur = payload?.totalDuration ?? 0;
    const win = controls.timeEnd - controls.timeStart;
    const e   = Math.min(dur, controls.timeEnd + win * 0.5);
    setControls({ timeStart: e - win, timeEnd: e });
  }, [controls.timeStart, controls.timeEnd, setControls, payload]);

  const doJump = useCallback(() => {
    const t = parseFloat(jumpInput);
    if (isNaN(t)) return;
    const dur = payload?.totalDuration ?? 0;
    const win = controls.timeEnd - controls.timeStart;
    const s   = Math.max(0, Math.min(t - win / 2, dur - win));
    setControls({ timeStart: s, timeEnd: s + win });
  }, [jumpInput, payload, controls.timeStart, controls.timeEnd, setControls]);

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
    events, setEvents,
    bandTintEnabled, setBandTintEnabled,
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
          style={{ outline: 'none', position: 'relative' }}
        >
          {payload ? (
            <>
              <canvas
                ref={canvasRef}
                className="ts-canvas"
                style={{ width: '100%', height: '100%', display: 'block', cursor: 'crosshair' }}
                onClick={handleCanvasClick}
                onMouseMove={handleMouseMove}
                onMouseLeave={() => { setTooltip(null); setChannelTooltip(null); }}
                onWheel={handleWheel}
              />
              {/* Vertical scrollbar — appears automatically when lanes overflow the canvas */}
              <MegVerticalScrollbar
                totalH={traces.length * controls.laneHeightPx}
                viewH={viewH}
                offset={scrollOffset}
                onChange={setScrollOffset}
              />
              {/* Event-marker hover tooltip */}
              {tooltip && (
                <div
                  className="meg-event-tooltip"
                  style={{ left: tooltip.x + 14, top: tooltip.y - 60 }}
                >
                  <span className="meg-event-tooltip__type" style={{ color: tooltip.evt.color }}>
                    {tooltip.evt.trialType}
                  </span>
                  <br />
                  <span>t = {tooltip.evt.onset.toFixed(3)} s</span>
                  {tooltip.evt.duration > 0 && (
                    <><br /><span>dur = {tooltip.evt.duration.toFixed(3)} s</span></>
                  )}
                  {tooltip.evt.responseTime != null && (
                    <><br /><span>RT = {tooltip.evt.responseTime.toFixed(3)} s</span></>
                  )}
                </div>
              )}
              {/* Channel label hover tooltip */}
              {channelTooltip && (
                <div
                  className="meg-event-tooltip"
                  style={{ left: channelTooltip.x + 14, top: Math.max(4, channelTooltip.y - 110) }}
                >
                  <span
                    className="meg-event-tooltip__type"
                    style={{ color: typeColor(channelTooltip.chType) }}
                  >
                    {channelTooltip.name}
                  </span>
                  <br />
                  <span>
                    {channelTooltip.chType === 'mag'
                      ? 'Magnetometer'
                      : channelTooltip.chType === 'grad'
                      ? 'Planar Gradiometer'
                      : channelTooltip.chType.toUpperCase()}
                  </span>
                  <br />
                  <span>Unit: {channelTooltip.unit}</span>
                  {channelTooltip.pod && (
                    <>
                      <br />
                      <span>Sensor pod: {channelTooltip.pod}</span>
                      {channelTooltip.siblings.length > 0 && (
                        <>
                          <br />
                          <span style={{ color: '#9a9ab8', fontSize: 9 }}>
                            Pod channels: {channelTooltip.siblings.join(', ')}
                          </span>
                        </>
                      )}
                    </>
                  )}
                </div>
              )}

              {/* ── Nav bar: pan arrows + jump-to-time input ── */}
              <div className="meg-viewer__nav">
                <button
                  className="btn btn--sm meg-viewer__pan-btn"
                  onClick={panLeft}
                  title="Pan left (← key)"
                  aria-label="Pan left"
                >◀</button>

                <div className="meg-viewer__jump">
                  <input
                    type="number"
                    className="meg-viewer__jump-input"
                    placeholder="time…"
                    step="0.1"
                    min={0}
                    max={payload?.totalDuration}
                    value={jumpInput}
                    onChange={(e) => setJumpInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') doJump(); }}
                    aria-label="Jump to time in seconds"
                  />
                  <span className="meg-viewer__jump-unit">s</span>
                  <button className="btn btn--sm" onClick={doJump} title="Jump to entered time">Go</button>
                </div>

                <span className="meg-viewer__time-range">
                  {controls.timeStart.toFixed(1)} – {controls.timeEnd.toFixed(1)} s
                </span>

                <button
                  className="btn btn--sm meg-viewer__pan-btn"
                  onClick={panRight}
                  title="Pan right (→ key)"
                  aria-label="Pan right"
                >▶</button>
              </div>

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
