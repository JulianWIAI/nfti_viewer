/**
 * MegPanel.tsx — MEG waveform panel for the multimodal workspace
 * ───────────────────────────────────────────────────────────────
 *
 * A compact, self-contained waveform canvas adapted from MegViewer.tsx for
 * use inside MultimodalWorkspace.  Key additions over the standalone viewer:
 *
 *   • Canvas click → writes currentTimeSec to SyncContext so the fMRI pane
 *     snaps to the BOLD volume corresponding to the clicked time point.
 *   • A thin vertical cursor line tracks the synced time position.
 *   • A labeled time axis (seconds) at the bottom of the canvas.
 *   • BIDS event-onset markers (colored triangles) on the time axis.
 *   • Hover tooltip on event markers showing trial type, onset, and RT.
 *
 * Data flow:
 *   MEG session loaded (MegSessionPayload)
 *     → megApi.getChannels() on window change
 *     → canvas draws waveform ribbons
 *     → user click → useSyncContext().setCurrentTimeSec(t)
 *     → FmriPanel.setTimepoint(Math.floor(t / tr))
 *
 * Layout:
 *   ┌─────────────────────────────────────────────┐
 *   │  Canvas (fills panel, ribbon per channel)   │
 *   │  Cursor line at currentTimeSec position     │
 *   ├─────────────────────────────────────────────┤  ← AXIS_H px
 *   │  Time axis: tick marks · event triangles    │
 *   └─────────────────────────────────────────────┘
 *   [ Pan controls bar ]
 */

import {
  type FC,
  useEffect,
  useRef,
  useCallback,
  useState,
} from 'react';
import { useSyncContext } from '../../contexts/SyncContext';
import type { MegSessionPayload } from '../../types/meg.types';
import type { ExperimentEvent } from '../../types/bids_events.types';
import type { ChannelTrace, ChannelDataResponse } from '../../services/megApi';
import { megApi } from '../../services/megApi';
import { typeColor } from '../../lib/meg/channelColors';
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
import MegVerticalScrollbar from '../meg/MegVerticalScrollbar';

// ── Layout constants ──────────────────────────────────────────────────────────

const LABEL_W  = 80;   // px reserved for channel names on the left
const AXIS_H   = 32;   // px for the time-axis strip at the bottom
const PROX_PX  = 12;   // px radius for event-marker hover detection
/** Fixed lane height.  Scroll kicks in automatically when lanes overflow. */
const LANE_H_PX = 52;  // px per channel lane

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Returns a "nice" tick interval that gives roughly 6–10 ticks. */
function niceInterval(span: number): number {
  const raw = span / 8;
  const candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1800];
  return candidates.find((c) => c >= raw) ?? candidates[candidates.length - 1]!;
}

/** Formats an absolute time in seconds as a concise axis label. */
function formatAxisTime(seconds: number, interval: number): string {
  const decimals = interval < 1 ? 1 : 0;
  return `${seconds.toFixed(decimals)}s`;
}

// ── Tooltip state ─────────────────────────────────────────────────────────────

interface TooltipState {
  /** Canvas-relative X (px) of the hovered event marker. */
  x:   number;
  /** Canvas-relative Y (px) of the hovered event marker. */
  y:   number;
  evt: ExperimentEvent;
}

// ── Component ─────────────────────────────────────────────────────────────────

interface MegPanelProps {
  /** Loaded MEG session payload; null if no MEG file is loaded yet. */
  payload: MegSessionPayload | null;
}

const MegPanel: FC<MegPanelProps> = ({ payload }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Time window state (seconds) for the visible waveform window
  const [timeStart,    setTimeStart]    = useState(0);
  const [timeEnd,      setTimeEnd]      = useState(5);
  const [traces,       setTraces]       = useState<ChannelTrace[]>([]);
  const [tooltip,      setTooltip]      = useState<TooltipState | null>(null);
  const [jumpInput,    setJumpInput]    = useState('');
  /** Vertical scroll offset in px — 0 = top of channel list. */
  const [scrollOffset, setScrollOffset] = useState(0);
  /** Visible waveform area height (canvas height − AXIS_H), updated by ResizeObserver. */
  const [viewH,        setViewH]        = useState(0);
  /** Toggles the per-lane spectral band background tint. */
  const [bandTintEnabled, setBandTintEnabled] = useState(false);
  /** Pre-computed dominant-band segments per channel name. */
  const [laneSegments,    setLaneSegments]    = useState<Map<string, BandSegment[]>>(new Map());

  // Time cursor + BIDS events from SyncContext
  const { currentTimeSec, setCurrentTimeSec, events, activeEvent } = useSyncContext();

  // ── Select the first 8 magnetometers (or first 8 channels) ───────────────
  const selectedChannels: string[] = payload
    ? (payload.channels.filter((c) => c.type === 'mag').slice(0, 8).map((c) => c.name).length > 0
        ? payload.channels.filter((c) => c.type === 'mag').slice(0, 8).map((c) => c.name)
        : payload.channels.slice(0, 8).map((c) => c.name))
    : [];

  // ── Fetch waveform data when session or window changes ────────────────────
  useEffect(() => {
    if (!payload || selectedChannels.length === 0) return;

    let aborted = false;
    const ctrl  = new AbortController();

    megApi.getChannelData(
      payload.sessionId,
      selectedChannels,
      timeStart,
      timeEnd,
      600,
    ).then((resp: ChannelDataResponse) => {
      if (!aborted) setTraces(resp.channels);
    }).catch(() => { /* ignore abort / network errors silently */ });

    return () => { aborted = true; ctrl.abort(); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [payload?.sessionId, timeStart, timeEnd]);

  // Reset window and scroll position when a new payload is loaded
  useEffect(() => {
    if (!payload) return;
    setTimeStart(0);
    setTimeEnd(Math.min(5, payload.totalDuration));
    setTraces([]);
    setScrollOffset(0);
  }, [payload?.sessionId]);

  // Keep viewH in sync with the canvas size for the vertical scrollbar
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      setViewH(Math.max(0, canvas.clientHeight - AXIS_H));
    });
    ro.observe(canvas);
    setViewH(Math.max(0, canvas.clientHeight - AXIS_H));
    return () => ro.disconnect();
  }, []);

  // ── Spectral tint: compute dominant-band segments per channel ─────────────
  // Separated from the canvas effect so scroll/cursor updates don't re-run DFTs.
  useEffect(() => {
    if (!bandTintEnabled || traces.length === 0) {
      setLaneSegments(new Map());
      return;
    }
    const windowSec   = timeEnd - timeStart;
    const effectiveSR = windowSec > 0 ? (traces[0]?.values.length ?? 0) / windowSec : 0;
    const map = new Map<string, BandSegment[]>();
    for (const trace of traces) {
      map.set(trace.name, computeLaneSpectrogram(trace.values, effectiveSR));
    }
    setLaneSegments(map);
  }, [traces, bandTintEnabled, timeStart, timeEnd]);

  // ── Canvas render ─────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx       = canvas.getContext('2d');
    if (!ctx) return;

    const W         = canvas.width  = canvas.clientWidth;
    const H         = canvas.height = canvas.clientHeight;
    const PLOT_H    = H - AXIS_H;
    const nCh       = traces.length;
    const windowSec = timeEnd - timeStart;
    const drawW     = W - LABEL_W;

    // Lane height: fixed LANE_H_PX when there are many channels; auto-expand
    // to fill the canvas when there are few so the panel never looks sparse.
    const laneH = nCh > 0
      ? Math.max(LANE_H_PX, Math.floor(PLOT_H / nCh))  // fills canvas if few channels
      : LANE_H_PX;

    // Scroll geometry
    const totalContentH = nCh * laneH;
    const maxScroll     = Math.max(0, totalContentH - PLOT_H);
    const clampedOffset = Math.min(scrollOffset, maxScroll);

    // Effective sample rate for band power estimation
    const effectiveSR = windowSec > 0 && nCh > 0
      ? (traces[0]?.values.length ?? 0) / windowSec
      : 0;

    // Backgrounds
    ctx.fillStyle = '#12121f';
    ctx.fillRect(0, 0, W, PLOT_H);
    ctx.fillStyle = '#0d0d1a';
    ctx.fillRect(0, PLOT_H, W, AXIS_H);

    if (nCh === 0) {
      ctx.fillStyle    = '#444';
      ctx.font         = '12px sans-serif';
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(payload ? 'Loading…' : 'No MEG session loaded', W / 2, PLOT_H / 2);
    } else {
      // ── Enter scrolled coordinate space ──────────────────────────────────────
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, W, PLOT_H);
      ctx.clip();
      ctx.translate(0, -clampedOffset);

      // ── BIDS event duration shading ─────────────────────────────────────────
      for (const evt of events) {
        const evtEnd = evt.onset + Math.max(evt.duration, 0.04);
        if (evtEnd < timeStart || evt.onset > timeEnd) continue;
        const x1 = LABEL_W + ((evt.onset - timeStart) / windowSec) * drawW;
        const x2 = LABEL_W + ((evtEnd  - timeStart) / windowSec) * drawW;
        ctx.fillStyle = evt.color + '20';
        ctx.fillRect(x1, 0, x2 - x1, totalContentH);
      }

      // ── Waveform lanes ──────────────────────────────────────────────────────
      traces.forEach((trace, ci) => {
        const colour = typeColor(
          payload?.channels.find((c) => c.name === trace.name)?.type ?? '',
        );
        const yBase = ci * laneH + laneH / 2;
        const n     = trace.times.length;
        if (n === 0) return;

        const vals = trace.maxs.concat(trace.mins);
        const aMax = Math.max(...vals.map(Math.abs)) || 1e-12;
        const scale = (laneH * 0.4) / aMax;

        // ── Band tint — lane background coloured by dominant frequency band ──────
        if (bandTintEnabled) {
          const segs = laneSegments.get(trace.name);
          if (segs && segs.length > 0) {
            ctx.save();
            ctx.globalAlpha = 0.22;
            for (const seg of segs) {
              ctx.fillStyle = BAND_COLORS[seg.band];
              ctx.fillRect(
                LABEL_W + seg.startFraction * drawW,
                ci * laneH,
                Math.max(1, (seg.endFraction - seg.startFraction) * drawW),
                laneH,
              );
            }
            ctx.restore();
          }
        }

        // Lane separator
        ctx.strokeStyle = '#1a1a2e';
        ctx.lineWidth   = 0.5;
        ctx.beginPath();
        ctx.moveTo(LABEL_W, ci * laneH);
        ctx.lineTo(W, ci * laneH);
        ctx.stroke();

        // Ribbon fill
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const x  = LABEL_W + ((trace.times[i]! - timeStart) / windowSec) * drawW;
          const yH = yBase - trace.maxs[i]! * scale;
          if (i === 0) ctx.moveTo(x, yH); else ctx.lineTo(x, yH);
        }
        for (let i = n - 1; i >= 0; i--) {
          const x  = LABEL_W + ((trace.times[i]! - timeStart) / windowSec) * drawW;
          const yL = yBase - trace.mins[i]! * scale;
          ctx.lineTo(x, yL);
        }
        ctx.closePath();
        ctx.fillStyle = colour + '40';
        ctx.fill();

        // Mean line
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const x = LABEL_W + ((trace.times[i]! - timeStart) / windowSec) * drawW;
          const y = yBase - trace.values[i]! * scale;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = colour;
        ctx.lineWidth   = 1;
        ctx.stroke();

        // Channel name label
        ctx.fillStyle    = colour;
        ctx.font         = '10px sans-serif';
        ctx.textAlign    = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(trace.name, 4, yBase);

        // ── Band power mini-bars ────────────────────────────────────────────────
        // Five 4 px-wide bars (δ θ α β γ) at the right edge of the label column.
        if (effectiveSR >= 2 && trace.values.length >= 8) {
          const powers  = computeBandPowers(trace.values, effectiveSR);
          const maxBarH = Math.max(4, laneH - 8);
          const barBase = ci * laneH + laneH - 4;
          BAND_ORDER.forEach((band, bi) => {
            const bx = 55 + bi * 5;
            const bh = Math.max(1, (powers as BandPowers)[band] * maxBarH);
            ctx.fillStyle = BAND_COLORS[band] + 'cc';
            ctx.fillRect(bx, barBase - bh, 4, bh);
          });
        }
      });

      // ── BIDS event onset lines + labels ────────────────────────────────────
      ctx.save();
      for (const evt of events) {
        if (evt.onset < timeStart || evt.onset > timeEnd) continue;
        const ex       = LABEL_W + ((evt.onset - timeStart) / windowSec) * drawW;
        const isActive = activeEvent?.id === evt.id;
        ctx.beginPath();
        ctx.moveTo(ex, 0);
        ctx.lineTo(ex, totalContentH);
        ctx.strokeStyle = isActive ? evt.color : evt.color + 'aa';
        ctx.lineWidth   = isActive ? 2 : 1;
        ctx.setLineDash([]);
        ctx.stroke();
        ctx.fillStyle    = evt.color;
        ctx.font         = '9px sans-serif';
        ctx.textAlign    = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(evt.trialType, ex + 2, 3);
      }
      ctx.restore();

      // ── Exit scrolled coordinate space ──────────────────────────────────────
      ctx.restore();
    }

    // ── Time axis (unscrolled) ─────────────────────────────────────────────────
    ctx.strokeStyle = 'rgba(200,200,220,0.35)';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(LABEL_W, PLOT_H);
    ctx.lineTo(W, PLOT_H);
    ctx.stroke();

    if (windowSec > 0 && drawW > 0) {
      const interval  = niceInterval(windowSec);
      const firstTick = Math.ceil(timeStart / interval) * interval;

      ctx.fillStyle    = 'rgba(180,180,200,0.75)';
      ctx.strokeStyle  = 'rgba(180,180,200,0.45)';
      ctx.lineWidth    = 1;
      ctx.font         = '10px monospace';
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'top';

      for (let t = firstTick; t <= timeEnd + 1e-9; t += interval) {
        const x = LABEL_W + ((t - timeStart) / windowSec) * drawW;
        if (x < LABEL_W || x > W + 1) continue;
        ctx.beginPath();
        ctx.moveTo(x, PLOT_H);
        ctx.lineTo(x, PLOT_H + 5);
        ctx.stroke();
        ctx.fillText(formatAxisTime(t, interval), x, PLOT_H + 7);
      }

      const TRI_H = 7;
      for (const evt of events) {
        if (evt.onset < timeStart || evt.onset > timeEnd) continue;
        const ex = LABEL_W + ((evt.onset - timeStart) / windowSec) * drawW;
        ctx.beginPath();
        ctx.moveTo(ex,              PLOT_H + 1);
        ctx.lineTo(ex - TRI_H / 2, PLOT_H + TRI_H + 1);
        ctx.lineTo(ex + TRI_H / 2, PLOT_H + TRI_H + 1);
        ctx.closePath();
        ctx.fillStyle = evt.color;
        ctx.fill();
      }
    }

    // ── Cursor line (unscrolled — spans full canvas height) ───────────────────
    if (windowSec > 0 && currentTimeSec >= timeStart && currentTimeSec <= timeEnd) {
      const cx = LABEL_W + ((currentTimeSec - timeStart) / windowSec) * drawW;
      ctx.beginPath();
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, H);
      ctx.strokeStyle = '#ffffff88';
      ctx.lineWidth   = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [traces, currentTimeSec, timeStart, timeEnd, payload, events, activeEvent, scrollOffset,
      laneSegments, bandTintEnabled]);

  // ── Click handler — maps canvas x-position to time, writes to SyncContext ─
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect  = canvas.getBoundingClientRect();
    const xRel  = e.clientX - rect.left - LABEL_W;
    const drawW = rect.width - LABEL_W;
    if (xRel < 0 || drawW <= 0) return;

    const t = timeStart + (xRel / drawW) * (timeEnd - timeStart);
    setCurrentTimeSec(Math.max(0, Math.min(t, payload?.totalDuration ?? t)));
  }, [timeStart, timeEnd, payload, setCurrentTimeSec]);

  // ── Hover handler — shows tooltip when near an event marker on the axis ──
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || events.length === 0) { setTooltip(null); return; }

    const rect      = canvas.getBoundingClientRect();
    const domX      = e.clientX - rect.left;
    const domY      = e.clientY - rect.top;
    const plotH     = canvas.clientHeight - AXIS_H;
    const drawW     = rect.width - LABEL_W;
    const windowSec = timeEnd - timeStart;

    // Only detect inside the axis strip
    if (domY < plotH || drawW <= 0) { setTooltip(null); return; }

    for (const evt of events) {
      if (evt.onset < timeStart || evt.onset > timeEnd) continue;
      const evtX = LABEL_W + ((evt.onset - timeStart) / windowSec) * drawW;
      if (Math.abs(domX - evtX) <= PROX_PX) {
        setTooltip({ x: domX, y: domY, evt });
        return;
      }
    }
    setTooltip(null);
  }, [timeStart, timeEnd, events]);

  // ── Mouse-wheel vertical scroll ──────────────────────────────────────────
  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    const laneH     = Math.max(LANE_H_PX, traces.length > 0
      ? Math.floor(viewH / traces.length) : LANE_H_PX);
    const totalH    = traces.length * laneH;
    const maxOff    = Math.max(0, totalH - viewH);
    if (maxOff <= 0) return;
    e.preventDefault();
    setScrollOffset((prev) => Math.max(0, Math.min(maxOff, prev + e.deltaY)));
  }, [traces.length, viewH]);

  // ── Pan + jump controls ───────────────────────────────────────────────────
  const windowSec = timeEnd - timeStart;
  const maxDur    = payload?.totalDuration ?? 60;

  const doJump = useCallback(() => {
    const t = parseFloat(jumpInput);
    if (isNaN(t)) return;
    const win = timeEnd - timeStart;
    const s   = Math.max(0, Math.min(t - win / 2, maxDur - win));
    setTimeStart(s);
    setTimeEnd(s + win);
  }, [jumpInput, timeStart, timeEnd, maxDur]);

  const panLeft = () => {
    const step = windowSec / 2;
    const ns   = Math.max(0, timeStart - step);
    setTimeStart(ns);
    setTimeEnd(ns + windowSec);
  };

  const panRight = () => {
    const step = windowSec / 2;
    const ne   = Math.min(maxDur, timeEnd + step);
    setTimeStart(ne - windowSec);
    setTimeEnd(ne);
  };

  return (
    <div className="multimodal-panel multimodal-panel--meg" style={{ position: 'relative' }}>
      {/* Waveform + time-axis canvas */}
      <canvas
        ref={canvasRef}
        className="multimodal-panel__canvas"
        style={{ width: '100%', height: '100%', cursor: 'crosshair' }}
        onClick={handleCanvasClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setTooltip(null)}
        onWheel={handleWheel}
        title="Click to place time cursor · scroll to pan channels · hover axis markers for event details"
      />
      {/* Vertical scrollbar — appears when channel lanes overflow the panel height */}
      <MegVerticalScrollbar
        totalH={traces.length * Math.max(LANE_H_PX, viewH > 0 && traces.length > 0
          ? Math.floor(viewH / traces.length) : LANE_H_PX)}
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

      {/* Nav bar: pan arrows + jump-to-time + time range */}
      {payload && (
        <div className="meg-panel__nav">
          <button className="btn meg-viewer__pan-btn" onClick={panLeft}  title="Pan left">◀</button>
          <button
            className={`btn btn--sm ${bandTintEnabled ? 'btn--primary' : 'btn--secondary'}`}
            onClick={() => setBandTintEnabled((v) => !v)}
            title="Colour each lane background with its dominant frequency band"
          >
            {bandTintEnabled ? 'Tint ✓' : 'Tint'}
          </button>
          <span className="meg-panel__time-label">
            {timeStart.toFixed(1)} – {timeEnd.toFixed(1)} s
          </span>
          <div className="meg-viewer__jump">
            <input
              type="number"
              className="meg-viewer__jump-input"
              placeholder="time…"
              step="0.1"
              min={0}
              value={jumpInput}
              onChange={(e) => setJumpInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') doJump(); }}
              aria-label="Jump to time in seconds"
            />
            <span className="meg-viewer__jump-unit">s</span>
            <button className="btn btn--sm" onClick={doJump} title="Jump to entered time">Go</button>
          </div>
          <button className="btn meg-viewer__pan-btn" onClick={panRight} title="Pan right">▶</button>
        </div>
      )}

      {/* Empty state */}
      {!payload && (
        <div className="multimodal-panel__empty">
          <p>Load a .fif MEG file to enable waveform view</p>
        </div>
      )}
    </div>
  );
};

export default MegPanel;
