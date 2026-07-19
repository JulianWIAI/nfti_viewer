/**
 * useCanvasRenderer.ts — EEG/MEG multi-channel canvas rendering hook
 * ────────────────────────────────────────────────────────────────────
 *
 * Renders up to 128 EEG/MEG channels into a 2D canvas using a stacked
 * multi-lane layout.  Each channel occupies a horizontal strip (lane)
 * of laneHeightPx pixels.
 *
 * RENDERING STRATEGY
 * ──────────────────
 * 1.  Decide the visible time window [timeStart, timeEnd].
 * 2.  Slice each channel's Float32Array to those sample indices
 *     (timeWindowToSampleRange from channelScaler).
 * 3.  Downsample to canvas pixel width using decimateMinMax (fills a
 *     min/max ribbon per pixel column — correct for noisy biosignals).
 * 4.  Draw each channel lane top-to-bottom:
 *       a. Faint separator line at the top of the lane.
 *       b. Channel label at the left edge.
 *       c. Signal as a filled ribbon between the min and max polylines.
 *          The ribbon is stroked with the channel colour; a zero-baseline
 *          rule is drawn as a hairline.
 *
 * PERFORMANCE
 * ───────────
 * • A single requestAnimationFrame loop drives redraws; the frame is
 *   skipped when nothing has changed (dirty flag).
 * • All canvas operations use integer pixel coordinates to avoid
 *   sub-pixel anti-aliasing overhead.
 * • The hook returns a RefObject<HTMLCanvasElement> that the component
 *   attaches to a <canvas> element; the hook owns the entire canvas
 *   lifecycle.
 */

import { useRef, useEffect, useCallback, type RefObject } from 'react';
import type { TimeseriesPayload, TimeseriesViewerControls } from '../../../types/timeseries.types';
import type { ArtifactAnnotation, SpikeMarker } from '../../../types/analysis.types';
import { decimateMinMax, timeWindowToSampleRange } from '../lib/channelScaler';

// Re-export so TimeseriesViewer can surface the types to its callers.
export type { ArtifactAnnotation, SpikeMarker };

// ── Colour palette for channels ───────────────────────────────────────────────

// 16-colour cycling palette — sufficient for typical EEG electrode groupings.
const PALETTE = [
  '#4e9af1', '#e05a5a', '#3cbf7a', '#f0a020', '#b36ee0',
  '#4ecdc4', '#ff6b6b', '#1abc9c', '#f39c12', '#8e44ad',
  '#2980b9', '#27ae60', '#d35400', '#c0392b', '#16a085',
  '#7f8c8d',
];

function channelColour(idx: number): string {
  return PALETTE[idx % PALETTE.length]!;
}

// ── Types ─────────────────────────────────────────────────────────────────────

interface RenderParams {
  payload:     TimeseriesPayload;
  controls:    TimeseriesViewerControls;
  width:       number;
  height:      number;
  /** Artefact time windows to render as coloured background spans. */
  annotations: ArtifactAnnotation[];
  /** Epileptiform spike timestamps to render as vertical red lines. */
  spikes:      SpikeMarker[];
}

// ── Hook ──────────────────────────────────────────────────────────────────────

/**
 * Attaches a render loop to a <canvas> element.
 *
 * Usage:
 *   const canvasRef = useCanvasRenderer(payload, controls);
 *   return <canvas ref={canvasRef} />;
 *
 * @param payload  Parsed timeseries data (null = empty state).
 * @param controls Current viewer controls (window, scale, visible channels).
 * @returns A RefObject<HTMLCanvasElement> to attach to the <canvas> element.
 */
export function useCanvasRenderer(
  payload:     TimeseriesPayload | null,
  controls:    TimeseriesViewerControls,
  /** Artefact annotation windows (blinks = yellow, muscle = grey). */
  annotations: ArtifactAnnotation[] = [],
  /** Spike timestamps (rendered as vertical red dashed lines). */
  spikes:      SpikeMarker[]        = [],
): RefObject<HTMLCanvasElement> {
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const paramsRef  = useRef<RenderParams | null>(null);
  const dirtyRef   = useRef(true);
  const rafRef     = useRef<number>(0);

  // ── Render one frame ──────────────────────────────────────────────────────

  const renderFrame = useCallback(() => {
    if (!dirtyRef.current) return;
    const canvas = canvasRef.current;
    const params = paramsRef.current;
    if (!canvas || !params) return;

    const { payload: p, controls: c, width, height } = params;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Device-pixel-ratio aware sizing
    canvas.width  = width;
    canvas.height = height;

    ctx.clearRect(0, 0, width, height);

    // ── Annotation overlays (drawn BEFORE waveforms so they stay behind) ────
    // Time → canvas-x mapping for the current view window.
    const { annotations: annots, spikes: spikeList } = params;
    const timeRange = c.timeEnd - c.timeStart;

    if ((annots.length > 0 || spikeList.length > 0) && timeRange > 0) {
      const toX = (t: number) => ((t - c.timeStart) / timeRange) * width;

      // ── Artefact spans (coloured background rectangles) ──────────────────
      // Blinks → translucent yellow  |  muscle → translucent grey
      for (const a of annots) {
        const x1 = toX(a.onset);
        const x2 = toX(a.onset + a.duration);
        // Skip if the annotation is completely outside the view window.
        if (x2 < 0 || x1 > width) continue;
        ctx.fillStyle = a.type === 'blink'
          ? 'rgba(255, 220, 50, 0.13)'    // yellow — eye blink
          : 'rgba(160, 160, 160, 0.10)';  // grey   — muscle / motion
        ctx.fillRect(
          Math.max(0, x1), 0,
          Math.min(width, x2) - Math.max(0, x1),
          height,
        );
        // Tiny type label at the top of the span.
        if (x2 - x1 > 12) {
          ctx.fillStyle = a.type === 'blink'
            ? 'rgba(255, 220, 50, 0.55)'
            : 'rgba(160, 160, 160, 0.40)';
          ctx.font = '8px monospace';
          ctx.fillText(a.type === 'blink' ? 'blink' : 'muscle', Math.max(0, x1) + 2, 9);
        }
      }

      // ── Spike markers (vertical red dashed lines + triangle) ─────────────
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 60, 60, 0.65)';
      ctx.lineWidth   = 1.2;
      ctx.setLineDash([3, 3]);

      for (const s of spikeList) {
        const x = Math.round(toX(s.time));
        if (x < 0 || x > width) continue;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      ctx.setLineDash([]);

      // Small downward triangle at the top of each spike line.
      ctx.fillStyle = 'rgba(255, 60, 60, 0.80)';
      for (const s of spikeList) {
        const x = Math.round(toX(s.time));
        if (x < 0 || x > width) continue;
        ctx.beginPath();
        ctx.moveTo(x - 4, 0);
        ctx.lineTo(x + 4, 0);
        ctx.lineTo(x,     7);
        ctx.closePath();
        ctx.fill();
      }

      ctx.restore();
    }
    // ── End annotation overlays ─────────────────────────────────────────────

    const { selectedChannels, timeStart, timeEnd, amplitudeScale, laneHeightPx } = c;
    const totalDuration = p.time[p.time.length - 1]! - p.time[0]!;
    if (totalDuration <= 0) return;

    const visibleChannels = selectedChannels.filter(
      (idx) => idx >= 0 && idx < p.channels.length,
    );

    const [sStart, sEnd] = timeWindowToSampleRange(p.time, timeStart, timeEnd);
    const windowSamples  = sEnd - sStart;
    if (windowSamples <= 0) return;

    const xScale = width / (timeEnd - timeStart);
    const halfLane = laneHeightPx / 2;
    const pixelsPerUnit = amplitudeScale > 0 ? (halfLane * 0.85) / amplitudeScale : 1;

    visibleChannels.forEach((chIdx, laneIdx) => {
      const chanData = p.data[chIdx];
      if (!chanData) return;

      const slice = chanData.subarray(sStart, sEnd);
      const sliceTime = p.time.subarray(sStart, sEnd);
      const laneTop = laneIdx * laneHeightPx;
      const baseline = laneTop + halfLane;

      // Lane separator
      ctx.strokeStyle = 'rgba(128,128,128,0.2)';
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.moveTo(0, laneTop);
      ctx.lineTo(width, laneTop);
      ctx.stroke();

      // Channel label
      ctx.fillStyle = 'rgba(180,180,180,0.85)';
      ctx.font      = '10px monospace';
      ctx.fillText(p.channels[chIdx]?.label ?? `Ch${chIdx}`, 4, laneTop + 12);

      // Baseline rule (zero line)
      ctx.strokeStyle = 'rgba(128,128,128,0.3)';
      ctx.lineWidth   = 0.5;
      ctx.beginPath();
      ctx.moveTo(0, baseline);
      ctx.lineTo(width, baseline);
      ctx.stroke();

      // Decimate to canvas pixel width
      const buckets = Math.max(1, Math.floor(width));
      const { mins, maxs, times } = decimateMinMax(slice, sliceTime, buckets);

      const colour = channelColour(laneIdx);
      ctx.strokeStyle = colour;
      ctx.fillStyle   = `${colour}22`; // ~13 % opacity fill for the ribbon

      // Build ribbon paths
      const minPath = new Path2D();
      const maxPath = new Path2D();

      for (let b = 0; b < times.length; b++) {
        const x    = Math.round((times[b]! - timeStart) * xScale);
        const yMin = Math.round(baseline - maxs[b]! * pixelsPerUnit);
        const yMax = Math.round(baseline - mins[b]! * pixelsPerUnit);

        if (b === 0) {
          maxPath.moveTo(x, yMin);
          minPath.moveTo(x, yMax);
        } else {
          maxPath.lineTo(x, yMin);
          minPath.lineTo(x, yMax);
        }
      }

      // Closed ribbon fill
      const ribbon = new Path2D(maxPath);
      // Traverse the minPath in reverse to close the ribbon
      for (let b = times.length - 1; b >= 0; b--) {
        const x    = Math.round((times[b]! - timeStart) * xScale);
        const yMax = Math.round(baseline - mins[b]! * pixelsPerUnit);
        ribbon.lineTo(x, yMax);
      }
      ribbon.closePath();

      ctx.fill(ribbon);
      ctx.lineWidth = 1;
      ctx.stroke(maxPath);
      ctx.stroke(minPath);
    });

    // Time axis at the bottom (if there's room)
    if (height > visibleChannels.length * laneHeightPx + 16) {
      drawTimeAxis(ctx, width, height, timeStart, timeEnd);
    }

    dirtyRef.current = false;
  }, []);

  // ── RAF loop ──────────────────────────────────────────────────────────────

  useEffect(() => {
    function loop() {
      renderFrame();
      rafRef.current = requestAnimationFrame(loop);
    }
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [renderFrame]);

  // ── Sync params → dirty on every render ──────────────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !payload) {
      paramsRef.current = null;
      dirtyRef.current  = true;
      return;
    }

    const { width, height } = canvas.getBoundingClientRect();
    paramsRef.current = {
      payload,
      controls,
      annotations,
      spikes,
      width:  Math.floor(width)  || 800,
      height: Math.floor(height) || 400,
    };
    dirtyRef.current = true;
  });

  // ── ResizeObserver ────────────────────────────────────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ro = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const { width, height } = entry.contentRect;
      if (paramsRef.current) {
        paramsRef.current.width  = Math.floor(width)  || 800;
        paramsRef.current.height = Math.floor(height) || 400;
        dirtyRef.current = true;
      }
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  return canvasRef;
}

// ── Time axis helper ──────────────────────────────────────────────────────────

function drawTimeAxis(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  tStart: number,
  tEnd: number,
): void {
  const axisY = height - 20;
  const span  = tEnd - tStart;

  ctx.strokeStyle = 'rgba(180,180,180,0.4)';
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(0, axisY);
  ctx.lineTo(width, axisY);
  ctx.stroke();

  // Pick a tick interval that gives roughly 8–12 ticks
  const rawInterval = span / 10;
  const magnitude   = Math.pow(10, Math.floor(Math.log10(rawInterval)));
  const interval    = Math.ceil(rawInterval / magnitude) * magnitude;

  ctx.fillStyle = 'rgba(180,180,180,0.7)';
  ctx.font      = '10px monospace';
  ctx.textAlign = 'center';

  const firstTick = Math.ceil(tStart / interval) * interval;
  for (let t = firstTick; t <= tEnd; t += interval) {
    const x = ((t - tStart) / span) * width;
    ctx.beginPath();
    ctx.moveTo(x, axisY);
    ctx.lineTo(x, axisY + 5);
    ctx.stroke();
    ctx.fillText(formatTime(t), x, axisY + 15);
  }
}

function formatTime(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(0).padStart(2, '0');
  return `${m}:${s}`;
}
