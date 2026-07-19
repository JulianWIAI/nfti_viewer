/**
 * DecodingTimeline.tsx — Lightweight SVG decoding time-course chart
 * ──────────────────────────────────────────────────────────────────
 *
 * Renders the MVPA AUC score vector returned by `POST /api/eeg/decode` as an
 * annotated line chart with:
 *
 *   • X axis  — time in seconds (EEG epoch window, e.g. -0.2 … 0.8 s)
 *   • Y axis  — AUC score, domain [yMin, yMax] (default 0.3 – 1.0)
 *   • Chance line   — dashed horizontal at `chanceLevel` (default 0.5)
 *   • Threshold band — region above `thresholdLevel` (default 0.65) is
 *     highlighted brighter via a clipPath so above-chance clusters are
 *     immediately visible without extra DOM nodes
 *   • Std-error band — optional shaded ±1 SD envelope around the mean
 *   • Playhead  — vertical cursor that tracks `currentTimeIndex`; synced
 *     with TimeScrubber so clicking the VTK scrubber moves the chart line
 *   • Click-to-seek — clicking anywhere on the SVG calls `onSeek(idx)` so
 *     the parent can keep TimeScrubber and this chart in lock-step
 *
 * SIZING
 * ───────
 * The SVG fills its container width via `width: 100%` and defaults to 180 px
 * tall; override with a CSS custom property or inline style on the wrapper.
 * A ResizeObserver keeps the scales reactive to container resizes without
 * remounting.
 *
 * PERFORMANCE
 * ────────────
 * All SVG path strings are memoised and only recomputed when `times`,
 * `scores`, or the plot dimensions change.  The playhead is a `<g
 * transform="translate(x,0)">` so GPU compositing moves it without
 * triggering a React re-render of the path strings.
 *
 * CLIP-PATH HIGHLIGHT
 * ────────────────────
 * Two clipPaths are defined in <defs>:
 *   {uid}-plot   clips all drawing to the plot area (prevents axis overflow)
 *   {uid}-above  clips to a rect whose height equals `thresholdY` (pixel Y of
 *                the threshold line from the top).  The area path is drawn
 *                twice: once with a subtle fill (visible everywhere), and once
 *                with a bright fill restricted to the {uid}-above clip region.
 *                The result is a clear visual distinction above the threshold
 *                with zero extra DOM nodes.
 *
 * clipPath IDs are scoped with a per-instance random suffix so multiple charts
 * on the same page never share IDs.
 *
 * CSS CLASSES  (defined in App.css under "── DecodingTimeline ──")
 * ───────────────────────────────────────────────────────────────
 *   .decoding-timeline              Outer container
 *   .decoding-timeline__svg         SVG element
 *   .decoding-timeline__grid-line   Faint horizontal grid line
 *   .decoding-timeline__axis-line   Solid X/Y axis line
 *   .decoding-timeline__tick        Tick mark line
 *   .decoding-timeline__tick-label  Tick label text
 *   .decoding-timeline__tick-label--y  Y-axis label (text-anchor: end)
 *   .decoding-timeline__chance-line  Dashed chance-level indicator
 *   .decoding-timeline__threshold-line  Dashed threshold indicator
 *   .decoding-timeline__area--base  Subtle base fill under the score line
 *   .decoding-timeline__area--above  Bright fill for above-threshold region
 *   .decoding-timeline__std-band    ±SD band fill
 *   .decoding-timeline__line        Score line stroke
 *   .decoding-timeline__playhead    Playhead group (line + dot)
 */

import {
  useRef,
  useState,
  useEffect,
  useMemo,
  useCallback,
  type JSX,
} from 'react';
import {
  createLinearScale,
  buildSvgLinePath,
  buildSvgAreaPath,
  buildSvgStdBandPath,
  findClosestTimeIndex,
  generateYTicks,
  generateXTicks,
  formatTimeMs,
} from '../lib/chart/decodingChartUtils';

// ── Props ─────────────────────────────────────────────────────────────────────

export interface DecodingTimelineProps {
  /**
   * Time axis values in seconds, one entry per decoded time point.
   * Must be the same length as `scores`.  Typically matches the `times`
   * array returned by `POST /api/eeg/decode`.
   */
  times: readonly number[];

  /**
   * Mean AUC score per time point.  Must be the same length as `times`.
   */
  scores: readonly number[];

  /**
   * Optional per-time-point standard deviation (or SEM).  When provided a
   * shaded ±1 SD band is drawn behind the score line.
   * Must be the same length as `times` if supplied.
   */
  scoresStd?: readonly number[];

  /** Y-axis lower bound (inclusive).  @default 0.3 */
  yMin?: number;

  /** Y-axis upper bound (inclusive).  @default 1.0 */
  yMax?: number;

  /**
   * Dashed horizontal chance-level line.
   * At 0.5 AUC a binary decoder is performing at chance.
   * @default 0.5
   */
  chanceLevel?: number;

  /**
   * Threshold AUC value above which the area fill is highlighted brighter.
   * A common heuristic for "reliably above chance" in EEG decoding.
   * @default 0.65
   */
  thresholdLevel?: number;

  /**
   * Index into `times` / `scores` of the currently displayed frame.
   * Passed in from the parent that also owns a TimeScrubber so both controls
   * stay synchronised.  @default 0
   */
  currentTimeIndex?: number;

  /**
   * Called when the user clicks on the chart area to seek to a time point.
   * Receives the index of the closest time sample to the click position.
   * The parent should propagate this to the TimeScrubber (and the VTK
   * temporal overlay) to keep everything in sync.
   */
  onSeek?: (timeIndex: number) => void;

  /** Optional extra CSS class on the outer container. */
  className?: string;
}

// ── Layout constants ──────────────────────────────────────────────────────────

/** Pixel margins around the SVG plot area (top, right, bottom, left). */
const MARGIN = { top: 18, right: 18, bottom: 32, left: 46 } as const;

/** Default SVG height in pixels (overridable via CSS on the SVG element). */
const DEFAULT_HEIGHT = 180;

// ── Component ─────────────────────────────────────────────────────────────────

export default function DecodingTimeline({
  times,
  scores,
  scoresStd,
  yMin            = 0.3,
  yMax            = 1.0,
  chanceLevel     = 0.5,
  thresholdLevel  = 0.65,
  currentTimeIndex = 0,
  onSeek,
  className,
}: DecodingTimelineProps): JSX.Element {

  // Stable UID per instance — prevents clipPath ID collisions when multiple
  // DecodingTimeline charts are mounted simultaneously.
  const uid = useRef(`dt${Math.random().toString(36).slice(2, 7)}`).current;

  // ── Responsive sizing via ResizeObserver ──────────────────────────────────
  const svgRef = useRef<SVGSVGElement>(null);
  const [dims, setDims] = useState({ width: 480, height: DEFAULT_HEIGHT });

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDims({
        width:  Math.max(width, 120),
        height: Math.max(height, 80),
      });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ── Derived plot area dimensions ──────────────────────────────────────────
  const plotW = dims.width  - MARGIN.left - MARGIN.right;
  const plotH = dims.height - MARGIN.top  - MARGIN.bottom;

  // ── Scales (memoised — only rebuild when data extent or dimensions change) ─
  const xScale = useMemo(() => {
    if (times.length < 2) return createLinearScale([0, 1], [0, plotW]);
    return createLinearScale(
      [times[0], times[times.length - 1]],
      [0, plotW],
    );
  }, [times, plotW]);

  const yScale = useMemo(
    () => createLinearScale([yMin, yMax], [plotH, 0]),
    [yMin, yMax, plotH],
  );

  // ── SVG path strings (memoised — recompute only on data or size change) ───
  const linePath = useMemo(
    () => buildSvgLinePath(times, scores, xScale, yScale),
    [times, scores, xScale, yScale],
  );

  const areaPath = useMemo(
    () => buildSvgAreaPath(times, scores, xScale, yScale, plotH),
    [times, scores, xScale, yScale, plotH],
  );

  const stdBandPath = useMemo(() => {
    if (!scoresStd || scoresStd.length !== scores.length) return null;
    const upper = scores.map((s, i) => s + scoresStd![i]);
    const lower = scores.map((s, i) => s - scoresStd![i]);
    return buildSvgStdBandPath(times, upper, lower, xScale, yScale);
  }, [times, scores, scoresStd, xScale, yScale]);

  // ── Reference line pixel positions ───────────────────────────────────────
  const chanceY    = yScale(chanceLevel);
  const thresholdY = yScale(thresholdLevel);

  // ── Axis ticks ────────────────────────────────────────────────────────────
  // Y: 0.1 AUC steps
  const yTicks = useMemo(
    () => generateYTicks(yMin, yMax, 0.1),
    [yMin, yMax],
  );

  // X: 6 evenly spaced ticks spanning the epoch window
  const xTicks = useMemo(
    () =>
      times.length > 0
        ? generateXTicks(times[0], times[times.length - 1], 6)
        : [],
    [times],
  );

  // ── Playhead x position in plot coordinates ───────────────────────────────
  const safeIdx    = Math.min(Math.max(currentTimeIndex, 0), times.length - 1);
  const playheadX  = times.length > 0 ? xScale(times[safeIdx]) : 0;
  const playheadY  = times.length > 0 && scores[safeIdx] !== undefined
    ? yScale(scores[safeIdx])
    : null;

  // ── Click-to-seek ─────────────────────────────────────────────────────────
  const handleSvgClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!onSeek || times.length === 0) return;
      const rect   = e.currentTarget.getBoundingClientRect();
      const clickX = e.clientX - rect.left - MARGIN.left;
      const tValue = xScale.invert(clickX);
      onSeek(findClosestTimeIndex(times, tValue));
    },
    [onSeek, xScale, times],
  );

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className={`decoding-timeline${className ? ` ${className}` : ''}`}>
      <svg
        ref={svgRef}
        className="decoding-timeline__svg"
        onClick={handleSvgClick}
        role="img"
        aria-label="Decoding accuracy over time"
      >
        <defs>
          {/* Clips all drawing to the plot rectangle — prevents lines from
              bleeding into the axis margin areas. */}
          <clipPath id={`${uid}-plot`}>
            <rect x={0} y={0} width={plotW} height={plotH} />
          </clipPath>

          {/* Clips to the region ABOVE the threshold line.
              Height = thresholdY because SVG Y grows downward — pixels
              between y=0 and y=thresholdY are above the threshold value. */}
          <clipPath id={`${uid}-above`}>
            <rect x={0} y={0} width={plotW} height={thresholdY} />
          </clipPath>
        </defs>

        {/* All drawing is in plot-origin coordinates */}
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>

          {/* ── Horizontal grid lines at each Y tick ──────────────────── */}
          {yTicks.map((v) => (
            <line
              key={v}
              className="decoding-timeline__grid-line"
              x1={0}
              y1={yScale(v)}
              x2={plotW}
              y2={yScale(v)}
            />
          ))}

          {/* ── Chance level — dashed horizontal ──────────────────────── */}
          <line
            className="decoding-timeline__chance-line"
            x1={0}
            y1={chanceY}
            x2={plotW}
            y2={chanceY}
          />

          {/* ── Area fill: base (subtle, full area) ───────────────────── */}
          {areaPath && (
            <path
              className="decoding-timeline__area decoding-timeline__area--base"
              d={areaPath}
              clipPath={`url(#${uid}-plot)`}
            />
          )}

          {/* ── Area fill: above-threshold (bright, clipped high) ──────── */}
          {areaPath && (
            <path
              className="decoding-timeline__area decoding-timeline__area--above"
              d={areaPath}
              clipPath={`url(#${uid}-above)`}
            />
          )}

          {/* ── Threshold dashed indicator line ───────────────────────── */}
          <line
            className="decoding-timeline__threshold-line"
            x1={0}
            y1={thresholdY}
            x2={plotW}
            y2={thresholdY}
          />

          {/* ── Optional ±SD band ─────────────────────────────────────── */}
          {stdBandPath && (
            <path
              className="decoding-timeline__std-band"
              d={stdBandPath}
              clipPath={`url(#${uid}-plot)`}
            />
          )}

          {/* ── Score line ────────────────────────────────────────────── */}
          {linePath && (
            <path
              className="decoding-timeline__line"
              d={linePath}
              clipPath={`url(#${uid}-plot)`}
            />
          )}

          {/* ── X axis ────────────────────────────────────────────────── */}
          <line
            className="decoding-timeline__axis-line"
            x1={0}
            y1={plotH}
            x2={plotW}
            y2={plotH}
          />
          {xTicks.map((t) => {
            const px = xScale(t);
            return (
              <g key={t} transform={`translate(${px},${plotH})`}>
                <line className="decoding-timeline__tick" y2={4} />
                <text className="decoding-timeline__tick-label" dy={14}>
                  {formatTimeMs(t)}
                </text>
              </g>
            );
          })}

          {/* ── Y axis ────────────────────────────────────────────────── */}
          <line
            className="decoding-timeline__axis-line"
            x1={0}
            y1={0}
            x2={0}
            y2={plotH}
          />
          {yTicks.map((v) => {
            const py = yScale(v);
            return (
              <g key={v} transform={`translate(0,${py})`}>
                <line className="decoding-timeline__tick" x2={-4} />
                <text
                  className="decoding-timeline__tick-label decoding-timeline__tick-label--y"
                  dx={-6}
                >
                  {v.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* ── Playhead ──────────────────────────────────────────────── */}
          {/* Wrapped in a <g transform> so GPU-composited CSS transforms move
              it without recalculating any path strings. */}
          {times.length > 0 && (
            <g
              className="decoding-timeline__playhead"
              transform={`translate(${playheadX.toFixed(2)},0)`}
            >
              {/* Vertical line spanning the full plot height */}
              <line y1={0} y2={plotH} />

              {/* Dot at the current score value */}
              {playheadY !== null && (
                <circle cy={playheadY} r={4} />
              )}
            </g>
          )}

        </g>{/* end plot group */}
      </svg>
    </div>
  );
}
