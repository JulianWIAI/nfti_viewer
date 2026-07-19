/**
 * BrainVolumetricsChart.tsx — SVG bar chart for hippocampal volumetrics
 * ───────────────────────────────────────────────────────────────────────
 * Displays left and right hippocampal volumes (in mm³) as vertical bars
 * alongside a normative reference band (mean ± 1 SD) derived from healthy
 * adult cohorts.  Also shows a compact tissue-volume breakdown.
 *
 * Implemented with plain SVG — no external charting library required.
 *
 * Props
 * ─────
 *   result   VolumetricsResult from /api/mri/volumetrics
 *
 * Visual design
 * ─────────────
 *   • Left bar  — sky blue (#4fc3f7)
 *   • Right bar — green   (#81c784)
 *   • Normative mean — amber dashed line
 *   • Normative ± 1 SD band — translucent amber rectangle
 */

import type { FC } from 'react';
import type { VolumetricsResult } from '../types/analysis.types';

// ── Colour constants ──────────────────────────────────────────────────────────

const COLOR_LEFT   = '#4fc3f7';  // sky blue
const COLOR_RIGHT  = '#81c784';  // green
const COLOR_NORM   = 'rgba(255, 215, 60, 0.22)';   // normative band fill
const COLOR_NORM_LINE = 'rgba(255, 215, 60, 0.55)'; // normative mean line

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Format mm³ values: if ≥ 1000 show in cm³, otherwise in mm³. */
function fmtVol(mm3: number): string {
  return mm3 >= 1000
    ? `${(mm3 / 1000).toFixed(2)} cm³`
    : `${mm3.toFixed(0)} mm³`;
}

/** Return "+" prefix for positive numbers (for the asymmetry index). */
function fmtAsym(v: number): string {
  return (v >= 0 ? '+' : '') + v.toFixed(1) + '%';
}

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  result: VolumetricsResult;
}

const BrainVolumetricsChart: FC<Props> = ({ result }) => {
  const { hippocampus: H, tissue_volumes: TV } = result;

  // ── Chart geometry ─────────────────────────────────────────────────────────
  const SVG_W  = 340;
  const SVG_H  = 210;
  const PAD_L  = 16;   // left padding
  const PAD_R  = 40;   // right padding (room for normative label)
  const PAD_T  = 28;   // top padding  (chart title)
  const PAD_B  = 48;   // bottom padding (x-axis labels)
  const chartW = SVG_W - PAD_L - PAD_R;
  const chartH = SVG_H - PAD_T - PAD_B;

  // Y-axis maximum: top of normative upper bound × 1.15 for headroom.
  const normHi  = H.normative_mean_mm3 + H.normative_sd_mm3;
  const normLo  = H.normative_mean_mm3 - H.normative_sd_mm3;
  const yMax    = Math.max(H.left_mm3, H.right_mm3, normHi) * 1.15;

  // Map a volume value to a Y coordinate (SVG y grows downward).
  const yOf = (v: number): number => PAD_T + chartH - (v / yMax) * chartH;

  // Bar geometry: two bars centred in the chart area.
  const barW   = chartW * 0.22;
  const bar1X  = PAD_L + chartW * 0.18;   // left hippocampus bar x
  const bar2X  = PAD_L + chartW * 0.58;   // right hippocampus bar x

  // Normative band Y coordinates.
  const normBandTop    = yOf(normHi);
  const normBandBottom = yOf(normLo);
  const normMeanY      = yOf(H.normative_mean_mm3);

  // Individual bar heights (bottom of chart is y = PAD_T + chartH).
  const barBottomY  = PAD_T + chartH;
  const leftBarTopY  = yOf(H.left_mm3);
  const rightBarTopY = yOf(H.right_mm3);

  return (
    <div>
      {/* ── SVG bar chart ─────────────────────────────────────────────────── */}
      <svg
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        style={{ width: '100%', height: 'auto' }}
        role="img"
        aria-label="Hippocampal volumetrics bar chart"
      >
        {/* Chart title */}
        <text
          x={PAD_L + chartW / 2} y={16}
          textAnchor="middle"
          fontSize={11} fontWeight="600" fill="#b0b0c8"
        >
          Hippocampal Volumes
        </text>

        {/* ── Normative range band ─────────────────────────────────────────── */}
        <rect
          x={PAD_L} y={normBandTop}
          width={chartW} height={normBandBottom - normBandTop}
          fill={COLOR_NORM}
        />
        {/* Normative mean dashed line */}
        <line
          x1={PAD_L} y1={normMeanY}
          x2={PAD_L + chartW} y2={normMeanY}
          stroke={COLOR_NORM_LINE} strokeWidth={1} strokeDasharray="4 3"
        />
        {/* "Norm." label to the right of the band */}
        <text
          x={PAD_L + chartW + 4} y={normMeanY + 3}
          fontSize={8} fill={COLOR_NORM_LINE} dominantBaseline="middle"
        >
          Norm.
        </text>

        {/* ── X-axis baseline ──────────────────────────────────────────────── */}
        <line
          x1={PAD_L} y1={barBottomY}
          x2={PAD_L + chartW} y2={barBottomY}
          stroke="#444" strokeWidth={0.5}
        />

        {/* ── Left hippocampus bar ─────────────────────────────────────────── */}
        <rect
          x={bar1X} y={leftBarTopY}
          width={barW} height={barBottomY - leftBarTopY}
          fill={COLOR_LEFT} opacity={0.85} rx={2}
        />
        {/* Value label above the bar */}
        <text
          x={bar1X + barW / 2} y={leftBarTopY - 5}
          textAnchor="middle" fontSize={9} fill={COLOR_LEFT}
        >
          {fmtVol(H.left_mm3)}
        </text>

        {/* ── Right hippocampus bar ────────────────────────────────────────── */}
        <rect
          x={bar2X} y={rightBarTopY}
          width={barW} height={barBottomY - rightBarTopY}
          fill={COLOR_RIGHT} opacity={0.85} rx={2}
        />
        <text
          x={bar2X + barW / 2} y={rightBarTopY - 5}
          textAnchor="middle" fontSize={9} fill={COLOR_RIGHT}
        >
          {fmtVol(H.right_mm3)}
        </text>

        {/* ── X-axis labels ────────────────────────────────────────────────── */}
        <text
          x={bar1X + barW / 2} y={barBottomY + 13}
          textAnchor="middle" fontSize={9} fill="#aaa"
        >
          Left
        </text>
        <text
          x={bar2X + barW / 2} y={barBottomY + 13}
          textAnchor="middle" fontSize={9} fill="#aaa"
        >
          Right
        </text>
        <text
          x={PAD_L + chartW / 2} y={barBottomY + 25}
          textAnchor="middle" fontSize={8} fill="#666"
        >
          Hippocampus
        </text>

        {/* ── Legend ───────────────────────────────────────────────────────── */}
        <rect x={bar1X} y={barBottomY + 35} width={8} height={8} fill={COLOR_LEFT} rx={1} />
        <text x={bar1X + 11} y={barBottomY + 42} fontSize={8} fill="#aaa">Left</text>
        <rect x={bar2X} y={barBottomY + 35} width={8} height={8} fill={COLOR_RIGHT} rx={1} />
        <text x={bar2X + 11} y={barBottomY + 42} fontSize={8} fill="#aaa">Right</text>
      </svg>

      {/* ── Stats summary ─────────────────────────────────────────────────── */}
      <div style={{ fontSize: 10, color: '#9a9ab8', lineHeight: 1.7, marginTop: 2 }}>
        <div>
          <span style={{ color: '#b0b0c8' }}>Asymmetry: </span>
          {fmtAsym(H.asymmetry_index)}
          <span style={{ color: '#666', marginLeft: 6 }}>(L vs R)</span>
        </div>
        <div style={{ marginTop: 4, color: '#666', fontSize: 9, lineHeight: 1.6 }}>
          GM {(TV.gm_mm3  / 1000).toFixed(0)} cm³ &nbsp;|&nbsp;
          WM {(TV.wm_mm3  / 1000).toFixed(0)} cm³ &nbsp;|&nbsp;
          CSF {(TV.csf_mm3 / 1000).toFixed(0)} cm³
        </div>
        <div style={{ color: '#555', fontSize: 9 }}>
          Total brain {(result.total_brain_mm3 / 1000).toFixed(0)} cm³
        </div>
      </div>
    </div>
  );
};

export default BrainVolumetricsChart;
