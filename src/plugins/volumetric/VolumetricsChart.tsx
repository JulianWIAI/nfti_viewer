/**
 * VolumetricsChart.tsx — Comparative grouped bar chart for hippocampal volumetrics.
 *
 * Renders an SVG grouped bar chart with two bar-groups (Left hippocampus, Right
 * hippocampus).  Each group contains one bar per loaded subject:
 *   • Subject A — sky blue (#4fc3f7)
 *   • Subject B — violet  (#ce93d8)  — only rendered when volumetricsB is provided
 *
 * When only Subject A data is available the chart renders a single-bar layout
 * identical to the original BrainVolumetricsChart.  Normative reference band
 * and mean dashed line are always drawn so the clinician can contextualise both
 * bars against the healthy-cohort distribution.
 *
 * Implemented with plain SVG — no external charting library required.
 *
 * Props
 * ─────
 *   volumetricsA  VolumetricsResult for Subject A (required — always present).
 *   volumetricsB  VolumetricsResult for Subject B (optional — null when B not loaded).
 */

import type { FC } from 'react';
import type { VolumetricsResult } from '../../types/analysis.types';

// ── Colour palette ─────────────────────────────────────────────────────────────

const COLOR_A        = '#4fc3f7';  // sky blue — Subject A
const COLOR_B        = '#ce93d8';  // violet   — Subject B
const COLOR_NORM     = 'rgba(255, 215, 60, 0.18)';   // normative band fill
const COLOR_NORM_LN  = 'rgba(255, 215, 60, 0.55)';   // normative mean line

// ── Helpers ────────────────────────────────────────────────────────────────────

/** Format mm³: show cm³ when ≥ 1000. */
function fmtVol(mm3: number): string {
  return mm3 >= 1000 ? `${(mm3 / 1000).toFixed(2)} cm³` : `${mm3.toFixed(0)} mm³`;
}

/** Render asymmetry index with explicit sign. */
function fmtAsym(v: number): string {
  return (v >= 0 ? '+' : '') + v.toFixed(1) + '%';
}

// ── Component ──────────────────────────────────────────────────────────────────

interface Props {
  /** Volumetrics for Subject A — always required. */
  volumetricsA: VolumetricsResult;
  /** Volumetrics for Subject B — null when only one brain is loaded. */
  volumetricsB: VolumetricsResult | null;
}

const VolumetricsChart: FC<Props> = ({ volumetricsA, volumetricsB }) => {
  const HA = volumetricsA.hippocampus;
  const HB = volumetricsB?.hippocampus ?? null;
  const TVA = volumetricsA.tissue_volumes;

  // ── Chart geometry ───────────────────────────────────────────────────────────
  const SVG_W  = 340;
  const SVG_H  = 220;
  const PAD_L  = 16;
  const PAD_R  = 40;   // room for "Norm." label
  const PAD_T  = 28;
  const PAD_B  = 56;   // extra row for B legend
  const chartW = SVG_W - PAD_L - PAD_R;
  const chartH = SVG_H - PAD_T - PAD_B;

  // Y-axis maximum — headroom above the tallest visible bar or normative band top.
  const normHi = HA.normative_mean_mm3 + HA.normative_sd_mm3;
  const normLo = HA.normative_mean_mm3 - HA.normative_sd_mm3;
  const allVals = [HA.left_mm3, HA.right_mm3, normHi];
  if (HB) allVals.push(HB.left_mm3, HB.right_mm3);
  const yMax = Math.max(...allVals) * 1.15;

  const yOf = (v: number): number => PAD_T + chartH - (v / yMax) * chartH;

  // Bar width depends on whether we have two bars per group.
  const dual    = HB !== null;
  const groupW  = chartW * 0.32;           // fraction of chart width per group
  const barW    = dual ? groupW * 0.42 : groupW * 0.55;
  const barGap  = dual ? groupW * 0.09 : 0;

  // Group centres
  const group1X = PAD_L + chartW * 0.20;  // left hippocampus group
  const group2X = PAD_L + chartW * 0.62;  // right hippocampus group

  // Bar X offsets within each group
  const aOffX = dual ? -(barW + barGap / 2) : -barW / 2;
  const bOffX = dual ? barGap / 2 : 0;

  const baseY = PAD_T + chartH;   // Y of the X-axis baseline
  const normBandTop    = yOf(normHi);
  const normBandBottom = yOf(normLo);
  const normMeanY      = yOf(HA.normative_mean_mm3);

  /** Render a vertical bar with optional value label. */
  function Bar({
    cx, volume, color,
  }: { cx: number; volume: number; color: string }) {
    const topY = yOf(volume);
    return (
      <>
        <rect x={cx} y={topY} width={barW} height={baseY - topY}
          fill={color} opacity={0.88} rx={2} />
        <text x={cx + barW / 2} y={topY - 4}
          textAnchor="middle" fontSize={8} fill={color}>
          {fmtVol(volume)}
        </text>
      </>
    );
  }

  return (
    <div>
      <svg
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        style={{ width: '100%', height: 'auto' }}
        role="img"
        aria-label="Comparative hippocampal volumetrics bar chart"
      >
        {/* Title */}
        <text x={PAD_L + chartW / 2} y={16}
          textAnchor="middle" fontSize={11} fontWeight="600" fill="#b0b0c8">
          Hippocampal Volumes
        </text>

        {/* Normative band */}
        <rect x={PAD_L} y={normBandTop}
          width={chartW} height={normBandBottom - normBandTop}
          fill={COLOR_NORM} />
        <line x1={PAD_L} y1={normMeanY}
          x2={PAD_L + chartW} y2={normMeanY}
          stroke={COLOR_NORM_LN} strokeWidth={1} strokeDasharray="4 3" />
        <text x={PAD_L + chartW + 4} y={normMeanY + 3}
          fontSize={8} fill={COLOR_NORM_LN} dominantBaseline="middle">
          Norm.
        </text>

        {/* X-axis baseline */}
        <line x1={PAD_L} y1={baseY}
          x2={PAD_L + chartW} y2={baseY}
          stroke="#444" strokeWidth={0.5} />

        {/* ── Left hippocampus group ─────────────────────────────────────────── */}
        <Bar cx={group1X + aOffX} volume={HA.left_mm3}  color={COLOR_A} />
        {HB && <Bar cx={group1X + bOffX} volume={HB.left_mm3}  color={COLOR_B} />}

        {/* ── Right hippocampus group ────────────────────────────────────────── */}
        <Bar cx={group2X + aOffX} volume={HA.right_mm3} color={COLOR_A} />
        {HB && <Bar cx={group2X + bOffX} volume={HB.right_mm3} color={COLOR_B} />}

        {/* X-axis group labels */}
        <text x={group1X} y={baseY + 12}
          textAnchor="middle" fontSize={9} fill="#aaa">
          Left
        </text>
        <text x={group2X} y={baseY + 12}
          textAnchor="middle" fontSize={9} fill="#aaa">
          Right
        </text>
        <text x={PAD_L + chartW / 2} y={baseY + 24}
          textAnchor="middle" fontSize={8} fill="#666">
          Hippocampus
        </text>

        {/* Legend row */}
        <rect x={group1X - 20} y={baseY + 34} width={8} height={8}
          fill={COLOR_A} rx={1} />
        <text x={group1X - 9} y={baseY + 41} fontSize={8} fill="#aaa">
          Subj. A
        </text>
        {HB && (
          <>
            <rect x={group2X - 20} y={baseY + 34} width={8} height={8}
              fill={COLOR_B} rx={1} />
            <text x={group2X - 9} y={baseY + 41} fontSize={8} fill="#aaa">
              Subj. B
            </text>
          </>
        )}
      </svg>

      {/* Stats summary */}
      <div style={{ fontSize: 10, color: '#9a9ab8', lineHeight: 1.7 }}>
        <div>
          <span style={{ color: COLOR_A }}>A</span>
          <span style={{ color: '#b0b0c8' }}> asymmetry: </span>
          {fmtAsym(HA.asymmetry_index)}
          {HB && (
            <>
              <span style={{ color: '#666', marginLeft: 8 }}>|</span>
              <span style={{ color: COLOR_B, marginLeft: 8 }}>B</span>
              <span style={{ color: '#b0b0c8' }}> asymmetry: </span>
              {fmtAsym(HB.asymmetry_index)}
            </>
          )}
        </div>
        {/* Tissue volumes for A */}
        <div style={{ marginTop: 4, color: '#666', fontSize: 9, lineHeight: 1.6 }}>
          <span style={{ color: '#888' }}>A: </span>
          GM {(TVA.gm_mm3  / 1000).toFixed(0)} cm³&nbsp;|&nbsp;
          WM {(TVA.wm_mm3  / 1000).toFixed(0)} cm³&nbsp;|&nbsp;
          CSF {(TVA.csf_mm3 / 1000).toFixed(0)} cm³
        </div>
        {/* Tissue volumes for B */}
        {volumetricsB && (
          <div style={{ color: '#666', fontSize: 9, lineHeight: 1.6 }}>
            <span style={{ color: '#888' }}>B: </span>
            GM {(volumetricsB.tissue_volumes.gm_mm3  / 1000).toFixed(0)} cm³&nbsp;|&nbsp;
            WM {(volumetricsB.tissue_volumes.wm_mm3  / 1000).toFixed(0)} cm³&nbsp;|&nbsp;
            CSF {(volumetricsB.tissue_volumes.csf_mm3 / 1000).toFixed(0)} cm³
          </div>
        )}
      </div>
    </div>
  );
};

export default VolumetricsChart;
