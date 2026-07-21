/**
 * InlineTaskProgress.tsx — Thin inline progress bar for analysis panels
 * ───────────────────────────────────────────────────────────────────────
 *
 * A lightweight progress bar designed to be embedded directly inside each
 * analysis panel section (AnomalyPanel, LongitudinalPanel, DecodingPanel,
 * VolumetricControls) right below the status indicator row.
 *
 * It is purely presentational: it receives the current phase and upload
 * percentage as props and renders the appropriate visual state.
 *
 * WHEN IT IS VISIBLE
 * ──────────────────
 * The component renders nothing (returns null) when phase is 'idle' or
 * 'done'.  Only 'uploading', 'running', and 'error' produce visible output,
 * so there is no layout impact during the idle state.
 *
 * VISUAL BEHAVIOUR
 * ─────────────────
 *   uploading → green determinate fill (width = uploadPct %)
 *   running   → animated indeterminate shimmer (CSS @keyframes)
 *   error     → full-width red fill
 *   idle/done → renders nothing
 *
 * CSS lives in ./progress.css.
 */

import type { FC } from 'react';

// ── Props ─────────────────────────────────────────────────────────────────────

/**
 * Props for InlineTaskProgress.
 *
 * phase     — the current task phase driving the visual style
 * uploadPct — byte-level upload percentage 0–100 (only used when phase === 'uploading')
 */
interface InlineTaskProgressProps {
  /** Current task phase.  'idle' and 'done' cause the component to render nothing. */
  phase:     'uploading' | 'running' | 'done' | 'error' | 'idle';
  /** Upload progress 0–100.  Only meaningful when phase === 'uploading'. */
  uploadPct: number;
}

// ── Component ─────────────────────────────────────────────────────────────────

/**
 * Thin (4px) inline progress bar for embedding inside sidebar panel sections.
 *
 * Returns null when phase is 'idle' or 'done' so it has zero layout impact
 * during those states.
 */
const InlineTaskProgress: FC<InlineTaskProgressProps> = ({ phase, uploadPct }) => {
  // Do not render anything when the task is idle or successfully completed.
  // The panel's own status row (status dot + text) handles those states.
  if (phase === 'idle' || phase === 'done') return null;

  return (
    // Wrapper div: full-width within the parent section, with small margins.
    <div className="inline-task-progress">

      {/* Track: the background container clipping the fill / shimmer */}
      <div className="inline-task-progress__track">

        {/* ── Determinate fill — uploading ──────────────────────────────── */}
        {phase === 'uploading' && (
          <div
            className="inline-task-progress__fill"
            // Width is driven by the uploadPct prop (0-100).
            // The CSS transition smoothes jumps between progress events.
            style={{ width: `${Math.min(100, Math.max(0, uploadPct))}%` }}
          />
        )}

        {/* ── Error fill — full width red ───────────────────────────────── */}
        {phase === 'error' && (
          <div className="inline-task-progress__fill inline-task-progress__fill--error" />
        )}

        {/* ── Indeterminate shimmer — running ───────────────────────────── */}
        {/* The shimmer is a CSS animated gradient — no JavaScript animation */}
        {phase === 'running' && (
          <div className="inline-task-progress__shimmer" />
        )}
      </div>
    </div>
  );
};

export default InlineTaskProgress;
