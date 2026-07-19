/**
 * AlignmentModeToggle.tsx — Segmented control for dual-viewer alignment mode.
 *
 * Renders two adjacent pill buttons that act as a single radio group:
 *
 *   [ Raw Overlay ]  |  [ SyN Alignment ]
 *
 * Behaviour
 * ─────────
 *   Raw Overlay   — both brains are displayed in their native voxel spaces;
 *                   cameras are decoupled (each pane can be inspected independently).
 *
 *   SyN Alignment — Subject B is warped into Subject A's space via Affine + SyN
 *                   diffeomorphic registration; cameras are locked together so any
 *                   rotation/pan on either pane is mirrored by the other.
 *                   Triggers the SyN pipeline on first click; subsequent clicks
 *                   restore the cached warped volume instantly.
 *
 * Props
 * ─────
 *   mode       Current alignment mode — controls which button is active.
 *   onChange   Called with the new mode on button click.
 *   disabled   Both buttons are non-interactive (e.g. SyN is in progress).
 *   loading    Shows a spinner inside the "SyN Alignment" button while the
 *              backend registration is running.
 */

import type { FC } from 'react';
import type { AlignmentMode } from './DualViewerContext';

// ── Spinner — tiny CSS animation inside the button ────────────────────────────

/** Inline spinner shown while SyN is processing. */
const Spinner: FC = () => (
  <span className="alignment-toggle__spinner" aria-hidden="true" />
);

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  /** Currently active alignment mode. */
  mode: AlignmentMode;
  /** Called with the new mode when the user clicks an option. */
  onChange: (mode: AlignmentMode) => void;
  /** Disable both options (e.g. during SyN upload). */
  disabled?: boolean;
  /** Show an inline spinner on the SyN button (registration in progress). */
  loading?: boolean;
}

const AlignmentModeToggle: FC<Props> = ({ mode, onChange, disabled = false, loading = false }) => {
  return (
    <div
      className="alignment-toggle"
      role="group"
      aria-label="Alignment mode"
    >
      {/* ── Raw Overlay option ───────────────────────────────────────────── */}
      <button
        type="button"
        className={[
          'alignment-toggle__option',
          mode === 'raw' ? 'alignment-toggle__option--active' : '',
        ].join(' ').trim()}
        onClick={() => onChange('raw')}
        disabled={disabled}
        aria-pressed={mode === 'raw'}
        title="Show both brains in their native voxel spaces (no registration)"
      >
        Raw Overlay
      </button>

      {/* ── SyN Alignment option ─────────────────────────────────────────── */}
      <button
        type="button"
        className={[
          'alignment-toggle__option',
          mode === 'registered' ? 'alignment-toggle__option--active' : '',
          loading ? 'alignment-toggle__option--loading' : '',
        ].join(' ').trim()}
        onClick={() => onChange('registered')}
        disabled={disabled}
        aria-pressed={mode === 'registered'}
        title={
          loading
            ? 'SyN registration in progress (2–5 min)…'
            : 'Warp Subject B into Subject A\'s space using Affine + SyN registration'
        }
      >
        {/* Spinner appears while backend is running — does NOT replace the label */}
        {loading && <Spinner />}
        SyN Alignment
      </button>
    </div>
  );
};

export default AlignmentModeToggle;
