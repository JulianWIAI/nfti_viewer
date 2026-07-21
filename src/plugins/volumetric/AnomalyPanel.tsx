/**
 * AnomalyPanel.tsx — sidebar controls for the anomaly detection overlay
 * ────────────────────────────────────────────────────────────────────────
 *
 * Self-contained panel that reads all state from VolumetricContext and drives
 * the anomaly pipeline entirely through context callbacks.  It does NOT own
 * any VTK actors — the bundle lifecycle (addVolume / removeActor) lives in
 * VolumetricViewer.tsx, matching the pattern used by ConnectomePanel.
 *
 * RENDERING PHASES
 * ─────────────────
 *   idle      → "Run Deep Sweep" button (enabled only when hasVolume)
 *   uploading → status dot + "Uploading…"   / button disabled
 *   running   → status dot + "Analysing…"   / button disabled
 *   done      → green dot + elapsed time + anomaly voxel count
 *               opacity slider  (RAF-debounced VTK call)
 *               visibility toggle
 *               "Clear Sweep" button
 *   error     → red dot + error message / "Retry" button
 *
 * OPACITY SLIDER — RAF DEBOUNCING
 * ─────────────────────────────────
 * The <input type="range"> fires onChange at ~60 events/s during drag.
 * We want the React controlled-input value to update immediately (so the
 * numeric readout stays live) but we debounce the VTK call to one update
 * per animation frame using requestAnimationFrame.  This prevents queuing
 * dozens of redundant render() calls per second.
 *
 * Pattern (mirrors ConnectomePanel filter sliders):
 *   pendingRafRef → stores the RAF id.
 *   On slider change → update local React state immediately →
 *     cancel any pending RAF → schedule a new RAF →
 *     RAF callback calls context.setAnomalyOpacity(value) which updates VTK.
 */

import { useRef, useCallback, type FC } from 'react';
import { useVolumetricContext } from './VolumetricViewer';
import type { AnomalyStatus } from './VolumetricViewer';

// ── Progress bar imports ──────────────────────────────────────────────────────
import InlineTaskProgress from '../../components/progress/InlineTaskProgress';
import { useTaskProgress } from '../../contexts/TaskProgressContext';

// ── Helper: human-readable status label ──────────────────────────────────────

function statusLabel(s: AnomalyStatus): string {
  switch (s.phase) {
    case 'idle':      return 'Idle';
    case 'uploading': return 'Uploading…';
    case 'running':   return 'Analysing…';
    case 'error':     return `Error: ${s.message}`;
    case 'done':      return `Done (${s.durationMs} ms)`;
  }
}

// ── Helper: CSS modifier class for the status dot ────────────────────────────

function dotClass(s: AnomalyStatus): string {
  switch (s.phase) {
    case 'done':      return 'status-dot--done';
    case 'error':     return 'status-dot--error';
    case 'uploading':
    case 'running':   return 'status-dot--running';
    default:          return 'status-dot--idle';
  }
}

// ── Component ─────────────────────────────────────────────────────────────────

const AnomalyPanel: FC = () => {
  const {
    hasVolume,
    anomalyStatus,
    hasAnomalyOverlay,
    showAnomalyOverlay,
    setShowAnomalyOverlay,
    anomalyOpacity,
    setAnomalyOpacity,
    nAnomalyVoxels,
    runAnomalyDetection,
    clearAnomalyOverlay,
  } = useVolumetricContext();

  // Read the global task registry to get the uploadPct for the 'anomaly' task.
  const { tasks } = useTaskProgress();

  // ── RAF debounce ref for the opacity slider ───────────────────────────────
  // Stores the pending requestAnimationFrame id so we can cancel it when a
  // new slider event arrives before the previous frame has fired.
  const pendingRafRef = useRef<number | null>(null);

  const isRunning =
    anomalyStatus.phase === 'uploading' || anomalyStatus.phase === 'running';

  // ── Slider onChange: update React state immediately, debounce VTK call ───
  const handleOpacityChange = useCallback(
    (rawValue: number) => {
      const factor = rawValue / 100; // slider is 0–100 integer

      // Optimistically update the context opacity so the numeric readout
      // reflects the drag position in real-time (no visual lag on the label).
      // setAnomalyOpacity will also schedule the VTK update below.

      // Cancel the previously scheduled RAF to avoid redundant renders.
      if (pendingRafRef.current !== null) {
        cancelAnimationFrame(pendingRafRef.current);
      }

      // Schedule one VTK update per animation frame.
      pendingRafRef.current = requestAnimationFrame(() => {
        pendingRafRef.current = null;
        setAnomalyOpacity(factor);
      });
    },
    [setAnomalyOpacity],
  );

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <section className="control-section anomaly-panel">
      <h3 className="section-title">Deep Sweep — Anomaly Detection</h3>

      {/* ── Hint text ───────────────────────────────────────────────────── */}
      <p className="section-hint anomaly-panel__hint">
        AI-powered lesion &amp; tumour detection (AnomalySeg ONNX).<br />
        Requires <code>AnomalySeg.onnx</code> in <code>backend/models/</code>.
      </p>

      {/* ── Status indicator ────────────────────────────────────────────── */}
      <div className="inference-status">
        <span className={`status-dot ${dotClass(anomalyStatus)}`} />
        <span className="anomaly-panel__status-text">{statusLabel(anomalyStatus)}</span>
      </div>

      {/* ── Inline progress bar — shown during uploading / running phases ── */}
      <InlineTaskProgress
        phase={anomalyStatus.phase}
        uploadPct={tasks.get('anomaly')?.uploadPct ?? 0}
      />

      {/* ── Error detail ────────────────────────────────────────────────── */}
      {anomalyStatus.phase === 'error' && (
        <p className="anomaly-panel__error">{anomalyStatus.message}</p>
      )}

      {/* ── Done: voxel count badge ──────────────────────────────────────── */}
      {anomalyStatus.phase === 'done' && (
        <p className="anomaly-panel__count">
          <span className="anomaly-panel__count-dot" aria-hidden />
          {nAnomalyVoxels.toLocaleString()} anomalous voxels detected
        </p>
      )}

      {/* ── Primary action button ────────────────────────────────────────── */}
      <button
        className="btn btn--primary anomaly-panel__run-btn"
        disabled={!hasVolume || isRunning}
        onClick={runAnomalyDetection}
        title={!hasVolume ? 'Upload a NIfTI scan first' : undefined}
      >
        {isRunning ? 'Running…' : hasAnomalyOverlay ? 'Re-run Deep Sweep' : 'Run Deep Sweep'}
      </button>

      {/* ── Post-detection controls (visible only after a successful run) ── */}
      {hasAnomalyOverlay && (
        <div className="anomaly-panel__overlay-controls">

          {/* Visibility toggle */}
          <div className="control-row control-row--toggle">
            <label className="control-label" htmlFor="anomaly-show">
              Show overlay
            </label>
            <input
              id="anomaly-show"
              type="checkbox"
              checked={showAnomalyOverlay}
              onChange={(e) => setShowAnomalyOverlay(e.target.checked)}
            />
          </div>

          {/* Opacity slider */}
          <div className="control-row anomaly-panel__opacity-row">
            <label className="control-label">
              <span>Opacity</span>
              {/* Live numeric readout — shows context value (updated by RAF) */}
              <span className="control-value">
                {Math.round(anomalyOpacity * 100)}%
              </span>
            </label>
            <input
              type="range"
              className="control-slider anomaly-panel__opacity-slider"
              min={0}
              max={100}
              step={1}
              disabled={!showAnomalyOverlay}
              value={Math.round(anomalyOpacity * 100)}
              onChange={(e) => handleOpacityChange(Number(e.target.value))}
            />
          </div>

          {/* Clear / remove overlay */}
          <button
            className="btn btn--ghost anomaly-panel__clear-btn"
            onClick={clearAnomalyOverlay}
          >
            Clear Sweep
          </button>
        </div>
      )}
    </section>
  );
};

export default AnomalyPanel;
