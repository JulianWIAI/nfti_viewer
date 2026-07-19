/**
 * ControlPanel.tsx — Live rendering controls
 * ──────────────────────────────────────────
 *
 * Exposes sliders and inputs for:
 *   • Slice position in each MPR plane (Axial K, Coronal J, Sagittal I)
 *   • Window/level (brightness & contrast)
 *   • 3-D volume opacity
 *   • Show/hide 3-D volume overlay
 *   • Trigger ONNX brain-tissue segmentation
 *
 * All state lives in App.tsx (passed down as `controls` + `onControlChange`).
 * This component is purely presentational — it calls the callback and re-renders
 * when props change. No local state except ONNX status.
 */

import { useState, useCallback } from 'react';
import type { ViewerControls, SliceMaxima } from '../types/nifti.types';
import type { InferenceStatus } from '../lib/onnx/inferenceEngine';

// ── Props ─────────────────────────────────────────────────────────────────────

interface ControlPanelProps {
  controls: ViewerControls;
  maxima: SliceMaxima;
  /** Whether a volume is currently loaded (enables the controls). */
  hasVolume: boolean;
  onControlChange: (partial: Partial<ViewerControls>) => void;
  /** Called when the user clicks "Run Segmentation". */
  onRunSegmentation: () => void;
  inferenceStatus: InferenceStatus;
}

// ── Sub-components ────────────────────────────────────────────────────────────

interface SliderRowProps {
  label: string;
  value: number;
  min: number;
  max: number;
  disabled: boolean;
  onChange: (v: number) => void;
}

function SliderRow({ label, value, min, max, disabled, onChange }: SliderRowProps) {
  return (
    <div className="control-row">
      <label className="control-label">
        <span>{label}</span>
        <span className="control-value">{value}</span>
      </label>
      <input
        type="range"
        className="control-slider"
        min={min}
        max={max}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ControlPanel({
  controls,
  maxima,
  hasVolume,
  onControlChange,
  onRunSegmentation,
  inferenceStatus,
}: ControlPanelProps) {
  // Track whether the ONNX model path input is expanded (advanced section)
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Stable callback factories to avoid inline functions in JSX
  const setSliceK = useCallback(
    (v: number) => onControlChange({ sliceK: v }), [onControlChange]);
  const setSliceJ = useCallback(
    (v: number) => onControlChange({ sliceJ: v }), [onControlChange]);
  const setSliceI = useCallback(
    (v: number) => onControlChange({ sliceI: v }), [onControlChange]);
  const setWindowWidth = useCallback(
    (v: number) => onControlChange({ windowWidth: v }), [onControlChange]);
  const setWindowCenter = useCallback(
    (v: number) => onControlChange({ windowCenter: v }), [onControlChange]);
  const setVolumeOpacity = useCallback(
    (v: number) => onControlChange({ volumeOpacity: v / 100 }), [onControlChange]);

  // ── Inference status label ──────────────────────────────────────────────
  const statusLabel = (): string => {
    switch (inferenceStatus.phase) {
      case 'idle':            return 'Idle';
      case 'loading_model':   return 'Loading model…';
      case 'preprocessing':   return 'Preprocessing…';
      case 'running':         return 'Running model…';
      case 'error':           return `Error: ${inferenceStatus.message}`;
      case 'done':            return `Done (${inferenceStatus.durationMs} ms)`;
    }
  };
  const isRunning = ['loading_model', 'preprocessing', 'running'].includes(
    inferenceStatus.phase,
  );

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <aside className="control-panel">
      <h2 className="panel-title">Controls</h2>

      {/* ── MPR Slices ─────────────────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">MPR Slices</h3>

        <SliderRow
          label="Axial (K)"
          value={controls.sliceK}
          min={0}
          max={Math.max(0, maxima.maxK - 1)}
          disabled={!hasVolume}
          onChange={setSliceK}
        />
        <SliderRow
          label="Coronal (J)"
          value={controls.sliceJ}
          min={0}
          max={Math.max(0, maxima.maxJ - 1)}
          disabled={!hasVolume}
          onChange={setSliceJ}
        />
        <SliderRow
          label="Sagittal (I)"
          value={controls.sliceI}
          min={0}
          max={Math.max(0, maxima.maxI - 1)}
          disabled={!hasVolume}
          onChange={setSliceI}
        />
      </section>

      {/* ── Window / Level ──────────────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">Window / Level</h3>

        <SliderRow
          label="Width"
          value={controls.windowWidth}
          min={1}
          max={4000}
          disabled={!hasVolume}
          onChange={setWindowWidth}
        />
        <SliderRow
          label="Centre"
          value={controls.windowCenter}
          min={-1000}
          max={3000}
          disabled={!hasVolume}
          onChange={setWindowCenter}
        />
      </section>

      {/* ── 3-D Volume ──────────────────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">3-D Volume</h3>

        <div className="control-row control-row--toggle">
          <label className="control-label" htmlFor="show-volume">
            Show volume
          </label>
          <input
            id="show-volume"
            type="checkbox"
            checked={controls.showVolume}
            disabled={!hasVolume}
            onChange={(e) => onControlChange({ showVolume: e.target.checked })}
          />
        </div>

        <SliderRow
          label="Opacity"
          value={Math.round(controls.volumeOpacity * 100)}
          min={0}
          max={100}
          disabled={!hasVolume || !controls.showVolume}
          onChange={setVolumeOpacity}
        />
      </section>

      {/* ── ONNX Segmentation ───────────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">AI Segmentation</h3>

        <p className="section-hint">
          Place a <code>.onnx</code> model at{' '}
          <code>public/models/brain_seg.onnx</code>, then click Run.
        </p>

        <div className="inference-status">
          <span
            className={`status-dot status-dot--${inferenceStatus.phase === 'done' ? 'done' : inferenceStatus.phase === 'error' ? 'error' : isRunning ? 'running' : 'idle'}`}
          />
          <span>{statusLabel()}</span>
        </div>

        <button
          className="btn btn--primary"
          disabled={!hasVolume || isRunning}
          onClick={onRunSegmentation}
        >
          {isRunning ? 'Running…' : 'Run Segmentation'}
        </button>

        {/* Advanced: custom model path */}
        <button
          className="btn btn--ghost"
          onClick={() => setShowAdvanced((v) => !v)}
        >
          {showAdvanced ? '▲ Hide advanced' : '▼ Advanced'}
        </button>

        {showAdvanced && (
          <p className="section-hint" style={{ marginTop: 8 }}>
            To use a custom model, update the <code>MODEL_PATH</code> constant
            in <code>src/components/Viewer.tsx</code> and rebuild.
          </p>
        )}
      </section>

      {/* ── Volume metadata ─────────────────────────────────────────────── */}
      {hasVolume && (
        <section className="control-section control-section--meta">
          <h3 className="section-title">Metadata</h3>
          <dl className="meta-list">
            <dt>Dimensions</dt>
            <dd>{maxima.maxI} × {maxima.maxJ} × {maxima.maxK}</dd>
          </dl>
        </section>
      )}
    </aside>
  );
}
