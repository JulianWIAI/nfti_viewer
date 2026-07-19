/**
 * NirsControls.tsx — fNIRS sidebar controls
 * ──────────────────────────────────────────
 *
 * Reads and updates state via NirsContext (provided by NirsViewer).
 * This is the ControlsComponent for the NIRS plugin.
 *
 * Controls exposed:
 *   • Wavelength selector (tabs for each wavelength in the probe)
 *   • Time window (start / width)
 *   • Amplitude scale
 *   • Recording metadata (read-only)
 */

import type { FC } from 'react';
import { useNirsContext } from './NirsViewer';
import type { PluginControlsProps } from '../../types/plugin.types';

// ── Reusable slider row ───────────────────────────────────────────────────────

interface SliderRowProps {
  label: string; value: number; min: number; max: number;
  step?: number; disabled: boolean;
  format?: (v: number) => string;
  onChange: (v: number) => void;
}

function SliderRow({ label, value, min, max, step = 1, disabled, format, onChange }: SliderRowProps) {
  return (
    <div className="control-row">
      <label className="control-label">
        <span>{label}</span>
        <span className="control-value">{format ? format(value) : value}</span>
      </label>
      <input type="range" className="control-slider"
        min={min} max={max} step={step} value={value} disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

const NirsControls: FC<PluginControlsProps> = () => {
  const {
    snirfPayload, controls, setControls,
    totalDuration, selectedWavelengthIdx, setSelectedWavelengthIdx,
  } = useNirsContext();

  const hasData    = snirfPayload !== null;
  const windowWidth = controls.timeEnd - controls.timeStart;

  const formatTime = (s: number) =>
    s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)}:${(s % 60).toFixed(0).padStart(2, '0')}`;

  return (
    <>
      {/* ── Wavelength selector ─────────────────────────────────────────── */}
      {hasData && snirfPayload!.wavelengths.length > 0 && (
        <section className="control-section">
          <h3 className="section-title">Wavelength</h3>
          <div className="control-row control-row--buttons">
            {Array.from(snirfPayload!.wavelengths).map((wl, i) => (
              <button
                key={i}
                className={`btn btn--secondary btn--sm${selectedWavelengthIdx === i + 1 ? ' btn--active' : ''}`}
                onClick={() => setSelectedWavelengthIdx(i + 1)}
              >
                {Math.round(wl)} nm
              </button>
            ))}
          </div>
        </section>
      )}

      {/* ── Time window ────────────────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">Time Window</h3>
        <SliderRow
          label="Start" value={Math.round(controls.timeStart)} min={0}
          max={Math.max(0, Math.round(totalDuration - windowWidth))}
          disabled={!hasData} format={formatTime}
          onChange={(v) => setControls({ timeStart: v, timeEnd: v + windowWidth })}
        />
        <SliderRow
          label="Width" value={Math.round(windowWidth)} min={1}
          max={Math.max(1, Math.round(totalDuration))}
          disabled={!hasData} format={formatTime}
          onChange={(v) => {
            const newEnd   = Math.min(totalDuration, controls.timeStart + v);
            const newStart = Math.max(0, newEnd - v);
            setControls({ timeStart: newStart, timeEnd: newEnd });
          }}
        />
      </section>

      {/* ── Amplitude ──────────────────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">Amplitude</h3>
        <SliderRow
          label="Scale" value={controls.amplitudeScale} min={0.001} max={0.5} step={0.001}
          disabled={!hasData} format={(v) => v.toFixed(3)}
          onChange={(v) => setControls({ amplitudeScale: v })}
        />
        <SliderRow
          label="Lane height" value={controls.laneHeightPx} min={20} max={200}
          disabled={!hasData} format={(v) => `${v} px`}
          onChange={(v) => setControls({ laneHeightPx: v })}
        />
      </section>

      {/* ── Metadata ───────────────────────────────────────────────────── */}
      {hasData && (
        <section className="control-section control-section--meta">
          <h3 className="section-title">Info</h3>
          <dl className="meta-list">
            <dt>Channels</dt>   <dd>{snirfPayload!.numChannels}</dd>
            <dt>Time points</dt><dd>{snirfPayload!.numTimePoints.toLocaleString()}</dd>
            <dt>Duration</dt>   <dd>{formatTime(totalDuration)}</dd>
            <dt>Sources</dt>    <dd>{snirfPayload!.sourcePositions.length}</dd>
            <dt>Detectors</dt>  <dd>{snirfPayload!.detectorPositions.length}</dd>
            <dt>Wavelengths</dt>
            <dd>
              {Array.from(snirfPayload!.wavelengths)
                .map((w) => `${Math.round(w)} nm`)
                .join(', ')}
            </dd>
          </dl>
        </section>
      )}
    </>
  );
};

export default NirsControls;
