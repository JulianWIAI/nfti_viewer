/**
 * TimeseriesControls.tsx — EEG / MEG sidebar controls
 * ──────────────────────────────────────────────────────
 *
 * Reads and updates state via TimeseriesContext (provided by TimeseriesViewer).
 * This is the ControlsComponent for the timeseries plugin.
 *
 * Controls exposed:
 *   • Time window (start + end via sliders, or direct epoch navigation)
 *   • Amplitude scale (µV / lane height)
 *   • Lane height in pixels
 *   • Channel visibility (virtual list with select-all / deselect-all)
 *   • Recording metadata (read-only)
 */

import { type FC, useMemo, useCallback } from 'react';
import { useTimeseriesContext } from './TimeseriesViewer';
import { useReferencePanel } from '../../contexts/ReferencePanelContext';
import type { PluginControlsProps } from '../../types/plugin.types';

// ── Reusable slider row ───────────────────────────────────────────────────────

interface SliderRowProps {
  label:    string;
  value:    number;
  min:      number;
  max:      number;
  step?:    number;
  disabled: boolean;
  format?:  (v: number) => string;
  onChange: (v: number) => void;
}

function SliderRow({ label, value, min, max, step = 1, disabled, format, onChange }: SliderRowProps) {
  const display = format ? format(value) : String(value);
  return (
    <div className="control-row">
      <label className="control-label">
        <span>{label}</span>
        <span className="control-value">{display}</span>
      </label>
      <input
        type="range"
        className="control-slider"
        min={min} max={max} step={step} value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

// ── Frequency band quick-nav descriptors ──────────────────────────────────────

const FREQ_BANDS = [
  { id: 'delta', label: 'δ', title: 'Delta  0.5 – 4 Hz'   },
  { id: 'theta', label: 'θ', title: 'Theta  4 – 8 Hz'     },
  { id: 'alpha', label: 'α', title: 'Alpha  8 – 13 Hz'    },
  { id: 'beta',  label: 'β', title: 'Beta  13 – 30 Hz'    },
  { id: 'gamma', label: 'γ', title: 'Gamma  30 – 100 Hz'  },
] as const;

// ── Component ─────────────────────────────────────────────────────────────────

const TimeseriesControls: FC<PluginControlsProps> = () => {
  const { payload, controls, setControls, totalDuration } = useTimeseriesContext();
  const { openDrawer, setActiveTab } = useReferencePanel();

  // Opens the Reference Drawer on the electrophysiology tab
  const openElectrophysiology = useCallback(() => {
    openDrawer();
    setActiveTab('electrophysiology');
  }, [openDrawer, setActiveTab]);

  const hasData = payload !== null;
  const windowWidth = controls.timeEnd - controls.timeStart;

  // Format seconds as mm:ss or s
  const formatTime = (s: number): string => {
    if (s < 60) return `${s.toFixed(1)} s`;
    return `${Math.floor(s / 60)}:${(s % 60).toFixed(0).padStart(2, '0')}`;
  };

  // All channel indices sorted by visibility
  const allChannels = useMemo(
    () => (payload ? payload.channels.map((ch, i) => ({ ch, i })) : []),
    [payload],
  );

  const selectedSet = useMemo(
    () => new Set(controls.selectedChannels),
    [controls.selectedChannels],
  );

  function toggleChannel(idx: number) {
    const next = selectedSet.has(idx)
      ? controls.selectedChannels.filter((i) => i !== idx)
      : [...controls.selectedChannels, idx].sort((a, b) => a - b);
    setControls({ selectedChannels: next });
  }

  function selectAll() {
    setControls({ selectedChannels: allChannels.map((c) => c.i) });
  }

  function deselectAll() {
    setControls({ selectedChannels: [] });
  }

  // Metadata block
  const meta = payload
    ? [
        ['File',       payload.filename],
        ['Modality',   payload.sourceModality.toUpperCase()],
        ['Channels',   String(payload.channels.length)],
        ['Duration',   formatTime(totalDuration)],
        ['Samples',    payload.numSamples.toLocaleString()],
        ['Rate',       payload.channels[0]
          ? `${payload.channels[0].sampleRate.toFixed(1)} Hz`
          : '—'],
      ]
    : [];

  return (
    <>
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
            const newEnd = Math.min(totalDuration, controls.timeStart + v);
            const newStart = Math.max(0, newEnd - v);
            setControls({ timeStart: newStart, timeEnd: newEnd });
          }}
        />
      </section>

      {/* ── Amplitude ──────────────────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">Amplitude</h3>
        <SliderRow
          label="Scale (µV)" value={Math.round(controls.amplitudeScale)} min={1} max={500}
          disabled={!hasData}
          onChange={(v) => setControls({ amplitudeScale: v })}
        />
        <SliderRow
          label="Lane height" value={controls.laneHeightPx} min={20} max={200}
          disabled={!hasData} format={(v) => `${v} px`}
          onChange={(v) => setControls({ laneHeightPx: v })}
        />
      </section>

      {/* ── Channel list ───────────────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">Channels</h3>
        {hasData && (
          <div className="control-row control-row--buttons">
            <button className="btn btn--secondary btn--sm" onClick={selectAll}>
              All
            </button>
            <button className="btn btn--secondary btn--sm" onClick={deselectAll}>
              None
            </button>
          </div>
        )}
        <div className="channel-list">
          {allChannels.map(({ ch, i }) => (
            <label key={i} className="channel-item">
              <input
                type="checkbox"
                checked={selectedSet.has(i)}
                onChange={() => toggleChannel(i)}
              />
              <span className="channel-label">{ch.label}</span>
              <span className="channel-unit">{ch.unit}</span>
            </label>
          ))}
          {!hasData && (
            <span className="channel-list__empty">No data loaded</span>
          )}
        </div>
      </section>

      {/* ── Electrophysiology Reference ─────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">EEG / MEG Reference</h3>
        {/* Frequency band badges — click to jump to electrophysiology tab */}
        <div className="ts-freq-badges">
          {FREQ_BANDS.map((band) => (
            <button
              key={band.id}
              className="ts-freq-badge"
              title={band.title}
              onClick={openElectrophysiology}
            >
              {band.label}
            </button>
          ))}
        </div>
        <button
          className="btn btn--secondary btn--sm"
          style={{ width: '100%', marginTop: 6 }}
          onClick={openElectrophysiology}
        >
          📖 Open ERP / HRF Reference
        </button>
      </section>

      {/* ── Metadata ───────────────────────────────────────────────────── */}
      {hasData && (
        <section className="control-section control-section--meta">
          <h3 className="section-title">Info</h3>
          <dl className="meta-list">
            {meta.map(([k, v]) => (
              <div key={k} style={{ display: 'contents' }}>
                <dt>{k}</dt><dd>{v}</dd>
              </div>
            ))}
          </dl>
        </section>
      )}
    </>
  );
};

export default TimeseriesControls;
