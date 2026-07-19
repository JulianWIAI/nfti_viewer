/**
 * EegControls.tsx — EEG sidebar controls
 * ─────────────────────────────────────────
 *
 * Reads and updates state via EegContext (provided by EegViewer).
 *
 * Controls:
 *   • Time window (start + width sliders)
 *   • Amplitude scale + lane height
 *   • Channel type tabs (eeg / eog / misc / all)
 *   • Channel visibility checkboxes with select-all / none
 *   • Recording metadata (read-only)
 */

import { type FC, useState } from 'react';
import type { PluginControlsProps } from '../../types/plugin.types';
import { useEegContext } from './EegViewer';
import { groupByType } from '../../services/megApi';

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

const EegControls: FC<PluginControlsProps> = () => {
  const { payload, controls, setControls, loadingChunk } = useEegContext();
  const [activeType, setActiveType] = useState<string>('eeg');

  const hasData     = payload !== null;
  const duration    = payload?.totalDuration ?? 0;
  const windowWidth = controls.timeEnd - controls.timeStart;

  const formatTime = (s: number): string => {
    if (s < 60) return `${s.toFixed(1)} s`;
    return `${Math.floor(s / 60)}:${(s % 60).toFixed(0).padStart(2, '0')}`;
  };

  const byType          = payload ? groupByType(payload.channels) : {};
  const types           = Object.keys(byType).sort();
  const showAll         = activeType === 'all';
  const displayChannels = showAll
    ? (payload?.channels ?? [])
    : (byType[activeType] ?? []);

  function toggleChannel(name: string) {
    const sel = controls.selectedChannels;
    setControls({
      selectedChannels: sel.includes(name)
        ? sel.filter((n) => n !== name)
        : [...sel, name],
    });
  }

  function selectAll() {
    const names = displayChannels.map((c) => c.name);
    setControls({ selectedChannels: [...new Set([...controls.selectedChannels, ...names])] });
  }

  function selectNone() {
    const names = new Set(displayChannels.map((c) => c.name));
    setControls({ selectedChannels: controls.selectedChannels.filter((n) => !names.has(n)) });
  }

  return (
    <>
      {/* ── Time window ──────────────────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">Time Window</h3>
        <SliderRow
          label="Start" value={parseFloat(controls.timeStart.toFixed(1))}
          min={0} max={Math.max(0, parseFloat((duration - windowWidth).toFixed(1)))}
          step={0.1} disabled={!hasData} format={formatTime}
          onChange={(v) => setControls({ timeStart: v, timeEnd: v + windowWidth })}
        />
        <SliderRow
          label="Width" value={parseFloat(windowWidth.toFixed(1))}
          min={0.1} max={Math.min(30, duration)} step={0.1}
          disabled={!hasData} format={formatTime}
          onChange={(v) => {
            const end   = Math.min(duration, controls.timeStart + v);
            const start = Math.max(0, end - v);
            setControls({ timeStart: start, timeEnd: end });
          }}
        />
      </section>

      {/* ── Amplitude ────────────────────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">Amplitude</h3>
        <SliderRow
          label="Scale" value={parseFloat(controls.amplitudeScale.toFixed(1))}
          min={0.1} max={10} step={0.1} disabled={!hasData}
          format={(v) => `×${v.toFixed(1)}`}
          onChange={(v) => setControls({ amplitudeScale: v })}
        />
        <SliderRow
          label="Lane height" value={controls.laneHeightPx}
          min={20} max={200} disabled={!hasData}
          format={(v) => `${v} px`}
          onChange={(v) => setControls({ laneHeightPx: v })}
        />
      </section>

      {/* ── Channel list ──────────────────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">
          Channels
          {loadingChunk && (
            <span
              className="status-dot status-dot--running"
              aria-label="Fetching data"
              style={{ marginLeft: 6 }}
            />
          )}
        </h3>

        {hasData && (
          <>
            <div className="control-row--buttons">
              <button
                className={`btn btn--sm ${activeType === 'all' ? 'btn--active' : 'btn--secondary'}`}
                onClick={() => setActiveType('all')}
              >
                All ({payload!.channels.length})
              </button>
              {types.map((t) => (
                <button
                  key={t}
                  className={`btn btn--sm ${activeType === t ? 'btn--active' : 'btn--secondary'}`}
                  onClick={() => setActiveType(t)}
                >
                  {t} ({byType[t]!.length})
                </button>
              ))}
            </div>

            <div className="control-row--buttons" style={{ marginBottom: 4 }}>
              <button className="btn btn--sm btn--secondary" onClick={selectAll}>All</button>
              <button className="btn btn--sm btn--secondary" onClick={selectNone}>None</button>
            </div>
          </>
        )}

        <div className="channel-list">
          {displayChannels.map((ch) => (
            <label key={ch.name} className="channel-item">
              <input
                type="checkbox"
                checked={controls.selectedChannels.includes(ch.name)}
                onChange={() => toggleChannel(ch.name)}
              />
              <span className="channel-label">{ch.name}</span>
              <span className="channel-unit">{ch.unit}</span>
            </label>
          ))}
          {!hasData && (
            <span className="channel-list__empty">
              Drop .vhdr + .eeg + .vmrk files together to load EEG
            </span>
          )}
        </div>
      </section>

      {/* ── Recording info ────────────────────────────────────────────────── */}
      {hasData && (
        <section className="control-section control-section--meta">
          <h3 className="section-title">Info</h3>
          <dl className="meta-list">
            {[
              ['File',     payload!.filename],
              ['Channels', String(payload!.nChannels)],
              ['Rate',     `${payload!.samplingRate} Hz`],
              ['Duration', formatTime(duration)],
              ['Samples',  payload!.nSamples.toLocaleString()],
            ].map(([k, v]) => (
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

export default EegControls;
