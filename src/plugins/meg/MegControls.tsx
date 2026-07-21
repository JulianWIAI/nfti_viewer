/**
 * MegControls.tsx — MEG sidebar controls
 * ──────────────────────────────────────────
 *
 * Reads and updates state via MegContext (provided by MegViewer).
 *
 * Controls:
 *   • Recording metadata (read-only)
 *   • Time window (start + width sliders)
 *   • Amplitude scale
 *   • Lane height
 *   • Channel type tabs (mag / grad / eeg / all)
 *   • Channel visibility (checkboxes, select-all / none per type)
 */

import { type FC, useState, useCallback, type DragEvent } from 'react';
import type { PluginControlsProps } from '../../types/plugin.types';
import { useMegContext } from './MegViewer';
import { groupByType } from '../../services/megApi';
import { bidsEventsApi } from '../../services/bidsEventsApi';
import FrequencyDashboard from '../../components/FrequencyDashboard';
import MegTopomap from '../../components/MegTopomap';
import { BAND_COLORS, BAND_LABELS, BAND_ORDER } from '../../lib/meg/bandPower';

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

const MegControls: FC<PluginControlsProps> = () => {
  const {
    payload, controls, setControls, loadingChunk,
    artifacts, spikes, artifactsLoading, spikesLoading,
    artifactsDone, spikesDone, artifactsError, spikesError,
    detectArtifacts, detectSpikes,
    events, setEvents,
    bandTintEnabled, setBandTintEnabled,
  } = useMegContext();
  const [activeType,    setActiveType]    = useState<string>('mag');
  const [tsvLoading,    setTsvLoading]    = useState(false);
  const [tsvError,      setTsvError]      = useState<string | null>(null);
  const [tsvFilename,   setTsvFilename]   = useState<string | null>(null);
  const [isDraggingOver, setIsDraggingOver] = useState(false);

  const handleTsvDragOver  = useCallback((e: DragEvent<HTMLDivElement>) => { e.preventDefault(); setIsDraggingOver(true); }, []);
  const handleTsvDragLeave = useCallback(() => setIsDraggingOver(false), []);
  const handleTsvDrop = useCallback(async (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingOver(false);
    const file = e.dataTransfer.files[0];
    if (!file || !file.name.toLowerCase().endsWith('.tsv')) {
      setTsvError('Please drop a BIDS *_events.tsv file.');
      return;
    }
    setTsvLoading(true); setTsvError(null);
    try {
      const result = await bidsEventsApi.uploadEventsFile(file);
      setEvents(result.events);
      setTsvFilename(file.name);
    } catch (err) {
      setTsvError(err instanceof Error ? err.message : String(err));
    } finally {
      setTsvLoading(false);
    }
  }, [setEvents]);

  const hasData    = payload !== null;
  const duration   = payload?.totalDuration ?? 0;
  const windowWidth = controls.timeEnd - controls.timeStart;

  const formatTime = (s: number): string => {
    if (s < 60) return `${s.toFixed(1)} s`;
    return `${Math.floor(s / 60)}:${(s % 60).toFixed(0).padStart(2, '0')}`;
  };

  // Channel list for the active type tab
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
      {/* ── BIDS Events ──────────────────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">BIDS Events</h3>
        <div
          className={`tsv-dropzone${isDraggingOver ? ' tsv-dropzone--over' : ''}`}
          onDragOver={handleTsvDragOver}
          onDragLeave={handleTsvDragLeave}
          onDrop={handleTsvDrop}
          aria-label="Drop BIDS events.tsv here"
        >
          {tsvLoading ? (
            <span className="tsv-dropzone__label">Parsing…</span>
          ) : tsvFilename ? (
            <span className="tsv-dropzone__label tsv-dropzone__label--loaded">
              ✓ {tsvFilename}
            </span>
          ) : (
            <span className="tsv-dropzone__label">Drop *_events.tsv</span>
          )}
        </div>
        {tsvError && (
          <p style={{ color: 'var(--accent-red, #ff4d4f)', fontSize: 10, marginTop: 4 }}>
            {tsvError}
          </p>
        )}
        {events.length > 0 && (
          <p style={{ fontSize: 10, color: '#9a9ab8', marginTop: 4 }}>
            {events.length} events · {[...new Set(events.map((e) => e.trialType))].join(', ')}
          </p>
        )}
      </section>

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

      {/* ── Signal Analysis ──────────────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">Signal Analysis</h3>
        <p className="section-hint">
          Runs on the server via MNE-Python.
        </p>

        {/* Artifact detection */}
        <button
          className="btn btn--primary"
          disabled={!hasData || artifactsLoading}
          onClick={() => void detectArtifacts()}
          style={{ width: '100%', marginBottom: 4 }}
        >
          {artifactsLoading ? 'Detecting…' : 'Detect Artefacts'}
        </button>
        {artifactsError && (
          <p style={{ color: 'var(--accent-red, #e05252)', fontSize: 9, margin: '0 0 6px' }}>
            {artifactsError}
          </p>
        )}
        {artifactsDone && !artifactsError && (
          <p style={{ color: '#9a9ab8', fontSize: 9, margin: '0 0 6px' }}>
            {artifacts.filter((a) => a.type === 'blink').length} blinks,&nbsp;
            {artifacts.filter((a) => a.type === 'muscle').length} muscle spans found
          </p>
        )}

        {/* Spike detection */}
        <button
          className="btn btn--secondary"
          disabled={!hasData || spikesLoading}
          onClick={() => void detectSpikes()}
          style={{ width: '100%', marginBottom: 4 }}
        >
          {spikesLoading ? 'Detecting…' : 'Detect Spikes'}
        </button>
        {spikesError && (
          <p style={{ color: 'var(--accent-red, #e05252)', fontSize: 9, margin: '0 0 6px' }}>
            {spikesError}
          </p>
        )}
        {spikesDone && !spikesError && (
          <p style={{ color: '#9a9ab8', fontSize: 9, margin: '0 0 6px' }}>
            {spikes.length} spike{spikes.length !== 1 ? 's' : ''} detected
          </p>
        )}
      </section>

      {/* ── Frequency Bands ──────────────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">Frequency Bands</h3>
        <FrequencyDashboard
          sessionId={payload?.sessionId}
          tStart={controls.timeStart}
          tEnd={controls.timeEnd}
        />
      </section>

      {/* ── Topomap ──────────────────────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">Topomap</h3>
        <p className="section-hint">Average field map over the current time window.</p>
        <MegTopomap
          sessionId={payload?.sessionId}
          tStart={controls.timeStart}
          tEnd={controls.timeEnd}
        />
      </section>

      {/* ── Amplitude ────────────────────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">Amplitude</h3>
        <SliderRow
          label="Scale" value={parseFloat(controls.amplitudeScale.toFixed(1))}
          min={0.1} max={5} step={0.1} disabled={!hasData}
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

      {/* ── Band Tint ────────────────────────────────────────────────────── */}
      {/* Paints each lane background with its sliding-window dominant frequency
          band colour so the user can see exactly where one oscillation ends and
          the next begins.  Toggle off for a clean waveform-only view. */}
      <section className="control-section">
        <h3 className="section-title">Lane Background Tint</h3>
        <button
          className={`btn ${bandTintEnabled ? 'btn--primary' : 'btn--secondary'}`}
          disabled={!hasData}
          onClick={() => setBandTintEnabled(!bandTintEnabled)}
          style={{ width: '100%', marginBottom: 6 }}
          title="Colour each channel lane with the dominant frequency band at each time point"
        >
          {bandTintEnabled ? 'Tint: ON' : 'Tint: OFF'}
        </button>
        {/* Band colour legend */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          {BAND_ORDER.map((band) => (
            <div key={band} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span
                style={{
                  display:      'inline-block',
                  width:        10,
                  height:       10,
                  borderRadius: 2,
                  background:   BAND_COLORS[band],
                  flexShrink:   0,
                  opacity:      0.85,
                }}
              />
              <span style={{ fontSize: 9, color: '#9a9ab8', fontFamily: 'monospace' }}>
                {BAND_LABELS[band]}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Channel type tabs + list ──────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">
          Channels
          {loadingChunk && (
            <span
              className="status-dot status-dot--running"
              aria-label="Fetching chunk"
              style={{ marginLeft: 6 }}
            />
          )}
        </h3>

        {hasData && (
          <>
            {/* Type tabs */}
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

            {/* Select-all / none for current type */}
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
            <span className="channel-list__empty">No MEG file loaded</span>
          )}
        </div>
      </section>

      {/* ── Recording metadata ────────────────────────────────────────────── */}
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

export default MegControls;
