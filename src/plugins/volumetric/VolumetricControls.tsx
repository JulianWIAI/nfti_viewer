/**
 * VolumetricControls.tsx — MRI/PET sidebar controls
 * ─────────────────────────────────────────────────────
 *
 * Reads and updates state via VolumetricContext (Subject A) and
 * ComparisonContext (Subject B, Phase 9).
 *
 * Controls exposed:
 *   Subject A
 *   • MPR slice positions (Axial K, Coronal J, Sagittal I)
 *   • Window / Level (brightness & contrast)
 *   • 3-D volume opacity + show/hide toggle
 *   • SynthSeg brain anatomy segmentation trigger + overlay toggle
 *   • Hippocampal volumetrics (comparative when B is loaded)
 *
 *   Subject B (Phase 9 — shown after SubjectBDropZone receives a file)
 *   • Drop zone — accepts .nii / .nii.gz drag-and-drop
 *   • Independent MPR slice + window/level controls
 *   • Independent tissue-class overlay toggles
 *   • Clear button to return to single-brain mode
 */

import type { FC } from 'react';
import { useVolumetricContext } from './VolumetricViewer';
import { useComparisonContext } from './RawComparisonContext';
import type { InferenceStatus } from './VolumetricViewer';
import type { PluginControlsProps } from '../../types/plugin.types';
import AnatomySelector       from '../../components/AnatomySelector';
import SubjectBAnatomySelector from './SubjectBAnatomySelector';
import DecodingPanel from './DecodingPanel';
import ConnectomePanel from './connectome/ConnectomePanel';
import AnomalyPanel from './AnomalyPanel';
import LongitudinalPanel from './LongitudinalPanel';
import SubjectBDropZone from './SubjectBDropZone';
import VolumetricsChart from './VolumetricsChart';

// ── Reusable slider row ───────────────────────────────────────────────────────

interface SliderRowProps {
  label: string; value: number; min: number; max: number;
  disabled: boolean; onChange: (v: number) => void;
}

function SliderRow({ label, value, min, max, disabled, onChange }: SliderRowProps) {
  return (
    <div className="control-row">
      <label className="control-label">
        <span>{label}</span>
        <span className="control-value">{value}</span>
      </label>
      <input type="range" className="control-slider"
        min={min} max={max} value={value} disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

// ── Status label helper ────────────────────────────────────────────────────────

function statusLabel(s: InferenceStatus): string {
  switch (s.phase) {
    case 'idle':      return 'Idle';
    case 'uploading': return 'Uploading…';
    case 'running':   return 'Segmenting…';
    case 'error':     return `Error: ${s.message}`;
    case 'done':      return `Done (${s.durationMs} ms)`;
  }
}

// ── Component ─────────────────────────────────────────────────────────────────

const VolumetricControls: FC<PluginControlsProps> = () => {
  const {
    controls, setControls, maxI, maxJ, maxK,
    inferenceStatus, hasVolume, hasOverlay, showOverlay,
    setShowOverlay, runSegmentation,
    volumetrics, volumetricsLoading, volumetricsError, runVolumetrics,
  } = useVolumetricContext();

  const {
    hasVolumeB, loadVolumeB, clearVolumeB,
    controlsB, setControlsB, maxIB, maxJB, maxKB,
    inferenceStatusB, hasOverlayB, showOverlayB, setShowOverlayB,
    volumetricsB, volumetricsLoadingB, volumetricsErrorB, runVolumetricsB,
  } = useComparisonContext();

  const isRunning = inferenceStatus.phase === 'uploading' || inferenceStatus.phase === 'running';

  const dotClass = (s: InferenceStatus): string =>
    s.phase === 'done'  ? 'status-dot--done'
    : s.phase === 'error' ? 'status-dot--error'
    : (s.phase === 'uploading' || s.phase === 'running') ? 'status-dot--running'
    : 'status-dot--idle';

  return (
    <>
      {/* ── Subject A — MPR slices ──────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">{hasVolumeB ? 'Subject A — MPR Slices' : 'MPR Slices'}</h3>
        <SliderRow label="Axial (K)"    value={controls.sliceK} min={0} max={Math.max(0, maxK - 1)} disabled={!hasVolume} onChange={(v) => setControls({ sliceK: v })} />
        <SliderRow label="Coronal (J)"  value={controls.sliceJ} min={0} max={Math.max(0, maxJ - 1)} disabled={!hasVolume} onChange={(v) => setControls({ sliceJ: v })} />
        <SliderRow label="Sagittal (I)" value={controls.sliceI} min={0} max={Math.max(0, maxI - 1)} disabled={!hasVolume} onChange={(v) => setControls({ sliceI: v })} />
      </section>

      {/* ── Subject A — Window / Level ──────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">{hasVolumeB ? 'Subject A — Window / Level' : 'Window / Level'}</h3>
        <SliderRow label="Width"  value={controls.windowWidth}  min={1}     max={4000} disabled={!hasVolume} onChange={(v) => setControls({ windowWidth: v })} />
        <SliderRow label="Centre" value={controls.windowCenter} min={-1000} max={3000} disabled={!hasVolume} onChange={(v) => setControls({ windowCenter: v })} />
      </section>

      {/* ── Subject A — 3-D Volume ──────────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">{hasVolumeB ? 'Subject A — 3-D Volume' : '3-D Volume'}</h3>
        <div className="control-row control-row--toggle">
          <label className="control-label" htmlFor="vol-show">Show volume</label>
          <input id="vol-show" type="checkbox" checked={controls.showVolume} disabled={!hasVolume}
            onChange={(e) => setControls({ showVolume: e.target.checked })} />
        </div>
        <SliderRow label="Opacity" value={Math.round(controls.volumeOpacity * 100)} min={0} max={100}
          disabled={!hasVolume || !controls.showVolume}
          onChange={(v) => setControls({ volumeOpacity: v / 100 })} />
      </section>

      {/* ── Subject A — AI Segmentation ─────────────────────────────────── */}
      <section className="control-section">
        <h3 className="section-title">{hasVolumeB ? 'Subject A — Segmentation' : 'SynthSeg Segmentation'}</h3>
        {!hasVolumeB && (
          <p className="section-hint">
            Anatomy: 33 brain structures via server-side SynthSeg.<br />
            Requires <code>python backend/download_models.py</code>.
          </p>
        )}
        <div className="inference-status">
          <span className={`status-dot ${dotClass(inferenceStatus)}`} />
          <span>{statusLabel(inferenceStatus)}</span>
        </div>
        {/* Manual single-subject run — available when B is NOT loaded */}
        {!hasVolumeB && (
          <button className="btn btn--primary"
            disabled={!hasVolume || isRunning}
            onClick={runSegmentation}>
            {isRunning ? 'Running…' : 'Run Segmentation'}
          </button>
        )}
        {/* When B is loaded, segmentation was auto-triggered; show status only */}
        {hasVolumeB && (
          <p style={{ fontSize: 10, color: '#666', marginTop: 4 }}>
            Auto-run concurrently with Subject B on drop.
          </p>
        )}
        {hasOverlay && (
          <div className="control-row control-row--toggle" style={{ marginTop: 8 }}>
            <label className="control-label" htmlFor="seg-show">Show overlay</label>
            <input id="seg-show" type="checkbox" checked={showOverlay}
              onChange={(e) => setShowOverlay(e.target.checked)} />
          </div>
        )}
      </section>

      {/* ── Subject A — Anatomy selector (per-structure toggles) ─────────── */}
      {hasOverlay && <AnatomySelector />}

      {/* ── Subject B — Anatomy selector (independent per-structure toggles) */}
      {hasOverlayB && <SubjectBAnatomySelector />}

      {/* ── Subject B drop zone (shown when A is loaded, B is not) ───────── */}
      {hasVolume && !hasVolumeB && (
        <SubjectBDropZone onFile={loadVolumeB} />
      )}

      {/* ── Subject B — controls (shown when B is loaded) ────────────────── */}
      {hasVolumeB && (
        <>
          {/* B status */}
          <section className="control-section">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <h3 className="section-title" style={{ margin: 0 }}>Subject B — Segmentation</h3>
              {/* Clear button returns to single-brain mode */}
              <button
                className="btn btn--secondary"
                style={{ fontSize: 10, padding: '2px 8px' }}
                onClick={clearVolumeB}
              >
                Clear B
              </button>
            </div>
            <div className="inference-status">
              <span className={`status-dot ${dotClass(inferenceStatusB)}`} />
              <span>{statusLabel(inferenceStatusB)}</span>
            </div>
            {hasOverlayB && (
              <div className="control-row control-row--toggle" style={{ marginTop: 8 }}>
                <label className="control-label" htmlFor="seg-show-b">Show overlay</label>
                <input id="seg-show-b" type="checkbox" checked={showOverlayB}
                  onChange={(e) => setShowOverlayB(e.target.checked)} />
              </div>
            )}
          </section>

          {/* B MPR slices */}
          <section className="control-section">
            <h3 className="section-title">Subject B — MPR Slices</h3>
            <SliderRow label="Axial (K)"    value={controlsB.sliceK} min={0} max={Math.max(0, maxKB - 1)} disabled={!hasVolumeB} onChange={(v) => setControlsB({ sliceK: v })} />
            <SliderRow label="Coronal (J)"  value={controlsB.sliceJ} min={0} max={Math.max(0, maxJB - 1)} disabled={!hasVolumeB} onChange={(v) => setControlsB({ sliceJ: v })} />
            <SliderRow label="Sagittal (I)" value={controlsB.sliceI} min={0} max={Math.max(0, maxIB - 1)} disabled={!hasVolumeB} onChange={(v) => setControlsB({ sliceI: v })} />
          </section>

          {/* B window/level */}
          <section className="control-section">
            <h3 className="section-title">Subject B — Window / Level</h3>
            <SliderRow label="Width"  value={controlsB.windowWidth}  min={1}     max={4000} disabled={!hasVolumeB} onChange={(v) => setControlsB({ windowWidth: v })} />
            <SliderRow label="Centre" value={controlsB.windowCenter} min={-1000} max={3000} disabled={!hasVolumeB} onChange={(v) => setControlsB({ windowCenter: v })} />
          </section>
        </>
      )}

      {/* ── Comparative volumetrics ────────────────────────────────────────── */}
      {(hasOverlay || hasOverlayB) && (
        <section className="control-section">
          <h3 className="section-title">Volumetrics</h3>

          {/* Subject A compute button */}
          {hasOverlay && (
            <>
              <button
                className="btn btn--primary"
                disabled={volumetricsLoading}
                onClick={runVolumetrics}
                style={{ width: '100%', marginBottom: 4 }}
              >
                {volumetricsLoading ? 'Computing A…' : hasVolumeB ? 'Compute A Volumes' : 'Compute Hippocampal Volumes'}
              </button>
              {volumetricsError && (
                <p style={{ color: 'var(--accent-red, #e05252)', fontSize: 10, marginBottom: 4 }}>
                  A: {volumetricsError}
                </p>
              )}
            </>
          )}

          {/* Subject B compute button */}
          {hasOverlayB && (
            <>
              <button
                className="btn btn--primary"
                disabled={volumetricsLoadingB}
                onClick={runVolumetricsB}
                style={{ width: '100%', marginBottom: 8 }}
              >
                {volumetricsLoadingB ? 'Computing B…' : 'Compute B Volumes'}
              </button>
              {volumetricsErrorB && (
                <p style={{ color: 'var(--accent-red, #e05252)', fontSize: 10, marginBottom: 4 }}>
                  B: {volumetricsErrorB}
                </p>
              )}
            </>
          )}

          {/* Chart — visible once at least A has been computed */}
          {volumetrics && (
            <VolumetricsChart
              volumetricsA={volumetrics}
              volumetricsB={volumetricsB}
            />
          )}

          {!volumetrics && !volumetricsLoading && (
            <p style={{ color: '#555', fontSize: 10, textAlign: 'center' }}>
              {hasOverlay ? 'Click above to measure hippocampal volumes.' : ''}
            </p>
          )}
        </section>
      )}

      {/* ── Metadata ────────────────────────────────────────────────────── */}
      {hasVolume && (
        <section className="control-section control-section--meta">
          <h3 className="section-title">Dimensions</h3>
          <dl className="meta-list">
            <dt>A voxels</dt><dd>{maxI} × {maxJ} × {maxK}</dd>
            {hasVolumeB && <><dt>B voxels</dt><dd>{maxIB} × {maxJB} × {maxKB}</dd></>}
          </dl>
        </section>
      )}

      {/* ── Neural Decoding (MVPA) ───────────────────────────────────────── */}
      <DecodingPanel />

      {/* ── Connectome ──────────────────────────────────────────────────── */}
      <ConnectomePanel />

      {/* ── Anomaly Detection (Deep Sweep) ───────────────────────────────── */}
      <AnomalyPanel />

      {/* ── Longitudinal Analysis ────────────────────────────────────────── */}
      <LongitudinalPanel />
    </>
  );
};

export default VolumetricControls;
