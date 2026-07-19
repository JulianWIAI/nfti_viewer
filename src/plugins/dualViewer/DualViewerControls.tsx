/**
 * DualViewerControls.tsx — Sidebar for the dual-viewer inter-subject comparison
 * ───────────────────────────────────────────────────────────────────────────────
 *
 * Reads all state from DualViewerContext and writes back through the context's
 * stable setter callbacks.  Zero VTK calls happen here — all rendering is
 * driven by useEffect hooks in DualVolumetricViewer.tsx.
 *
 * LAYOUT (top → bottom)
 *   0. Alignment mode toggle (Raw / SyN) — visible once Subject B is parsed
 *   1. Subject B file picker + SyN status indicator
 *   2. "Run SyN" button
 *   3. Shared slice sliders (K / J / I) — applied to both viewers
 *   4. Window / Level sliders
 *   5. Volume opacity slider
 *
 * OPACITY SLIDER — RAF DEBOUNCING
 * ─────────────────────────────────
 * All range inputs use the same RAF-debounce pattern as AnomalyPanel.tsx:
 *   • onChange updates local React state immediately (no visual lag on the label).
 *   • The actual context setter (which triggers VTK re-renders) is called inside a
 *     requestAnimationFrame callback so at most one VTK render fires per frame.
 */

import { useRef, useCallback, type FC, type ChangeEvent } from 'react';
import { useDualViewerContext }                             from './DualViewerContext';
import type { SynStatus }                                  from './DualViewerContext';
import AlignmentModeToggle                                  from './AlignmentModeToggle';
import DualAnatomySelector                                  from './DualAnatomySelector';

// ── Status helpers ────────────────────────────────────────────────────────────

function statusLabel(s: SynStatus): string {
  switch (s.phase) {
    case 'idle':      return 'Idle — no registration yet';
    case 'uploading': return 'Registering… (2–5 min)';
    case 'done':      return `Done (${Math.round(s.durationMs / 1000)} s)`;
    case 'error':     return `Error: ${s.message}`;
  }
}

function dotClass(s: SynStatus): string {
  switch (s.phase) {
    case 'done':      return 'status-dot--done';
    case 'error':     return 'status-dot--error';
    case 'uploading': return 'status-dot--running';
    default:          return 'status-dot--idle';
  }
}

// ── Component ─────────────────────────────────────────────────────────────────

const DualViewerControls: FC = () => {
  const {
    subjectALoaded,
    subjectBFile, setSubjectBFile,
    synStatus, warpedLoaded,
    runSyn,
    alignmentMode, setAlignmentMode,
    rawBLoaded, synCached,
    sliceK, maxK, setSliceK,
    sliceJ, maxJ, setSliceJ,
    sliceI, maxI, setSliceI,
    windowCenter, setWindowCenter,
    windowWidth,  setWindowWidth,
    volumeOpacity, setVolumeOpacity,
    hasSegmentation,
    volumetrics,
  } = useDualViewerContext();


  // ── RAF debounce refs — one per slider ──────────────────────────────────
  const rafSliceK  = useRef<number | null>(null);
  const rafSliceJ  = useRef<number | null>(null);
  const rafSliceI  = useRef<number | null>(null);
  const rafCenter  = useRef<number | null>(null);
  const rafWidth   = useRef<number | null>(null);
  const rafOpacity = useRef<number | null>(null);

  const isRunning = synStatus.phase === 'uploading';
  const canRunSyn = subjectALoaded && !!subjectBFile && !isRunning;

  // ── RAF helper: cancel pending frame, schedule new one ──────────────────
  function rafSet<T>(rafRef: React.MutableRefObject<number | null>, setter: (v: T) => void, value: T): void {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      setter(value);
    });
  }

  // ── File picker ──────────────────────────────────────────────────────────
  const handleFileChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setSubjectBFile(e.target.files?.[0] ?? null);
  }, [setSubjectBFile]);

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <aside className="dual-viewer__sidebar">

      {/* ── Alignment mode toggle — visible once Subject B raw actors are ready ── */}
      {rawBLoaded && (
        <section className="control-section dual-viewer-controls__section dual-viewer-controls__alignment">
          <h3 className="section-title">View Mode</h3>
          <AlignmentModeToggle
            mode={alignmentMode}
            onChange={setAlignmentMode}
            disabled={isRunning}
            loading={isRunning}
          />
          {/* Cached hint: instant switch available without another API call */}
          {synCached && alignmentMode === 'raw' && (
            <p className="dual-viewer-controls__hint dual-viewer-controls__hint--cached">
              SyN result cached — switching to Registered is instant.
            </p>
          )}
        </section>
      )}

      {/* ── Subject B selection ─────────────────────────────────────────── */}
      <section className="control-section dual-viewer-controls__section">
        <h3 className="section-title">SyN Registration</h3>

        <p className="dual-viewer-controls__hint">
          Select Subject B (.nii/.nii.gz) to co-register against Subject A
          using dipy Affine + SyN diffeomorphic registration.
        </p>

        {/* File picker */}
        <div className="dual-viewer-controls__file-row">
          <label className="dual-viewer-controls__file-label" htmlFor="dual-subject-b">
            Subject B (moving)
          </label>
          <input
            id="dual-subject-b"
            type="file"
            accept=".nii,.nii.gz"
            className="dual-viewer-controls__file-input"
            disabled={isRunning}
            onChange={handleFileChange}
          />
          {subjectBFile && (
            <span className="dual-viewer-controls__file-name" title={subjectBFile.name}>
              {subjectBFile.name}
            </span>
          )}
        </div>

        {/* SyN status indicator */}
        <div className="inference-status">
          <span className={`status-dot ${dotClass(synStatus)}`} />
          <span className="dual-viewer-controls__status-text">
            {statusLabel(synStatus)}
          </span>
        </div>

        {/* Error detail */}
        {synStatus.phase === 'error' && (
          <p className="dual-viewer-controls__error">{synStatus.message}</p>
        )}

        {/* Run / re-run button */}
        <button
          className="btn btn--primary dual-viewer-controls__run-btn"
          disabled={!canRunSyn}
          onClick={runSyn}
          title={
            !subjectALoaded ? 'Upload Subject A first'
            : !subjectBFile ? 'Select a Subject B file first'
            : undefined
          }
        >
          {isRunning
            ? 'Registering…'
            : warpedLoaded
              ? 'Re-run SyN'
              : 'Run SyN Registration'}
        </button>
      </section>

      {/* ── Slice controls ──────────────────────────────────────────────── */}
      <section className="control-section dual-viewer-controls__section">
        <h3 className="section-title">Slices (shared)</h3>

        {/* Axial (K) */}
        <div className="control-row">
          <label className="control-label">
            <span>Axial K</span>
            <span className="control-value">{sliceK}</span>
          </label>
          <input
            type="range"
            className="control-slider"
            min={0}
            max={Math.max(0, maxK - 1)}
            step={1}
            value={sliceK}
            disabled={!subjectALoaded}
            onChange={(e) => rafSet(rafSliceK, setSliceK, Number(e.target.value))}
          />
        </div>

        {/* Coronal (J) */}
        <div className="control-row">
          <label className="control-label">
            <span>Coronal J</span>
            <span className="control-value">{sliceJ}</span>
          </label>
          <input
            type="range"
            className="control-slider"
            min={0}
            max={Math.max(0, maxJ - 1)}
            step={1}
            value={sliceJ}
            disabled={!subjectALoaded}
            onChange={(e) => rafSet(rafSliceJ, setSliceJ, Number(e.target.value))}
          />
        </div>

        {/* Sagittal (I) */}
        <div className="control-row">
          <label className="control-label">
            <span>Sagittal I</span>
            <span className="control-value">{sliceI}</span>
          </label>
          <input
            type="range"
            className="control-slider"
            min={0}
            max={Math.max(0, maxI - 1)}
            step={1}
            value={sliceI}
            disabled={!subjectALoaded}
            onChange={(e) => rafSet(rafSliceI, setSliceI, Number(e.target.value))}
          />
        </div>
      </section>

      {/* ── Window / Level ──────────────────────────────────────────────── */}
      <section className="control-section dual-viewer-controls__section">
        <h3 className="section-title">Window / Level (shared)</h3>

        {/* Window Centre */}
        <div className="control-row">
          <label className="control-label">
            <span>Centre</span>
            <span className="control-value">{Math.round(windowCenter)}</span>
          </label>
          <input
            type="range"
            className="control-slider"
            min={-1000}
            max={4000}
            step={10}
            value={windowCenter}
            disabled={!subjectALoaded}
            onChange={(e) => rafSet(rafCenter, setWindowCenter, Number(e.target.value))}
          />
        </div>

        {/* Window Width */}
        <div className="control-row">
          <label className="control-label">
            <span>Width</span>
            <span className="control-value">{Math.round(windowWidth)}</span>
          </label>
          <input
            type="range"
            className="control-slider"
            min={1}
            max={5000}
            step={10}
            value={windowWidth}
            disabled={!subjectALoaded}
            onChange={(e) => rafSet(rafWidth, setWindowWidth, Number(e.target.value))}
          />
        </div>
      </section>

      {/* ── Volume opacity ───────────────────────────────────────────────── */}
      <section className="control-section dual-viewer-controls__section">
        <h3 className="section-title">3-D Volume Opacity (shared)</h3>

        <div className="control-row dual-viewer-controls__opacity-row">
          <label className="control-label">
            <span>Opacity</span>
            <span className="control-value">{Math.round(volumeOpacity * 100)}%</span>
          </label>
          <input
            type="range"
            className="control-slider dual-viewer-controls__opacity-slider"
            min={0}
            max={100}
            step={1}
            value={Math.round(volumeOpacity * 100)}
            disabled={!subjectALoaded}
            onChange={(e) => rafSet(rafOpacity, setVolumeOpacity, Number(e.target.value) / 100)}
          />
        </div>
      </section>

      {/* ── Per-structure anatomy selector — shown when segmentation overlay is ready ── */}
      {/* DualAnatomySelector provides the same GM/WM/CSF group toggles AND          */}
      {/* per-structure checkboxes as the single-brain AnatomySelector, but controls  */}
      {/* both viewers simultaneously via dualSegBundle.updateLabelVisibility().      */}
      {hasSegmentation && <DualAnatomySelector />}

      {/* ── Comparative hippocampal volumetrics — shown when available ──── */}
      {volumetrics && (
        <section className="control-section dual-viewer-controls__section dual-volumetrics">
          <h3 className="section-title">Hippocampal Volumes</h3>
          <p className="dual-viewer-controls__hint">
            Computed from each subject's un-warped NIfTI using voxel geometry.
          </p>
          <table className="dual-volumetrics__table">
            <thead>
              <tr>
                <th className="dual-volumetrics__th" />
                <th className="dual-volumetrics__th">Subject A</th>
                <th className="dual-volumetrics__th">Subject B</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="dual-volumetrics__label">LH (cm³)</td>
                <td className="dual-volumetrics__value">{volumetrics.subjectA.lh.toFixed(3)}</td>
                <td className="dual-volumetrics__value">{volumetrics.subjectB.lh.toFixed(3)}</td>
              </tr>
              <tr>
                <td className="dual-volumetrics__label">RH (cm³)</td>
                <td className="dual-volumetrics__value">{volumetrics.subjectA.rh.toFixed(3)}</td>
                <td className="dual-volumetrics__value">{volumetrics.subjectB.rh.toFixed(3)}</td>
              </tr>
              <tr>
                <td className="dual-volumetrics__label">Total (cm³)</td>
                <td className="dual-volumetrics__value">
                  {(volumetrics.subjectA.lh + volumetrics.subjectA.rh).toFixed(3)}
                </td>
                <td className="dual-volumetrics__value">
                  {(volumetrics.subjectB.lh + volumetrics.subjectB.rh).toFixed(3)}
                </td>
              </tr>
            </tbody>
          </table>
        </section>
      )}

    </aside>
  );
};

export default DualViewerControls;
