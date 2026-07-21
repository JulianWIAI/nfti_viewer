/**
 * MultimodalWorkspace.tsx — Split-screen fMRI + MEG multimodal viewer
 * ─────────────────────────────────────────────────────────────────────
 *
 * Renders a 50/50 horizontal split with:
 *   LEFT  — FmriPanel (4-D BOLD volume renderer, synced to MEG time)
 *   RIGHT — MegPanel  (waveform canvas; click sets shared time cursor)
 *
 * The SyncProvider sits at the top of this subtree so both panels
 * can read and write the shared currentTimeSec without prop drilling.
 *
 * A sidebar strip (right edge) hosts:
 *   • TimeSlider for scrubbing through fMRI volumes
 *   • fMRI window/level controls
 *   • Source estimate trigger and overlay toggle
 *
 * File loading:
 *   This component does NOT own file loading — the PluginData discriminated
 *   union is received from the host (Viewer.tsx / App.tsx) via props.
 *   The host's BidsRouter has already routed .nii/.nii.gz → FmriPayload
 *   and .fif → MegSessionPayload.
 *
 *   In the multimodal workspace BOTH payloads may be present simultaneously.
 *   The user loads them sequentially via drag-and-drop (any order).
 *
 * Layout:
 *   ┌──────────────────────────┬──────────────────────────┬────────┐
 *   │   FmriPanel              │   MegPanel               │Sidebar │
 *   │  (vtk.js BOLD volume)    │  (waveform canvas)       │controls│
 *   │  currentVolumeIdx        │  click → setTimeSec      │        │
 *   └──────────────────────────┴──────────────────────────┴────────┘
 *   [  TimeSlider  (bottom bar, spans both panels)                  ]
 */

import { type FC, useState, useCallback, type DragEvent } from 'react';
import { SyncProvider, useSyncContext } from '../../contexts/SyncContext';
import FmriPanel from './FmriPanel';
import MegPanel  from './MegPanel';
import TimeSlider from '../../components/TimeSlider';
import EventTimelineRibbon from '../../components/EventTimelineRibbon';
import { requestSourceEstimate } from '../../services/megSourceApi';
import { bidsEventsApi } from '../../services/bidsEventsApi';
import type { FmriPayload } from '../../types/fmri.types';
import type { MegSessionPayload } from '../../types/meg.types';

// ── Inner workspace (inside SyncProvider so it can use useSyncContext) ────────

interface WorkspaceInnerProps {
  fmriPayload: FmriPayload | null;
  megPayload:  MegSessionPayload | null;
}

const WorkspaceInner: FC<WorkspaceInnerProps> = ({ fmriPayload, megPayload }) => {
  const { currentTimeSec, tr, currentVolumeIdx, setEvents } = useSyncContext();

  // Source-estimate computation state
  const [srcLoading,  setSrcLoading]  = useState(false);
  const [srcError,    setSrcError]    = useState<string | null>(null);

  // BIDS events.tsv drag-drop state
  const [tsvLoading,        setTsvLoading]        = useState(false);
  const [tsvError,          setTsvError]          = useState<string | null>(null);
  const [tsvFilename,       setTsvFilename]        = useState<string | null>(null);
  const [eventsTotalDur,    setEventsTotalDur]    = useState(0);
  const [isDraggingOver,    setIsDraggingOver]    = useState(false);

  // ── TSV drag-and-drop handlers ────────────────────────────────────────────
  const handleTsvDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingOver(true);
  }, []);

  const handleTsvDragLeave = useCallback(() => {
    setIsDraggingOver(false);
  }, []);

  const handleTsvDrop = useCallback(async (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingOver(false);

    const file = e.dataTransfer.files[0];
    if (!file || !file.name.toLowerCase().endsWith('.tsv')) {
      setTsvError('Please drop a BIDS *_events.tsv file.');
      return;
    }

    setTsvLoading(true);
    setTsvError(null);
    try {
      const result = await bidsEventsApi.uploadEventsFile(file);
      setEvents(result.events);
      setEventsTotalDur(result.totalDuration);
      setTsvFilename(file.name);
    } catch (err) {
      setTsvError(err instanceof Error ? err.message : String(err));
    } finally {
      setTsvLoading(false);
    }
  }, [setEvents]);

  // Access the fMRI panel context through a ref callback isn't possible directly,
  // so we manage overlay state here and pass it down via FmriPanel props.
  // FmriPanel owns the VTK actor; we only need to trigger the API call here.
  const [srcResult, setSrcResult] = useState<import('../../types/fmri.types').SourceEstimateResult | null>(null);
  const [showOverlay, setShowOverlay] = useState(false);

  // ── Trigger source estimate from the sidebar ──────────────────────────────
  const runSourceEstimate = useCallback(async () => {
    if (!megPayload) return;
    setSrcLoading(true);
    setSrcError(null);
    try {
      const result = await requestSourceEstimate({
        sessionId: megPayload.sessionId,
        method:    'dSPM',
        tMin:      currentTimeSec,
        tMax:      currentTimeSec + (tr > 0 ? tr : 1),
      });
      setSrcResult(result);
      setShowOverlay(true);
    } catch (err) {
      setSrcError(err instanceof Error ? err.message : String(err));
    } finally {
      setSrcLoading(false);
    }
  }, [megPayload, currentTimeSec, tr]);

  // ── fMRI panel controls sidebar ───────────────────────────────────────────
  const sidebar = (
    <aside className="multimodal-sidebar">
      <h3 className="section-title">fMRI Controls</h3>

      {/* Volume info */}
      {fmriPayload && (
        <div style={{ fontSize: 10, color: '#888', marginBottom: 8 }}>
          {fmriPayload.nTimepoints} volumes · TR {fmriPayload.tr.toFixed(2)} s
        </div>
      )}

      {/* Sync status */}
      <div style={{ fontSize: 10, color: '#4fc3f7', marginBottom: 12 }}>
        {tr > 0
          ? `Vol ${currentVolumeIdx + 1} · t = ${currentTimeSec.toFixed(1)} s`
          : 'Click MEG waveform to sync'}
      </div>

      {/* BIDS events.tsv loader */}
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
      </section>

      {/* Source estimate section */}
      {megPayload && (
        <section className="control-section">
          <h3 className="section-title">MEG Source Estimate</h3>
          <p style={{ fontSize: 10, color: '#666', marginBottom: 6 }}>
            Runs dSPM on the current MEG time window and overlays the
            source-space activation on the fMRI volume.
          </p>
          <button
            className="btn btn--primary"
            style={{ width: '100%', marginBottom: 6 }}
            disabled={srcLoading || !fmriPayload}
            onClick={runSourceEstimate}
          >
            {srcLoading ? 'Computing…' : 'Run Source Estimate'}
          </button>
          {srcError && (
            <p style={{ color: 'var(--accent-red, #e05252)', fontSize: 10 }}>
              {srcError}
            </p>
          )}
          {srcResult && (
            <div className="control-row control-row--toggle" style={{ marginTop: 4 }}>
              <label className="control-label" htmlFor="src-overlay-toggle">
                Show overlay
              </label>
              <input
                id="src-overlay-toggle"
                type="checkbox"
                checked={showOverlay}
                onChange={(e) => setShowOverlay(e.target.checked)}
              />
            </div>
          )}
          {srcResult && (
            <p style={{ fontSize: 10, color: '#666', marginTop: 4 }}>
              {srcResult.vertices.length} vertices · peak {srcResult.peakAmplitude.toFixed(2)} ·
              {' '}{srcResult.method} · {srcResult.durationMs.toFixed(0)} ms
            </p>
          )}
        </section>
      )}
    </aside>
  );

  return (
    <div className="multimodal-workspace">
      {/* Split panels */}
      <div className="multimodal-workspace__panels">
        {/* fMRI pane */}
        <FmriPanel
          payload={fmriPayload}
          controlsSlot={sidebar}
        />

        {/* MEG pane */}
        <MegPanel payload={megPayload} />
      </div>

      {/* BIDS event timeline ribbon — shown when events are loaded */}
      <EventTimelineRibbon totalDuration={eventsTotalDur} />

      {/* TimeSlider spanning the bottom */}
      <div className="multimodal-workspace__timebar">
        <TimeSlider
          totalDuration={fmriPayload ? fmriPayload.nTimepoints * fmriPayload.tr : 0}
          tr={fmriPayload?.tr ?? 0}
          nTimepoints={fmriPayload?.nTimepoints ?? 0}
        />
      </div>
    </div>
  );
};

// ── Exported outer component (provides SyncContext) ───────────────────────────

interface MultimodalWorkspaceProps {
  fmriPayload: FmriPayload | null;
  megPayload:  MegSessionPayload | null;
}

/**
 * Top-level multimodal workspace.
 * Wraps WorkspaceInner in SyncProvider for shared time state.
 * ReferencePanelProvider now lives at the App root — no local wrapper needed.
 */
const MultimodalWorkspace: FC<MultimodalWorkspaceProps> = (props) => (
  <SyncProvider>
    <WorkspaceInner {...props} />
  </SyncProvider>
);

export default MultimodalWorkspace;
