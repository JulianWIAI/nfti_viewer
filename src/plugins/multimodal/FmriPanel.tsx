/**
 * FmriPanel.tsx — 4-D fMRI viewer panel for the multimodal workspace
 * ────────────────────────────────────────────────────────────────────
 *
 * Renders a single vtk.js volume renderer pane that:
 *   1. Builds the fMRI volume pipeline from FmriPayload.
 *   2. Listens to currentVolumeIdx from SyncContext and updates the
 *      displayed BOLD volume whenever the MEG panel advances time.
 *   3. Optionally renders a BoldOverlay point cloud when source estimate
 *      results are available.
 *   4. Exposes slice / window-level controls in the sidebar slot via
 *      the controls prop.
 *
 * VTK lifecycle:
 *   • initVtk() on mount → renderWindow + renderer
 *   • buildFmriVolumeActor() when payload arrives
 *   • setTimepoint() on every currentVolumeIdx change
 *   • destroyVtk() on unmount
 *
 * The component intentionally does NOT own playback state — that lives in
 * SyncContext so MegPanel can drive it.
 */

import {
  type FC,
  useEffect,
  useRef,
  useCallback,
  createContext,
  useContext,
  useState,
} from 'react';
import { initVtk, resizeVtk, destroyVtk } from '../../lib/vtk/vtkSetup';
import { buildFmriVolumeActor, type FmriActorBundle } from '../../lib/vtk/fmriVolumeRenderer';
import { buildBoldOverlay, type BoldOverlayBundle } from '../../lib/vtk/boldOverlay';
import { useSyncContext } from '../../contexts/SyncContext';
import type { FmriPayload, SourceEstimateResult } from '../../types/fmri.types';
import type { VtkContext } from '../../lib/vtk/vtkSetup';

// ── Context ───────────────────────────────────────────────────────────────────

export interface FmriPanelContextValue {
  /** True once the fMRI volume actor has been created. */
  hasFmri: boolean;
  /** The parsed fMRI payload (null until a file is loaded). */
  payload: FmriPayload | null;
  /** Update window/level from the controls sidebar. */
  updateWindowLevel: (center: number, width: number) => void;
  /** Update 3-D volume opacity from the controls sidebar. */
  updateOpacity: (factor: number) => void;
  /** Current window width (for slider initialisation). */
  windowWidth: number;
  /** Current window center (for slider initialisation). */
  windowCenter: number;
  /** Show/hide the source-estimate overlay. */
  showOverlay: boolean;
  setShowOverlay: (v: boolean) => void;
  /** Source estimate result (null until computed). */
  sourceEstimate: SourceEstimateResult | null;
  /** Store a new source estimate result (called from sidebar). */
  setSourceEstimate: (result: SourceEstimateResult) => void;
}

const FmriPanelContext = createContext<FmriPanelContextValue | null>(null);

export function useFmriPanelContext(): FmriPanelContextValue {
  const ctx = useContext(FmriPanelContext);
  if (!ctx) throw new Error('useFmriPanelContext must be inside FmriPanel');
  return ctx;
}

// ── VTK refs container ────────────────────────────────────────────────────────

interface VtkFmriRefs {
  vtk:      VtkContext | null;
  bundle:   FmriActorBundle | null;
  overlay:  BoldOverlayBundle | null;
}

// ── Component ─────────────────────────────────────────────────────────────────

interface FmriPanelProps {
  /** Parsed 4-D NIfTI payload from the file-load pipeline. */
  payload: FmriPayload | null;
  /** Sidebar controls slot — rendered as a sibling to the canvas. */
  controlsSlot?: React.ReactNode;
}

const FmriPanel: FC<FmriPanelProps> = ({ payload, controlsSlot }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const vtkRef       = useRef<VtkFmriRefs>({ vtk: null, bundle: null, overlay: null });

  // Pull the synced volume index from the shared multimodal context
  const { currentVolumeIdx, setTr } = useSyncContext();

  // Local sidebar state
  const [windowWidth,  setWindowWidth]  = useState(2000);
  const [windowCenter, setWindowCenter] = useState(500);
  const [showOverlay,  setShowOverlay]  = useState(false);
  const [sourceEstimate, setSourceEstimate] = useState<SourceEstimateResult | null>(null);

  // ── VTK init on mount ─────────────────────────────────────────────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const vtk = initVtk(el);
    vtkRef.current.vtk = vtk;
    return () => {
      destroyVtk(vtk);
      vtkRef.current = { vtk: null, bundle: null, overlay: null };
    };
  }, []);

  // ── Resize handler ────────────────────────────────────────────────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(() => {
      if (vtkRef.current.vtk) resizeVtk(vtkRef.current.vtk, el);
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // ── Build / replace the fMRI actor when payload changes ──────────────────
  useEffect(() => {
    const { vtk } = vtkRef.current;
    if (!vtk || !payload) return;

    // Remove old actor if present
    if (vtkRef.current.bundle) {
      vtk.volumeRenderer.removeVolume(vtkRef.current.bundle.actor);
    }

    const bundle = buildFmriVolumeActor(payload);
    vtk.volumeRenderer.addVolume(bundle.actor);
    vtk.volumeRenderer.resetCamera();
    vtkRef.current.bundle = bundle;

    // Initialise window/level from the data range
    const [lo, hi] = bundle.dataRange;
    const center = (lo + hi) / 2;
    const width  = hi - lo;
    setWindowWidth(Math.round(width));
    setWindowCenter(Math.round(center));

    // Register the TR with SyncContext so MegPanel can compute volume indices
    setTr(payload.tr);

    vtk.renderWindow.render();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [payload]);

  // ── Advance to the synced timepoint ──────────────────────────────────────
  // Runs whenever MegPanel updates currentTimeSec in SyncContext
  useEffect(() => {
    const { vtk, bundle } = vtkRef.current;
    if (!vtk || !bundle) return;
    bundle.setTimepoint(currentVolumeIdx);
    vtk.renderWindow.render();
  }, [currentVolumeIdx]);

  // ── Show / hide the source-estimate overlay ───────────────────────────────
  useEffect(() => {
    const { vtk, overlay } = vtkRef.current;
    if (!vtk || !overlay) return;
    overlay.actor.setVisibility(showOverlay);
    vtk.renderWindow.render();
  }, [showOverlay]);

  // ── Build / rebuild overlay when a new source estimate arrives ───────────
  useEffect(() => {
    const { vtk } = vtkRef.current;
    if (!vtk || !sourceEstimate) return;

    // If an overlay already exists update its amplitudes in-place
    if (vtkRef.current.overlay) {
      vtkRef.current.overlay.updateAmplitudes(sourceEstimate.vertices);
    } else {
      const newOverlay = buildBoldOverlay(sourceEstimate.vertices);
      vtk.volumeRenderer.addActor(newOverlay.actor);
      vtkRef.current.overlay = newOverlay;
    }

    vtkRef.current.overlay?.actor.setVisibility(showOverlay);
    vtk.renderWindow.render();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceEstimate]);

  // ── Stable callbacks for sidebar controls ─────────────────────────────────
  const updateWindowLevel = useCallback((center: number, width: number) => {
    setWindowCenter(center);
    setWindowWidth(width);
    const { vtk, bundle } = vtkRef.current;
    if (!vtk || !bundle) return;
    bundle.updateWindowLevel(center, width);
    vtk.renderWindow.render();
  }, []);

  const updateOpacity = useCallback((factor: number) => {
    const { vtk, bundle } = vtkRef.current;
    if (!vtk || !bundle) return;
    bundle.updateOpacity(factor);
    vtk.renderWindow.render();
  }, []);

  // ── Context value ─────────────────────────────────────────────────────────
  const ctxValue: FmriPanelContextValue = {
    hasFmri:  !!payload,
    payload,
    updateWindowLevel,
    updateOpacity,
    windowWidth,
    windowCenter,
    showOverlay,
    setShowOverlay,
    sourceEstimate,
    setSourceEstimate,
  };

  return (
    <FmriPanelContext.Provider value={ctxValue}>
      <div className="multimodal-panel multimodal-panel--fmri">
        {/* vtk.js render target — fills the panel */}
        <div
          ref={containerRef}
          className="multimodal-panel__canvas"
          style={{ width: '100%', height: '100%' }}
        />
        {/* Hint when no file is loaded */}
        {!payload && (
          <div className="multimodal-panel__empty">
            <p>Drop a 4-D NIfTI (.nii / .nii.gz) here to load fMRI data</p>
          </div>
        )}
        {/* Sidebar controls slot rendered outside the canvas */}
        {controlsSlot}
      </div>
    </FmriPanelContext.Provider>
  );
};

export default FmriPanel;
