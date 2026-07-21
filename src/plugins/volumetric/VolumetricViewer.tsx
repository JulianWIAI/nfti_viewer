/**
 * VolumetricViewer.tsx — vtk.js plugin viewer component
 * ────────────────────────────────────────────────────────
 *
 * Thin adapter between the NeuroimagingPlugin contract (PluginViewerProps) and
 * the vtk.js rendering pipeline in ../../lib/vtk/.
 *
 * Phase 9 — Dynamic split-screen:
 *   When only Subject A is loaded the component renders a single full-width
 *   4-panel VTK canvas (standard MPR + 3D layout).  When the user drops a
 *   Subject B NIfTI via SubjectBDropZone a second, independent VTK context
 *   is initialised in the right half of the workspace.  No SyN registration
 *   is performed — each brain lives in its native voxel space.
 *
 *   On B load, /api/segment is called concurrently for both subjects
 *   (Promise.all) and the resulting overlays are applied independently to
 *   each pane.  Tissue-class visibility toggles are independent per pane.
 *
 * AI Segmentation flow:
 *   1. runSegmentation() POSTs the original NIfTI File to /api/segment.
 *   2. The backend runs SynthSeg and returns a base64 uint8 label map.
 *   3. buildSegmentationOverlay() creates semi-transparent RGBA vtkImageSlice
 *      actors on top of the MPR greyscale slices.
 *   4. Overlay slice positions are kept in sync with the MPR controls.
 */

import {
  useEffect,
  useRef,
  useState,
  useMemo,
  useCallback,
  createContext,
  useContext,
  type FC,
  type RefObject,
} from 'react';

import { initVtk, resizeVtk, destroyVtk } from '../../lib/vtk/vtkSetup';
import { buildVolumeActor, type VolumeActorBundle } from '../../lib/vtk/volumeRenderer';
import { buildMprActors, setupMprCameras, type MprBundle } from '../../lib/vtk/mprRenderer';
import { attachLod, type LodHandle } from '../../lib/vtk/lodManager';
import {
  buildSegmentationOverlay,
  type SegmentationBundle,
} from '../../lib/vtk/segmentationOverlay';
import {
  buildAnomalyOverlay,
  type AnomalyBundle,
} from '../../lib/vtk/anomalyOverlay';
import { anomalyApi } from '../../services/anomalyApi';
import {
  buildLongitudinalOverlay,
  type LongitudinalBundle,
} from '../../lib/vtk/longitudinalOverlay';
import { longitudinalApi } from '../../services/longitudinalApi';
import type { VtkContext } from '../../lib/vtk/vtkSetup';
import { segmentApi } from '../../services/segmentApi';
import { dualSegmentApi } from '../../services/dualSegmentApi';
import { useNiftiWorker } from '../../hooks/useNiftiWorker';
import type { TissueClass } from '../../lib/vtk/tissueGroups';
import {
  type MacroState,
  BRAIN_LABELS,
  LABELS_BY_GROUP,
  defaultLabelVisibility,
  getMacroGroupState,
} from '../../lib/vtk/labelVisibility';
import type { VolumetricsResult } from '../../types/analysis.types';
import { mriApi } from '../../services/mriApi';

import type { PluginViewerProps } from '../../types/plugin.types';
import type { ViewerControls } from '../../types/nifti.types';
import type { ConnectomeApiResponse } from '../../lib/vtk/connectome/connectomeTypes';

import { useNeuralDecoding } from './useNeuralDecoding';
import type { NeuralDecodingStatus } from './useNeuralDecoding';
import type { DecodingRequest, DecodingResult } from '../../services/decodingApi';

import {
  ComparisonContext,
  type ComparisonContextValue,
} from './RawComparisonContext';

import { useReferencePanel } from '../../contexts/ReferencePanelContext';
import { useSegmentationPicker } from '../../hooks/useSegmentationPicker';

// ── InferenceStatus (reused by VolumetricControls for the progress UI) ────────

export type InferenceStatus =
  | { phase: 'idle' }
  | { phase: 'uploading' }
  | { phase: 'running' }
  | { phase: 'done'; durationMs: number }
  | { phase: 'error'; message: string };

/** Progress phases for the anomaly detection pipeline. */
export type AnomalyStatus =
  | { phase: 'idle' }
  | { phase: 'uploading' }
  | { phase: 'running' }
  | { phase: 'done'; durationMs: number; nAnomaly: number }
  | { phase: 'error'; message: string };

/** Progress phases for the longitudinal delta pipeline. */
export type LongitudinalStatus =
  | { phase: 'idle' }
  | { phase: 'uploading' }
  | { phase: 'running' }
  | { phase: 'done'; durationMs: number; nPositive: number; nNegative: number }
  | { phase: 'error'; message: string };

// ── Shared context (consumed by VolumetricControls) ───────────────────────────

interface VolumetricContextValue {
  controls:              ViewerControls;
  setControls:           (partial: Partial<ViewerControls>) => void;
  maxI: number; maxJ: number; maxK: number;
  inferenceStatus:       InferenceStatus;
  hasVolume:             boolean;
  hasOverlay:            boolean;
  showOverlay:           boolean;
  setShowOverlay:        (v: boolean) => void;
  runSegmentation:       () => void;
  /** Per-label (FreeSurfer ID → visible) visibility record. */
  labelVisibility:  Record<number, boolean>;
  /** Derived tristate (all/partial/none) for each tissue macro-group. */
  macroGroupState:  Record<TissueClass, MacroState>;
  /** Toggle a single brain structure on or off by FreeSurfer label ID. */
  setLabelVisible:  (labelId: number, visible: boolean) => void;
  /** Set all structures belonging to a tissue class to visible/hidden. */
  setGroupVisible:  (group: TissueClass, visible: boolean) => void;
  /** Make all 32 non-background structures visible. */
  showAllLabels:    () => void;
  /** Hide all 32 non-background structures. */
  hideAllLabels:    () => void;
  /** Hippocampal volumetrics result (null = not yet computed). */
  volumetrics:           VolumetricsResult | null;
  volumetricsLoading:    boolean;
  volumetricsError:      string | null;
  runVolumetrics:        () => void;

  // ── Neural decoding (MVPA) ──────────────────────────────────────────────
  decodingData:         DecodingResult | null;
  decodingStatus:       NeuralDecodingStatus;
  runNeuralDecoding:    (req: DecodingRequest) => Promise<void>;
  currentTimeIndex:     number;
  setCurrentTimeIndex:  (idx: number) => void;

  // ── Connectome ──────────────────────────────────────────────────────────────
  connectomeData:      ConnectomeApiResponse | null;
  setConnectomeData:   (data: ConnectomeApiResponse | null) => void;
  getVtkCtx: () => import('../../lib/vtk/vtkSetup').VtkContext | null;

  // ── Cross-filter: heatmap cell → VTK edge highlight ─────────────────────────
  selectedEdge: { source: number; target: number } | null;
  setSelectedEdge: (edge: { source: number; target: number } | null) => void;

  // ── Anomaly detection overlay ────────────────────────────────────────────
  anomalyStatus:         AnomalyStatus;
  hasAnomalyOverlay:     boolean;
  showAnomalyOverlay:    boolean;
  setShowAnomalyOverlay: (v: boolean) => void;
  anomalyOpacity:        number;
  setAnomalyOpacity:     (factor: number) => void;
  nAnomalyVoxels:        number;
  runAnomalyDetection:   () => void;
  clearAnomalyOverlay:   () => void;

  // ── Longitudinal delta overlay ───────────────────────────────────────────
  longitudinalStatus:           LongitudinalStatus;
  hasLongitudinalOverlay:       boolean;
  showLongitudinalOverlay:      boolean;
  setShowLongitudinalOverlay:   (v: boolean) => void;
  longitudinalOpacity:          number;
  setLongitudinalOpacity:       (factor: number) => void;
  nPositive:                    number;
  nNegative:                    number;
  runLongitudinalDelta:         (baseline: File, followup: File, transformType: 'rigid' | 'affine') => void;
  clearLongitudinalOverlay:     () => void;
}

export const VolumetricContext = createContext<VolumetricContextValue | null>(null);

export function useVolumetricContext(): VolumetricContextValue {
  const ctx = useContext(VolumetricContext);
  if (!ctx) throw new Error('useVolumetricContext must be used inside VolumetricViewer');
  return ctx;
}

// ── Default controls ──────────────────────────────────────────────────────────

const DEFAULT_CONTROLS: ViewerControls = {
  sliceK: 0, sliceJ: 0, sliceI: 0,
  windowWidth: 1000, windowCenter: 500,
  volumeOpacity: 0.8, showVolume: true,
};

// ── Mutable vtk refs ──────────────────────────────────────────────────────────

interface VtkRefs {
  ctx: VtkContext | null;
  volumeBundle: VolumeActorBundle | null;
  mprBundle: MprBundle | null;
  lodHandle: LodHandle | null;
  segBundle: SegmentationBundle | null;
  anomalyBundle: AnomalyBundle | null;
  longitudinalBundle: LongitudinalBundle | null;
}

/** Mutable refs for Subject B's VTK pane (no anomaly/longitudinal overlays). */
interface VtkRefsB {
  ctx: VtkContext | null;
  volumeBundle: VolumeActorBundle | null;
  mprBundle: MprBundle | null;
  lodHandle: LodHandle | null;
  segBundle: SegmentationBundle | null;
  ro: ResizeObserver | null;  // stored for cleanup on clear/unmount
}

const EMPTY_REFS_B: VtkRefsB = {
  ctx: null, volumeBundle: null, mprBundle: null, lodHandle: null, segBundle: null, ro: null,
};

// ── Component ─────────────────────────────────────────────────────────────────

const VolumetricViewer: FC<PluginViewerProps> = ({ data, controlsSlot }) => {
  // ── Subject A refs & state ──────────────────────────────────────────────────
  const containerRef = useRef<HTMLDivElement>(null);
  const vtkRef       = useRef<VtkRefs>({
    ctx: null, volumeBundle: null, mprBundle: null, lodHandle: null,
    segBundle: null, anomalyBundle: null, longitudinalBundle: null,
  });

  const volume = data?.kind === 'volumetric' ? data.payload : null;

  const [controls,        setControlsState]   = useState<ViewerControls>(DEFAULT_CONTROLS);
  const [maxI,            setMaxI]            = useState(1);
  const [maxJ,            setMaxJ]            = useState(1);
  const [maxK,            setMaxK]            = useState(1);
  const [inferenceStatus, setInferenceStatus] = useState<InferenceStatus>({ phase: 'idle' });
  const [hasOverlay,      setHasOverlay]      = useState(false);
  const [showOverlay,     setShowOverlayState]= useState(true);
  const [vtkError,        setVtkError]        = useState<string | null>(null);

  const [labelVisibility,     setLabelVisibility]    = useState<Record<number, boolean>>(defaultLabelVisibility);
  const labelVisibilityRef  = useRef<Record<number, boolean>>(defaultLabelVisibility());
  const [volumetrics,         setVolumetrics]        = useState<VolumetricsResult | null>(null);
  const [volumetricsLoading,  setVolumetricsLoading] = useState(false);
  const [volumetricsError,    setVolumetricsError]   = useState<string | null>(null);

  // ── Subject B refs & state ──────────────────────────────────────────────────
  const containerBRef  = useRef<HTMLDivElement>(null);
  const vtkRefB        = useRef<VtkRefsB>({ ...EMPTY_REFS_B });
  const fileBRef       = useRef<File | null>(null);   // original File kept for API calls

  const workerB = useNiftiWorker();  // independent worker for parsing Subject B

  const [hasVolumeB,          setHasVolumeB]         = useState(false);
  const [controlsB,           setControlsBState]     = useState<ViewerControls>(DEFAULT_CONTROLS);
  const [maxIB,               setMaxIB]              = useState(1);
  const [maxJB,               setMaxJB]              = useState(1);
  const [maxKB,               setMaxKB]              = useState(1);
  const [inferenceStatusB,    setInferenceStatusB]   = useState<InferenceStatus>({ phase: 'idle' });
  const [hasOverlayB,         setHasOverlayB]        = useState(false);
  const [showOverlayB,        setShowOverlayBState]  = useState(true);
  const [labelVisibilityB,    setLabelVisibilityB]   = useState<Record<number, boolean>>(defaultLabelVisibility);
  const labelVisibilityBRef = useRef<Record<number, boolean>>(defaultLabelVisibility());
  const [volumetricsB,        setVolumetricsB]       = useState<VolumetricsResult | null>(null);
  const [volumetricsLoadingB, setVolumetricsLoadingB]= useState(false);
  const [volumetricsErrorB,   setVolumetricsErrorB]  = useState<string | null>(null);

  // ── Neural decoding (MVPA) ──────────────────────────────────────────────────
  const { decodingData, decodingStatus, runNeuralDecoding } = useNeuralDecoding();
  const [currentTimeIndex, setCurrentTimeIndex] = useState(0);

  // ── Connectome ──────────────────────────────────────────────────────────────
  const [connectomeData, setConnectomeDataState] = useState<ConnectomeApiResponse | null>(null);
  const setConnectomeData = useCallback((data: ConnectomeApiResponse | null) => {
    setConnectomeDataState(data);
  }, []);

  const [selectedEdge, setSelectedEdgeState] = useState<{ source: number; target: number } | null>(null);
  const setSelectedEdge = useCallback((edge: { source: number; target: number } | null) => {
    setSelectedEdgeState(edge);
  }, []);

  // ── Anomaly detection state ─────────────────────────────────────────────────
  const [anomalyStatus,      setAnomalyStatus]      = useState<AnomalyStatus>({ phase: 'idle' });
  const [hasAnomalyOverlay,  setHasAnomalyOverlay]  = useState(false);
  const [showAnomalyOverlay, setShowAnomalyOverlayState] = useState(true);
  const [anomalyOpacity,     setAnomalyOpacityState] = useState(0.8);
  const [nAnomalyVoxels,     setNAnomalyVoxels]     = useState(0);

  // ── Longitudinal delta state ────────────────────────────────────────────────
  const [longitudinalStatus,      setLongitudinalStatus]      = useState<LongitudinalStatus>({ phase: 'idle' });
  const [hasLongitudinalOverlay,  setHasLongitudinalOverlay]  = useState(false);
  const [showLongitudinalOverlay, setShowLongitudinalOverlayState] = useState(true);
  const [longitudinalOpacity,     setLongitudinalOpacityState] = useState(0.8);
  const [nPositive,               setNPositive]               = useState(0);
  const [nNegative,               setNNegative]               = useState(0);

  const getVtkCtx = useCallback(() => vtkRef.current.ctx, []);

  // ── Segmentation label picking → Reference Panel ───────────────────────────
  const { navigateToRegion } = useReferencePanel();

  // Stable callbacks that read mutable refs — never cause hook re-runs on render
  const getPickerRefs = useCallback(() => ({
    ctx:          vtkRef.current.ctx,
    segBundle:    vtkRef.current.segBundle,
    volumeBundle: vtkRef.current.volumeBundle,
  }), []);

  const getVolumeDims = useCallback((): [number, number, number] | null => {
    if (!volume) return null;
    const { dims } = volume.header;
    const nx = dims[1]; const ny = dims[2]; const nz = dims[3];
    if (!nx || !ny || !nz) return null;
    return [nx, ny, nz];
  }, [volume]);

  // Clicking an MPR quadrant while a segmentation overlay is visible opens
  // the Reference Drawer and navigates to the anatomical entry for that label.
  useSegmentationPicker({
    containerRef,
    getPickerRefs,
    getVolumeDims,
    onLabelPick: navigateToRegion,
    enabled:     hasOverlay,
  });

  useEffect(() => {
    setCurrentTimeIndex(0);
  }, [decodingData]);

  // ── Subject A stable control updaters ─────────────────────────────────────
  const setControls = useCallback((partial: Partial<ViewerControls>) => {
    setControlsState((prev) => ({ ...prev, ...partial }));
  }, []);

  const setShowOverlay = useCallback((v: boolean) => {
    setShowOverlayState(v);
    const { segBundle, ctx } = vtkRef.current;
    if (!segBundle || !ctx) return;
    segBundle.axial.setVisibility(v);
    segBundle.coronal.setVisibility(v);
    segBundle.sagittal.setVisibility(v);
    ctx.renderWindow.render();
  }, []);

  const setLabelVisible = useCallback((labelId: number, visible: boolean) => {
    setLabelVisibility((prev) => ({ ...prev, [labelId]: visible }));
  }, []);

  const setGroupVisible = useCallback((group: TissueClass, visible: boolean) => {
    setLabelVisibility((prev) => {
      const updated = { ...prev };
      for (const lbl of LABELS_BY_GROUP[group]) updated[lbl.id] = visible;
      return updated;
    });
  }, []);

  const showAllLabels = useCallback(() => {
    setLabelVisibility(defaultLabelVisibility());
  }, []);

  const hideAllLabels = useCallback(() => {
    setLabelVisibility(() => {
      const all: Record<number, boolean> = {};
      for (const lbl of BRAIN_LABELS) all[lbl.id] = false;
      return all;
    });
  }, []);

  const macroGroupState = useMemo<Record<TissueClass, MacroState>>(() => ({
    gm:  getMacroGroupState('gm',  labelVisibility),
    wm:  getMacroGroupState('wm',  labelVisibility),
    csf: getMacroGroupState('csf', labelVisibility),
  }), [labelVisibility]);

  // ── Subject B stable control updaters ─────────────────────────────────────

  const setControlsB = useCallback((partial: Partial<ViewerControls>) => {
    setControlsBState((prev) => ({ ...prev, ...partial }));
  }, []);

  const setShowOverlayB = useCallback((v: boolean) => {
    setShowOverlayBState(v);
    const { segBundle, ctx } = vtkRefB.current;
    if (!segBundle || !ctx) return;
    segBundle.axial.setVisibility(v);
    segBundle.coronal.setVisibility(v);
    segBundle.sagittal.setVisibility(v);
    ctx.renderWindow.render();
  }, []);

  const setLabelVisibleB = useCallback((labelId: number, visible: boolean) => {
    setLabelVisibilityB((prev) => ({ ...prev, [labelId]: visible }));
  }, []);

  const setGroupVisibleB = useCallback((group: TissueClass, visible: boolean) => {
    setLabelVisibilityB((prev) => {
      const updated = { ...prev };
      for (const lbl of LABELS_BY_GROUP[group]) updated[lbl.id] = visible;
      return updated;
    });
  }, []);

  const showAllLabelsB = useCallback(() => {
    setLabelVisibilityB(defaultLabelVisibility());
  }, []);

  const hideAllLabelsB = useCallback(() => {
    setLabelVisibilityB(() => {
      const all: Record<number, boolean> = {};
      for (const lbl of BRAIN_LABELS) all[lbl.id] = false;
      return all;
    });
  }, []);

  const macroGroupStateB = useMemo<Record<TissueClass, MacroState>>(() => ({
    gm:  getMacroGroupState('gm',  labelVisibilityB),
    wm:  getMacroGroupState('wm',  labelVisibilityB),
    csf: getMacroGroupState('csf', labelVisibilityB),
  }), [labelVisibilityB]);

  // Accept a new Subject B NIfTI file: store the File ref then parse via worker.
  const { processFile: processFileB } = workerB;
  const loadVolumeB = useCallback(async (file: File) => {
    fileBRef.current = file;
    await processFileB(file);
  }, [processFileB]);

  // Remove Subject B: tear down VTK B context and reset all B state.
  const clearVolumeB = useCallback(() => {
    const { ctx, lodHandle, segBundle, volumeBundle, mprBundle, ro } = vtkRefB.current;
    ro?.disconnect();
    lodHandle?.dispose();
    if (ctx) {
      if (volumeBundle) ctx.volumeRenderer.removeVolume(volumeBundle.actor);
      if (mprBundle) {
        ctx.axialRenderer.removeActor(mprBundle.axial.actor);
        ctx.coronalRenderer.removeActor(mprBundle.coronal.actor);
        ctx.sagittalRenderer.removeActor(mprBundle.sagittal.actor);
      }
      if (segBundle) {
        ctx.axialRenderer.removeActor(segBundle.axial);
        ctx.coronalRenderer.removeActor(segBundle.coronal);
        ctx.sagittalRenderer.removeActor(segBundle.sagittal);
      }
      destroyVtk(ctx);
    }
    vtkRefB.current = { ...EMPTY_REFS_B };
    fileBRef.current = null;
    setHasVolumeB(false);
    setMaxIB(1); setMaxJB(1); setMaxKB(1);
    setControlsBState(DEFAULT_CONTROLS);
    setInferenceStatusB({ phase: 'idle' });
    setHasOverlayB(false);
    setShowOverlayBState(true);
    setLabelVisibilityB(defaultLabelVisibility());
    setVolumetricsB(null);
    setVolumetricsLoadingB(false);
    setVolumetricsErrorB(null);
  }, []);

  // ── vtk.js init — Subject A (on mount) ─────────────────────────────────────
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const ctx = initVtk(container);
    vtkRef.current.ctx = ctx;

    const ro = new ResizeObserver(() => {
      if (vtkRef.current.ctx) resizeVtk(vtkRef.current.ctx, container);
    });
    ro.observe(container);

    return () => {
      ro.disconnect();
      vtkRef.current.lodHandle?.dispose();
      if (vtkRef.current.ctx) destroyVtk(vtkRef.current.ctx);
      vtkRef.current = { ctx: null, volumeBundle: null, mprBundle: null, lodHandle: null, segBundle: null, anomalyBundle: null, longitudinalBundle: null };
    };
  }, []);

  // Teardown Subject B VTK on component unmount (clearVolumeB is user-triggered).
  useEffect(() => {
    return () => {
      const { ctx, lodHandle, ro } = vtkRefB.current;
      ro?.disconnect();
      lodHandle?.dispose();
      if (ctx) destroyVtk(ctx);
    };
  }, []);

  // ── Subject A volume load ───────────────────────────────────────────────────
  useEffect(() => {
    const { ctx } = vtkRef.current;
    if (!ctx || !volume) return;

    setVtkError(null);
    setInferenceStatus({ phase: 'idle' });
    setHasOverlay(false);
    setShowOverlayState(true);

    vtkRef.current.lodHandle?.dispose();
    if (vtkRef.current.volumeBundle) ctx.volumeRenderer.removeVolume(vtkRef.current.volumeBundle.actor);
    if (vtkRef.current.mprBundle) {
      ctx.axialRenderer.removeActor(vtkRef.current.mprBundle.axial.actor);
      ctx.coronalRenderer.removeActor(vtkRef.current.mprBundle.coronal.actor);
      ctx.sagittalRenderer.removeActor(vtkRef.current.mprBundle.sagittal.actor);
    }
    if (vtkRef.current.segBundle) {
      ctx.axialRenderer.removeActor(vtkRef.current.segBundle.axial);
      ctx.coronalRenderer.removeActor(vtkRef.current.segBundle.coronal);
      ctx.sagittalRenderer.removeActor(vtkRef.current.segBundle.sagittal);
    }
    if (vtkRef.current.anomalyBundle) {
      ctx.axialRenderer.removeActor(vtkRef.current.anomalyBundle.axial);
      ctx.coronalRenderer.removeActor(vtkRef.current.anomalyBundle.coronal);
      ctx.sagittalRenderer.removeActor(vtkRef.current.anomalyBundle.sagittal);
      ctx.volumeRenderer.removeVolume(vtkRef.current.anomalyBundle.volume3d);
    }
    if (vtkRef.current.longitudinalBundle) {
      ctx.axialRenderer.removeActor(vtkRef.current.longitudinalBundle.axial);
      ctx.coronalRenderer.removeActor(vtkRef.current.longitudinalBundle.coronal);
      ctx.sagittalRenderer.removeActor(vtkRef.current.longitudinalBundle.sagittal);
      ctx.volumeRenderer.removeVolume(vtkRef.current.longitudinalBundle.volume3d);
    }
    vtkRef.current.volumeBundle      = null;
    vtkRef.current.mprBundle         = null;
    vtkRef.current.lodHandle         = null;
    vtkRef.current.segBundle         = null;
    vtkRef.current.anomalyBundle     = null;
    vtkRef.current.longitudinalBundle = null;
    setAnomalyStatus({ phase: 'idle' });
    setHasAnomalyOverlay(false);
    setShowAnomalyOverlayState(true);
    setNAnomalyVoxels(0);
    setLongitudinalStatus({ phase: 'idle' });
    setHasLongitudinalOverlay(false);
    setShowLongitudinalOverlayState(true);
    setNPositive(0);
    setNNegative(0);

    try {
      const { dims } = volume.header;
      if (!dims[1] || !dims[2] || !dims[3]) {
        throw new Error(`Invalid volume dimensions: ${dims[1]}×${dims[2]}×${dims[3]}`);
      }

      const volumeBundle = buildVolumeActor(volume);
      ctx.volumeRenderer.addVolume(volumeBundle.actor);

      const [dataLo, dataHi] = volumeBundle.dataRange;
      const autoCenter = Math.round((dataLo + dataHi) / 2);
      const autoWidth  = Math.round(dataHi - dataLo);

      const mprBundle = buildMprActors(volumeBundle.imageData, dims, autoCenter, autoWidth);
      ctx.axialRenderer.addActor(mprBundle.axial.actor);
      ctx.coronalRenderer.addActor(mprBundle.coronal.actor);
      ctx.sagittalRenderer.addActor(mprBundle.sagittal.actor);

      const lodHandle = attachLod(ctx.interactor, volumeBundle.mapper, ctx.renderWindow);

      ctx.axialRenderer.resetCamera();
      ctx.coronalRenderer.resetCamera();
      ctx.sagittalRenderer.resetCamera();
      ctx.volumeRenderer.resetCamera();
      setupMprCameras({
        axial:    ctx.axialRenderer,
        coronal:  ctx.coronalRenderer,
        sagittal: ctx.sagittalRenderer,
      });
      ctx.renderWindow.render();

      const midK = Math.floor(dims[3] / 2);
      const midJ = Math.floor(dims[2] / 2);
      const midI = Math.floor(dims[1] / 2);
      setMaxK(dims[3]); setMaxJ(dims[2]); setMaxI(dims[1]);
      setControlsState((prev) => ({
        ...prev,
        sliceK: midK, sliceJ: midJ, sliceI: midI,
        windowCenter: autoCenter, windowWidth: autoWidth,
      }));

      vtkRef.current = { ctx, volumeBundle, mprBundle, lodHandle, segBundle: null, anomalyBundle: null, longitudinalBundle: null };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error('VTK rendering error:', err);
      setVtkError(msg);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [volume]);

  // ── Step 1: mark B as "pending init" so its container div renders ───────────
  // workerB.volume becomes non-null once the NIfTI worker finishes parsing.
  // Setting hasVolumeB causes a re-render that inserts containerBRef into the DOM.
  useEffect(() => {
    if (workerB.volume) setHasVolumeB(true);
  }, [workerB.volume]);

  // ── Step 2: init VTK B — runs after containerBRef is in the DOM ─────────────
  // Fires when hasVolumeB transitions true; at that point containerBRef.current
  // is guaranteed to exist because the render in Step 1 already completed.
  useEffect(() => {
    if (!hasVolumeB || !workerB.volume || vtkRefB.current.ctx) return;
    const container = containerBRef.current;
    if (!container) return;

    const volumeB = workerB.volume;
    const { dims } = volumeB.header;

    const ctxB = initVtk(container);

    const ro = new ResizeObserver(() => {
      if (vtkRefB.current.ctx) resizeVtk(vtkRefB.current.ctx, container);
    });
    ro.observe(container);

    const volumeBundleB = buildVolumeActor(volumeB);
    ctxB.volumeRenderer.addVolume(volumeBundleB.actor);

    const [lo, hi] = volumeBundleB.dataRange;
    const autoCenter = Math.round((lo + hi) / 2);
    const autoWidth  = Math.round(hi - lo);

    const mprBundleB = buildMprActors(volumeBundleB.imageData, dims, autoCenter, autoWidth);
    ctxB.axialRenderer.addActor(mprBundleB.axial.actor);
    ctxB.coronalRenderer.addActor(mprBundleB.coronal.actor);
    ctxB.sagittalRenderer.addActor(mprBundleB.sagittal.actor);

    const lodB = attachLod(ctxB.interactor, volumeBundleB.mapper, ctxB.renderWindow);

    ctxB.axialRenderer.resetCamera();
    ctxB.coronalRenderer.resetCamera();
    ctxB.sagittalRenderer.resetCamera();
    ctxB.volumeRenderer.resetCamera();
    setupMprCameras({
      axial:    ctxB.axialRenderer,
      coronal:  ctxB.coronalRenderer,
      sagittal: ctxB.sagittalRenderer,
    });
    ctxB.renderWindow.render();

    const midK = Math.floor(dims[3] / 2);
    const midJ = Math.floor(dims[2] / 2);
    const midI = Math.floor(dims[1] / 2);
    setMaxKB(dims[3]); setMaxJB(dims[2]); setMaxIB(dims[1]);
    setControlsBState((prev) => ({
      ...prev,
      sliceK: midK, sliceJ: midJ, sliceI: midI,
      windowCenter: autoCenter, windowWidth: autoWidth,
    }));

    vtkRefB.current = { ctx: ctxB, volumeBundle: volumeBundleB, mprBundle: mprBundleB, lodHandle: lodB, segBundle: null, ro };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasVolumeB, workerB.volume]);

  // ── Step 3: auto dual segmentation when Subject B first loads ───────────────
  // Runs concurrently for both subjects immediately after VTK B is ready.
  // Overwrites any existing segmentation on A so both are fresh and comparable.
  useEffect(() => {
    if (!hasVolumeB || !volume?.file || !fileBRef.current) return;
    if (!vtkRefB.current.ctx) return;  // VTK B not yet initialised — skip

    const { ctx: ctxA, volumeBundle: vbA } = vtkRef.current;
    const { ctx: ctxB, volumeBundle: vbB } = vtkRefB.current;
    if (!ctxA || !vbA || !ctxB || !vbB) return;

    // Remove stale seg overlays before the new concurrent run.
    if (vtkRef.current.segBundle) {
      ctxA.axialRenderer.removeActor(vtkRef.current.segBundle.axial);
      ctxA.coronalRenderer.removeActor(vtkRef.current.segBundle.coronal);
      ctxA.sagittalRenderer.removeActor(vtkRef.current.segBundle.sagittal);
      vtkRef.current.segBundle = null;
      setHasOverlay(false);
    }
    if (vtkRefB.current.segBundle) {
      ctxB.axialRenderer.removeActor(vtkRefB.current.segBundle.axial);
      ctxB.coronalRenderer.removeActor(vtkRefB.current.segBundle.coronal);
      ctxB.sagittalRenderer.removeActor(vtkRefB.current.segBundle.sagittal);
      vtkRefB.current.segBundle = null;
      setHasOverlayB(false);
    }

    setInferenceStatus({ phase: 'uploading' });
    setInferenceStatusB({ phase: 'uploading' });

    const fileA = volume.file;
    const fileB = fileBRef.current;

    // Compute midpoints from workerB.volume header for initial slice sync on B.
    const bDims = workerB.volume?.header.dims;
    const bMidK = bDims ? Math.floor(bDims[3] / 2) : 0;
    const bMidJ = bDims ? Math.floor(bDims[2] / 2) : 0;
    const bMidI = bDims ? Math.floor(bDims[1] / 2) : 0;

    dualSegmentApi.segmentBoth(fileA, fileB)
      .then(([resultA, resultB]) => {
        setInferenceStatus({ phase: 'running' });
        setInferenceStatusB({ phase: 'running' });

        // Decode A labels and mount overlay.
        const binA = atob(resultA.labels);
        const labA = new Uint8Array(binA.length);
        for (let i = 0; i < binA.length; i++) labA[i] = binA.charCodeAt(i);
        const sbA = buildSegmentationOverlay(
          vbA.imageData, labA, resultA.dims as [number, number, number], labelVisibilityRef.current,
        );
        // Use the captured controls from this render cycle for A's slice sync.
        sbA.setSlices(controls.sliceK, controls.sliceJ, controls.sliceI);
        ctxA.axialRenderer.addActor(sbA.axial);
        ctxA.coronalRenderer.addActor(sbA.coronal);
        ctxA.sagittalRenderer.addActor(sbA.sagittal);
        vtkRef.current.segBundle = sbA;
        ctxA.renderWindow.render();
        setHasOverlay(true);
        setShowOverlayState(true);
        setInferenceStatus({ phase: 'done', durationMs: Math.round(resultA.duration_ms) });

        // Decode B labels and mount overlay.
        const binB = atob(resultB.labels);
        const labB = new Uint8Array(binB.length);
        for (let i = 0; i < binB.length; i++) labB[i] = binB.charCodeAt(i);
        const sbB = buildSegmentationOverlay(
          vbB.imageData, labB, resultB.dims as [number, number, number], labelVisibilityBRef.current,
        );
        sbB.setSlices(bMidK, bMidJ, bMidI);
        ctxB.axialRenderer.addActor(sbB.axial);
        ctxB.coronalRenderer.addActor(sbB.coronal);
        ctxB.sagittalRenderer.addActor(sbB.sagittal);
        vtkRefB.current.segBundle = sbB;
        ctxB.renderWindow.render();
        setHasOverlayB(true);
        setShowOverlayBState(true);
        setInferenceStatusB({ phase: 'done', durationMs: Math.round(resultB.duration_ms) });
      })
      .catch((err) => {
        const msg = err instanceof Error ? err.message : String(err);
        setInferenceStatus({ phase: 'error', message: msg });
        setInferenceStatusB({ phase: 'error', message: msg });
      });
  // Intentionally only fires when hasVolumeB transitions false→true.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasVolumeB]);

  // ── A slice position ────────────────────────────────────────────────────────
  useEffect(() => {
    const { mprBundle, segBundle, ctx } = vtkRef.current;
    if (!mprBundle || !ctx) return;
    mprBundle.axial.setSlice(controls.sliceK);
    mprBundle.coronal.setSlice(controls.sliceJ);
    mprBundle.sagittal.setSlice(controls.sliceI);
    segBundle?.setSlices(controls.sliceK, controls.sliceJ, controls.sliceI);
    vtkRef.current.anomalyBundle?.setSlices(controls.sliceK, controls.sliceJ, controls.sliceI);
    vtkRef.current.longitudinalBundle?.setSlices(controls.sliceK, controls.sliceJ, controls.sliceI);
    ctx.renderWindow.render();
  }, [controls.sliceK, controls.sliceJ, controls.sliceI]);

  // ── A window/level ──────────────────────────────────────────────────────────
  useEffect(() => {
    const { mprBundle, volumeBundle, ctx } = vtkRef.current;
    if (!ctx) return;
    mprBundle?.axial.setWindowLevel(controls.windowCenter, controls.windowWidth);
    mprBundle?.coronal.setWindowLevel(controls.windowCenter, controls.windowWidth);
    mprBundle?.sagittal.setWindowLevel(controls.windowCenter, controls.windowWidth);
    volumeBundle?.updateWindowLevel(controls.windowCenter, controls.windowWidth);
    ctx.renderWindow.render();
  }, [controls.windowCenter, controls.windowWidth]);

  // ── A opacity & visibility ──────────────────────────────────────────────────
  useEffect(() => {
    const { volumeBundle, ctx } = vtkRef.current;
    if (!volumeBundle || !ctx) return;
    volumeBundle.updateOpacity(controls.volumeOpacity);
    volumeBundle.actor.setVisibility(controls.showVolume);
    ctx.renderWindow.render();
  }, [controls.volumeOpacity, controls.showVolume]);

  // ── A label visibility ref sync ─────────────────────────────────────────────
  useEffect(() => {
    labelVisibilityRef.current = labelVisibility;
  }, [labelVisibility]);

  // ── A per-label visibility → VTK ───────────────────────────────────────────
  useEffect(() => {
    const { segBundle, ctx } = vtkRef.current;
    if (!segBundle || !ctx) return;
    segBundle.updateLabelVisibility(labelVisibility);
    ctx.renderWindow.render();
  }, [labelVisibility]);

  // ── B slice position ────────────────────────────────────────────────────────
  useEffect(() => {
    const { mprBundle, segBundle, ctx } = vtkRefB.current;
    if (!mprBundle || !ctx) return;
    mprBundle.axial.setSlice(controlsB.sliceK);
    mprBundle.coronal.setSlice(controlsB.sliceJ);
    mprBundle.sagittal.setSlice(controlsB.sliceI);
    segBundle?.setSlices(controlsB.sliceK, controlsB.sliceJ, controlsB.sliceI);
    ctx.renderWindow.render();
  }, [controlsB.sliceK, controlsB.sliceJ, controlsB.sliceI]);

  // ── B window/level ──────────────────────────────────────────────────────────
  useEffect(() => {
    const { mprBundle, volumeBundle, ctx } = vtkRefB.current;
    if (!ctx) return;
    mprBundle?.axial.setWindowLevel(controlsB.windowCenter, controlsB.windowWidth);
    mprBundle?.coronal.setWindowLevel(controlsB.windowCenter, controlsB.windowWidth);
    mprBundle?.sagittal.setWindowLevel(controlsB.windowCenter, controlsB.windowWidth);
    volumeBundle?.updateWindowLevel(controlsB.windowCenter, controlsB.windowWidth);
    ctx.renderWindow.render();
  }, [controlsB.windowCenter, controlsB.windowWidth]);

  // ── B opacity & visibility ──────────────────────────────────────────────────
  useEffect(() => {
    const { volumeBundle, ctx } = vtkRefB.current;
    if (!volumeBundle || !ctx) return;
    volumeBundle.updateOpacity(controlsB.volumeOpacity);
    volumeBundle.actor.setVisibility(controlsB.showVolume);
    ctx.renderWindow.render();
  }, [controlsB.volumeOpacity, controlsB.showVolume]);

  // ── B label visibility ref sync ─────────────────────────────────────────────
  useEffect(() => {
    labelVisibilityBRef.current = labelVisibilityB;
  }, [labelVisibilityB]);

  // ── B per-label visibility → VTK ───────────────────────────────────────────
  useEffect(() => {
    const { segBundle, ctx } = vtkRefB.current;
    if (!segBundle || !ctx) return;
    segBundle.updateLabelVisibility(labelVisibilityB);
    ctx.renderWindow.render();
  }, [labelVisibilityB]);

  // ── Subject A — segmentation (manual trigger) ──────────────────────────────
  const runSegmentation = useCallback(async () => {
    if (!volume?.file) return;
    const { ctx, volumeBundle, mprBundle } = vtkRef.current;
    if (!ctx || !volumeBundle || !mprBundle) return;

    if (vtkRef.current.segBundle) {
      ctx.axialRenderer.removeActor(vtkRef.current.segBundle.axial);
      ctx.coronalRenderer.removeActor(vtkRef.current.segBundle.coronal);
      ctx.sagittalRenderer.removeActor(vtkRef.current.segBundle.sagittal);
      vtkRef.current.segBundle = null;
      setHasOverlay(false);
    }

    try {
      setInferenceStatus({ phase: 'uploading' });
      const result = await segmentApi.segment(volume.file!);
      setInferenceStatus({ phase: 'running' });

      const bin = atob(result.labels);
      const labelArray = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) labelArray[i] = bin.charCodeAt(i);

      const segBundle = buildSegmentationOverlay(
        volumeBundle.imageData,
        labelArray,
        result.dims as [number, number, number],
        labelVisibilityRef.current,
      );
      segBundle.setSlices(controls.sliceK, controls.sliceJ, controls.sliceI);
      ctx.axialRenderer.addActor(segBundle.axial);
      ctx.coronalRenderer.addActor(segBundle.coronal);
      ctx.sagittalRenderer.addActor(segBundle.sagittal);
      vtkRef.current.segBundle = segBundle;
      ctx.renderWindow.render();
      setHasOverlay(true);
      setShowOverlayState(true);
      setInferenceStatus({ phase: 'done', durationMs: Math.round(result.duration_ms) });
    } catch (err) {
      setInferenceStatus({
        phase: 'error',
        message: err instanceof Error ? err.message : String(err),
      });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [volume, controls.sliceK, controls.sliceJ, controls.sliceI]);

  // ── Hippocampal volumetrics — Subject A ────────────────────────────────────
  const runVolumetrics = useCallback(async () => {
    const { segBundle } = vtkRef.current;
    if (!segBundle || !volume) return;

    const { dims, pixDims } = volume.header;
    const nx = dims[1] ?? 1; const ny = dims[2] ?? 1; const nz = dims[3] ?? 1;
    const dx = pixDims[1] ?? 1; const dy = pixDims[2] ?? 1; const dz = pixDims[3] ?? 1;

    setVolumetricsLoading(true);
    setVolumetricsError(null);
    try {
      const result = await mriApi.computeVolumetrics(segBundle.labelFlat, [nx, ny, nz], [dx, dy, dz]);
      setVolumetrics(result);
    } catch (err) {
      setVolumetricsError(err instanceof Error ? err.message : String(err));
    } finally {
      setVolumetricsLoading(false);
    }
  }, [volume]);

  // ── Hippocampal volumetrics — Subject B ────────────────────────────────────
  const runVolumetricsB = useCallback(async () => {
    const { segBundle } = vtkRefB.current;
    const volB = workerB.volume;
    if (!segBundle || !volB) return;

    const { dims, pixDims } = volB.header;
    const nx = dims[1] ?? 1; const ny = dims[2] ?? 1; const nz = dims[3] ?? 1;
    const dx = pixDims[1] ?? 1; const dy = pixDims[2] ?? 1; const dz = pixDims[3] ?? 1;

    setVolumetricsLoadingB(true);
    setVolumetricsErrorB(null);
    try {
      const result = await mriApi.computeVolumetrics(segBundle.labelFlat, [nx, ny, nz], [dx, dy, dz]);
      setVolumetricsB(result);
    } catch (err) {
      setVolumetricsErrorB(err instanceof Error ? err.message : String(err));
    } finally {
      setVolumetricsLoadingB(false);
    }
  }, [workerB.volume]);

  // ── Anomaly overlay callbacks ───────────────────────────────────────────────

  const setShowAnomalyOverlay = useCallback((v: boolean) => {
    setShowAnomalyOverlayState(v);
    const { anomalyBundle, ctx } = vtkRef.current;
    if (!anomalyBundle || !ctx) return;
    anomalyBundle.axial.setVisibility(v);
    anomalyBundle.coronal.setVisibility(v);
    anomalyBundle.sagittal.setVisibility(v);
    anomalyBundle.volume3d.setVisibility(v);
    ctx.renderWindow.render();
  }, []);

  const setAnomalyOpacity = useCallback((factor: number) => {
    setAnomalyOpacityState(factor);
    const { anomalyBundle, ctx } = vtkRef.current;
    if (!anomalyBundle || !ctx) return;
    anomalyBundle.setOpacity(factor);
    ctx.renderWindow.render();
  }, []);

  const runAnomalyDetection = useCallback(async () => {
    if (!volume?.file) return;
    const { ctx, volumeBundle } = vtkRef.current;
    if (!ctx || !volumeBundle) return;

    if (vtkRef.current.anomalyBundle) {
      ctx.axialRenderer.removeActor(vtkRef.current.anomalyBundle.axial);
      ctx.coronalRenderer.removeActor(vtkRef.current.anomalyBundle.coronal);
      ctx.sagittalRenderer.removeActor(vtkRef.current.anomalyBundle.sagittal);
      ctx.volumeRenderer.removeVolume(vtkRef.current.anomalyBundle.volume3d);
      vtkRef.current.anomalyBundle = null;
      setHasAnomalyOverlay(false);
    }

    try {
      setAnomalyStatus({ phase: 'uploading' });
      const result = await anomalyApi.detect(volume.file!);
      setAnomalyStatus({ phase: 'running' });

      const bin = atob(result.mask);
      const maskArray = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) maskArray[i] = bin.charCodeAt(i);

      const anomalyBundle = buildAnomalyOverlay(volumeBundle.imageData, maskArray, result.dims, anomalyOpacity);
      anomalyBundle.setSlices(controls.sliceK, controls.sliceJ, controls.sliceI);
      ctx.axialRenderer.addActor(anomalyBundle.axial);
      ctx.coronalRenderer.addActor(anomalyBundle.coronal);
      ctx.sagittalRenderer.addActor(anomalyBundle.sagittal);
      ctx.volumeRenderer.addVolume(anomalyBundle.volume3d);
      vtkRef.current.anomalyBundle = anomalyBundle;
      ctx.renderWindow.render();
      setHasAnomalyOverlay(true);
      setShowAnomalyOverlayState(true);
      setNAnomalyVoxels(result.n_anomaly);
      setAnomalyStatus({ phase: 'done', durationMs: Math.round(result.duration_ms), nAnomaly: result.n_anomaly });
    } catch (err) {
      setAnomalyStatus({ phase: 'error', message: err instanceof Error ? err.message : String(err) });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [volume, controls.sliceK, controls.sliceJ, controls.sliceI, anomalyOpacity]);

  const clearAnomalyOverlay = useCallback(() => {
    const { anomalyBundle, ctx } = vtkRef.current;
    if (!anomalyBundle || !ctx) return;
    ctx.axialRenderer.removeActor(anomalyBundle.axial);
    ctx.coronalRenderer.removeActor(anomalyBundle.coronal);
    ctx.sagittalRenderer.removeActor(anomalyBundle.sagittal);
    ctx.volumeRenderer.removeVolume(anomalyBundle.volume3d);
    vtkRef.current.anomalyBundle = null;
    ctx.renderWindow.render();
    setHasAnomalyOverlay(false);
    setShowAnomalyOverlayState(true);
    setAnomalyOpacityState(0.8);
    setNAnomalyVoxels(0);
    setAnomalyStatus({ phase: 'idle' });
  }, []);

  // ── Longitudinal overlay callbacks ──────────────────────────────────────────

  const setShowLongitudinalOverlay = useCallback((v: boolean) => {
    setShowLongitudinalOverlayState(v);
    const { longitudinalBundle, ctx } = vtkRef.current;
    if (!longitudinalBundle || !ctx) return;
    longitudinalBundle.axial.setVisibility(v);
    longitudinalBundle.coronal.setVisibility(v);
    longitudinalBundle.sagittal.setVisibility(v);
    longitudinalBundle.volume3d.setVisibility(v);
    ctx.renderWindow.render();
  }, []);

  const setLongitudinalOpacity = useCallback((factor: number) => {
    setLongitudinalOpacityState(factor);
    const { longitudinalBundle, ctx } = vtkRef.current;
    if (!longitudinalBundle || !ctx) return;
    longitudinalBundle.setOpacity(factor);
    ctx.renderWindow.render();
  }, []);

  const runLongitudinalDelta = useCallback(async (
    baseline: File, followup: File, transformType: 'rigid' | 'affine',
  ) => {
    const { ctx, volumeBundle } = vtkRef.current;
    if (!ctx || !volumeBundle) return;

    if (vtkRef.current.longitudinalBundle) {
      ctx.axialRenderer.removeActor(vtkRef.current.longitudinalBundle.axial);
      ctx.coronalRenderer.removeActor(vtkRef.current.longitudinalBundle.coronal);
      ctx.sagittalRenderer.removeActor(vtkRef.current.longitudinalBundle.sagittal);
      ctx.volumeRenderer.removeVolume(vtkRef.current.longitudinalBundle.volume3d);
      vtkRef.current.longitudinalBundle = null;
      setHasLongitudinalOverlay(false);
    }

    try {
      setLongitudinalStatus({ phase: 'uploading' });
      const result = await longitudinalApi.computeDelta(baseline, followup, transformType);
      setLongitudinalStatus({ phase: 'running' });

      const bin       = atob(result.delta);
      const byteArray = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) byteArray[i] = bin.charCodeAt(i);
      const deltaF32  = new Float32Array(byteArray.buffer);

      const longitudinalBundle = buildLongitudinalOverlay(
        volumeBundle.imageData, deltaF32, result.dims, result.min_val, result.max_val, longitudinalOpacity,
      );
      longitudinalBundle.setSlices(controls.sliceK, controls.sliceJ, controls.sliceI);
      ctx.axialRenderer.addActor(longitudinalBundle.axial);
      ctx.coronalRenderer.addActor(longitudinalBundle.coronal);
      ctx.sagittalRenderer.addActor(longitudinalBundle.sagittal);
      ctx.volumeRenderer.addVolume(longitudinalBundle.volume3d);
      vtkRef.current.longitudinalBundle = longitudinalBundle;
      ctx.renderWindow.render();
      setHasLongitudinalOverlay(true);
      setShowLongitudinalOverlayState(true);
      setNPositive(longitudinalBundle.nPositive);
      setNNegative(longitudinalBundle.nNegative);
      setLongitudinalStatus({
        phase: 'done',
        durationMs: Math.round(result.duration_ms),
        nPositive:  longitudinalBundle.nPositive,
        nNegative:  longitudinalBundle.nNegative,
      });
    } catch (err) {
      setLongitudinalStatus({ phase: 'error', message: err instanceof Error ? err.message : String(err) });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [controls.sliceK, controls.sliceJ, controls.sliceI, longitudinalOpacity]);

  const clearLongitudinalOverlay = useCallback(() => {
    const { longitudinalBundle, ctx } = vtkRef.current;
    if (!longitudinalBundle || !ctx) return;
    ctx.axialRenderer.removeActor(longitudinalBundle.axial);
    ctx.coronalRenderer.removeActor(longitudinalBundle.coronal);
    ctx.sagittalRenderer.removeActor(longitudinalBundle.sagittal);
    ctx.volumeRenderer.removeVolume(longitudinalBundle.volume3d);
    vtkRef.current.longitudinalBundle = null;
    ctx.renderWindow.render();
    setHasLongitudinalOverlay(false);
    setShowLongitudinalOverlayState(true);
    setLongitudinalOpacityState(0.8);
    setNPositive(0);
    setNNegative(0);
    setLongitudinalStatus({ phase: 'idle' });
  }, []);

  // ── Context values ──────────────────────────────────────────────────────────

  const contextValue: VolumetricContextValue = {
    controls, setControls,
    maxI, maxJ, maxK,
    inferenceStatus,
    hasVolume:   volume !== null && !!volume?.file,
    hasOverlay,
    showOverlay,
    setShowOverlay,
    runSegmentation,
    labelVisibility,
    macroGroupState,
    setLabelVisible,
    setGroupVisible,
    showAllLabels,
    hideAllLabels,
    volumetrics,
    volumetricsLoading,
    volumetricsError,
    runVolumetrics,
    decodingData,
    decodingStatus,
    runNeuralDecoding,
    currentTimeIndex,
    setCurrentTimeIndex,
    connectomeData,
    setConnectomeData,
    getVtkCtx,
    selectedEdge,
    setSelectedEdge,
    anomalyStatus,
    hasAnomalyOverlay,
    showAnomalyOverlay,
    setShowAnomalyOverlay,
    anomalyOpacity,
    setAnomalyOpacity,
    nAnomalyVoxels,
    runAnomalyDetection,
    clearAnomalyOverlay,
    longitudinalStatus,
    hasLongitudinalOverlay,
    showLongitudinalOverlay,
    setShowLongitudinalOverlay,
    longitudinalOpacity,
    setLongitudinalOpacity,
    nPositive,
    nNegative,
    runLongitudinalDelta,
    clearLongitudinalOverlay,
  };

  const comparisonValue: ComparisonContextValue = {
    hasVolumeB,
    loadVolumeB,
    clearVolumeB,
    controlsB,
    setControlsB,
    maxIB, maxJB, maxKB,
    inferenceStatusB,
    hasOverlayB,
    showOverlayB,
    setShowOverlayB,
    labelVisibilityB,
    macroGroupStateB,
    setLabelVisibleB,
    setGroupVisibleB,
    showAllLabelsB,
    hideAllLabelsB,
    volumetricsB,
    volumetricsLoadingB,
    volumetricsErrorB,
    runVolumetricsB,
  };

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <VolumetricContext.Provider value={contextValue}>
      <ComparisonContext.Provider value={comparisonValue}>
        <div className={`plugin-workspace${hasVolumeB ? ' plugin-workspace--split' : ''}`}>

          {/* Subject A pane — always present */}
          <div
            className={hasVolumeB ? 'vtk-pane' : 'vtk-container'}
            ref={containerRef as RefObject<HTMLDivElement>}
          >
            {hasVolumeB && (
              <div className="vtk-pane-label">Subject A</div>
            )}
            <div className="quadrant-label quadrant-label--tl" aria-hidden>Axial</div>
            <div className="quadrant-label quadrant-label--tr" aria-hidden>Coronal</div>
            <div className="quadrant-label quadrant-label--bl" aria-hidden>Sagittal</div>
            <div className="quadrant-label quadrant-label--br" aria-hidden>3D Volume</div>
            <div className="panel-divider panel-divider--h" aria-hidden />
            <div className="panel-divider panel-divider--v" aria-hidden />
            {!volume && !vtkError && (
              <div className="empty-state"><p>Upload a NIfTI scan (.nii / .nii.gz)</p></div>
            )}
            {vtkError && (
              <div className="empty-state">
                <p style={{ color: 'var(--accent-red, #e05252)' }}>Rendering error: {vtkError}</p>
              </div>
            )}
          </div>

          {/* Subject B pane — rendered only once B is loaded */}
          {hasVolumeB && (
            <div className="vtk-pane" ref={containerBRef as RefObject<HTMLDivElement>}>
              <div className="vtk-pane-label">Subject B</div>
              <div className="quadrant-label quadrant-label--tl" aria-hidden>Axial</div>
              <div className="quadrant-label quadrant-label--tr" aria-hidden>Coronal</div>
              <div className="quadrant-label quadrant-label--bl" aria-hidden>Sagittal</div>
              <div className="quadrant-label quadrant-label--br" aria-hidden>3D Volume</div>
              <div className="panel-divider panel-divider--h" aria-hidden />
              <div className="panel-divider panel-divider--v" aria-hidden />
              {workerB.loading && (
                <div className="empty-state"><p>Parsing Subject B…</p></div>
              )}
              {workerB.error && (
                <div className="empty-state">
                  <p style={{ color: 'var(--accent-red, #e05252)' }}>{workerB.error}</p>
                </div>
              )}
            </div>
          )}

          {controlsSlot}
        </div>
      </ComparisonContext.Provider>
    </VolumetricContext.Provider>
  );
};

export default VolumetricViewer;
