/**
 * DualVolumetricViewer.tsx — Side-by-side inter-subject comparison viewer
 * ──────────────────────────────────────────────────────────────────────────
 *
 * Renders two independent vtk.js canvases:
 *   Left  — Subject A (reference, always in its native space).
 *   Right — Subject B, whose content depends on the current alignment mode:
 *
 *     'raw'        Displays Subject B's original unwarped NIfTI immediately
 *                  after the file is selected.  Cameras are DECOUPLED so each
 *                  pane can be inspected independently.  Load time < 2 s.
 *
 *     'registered' Warps Subject B into Subject A's space via Affine + SyN
 *                  diffeomorphic registration (POST /api/registration/syn).
 *                  Cameras are LOCKED so any rotation/zoom/pan is mirrored.
 *                  The first switch triggers the backend call (~2-5 min).
 *                  Subsequent switches use the in-memory actor cache — instant.
 *
 * Phase 10 changes (alignment mode toggle)
 * ─────────────────────────────────────────
 *   • useNiftiWorker() is used to parse Subject B's raw NIfTI in a worker
 *     thread, producing raw VTK actors that are mounted immediately on load.
 *   • DualVtkRefs gains rawRight* fields for the raw-B actor bundle.
 *   • alignmentMode effect handles the actor swap:
 *       'raw'        → unlock cameras, unmount warped B, mount raw B.
 *       'registered' → unmount raw B, run SyN or restore from actor cache,
 *                       mount warped B, lock cameras.
 *   • runSyn() now pre-removes raw B before the API call and restores it on
 *     error to ensure the right pane always shows something.
 *   • synCachedRef mirrors synCached state so the alignmentMode effect can
 *     branch without synCached in its deps array (prevents extra firings).
 *
 * VTK ACTOR LIFECYCLE
 * ────────────────────
 *   Actors are built once and kept in vtkRef.  Switching modes only calls
 *   addVolume/addActor and removeVolume/removeActor — the underlying
 *   vtkImageData and GPU texture are NOT re-uploaded.  This keeps mode
 *   switching instantaneous after the first SyN run.
 */

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  useCallback,
  type FC,
} from 'react';

import { initDualVtk, lockCameras }        from '../../lib/vtk/dualVtkSetup';
import { buildVolumeActor }                 from '../../lib/vtk/volumeRenderer';
import { buildMprActors, setupMprCameras }  from '../../lib/vtk/mprRenderer';
import { buildRawVolumeActor }              from '../../lib/vtk/rawVolumeBuilder';
import { attachLod }                        from '../../lib/vtk/lodManager';
import { buildDualSegOverlay }              from '../../lib/vtk/dualSegmentationOverlay';
import { DEFAULT_TISSUE_VISIBILITY }        from '../../lib/vtk/tissueGroups';
import {
  defaultLabelVisibility,
  getMacroGroupState,
  LABELS_BY_GROUP,
  BRAIN_LABELS,
  type MacroState,
}                                           from '../../lib/vtk/labelVisibility';
import { synRegistrationApi }               from '../../services/synRegistrationApi';
import { resizeVtk }                        from '../../lib/vtk/vtkSetup';
import { useNiftiWorker }                   from '../../hooks/useNiftiWorker';

import type { DualVtkContext }              from '../../lib/vtk/dualVtkSetup';
import type { VolumeActorBundle }           from '../../lib/vtk/volumeRenderer';
import type { MprBundle }                   from '../../lib/vtk/mprRenderer';
import type { LodHandle }                   from '../../lib/vtk/lodManager';
import type { DualSegBundle }               from '../../lib/vtk/dualSegmentationOverlay';
import type { TissueGroupVisibility, TissueClass } from '../../lib/vtk/tissueGroups';
import type { VolumePayload }               from '../../types/nifti.types';
import type { SynVolumetrics }              from '../../services/synRegistrationApi';

import {
  DualViewerContext,
  type AlignmentMode,
  type SynStatus,
  type DualViewerContextValue,
} from './DualViewerContext';
import DualViewerControls from './DualViewerControls';

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Decode a base64 string → Uint8Array without using the spread operator. */
function base64ToUint8(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// ── Mutable VTK refs ───────────────────────────────────────────────────────────

interface DualVtkRefs {
  /** Both render windows + contexts created on mount. */
  dual: DualVtkContext | null;

  // Left viewer (Subject A, reference) — unchanged from previous version.
  leftVolumeBundle:   VolumeActorBundle | null;
  leftMprBundle:      MprBundle | null;
  leftLodHandle:      LodHandle | null;

  // Right viewer warped actors (Subject B registered into A's space).
  // Built during runSyn(); kept in memory as the cache.
  // In 'raw' mode these are NOT added to any renderer; in 'registered' they are.
  rightVolumeBundle:  VolumeActorBundle | null;
  rightMprBundle:     MprBundle | null;
  rightLodHandle:     LodHandle | null;

  // Right viewer raw actors (Subject B in its own native space).
  // Built by useNiftiWorker when the user picks a Subject B file.
  // In 'raw' mode these ARE added to the right renderer; in 'registered' they are not.
  rawRightVolumeBundle: VolumeActorBundle | null;
  rawRightMprBundle:    MprBundle | null;
  rawRightLodHandle:    LodHandle | null;

  /**
   * Cleanup function returned by lockCameras().
   * Non-null only while 'registered' mode is active.
   * Calling it decouples the camera event listeners.
   */
  cameraCleanup: (() => void) | null;

  /** The original NIfTI File for Subject A — kept for the SyN upload. */
  subjectAFile: File | null;

  /** Paired segmentation overlay for both viewers (registered mode only). */
  dualSegBundle: DualSegBundle | null;
}

const EMPTY_REFS: DualVtkRefs = {
  dual:                 null,
  leftVolumeBundle:     null,
  leftMprBundle:        null,
  leftLodHandle:        null,
  rightVolumeBundle:    null,
  rightMprBundle:       null,
  rightLodHandle:       null,
  rawRightVolumeBundle: null,
  rawRightMprBundle:    null,
  rawRightLodHandle:    null,
  cameraCleanup:        null,
  subjectAFile:         null,
  dualSegBundle:        null,
};

// ── Component ─────────────────────────────────────────────────────────────────

interface DualVolumetricViewerProps {
  /** Parsed Subject A NIfTI — with .file present for the SyN upload. */
  subjectAPayload: VolumePayload | null;
}

const DualVolumetricViewer: FC<DualVolumetricViewerProps> = ({ subjectAPayload }) => {
  const leftContainerRef  = useRef<HTMLDivElement>(null);
  const rightContainerRef = useRef<HTMLDivElement>(null);
  const vtkRef            = useRef<DualVtkRefs>({ ...EMPTY_REFS });

  // ── Worker for Subject B raw NIfTI parsing ─────────────────────────────────
  // A separate worker thread so the main thread (and thus the VTK canvas) stays
  // responsive while large .nii.gz files are being decompressed and parsed.
  const workerB = useNiftiWorker();
  const { processFile: processBFile } = workerB;

  // Holds the raw File reference for Subject B so runSyn() can upload it even
  // when the NIfTI worker has already consumed (transferred) the ArrayBuffer.
  const fileBRef = useRef<File | null>(null);

  // ── React state ──────────────────────────────────────────────────────────
  const [subjectALoaded,  setSubjectALoaded]  = useState(false);
  const [warpedLoaded,    setWarpedLoaded]    = useState(false);
  const [subjectBFile,    setSubjectBFileState] = useState<File | null>(null);
  const [synStatus,       setSynStatus]       = useState<SynStatus>({ phase: 'idle' });

  // Alignment mode state — 'raw' by default; toggles via AlignmentModeToggle.
  const [alignmentMode,   setAlignmentModeState] = useState<AlignmentMode>('raw');
  // rawBLoaded: true once workerB has parsed B and raw VTK actors are ready.
  const [rawBLoaded,      setRawBLoaded]      = useState(false);
  // synCached: true once SyN has run at least once and warped actors are built.
  const [synCached,       setSynCached]       = useState(false);

  // Ref mirror for synCached — lets the alignmentMode effect branch without
  // including synCached in its deps (which would cause extra spurious firings).
  const synCachedRef    = useRef(false);
  // Ref mirror for alignmentMode — used by the workerB.volume effect without
  // re-triggering the effect when alignmentMode changes.
  const alignmentModeRef = useRef<AlignmentMode>('raw');

  // Shared slice indices — applied to both viewers via effects.
  const [sliceK, setSliceKState] = useState(0);
  const [sliceJ, setSliceJState] = useState(0);
  const [sliceI, setSliceIState] = useState(0);
  const [maxK,   setMaxK]        = useState(1);
  const [maxJ,   setMaxJ]        = useState(1);
  const [maxI,   setMaxI]        = useState(1);

  const [windowCenter,  setWindowCenterState] = useState(500);
  const [windowWidth,   setWindowWidthState]  = useState(1000);
  const [volumeOpacity, setVolumeOpacityState] = useState(0.8);

  const [hasSegmentation,  setHasSegmentation]    = useState(false);
  const [tissueVisibility, setTissueVisibilityState] = useState<TissueGroupVisibility>(DEFAULT_TISSUE_VISIBILITY);
  const [volumetrics,      setVolumetrics]         = useState<SynVolumetrics | null>(null);

  // Per-structure label visibility — drives fine-grained overlay control.
  // Initialized to all-visible; reset to default when Subject A changes.
  const [labelVisibility, setLabelVisibilityState] = useState<Record<number, boolean>>(
    () => defaultLabelVisibility(),
  );

  // Derived tristate per tissue class — used by DualAnatomySelector macro checkboxes.
  const macroGroupState = useMemo<Record<TissueClass, MacroState>>(() => ({
    gm:  getMacroGroupState('gm',  labelVisibility),
    wm:  getMacroGroupState('wm',  labelVisibility),
    csf: getMacroGroupState('csf', labelVisibility),
  }), [labelVisibility]);

  // ── Stable setters ──────────────────────────────────────────────────────────
  const setSubjectBFile   = useCallback((f: File | null)           => setSubjectBFileState(f),        []);
  const setSliceK         = useCallback((v: number)                => setSliceKState(v),               []);
  const setSliceJ         = useCallback((v: number)                => setSliceJState(v),               []);
  const setSliceI         = useCallback((v: number)                => setSliceIState(v),               []);
  const setWindowCenter   = useCallback((v: number)                => setWindowCenterState(v),         []);
  const setWindowWidth    = useCallback((v: number)                => setWindowWidthState(v),          []);
  const setVolumeOpacity  = useCallback((v: number)                => setVolumeOpacityState(v),        []);
  const setTissueVisibility = useCallback((vis: TissueGroupVisibility) => setTissueVisibilityState(vis), []);

  // ── Per-label visibility callbacks ──────────────────────────────────────────
  // These are consumed by DualAnatomySelector; each produces a new labelVisibility
  // object so the [labelVisibility] effect fires and updates the VTK overlay.
  const setLabelVisible = useCallback((id: number, visible: boolean) => {
    setLabelVisibilityState((prev) => ({ ...prev, [id]: visible }));
  }, []);

  const setGroupVisible = useCallback((cls: TissueClass, visible: boolean) => {
    setLabelVisibilityState((prev) => {
      const next = { ...prev };
      for (const lbl of LABELS_BY_GROUP[cls]) next[lbl.id] = visible;
      return next;
    });
  }, []);

  const showAllLabels = useCallback(() => {
    setLabelVisibilityState(defaultLabelVisibility());
  }, []);

  const hideAllLabels = useCallback(() => {
    const hidden: Record<number, boolean> = {};
    for (const lbl of BRAIN_LABELS) hidden[lbl.id] = false;
    setLabelVisibilityState(hidden);
  }, []);

  // Exposed to DualViewerControls via context; triggers the alignmentMode effect.
  const setAlignmentMode  = useCallback((mode: AlignmentMode)      => setAlignmentModeState(mode),    []);

  // ── Keep ref mirrors current ────────────────────────────────────────────────
  useEffect(() => { synCachedRef.current = synCached; },       [synCached]);
  useEffect(() => { alignmentModeRef.current = alignmentMode; }, [alignmentMode]);

  // ── VTK init on mount ───────────────────────────────────────────────────────
  useEffect(() => {
    const leftEl  = leftContainerRef.current;
    const rightEl = rightContainerRef.current;
    if (!leftEl || !rightEl) return;

    const dual = initDualVtk(leftEl, rightEl);
    vtkRef.current.dual = dual;

    const roLeft = new ResizeObserver(() => {
      if (vtkRef.current.dual) resizeVtk(vtkRef.current.dual.left, leftEl);
    });
    const roRight = new ResizeObserver(() => {
      if (vtkRef.current.dual) resizeVtk(vtkRef.current.dual.right, rightEl);
    });
    roLeft.observe(leftEl);
    roRight.observe(rightEl);

    return () => {
      roLeft.disconnect();
      roRight.disconnect();
      vtkRef.current.cameraCleanup?.();
      vtkRef.current.leftLodHandle?.dispose();
      vtkRef.current.rightLodHandle?.dispose();
      vtkRef.current.rawRightLodHandle?.dispose();
      vtkRef.current.dual?.dispose();
      vtkRef.current = { ...EMPTY_REFS };
    };
  }, []);

  // ── Subject A load ──────────────────────────────────────────────────────────
  useEffect(() => {
    const { dual } = vtkRef.current;
    if (!dual || !subjectAPayload) {
      setSubjectALoaded(false);
      return;
    }

    const { left } = dual;

    // Tear down previous left content.
    vtkRef.current.leftLodHandle?.dispose();
    if (vtkRef.current.leftVolumeBundle) {
      left.volumeRenderer.removeVolume(vtkRef.current.leftVolumeBundle.actor);
    }
    if (vtkRef.current.leftMprBundle) {
      left.axialRenderer.removeActor(vtkRef.current.leftMprBundle.axial.actor);
      left.coronalRenderer.removeActor(vtkRef.current.leftMprBundle.coronal.actor);
      left.sagittalRenderer.removeActor(vtkRef.current.leftMprBundle.sagittal.actor);
    }

    // Release camera lock before reseating left cameras.
    vtkRef.current.cameraCleanup?.();
    vtkRef.current.cameraCleanup = null;

    // Tear down right viewer — warped volume and raw B are now stale.
    vtkRef.current.rightLodHandle?.dispose();
    if (vtkRef.current.rightVolumeBundle) {
      dual.right.volumeRenderer.removeVolume(vtkRef.current.rightVolumeBundle.actor);
    }
    if (vtkRef.current.rightMprBundle) {
      dual.right.axialRenderer.removeActor(vtkRef.current.rightMprBundle.axial.actor);
      dual.right.coronalRenderer.removeActor(vtkRef.current.rightMprBundle.coronal.actor);
      dual.right.sagittalRenderer.removeActor(vtkRef.current.rightMprBundle.sagittal.actor);
    }

    // Tear down raw B actors from right pane.
    vtkRef.current.rawRightLodHandle?.dispose();
    if (vtkRef.current.rawRightVolumeBundle) {
      dual.right.volumeRenderer.removeVolume(vtkRef.current.rawRightVolumeBundle.actor);
    }
    if (vtkRef.current.rawRightMprBundle) {
      dual.right.axialRenderer.removeActor(vtkRef.current.rawRightMprBundle.axial.actor);
      dual.right.coronalRenderer.removeActor(vtkRef.current.rawRightMprBundle.coronal.actor);
      dual.right.sagittalRenderer.removeActor(vtkRef.current.rawRightMprBundle.sagittal.actor);
    }

    // Remove seg overlay if present.
    if (vtkRef.current.dualSegBundle) {
      const sb = vtkRef.current.dualSegBundle;
      dual.left.axialRenderer.removeActor(sb.leftAxial);
      dual.left.coronalRenderer.removeActor(sb.leftCoronal);
      dual.left.sagittalRenderer.removeActor(sb.leftSagittal);
      dual.right.axialRenderer.removeActor(sb.rightAxial);
      dual.right.coronalRenderer.removeActor(sb.rightCoronal);
      dual.right.sagittalRenderer.removeActor(sb.rightSagittal);
    }

    vtkRef.current.leftVolumeBundle     = null;
    vtkRef.current.leftMprBundle        = null;
    vtkRef.current.leftLodHandle        = null;
    vtkRef.current.rightVolumeBundle    = null;
    vtkRef.current.rightMprBundle       = null;
    vtkRef.current.rightLodHandle       = null;
    vtkRef.current.rawRightVolumeBundle = null;
    vtkRef.current.rawRightMprBundle    = null;
    vtkRef.current.rawRightLodHandle    = null;
    vtkRef.current.dualSegBundle        = null;

    setWarpedLoaded(false);
    setRawBLoaded(false);
    setSynCached(false);
    synCachedRef.current = false;
    setHasSegmentation(false);
    setVolumetrics(null);
    setSynStatus({ phase: 'idle' });
    setAlignmentModeState('raw');
    // Reset per-label visibility to all-visible for the new subject.
    setLabelVisibilityState(defaultLabelVisibility());

    try {
      const { dims } = subjectAPayload.header;

      const leftBundle = buildVolumeActor(subjectAPayload);
      const [lo, hi]   = leftBundle.dataRange;
      const autoCenter = Math.round((lo + hi) / 2);
      const autoWidth  = Math.round(hi - lo);

      left.volumeRenderer.addVolume(leftBundle.actor);

      const leftMpr = buildMprActors(leftBundle.imageData, dims, autoCenter, autoWidth);
      left.axialRenderer.addActor(leftMpr.axial.actor);
      left.coronalRenderer.addActor(leftMpr.coronal.actor);
      left.sagittalRenderer.addActor(leftMpr.sagittal.actor);

      const leftLod = attachLod(left.interactor, leftBundle.mapper, left.renderWindow);

      left.axialRenderer.resetCamera();
      left.coronalRenderer.resetCamera();
      left.sagittalRenderer.resetCamera();
      left.volumeRenderer.resetCamera();
      setupMprCameras({
        axial:    left.axialRenderer,
        coronal:  left.coronalRenderer,
        sagittal: left.sagittalRenderer,
      });
      left.renderWindow.render();

      const midK = Math.floor(dims[3] / 2);
      const midJ = Math.floor(dims[2] / 2);
      const midI = Math.floor(dims[1] / 2);

      setMaxK(dims[3]);
      setMaxJ(dims[2]);
      setMaxI(dims[1]);
      setSliceKState(midK);
      setSliceJState(midJ);
      setSliceIState(midI);
      setWindowCenterState(autoCenter);
      setWindowWidthState(autoWidth);

      vtkRef.current.leftVolumeBundle = leftBundle;
      vtkRef.current.leftMprBundle    = leftMpr;
      vtkRef.current.leftLodHandle    = leftLod;
      vtkRef.current.subjectAFile     = subjectAPayload.file ?? null;

      setSubjectALoaded(true);
    } catch (err) {
      console.error('[DualVolumetricViewer] Subject A load error:', err);
      setSubjectALoaded(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [subjectAPayload]);

  // ── Subject B file selection → parse in worker ──────────────────────────────
  // Whenever the user picks a new Subject B file (via the sidebar file picker),
  // kick off the NIfTI worker.  fileBRef is updated first so runSyn can still
  // upload the original File even after the ArrayBuffer has been transferred.
  useEffect(() => {
    if (!subjectBFile) return;
    fileBRef.current = subjectBFile;
    setRawBLoaded(false);
    void processBFile(subjectBFile);
  }, [subjectBFile, processBFile]);

  // ── Worker result → build raw B VTK actors + mount if mode is 'raw' ─────────
  // Fires when the NIfTI worker finishes parsing Subject B.
  // Builds the raw actor bundle once; subsequent mode switches add/remove actors
  // without rebuilding the underlying vtkImageData.
  useEffect(() => {
    const payload = workerB.volume;
    if (!payload) return;

    const { dual } = vtkRef.current;
    if (!dual) return;

    // Dispose any previous raw B bundle (e.g. user picks a second file).
    vtkRef.current.rawRightLodHandle?.dispose();
    if (vtkRef.current.rawRightVolumeBundle) {
      dual.right.volumeRenderer.removeVolume(vtkRef.current.rawRightVolumeBundle.actor);
    }
    if (vtkRef.current.rawRightMprBundle) {
      dual.right.axialRenderer.removeActor(vtkRef.current.rawRightMprBundle.axial.actor);
      dual.right.coronalRenderer.removeActor(vtkRef.current.rawRightMprBundle.coronal.actor);
      dual.right.sagittalRenderer.removeActor(vtkRef.current.rawRightMprBundle.sagittal.actor);
    }
    vtkRef.current.rawRightVolumeBundle = null;
    vtkRef.current.rawRightMprBundle    = null;
    vtkRef.current.rawRightLodHandle    = null;

    // Build raw B VTK pipeline — identical path to Subject A.
    const { dims } = payload.header;
    const rawBundle = buildVolumeActor(payload);
    const rawMpr    = buildMprActors(rawBundle.imageData, dims, windowCenter, windowWidth);
    const rawLod    = attachLod(dual.right.interactor, rawBundle.mapper, dual.right.renderWindow);

    // Sync to current slice positions so actors are already positioned correctly
    // when they land in the right renderer.
    rawMpr.axial.setSlice(sliceK);
    rawMpr.coronal.setSlice(sliceJ);
    rawMpr.sagittal.setSlice(sliceI);
    rawBundle.updateOpacity(volumeOpacity);

    vtkRef.current.rawRightVolumeBundle = rawBundle;
    vtkRef.current.rawRightMprBundle    = rawMpr;
    vtkRef.current.rawRightLodHandle    = rawLod;

    // Immediately mount in the right pane when mode is 'raw'.
    // (If mode is 'registered' the user will have to already be in registered mode
    // which isn't possible before rawBLoaded becomes true, so this branch is always taken
    // on first B load.)
    if (alignmentModeRef.current === 'raw') {
      dual.right.volumeRenderer.addVolume(rawBundle.actor);
      dual.right.axialRenderer.addActor(rawMpr.axial.actor);
      dual.right.coronalRenderer.addActor(rawMpr.coronal.actor);
      dual.right.sagittalRenderer.addActor(rawMpr.sagittal.actor);

      // Right cameras are independent in raw mode — reset to frame raw B.
      dual.right.volumeRenderer.resetCamera();
      dual.right.axialRenderer.resetCamera();
      dual.right.coronalRenderer.resetCamera();
      dual.right.sagittalRenderer.resetCamera();
      setupMprCameras({
        axial:    dual.right.axialRenderer,
        coronal:  dual.right.coronalRenderer,
        sagittal: dual.right.sagittalRenderer,
      });

      dual.right.renderWindow.render();
    }

    // Reset mode to 'raw' when a new B file is loaded (clears any previous SyN).
    setAlignmentModeState('raw');
    setSynCached(false);
    synCachedRef.current = false;
    setWarpedLoaded(false);
    setRawBLoaded(true);
  // sliceK/J/I, windowCenter/Width, volumeOpacity are captured at parse time;
  // subsequent slider moves are handled by the dedicated sync effects.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workerB.volume]);

  // ── Alignment mode effect — swaps right pane actors ─────────────────────────
  // Fires whenever the user clicks the AlignmentModeToggle.
  // Uses synCachedRef (not synCached state) to avoid extra firings on SyN completion.
  useEffect(() => {
    const { dual } = vtkRef.current;
    if (!dual) return;

    const rvb = vtkRef.current.rawRightVolumeBundle;
    const rmb = vtkRef.current.rawRightMprBundle;

    if (alignmentMode === 'raw') {
      // ── Switch to Raw Mode ────────────────────────────────────────────────

      // 1. Decouple cameras.
      vtkRef.current.cameraCleanup?.();
      vtkRef.current.cameraCleanup = null;

      // 2. Remove warped B actors from right pane (kept in refs as cache).
      if (vtkRef.current.rightVolumeBundle) {
        dual.right.volumeRenderer.removeVolume(vtkRef.current.rightVolumeBundle.actor);
      }
      if (vtkRef.current.rightMprBundle) {
        dual.right.axialRenderer.removeActor(vtkRef.current.rightMprBundle.axial.actor);
        dual.right.coronalRenderer.removeActor(vtkRef.current.rightMprBundle.coronal.actor);
        dual.right.sagittalRenderer.removeActor(vtkRef.current.rightMprBundle.sagittal.actor);
      }

      // 3. Remove seg overlays — only meaningful in registered (shared) space.
      if (vtkRef.current.dualSegBundle) {
        const sb = vtkRef.current.dualSegBundle;
        dual.left.axialRenderer.removeActor(sb.leftAxial);
        dual.left.coronalRenderer.removeActor(sb.leftCoronal);
        dual.left.sagittalRenderer.removeActor(sb.leftSagittal);
        dual.right.axialRenderer.removeActor(sb.rightAxial);
        dual.right.coronalRenderer.removeActor(sb.rightCoronal);
        dual.right.sagittalRenderer.removeActor(sb.rightSagittal);
      }

      // 4. Mount raw B (if available — it should be if rawBLoaded is true).
      if (rvb && rmb) {
        dual.right.volumeRenderer.addVolume(rvb.actor);
        dual.right.axialRenderer.addActor(rmb.axial.actor);
        dual.right.coronalRenderer.addActor(rmb.coronal.actor);
        dual.right.sagittalRenderer.addActor(rmb.sagittal.actor);

        // Sync raw B slices to the shared slider positions.
        rmb.axial.setSlice(sliceK);
        rmb.coronal.setSlice(sliceJ);
        rmb.sagittal.setSlice(sliceI);

        // Reset right cameras — now decoupled from left.
        dual.right.volumeRenderer.resetCamera();
        dual.right.axialRenderer.resetCamera();
        dual.right.coronalRenderer.resetCamera();
        dual.right.sagittalRenderer.resetCamera();
        setupMprCameras({
          axial:    dual.right.axialRenderer,
          coronal:  dual.right.coronalRenderer,
          sagittal: dual.right.sagittalRenderer,
        });
      }

      dual.left.renderWindow.render();
      dual.right.renderWindow.render();

    } else {
      // ── Switch to Registered Mode ─────────────────────────────────────────

      // 1. Remove raw B from right pane (kept in refs for fast restore).
      if (rvb && rmb) {
        dual.right.volumeRenderer.removeVolume(rvb.actor);
        dual.right.axialRenderer.removeActor(rmb.axial.actor);
        dual.right.coronalRenderer.removeActor(rmb.coronal.actor);
        dual.right.sagittalRenderer.removeActor(rmb.sagittal.actor);
        dual.right.renderWindow.render();
      }

      if (synCachedRef.current && vtkRef.current.rightVolumeBundle && vtkRef.current.rightMprBundle) {
        // 2a. Cache hit — instantly restore warped B actors.
        const wvb = vtkRef.current.rightVolumeBundle;
        const wmb = vtkRef.current.rightMprBundle;

        dual.right.volumeRenderer.addVolume(wvb.actor);
        dual.right.axialRenderer.addActor(wmb.axial.actor);
        dual.right.coronalRenderer.addActor(wmb.coronal.actor);
        dual.right.sagittalRenderer.addActor(wmb.sagittal.actor);

        // Sync warped B to current slice positions.
        wmb.axial.setSlice(sliceK);
        wmb.coronal.setSlice(sliceJ);
        wmb.sagittal.setSlice(sliceI);

        // Re-lock cameras (they were decoupled while in raw mode).
        const cleanup = lockCameras(dual.left, dual.right);
        vtkRef.current.cameraCleanup = cleanup;

        // Re-add seg overlays if SyN had returned them.
        if (vtkRef.current.dualSegBundle) {
          const sb = vtkRef.current.dualSegBundle;
          dual.left.axialRenderer.addActor(sb.leftAxial);
          dual.left.coronalRenderer.addActor(sb.leftCoronal);
          dual.left.sagittalRenderer.addActor(sb.leftSagittal);
          dual.right.axialRenderer.addActor(sb.rightAxial);
          dual.right.coronalRenderer.addActor(sb.rightCoronal);
          dual.right.sagittalRenderer.addActor(sb.rightSagittal);
          dual.left.renderWindow.render();
        }

        dual.right.renderWindow.render();

      } else {
        // 2b. No cache — trigger SyN registration.
        // runSynRef.current is a stable ref to the latest runSyn callback,
        // avoiding runSyn as a dep in this effect (which would cause extra firings).
        runSynRef.current();
      }
    }
  // Intentionally exclude sliceK/J/I — they have their own sync effect.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [alignmentMode]);

  // ── Stable ref to runSyn — avoids stale closure in alignmentMode effect ─────
  // runSyn itself is a useCallback that changes when subjectBFile / synStatus change.
  // We mirror it to a ref so the alignmentMode effect always calls the latest version
  // without having runSyn in its dep array.
  const runSynRef = useRef<() => void>(() => {});

  // ── Slice position sync → both viewers ─────────────────────────────────────
  useEffect(() => {
    const { dual, leftMprBundle, rightMprBundle, rawRightMprBundle, dualSegBundle } = vtkRef.current;
    if (!dual) return;
    // Left (Subject A).
    leftMprBundle?.axial.setSlice(sliceK);
    leftMprBundle?.coronal.setSlice(sliceJ);
    leftMprBundle?.sagittal.setSlice(sliceI);
    // Right warped (always updated even if not currently mounted — keeps actor in sync).
    rightMprBundle?.axial.setSlice(sliceK);
    rightMprBundle?.coronal.setSlice(sliceJ);
    rightMprBundle?.sagittal.setSlice(sliceI);
    // Right raw (same — updated regardless of current mode).
    rawRightMprBundle?.axial.setSlice(sliceK);
    rawRightMprBundle?.coronal.setSlice(sliceJ);
    rawRightMprBundle?.sagittal.setSlice(sliceI);
    // Seg overlay.
    dualSegBundle?.setSlices(sliceK, sliceJ, sliceI);
    dual.left.renderWindow.render();
    dual.right.renderWindow.render();
  }, [sliceK, sliceJ, sliceI]);

  // ── Window / level sync → both viewers ─────────────────────────────────────
  useEffect(() => {
    const { dual, leftMprBundle, leftVolumeBundle, rightMprBundle, rightVolumeBundle, rawRightMprBundle, rawRightVolumeBundle } = vtkRef.current;
    if (!dual) return;
    leftMprBundle?.axial.setWindowLevel(windowCenter, windowWidth);
    leftMprBundle?.coronal.setWindowLevel(windowCenter, windowWidth);
    leftMprBundle?.sagittal.setWindowLevel(windowCenter, windowWidth);
    leftVolumeBundle?.updateWindowLevel(windowCenter, windowWidth);
    rightMprBundle?.axial.setWindowLevel(windowCenter, windowWidth);
    rightMprBundle?.coronal.setWindowLevel(windowCenter, windowWidth);
    rightMprBundle?.sagittal.setWindowLevel(windowCenter, windowWidth);
    rightVolumeBundle?.updateWindowLevel(windowCenter, windowWidth);
    rawRightMprBundle?.axial.setWindowLevel(windowCenter, windowWidth);
    rawRightMprBundle?.coronal.setWindowLevel(windowCenter, windowWidth);
    rawRightMprBundle?.sagittal.setWindowLevel(windowCenter, windowWidth);
    rawRightVolumeBundle?.updateWindowLevel(windowCenter, windowWidth);
    dual.left.renderWindow.render();
    dual.right.renderWindow.render();
  }, [windowCenter, windowWidth]);

  // ── Volume opacity sync → both viewers ─────────────────────────────────────
  useEffect(() => {
    const { dual, leftVolumeBundle, rightVolumeBundle, rawRightVolumeBundle } = vtkRef.current;
    if (!dual) return;
    leftVolumeBundle?.updateOpacity(volumeOpacity);
    rightVolumeBundle?.updateOpacity(volumeOpacity);
    rawRightVolumeBundle?.updateOpacity(volumeOpacity);
    dual.left.renderWindow.render();
    dual.right.renderWindow.render();
  }, [volumeOpacity]);

  // ── Per-label visibility sync → seg overlays ────────────────────────────────
  // Fires whenever any label is toggled in DualAnatomySelector.
  // Uses updateLabelVisibility (per-structure) rather than updateGroupVisibility
  // (group-level) so individual structures can be shown/hidden independently.
  useEffect(() => {
    const { dual, dualSegBundle } = vtkRef.current;
    if (!dual || !dualSegBundle) return;
    dualSegBundle.updateLabelVisibility(labelVisibility);
    dual.left.renderWindow.render();
    dual.right.renderWindow.render();
  }, [labelVisibility]);

  // ── SyN registration ────────────────────────────────────────────────────────
  const runSyn = useCallback(async () => {
    const { dual, subjectAFile } = vtkRef.current;
    if (!dual || !subjectAFile || !fileBRef.current) return;
    if (synStatus.phase === 'uploading') return;

    const { right } = dual;
    const rvb = vtkRef.current.rawRightVolumeBundle;
    const rmb = vtkRef.current.rawRightMprBundle;

    // Remove raw B from right pane while registration runs.
    // The right pane goes blank with a spinner overlay (see render below).
    if (rvb) right.volumeRenderer.removeVolume(rvb.actor);
    if (rmb) {
      right.axialRenderer.removeActor(rmb.axial.actor);
      right.coronalRenderer.removeActor(rmb.coronal.actor);
      right.sagittalRenderer.removeActor(rmb.sagittal.actor);
    }
    right.renderWindow.render();

    // Remove previous warped bundle and seg overlays.
    vtkRef.current.cameraCleanup?.();
    vtkRef.current.cameraCleanup = null;
    vtkRef.current.rightLodHandle?.dispose();
    if (vtkRef.current.rightVolumeBundle) {
      right.volumeRenderer.removeVolume(vtkRef.current.rightVolumeBundle.actor);
    }
    if (vtkRef.current.rightMprBundle) {
      right.axialRenderer.removeActor(vtkRef.current.rightMprBundle.axial.actor);
      right.coronalRenderer.removeActor(vtkRef.current.rightMprBundle.coronal.actor);
      right.sagittalRenderer.removeActor(vtkRef.current.rightMprBundle.sagittal.actor);
    }
    if (vtkRef.current.dualSegBundle) {
      const sb = vtkRef.current.dualSegBundle;
      dual.left.axialRenderer.removeActor(sb.leftAxial);
      dual.left.coronalRenderer.removeActor(sb.leftCoronal);
      dual.left.sagittalRenderer.removeActor(sb.leftSagittal);
      right.axialRenderer.removeActor(sb.rightAxial);
      right.coronalRenderer.removeActor(sb.rightCoronal);
      right.sagittalRenderer.removeActor(sb.rightSagittal);
    }
    vtkRef.current.rightVolumeBundle = null;
    vtkRef.current.rightMprBundle    = null;
    vtkRef.current.rightLodHandle    = null;
    vtkRef.current.dualSegBundle     = null;
    setWarpedLoaded(false);
    setHasSegmentation(false);
    setVolumetrics(null);

    setSynStatus({ phase: 'uploading' });

    try {
      const result = await synRegistrationApi.register(subjectAFile, fileBRef.current);

      // Decode Fortran-order base64 float32 → Float32Array.
      const bin       = atob(result.warped);
      const byteArray = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) byteArray[i] = bin.charCodeAt(i);
      const warpedF32 = new Float32Array(byteArray.buffer);

      // Build warped B pipeline.
      const rightBundle = buildRawVolumeActor(
        warpedF32,
        result.dims as [number, number, number],
        result.affine,
      );

      right.volumeRenderer.addVolume(rightBundle.actor);

      const dims = [0, result.dims[0], result.dims[1], result.dims[2]];
      const rightMpr = buildMprActors(rightBundle.imageData, dims, windowCenter, windowWidth);
      right.axialRenderer.addActor(rightMpr.axial.actor);
      right.coronalRenderer.addActor(rightMpr.coronal.actor);
      right.sagittalRenderer.addActor(rightMpr.sagittal.actor);

      const rightLod = attachLod(right.interactor, rightBundle.mapper, right.renderWindow);

      // Lock cameras — right inherits left's already-positioned cameras.
      const cleanup = lockCameras(dual.left, dual.right);

      // Sync current slice positions into right MPR actors.
      rightMpr.axial.setSlice(sliceK);
      rightMpr.coronal.setSlice(sliceJ);
      rightMpr.sagittal.setSlice(sliceI);

      right.renderWindow.render();

      vtkRef.current.rightVolumeBundle = rightBundle;
      vtkRef.current.rightMprBundle    = rightMpr;
      vtkRef.current.rightLodHandle    = rightLod;
      vtkRef.current.cameraCleanup     = cleanup;

      // Optional SynthSeg seg overlay.
      if (result.seg_a && result.seg_b_warped) {
        const segA = base64ToUint8(result.seg_a);
        const segB = base64ToUint8(result.seg_b_warped);

        // Build overlay starting fully visible; labelVisibility is applied
        // immediately below so the user's current structure selection is preserved.
        const segBundle = buildDualSegOverlay(
          vtkRef.current.leftVolumeBundle!.imageData,
          segA, segB,
          result.dims as [number, number, number],
        );

        segBundle.setSlices(sliceK, sliceJ, sliceI);

        // Apply current per-label selection immediately so the overlay reflects
        // whatever the user had configured before re-running SyN.
        segBundle.updateLabelVisibility(labelVisibility);

        dual.left.axialRenderer.addActor(segBundle.leftAxial);
        dual.left.coronalRenderer.addActor(segBundle.leftCoronal);
        dual.left.sagittalRenderer.addActor(segBundle.leftSagittal);
        right.axialRenderer.addActor(segBundle.rightAxial);
        right.coronalRenderer.addActor(segBundle.rightCoronal);
        right.sagittalRenderer.addActor(segBundle.rightSagittal);

        dual.left.renderWindow.render();
        right.renderWindow.render();

        vtkRef.current.dualSegBundle = segBundle;
        setHasSegmentation(true);
      }

      setVolumetrics(result.volumetrics);

      // Mark warped bundle as cached — future mode switches are instant.
      synCachedRef.current = true;
      setSynCached(true);
      setWarpedLoaded(true);
      setSynStatus({ phase: 'done', durationMs: result.duration_ms });

    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error('[DualVolumetricViewer] SyN registration error:', err);
      setSynStatus({ phase: 'error', message: msg });

      // Restore raw B to the right pane so the user still sees something.
      if (rvb) right.volumeRenderer.addVolume(rvb.actor);
      if (rmb) {
        right.axialRenderer.addActor(rmb.axial.actor);
        right.coronalRenderer.addActor(rmb.coronal.actor);
        right.sagittalRenderer.addActor(rmb.sagittal.actor);
      }
      right.renderWindow.render();

      // Revert to raw mode — the toggle snaps back.
      setAlignmentModeState('raw');
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [subjectBFile, synStatus.phase, labelVisibility, sliceK, sliceJ, sliceI]);

  // Keep the runSynRef current so the alignmentMode effect always calls the
  // latest version without adding runSyn itself to its deps array.
  useEffect(() => {
    runSynRef.current = runSyn;
  }, [runSyn]);

  // ── Build context value ─────────────────────────────────────────────────────
  const contextValue: DualViewerContextValue = {
    subjectALoaded,
    subjectBFile,    setSubjectBFile,
    synStatus,       warpedLoaded,
    runSyn,
    alignmentMode,   setAlignmentMode,
    rawBLoaded,      synCached,
    sliceK, maxK, setSliceK,
    sliceJ, maxJ, setSliceJ,
    sliceI, maxI, setSliceI,
    windowCenter,  setWindowCenter,
    windowWidth,   setWindowWidth,
    volumeOpacity, setVolumeOpacity,
    hasSegmentation,
    tissueVisibility, setTissueVisibility,
    labelVisibility, macroGroupState,
    setLabelVisible, setGroupVisible, showAllLabels, hideAllLabels,
    volumetrics,
  };

  // ── Render ──────────────────────────────────────────────────────────────────
  const isRegistering = synStatus.phase === 'uploading';

  return (
    <DualViewerContext.Provider value={contextValue}>
      <div className="dual-viewer">

        {/* Two side-by-side VTK canvases */}
        <div className="dual-viewer__panes">

          {/* Left pane — Subject A (always reference) */}
          <div className="dual-viewer__pane">
            <span className="dual-viewer__pane-label">Subject A (Reference)</span>
            <div ref={leftContainerRef} className="dual-viewer__canvas" />
          </div>

          {/* Right pane — Subject B (raw or warped depending on mode) */}
          <div className="dual-viewer__pane">
            <span className="dual-viewer__pane-label">
              {alignmentMode === 'registered' ? 'Subject B (Warped)' : 'Subject B (Raw)'}
            </span>
            <div ref={rightContainerRef} className="dual-viewer__canvas dual-viewer__canvas--right" />

            {/* Empty hint before Subject B is picked */}
            {!rawBLoaded && !isRegistering && synStatus.phase === 'idle' && (
              <div className="dual-viewer__empty-hint">
                Select Subject B in the sidebar to compare
              </div>
            )}

            {/* Parsing spinner — Subject B NIfTI is being decompressed in worker */}
            {workerB.loading && (
              <div className="dual-viewer__loading-overlay">
                <div className="dual-viewer__loading-spinner" />
                <p className="dual-viewer__loading-text">Parsing Subject B…</p>
              </div>
            )}

            {/* Registration spinner — SyN backend call in progress */}
            {isRegistering && (
              <div className="dual-viewer__loading-overlay">
                <div className="dual-viewer__loading-spinner" />
                <p className="dual-viewer__loading-text">SyN registration in progress…</p>
                <p className="dual-viewer__loading-subtext">This may take 2–5 minutes</p>
              </div>
            )}

            {/* Worker parse error */}
            {workerB.error && !rawBLoaded && (
              <div className="dual-viewer__empty-hint dual-viewer__empty-hint--error">
                {workerB.error}
              </div>
            )}
          </div>
        </div>

        {/* Sidebar controls */}
        <DualViewerControls />
      </div>
    </DualViewerContext.Provider>
  );
};

export default DualVolumetricViewer;
