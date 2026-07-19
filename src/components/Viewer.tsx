/**
 * Viewer.tsx — Main vtk.js rendering component
 * ─────────────────────────────────────────────
 *
 * Responsibilities:
 *   1. Own the DOM container that vtk.js binds its WebGL canvas to.
 *   2. Initialise the vtk.js context (vtkSetup.initVtk) on mount.
 *   3. React to `volume` prop changes: build imageData + actors, add to
 *      renderers, attach LOD, reset cameras, render once.
 *   4. React to `controls` prop changes: update slice positions, window/level,
 *      opacity, visibility — each in a targeted useEffect.
 *   5. Expose a stable `runSegmentation` callback to the parent (App.tsx).
 *   6. Handle resize via ResizeObserver.
 *   7. Tear everything down on unmount.
 *
 * DESIGN NOTES
 * ─────────────
 *  • All vtk.js objects are stored in a single mutable ref (`vtkRef`) so they
 *    don't trigger re-renders. React state is used only for things the UI needs
 *    to display (inferenceStatus).
 *  • The `volume` and `controls` props are the reactive inputs; vtk.js is the
 *    imperative output. useEffect bridges the two worlds.
 *  • We do NOT store the VolumeActorBundle or MprBundle in React state because
 *    vtk.js objects are not serialisable and would confuse React reconciliation.
 */

import {
  useEffect,
  useRef,
  useCallback,
  useState,
  type FC,
} from 'react';

// vtk.js modules
import { initVtk, resizeVtk, destroyVtk } from '../lib/vtk/vtkSetup';
import { buildVolumeActor, type VolumeActorBundle } from '../lib/vtk/volumeRenderer';
import { buildMprActors, setupMprCameras, type MprBundle } from '../lib/vtk/mprRenderer';
import { attachLod, type LodHandle } from '../lib/vtk/lodManager';
import type { VtkContext } from '../lib/vtk/vtkSetup';

// ONNX
import { inferenceEngine, type InferenceStatus } from '../lib/onnx/inferenceEngine';

// Shared types
import type { VolumePayload, ViewerControls } from '../types/nifti.types';

// ── Constants ─────────────────────────────────────────────────────────────────

/**
 * Path to the ONNX model file, relative to the public directory.
 * Change this if you place your model somewhere else.
 */
const MODEL_PATH = `${import.meta.env.BASE_URL}models/brain_seg.onnx`;

// ── Props ─────────────────────────────────────────────────────────────────────

interface ViewerProps {
  /** Parsed NIfTI payload from useNiftiWorker; null until a file is loaded. */
  volume: VolumePayload | null;
  /** Live control values from App.tsx (slices, W/L, opacity). */
  controls: ViewerControls;
  /** Called when inferenceStatus changes so the ControlPanel can display it. */
  onInferenceStatus: (s: InferenceStatus) => void;
}

// ── Mutable vtk state (not React state) ──────────────────────────────────────

interface VtkRefs {
  ctx: VtkContext | null;
  volumeBundle: VolumeActorBundle | null;
  mprBundle: MprBundle | null;
  lodHandle: LodHandle | null;
}

// ── Component ─────────────────────────────────────────────────────────────────

const Viewer: FC<ViewerProps> = ({ volume, controls, onInferenceStatus }) => {
  // The DOM element vtk.js binds to — we need its pixel dimensions at init time
  const containerRef = useRef<HTMLDivElement>(null);

  // Mutable vtk.js state — stored in a ref to avoid re-render churn
  const vtkRef = useRef<VtkRefs>({
    ctx: null,
    volumeBundle: null,
    mprBundle: null,
    lodHandle: null,
  });

  // Inference status is displayed in the ControlPanel — keep in React state
  const [, setInferenceStatus] = useState<InferenceStatus>({ phase: 'idle' });

  const handleInferenceStatus = useCallback(
    (s: InferenceStatus) => {
      setInferenceStatus(s);
      onInferenceStatus(s);
    },
    [onInferenceStatus],
  );

  // ── 1. vtk.js initialisation on mount ────────────────────────────────────
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const ctx = initVtk(container);
    vtkRef.current.ctx = ctx;

    // ── ResizeObserver — keeps the WebGL canvas in sync with CSS layout ──
    const resizeObserver = new ResizeObserver(() => {
      if (vtkRef.current.ctx) resizeVtk(vtkRef.current.ctx, container);
    });
    resizeObserver.observe(container);

    // ── Cleanup on unmount ───────────────────────────────────────────────
    return () => {
      resizeObserver.disconnect();
      vtkRef.current.lodHandle?.dispose();
      if (vtkRef.current.ctx) destroyVtk(vtkRef.current.ctx);
      vtkRef.current = { ctx: null, volumeBundle: null, mprBundle: null, lodHandle: null };
      inferenceEngine.dispose();
    };
  }, []); // run once on mount

  // ── 2. Volume load — rebuild actors when a new file is parsed ────────────
  useEffect(() => {
    const { ctx } = vtkRef.current;
    if (!ctx || !volume) return;

    // ── 2a. Clean up any previously loaded volume ────────────────────────
    if (vtkRef.current.lodHandle) {
      vtkRef.current.lodHandle.dispose();
      vtkRef.current.lodHandle = null;
    }
    if (vtkRef.current.volumeBundle) {
      ctx.volumeRenderer.removeVolume(vtkRef.current.volumeBundle.actor);
    }
    if (vtkRef.current.mprBundle) {
      ctx.axialRenderer.removeActor(vtkRef.current.mprBundle.axial.actor);
      ctx.coronalRenderer.removeActor(vtkRef.current.mprBundle.coronal.actor);
      ctx.sagittalRenderer.removeActor(vtkRef.current.mprBundle.sagittal.actor);
    }

    // ── 2b. Build volume rendering pipeline ──────────────────────────────
    const volumeBundle = buildVolumeActor(volume);
    ctx.volumeRenderer.addVolume(volumeBundle.actor);

    // ── 2c. Build MPR slice actors (share the same imageData) ─────────────
    const { dims } = volume.header;
    const mprBundle = buildMprActors(
      volumeBundle.imageData,
      dims,
      controls.windowCenter,
      controls.windowWidth,
    );
    ctx.axialRenderer.addActor(mprBundle.axial.actor);
    ctx.coronalRenderer.addActor(mprBundle.coronal.actor);
    ctx.sagittalRenderer.addActor(mprBundle.sagittal.actor);

    // ── 2d. Attach LOD to the 3-D volume mapper ───────────────────────────
    // Pass renderWindow explicitly — vtkRenderWindowInteractor's TS types in
    // vtk.js v36 don't expose getRenderWindow(), so we supply it here instead.
    const lodHandle = attachLod(ctx.interactor, volumeBundle.mapper, ctx.renderWindow);

    // ── 2e. Camera reset ──────────────────────────────────────────────────
    ctx.axialRenderer.resetCamera();
    ctx.coronalRenderer.resetCamera();
    ctx.sagittalRenderer.resetCamera();
    ctx.volumeRenderer.resetCamera();

    // Apply parallel projection to MPR cameras after reset
    setupMprCameras({
      axial:    ctx.axialRenderer,
      coronal:  ctx.coronalRenderer,
      sagittal: ctx.sagittalRenderer,
    });

    // ── 2f. Initial render ────────────────────────────────────────────────
    ctx.renderWindow.render();

    // Store refs for later updates
    vtkRef.current.volumeBundle = volumeBundle;
    vtkRef.current.mprBundle    = mprBundle;
    vtkRef.current.lodHandle    = lodHandle;
  }, [volume]); // re-run only when the volume changes

  // ── 3. Slice position updates ─────────────────────────────────────────────
  useEffect(() => {
    const { mprBundle, ctx } = vtkRef.current;
    if (!mprBundle || !ctx) return;

    mprBundle.axial.setSlice(controls.sliceK);
    mprBundle.coronal.setSlice(controls.sliceJ);
    mprBundle.sagittal.setSlice(controls.sliceI);
    ctx.renderWindow.render();
  }, [controls.sliceK, controls.sliceJ, controls.sliceI]);

  // ── 4. Window / level updates ─────────────────────────────────────────────
  useEffect(() => {
    const { mprBundle, volumeBundle, ctx } = vtkRef.current;
    if (!ctx) return;

    mprBundle?.axial.setWindowLevel(controls.windowCenter, controls.windowWidth);
    mprBundle?.coronal.setWindowLevel(controls.windowCenter, controls.windowWidth);
    mprBundle?.sagittal.setWindowLevel(controls.windowCenter, controls.windowWidth);
    volumeBundle?.updateWindowLevel(controls.windowCenter, controls.windowWidth);
    ctx.renderWindow.render();
  }, [controls.windowCenter, controls.windowWidth]);

  // ── 5. Opacity & visibility updates ──────────────────────────────────────
  useEffect(() => {
    const { volumeBundle, ctx } = vtkRef.current;
    if (!volumeBundle || !ctx) return;

    volumeBundle.updateOpacity(controls.volumeOpacity);
    volumeBundle.actor.setVisibility(controls.showVolume);
    ctx.renderWindow.render();
  }, [controls.volumeOpacity, controls.showVolume]);

  // ── 6. ONNX segmentation ──────────────────────────────────────────────────
  const runSegmentation = useCallback(async () => {
    if (!volume) return;

    try {
      // Load model (no-op if already cached)
      await inferenceEngine.loadModel(MODEL_PATH, handleInferenceStatus);
      // Run segmentation — this is the INTERCEPT POINT described in inferenceEngine.ts
      await inferenceEngine.segment(volume, handleInferenceStatus);
      // TODO: convert mask → vtkImageData overlay and add to MPR renderers
    } catch (err) {
      handleInferenceStatus({
        phase: 'error',
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }, [volume, handleInferenceStatus]);

  // Expose runSegmentation to App.tsx via the ref trick
  // (We don't need useImperativeHandle here because App.tsx calls a callback
  // prop — but we attach it to the DOM node's dataset as a shorthand.)
  // Instead, the cleanest approach: App.tsx passes `onRunSegmentation` prop
  // through ControlPanel, and ControlPanel calls it. We expose it via the
  // returned JSX (see the hidden button approach below).

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="viewer-root">
      {/* vtk.js canvas lives here — fills the remaining space */}
      <div className="vtk-container" ref={containerRef}>
        {/* Overlay labels for each quadrant (pointer-events: none so they
            don't interfere with vtk.js mouse events) */}
        <div className="quadrant-label quadrant-label--tl" aria-hidden="true">
          Axial
        </div>
        <div className="quadrant-label quadrant-label--tr" aria-hidden="true">
          Coronal
        </div>
        <div className="quadrant-label quadrant-label--bl" aria-hidden="true">
          Sagittal
        </div>
        <div className="quadrant-label quadrant-label--br" aria-hidden="true">
          3D Volume
        </div>

        {/* Visual dividers between the four quadrants */}
        <div className="panel-divider panel-divider--h" aria-hidden="true" />
        <div className="panel-divider panel-divider--v" aria-hidden="true" />

        {/* Empty-state message when no file is loaded */}
        {!volume && (
          <div className="empty-state">
            <p>Upload a NIfTI scan to begin</p>
          </div>
        )}
      </div>

      {/*
        Hidden button used as a stable imperative handle.
        App.tsx passes `onRunSegmentation` to ControlPanel which calls it,
        and ControlPanel calls the prop directly. The button here is a
        fallback ref entry point for keyboard / accessibility callers.
      */}
      <button
        style={{ display: 'none' }}
        id="vtk-run-segmentation"
        onClick={runSegmentation}
        aria-hidden="true"
      />
    </div>
  );
};

export default Viewer;

/**
 * Imperative trigger for the segmentation button in ControlPanel.
 * App.tsx creates a callback that calls this:
 *   const handleRunSeg = () => document.getElementById('vtk-run-segmentation')?.click();
 * This avoids prop-drilling a ref through three layers.
 */
export function triggerSegmentation(): void {
  (document.getElementById('vtk-run-segmentation') as HTMLButtonElement | null)?.click();
}
