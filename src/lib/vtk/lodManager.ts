/**
 * lodManager.ts — Level-of-Detail (LOD) strategy for vtk.js
 * ───────────────────────────────────────────────────────────
 *
 * CONCEPT
 * ───────
 * Volume rendering (ray-casting) is expensive: every pixel in the 3-D viewport
 * must march a ray through hundreds of voxels, evaluate transfer functions, and
 * accumulate colour+opacity. At full quality this can take 50–200 ms per frame
 * on integrated GPU hardware — far too slow for interactive rotation.
 *
 * vtk.js's volume mapper exposes `sampleDistance`, which controls how far apart
 * the ray-marching samples are (in world-space mm). A larger value = fewer
 * samples per ray = faster rendering but coarser result.
 *
 * We exploit TWO complementary vtk.js mechanisms:
 *
 * 1. Interactor update rates
 *    `interactor.setDesiredUpdateRate(N)` — frames/s the render window tries to
 *    hit during active interaction (mouse drag). vtk.js internally scales the
 *    mapper quality to meet this budget.
 *    `interactor.setStillUpdateRate(N)` — frames/s when the user has stopped.
 *    Setting this very low (e.g. 0.1 fps) means vtk.js renders one high-quality
 *    frame after the interaction ends instead of churning.
 *
 * 2. Explicit sample-distance switching
 *    We additionally register start/end interaction callbacks to manually set a
 *    coarse sample distance during dragging and restore the fine one once the
 *    mouse is released. This gives deterministic quality transitions regardless
 *    of GPU speed.
 *
 * INTERACTION ←→ CAMERA
 * ─────────────────────
 * The interactor fires:
 *   • StartInteractionEvent — on mousedown / touchstart
 *   • EndInteractionEvent   — on mouseup / touchend
 *
 * These are NOT camera events; they are interactor-level events. The camera
 * transform (pan/rotate/zoom) is applied by the interactor style BETWEEN those
 * two events. The LOD manager reacts to the bookend events, so the quality drop
 * is immediate on grab and the quality restore happens the first frame the scene
 * is static.
 *
 * ONNX INTERCEPT POINT (for documentation)
 * ─────────────────────────────────────────
 * The ONNX inference pipeline does NOT interact with vtk.js rendering quality.
 * It receives the raw VolumePayload (the same ArrayBuffer the worker returned)
 * and runs asynchronously. Its output (a segmentation mask) is later fed back
 * into vtk.js as a second vtkImageData overlaid on the slices, at full quality,
 * after inference is complete.
 */

// ── Types ─────────────────────────────────────────────────────────────────────

/** Subset of vtkVolumeMapper we actually call in this module. */
interface MapperLOD {
  setSampleDistance(d: number): void;
  setAutoAdjustSampleDistances(v: boolean): void;
  setMaximumSamplesPerRay(n: number): void;
}

/** Subset of vtkRenderWindowInteractor we call here. */
interface InteractorLOD {
  setDesiredUpdateRate(fps: number): void;
  setStillUpdateRate(fps: number): void;
  // vtk.js macro registers events as "StartInteraction"/"EndInteraction" →
  // generates onStartInteraction / onEndInteraction (NOT the *Event suffix the
  // .d.ts incorrectly declares).
  onStartInteraction(cb: () => void): { unsubscribe(): void };
  onEndInteraction(cb: () => void): { unsubscribe(): void };
}

/** Subset of vtkRenderWindow we need to trigger re-renders. */
interface RenderWindowLOD {
  render(): void;
}

// ── Configuration ─────────────────────────────────────────────────────────────

/**
 * Tunable LOD parameters. Tweak these based on your target hardware.
 *
 * A sampleDistance of ~2 mm gives ~4–8× speedup over 0.5 mm on a typical
 * 1 mm-isotropic MRI brain scan. The quality is noticeably coarser but
 * perfectly acceptable for real-time rotation feedback.
 */
const LOD_CONFIG = {
  /** Sample distance during active interaction (coarse, fast). */
  interactSampleDist: 2.0,
  /** Sample distance when idle (fine, slow — rendered once). */
  stillSampleDist: 0.5,
  /** Target frame rate during interaction. vtk.js uses this as a budget. */
  desiredFps: 15,
  /** Idle update rate: 0.5 fps means one high-quality render every 2 s. */
  stillFps: 0.5,
  /** Cap rays to avoid extreme overdraw on very thick volumes. */
  maxSamplesPerRay: 1000,
} as const;

// ── LOD subscription handle ───────────────────────────────────────────────────

/** Returned by `attachLod` so the caller can clean up subscriptions. */
export interface LodHandle {
  /** Removes the start/end event listeners from the interactor. */
  dispose(): void;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Attaches LOD behaviour to `mapper` via `interactor` events.
 *
 * @param interactor   - The render-window interactor from VtkContext.
 * @param mapper       - The volume mapper whose sample distance will be switched.
 * @param renderWindow - The render window to call render() on after quality restore.
 *                       Passed explicitly because vtkRenderWindowInteractor does not
 *                       expose getRenderWindow() in its public TypeScript API (v36).
 * @returns            - A handle with a dispose() method for cleanup.
 */
export function attachLod(
  // vtk.js .d.ts does not declare onStartInteraction / onEndInteraction even
  // though the macro system generates them at runtime; accept unknown and cast.
  interactor: unknown,
  mapper: MapperLOD,
  renderWindow: RenderWindowLOD,
): LodHandle {
  const lod = interactor as InteractorLOD;
  // ── Mapper baseline setup ────────────────────────────────────────────────
  // Disable vtk.js's own automatic adjustment so our manual switching takes
  // full effect. If autoAdjust is on, vtk.js can re-override our values.
  mapper.setAutoAdjustSampleDistances(false);
  mapper.setMaximumSamplesPerRay(LOD_CONFIG.maxSamplesPerRay);
  mapper.setSampleDistance(LOD_CONFIG.stillSampleDist); // start at full quality

  // ── Interactor update-rate budget ────────────────────────────────────────
  // The interactor uses these to decide whether to skip renders while the
  // mouse is moving. Together with our explicit sample-distance switching,
  // this gives a double guarantee: vtk.js won't over-render and the mapper
  // quality will be coarse during the drag.
  lod.setDesiredUpdateRate(LOD_CONFIG.desiredFps);
  lod.setStillUpdateRate(LOD_CONFIG.stillFps);

  // ── Event subscriptions ──────────────────────────────────────────────────
  const startSub = lod.onStartInteraction(() => {
    mapper.setSampleDistance(LOD_CONFIG.interactSampleDist);
  });

  const endSub = lod.onEndInteraction(() => {
    mapper.setSampleDistance(LOD_CONFIG.stillSampleDist);
    renderWindow.render();
  });

  return {
    dispose() {
      startSub.unsubscribe();
      endSub.unsubscribe();
      // Restore safe defaults in case the mapper outlives this LOD session
      mapper.setAutoAdjustSampleDistances(true);
      mapper.setSampleDistance(LOD_CONFIG.stillSampleDist);
    },
  };
}
