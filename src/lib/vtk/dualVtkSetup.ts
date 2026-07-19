/**
 * dualVtkSetup.ts — Dual VTK context initialisation and camera locking
 * ──────────────────────────────────────────────────────────────────────
 *
 * Creates TWO independent vtk.js render windows (left + right), each with
 * four viewports (axial / coronal / sagittal / 3-D), then wires them together
 * so that any camera movement in either window is immediately mirrored in the
 * other.
 *
 * WHY TWO SEPARATE RENDER WINDOWS?
 * ─────────────────────────────────
 * A dual-viewport side-by-side comparison needs two independent canvases bound
 * to two separate DOM containers (left pane / right pane).  Each canvas gets
 * its own WebGL surface via vtkOpenGLRenderWindow and its own event binding via
 * vtkRenderWindowInteractor.  Camera state is shared via shared camera object
 * references — not copied — so there is no per-frame synchronisation cost.
 *
 * CAMERA LOCKING MECHANISM
 * ─────────────────────────
 * After both volumes are loaded (call lockCameras() then):
 *
 *   1. All four of the RIGHT viewer's renderers are set to use the LEFT
 *      viewer's active camera object via:
 *        rightRenderer.setActiveCamera(leftRenderer.getActiveCamera())
 *
 *   2. Because the same object is referenced by both renderers, any
 *      interaction that modifies the camera (rotation, zoom, pan) is
 *      instantly reflected in both viewers' projection matrices.
 *
 *   3. The remaining problem is triggering renders on BOTH windows.
 *      vtk.js fires an `Animation` event from the interactor every
 *      animation frame during active interaction.  We subscribe to this
 *      on each interactor and call render on the OTHER window, producing
 *      frame-accurate synchronisation:
 *
 *        left.interactor.onAnimation  → right.renderWindow.render()
 *        right.interactor.onAnimation → left.renderWindow.render()
 *
 *   4. lockCameras() returns a cleanup function that calls unsubscribe()
 *      on both Animation subscriptions when the dual viewer unmounts.
 *
 * WEBGL CONTEXT BUDGET
 * ─────────────────────
 * Each initVtk() call creates ONE WebGL context (4 viewports share one canvas).
 * Total for the dual viewer: 2 WebGL contexts.  Browsers typically allow 8–16
 * contexts per page, so two is well within the budget even if other tabs are open.
 */

import { initVtk, destroyVtk } from './vtkSetup';
import type { VtkContext } from './vtkSetup';

// ── Private interactor interface ──────────────────────────────────────────────
// vtk.js generates `onAnimation / onStartInteraction / onEndInteraction`
// at runtime via the macro system. The TypeScript .d.ts incorrectly lists them
// with an "Event" suffix; we define our own interface and cast explicitly.
// See also lodManager.ts which uses the same pattern for `onStartInteraction`.

interface SyncableInteractor {
  /** Fires every animation frame while the interactor is active (drag). */
  onAnimation(cb: () => void): { unsubscribe(): void };
  /** Fires once when the user ends mouse interaction (mouseup / touchend). */
  onEndInteraction(cb: () => void): { unsubscribe(): void };
}

// ── Public interface ──────────────────────────────────────────────────────────

/** Both VTK contexts created by initDualVtk(). */
export interface DualVtkContext {
  /** Left viewer — Subject A (reference). */
  left:  VtkContext;
  /** Right viewer — Subject B (warped). */
  right: VtkContext;
  /**
   * Dispose both VTK render windows and remove all event bindings.
   * Call from the React useEffect cleanup when DualVolumetricViewer unmounts.
   */
  dispose(): void;
}

// ── Factory ──────────────────────────────────────────────────────────────────

/**
 * Initialise two independent vtk.js render windows and bind them to the
 * provided DOM containers.
 *
 * @param leftContainer  DOM element for the left (Subject A) canvas.
 * @param rightContainer DOM element for the right (Subject B) canvas.
 * @returns DualVtkContext with left + right VtkContexts and a dispose() fn.
 */
export function initDualVtk(
  leftContainer:  HTMLDivElement,
  rightContainer: HTMLDivElement,
): DualVtkContext {
  const left  = initVtk(leftContainer);
  const right = initVtk(rightContainer);

  function dispose(): void {
    destroyVtk(left);
    destroyVtk(right);
  }

  return { left, right, dispose };
}

// ── Camera locking ────────────────────────────────────────────────────────────

/**
 * Share the left viewer's cameras with the right viewer and synchronise renders.
 *
 * MUST be called after both volumes are loaded and both viewers' cameras have
 * been positioned via resetCamera() + setupMprCameras().  The right viewer's
 * camera is REPLACED by the left viewer's camera object at this point —
 * therefore resetCamera() must NOT be called on the right viewer afterwards,
 * as it would re-position the shared camera.
 *
 * @param left  The left VtkContext (Subject A, already positioned).
 * @param right The right VtkContext (Subject B, just loaded — no resetCamera yet).
 * @returns Cleanup function — call it on unmount or before re-locking.
 */
export function lockCameras(left: VtkContext, right: VtkContext): () => void {
  // ── 1. Inject left's cameras into right's renderers ───────────────────────
  // After this, both renderers reference the SAME camera object in memory.
  // Rotating/zooming either window modifies the single shared object, so the
  // next render on either window will reflect the updated transform.
  right.axialRenderer   .setActiveCamera(left.axialRenderer   .getActiveCamera());
  right.coronalRenderer .setActiveCamera(left.coronalRenderer .getActiveCamera());
  right.sagittalRenderer.setActiveCamera(left.sagittalRenderer.getActiveCamera());
  right.volumeRenderer  .setActiveCamera(left.volumeRenderer  .getActiveCamera());

  // ── 2. Frame-accurate render synchronisation ──────────────────────────────
  // The vtk.js interactor fires `Animation` every requestAnimationFrame during
  // active interaction (mouse drag, touchmove).  We subscribe to each interactor
  // and immediately render the OTHER window so both views update in lock-step.
  //
  // The `syncing` flag prevents the pathological case where render() on the
  // other window internally causes another interaction event.
  let syncing = false;

  const leftInteractor  = left.interactor  as unknown as SyncableInteractor;
  const rightInteractor = right.interactor as unknown as SyncableInteractor;

  // Left interactor fires → render right
  const subLeftAnim = leftInteractor.onAnimation(() => {
    if (syncing) return;
    syncing = true;
    right.renderWindow.render();
    syncing = false;
  });

  // Right interactor fires → render left
  const subRightAnim = rightInteractor.onAnimation(() => {
    if (syncing) return;
    syncing = true;
    left.renderWindow.render();
    syncing = false;
  });

  // EndInteraction ensures a final high-quality render on both windows after
  // the drag ends (vtk.js drops to stillUpdateRate after interaction ends).
  const subLeftEnd = leftInteractor.onEndInteraction(() => {
    right.renderWindow.render();
  });

  const subRightEnd = rightInteractor.onEndInteraction(() => {
    left.renderWindow.render();
  });

  // ── Cleanup ───────────────────────────────────────────────────────────────
  return () => {
    subLeftAnim .unsubscribe();
    subRightAnim.unsubscribe();
    subLeftEnd  .unsubscribe();
    subRightEnd .unsubscribe();
  };
}
