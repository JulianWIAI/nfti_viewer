/**
 * vtkSetup.ts — vtk.js render-window initialisation
 * ──────────────────────────────────────────────────
 *
 * Creates ONE WebGL canvas with FOUR independent vtk.js Renderers laid out
 * as a 2×2 grid via normalised viewport coordinates:
 *
 *   ┌─────────────────┬─────────────────┐
 *   │  Axial  (K)     │  Coronal  (J)   │
 *   ├─────────────────┼─────────────────┤
 *   │  Sagittal (I)   │  3-D Volume     │
 *   └─────────────────┴─────────────────┘
 *
 * Each Renderer has its own camera. The Interactor is bound to the OpenGL
 * window; the LOD manager will hook into its start/end events separately.
 *
 * Why one canvas instead of four <div>s?
 *   Browsers cap WebGL contexts per page (usually 8–16). Using one context
 *   for all four views is more efficient and avoids hitting that limit when
 *   other browser extensions/tabs also consume contexts.
 *
 * Rendering profiles must be imported before creating any Renderers because
 * vtk.js uses a late-binding factory pattern: the profile side-effects
 * register the concrete WebGL implementations against the abstract keys.
 */

// ── Profile import (MUST come first) ────────────────────────────────────────
// 'All' registers implementations for Geometry, Volume, and ImageSlice/Glyph.
import '@kitware/vtk.js/Rendering/Profiles/All';

import vtkRenderWindow from '@kitware/vtk.js/Rendering/Core/RenderWindow';
import vtkRenderer from '@kitware/vtk.js/Rendering/Core/Renderer';
import vtkOpenGLRenderWindow from '@kitware/vtk.js/Rendering/OpenGL/RenderWindow';
import vtkRenderWindowInteractor from '@kitware/vtk.js/Rendering/Core/RenderWindowInteractor';
import vtkInteractorStyleTrackballCamera from '@kitware/vtk.js/Interaction/Style/InteractorStyleTrackballCamera';

// ── Public interface ─────────────────────────────────────────────────────────

/**
 * All vtk.js objects created during setup, returned as a bundle so callers
 * can pass individual pieces to the renderer and LOD modules.
 */
export interface VtkContext {
  /** The logical render window — owns all renderers. */
  renderWindow: ReturnType<typeof vtkRenderWindow.newInstance>;
  /** The WebGL surface bound to the DOM container. */
  openGLWindow: ReturnType<typeof vtkOpenGLRenderWindow.newInstance>;
  /** Mouse / keyboard interactor. */
  interactor: ReturnType<typeof vtkRenderWindowInteractor.newInstance>;
  /** Top-left quadrant — axial (K) slices. */
  axialRenderer: ReturnType<typeof vtkRenderer.newInstance>;
  /** Top-right quadrant — coronal (J) slices. */
  coronalRenderer: ReturnType<typeof vtkRenderer.newInstance>;
  /** Bottom-left quadrant — sagittal (I) slices. */
  sagittalRenderer: ReturnType<typeof vtkRenderer.newInstance>;
  /** Bottom-right quadrant — 3-D volume rendering. */
  volumeRenderer: ReturnType<typeof vtkRenderer.newInstance>;
}

// ── Colour constants ─────────────────────────────────────────────────────────

/** Dark background for all 2-D MPR panels (medical convention). */
const BG_MPR: [number, number, number] = [0.05, 0.05, 0.05];
/** Slightly different shade for the 3-D panel to distinguish it visually. */
const BG_VOL: [number, number, number] = [0.02, 0.02, 0.08];

// ── Factory ──────────────────────────────────────────────────────────────────

/**
 * Initialises the full vtk.js render stack and binds it to `container`.
 *
 * @param container - The DOM element that will host the WebGL canvas.
 *                    Should have a known pixel size before this is called.
 * @returns VtkContext with all renderers and the interactor.
 */
export function initVtk(container: HTMLDivElement): VtkContext {
  // ── 1. Logical render window ─────────────────────────────────────────────
  const renderWindow = vtkRenderWindow.newInstance();

  // ── 2. OpenGL surface (owns the <canvas> element) ────────────────────────
  const openGLWindow = vtkOpenGLRenderWindow.newInstance();
  openGLWindow.setContainer(container);
  openGLWindow.setSize(container.clientWidth, container.clientHeight);
  renderWindow.addView(openGLWindow);

  // ── 3. Interactor ────────────────────────────────────────────────────────
  const interactor = vtkRenderWindowInteractor.newInstance();
  interactor.setView(openGLWindow);
  interactor.initialize();
  interactor.bindEvents(container);

  // Default style for the 3-D viewport; LOD manager will hook into its events.
  const style = vtkInteractorStyleTrackballCamera.newInstance();
  interactor.setInteractorStyle(style);

  // ── 4. Renderers with viewport coordinates ───────────────────────────────
  // vtk.js viewport coordinates: [xMin, yMin, xMax, yMax] in normalised 0→1,
  // origin at BOTTOM-LEFT (OpenGL convention).
  //
  //   yMax=1.0 ┌──────────┬──────────┐
  //             │  Axial   │ Coronal  │
  //   yMid=0.5 ├──────────┼──────────┤
  //             │ Sagittal │  3-D     │
  //   yMin=0.0 └──────────┴──────────┘
  //           xMin=0.0  xMid=0.5  xMax=1.0

  const axialRenderer = vtkRenderer.newInstance({ background: BG_MPR });
  axialRenderer.setViewport(0.0, 0.5, 0.5, 1.0); // top-left

  const coronalRenderer = vtkRenderer.newInstance({ background: BG_MPR });
  coronalRenderer.setViewport(0.5, 0.5, 1.0, 1.0); // top-right

  const sagittalRenderer = vtkRenderer.newInstance({ background: BG_MPR });
  sagittalRenderer.setViewport(0.0, 0.0, 0.5, 0.5); // bottom-left

  const volumeRenderer = vtkRenderer.newInstance({ background: BG_VOL });
  volumeRenderer.setViewport(0.5, 0.0, 1.0, 0.5); // bottom-right

  renderWindow.addRenderer(axialRenderer);
  renderWindow.addRenderer(coronalRenderer);
  renderWindow.addRenderer(sagittalRenderer);
  renderWindow.addRenderer(volumeRenderer);

  return {
    renderWindow,
    openGLWindow,
    interactor,
    axialRenderer,
    coronalRenderer,
    sagittalRenderer,
    volumeRenderer,
  };
}

/**
 * Handles a container resize event.
 * Call this from a ResizeObserver callback to keep the WebGL surface in sync
 * with the DOM layout — vtk.js does not auto-resize on its own.
 */
export function resizeVtk(ctx: VtkContext, container: HTMLDivElement): void {
  ctx.openGLWindow.setSize(container.clientWidth, container.clientHeight);
  ctx.renderWindow.render();
}

/**
 * Tears down all vtk.js resources created by `initVtk`.
 * Call from a React useEffect cleanup to avoid WebGL context leaks when the
 * Viewer component unmounts.
 */
export function destroyVtk(ctx: VtkContext): void {
  ctx.interactor.unbindEvents();
  ctx.renderWindow.getRenderers().forEach((r: ReturnType<typeof vtkRenderer.newInstance>) => {
    r.getActors().forEach((a: unknown) => r.removeActor(a as Parameters<typeof r.removeActor>[0]));
    r.getVolumes().forEach((v: unknown) => r.removeVolume(v as Parameters<typeof r.removeVolume>[0]));
  });
  ctx.openGLWindow.delete();
  ctx.renderWindow.delete();
}
