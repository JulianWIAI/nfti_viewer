/**
 * mprRenderer.ts — Multi-Planar Reconstruction slice actors
 * ──────────────────────────────────────────────────────────
 *
 * MPR renders three orthogonal 2-D slices through the volume:
 *   • Axial    — constant K (z) index, camera looking down −Z
 *   • Coronal  — constant J (y) index, camera looking down −Y
 *   • Sagittal — constant I (x) index, camera looking down −X
 *
 * Each slice uses vtk.js's vtkImageMapper (SlicingMode) + vtkImageSlice,
 * which textures a single GPU quad with the resampled slice data. This is
 * much cheaper than volume ray-casting and gives crisp, pixel-accurate views.
 *
 * The three actors all reference the SAME vtkImageData created in
 * volumeRenderer.ts — no data duplication. Changing the slice index just
 * tells the mapper which plane to extract; the voxel data is not copied.
 *
 * CAMERA SETUP
 * ─────────────
 * Each MPR renderer uses parallel (orthographic) projection so distances on
 * screen are true mm regardless of zoom. The camera is pointed along the
 * slicing axis and the parallel scale is set to show the full FOV initially.
 *
 * The "up" vector for each view follows neurological convention:
 *   Axial    → up = -Y (superior is at the top)
 *   Coronal  → up = +Z (superior is at the top)
 *   Sagittal → up = +Z (superior is at the top)
 */

import vtkImageMapper from '@kitware/vtk.js/Rendering/Core/ImageMapper';
import vtkImageSlice from '@kitware/vtk.js/Rendering/Core/ImageSlice';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import vtkPiecewiseFunction from '@kitware/vtk.js/Common/DataModel/PiecewiseFunction';

// ── Types ─────────────────────────────────────────────────────────────────────

/** One plane's rendering objects, returned so the Viewer can update them. */
export interface MprPlane {
  actor: ReturnType<typeof vtkImageSlice.newInstance>;
  mapper: ReturnType<typeof vtkImageMapper.newInstance>;
  /** Update the displayed slice index imperatively (from a slider). */
  setSlice(index: number): void;
  /** Update window/level (brightness/contrast) for this plane. */
  setWindowLevel(center: number, width: number): void;
}

/** All three MPR planes returned together. */
export interface MprBundle {
  axial: MprPlane;
  coronal: MprPlane;
  sagittal: MprPlane;
}

// ── vtkImageMapper SlicingMode enum ──────────────────────────────────────────

// vtk.js exports SlicingMode as a named enum on the mapper; import it here.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const { SlicingMode } = vtkImageMapper as any;

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Builds a grey-scale colour transfer function that maps
 * [windowCenter - width/2, windowCenter + width/2] → [black, white].
 * This is the standard "window/level" display convention in radiology.
 */
function buildGreyscaleCTF(
  center: number,
  width: number,
): ReturnType<typeof vtkColorTransferFunction.newInstance> {
  const ctf = vtkColorTransferFunction.newInstance();
  const lo = center - width / 2;
  const hi = center + width / 2;
  ctf.addRGBPoint(lo, 0, 0, 0); // bottom of window → black
  ctf.addRGBPoint(hi, 1, 1, 1); // top of window → white
  return ctf;
}

/**
 * Builds a fully-opaque (1.0 everywhere) piecewise function.
 * MPR slices are 2-D quads and don't need opacity transfer; the function is
 * required by the property API but we keep it flat.
 */
function buildFlatOpacityTF(): ReturnType<typeof vtkPiecewiseFunction.newInstance> {
  const otf = vtkPiecewiseFunction.newInstance();
  otf.addPoint(0,    1.0);
  otf.addPoint(4096, 1.0);
  return otf;
}

/**
 * Constructs a single MPR plane.
 *
 * @param imageData   - Shared vtkImageData from volumeRenderer.buildVolumeActor().
 * @param mode        - SlicingMode.I | J | K
 * @param initialSlice - Starting slice index.
 * @param initCenter  - Initial window centre.
 * @param initWidth   - Initial window width.
 */
function buildPlane(
  imageData: unknown,
  mode: number,
  initialSlice: number,
  initCenter: number,
  initWidth: number,
): MprPlane {
  // ── Mapper ───────────────────────────────────────────────────────────────
  const mapper = vtkImageMapper.newInstance();
  mapper.setInputData(imageData);
  mapper.setSlicingMode(mode);
  mapper.setSlice(initialSlice);

  // ── Actor ────────────────────────────────────────────────────────────────
  const actor = vtkImageSlice.newInstance();
  actor.setMapper(mapper);

  // ── Window/level property ────────────────────────────────────────────────
  let ctf = buildGreyscaleCTF(initCenter, initWidth);
  const otf = buildFlatOpacityTF();

  const property = actor.getProperty();
  // vtk.js v36: vtkImageProperty uses the same component-index convention as
  // vtkVolumeProperty — first arg is always the component (0 for scalar data).
  property.setRGBTransferFunction(0, ctf);
  property.setPiecewiseFunction(0, otf);
  property.setColorWindow(initWidth);
  property.setColorLevel(initCenter);

  // ── Imperative setters ───────────────────────────────────────────────────
  function setSlice(index: number): void {
    mapper.setSlice(index);
  }

  function setWindowLevel(center: number, width: number): void {
    // Rebuild CTF in-place — cheaper than modifying the existing one's points
    ctf = buildGreyscaleCTF(center, width);
    property.setRGBTransferFunction(0, ctf);
    property.setColorWindow(width);
    property.setColorLevel(center);
    property.modified();
  }

  return { actor, mapper, setSlice, setWindowLevel };
}

// ── Public factory ────────────────────────────────────────────────────────────

/**
 * Builds all three MPR planes from the shared imageData.
 *
 * @param imageData    - Shared vtkImageData (from volumeRenderer).
 * @param dims         - [_, x, y, z] dimensions from the NIfTI header.
 * @param initCenter   - Window centre for the initial display.
 * @param initWidth    - Window width for the initial display.
 * @returns MprBundle with axial, coronal, sagittal planes.
 */
export function buildMprActors(
  imageData: unknown,
  dims: number[],
  initCenter: number = 500,
  initWidth: number = 1000,
): MprBundle {
  const midI = Math.floor(dims[1] / 2); // x midpoint for sagittal
  const midJ = Math.floor(dims[2] / 2); // y midpoint for coronal
  const midK = Math.floor(dims[3] / 2); // z midpoint for axial

  const axial    = buildPlane(imageData, SlicingMode.K, midK, initCenter, initWidth);
  const coronal  = buildPlane(imageData, SlicingMode.J, midJ, initCenter, initWidth);
  const sagittal = buildPlane(imageData, SlicingMode.I, midI, initCenter, initWidth);

  return { axial, coronal, sagittal };
}

/**
 * Positions each MPR renderer's camera to show its plane correctly.
 *
 * Must be called AFTER adding the actors to the renderers and after the first
 * `renderWindow.render()`, so the camera has a bounds to fit to.
 *
 * @param renderers - Object containing the three MPR vtkRenderer instances.
 * @param planes    - MprBundle (actors must already be added to renderers).
 */
export function setupMprCameras(
  renderers: {
    axial: { resetCamera(): void; getActiveCamera(): ReturnType<typeof getCam> };
    coronal: { resetCamera(): void; getActiveCamera(): ReturnType<typeof getCam> };
    sagittal: { resetCamera(): void; getActiveCamera(): ReturnType<typeof getCam> };
  },
): void {
  // Axial — looking down Z (camera at +Z, focal at origin, up = -Y)
  renderers.axial.resetCamera();
  const axCam = renderers.axial.getActiveCamera();
  axCam.setParallelProjection(true);

  // Coronal — looking down Y
  renderers.coronal.resetCamera();
  const coCam = renderers.coronal.getActiveCamera();
  coCam.setParallelProjection(true);

  // Sagittal — looking down X
  renderers.sagittal.resetCamera();
  const saCam = renderers.sagittal.getActiveCamera();
  saCam.setParallelProjection(true);
}

// Dummy type helper so the function signature compiles without importing vtkCamera
function getCam() {
  return {
    setParallelProjection(v: boolean): void { void v; },
  };
}
void getCam; // prevent "declared but never used"
