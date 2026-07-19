/**
 * dualSegmentationOverlay.ts — Paired VTK overlay actors for comparative SynthSeg maps
 * ──────────────────────────────────────────────────────────────────────────────────────
 *
 * Builds SIX vtkImageSlice actors from two flat Uint8Array label maps:
 *   • Three actors for the LEFT viewer  (Subject A — reference, un-warped)
 *   • Three actors for the RIGHT viewer (Subject B — warped via SyN, NN-interpolated)
 *
 * Both label maps are in Subject A's voxel space after SyN registration, so they
 * share the same vtkImageData geometry (spacing, origin, direction).
 *
 * RGBA BAKING
 * ───────────
 * Identical strategy to segmentationOverlay.ts: 4-component RGBA scalars are baked
 * per-voxel from the 256-entry LUT so vtkImageSlice can composite the overlay over
 * the underlying MRI slice using standard GL_BLEND.  Background (label 0) is always
 * transparent (α = 0).  All other non-background labels are at α = 153 (≈60%).
 *
 * TISSUE GROUP VISIBILITY
 * ───────────────────────
 * updateGroupVisibility(vis) rebuilds BOTH imageData scalar arrays simultaneously
 * using buildGroupVisibilityLut() from tissueGroups.ts.  The caller only needs to
 * call renderWindow.render() on both viewers after this to see the change.
 *
 * INVARIANT
 * ─────────
 * The label arrays passed in must be in Fortran order (X-fastest), matching the
 * np.asfortranarray().tobytes(order='F') serialisation produced by the backend.
 */

import vtkImageData   from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkDataArray   from '@kitware/vtk.js/Common/Core/DataArray';
import vtkImageMapper from '@kitware/vtk.js/Rendering/Core/ImageMapper';
import vtkImageSlice  from '@kitware/vtk.js/Rendering/Core/ImageSlice';

import { LABEL_RGBA_LUT_BASE }         from './segmentationOverlay';
import {
  applyLutToLabels,
  buildGroupVisibilityLut,
  type TissueGroupVisibility,
} from './tissueGroups';
import { buildLabelVisibilityLut }     from './labelVisibility';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const { SlicingMode } = vtkImageMapper as any;

// ── Public interface ──────────────────────────────────────────────────────────

export interface DualSegBundle {
  /** Subject A overlay actors for the LEFT viewer — add to left renderers. */
  leftAxial:    ReturnType<typeof vtkImageSlice.newInstance>;
  leftCoronal:  ReturnType<typeof vtkImageSlice.newInstance>;
  leftSagittal: ReturnType<typeof vtkImageSlice.newInstance>;

  /** Subject B (warped) overlay actors for the RIGHT viewer — add to right renderers. */
  rightAxial:    ReturnType<typeof vtkImageSlice.newInstance>;
  rightCoronal:  ReturnType<typeof vtkImageSlice.newInstance>;
  rightSagittal: ReturnType<typeof vtkImageSlice.newInstance>;

  /**
   * Sync overlay slice positions with the current MPR slice controls.
   * Applied to all six actors — call before renderWindow.render().
   */
  setSlices(k: number, j: number, i: number): void;

  /**
   * Rebuild RGBA scalars for both imageData objects to hide/show tissue groups.
   * Uses buildGroupVisibilityLut (GM / WM / CSF group-level control).
   * The caller must still call renderWindow.render() on both windows to redraw.
   */
  updateGroupVisibility(vis: TissueGroupVisibility): void;

  /**
   * Rebuild RGBA scalars for both imageData objects using per-structure visibility.
   * Uses buildLabelVisibilityLut (individual FreeSurfer label control).
   * This is the fine-grained counterpart to updateGroupVisibility.
   * The caller must still call renderWindow.render() on both windows to redraw.
   */
  updateLabelVisibility(vis: Record<number, boolean>): void;
}

// ── Factory ───────────────────────────────────────────────────────────────────

/**
 * Build paired semi-transparent RGBA overlay actors for both dual-viewer panes.
 *
 * @param sourceImageData  The left viewer's MRI vtkImageData — copied for geometry
 *                         (spacing, origin, direction, dimensions).  Both overlay
 *                         imageData objects use the same geometry since both label
 *                         maps are in Subject A's voxel space.
 * @param leftLabels       Flat Uint8Array of FreeSurfer label IDs for Subject A,
 *                         Fortran order.
 * @param rightLabels      Flat Uint8Array of FreeSurfer label IDs for Subject B
 *                         warped, Fortran order — same shape as leftLabels.
 * @param dims             [X, Y, Z] voxel dimensions shared by both label maps.
 * @param initialVis       Optional: apply current tissue-group visibility immediately
 *                         so toggled-off groups don't flash on at construction time.
 */
export function buildDualSegOverlay(
  sourceImageData: unknown,
  leftLabels:      Uint8Array,
  rightLabels:     Uint8Array,
  dims:            [number, number, number],
  initialVis?:     TissueGroupVisibility,
): DualSegBundle {

  // ── 1. Compute initial RGBA arrays ────────────────────────────────────────
  const lut = initialVis
    ? buildGroupVisibilityLut(initialVis, LABEL_RGBA_LUT_BASE as Uint8Array)
    : (LABEL_RGBA_LUT_BASE as Uint8Array);

  const leftRgba  = applyLutToLabels(leftLabels,  lut);
  const rightRgba = applyLutToLabels(rightLabels, lut);

  // ── 2. Create two vtkImageData objects with shared geometry ───────────────
  // Both overlay volumes are in Subject A's voxel space — same geometry.
  const src = sourceImageData as ReturnType<typeof vtkImageData.newInstance>;

  const leftImageData  = makeImageData(src, dims, leftRgba);
  const rightImageData = makeImageData(src, dims, rightRgba);

  // ── 3. Build six overlay actors (three per viewer) ────────────────────────
  const midK = Math.floor(dims[2] / 2);
  const midJ = Math.floor(dims[1] / 2);
  const midI = Math.floor(dims[0] / 2);

  const leftAx  = buildSliceActor(leftImageData,  SlicingMode.K, midK);
  const leftCo  = buildSliceActor(leftImageData,  SlicingMode.J, midJ);
  const leftSa  = buildSliceActor(leftImageData,  SlicingMode.I, midI);
  const rightAx = buildSliceActor(rightImageData, SlicingMode.K, midK);
  const rightCo = buildSliceActor(rightImageData, SlicingMode.J, midJ);
  const rightSa = buildSliceActor(rightImageData, SlicingMode.I, midI);

  // ── 4. Slice sync helper ──────────────────────────────────────────────────
  function setSlices(k: number, j: number, i: number): void {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (leftAx .getMapper() as any).setSlice(k);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (leftCo .getMapper() as any).setSlice(j);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (leftSa .getMapper() as any).setSlice(i);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (rightAx.getMapper() as any).setSlice(k);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (rightCo.getMapper() as any).setSlice(j);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (rightSa.getMapper() as any).setSlice(i);
  }

  // ── 5. Tissue group visibility (group-level: GM / WM / CSF) ──────────────
  // Rebuilds both imageData scalar arrays so both sides update together.
  // Replaces the vtkDataArray object (not mutates in-place) to guarantee the
  // vtk.js dirty chain fires:  setScalars() → pointData.modified() → imageData.modified().
  function updateGroupVisibility(vis: TissueGroupVisibility): void {
    const newLut = buildGroupVisibilityLut(vis, LABEL_RGBA_LUT_BASE as Uint8Array);

    const newLeftRgba  = applyLutToLabels(leftLabels,  newLut);
    replaceScalars(leftImageData,  newLeftRgba);

    const newRightRgba = applyLutToLabels(rightLabels, newLut);
    replaceScalars(rightImageData, newRightRgba);
  }

  // ── 6. Per-label visibility (individual FreeSurfer structure control) ─────
  // Fine-grained counterpart to updateGroupVisibility: zeros alpha for specific
  // label IDs rather than entire tissue classes.
  function updateLabelVisibility(vis: Record<number, boolean>): void {
    const newLut = buildLabelVisibilityLut(vis, LABEL_RGBA_LUT_BASE as Uint8Array);

    const newLeftRgba  = applyLutToLabels(leftLabels,  newLut);
    replaceScalars(leftImageData,  newLeftRgba);

    const newRightRgba = applyLutToLabels(rightLabels, newLut);
    replaceScalars(rightImageData, newRightRgba);
  }

  return {
    leftAxial:    leftAx,
    leftCoronal:  leftCo,
    leftSagittal: leftSa,
    rightAxial:    rightAx,
    rightCoronal:  rightCo,
    rightSagittal: rightSa,
    setSlices,
    updateGroupVisibility,
    updateLabelVisibility,
  };
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Create a vtkImageData inheriting geometry from src, populated with RGBA scalars. */
function makeImageData(
  src:  ReturnType<typeof vtkImageData.newInstance>,
  dims: [number, number, number],
  rgba: Uint8Array,
): ReturnType<typeof vtkImageData.newInstance> {
  const imgData = vtkImageData.newInstance();
  imgData.setSpacing(src.getSpacing());
  imgData.setOrigin(src.getOrigin());
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  imgData.setDirection((src as any).getDirection());
  imgData.setDimensions(dims[0], dims[1], dims[2]);

  const scalars = vtkDataArray.newInstance({
    name:               'SynthSegRGBA',
    numberOfComponents: 4,
    values:             rgba,
  });
  imgData.getPointData().setScalars(scalars);
  return imgData;
}

/** Build one semi-transparent vtkImageSlice actor for a segmentation overlay. */
function buildSliceActor(
  imageData: ReturnType<typeof vtkImageData.newInstance>,
  mode:      number,
  slice:     number,
): ReturnType<typeof vtkImageSlice.newInstance> {
  const mapper = vtkImageMapper.newInstance();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const m = mapper as any;
  m.setInputData(imageData);
  m.setSlicingMode(mode);
  m.setSlice(slice);

  const actor = vtkImageSlice.newInstance();
  actor.setMapper(mapper);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const prop = actor.getProperty() as any;
  // 4-component RGBA path: component 3 is per-pixel alpha, not an independent channel.
  prop.setIndependentComponents(false);
  // Force translucent rendering pass (GL_BLEND on) so background voxels (α=0) are
  // transparent rather than black.  0.9999 is visually indistinguishable from 1.0.
  prop.setOpacity(0.9999);

  return actor;
}

/** Replace the vtkDataArray on an imageData to propagate dirty flag through the pipeline. */
function replaceScalars(
  imgData: ReturnType<typeof vtkImageData.newInstance>,
  rgba:    Uint8Array,
): void {
  const scalars = vtkDataArray.newInstance({
    name:               'SynthSegRGBA',
    numberOfComponents: 4,
    values:             rgba,
  });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (imgData as any).getPointData().setScalars(scalars);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (imgData as any).modified();
}
