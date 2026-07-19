/**
 * segmentationOverlay.ts — vtk.js overlay actors for SynthSeg label maps
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Builds three vtkImageSlice actors (axial / coronal / sagittal) from a flat
 * Uint8Array of FreeSurfer label IDs returned by the /api/segment backend.
 *
 * WHY 4-component RGBA scalars (not CTF + OTF)
 * ─────────────────────────────────────────────
 * vtkImageSlice / vtkImageMapper use the PiecewiseFunction (OTF) only in
 * volume-ray-cast rendering, NOT for MPR slice quads.  Per-pixel alpha must
 * therefore be baked directly into the scalar array as a 4th UNSIGNED_CHAR
 * component.  vtk.js reads 4-component Uint8 data as RGBA and composites it
 * with the underlying actor using standard alpha-blending.
 *
 *   Background (label 0) → A = 0   → fully transparent  → MRI shows through
 *   All other labels      → A = 153 → ≈ 60 % opaque      → coloured overlay
 *
 * Byte-order contract
 * ───────────────────
 * The flat array must be in Fortran order (X varies fastest), matching the
 * layout vtk.js uses for vtkImageData point scalars.  The backend serialises
 * with  np.asfortranarray(labels).tobytes()  to guarantee this.
 */

import vtkImageData   from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkDataArray   from '@kitware/vtk.js/Common/Core/DataArray';
import vtkImageMapper from '@kitware/vtk.js/Rendering/Core/ImageMapper';
import vtkImageSlice  from '@kitware/vtk.js/Rendering/Core/ImageSlice';
import { applyLutToLabels } from './tissueGroups';
import { buildLabelVisibilityLut } from './labelVisibility';

// ── FreeSurfer colour table for SynthSeg output ───────────────────────────────
// [labelId, R, G, B]  (R/G/B in 0–255)

const SYNTHSEG_COLORS: ReadonlyArray<[number, number, number, number]> = [
  [  2, 245, 245, 245],  // left cerebral white matter
  [  3, 205,  62,  78],  // left cerebral cortex
  [  4,  30, 118, 161],  // left lateral ventricle
  [  5, 101,  60, 128],  // left inferior lateral ventricle
  [  7, 220, 248, 164],  // left cerebellum white matter
  [  8, 231, 148,  34],  // left cerebellum cortex
  [ 10,   0, 118,  14],  // left thalamus
  [ 11, 122, 186, 220],  // left caudate
  [ 12, 236,  13, 176],  // left putamen
  [ 13,  12,  48, 255],  // left pallidum
  [ 14, 204, 182, 142],  // 3rd ventricle
  [ 15,  42, 204, 164],  // 4th ventricle
  [ 16, 119, 159, 176],  // brain stem
  [ 17, 220, 216,  20],  // left hippocampus
  [ 18, 103, 255, 255],  // left amygdala
  [ 24,  60,  60,  60],  // CSF
  [ 26, 255, 165,   0],  // left accumbens area
  [ 28, 165,  42,  42],  // left ventral DC
  [ 41, 200, 200, 200],  // right cerebral white matter
  [ 42, 237, 169,  18],  // right cerebral cortex
  [ 43,  25, 100, 140],  // right lateral ventricle
  [ 44,  90,  60, 110],  // right inferior lateral ventricle
  [ 46, 200, 230, 145],  // right cerebellum white matter
  [ 47, 206, 128,  38],  // right cerebellum cortex
  [ 49,   0, 100,  12],  // right thalamus
  [ 50, 100, 166, 200],  // right caudate
  [ 51, 214,  11, 154],  // right putamen
  [ 52,  10,  43, 235],  // right pallidum
  [ 53, 210, 206,  10],  // right hippocampus
  [ 54,  93, 245, 245],  // right amygdala
  [ 58, 255, 140,   0],  // right accumbens area
  [ 60, 150,  37,  37],  // right ventral DC
];

// Pre-built 256-entry RGBA lookup table indexed by FreeSurfer label ID.
// label 0 (background) stays at (0, 0, 0, 0) → fully transparent.
const LABEL_RGBA_LUT = new Uint8Array(256 * 4); // all zeros by default
for (const [id, r, g, b] of SYNTHSEG_COLORS) {
  LABEL_RGBA_LUT[id * 4    ] = r;
  LABEL_RGBA_LUT[id * 4 + 1] = g;
  LABEL_RGBA_LUT[id * 4 + 2] = b;
  LABEL_RGBA_LUT[id * 4 + 3] = 153; // ≈ 60 % opacity
}

/**
 * The base RGBA LUT before any tissue-group visibility is applied.
 * Exported so labelVisibility.buildLabelVisibilityLut() can clone and modify it
 * without mutating the original.
 */
export const LABEL_RGBA_LUT_BASE: Readonly<Uint8Array> = LABEL_RGBA_LUT;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const { SlicingMode } = vtkImageMapper as any;

// ── Public interface ──────────────────────────────────────────────────────────

export interface SegmentationBundle {
  axial:    ReturnType<typeof vtkImageSlice.newInstance>;
  coronal:  ReturnType<typeof vtkImageSlice.newInstance>;
  sagittal: ReturnType<typeof vtkImageSlice.newInstance>;
  /** Sync overlay slice positions with the current MPR slice controls. */
  setSlices(k: number, j: number, i: number): void;
  /**
   * Rebuild the RGBA scalar buffer to hide or show individual brain structures.
   * Labels with visibility[id] === false get alpha = 0 (fully transparent).
   * Calls setScalars() + vtkImageData.modified() automatically — the caller
   * only needs to trigger renderWindow.render() afterwards.
   */
  updateLabelVisibility(visibility: Record<number, boolean>): void;
  /**
   * The original flat uint8 label array in Fortran order (X-fastest).
   * Exposed so callers can send it to /api/mri/volumetrics without storing
   * a separate copy of the data.
   */
  readonly labelFlat: Uint8Array;
}

// ── Public factory ────────────────────────────────────────────────────────────

/**
 * Build three semi-transparent RGBA overlay actors from a SynthSeg label map.
 *
 * @param sourceImageData  The MRI vtkImageData — used only for geometry
 *                         (spacing, origin, direction, dimensions).
 * @param labelFlat        Flat Uint8Array of FreeSurfer label IDs,
 *                         serialised in Fortran order (X varies fastest).
 * @param dims             [x, y, z] dimensions returned by the backend.
 * @param initialVisibility  Optional: apply the current per-label toggle state
 *                           immediately at construction time so the overlay
 *                           respects whatever structures the user has hidden
 *                           before the segmentation finished.  When omitted,
 *                           all 32 non-background labels are visible.
 */
export function buildSegmentationOverlay(
  sourceImageData: unknown,
  labelFlat: Uint8Array,
  dims: [number, number, number],
  initialVisibility?: Record<number, boolean>,
): SegmentationBundle {

  // ── 1. Build per-voxel RGBA array ─────────────────────────────────────────
  // Apply the initial visibility filter if supplied, otherwise use the base LUT
  // (all 32 non-background FreeSurfer labels at α=153 ≈ 60 % opacity).
  const initialLut = initialVisibility
    ? buildLabelVisibilityLut(initialVisibility, LABEL_RGBA_LUT_BASE as Uint8Array)
    : (LABEL_RGBA_LUT_BASE as Uint8Array);
  const rgba = applyLutToLabels(labelFlat, initialLut);

  // ── 2. vtkImageData with the same geometry as the MRI ─────────────────────
  const labelData = vtkImageData.newInstance();
  const src = sourceImageData as ReturnType<typeof vtkImageData.newInstance>;
  labelData.setSpacing(src.getSpacing());
  labelData.setOrigin(src.getOrigin());
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  labelData.setDirection((src as any).getDirection());
  labelData.setDimensions(dims[0], dims[1], dims[2]);

  // Pass the Uint8Array directly — vtk.js infers UNSIGNED_CHAR from the
  // TypedArray constructor, same pattern as volumeRenderer.ts.
  const scalars = vtkDataArray.newInstance({
    name:               'SynthSegRGBA',
    numberOfComponents: 4,
    values:             rgba,
  });
  labelData.getPointData().setScalars(scalars);

  // ── 3. Three overlay actors ────────────────────────────────────────────────
  const midK = Math.floor(dims[2] / 2);
  const midJ = Math.floor(dims[1] / 2);
  const midI = Math.floor(dims[0] / 2);

  const axActor = buildOverlayActor(labelData, SlicingMode.K, midK);
  const coActor = buildOverlayActor(labelData, SlicingMode.J, midJ);
  const saActor = buildOverlayActor(labelData, SlicingMode.I, midI);

  function setSlices(k: number, j: number, i: number): void {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (axActor.getMapper() as any).setSlice(k);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (coActor.getMapper() as any).setSlice(j);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (saActor.getMapper() as any).setSlice(i);
  }

  /**
   * Rebuild the scalar buffer to reflect new per-label visibility.
   *
   * Replaces the vtkDataArray on the imageData's PointData each time rather
   * than mutating the existing buffer in-place.  In-place mutation via
   * getData().set() can fail silently in some vtk.js versions because the
   * mapper caches a reference to the old TypedArray and getData() may return
   * a stale view.  Replacing the array object guarantees vtk.js notices the
   * change through the full dirty chain:
   *   new vtkDataArray  →  setScalars()  →  pointData.modified()
   *   →  labelData.modified()  →  mapper re-uploads on next render.
   *
   * The caller must still call renderWindow.render() to see the change.
   */
  function updateLabelVisibility(visibility: Record<number, boolean>): void {
    // 1. Build a modified 256-entry RGBA LUT with hidden labels' alpha → 0.
    //    buildLabelVisibilityLut clones LABEL_RGBA_LUT_BASE (never mutates it).
    const lut = buildLabelVisibilityLut(visibility, LABEL_RGBA_LUT_BASE as Uint8Array);

    // 2. Apply per-voxel — O(n_voxels), new Uint8Array output.
    const newRgba = applyLutToLabels(labelFlat, lut);

    // 3. Replace the vtkDataArray to guarantee vtk.js sees the new bytes.
    //    setScalars() increments the PointData mtime; labelData.modified()
    //    propagates it up the pipeline so the mapper re-uploads the texture.
    const newScalars = vtkDataArray.newInstance({
      name:               'SynthSegRGBA',
      numberOfComponents: 4,
      values:             newRgba,
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (labelData as any).getPointData().setScalars(newScalars);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (labelData as any).modified();
  }

  return {
    axial: axActor, coronal: coActor, sagittal: saActor,
    setSlices,
    updateLabelVisibility,
    get labelFlat() { return labelFlat; },
  };
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function buildOverlayActor(
  imageData: ReturnType<typeof vtkImageData.newInstance>,
  mode:  number,
  slice: number,
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

  // Tell vtk.js the 4 components are NOT independent channels but a single
  // RGBA pixel — this activates the 4-component RGBA rendering path where
  // component 3 (A) controls per-pixel transparency.
  prop.setIndependentComponents(false);

  // vtkImageSlice.hasTranslucentPolygonalGeometry() returns (opacity < 1.0).
  // Without this, opacity = 1.0 (default) places the actor in the OPAQUE
  // rendering pass where GL_BLEND is disabled, so background voxels (A=0)
  // write black to the framebuffer instead of being transparent, and the two
  // coplanar actors Z-fight.  Setting 0.9999 forces the translucent pass
  // (GL_BLEND on, depth test LEQUAL) so the overlay correctly composites
  // over the MRI slice.  The visual difference from 1.0 is imperceptible.
  prop.setOpacity(0.9999);

  return actor;
}
