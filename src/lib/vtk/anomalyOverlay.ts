/**
 * anomalyOverlay.ts — vtk.js overlay actors for binary anomaly masks
 * ────────────────────────────────────────────────────────────────────
 *
 * Builds four actors from a flat binary Uint8Array {0, 1} returned by the
 * /api/anomalies/detect endpoint:
 *
 *   axial / coronal / sagittal  — vtkImageSlice RGBA quads for MPR views
 *   volume3d                    — vtkVolume with CTF + OTF for the 3-D viewport
 *
 * WHY TWO SEPARATE vtkImageData OBJECTS
 * ──────────────────────────────────────
 * vtkImageSlice requires 4-component UNSIGNED_CHAR RGBA scalars so that the
 * alpha channel controls per-pixel transparency on each MPR quad.
 * vtkVolumeMapper requires 1-component scalars for CTF / OTF ray-casting.
 * Both share the same spatial geometry (spacing, origin, direction) cloned
 * from the MRI imageData, but carry different scalar payloads.
 *
 * OPACITY CONTROL — NO RGBA REBUILD ON SLIDER DRAG
 * ──────────────────────────────────────────────────
 * Per-voxel RGBA alpha is baked once at construction time (anomaly voxels
 * get A = ANOMALY_ALPHA; background voxels get A = 0).  Runtime opacity
 * changes from the slider are applied via:
 *
 *   MPR slices: actor.getProperty().setOpacity(factor)
 *     → scales the full actor uniformly — the GL compositor multiplies the
 *       per-pixel A value by the actor opacity, so the combined alpha for an
 *       anomaly voxel is  ANOMALY_ALPHA/255 × factor.
 *     → No new Uint8Array allocated; no vtkDataArray replacement.
 *     → O(1) cost — safe to call on every slider onChange event.
 *
 *   3-D volume: rebuild OTF (2 points only) + volProp.modified()
 *     → Trivially cheap.
 *
 * BYTE ORDER CONTRACT
 * ─────────────────────
 * maskFlat must be in Fortran order (X varies fastest), exactly as serialised
 * by the backend's  np.asfortranarray(mask.astype(np.uint8)).tobytes(order='F').
 * vtk.js internally stores vtkImageData point scalars in this same X-fastest
 * layout, so no transposition is needed on the JavaScript side.
 */

import vtkImageData             from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkDataArray             from '@kitware/vtk.js/Common/Core/DataArray';
import vtkImageMapper           from '@kitware/vtk.js/Rendering/Core/ImageMapper';
import vtkImageSlice            from '@kitware/vtk.js/Rendering/Core/ImageSlice';
import vtkVolume                from '@kitware/vtk.js/Rendering/Core/Volume';
import vtkVolumeMapper          from '@kitware/vtk.js/Rendering/Core/VolumeMapper';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import vtkPiecewiseFunction     from '@kitware/vtk.js/Common/DataModel/PiecewiseFunction';

// SlicingMode constants (K = axial, J = coronal, I = sagittal).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const { SlicingMode } = vtkImageMapper as any;

// ── Anomaly colour: bright red ────────────────────────────────────────────────
// Per-voxel RGBA for anomalous voxels.  The per-pixel alpha (ANOMALY_ALPHA)
// sets the ceiling; the slider multiplies it via actor property opacity.
const ANOMALY_R     = 255;
const ANOMALY_G     = 0;
const ANOMALY_B     = 0;
const ANOMALY_ALPHA = 220;   // ≈ 86 % when actor opacity = 1.0

// Peak opacity for the 3-D ray-cast volume at factor = 1.0.
// Lower than 1.0 to avoid completely occluding the underlying MRI volume.
const MAX_VOLUME_OPACITY = 0.80;

// ── Public interface ──────────────────────────────────────────────────────────

export interface AnomalyBundle {
  /** RGBA overlay quad for the axial (K-slice) viewport. */
  axial:    ReturnType<typeof vtkImageSlice.newInstance>;
  /** RGBA overlay quad for the coronal (J-slice) viewport. */
  coronal:  ReturnType<typeof vtkImageSlice.newInstance>;
  /** RGBA overlay quad for the sagittal (I-slice) viewport. */
  sagittal: ReturnType<typeof vtkImageSlice.newInstance>;
  /** Full 3-D ray-cast volume for the volumetric viewport. */
  volume3d: ReturnType<typeof vtkVolume.newInstance>;

  /**
   * Sync overlay slice positions with the MPR controls.
   * Call after any sliceK / sliceJ / sliceI change and trigger a render.
   */
  setSlices(k: number, j: number, i: number): void;

  /**
   * Set global overlay opacity.  factor ∈ [0, 1].
   *
   * For MPR slices:  actor property opacity is updated in O(1) — no RGBA
   *   buffer is rebuilt.
   * For 3-D volume:  the opacity transfer function is rebuilt (2 addPoint calls).
   *
   * The caller must trigger renderWindow.render() after this call.
   */
  setOpacity(factor: number): void;

  /** Count of anomalous voxels (mask = 1) detected by the model. */
  readonly nAnomaly: number;
}

// ── Factory function ──────────────────────────────────────────────────────────

/**
 * Build a four-actor anomaly overlay from a flat binary mask.
 *
 * @param sourceImageData  MRI vtkImageData — only its geometry is used
 *                         (spacing, origin, direction, dims).
 * @param maskFlat         Flat Uint8Array {0, 1} in Fortran order (X-fastest).
 * @param dims             [X, Y, Z] voxel dimensions matching maskFlat.
 * @param initialOpacity   Starting opacity factor ∈ [0, 1].  Defaults to 0.8.
 */
export function buildAnomalyOverlay(
  sourceImageData: unknown,
  maskFlat:        Uint8Array,
  dims:            [number, number, number],
  initialOpacity   = 0.8,
): AnomalyBundle {

  const [nx, ny, nz] = dims;
  const nVox = nx * ny * nz;

  // ── 1. Build 4-component RGBA buffer for MPR slices ──────────────────────
  // One pass over the flat mask: anomaly voxels get full red with ANOMALY_ALPHA;
  // background voxels stay (0, 0, 0, 0) — Uint8Array is zero-initialised.
  const rgba = new Uint8Array(nVox * 4);
  let nAnomaly = 0;
  for (let i = 0; i < nVox; i++) {
    if (maskFlat[i] === 1) {
      rgba[i * 4    ] = ANOMALY_R;
      rgba[i * 4 + 1] = ANOMALY_G;
      rgba[i * 4 + 2] = ANOMALY_B;
      rgba[i * 4 + 3] = ANOMALY_ALPHA;
      nAnomaly++;
    }
  }

  // ── 2. vtkImageData for MPR slices (4-component RGBA) ────────────────────
  const src = sourceImageData as ReturnType<typeof vtkImageData.newInstance>;

  const sliceData = vtkImageData.newInstance();
  sliceData.setSpacing(src.getSpacing());
  sliceData.setOrigin(src.getOrigin());
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  sliceData.setDirection((src as any).getDirection());
  sliceData.setDimensions(nx, ny, nz);
  sliceData.getPointData().setScalars(
    vtkDataArray.newInstance({
      name:               'AnomalyRGBA',
      numberOfComponents: 4,
      values:             rgba,
    }),
  );

  // ── 3. MPR overlay slice actors ──────────────────────────────────────────
  const midK = Math.floor(nz / 2);
  const midJ = Math.floor(ny / 2);
  const midI = Math.floor(nx / 2);

  const axActor = _buildSliceActor(sliceData, SlicingMode.K, midK, initialOpacity);
  const coActor = _buildSliceActor(sliceData, SlicingMode.J, midJ, initialOpacity);
  const saActor = _buildSliceActor(sliceData, SlicingMode.I, midI, initialOpacity);

  // ── 4. vtkImageData for 3-D volume (1-component scalar, values {0, 1}) ──
  // The volume mapper needs a single scalar channel for CTF/OTF ray-casting.
  // We share the original maskFlat array — no copy needed.
  const volData = vtkImageData.newInstance();
  volData.setSpacing(src.getSpacing());
  volData.setOrigin(src.getOrigin());
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  volData.setDirection((src as any).getDirection());
  volData.setDimensions(nx, ny, nz);
  volData.getPointData().setScalars(
    vtkDataArray.newInstance({
      name:               'AnomalyMask',
      numberOfComponents: 1,
      values:             maskFlat,
    }),
  );

  // Color transfer function: scalar 0 → black (never seen — OTF is 0 there),
  // scalar 1 → pure red.
  const ctf = vtkColorTransferFunction.newInstance();
  ctf.addRGBPoint(0, 0, 0, 0);
  ctf.addRGBPoint(1, 1, 0, 0);

  // Opacity transfer function: background transparent, anomaly adjustable.
  const otf = vtkPiecewiseFunction.newInstance();
  otf.addPoint(0, 0);
  otf.addPoint(1, initialOpacity * MAX_VOLUME_OPACITY);

  const volMapper = vtkVolumeMapper.newInstance();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (volMapper as any).setInputData(volData);

  const vol3d = vtkVolume.newInstance();
  vol3d.setMapper(volMapper);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const volProp = vol3d.getProperty() as any;
  volProp.setRGBTransferFunction(0, ctf);
  volProp.setScalarOpacity(0, otf);
  // Shading off: avoids dark shadow artefacts on the thin binary anomaly surface.
  volProp.setShade(false);

  // ── 5. setSlices ─────────────────────────────────────────────────────────
  function setSlices(k: number, j: number, i: number): void {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (axActor.getMapper() as any).setSlice(k);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (coActor.getMapper() as any).setSlice(j);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (saActor.getMapper() as any).setSlice(i);
  }

  // ── 6. setOpacity ────────────────────────────────────────────────────────
  // MPR: actor property opacity — O(1), no buffer rebuild.
  // 3-D:  rebuild OTF (2 addPoint calls) + mark property modified.
  function setOpacity(factor: number): void {
    const clamped = Math.max(0, Math.min(1, factor));
    // Keep a tiny epsilon so the actor stays in the translucent render pass.
    // At absolute 0 the pass switch can produce a one-frame black flash.
    const sliceOpacity = Math.max(0.001, clamped);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (axActor.getProperty() as any).setOpacity(sliceOpacity);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (coActor.getProperty() as any).setOpacity(sliceOpacity);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (saActor.getProperty() as any).setOpacity(sliceOpacity);

    otf.removeAllPoints();
    otf.addPoint(0, 0);
    otf.addPoint(1, clamped * MAX_VOLUME_OPACITY);
    volProp.modified();
  }

  return {
    axial:    axActor,
    coronal:  coActor,
    sagittal: saActor,
    volume3d: vol3d,
    setSlices,
    setOpacity,
    get nAnomaly() { return nAnomaly; },
  };
}

// ── Private helper ────────────────────────────────────────────────────────────

/**
 * Build a single MPR RGBA overlay actor.
 *
 * prop.setIndependentComponents(false) tells vtk.js to treat the 4 scalar
 * components as a single RGBA pixel rather than four independent channels,
 * activating the per-pixel alpha-blending path.
 *
 * prop.setOpacity(< 1.0) forces the translucent rendering pass (GL_BLEND on)
 * so background voxels (A = 0) remain transparent instead of writing black
 * to the framebuffer.  See segmentationOverlay.ts for the full explanation.
 */
function _buildSliceActor(
  imageData:      ReturnType<typeof vtkImageData.newInstance>,
  slicingMode:    number,
  initialSlice:   number,
  initialOpacity: number,
): ReturnType<typeof vtkImageSlice.newInstance> {
  const mapper = vtkImageMapper.newInstance();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const m = mapper as any;
  m.setInputData(imageData);
  m.setSlicingMode(slicingMode);
  m.setSlice(initialSlice);

  const actor = vtkImageSlice.newInstance();
  actor.setMapper(mapper);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const prop = actor.getProperty() as any;
  prop.setIndependentComponents(false);
  prop.setOpacity(Math.max(0.001, initialOpacity));

  return actor;
}
