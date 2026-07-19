/**
 * longitudinalOverlay.ts — vtk.js overlay actors for float32 longitudinal delta maps
 * ─────────────────────────────────────────────────────────────────────────────────────
 *
 * Builds four actors from a Float32Array delta volume returned by the
 * /api/longitudinal/delta endpoint.  The delta encodes:
 *
 *   Positive (+) → tissue growth, fluid expansion, lesion progression  (RED)
 *   Negative (−) → grey-matter atrophy, tissue loss, lesion shrinkage  (BLUE)
 *   Near zero    → within noise floor — rendered fully transparent
 *
 * FOUR ACTORS
 * ────────────
 *   axial / coronal / sagittal  — vtkImageSlice RGBA quads for the three MPR views
 *   volume3d                    — vtkVolume with divergent CTF + noise-floor OTF
 *
 * WHY TWO SEPARATE vtkImageData OBJECTS
 * ──────────────────────────────────────
 * Identical to anomalyOverlay.ts: vtkImageSlice needs 4-component UNSIGNED_CHAR RGBA
 * scalars, while vtkVolumeMapper needs 1-component Float32 scalars for the CTF/OTF
 * ray-casting pipeline.  Both share geometry from the source MRI imageData.
 *
 * NOISE FLOOR — CRITICAL DESIGN DECISION
 * ────────────────────────────────────────
 * Registration residuals and MRI noise produce a band of small delta values around
 * zero that are NOT clinically meaningful.  We suppress them completely by setting
 * alpha = 0 for any voxel whose |delta| < NOISE_FRACTION × absMax.
 *
 *   absMax        = max(|minVal|, |maxVal|, 1e-6)          — symmetric range ceiling
 *   noiseCeiling  = absMax × NOISE_FRACTION   (default 5%) — invisible below this
 *   transitionPt  = absMax × TRANSITION_FRACTION (10%)     — full opacity above this
 *
 * Between noiseCeiling and transitionPt, opacity ramps linearly from 0 to PEAK_ALPHA,
 * producing a smooth threshold rather than a hard cutoff.
 *
 * DIVERGENT COLORMAP — MPR SLICES
 * ────────────────────────────────
 * RGBA is baked once at construction time from the Float32Array delta values:
 *
 *   |val| < noiseCeiling  → (0, 0, 0, 0)          — fully transparent
 *   val < 0 (atrophy)     → (0, 0, 255, α)         — blue, α proportional to |val|
 *   val > 0 (growth)      → (255, 0, 0, α)          — red,  α proportional to val
 *
 * Runtime opacity slider: actor.getProperty().setOpacity(factor) — O(1), no rebuild.
 *
 * DIVERGENT COLORMAP — 3D VOLUME
 * ────────────────────────────────
 * CTF spans [effectiveMin → blue] via [0 → white] to [effectiveMax → red].
 * OTF suppresses the noise floor band [-noiseCeiling, +noiseCeiling] → opacity 0,
 * ramps to full opacity beyond transitionPt.
 *
 * Runtime opacity slider: rebuilds the OTF (7 addPoint calls) + volProp.modified().
 * Negligibly cheap compared to a buffer rebuild.
 *
 * BYTE ORDER CONTRACT
 * ─────────────────────
 * deltaFlat must be in Fortran order (X varies fastest), exactly as serialised by
 * np.asfortranarray(delta.astype(np.float32)).tobytes(order='F').
 * vtk.js stores vtkImageData point scalars in this same X-fastest layout.
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

// ── Colormap constants ─────────────────────────────────────────────────────────

// Per-voxel alpha ceiling for MPR slices (at maximum |delta|, full slider).
// Multiplied by the actor property opacity at runtime — so the actual alpha
// for a maximally-changed voxel is PEAK_ALPHA/255 × actor.getProperty().opacity.
const PEAK_ALPHA = 220;

// Peak opacity for the 3-D ray-cast volume at slider factor = 1.0.
const MAX_VOLUME_OPACITY = 0.75;

// Fraction of absMax below which voxels are considered noise and hidden.
const NOISE_FRACTION = 0.05;

// Fraction of absMax above which voxels reach full opacity.
const TRANSITION_FRACTION = 0.10;

// ── Public interface ──────────────────────────────────────────────────────────

export interface LongitudinalBundle {
  /** RGBA overlay quad for the axial (K-slice) viewport. */
  axial:    ReturnType<typeof vtkImageSlice.newInstance>;
  /** RGBA overlay quad for the coronal (J-slice) viewport. */
  coronal:  ReturnType<typeof vtkImageSlice.newInstance>;
  /** RGBA overlay quad for the sagittal (I-slice) viewport. */
  sagittal: ReturnType<typeof vtkImageSlice.newInstance>;
  /** Full 3-D ray-cast volume with divergent CTF + noise-floor OTF. */
  volume3d: ReturnType<typeof vtkVolume.newInstance>;

  /**
   * Sync overlay slice positions with the MPR controls.
   * Must be followed by renderWindow.render().
   */
  setSlices(k: number, j: number, i: number): void;

  /**
   * Set global overlay opacity.  factor ∈ [0, 1].
   *
   * MPR slices: actor.getProperty().setOpacity(factor) — O(1), no buffer rebuild.
   * 3-D volume: rebuilds the 7-point OTF + marks property modified (cheap).
   *
   * Caller must trigger renderWindow.render() after this call.
   */
  setOpacity(factor: number): void;

  /** Minimum delta intensity from the backend (before noise-floor clipping). */
  readonly minVal:    number;
  /** Maximum delta intensity from the backend (before noise-floor clipping). */
  readonly maxVal:    number;
  /** Count of voxels with |delta| ≥ noiseCeiling and delta > 0 (growth). */
  readonly nPositive: number;
  /** Count of voxels with |delta| ≥ noiseCeiling and delta < 0 (atrophy). */
  readonly nNegative: number;
}

// ── Factory function ──────────────────────────────────────────────────────────

/**
 * Build a four-actor longitudinal overlay from a float32 delta volume.
 *
 * @param sourceImageData  MRI vtkImageData — only geometry is used
 *                         (spacing, origin, direction, dims).
 * @param deltaFlat        Float32Array in Fortran order (X-fastest), decoded
 *                         from the base64 response of /api/longitudinal/delta.
 * @param dims             [X, Y, Z] voxel dimensions matching deltaFlat.
 * @param minVal           Minimum delta intensity (from backend response).
 * @param maxVal           Maximum delta intensity (from backend response).
 * @param initialOpacity   Starting opacity factor ∈ [0, 1].  Defaults to 0.8.
 */
export function buildLongitudinalOverlay(
  sourceImageData: unknown,
  deltaFlat:       Float32Array,
  dims:            [number, number, number],
  minVal:          number,
  maxVal:          number,
  initialOpacity   = 0.8,
): LongitudinalBundle {

  const [nx, ny, nz] = dims;
  const nVox = nx * ny * nz;

  // ── Noise floor thresholds ──────────────────────────────────────────────────
  // absMax is the symmetric range ceiling; all fractions are relative to it.
  // The 1e-6 guard prevents division by zero for a uniform (all-zero) delta.
  const absMax       = Math.max(Math.abs(minVal), Math.abs(maxVal), 1e-6);
  const noiseCeiling = absMax * NOISE_FRACTION;
  const transitionPt = absMax * TRANSITION_FRACTION;

  // effectiveMin / effectiveMax: the actual scalar extent for CTF/OTF anchor points.
  // Clamped outward so OTF/CTF control points are always strictly ascending.
  const effectiveMin = Math.min(minVal, -transitionPt - 1e-6);
  const effectiveMax = Math.max(maxVal,  transitionPt + 1e-6);

  // ── 1. Build 4-component RGBA buffer for MPR slices ────────────────────────
  // One pass over the float32 delta:
  //   |val| < noiseCeiling  → (0,0,0,0)    transparent
  //   val < 0 (atrophy)     → (0,0,255,α)  blue,  α proportional to distance from noise
  //   val > 0 (growth)      → (255,0,0,α)  red,   α proportional to distance from noise
  //
  // Uint8Array is zero-initialised, so noise-floor voxels require no writes.
  const rgba = new Uint8Array(nVox * 4);
  let nPositive = 0;
  let nNegative = 0;

  // Denominator for the linear ramp: distance from noiseCeiling to absMax.
  // At least 1e-9 to prevent division by zero when absMax ≈ noiseCeiling.
  const rampDenom = Math.max(absMax - noiseCeiling, 1e-9);

  for (let i = 0; i < nVox; i++) {
    const val    = deltaFlat[i];
    const absVal = Math.abs(val);

    // Below the noise floor — leave as transparent (Uint8Array zero-initialised).
    if (absVal < noiseCeiling) continue;

    // Linear ramp: 0 at noiseCeiling → PEAK_ALPHA at absMax.
    const t     = Math.min(1, (absVal - noiseCeiling) / rampDenom);
    const alpha = Math.round(t * PEAK_ALPHA);

    rgba[i * 4 + 3] = alpha;  // A

    if (val < 0) {
      // Atrophy — blue
      rgba[i * 4 + 2] = 255; // B = 255, R = G = 0
      nNegative++;
    } else {
      // Growth — red
      rgba[i * 4]     = 255; // R = 255, G = B = 0
      nPositive++;
    }
  }

  // ── 2. vtkImageData for MPR slices (4-component RGBA UNSIGNED_CHAR) ────────
  const src = sourceImageData as ReturnType<typeof vtkImageData.newInstance>;

  const sliceData = vtkImageData.newInstance();
  sliceData.setSpacing(src.getSpacing());
  sliceData.setOrigin(src.getOrigin());
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  sliceData.setDirection((src as any).getDirection());
  sliceData.setDimensions(nx, ny, nz);
  sliceData.getPointData().setScalars(
    vtkDataArray.newInstance({
      name:               'LongitudinalRGBA',
      numberOfComponents: 4,
      values:             rgba,
    }),
  );

  // ── 3. MPR overlay slice actors ────────────────────────────────────────────
  const midK = Math.floor(nz / 2);
  const midJ = Math.floor(ny / 2);
  const midI = Math.floor(nx / 2);

  const axActor = _buildSliceActor(sliceData, SlicingMode.K, midK, initialOpacity);
  const coActor = _buildSliceActor(sliceData, SlicingMode.J, midJ, initialOpacity);
  const saActor = _buildSliceActor(sliceData, SlicingMode.I, midI, initialOpacity);

  // ── 4. vtkImageData for 3-D volume (1-component Float32 scalars) ──────────
  // The volume mapper receives the raw float32 delta values and maps them to
  // colour + opacity using the CTF / OTF defined below.
  const volData = vtkImageData.newInstance();
  volData.setSpacing(src.getSpacing());
  volData.setOrigin(src.getOrigin());
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  volData.setDirection((src as any).getDirection());
  volData.setDimensions(nx, ny, nz);
  volData.getPointData().setScalars(
    vtkDataArray.newInstance({
      name:               'LongitudinalDelta',
      numberOfComponents: 1,
      values:             deltaFlat, // Float32Array — type inferred by vtk.js
    }),
  );

  // ── 5. Divergent CTF: blue ← atrophy ← 0 → growth → red ─────────────────
  // Seven control points give a smooth blue → near-white → near-white → red
  // gradient, with white at zero (where OTF opacity = 0 so it never shows).
  const ctf = vtkColorTransferFunction.newInstance();
  ctf.addRGBPoint(effectiveMin,      0,   0,   1);   // max atrophy  → pure blue
  ctf.addRGBPoint(-transitionPt,     0.2, 0.2, 1);   // transition   → deep blue
  ctf.addRGBPoint(-noiseCeiling,     0.7, 0.7, 1);   // near noise   → light blue (OTF = 0 here)
  ctf.addRGBPoint(0,                 1,   1,   1);   // zero         → white      (OTF = 0)
  ctf.addRGBPoint(noiseCeiling,      1,   0.7, 0.7); // near noise   → light red  (OTF = 0 here)
  ctf.addRGBPoint(transitionPt,      1,   0.2, 0.2); // transition   → deep red
  ctf.addRGBPoint(effectiveMax,      1,   0,   0);   // max growth   → pure red

  // ── 6. OTF: transparent noise band, opaque signal band ────────────────────
  const otf = vtkPiecewiseFunction.newInstance();
  const initMaxVolOpacity = initialOpacity * MAX_VOLUME_OPACITY;
  _buildOtf(otf, effectiveMin, effectiveMax, noiseCeiling, transitionPt, initMaxVolOpacity);

  // ── 7. Assemble the 3-D volume actor ──────────────────────────────────────
  const volMapper = vtkVolumeMapper.newInstance();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (volMapper as any).setInputData(volData);

  const vol3d  = vtkVolume.newInstance();
  vol3d.setMapper(volMapper);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const volProp = vol3d.getProperty() as any;
  volProp.setRGBTransferFunction(0, ctf);
  volProp.setScalarOpacity(0, otf);
  // Shading off: avoids dark shadow artefacts on the thin delta surface.
  volProp.setShade(false);

  // ── 8. setSlices ───────────────────────────────────────────────────────────
  function setSlices(k: number, j: number, i: number): void {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (axActor.getMapper() as any).setSlice(k);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (coActor.getMapper() as any).setSlice(j);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (saActor.getMapper() as any).setSlice(i);
  }

  // ── 9. setOpacity ──────────────────────────────────────────────────────────
  // MPR: actor property opacity — O(1), no RGBA buffer rebuild.
  // 3-D: rebuild 7-point OTF + mark property modified (trivially cheap).
  function setOpacity(factor: number): void {
    const clamped = Math.max(0, Math.min(1, factor));
    // Keep a tiny epsilon so actors stay in the translucent render pass;
    // at absolute 0 the pass switch can produce a one-frame black flash.
    const sliceOpacity = Math.max(0.001, clamped);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (axActor.getProperty() as any).setOpacity(sliceOpacity);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (coActor.getProperty() as any).setOpacity(sliceOpacity);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (saActor.getProperty() as any).setOpacity(sliceOpacity);

    _buildOtf(otf, effectiveMin, effectiveMax, noiseCeiling, transitionPt, clamped * MAX_VOLUME_OPACITY);
    volProp.modified();
  }

  return {
    axial:    axActor,
    coronal:  coActor,
    sagittal: saActor,
    volume3d: vol3d,
    setSlices,
    setOpacity,
    get minVal()    { return minVal;    },
    get maxVal()    { return maxVal;    },
    get nPositive() { return nPositive; },
    get nNegative() { return nNegative; },
  };
}

// ── Private helpers ───────────────────────────────────────────────────────────

/**
 * Build a single MPR RGBA overlay actor for a given slicing axis.
 *
 * prop.setIndependentComponents(false) — 4 scalars treated as a single RGBA
 * pixel, activating the per-pixel alpha-blending path in vtk.js.
 *
 * prop.setOpacity(< 1.0) — forces the translucent rendering pass (GL_BLEND),
 * so background voxels (A = 0) remain transparent.
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

/**
 * Populate a PiecewiseFunction with the 7-point noise-suppressing OTF.
 *
 * Control point layout (ascending scalar value):
 *   effectiveMin  → maxOpacity   (full opacity at maximum atrophy)
 *   -transitionPt → maxOpacity   (still full opacity at transition boundary)
 *   -noiseCeiling → 0            (drops to zero just outside the noise band)
 *   0             → 0            (zero delta is always transparent)
 *   +noiseCeiling → 0            (drops to zero just outside the noise band)
 *   +transitionPt → maxOpacity   (ramps back to full opacity at transition)
 *   effectiveMax  → maxOpacity   (full opacity at maximum growth)
 *
 * effectiveMin and effectiveMax are pre-clamped so they never violate the
 * ascending-value contract required by vtkPiecewiseFunction.
 */
function _buildOtf(
  otf:           ReturnType<typeof vtkPiecewiseFunction.newInstance>,
  effectiveMin:  number,
  effectiveMax:  number,
  noiseCeiling:  number,
  transitionPt:  number,
  maxOpacity:    number,
): void {
  otf.removeAllPoints();
  otf.addPoint(effectiveMin,   maxOpacity);
  otf.addPoint(-transitionPt,  maxOpacity);
  otf.addPoint(-noiseCeiling,  0);
  otf.addPoint(0,              0);
  otf.addPoint(noiseCeiling,   0);
  otf.addPoint(transitionPt,   maxOpacity);
  otf.addPoint(effectiveMax,   maxOpacity);
}
