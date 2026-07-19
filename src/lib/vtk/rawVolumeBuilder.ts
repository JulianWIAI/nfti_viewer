/**
 * rawVolumeBuilder.ts — Build VTK rendering pipeline from raw float32 data
 * ──────────────────────────────────────────────────────────────────────────
 *
 * Mirrors buildVolumeActor() from volumeRenderer.ts, but starts from a raw
 * Float32Array + flat affine matrix rather than a parsed NIfTI VolumePayload.
 *
 * Used by DualVolumetricViewer to load the SyN-warped Subject B volume that
 * comes back from /api/registration/syn as a base64 Fortran-order float32
 * blob (no NIfTI container).
 *
 * AFFINE DECODING
 * ──────────────
 * The backend returns the static scan's 4×4 RAS-mm affine row-major flattened
 * into 16 float64 values.  Layout (flat index → row/column):
 *
 *   [a0,  a1,  a2,  a3,   // row 0: Sxx Sxy Sxz Tx
 *    a4,  a5,  a6,  a7,   // row 1: Syx Syy Syz Ty
 *    a8,  a9,  a10, a11,  // row 2: Szx Szy Szz Tz
 *    a12, a13, a14, a15]  // row 3: 0   0   0   1
 *
 * Spacing (voxel size in mm) = L2 norm of each column of the 3×3 block:
 *   sx = √(a0² + a4² + a8²)
 *   sy = √(a1² + a5² + a9²)
 *   sz = √(a2² + a6² + a10²)
 *
 * Origin = translation column:
 *   (a3, a7, a11)
 *
 * Direction cosines (vtk.js column-major 9-element Float32Array):
 *   [a0/sx, a4/sx, a8/sx,   // x column unit vector
 *    a1/sy, a5/sy, a9/sy,   // y column unit vector
 *    a2/sz, a6/sz, a10/sz]  // z column unit vector
 *
 * BYTE ORDER
 * ───────────
 * The warped float32 data is in Fortran order (X-fastest, matching the
 * np.asfortranarray().tobytes(order='F') serialisation on the backend).
 * vtk.js vtkImageData stores scalars in X-fastest order internally, so
 * no transposition is needed — the bytes are used directly.
 *
 * OUTPUT
 * ───────
 * Returns the same VolumeActorBundle type as buildVolumeActor() so the
 * dual viewer can treat both actors uniformly.
 */

import vtkImageData             from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkDataArray             from '@kitware/vtk.js/Common/Core/DataArray';
import vtkVolume                from '@kitware/vtk.js/Rendering/Core/Volume';
import vtkVolumeMapper          from '@kitware/vtk.js/Rendering/Core/VolumeMapper';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import vtkPiecewiseFunction     from '@kitware/vtk.js/Common/DataModel/PiecewiseFunction';

import type { VolumeActorBundle } from './volumeRenderer';

// ── Data range sampler ────────────────────────────────────────────────────────

/** Samples up to 5 000 voxels to estimate [min, max] without a full scan. */
function computeDataRange(data: Float32Array): [number, number] {
  const step = Math.max(1, Math.floor(data.length / 5000));
  let lo =  Infinity;
  let hi = -Infinity;
  for (let i = 0; i < data.length; i += step) {
    const v = data[i];
    if (!isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo < hi ? [lo, hi] : [0, 1];
}

// ── Default T1-MRI transfer functions ─────────────────────────────────────────

function buildColorTF(lo: number, hi: number): ReturnType<typeof vtkColorTransferFunction.newInstance> {
  const span = hi - lo || 1;
  const ctf  = vtkColorTransferFunction.newInstance();
  ctf.addRGBPoint(lo,                0.0,  0.0,  0.0);
  ctf.addRGBPoint(lo + 0.10 * span,  0.1,  0.1,  0.1);
  ctf.addRGBPoint(lo + 0.30 * span,  0.6,  0.35, 0.25);
  ctf.addRGBPoint(lo + 0.60 * span,  0.9,  0.9,  0.7);
  ctf.addRGBPoint(hi,                1.0,  1.0,  1.0);
  return ctf;
}

function buildOpacityTF(lo: number, hi: number, factor = 1): ReturnType<typeof vtkPiecewiseFunction.newInstance> {
  const span = hi - lo || 1;
  const otf  = vtkPiecewiseFunction.newInstance();
  otf.addPoint(lo,                   0.00 * factor);
  otf.addPoint(lo + 0.15 * span,     0.00 * factor);
  otf.addPoint(lo + 0.30 * span,     0.15 * factor);
  otf.addPoint(lo + 0.60 * span,     0.50 * factor);
  otf.addPoint(hi,                   0.85 * factor);
  return otf;
}

// ── Public factory ────────────────────────────────────────────────────────────

/**
 * Build a complete volume rendering pipeline from raw float32 voxel data and
 * a flat row-major 4×4 RAS affine.
 *
 * @param data        Float32Array in Fortran order (X-fastest), decoded from
 *                    the base64 payload of /api/registration/syn.
 * @param dims        [X, Y, Z] voxel dimensions matching `data`.
 * @param affineFlat  16 float64 values, row-major, from the backend response.
 *                    Used to extract spacing, origin, and direction cosines.
 * @returns           VolumeActorBundle compatible with mprRenderer.buildMprActors()
 *                    and lodManager.attachLod().
 */
export function buildRawVolumeActor(
  data:       Float32Array,
  dims:       [number, number, number],
  affineFlat: number[],
): VolumeActorBundle {
  const [nx, ny, nz] = dims;

  // ── Decode affine → spacing, origin, direction ────────────────────────────
  // Column 0 of the 3×3 block: (a[0], a[4], a[8])
  const sx = Math.sqrt(affineFlat[0] ** 2 + affineFlat[4] ** 2 + affineFlat[8]  ** 2);
  // Column 1: (a[1], a[5], a[9])
  const sy = Math.sqrt(affineFlat[1] ** 2 + affineFlat[5] ** 2 + affineFlat[9]  ** 2);
  // Column 2: (a[2], a[6], a[10])
  const sz = Math.sqrt(affineFlat[2] ** 2 + affineFlat[6] ** 2 + affineFlat[10] ** 2);

  // Guard against degenerate affines (zero voxel size)
  const safeSx = sx || 1;
  const safeSy = sy || 1;
  const safeSz = sz || 1;

  // Direction cosines — vtk.js expects column-major 9-element Float32Array:
  //   [xCol_row0, xCol_row1, xCol_row2, yCol_row0, ..., zCol_row2]
  const direction = Float32Array.from([
    affineFlat[0] / safeSx, affineFlat[4] / safeSx, affineFlat[8]  / safeSx, // x column
    affineFlat[1] / safeSy, affineFlat[5] / safeSy, affineFlat[9]  / safeSy, // y column
    affineFlat[2] / safeSz, affineFlat[6] / safeSz, affineFlat[10] / safeSz, // z column
  ]);

  // Translation column → vtkImageData origin
  const origin: [number, number, number] = [affineFlat[3], affineFlat[7], affineFlat[11]];

  // ── vtkImageData ──────────────────────────────────────────────────────────
  const imageData = vtkImageData.newInstance();
  imageData.setDimensions(nx, ny, nz);
  imageData.setSpacing([safeSx, safeSy, safeSz]);
  imageData.setOrigin(origin);
  // Float32Array satisfies the gl-matrix mat3 type vtk.js v36 expects.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  imageData.setDirection(direction as any);

  imageData.getPointData().setScalars(
    vtkDataArray.newInstance({
      name:               'Scalars',
      numberOfComponents: 1,
      values:             data, // Float32Array — type inferred from values
    }),
  );

  // ── Data range for transfer functions ─────────────────────────────────────
  const [lo, hi] = computeDataRange(data);

  // ── Volume mapper ─────────────────────────────────────────────────────────
  const mapper = vtkVolumeMapper.newInstance();
  mapper.setInputData(imageData);
  mapper.setSampleDistance(0.5);
  mapper.setAutoAdjustSampleDistances(false);

  // ── Transfer functions ────────────────────────────────────────────────────
  const colorTF   = buildColorTF(lo, hi);
  const opacityTF = buildOpacityTF(lo, hi);

  // ── Volume actor + property ───────────────────────────────────────────────
  const actor = vtkVolume.newInstance();
  actor.setMapper(mapper);

  const property = actor.getProperty();
  property.setRGBTransferFunction(0, colorTF);
  property.setScalarOpacity(0, opacityTF);
  property.setInterpolationTypeToLinear();
  property.setShade(true);
  property.setAmbient(0.2);
  property.setDiffuse(0.7);
  property.setSpecular(0.3);
  property.setSpecularPower(8.0);

  // ── Window/level + opacity helpers ────────────────────────────────────────
  function updateWindowLevel(center: number, width: number): void {
    const wlo = center - width / 2;
    const whi = center + width / 2;
    colorTF.removeAllPoints();
    colorTF.addRGBPoint(wlo - 1, 0, 0, 0);
    colorTF.addRGBPoint(wlo,     0, 0, 0);
    colorTF.addRGBPoint(whi,     1, 1, 1);
    colorTF.addRGBPoint(whi + 1, 1, 1, 1);
    opacityTF.removeAllPoints();
    opacityTF.addPoint(wlo - 1,                           0.0);
    opacityTF.addPoint(wlo,                               0.0);
    opacityTF.addPoint(wlo + (whi - wlo) * 0.2,           0.1);
    opacityTF.addPoint(whi,                               0.85);
    property.setRGBTransferFunction(0, colorTF);
    property.setScalarOpacity(0, opacityTF);
    property.modified();
  }

  function updateOpacity(factor: number): void {
    const span = hi - lo || 1;
    opacityTF.removeAllPoints();
    opacityTF.addPoint(lo,                   0.00 * factor);
    opacityTF.addPoint(lo + 0.15 * span,     0.00 * factor);
    opacityTF.addPoint(lo + 0.30 * span,     0.15 * factor);
    opacityTF.addPoint(lo + 0.60 * span,     0.50 * factor);
    opacityTF.addPoint(hi,                   0.85 * factor);
    property.modified();
  }

  return { actor, mapper, imageData, dataRange: [lo, hi], updateWindowLevel, updateOpacity };
}
