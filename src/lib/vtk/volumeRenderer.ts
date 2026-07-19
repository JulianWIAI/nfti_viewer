/**
 * volumeRenderer.ts — Builds a vtk.js vtkVolume from a parsed NIfTI payload
 * ───────────────────────────────────────────────────────────────────────────
 *
 * Responsibilities:
 *   1. Convert the raw voxel ArrayBuffer → vtkImageData with correct
 *      spacing, origin, and direction cosines from the NIfTI affine.
 *   2. Create a vtkVolumeMapper wired to that imageData.
 *   3. Apply default transfer functions (colour + opacity) appropriate for a
 *      T1-weighted MRI brain scan. These are the first things a user will want
 *      to adjust, so they are exposed via `updateTransferFunctions`.
 *   4. Create a vtkVolume actor combining mapper + property.
 *
 * ONNX INTERCEPT POINT
 * ─────────────────────
 * After inference (inferenceEngine.ts), the caller receives a Float32Array
 * segmentation mask with the same (x,y,z) dimensions as the original volume.
 * That mask is wrapped in a *second* vtkImageData and added to the MPR
 * renderers as a semi-transparent colour-mapped overlay. This file does NOT
 * handle the overlay — see mprRenderer.ts where the slice actors live.
 * The key is that both the raw volume and the mask share the same
 * `imageData.getSpacing()` and `imageData.getOrigin()` so they align.
 */

import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkVolume from '@kitware/vtk.js/Rendering/Core/Volume';
import vtkVolumeMapper from '@kitware/vtk.js/Rendering/Core/VolumeMapper';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import vtkPiecewiseFunction from '@kitware/vtk.js/Common/DataModel/PiecewiseFunction';
import type { VolumePayload } from '../../types/nifti.types';

// ── Public types ─────────────────────────────────────────────────────────────

/** Everything the caller needs to add the actor to a renderer and do LOD. */
export interface VolumeActorBundle {
  /** Add to the 3-D renderer. */
  actor: ReturnType<typeof vtkVolume.newInstance>;
  /** Pass to lodManager.attachLod(). */
  mapper: ReturnType<typeof vtkVolumeMapper.newInstance>;
  /**
   * The shared vtkImageData.
   * Passed to mprRenderer.buildMprActors() so slice mappers reference the
   * same data object — no duplication of the voxel buffer.
   */
  imageData: ReturnType<typeof vtkImageData.newInstance>;
  /**
   * Sampled [min, max] of the actual voxel intensities in this volume.
   * Use this to initialise window/level controls to the correct range.
   */
  dataRange: [number, number];
  /** Update window/level without rebuilding the whole pipeline. */
  updateWindowLevel(center: number, width: number): void;
  /** Update the 3-D volume opacity (0 = transparent, 1 = opaque). */
  updateOpacity(factor: number): void;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Samples up to 5 000 voxels evenly distributed through the typed array to
 * estimate the [min, max] intensity range without scanning every voxel.
 * Non-finite values (NaN, ±Inf) are skipped so they don't corrupt the range.
 */
function computeDataRange(typed: Float32Array | Int16Array | Uint8Array): [number, number] {
  const step = Math.max(1, Math.floor(typed.length / 5000));
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < typed.length; i += step) {
    const v = typed[i];
    if (!isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return min < max ? [min, max] : [0, 1];
}

/**
 * Extracts the vtkImageData direction matrix (column-major 3×3, 9 elements)
 * from the NIfTI affine.
 *
 * vtk.js expects direction cosines: each column is a unit vector in world space
 * for the corresponding voxel axis. We divide by the spacing (pixDims) to
 * recover pure rotation from the affine scale+rotation composite.
 */
/**
 * Returns a Float32Array (9 elements) so it is directly assignable to the
 * gl-matrix `mat3` type that `vtkImageData.setDirection` requires in vtk.js v36.
 * Column-major layout: col0, col1, col2.
 */
function extractDirection(affine: number[][], pixDims: number[]): Float32Array {
  const dx = pixDims[1] || 1;
  const dy = pixDims[2] || 1;
  const dz = pixDims[3] || 1;

  // affine row 0-2, column 0-2 encodes R*diag(dx,dy,dz)
  // Divide each column element by the corresponding spacing to get R alone.
  // vtk.js direction is column-major: [col0_row0, col0_row1, col0_row2, col1..., col2...]
  return Float32Array.from([
    affine[0][0] / dx, affine[1][0] / dx, affine[2][0] / dx, // x column
    affine[0][1] / dy, affine[1][1] / dy, affine[2][1] / dy, // y column
    affine[0][2] / dz, affine[1][2] / dz, affine[2][2] / dz, // z column
  ]);
}

/**
 * Wraps the raw ArrayBuffer returned by the worker in the correct TypedArray
 * and creates a vtkDataArray from it.
 *
 * We do NOT copy the buffer — we create a view over the existing ArrayBuffer.
 * vtk.js's DataArray keeps a reference, so the buffer must stay alive as long
 * as the imageData is in use (it will, because imageData holds the DataArray).
 */
function wrapScalars(
  buffer: ArrayBuffer,
  tag: VolumePayload['dataType'],
): Float32Array | Int16Array | Uint8Array {
  switch (tag) {
    case 'Uint8Array':  return new Uint8Array(buffer);
    case 'Int16Array':  return new Int16Array(buffer);
    case 'Float32Array':
    default:            return new Float32Array(buffer);
  }
}

// ── Default transfer functions for a T1 MRI brain ────────────────────────────

/**
 * Builds a colour transfer function scaled to the actual data range.
 * Proportions: 0% → black (background), 10% → near-black (CSF),
 * 30% → brownish (grey matter), 60% → off-white (white matter), 100% → white.
 */
function buildColorTF(lo: number, hi: number): ReturnType<typeof vtkColorTransferFunction.newInstance> {
  const span = hi - lo || 1;
  const ctf = vtkColorTransferFunction.newInstance();
  ctf.addRGBPoint(lo,                0.0,  0.0,  0.0);  // background → black
  ctf.addRGBPoint(lo + 0.10 * span,  0.1,  0.1,  0.1);  // CSF
  ctf.addRGBPoint(lo + 0.30 * span,  0.6,  0.35, 0.25); // grey matter
  ctf.addRGBPoint(lo + 0.60 * span,  0.9,  0.9,  0.7);  // white matter
  ctf.addRGBPoint(hi,                1.0,  1.0,  1.0);  // bright structures
  return ctf;
}

/**
 * Builds a piecewise-linear opacity TF scaled to [lo, hi].
 * The lowest 15% of the range stays transparent (background air).
 */
function buildOpacityTF(lo: number, hi: number, factor: number = 1): ReturnType<typeof vtkPiecewiseFunction.newInstance> {
  const span = hi - lo || 1;
  const otf = vtkPiecewiseFunction.newInstance();
  otf.addPoint(lo,                   0.00 * factor); // background invisible
  otf.addPoint(lo + 0.15 * span,     0.00 * factor); // still background
  otf.addPoint(lo + 0.30 * span,     0.15 * factor); // tissue starts
  otf.addPoint(lo + 0.60 * span,     0.50 * factor); // mid-tissue
  otf.addPoint(hi,                   0.85 * factor); // bright structures
  return otf;
}

// ── Factory ──────────────────────────────────────────────────────────────────

/**
 * Builds a complete volume rendering pipeline from a decoded NIfTI payload.
 *
 * @param payload - Output from the NIfTI worker (header + raw voxel buffer).
 * @returns VolumeActorBundle with the actor, mapper, and imageData.
 */
export function buildVolumeActor(payload: VolumePayload): VolumeActorBundle {
  const { header, volumeData, dataType } = payload;
  const { dims, pixDims, affine } = header;

  // ── 1. vtkImageData ──────────────────────────────────────────────────────
  const imageData = vtkImageData.newInstance();

  // NIfTI dims: [nDims, x, y, z, ...]
  // vtk.js v36 API: setDimensions accepts a 3-tuple or 3 spread args;
  // setSpacing and setOrigin only have the array overload.
  imageData.setDimensions(dims[1], dims[2], dims[3]);
  imageData.setSpacing([pixDims[1], pixDims[2], pixDims[3]]);
  // Origin is the translation column of the affine matrix
  imageData.setOrigin([affine[0][3], affine[1][3], affine[2][3]] as [number, number, number]);
  // Direction cosines encode RAS orientation.
  // Float32Array satisfies the gl-matrix mat3 type vtk.js v36 expects.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  imageData.setDirection(extractDirection(affine, pixDims) as any);

  // Wrap raw buffer as typed array and hand to vtk DataArray
  const typedValues = wrapScalars(volumeData, dataType);
  const scalars = vtkDataArray.newInstance({
    name: 'Scalars',
    numberOfComponents: 1,
    values: typedValues,
  });
  imageData.getPointData().setScalars(scalars);

  // ── Compute actual intensity range for auto-windowing ────────────────────
  // Prefer the NIfTI header calibration values when they are valid (both
  // non-zero and calMax > calMin). Fall back to sampling the voxel data.
  const dataRange: [number, number] =
    header.calMin !== 0 || header.calMax !== 0
      ? [header.calMin, header.calMax]
      : computeDataRange(typedValues);
  const [lo, hi] = dataRange;

  // ── 2. Volume mapper ─────────────────────────────────────────────────────
  const mapper = vtkVolumeMapper.newInstance();
  mapper.setInputData(imageData);
  // LOD manager will override this; starting at fine quality for the first render
  mapper.setSampleDistance(0.5);
  mapper.setAutoAdjustSampleDistances(false);

  // ── 3. Transfer functions scaled to the actual data range ────────────────
  const colorTF   = buildColorTF(lo, hi);
  const opacityTF = buildOpacityTF(lo, hi, 1);

  // ── 4. Volume actor + property ───────────────────────────────────────────
  const actor = vtkVolume.newInstance();
  actor.setMapper(mapper);

  const property = actor.getProperty();
  property.setRGBTransferFunction(0, colorTF);
  property.setScalarOpacity(0, opacityTF);
  property.setInterpolationTypeToLinear();
  // Phong shading — gives tissue a sense of depth
  property.setShade(true);
  property.setAmbient(0.2);
  property.setDiffuse(0.7);
  property.setSpecular(0.3);
  property.setSpecularPower(8.0);

  // ── 5. Helpers exposed to the Viewer for live control-panel updates ──────

  function updateWindowLevel(center: number, width: number): void {
    // Rebuild the colour TF around [center - width/2, center + width/2]
    const lo = center - width / 2;
    const hi = center + width / 2;
    colorTF.removeAllPoints();
    colorTF.addRGBPoint(lo - 1, 0, 0, 0);
    colorTF.addRGBPoint(lo,     0, 0, 0);
    colorTF.addRGBPoint(hi,     1, 1, 1);
    colorTF.addRGBPoint(hi + 1, 1, 1, 1);

    opacityTF.removeAllPoints();
    opacityTF.addPoint(lo - 1, 0.0);
    opacityTF.addPoint(lo,     0.0);
    opacityTF.addPoint(lo + (hi - lo) * 0.2, 0.1);
    opacityTF.addPoint(hi,     0.85);

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

  return { actor, mapper, imageData, dataRange, updateWindowLevel, updateOpacity };
}
