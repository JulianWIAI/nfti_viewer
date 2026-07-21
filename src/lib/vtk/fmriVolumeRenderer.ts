/**
 * fmriVolumeRenderer.ts — 4-D fMRI volume rendering pipeline
 * ────────────────────────────────────────────────────────────
 *
 * Wraps the standard volumeRenderer pipeline with support for 4-D NIfTI
 * data.  A 4-D NIfTI stores N successive 3-D brain volumes (one per TR)
 * as a single contiguous flat array.  This module slices out one frame
 * at a time and updates the vtkImageData scalars in-place, which lets
 * the renderer display a different temporal volume without rebuilding the
 * entire VTK pipeline (no vtkImageData recreation, no mapper reconnect).
 *
 * FRAME SELECTION
 * ────────────────
 * Given dims [x, y, z] and N timepoints the flat buffer is laid out as:
 *
 *   [ volume_0 | volume_1 | … | volume_{N-1} ]
 *   each sub-block has size  x * y * z  scalars.
 *
 * setTimepoint(t) copies volume t's sub-block into the active vtkDataArray,
 * then calls imageData.modified() to notify the mapper that the data changed.
 * vtk.js propagates the dirty flag to the GPU upload on the next render.
 *
 * DESIGN DECISION: copy vs. view
 * ───────────────────────────────
 * We copy one frame into a pre-allocated Float32Array rather than creating
 * a new view per call.  Rationale:
 *  1. vtkDataArray holds a reference to the typed array — reassigning a new
 *     typed array with setData() works but forces the mapper to re-validate
 *     the data layout on every frame, adding ~1 ms per call at 256³ voxels.
 *  2. A fixed-size scratch buffer avoids GC pressure during playback.
 *
 * USAGE
 * ──────
 *   import { buildFmriVolumeActor } from './fmriVolumeRenderer';
 *
 *   const bundle = buildFmriVolumeActor(fmriPayload);
 *   renderer.addVolume(bundle.actor);
 *   bundle.setTimepoint(3);     // jump to the 4th TR
 *   renderWindow.render();
 */

import vtkImageData   from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkDataArray   from '@kitware/vtk.js/Common/Core/DataArray';
import vtkVolume      from '@kitware/vtk.js/Rendering/Core/Volume';
import vtkVolumeMapper from '@kitware/vtk.js/Rendering/Core/VolumeMapper';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import vtkPiecewiseFunction     from '@kitware/vtk.js/Common/DataModel/PiecewiseFunction';
import type { FmriPayload } from '../../types/fmri.types';

// ── Public types ─────────────────────────────────────────────────────────────

/**
 * Everything the caller needs to drive the 4-D volume renderer.
 * Extends the standard VolumeActorBundle pattern with setTimepoint().
 */
export interface FmriActorBundle {
  /** VTK volume actor — add to the 3-D renderer once at startup. */
  actor:     ReturnType<typeof vtkVolume.newInstance>;
  /** Volume mapper — pass to lodManager if desired. */
  mapper:    ReturnType<typeof vtkVolumeMapper.newInstance>;
  /** Shared vtkImageData — pass to mprRenderer for slice actors. */
  imageData: ReturnType<typeof vtkImageData.newInstance>;
  /** Sampled [min, max] of the scalar intensities of volume 0. */
  dataRange: [number, number];
  /**
   * Switch the displayed volume to the t-th timepoint (0-indexed).
   * Clamps silently to [0, nTimepoints-1].
   * Does NOT call renderWindow.render() — the caller is responsible.
   */
  setTimepoint(t: number): void;
  /** Update window/level without rebuilding the pipeline. */
  updateWindowLevel(center: number, width: number): void;
  /** Update 3-D volume opacity (0 = transparent, 1 = opaque). */
  updateOpacity(factor: number): void;
  /** Number of temporal volumes — useful for slider max. */
  nTimepoints: number;
  /** TR in seconds — used by SyncContext to convert seconds ↔ volume index. */
  tr: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Samples at most 5 000 evenly spaced voxels from a typed array to estimate
 * [min, max] without scanning the entire volume (which can be 512³ voxels).
 */
function sampleRange(arr: Float32Array | Int16Array | Uint8Array): [number, number] {
  const step = Math.max(1, Math.floor(arr.length / 5000));
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < arr.length; i += step) {
    const v = arr[i];
    if (!isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return min < max ? [min, max] : [0, 1];
}

/**
 * Derives the vtkImageData direction matrix (column-major 3×3 Float32Array)
 * from the NIfTI affine.  Divides each column by its corresponding spacing
 * value to recover the pure rotation component.
 */
function extractDirection(affine: number[][], pixDims: number[]): Float32Array {
  const dx = pixDims[1] || 1;
  const dy = pixDims[2] || 1;
  const dz = pixDims[3] || 1;
  return Float32Array.from([
    affine[0][0] / dx, affine[1][0] / dx, affine[2][0] / dx,
    affine[0][1] / dy, affine[1][1] / dy, affine[2][1] / dy,
    affine[0][2] / dz, affine[1][2] / dz, affine[2][2] / dz,
  ]);
}

// ── Default transfer functions ────────────────────────────────────────────────

function buildColorTF(lo: number, hi: number): ReturnType<typeof vtkColorTransferFunction.newInstance> {
  const span = hi - lo || 1;
  const ctf  = vtkColorTransferFunction.newInstance();
  // Standard T1-weighted MRI palette (black→grey matter→white matter→white)
  ctf.addRGBPoint(lo,                0.00, 0.00, 0.00);
  ctf.addRGBPoint(lo + 0.10 * span,  0.10, 0.10, 0.10);
  ctf.addRGBPoint(lo + 0.30 * span,  0.60, 0.35, 0.25);
  ctf.addRGBPoint(lo + 0.60 * span,  0.90, 0.90, 0.70);
  ctf.addRGBPoint(hi,                1.00, 1.00, 1.00);
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

// ── Factory ──────────────────────────────────────────────────────────────────

/**
 * Builds a 4-D-aware volume rendering pipeline from an FmriPayload.
 *
 * The initial display shows timepoint 0.  Call bundle.setTimepoint(t) +
 * renderWindow.render() to advance through the timeseries.
 *
 * @param payload - Parsed 4-D NIfTI output from the NIfTI worker.
 */
export function buildFmriVolumeActor(payload: FmriPayload): FmriActorBundle {
  const { header, volumeData, dataType, nTimepoints, tr } = payload;
  const { dims, pixDims, affine } = header;

  // Number of voxels in a single 3-D frame
  const frameSize = dims[1] * dims[2] * dims[3];

  // ── Wrap the full 4-D buffer in the correct typed array ──────────────────
  let fullData: Float32Array | Int16Array | Uint8Array;
  switch (dataType) {
    case 'Uint8Array':  fullData = new Uint8Array(volumeData);   break;
    case 'Int16Array':  fullData = new Int16Array(volumeData);   break;
    case 'Float32Array':
    default:            fullData = new Float32Array(volumeData); break;
  }

  // ── Scratch buffer for a single frame (avoids per-frame allocation) ───────
  // Int16Array and Uint8Array cannot be set to vtkDataArray as Float32 scalars,
  // so we always promote to Float32 for the active-frame scratch buffer.
  const frameBuffer = new Float32Array(frameSize);

  // Copy the first frame into the scratch buffer
  for (let i = 0; i < frameSize; i++) {
    frameBuffer[i] = fullData[i];
  }

  // ── Compute intensity range from the first frame only ────────────────────
  const dataRange: [number, number] = sampleRange(frameBuffer);
  const [lo, hi] = dataRange;

  // ── vtkImageData ──────────────────────────────────────────────────────────
  const imageData = vtkImageData.newInstance();
  imageData.setDimensions(dims[1], dims[2], dims[3]);
  imageData.setSpacing([pixDims[1], pixDims[2], pixDims[3]]);
  imageData.setOrigin([affine[0][3], affine[1][3], affine[2][3]] as [number, number, number]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  imageData.setDirection(extractDirection(affine, pixDims) as any);

  // Wrap the scratch buffer in a vtkDataArray and attach it as scalars.
  // After this the mapper always reads from frameBuffer — no ImageData rebuild needed.
  const scalars = vtkDataArray.newInstance({
    name: 'Scalars',
    numberOfComponents: 1,
    values: frameBuffer,
  });
  imageData.getPointData().setScalars(scalars);

  // ── Volume mapper ─────────────────────────────────────────────────────────
  const mapper = vtkVolumeMapper.newInstance();
  mapper.setInputData(imageData);
  mapper.setSampleDistance(0.5);
  mapper.setAutoAdjustSampleDistances(false);

  // ── Transfer functions ────────────────────────────────────────────────────
  const colorTF   = buildColorTF(lo, hi);
  const opacityTF = buildOpacityTF(lo, hi);

  // ── Volume actor ──────────────────────────────────────────────────────────
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

  // ── setTimepoint — hot path called once per TR during playback ───────────
  function setTimepoint(t: number): void {
    // Clamp to valid range
    const idx = Math.max(0, Math.min(Math.round(t), nTimepoints - 1));
    const offset = idx * frameSize;

    // Copy this frame's scalars into the shared scratch buffer.
    // Float32Array.set() is a single typed-array copy — typically ~1 µs for 256³.
    for (let i = 0; i < frameSize; i++) {
      frameBuffer[i] = fullData[offset + i];
    }

    // Notify the mapper that scalar data has changed.
    // modified() sets an internal timestamp; the GPU upload happens on the next render() call.
    imageData.modified();
  }

  // ── Window/Level update ───────────────────────────────────────────────────
  function updateWindowLevel(center: number, width: number): void {
    const wlo = center - width / 2;
    const whi = center + width / 2;
    colorTF.removeAllPoints();
    colorTF.addRGBPoint(wlo - 1, 0, 0, 0);
    colorTF.addRGBPoint(wlo,     0, 0, 0);
    colorTF.addRGBPoint(whi,     1, 1, 1);
    colorTF.addRGBPoint(whi + 1, 1, 1, 1);
    opacityTF.removeAllPoints();
    opacityTF.addPoint(wlo - 1, 0.0);
    opacityTF.addPoint(wlo,     0.0);
    opacityTF.addPoint(wlo + (whi - wlo) * 0.2, 0.1);
    opacityTF.addPoint(whi,     0.85);
    property.setRGBTransferFunction(0, colorTF);
    property.setScalarOpacity(0, opacityTF);
    property.modified();
  }

  // ── Opacity update ────────────────────────────────────────────────────────
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

  return {
    actor, mapper, imageData, dataRange,
    setTimepoint, updateWindowLevel, updateOpacity,
    nTimepoints, tr,
  };
}
