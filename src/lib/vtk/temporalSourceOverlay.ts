/**
 * temporalSourceOverlay.ts — Time-series EEG source localisation overlay
 * ────────────────────────────────────────────────────────────────────────
 *
 * Extends the static vtkGlyph3DMapper pipeline from sourceLocalizationOverlay.ts
 * to support a 2-D amplitude matrix.  The core performance insight is that
 * the coordinates (vtkPoints) are written ONCE at build time, and each call to
 * updateTimeFrame() writes into the SAME pre-allocated Float32Array buffers that
 * the vtkDataArrays already reference — producing zero heap allocations per frame.
 *
 * TEMPORAL DATA SHAPE
 * ────────────────────
 *   {
 *     coordinates: [{ x, y, z }, …],          // N sources, MNI-152 mm
 *     amplitudes:  [[t0_amp0, …, t0_ampN-1],  // T×N matrix
 *                   [t1_amp0, …, t1_ampN-1],  //   outer index = time step
 *                   …]                        //   inner index = source index
 *   }
 *
 * PERFORMANCE STRATEGY
 * ─────────────────────
 * Rebuilding vtk.js pipeline objects (new vtkDataArray, new Float32Array) on
 * every slider tick would trigger garbage collection and pipeline re-validation,
 * causing visible frame drops at ≥15 frames/s.  Instead:
 *
 *   Build time (once):
 *     ampBuffer    = new Float32Array(N)   ← owned by ampScalars DataArray
 *     radiusBuffer = new Float32Array(N)   ← owned by radiusScalars DataArray
 *
 *   updateTimeFrame(t) — zero allocations:
 *     1. Write amplitudes[t] into ampBuffer    in-place
 *     2. Recompute radii from amplitudes[t]    into radiusBuffer in-place
 *     3. Call dataChange() on both DataArrays  (fires modified(), clears range cache)
 *     4. Call polyData.modified()              (propagates dirty up to the mapper)
 *     5. Caller calls renderWindow.render()    (redraws the scene)
 *
 * The global amplitude range is computed once across ALL frames so the hot CTF
 * and sphere scale remain stable as the user scrubs — the colour assignment
 * for a given amplitude is identical at t=0 and t=T-1.
 *
 * COORDINATE FRAMES
 * ──────────────────
 * Same as sourceLocalizationOverlay.ts — see mniCoordinateTransform.ts for the
 * full HEAD→MNI→vtk-world chain.  For MNI-registered NIfTIs the transform is
 * identity and the MNI mm coordinates are written directly to vtkPoints.
 *
 * USAGE
 * ──────
 *   import { buildTemporalSourceLocalizationOverlay }
 *     from '../../lib/vtk/temporalSourceOverlay';
 *
 *   const bundle = buildTemporalSourceLocalizationOverlay(apiResponse);
 *   ctx.volumeRenderer.addActor(bundle.actor);
 *   ctx.renderWindow.render();           // render frame 0
 *
 *   // Inside TimeScrubber.onScrub callback:
 *   bundle.updateTimeFrame(timeIndex);
 *   ctx.renderWindow.render();
 *
 *   // On unmount / new data:
 *   ctx.volumeRenderer.removeActor(bundle.actor);
 *   bundle.dispose();
 */

import vtkActor         from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkGlyph3DMapper from '@kitware/vtk.js/Rendering/Core/Glyph3DMapper';
import vtkSphereSource  from '@kitware/vtk.js/Filters/Sources/SphereSource';
import vtkPolyData      from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkDataArray     from '@kitware/vtk.js/Common/Core/DataArray';
import { buildHotColorTransferFunction } from './hotColorMap';
import {
  buildMniToVtkWorldTransform,
  transformSourcesToFlat,
  type Matrix4x4,
  type Point3,
} from './mniCoordinateTransform';

// ── Public interfaces ─────────────────────────────────────────────────────────

/**
 * The 2-D temporal payload returned by POST /api/eeg/localize when the backend
 * supplies time-resolved amplitudes (e.g., from a sliding-window dSPM/sLORETA).
 *
 * amplitudes[timeIndex][sourceIndex] = scalar activation value
 *
 * All rows must have the same length as coordinates.
 */
export interface TemporalSourceData {
  /** MNI-152 (x, y, z) mm for each of the N localised source peaks. */
  coordinates: Point3[];
  /**
   * T×N amplitude matrix.  Outer index is the time step; inner index is the
   * source index (aligned to `coordinates`).
   */
  amplitudes: number[][];
}

/** Tuning options — identical to SourceLocalizationOptions but ampMin/ampMax
 *  are intentionally omitted: they are always derived globally across all frames
 *  to keep the CTF and sphere sizes stable during scrubbing. */
export interface TemporalSourceOptions {
  /** Sphere radius (mm) for the source with the lowest amplitude across all frames. @default 2.0 */
  minSphereRadius?: number;
  /** Sphere radius (mm) for the source with the highest amplitude across all frames. @default 7.0 */
  maxSphereRadius?: number;
  /** Longitudinal sphere divisions. Higher = smoother. @default 16 */
  thetaResolution?: number;
  /** Latitudinal sphere divisions. @default 12 */
  phiResolution?: number;
  /**
   * Optional 4×4 row-major affine (see mniCoordinateTransform.ts).
   * Omit for MNI-registered NIfTIs (identity is correct).
   */
  mniToWorldMatrix?: Matrix4x4;
}

/**
 * Live bundle returned by buildTemporalSourceLocalizationOverlay.
 * Add `actor` to `ctx.volumeRenderer`; call `dispose()` before removing it.
 */
export interface TemporalSourceBundle {
  /** Add to ctx.volumeRenderer. */
  actor: ReturnType<typeof vtkActor.newInstance>;
  /** Number of time frames (length of the amplitudes outer array). */
  frameCount: number;
  /**
   * High-performance in-place frame update.
   *
   * Writes the amplitude/radius values for `timeIndex` into the pre-allocated
   * Float32Array buffers that the vtkDataArrays reference.  Zero heap
   * allocations.  Caller must call `ctx.renderWindow.render()` afterwards.
   *
   * @param timeIndex  Frame index in [0, frameCount - 1].  Out-of-range values
   *                   are clamped rather than throwing.
   */
  updateTimeFrame(timeIndex: number): void;
  /** Show or hide the overlay. Caller must render afterwards. */
  setVisible(visible: boolean): void;
  /** Release all vtk.js pipeline objects.  Call before removing the actor. */
  dispose(): void;
}

// ── Factory ───────────────────────────────────────────────────────────────────

/**
 * Build a vtk.js Glyph3DMapper pipeline for time-resolved EEG source data.
 *
 * The pipeline is initialised with frame 0.  Call `bundle.updateTimeFrame(t)`
 * + `renderWindow.render()` on each slider tick to advance to frame t.
 *
 * @param data     Temporal source data from the backend.
 * @param options  Optional sphere geometry and world-transform tuning.
 */
export function buildTemporalSourceLocalizationOverlay(
  data:    TemporalSourceData,
  options: TemporalSourceOptions = {},
): TemporalSourceBundle {
  const { coordinates, amplitudes } = data;
  const n          = coordinates.length;
  const frameCount = amplitudes.length;

  if (n === 0 || frameCount === 0) {
    throw new Error('temporalSourceOverlay: coordinates and amplitudes must be non-empty');
  }

  // ── 1. Global amplitude range ────────────────────────────────────────────
  // Computed ONCE across all frames so the hot CTF and sphere sizes are
  // identical for a given amplitude regardless of which frame is active.
  // This prevents the jarring colour/size jumps that would occur if the range
  // were recomputed per-frame.
  let globalAmpMin =  Infinity;
  let globalAmpMax = -Infinity;
  for (const frame of amplitudes) {
    for (const amp of frame) {
      if (amp < globalAmpMin) globalAmpMin = amp;
      if (amp > globalAmpMax) globalAmpMax = amp;
    }
  }
  // Guard: single amplitude value or identical across all frames.
  const globalAmpSpan = globalAmpMax - globalAmpMin || 1.0;

  const minR = options.minSphereRadius ?? 2.0;
  const maxR = options.maxSphereRadius ?? 7.0;

  // ── 2. Static coordinate geometry ───────────────────────────────────────
  // Coordinates are fixed — only scalar values change per frame.
  // Transform MNI mm → vtk world mm once at build time.
  const M          = buildMniToVtkWorldTransform(options.mniToWorldMatrix);
  const flatCoords = transformSourcesToFlat(M, coordinates);

  const polyData = vtkPolyData.newInstance();
  const ptsArray = vtkDataArray.newInstance({
    numberOfComponents: 3,
    values:             flatCoords,
    name:               'Points',
  });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (polyData as any).setPoints(ptsArray);

  // ── 3. Pre-allocated scalar buffers ─────────────────────────────────────
  // These Float32Arrays are owned by the vtkDataArrays below.  vtk.js stores
  // a reference (not a copy), so writing into them directly modifies what the
  // mapper reads on the next render() call.
  const ampBuffer    = new Float32Array(n); // amplitude for colour mapping
  const radiusBuffer = new Float32Array(n); // per-point glyph radius (mm)

  const ampScalars = vtkDataArray.newInstance({
    name:               'Amplitude',
    numberOfComponents: 1,
    values:             ampBuffer,
  });
  const radiusScalars = vtkDataArray.newInstance({
    name:               'GlyphRadius',
    numberOfComponents: 1,
    values:             radiusBuffer,
  });

  const pd = polyData.getPointData();
  pd.setScalars(ampScalars);                     // active scalars → colour mapping
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (pd as any).addArray(radiusScalars);            // named array → scale mapping

  // ── 4. Sphere glyph template ─────────────────────────────────────────────
  // Radius 1.0 — actual size per instance is driven by the GlyphRadius array
  // via SCALE_BY_MAGNITUDE with scaleFactor 1.0 (values already in mm).
  const sphere = vtkSphereSource.newInstance({
    radius:          1.0,
    thetaResolution: options.thetaResolution ?? 16,
    phiResolution:   options.phiResolution   ?? 12,
    center:          [0, 0, 0],
  });

  // ── 5. Hot colour transfer function (stable across all frames) ───────────
  const ctf = buildHotColorTransferFunction({
    minValue: globalAmpMin,
    maxValue: globalAmpMax,
  });

  // ── 6. Glyph3DMapper ─────────────────────────────────────────────────────
  const mapper = vtkGlyph3DMapper.newInstance({
    scaling:     true,
    scaleFactor: 1.0,
    scaleArray:  'GlyphRadius',
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const m = mapper as any;
  m.setInputData(polyData);
  m.setSourceConnection(sphere.getOutputPort());
  mapper.setScaleModeToScaleByMagnitude();
  m.setColorByArrayName('Amplitude');
  m.setColorModeToMapScalars();
  m.setScalarModeToUsePointFieldData();
  m.setLookupTable(ctf);
  // Pin the mapper's scalar range to [globalAmpMin, globalAmpMax] so the CTF
  // is not re-derived from the current frame's data range on each render call.
  m.setUseLookupTableScalarRange(true);

  // ── 7. Actor with Phong shading ──────────────────────────────────────────
  const actor = vtkActor.newInstance();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  actor.setMapper(mapper as any);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const prop = (actor as any).getProperty();
  prop.setAmbient(0.3);
  prop.setDiffuse(0.7);
  prop.setSpecular(0.4);
  prop.setSpecularPower(20.0);

  // ── 8. Populate frame 0 ──────────────────────────────────────────────────
  _writeFrame(
    amplitudes[0], n, globalAmpMin, globalAmpSpan, minR, maxR,
    ampBuffer, radiusBuffer, ampScalars, radiusScalars, polyData,
  );

  // ── 9. Lifecycle helpers ─────────────────────────────────────────────────

  /**
   * In-place frame update — zero allocations.
   *
   * Writes the amplitude values for `timeIndex` directly into the existing
   * Float32Array buffers and signals vtk.js to re-read them on the next render.
   *
   * The caller must call ctx.renderWindow.render() after this to see the update.
   */
  function updateTimeFrame(timeIndex: number): void {
    const clamped = Math.max(0, Math.min(frameCount - 1, timeIndex));
    _writeFrame(
      amplitudes[clamped], n, globalAmpMin, globalAmpSpan, minR, maxR,
      ampBuffer, radiusBuffer, ampScalars, radiusScalars, polyData,
    );
  }

  function setVisible(visible: boolean): void {
    actor.setVisibility(visible);
  }

  function dispose(): void {
    sphere.delete();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (polyData as any).delete();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (mapper  as any).delete();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (actor   as any).delete();
  }

  return { actor, frameCount, updateTimeFrame, setVisible, dispose };
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/**
 * Write one frame of amplitude/radius data into the pre-allocated buffers.
 *
 * This is the hot path — called on every slider tick.  It does no heap
 * allocation: all writes go into existing Float32Array memory.
 *
 * After writing, dataChange() is called on each DataArray to:
 *   1. Increment the DataArray's mtime so vtk.js knows the buffer changed.
 *   2. Clear the cached range (so the mapper re-reads the actual data range
 *      if needed, though setUseLookupTableScalarRange prevents that here).
 *   3. Internally call modified() — propagates dirty up to the PointData.
 *
 * polyData.modified() propagates the dirty flag to the Glyph3DMapper, which
 * schedules a full GPU buffer re-upload on the next renderWindow.render().
 */
function _writeFrame(
  frameAmps:     number[],
  n:             number,
  ampMin:        number,
  ampSpan:       number,
  minR:          number,
  maxR:          number,
  ampBuffer:     Float32Array,
  radiusBuffer:  Float32Array,
  ampScalars:    ReturnType<typeof vtkDataArray.newInstance>,
  radiusScalars: ReturnType<typeof vtkDataArray.newInstance>,
  polyData:      ReturnType<typeof vtkPolyData.newInstance>,
): void {
  // Write amplitude and radius into the pre-allocated buffers in a single pass.
  for (let i = 0; i < n; i++) {
    const amp        = frameAmps[i];
    const t          = (amp - ampMin) / ampSpan; // normalised 0→1
    ampBuffer[i]    = amp;
    radiusBuffer[i] = minR + t * (maxR - minR);
  }

  // Signal vtk.js that the underlying TypedArray data has changed.
  // dataChange() is the correct API for this pattern — it increments mtime
  // and clears range caches without requiring a new DataArray object.
  ampScalars.dataChange();
  radiusScalars.dataChange();

  // Propagate dirty up to the mapper so it re-uploads the buffers on render().
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (polyData as any).modified();
}
