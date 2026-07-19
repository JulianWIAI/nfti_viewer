/**
 * sourceLocalizationOverlay.ts — vtk.js 3-D heatmap for EEG source localisation
 * ────────────────────────────────────────────────────────────────────────────────
 *
 * Renders the MNI-space peak coordinates returned by POST /api/eeg/localize as
 * amplitude-coloured, amplitude-scaled spheres overlaid on the 3-D volume view.
 *
 * PIPELINE OVERVIEW
 * ──────────────────
 *
 *   Backend JSON  →  SourcePoint[]
 *         │
 *         ▼  mniCoordinateTransform.transformSourcesToFlat()
 *   Float64Array  (vtk world mm, flat [x0,y0,z0, x1,y1,z1, …])
 *         │
 *         ▼  vtkPoints  ←─────────────────────────────────────────────┐
 *   vtkPolyData                                                        │
 *     ├─ PointData['Amplitude']   Float32Array  (for CTF coloring)     │
 *     └─ PointData['GlyphRadius'] Float32Array  (per-point sphere size)│
 *         │                                                            │
 *         ▼                                                            │
 *   vtkGlyph3DMapper  ◄── vtkSphereSource.getOutputPort() ────────────┘
 *     ├─ setScaleArray('GlyphRadius')         per-point sphere radius
 *     ├─ setScaleModeToScaleByMagnitude()     scale = GlyphRadius * scaleFactor
 *     ├─ setScaleFactor(1.0)                  radius values are already in mm
 *     ├─ setColorByArrayName('Amplitude')     drive color from amplitude scalar
 *     ├─ setLookupTable(hotCTF)               dark-red → near-white
 *     └─ setUseLookupTableScalarRange(true)   pin CTF to [minAmplitude, maxAmplitude]
 *         │
 *         ▼
 *   vtkActor  (add to ctx.volumeRenderer)
 *
 * ADDING THE ACTOR TO THE SCENE
 * ──────────────────────────────
 * In VolumetricViewer.tsx, after receiving the localize response:
 *
 *   import { buildSourceLocalizationOverlay } from '../../lib/vtk/sourceLocalizationOverlay';
 *
 *   const bundle = buildSourceLocalizationOverlay(response.peaks);
 *   ctx.volumeRenderer.addActor(bundle.actor);
 *   ctx.renderWindow.render();
 *
 *   // Later (e.g., on new upload or component unmount):
 *   bundle.dispose();
 *   ctx.renderWindow.render();
 *
 * COORDINATE ALIGNMENT
 * ──────────────────────
 * Source points are in MNI-152 mm.  For a standard MNI-registered NIfTI the
 * vtk world frame is also MNI mm, so points are transformed by the identity
 * matrix (buildMniToVtkWorldTransform() with no args).  Pass the NIfTI sform
 * affine if the loaded scan uses a different world origin or orientation.
 * See mniCoordinateTransform.ts for the full frame-of-reference discussion.
 *
 * SPHERE SIZING
 * ──────────────
 * Each sphere radius is linearly interpolated between options.minSphereRadius
 * and options.maxSphereRadius proportionally to its amplitude within the
 * [min, max] amplitude range of the current source set.
 *
 *   radius(i) = minR + (maxR - minR) * (amp(i) - ampMin) / (ampMax - ampMin)
 *
 * The vtkSphereSource template is created at radius = 1.0 (unit sphere); the
 * GlyphRadius scalar array then scales each instance independently via the
 * mapper's SCALE_BY_MAGNITUDE mode with scaleFactor = 1.0.  This avoids
 * rebuilding the sphere geometry for every point.
 */

import vtkActor           from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkGlyph3DMapper   from '@kitware/vtk.js/Rendering/Core/Glyph3DMapper';
import vtkSphereSource    from '@kitware/vtk.js/Filters/Sources/SphereSource';
import vtkPolyData        from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkDataArray       from '@kitware/vtk.js/Common/Core/DataArray';
import { buildHotColorTransferFunction } from './hotColorMap';
import {
  buildMniToVtkWorldTransform,
  transformSourcesToFlat,
  type Matrix4x4,
} from './mniCoordinateTransform';

// ── Public interfaces ─────────────────────────────────────────────────────────

/**
 * A single localised source point as returned by POST /api/eeg/localize.
 * Matches the `SourcePeak` Pydantic model from backend/routers/localization.py.
 */
export interface SourcePoint {
  /** MNI-152 X coordinate (mm, left–right; negative = left hemisphere). */
  x: number;
  /** MNI-152 Y coordinate (mm, anterior–posterior; positive = anterior). */
  y: number;
  /** MNI-152 Z coordinate (mm, inferior–superior; positive = superior). */
  z: number;
  /**
   * dSPM / sLORETA amplitude (normalised, unitless).
   * Higher = stronger activation.  Used for both colour and sphere size.
   */
  amplitude: number;
  /** Optional hemisphere tag from the backend's greedy NMS step. */
  hemisphere?: 'lh' | 'rh';
}

/** Tuning parameters for the overlay. All sizes in millimetres. */
export interface SourceLocalizationOptions {
  /**
   * Sphere radius (mm) for the source with the lowest amplitude.
   * @default 2.0
   */
  minSphereRadius?: number;
  /**
   * Sphere radius (mm) for the source with the highest amplitude.
   * @default 7.0
   */
  maxSphereRadius?: number;
  /**
   * Number of longitudinal divisions on each sphere.
   * Higher = smoother, but more GPU geometry.
   * @default 16
   */
  thetaResolution?: number;
  /**
   * Number of latitudinal divisions on each sphere.
   * @default 12
   */
  phiResolution?: number;
  /**
   * Override the minimum amplitude for the colour/size mapping.
   * Useful when comparing multiple localization runs on the same scale.
   * When omitted, derived from the current source set.
   */
  ampMin?: number;
  /**
   * Override the maximum amplitude for the colour/size mapping.
   * When omitted, derived from the current source set.
   */
  ampMax?: number;
  /**
   * Optional 4×4 row-major affine (see mniCoordinateTransform.ts).
   * Defaults to identity (MNI == vtk world — correct for MNI-registered scans).
   */
  mniToWorldMatrix?: Matrix4x4;
}

/**
 * The live bundle returned by buildSourceLocalizationOverlay.
 *
 * Add `actor` to `ctx.volumeRenderer`, then call `dispose()` before removing
 * it to release vtk.js pipeline objects.
 */
export interface SourceLocalizationBundle {
  /** vtk.js actor — add to `ctx.volumeRenderer` to make it visible. */
  actor: ReturnType<typeof vtkActor.newInstance>;
  /**
   * Replace the displayed source set without rebuilding the full pipeline.
   * Updates the vtkPolyData points + scalar arrays in-place and triggers
   * the vtk.js dirty chain (polydata.modified() → mapper re-renders).
   * Caller must call `ctx.renderWindow.render()` afterwards.
   */
  updateSources(sources: SourcePoint[], options?: SourceLocalizationOptions): void;
  /**
   * Show or hide the entire overlay.
   * Caller must call `ctx.renderWindow.render()` afterwards.
   */
  setVisible(visible: boolean): void;
  /**
   * Release all vtk.js pipeline objects.
   * Always call this before removing the actor from the renderer.
   */
  dispose(): void;
}

// ── Factory ───────────────────────────────────────────────────────────────────

/**
 * Build a vtk.js Glyph3DMapper pipeline that renders the given source points
 * as amplitude-coloured, amplitude-scaled spheres in MNI / vtk world space.
 *
 * @param sources  Peak source locations from POST /api/eeg/localize.
 * @param options  Optional tuning parameters (sphere size range, resolution,
 *                 amplitude overrides, world transform).
 * @returns        Bundle with the actor and lifecycle helpers.
 */
export function buildSourceLocalizationOverlay(
  sources: SourcePoint[],
  options: SourceLocalizationOptions = {},
): SourceLocalizationBundle {

  // ── 1. Sphere glyph template ─────────────────────────────────────────────
  // Unit-radius sphere — each instance is scaled by the GlyphRadius array.
  // Centred at origin (0, 0, 0) so the Glyph3DMapper positions it correctly.
  const sphere = vtkSphereSource.newInstance({
    radius: 1.0,
    thetaResolution: options.thetaResolution ?? 16,
    phiResolution:   options.phiResolution   ?? 12,
    center: [0, 0, 0],
  });

  // ── 2. vtkPolyData holding the point cloud ───────────────────────────────
  // vtkPolyData holds the per-point coordinates as a vtkDataArray with 3
  // components (XYZ).  We use setPoints() with a plain vtkDataArray because
  // vtkPoints is a thin type alias and vtk.js accepts any 3-component array.
  const polyData = vtkPolyData.newInstance();
  // Placeholder empty array — replaced by _populatePolyData below.
  const pts = vtkDataArray.newInstance({ numberOfComponents: 3, values: new Float64Array(0) });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (polyData as any).setPoints(pts);

  // ── 3. Mapper ────────────────────────────────────────────────────────────
  const mapper = vtkGlyph3DMapper.newInstance({
    scaling:     true,
    scaleFactor: 1.0,   // GlyphRadius values are already in mm
    scaleArray:  'GlyphRadius',
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const m = mapper as any;
  m.setInputData(polyData);
  m.setSourceConnection(sphere.getOutputPort());

  // Scale by the magnitude of the single-component GlyphRadius scalar array.
  mapper.setScaleModeToScaleByMagnitude();

  // Color by the Amplitude array, mapped through the hot CTF.
  m.setColorByArrayName('Amplitude');
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (mapper as any).setColorModeToMapScalars();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (mapper as any).setScalarModeToUsePointFieldData();

  // ── 4. Actor + Phong property ────────────────────────────────────────────
  const actor = vtkActor.newInstance();
  actor.setMapper(mapper as unknown as Parameters<typeof actor.setMapper>[0]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const prop = (actor as any).getProperty();
  prop.setAmbient(0.3);
  prop.setDiffuse(0.7);
  prop.setSpecular(0.4);
  prop.setSpecularPower(20.0);

  // ── 5. Populate with initial source data ────────────────────────────────
  _populatePolyData(polyData, pts, mapper, sources, options);

  // ── 6. Lifecycle helpers ──────────────────────────────────────────────────

  function updateSources(
    newSources: SourcePoint[],
    newOptions: SourceLocalizationOptions = {},
  ): void {
    _populatePolyData(polyData, pts, mapper, newSources, newOptions);
  }

  function setVisible(visible: boolean): void {
    actor.setVisibility(visible);
  }

  function dispose(): void {
    sphere.delete();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (polyData as any).delete();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (mapper   as any).delete();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (actor    as any).delete();
  }

  return { actor, updateSources, setVisible, dispose };
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/**
 * Write source coordinates and scalar arrays into the given vtkPolyData,
 * rebuild the colour transfer function for the current amplitude range, and
 * mark the polyData modified so vtk.js re-renders on the next render() call.
 *
 * Called at construction and again by updateSources() on live data refresh.
 */
function _populatePolyData(
  polyData:  ReturnType<typeof vtkPolyData.newInstance>,
  pts:       ReturnType<typeof vtkDataArray.newInstance>,
  mapper:    ReturnType<typeof vtkGlyph3DMapper.newInstance>,
  sources:   SourcePoint[],
  options:   SourceLocalizationOptions,
): void {
  const n = sources.length;

  // ── a. Derive amplitude range ────────────────────────────────────────────
  let ampMin = options.ampMin ?? Infinity;
  let ampMax = options.ampMax ?? -Infinity;
  if (options.ampMin === undefined || options.ampMax === undefined) {
    for (const s of sources) {
      if (s.amplitude < ampMin) ampMin = s.amplitude;
      if (s.amplitude > ampMax) ampMax = s.amplitude;
    }
  }
  // Guard against degenerate (single-source or all-equal) amplitude range.
  const ampSpan = ampMax - ampMin || 1.0;

  const minR = options.minSphereRadius ?? 2.0;
  const maxR = options.maxSphereRadius ?? 7.0;

  // ── b. Coordinate transform: MNI mm → vtk world mm ──────────────────────
  const M = buildMniToVtkWorldTransform(options.mniToWorldMatrix);
  const flatCoords = transformSourcesToFlat(M, sources);

  // ── c. Fill vtkPoints ────────────────────────────────────────────────────
  // Replace the entire DataArray backing the points.  setTuples() writes
  // the flat [x0,y0,z0, x1,y1,z1, …] Float64Array starting at tuple 0.
  // We resize first so the internal buffer is large enough.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (pts as any).setData(flatCoords, 3);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (polyData as any).setPoints(pts);

  // ── d. Build amplitude and radius scalar arrays ──────────────────────────
  const amplitudeArr = new Float32Array(n);
  const radiusArr    = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const amp    = sources[i].amplitude;
    const t      = (amp - ampMin) / ampSpan;   // normalised 0→1
    amplitudeArr[i] = amp;
    radiusArr[i]    = minR + t * (maxR - minR);
  }

  const ampScalars = vtkDataArray.newInstance({
    name:               'Amplitude',
    numberOfComponents: 1,
    values:             amplitudeArr,
  });
  const radiusScalars = vtkDataArray.newInstance({
    name:               'GlyphRadius',
    numberOfComponents: 1,
    values:             radiusArr,
  });

  // Replace existing arrays (guarantees vtk.js sees the new data).
  const pd = polyData.getPointData();
  pd.setScalars(ampScalars);                    // active scalars → Amplitude
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (pd as any).addArray(radiusScalars);           // named array for scale

  // ── e. Rebuild the hot CTF for the current amplitude range ───────────────
  const ctf = buildHotColorTransferFunction({ minValue: ampMin, maxValue: ampMax });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (mapper as any).setLookupTable(ctf);
  // Pin the CTF's own scalar range so the mapper does not override it.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (mapper as any).setUseLookupTableScalarRange(true);

  // ── f. Propagate dirty state up the pipeline ─────────────────────────────
  // setData() on vtkPoints increments its mtime automatically.
  // Replacing the scalar arrays increments pointData mtime.
  // Calling polyData.modified() propagates to the mapper, which schedules
  // a full GPU buffer re-upload on the next renderWindow.render() call.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (polyData as any).modified();
}
