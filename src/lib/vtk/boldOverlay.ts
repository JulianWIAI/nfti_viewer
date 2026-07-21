/**
 * boldOverlay.ts — vtk.js point-cloud overlay for MEG source estimates / BOLD activation
 * ─────────────────────────────────────────────────────────────────────────────────────────
 *
 * Renders a set of 3-D source-space vertices (returned by POST /api/meg/source-estimate)
 * as colour-coded spheres using a vtkGlyph3DMapper pipeline:
 *
 *   vtkPolyData (points + scalars)
 *     → vtkGlyph3DMapper (glyph = small sphere, lookup = inferno CTF)
 *       → vtkActor (added to the 3-D renderer)
 *
 * The approach deliberately avoids vtkSphereSource per vertex because Glyph3DMapper
 * handles all N glyphs in a single draw call, which is GPU-friendly even at
 * ~10 000 cortical source points.
 *
 * DESIGN: immutable on creation, mutable amplitude range
 * ───────────────────────────────────────────────────────
 * The polyData (positions) is fixed once — re-creating the overlay for each
 * source estimate result would force a VBO upload every time.  Instead we
 * keep the actor alive and only replace the scalar array + colour-map range
 * via updateAmplitudes() when the backend returns a new result.
 *
 * USAGE
 * ──────
 *   import { buildBoldOverlay } from './boldOverlay';
 *
 *   const overlay = buildBoldOverlay(vertices, renderer);
 *   renderer.addActor(overlay.actor);
 *   renderWindow.render();
 *
 *   // When new source estimate arrives:
 *   overlay.updateAmplitudes(newVertices);
 *   renderWindow.render();
 *
 *   // To hide without removing:
 *   overlay.actor.setVisibility(false);
 */

import vtkPolyData    from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkDataArray   from '@kitware/vtk.js/Common/Core/DataArray';
import vtkActor       from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkSphereSource from '@kitware/vtk.js/Filters/Sources/SphereSource';
import { buildInfernoColorTransferFunction } from './infernoColorMap';
import type { SourceVertex } from '../../types/fmri.types';

// We use Glyph3DMapper if available in the vtk.js build, but fall back to
// a simple PointsMapper for environments where the Glyph filter is omitted.
// Dynamic import avoids hard-dep on the Glyph module.
let _Glyph3DMapper: any = null;
let _PointsMapper:  any = null;

async function _loadMappers() {
  try {
    const mod = await import('@kitware/vtk.js/Rendering/Core/Glyph3DMapper');
    _Glyph3DMapper = mod.default;
  } catch {
    const mod = await import('@kitware/vtk.js/Rendering/Core/Mapper');
    _PointsMapper = mod.default;
  }
}

// Pre-load mappers at module init time (fire-and-forget)
_loadMappers();

// ── Public types ─────────────────────────────────────────────────────────────

export interface BoldOverlayBundle {
  /** The vtk.js actor — add to a renderer and it becomes visible. */
  actor: ReturnType<typeof vtkActor.newInstance>;
  /**
   * Update the displayed activation amplitudes without recreating the overlay.
   * Also recalibrates the inferno colour map to the new peak amplitude.
   * Does NOT call renderWindow.render() — the caller is responsible.
   */
  updateAmplitudes(vertices: SourceVertex[]): void;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Build a Float32Array of [x, y, z] triples from a vertex list. */
function buildPointsArray(vertices: SourceVertex[]): Float32Array {
  const pts = new Float32Array(vertices.length * 3);
  for (let i = 0; i < vertices.length; i++) {
    pts[i * 3]     = vertices[i].x;
    pts[i * 3 + 1] = vertices[i].y;
    pts[i * 3 + 2] = vertices[i].z;
  }
  return pts;
}

/** Build a Float32Array of scalar amplitudes (one per vertex). */
function buildAmplitudesArray(vertices: SourceVertex[]): Float32Array {
  return Float32Array.from(vertices, (v) => v.amplitude);
}

// ── Factory ──────────────────────────────────────────────────────────────────

/**
 * Build a BOLD/source-estimate point-cloud overlay.
 *
 * @param vertices - Initial set of source vertices with amplitudes.
 * @returns BoldOverlayBundle with actor + updateAmplitudes().
 */
export function buildBoldOverlay(vertices: SourceVertex[]): BoldOverlayBundle {
  // ── vtkPolyData — position cloud ─────────────────────────────────────────
  const polyData = vtkPolyData.newInstance();

  // setData on vtkPoints expects a raw TypedArray (not a vtkDataArray wrapper)
  polyData.getPoints().setData(buildPointsArray(vertices), 3);

  // ── Scalar amplitude array (drives colour map) ────────────────────────────
  const amps = buildAmplitudesArray(vertices);
  const scalarArray = vtkDataArray.newInstance({
    name: 'Amplitude',
    numberOfComponents: 1,
    values: amps,
  });
  polyData.getPointData().setScalars(scalarArray);

  // ── Inferno colour-transfer function scaled to the amplitude range ─────────
  const peakAmp = amps.length ? Math.max(...Array.from(amps)) : 1.0;
  const ctf = buildInfernoColorTransferFunction({ minValue: 0, maxValue: peakAmp || 1 });

  // ── Glyph sphere source (radius ≈ 3 mm — anatomically appropriate) ────────
  const sphere = vtkSphereSource.newInstance({ radius: 3.0, thetaResolution: 8, phiResolution: 8 });

  // ── Actor ─────────────────────────────────────────────────────────────────
  const actor = vtkActor.newInstance();

  // Try Glyph3DMapper first; fall back to a simple Points mapper.
  // The type is intentionally loose (any) because the two mappers share
  // the setInputData / setLookupTable interface but differ in other methods.
  let mapper: any;

  if (_Glyph3DMapper) {
    mapper = _Glyph3DMapper.newInstance();
    mapper.setInputData(polyData);
    mapper.setSourceConnection(sphere.getOutputPort());
    mapper.setScaleModeToScaleByMagnitude();
    mapper.setScaleFactor(1.0);
    mapper.setColorModeToMapScalars();
    mapper.setLookupTable(ctf);
    mapper.setUseLookupTableScalarRange(true);
  } else {
    // Fallback: render points without glyphs
    const PointsMapper = _PointsMapper || (() => { throw new Error('No mapper available'); })();
    mapper = PointsMapper.newInstance();
    mapper.setInputData(polyData);
    mapper.setLookupTable(ctf);
    mapper.setUseLookupTableScalarRange(true);
  }

  actor.setMapper(mapper);
  actor.getProperty().setPointSize(6);

  // ── updateAmplitudes — called when the backend returns new source estimate ─
  function updateAmplitudes(newVertices: SourceVertex[]): void {
    // Update scalar amplitudes in-place (positions stay fixed)
    const newAmps = buildAmplitudesArray(newVertices);
    const newPeak = newAmps.length ? Math.max(...Array.from(newAmps)) : 1.0;

    // Replace the typed array backing the scalar DataArray.
    // vtkDataArray.setData() accepts a TypedArray + numberOfComponents,
    // but the TS types don't expose this overload — cast to bypass the check.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (scalarArray as unknown as { setData(d: Float32Array, nc: number): void }).setData(newAmps, 1);
    polyData.getPointData().setScalars(scalarArray);

    // Recalibrate the colour-transfer function to the new amplitude range
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ctf.removeAllPoints();
    const updatedCtf = buildInfernoColorTransferFunction({ minValue: 0, maxValue: newPeak || 1 });
    // Copy the updated CTF's control points into the existing instance so the
    // mapper reference stays valid (no mapper.setLookupTable re-call needed).
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const pts = (updatedCtf as any).getNodes ? (updatedCtf as any).getNodes() : [];
    if (pts.length > 0) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (ctf as any).setNodes(pts);
    }

    polyData.modified();
  }

  return { actor, updateAmplitudes };
}
