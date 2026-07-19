/**
 * tractographyOverlay.ts — vtk.js pipeline for rendering DTI streamlines
 * ────────────────────────────────────────────────────────────────────────
 *
 * PIPELINE OVERVIEW
 * ──────────────────
 *
 *   StreamlineArray (JSON from backend)
 *         │
 *         ▼  buildTractographyBuffers()          [tractographyGeometry.ts]
 *   Float32Array  pointsFlat   [x₀,y₀,z₀, x₁,y₁,z₁, …]
 *   Uint32Array   cellData     [n₀, id₀₀, id₀₁, …, n₁, id₁₀, …]
 *   Uint8Array    colorsFlat   [R₀,G₀,B₀, R₁,G₁,B₁, …]   (DEC)
 *         │
 *         ▼
 *   vtkDataArray  (pts)        numberOfComponents=3, Float32Array
 *   vtkCellArray  (lines)      linesArray.setData(cellData)
 *   vtkDataArray  (colors)     name='DEC', numberOfComponents=3, Uint8Array
 *         │
 *         ▼
 *   vtkPolyData
 *     ├─ setPoints(pts)
 *     ├─ setLines(lines)
 *     └─ getPointData().setScalars(colors)
 *         │
 *         ▼
 *   vtkMapper
 *     ├─ setInputData(polyData)
 *     ├─ setColorModeToDirectScalars()   ← interpret Uint8 RGB as-is, no CTF
 *     ├─ setScalarModeToUsePointData()   ← colours come from point attributes
 *     └─ setScalarVisibility(true)
 *         │
 *         ▼
 *   vtkActor
 *     └─ property: opacity=0.8, lineWidth=2.0, lighting=false
 *         │
 *         ▼
 *   ctx.volumeRenderer.addActor(bundle.actor)
 *
 * DIRECT-SCALAR COLOR MODE
 * ──────────────────────────
 * `setColorModeToDirectScalars()` tells the mapper to treat the Uint8Array
 * values as literal [R, G, B] bytes without passing them through a colour
 * transfer function.  This is the correct mode for DEC because we have already
 * computed the desired colour in the CPU buffer — no additional mapping needed.
 *
 * LIGHTING
 * ─────────
 * Lines have no surface normal, so Phong shading is meaningless.  Disabling
 * lighting (`setLighting(false)` on the actor property) renders lines with
 * their unshaded DEC colour, which produces the crisp, high-contrast appearance
 * expected in DTI tractography visualisations.
 *
 * COORDINATE FRAME
 * ─────────────────
 * streamlines must be in vtk world space (RAS mm), identical to the frame of
 * the existing structural MRI renderer.  The backend serialises coordinates in
 * the NIfTI sform world frame, which is the same frame vtk.js uses for the
 * volume after loading the NIfTI through vtkITKHelper — no transform needed.
 *
 * USAGE
 * ──────
 *   import { buildTractographyOverlay } from '../../lib/vtk/tractography/tractographyOverlay';
 *
 *   const bundle = buildTractographyOverlay(apiResponse.streamlines);
 *   ctx.volumeRenderer.addActor(bundle.actor);
 *   ctx.renderWindow.render();
 *
 *   // On unmount or new upload:
 *   bundle.dispose();
 *   ctx.volumeRenderer.removeActor(bundle.actor);
 *   ctx.renderWindow.render();
 */

import vtkActor     from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper    from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkPolyData  from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkCellArray from '@kitware/vtk.js/Common/Core/CellArray';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';

import { buildTractographyBuffers }   from './tractographyGeometry';
import type { StreamlineArray, TractographyBundle } from './tractographyTypes';

/**
 * Build a vtk.js vtkPolyData → vtkMapper → vtkActor pipeline that renders the
 * given streamlines as Direction-Encoded Colour lines in 3-D world space.
 *
 * The entire pipeline is constructed on the CPU (in JS) and handed to the GPU
 * on the first call to ctx.renderWindow.render().  The function itself takes
 * roughly 5–15 ms for a 10 000-streamline tractogram.
 *
 * @param streamlines  Nested coordinate array from POST /api/dti/tractography.
 * @returns            TractographyBundle with actor + lifecycle helpers.
 */
export function buildTractographyOverlay(
  streamlines: StreamlineArray,
): TractographyBundle {

  // ── 1. Build flat GPU buffers (two-pass, single allocation) ─────────────
  const { pointsFlat, cellData, colorsFlat, stats } =
    buildTractographyBuffers(streamlines);

  // ── 2. vtkPoints — 3-component Float32Array of (x, y, z) coordinates ────
  // We use vtkDataArray with numberOfComponents=3 instead of vtkPoints because
  // vtkPoints is a thin type alias; vtk.js accepts any 3-component DataArray as
  // the point set of a vtkPolyData (see sourceLocalizationOverlay.ts precedent).
  const pts = vtkDataArray.newInstance({
    numberOfComponents: 3,
    values: pointsFlat,
  });

  // ── 3. vtkCellArray — line-cell connectivity in VTK legacy flat format ───
  // Format: [n₀, id₀₀, …, id₀_{n₀-1}, n₁, id₁₀, …]
  // setData() replaces the internal buffer directly — no copy.
  const linesArray = vtkCellArray.newInstance();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (linesArray as any).setData(cellData);

  // ── 4. vtkDataArray — per-point DEC RGB colours ──────────────────────────
  // numberOfComponents=3 matches the [R,G,B] layout of colorsFlat.
  // The 'DEC' name is not strictly required for direct-scalar mode but helps
  // with debugging (visible in ParaView-style pipeline inspectors).
  const colorArray = vtkDataArray.newInstance({
    name:               'DEC',
    numberOfComponents: 3,
    values:             colorsFlat,
    dataType:           'Uint8Array',
  });

  // ── 5. vtkPolyData — assemble points + line topology + point colors ──────
  const polyData = vtkPolyData.newInstance();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (polyData as any).setPoints(pts);
  polyData.setLines(linesArray);
  // setScalars() registers colorArray as the active scalars — the mapper will
  // pick it up automatically under ScalarModeToUsePointData.
  polyData.getPointData().setScalars(colorArray);

  // ── 6. vtkMapper — configure direct-scalar colour mode ───────────────────
  const mapper = vtkMapper.newInstance();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (mapper as any).setInputData(polyData);

  // setColorModeToDirectScalars(): pass Uint8 RGB through to the GPU unchanged,
  // bypassing the colour transfer function (CTF).  Required for DEC rendering.
  mapper.setColorModeToDirectScalars();
  // Pull colours from point-data scalars (the 'DEC' array set above).
  mapper.setScalarModeToUsePointData();
  // Enable scalar-driven colouring.
  mapper.setScalarVisibility(true);

  // ── 7. vtkActor — visual appearance ──────────────────────────────────────
  const actor = vtkActor.newInstance();
  actor.setMapper(mapper as unknown as Parameters<typeof actor.setMapper>[0]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const prop = (actor as any).getProperty();
  // 80% opacity: allow the structural MRI to show through dense bundles.
  prop.setOpacity(0.8);
  // 2 px wide lines — visible without obscuring anatomy at standard zoom.
  prop.setLineWidth(2.0);
  // Disable Phong shading: lines have no surface normal; flat colour is correct.
  prop.setLighting(false);

  // ── 8. Lifecycle helpers ──────────────────────────────────────────────────

  function setVisible(visible: boolean): void {
    actor.setVisibility(visible);
  }

  function dispose(): void {
    // Release GPU pipeline objects.  Order: actor → mapper → polyData.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (actor    as any).delete();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (mapper   as any).delete();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (polyData as any).delete();
  }

  return { actor, setVisible, dispose, stats };
}
