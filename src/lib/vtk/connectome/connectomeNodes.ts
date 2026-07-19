/**
 * connectomeNodes.ts — 3-D sphere overlay for connectome nodes
 * ─────────────────────────────────────────────────────────────
 *
 * PIPELINE OVERVIEW
 * ──────────────────
 *
 *   nodes dict   (ConnectomeNode[])
 *         │
 *         ▼
 *   coordsFlat   Float32Array(N × 3)    — centre-of-mass positions
 *   colorsFlat   Uint8Array(N × 3)      — SYNTHSEG RGB per node
 *   radiusArr    Float32Array(N)        — betweenness-scaled radii
 *         │
 *         ▼  vtkDataArray (3-component, Float32) → pts
 *         ▼  vtkDataArray 'NodeColor'  (3-component, Uint8) → active scalars
 *         ▼  vtkDataArray 'GlyphRadius' (1-component, Float32) → named array
 *         │
 *         ▼
 *   vtkPolyData
 *     ├─ setPoints(pts)
 *     ├─ getPointData().setScalars(colorArray)    active = NodeColor
 *     └─ getPointData().addArray(radiusArray)     named  = GlyphRadius
 *         │
 *         ▼
 *   vtkGlyph3DMapper
 *     ├─ setInputData(polyData)
 *     ├─ setSourceConnection(sphere.getOutputPort())
 *     ├─ scaleArray  = 'GlyphRadius'       uses radius per glyph
 *     ├─ setScaleModeToScaleByMagnitude()  scale = radius × scaleFactor(1.0)
 *     ├─ setColorModeToDirectScalars()     use Uint8 RGB unchanged, no CTF
 *     ├─ setScalarModeToUsePointData()     color from active point scalars
 *     └─ setScalarVisibility(true)
 *         │
 *         ▼
 *   vtkActor  (opacity = nodeOpacity, Phong shading ON for spheres)
 *
 * BETWEENNESS → RADIUS SCALING
 * ──────────────────────────────
 * Radius is a linear interpolation between minNodeRadius and maxNodeRadius:
 *
 *   t      = (B(v) - B_min) / (B_max - B_min)   ∈ [0, 1]
 *   radius = minR + t × (maxR - minR)
 *
 * where B(v) is the normalised betweenness centrality of node v.
 * A degenerate range (all nodes identical betweenness) uses t = 0 for all,
 * so every node renders at minNodeRadius.
 *
 * COLOR
 * ──────
 * Each node's color is the SYNTHSEG palette color for its FreeSurfer label.
 * Colors are stored as Uint8Array and passed through with
 * setColorModeToDirectScalars() — no colour transfer function is used.
 * This matches the segmentation overlay color exactly (same RGBA LUT).
 */

import vtkActor         from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkGlyph3DMapper from '@kitware/vtk.js/Rendering/Core/Glyph3DMapper';
import vtkSphereSource  from '@kitware/vtk.js/Filters/Sources/SphereSource';
import vtkPolyData      from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkDataArray     from '@kitware/vtk.js/Common/Core/DataArray';

import { labelToRgb }          from './connectomeLabelColor';
import type {
  ConnectomeNode,
  ConnectomeOptions,
  ConnectomeNodeBundle,
} from './connectomeTypes';

/**
 * Build a vtk.js Glyph3DMapper pipeline that renders each connectome node as
 * a betweenness-scaled, SYNTHSEG-colored sphere in RAS mm world space.
 *
 * @param nodes    Node dictionary from ConnectomeApiResponse.nodes.
 * @param options  Optional visual tuning (radius range, resolution, opacity).
 * @returns        ConnectomeActorBundle with actor + lifecycle helpers.
 */
export function buildConnectomeNodes(
  nodes:   Record<string, ConnectomeNode>,
  options: ConnectomeOptions = {},
): ConnectomeNodeBundle {

  // ── Configuration defaults ────────────────────────────────────────────────
  const minR       = options.minNodeRadius  ?? 2.0;
  const maxR       = options.maxNodeRadius  ?? 8.0;
  const resolution = options.nodeResolution ?? 14;
  const opacity    = options.nodeOpacity    ?? 0.92;

  // ── Flatten node entries for indexed access ───────────────────────────────
  // Object.entries preserves insertion order for string-keyed objects in V8,
  // but the order here doesn't matter — we only need (label, node) pairs.
  const nodeEntries = Object.entries(nodes) as [string, ConnectomeNode][];
  const n = nodeEntries.length;

  // ── Derive betweenness range for radius normalisation ─────────────────────
  let bMin = Infinity;
  let bMax = -Infinity;
  for (const [, node] of nodeEntries) {
    if (node.betweenness < bMin) bMin = node.betweenness;
    if (node.betweenness > bMax) bMax = node.betweenness;
  }
  // Guard: if all nodes share the same betweenness (e.g. single-node graph),
  // bSpan = 0 would cause division-by-zero.  Map everything to t = 0.
  const bSpan = bMax - bMin || 1.0;

  // ── Allocate flat buffers ─────────────────────────────────────────────────
  const coordsFlat = new Float32Array(n * 3);  // (x, y, z) per node
  const radiusArr  = new Float32Array(n);       // mm radius per node
  const colorsFlat = new Uint8Array(n * 3);     // (R, G, B) per node

  // ── Fill buffers ──────────────────────────────────────────────────────────
  for (let i = 0; i < n; i++) {
    const [label, node] = nodeEntries[i]!;
    const [x, y, z]     = node.center_of_mass;

    // World-space position (RAS mm — already in vtk frame).
    coordsFlat[i * 3    ] = x;
    coordsFlat[i * 3 + 1] = y;
    coordsFlat[i * 3 + 2] = z;

    // Betweenness → radius via linear interpolation.
    const t       = (node.betweenness - bMin) / bSpan;   // normalised [0,1]
    radiusArr[i]  = minR + t * (maxR - minR);

    // SYNTHSEG palette color (matches segmentation overlay).
    const [r, g, b]       = labelToRgb(label);
    colorsFlat[i * 3    ] = r;
    colorsFlat[i * 3 + 1] = g;
    colorsFlat[i * 3 + 2] = b;
  }

  // ── vtkDataArrays ──────────────────────────────────────────────────────────
  // Point positions — 3-component Float32 (same pattern as sourceLocalizationOverlay).
  const pts = vtkDataArray.newInstance({
    numberOfComponents: 3,
    values: coordsFlat,
  });

  // Active scalars: Uint8 RGB — interpreted as direct colors by the mapper
  // when setColorModeToDirectScalars() is active.
  const colorArray = vtkDataArray.newInstance({
    name:               'NodeColor',
    numberOfComponents: 3,
    values:             colorsFlat,
    dataType:           'Uint8Array',
  });

  // Named (non-active) scalar for per-glyph scaling.
  const radiusArray = vtkDataArray.newInstance({
    name:               'GlyphRadius',
    numberOfComponents: 1,
    values:             radiusArr,
  });

  // ── vtkPolyData ───────────────────────────────────────────────────────────
  const polyData = vtkPolyData.newInstance();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (polyData as any).setPoints(pts);

  // NodeColor → active scalars (drives direct color without a CTF).
  polyData.getPointData().setScalars(colorArray);
  // GlyphRadius → non-active named array (consumed by mapper's scaleArray).
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (polyData.getPointData() as any).addArray(radiusArray);

  // ── Sphere glyph template (unit radius — scaled per-instance) ─────────────
  // A single vtkSphereSource is shared across all glyph instances.  The
  // Glyph3DMapper positions and scales each copy using the point array.
  const sphere = vtkSphereSource.newInstance({
    radius:          1.0,   // unit sphere — GlyphRadius scales each instance
    thetaResolution: resolution,
    phiResolution:   resolution,
    center: [0, 0, 0],
  });

  // ── vtkGlyph3DMapper ───────────────────────────────────────────────────────
  const mapper = vtkGlyph3DMapper.newInstance({
    scaling:     true,
    scaleFactor: 1.0,       // GlyphRadius values are already in mm
    scaleArray:  'GlyphRadius',
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const m = mapper as any;
  m.setInputData(polyData);
  m.setSourceConnection(sphere.getOutputPort());

  // Scale each sphere instance by the magnitude of its GlyphRadius scalar.
  mapper.setScaleModeToScaleByMagnitude();

  // Direct-scalar color: pass Uint8 RGB bytes straight to the GPU without a CTF.
  // This is the correct mode when the color is pre-computed per-point rather
  // than derived from a numeric scalar through a transfer function.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (mapper as any).setColorModeToDirectScalars();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (mapper as any).setScalarModeToUsePointData();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (mapper as any).setScalarVisibility(true);

  // ── vtkActor ───────────────────────────────────────────────────────────────
  const actor = vtkActor.newInstance();
  actor.setMapper(mapper as unknown as Parameters<typeof actor.setMapper>[0]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const prop = (actor as any).getProperty();
  prop.setOpacity(opacity);
  // Phong shading ON — spheres have surface normals, so shading improves depth cues.
  prop.setAmbient(0.25);
  prop.setDiffuse(0.70);
  prop.setSpecular(0.35);
  prop.setSpecularPower(15.0);

  // ── Lifecycle helpers ──────────────────────────────────────────────────────

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

  // ── In-place centrality filter ─────────────────────────────────────────────
  //
  // How it works (no actor or mapper rebuild):
  //   1. Walk nodeEntries[].  For nodes below minBetweenness, zero the radius.
  //      For nodes at or above it, recompute from the original betweenness
  //      value using the same [bMin, bSpan, minR, maxR] captured at build time.
  //   2. Replace the backing Float32Array of the GlyphRadius vtkDataArray via
  //      setData().  vtk.js marks the DataArray mtime dirty automatically.
  //   3. Call polyData.modified() to propagate the dirty flag up through the
  //      mapper to the renderer.  The next renderWindow.render() call picks it
  //      up and re-uploads the buffer to the GPU without rebuilding any pipeline
  //      objects.
  //
  // Caller is responsible for calling ctx.renderWindow.render() after this.
  //
  function filterByCentrality(minBetweenness: number): void {
    for (let i = 0; i < n; i++) {
      const [, node] = nodeEntries[i]!;
      if (node.betweenness < minBetweenness) {
        // Zero radius → glyph becomes a degenerate sphere with no visible geometry.
        radiusArr[i] = 0;
      } else {
        // Restore the original betweenness-proportional radius.
        const t     = (node.betweenness - bMin) / bSpan;
        radiusArr[i] = minR + t * (maxR - minR);
      }
    }
    // Swap the backing data buffer — vtk.js tracks the change via mtime.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (radiusArray as any).setData(radiusArr);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (polyData    as any).modified();
  }

  return { actor, setVisible, dispose, filterByCentrality };
}
