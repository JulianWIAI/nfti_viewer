/**
 * connectomeTypes.ts — Shared TypeScript type definitions for the 3-D
 * connectome node-and-edge vtk.js rendering pipeline.
 *
 * DATA MODEL
 * ───────────
 * The backend endpoint POST /api/connectomics/matrix returns a JSON payload
 * with three top-level keys:
 *
 *   matrix   : number[][]
 *     Structural connectivity matrix A where A[i][j] is the number of DTI
 *     streamlines connecting region node_ids[i] to region node_ids[j].
 *     The matrix is symmetric for undirected tractography (A[i][j] = A[j][i]),
 *     but this is NOT assumed — we take max(A[i][j], A[j][i]) when drawing
 *     undirected edges to handle minor asymmetries from the tracking algorithm.
 *
 *   nodes    : Record<string, ConnectomeNode>
 *     Dictionary keyed by FreeSurfer integer label ID (as a decimal string).
 *     Only labels that have at least one streamline to any other region are
 *     included — isolated regions are absent.
 *
 *   node_ids : string[]
 *     Ordered list of FreeSurfer label strings whose index corresponds to the
 *     row/column index in `matrix`.  Required because plain JS objects do not
 *     guarantee insertion order when keys are integers.
 *
 * GRAPH METRICS
 * ──────────────
 * Graph metrics are computed server-side from the weighted adjacency matrix
 * using NetworkX.  The metrics stored per node are:
 *
 *   degree       : number of edges incident to this node (after thresholding)
 *   betweenness  : normalised betweenness centrality ∈ [0, 1]
 *                  B(v) = Σ_{s≠v≠t} σ(s,t|v) / σ(s,t)
 *                  where σ(s,t) = total shortest paths from s to t in the
 *                  weighted graph, σ(s,t|v) = those that pass through v.
 *                  High betweenness marks structural hubs (corpus callosum,
 *                  thalamus, putamen) that carry most inter-regional traffic.
 *   center_of_mass : [x, y, z] in NIfTI world (RAS mm) space — identical
 *                  frame to the vtk.js volume renderer, so no further
 *                  coordinate transform is needed.
 */

import type vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';

// ── Per-node data from the API ────────────────────────────────────────────────

/**
 * Graph-theoretic properties and spatial position for one brain region.
 * Matches the per-entry schema of the `nodes` dict from the backend.
 */
export interface ConnectomeNode {
  /** Number of suprathreshold connections incident to this region. */
  degree: number;
  /**
   * Normalised betweenness centrality ∈ [0, 1].
   * Drives sphere radius: higher betweenness → larger, more prominent sphere.
   */
  betweenness: number;
  /**
   * Centre of mass of the SynthSeg label mask in RAS mm world coordinates.
   * These coordinates are in the same frame as the vtk.js volume renderer —
   * no additional affine transform is required.
   */
  center_of_mass: [number, number, number];
}

// ── Full API response ─────────────────────────────────────────────────────────

/**
 * Complete JSON payload from POST /api/connectomics/matrix.
 *
 * `node_ids[i]` gives the FreeSurfer label string for row/column i of
 * `matrix`.  This ordering is required for correct index-to-region lookup
 * when iterating over the matrix to build edge geometry.
 */
export interface ConnectomeApiResponse {
  /** Symmetric structural connectivity matrix; entry [i][j] = fibre count. */
  matrix:   number[][];
  /** FreeSurfer label → node properties, for all regions with ≥ 1 connection. */
  nodes:    Record<string, ConnectomeNode>;
  /** Ordered FreeSurfer label strings that index the rows/columns of `matrix`. */
  node_ids: string[];
}

// ── Rendering options ─────────────────────────────────────────────────────────

/**
 * Tuning knobs for the visual appearance of the connectome overlay.
 * All defaults produce a visually balanced, publication-quality rendering.
 */
export interface ConnectomeOptions {
  /**
   * Minimum fibre count required to draw an edge (A[i][j] threshold).
   *
   * MATHEMATICAL RATIONALE
   * ──────────────────────
   * Even with three-stage compression, DTI tractography produces spurious
   * connections from tracking noise at WM/GM boundaries (false positives).
   * Empirically, connections with < 10 streamlines arise almost entirely
   * from:
   *   • Gyral bias: the gyral crown is always closer to the WM skeleton
   *     and accumulates spurious short connections.
   *   • Anatomical adjacency: neighbouring voxels that share a thin WM
   *     bridge generate 1–5 streamlines even without a real tract.
   *
   * A threshold of 10 removes ~70% of edges from a whole-brain connectome
   * while retaining > 99% of the anatomically verified connections (which
   * are typically > 50 streamlines in a clinical DTI acquisition).
   *
   * Increase to 50–100 for a sparser, more legible graph; decrease to 5
   * for exploratory analysis where missing weak connections matters more
   * than visual clarity.
   *
   * @default 10
   */
  edgeThreshold?: number;

  /**
   * Sphere radius (mm) for the node with the lowest betweenness centrality.
   * @default 2.0
   */
  minNodeRadius?: number;

  /**
   * Sphere radius (mm) for the node with the highest betweenness centrality.
   * Large radius (8–10 mm) makes structural hubs visually dominant, which is
   * the primary diagnostic value of betweenness centrality visualisation.
   * @default 8.0
   */
  maxNodeRadius?: number;

  /**
   * Sphere tessellation resolution (theta and phi divisions).
   * Higher = smoother spheres but more GPU geometry per node.
   * @default 14
   */
  nodeResolution?: number;

  /** Global opacity for the node actor (0 = transparent, 1 = opaque). @default 0.92 */
  nodeOpacity?: number;

  /** Global opacity for the edge actor. @default 0.70 */
  edgeOpacity?: number;

  /**
   * Rendered line width (screen pixels) for edges.
   * vtk.js does not support per-line width; this applies uniformly.
   * Edge weight is instead encoded as colour intensity (pale → vivid blue).
   * @default 1.5
   */
  edgeLineWidth?: number;
}

// ── Live overlay handles ──────────────────────────────────────────────────────

/**
 * Handle to a single-actor overlay (nodes or edges separately).
 *
 * Add `actor` to `ctx.volumeRenderer`, call `dispose()` before removing it.
 */
export interface ConnectomeActorBundle {
  /** Add to ctx.volumeRenderer.addActor(bundle.actor). */
  actor:      ReturnType<typeof vtkActor.newInstance>;
  /** Toggle visibility without rebuilding.  Caller must call renderWindow.render(). */
  setVisible: (visible: boolean) => void;
  /** Release all vtk.js pipeline objects.  Call before removeActor(). */
  dispose:    () => void;
}

/**
 * Extended bundle returned by buildConnectomeNodes().
 *
 * `filterByCentrality` performs an in-place update of the GlyphRadius array —
 * nodes below `minBetweenness` have their radius zeroed (invisible), others are
 * restored to their original betweenness-scaled radius.
 * Caller must call `ctx.renderWindow.render()` after this returns.
 *
 * NO actor or mapper rebuild is performed — only the underlying Float32Array
 * backing the GlyphRadius vtkDataArray is modified and vtk.js is notified via
 * `polyData.modified()`.
 */
export interface ConnectomeNodeBundle extends ConnectomeActorBundle {
  /**
   * Hide all nodes whose betweenness centrality is below `minBetweenness`.
   * Pass 0 to restore all nodes to full visibility.
   * Caller must call `ctx.renderWindow.render()` after this returns.
   */
  filterByCentrality: (minBetweenness: number) => void;
}

/**
 * Extended bundle returned by buildConnectomeEdges().
 *
 * `filterByWeight` rebuilds the vtkCellArray and cell-scalar array in-place
 * so that only edges with `max(A[i][j], A[j][i]) > newThreshold` are drawn.
 * The colour transfer function range is also updated to the new [newThreshold,
 * maxWeight] span.  The vtkPolyData, vtkMapper, and vtkActor are NOT recreated.
 * Caller must call `ctx.renderWindow.render()` after this returns.
 *
 * `highlightEdge` switches the mapper between two coloring modes:
 *
 *   null      → normal mode: all suprathreshold edges drawn with the blue CTF,
 *               actor opacity restored to edgeOpacity.
 *
 *   non-null  → highlight mode: selected edge is accent-green (full opacity),
 *               all other edges are dim grey (~5% opacity).  Internally uses
 *               per-cell RGBA Uint8 scalars with setColorModeToDirectScalars().
 *
 * Caller must call `ctx.renderWindow.render()` after this returns.
 */
export interface ConnectomeEdgeBundle extends ConnectomeActorBundle {
  /**
   * Update the displayed edges to those with fibre count > `newThreshold`.
   * Caller must call `ctx.renderWindow.render()` after this returns.
   */
  filterByWeight: (newThreshold: number) => void;

  /**
   * Enter or exit edge-highlight mode.
   *
   * @param sourceIdx  Global point index of the source region (= matrix row index
   *                   in `node_ids`), OR null to exit highlight mode.
   * @param targetIdx  Global point index of the target region (= matrix col index),
   *                   OR null to exit highlight mode.
   *
   * When both are null the mapper reverts to normal CTF-mapped colours.
   * Caller must call `ctx.renderWindow.render()` after this returns.
   */
  highlightEdge: (sourceIdx: number | null, targetIdx: number | null) => void;
}

/**
 * Combined handle returned by buildConnectomeOverlay().
 * Provides independent toggle for nodes and edges plus a convenience
 * method that toggles both simultaneously.
 *
 * USAGE
 * ──────
 *   const bundle = buildConnectomeOverlay(apiResponse);
 *   ctx.volumeRenderer.addActor(bundle.nodeBundle.actor);
 *   ctx.volumeRenderer.addActor(bundle.edgeBundle.actor);
 *   ctx.renderWindow.render();
 *
 *   // Dynamic threshold update (no actor rebuild):
 *   bundle.edgeBundle.filterByWeight(25);
 *   bundle.nodeBundle.filterByCentrality(0.10);
 *   ctx.renderWindow.render();
 *
 *   // Full teardown:
 *   bundle.dispose();
 *   ctx.volumeRenderer.removeActor(bundle.nodeBundle.actor);
 *   ctx.volumeRenderer.removeActor(bundle.edgeBundle.actor);
 *   ctx.renderWindow.render();
 */
export interface ConnectomeBundle {
  /** Node sphere glyph overlay with in-place centrality filter. */
  nodeBundle:  ConnectomeNodeBundle;
  /** Edge line overlay with in-place weight threshold filter. */
  edgeBundle:  ConnectomeEdgeBundle;
  /** Hide or show both nodes and edges simultaneously. */
  setVisible:  (visible: boolean) => void;
  /** Dispose both actors; always call before removing from renderer. */
  dispose:     () => void;
}
