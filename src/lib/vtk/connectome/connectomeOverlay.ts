/**
 * connectomeOverlay.ts — Top-level factory for the 3-D connectome overlay
 * ─────────────────────────────────────────────────────────────────────────
 *
 * ARCHITECTURE
 * ─────────────
 * This module is the public entry point for the connectome vtk.js pipeline.
 * It delegates to two specialised builders:
 *
 *   buildConnectomeNodes()   [connectomeNodes.ts]
 *     → vtkGlyph3DMapper spheres, colored by SYNTHSEG label, scaled by
 *       betweenness centrality.
 *
 *   buildConnectomeEdges()   [connectomeEdges.ts]
 *     → vtkPolyData lines, drawn only for matrix entries above edgeThreshold,
 *       colored by fibre count through a sequential blue CTF.
 *
 * The two actors are independent — they can be toggled on/off separately
 * in the UI without affecting each other.  Both are in the same vtk world
 * space (RAS mm, same as the structural MRI volume renderer).
 *
 * USAGE
 * ──────
 *   import { buildConnectomeOverlay } from '../../lib/vtk/connectome/connectomeOverlay';
 *
 *   // On receiving the API response:
 *   const bundle = buildConnectomeOverlay(response, { edgeThreshold: 20 });
 *   ctx.volumeRenderer.addActor(bundle.nodeBundle.actor);
 *   ctx.volumeRenderer.addActor(bundle.edgeBundle.actor);
 *   ctx.renderWindow.render();
 *
 *   // Toggle edges off (nodes stay visible):
 *   bundle.edgeBundle.setVisible(false);
 *   ctx.renderWindow.render();
 *
 *   // Toggle both off:
 *   bundle.setVisible(false);
 *   ctx.renderWindow.render();
 *
 *   // Full teardown (always call dispose before removeActor):
 *   bundle.dispose();
 *   ctx.volumeRenderer.removeActor(bundle.nodeBundle.actor);
 *   ctx.volumeRenderer.removeActor(bundle.edgeBundle.actor);
 *   ctx.renderWindow.render();
 *
 * COORDINATE FRAME
 * ─────────────────
 * The backend computes center_of_mass in the NIfTI sform world frame (RAS mm).
 * This is the same frame the volumetric MRI renderer uses after loading the
 * NIfTI through vtkITKHelper — no additional affine transform is needed.
 * The node positions and edge endpoints will overlay exactly on the structural
 * anatomy when the two actors are added to ctx.volumeRenderer.
 */

import { buildConnectomeNodes } from './connectomeNodes';
import { buildConnectomeEdges } from './connectomeEdges';
import type {
  ConnectomeApiResponse,
  ConnectomeOptions,
  ConnectomeBundle,
} from './connectomeTypes';

// Re-export all types from this module so callers only need one import.
export type {
  ConnectomeNode,
  ConnectomeApiResponse,
  ConnectomeOptions,
  ConnectomeActorBundle,
  ConnectomeNodeBundle,
  ConnectomeEdgeBundle,
  ConnectomeBundle,
} from './connectomeTypes';

/**
 * Build the complete 3-D connectome overlay (nodes + edges) from an API response.
 *
 * Both actors are constructed and returned together but are fully independent:
 * add them to the renderer separately so each can be individually removed.
 *
 * @param response  Full JSON payload from POST /api/connectomics/matrix.
 * @param options   Optional visual tuning applied to both sub-overlays.
 * @returns         ConnectomeBundle with node actor, edge actor, and helpers.
 */
export function buildConnectomeOverlay(
  response: ConnectomeApiResponse,
  options:  ConnectomeOptions = {},
): ConnectomeBundle {

  // ── Build node sphere overlay ─────────────────────────────────────────────
  // Positions = center_of_mass per node.
  // Color     = SYNTHSEG palette by FreeSurfer label.
  // Radius    = betweenness centrality → [minNodeRadius, maxNodeRadius].
  const nodeBundle = buildConnectomeNodes(response.nodes, options);

  // ── Build edge line overlay ───────────────────────────────────────────────
  // Only edges where max(A[i][j], A[j][i]) > edgeThreshold are drawn.
  // Color = fibre count mapped through pale-blue → cobalt sequential CTF.
  // node_ids provides the ordered index into the matrix rows/columns.
  const edgeBundle = buildConnectomeEdges(
    response.matrix,
    response.node_ids,
    response.nodes,
    options,
  );

  // ── Combined lifecycle helpers ─────────────────────────────────────────────

  /** Toggle both node spheres and edge lines simultaneously. */
  function setVisible(visible: boolean): void {
    nodeBundle.setVisible(visible);
    edgeBundle.setVisible(visible);
  }

  /** Release all vtk.js objects for both sub-overlays. */
  function dispose(): void {
    nodeBundle.dispose();
    edgeBundle.dispose();
  }

  return { nodeBundle, edgeBundle, setVisible, dispose };
}
