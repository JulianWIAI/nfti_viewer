/**
 * connectomePanelTypes.ts — Local type definitions for the ConnectomePanel UI
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * These types are intentionally local to the panel layer (not in the vtk lib)
 * because they describe ephemeral UI state: which node is selected, what the
 * current filter thresholds are.  They do not belong in the VTK pipeline types.
 */

import type { ConnectomeNode } from '../../../lib/vtk/connectome/connectomeTypes';

// ── Filter state (managed by useConnectomeFilters via useReducer) ─────────────

/**
 * The complete filter state for the connectome overlay.
 * Tracked in a single useReducer to make state transitions explicit and
 * debuggable (each dispatch has a typed action with a clear `type`).
 */
export interface ConnectomeFilterState {
  /**
   * Minimum fibre count required to display an edge.
   * Edges with `max(A[i][j], A[j][i]) <= edgeThreshold` are hidden.
   * Corresponds to the "Edge Threshold" range slider.
   * @default 10
   */
  edgeThreshold: number;

  /**
   * Minimum betweenness centrality for a node to be displayed.
   * Nodes with `betweenness < centralityMin` have their GlyphRadius zeroed.
   * Corresponds to the "Min. Betweenness" range slider.
   * @default 0   (show all)
   */
  centralityMin: number;

  /**
   * Master visibility flag for the entire connectome overlay.
   * When false, both node and edge actors have setVisible(false) applied.
   * Corresponds to the "Show Connectome" top-level toggle checkbox.
   * @default true
   */
  overlayVisible: boolean;
}

// ── Action union for the filter reducer ──────────────────────────────────────

export type ConnectomeFilterAction =
  | { type: 'SET_EDGE_THRESHOLD';  value: number }
  | { type: 'SET_CENTRALITY_MIN';  value: number }
  | { type: 'SET_OVERLAY_VISIBLE'; value: boolean }
  | { type: 'RESET' };

// ── Selected node (populated on node row click or VTK picker) ────────────────

/**
 * Data shown in the ConnectomeNodeDossier card.
 * Populated when the user clicks a node row in the list or
 * (future) when the VTK hardware picker fires an onNodeClick callback.
 */
export interface SelectedNodeInfo {
  /** FreeSurfer label ID as a decimal string, e.g. "17" for L-hippocampus. */
  labelId: string;

  /**
   * Human-readable anatomy name from BRAIN_LABELS, or a fallback string
   * "Label {id}" for regions not in the 32-structure SynthSeg set.
   */
  name: string;

  /**
   * Tissue macro-class ("gm" | "wm" | "csf") or "unknown" for unmapped labels.
   * Shown as a coloured badge in the dossier header.
   */
  group: string;

  /** Full ConnectomeNode from the API response. */
  node: ConnectomeNode;
}
