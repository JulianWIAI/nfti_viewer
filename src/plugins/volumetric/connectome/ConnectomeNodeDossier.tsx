/**
 * ConnectomeNodeDossier.tsx — Detail card for a selected connectome node
 * ────────────────────────────────────────────────────────────────────────
 *
 * Rendered by ConnectomePanel beneath the node list when a node is selected.
 * Displays the full per-region data returned by POST /api/connectomics/matrix.
 *
 * DATA SHOWN
 * ───────────
 *   Region name      — from BRAIN_LABELS, e.g. "Hippocampus (L)"
 *   Tissue badge     — GM / WM / CSF coloured pill
 *   FreeSurfer ID    — the numeric label, e.g. 17
 *   Degree           — number of suprathreshold connections (integer)
 *   Betweenness      — normalised betweenness centrality, 4 decimal places
 *   Center of mass   — RAS mm coordinates, 1 decimal place
 *
 * FUTURE: VTK PICKER BINDING
 * ───────────────────────────
 * The `onNodeClick(labelId)` callback in ConnectomePanel is the entry point
 * for the VTK hardware picker.  When the picker fires on a sphere glyph, the
 * picked FreeSurfer label (encoded in the glyph instance index) should be
 * passed to `onNodeClick` which updates `selectedNode` state.
 *
 * VTK hardware picker integration sketch:
 *
 *   // In ConnectomePanel, after building the bundle:
 *   const picker = vtkCellPicker.newInstance();
 *   picker.setPickFromList(true);
 *   picker.addPickList(bundle.nodeBundle.actor);
 *
 *   ctx.interactor.onLeftButtonPress((event) => {
 *     const [x, y] = event.position;
 *     picker.pick([x, y, 0], ctx.volumeRenderer);
 *     const cellId = picker.getCellId();
 *     if (cellId >= 0) {
 *       const labelId = nodeIds[cellId];   // nodeIds ordered matches nodeEntries
 *       if (labelId) onNodeClick(labelId);
 *     }
 *   });
 *
 * CSS CLASSES  (defined in App.css under "── ConnectomePanel ──")
 * ────────────────────────────────────────────────────────────────
 *   .connectome-panel__dossier          card container
 *   .connectome-panel__dossier-header   name + badge row
 *   .connectome-panel__dossier-title    region name (bold)
 *   .connectome-panel__dossier-badge    tissue class pill (gm/wm/csf colours)
 *   .connectome-panel__dossier-badge--gm/.--wm/.--csf  color modifiers
 *   .connectome-panel__dossier-grid     2-column metric grid
 *   .connectome-panel__dossier-metric   label + value pair
 *   .connectome-panel__dossier-metric-label  dim small-caps label
 *   .connectome-panel__dossier-metric-value  mono value
 *   .connectome-panel__dossier-coords   RAS mm coordinate row
 */

import type { JSX } from 'react';
import type { SelectedNodeInfo } from './connectomePanelTypes';

// ── Props ─────────────────────────────────────────────────────────────────────

interface ConnectomeNodeDossierProps {
  /** The node to display.  When null the card is not rendered. */
  info: SelectedNodeInfo;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function ConnectomeNodeDossier({ info }: ConnectomeNodeDossierProps): JSX.Element {
  const { node } = info;
  const [cx, cy, cz] = node.center_of_mass;

  return (
    <div className="connectome-panel__dossier">

      {/* ── Header: name + tissue badge ──────────────────────────────── */}
      <div className="connectome-panel__dossier-header">
        <span className="connectome-panel__dossier-title">{info.name}</span>
        {info.group !== 'unknown' && (
          <span className={`connectome-panel__dossier-badge connectome-panel__dossier-badge--${info.group}`}>
            {info.group.toUpperCase()}
          </span>
        )}
      </div>

      {/* ── Metrics grid (2 columns) ─────────────────────────────────── */}
      <div className="connectome-panel__dossier-grid">

        {/* FreeSurfer label ID */}
        <DossierMetric label="Label ID" value={info.labelId} />

        {/* Degree: number of connected regions (integer) */}
        <DossierMetric label="Degree" value={String(node.degree)} />

        {/* Betweenness centrality: how often this node sits on shortest paths */}
        <DossierMetric label="Betweenness" value={node.betweenness.toFixed(4)} />

        {/* Connection count would go here once exposed by the API */}
        <DossierMetric label="Connections" value={`${node.degree} edges`} />

      </div>

      {/* ── Center of mass (RAS mm) ───────────────────────────────────── */}
      {/*                                                                   */}
      {/* These coordinates are in the same RAS mm world space as the       */}
      {/* vtk.js volume renderer.  Useful for cross-referencing the node    */}
      {/* position with an atlas or for debugging coordinate alignment.     */}
      <div className="connectome-panel__dossier-coords">
        <span className="connectome-panel__dossier-metric-label">CoM (RAS mm)</span>
        <span className="connectome-panel__dossier-metric-value">
          {cx.toFixed(1)}, {cy.toFixed(1)}, {cz.toFixed(1)}
        </span>
      </div>

    </div>
  );
}

// ── Private sub-component ─────────────────────────────────────────────────────

/** Single metric row: small-caps label + monospace value. */
function DossierMetric({ label, value }: { label: string; value: string }): JSX.Element {
  return (
    <div className="connectome-panel__dossier-metric">
      <span className="connectome-panel__dossier-metric-label">{label}</span>
      <span className="connectome-panel__dossier-metric-value">{value}</span>
    </div>
  );
}
