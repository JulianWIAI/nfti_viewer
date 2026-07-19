/**
 * ConnectomePanel.tsx — Sidebar control section for the 3-D connectome overlay
 * ──────────────────────────────────────────────────────────────────────────────
 *
 * STRUCTURE
 * ──────────
 *
 *   [section] Connectome
 *   │
 *   ├── Master toggle  "Show Connectome"  (checkbox)
 *   │     → calls bundle.setVisible() + renderWindow.render()
 *   │     → useEffect[overlayVisible] handles the VTK update
 *   │
 *   ├── (placeholder when connectomeData is null)
 *   │
 *   └── (when connectomeData is available):
 *       ├── Edge Threshold slider  [0 … maxFiberCount]
 *       │     → RAF-debounced → bundle.edgeBundle.filterByWeight(v) + render()
 *       │     → Why RAF? slider fires onChange ~60+/s; RAF caps VTK calls to 60fps
 *       │
 *       ├── Min. Betweenness slider  [0 … maxBetweenness]
 *       │     → RAF-debounced → bundle.nodeBundle.filterByCentrality(v) + render()
 *       │
 *       ├── Stats line  "N edges · M regions"
 *       │
 *       ├── Node list  (sorted by betweenness, highest first)
 *       │     Each row: [color swatch] [name] [betweenness %]
 *       │     Click row → sets selectedNode → shows dossier
 *       │     Dim rows for nodes below centralityMin
 *       │
 *       └── ConnectomeNodeDossier  (shown when selectedNode is non-null)
 *             Displays name, tissue class, degree, betweenness, CoM
 *
 * VTK LIFECYCLE
 * ──────────────
 * The VTK bundle (ConnectomeBundle) is built when connectomeData arrives and
 * disposed when the data is cleared or the component unmounts.  This avoids
 * the actor persisting in the renderer after the user loads a new scan.
 *
 *   connectomeData changes → useEffect builds new bundle, adds actors to
 *                             ctx.volumeRenderer, returns cleanup that removes
 *                             actors and calls bundle.dispose() on next change
 *                             or unmount.
 *
 * VTK UPDATE STRATEGIES
 * ──────────────────────
 * Two strategies are used depending on update frequency and vtk.js capability:
 *
 *   1. useEffect[overlayVisible]
 *      Low frequency (user toggles checkbox), fine to go through React's
 *      render cycle.  Calls bundle.setVisible(v) + render().
 *
 *   2. requestAnimationFrame (RAF) debounce — for sliders
 *      High frequency (slider onChange fires ~60-200/s on pointer drag).
 *      Scheduling the vtk update on the next animation frame ensures:
 *        • At most one VTK update per screen refresh (~16ms).
 *        • The slider thumb moves at native OS speed (no lag) because we
 *          only update React state once per drag event, not VTK.
 *        • No extra React re-render cycles are triggered by the VTK call.
 *
 *      Pattern:
 *        if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
 *        rafRef.current = requestAnimationFrame(() => {
 *          bundle.edgeBundle.filterByWeight(v);
 *          ctx.renderWindow.render();
 *          rafRef.current = null;
 *        });
 *
 * FUTURE: VTK HARDWARE PICKER
 * ─────────────────────────────
 * The `handleNodeClick(labelId)` function at the bottom of this component is
 * the designated binding point for the VTK hardware picker.  When implemented,
 * the picker would call it with the FreeSurfer label of the clicked sphere.
 * See ConnectomeNodeDossier.tsx for the integration sketch.
 *
 * CSS CLASSES  (defined in App.css)
 * ──────────────────────────────────
 *   .connectome-panel__slider-group     wrapper for label + slider + value
 *   .connectome-panel__slider-label     flex row: name left, value right
 *   .connectome-panel__slider-hint      dim explanatory text beneath slider
 *   .connectome-panel__divider          thin rule between sections
 *   .connectome-panel__stats            dim footer line
 *   .connectome-panel__node-list        scrollable region list
 *   .connectome-panel__node-item        clickable node row
 *   .connectome-panel__node-item--selected  highlighted row
 *   .connectome-panel__node-item--dimmed    below centrality threshold
 *   .connectome-panel__node-swatch      small colored square
 *   .connectome-panel__node-name        anatomy label text
 *   .connectome-panel__node-centrality  betweenness value (right-aligned)
 */

import {
  useRef,
  useState,
  useEffect,
  useCallback,
  useMemo,
  type JSX,
} from 'react';
import { useVolumetricContext } from '../VolumetricViewer';
import { buildConnectomeOverlay } from '../../../lib/vtk/connectome/connectomeOverlay';
import { BRAIN_LABELS }           from '../../../lib/vtk/labelVisibility';
import { labelToRgb }             from '../../../lib/vtk/connectome/connectomeLabelColor';
import { useConnectomeFilters }   from './useConnectomeFilters';
import ConnectomeNodeDossier      from './ConnectomeNodeDossier';
import ConnectomeHeatmap          from '../../../components/ConnectomeHeatmap';
import type { ConnectomeBundle }  from '../../../lib/vtk/connectome/connectomeTypes';
import type { SelectedNodeInfo }  from './connectomePanelTypes';

// ── Fast label-ID-to-name lookup (built once at module load) ──────────────────
// Keys are string versions of the numeric IDs to match ConnectomeApiResponse.nodes.
const LABEL_NAME_MAP = new Map<string, typeof BRAIN_LABELS[number]>(
  BRAIN_LABELS.map((l) => [String(l.id), l]),
);

// ── Component ─────────────────────────────────────────────────────────────────

export default function ConnectomePanel(): JSX.Element {
  // ── Context (shared state from VolumetricViewer) ────────────────────────────
  const { connectomeData, getVtkCtx, selectedEdge, setSelectedEdge } = useVolumetricContext();

  // ── Filter state — managed by useReducer for explicit, typed transitions ────
  const {
    state: filters,
    setEdgeThreshold,
    setCentralityMin,
    setOverlayVisible,
    reset: resetFilters,
  } = useConnectomeFilters();

  // ── Node selection for the dossier card ─────────────────────────────────────
  const [selectedNode, setSelectedNode] = useState<SelectedNodeInfo | null>(null);

  // ── VTK bundle ref — does NOT go into React state ──────────────────────────
  //
  // Storing the bundle in a ref (not useState) prevents React from re-rendering
  // when the bundle is replaced.  VTK state is imperative; React state is
  // declarative — they must be kept separate.
  const bundleRef = useRef<ConnectomeBundle | null>(null);

  // ── RAF handle — used to debounce slider → VTK updates ─────────────────────
  //
  // One shared ref for both sliders is fine because only one slider can be
  // dragged at a time.  The pending RAF is cancelled before scheduling a new
  // one, so rapid alternating slider drags never queue more than one frame.
  const rafRef = useRef<number | null>(null);

  // ── Derived slider maxima (from API data) ────────────────────────────────────

  /** Maximum fibre count in the matrix — sets the right edge of the edge-threshold slider. */
  const maxFiberCount = useMemo(() => {
    if (!connectomeData) return 200;
    let max = 0;
    for (const row of connectomeData.matrix) {
      for (const v of row) if (v > max) max = v;
    }
    return Math.max(max, 1);
  }, [connectomeData]);

  /** Maximum betweenness in the node set — sets the right edge of the centrality slider. */
  const maxBetweenness = useMemo(() => {
    if (!connectomeData) return 1;
    let max = 0;
    for (const node of Object.values(connectomeData.nodes)) {
      if (node.betweenness > max) max = node.betweenness;
    }
    return Math.max(max, 0.0001);
  }, [connectomeData]);

  // ── Live counts shown in the stats footer ────────────────────────────────────

  /** Number of edges currently above the threshold — recomputed on slider change. */
  const activeEdgeCount = useMemo(() => {
    if (!connectomeData) return 0;
    const { matrix, node_ids: nodeIds } = connectomeData;
    const n = nodeIds.length;
    let count = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const w = Math.max(matrix[i]?.[j] ?? 0, matrix[j]?.[i] ?? 0);
        if (w > filters.edgeThreshold) count++;
      }
    }
    return count;
  }, [connectomeData, filters.edgeThreshold]);

  /** Number of nodes whose betweenness is at or above the centrality slider. */
  const activeNodeCount = useMemo(() => {
    if (!connectomeData) return 0;
    return Object.values(connectomeData.nodes).filter(
      (n) => n.betweenness >= filters.centralityMin,
    ).length;
  }, [connectomeData, filters.centralityMin]);

  // ── Sorted node list (betweenness descending) ─────────────────────────────
  //
  // Sorting by betweenness puts the structural hubs at the top of the list —
  // the regions a neuroscientist is most likely to want to inspect first.
  const sortedNodes = useMemo(() => {
    if (!connectomeData) return [];
    return Object.entries(connectomeData.nodes).sort(
      ([, a], [, b]) => b.betweenness - a.betweenness,
    );
  }, [connectomeData]);

  // ── VTK bundle lifecycle ──────────────────────────────────────────────────
  //
  // This effect runs when `connectomeData` changes (null → data or data → null).
  // It does NOT run when filter thresholds change — those are handled by
  // the RAF callbacks below.  This separation avoids rebuilding expensive VTK
  // objects on every slider tick.
  //
  useEffect(() => {
    const ctx = getVtkCtx();
    if (!ctx || !connectomeData) return;

    // Dispose the previous bundle if one exists (handles data refresh).
    if (bundleRef.current) {
      ctx.volumeRenderer.removeActor(bundleRef.current.nodeBundle.actor);
      ctx.volumeRenderer.removeActor(bundleRef.current.edgeBundle.actor);
      bundleRef.current.dispose();
      bundleRef.current = null;
    }

    // Build fresh overlay from the API data.
    // Use the current filter state so the initial render respects
    // whatever threshold the user had set before the data arrived.
    const bundle = buildConnectomeOverlay(connectomeData, {
      edgeThreshold: filters.edgeThreshold,
    });

    // Register actors with the 3-D volume renderer.
    // NOTE: these actors share the same world coordinate frame as the
    //       structural MRI volume — no additional transform is needed.
    ctx.volumeRenderer.addActor(bundle.nodeBundle.actor);
    ctx.volumeRenderer.addActor(bundle.edgeBundle.actor);
    bundleRef.current = bundle;

    // Apply the current visibility toggle in case the overlay was previously
    // hidden by the user before new data arrived.
    bundle.setVisible(filters.overlayVisible);

    ctx.renderWindow.render();

    // ── Cleanup ──────────────────────────────────────────────────────────────
    // Called on next connectomeData change or component unmount.
    // Ensures VTK actors are removed from the renderer before the JS objects
    // are garbage-collected.
    return () => {
      if (bundleRef.current) {
        const cleanCtx = getVtkCtx();
        if (cleanCtx) {
          cleanCtx.volumeRenderer.removeActor(bundleRef.current.nodeBundle.actor);
          cleanCtx.volumeRenderer.removeActor(bundleRef.current.edgeBundle.actor);
          cleanCtx.renderWindow.render();
        }
        bundleRef.current.dispose();
        bundleRef.current = null;
      }
    };
    // Intentionally omit `filters` from deps — we only want to rebuild on
    // new API data.  Threshold changes are handled via RAF, not this effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connectomeData, getVtkCtx]);

  // ── VTK visibility update — useEffect (low frequency) ────────────────────
  //
  // useEffect is appropriate here because visibility is toggled at most once
  // per user interaction (a checkbox click), not at 60+/s.
  //
  useEffect(() => {
    const bundle = bundleRef.current;
    const ctx    = getVtkCtx();
    if (!bundle || !ctx) return;
    bundle.setVisible(filters.overlayVisible);
    ctx.renderWindow.render();
  }, [filters.overlayVisible, getVtkCtx]);

  // ── RAF-debounced VTK update handlers ─────────────────────────────────────
  //
  // These handlers update React state immediately (for the slider thumb and
  // the value readout) but schedule the expensive VTK update on the NEXT
  // animation frame.  If the user drags faster than 60fps, only the final
  // position within each 16ms frame window is forwarded to VTK.
  //
  // IMPORTANT: We read the bundle from bundleRef.current *inside* the RAF
  // callback, not at handler-creation time, so we always use the latest
  // bundle even if data was refreshed while the user was dragging.
  //

  /** Handler for the Edge Threshold range slider. */
  const handleEdgeThreshold = useCallback((v: number) => {
    // 1. Update React state immediately — the slider thumb and value display
    //    respond at native OS speed.
    setEdgeThreshold(v);

    // 2. Schedule a VTK update on the next animation frame.
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      const bundle = bundleRef.current;
      const ctx    = getVtkCtx();
      if (bundle && ctx) {
        // In-place edge rebuild: replaces only the vtkCellArray + cell scalars,
        // NOT the actor, mapper, or polyData object.
        // See connectomeEdges.ts → filterByWeight() for the implementation.
        bundle.edgeBundle.filterByWeight(v);
        ctx.renderWindow.render();
      }
      rafRef.current = null;
    });
  }, [setEdgeThreshold, getVtkCtx]);

  /** Handler for the Minimum Betweenness Centrality range slider. */
  const handleCentralityMin = useCallback((v: number) => {
    // Same RAF-debounce pattern as handleEdgeThreshold.
    setCentralityMin(v);

    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      const bundle = bundleRef.current;
      const ctx    = getVtkCtx();
      if (bundle && ctx) {
        // In-place radius update: sets GlyphRadius to 0 for nodes below
        // minBetweenness (zero-radius glyphs render as nothing), restores
        // full radius for nodes at or above the threshold.
        // See connectomeNodes.ts → filterByCentrality() for the implementation.
        bundle.nodeBundle.filterByCentrality(v);
        ctx.renderWindow.render();
      }
      rafRef.current = null;
    });
  }, [setCentralityMin, getVtkCtx]);

  // ── Cross-filter: heatmap cell click → VTK edge highlight ────────────────
  //
  // DATA FLOW
  // ──────────
  //   ConnectomeHeatmap.onCellSelect(i, j, count)
  //     → handleCellSelect(i, j, count)
  //       → setSelectedEdge({ source: i, target: j })   [VolumetricContext state]
  //         → useEffect[selectedEdge] fires
  //           → bundle.edgeBundle.highlightEdge(i, j)   [VTK in-place update]
  //             → ctx.renderWindow.render()              [GPU re-upload]
  //
  // The two-step path (React state → effect → VTK) is intentional:
  //   • setSelectedEdge lifts the selection into shared context so any future
  //     subscriber (e.g. a stats panel) can also read it.
  //   • The useEffect is the correct React pattern for imperative VTK side-effects
  //     that depend on React state changes.
  //

  /**
   * Wired to ConnectomeHeatmap.onCellSelect.
   * Translates the nullable (sourceIdx, targetIdx) pair into a VolumetricContext
   * selectedEdge update.  Passing null for either arg clears the selection.
   */
  const handleCellSelect = useCallback((
    sourceIdx:  number | null,
    targetIdx:  number | null,
    _fiberCount: number | null,
  ) => {
    setSelectedEdge(
      sourceIdx !== null && targetIdx !== null
        ? { source: sourceIdx, target: targetIdx }
        : null,
    );
  }, [setSelectedEdge]);

  /**
   * useEffect[selectedEdge] — synchronises VolumetricContext selection state
   * with the VTK edge highlight.
   *
   * WHY useEffect AND NOT A RAF CALLBACK?
   * ──────────────────────────────────────
   * Highlight changes occur at most once per click (not 60+/s like a slider
   * drag), so going through React's commit phase is acceptable.  The effect
   * fires synchronously after the paint, ensuring the 3D view updates on the
   * very next frame after the heatmap click.
   */
  useEffect(() => {
    const bundle = bundleRef.current;
    const ctx    = getVtkCtx();
    if (!bundle || !ctx) return;

    // Delegate to the edge bundle's in-place highlight method.
    // null → exit highlight mode (restore full CTF-based rendering).
    // non-null → enter highlight mode (accent-green selected, rest ghosted).
    bundle.edgeBundle.highlightEdge(
      selectedEdge?.source ?? null,
      selectedEdge?.target ?? null,
    );

    ctx.renderWindow.render();
  }, [selectedEdge, getVtkCtx]);

  // ── Node selection handler ────────────────────────────────────────────────
  //
  // PLACEHOLDER for VTK hardware picker binding.
  //
  // This function is called either:
  //   a) By a node-list row click (implemented below).
  //   b) [FUTURE] By the VTK vtkCellPicker when the user clicks a sphere in
  //      the 3-D volume view.  The picker resolves the glyph instance index
  //      to a FreeSurfer label via nodeIds[], then calls onNodeClick(labelId).
  //
  // TO IMPLEMENT THE VTK PICKER:
  //   1. After building the bundle, create a vtkCellPicker.
  //   2. Set it to pick only the node actor.
  //   3. In ctx.interactor.onLeftButtonPress, call picker.pick(), read
  //      picker.getCellId(), map to labelId via nodeIds[], call handleNodeClick.
  //   See ConnectomeNodeDossier.tsx for the full code sketch.
  //
  const handleNodeClick = useCallback((labelId: string) => {
    if (!connectomeData) return;
    const node = connectomeData.nodes[labelId];
    if (!node) return;

    const meta = LABEL_NAME_MAP.get(labelId);
    setSelectedNode({
      labelId,
      name:  meta?.name  ?? `Label ${labelId}`,
      group: meta?.group ?? 'unknown',
      node,
    });
  }, [connectomeData]);

  // ── Cleanup RAF on unmount ─────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, []);

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <section className="control-section">
      <h3 className="section-title">Connectome</h3>

      {/* ── Master toggle ───────────────────────────────────────────────── */}
      {/*                                                                    */}
      {/* Visibility is handled by a useEffect (not a RAF callback) because  */}
      {/* it fires at most once per checkbox click, not during drag.         */}

      <div className="control-row control-row--toggle">
        <label className="control-label" htmlFor="connectome-show">
          Show Connectome
        </label>
        <input
          id="connectome-show"
          type="checkbox"
          checked={filters.overlayVisible}
          disabled={!connectomeData}
          onChange={(e) => setOverlayVisible(e.target.checked)}
        />
      </div>

      {/* ── Empty state ─────────────────────────────────────────────────── */}
      {!connectomeData && (
        <p className="section-hint">
          No connectome loaded.{' '}
          Call <code>POST /api/connectomics/matrix</code> and pass the
          result to <code>setConnectomeData()</code>.
        </p>
      )}

      {/* ── Controls (only when data is available) ──────────────────────── */}
      {connectomeData && (
        <>
          {/* ── Edge threshold slider ──────────────────────────────────── */}
          {/*                                                                */}
          {/* THRESHOLDING RATIONALE                                         */}
          {/* A full-brain DTI connectome at threshold=0 contains hundreds   */}
          {/* of edges, most from tracking noise and gyral-bias artefacts.   */}
          {/* Raising the threshold progressively removes weak connections   */}
          {/* until only the major structural highways remain.               */}
          {/*                                                                */}
          {/* The VTK update is RAF-debounced in handleEdgeThreshold() so    */}
          {/* scrubbing the slider is smooth at 60fps even on slow hardware. */}

          <div className="connectome-panel__slider-group">
            <div className="connectome-panel__slider-label">
              <span>Edge Threshold</span>
              <span className="control-value">≥ {filters.edgeThreshold} fibres</span>
            </div>
            <p className="connectome-panel__slider-hint">
              Hides edges with fewer than this many streamlines.
            </p>
            <input
              type="range"
              className="control-slider"
              min={0}
              max={maxFiberCount}
              step={1}
              value={filters.edgeThreshold}
              onChange={(e) => handleEdgeThreshold(Number(e.target.value))}
            />
          </div>

          {/* ── Centrality slider ──────────────────────────────────────── */}
          {/*                                                                */}
          {/* BETWEENNESS CENTRALITY                                         */}
          {/* B(v) = Σ_{s≠v≠t} σ(s,t|v) / σ(s,t)                          */}
          {/* Nodes with high betweenness sit on many shortest paths —       */}
          {/* they are structural hubs (thalamus, corpus callosum, putamen). */}
          {/* Raising this threshold isolates only the most critical hubs.   */}
          {/*                                                                */}
          {/* In-place VTK update via connectomeNodes.filterByCentrality():  */}
          {/*   Zero the GlyphRadius for hidden nodes → degenerate spheres   */}
          {/*   are not rendered (no geometry emitted by the Glyph3DMapper). */}

          <div className="connectome-panel__slider-group">
            <div className="connectome-panel__slider-label">
              <span>Min. Betweenness</span>
              <span className="control-value">
                ≥ {(filters.centralityMin / maxBetweenness * 100).toFixed(1)}%
              </span>
            </div>
            <p className="connectome-panel__slider-hint">
              Hides nodes below this betweenness centrality.
            </p>
            <input
              type="range"
              className="control-slider"
              min={0}
              max={maxBetweenness}
              step={maxBetweenness / 200}   // 200 steps across the range
              value={filters.centralityMin}
              onChange={(e) => handleCentralityMin(Number(e.target.value))}
            />
          </div>

          {/* ── Live stats + reset ─────────────────────────────────────── */}
          <div className="connectome-panel__stats-row">
            <span className="connectome-panel__stats">
              {activeEdgeCount} edges · {activeNodeCount} regions
            </span>
            <button
              type="button"
              className="connectome-panel__reset-btn"
              onClick={resetFilters}
              title="Reset filters to defaults"
            >
              Reset
            </button>
          </div>

          <hr className="connectome-panel__divider" />

          {/* ── Node list ──────────────────────────────────────────────── */}
          {/*                                                                */}
          {/* Sorted by betweenness descending so hubs appear at the top.   */}
          {/* Rows below the centrality threshold are dimmed but still       */}
          {/* listed so the user can drag the slider back to reveal them.   */}
          {/*                                                                */}
          {/* Clicking a row calls handleNodeClick(labelId) which populates  */}
          {/* the SelectedNodeInfo state and reveals the dossier card.       */}
          {/* The same handleNodeClick is the binding point for the future   */}
          {/* VTK vtkCellPicker integration.                                 */}

          <div className="connectome-panel__node-list" role="listbox" aria-label="Brain regions">
            {sortedNodes.map(([labelId, node]) => {
              const meta       = LABEL_NAME_MAP.get(labelId);
              const name       = meta?.name  ?? `Label ${labelId}`;
              const isSelected = selectedNode?.labelId === labelId;
              const isActive   = node.betweenness >= filters.centralityMin;
              const [r, g, b]  = labelToRgb(labelId);

              return (
                <button
                  key={labelId}
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  className={[
                    'connectome-panel__node-item',
                    isSelected ? 'connectome-panel__node-item--selected' : '',
                    !isActive  ? 'connectome-panel__node-item--dimmed'   : '',
                  ].join(' ').trim()}
                  onClick={() => handleNodeClick(labelId)}
                >
                  {/* SYNTHSEG colour swatch */}
                  <span
                    className="connectome-panel__node-swatch"
                    style={{ background: `rgb(${r},${g},${b})` }}
                    aria-hidden
                  />
                  {/* Anatomy name */}
                  <span className="connectome-panel__node-name">{name}</span>
                  {/* Betweenness percentage (relative to max) */}
                  <span className="connectome-panel__node-centrality">
                    {(node.betweenness / maxBetweenness * 100).toFixed(1)}%
                  </span>
                </button>
              );
            })}
          </div>

          {/* ── Node dossier (selected region detail card) ─────────────── */}
          {/*                                                                */}
          {/* Populated by handleNodeClick — either from a list-row click    */}
          {/* or (future) from the VTK hardware picker event.                */}
          {/* When null the card is not mounted (no empty-card flicker).     */}

          {selectedNode && (
            <ConnectomeNodeDossier info={selectedNode} />
          )}

          {/* ── 2D Connectivity Heatmap ────────────────────────────────────── */}
          {/*                                                                    */}
          {/* Cross-filter bridge:                                               */}
          {/*   onCellSelect → handleCellSelect → setSelectedEdge               */}
          {/*     → useEffect[selectedEdge] → bundle.edgeBundle.highlightEdge   */}
          {/*                                                                    */}
          {/* activeRow / activeCol (= selectedEdge.source / .target) are fed   */}
          {/* back into the heatmap so it draws a persistent white selection     */}
          {/* border on the chosen cell and its symmetric counterpart.           */}

          <hr className="connectome-panel__divider" style={{ marginTop: 10 }} />

          <ConnectomeHeatmap
            matrix={connectomeData.matrix}
            nodeIds={connectomeData.node_ids}
            activeRow={selectedEdge?.source ?? null}
            activeCol={selectedEdge?.target ?? null}
            onCellSelect={handleCellSelect}
          />
        </>
      )}
    </section>
  );
}
