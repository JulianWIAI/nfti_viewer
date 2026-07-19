/**
 * DualAnatomySelector.tsx — Per-structure segmentation overlay controls for the
 * dual-viewer inter-subject comparison.
 *
 * Functionally identical to AnatomySelector.tsx but reads from DualViewerContext
 * instead of VolumetricContext, so it can control BOTH viewers simultaneously.
 *
 * When a label is toggled here, dualSegBundle.updateLabelVisibility() rebuilds
 * the RGBA scalar arrays for BOTH the left (A) and right (B warped) overlay
 * actors in a single call — the two panes always stay in sync.
 *
 * Features (same as single-brain AnatomySelector)
 * ────────────────────────────────────────────────
 *   • Collapsible panel (default: expanded)
 *   • Text search filtering structure names across all groups
 *   • "Show All" / "Hide All" bulk action buttons
 *   • GM / WM / CSF group-level macro checkboxes with indeterminate state
 *   • Per-structure checkboxes with name and group-colour dot
 */

import { useState, type FC } from 'react';
import { useDualViewerContext } from './DualViewerContext';
import IndeterminateCheckbox    from '../../components/IndeterminateCheckbox';
import { LABELS_BY_GROUP }      from '../../lib/vtk/labelVisibility';
import {
  type TissueClass,
  TISSUE_CLASS_LABELS,
  TISSUE_CSS_COLORS,
} from '../../lib/vtk/tissueGroups';

// Display order in the panel.
const TISSUE_CLASSES: TissueClass[] = ['gm', 'wm', 'csf'];

// ── Inline styles ────────────────────────────────────────────────────────────

const searchStyle: React.CSSProperties = {
  width: '100%', boxSizing: 'border-box',
  marginTop: 6, marginBottom: 4,
  padding: '3px 6px', fontSize: 11,
  background: '#1a1a1a', color: '#ccc',
  border: '1px solid #333', borderRadius: 3,
  outline: 'none',
};

const bulkBtnStyle: React.CSSProperties = {
  flex: 1, fontSize: 10, padding: '2px 0',
  cursor: 'pointer',
};

// ── Component ─────────────────────────────────────────────────────────────────

const DualAnatomySelector: FC = () => {
  const {
    labelVisibility, macroGroupState,
    setLabelVisible, setGroupVisible,
    showAllLabels,   hideAllLabels,
  } = useDualViewerContext();

  // Local UI state — does not affect VTK actors.
  const [expanded, setExpanded] = useState(true);
  const [query,    setQuery]    = useState('');

  const normQuery = query.trim().toLowerCase();

  return (
    <section className="control-section dual-viewer-controls__section">

      {/* ── Panel header with collapse toggle ────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h3 className="section-title" style={{ margin: 0 }}>Anatomy (both viewers)</h3>
        <button
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
          style={{
            background: 'none', border: 'none', cursor: 'pointer',
            fontSize: 11, color: '#888', padding: 0, lineHeight: 1,
          }}
        >
          {expanded ? '▲' : '▼'}
        </button>
      </div>

      {expanded && (
        <>
          {/* ── Search ──────────────────────────────────────────────────── */}
          <input
            type="search"
            placeholder="Filter structures…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={searchStyle}
          />

          {/* ── Bulk actions ─────────────────────────────────────────────── */}
          <div style={{ display: 'flex', gap: 4, marginBottom: 6 }}>
            <button className="btn" style={bulkBtnStyle} onClick={showAllLabels}>
              Show All
            </button>
            <button className="btn" style={bulkBtnStyle} onClick={hideAllLabels}>
              Hide All
            </button>
          </div>

          {/* ── One collapsible group per tissue class ────────────────────── */}
          {TISSUE_CLASSES.map((cls) => {
            const groupLabels = LABELS_BY_GROUP[cls];
            const filtered = normQuery
              ? groupLabels.filter((l) => l.name.toLowerCase().includes(normQuery))
              : groupLabels;

            if (filtered.length === 0) return null;

            const state = macroGroupState[cls];
            const color = TISSUE_CSS_COLORS[cls];

            return (
              <div key={cls} style={{ marginBottom: 8 }}>

                {/* Group header — macro checkbox + group name */}
                <div className="control-row control-row--toggle" style={{ marginBottom: 2 }}>
                  <label
                    className="control-label"
                    htmlFor={`dual-anatomy-group-${cls}`}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 5,
                      fontWeight: 600, fontSize: 11,
                    }}
                  >
                    <span style={{
                      display: 'inline-block', width: 8, height: 8,
                      borderRadius: 2, background: color, flexShrink: 0,
                    }} />
                    {TISSUE_CLASS_LABELS[cls]}
                  </label>
                  <IndeterminateCheckbox
                    id={`dual-anatomy-group-${cls}`}
                    checked={state === 'all'}
                    indeterminate={state === 'partial'}
                    onChange={(checked) => setGroupVisible(cls, checked)}
                  />
                </div>

                {/* Per-structure rows */}
                {filtered.map((lbl) => (
                  <div
                    key={lbl.id}
                    className="control-row control-row--toggle"
                    style={{ paddingLeft: 14, marginBottom: 1 }}
                  >
                    <label
                      className="control-label"
                      htmlFor={`dual-anatomy-label-${lbl.id}`}
                      style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 10 }}
                    >
                      <span style={{
                        display: 'inline-block', width: 6, height: 6,
                        borderRadius: 1, background: color, flexShrink: 0, opacity: 0.65,
                      }} />
                      {lbl.name}
                    </label>
                    <input
                      id={`dual-anatomy-label-${lbl.id}`}
                      type="checkbox"
                      checked={!!labelVisibility[lbl.id]}
                      onChange={(e) => setLabelVisible(lbl.id, e.target.checked)}
                    />
                  </div>
                ))}
              </div>
            );
          })}
        </>
      )}
    </section>
  );
};

export default DualAnatomySelector;
