/**
 * AnatomySelector.tsx — Searchable per-structure visibility panel
 * ──────────────────────────────────────────────────────────────────
 *
 * Displays the 32 SynthSeg brain structures grouped by tissue class
 * (Gray Matter / White Matter / CSF).  Each group has a macro-level checkbox
 * with indeterminate state reflecting whether all, some, or none of its
 * structures are visible.  Individual structures can be toggled within groups.
 *
 * UI features
 * ───────────
 *  • Collapsible panel (default: expanded)
 *  • Text search that filters structure names across all groups in real time
 *  • "Show All" / "Hide All" bulk action buttons
 *  • Macro group checkboxes with native indeterminate state for partial selections
 *  • Per-label checkboxes with structure name and group-colour dot
 *
 * State ownership
 * ───────────────
 * All visibility state lives in VolumetricViewer (via VolumetricContext).
 * This component is a pure UI consumer — it reads labelVisibility /
 * macroGroupState and dispatches setLabelVisible / setGroupVisible /
 * showAllLabels / hideAllLabels.  No local visibility state is kept here.
 */

import { useState, type FC } from 'react';
import { useVolumetricContext } from '../plugins/volumetric/VolumetricViewer';
import IndeterminateCheckbox from './IndeterminateCheckbox';
import { LABELS_BY_GROUP } from '../lib/vtk/labelVisibility';
import {
  type TissueClass,
  TISSUE_CLASS_LABELS,
  TISSUE_CSS_COLORS,
} from '../lib/vtk/tissueGroups';

// Display order of tissue classes in the panel.
const TISSUE_CLASSES: TissueClass[] = ['gm', 'wm', 'csf'];

// ── Styles (inline, scoped to this component) ─────────────────────────────────

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

const AnatomySelector: FC = () => {
  const {
    labelVisibility, macroGroupState,
    setLabelVisible, setGroupVisible,
    showAllLabels, hideAllLabels,
  } = useVolumetricContext();

  // Local UI state only — does not affect the vtk.js overlay.
  const [expanded, setExpanded] = useState(true);
  const [query,    setQuery]    = useState('');

  const normQuery = query.trim().toLowerCase();

  return (
    <section className="control-section">

      {/* ── Panel header with collapse toggle ────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h3 className="section-title" style={{ margin: 0 }}>Anatomy</h3>
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
          {/* ── Search field ─────────────────────────────────────────────── */}
          <input
            type="search"
            placeholder="Filter structures…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={searchStyle}
          />

          {/* ── Bulk actions ──────────────────────────────────────────────── */}
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
            // Filter structures by the search query.
            const filtered = normQuery
              ? groupLabels.filter((l) => l.name.toLowerCase().includes(normQuery))
              : groupLabels;

            // If the search hides every structure in this group, skip the group.
            if (filtered.length === 0) return null;

            const state = macroGroupState[cls];
            const color = TISSUE_CSS_COLORS[cls];

            return (
              <div key={cls} style={{ marginBottom: 8 }}>

                {/* Group header row — macro checkbox + group name */}
                <div className="control-row control-row--toggle" style={{ marginBottom: 2 }}>
                  <label
                    className="control-label"
                    htmlFor={`anatomy-group-${cls}`}
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
                    id={`anatomy-group-${cls}`}
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
                      htmlFor={`anatomy-label-${lbl.id}`}
                      style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 10 }}
                    >
                      <span style={{
                        display: 'inline-block', width: 6, height: 6,
                        borderRadius: 1, background: color, flexShrink: 0, opacity: 0.65,
                      }} />
                      {lbl.name}
                    </label>
                    <input
                      id={`anatomy-label-${lbl.id}`}
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

export default AnatomySelector;
