/**
 * ReferenceDrawer.tsx — Contextual Reference Panel ("Biological Dictionary")
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * A collapsible right-hand drawer that decodes:
 *   • Anatomical Cipher (MRI) — all 32 FreeSurfer SynthSeg structures, grouped
 *     by functional system with colour swatches matched to the vtk.js overlay.
 *   • Electrophysiological Cipher (MEG) — five frequency bands, ERP waves
 *     (MMN, P300), and fMRI HRF methodology.
 *
 * REACTIVE NAVIGATION
 * ────────────────────
 * When the user clicks a coloured region on the vtk.js 3-D brain model,
 * ReferencePanelContext.navigateToRegion(label) fires.  This drawer reacts via
 * useEffect: the matching anatomy group auto-expands and the specific region
 * gets a coloured ring highlight so the user immediately knows which structure
 * was picked.
 *
 * SEARCH / FILTER + HIGHLIGHT
 * ────────────────────────────
 * The search bar sits above the tabs and is always visible.  On the Anatomy tab
 * it also filters which groups/regions are shown.  On both tabs, every occurrence
 * of the query string is wrapped in <mark class="ref-highlight"> so the user can
 * instantly see where the term appears in the text.
 *
 * The <Highlight> helper component does the marking: it splits the text on each
 * match (case-insensitive) and wraps matched segments with the mark element.
 *
 * CSS classes live in App.css under the "ReferenceDrawer" section appended at
 * the end of that file.
 */

import {
  type FC,
  type KeyboardEvent,
  useState,
  useEffect,
  useCallback,
  useRef,
} from 'react';
import {
  useReferencePanel,
  type ReferenceTab,
} from '../contexts/ReferencePanelContext';
import {
  neuroDictionary,
  type NeuroGroup,
  type NeuroRegion,
  type MegBand,
  type ErpWave,
} from '../lib/neuroData';

// ── Highlight helper ──────────────────────────────────────────────────────────
// Renders `text` with every case-insensitive occurrence of `query` wrapped in
// <mark class="ref-highlight">.  Returns a plain string when query is empty
// so React doesn't create unnecessary extra nodes.

interface HighlightProps {
  text:  string;
  query: string;
}

const Highlight: FC<HighlightProps> = ({ text, query }) => {
  // No query → render text as-is (no JSX overhead)
  if (!query) return <>{text}</>;

  const q        = query.toLowerCase();
  const segments: React.ReactNode[] = [];
  let remaining      = text;
  let lowerRemaining = text.toLowerCase();
  let key            = 0;

  while (remaining.length > 0) {
    const idx = lowerRemaining.indexOf(q);
    if (idx === -1) {
      // No more matches — flush the rest
      segments.push(remaining);
      break;
    }
    // Text before the match
    if (idx > 0) segments.push(remaining.slice(0, idx));
    // The match itself — preserve original casing, just wrap it
    segments.push(
      <mark key={key++} className="ref-highlight">
        {remaining.slice(idx, idx + q.length)}
      </mark>,
    );
    remaining      = remaining.slice(idx + q.length);
    lowerRemaining = remaining.toLowerCase();
  }

  return <>{segments}</>;
};

// ── Sub-components ─────────────────────────────────────────────────────────────

// ── RegionRow — one anatomy region inside a group accordion ──────────────────

interface RegionRowProps {
  region:        NeuroRegion;
  isHighlighted: boolean;
  isExpanded:    boolean;
  query:         string;
  onToggle:      (id: string) => void;
}

const RegionRow: FC<RegionRowProps> = ({
  region, isHighlighted, isExpanded, query, onToggle,
}) => {
  const rowRef = useRef<HTMLDivElement>(null);

  // Scroll into view when this region is highlighted from a 3D click
  useEffect(() => {
    if (isHighlighted && rowRef.current) {
      rowRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [isHighlighted]);

  return (
    <div
      ref={rowRef}
      className={`ref-region${isHighlighted ? ' ref-region--highlighted' : ''}`}
      style={isHighlighted ? ({ '--highlight-color': region.color } as React.CSSProperties) : undefined}
    >
      {/* Region header — click to expand detail */}
      <button
        className="ref-region__header"
        onClick={() => onToggle(region.id)}
        aria-expanded={isExpanded}
      >
        {/* FreeSurfer colour swatch — pixel-matched to vtk.js overlay */}
        <span
          className="ref-region__swatch"
          style={{ background: region.color }}
          title={`FreeSurfer labels: ${region.fsLabels.join(', ')}`}
        />
        <span className="ref-region__name">
          <Highlight text={region.name} query={query} />
        </span>
        <span className="ref-region__fn">
          <Highlight text={region.function} query={query} />
        </span>
        <span className="ref-region__chevron">{isExpanded ? '▾' : '▸'}</span>
      </button>

      {/* Expandable detail panel */}
      <div className={`ref-region__body${isExpanded ? ' ref-region__body--open' : ''}`}>
        <p className="ref-region__detail">
          <Highlight text={region.detail} query={query} />
        </p>
        {/* Label IDs for cross-reference with FreeSurfer tools */}
        <p className="ref-region__labels">
          FreeSurfer label{region.fsLabels.length > 1 ? 's' : ''}:{' '}
          {region.fsLabels.map((l) => (
            <code key={l} className="ref-label-badge">{l}</code>
          ))}
        </p>
      </div>
    </div>
  );
};

// ── GroupAccordion — one accordion panel containing multiple regions ──────────

interface GroupAccordionProps {
  group:              NeuroGroup;
  isOpen:             boolean;
  visibleRegions:     NeuroRegion[];    // filtered subset; may equal group.regions
  highlightedRegionId: string | null;
  expandedRegionIds:  Set<string>;
  query:              string;
  onGroupToggle:      (id: string) => void;
  onRegionToggle:     (id: string) => void;
}

const GroupAccordion: FC<GroupAccordionProps> = ({
  group,
  isOpen,
  visibleRegions,
  highlightedRegionId,
  expandedRegionIds,
  query,
  onGroupToggle,
  onRegionToggle,
}) => (
  <section className="ref-group">
    {/* Group header row */}
    <button
      className={`ref-group__header${isOpen ? ' ref-group__header--open' : ''}`}
      onClick={() => onGroupToggle(group.id)}
      aria-expanded={isOpen}
    >
      <span className="ref-group__icon">{group.icon}</span>
      <span className="ref-group__label">
        <Highlight text={group.label} query={query} />
      </span>
      <span className="ref-group__count">{visibleRegions.length}</span>
      <span className="ref-group__arrow">{isOpen ? '▾' : '▸'}</span>
    </button>

    {/* Collapsible body */}
    <div className={`ref-group__body${isOpen ? ' ref-group__body--open' : ''}`}>
      {/* Group description — shown at top of expanded panel */}
      <p className="ref-group__desc">
        <Highlight text={group.description} query={query} />
      </p>

      {/* Region list */}
      {visibleRegions.map((region) => (
        <RegionRow
          key={region.id}
          region={region}
          isHighlighted={highlightedRegionId === region.id}
          isExpanded={expandedRegionIds.has(region.id)}
          query={query}
          onToggle={onRegionToggle}
        />
      ))}
    </div>
  </section>
);

// ── MegBandRow — one frequency-band entry in the MEG tab ──────────────────────

interface MegBandRowProps {
  band:       MegBand;
  isExpanded: boolean;
  query:      string;
  onToggle:   (id: string) => void;
}

const MegBandRow: FC<MegBandRowProps> = ({ band, isExpanded, query, onToggle }) => (
  <div className="ref-meg-band">
    <button
      className="ref-meg-band__header"
      onClick={() => onToggle(band.id)}
      aria-expanded={isExpanded}
    >
      <span className="ref-meg-band__label">
        <Highlight text={band.label} query={query} />
      </span>
      <span className="ref-meg-band__range">
        <Highlight text={band.freqRange} query={query} />
      </span>
      <span className="ref-meg-band__fn">
        <Highlight text={band.function} query={query} />
      </span>
      <span className="ref-meg-band__chevron">{isExpanded ? '▾' : '▸'}</span>
    </button>
    <div className={`ref-meg-band__body${isExpanded ? ' ref-meg-band__body--open' : ''}`}>
      <p className="ref-meg-band__detail">
        <Highlight text={band.detail} query={query} />
      </p>
    </div>
  </div>
);

// ── ErpWaveCard — a single ERP wave definition card ───────────────────────────

interface ErpWaveCardProps {
  wave:  ErpWave;
  query: string;
}

const ErpWaveCard: FC<ErpWaveCardProps> = ({ wave, query }) => (
  <div className="ref-erp-wave">
    <h4 className="ref-erp-wave__title">
      <Highlight text={wave.label} query={query} />
    </h4>
    <dl className="ref-erp-wave__meta">
      <dt>Latency</dt>
      <dd><Highlight text={wave.latency} query={query} /></dd>
      <dt>Distribution</dt>
      <dd><Highlight text={wave.distribution} query={query} /></dd>
      <dt>Interpretation</dt>
      <dd><Highlight text={wave.interpretation} query={query} /></dd>
      <dt>Computation</dt>
      <dd><Highlight text={wave.mathematicalBasis} query={query} /></dd>
    </dl>
  </div>
);

// ── Main drawer component ──────────────────────────────────────────────────────

/**
 * The collapsible Contextual Reference Panel.
 * Mount anywhere inside ReferencePanelProvider — all state comes from context.
 */
const ReferenceDrawer: FC = () => {
  const {
    isOpen,
    activeTab,
    expandedGroupId,
    highlightedRegionId,
    closeDrawer,
    setActiveTab,
  } = useReferencePanel();

  // ── Local state: search query (global — persists across tab switches) ─────
  const [query, setQuery] = useState('');

  // Normalised query for matching (trimmed, lower-cased)
  const q = query.toLowerCase().trim();

  // ── Local state: which anatomy GROUPS are expanded ────────────────────────
  const [openGroupIds, setOpenGroupIds] = useState<Set<string>>(new Set());

  // ── Local state: which anatomy REGIONS are expanded (within open groups) ──
  const [openRegionIds, setOpenRegionIds] = useState<Set<string>>(new Set());

  // ── Local state: which MEG band rows are expanded ─────────────────────────
  const [openBandIds,  setOpenBandIds]  = useState<Set<string>>(new Set());
  const [erpOpen,      setErpOpen]      = useState(false);
  const [hrfOpen,      setHrfOpen]      = useState(false);
  const [topomapOpen,  setTopomapOpen]  = useState(false);

  // Search input ref for keyboard shortcut focus
  const searchRef = useRef<HTMLInputElement>(null);

  // ── React to 3D navigation: auto-expand the target group + region ─────────
  useEffect(() => {
    if (!expandedGroupId) return;
    setOpenGroupIds((prev) => new Set([...prev, expandedGroupId]));
    if (highlightedRegionId) {
      setOpenRegionIds((prev) => new Set([...prev, highlightedRegionId]));
    }
    setActiveTab('anatomy');
  }, [expandedGroupId, highlightedRegionId, setActiveTab]);

  // ── Toggle helpers ────────────────────────────────────────────────────────
  const toggleGroup = useCallback((id: string) => {
    setOpenGroupIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) { next.delete(id); } else { next.add(id); }
      return next;
    });
  }, []);

  const toggleRegion = useCallback((id: string) => {
    setOpenRegionIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) { next.delete(id); } else { next.add(id); }
      return next;
    });
  }, []);

  const toggleBand = useCallback((id: string) => {
    setOpenBandIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) { next.delete(id); } else { next.add(id); }
      return next;
    });
  }, []);

  // ── Anatomy search filter ─────────────────────────────────────────────────
  // Returns the subset of regions in this group that match the query.
  const filteredRegions = useCallback(
    (group: NeuroGroup): NeuroRegion[] => {
      if (!q) return group.regions;
      return group.regions.filter(
        (r) =>
          r.name.toLowerCase().includes(q) ||
          r.function.toLowerCase().includes(q) ||
          r.detail.toLowerCase().includes(q),
      );
    },
    [q],
  );

  // Whether a group should be shown on the anatomy tab at all
  const isGroupVisible = useCallback(
    (group: NeuroGroup): boolean => {
      if (!q) return true;
      return (
        group.label.toLowerCase().includes(q) ||
        group.description.toLowerCase().includes(q) ||
        filteredRegions(group).length > 0
      );
    },
    [q, filteredRegions],
  );

  // Whether a group is forced open (manual toggle OR search match)
  const isGroupOpen = useCallback(
    (group: NeuroGroup): boolean =>
      openGroupIds.has(group.id) || (q.length > 0 && filteredRegions(group).length > 0),
    [openGroupIds, q, filteredRegions],
  );

  // Escape key closes the drawer
  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      if (e.key === 'Escape') closeDrawer();
    },
    [closeDrawer],
  );

  const { anatomy, megElectrophysiology, fmriHemodynamics } = neuroDictionary;

  return (
    <div
      className={`ref-drawer${isOpen ? ' ref-drawer--open' : ''}`}
      role="complementary"
      aria-label="Contextual Reference Panel"
      onKeyDown={handleKeyDown}
    >
      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <div className="ref-drawer__header">
        <span className="ref-drawer__title">
          <span className="ref-drawer__title-icon">📖</span>
          Contextual Reference
        </span>
        <button
          className="ref-drawer__close"
          onClick={closeDrawer}
          aria-label="Close reference panel"
          title="Close (Esc)"
        >
          ✕
        </button>
      </div>

      {/* ── Search bar — global, always visible, marks text on both tabs ──────── */}
      <div className="ref-drawer__search-wrap">
        <span className="ref-drawer__search-icon">🔍</span>
        <input
          ref={searchRef}
          className="ref-drawer__search"
          type="search"
          placeholder="Search & highlight — e.g. memory, gamma, P300…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search and highlight guide entries"
        />
        {query && (
          <button
            className="ref-drawer__search-clear"
            onClick={() => setQuery('')}
            aria-label="Clear search"
          >
            ✕
          </button>
        )}
      </div>

      {/* ── Tabs ────────────────────────────────────────────────────────────── */}
      <div className="ref-drawer__tabs" role="tablist">
        {(
          [
            { id: 'anatomy',           label: 'Anatomical Cipher'           },
            { id: 'electrophysiology', label: 'Electrophysiological Cipher' },
          ] as { id: ReferenceTab; label: string }[]
        ).map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`ref-tab${activeTab === tab.id ? ' ref-tab--active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* ── Scrollable body ──────────────────────────────────────────────────── */}
      <div className="ref-drawer__body">

        {/* ══ ANATOMY TAB ════════════════════════════════════════════════════ */}
        {activeTab === 'anatomy' && (
          <div role="tabpanel">
            {anatomy
              .filter(isGroupVisible)
              .map((group) => (
                <GroupAccordion
                  key={group.id}
                  group={group}
                  isOpen={isGroupOpen(group)}
                  visibleRegions={filteredRegions(group)}
                  highlightedRegionId={highlightedRegionId}
                  expandedRegionIds={openRegionIds}
                  query={q}
                  onGroupToggle={toggleGroup}
                  onRegionToggle={toggleRegion}
                />
              ))}

            {/* Empty-state when search has no matches */}
            {anatomy.filter(isGroupVisible).length === 0 && (
              <p className="ref-empty">No structures match "{query}".</p>
            )}
          </div>
        )}

        {/* ══ ELECTROPHYSIOLOGY TAB ══════════════════════════════════════════ */}
        {activeTab === 'electrophysiology' && (
          <div role="tabpanel">

            {/* MEG frequency bands */}
            <h3 className="ref-section-heading">MEG Frequency Bands</h3>
            {megElectrophysiology.bands.map((band) => (
              <MegBandRow
                key={band.id}
                band={band}
                isExpanded={openBandIds.has(band.id)}
                query={q}
                onToggle={toggleBand}
              />
            ))}

            {/* ERP section */}
            <h3 className="ref-section-heading" style={{ marginTop: 20 }}>
              Event-Related Potentials
            </h3>
            <div className="ref-meg-band">
              <button
                className="ref-meg-band__header"
                onClick={() => setErpOpen((v) => !v)}
                aria-expanded={erpOpen}
              >
                <span className="ref-meg-band__label">
                  <Highlight text={megElectrophysiology.erp.label} query={q} />
                </span>
                <span className="ref-meg-band__fn">
                  <Highlight
                    text={megElectrophysiology.erp.description.slice(0, 60) + '…'}
                    query={q}
                  />
                </span>
                <span className="ref-meg-band__chevron">{erpOpen ? '▾' : '▸'}</span>
              </button>
              <div className={`ref-meg-band__body${erpOpen ? ' ref-meg-band__body--open' : ''}`}>
                <p className="ref-meg-band__detail" style={{ marginBottom: 12 }}>
                  <Highlight text={megElectrophysiology.erp.description} query={q} />
                </p>
                {megElectrophysiology.erp.waves.map((wave: ErpWave) => (
                  <ErpWaveCard key={wave.id} wave={wave} query={q} />
                ))}
              </div>
            </div>

            {/* MEG Topomap */}
            <h3 className="ref-section-heading" style={{ marginTop: 20 }}>MEG Topomap</h3>
            <div className="ref-meg-band">
              <button
                className="ref-meg-band__header"
                onClick={() => setTopomapOpen((v) => !v)}
                aria-expanded={topomapOpen}
              >
                <span className="ref-meg-band__label">
                  <Highlight text="Field Map (Topomap)" query={q} />
                </span>
                <span className="ref-meg-band__fn">
                  <Highlight text="Scalp magnetic field at a time window" query={q} />
                </span>
                <span className="ref-meg-band__chevron">{topomapOpen ? '▾' : '▸'}</span>
              </button>
              <div className={`ref-meg-band__body${topomapOpen ? ' ref-meg-band__body--open' : ''}`}>
                <p className="ref-meg-band__detail">
                  <Highlight
                    text="A topomap (topographic map) shows the instantaneous magnetic field across the scalp surface, averaged over the selected time window. Each point on the scalp disc is colour-coded: blue = field directed into the head; red = field directed out; white = near zero."
                    query={q}
                  />
                </p>
                <p className="ref-meg-band__detail" style={{ marginTop: 8 }}>
                  <Highlight
                    text="Reading a dipolar pattern: a paired red + blue blob is the hallmark of a single equivalent current dipole — a small patch of synchronously firing cortex. The source lies roughly where the field gradient between the two lobes is steepest (the zero-crossing line). The red lobe marks where field exits the skull; blue where it re-enters."
                    query={q}
                  />
                </p>
                <p className="ref-meg-band__detail" style={{ marginTop: 8 }}>
                  <Highlight
                    text="Diffuse or symmetric maps without a clear dipole usually mean multiple overlapping sources, or a radially oriented source. MEG is blind to radial currents — only tangential sources in sulcal walls produce detectable external magnetic flux. The ± scale updates automatically; a larger range indicates stronger activation or artifact."
                    query={q}
                  />
                </p>
              </div>
            </div>

            {/* fMRI HRF methodology */}
            <h3 className="ref-section-heading" style={{ marginTop: 20 }}>
              fMRI Hemodynamics
            </h3>
            <div className="ref-meg-band">
              <button
                className="ref-meg-band__header"
                onClick={() => setHrfOpen((v) => !v)}
                aria-expanded={hrfOpen}
              >
                <span className="ref-meg-band__label">
                  <Highlight text={fmriHemodynamics.label} query={q} />
                </span>
                <span className="ref-meg-band__fn">
                  <Highlight text={fmriHemodynamics.function} query={q} />
                </span>
                <span className="ref-meg-band__chevron">{hrfOpen ? '▾' : '▸'}</span>
              </button>
              <div className={`ref-meg-band__body${hrfOpen ? ' ref-meg-band__body--open' : ''}`}>
                <p className="ref-meg-band__detail">
                  <Highlight text={fmriHemodynamics.detail} query={q} />
                </p>
              </div>
            </div>

          </div>
        )}
      </div>
    </div>
  );
};

export default ReferenceDrawer;
