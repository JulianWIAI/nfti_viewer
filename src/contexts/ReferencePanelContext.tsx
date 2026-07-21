/**
 * ReferencePanelContext.tsx — Reactive bridge between 3-D brain picks and the drawer
 * ─────────────────────────────────────────────────────────────────────────────────────
 *
 * Provides a `useReferencePanel()` hook consumed by any descendant.
 * The key entry point is `navigateToRegion(freeSurferLabel)` — call this from
 * a vtk.js pick event and the Reference Drawer automatically opens, switches to
 * the Anatomy tab, expands the matching accordion group, and highlights the
 * specific region.
 *
 * VTK.JS INTEGRATION PATTERN
 * ────────────────────────────
 * In FmriPanel or VolumetricViewer, attach a pick callback after building
 * the segmentation actor:
 *
 *   import vtkCellPicker from '@kitware/vtk.js/Rendering/Core/CellPicker';
 *   const { navigateToRegion } = useReferencePanel();
 *
 *   const picker = vtkCellPicker.newInstance();
 *   picker.setPickFromList(true);
 *   picker.addPickList(segmentationActor);
 *
 *   vtk.interactor.onLeftButtonRelease(() => {
 *     const [x, y] = vtk.interactor.getEventPosition();
 *     picker.pick([x, y, 0], vtk.renderer);
 *     if (picker.getCellId() >= 0) {
 *       const label = picker
 *         .getDataSet()
 *         .getPointData()
 *         .getScalars()
 *         .getValue(picker.getCellId());
 *       navigateToRegion(label);
 *     }
 *   });
 */

import {
  createContext,
  useContext,
  useState,
  useMemo,
  useCallback,
  type FC,
  type ReactNode,
} from 'react';
import { buildLabelLookup } from '../lib/neuroData';

// ── Context value shape ────────────────────────────────────────────────────────

export type ReferenceTab = 'anatomy' | 'electrophysiology';

export interface ReferencePanelContextValue {
  /** Whether the drawer is currently visible. */
  isOpen: boolean;

  /** Which of the two main tabs is active. */
  activeTab: ReferenceTab;

  /**
   * The anatomy accordion group that should be forced open.
   * Set by navigateToRegion(); the drawer reacts via useEffect.
   * null when no navigation target is pending.
   */
  expandedGroupId: string | null;

  /**
   * The specific region entry to highlight (ring glow effect in the drawer).
   * null when only a group-level navigation was requested.
   */
  highlightedRegionId: string | null;

  // ── Actions ───────────────────────────────────────────────────────────────

  /** Toggle drawer open/closed (used by the header button). */
  toggleDrawer: () => void;

  /** Open the drawer without toggling. */
  openDrawer: () => void;

  /** Close the drawer. */
  closeDrawer: () => void;

  /**
   * Switch the active tab.
   * Called programmatically by navigateToRegion (always switches to 'anatomy').
   */
  setActiveTab: (tab: ReferenceTab) => void;

  /**
   * Navigate the drawer to the anatomy entry for the given FreeSurfer label ID.
   *
   * Effects:
   *   1. Opens the drawer if it is closed.
   *   2. Switches to the 'anatomy' tab.
   *   3. Sets expandedGroupId to the group containing the label.
   *   4. Sets highlightedRegionId to the specific region.
   *
   * No-ops silently for label IDs not in the dictionary (e.g. background = 0).
   */
  navigateToRegion: (freeSurferLabel: number) => void;
}

// ── Context creation ──────────────────────────────────────────────────────────

const ReferencePanelContext = createContext<ReferencePanelContextValue | null>(null);

// ── Provider ──────────────────────────────────────────────────────────────────

interface ReferencePanelProviderProps {
  children: ReactNode;
}

/**
 * Provides the reference panel state to the entire workspace subtree.
 * Mount this above MultimodalWorkspace so that FmriPanel and VolumetricViewer
 * can both call navigateToRegion() from their vtk.js pick handlers.
 */
export const ReferencePanelProvider: FC<ReferencePanelProviderProps> = ({ children }) => {
  const [isOpen,              setIsOpen]              = useState(false);
  const [activeTab,           setActiveTab]           = useState<ReferenceTab>('anatomy');
  const [expandedGroupId,     setExpandedGroupId]     = useState<string | null>(null);
  const [highlightedRegionId, setHighlightedRegionId] = useState<string | null>(null);

  // Build the label → {groupId, regionId} lookup once (data is static)
  const labelLookup = useMemo(() => buildLabelLookup(), []);

  const toggleDrawer = useCallback(() => setIsOpen((o) => !o), []);
  const openDrawer   = useCallback(() => setIsOpen(true),      []);
  const closeDrawer  = useCallback(() => setIsOpen(false),     []);

  // navigateToRegion — the primary integration point for vtk.js click handlers
  const navigateToRegion = useCallback(
    (freeSurferLabel: number) => {
      const target = labelLookup.get(freeSurferLabel);
      if (!target) return; // unknown label (background, WM, etc.)

      setIsOpen(true);
      setActiveTab('anatomy');
      setExpandedGroupId(target.groupId);
      setHighlightedRegionId(target.regionId);
    },
    [labelLookup],
  );

  const value: ReferencePanelContextValue = {
    isOpen,
    activeTab,
    expandedGroupId,
    highlightedRegionId,
    toggleDrawer,
    openDrawer,
    closeDrawer,
    setActiveTab,
    navigateToRegion,
  };

  return (
    <ReferencePanelContext.Provider value={value}>
      {children}
    </ReferencePanelContext.Provider>
  );
};

// ── Hook ──────────────────────────────────────────────────────────────────────

/**
 * Consume the reference panel state and actions.
 * Must be called inside a ReferencePanelProvider — throws otherwise.
 */
export function useReferencePanel(): ReferencePanelContextValue {
  const ctx = useContext(ReferencePanelContext);
  if (!ctx) {
    throw new Error('useReferencePanel must be called inside a ReferencePanelProvider');
  }
  return ctx;
}
