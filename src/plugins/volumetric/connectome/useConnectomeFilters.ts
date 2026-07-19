/**
 * useConnectomeFilters.ts — useReducer hook for connectome filter state
 * ──────────────────────────────────────────────────────────────────────
 *
 * WHY useReducer INSTEAD OF MULTIPLE useStates?
 * ───────────────────────────────────────────────
 * The three filter fields (edgeThreshold, centralityMin, overlayVisible) are
 * logically coupled:
 *   • Both threshold sliders share a "reset" action that snaps all three back
 *     to defaults in a single dispatch (avoids three re-renders).
 *   • The reducer makes each state transition explicit and serialisable, which
 *     simplifies future additions (e.g. "IMPORT_PRESET", "LINK_TO_URL").
 *   • useReducer produces a stable `dispatch` reference — used as the stable
 *     callback identity in the RAF debounce handlers.
 *
 * USAGE
 * ──────
 *   const { state, setEdgeThreshold, setCentralityMin, setOverlayVisible, reset }
 *     = useConnectomeFilters();
 *
 * The four named helpers are memoised with useCallback so they are safe to pass
 * as event handler props to child components without causing re-renders.
 */

import { useReducer, useCallback } from 'react';
import type {
  ConnectomeFilterState,
  ConnectomeFilterAction,
} from './connectomePanelTypes';

// ── Default state ─────────────────────────────────────────────────────────────

export const DEFAULT_FILTER_STATE: ConnectomeFilterState = {
  edgeThreshold:  10,    // matches ConnectomeOptions.edgeThreshold default
  centralityMin:  0,     // show all nodes
  overlayVisible: true,
};

// ── Reducer ───────────────────────────────────────────────────────────────────

function filterReducer(
  state:  ConnectomeFilterState,
  action: ConnectomeFilterAction,
): ConnectomeFilterState {
  switch (action.type) {
    case 'SET_EDGE_THRESHOLD':
      return { ...state, edgeThreshold: action.value };
    case 'SET_CENTRALITY_MIN':
      return { ...state, centralityMin: action.value };
    case 'SET_OVERLAY_VISIBLE':
      return { ...state, overlayVisible: action.value };
    case 'RESET':
      return DEFAULT_FILTER_STATE;
    default:
      return state;
  }
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export interface ConnectomeFiltersHook {
  /** Current filter state — safe to read in render. */
  state:             ConnectomeFilterState;
  /**
   * Set the edge threshold slider value.
   * VTK update is handled by the RAF callback in ConnectomePanel, NOT by a
   * useEffect watching this value — that keeps the update off the React
   * render-flush cycle for maximum slider responsiveness.
   */
  setEdgeThreshold:  (v: number) => void;
  /**
   * Set the minimum betweenness centrality slider value.
   * Same scheduling note as setEdgeThreshold.
   */
  setCentralityMin:  (v: number) => void;
  /** Toggle the entire connectome overlay on/off. */
  setOverlayVisible: (v: boolean) => void;
  /** Reset all filters to their defaults. */
  reset:             () => void;
}

export function useConnectomeFilters(): ConnectomeFiltersHook {
  const [state, dispatch] = useReducer(filterReducer, DEFAULT_FILTER_STATE);

  const setEdgeThreshold  = useCallback((v: number)  => dispatch({ type: 'SET_EDGE_THRESHOLD',  value: v }), []);
  const setCentralityMin  = useCallback((v: number)  => dispatch({ type: 'SET_CENTRALITY_MIN',  value: v }), []);
  const setOverlayVisible = useCallback((v: boolean) => dispatch({ type: 'SET_OVERLAY_VISIBLE', value: v }), []);
  const reset             = useCallback(()           => dispatch({ type: 'RESET' }), []);

  return { state, setEdgeThreshold, setCentralityMin, setOverlayVisible, reset };
}
