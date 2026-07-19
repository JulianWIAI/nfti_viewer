/**
 * RawComparisonContext.ts — React context for Subject B raw-comparison state.
 *
 * Separates Subject B's rendering and segmentation state from VolumetricContext
 * (which owns Subject A).  Both contexts are provided by VolumetricViewer and
 * consumed by VolumetricControls.
 *
 * Shape
 * ─────
 *   hasVolumeB         — true once Subject B has been parsed and its VTK
 *                        pane is initialised.
 *   loadVolumeB        — callback exposed to SubjectBDropZone; parses the
 *                        NIfTI in a worker then triggers VTK B setup.
 *   controlsB          — independent MPR slice & window/level state for B.
 *   setControlsB       — partial-merge updater (same API as setControls for A).
 *   maxIB/J/KB         — dimension maxima derived from Subject B's header.
 *   inferenceStatusB   — segmentation lifecycle phase for Subject B.
 *   hasOverlayB        — true once B's seg overlay actors are mounted.
 *   showOverlayB       — visibility flag for B's seg overlay actors.
 *   setShowOverlayB    — toggle B's seg overlay visibility.
 *   labelVisibilityB   — per-FreeSurfer-ID visibility map for B's overlay.
 *   macroGroupStateB   — derived tristate (all/partial/none) per tissue class.
 *   setLabelVisibleB   — toggle a single structure by FreeSurfer label ID.
 *   setGroupVisibleB   — show/hide all structures in a tissue class.
 *   showAllLabelsB     — make every structure visible.
 *   hideAllLabelsB     — hide every structure.
 *   volumetricsB       — hippocampal volumetrics for Subject B (null until computed).
 *   volumetricsLoadingB
 *   volumetricsErrorB
 *   runVolumetricsB    — trigger volumetrics computation for Subject B.
 *   clearVolumeB       — remove Subject B and reset to single-brain mode.
 */

import { createContext, useContext } from 'react';
import type { ViewerControls } from '../../types/nifti.types';
import type { VolumetricsResult } from '../../types/analysis.types';
import type { TissueClass } from '../../lib/vtk/tissueGroups';
import type { MacroState } from '../../lib/vtk/labelVisibility';
import type { InferenceStatus } from './VolumetricViewer';

// ── Context value shape ────────────────────────────────────────────────────────

export interface ComparisonContextValue {
  /** True when Subject B's VolumePayload has been parsed and VTK B is ready. */
  hasVolumeB: boolean;
  /** Accept a new Subject B file — parses it in a worker, then builds VTK B. */
  loadVolumeB: (file: File) => void;
  /** Remove Subject B and return to single-brain layout. */
  clearVolumeB: () => void;

  // ── B slice controls ───────────────────────────────────────────────────────
  controlsB:    ViewerControls;
  setControlsB: (partial: Partial<ViewerControls>) => void;
  maxIB: number; maxJB: number; maxKB: number;

  // ── B segmentation ─────────────────────────────────────────────────────────
  inferenceStatusB:  InferenceStatus;
  hasOverlayB:       boolean;
  showOverlayB:      boolean;
  setShowOverlayB:   (v: boolean) => void;

  // ── B per-label visibility ─────────────────────────────────────────────────
  labelVisibilityB:  Record<number, boolean>;
  macroGroupStateB:  Record<TissueClass, MacroState>;
  setLabelVisibleB:  (labelId: number, visible: boolean) => void;
  setGroupVisibleB:  (group: TissueClass, visible: boolean) => void;
  showAllLabelsB:    () => void;
  hideAllLabelsB:    () => void;

  // ── B volumetrics ──────────────────────────────────────────────────────────
  volumetricsB:        VolumetricsResult | null;
  volumetricsLoadingB: boolean;
  volumetricsErrorB:   string | null;
  runVolumetricsB:     () => void;
}

// ── Context object ─────────────────────────────────────────────────────────────

export const ComparisonContext = createContext<ComparisonContextValue | null>(null);

// ── Convenience hook ───────────────────────────────────────────────────────────

export function useComparisonContext(): ComparisonContextValue {
  const ctx = useContext(ComparisonContext);
  if (!ctx) throw new Error('useComparisonContext must be used inside VolumetricViewer');
  return ctx;
}
