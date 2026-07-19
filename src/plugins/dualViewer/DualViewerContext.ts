/**
 * DualViewerContext.ts — React context for the dual-viewer side-by-side comparison
 * ──────────────────────────────────────────────────────────────────────────────────
 *
 * Provides shared state (sliders, SyN status, alignment mode, subject B file)
 * to DualVolumetricViewer and DualViewerControls without prop-drilling.
 *
 * All VTK imperative operations (adding actors, locking cameras) live in
 * DualVolumetricViewer.tsx; this context only carries plain React state that
 * the sidebar controls need to read and write.
 *
 * Phase 10 additions
 * ──────────────────
 *   AlignmentMode — discriminated union: 'raw' | 'registered'.
 *   alignmentMode — current value of the toggle.
 *   setAlignmentMode — called by AlignmentModeToggle; the viewer responds via
 *                      a useEffect that swaps actors and locks/unlocks cameras.
 *   rawBLoaded — true once Subject B has been parsed and its raw VTK actors are
 *                ready (drives toggle visibility).
 *   synCached  — true once SyN has completed at least once and the warped actor
 *                bundle is held in vtkRef (drives instant mode switching).
 */

import { createContext, useContext } from 'react';
import type { TissueGroupVisibility, TissueClass } from '../../lib/vtk/tissueGroups';
import type { MacroState } from '../../lib/vtk/labelVisibility';
import type { SynVolumetrics } from '../../services/synRegistrationApi';

export type { TissueGroupVisibility, TissueClass, MacroState, SynVolumetrics };

// ── Alignment mode ────────────────────────────────────────────────────────────

/**
 * Discriminated literal union for the dual-viewer alignment toggle.
 *
 *   'raw'        — both panes show native voxel spaces; cameras are decoupled.
 *   'registered' — Subject B is SyN-warped into Subject A's space; cameras locked.
 */
export type AlignmentMode = 'raw' | 'registered';

// ── SyN pipeline status ───────────────────────────────────────────────────────

/**
 * Discriminated union tracking the lifecycle of POST /api/registration/syn.
 *
 *   idle      → no request yet
 *   uploading → files are being sent (fetch in progress)
 *   done      → warped volume loaded into the right viewer
 *   error     → request failed, message contains a human-readable reason
 */
export type SynStatus =
  | { phase: 'idle' }
  | { phase: 'uploading' }
  | { phase: 'done'; durationMs: number }
  | { phase: 'error'; message: string };

// ── Context value ─────────────────────────────────────────────────────────────

export interface DualViewerContextValue {
  // ── Subject A ─────────────────────────────────────────────────────────────
  /** True once Subject A is loaded into the left viewer. */
  subjectALoaded: boolean;

  // ── Subject B / SyN ──────────────────────────────────────────────────────
  /** The NIfTI file the user selected for Subject B (moving scan). */
  subjectBFile: File | null;
  /** Replace the selected Subject B file. */
  setSubjectBFile: (f: File | null) => void;
  /** Current state of the SyN registration pipeline. */
  synStatus: SynStatus;
  /** True once Subject B has been warped and loaded into the right viewer. */
  warpedLoaded: boolean;
  /**
   * Start the SyN registration pipeline.
   * Reads subjectBFile and the cached Subject A File from the viewer ref.
   * No-op if either file is missing or a run is already in progress.
   */
  runSyn: () => void;

  // ── Alignment mode (Phase 10) ─────────────────────────────────────────────
  /**
   * Current alignment mode.
   *   'raw'        — both panes in native space; cameras decoupled.
   *   'registered' — Subject B SyN-warped; cameras locked.
   * Defaults to 'raw' when a new Subject B file is dropped.
   */
  alignmentMode: AlignmentMode;
  /**
   * Switch alignment mode.  Called by AlignmentModeToggle.
   * The viewer responds via a useEffect that swaps actors and adjusts cameras.
   */
  setAlignmentMode: (mode: AlignmentMode) => void;
  /**
   * True once Subject B's NIfTI has been parsed by the frontend worker and the
   * raw VTK actor bundle is built.  Controls toggle visibility.
   */
  rawBLoaded: boolean;
  /**
   * True once SyN has completed at least once and the warped actor bundle is
   * held in memory.  When true, switching to 'registered' is instant (no API call).
   */
  synCached: boolean;

  // ── Shared slice controls ─────────────────────────────────────────────────
  /** Current axial (K) slice index — applied to both left and right viewers. */
  sliceK: number;
  /** Maximum valid K index (volume depth − 1). */
  maxK: number;
  setSliceK: (v: number) => void;

  /** Current coronal (J) slice index. */
  sliceJ: number;
  /** Maximum valid J index. */
  maxJ: number;
  setSliceJ: (v: number) => void;

  /** Current sagittal (I) slice index. */
  sliceI: number;
  /** Maximum valid I index. */
  maxI: number;
  setSliceI: (v: number) => void;

  // ── Window / level ───────────────────────────────────────────────────────
  windowCenter: number;
  setWindowCenter: (v: number) => void;
  windowWidth: number;
  setWindowWidth: (v: number) => void;

  // ── Volume opacity ────────────────────────────────────────────────────────
  /** Shared 3-D volume opacity factor ∈ [0, 1] for both viewers. */
  volumeOpacity: number;
  setVolumeOpacity: (v: number) => void;

  // ── Segmentation overlay ──────────────────────────────────────────────────
  /**
   * True when SynthSeg label maps for both subjects have been decoded and
   * injected into both viewers as RGBA overlay actors.
   */
  hasSegmentation: boolean;
  /**
   * Current GM / WM / CSF group visibility flags.
   * Kept for interface compatibility; fine-grained control is via labelVisibility.
   */
  tissueVisibility: TissueGroupVisibility;
  /** Replace the tissue-group visibility record (all three classes at once). */
  setTissueVisibility: (vis: TissueGroupVisibility) => void;

  // ── Per-structure label visibility (mirrors AnatomySelector in single-brain) ──
  /**
   * Per-structure visibility keyed by FreeSurfer label ID.
   * Changing any entry triggers a rebuild of both overlay RGBA arrays via
   * buildLabelVisibilityLut (DualAnatomySelector is the primary UI for this).
   */
  labelVisibility: Record<number, boolean>;
  /**
   * Derived tristate (all / partial / none) per tissue class.
   * Drives the indeterminate state of the group-level macro checkboxes.
   */
  macroGroupState: Record<TissueClass, MacroState>;
  /** Show or hide one structure by its FreeSurfer label ID. */
  setLabelVisible: (id: number, visible: boolean) => void;
  /** Show or hide all structures that belong to a tissue class. */
  setGroupVisible: (cls: TissueClass, visible: boolean) => void;
  /** Restore visibility for all 32 SynthSeg structures. */
  showAllLabels: () => void;
  /** Hide all 32 SynthSeg structures (blank overlay). */
  hideAllLabels: () => void;

  // ── Comparative volumetrics ───────────────────────────────────────────────
  /**
   * Hippocampal volumes from each subject's raw NIfTI.
   * Null before SyN completes or when SynthSeg is unavailable.
   */
  volumetrics: SynVolumetrics | null;
}

// ── Context object and hook ───────────────────────────────────────────────────

export const DualViewerContext = createContext<DualViewerContextValue | null>(null);

/**
 * Typed hook — throws a clear error if used outside DualVolumetricViewer's
 * provider, so we catch wiring mistakes at runtime rather than seeing undefined
 * cascade failures deeper in the component tree.
 */
export function useDualViewerContext(): DualViewerContextValue {
  const ctx = useContext(DualViewerContext);
  if (!ctx) throw new Error('useDualViewerContext must be used inside DualVolumetricViewer');
  return ctx;
}
