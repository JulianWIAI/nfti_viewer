/**
 * labelVisibility.ts — Per-label visibility state for the SynthSeg overlay
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Provides the data model and LUT builder for per-structure visibility control.
 * The SynthSeg pipeline produces 32 non-background FreeSurfer label IDs.
 * This module lets callers show or hide each structure independently.
 *
 * RELATIONSHIP WITH tissueGroups.ts
 * ──────────────────────────────────
 * tissueGroups.ts provides the GM/WM/CSF class taxonomy and the base
 * RGBA LUT builder for group-level control.  This module builds on top of
 * that by adding per-label granularity:
 *
 *   tissueGroups.applyLutToLabels()   — shared LUT → voxel RGBA helper
 *   buildLabelVisibilityLut()         — per-label version, defined here
 *
 * MacroState (all/partial/none) is derived from the per-label record and
 * drives the indeterminate state of the group-level checkboxes in the UI.
 * The state itself always lives in VolumetricViewer; this module is pure data.
 */

import type { TissueClass } from './tissueGroups';

// ── MacroState — tristate reflecting aggregate visibility within a group ───────
//
// 'all'     → every label in the group is visible → group checkbox checked
// 'none'    → no label in the group is visible    → group checkbox unchecked
// 'partial' → mix of visible and hidden            → group checkbox indeterminate

export type MacroState = 'all' | 'none' | 'partial';

// ── BrainLabel — metadata for one SynthSeg structure ─────────────────────────

export interface BrainLabel {
  /** FreeSurfer integer label ID (1–255, non-zero). */
  id:    number;
  /** Human-readable anatomy name for the UI. */
  name:  string;
  /** Tissue macro-class this structure belongs to. */
  group: TissueClass;
  /** Laterality — used to pair / group left/right structures. */
  side:  'left' | 'right' | 'midline';
}

// ── BRAIN_LABELS — all 32 non-background SynthSeg structures ─────────────────
// Ordered: WM first, then GM (cortex → subcortical), then CSF.
// The order here controls display order in the AnatomySelector UI.

export const BRAIN_LABELS: ReadonlyArray<BrainLabel> = [
  // ── White Matter ─────────────────────────────────────────────────────────────
  { id:  2, name: 'Cerebral WM (L)',      group: 'wm',  side: 'left'    },
  { id: 41, name: 'Cerebral WM (R)',      group: 'wm',  side: 'right'   },
  { id:  7, name: 'Cerebellar WM (L)',    group: 'wm',  side: 'left'    },
  { id: 46, name: 'Cerebellar WM (R)',    group: 'wm',  side: 'right'   },
  { id: 16, name: 'Brain Stem',           group: 'wm',  side: 'midline' },
  // ── Gray Matter — Cortex ─────────────────────────────────────────────────────
  { id:  3, name: 'Cerebral Cortex (L)',  group: 'gm',  side: 'left'    },
  { id: 42, name: 'Cerebral Cortex (R)',  group: 'gm',  side: 'right'   },
  { id:  8, name: 'Cerebellar Ctx (L)',   group: 'gm',  side: 'left'    },
  { id: 47, name: 'Cerebellar Ctx (R)',   group: 'gm',  side: 'right'   },
  // ── Gray Matter — Subcortical ─────────────────────────────────────────────────
  { id: 10, name: 'Thalamus (L)',         group: 'gm',  side: 'left'    },
  { id: 49, name: 'Thalamus (R)',         group: 'gm',  side: 'right'   },
  { id: 11, name: 'Caudate (L)',          group: 'gm',  side: 'left'    },
  { id: 50, name: 'Caudate (R)',          group: 'gm',  side: 'right'   },
  { id: 12, name: 'Putamen (L)',          group: 'gm',  side: 'left'    },
  { id: 51, name: 'Putamen (R)',          group: 'gm',  side: 'right'   },
  { id: 13, name: 'Pallidum (L)',         group: 'gm',  side: 'left'    },
  { id: 52, name: 'Pallidum (R)',         group: 'gm',  side: 'right'   },
  { id: 17, name: 'Hippocampus (L)',      group: 'gm',  side: 'left'    },
  { id: 53, name: 'Hippocampus (R)',      group: 'gm',  side: 'right'   },
  { id: 18, name: 'Amygdala (L)',         group: 'gm',  side: 'left'    },
  { id: 54, name: 'Amygdala (R)',         group: 'gm',  side: 'right'   },
  { id: 26, name: 'Accumbens (L)',        group: 'gm',  side: 'left'    },
  { id: 58, name: 'Accumbens (R)',        group: 'gm',  side: 'right'   },
  { id: 28, name: 'Ventral DC (L)',       group: 'gm',  side: 'left'    },
  { id: 60, name: 'Ventral DC (R)',       group: 'gm',  side: 'right'   },
  // ── CSF — Ventricles ─────────────────────────────────────────────────────────
  { id:  4, name: 'Lat. Ventricle (L)',   group: 'csf', side: 'left'    },
  { id: 43, name: 'Lat. Ventricle (R)',   group: 'csf', side: 'right'   },
  { id:  5, name: 'Inf. Lat. Vent. (L)',  group: 'csf', side: 'left'    },
  { id: 44, name: 'Inf. Lat. Vent. (R)',  group: 'csf', side: 'right'   },
  { id: 14, name: '3rd Ventricle',        group: 'csf', side: 'midline' },
  { id: 15, name: '4th Ventricle',        group: 'csf', side: 'midline' },
  { id: 24, name: 'CSF',                  group: 'csf', side: 'midline' },
];

// ── LABELS_BY_GROUP — fast access by tissue class ─────────────────────────────

export const LABELS_BY_GROUP: Record<TissueClass, ReadonlyArray<BrainLabel>> = {
  gm:  BRAIN_LABELS.filter((l) => l.group === 'gm'),
  wm:  BRAIN_LABELS.filter((l) => l.group === 'wm'),
  csf: BRAIN_LABELS.filter((l) => l.group === 'csf'),
};

// ── defaultLabelVisibility — all 32 labels visible ───────────────────────────

export function defaultLabelVisibility(): Record<number, boolean> {
  const out: Record<number, boolean> = {};
  for (const label of BRAIN_LABELS) out[label.id] = true;
  return out;
}

// ── getMacroGroupState ────────────────────────────────────────────────────────

/**
 * Derive the tristate for a tissue group from the per-label visibility record.
 * Used to drive indeterminate checkboxes in the AnatomySelector UI.
 */
export function getMacroGroupState(
  group:      TissueClass,
  visibility: Record<number, boolean>,
): MacroState {
  const labels   = LABELS_BY_GROUP[group];
  let   visCount = 0;
  for (const lbl of labels) {
    if (visibility[lbl.id]) visCount++;
  }
  if (visCount === 0)             return 'none';
  if (visCount === labels.length) return 'all';
  return 'partial';
}

// ── buildLabelVisibilityLut ───────────────────────────────────────────────────

/**
 * Build a 256×4 RGBA lookup table that respects per-label visibility flags.
 *
 * For each of the 32 non-background SynthSeg labels, if the corresponding
 * entry in `visibility` is false, the alpha channel is zeroed so that label
 * becomes fully transparent in the vtk.js overlay.  All other labels retain
 * their colour and opacity from `baseLut`.
 *
 * @param visibility  Record mapping FreeSurfer label ID → visible (true/false).
 *                    IDs not present in BRAIN_LABELS are ignored.
 * @param baseLut     The 1024-byte base LUT (LABEL_RGBA_LUT_BASE from
 *                    segmentationOverlay.ts).  Cloned — never mutated.
 * @returns           New Uint8Array(1024) ready to pass to applyLutToLabels().
 */
export function buildLabelVisibilityLut(
  visibility: Record<number, boolean>,
  baseLut:    Uint8Array,
): Uint8Array {
  const lut = new Uint8Array(baseLut);
  for (const label of BRAIN_LABELS) {
    if (!visibility[label.id]) {
      // Zero the alpha byte for this label → fully transparent in the overlay.
      lut[label.id * 4 + 3] = 0;
    }
  }
  return lut;
}
