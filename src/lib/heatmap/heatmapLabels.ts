/**
 * heatmapLabels.ts — Axis label abbreviations and per-label metadata
 * ────────────────────────────────────────────────────────────────────
 *
 * Two public exports:
 *
 *   NAME_ABBREV    — static map from BRAIN_LABELS.name → short axis label
 *   buildLabelInfoArray() — converts an ordered node_ids list into the
 *                           HeatmapLabelInfo[] array used by the renderers
 *
 * ABBREVIATION STRATEGY
 * ──────────────────────
 * Each abbreviation is:
 *   • ≤ 8 characters so it fits in the ~72 px rotated label band
 *   • anatomically unambiguous within the 32-structure SynthSeg set
 *   • uses (L)/(R) suffixes for bilateral structures
 *   • falls back to first-7-chars + "…" for any unlisted name
 *
 * TISSUE GROUP COLORS
 * ────────────────────
 *   GM   →  #e87d7d  (warm rose)
 *   WM   →  #dcdcdc  (light grey)
 *   CSF  →  #64aae0  (sky blue)
 *   ?    →  #888888  (neutral — for unmapped FreeSurfer labels)
 */

import { BRAIN_LABELS }     from '../vtk/labelVisibility';
import type { HeatmapLabelInfo } from './heatmapTypes';

// ── Tissue group → CSS colour ─────────────────────────────────────────────────

const GROUP_RGB: Record<string, string> = {
  gm:      '#e87d7d',
  wm:      '#dcdcdc',
  csf:     '#64aae0',
  unknown: '#888888',
};

// ── Static abbreviation map ───────────────────────────────────────────────────
// Keys match BRAIN_LABELS[*].name exactly.

export const NAME_ABBREV: ReadonlyMap<string, string> = new Map<string, string>([
  // White Matter
  ['Cerebral WM (L)',      'CerbWM-L'],
  ['Cerebral WM (R)',      'CerbWM-R'],
  ['Cerebellar WM (L)',    'CblrWM-L'],
  ['Cerebellar WM (R)',    'CblrWM-R'],
  ['Brain Stem',           'BStem'],
  // Cortex
  ['Cerebral Cortex (L)',  'CerbCx-L'],
  ['Cerebral Cortex (R)',  'CerbCx-R'],
  ['Cerebellar Ctx (L)',   'CblrCx-L'],
  ['Cerebellar Ctx (R)',   'CblrCx-R'],
  // Subcortical GM
  ['Thalamus (L)',         'Thal-L'],
  ['Thalamus (R)',         'Thal-R'],
  ['Caudate (L)',          'Caud-L'],
  ['Caudate (R)',          'Caud-R'],
  ['Putamen (L)',          'Puta-L'],
  ['Putamen (R)',          'Puta-R'],
  ['Pallidum (L)',         'Pall-L'],
  ['Pallidum (R)',         'Pall-R'],
  ['Hippocampus (L)',      'Hipp-L'],
  ['Hippocampus (R)',      'Hipp-R'],
  ['Amygdala (L)',         'Amyg-L'],
  ['Amygdala (R)',         'Amyg-R'],
  ['Accumbens (L)',        'Accu-L'],
  ['Accumbens (R)',        'Accu-R'],
  ['Ventral DC (L)',       'VDC-L'],
  ['Ventral DC (R)',       'VDC-R'],
  // CSF
  ['Lat. Ventricle (L)',   'LatV-L'],
  ['Lat. Ventricle (R)',   'LatV-R'],
  ['Inf. Lat. Vent. (L)',  'ILV-L'],
  ['Inf. Lat. Vent. (R)',  'ILV-R'],
  ['3rd Ventricle',        '3rdV'],
  ['4th Ventricle',        '4thV'],
  ['CSF',                  'CSF'],
]);

// ── Module-load lookup from FreeSurfer ID string to BRAIN_LABELS entry ────────

const ID_TO_LABEL = new Map(BRAIN_LABELS.map((l) => [String(l.id), l]));

// ── buildLabelInfoArray ───────────────────────────────────────────────────────

/**
 * Build the per-position label metadata array consumed by the axis renderers.
 *
 * @param nodeIds  Ordered list of FreeSurfer label strings from the API response.
 *                 The index matches the matrix row/column index.
 * @returns        One HeatmapLabelInfo per entry in nodeIds, in the same order.
 */
export function buildLabelInfoArray(nodeIds: string[]): HeatmapLabelInfo[] {
  return nodeIds.map((labelId) => {
    const meta  = ID_TO_LABEL.get(labelId);
    const name  = meta?.name  ?? `Lbl${labelId}`;
    const group = meta?.group ?? 'unknown';

    // Abbreviation: known name → static map; unknown → truncate + ellipsis.
    const abbrev = NAME_ABBREV.get(name) ?? (name.length > 8 ? name.slice(0, 7) + '…' : name);

    return {
      labelId,
      abbrev,
      group,
      groupRgb: GROUP_RGB[group] ?? GROUP_RGB['unknown']!,
    };
  });
}
