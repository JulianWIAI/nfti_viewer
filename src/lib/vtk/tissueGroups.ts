/**
 * tissueGroups.ts — FreeSurfer label → tissue class mapping for group-level
 * visibility control in the segmentation overlay.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * The SynthSeg segmentation produces 33 individual brain-structure labels
 * (FreeSurfer IDs).  This module groups them into three clinical macrostructures:
 *
 *   Gray Matter  (gm)  — cortex, hippocampus, subcortical nuclei, cerebellar cortex
 *   White Matter (wm)  — cerebral WM, cerebellar WM, brain stem
 *   CSF                — lateral / 3rd / 4th ventricles, extra-ventricular CSF
 *
 * USAGE IN THE OVERLAY
 * ────────────────────
 * buildGroupVisibilityLut() takes the current GM/WM/CSF visibility flags and
 * returns a modified 256×4 RGBA lookup table (same format as LABEL_RGBA_LUT_BASE
 * in segmentationOverlay.ts) with the alpha channel zeroed for any hidden group.
 * The overlay's updateGroupVisibility() method calls this, rebuilds the per-voxel
 * RGBA array in-place, and calls vtkDataArray.modified() to trigger a re-render.
 */

// ── Tissue class identifier ───────────────────────────────────────────────────

export type TissueClass = 'gm' | 'wm' | 'csf';

/** Which tissue macro-classes are currently visible in the overlay. */
export type TissueGroupVisibility = Record<TissueClass, boolean>;

// ── Default visibility (all groups on) ───────────────────────────────────────

export const DEFAULT_TISSUE_VISIBILITY: TissueGroupVisibility = {
  gm: true, wm: true, csf: true,
};

// ── FreeSurfer label ID → tissue class ───────────────────────────────────────
// Keys are the FreeSurfer integer label IDs produced by SynthSeg v2.
// Background (0) is absent — it is always transparent regardless of visibility.

export const LABEL_TISSUE_MAP: ReadonlyMap<number, TissueClass> = new Map<number, TissueClass>([
  // ── Left hemisphere ────────────────────────────────────────────────────────
  [ 2, 'wm'],  // left cerebral white matter
  [ 3, 'gm'],  // left cerebral cortex
  [ 4, 'csf'], // left lateral ventricle
  [ 5, 'csf'], // left inferior lateral ventricle
  [ 7, 'wm'],  // left cerebellum white matter
  [ 8, 'gm'],  // left cerebellum cortex
  [10, 'gm'],  // left thalamus        (subcortical gray matter)
  [11, 'gm'],  // left caudate         (subcortical GM)
  [12, 'gm'],  // left putamen         (subcortical GM)
  [13, 'gm'],  // left pallidum        (subcortical GM)
  [14, 'csf'], // 3rd ventricle
  [15, 'csf'], // 4th ventricle
  [16, 'wm'],  // brain stem           (predominantly myelinated WM)
  [17, 'gm'],  // left hippocampus     (medial temporal GM)
  [18, 'gm'],  // left amygdala        (medial temporal GM)
  [24, 'csf'], // CSF                  (extra-ventricular / sulcal)
  [26, 'gm'],  // left accumbens area  (subcortical GM)
  [28, 'gm'],  // left ventral DC      (subcortical GM)
  // ── Right hemisphere ───────────────────────────────────────────────────────
  [41, 'wm'],  // right cerebral white matter
  [42, 'gm'],  // right cerebral cortex
  [43, 'csf'], // right lateral ventricle
  [44, 'csf'], // right inferior lateral ventricle
  [46, 'wm'],  // right cerebellum white matter
  [47, 'gm'],  // right cerebellum cortex
  [49, 'gm'],  // right thalamus
  [50, 'gm'],  // right caudate
  [51, 'gm'],  // right putamen
  [52, 'gm'],  // right pallidum
  [53, 'gm'],  // right hippocampus
  [54, 'gm'],  // right amygdala
  [58, 'gm'],  // right accumbens area
  [60, 'gm'],  // right ventral DC
]);

// ── Human-readable labels for UI controls ─────────────────────────────────────

export const TISSUE_CLASS_LABELS: Record<TissueClass, string> = {
  gm:  'Gray Matter',
  wm:  'White Matter',
  csf: 'CSF',
};

// ── Accent colours for each class (for UI badge dots, legends, etc.) ──────────
// These are CSS hex strings — NOT the colors in the vtk.js RGBA LUT.
// The vtk.js overlay retains per-structure SYNTHSEG_COLORS even when grouped.

export const TISSUE_CSS_COLORS: Record<TissueClass, string> = {
  gm:  '#e87d7d', // warm rose
  wm:  '#dcdcdc', // near-white
  csf: '#64aae0', // sky blue
};

// ── Pre-built sets for fast membership checks ─────────────────────────────────
// Used by mriApi.ts to avoid iterating the full map on every render.

export const LABELS_BY_CLASS: Record<TissueClass, ReadonlySet<number>> = {
  gm:  new Set([ 3, 8, 10, 11, 12, 13, 17, 18, 26, 28, 42, 47, 49, 50, 51, 52, 53, 54, 58, 60]),
  wm:  new Set([ 2, 7, 16, 41, 46]),
  csf: new Set([ 4, 5, 14, 15, 24, 43, 44]),
};

// ── LUT manipulation ──────────────────────────────────────────────────────────

/**
 * Build a 256×4 RGBA lookup table identical to the base SYNTHSEG LUT but with
 * the alpha channel zeroed for every label that belongs to a hidden tissue class.
 *
 * @param visibility   Current GM/WM/CSF visibility flags.
 * @param baseLut      The 1024-byte base LUT from segmentationOverlay.ts
 *                     (``LABEL_RGBA_LUT_BASE``).  Cloned — never mutated.
 * @returns            A new Uint8Array(1024) with hidden groups made transparent.
 */
export function buildGroupVisibilityLut(
  visibility: TissueGroupVisibility,
  baseLut:    Uint8Array,
): Uint8Array {
  // Clone so we never mutate the module-level constant.
  const lut = new Uint8Array(baseLut);

  for (const [labelId, tissueClass] of LABEL_TISSUE_MAP) {
    if (!visibility[tissueClass]) {
      // Zero the alpha component for this label ID → fully transparent.
      lut[labelId * 4 + 3] = 0;
    }
  }

  return lut;
}

/**
 * Rebuild the per-voxel RGBA flat array by applying a custom LUT.
 * This is O(n_voxels) and suitable for interactive use (a typical 256³ scan
 * has ~16M voxels; a modern JS engine processes this in < 100 ms).
 *
 * @param labelFlat   Original flat uint8 label array (Fortran order, read-only).
 * @param lut         256×4 RGBA lookup table (1024 bytes).
 * @returns           New Uint8Array of length labelFlat.length × 4.
 */
export function applyLutToLabels(labelFlat: Uint8Array, lut: Uint8Array): Uint8Array {
  const rgba = new Uint8Array(labelFlat.length * 4);
  for (let i = 0; i < labelFlat.length; i++) {
    const id  = labelFlat[i]! & 0xff;  // clamp to uint8 range
    const src = id * 4;
    const dst = i  * 4;
    rgba[dst    ] = lut[src    ]!;
    rgba[dst + 1] = lut[src + 1]!;
    rgba[dst + 2] = lut[src + 2]!;
    rgba[dst + 3] = lut[src + 3]!;
  }
  return rgba;
}
