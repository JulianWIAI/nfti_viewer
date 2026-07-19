/**
 * connectomeLabelColor.ts — FreeSurfer label ID → RGB color for connectome nodes
 * ────────────────────────────────────────────────────────────────────────────────
 *
 * PURPOSE
 * ────────
 * Maps each connectome node (identified by its FreeSurfer integer label string)
 * to an RGB color triple using the same SYNTHSEG palette that the segmentation
 * overlay uses.  This ensures visual coherence: a node sphere rendered in the
 * 3-D connectome view has exactly the same color as the corresponding label
 * region in the 2-D segmentation overlay.
 *
 * SOURCE OF COLORS
 * ─────────────────
 * Colors are read directly from `LABEL_RGBA_LUT_BASE` exported by
 * segmentationOverlay.ts.  That constant is a pre-built 256-entry RGBA LUT
 * (1024 bytes) where `LUT[id * 4 .. id * 4 + 2]` = (R, G, B) in [0, 255].
 *
 * The alpha component is ignored — the node actor's global opacity is
 * controlled separately via `ConnectomeOptions.nodeOpacity`.
 *
 * FALLBACK HIERARCHY
 * ───────────────────
 * If a label ID is not in the SYNTHSEG palette (e.g., a label produced by
 * a future SynthSeg model version), we fall back:
 *   1. Tissue class color from LABEL_TISSUE_MAP (gm / wm / csf).
 *   2. Neutral gray (200, 200, 200) for completely unknown labels.
 *
 * TISSUE CLASS COLORS (fallback RGB)
 * ────────────────────────────────────
 *   Gray matter  → (232, 125, 125)   warm rose   — matches TISSUE_CSS_COLORS.gm
 *   White matter → (220, 220, 220)   near-white  — matches TISSUE_CSS_COLORS.wm
 *   CSF          → (100, 170, 224)   sky blue    — matches TISSUE_CSS_COLORS.csf
 */

import { LABEL_RGBA_LUT_BASE }  from '../segmentationOverlay';
import { LABEL_TISSUE_MAP }      from '../tissueGroups';
import type { TissueClass }      from '../tissueGroups';

// ── Tissue class fallback RGB values ─────────────────────────────────────────
// Numeric equivalents of TISSUE_CSS_COLORS from tissueGroups.ts.
// Kept here as numbers so we never need to parse CSS strings at runtime.

const TISSUE_RGB: Record<TissueClass, [number, number, number]> = {
  gm:  [232, 125, 125],   // #e87d7d — warm rose
  wm:  [220, 220, 220],   // #dcdcdc — near-white
  csf: [100, 170, 224],   // #64aae0 — sky blue
};

/** Neutral gray returned for label IDs not found in either the LUT or the tissue map. */
const UNKNOWN_RGB: [number, number, number] = [200, 200, 200];

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Convert a FreeSurfer label ID (as a decimal string) to an 8-bit RGB triple.
 *
 * The returned array is a new tuple — safe to write into a Uint8Array without
 * worrying about aliasing.
 *
 * @param labelStr  FreeSurfer label as a decimal string, e.g. "17" for
 *                  left hippocampus.
 * @returns         [R, G, B] in [0–255].
 *
 * @example
 *   labelToRgb("17")  // → [220, 216, 20]  left hippocampus (gold)
 *   labelToRgb("2")   // → [245, 245, 245]  left cerebral WM (white)
 *   labelToRgb("999") // → [200, 200, 200]  unknown (neutral gray)
 */
export function labelToRgb(labelStr: string): [number, number, number] {
  const id = parseInt(labelStr, 10);

  // Reject non-integer or out-of-range IDs immediately.
  if (isNaN(id) || id < 0 || id > 255) return UNKNOWN_RGB;

  // ── Primary lookup: SYNTHSEG RGBA LUT ─────────────────────────────────────
  // LABEL_RGBA_LUT_BASE is a Readonly<Uint8Array> of length 1024 (256 × 4).
  // Background (id=0) is (0,0,0,0) — both RGB and alpha are zero.
  // Real structures have at least one non-zero color channel.
  const r = LABEL_RGBA_LUT_BASE[id * 4    ] ?? 0;
  const g = LABEL_RGBA_LUT_BASE[id * 4 + 1] ?? 0;
  const b = LABEL_RGBA_LUT_BASE[id * 4 + 2] ?? 0;

  // If any channel is non-zero the LUT has a valid entry for this label.
  if (r !== 0 || g !== 0 || b !== 0) {
    return [r, g, b];
  }

  // ── Fallback: tissue class color ──────────────────────────────────────────
  const tissueClass = LABEL_TISSUE_MAP.get(id);
  if (tissueClass !== undefined) {
    return TISSUE_RGB[tissueClass];
  }

  // ── Last resort: neutral gray ─────────────────────────────────────────────
  return UNKNOWN_RGB;
}
