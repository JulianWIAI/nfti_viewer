/**
 * channelColors.ts — Semantic colour mapping for MEG / EEG channel types
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Single source of truth for channel-type colours shared by the standalone
 * MegViewer and the multimodal MegPanel canvas renderers.  Semantic colours
 * (mag = sky-blue, grad = green) make it immediately obvious which traces
 * are magnetometers vs. planar gradiometers — sensors that have different
 * sensitivity profiles and complementary information content.
 *
 * Elekta / MEGIN Neuromag TRIUX sensor layout (most modern MEG labs):
 *   102 sensor pods  ×  3 channels each  =  306 MEG channels total
 *   Each pod contains:
 *     1 magnetometer   (type 'mag')  — measures absolute radial field   (unit: T)
 *     2 planar grads   (type 'grad') — measure tangential field gradient (unit: T/m)
 *
 * Channel naming convention:  'MEG XYZN'
 *   XYZ = 3-digit sensor pod number (001–102)
 *   N   = 1 (magnetometer), 2 or 3 (planar gradiometers of the same pod)
 *   MNE-Python inserts a space: 'MEG 0111', 'MEG 0112', 'MEG 0113'
 */

/** MNE channel-type string → display colour. */
export const TYPE_COLORS: Readonly<Record<string, string>> = {
  mag:  '#4fc3f7',   // sky-blue — magnetometers
  grad: '#81c784',   // green    — planar gradiometers
};

/** Fallback colour for EEG, EOG, ECG, STIM, MISC, and any unrecognised type. */
export const TYPE_COLOR_DEFAULT = '#9090a0';

/**
 * Returns the display colour for a given MNE channel-type string.
 * Unknown types fall back to TYPE_COLOR_DEFAULT (neutral grey).
 */
export function typeColor(chType: string): string {
  return TYPE_COLORS[chType] ?? TYPE_COLOR_DEFAULT;
}

/**
 * Extracts the 3-digit sensor pod number from an Elekta/MEGIN channel name.
 *
 * Examples:
 *   sensorPod('MEG 0111') → '011'   (magnetometer of pod 011)
 *   sensorPod('MEG 0113') → '011'   (gradiometer of the same pod)
 *   sensorPod('EOG 061')  → null    (not a MEG channel)
 *
 * The regex handles both 'MEG0111' (no space) and 'MEG 0111' (MNE format).
 */
export function sensorPod(name: string): string | null {
  const m = /MEG\s*(\d{3})\d/.exec(name);
  return m ? m[1]! : null;
}
