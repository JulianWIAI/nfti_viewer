/**
 * analysis.types.ts — Shared types for MEG/MRI analysis API responses
 * ─────────────────────────────────────────────────────────────────────
 * Centralises the data-transfer object shapes used by megAnalysisApi.ts,
 * mriApi.ts, the canvas renderers, and the UI components so they all agree
 * on the same structure without circular imports.
 */

// ── MRI Volumetrics ───────────────────────────────────────────────────────────

export interface HippocampalVolumes {
  /** Left hippocampal volume in mm³ (FreeSurfer label 17). */
  left_mm3:           number;
  /** Right hippocampal volume in mm³ (FreeSurfer label 53). */
  right_mm3:          number;
  /** (L − R) / mean × 100 % — positive means left larger. */
  asymmetry_index:    number;
  /** Normative mean per hemisphere used as the reference line in the chart. */
  normative_mean_mm3: number;
  /** Normative 1-SD used to draw the reference band. */
  normative_sd_mm3:   number;
}

export interface TissueVolumes {
  gm_mm3:      number;   // total gray matter in mm³
  wm_mm3:      number;   // total white matter in mm³
  csf_mm3:     number;   // total CSF in mm³
  gm_fraction: number;   // GM / (GM + WM + CSF)
  wm_fraction: number;
  csf_fraction: number;
}

export interface VolumetricsResult {
  hippocampus:      HippocampalVolumes;
  tissue_volumes:   TissueVolumes;
  voxel_volume_mm3: number;  // single-voxel volume in mm³
  total_brain_mm3:  number;  // GM + WM in mm³ (brain parenchyma)
}

// ── MEG Artefact Annotations ──────────────────────────────────────────────────

/** A single time-stamped artefact window returned by /api/meg/detect-artifacts. */
export interface ArtifactAnnotation {
  /** 'blink' = EOG eye blink   |   'muscle' = high-frequency muscle burst */
  type:     'blink' | 'muscle';
  /** Artefact start time in seconds from recording start. */
  onset:    number;
  /** Artefact window duration in seconds. */
  duration: number;
  /** Source channel name (e.g. 'EOG') or 'all' for sensor-space artefacts. */
  channel:  string;
}

export interface ArtifactResult {
  session_id:  string;
  n_blinks:    number;
  n_muscle:    number;
  annotations: ArtifactAnnotation[];
}

// ── MEG Spike Markers ─────────────────────────────────────────────────────────

/** A single epileptiform-like transient event from /api/meg/detect-spikes. */
export interface SpikeMarker {
  /** Peak crossing time in seconds. */
  time:      number;
  /** Channel names whose amplitude exceeded the MAD threshold. */
  channels:  string[];
  /** Maximum absolute amplitude across active channels (SI units). */
  amplitude: number;
}

export interface SpikeResult {
  session_id:    string;
  n_spikes:      number;
  threshold_mad: number;
  spikes:        SpikeMarker[];
}

// ── MEG Frequency Band Power ──────────────────────────────────────────────────

/** Relative PSD power for the five standard neurological frequency bands. */
export interface BandPower {
  delta: number;   // 1–4 Hz
  theta: number;   // 4–8 Hz
  alpha: number;   // 8–12 Hz
  beta:  number;   // 12–30 Hz
  gamma: number;   // 30–50 Hz
}

export interface FrequencyBandsResult {
  session_id:    string;
  n_channels:    number;
  channel_types: string[];
  bands:         BandPower;
}
