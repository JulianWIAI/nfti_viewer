/**
 * bids.types.ts — Brain Imaging Data Structure (BIDS) type definitions
 * ─────────────────────────────────────────────────────────────────────
 *
 * BIDS is the community standard for organising neuroimaging data on OpenNeuro.
 * Directory layout:
 *
 *   dataset/
 *     sub-01/
 *       ses-01/
 *         anat/   ← structural MRI    (.nii.gz)
 *         func/   ← functional MRI    (.nii.gz)
 *         eeg/    ← electroenceph.    (.edf, .vhdr, .bdf)
 *         meg/    ← magnetoenceph.    (.fif, .ds, .con)
 *         ieeg/   ← intracranial EEG  (.edf, .vhdr, .nwb)
 *         nirs/   ← optical imaging   (.snirf)
 *         pet/    ← PET               (.nii.gz)
 *
 * The BidsRouter analyses file names / directory paths and populates these types.
 */

// ── Modality ─────────────────────────────────────────────────────────────────

/**
 * Top-level neuroimaging modality.
 * Drives which plugin gets mounted in the Viewer host.
 */
export type Modality =
  | 'volumetric'   // MRI (anat/func), PET  → vtk.js plugin
  | 'timeseries'   // EEG/iEEG (.edf/.bdf)  → browser-side canvas plugin
  | 'eeg'          // BrainVision EEG       → backend-powered EEG plugin
  | 'meg'          // MEG (.fif)            → backend-powered MEG plugin
  | 'nirs'         // fNIRS (.snirf)        → NIRS plugin (wraps timeseries)
  | 'unknown';     // unrecognised extension

// ── BIDS path components ──────────────────────────────────────────────────────

/**
 * Parsed components from a BIDS-conformant file path.
 * All fields are optional because users may upload individual files without
 * a full BIDS directory structure.
 */
export interface BidsPath {
  /** Subject label, e.g. "01" from sub-01. */
  subject?: string;
  /** Session label, e.g. "01" from ses-01. */
  session?: string;
  /**
   * BIDS data-type directory, e.g. "anat", "eeg", "nirs".
   * Takes priority over extension-based modality detection.
   */
  dataType?: BidsDataType;
  /** BIDS task label extracted from the filename. */
  task?: string;
  /** BIDS acquisition label. */
  acquisition?: string;
  /** BIDS run index. */
  run?: string;
}

/**
 * Valid BIDS data-type directory names.
 * Each maps to a canonical modality.
 */
export type BidsDataType =
  | 'anat' | 'func' | 'pet'          // → volumetric
  | 'eeg'  | 'meg'  | 'ieeg'         // → timeseries
  | 'nirs';                           // → nirs

// ── Per-file descriptor ───────────────────────────────────────────────────────

/**
 * Enriched file descriptor produced by BidsRouter.route().
 * This is what the hook layer and workers consume.
 */
export interface BidsFile {
  /** Original File object from the browser's FileList or DataTransfer. */
  file: File;
  /** Detected modality — drives plugin selection. */
  modality: Modality;
  /** Parsed BIDS path components (null if the path is not BIDS-conformant). */
  bidsPath: BidsPath | null;
  /** Lower-cased file extension including the leading dot, e.g. ".nii.gz". */
  extension: string;
  /**
   * True when the modality was derived from the BIDS directory path rather
   * than the file extension. Path-based detection is more reliable for
   * ambiguous extensions like ".tsv" (which can appear in multiple modalities).
   */
  pathBased: boolean;
}

// ── Dataset descriptor ────────────────────────────────────────────────────────

/**
 * Represents a collection of BidsFile objects from a single upload session.
 * Produced when the user drops an entire BIDS dataset directory.
 */
export interface BidsDataset {
  /** All files detected in the upload. */
  files: BidsFile[];
  /** Primary modality (most frequently occurring). */
  primaryModality: Modality;
  /** Unique subject labels found in the dataset. */
  subjects: string[];
  /** Unique session labels found. */
  sessions: string[];
}

// ── File extension → modality mapping (used by BidsRouter) ───────────────────

/**
 * Canonical mapping from lower-cased extensions to modalities.
 * Exported so other modules can inspect or extend it.
 */
export const EXTENSION_MODALITY_MAP: Readonly<Record<string, Modality>> = {
  '.nii':    'volumetric',
  '.nii.gz': 'volumetric',
  '.edf':    'timeseries',
  '.bdf':    'timeseries',
  '.vhdr':   'eeg',   // BrainVision header — entry point for the EEG trio
  '.eeg':    'eeg',   // BrainVision binary data (companion to .vhdr)
  '.vmrk':   'eeg',   // BrainVision marker file (companion to .vhdr)
  '.fif':    'meg',
  '.meg4':   'meg',
  '.con':    'meg',   // KIT/Yokogawa raw data
  '.mrk':    'meg',   // KIT HPI marker positions (companion to .con)
  '.pos':    'meg',   // KIT sensor positions    (companion to .con)
  '.snirf':  'nirs',
};

/**
 * Canonical mapping from BIDS data-type directory names to modalities.
 */
export const BIDS_DATATYPE_MODALITY_MAP: Readonly<Record<BidsDataType, Modality>> = {
  anat:  'volumetric',
  func:  'volumetric',
  pet:   'volumetric',
  eeg:   'eeg',
  meg:   'meg',
  ieeg:  'timeseries',
  nirs:  'nirs',
};
