/**
 * BidsRouter.ts — Central file ingestion and modality dispatcher
 * ───────────────────────────────────────────────────────────────
 *
 * This is the single entry-point for all files that enter the application.
 * It sits between the FileUpload component and the plugin system:
 *
 *   FileUpload (drops files)
 *     → BidsRouter.route()           ← analyses path + extension
 *       → returns BidsFile[]
 *         → ModalityRegistry.getPlugin(modality)
 *           → plugin.processFile(bidsFile) → PluginData
 *             → Viewer host renders the correct plugin
 *
 * BIDS detection strategy (in order of priority):
 *   1. BIDS directory path component (most reliable for full datasets)
 *      e.g. "sub-01/ses-01/eeg/sub-01_task-rest_eeg.edf" → timeseries
 *   2. File extension (reliable for individual file uploads)
 *      e.g. ".snirf" → nirs
 *   3. Unknown (file passes through, plugin must handle gracefully)
 *
 * BIDS entity parsing:
 *   BIDS filenames follow the pattern:
 *     sub-<label>[_ses-<label>][_task-<label>][_acq-<label>][_run-<index>]_<suffix>.<ext>
 *   We extract the entity key-value pairs from the stem.
 */

import type {
  BidsFile,
  BidsDataset,
  BidsPath,
  BidsDataType,
  Modality,
} from '../types/bids.types';
import {
  EXTENSION_MODALITY_MAP,
  BIDS_DATATYPE_MODALITY_MAP,
} from '../types/bids.types';

// ── Path analysis helpers ─────────────────────────────────────────────────────

/**
 * Extracts the lower-cased file extension, handling compound extensions like
 * ".nii.gz" which `path.extname` / `String.prototype.split('.').at(-1)` miss.
 */
function getExtension(filename: string): string {
  const lower = filename.toLowerCase();
  // Check compound extensions first (longest match wins)
  if (lower.endsWith('.nii.gz')) return '.nii.gz';
  if (lower.endsWith('.tar.gz')) return '.tar.gz';
  const dotIdx = lower.lastIndexOf('.');
  return dotIdx >= 0 ? lower.slice(dotIdx) : '';
}

/**
 * Extracts modality from the BIDS data-type directory name in the file path.
 *
 * The browser's File.webkitRelativePath looks like:
 *   "dataset/sub-01/ses-01/eeg/sub-01_task-rest_eeg.edf"
 * We split on '/' and look for a known BIDS data-type segment.
 */
function getModalityFromPath(relativePath: string): Modality | null {
  const segments = relativePath.toLowerCase().split('/');
  for (const segment of segments) {
    const modality = BIDS_DATATYPE_MODALITY_MAP[segment as BidsDataType];
    if (modality) return modality;
  }
  return null;
}

/**
 * Parses BIDS entities from a file's relative path.
 *
 * Given "dataset/sub-01/ses-02/eeg/sub-01_ses-02_task-rest_run-1_eeg.edf",
 * returns { subject: '01', session: '02', task: 'rest', run: '1', dataType: 'eeg' }.
 */
function parseBidsPath(relativePath: string): BidsPath | null {
  const segments = relativePath.split('/');

  let subject: string | undefined;
  let session: string | undefined;
  let dataType: BidsDataType | undefined;
  let task: string | undefined;
  let acquisition: string | undefined;
  let run: string | undefined;

  // Walk directory segments for sub-, ses-, and data-type
  for (const seg of segments.slice(0, -1)) {
    // Remove leading/trailing whitespace for safety
    const s = seg.trim().toLowerCase();
    if (s.startsWith('sub-')) subject = seg.slice(4);
    else if (s.startsWith('ses-')) session = seg.slice(4);
    else if (BIDS_DATATYPE_MODALITY_MAP[s as BidsDataType]) {
      dataType = s as BidsDataType;
    }
  }

  // Parse BIDS entities from the filename (last segment, without extension)
  const filename = segments[segments.length - 1] ?? '';
  const stem = filename.split('.')[0] ?? '';
  const entityPairs = stem.split('_');
  for (const pair of entityPairs) {
    const dashIdx = pair.indexOf('-');
    if (dashIdx < 0) continue;
    const key = pair.slice(0, dashIdx).toLowerCase();
    const val = pair.slice(dashIdx + 1);
    switch (key) {
      case 'sub':  subject     = val; break;
      case 'ses':  session     = val; break;
      case 'task': task        = val; break;
      case 'acq':  acquisition = val; break;
      case 'run':  run         = val; break;
    }
  }

  // Return null if no BIDS entities were found (plain file, not a BIDS dataset)
  if (!subject && !dataType) return null;

  return { subject, session, dataType, task, acquisition, run };
}

// ── Public router ─────────────────────────────────────────────────────────────

/**
 * BidsRouter analyses an array of File objects and returns one BidsFile
 * descriptor per file with modality, BIDS path, and extension resolved.
 *
 * Designed to be stateless and synchronous — all heavy work (parsing, decompression)
 * happens inside Web Workers launched by the plugin's processFile() method.
 */
export class BidsRouter {
  /**
   * Analyse a single File object.
   *
   * @param file - Browser File (may have .webkitRelativePath set if from a
   *               directory picker or drag-and-drop of a folder).
   * @returns BidsFile with modality and path resolved.
   */
  static route(file: File): BidsFile {
    const relativePath = (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
    const extension    = getExtension(file.name);

    // ── Priority 1: BIDS directory path ──────────────────────────────────
    const pathModality = getModalityFromPath(relativePath);
    if (pathModality && pathModality !== 'unknown') {
      return {
        file,
        modality: pathModality,
        bidsPath: parseBidsPath(relativePath),
        extension,
        pathBased: true,
      };
    }

    // ── Priority 2: File extension ────────────────────────────────────────
    const extModality = EXTENSION_MODALITY_MAP[extension] ?? 'unknown';
    return {
      file,
      modality: extModality,
      bidsPath: parseBidsPath(relativePath),
      extension,
      pathBased: false,
    };
  }

  /**
   * Analyse a FileList or File array (e.g. from a directory upload) and group
   * results into a BidsDataset.
   *
   * Skips sidecar files (.json, .tsv, .txt) that are metadata, not data.
   *
   * @param files - FileList from input[type=file] or DataTransfer.files.
   */
  static routeAll(files: FileList | File[]): BidsDataset {
    const arr = Array.from(files);

    // Filter to only data files (skip JSON sidecars, README, etc.)
    const dataFiles = arr.filter((f) => !BidsRouter.isSidecar(f.name));

    const bidsFiles = dataFiles.map((f) => BidsRouter.route(f));

    // Determine primary modality by frequency
    const counts: Partial<Record<Modality, number>> = {};
    for (const bf of bidsFiles) {
      counts[bf.modality] = (counts[bf.modality] ?? 0) + 1;
    }
    const primaryModality = (Object.entries(counts).sort(([, a], [, b]) => b - a)[0]?.[0] ?? 'unknown') as Modality;

    // Collect unique subjects and sessions
    const subjects = [...new Set(bidsFiles.map((f) => f.bidsPath?.subject).filter(Boolean) as string[])];
    const sessions = [...new Set(bidsFiles.map((f) => f.bidsPath?.session).filter(Boolean) as string[])];

    return { files: bidsFiles, primaryModality, subjects, sessions };
  }

  /**
   * Returns true for BIDS sidecar files that carry metadata but no raw data.
   * These are silently ignored by routeAll().
   */
  static isSidecar(filename: string): boolean {
    const lower = filename.toLowerCase();
    return (
      lower.endsWith('.json')   ||
      lower.endsWith('.tsv')    ||
      lower.endsWith('.txt')    ||
      lower.endsWith('.md')     ||
      lower === 'readme'        ||
      lower === 'dataset_description.json' ||
      lower === 'participants.tsv'
    );
  }

  /**
   * Quick check — does this file's extension suggest we can handle it?
   * Used by FileUpload to give immediate feedback before routing.
   */
  static isSupportedExtension(filename: string): boolean {
    const ext = getExtension(filename);
    return ext in EXTENSION_MODALITY_MAP;
  }
}
