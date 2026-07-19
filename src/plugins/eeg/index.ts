/**
 * plugins/eeg/index.ts — BrainVision EEG plugin manifest
 * ────────────────────────────────────────────────────────
 *
 * Handles BrainVision EEG files (.vhdr + .eeg + .vmrk) via the Python
 * FastAPI backend.  All three files must be uploaded together; the backend
 * writes them to a temp directory and reads the .vhdr with MNE-Python.
 *
 * processFile():
 *   1. POST /api/load-eeg   — upload .vhdr + .eeg + .vmrk as FormData
 *   2. GET  /api/eeg/metadata — fetch channel list + recording parameters
 *   3. Resolve with { kind: 'eeg', payload: EegSessionPayload }
 */

import type { NeuroimagingPlugin, PluginData } from '../../types/plugin.types';
import type { BidsFile } from '../../types/bids.types';
import type { EegSessionPayload } from '../../types/eeg.types';
import { eegApi } from '../../services/eegApi';
import EegViewer from './EegViewer';
import EegControls from './EegControls';

// ── processFile ───────────────────────────────────────────────────────────────

async function processEegFile(bidsFile: BidsFile): Promise<PluginData> {
  // The .vhdr is the primary file; App.tsx attaches the full set as bidsFile.files
  // when the user drops multiple files containing a .vhdr.
  const bidsFileWithCompanions = bidsFile as typeof bidsFile & { files?: File[] };
  const filesToUpload: File[] = bidsFileWithCompanions.files ?? [bidsFile.file];

  if (!filesToUpload.some((f) => f.name.toLowerCase().endsWith('.vhdr'))) {
    throw new Error(
      'BrainVision EEG requires three files: drop the .vhdr, .eeg and .vmrk together.',
    );
  }
  if (!filesToUpload.some((f) => f.name.toLowerCase().endsWith('.eeg'))) {
    throw new Error(
      'Missing .eeg file. Drop the .vhdr, .eeg and .vmrk together.',
    );
  }

  const loadResult = await eegApi.loadFiles(filesToUpload);
  const metadata   = await eegApi.getMetadata(loadResult.session_id);

  const payload: EegSessionPayload = {
    sessionId:     metadata.session_id,
    filename:      metadata.filename,
    samplingRate:  metadata.sampling_rate,
    totalDuration: metadata.total_duration,
    nSamples:      metadata.n_samples,
    nChannels:     metadata.n_channels,
    channels:      metadata.channels,
  };

  return { kind: 'eeg', payload };
}

// ── Plugin manifest ───────────────────────────────────────────────────────────

const eegPlugin: NeuroimagingPlugin = {
  id:                  'eeg',
  name:                'EEG Viewer',
  modality:            'eeg',
  supportedExtensions: ['.vhdr', '.eeg', '.vmrk'],
  ViewerComponent:     EegViewer,
  ControlsComponent:   EegControls,
  processFile:         processEegFile,
};

export default eegPlugin;
