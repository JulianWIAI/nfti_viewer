/**
 * plugins/meg/index.ts — MEG plugin manifest
 * ─────────────────────────────────────────────
 *
 * Handles .fif (MNE / Elekta) and .meg4 (CTF) recordings via the Python
 * FastAPI backend. Unlike EEG/NIRS plugins, MEG data is never transferred to
 * the browser — only session metadata is fetched here. The MegViewer streams
 * decimated chunks on demand via megApi.getChannelData().
 *
 * processFile():
 *   1. POST /api/load-meg  — uploads the file, backend returns session_id
 *   2. GET  /api/meg/metadata — fetches channel list + recording parameters
 *   3. Resolves with { kind: 'meg', payload: MegSessionPayload }
 */

import type { NeuroimagingPlugin, PluginData } from '../../types/plugin.types';
import type { BidsFile } from '../../types/bids.types';
import type { MegSessionPayload } from '../../types/meg.types';
import { megApi } from '../../services/megApi';
import MegViewer from './MegViewer';
import MegControls from './MegControls';

// ── processFile ───────────────────────────────────────────────────────────────

async function processMegFile(bidsFile: BidsFile): Promise<PluginData> {
  const ext = bidsFile.extension.toLowerCase();

  // Resolve the load result based on file format.
  let loadResult;

  if (ext === '.con') {
    // KIT/Yokogawa file set — App.tsx attaches all dropped files to bidsFile.files
    // when a .con is dropped together with companion files.
    const allFiles: File[] =
      (bidsFile as BidsFile & { files?: File[] }).files ?? [bidsFile.file];

    const conFile = allFiles.find((f) => f.name.toLowerCase().endsWith('.con')) ?? bidsFile.file;
    const mrkFile = allFiles.find((f) => f.name.toLowerCase().endsWith('.mrk'));
    const posFile = allFiles.find((f) => f.name.toLowerCase().endsWith('.pos'));

    loadResult = await megApi.loadKitFiles(conFile, mrkFile, posFile);

  } else if (ext === '.fif' || ext === '.meg4') {
    // Multi-file split recordings: App.tsx attaches all .fif parts to bidsFile.files
    const allFiles = (bidsFile as BidsFile & { files?: File[] }).files;
    const fifFiles = allFiles?.filter((f) => f.name.toLowerCase().endsWith('.fif'));
    if (fifFiles && fifFiles.length > 1) {
      loadResult = await megApi.loadFifFiles(fifFiles);
    } else {
      loadResult = await megApi.loadFile(bidsFile.file);
    }

  } else {
    // .mrk or .pos dropped without a .con — cannot open alone
    throw new Error(
      `${ext} files cannot be opened on their own. ` +
      'Drop the .con file together with the .mrk and .pos files.',
    );
  }

  // Fetch full channel inventory and recording parameters
  const metadata = await megApi.getMetadata(loadResult.session_id);

  const payload: MegSessionPayload = {
    sessionId:     metadata.session_id,
    filename:      metadata.filename,
    samplingRate:  metadata.sampling_rate,
    totalDuration: metadata.total_duration,
    nSamples:      metadata.n_samples,
    nChannels:     metadata.n_channels,
    channels:      metadata.channels,
  };

  return { kind: 'meg', payload };
}

// ── Plugin manifest ───────────────────────────────────────────────────────────

const megPlugin: NeuroimagingPlugin = {
  id:                  'meg',
  name:                'MEG Viewer',
  modality:            'meg',
  supportedExtensions: ['.fif', '.meg4', '.con'],
  ViewerComponent:     MegViewer,
  ControlsComponent:   MegControls,
  processFile:         processMegFile,
};

export default megPlugin;
