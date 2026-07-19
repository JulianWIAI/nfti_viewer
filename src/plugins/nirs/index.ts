/**
 * plugins/nirs/index.ts — NIRS plugin manifest
 * ──────────────────────────────────────────────
 *
 * Wraps SNIRF (fNIRS) file support inside the NeuroimagingPlugin contract.
 * Supported format: .snirf (HDF5-based Shared Near-Infrared Format)
 *
 * processFile() spawns snirf.worker.ts, passes the ArrayBuffer (zero-copy),
 * and resolves with PluginData<'nirs'> when parsing is complete.
 *
 * DEPENDENCY: jsfive must be installed:
 *   npm install jsfive
 */

import type { NeuroimagingPlugin, PluginData } from '../../types/plugin.types';
import type { BidsFile } from '../../types/bids.types';
import type { SnirfPayload } from '../../types/timeseries.types';
import NirsViewer from './NirsViewer';
import NirsControls from './NirsControls';

// ── Worker message types (mirror snirf.worker.ts) ─────────────────────────────

interface SnirfWorkerSuccess {
  type:    'SUCCESS';
  payload: SnirfPayload;
}

interface SnirfWorkerError {
  type:  'ERROR';
  error: string;
}

type SnirfWorkerOutput = SnirfWorkerSuccess | SnirfWorkerError;

// ── processFile ───────────────────────────────────────────────────────────────

async function processSnirfFile(bidsFile: BidsFile): Promise<PluginData> {
  const buffer = await bidsFile.file.arrayBuffer();

  return new Promise<PluginData>((resolve, reject) => {
    const worker = new Worker(
      new URL('../../workers/snirf.worker.ts', import.meta.url),
      { type: 'module' },
    );

    worker.onmessage = (event: MessageEvent<SnirfWorkerOutput>) => {
      worker.terminate();
      const msg = event.data;
      if (msg.type === 'SUCCESS') {
        resolve({ kind: 'nirs', payload: msg.payload });
      } else {
        reject(new Error(msg.error));
      }
    };

    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error(`snirf.worker uncaught error: ${e.message}`));
    };

    worker.postMessage({ buffer, filename: bidsFile.file.name }, [buffer]);
  });
}

// ── Plugin manifest ───────────────────────────────────────────────────────────

const nirsPlugin: NeuroimagingPlugin = {
  id:                  'nirs',
  name:                'fNIRS Viewer',
  modality:            'nirs',
  supportedExtensions: ['.snirf'],
  ViewerComponent:     NirsViewer,
  ControlsComponent:   NirsControls,
  processFile:         processSnirfFile,
};

export default nirsPlugin;
