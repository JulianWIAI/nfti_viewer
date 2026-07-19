/**
 * plugins/timeseries/index.ts — Timeseries plugin manifest
 * ──────────────────────────────────────────────────────────
 *
 * Wraps EDF / BDF EEG recording support inside the NeuroimagingPlugin contract
 * so the ModalityRegistry can host it alongside the volumetric plugin.
 *
 * Supported formats: .edf (European Data Format), .bdf (BioSemi 24-bit)
 *
 * processFile() spawns edf.worker.ts, posts the raw ArrayBuffer (zero-copy),
 * and resolves with a PluginData when the worker succeeds.
 */

import type { NeuroimagingPlugin } from '../../types/plugin.types';
import type { PluginData } from '../../types/plugin.types';
import type { BidsFile } from '../../types/bids.types';
import type { TimeseriesPayload } from '../../types/timeseries.types';
import TimeseriesViewer from './TimeseriesViewer';
import TimeseriesControls from './TimeseriesControls';

// ── Worker message types (mirror edf.worker.ts) ───────────────────────────────

interface EdfWorkerSuccess {
  type:    'SUCCESS';
  payload: TimeseriesPayload;
}

interface EdfWorkerError {
  type:  'ERROR';
  error: string;
}

type EdfWorkerOutput = EdfWorkerSuccess | EdfWorkerError;

// ── processFile ───────────────────────────────────────────────────────────────

async function processEdfFile(bidsFile: BidsFile): Promise<PluginData> {
  const buffer = await bidsFile.file.arrayBuffer();

  return new Promise<PluginData>((resolve, reject) => {
    const worker = new Worker(
      new URL('../../workers/edf.worker.ts', import.meta.url),
      { type: 'module' },
    );

    worker.onmessage = (event: MessageEvent<EdfWorkerOutput>) => {
      worker.terminate();
      const msg = event.data;
      if (msg.type === 'SUCCESS') {
        resolve({ kind: 'timeseries', payload: msg.payload });
      } else {
        reject(new Error(msg.error));
      }
    };

    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error(`edf.worker uncaught error: ${e.message}`));
    };

    worker.postMessage({ buffer, filename: bidsFile.file.name }, [buffer]);
  });
}

// ── Plugin manifest ───────────────────────────────────────────────────────────

const timeseriesPlugin: NeuroimagingPlugin = {
  id:                  'timeseries',
  name:                'EEG / MEG Viewer',
  modality:            'timeseries',
  supportedExtensions: ['.edf', '.bdf'],
  ViewerComponent:     TimeseriesViewer,
  ControlsComponent:   TimeseriesControls,
  processFile:         processEdfFile,
};

export default timeseriesPlugin;
