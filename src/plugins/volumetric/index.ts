/**
 * plugins/volumetric/index.ts — Volumetric plugin manifest
 * ──────────────────────────────────────────────────────────
 *
 * Wraps the existing vtk.js NIfTI viewer (unchanged from the original
 * implementation in lib/vtk/ and hooks/useNiftiWorker.ts) inside the
 * NeuroimagingPlugin contract so the ModalityRegistry can host it.
 *
 * Supported formats: .nii, .nii.gz (MRI structural, functional, PET)
 *
 * The processFile() method spawns the existing nifti.worker.ts and returns
 * a PluginData<'volumetric'> when the worker resolves.
 */

import type { NeuroimagingPlugin } from '../../types/plugin.types';
import type { BidsFile } from '../../types/bids.types';
import type { PluginData } from '../../types/plugin.types';
import VolumetricViewer from './VolumetricViewer';
import VolumetricControls from './VolumetricControls';
import type { WorkerInputMessage, WorkerOutboundMessage } from '../../types/nifti.types';

/**
 * Spawns a nifti.worker.ts instance, posts the file buffer, and resolves
 * when the worker returns a SUCCESS or ERROR message.
 * Ownership of the ArrayBuffer is transferred to the worker (zero-copy).
 */
async function processNiftiFile(bidsFile: BidsFile): Promise<PluginData> {
  const buffer = await bidsFile.file.arrayBuffer();

  return new Promise<PluginData>((resolve, reject) => {
    const worker = new Worker(
      new URL('../../workers/nifti.worker.ts', import.meta.url),
      { type: 'module' },
    );

    worker.onmessage = (event: MessageEvent<WorkerOutboundMessage>) => {
      worker.terminate();
      const msg = event.data;
      if (msg.type === 'SUCCESS') {
        resolve({
          kind: 'volumetric',
          payload: {
            header:     msg.header,
            volumeData: msg.volumeData,
            dataType:   msg.dataType,
            file:       bidsFile.file,
          },
        });
      } else {
        reject(new Error(msg.error));
      }
    };

    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error(`nifti.worker uncaught error: ${e.message}`));
    };

    const msg: WorkerInputMessage = { buffer, filename: bidsFile.file.name };
    worker.postMessage(msg, [buffer]);
  });
}

// ── Plugin manifest ───────────────────────────────────────────────────────────

const volumetricPlugin: NeuroimagingPlugin = {
  id:                  'volumetric',
  name:                'MRI / PET Viewer',
  modality:            'volumetric',
  supportedExtensions: ['.nii', '.nii.gz'],
  ViewerComponent:     VolumetricViewer,
  ControlsComponent:   VolumetricControls,
  processFile:         processNiftiFile,
};

export default volumetricPlugin;
