/**
 * src/plugins/multimodal/index.ts — Plugin manifest for the multimodal fMRI+MEG viewer
 * ──────────────────────────────────────────────────────────────────────────────────────
 *
 * This module registers the multimodal plugin with the ModalityRegistry so that
 * the BidsRouter can dispatch 4-D NIfTI files (.nii / .nii.gz with dims[4] > 1)
 * to the MultimodalWorkspace instead of the single-brain volumetric viewer.
 *
 * Plugin responsibilities:
 *   • processFile()          — parses a 4-D NIfTI via the Web Worker and returns
 *                              { kind: 'fmri', payload: FmriPayload }
 *   • ViewerComponent        — MultimodalWorkspace (split fMRI + MEG panes)
 *   • ControlsComponent      — stub (each panel owns its own sidebar section)
 *
 * How the ModalityRegistry chooses this plugin:
 *   The registry iterates plugins in registration order. When the BidsRouter
 *   identifies a .nii or .nii.gz it first asks the volumetric plugin; that plugin
 *   declares modality: 'mri'.  The multimodal plugin declares modality: 'fmri'
 *   and the BidsRouter inspects the parsed header after loading to see whether
 *   dims[4] > 1; if so it re-dispatches to this plugin.
 *
 *   Alternatively, the user can select the "Multimodal" viewer explicitly from
 *   a dropdown in the UI before dragging the file.
 */

import type { NeuroimagingPlugin }    from '../../types/plugin.types';
import type { BidsFile }              from '../../types/bids.types';
import type { PluginData }            from '../../types/plugin.types';
import type { FmriPayload }           from '../../types/fmri.types';
import type { WorkerInputMessage, WorkerOutboundMessage } from '../../types/nifti.types';
import {
  MultimodalViewerComponent,
  MultimodalControlsComponent,
} from './MultimodalViewerAdapter';

// ── File processing ──────────────────────────────────────────────────────────

/**
 * Parse a 4-D NIfTI file via the shared NIfTI Web Worker and return
 * { kind: 'fmri', payload: FmriPayload }.
 *
 * Spawns a dedicated worker instance (same pattern as volumetric/index.ts).
 * We read dims[4] from the parsed header to compute nTimepoints and use
 * pixDims[4] as the TR.
 */
async function processFile(bidsFile: BidsFile): Promise<PluginData> {
  const buffer = await bidsFile.file.arrayBuffer();

  return new Promise<PluginData>((resolve, reject) => {
    const worker = new Worker(
      new URL('../../workers/nifti.worker.ts', import.meta.url),
      { type: 'module' },
    );

    worker.onmessage = (event: MessageEvent<WorkerOutboundMessage>) => {
      worker.terminate();
      const msg = event.data;

      if (msg.type === 'ERROR') {
        reject(new Error(msg.error));
        return;
      }

      const { header, volumeData, dataType } = msg;

      // Extract 4-D metadata from the NIfTI header
      const nTimepoints = header.dims[4] ?? 1;
      const tr          = header.pixDims[4] ?? 0;

      const fmriPayload: FmriPayload = {
        header,
        volumeData,
        dataType,
        file:       bidsFile.file,
        nTimepoints,
        tr,
      };

      resolve({ kind: 'fmri', payload: fmriPayload });
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

const multimodalPlugin: NeuroimagingPlugin = {
  id:         'multimodal',
  name:       'Multimodal fMRI + MEG',
  modality:   'fmri' as never, // extended modality — cast required until bids.types adds 'fmri'
  supportedExtensions: ['.nii', '.nii.gz'],

  ViewerComponent:    MultimodalViewerComponent,
  ControlsComponent:  MultimodalControlsComponent,

  processFile,
};

export default multimodalPlugin;
