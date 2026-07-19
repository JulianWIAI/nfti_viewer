/// <reference lib="webworker" />
/**
 * snirf.worker.ts — SNIRF (fNIRS HDF5) file parsing Web Worker
 * ──────────────────────────────────────────────────────────────
 *
 * Receives:  { buffer: ArrayBuffer, filename: string }
 * Posts:     { type: 'SUCCESS', payload: SnirfPayload }
 *         or { type: 'ERROR',   error:   string }
 *
 * The heavy jsfive HDF5 parsing happens off the main thread so the UI
 * remains responsive during file loading.
 *
 * NOTE: transferable arrays in SnirfPayload are transferred back to
 * the main thread to avoid copying large Float32Arrays.
 */

import type { SnirfPayload } from '../types/timeseries.types';
import { parseSnirfFile } from '../plugins/nirs/lib/snirfParser';

// ── Message contract ──────────────────────────────────────────────────────────

interface WorkerInput {
  buffer:   ArrayBuffer;
  filename: string;
}

interface WorkerSuccess {
  type:    'SUCCESS';
  payload: SnirfPayload;
}

interface WorkerError {
  type:  'ERROR';
  error: string;
}

// ── Message handler ───────────────────────────────────────────────────────────

self.onmessage = async (event: MessageEvent<WorkerInput>) => {
  const { buffer, filename } = event.data;

  try {
    const payload = await parseSnirfFile(buffer, filename);

    // Collect transferable buffers for zero-copy postMessage
    const transferables: ArrayBuffer[] = [
      payload.dataTimeSeries.buffer as ArrayBuffer,
      payload.time.buffer as ArrayBuffer,
      payload.wavelengths.buffer as ArrayBuffer,
    ];

    const response: WorkerSuccess = { type: 'SUCCESS', payload };
    self.postMessage(response, transferables);

  } catch (err) {
    const response: WorkerError = {
      type:  'ERROR',
      error: err instanceof Error ? err.message : String(err),
    };
    self.postMessage(response);
  }
};
