/// <reference lib="webworker" />
/**
 * edf.worker.ts — EDF/BDF file parsing Web Worker
 * ──────────────────────────────────────────────────
 *
 * Receives:  { buffer: ArrayBuffer, filename: string }
 *            The ArrayBuffer ownership is transferred (zero-copy).
 *
 * Posts back (on success):
 *   { type: 'SUCCESS', payload: TimeseriesPayload }
 *   The Float32Arrays inside payload are transferred back to the main thread.
 *
 * Posts back (on error):
 *   { type: 'ERROR', error: string }
 *
 * NOTE: The worker detects BDF vs EDF from the file extension (.bdf → 3 bytes
 * per sample).  The EDF parser in edfParser.ts handles both.
 */

import type { TimeseriesPayload, ChannelMeta } from '../types/timeseries.types';
import { parseEdfHeader, readEdfData, buildTimeAxis } from '../plugins/timeseries/lib/edfParser';

// ── Message contract ──────────────────────────────────────────────────────────

interface WorkerInput {
  buffer:   ArrayBuffer;
  filename: string;
}

interface WorkerSuccess {
  type:    'SUCCESS';
  payload: TimeseriesPayload;
}

interface WorkerError {
  type:  'ERROR';
  error: string;
}

// ── Message handler ───────────────────────────────────────────────────────────

self.onmessage = (event: MessageEvent<WorkerInput>) => {
  const { buffer, filename } = event.data;

  try {
    const lowerName = filename.toLowerCase();
    const isBdf     = lowerName.endsWith('.bdf');
    const modality  = lowerName.includes('ieeg') ? 'ieeg'
      : lowerName.includes('meg')                ? 'meg'
      : 'eeg';

    // 1. Parse header
    const header = parseEdfHeader(buffer);

    // 2. Read data records → one Float32Array per channel (physical units)
    const channelData = readEdfData(buffer, header, isBdf ? 3 : 2);

    // 3. Build shared time axis from the highest-rate channel
    const time = buildTimeAxis(header);

    // 4. Build unified ChannelMeta[]
    const channels: ChannelMeta[] = header.signals.map((sig) => ({
      label:      sig.label || `Ch${header.signals.indexOf(sig)}`,
      unit:       sig.physicalDimension || (modality === 'eeg' ? 'µV' : ''),
      sampleRate: sig.sampleRate,
      visible:    true,
    }));

    const numSamples = time.length;

    const payload: TimeseriesPayload = {
      time,
      numSamples,
      channels,
      data:           channelData,
      sourceModality: modality,
      filename,
    };

    // 5. Collect all transferable ArrayBuffers for zero-copy postMessage
    const transferables: ArrayBuffer[] = [
      time.buffer as ArrayBuffer,
      ...channelData.map((ch) => ch.buffer as ArrayBuffer),
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
