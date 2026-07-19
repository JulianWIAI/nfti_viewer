/**
 * useNiftiWorker.ts — React hook that manages the NIfTI Web Worker lifecycle
 * ──────────────────────────────────────────────────────────────────────────
 *
 * This hook is the bridge between the React world (File objects, state,
 * callbacks) and the isolated worker thread (ArrayBuffers, postMessage).
 *
 * LIFECYCLE
 * ──────────
 *  • The Worker is instantiated once on mount (stored in a ref so it survives
 *    re-renders). It is terminated on unmount to free the worker thread.
 *  • Each call to `processFile()` posts a new job. If the user drops a second
 *    file before the first job completes, the old job is implicitly superseded
 *    (the worker handles one message at a time; subsequent messages queue).
 *    For a production app you might terminate-and-recreate the worker to cancel
 *    in-flight work — left as a TODO here.
 *
 * ZERO-COPY TRANSFERS
 * ────────────────────
 *  • `file.arrayBuffer()` reads the file into memory on the main thread.
 *  • `worker.postMessage(msg, [msg.buffer])` transfers (not copies) the buffer
 *    to the worker thread: the main thread loses access, and the worker thread
 *    gains it at zero allocation cost.
 *  • The worker similarly transfers `imageBuffer` back via postMessage.
 *    By the time the hook receives the SUCCESS message, no large copies were made.
 *
 * WORKER INSTANTIATION (Vite syntax)
 * ────────────────────────────────────
 *   new Worker(new URL('../workers/nifti.worker.ts', import.meta.url), { type: 'module' })
 *
 *   Vite detects this pattern and bundles the worker into a separate chunk.
 *   The `{ type: 'module' }` option activates ES-module mode in the worker,
 *   which is required because our worker uses ES import statements.
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import type {
  VolumePayload,
  WorkerOutboundMessage,
  WorkerInputMessage,
} from '../types/nifti.types';

// ── Hook return type ─────────────────────────────────────────────────────────

export interface UseNiftiWorkerReturn {
  /** True while the worker is actively processing a file. */
  loading: boolean;
  /** Set when the worker reports an error; null otherwise. */
  error: string | null;
  /** The successfully parsed volume; null until a file is processed. */
  volume: VolumePayload | null;
  /**
   * Call this with a File object (from an <input> or drag event) to kick off
   * the parse pipeline. Safe to call while another parse is in flight.
   */
  processFile: (file: File) => Promise<void>;
}

// ── Hook ─────────────────────────────────────────────────────────────────────

export function useNiftiWorker(): UseNiftiWorkerReturn {
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);
  const [volume, setVolume]   = useState<VolumePayload | null>(null);

  // Keep the worker in a ref so it's created once and not re-created on render.
  const workerRef = useRef<Worker | null>(null);

  // ── Worker creation & teardown ───────────────────────────────────────────
  useEffect(() => {
    // Vite resolves this URL at build time and emits a separate worker chunk.
    workerRef.current = new Worker(
      new URL('../workers/nifti.worker.ts', import.meta.url),
      { type: 'module' },
    );

    // Register the message handler once; the handler reads current state via
    // closures over the state setters (which are stable references in React).
    workerRef.current.onmessage = (event: MessageEvent<WorkerOutboundMessage>) => {
      const msg = event.data;

      if (msg.type === 'SUCCESS') {
        setVolume({
          header: msg.header,
          volumeData: msg.volumeData,
          dataType: msg.dataType,
        });
        setError(null);
      } else {
        // msg.type === 'ERROR'
        setError(msg.error);
        setVolume(null);
      }

      setLoading(false);
    };

    workerRef.current.onerror = (event: ErrorEvent) => {
      setError(`Worker uncaught error: ${event.message}`);
      setLoading(false);
    };

    // Terminate the worker when the hook's host component unmounts.
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []); // empty deps — run once on mount

  // ── File processing ──────────────────────────────────────────────────────
  const processFile = useCallback(async (file: File): Promise<void> => {
    if (!workerRef.current) {
      setError('Worker not initialised. This is a bug.');
      return;
    }

    // Quick MIME-type and extension guard before paying the read cost.
    const name = file.name.toLowerCase();
    if (!name.endsWith('.nii') && !name.endsWith('.nii.gz')) {
      setError(`Unsupported file type: "${file.name}". Please select a .nii or .nii.gz file.`);
      return;
    }

    setLoading(true);
    setError(null);
    setVolume(null);

    // Read the entire file into an ArrayBuffer on the main thread.
    // For large files (> 200 MB) this can take 1–3 s; the UI should show a
    // spinner during this time (loading=true handles that).
    let buffer: ArrayBuffer;
    try {
      buffer = await file.arrayBuffer();
    } catch (readErr) {
      setError(`Could not read file: ${readErr instanceof Error ? readErr.message : String(readErr)}`);
      setLoading(false);
      return;
    }

    // Build the worker input and transfer the buffer (zero-copy).
    const msg: WorkerInputMessage = { buffer, filename: file.name };
    workerRef.current.postMessage(msg, [buffer]);
    // After this line, `buffer` is a detached (zero-byteLength) ArrayBuffer
    // on the main thread. The worker owns it now.
  }, []); // no deps — processFile identity is stable

  return { loading, error, volume, processFile };
}
