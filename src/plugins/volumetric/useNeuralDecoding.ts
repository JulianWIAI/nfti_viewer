/**
 * useNeuralDecoding.ts — React hook for the MVPA time-resolved decoding workflow
 * ─────────────────────────────────────────────────────────────────────────────────
 *
 * Encapsulates the three pieces of state that drive the neural decoding UI:
 *
 *   decodingData    — the full DecodeResult once a pipeline run completes
 *   decodingStatus  — discriminated-union phase for UI feedback (idle / running /
 *                     done / error)
 *   runNeuralDecoding — async callback that fires the POST /api/eeg/decode request
 *
 * By lifting this logic into a dedicated hook the VolumetricViewer component body
 * stays readable even as the number of context fields grows.  The hook is a pure
 * function of useState / useCallback with no side-effects beyond the fetch call,
 * so it is trivially unit-testable in isolation.
 *
 * UPLOAD PROGRESS
 * ─────────────────
 * runNeuralDecoding now accepts an optional second argument `options` that
 * carries onUploadProgress and onUploadComplete callbacks.  These are forwarded
 * to decodingApi.runMvpaDecoding which in turn forwards them to xhrPost, giving
 * the caller (VolumetricViewer) byte-level upload progress.
 *
 * ERROR RETHROWING
 * ─────────────────
 * After setting the error state, the hook NOW rethrows the error so callers that
 * wrap runNeuralDecoding in a try/catch (e.g. VolumetricViewer's wrappedRun) can
 * call failTask() on the global task registry with the error message.
 *
 * USAGE
 * ───────
 *   // Inside VolumetricViewer:
 *   const { decodingData, decodingStatus, runNeuralDecoding } = useNeuralDecoding();
 *
 *   // Then surface these three values through VolumetricContext so that
 *   // DecodingPanel (rendered inside VolumetricControls) can call
 *   // runNeuralDecoding(req) and react to decodingStatus changes.
 */

import { useState, useCallback } from 'react';
import { decodingApi } from '../../services/decodingApi';
import type { DecodingRequest, DecodingResult } from '../../services/decodingApi';

// ── Status discriminated union ────────────────────────────────────────────────

/**
 * Represents the current phase of the MVPA pipeline.
 *
 * Matches the shape of InferenceStatus used by the SynthSeg segmentation
 * workflow — a discriminated union on `phase` — so UI components can follow
 * the same switch / conditional pattern for both pipelines.
 *
 *   idle    — no run has been triggered yet (or the result was cleared)
 *   running — a fetch is in-flight; disable the Run button + show spinner
 *   done    — pipeline completed successfully; decodingData is non-null
 *   error   — something went wrong; message contains the backend detail
 */
export type NeuralDecodingStatus =
  | { phase: 'idle' }
  | { phase: 'running' }
  | { phase: 'done'; durationMs: number; peakScore: number; peakTimeS: number }
  | { phase: 'error'; message: string };

// ── Optional progress callbacks ────────────────────────────────────────────────

/**
 * Upload-progress options for a single runNeuralDecoding call.
 *
 * Forwarded to decodingApi.runMvpaDecoding → xhrPost so the caller can track
 * byte-level upload progress and the server processing phase independently.
 *
 * onUploadProgress — called with pct 0–100 as bytes are sent
 * onUploadComplete — called once all bytes have been transmitted
 */
export interface NeuralDecodingRunOptions {
  onUploadProgress?: (pct: number) => void;
  onUploadComplete?: () => void;
}

// ── Hook return shape ─────────────────────────────────────────────────────────

export interface NeuralDecodingState {
  /** Full decoding time-course; null until a pipeline run completes. */
  decodingData: DecodingResult | null;
  /** Current pipeline phase — drives UI feedback (spinner, error, summary). */
  decodingStatus: NeuralDecodingStatus;
  /**
   * Kick off a new MVPA run.
   *
   * Sets status to `running`, awaits the fetch, then transitions to `done`
   * (with summary statistics) or `error` (with the backend message).
   *
   * On error the hook sets its own error state AND rethrows the error so
   * callers that wrap this in try/catch (e.g. VolumetricViewer) can call
   * failTask() on the global task registry.
   *
   * @param req     Files and analysis parameters.
   * @param options Optional upload-progress callbacks.
   */
  runNeuralDecoding: (req: DecodingRequest, options?: NeuralDecodingRunOptions) => Promise<void>;
}

// ── Hook ─────────────────────────────────────────────────────────────────────

export function useNeuralDecoding(): NeuralDecodingState {
  const [decodingData,   setDecodingData]   = useState<DecodingResult | null>(null);
  const [decodingStatus, setDecodingStatus] = useState<NeuralDecodingStatus>({ phase: 'idle' });

  /**
   * runNeuralDecoding — assemble a FormData payload and call the API client.
   *
   * The dep array is empty because `decodingApi.runMvpaDecoding` is a module-
   * level constant and the setters from useState are referentially stable.
   * The `req` and `options` arguments are read at call time so they are always
   * current.
   *
   * ERROR BEHAVIOUR:
   * After catching an error, the hook sets the 'error' phase state so the
   * DecodingPanel can display the error message via decodingStatus.
   * It then RETHROWS the error so VolumetricViewer's wrappedRun can call
   * failTask() on the global task registry.
   */
  const runNeuralDecoding = useCallback(
    async (req: DecodingRequest, options?: NeuralDecodingRunOptions): Promise<void> => {
      // Transition to 'running' immediately to disable the Run button.
      // Note: this shows 'running' before upload starts; the caller can
      // use onUploadProgress to differentiate upload vs. server phases.
      setDecodingStatus({ phase: 'running' });

      try {
        // Delegate to the API client, forwarding any progress callbacks.
        const result = await decodingApi.runMvpaDecoding(req, {
          onUploadProgress: options?.onUploadProgress,
          onUploadComplete: options?.onUploadComplete,
        });

        // Pipeline succeeded — store the result and update status.
        setDecodingData(result);
        setDecodingStatus({
          phase:      'done',
          durationMs: result.durationMs,
          peakScore:  result.peakScore,
          peakTimeS:  result.peakTimeS,
        });
      } catch (err) {
        // Set error state so the DecodingPanel can display the message.
        setDecodingStatus({
          phase:   'error',
          message: err instanceof Error ? err.message : String(err),
        });

        // Rethrow so VolumetricViewer's try/catch can call failTask()
        // on the global task registry with the same error message.
        throw err;
      }
    },
    [], // No reactive deps — setters and API client are stable module constants.
  );

  return { decodingData, decodingStatus, runNeuralDecoding };
}
