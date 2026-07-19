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
   * Safe to call while another run is in progress — the caller should guard
   * with `decodingStatus.phase === 'running'` before showing the button as
   * enabled.
   */
  runNeuralDecoding: (req: DecodingRequest) => Promise<void>;
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
   * The `req` argument is read at call time (not captured at creation time),
   * so the files and parameters are always current.
   */
  const runNeuralDecoding = useCallback(async (req: DecodingRequest): Promise<void> => {
    setDecodingStatus({ phase: 'running' });

    try {
      const result = await decodingApi.runMvpaDecoding(req);

      setDecodingData(result);
      setDecodingStatus({
        phase:      'done',
        durationMs: result.durationMs,
        peakScore:  result.peakScore,
        peakTimeS:  result.peakTimeS,
      });
    } catch (err) {
      setDecodingStatus({
        phase:   'error',
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }, []);

  return { decodingData, decodingStatus, runNeuralDecoding };
}
