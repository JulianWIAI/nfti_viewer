/**
 * inferenceEngine.ts — In-browser ONNX brain-tissue segmentation pipeline
 * ────────────────────────────────────────────────────────────────────────
 *
 * WHY in-browser ONNX?
 * Sending raw MRI data to a server raises HIPAA / GDPR concerns. Running the
 * model locally in WebAssembly keeps patient data entirely on the client while
 * still leveraging a real trained neural network.
 *
 * HOW it integrates with the rendering pipeline
 * ─────────────────────────────────────────────
 * The ONNX inference pipeline intercepts the volume data at TWO possible points:
 *
 *   A) Immediately after the NIfTI worker posts SUCCESS — the raw ArrayBuffer
 *      is still in Float32 / Int16 / Uint8 form before vtk.js has touched it.
 *      This is the RECOMMENDED point: the data is clean and no vtk.js transform
 *      has been applied. The Viewer component calls `inferenceEngine.segment()`
 *      right after receiving the VolumePayload from `useNiftiWorker`.
 *
 *   B) After the first vtk.js render — useful if you need to read back a
 *      pre-processed texture. Generally not recommended because it introduces
 *      a GPU→CPU round-trip.
 *
 * Once inference is complete, the segmentation mask is a flat Float32Array
 * with the same (x × y × z) shape as the original volume. The Viewer passes
 * this mask to a future `buildSegmentationOverlay()` function in mprRenderer.ts
 * to create a colour-coded transparent overlay on the MPR slices.
 *
 * SETUP CHECKLIST (for when you have a real .onnx model)
 * ───────────────────────────────────────────────────────
 *  1. Place your model at `public/models/brain_seg.onnx`  (served as a static
 *     asset). ONNX Runtime Web will fetch it over HTTP at session create time.
 *  2. Input name and shape must match what your Python export used:
 *       - Name: 'input' (or whatever `session.inputNames[0]` returns)
 *       - Shape: [1, 1, Z, Y, X]  (batch, channel, depth, height, width)
 *       - Dtype: float32
 *  3. If the model needs > 1 GB WASM heap, enable multi-threading:
 *       ort.env.wasm.numThreads = navigator.hardwareConcurrency
 *     (requires COOP/COEP headers — already configured in vite.config.ts).
 */

import * as ort from 'onnxruntime-web';
import type { VolumePayload } from '../../types/nifti.types';

// ── Status type ───────────────────────────────────────────────────────────────

export type InferenceStatus =
  | { phase: 'idle' }
  | { phase: 'loading_model' }
  | { phase: 'preprocessing' }
  | { phase: 'running' }
  | { phase: 'done'; durationMs: number }
  | { phase: 'error'; message: string };

// ── WASM path configuration ───────────────────────────────────────────────────

/**
 * Tells ONNX Runtime where to find its WASM blobs.
 * vite.config.ts copies them from node_modules → public/wasm/ at build time,
 * so they are served as static assets at BASE_URL + 'wasm/'.
 *
 * This must be called ONCE before any InferenceSession is created.
 */
function configureWasmPaths(): void {
  // import.meta.env.BASE_URL is '/nfti_viewer/' in production, '/' in dev.
  ort.env.wasm.wasmPaths = `${import.meta.env.BASE_URL}wasm/`;

  // Disable WASM multi-threading by default for maximum compatibility.
  // Enable with: ort.env.wasm.numThreads = navigator.hardwareConcurrency;
  ort.env.wasm.numThreads = 1;
}

// ── Engine class ──────────────────────────────────────────────────────────────

/**
 * Manages the lifecycle of one ONNX InferenceSession and exposes a
 * `segment()` method that the Viewer component calls after file load.
 *
 * Usage:
 *   const engine = new InferenceEngine();
 *   await engine.loadModel(`${import.meta.env.BASE_URL}models/brain_seg.onnx`);
 *   const mask = await engine.segment(volumePayload, onStatusChange);
 */
export class InferenceEngine {
  private session: ort.InferenceSession | null = null;
  private modelPath: string | null = null;

  // ── Model loading ─────────────────────────────────────────────────────────

  /**
   * Fetches and compiles the ONNX model.
   * Subsequent calls with the same path are a no-op (session is cached).
   *
   * @param modelPath - URL path to the .onnx file (relative to origin).
   * @param onStatus  - Optional callback for UI progress reporting.
   */
  async loadModel(
    modelPath: string,
    onStatus?: (s: InferenceStatus) => void,
  ): Promise<void> {
    if (this.session && this.modelPath === modelPath) return; // already loaded

    onStatus?.({ phase: 'loading_model' });
    configureWasmPaths();

    this.session = await ort.InferenceSession.create(modelPath, {
      // 'wasm' runs on the CPU via WebAssembly — always available.
      // Add 'webgl' or 'webgpu' before 'wasm' for GPU acceleration when
      // available:  executionProviders: ['webgpu', 'webgl', 'wasm']
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    this.modelPath = modelPath;
  }

  // ── Inference ─────────────────────────────────────────────────────────────

  /**
   * Runs brain tissue segmentation on the loaded volume.
   *
   * INTERCEPT POINT — this is where volumeData (from the NIfTI worker) enters
   * the AI pipeline. The steps:
   *   1. Convert the raw typed-array to Float32 (models expect float32 input).
   *   2. Normalise intensity to [0, 1] using the header cal_min/cal_max hints
   *      (or the actual min/max if those are unset).
   *   3. Create an ort.Tensor with shape [1, 1, Z, Y, X].
   *   4. Run session.run() — this is the blocking/async CPU-WASM computation.
   *   5. Extract the output tensor and return as Float32Array.
   *
   * @param payload  - VolumePayload from useNiftiWorker (header + raw buffer).
   * @param onStatus - Optional callback for UI progress updates.
   * @returns        - Flat Float32Array of class probabilities or labels,
   *                   same length as (X × Y × Z). Shape matches the input volume.
   */
  async segment(
    payload: VolumePayload,
    onStatus?: (s: InferenceStatus) => void,
  ): Promise<Float32Array> {
    if (!this.session) {
      throw new Error(
        'Model not loaded. Call loadModel() before segment().',
      );
    }

    // ── Step 1: Cast to Float32 ──────────────────────────────────────────
    onStatus?.({ phase: 'preprocessing' });

    const { header, volumeData, dataType } = payload;
    const [, x, y, z] = header.dims;
    const totalVoxels = x * y * z;

    let rawArray: Int16Array | Uint8Array | Float32Array;
    switch (dataType) {
      case 'Uint8Array':  rawArray = new Uint8Array(volumeData);  break;
      case 'Int16Array':  rawArray = new Int16Array(volumeData);  break;
      default:            rawArray = new Float32Array(volumeData); break;
    }

    // ── Step 2: Normalise to [0, 1] ──────────────────────────────────────
    // Use cal_min / cal_max if the file set them; otherwise scan for actual extremes.
    let lo = header.calMin;
    let hi = header.calMax;
    if (lo === 0 && hi === 0) {
      // cal_min/max not set — compute from data (O(n) scan, acceptable here)
      lo = Infinity;
      hi = -Infinity;
      for (let i = 0; i < totalVoxels; i++) {
        const v = rawArray[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    const range = hi - lo || 1; // guard division-by-zero on flat volumes

    const float32Input = new Float32Array(totalVoxels);
    for (let i = 0; i < totalVoxels; i++) {
      float32Input[i] = (rawArray[i] - lo) / range;
    }

    // ── Step 3: Build input tensor ────────────────────────────────────────
    // Shape: [batch=1, channels=1, depth=Z, height=Y, width=X]
    // This matches the standard 3-D UNet / nnU-Net convention.
    const inputTensor = new ort.Tensor('float32', float32Input, [1, 1, z, y, x]);

    // ── Step 4: Run inference ─────────────────────────────────────────────
    onStatus?.({ phase: 'running' });
    const t0 = performance.now();

    // Use the model's first input name dynamically so this code works with
    // any exported model regardless of what the exporter called the input.
    const inputName = this.session.inputNames[0];
    const outputName = this.session.outputNames[0];
    const feeds = { [inputName]: inputTensor };
    const results = await this.session.run(feeds);

    const durationMs = Math.round(performance.now() - t0);

    // ── Step 5: Extract output ────────────────────────────────────────────
    const outputTensor = results[outputName];
    if (!outputTensor) {
      throw new Error(`Model output "${outputName}" not found in inference results.`);
    }

    onStatus?.({ phase: 'done', durationMs });

    // Return the flat data — callers convert to vtkImageData for overlay
    return outputTensor.data as Float32Array;
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────

  /**
   * Releases the ONNX session and frees WASM heap memory.
   * Call on Viewer unmount to prevent memory leaks in long-running sessions.
   */
  dispose(): void {
    // ort.InferenceSession does not expose a synchronous destroy() in all
    // backends; setting to null lets the GC reclaim the JS wrapper.
    this.session = null;
    this.modelPath = null;
  }
}

// ── Singleton ─────────────────────────────────────────────────────────────────

/**
 * Application-wide singleton. Importing this constant from Viewer.tsx means
 * the WASM runtime is initialised only once even if the component re-mounts.
 */
export const inferenceEngine = new InferenceEngine();
