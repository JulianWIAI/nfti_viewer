/**
 * nifti.types.ts — Shared TypeScript interfaces for the NIfTI data pipeline
 * ──────────────────────────────────────────────────────────────────────────
 *
 * Data flow:
 *   File (drag-drop) → FileUpload component
 *     → useNiftiWorker hook  (posts ArrayBuffer to worker)
 *       → nifti.worker.ts    (parses header + image data)
 *         → useNiftiWorker   (receives VolumePayload)
 *           → Viewer component → vtk.js | ONNX InferenceEngine
 *
 * Keeping all cross-boundary types here means the worker script and the
 * React components never import from each other, which keeps the worker
 * isolation boundary clean.
 */

// ── NIfTI header ────────────────────────────────────────────────────────────

/**
 * Subset of a parsed NIfTI-1/2 header that the rendering and inference
 * pipelines actually need. The raw nifti-reader-js objects have dozens of
 * rarely-used fields; we strip them down here.
 */
export interface NiftiHeader {
  /**
   * Image dimensions array.
   * dims[0] = number of active dimensions (usually 3 or 4).
   * dims[1] = x voxels, dims[2] = y voxels, dims[3] = z voxels.
   * dims[4] = number of time points (1 for a single 3-D scan).
   */
  dims: number[];

  /**
   * Voxel size in mm (and TR in seconds for dim 4).
   * pixDims[0] = qfac (±1, used internally by qform).
   * pixDims[1..3] = voxel width in x, y, z.
   */
  pixDims: number[];

  /**
   * NIfTI scalar data-type code.
   * Common values: 2 = uint8, 4 = int16, 8 = int32, 16 = float32.
   */
  datatypeCode: number;

  /** Intended display minimum; may be 0 if unset in the file. */
  calMin: number;

  /** Intended display maximum; may be 0 if unset in the file. */
  calMax: number;

  /**
   * 4×4 RAS affine matrix (row-major).
   * Maps voxel indices (i,j,k) to world-space mm (x,y,z).
   * Derived from sform if sform_code > 0, else from qform.
   */
  affine: number[][];
}

// ── Worker message types ─────────────────────────────────────────────────────

/**
 * Message the main thread sends TO the worker.
 * The buffer's ownership is transferred (Transferable) — zero-copy.
 */
export interface WorkerInputMessage {
  /** Raw bytes of the uploaded .nii or .nii.gz file. */
  buffer: ArrayBuffer;
  /** Original filename, used as a hint to detect .gz before inspecting bytes. */
  filename: string;
}

/** Sent FROM the worker when parsing succeeds. */
export interface WorkerSuccessMessage {
  type: 'SUCCESS';
  header: NiftiHeader;
  /**
   * Flat voxel array as a plain ArrayBuffer (ownership transferred back).
   * Wrap with the constructor named in `dataType` before use.
   */
  volumeData: ArrayBuffer;
  /**
   * Which TypedArray wraps the raw buffer.
   * The vtk.js DataArray constructor and ONNX tensor will both need this.
   */
  dataType: 'Uint8Array' | 'Int16Array' | 'Float32Array';
}

/** Sent FROM the worker when parsing fails. */
export interface WorkerErrorMessage {
  type: 'ERROR';
  error: string;
}

/** Discriminated union of all worker → main messages. */
export type WorkerOutboundMessage = WorkerSuccessMessage | WorkerErrorMessage;

// ── Parsed volume ready for rendering ───────────────────────────────────────

/**
 * The fully parsed, ready-to-render volume handed off from the worker hook
 * to the Viewer component and eventually to the segmentation API.
 */
export interface VolumePayload {
  header: NiftiHeader;
  volumeData: ArrayBuffer;
  dataType: 'Uint8Array' | 'Int16Array' | 'Float32Array';
  /**
   * Original File object kept for backend segmentation upload (zero-copy reference).
   * Present when loaded via the plugin pipeline; absent in the legacy hook path.
   */
  file?: File;
}

// ── Viewer control state ─────────────────────────────────────────────────────

/**
 * State held in App.tsx and threaded down to both ControlPanel (for sliders)
 * and Viewer (to drive vtk.js mapper updates via useEffect).
 */
export interface ViewerControls {
  /** Current K slice index (axial plane). */
  sliceK: number;
  /** Current J slice index (coronal plane). */
  sliceJ: number;
  /** Current I slice index (sagittal plane). */
  sliceI: number;
  /** Window width for brightness/contrast (Hounsfield Units for CT). */
  windowWidth: number;
  /** Window centre/level for brightness/contrast. */
  windowCenter: number;
  /** Volume rendering opacity factor 0–1. */
  volumeOpacity: number;
  /** Whether the 3D volume renderer is visible alongside MPR slices. */
  showVolume: boolean;
}

/** Maximum slice indices derived from the volume header. */
export interface SliceMaxima {
  maxI: number;
  maxJ: number;
  maxK: number;
}
