/**
 * nifti.worker.ts — Dedicated Web Worker for NIfTI file parsing
 * ──────────────────────────────────────────────────────────────
 *
 * WHY a worker?
 * Parsing a compressed 3-D brain scan can take several hundred milliseconds
 * and involves pako inflate (CPU-bound) and nifti-reader-js header walking.
 * Running this on the main thread would freeze React's event loop, stutter
 * the UI, and block vtk.js from rendering. Moving it here keeps the frame
 * rate smooth regardless of file size.
 *
 * Pipeline (all in this thread):
 *   1. Receive { buffer, filename } — ArrayBuffer transferred (zero-copy).
 *   2. Detect gzip via magic bytes; decompress with nifti-reader-js / pako.
 *   3. Validate the NIfTI magic bytes with nifti-reader-js.
 *   4. Parse the NIfTI-1/2 header.
 *   5. Extract the raw image data buffer.
 *   6. Build our typed NiftiHeader (affine from sform or qform).
 *   7. Transfer { header, volumeData } back — zero-copy.
 *
 * The `/// <reference lib="webworker" />` directive tells TypeScript to use
 * the WebWorker type definitions (DedicatedWorkerGlobalScope) instead of the
 * browser lib for this file.
 */

/// <reference lib="webworker" />

import * as nifti from 'nifti-reader-js';
import { inflate } from 'pako';
import type {
  WorkerInputMessage,
  WorkerSuccessMessage,
  WorkerErrorMessage,
  NiftiHeader,
} from '../types/nifti.types';

// ── Type helpers ─────────────────────────────────────────────────────────────

/**
 * nifti-reader-js doesn't always ship complete .d.ts files so we type the
 * parts we actually use to keep TypeScript happy without resorting to `any`.
 */
interface NiftiParsedHeader {
  dims: number[];
  pixDims: number[];
  datatypeCode: number;
  cal_min: number;
  cal_max: number;
  qform_code: number;
  sform_code: number;
  qfac: number;
  quatern_b: number;
  quatern_c: number;
  quatern_d: number;
  qoffset_x: number;
  qoffset_y: number;
  qoffset_z: number;
  srow_x: number[];
  srow_y: number[];
  srow_z: number[];
}

// ── Decompression ─────────────────────────────────────────────────────────────

/**
 * Returns true when the first two bytes match the gzip magic number (0x1F 0x8B).
 * More reliable than checking the file extension, which users sometimes omit.
 */
function isGzipped(buf: ArrayBuffer): boolean {
  if (buf.byteLength < 2) return false;
  const view = new Uint8Array(buf, 0, 2);
  return view[0] === 0x1f && view[1] === 0x8b;
}

/**
 * Decompresses a gzip buffer.
 * We prefer nifti-reader-js's own decompress() (which calls pako internally)
 * for simplicity. We call pako's inflate directly as a fallback in case the
 * nifti library's version has issues with an edge-case file.
 */
function decompress(buf: ArrayBuffer): ArrayBuffer {
  // Primary path: let nifti-reader-js handle it
  try {
    // nifti.decompress returns ArrayBuffer
    const result = nifti.decompress(buf) as unknown as ArrayBuffer;
    if (result && result.byteLength > 0) return result;
  } catch {
    // fall through to pako
  }

  // Fallback: pako inflate
  const u8 = new Uint8Array(buf);
  const inflated = inflate(u8);
  // `inflated.buffer` may be a larger shared ArrayBuffer; slice to own copy
  return inflated.buffer.slice(inflated.byteOffset, inflated.byteOffset + inflated.byteLength);
}

// ── Affine matrix construction ───────────────────────────────────────────────

/**
 * Constructs a 4×4 RAS affine matrix from the parsed NIfTI header.
 *
 * Priority:
 *   1. sform (sform_code > 0) — most general; stored as three explicit rows.
 *   2. qform (qform_code > 0) — stored as a quaternion; we reconstruct.
 *   3. Identity fallback.
 *
 * The affine maps voxel index (i, j, k, 1)ᵀ → world RAS mm (x, y, z, 1)ᵀ.
 */
function buildAffine(h: NiftiParsedHeader): number[][] {
  // ── sform path ───────────────────────────────────────────────────────────
  if (h.sform_code > 0 && h.srow_x && h.srow_y && h.srow_z) {
    return [
      [...h.srow_x], // [r11*dx, r12*dy, r13*dz, tx]
      [...h.srow_y], // [r21*dx, r22*dy, r23*dz, ty]
      [...h.srow_z], // [r31*dx, r32*dy, r33*dz, tz]
      [0, 0, 0, 1],
    ];
  }

  // ── qform path ───────────────────────────────────────────────────────────
  if (h.qform_code > 0) {
    // Recover the quaternion scalar component from the unit-quaternion constraint:
    //   a² + b² + c² + d² = 1
    const b = h.quatern_b ?? 0;
    const c = h.quatern_c ?? 0;
    const d = h.quatern_d ?? 0;
    const a = Math.sqrt(Math.max(0, 1 - b * b - c * c - d * d));

    const dx = h.pixDims[1];
    const dy = h.pixDims[2];
    // qfac is stored in pixDims[0]: ±1, encodes L-R handedness of the volume
    const dz = h.pixDims[3] * (h.pixDims[0] < 0 ? -1 : 1);

    const tx = h.qoffset_x ?? 0;
    const ty = h.qoffset_y ?? 0;
    const tz = h.qoffset_z ?? 0;

    return [
      [(a * a + b * b - c * c - d * d) * dx, 2 * (b * c - a * d) * dy, 2 * (b * d + a * c) * dz, tx],
      [2 * (b * c + a * d) * dx, (a * a + c * c - b * b - d * d) * dy, 2 * (c * d - a * b) * dz, ty],
      [2 * (b * d - a * c) * dx, 2 * (c * d + a * b) * dy, (a * a + d * d - b * b - c * c) * dz, tz],
      [0, 0, 0, 1],
    ];
  }

  // ── identity fallback (no spatial info in the file) ──────────────────────
  const dx = h.pixDims[1] || 1;
  const dy = h.pixDims[2] || 1;
  const dz = h.pixDims[3] || 1;
  return [
    [dx, 0, 0, 0],
    [0, dy, 0, 0],
    [0, 0, dz, 0],
    [0, 0, 0, 1],
  ];
}

// ── Data-type mapping ────────────────────────────────────────────────────────

/**
 * Maps a NIfTI datatype code to the closest JS TypedArray tag that vtk.js
 * and ONNX Runtime can both consume without manual conversion.
 *
 * NIfTI codes (DT_*):
 *   2 = UINT8, 4 = INT16, 8 = INT32, 16 = FLOAT32, 64 = FLOAT64,
 *   256 = INT8, 512 = UINT16, 768 = UINT32
 */
function resolveDataType(code: number): 'Uint8Array' | 'Int16Array' | 'Float32Array' {
  switch (code) {
    case 2:   // DT_UINT8
    case 256: // DT_INT8 (treat signed bytes as unsigned for display)
      return 'Uint8Array';

    case 4:  // DT_INT16
    case 8:  // DT_INT32 — downcast is acceptable for display; intensity range is preserved
      return 'Int16Array';

    default:
      // Float32 is the safest GPU-friendly default for float64, uint16, uint32
      return 'Float32Array';
  }
}

// ── Main message handler ─────────────────────────────────────────────────────

self.onmessage = (event: MessageEvent<WorkerInputMessage>): void => {
  const { buffer, filename } = event.data;

  try {
    // ── Step 0: Detect git-annex / DataLad pointer files ─────────────────
    // OpenNeuro and many BIDS datasets are distributed via DataLad. When the
    // actual content hasn't been fetched yet, the visible "file" is a small
    // text stub containing the annex object path (e.g. "../../.git/annex/…").
    // These stubs are far too small to be a real NIfTI and contain ASCII text.
    if (buffer.byteLength < 1000) {
      const peek = new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 256));
      const text = String.fromCharCode(...peek);
      if (text.includes('/annex/objects/') || text.startsWith('../') || text.startsWith('/annex/')) {
        throw new Error(
          `"${filename}" is a DataLad/git-annex pointer, not actual image data. ` +
          'Run "datalad get <filename>" (or "git annex get <filename>") to download the content first.',
        );
      }
    }

    // ── Step 1: Decompress if needed ───────────────────────────────────────
    let rawBuffer = isGzipped(buffer) ? decompress(buffer) : buffer;

    // ── Step 2 + 3: Parse header ──────────────────────────────────────────
    // nifti.readHeader() throws when magic bytes don't match.
    //   NIfTI-1: magic 'n+1' at offset 344 (sizeof_hdr=348)
    //   NIfTI-2: magic 'n+2' at offset 4   (sizeof_hdr=540)
    //
    // Fallback: if the primary parse fails, check sizeof_hdr at offset 0.
    //   If it equals 348 → clone + patch NIfTI-1 magic at 344.
    //   If it equals 540 → clone + patch NIfTI-2 magic at 4.
    // This recovers Analyze-7.5 / older FSL / SPM files with missing magic.
    let workBuf = rawBuffer; // may be replaced with a patched copy below
    let raw: NiftiParsedHeader | null = null;

    // First attempt — standard path
    try {
      raw = nifti.readHeader(workBuf) as unknown as NiftiParsedHeader;
    } catch { /* fall through to sizeof_hdr fallback */ }

    // Second attempt — patch magic bytes based on sizeof_hdr heuristic
    if (!raw) {
      if (workBuf.byteLength >= 352) {
        const peek = new DataView(workBuf);
        const hLE  = peek.getInt32(0, true);
        const hBE  = peek.getInt32(0, false);

        if (hLE === 348 || hBE === 348) {
          // Looks like NIfTI-1: patch magic 'n+1\0' at offset 344
          workBuf = workBuf.slice(0);
          const pv = new DataView(workBuf);
          pv.setUint8(344, 0x6e); pv.setUint8(345, 0x2b);
          pv.setUint8(346, 0x31); pv.setUint8(347, 0x00);
          try {
            raw = nifti.readHeader(workBuf) as unknown as NiftiParsedHeader;
          } catch { /* will be caught below */ }
        } else if (hLE === 540 || hBE === 540) {
          // Looks like NIfTI-2: patch magic 'n+2\0' at offset 4
          workBuf = workBuf.slice(0);
          const pv = new DataView(workBuf);
          pv.setUint8(4, 0x6e); pv.setUint8(5, 0x2b);
          pv.setUint8(6, 0x32); pv.setUint8(7, 0x00);
          try {
            raw = nifti.readHeader(workBuf) as unknown as NiftiParsedHeader;
          } catch { /* will be caught below */ }
        }
      }
    }

    if (!raw) {
      // Build a diagnostic showing what was found at the expected magic locations
      // (rawBuffer = decompressed, unpatched bytes; workBuf may be a patched clone)
      const dv = new DataView(rawBuffer);
      const hex = (offset: number) =>
        rawBuffer.byteLength > offset
          ? dv.getUint8(offset).toString(16).padStart(2, '0')
          : '--';
      const at4   = `${hex(4)} ${hex(5)} ${hex(6)}`;
      const at344 = `${hex(344)} ${hex(345)} ${hex(346)}`;
      throw new Error(
        `"${filename}" could not be parsed as a NIfTI file. ` +
        `(magic@4: ${at4}, magic@344: ${at344}) ` +
        'Make sure it is a valid .nii or .nii.gz and is not corrupt.',
      );
    }

    // ── Step 4: Build our typed header ────────────────────────────────────
    const header: NiftiHeader = {
      dims: Array.from(raw.dims),
      pixDims: Array.from(raw.pixDims),
      datatypeCode: raw.datatypeCode,
      calMin: raw.cal_min,
      calMax: raw.cal_max,
      affine: buildAffine(raw),
    };

    // ── Step 5: Extract image data ────────────────────────────────────────
    // nifti.readImage returns a plain ArrayBuffer of the voxel scalars,
    // in the native datatype specified by header.datatypeCode.
    // Use workBuf (which may be the patched copy) so readImage sees the
    // same bytes that readHeader used.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const imageBuffer = nifti.readImage(raw as any, workBuf) as ArrayBuffer;
    if (!imageBuffer || imageBuffer.byteLength === 0) {
      throw new Error('NIfTI image data is empty after parsing. The file may contain only a header.');
    }

    // ── Step 6: Post back with zero-copy transfer ─────────────────────────
    const success: WorkerSuccessMessage = {
      type: 'SUCCESS',
      header,
      volumeData: imageBuffer,
      dataType: resolveDataType(raw.datatypeCode),
    };
    // Listing imageBuffer in the transfer list moves ownership to the main
    // thread — no copying, instant for even multi-hundred-MB volumes.
    self.postMessage(success, [imageBuffer]);

  } catch (err) {
    const failure: WorkerErrorMessage = {
      type: 'ERROR',
      error: err instanceof Error ? err.message : String(err),
    };
    self.postMessage(failure);
  }
};
