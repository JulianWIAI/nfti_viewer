/**
 * snirfParser.ts — SNIRF (HDF5) file parser using h5wasm
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * SNIRF (Shared Near-Infrared Spectroscopy Format) is an HDF5-based file
 * format for fNIRS data.  This parser uses h5wasm (the HDF5 C library
 * compiled to WebAssembly) to read the HDF5 structure in the browser without
 * native bindings.
 *
 * WHY h5wasm instead of jsfive?
 *   jsfive does not implement all HDF5 filter codecs.  In particular, filter
 *   id 6 (SCALEOFFSET) used by many SNIRF datasets (e.g. ds008192 from
 *   OpenNeuro) causes a "Filter with id:6 not supported" error.  h5wasm is
 *   the genuine HDF5 C library and handles all standard filters.
 *
 * SNIRF v1.0 HDF5 STRUCTURE (relevant paths only):
 *
 *   /nirs{i}/                          — one or more acquisition blocks
 *     data{j}/
 *       dataTimeSeries                 — float64 [numTimePoints × numChannels]
 *       time                           — float64 [numTimePoints]
 *       measurementList{k}/
 *         sourceIndex                  — int
 *         detectorIndex                — int
 *         wavelengthIndex              — int
 *         dataType                     — int (1=raw, 99999=processed)
 *     probe/
 *       wavelengths                    — float64 [numWavelengths]
 *       sourcePos3D                    — float64 [numSources × 3]
 *       detectorPos3D                  — float64 [numDetectors × 3]
 *       sourceLabels                   — string[] (optional)
 *       detectorLabels                 — string[] (optional)
 *
 * DEPENDENCY: h5wasm (npm install h5wasm)
 */

import type { Dataset } from 'h5wasm';
import type { SnirfPayload, SnirfMeasurementListEntry, OptodePosition } from '../../../types/timeseries.types';

// h5wasm Dataset.value type alias
type H5Value = Dataset['value'];

// Minimal FS interface (Emscripten in-memory filesystem)
interface EmFS {
  writeFile(path: string, data: Uint8Array): void;
  unlink(path: string): void;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Gets a Dataset from an h5wasm Group/File.
 * Returns null if the path is missing, is a Group, or throws.
 * h5wasm's get() returns null for missing paths, but we wrap in try-catch
 * for any unexpected HDF5 errors on malformed files.
 */
function safeGet(group: { get(path: string): unknown }, path: string): Dataset | null {
  try {
    const result = group.get(path);
    if (result == null) return null;
    // Dataset has a 'value' property; Group has 'keys()' but not 'value'
    if (typeof result === 'object' && 'value' in result) return result as Dataset;
    return null;
  } catch {
    return null;
  }
}

/** Coerces an h5wasm dataset value to a Float32Array. */
function toFloat32(val: H5Value): Float32Array {
  if (val instanceof Float32Array) return val;
  if (val instanceof Float64Array) {
    const out = new Float32Array(val.length);
    for (let i = 0; i < val.length; i++) out[i] = val[i]!;
    return out;
  }
  if (val instanceof BigInt64Array || val instanceof BigUint64Array) {
    const out = new Float32Array(val.length);
    for (let i = 0; i < val.length; i++) out[i] = Number(val[i]);
    return out;
  }
  if (ArrayBuffer.isView(val)) return Float32Array.from(val as unknown as ArrayLike<number>);
  if (Array.isArray(val))      return Float32Array.from(val as number[]);
  if (typeof val === 'number') return new Float32Array([val]);
  if (typeof val === 'bigint') return new Float32Array([Number(val)]);
  return new Float32Array(0);
}

/** Coerces an h5wasm dataset value to a number. */
function toNumber(val: H5Value): number {
  if (typeof val === 'number') return val;
  if (typeof val === 'bigint') return Number(val);
  if (val instanceof Float64Array && val.length > 0) return val[0]!;
  if (val instanceof BigInt64Array && val.length > 0) return Number(val[0]);
  if (ArrayBuffer.isView(val) && (val as unknown as ArrayLike<number>).length > 0) {
    return Number((val as unknown as ArrayLike<number>)[0]);
  }
  if (Array.isArray(val) && val.length > 0) return Number(val[0]);
  return 0;
}

/** Coerces an h5wasm dataset value to a string array. */
function toStringArray(val: H5Value): string[] {
  if (typeof val === 'string') return [val];
  if (Array.isArray(val))     return val.map(String);
  return [];
}

// ── Main parser ───────────────────────────────────────────────────────────────

/**
 * Parses a SNIRF file from its raw ArrayBuffer using h5wasm.
 *
 * @param buffer   The full SNIRF file as an ArrayBuffer.
 * @param filename The original filename (used in error messages only).
 * @returns        A fully populated SnirfPayload.
 * @throws         Error if the file cannot be parsed or required fields are missing.
 */
export async function parseSnirfFile(buffer: ArrayBuffer, filename: string): Promise<SnirfPayload> {
  // Dynamic import keeps h5wasm (large WASM bundle) out of the initial chunk
  const h5mod = await import('h5wasm');
  await h5mod.ready;

  const FS = h5mod.FS as EmFS | null;
  if (!FS) throw new Error('SNIRF: h5wasm FS not available after ready');

  // Write the ArrayBuffer into Emscripten's in-memory virtual filesystem
  const fname = `_snirf_${Date.now()}_${Math.random().toString(36).slice(2)}.snirf`;
  FS.writeFile(fname, new Uint8Array(buffer));

  const f = new h5mod.File(fname, 'r');
  try {
    // ── Determine nirs/data root paths ──────────────────────────────────────
    // The SNIRF spec uses /nirs1, /nirs2, … but older implementations use /nirs.
    const rootKeys = f.keys();
    const nirsRoot =
      rootKeys.includes('nirs1') ? 'nirs1' :
      rootKeys.includes('nirs')  ? 'nirs'  :
      rootKeys.find(k => /^nirs/i.test(k)) ?? null;

    if (!nirsRoot) {
      throw new Error(
        `SNIRF: no /nirs root group found in "${filename}". ` +
        `Top-level keys: [${rootKeys.join(', ') || '(empty)'}]`,
      );
    }

    const dataRoot  = `${nirsRoot}/data1`;
    const probeRoot = `${nirsRoot}/probe`;

    // ── Required: dataTimeSeries ─────────────────────────────────────────────
    const dtsDs = safeGet(f, `${dataRoot}/dataTimeSeries`);
    if (!dtsDs) throw new Error('SNIRF: dataTimeSeries not found');
    const dtsRaw = toFloat32(dtsDs.value);

    // ── Required: time ───────────────────────────────────────────────────────
    const timeDs = safeGet(f, `${dataRoot}/time`);
    if (!timeDs) throw new Error('SNIRF: time not found');
    const time = toFloat32(timeDs.value);

    const numTimePoints = time.length;
    const numChannels   = numTimePoints > 0 ? dtsRaw.length / numTimePoints : 0;

    if (!Number.isInteger(numChannels) || numChannels === 0) {
      throw new Error(
        `SNIRF: dataTimeSeries length (${dtsRaw.length}) is not divisible by numTimePoints (${numTimePoints})`,
      );
    }

    // ── Measurement list ─────────────────────────────────────────────────────
    const measurementList: SnirfMeasurementListEntry[] = [];
    for (let k = 1; k <= numChannels; k++) {
      const mlRoot = `${dataRoot}/measurementList${k}`;
      const srcDs  = safeGet(f, `${mlRoot}/sourceIndex`);
      const detDs  = safeGet(f, `${mlRoot}/detectorIndex`);
      const wlDs   = safeGet(f, `${mlRoot}/wavelengthIndex`);
      const dtDs   = safeGet(f, `${mlRoot}/dataType`);

      measurementList.push({
        sourceIndex:     srcDs ? toNumber(srcDs.value) : k,
        detectorIndex:   detDs ? toNumber(detDs.value) : k,
        wavelengthIndex: wlDs  ? toNumber(wlDs.value)  : 1,
        dataType:        dtDs  ? toNumber(dtDs.value)  : 1,
      });
    }

    // ── Probe ────────────────────────────────────────────────────────────────
    const wlDs   = safeGet(f, `${probeRoot}/wavelengths`);
    const srcDs  = safeGet(f, `${probeRoot}/sourcePos3D`);
    const detDs  = safeGet(f, `${probeRoot}/detectorPos3D`);
    const srcLbl = safeGet(f, `${probeRoot}/sourceLabels`);
    const detLbl = safeGet(f, `${probeRoot}/detectorLabels`);

    const wavelengths       = wlDs  ? toFloat32(wlDs.value)  : new Float32Array(0);
    const srcPosFlat        = srcDs ? toFloat32(srcDs.value) : new Float32Array(0);
    const detPosFlat        = detDs ? toFloat32(detDs.value) : new Float32Array(0);
    const sourcePositions   = flatToPositions(srcPosFlat);
    const detectorPositions = flatToPositions(detPosFlat);
    const sourceLabels      = srcLbl ? toStringArray(srcLbl.value) : [];
    const detectorLabels    = detLbl ? toStringArray(detLbl.value) : [];

    return {
      dataTimeSeries: dtsRaw,
      numTimePoints,
      numChannels,
      time,
      measurementList,
      wavelengths,
      sourcePositions,
      detectorPositions,
      sourceLabels,
      detectorLabels,
    };
  } finally {
    try { f.close(); }   catch { /* ignore */ }
    try { FS.unlink(fname); } catch { /* ignore */ }
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

/** Converts a flat [x0,y0,z0, x1,y1,z1, …] array into OptodePosition[]. */
function flatToPositions(flat: Float32Array): OptodePosition[] {
  const positions: OptodePosition[] = [];
  for (let i = 0; i + 2 < flat.length; i += 3) {
    positions.push({ x: flat[i]!, y: flat[i + 1]!, z: flat[i + 2]! });
  }
  return positions;
}
