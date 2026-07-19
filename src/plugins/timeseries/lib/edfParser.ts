/**
 * edfParser.ts — EDF/BDF file format parser (pure utility, no DOM/Worker)
 * ────────────────────────────────────────────────────────────────────────
 *
 * European Data Format (EDF) specification (Kemp et al., 1992):
 *
 * FILE LAYOUT
 * ───────────
 *   [Global header — 256 bytes]
 *   [Per-signal header — ns × 256 bytes]  (all labels, then all transducers, …)
 *   [Data records — nr × (Σ samplesPerRecord[i] × 2 bytes)]
 *
 * PER-SIGNAL HEADER FIELD SIZES (bytes, repeated ns times, grouped by field):
 *   label           : 16  — e.g. "EEG Fp1     "
 *   transducer_type : 80
 *   physical_dim    :  8  — e.g. "uV      "
 *   physical_min    :  8  — ASCII decimal
 *   physical_max    :  8
 *   digital_min     :  8  — usually -32768
 *   digital_max     :  8  — usually 32767
 *   prefiltering    : 80
 *   samples_per_rec :  8  — key: sampleRate × recordDuration
 *   reserved        : 32
 *   ──────────────────────────────────────────────────────
 *   Total per signal: 256 bytes × ns signals
 *
 * DATA RECORDS
 * ────────────
 * Each data record contains samplesPerRecord[i] int16 values for each signal i,
 * stored channel-by-channel (not interleaved sample-by-sample).
 *
 * PHYSICAL VALUE CONVERSION
 * ─────────────────────────
 *   gain   = (physMax - physMin) / (digMax - digMin)
 *   offset = physMax / gain - digMax
 *   physValue = (digitalValue + offset) × gain
 *
 * BDF (24-bit biosemi) is handled by the same parser with 3-byte samples.
 */

import type { EdfHeader, EdfSignalHeader } from '../../../types/timeseries.types';

// ── ASCII helpers ─────────────────────────────────────────────────────────────

const ASCII = new TextDecoder('ascii');

/** Reads a fixed-width ASCII field from a Uint8Array and trims whitespace. */
function readField(view: Uint8Array, offset: number, length: number): string {
  return ASCII.decode(view.subarray(offset, offset + length)).trim();
}

// ── Header parser ─────────────────────────────────────────────────────────────

/**
 * Parses the global + per-signal EDF/BDF header from a raw ArrayBuffer.
 * The caller must pass the FULL file buffer; only the header bytes are read here.
 */
export function parseEdfHeader(buffer: ArrayBuffer): EdfHeader {
  const view = new Uint8Array(buffer);

  // ── Global header (256 bytes) ──────────────────────────────────────────
  const version      = readField(view,   0,   8);
  const patientId    = readField(view,   8,  80);
  const recordingId  = readField(view,  88,  80);
  const startDate    = readField(view, 168,   8); // dd.mm.yy
  const startTime    = readField(view, 176,   8); // hh.mm.ss
  // headerBytes      = readField(view, 184,   8)  — unused, can be derived
  // reserved         = readField(view, 192,  44)
  const numRecords   = parseInt(readField(view, 236, 8));
  const recordDur    = parseFloat(readField(view, 244, 8));
  const numChannels  = parseInt(readField(view, 252, 4));

  // ── Per-signal header (ns × 256 bytes, grouped by field) ──────────────
  // EDF stores ALL labels for ALL channels first, then all transducer types, …
  let off = 256; // start of per-signal header block

  const fieldDefs: Array<{ name: keyof EdfSignalHeader | '_reserved'; size: number }> = [
    { name: 'label',           size: 16 },
    { name: 'transducerType',  size: 80 },
    { name: 'physicalDimension', size:  8 },
    { name: 'physicalMin',     size:  8 },
    { name: 'physicalMax',     size:  8 },
    { name: 'digitalMin',      size:  8 },
    { name: 'digitalMax',      size:  8 },
    { name: 'prefiltering',    size: 80 },
    { name: 'samplesPerRecord', size: 8 },
    { name: '_reserved',       size: 32 },
  ];

  // Initialise per-channel record objects
  const raw: Record<string, string>[] = Array.from({ length: numChannels }, () => ({}));

  for (const { name, size } of fieldDefs) {
    for (let ch = 0; ch < numChannels; ch++) {
      const value = readField(view, off, size);
      (raw[ch] as Record<string, string>)[name as string] = value;
      off += size;
    }
  }

  // ── Build typed signal headers ──────────────────────────────────────────
  const signals: EdfSignalHeader[] = raw.map((r) => {
    const physMin  = parseFloat(r['physicalMin']  ?? '0');
    const physMax  = parseFloat(r['physicalMax']  ?? '1');
    const digMin   = parseFloat(r['digitalMin']   ?? '-32768');
    const digMax   = parseFloat(r['digitalMax']   ?? '32767');
    const spr      = parseInt(r['samplesPerRecord'] ?? '256');
    const sampleRate = recordDur > 0 ? spr / recordDur : 0;

    return {
      label:            r['label']           ?? '',
      transducerType:   r['transducerType']  ?? '',
      physicalDimension: r['physicalDimension'] ?? '',
      physicalMin: physMin,
      physicalMax: physMax,
      digitalMin:  digMin,
      digitalMax:  digMax,
      prefiltering: r['prefiltering'] ?? '',
      samplesPerRecord: spr,
      sampleRate,
    };
  });

  return { version, patientId, recordingId, startDate, startTime, numRecords, recordDuration: recordDur, numChannels, signals };
}

// ── Data record reader ────────────────────────────────────────────────────────

/**
 * Reads all data records from an EDF/BDF buffer and converts int16 values to
 * physical Float32 values for each channel.
 *
 * Returns one Float32Array per channel, each of length
 *   header.numRecords × header.signals[i].samplesPerRecord
 *
 * For BDF (24-bit / 3 bytes per sample), set `bytesPerSample = 3`.
 */
export function readEdfData(
  buffer: ArrayBuffer,
  header: EdfHeader,
  bytesPerSample: 2 | 3 = 2,
): Float32Array[] {
  const { signals, numRecords } = header;
  const ns = signals.length;

  // Total samples per channel
  const totalSamples = signals.map((s) => s.samplesPerRecord * numRecords);

  // Allocate output arrays
  const output: Float32Array[] = signals.map((_, i) => new Float32Array(totalSamples[i]!));

  // Pre-compute gain/offset per channel (avoids division in the inner loop)
  const gain:   number[] = signals.map((s) =>
    (s.physicalMax - s.physicalMin) / (s.digitalMax - s.digitalMin || 1),
  );
  const physOffset: number[] = signals.map((s, i) =>
    s.physicalMin - gain[i]! * s.digitalMin,
  );

  // Header size in bytes: 256 (global) + ns*256 (per-signal)
  const headerBytes = 256 + ns * 256;
  const view = new DataView(buffer, headerBytes);
  let byteOffset = 0;

  // Fill sample index per channel
  const sampleIdx = new Int32Array(ns); // tracks where to write next sample

  for (let rec = 0; rec < numRecords; rec++) {
    for (let ch = 0; ch < ns; ch++) {
      const spr = signals[ch]!.samplesPerRecord;
      const g   = gain[ch]!;
      const po  = physOffset[ch]!;
      const out = output[ch]!;
      let   si  = sampleIdx[ch]!;

      if (bytesPerSample === 2) {
        // EDF: 16-bit signed integers, little-endian
        for (let s = 0; s < spr; s++) {
          const raw = view.getInt16(byteOffset, true);
          out[si++]  = raw * g + po;
          byteOffset += 2;
        }
      } else {
        // BDF: 24-bit signed integers, little-endian
        for (let s = 0; s < spr; s++) {
          const b0 = view.getUint8(byteOffset);
          const b1 = view.getUint8(byteOffset + 1);
          const b2 = view.getUint8(byteOffset + 2);
          // Reconstruct signed 24-bit value
          let raw = b0 | (b1 << 8) | (b2 << 16);
          if (raw & 0x800000) raw |= 0xff000000; // sign-extend to 32 bits
          out[si++]  = raw * g + po;
          byteOffset += 3;
        }
      }

      sampleIdx[ch] = si;
    }
  }

  return output;
}

// ── Time axis builder ─────────────────────────────────────────────────────────

/**
 * Builds a shared time axis in seconds for channel 0 (the channel with the
 * highest sample rate or the first channel, depending on the file).
 *
 * In mixed-frequency EDF files, each channel may have a different sample rate.
 * For display purposes we use the highest-rate channel as the time reference.
 */
export function buildTimeAxis(header: EdfHeader): Float32Array {
  const maxSpr = Math.max(...header.signals.map((s) => s.samplesPerRecord));
  const totalSamples = maxSpr * header.numRecords;
  const maxRate = maxSpr / (header.recordDuration || 1);
  const time = new Float32Array(totalSamples);
  for (let i = 0; i < totalSamples; i++) {
    time[i] = i / maxRate;
  }
  return time;
}
