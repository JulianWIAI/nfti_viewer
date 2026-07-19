/**
 * timeseries.types.ts — EEG / iEEG / MEG / fNIRS shared data types
 * ──────────────────────────────────────────────────────────────────
 *
 * Both the EDF worker and the SNIRF worker produce data that ultimately ends up
 * in a TimeseriesPayload. The rendering layer (TimeseriesViewer, NirsViewer)
 * depends only on this file — it never imports from the workers directly.
 *
 * Memory layout convention:
 *   channels[channelIndex][sampleIndex]
 * Each channel is a Float32Array for GPU/canvas efficiency.
 * The time axis is a shared Float32Array (seconds from recording start).
 */

// ── EDF-specific types ────────────────────────────────────────────────────────

/**
 * Metadata for one signal (channel) as parsed from an EDF/BDF header.
 * These are the 11 per-signal fields defined in the EDF specification.
 */
export interface EdfSignalHeader {
  /** Channel label, e.g. "EEG Fp1", "ECG", "EOG". */
  label: string;
  /** Transducer type, e.g. "AgAgCl electrode". */
  transducerType: string;
  /** Physical unit, e.g. "uV", "mV", "°C". */
  physicalDimension: string;
  /** Physical minimum value (for digital → physical conversion). */
  physicalMin: number;
  /** Physical maximum value. */
  physicalMax: number;
  /** Digital minimum (raw ADC value). */
  digitalMin: number;
  /** Digital maximum. */
  digitalMax: number;
  /** Pre-filter description, e.g. "HP:0.1Hz LP:70Hz N:50Hz". */
  prefiltering: string;
  /** Number of samples in each data record (= sampleRate × recordDuration). */
  samplesPerRecord: number;
  /**
   * Derived sampling rate in Hz (samplesPerRecord / recordDuration).
   * Computed by the parser — not stored in the EDF header explicitly.
   */
  sampleRate: number;
}

/**
 * Global EDF/BDF file header.
 */
export interface EdfHeader {
  /** EDF version string, usually "0". */
  version: string;
  /** Patient identification string. */
  patientId: string;
  /** Recording identification string. */
  recordingId: string;
  /** Recording start date (dd.mm.yy). */
  startDate: string;
  /** Recording start time (hh.mm.ss). */
  startTime: string;
  /** Total number of data records in the file. */
  numRecords: number;
  /** Duration of each data record in seconds (usually 1 or 30). */
  recordDuration: number;
  /** Total number of signals (channels) in the file. */
  numChannels: number;
  /** Per-channel metadata, one entry per signal. */
  signals: EdfSignalHeader[];
}

// ── SNIRF / fNIRS-specific types ─────────────────────────────────────────────

/**
 * One entry in the SNIRF measurement list.
 * Describes the source–detector pair and wavelength for a single data column.
 */
export interface SnirfMeasurementListEntry {
  /** 1-based index into the probe source array. */
  sourceIndex: number;
  /** 1-based index into the probe detector array. */
  detectorIndex: number;
  /** 1-based index into probe.wavelengths. */
  wavelengthIndex: number;
  /**
   * SNIRF dataType code:
   *  1 = raw optical intensity,
   *  99999 = processed (HbO/HbR/HbT),
   *  etc.
   */
  dataType: number;
}

/**
 * 3-D position of an optode (source or detector).
 */
export interface OptodePosition {
  x: number;
  y: number;
  z: number;
}

/**
 * Parsed SNIRF file payload, ready for the NIRS viewer.
 */
export interface SnirfPayload {
  /**
   * Data matrix: rows = time points, columns = channels.
   * Stored as a flat Float32Array; reshape as [numTimePoints × numChannels].
   */
  dataTimeSeries: Float32Array;
  /** Number of time points (rows in dataTimeSeries). */
  numTimePoints: number;
  /** Number of channels (columns in dataTimeSeries). */
  numChannels: number;
  /** Time axis in seconds (length = numTimePoints). */
  time: Float32Array;
  /** Measurement list — one entry per channel column. */
  measurementList: SnirfMeasurementListEntry[];
  /** Wavelengths in nm, e.g. [690, 830]. */
  wavelengths: Float32Array;
  /** Source 3-D positions (in mm). */
  sourcePositions: OptodePosition[];
  /** Detector 3-D positions (in mm). */
  detectorPositions: OptodePosition[];
  /** Source labels (if available in the file). */
  sourceLabels: string[];
  /** Detector labels (if available in the file). */
  detectorLabels: string[];
}

// ── Unified timeseries payload ────────────────────────────────────────────────

/**
 * Normalised time-series payload consumed by TimeseriesViewer.
 * Both EDF and SNIRF data are converted into this format after parsing.
 *
 * channels[i] is a Float32Array of `numSamples` physical values in the
 * unit given by channels[i].unit (μV for EEG, optical density or Δ[Hb] for NIRS).
 */
export interface TimeseriesPayload {
  /** Time axis in seconds, starting from 0. Length = numSamples. */
  time: Float32Array;
  /** Total number of samples (= time.length). */
  numSamples: number;
  /** Per-channel metadata. */
  channels: ChannelMeta[];
  /**
   * Flat channel data: one Float32Array per channel, all length = numSamples.
   * Indexed in the same order as `channels`.
   */
  data: Float32Array[];
  /** Source modality — used to pick default display settings. */
  sourceModality: 'eeg' | 'ieeg' | 'meg' | 'nirs';
  /** Original filename for display. */
  filename: string;
}

/**
 * Metadata for one channel within a TimeseriesPayload.
 */
export interface ChannelMeta {
  /** Display label, e.g. "Fp1", "S1-D1 850nm". */
  label: string;
  /** Physical unit, e.g. "μV", "OD", "μmol/L". */
  unit: string;
  /** Nominal sampling rate in Hz. */
  sampleRate: number;
  /** Whether this channel is currently selected for display. */
  visible: boolean;
  /** Optional BIDS channel type, e.g. "EEG", "EOG", "ECG", "MISC". */
  type?: string;
}

// ── Viewer control types ──────────────────────────────────────────────────────

/**
 * UI control state for the time-series viewer.
 * Lives in App.tsx and is threaded down to TimeseriesControls + TimeseriesViewer.
 */
export interface TimeseriesViewerControls {
  /** First visible time in seconds. */
  timeStart: number;
  /** Last visible time in seconds (defines the time window). */
  timeEnd: number;
  /**
   * Amplitude scale in physical units per pixel (or per lane).
   * For EEG: μV; for NIRS: OD or μmol/L.
   */
  amplitudeScale: number;
  /** Height in pixels of each channel lane. */
  laneHeightPx: number;
  /** First visible channel index (for virtual scrolling). */
  firstVisibleChannel: number;
  /** Number of channels to display simultaneously. */
  visibleChannelCount: number;
  /** Indices of channels that are user-selected / visible. */
  selectedChannels: number[];
  /** Low-cut filter frequency in Hz (0 = off). */
  filterLowHz: number;
  /** High-cut filter frequency in Hz (0 = off). */
  filterHighHz: number;
}

/**
 * NIRS-specific controls layered on top of the timeseries controls.
 */
export interface NirsViewerControls extends TimeseriesViewerControls {
  /** Which optical signal type to display. */
  signalType: 'raw' | 'od' | 'hbo' | 'hbr' | 'hbt';
  /** Selected wavelength index (0-based into SnirfPayload.wavelengths). */
  wavelengthIndex: number;
}
