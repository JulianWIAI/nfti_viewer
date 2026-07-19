/**
 * meg.types.ts — MEG session payload types
 * ──────────────────────────────────────────
 *
 * Returned by the MEG plugin's processFile() after a successful load via the
 * Python FastAPI backend. Unlike the EDF/SNIRF plugins, the MEG plugin does
 * NOT transfer the raw signal data to the frontend upfront — it only transfers
 * the session metadata. Signal data is fetched in viewport-sized chunks on
 * demand via megApi.getChannelData().
 */

import type { ChannelInfo } from '../services/megApi';

/** Session metadata returned after a successful /api/load-meg call. */
export interface MegSessionPayload {
  /** UUID assigned by the FastAPI backend — required for all data requests. */
  sessionId:     string;
  /** Original filename (for display). */
  filename:      string;
  /** Sampling frequency in Hz (e.g. 1000 for Elekta Neuromag). */
  samplingRate:  number;
  /** Total recording duration in seconds. */
  totalDuration: number;
  /** Total sample count. */
  nSamples:      number;
  /** Total channel count. */
  nChannels:     number;
  /** Full channel list with name, type, unit, and index. */
  channels:      ChannelInfo[];
}
