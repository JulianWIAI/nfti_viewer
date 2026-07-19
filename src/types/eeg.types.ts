/**
 * eeg.types.ts — BrainVision EEG plugin data types
 */

import type { ChannelInfo } from '../services/eegApi';

/**
 * Payload returned after a successful /api/load-eeg call.
 * Held in App state and passed to EegViewer via PluginData.
 */
export interface EegSessionPayload {
  sessionId:     string;
  filename:      string;
  samplingRate:  number;
  totalDuration: number;
  nSamples:      number;
  nChannels:     number;
  channels:      ChannelInfo[];
}
