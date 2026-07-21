/**
 * plugin.types.ts — Plugin system interface contract
 * ────────────────────────────────────────────────────
 *
 * Every neuroimaging plugin (volumetric, timeseries, nirs) exports one object
 * conforming to NeuroimagingPlugin. The ModalityRegistry holds all registered
 * plugins and the Viewer host queries it to decide what to render.
 *
 * Design goal: plugins are entirely self-contained — they own their viewer,
 * controls, and internal state. The host (Viewer.tsx) knows nothing about
 * vtk.js, uPlot, or HDF5; it only calls the plugin's React components.
 */

import type { ComponentType, ReactNode } from 'react';
import type { Modality, BidsFile } from './bids.types';
import type { VolumePayload } from './nifti.types';
import type { TimeseriesPayload, NirsViewerControls, TimeseriesViewerControls, SnirfPayload } from './timeseries.types';
import type { ViewerControls, SliceMaxima } from './nifti.types';
import type { MegSessionPayload } from './meg.types';
import type { EegSessionPayload } from './eeg.types';
import type { FmriPayload } from './fmri.types';

// ── Generic plugin data union ─────────────────────────────────────────────────

/**
 * Union of all possible parsed-data payloads.
 * The plugin host receives one of these and passes it to the correct plugin.
 */
export type PluginData =
  | { kind: 'volumetric'; payload: VolumePayload }
  | { kind: 'fmri';       payload: FmriPayload }
  | { kind: 'timeseries'; payload: TimeseriesPayload }
  | { kind: 'nirs';       payload: SnirfPayload }
  | { kind: 'meg';        payload: MegSessionPayload }
  | { kind: 'eeg';        payload: EegSessionPayload }
  | null;

// ── Plugin viewer props ───────────────────────────────────────────────────────

/** Props passed by the host to every plugin's Viewer component. */
export interface PluginViewerProps {
  /** Parsed data — null while loading or before a file is opened. */
  data: PluginData;
  /**
   * The ControlsComponent instance to render inside the viewer's Context.Provider.
   * Each plugin's ViewerComponent owns the full workspace layout (canvas + sidebar).
   * The host (App.tsx) passes the controls here so they become descendants of the
   * Provider, which allows them to consume the plugin's React context.
   */
  controlsSlot?: ReactNode;
}

// ── Plugin controls props ─────────────────────────────────────────────────────

/**
 * Props passed by the host to every plugin's Controls component.
 * Controls are intentionally generic; each plugin narrows the control types
 * in its own ControlsComponent implementation.
 */
export interface PluginControlsProps {
  /** Parsed data (read-only, for computing slider maxima). */
  data: PluginData;
  /** True while the worker is parsing a file. */
  loading: boolean;
  /** Worker error message, or null. */
  error: string | null;
  /** Called when the user triggers ONNX inference (volumetric only). */
  onRunSegmentation?: () => void;
}

// ── Plugin manifest ───────────────────────────────────────────────────────────

/**
 * The manifest that every plugin must export from its index.ts.
 * Registered with ModalityRegistry at app startup.
 */
export interface NeuroimagingPlugin {
  /** Unique identifier, e.g. "volumetric", "timeseries", "nirs". */
  id: string;

  /** Human-readable name shown in the UI header. */
  name: string;

  /** Which top-level modality this plugin handles. */
  modality: Modality;

  /**
   * File extensions this plugin can process (lower-cased, with leading dot).
   * Used by BidsRouter to verify the routed file matches the plugin.
   */
  supportedExtensions: string[];

  /**
   * React component that renders the main data visualisation.
   * Mounted by the Viewer host in the central canvas area.
   */
  ViewerComponent: ComponentType<PluginViewerProps>;

  /**
   * React component that renders the sidebar controls for this modality.
   * Mounted by Viewer host in the right panel.
   */
  ControlsComponent: ComponentType<PluginControlsProps>;

  /**
   * Process a raw BidsFile into the appropriate PluginData.
   * This is called on the main thread but immediately hands off to a Web Worker.
   * Returns a Promise that resolves to a PluginData when the worker is done.
   *
   * The plugin is responsible for spawning and managing its own worker.
   * The host does not care about the internals — it only awaits the result.
   */
  processFile(bidsFile: BidsFile): Promise<PluginData>;
}

// ── Active plugin state (held in App.tsx) ────────────────────────────────────

/**
 * The complete state for the currently active plugin, held in App.tsx.
 * When the user loads a new file, this state is replaced wholesale.
 */
export interface ActivePluginState {
  pluginId: string;
  modality: Modality;
  data: PluginData;
  loading: boolean;
  error: string | null;
  filename: string;
}

// ── Volumetric-specific re-export for the control bridge ─────────────────────
// These are re-exported so ControlPanel.tsx can import everything from one place.
export type { ViewerControls, SliceMaxima, TimeseriesViewerControls, NirsViewerControls };
