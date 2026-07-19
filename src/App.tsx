/**
 * App.tsx — Multi-modal neuroimaging platform root
 * ──────────────────────────────────────────────────
 *
 * Manages:
 *   • Plugin registration (volumetric, timeseries, nirs)
 *   • File routing via BidsRouter → ModalityRegistry → plugin.processFile()
 *   • Active plugin + parsed data state
 *   • Layout: header + plugin workspace (viewer canvas + sidebar controls)
 *
 * The sidebar layout is owned by each plugin's ViewerComponent (via the
 * `controlsSlot` prop pattern), which allows ControlsComponents to be
 * descendants of their plugin's Context.Provider.
 *
 * Data flow:
 *   File drop → BidsRouter.route() → getPlugin() → processFile()
 *     → PluginData → ViewerComponent + ControlsComponent (via context)
 */

import { useState, useCallback } from 'react';
import './App.css';

import { registerPlugin, getPlugin } from './core/ModalityRegistry';
import { BidsRouter } from './core/BidsRouter';

import volumetricPlugin from './plugins/volumetric/index';
import timeseriesPlugin from './plugins/timeseries/index';
import nirsPlugin       from './plugins/nirs/index';
import megPlugin        from './plugins/meg/index';
import eegPlugin        from './plugins/eeg/index';

import FileUpload from './components/FileUpload';

import type { PluginData, NeuroimagingPlugin } from './types/plugin.types';

// ── Register all plugins once at module level ────────────────────────────────
// (Guard against HMR double-registration in dev mode)

try { registerPlugin(volumetricPlugin); } catch { /* already registered */ }
try { registerPlugin(timeseriesPlugin); } catch { /* already registered */ }
try { registerPlugin(nirsPlugin);       } catch { /* already registered */ }
try { registerPlugin(megPlugin);        } catch { /* already registered */ }
try { registerPlugin(eegPlugin);        } catch { /* already registered */ }

// ── App state ─────────────────────────────────────────────────────────────────

interface AppState {
  plugin:   NeuroimagingPlugin | null;
  data:     PluginData;
  loading:  boolean;
  error:    string | null;
  filename: string;
}

const INITIAL_STATE: AppState = {
  plugin: null, data: null, loading: false, error: null, filename: '',
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function App() {
  const [state, setState] = useState<AppState>(INITIAL_STATE);

  const handleFiles = useCallback(async (files: File[]) => {
    if (files.length === 0) return;

    // ── BrainVision EEG trio detection (.vhdr + .eeg + .vmrk) ───────────────
    const vhdrFile = files.find((f) => f.name.toLowerCase().endsWith('.vhdr'));
    // ── KIT/Yokogawa MEG trio detection (.con + .mrk + .pos) ─────────────────
    const conFile  = files.find((f) => f.name.toLowerCase().endsWith('.con'));
    // ── Split FIF detection (multiple .fif files dropped together) ────────────
    const fifFiles = files.filter((f) => f.name.toLowerCase().endsWith('.fif'));
    // Primary FIF = the one without a split suffix, or lowest split number
    const primaryFif = fifFiles.length > 1
      ? fifFiles.slice().sort((a, b) => {
          const splitNum = (name: string) => {
            const m = name.match(/[_-]split[_-](\d+)/i) ?? name.match(/-(\d+)\.fif$/i);
            return m ? parseInt(m[1]!, 10) : 0;
          };
          return splitNum(a.name) - splitNum(b.name);
        })[0]!
      : fifFiles[0];

    // Use the specialised entry-point file as primary; otherwise first file
    const primaryFile = vhdrFile ?? conFile ?? primaryFif ?? files[0]!;

    const bidsFile = BidsRouter.route(primaryFile);

    // Attach all companion files so the plugin can upload the full file set.
    if (vhdrFile && files.length > 1) {
      (bidsFile as typeof bidsFile & { files: File[] }).files = files;
    } else if (conFile) {
      (bidsFile as typeof bidsFile & { files: File[] }).files = files;
    } else if (fifFiles.length > 1) {
      (bidsFile as typeof bidsFile & { files: File[] }).files = fifFiles;
    }

    const plugin = getPlugin(bidsFile.modality);

    if (!plugin) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error:   `No plugin registered for modality "${bidsFile.modality}" `
               + `(extension: ${bidsFile.extension})`,
      }));
      return;
    }

    setState({ plugin, data: null, loading: true, error: null, filename: primaryFile.name });

    try {
      const data = await plugin.processFile(bidsFile);
      setState((prev) => ({ ...prev, data, loading: false }));
    } catch (err) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error:   err instanceof Error ? err.message : String(err),
      }));
    }
  }, []);

  const { plugin, data, loading, error, filename } = state;

  // Build the sidebar (file upload + plugin controls if a plugin is active)
  const sidebarContent = (
    <aside className="sidebar">
      <FileUpload
        onFiles={handleFiles}
        loading={loading}
        error={error}
        loadedFileName={filename || undefined}
      />
      {plugin && (
        <plugin.ControlsComponent
          data={data}
          loading={loading}
          error={error}
        />
      )}
    </aside>
  );

  return (
    <div className="app">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="app-header">
        <h1>NeuroViewer</h1>
        <span className="app-header__badge">
          {plugin ? plugin.name : 'Multi-modal neuroimaging platform'}
        </span>
      </header>

      {/* ── Main content ────────────────────────────────────────────────── */}
      <div className="app-body">
        {plugin ? (
          // Active plugin: ViewerComponent owns the workspace layout
          // and renders controlsSlot (the sidebar) inside its Context.Provider
          <plugin.ViewerComponent data={data} controlsSlot={sidebarContent} />
        ) : (
          // No plugin yet: show a welcoming empty state with the file upload
          <div className="plugin-workspace">
            <div className="welcome-state">
              <div className="welcome-state__inner">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth="1" aria-hidden="true">
                  <path d="M12 2C9.5 2 7.5 3.5 7 5.5C5.3 5.8 4 7.2 4 9c0 1.1.4 2.1 1.1 2.8C4.4 12.4 4 13.2 4 14c0 2.2 1.8 4 4 4h8c2.2 0 4-1.8 4-4 0-.8-.4-1.6-1.1-2.2C19.6 11.1 20 10.1 20 9c0-1.8-1.3-3.2-3-3.5C16.5 3.5 14.5 2 12 2z" />
                </svg>
                <h2>Drop a neuroimaging file to begin</h2>
                <p className="welcome-state__formats">
                  MRI / PET: <code>.nii</code> · <code>.nii.gz</code>
                  &ensp;|&ensp;
                  EEG: <code>.edf</code> · <code>.bdf</code> · <code>.vhdr</code>
                  &ensp;|&ensp;
                  MEG: <code>.fif</code> · <code>.con+.mrk+.pos</code>
                  &ensp;|&ensp;
                  fNIRS: <code>.snirf</code>
                </p>
              </div>
            </div>
            {sidebarContent}
          </div>
        )}
      </div>
    </div>
  );
}
