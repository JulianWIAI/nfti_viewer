/**
 * ModalityRegistry.ts — Plugin registration and lookup
 * ──────────────────────────────────────────────────────
 *
 * A simple registry that maps Modality → NeuroimagingPlugin.
 * Plugins are registered at app startup (main.tsx) and retrieved at runtime
 * by the BidsRouter → Viewer pipeline.
 *
 * WHY a registry instead of hard-coded imports?
 *   • Plugins can be added without touching App.tsx or Viewer.tsx.
 *   • Future: plugins could be loaded dynamically (async import) from a URL,
 *     enabling a true extension marketplace for the platform.
 *   • Keeps the host code dependency-free from vtk.js, uPlot, or HDF5 — only
 *     the concrete plugin files import those heavy libraries.
 */

import type { NeuroimagingPlugin } from '../types/plugin.types';
import type { Modality } from '../types/bids.types';

// ── Registry store ────────────────────────────────────────────────────────────

// Map from modality string to plugin descriptor.
// Using a plain Map keeps lookup O(1) and preserves insertion order.
const registry = new Map<Modality, NeuroimagingPlugin>();

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Register a plugin with the registry.
 * Should be called once per plugin at application startup, before any files
 * are loaded.
 *
 * Throws if a plugin for the same modality is already registered, to catch
 * accidental double-registration in development.
 *
 * @param plugin - Plugin manifest conforming to NeuroimagingPlugin.
 */
export function registerPlugin(plugin: NeuroimagingPlugin): void {
  if (registry.has(plugin.modality)) {
    throw new Error(
      `ModalityRegistry: a plugin for modality "${plugin.modality}" is already registered ` +
        `(existing: "${registry.get(plugin.modality)!.id}", new: "${plugin.id}"). ` +
        'Call unregisterPlugin() first if you intend to replace it.',
    );
  }
  registry.set(plugin.modality, plugin);
}

/**
 * Remove a plugin from the registry by modality.
 * Useful in testing and for hot-swapping plugins in development.
 */
export function unregisterPlugin(modality: Modality): boolean {
  return registry.delete(modality);
}

/**
 * Retrieve the plugin registered for a given modality.
 * Returns null (not an error) when no plugin handles the modality — the caller
 * should show a "modality not supported" message instead of crashing.
 */
export function getPlugin(modality: Modality): NeuroimagingPlugin | null {
  return registry.get(modality) ?? null;
}

/**
 * Returns all registered plugins in registration order.
 * Used by the UI to list available modalities.
 */
export function getAllPlugins(): NeuroimagingPlugin[] {
  return Array.from(registry.values());
}

/**
 * Returns the modalities that have a registered plugin.
 */
export function getSupportedModalities(): Modality[] {
  return Array.from(registry.keys());
}

/**
 * True if a plugin is registered for this modality.
 */
export function isSupported(modality: Modality): boolean {
  return registry.has(modality);
}
