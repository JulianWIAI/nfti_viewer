/**
 * MultimodalViewerAdapter.tsx — Adapts PluginViewerProps to MultimodalWorkspace
 *
 * Thin wrapper that extracts the fMRI payload from PluginViewerProps.data
 * and forwards it to MultimodalWorkspace.  MEG is loaded separately via
 * drag-and-drop inside the workspace itself.
 */

import type { FC }             from 'react';
import type { PluginViewerProps } from '../../types/plugin.types';
import type { PluginControlsProps } from '../../types/plugin.types';
import MultimodalWorkspace       from './MultimodalWorkspace';

export const MultimodalViewerComponent: FC<PluginViewerProps> = ({ data }) => {
  const fmriPayload = data?.kind === 'fmri' ? data.payload : null;
  // MEG is loaded inside MultimodalWorkspace via its own drop zone
  return <MultimodalWorkspace fmriPayload={fmriPayload} megPayload={null} />;
};

// Controls stub — each panel renders its own inline sidebar sections
export const MultimodalControlsComponent: FC<PluginControlsProps> = () => null;
