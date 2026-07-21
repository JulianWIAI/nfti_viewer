/**
 * useSegmentationPicker.ts — vtk.js DOM-click → FreeSurfer label → onLabelPick()
 * ─────────────────────────────────────────────────────────────────────────────────
 *
 * Attaches a native click listener to the VTK container div.
 * On click:
 *   1. Determine which of the four renderers was clicked (quadrant hit-test).
 *   2. Convert DOM pixel → vtk.js display coord → world point via vtkCoordinate.
 *   3. Map world point → voxel index using imageData origin + spacing.
 *   4. Look up the FreeSurfer label in the flat Fortran-order label array.
 *   5. Call onLabelPick(label) for any non-background (> 0) label.
 *
 * The 3-D volume quadrant (bottom-right) is excluded — point picking on a
 * ray-cast volume is ambiguous without a dedicated vtkCellPicker.
 *
 * VIEWPORT LAYOUT (normalised coords, y from bottom — same as vtkSetup.ts):
 *   axial    [0.0, 0.5, 0.5, 1.0]  top-left
 *   coronal  [0.5, 0.5, 1.0, 1.0]  top-right
 *   sagittal [0.0, 0.0, 0.5, 0.5]  bottom-left
 *   volume   [0.5, 0.0, 1.0, 0.5]  bottom-right  ← skipped
 */

import { useEffect, type RefObject } from 'react';
import vtkCoordinate from '@kitware/vtk.js/Rendering/Core/Coordinate';
import type { VtkContext } from '../lib/vtk/vtkSetup';
import type { SegmentationBundle } from '../lib/vtk/segmentationOverlay';
import type { VolumeActorBundle } from '../lib/vtk/volumeRenderer';

// ── Caller-provided refs shape ────────────────────────────────────────────────

export interface PickerRefs {
  ctx:          VtkContext | null;
  segBundle:    SegmentationBundle | null;
  volumeBundle: VolumeActorBundle | null;
}

export interface UseSegmentationPickerOptions {
  /** The div wrapping the vtk.js WebGL canvas (containerRef from VolumetricViewer). */
  containerRef: RefObject<HTMLDivElement>;
  /** Stable callback returning the current mutable vtk refs without causing re-renders. */
  getPickerRefs: () => PickerRefs;
  /** Stable callback returning [nx, ny, nz] from the volume header, or null if no volume. */
  getVolumeDims: () => [number, number, number] | null;
  /** Called with a non-zero FreeSurfer label ID when the user clicks a labelled voxel. */
  onLabelPick: (label: number) => void;
  /** Disable the listener (e.g. before segmentation finishes). */
  enabled: boolean;
}

// ── Quadrant → renderer mapping ───────────────────────────────────────────────

/**
 * Returns the vtk renderer that owns the click position, or null for the
 * 3-D volume pane (bottom-right quadrant) where picking is skipped.
 *
 * normX / normY are vtk-normalised coords: 0-1 from left / bottom.
 */
function rendererForClick(
  normX: number,
  normY: number,
  ctx: VtkContext,
): typeof ctx.axialRenderer | null {
  if (normX < 0.5 && normY >= 0.5) return ctx.axialRenderer;
  if (normX >= 0.5 && normY >= 0.5) return ctx.coronalRenderer;
  if (normX < 0.5 && normY < 0.5)  return ctx.sagittalRenderer;
  return null; // volume pane
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useSegmentationPicker({
  containerRef,
  getPickerRefs,
  getVolumeDims,
  onLabelPick,
  enabled,
}: UseSegmentationPickerOptions): void {
  useEffect(() => {
    if (!enabled) return;
    const container = containerRef.current;
    if (!container) return;

    const handleClick = (e: MouseEvent) => {
      const { ctx, segBundle, volumeBundle } = getPickerRefs();
      if (!ctx || !segBundle || !volumeBundle) return;

      const dims = getVolumeDims();
      if (!dims) return;
      const [nX, nY, nZ] = dims;

      // ── 1. DOM → vtk display coordinates ─────────────────────────────────
      const rect  = container.getBoundingClientRect();
      const domX  = e.clientX - rect.left;
      const domY  = e.clientY - rect.top;
      const W     = rect.width;
      const H     = rect.height;
      // vtk.js display y is measured from the bottom of the canvas (OpenGL convention)
      const dispX = domX;
      const dispY = H - domY;

      // ── 2. Determine quadrant renderer ────────────────────────────────────
      const renderer = rendererForClick(dispX / W, dispY / H, ctx);
      if (!renderer) return;

      // ── 3. Display → world via vtkCoordinate ──────────────────────────────
      const coord = vtkCoordinate.newInstance();
      coord.setCoordinateSystemToDisplay();
      coord.setValue([dispX, dispY, 0]);
      const world = coord.getComputedWorldValue(renderer) as
        [number, number, number] | undefined;
      if (!world) return;

      // ── 4. World → voxel index ────────────────────────────────────────────
      const origin  = volumeBundle.imageData.getOrigin()  as [number, number, number];
      const spacing = volumeBundle.imageData.getSpacing() as [number, number, number];

      const i = Math.round((world[0] - origin[0]) / spacing[0]);
      const j = Math.round((world[1] - origin[1]) / spacing[1]);
      const k = Math.round((world[2] - origin[2]) / spacing[2]);

      if (i < 0 || i >= nX || j < 0 || j >= nY || k < 0 || k >= nZ) return;

      // ── 5. Fortran-order label lookup ─────────────────────────────────────
      const idx   = i + j * nX + k * nX * nY;
      const label = segBundle.labelFlat[idx] ?? 0;
      if (label > 0) {
        onLabelPick(label);
      }
    };

    container.addEventListener('click', handleClick);
    return () => {
      container.removeEventListener('click', handleClick);
    };
  }, [containerRef, getPickerRefs, getVolumeDims, onLabelPick, enabled]);
}
