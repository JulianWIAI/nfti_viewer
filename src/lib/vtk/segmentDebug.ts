/**
 * segmentDebug.ts — Diagnostic utilities for the SynthSeg segmentation pipeline
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * All functions here are development-only helpers.  None are imported by
 * production code.  Enable individual checks by un-commenting the call sites
 * you care about and importing from this module.
 *
 * Root cause of diagonal stripe artefacts
 * ─────────────────────────────────────────
 * vtk.js vtkImageData stores voxels in Fortran order (X varies fastest).
 * If the backend returns dims = [257, 300, 180] while the MRI vtkImageData
 * has dims = [256, 300, 180], every displayed row starts 1 pixel further
 * along the flat buffer than the MRI row → slanted stripes.
 *
 * The backend fix (backend/main.py, /api/segment) clamps the inverse-zoom
 * output to orig_shape, so dims always matches the NIfTI header exactly.
 * Use verifyDimsMatch() below to assert this at runtime during testing.
 */

import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';

// ── 1. Python serialisation reference (documented here, runs on backend) ──────
//
// THE CRITICAL TRAP
// ─────────────────
// np.ndarray.tobytes() has an `order` kwarg that defaults to 'C'.
// It controls the TRAVERSAL ORDER, NOT the array's memory layout.
// So np.asfortranarray(arr).tobytes() STILL produces C-order bytes!
//
//   # WRONG — creates F-contiguous array then serialises in C order:
//   np.asfortranarray(labels_orig).tobytes()        # last axis fastest (Z)
//
//   # RIGHT — serialises in Fortran order regardless of memory layout:
//   labels_orig.tobytes(order='F')                  # first axis fastest (X)
//
// vtk.js vtkImageData point scalars are stored with the first dimension
// (X / setDimensions arg 0) varying fastest, which is Fortran order.
// NIfTI files on disk also store data this way (dim[1] varies fastest).
//
// Correct Fortran-order serialisation in Python:
//
//   import numpy as np, base64
//
//   def serialize_labels(labels_orig: np.ndarray) -> str:
//       """
//       Serialise a 3-D uint8 label volume so voxel [i,j,k] lands at
//       flat offset  i + j*nx + k*nx*ny  — Fortran order (X fastest).
//       """
//       assert labels_orig.ndim == 3, "labels must be 3-D"
//       arr = labels_orig.astype(np.uint8, copy=False)
//       return base64.b64encode(arr.tobytes(order='F')).decode("ascii")
//
// Shape clamp (prevents ±1 scipy.zoom rounding → stride mismatch → stripes):
//
//   def clamp_to_shape(arr: np.ndarray, target: tuple) -> np.ndarray:
//       if arr.shape == target:
//           return arr
//       canvas = np.zeros(target, dtype=np.uint8)
//       clip   = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target))
//       canvas[clip] = arr[clip]
//       return canvas

// ── 2. vtk.js deserialisation reference ──────────────────────────────────────
//
// Correct pattern (mirrors what segmentationOverlay.ts already does):
//
//   function buildLabelImageData(
//     labelFlat : Uint8Array,          // raw Fortran-order bytes from backend
//     dims      : [number,number,number], // [x, y, z] — must equal MRI dims
//     mriSource : ReturnType<typeof vtkImageData.newInstance>,
//   ) {
//     const imgData = vtkImageData.newInstance();
//
//     // Copy geometry from the MRI so both actors share the same world space.
//     imgData.setSpacing(mriSource.getSpacing());
//     imgData.setOrigin(mriSource.getOrigin());
//     (imgData as any).setDirection((mriSource as any).getDirection());
//
//     // dims must equal the MRI dimensions — any mismatch causes stripes.
//     imgData.setDimensions(dims[0], dims[1], dims[2]);
//
//     // Build RGBA by LUT-lookup over the Fortran-order flat array.
//     // Order is preserved: voxel index i in labelFlat corresponds to the
//     // same physical location i in the MRI flat buffer (both Fortran order).
//     const rgba = new Uint8Array(labelFlat.length * 4);
//     for (let i = 0; i < labelFlat.length; i++) {
//       const label = labelFlat[i]! & 0xff;
//       rgba[i * 4    ] = LUT[label * 4    ];  // R
//       rgba[i * 4 + 1] = LUT[label * 4 + 1]; // G
//       rgba[i * 4 + 2] = LUT[label * 4 + 2]; // B
//       rgba[i * 4 + 3] = label === 0 ? 0 : 153; // A: transparent bg
//     }
//
//     const scalars = vtkDataArray.newInstance({
//       name: 'SynthSegRGBA',
//       numberOfComponents: 4,
//       values: rgba,
//     });
//     imgData.getPointData().setScalars(scalars);
//     return imgData;
//   }

// ── 3. Runtime assertion helpers (import and call during testing) ─────────────

/**
 * Assert that `overlayDims` returned by the backend exactly matches the MRI
 * vtkImageData dimensions.  Logs a warning (never throws) so it's safe to
 * keep in a staging build.
 *
 * Usage:
 *   import { verifyDimsMatch } from '../lib/vtk/segmentDebug';
 *   verifyDimsMatch(result.dims as [number,number,number], mriImageData);
 */
export function verifyDimsMatch(
  overlayDims: [number, number, number],
  mriImageData: ReturnType<typeof vtkImageData.newInstance>,
): boolean {
  const mriDims = mriImageData.getDimensions() as [number, number, number];
  const match =
    overlayDims[0] === mriDims[0] &&
    overlayDims[1] === mriDims[1] &&
    overlayDims[2] === mriDims[2];

  if (!match) {
    console.warn(
      '[segmentDebug] Dimension mismatch — will cause diagonal stripes!\n' +
      `  overlay: [${overlayDims.join(', ')}]\n` +
      `  MRI:     [${mriDims.join(', ')}]`,
    );
  }
  return match;
}

/**
 * Assert that a flat label buffer has the correct voxel count for the given dims.
 *
 * Usage:
 *   import { verifyBufferSize } from '../lib/vtk/segmentDebug';
 *   verifyBufferSize(labelFlat, result.dims as [number,number,number]);
 */
export function verifyBufferSize(
  labelFlat: Uint8Array,
  dims: [number, number, number],
): boolean {
  const expected = dims[0] * dims[1] * dims[2];
  const ok = labelFlat.length === expected;
  if (!ok) {
    console.warn(
      '[segmentDebug] Buffer size mismatch!\n' +
      `  labelFlat.length: ${labelFlat.length}\n` +
      `  dims product:     ${expected}  (${dims.join('×')})`,
    );
  }
  return ok;
}

/**
 * Spot-check that the flat buffer is in Fortran order by sampling a known
 * voxel and comparing it to what C order would place there.
 *
 * Because the backend uses np.asfortranarray().tobytes(), the voxel at
 * logical position (x, y, z) is at flat index:  x + y*dimX + z*dimX*dimY
 * (Fortran / column-major).  A C-order buffer would place it at:
 *   z*dimY*dimX + y*dimX + x — different for non-trivial coordinates.
 *
 * This function cannot distinguish correct Fortran from accidental alignment
 * without ground-truth values, so it only prints the two candidate indices
 * for a human to inspect in the browser debugger.
 *
 * Usage:
 *   import { logOrderProbe } from '../lib/vtk/segmentDebug';
 *   logOrderProbe(labelFlat, result.dims as [number,number,number], [10,20,5]);
 */
export function logOrderProbe(
  labelFlat: Uint8Array,
  dims: [number, number, number],
  voxel: [number, number, number],
): void {
  const [x, y, z] = voxel;
  const [dx, dy] = dims;
  const fortranIdx = x + y * dx + z * dx * dy;
  const cIdx       = z * dy * dx + y * dx + x;

  console.group('[segmentDebug] Order probe at voxel', voxel);
  console.log('Fortran index:', fortranIdx, '→ label', labelFlat[fortranIdx]);
  console.log('C-order index:', cIdx,       '→ label', labelFlat[cIdx]);
  console.log('(Both are the same iff x=0 or single-column data)');
  console.groupEnd();
}
