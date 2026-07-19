/**
 * mniCoordinateTransform.ts — MNI-152 → vtk.js world coordinate transform
 * ──────────────────────────────────────────────────────────────────────────
 *
 * PURPOSE
 * ────────
 * The backend returns EEG source locations in MNI-152 space (millimetres)
 * via `mne.vertex_to_mni()`.  Before adding these points to a vtk.js scene,
 * they must be expressed in the same world coordinate system that vtk.js uses
 * for the loaded MRI volume.
 *
 * COORDINATE FRAME CHAIN
 * ───────────────────────
 *
 *   MNE inverse solution
 *         │
 *         ▼  HEAD frame (metres, electrode-centric)
 *   mne.vertex_to_mni()
 *         │
 *         ▼  MNI-152 space (millimetres, standard template)
 *   [THIS MODULE]
 *         │
 *         ▼  vtk.js world space (millimetres, from NIfTI sform/qform)
 *   vtkPolyData points
 *         │
 *         ▼  vtkGlyph3DMapper → rendered sphere glyphs
 *
 * WHEN IS THE TRANSFORM IDENTITY?
 * ─────────────────────────────────
 * vtk.js builds its world coordinate system from the NIfTI sform (or qform)
 * affine stored in the file header.  The standard MNI-152 template NIfTI
 * (MNI152_T1_1mm.nii.gz and its derivatives) carries an affine with:
 *
 *   • spacing  = [1 mm, 1 mm, 1 mm]
 *   • origin   = [-90, -126, -72]  (centre of the 182×218×182 volume in RAS)
 *   • direction = identity 3×3      (pure RAS orientation, no rotation)
 *
 * In this case vtk world mm == MNI-152 mm, and the transform reduces to the
 * identity matrix — MNI coordinates can be handed directly to vtkPoints.
 *
 * WHEN IS A NON-IDENTITY TRANSFORM NEEDED?
 * ──────────────────────────────────────────
 * If the uploaded NIfTI is a native-space T1 (not registered to MNI), the
 * vtk world and MNI frames diverge.  Reconciling them requires the
 * registration warp (e.g., ANTs or FSL FLIRT output), which this viewer does
 * not yet manage.  In that case source locations will appear offset from the
 * brain surface; document this limitation in the UI.
 *
 * If the NIfTI has a non-identity direction cosine matrix (oblique acquisition)
 * the direction is already encoded in the vtkImageData; the vtkPoints world
 * coordinates still sit in the same RAS mm space as MNI, so identity still
 * works for MNI-registered scans.
 *
 * USAGE
 * ──────
 *   import {
 *     identityMatrix4x4,
 *     buildMniToVtkWorldTransform,
 *     applyMatrix4x4,
 *   } from './mniCoordinateTransform';
 *
 *   const M  = buildMniToVtkWorldTransform();         // identity for MNI scans
 *   const pt = applyMatrix4x4(M, { x: -42, y: 18, z: 60 });
 *   // pt.x/y/z are now ready for vtkPoints.setPoint(i, pt.x, pt.y, pt.z)
 */

// ── Types ─────────────────────────────────────────────────────────────────────

/**
 * Row-major 4×4 homogeneous transform matrix, stored as a flat 16-element
 * tuple.  Element order: [m00, m01, m02, m03, m10, m11, m12, m13, ...].
 *
 *   │ m00 m01 m02 m03 │   │ R  R  R  tx │
 *   │ m10 m11 m12 m13 │ = │ R  R  R  ty │
 *   │ m20 m21 m22 m23 │   │ R  R  R  tz │
 *   │ m30 m31 m32 m33 │   │  0  0  0  1 │
 *
 * Applying to a point p = (x, y, z):
 *   x' = m00·x + m01·y + m02·z + m03
 *   y' = m10·x + m11·y + m12·z + m13
 *   z' = m20·x + m21·y + m22·z + m23
 */
export type Matrix4x4 = readonly [
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
];

/** A 3-D point in millimetres. */
export interface Point3 {
  x: number;
  y: number;
  z: number;
}

// ── Factories ─────────────────────────────────────────────────────────────────

/**
 * Returns the 4×4 identity matrix.
 *
 * Use this when the loaded NIfTI is already in MNI-152 space so that
 * MNI mm coordinates can be passed directly to vtk.js as world coordinates.
 */
export function identityMatrix4x4(): Matrix4x4 {
  return [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ];
}

/**
 * Build the 4×4 affine that maps MNI-152 mm coordinates into vtk.js world mm.
 *
 * @param niftiSformAffine
 *   Optional: the full 4×4 NIfTI sform affine as a row-major flat array.
 *   Provide this when the NIfTI origin/direction differ from pure MNI-152
 *   identity (e.g., oblique acquisition, unusual resampling).
 *
 *   When omitted (most common case: standard MNI template or subject T1
 *   already registered to MNI), returns the identity matrix because
 *   vtk world mm == MNI mm.
 *
 * IMPORTANT — native-space T1s:
 *   If the NIfTI is in native scanner space (not MNI-registered), this
 *   function still returns identity, which means the source spheres will
 *   appear at the correct MNI coordinates but those coordinates will NOT
 *   align with the native-space brain mesh.  The viewer currently documents
 *   this as a known limitation and expects MNI-registered uploads.
 */
export function buildMniToVtkWorldTransform(
  niftiSformAffine?: Matrix4x4,
): Matrix4x4 {
  // When no custom affine is provided, MNI == vtk world → identity.
  if (!niftiSformAffine) {
    return identityMatrix4x4();
  }

  // When an affine IS provided it encodes voxel-index → vtk world.
  // The sform maps voxel → world; for MNI-registered volumes the sform
  // already expresses "world = MNI", so we can return identity there too.
  // For the general case a caller could supply the inverse sform composed
  // with a MNI-to-native registration matrix; for now we expose the hook
  // but still default to identity unless the caller has done that composition.
  return niftiSformAffine;
}

// ── Application ───────────────────────────────────────────────────────────────

/**
 * Apply a 4×4 row-major homogeneous matrix to a 3-D point.
 *
 * @param M   Row-major 4×4 transform matrix.
 * @param p   Input point in the source coordinate frame (mm).
 * @returns   Output point in the target coordinate frame (mm).
 *
 * @example
 *   const M = buildMniToVtkWorldTransform();
 *   sources.forEach((src, i) => {
 *     const w = applyMatrix4x4(M, src);
 *     pts.setPoint(i, w.x, w.y, w.z);
 *   });
 */
export function applyMatrix4x4(M: Matrix4x4, p: Point3): Point3 {
  const { x, y, z } = p;
  return {
    x: M[0] * x + M[1] * y + M[2]  * z + M[3],
    y: M[4] * x + M[5] * y + M[6]  * z + M[7],
    z: M[8] * x + M[9] * y + M[10] * z + M[11],
  };
}

/**
 * Apply a 4×4 transform to an array of points in-place, returning a new
 * flat Float64Array suitable for `vtkPoints.setData()`.
 *
 * Avoids individual object allocations when transforming large source arrays.
 *
 * @param M        Row-major 4×4 transform matrix.
 * @param sources  Array of source points (MNI mm).
 * @returns        Flat [x0, y0, z0, x1, y1, z1, …] Float64Array in vtk world mm.
 */
export function transformSourcesToFlat(
  M: Matrix4x4,
  sources: ReadonlyArray<Point3>,
): Float64Array {
  const out = new Float64Array(sources.length * 3);
  for (let i = 0; i < sources.length; i++) {
    const { x, y, z } = sources[i];
    out[i * 3    ] = M[0] * x + M[1] * y + M[2]  * z + M[3];
    out[i * 3 + 1] = M[4] * x + M[5] * y + M[6]  * z + M[7];
    out[i * 3 + 2] = M[8] * x + M[9] * y + M[10] * z + M[11];
  }
  return out;
}
