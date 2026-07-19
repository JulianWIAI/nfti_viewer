/**
 * synRegistrationApi.ts — HTTP client for POST /api/registration/syn
 * ────────────────────────────────────────────────────────────────────
 *
 * Thin typed fetch wrapper that sends two NIfTI files to the SyN registration
 * endpoint and returns the warped Subject B volume as a base64 float32 payload.
 *
 * The endpoint blocks for ~2–5 minutes while dipy runs the Affine + SyN
 * pipeline server-side.  fetch() has no built-in timeout, so the browser will
 * wait until the server responds (subject to browser/proxy limits, typically
 * 5–30 minutes).  No progress streaming is available; the UI shows a spinner.
 */

// ── Volumetrics types ─────────────────────────────────────────────────────────

/** Left- and right-hemisphere hippocampal volumes in cm³ for one subject. */
export interface HippocampalVolumes {
  /** Left hippocampus volume in cm³ (FreeSurfer label 17). */
  lh: number;
  /** Right hippocampus volume in cm³ (FreeSurfer label 53). */
  rh: number;
}

/**
 * Comparative hippocampal volumetrics — both subjects, from their raw
 * (un-warped) NIfTIs so values reflect true anatomy, not registration geometry.
 */
export interface SynVolumetrics {
  subjectA: HippocampalVolumes;
  subjectB: HippocampalVolumes;
}

/** Typed response from POST /api/registration/syn. */
export interface SynApiResult {
  /**
   * Base64-encoded float32 volume in Fortran order (X varies fastest).
   * Subject B warped into Subject A's voxel space.
   * Decode with:
   *   const bin       = atob(result.warped);
   *   const byteArray = new Uint8Array(bin.length);
   *   for (let i = 0; i < bin.length; i++) byteArray[i] = bin.charCodeAt(i);
   *   const warpedF32 = new Float32Array(byteArray.buffer);
   */
  warped: string;

  /**
   * [X, Y, Z] voxel dimensions of the warped volume.
   * Matches Subject A's voxel space (static reference scan).
   */
  dims: [number, number, number];

  /**
   * 4×4 RAS-mm affine of Subject A (the reference), row-major flattened into
   * 16 float64 values.  The warped volume occupies this coordinate space.
   */
  affine: number[];

  /** Minimum intensity of the warped volume (for auto-windowing). */
  min_val: number;

  /** Maximum intensity of the warped volume (for auto-windowing). */
  max_val: number;

  /** Total server-side wall-clock time in milliseconds. */
  duration_ms: number;

  /**
   * Base64-encoded uint8 Fortran-order FreeSurfer label map for Subject A in
   * Subject A's original voxel space.  Empty string when SynthSeg unavailable.
   */
  seg_a: string;

  /**
   * Base64-encoded uint8 Fortran-order FreeSurfer label map for Subject B,
   * warped into Subject A's voxel space via nearest-neighbour interpolation.
   * Empty string when SynthSeg unavailable.
   */
  seg_b_warped: string;

  /**
   * Hippocampal volumetrics from each subject's raw (un-warped) label map.
   * Null when SynthSeg unavailable.
   */
  volumetrics: SynVolumetrics | null;
}

export const synRegistrationApi = {
  /**
   * POST /api/registration/syn
   *
   * Upload Subject A (reference) and Subject B (moving), co-register B to A
   * using dipy affine + SyN, and receive the warped Subject B volume.
   *
   * @param subjectA  The reference NIfTI file (.nii / .nii.gz).
   * @param subjectB  The moving NIfTI file (.nii / .nii.gz).
   * @throws Error with a human-readable message on network failure or non-OK status.
   */
  async register(subjectA: File, subjectB: File): Promise<SynApiResult> {
    const form = new FormData();
    form.append('subject_a_ref',    subjectA);
    form.append('subject_b_moving', subjectB);

    const res = await fetch('/api/registration/syn', {
      method: 'POST',
      body:   form,
    });

    if (!res.ok) {
      // Surface the FastAPI detail message when available.
      const text = await res.text().catch(() => res.statusText);
      let detail = text;
      try {
        const json = JSON.parse(text) as { detail?: string };
        if (json.detail) detail = json.detail;
      } catch { /* not JSON — use raw text */ }
      throw new Error(`SyN registration failed (${res.status}): ${detail}`);
    }

    return res.json() as Promise<SynApiResult>;
  },
};
