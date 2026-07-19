/**
 * LongitudinalPanel.tsx — sidebar controls for the longitudinal delta overlay
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Self-contained panel that manages two NIfTI file uploads (baseline + follow-up),
 * calls context.runLongitudinalDelta(), and provides opacity/visibility controls
 * for the resulting delta overlay.  Does NOT own any VTK actors — the bundle
 * lifecycle lives in VolumetricViewer.tsx.
 *
 * RENDERING PHASES
 * ─────────────────
 *   idle      → file pickers + "Compute Delta" button
 *   uploading → status dot + "Uploading…" / button disabled
 *   running   → status dot + "Registering…" / button disabled
 *   done      → green dot + elapsed time + voxel counts
 *               opacity slider (RAF-debounced)
 *               visibility toggle
 *               blink compare toggle
 *               "Clear" button
 *   error     → red dot + error message / button re-enabled
 *
 * OPACITY SLIDER — RAF DEBOUNCING
 * ─────────────────────────────────
 * The <input type="range"> fires onChange at ~60 events/s.  We debounce the
 * VTK call to one update per animation frame using requestAnimationFrame so
 * the event loop does not queue dozens of redundant render() calls per second.
 *
 * BLINK COMPARE
 * ─────────────
 * A 500 ms setInterval alternates the overlay visibility flag so the user can
 * compare the baseline MRI against the delta overlay to verify registration
 * accuracy.  The interval is cleared on unmount and when the overlay is removed.
 */

import { useState, useRef, useEffect, useCallback, type FC } from 'react';
import { useVolumetricContext } from './VolumetricViewer';
import type { LongitudinalStatus } from './VolumetricViewer';

// ── Helper: human-readable status label ──────────────────────────────────────

function statusLabel(s: LongitudinalStatus): string {
  switch (s.phase) {
    case 'idle':      return 'Idle';
    case 'uploading': return 'Uploading…';
    case 'running':   return 'Registering…';
    case 'error':     return `Error: ${s.message}`;
    case 'done':      return `Done (${s.durationMs} ms)`;
  }
}

// ── Helper: CSS modifier class for the status dot ────────────────────────────

function dotClass(s: LongitudinalStatus): string {
  switch (s.phase) {
    case 'done':      return 'status-dot--done';
    case 'error':     return 'status-dot--error';
    case 'uploading':
    case 'running':   return 'status-dot--running';
    default:          return 'status-dot--idle';
  }
}

// ── Component ─────────────────────────────────────────────────────────────────

const LongitudinalPanel: FC = () => {
  const {
    longitudinalStatus,
    hasLongitudinalOverlay,
    showLongitudinalOverlay,
    setShowLongitudinalOverlay,
    longitudinalOpacity,
    setLongitudinalOpacity,
    nPositive,
    nNegative,
    runLongitudinalDelta,
    clearLongitudinalOverlay,
  } = useVolumetricContext();

  // ── Local file + transform state ──────────────────────────────────────────
  const [baselineFile,   setBaselineFile]   = useState<File | null>(null);
  const [followupFile,   setFollowupFile]   = useState<File | null>(null);
  const [transformType,  setTransformType]  = useState<'rigid' | 'affine'>('rigid');

  // ── RAF debounce ref for the opacity slider ───────────────────────────────
  const pendingRafRef = useRef<number | null>(null);

  // ── Blink compare refs ────────────────────────────────────────────────────
  // isBlinking: local React state (drives button label / style)
  // blinkIntervalRef: stores the setInterval handle for cleanup
  // blinkVisibleRef:  tracks the current flip state inside the interval closure
  const [isBlinking,       setIsBlinking]       = useState(false);
  const blinkIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const blinkVisibleRef  = useRef(true);

  const isRunning =
    longitudinalStatus.phase === 'uploading' ||
    longitudinalStatus.phase === 'running';

  const canCompute = baselineFile !== null && followupFile !== null && !isRunning;

  // ── Stop blink when overlay is removed ────────────────────────────────────
  // If the user clears the overlay while blinking, we need to clean up the
  // interval; otherwise it keeps calling setShowLongitudinalOverlay on stale state.
  useEffect(() => {
    if (!hasLongitudinalOverlay && isBlinking) {
      if (blinkIntervalRef.current !== null) {
        clearInterval(blinkIntervalRef.current);
        blinkIntervalRef.current = null;
      }
      setIsBlinking(false);
    }
  }, [hasLongitudinalOverlay, isBlinking]);

  // ── Cleanup interval on unmount ───────────────────────────────────────────
  useEffect(() => {
    return () => {
      if (blinkIntervalRef.current !== null) {
        clearInterval(blinkIntervalRef.current);
      }
      if (pendingRafRef.current !== null) {
        cancelAnimationFrame(pendingRafRef.current);
      }
    };
  }, []);

  // ── Opacity slider: update React state immediately, debounce VTK call ─────
  const handleOpacityChange = useCallback(
    (rawValue: number) => {
      const factor = rawValue / 100;

      if (pendingRafRef.current !== null) {
        cancelAnimationFrame(pendingRafRef.current);
      }
      pendingRafRef.current = requestAnimationFrame(() => {
        pendingRafRef.current = null;
        setLongitudinalOpacity(factor);
      });
    },
    [setLongitudinalOpacity],
  );

  // ── Blink compare toggle ──────────────────────────────────────────────────
  const handleBlinkToggle = useCallback(() => {
    if (isBlinking) {
      // Stop blinking and restore visibility.
      if (blinkIntervalRef.current !== null) {
        clearInterval(blinkIntervalRef.current);
        blinkIntervalRef.current = null;
      }
      setShowLongitudinalOverlay(true);
      setIsBlinking(false);
    } else {
      // Start blinking at 500 ms cadence.
      blinkVisibleRef.current = true;
      blinkIntervalRef.current = setInterval(() => {
        blinkVisibleRef.current = !blinkVisibleRef.current;
        setShowLongitudinalOverlay(blinkVisibleRef.current);
      }, 500);
      setIsBlinking(true);
    }
  }, [isBlinking, setShowLongitudinalOverlay]);

  // ── Compute delta ─────────────────────────────────────────────────────────
  const handleCompute = useCallback(() => {
    if (!baselineFile || !followupFile) return;
    runLongitudinalDelta(baselineFile, followupFile, transformType);
  }, [baselineFile, followupFile, transformType, runLongitudinalDelta]);

  // ── Clear overlay ─────────────────────────────────────────────────────────
  const handleClear = useCallback(() => {
    // Stop blinking first so the interval does not outlive the bundle.
    if (blinkIntervalRef.current !== null) {
      clearInterval(blinkIntervalRef.current);
      blinkIntervalRef.current = null;
    }
    setIsBlinking(false);
    clearLongitudinalOverlay();
  }, [clearLongitudinalOverlay]);

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <section className="control-section longitudinal-panel">
      <h3 className="section-title">Longitudinal Analysis</h3>

      {/* ── Hint text ───────────────────────────────────────────────────── */}
      <p className="section-hint longitudinal-panel__hint">
        Co-register two sessions and visualise voxel-wise brain change.<br />
        <span className="longitudinal-panel__hint-red">Red</span> = growth &nbsp;·&nbsp;
        <span className="longitudinal-panel__hint-blue">Blue</span> = atrophy.
        Requires <code>pip install dipy</code>.
      </p>

      {/* ── Baseline file input ──────────────────────────────────────────── */}
      <div className="longitudinal-panel__file-row">
        <label className="longitudinal-panel__file-label" htmlFor="longitudinal-baseline">
          Baseline NIfTI
        </label>
        <input
          id="longitudinal-baseline"
          type="file"
          accept=".nii,.nii.gz"
          className="longitudinal-panel__file-input"
          onChange={(e) => setBaselineFile(e.target.files?.[0] ?? null)}
        />
        {baselineFile && (
          <span className="longitudinal-panel__file-name" title={baselineFile.name}>
            {baselineFile.name}
          </span>
        )}
      </div>

      {/* ── Follow-up file input ─────────────────────────────────────────── */}
      <div className="longitudinal-panel__file-row">
        <label className="longitudinal-panel__file-label" htmlFor="longitudinal-followup">
          Follow-up NIfTI
        </label>
        <input
          id="longitudinal-followup"
          type="file"
          accept=".nii,.nii.gz"
          className="longitudinal-panel__file-input"
          onChange={(e) => setFollowupFile(e.target.files?.[0] ?? null)}
        />
        {followupFile && (
          <span className="longitudinal-panel__file-name" title={followupFile.name}>
            {followupFile.name}
          </span>
        )}
      </div>

      {/* ── Registration strategy ───────────────────────────────────────── */}
      <div className="longitudinal-panel__xform-row">
        <label className="longitudinal-panel__xform-label" htmlFor="longitudinal-xform">
          Registration
        </label>
        <select
          id="longitudinal-xform"
          className="longitudinal-panel__xform-select"
          value={transformType}
          onChange={(e) => setTransformType(e.target.value as 'rigid' | 'affine')}
          disabled={isRunning}
        >
          <option value="rigid">Rigid (6 DOF, ~30–60 s)</option>
          <option value="affine">Affine (12 DOF, ~60–120 s)</option>
        </select>
      </div>

      {/* ── Status indicator ────────────────────────────────────────────── */}
      <div className="inference-status">
        <span className={`status-dot ${dotClass(longitudinalStatus)}`} />
        <span className="longitudinal-panel__status-text">
          {statusLabel(longitudinalStatus)}
        </span>
      </div>

      {/* ── Error detail ────────────────────────────────────────────────── */}
      {longitudinalStatus.phase === 'error' && (
        <p className="longitudinal-panel__error">{longitudinalStatus.message}</p>
      )}

      {/* ── Done: voxel counts badge ─────────────────────────────────────── */}
      {longitudinalStatus.phase === 'done' && (
        <p className="longitudinal-panel__counts">
          <span className="longitudinal-panel__counts-red" aria-hidden />
          {nPositive.toLocaleString()} growth&nbsp;&nbsp;
          <span className="longitudinal-panel__counts-blue" aria-hidden />
          {nNegative.toLocaleString()} atrophy
        </p>
      )}

      {/* ── Primary action button ────────────────────────────────────────── */}
      <button
        className="btn btn--primary longitudinal-panel__run-btn"
        disabled={!canCompute}
        onClick={handleCompute}
        title={
          !baselineFile ? 'Select a baseline scan first'
          : !followupFile ? 'Select a follow-up scan'
          : undefined
        }
      >
        {isRunning
          ? longitudinalStatus.phase === 'uploading' ? 'Uploading…' : 'Registering…'
          : hasLongitudinalOverlay
            ? 'Re-compute Delta'
            : 'Compute Delta'}
      </button>

      {/* ── Post-detection controls (visible after a successful run) ─────── */}
      {hasLongitudinalOverlay && (
        <div className="longitudinal-panel__overlay-controls">

          {/* Visibility toggle */}
          <div className="control-row control-row--toggle">
            <label className="control-label" htmlFor="longitudinal-show">
              Show overlay
            </label>
            <input
              id="longitudinal-show"
              type="checkbox"
              checked={showLongitudinalOverlay}
              onChange={(e) => {
                // Stop blinking if the user manually toggles visibility.
                if (isBlinking) {
                  if (blinkIntervalRef.current !== null) {
                    clearInterval(blinkIntervalRef.current);
                    blinkIntervalRef.current = null;
                  }
                  setIsBlinking(false);
                }
                setShowLongitudinalOverlay(e.target.checked);
              }}
            />
          </div>

          {/* Opacity slider */}
          <div className="control-row longitudinal-panel__opacity-row">
            <label className="control-label">
              <span>Opacity</span>
              <span className="control-value">
                {Math.round(longitudinalOpacity * 100)}%
              </span>
            </label>
            <input
              type="range"
              className="control-slider longitudinal-panel__opacity-slider"
              min={0}
              max={100}
              step={1}
              disabled={!showLongitudinalOverlay}
              value={Math.round(longitudinalOpacity * 100)}
              onChange={(e) => handleOpacityChange(Number(e.target.value))}
            />
          </div>

          {/* Blink compare button */}
          <button
            className={`btn longitudinal-panel__blink-btn${isBlinking ? ' longitudinal-panel__blink-btn--active' : ''}`}
            onClick={handleBlinkToggle}
            title="Rapidly alternate overlay visibility to check registration accuracy"
          >
            {isBlinking ? '■ Stop Blink' : '▶ Blink Compare'}
          </button>

          {/* Clear / remove overlay */}
          <button
            className="btn btn--ghost longitudinal-panel__clear-btn"
            onClick={handleClear}
          >
            Clear Delta
          </button>
        </div>
      )}
    </section>
  );
};

export default LongitudinalPanel;
