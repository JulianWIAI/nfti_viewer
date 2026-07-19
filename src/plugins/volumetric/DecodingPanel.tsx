/**
 * DecodingPanel.tsx — Neural decoding controls + timeline chart
 * ──────────────────────────────────────────────────────────────
 *
 * Rendered inside VolumetricControls as a self-contained `<section>`.
 * Reads shared state from VolumetricContext (provided by VolumetricViewer)
 * rather than accepting props, so no prop-drilling is needed between the
 * viewer and the controls panel.
 *
 * STRUCTURE
 * ──────────
 *
 *   [section] Neural Decoding (MVPA)
 *   │
 *   ├── File pickers (.vhdr / .eeg|.dat / .vmrk)
 *   │     Hidden <input type="file"> triggered by styled buttons.
 *   │     Filename is displayed after selection so the user can confirm
 *   │     they have the right file without expanding a long path.
 *   │
 *   ├── Parameter inputs (Class A, Class B, tmin, tmax, n_folds)
 *   │     Compact number inputs inside a 2-column grid.
 *   │
 *   ├── Status row (idle / running spinner / error / done summary)
 *   │
 *   ├── [Run Decoding] button
 *   │     Disabled while running or if not all 3 files are selected.
 *   │
 *   └── (shown only after successful run)
 *       ├── TimeScrubber — plays through decoded time points
 *       │     onScrub → setCurrentTimeIndex (context) so the playhead in
 *       │     DecodingTimeline moves in lock-step with slider / playback.
 *       └── DecodingTimeline — SVG chart of AUC over time
 *             currentTimeIndex ← from context (driven by TimeScrubber)
 *             onSeek → setCurrentTimeIndex (click-to-seek in chart)
 *
 * FILE PICKER PATTERN
 * ─────────────────────
 * The browser's native file picker button is nearly impossible to style
 * consistently.  Instead, each picker uses:
 *   1. A hidden `<input type="file">` with a React ref.
 *   2. A visible styled `<button>` that calls `ref.current.click()`.
 *   3. A `<span>` that shows the selected filename (or a placeholder).
 *
 * FILE STATE IS LOCAL
 * ─────────────────────
 * The three File objects (vhdrFile, eegFile, vmrkFile) and the parameter
 * values (classA, classB, tmin, tmax, nFolds) live in local state here.
 * They do not belong in VolumetricContext because they are ephemeral form
 * state used only to assemble the DecodingRequest payload — once the run
 * starts the files are no longer needed.  The result (decodingData) lives
 * in the context.
 *
 * CSS CLASSES  (defined in App.css under "── DecodingPanel ──")
 * ──────────────────────────────────────────────────────────────
 *   .decoding-panel__file-row     Row: [icon btn] [filename] per file
 *   .decoding-panel__file-btn     Styled trigger button (ext badge)
 *   .decoding-panel__file-name    Filename span (truncated with ellipsis)
 *   .decoding-panel__params       2-column grid for numeric params
 *   .decoding-panel__param-field  Label + input pair
 *   .decoding-panel__status       Status row between params and run btn
 *   .decoding-panel__error        Red error message
 *   .decoding-panel__meta         Green success summary (peak AUC + latency)
 *   .decoding-panel__divider      Thin rule before the scrubber block
 */

import {
  useRef,
  useState,
  useCallback,
  type JSX,
  type ChangeEvent,
} from 'react';
import { useVolumetricContext } from './VolumetricViewer';
import TimeScrubber from '../../components/TimeScrubber';
import DecodingTimeline from '../../components/DecodingTimeline';
import type { DecodingRequest } from '../../services/decodingApi';

// ── Component ─────────────────────────────────────────────────────────────────

export default function DecodingPanel(): JSX.Element {
  // Pull shared state from the context that VolumetricViewer provides.
  const {
    decodingData,
    decodingStatus,
    runNeuralDecoding,
    currentTimeIndex,
    setCurrentTimeIndex,
  } = useVolumetricContext();

  // ── Local form state (ephemeral; not in context) ──────────────────────────

  // The three BrainVision files — all null until the user selects them.
  const [vhdrFile, setVhdrFile] = useState<File | null>(null);
  const [eegFile,  setEegFile]  = useState<File | null>(null);
  const [vmrkFile, setVmrkFile] = useState<File | null>(null);

  // Pipeline parameters — defaulting to the FastAPI endpoint's defaults.
  const [classA,  setClassA]  = useState<number>(1);
  const [classB,  setClassB]  = useState<number>(2);
  const [tmin,    setTmin]    = useState<number>(-0.2);
  const [tmax,    setTmax]    = useState<number>(0.8);
  const [nFolds,  setNFolds]  = useState<number>(5);

  // ── Hidden file input refs — triggered programmatically by styled buttons ──

  const vhdrInputRef = useRef<HTMLInputElement>(null);
  const eegInputRef  = useRef<HTMLInputElement>(null);
  const vmrkInputRef = useRef<HTMLInputElement>(null);

  // ── Derived state ─────────────────────────────────────────────────────────

  const isRunning = decodingStatus.phase === 'running';

  // All three files must be selected before the run button is enabled.
  const canRun = vhdrFile !== null && eegFile !== null && vmrkFile !== null && !isRunning;

  // The scrubber's frame count equals the number of decoded time points.
  // Falls back to 1 (TimeScrubber requires frameCount ≥ 1) before a run.
  const frameCount = decodingData ? decodingData.times.length : 1;

  // ── Handlers ──────────────────────────────────────────────────────────────

  /** Assemble the DecodingRequest and fire the pipeline via context. */
  const handleRun = useCallback(() => {
    if (!vhdrFile || !eegFile || !vmrkFile) return;

    const req: DecodingRequest = {
      vhdrFile,
      eegFile,
      vmrkFile,
      classA,
      classB,
      tmin,
      tmax,
      nFolds,
    };

    // runNeuralDecoding (from context) is the async handler in VolumetricViewer.
    // It sets decodingStatus to 'running', awaits the fetch, then lands in
    // 'done' or 'error'.  We intentionally don't await here — the void return
    // is intentional; status updates propagate via the context.
    void runNeuralDecoding(req);
  }, [vhdrFile, eegFile, vmrkFile, classA, classB, tmin, tmax, nFolds, runNeuralDecoding]);

  /**
   * Called by TimeScrubber on every scrub event (RAF-throttled, ~60 Hz max).
   * Propagates the frame index to the context so DecodingTimeline's playhead
   * moves in lock-step.
   */
  const handleScrub = useCallback((idx: number) => {
    setCurrentTimeIndex(idx);
  }, [setCurrentTimeIndex]);

  /**
   * Called by DecodingTimeline when the user clicks a point on the chart.
   * Propagates the time index back so the TimeScrubber thumb jumps to that
   * position.
   *
   * Because both components read `currentTimeIndex` from the same context,
   * setting it once here is sufficient to synchronise both controls.
   */
  const handleSeek = useCallback((idx: number) => {
    setCurrentTimeIndex(idx);
  }, [setCurrentTimeIndex]);

  /** Generic file-change handler that validates a required extension. */
  function makeFileHandler(
    set: (f: File | null) => void,
  ): (e: ChangeEvent<HTMLInputElement>) => void {
    return (e) => set(e.target.files?.[0] ?? null);
  }

  // ── Status display helpers ────────────────────────────────────────────────

  /** Returns the human-readable status line for the current pipeline phase. */
  function statusLabel(): string {
    switch (decodingStatus.phase) {
      case 'idle':    return 'Ready';
      case 'running': return 'Running MVPA…';
      case 'error':   return `Error: ${decodingStatus.message}`;
      case 'done':
        return (
          `Peak AUC ${decodingStatus.peakScore.toFixed(3)}` +
          ` @ ${Math.round(decodingStatus.peakTimeS * 1000)} ms` +
          ` · ${Math.round(decodingStatus.durationMs)} ms`
        );
    }
  }

  /** CSS modifier class on the status dot (mirrors SynthSeg pattern). */
  const dotClass =
    decodingStatus.phase === 'done'    ? 'status-dot--done'
    : decodingStatus.phase === 'error' ? 'status-dot--error'
    : decodingStatus.phase === 'running' ? 'status-dot--running'
    :                                      'status-dot--idle';

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <section className="control-section">
      <h3 className="section-title">Neural Decoding (MVPA)</h3>

      {/* ── File pickers ────────────────────────────────────────────────── */}
      {/* Each row: styled trigger button + truncated filename display.     */}

      <FilePicker
        label=".vhdr"
        accept=".vhdr"
        file={vhdrFile}
        inputRef={vhdrInputRef}
        onChange={makeFileHandler(setVhdrFile)}
      />
      <FilePicker
        label=".eeg"
        accept=".eeg,.dat"
        file={eegFile}
        inputRef={eegInputRef}
        onChange={makeFileHandler(setEegFile)}
      />
      <FilePicker
        label=".vmrk"
        accept=".vmrk"
        file={vmrkFile}
        inputRef={vmrkInputRef}
        onChange={makeFileHandler(setVmrkFile)}
      />

      {/* ── Parameters ──────────────────────────────────────────────────── */}

      <div className="decoding-panel__params">
        <NumberField
          label="Class A"
          value={classA}
          min={1}
          onChange={setClassA}
        />
        <NumberField
          label="Class B"
          value={classB}
          min={1}
          onChange={setClassB}
        />
        <NumberField
          label="tmin (s)"
          value={tmin}
          step={0.1}
          onChange={setTmin}
        />
        <NumberField
          label="tmax (s)"
          value={tmax}
          step={0.1}
          onChange={setTmax}
        />
        <NumberField
          label="k-folds"
          value={nFolds}
          min={2}
          max={20}
          onChange={setNFolds}
        />
      </div>

      {/* ── Status row ──────────────────────────────────────────────────── */}

      <div className="inference-status decoding-panel__status">
        <span className={`status-dot ${dotClass}`} />
        <span
          className={
            decodingStatus.phase === 'error' ? 'decoding-panel__error'
            : decodingStatus.phase === 'done' ? 'decoding-panel__meta'
            : undefined
          }
        >
          {statusLabel()}
        </span>
      </div>

      {/* ── Run button ──────────────────────────────────────────────────── */}

      <button
        className="btn btn--primary"
        style={{ width: '100%', marginTop: 4 }}
        disabled={!canRun}
        onClick={handleRun}
        type="button"
      >
        {isRunning ? 'Running…' : 'Run Decoding'}
      </button>

      {/* ── Playback + Timeline ─────────────────────────────────────────── */}
      {/*                                                                   */}
      {/* Shown only after at least one successful decode so there is data  */}
      {/* for the chart and a meaningful frame count for the scrubber.      */}

      {decodingData && (
        <>
          <hr className="decoding-panel__divider" />

          {/* TimeScrubber: drives currentTimeIndex in context via onScrub.
              The DecodingTimeline playhead then reads that same index.    */}
          <TimeScrubber
            frameCount={frameCount}
            onScrub={handleScrub}
            fps={30}
          />

          {/* DecodingTimeline: SVG chart with playhead + click-to-seek.   */}
          <DecodingTimeline
            times={decodingData.times}
            scores={decodingData.scores}
            scoresStd={decodingData.scoresStd}
            currentTimeIndex={currentTimeIndex}
            onSeek={handleSeek}
          />

          {/* Dataset summary — compact line below the chart. */}
          <p className="decoding-panel__summary">
            {decodingData.nEpochs} epochs
            &thinsp;·&thinsp;{decodingData.nChannels} ch
            &thinsp;·&thinsp;A={decodingData.nEpochsClassA} / B={decodingData.nEpochsClassB}
          </p>
        </>
      )}
    </section>
  );
}

// ── Private sub-components ────────────────────────────────────────────────────

/**
 * FilePicker — hidden `<input type="file">` with a styled trigger button.
 *
 * The native file input is hidden (width: 0 / height: 0 / opacity: 0) to
 * avoid cross-browser styling limitations.  Clicking the visible button
 * calls `inputRef.current.click()` to open the OS file dialog.
 *
 * Props:
 *   label    — short extension label displayed on the trigger button
 *   accept   — value passed to the input's `accept` attribute
 *   file     — currently selected File (or null)
 *   inputRef — ref attached to the hidden input
 *   onChange — called when the user selects a file
 */
interface FilePickerProps {
  label:    string;
  accept:   string;
  file:     File | null;
  // React's DOM ref type: RefObject<T> where T is the element, not T | null —
  // null is already embedded inside RefObject as `current: T | null`.
  inputRef: React.RefObject<HTMLInputElement>;
  onChange: (e: ChangeEvent<HTMLInputElement>) => void;
}

function FilePicker({ label, accept, file, inputRef, onChange }: FilePickerProps): JSX.Element {
  return (
    <div className="decoding-panel__file-row">
      {/* Hidden native input */}
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={onChange}
        style={{ display: 'none' }}
        tabIndex={-1}
        aria-hidden
      />
      {/* Visible styled trigger */}
      <button
        type="button"
        className="decoding-panel__file-btn"
        onClick={() => inputRef.current?.click()}
        aria-label={`Select ${label} file`}
      >
        {label}
      </button>
      {/* Selected filename — truncated with CSS ellipsis */}
      <span className="decoding-panel__file-name" title={file?.name ?? ''}>
        {file ? file.name : 'No file selected'}
      </span>
    </div>
  );
}

/**
 * NumberField — compact label + number input pair for the parameter grid.
 *
 * Renders a column (label above, input below) that fits naturally in the
 * 2-column `.decoding-panel__params` grid.
 */
interface NumberFieldProps {
  label:    string;
  value:    number;
  min?:     number;
  max?:     number;
  step?:    number;
  onChange: (v: number) => void;
}

function NumberField({ label, value, min, max, step = 1, onChange }: NumberFieldProps): JSX.Element {
  return (
    <label className="decoding-panel__param-field">
      <span className="decoding-panel__param-label">{label}</span>
      <input
        type="number"
        className="decoding-panel__param-input"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}
