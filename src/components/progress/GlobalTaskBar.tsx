/**
 * GlobalTaskBar.tsx — Floating task-progress panel (bottom-right corner)
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Reads the live task registry from TaskProgressContext and renders a
 * floating card for every active or recently-completed task.
 *
 * WHEN IT IS VISIBLE
 * ──────────────────
 * The panel renders nothing when the task Map is empty (no active tasks),
 * so it does not occupy any space or DOM nodes during idle periods.
 *
 * CARD ANATOMY
 * ─────────────
 *   ┌────────────────────────────── card ──────────────────────────────────┐
 *   │  [label]                                              [phase icon]   │
 *   │  [progress bar track ─────────────────────────────────────────────]  │
 *   │  [pct text – uploading only]                                         │
 *   │  [message – done summary or error detail]                            │
 *   │  [Dismiss button – error only]                                       │
 *   └──────────────────────────────────────────────────────────────────────┘
 *
 * PROGRESS BAR BEHAVIOUR
 * ──────────────────────
 *   uploading → green determinate fill (width = uploadPct %)
 *   running   → animated shimmer (indeterminate)
 *   done      → full green bar + flash animation + auto-dismissed in 3 s
 *   error     → full red bar + error message + manual dismiss button
 *
 * CSS lives in ./progress.css.
 */

import type { FC } from 'react';
import { useTaskProgress } from '../../contexts/TaskProgressContext';
import type { TaskEntry } from '../../contexts/TaskProgressContext';

// ── Phase icon map ────────────────────────────────────────────────────────────

/**
 * Returns a single Unicode symbol representing the current task phase.
 * These are plain text characters, not emoji, so they render consistently
 * across Windows/macOS/Linux without font-dependency issues.
 *
 *   ↑  uploading — bytes going up to the server
 *   ⟳  running   — server is processing
 *   ✓  done      — completed successfully
 *   ✗  error     — failed
 */
function phaseIcon(phase: TaskEntry['phase']): string {
  switch (phase) {
    case 'uploading': return '↑';
    case 'running':   return '⟳';
    case 'done':      return '✓';
    case 'error':     return '✗';
  }
}

// ── Single task card ──────────────────────────────────────────────────────────

/**
 * Renders one task card inside the GlobalTaskBar.
 * Stateless — all data comes from the TaskEntry prop.
 */
function TaskCard({ entry, onDismiss }: { entry: TaskEntry; onDismiss: () => void }) {
  const { phase, label, uploadPct, message } = entry;

  return (
    <div className="task-card">

      {/* ── Header: label + phase icon ──────────────────────────────────── */}
      <div className="task-card__header">
        {/* Human-readable task name (e.g. "Segmentation — Subject A") */}
        <span className="task-card__label" title={label}>{label}</span>

        {/* Phase icon with per-phase colour modifier class */}
        <span className={`task-card__icon task-card__icon--${phase}`}>
          {phaseIcon(phase)}
        </span>
      </div>

      {/* ── Progress bar track ──────────────────────────────────────────── */}
      <div className="task-card__bar-track">

        {/* Determinate fill — shown for uploading, done, and error phases */}
        {phase !== 'running' && (
          <div
            className={`task-card__bar-fill task-card__bar-fill--${phase === 'done' ? 'done' : phase === 'error' ? 'error' : 'uploading'}`}
            style={
              /* For uploading, width reflects actual byte progress.
                 For done/error, width is forced to 100% via the CSS class. */
              phase === 'uploading' ? { width: `${uploadPct}%` } : undefined
            }
          />
        )}

        {/* Indeterminate shimmer — shown only during 'running' phase */}
        {phase === 'running' && (
          <div className="task-card__bar-shimmer" />
        )}
      </div>

      {/* ── Upload percentage text — shown only during uploading ─────────── */}
      {phase === 'uploading' && (
        <div className="task-card__pct">{uploadPct}%</div>
      )}

      {/* ── Message row — shown when there is a done summary or error ────── */}
      {message && (
        <div className={`task-card__message task-card__message--${phase === 'error' ? 'error' : 'done'}`}>
          {message}
        </div>
      )}

      {/* ── Dismiss button — only for error cards (done is auto-dismissed) ── */}
      {phase === 'error' && (
        <button
          className="task-card__dismiss"
          onClick={onDismiss}
          aria-label={`Dismiss error: ${label}`}
        >
          Dismiss
        </button>
      )}

      {/* Show a hint for active (non-error) tasks that they will auto-dismiss. */}
      {phase === 'running' && (
        // subtle secondary text — server is working
        <div className="task-card__pct">Processing…</div>
      )}
    </div>
  );
}

// ── GlobalTaskBar component ───────────────────────────────────────────────────

/**
 * Floating task panel fixed to the bottom-right corner of the viewport.
 *
 * Renders nothing when no tasks are registered (Map is empty).
 * Must be rendered inside <TaskProgressProvider>.
 */
const GlobalTaskBar: FC = () => {
  // Read the live task registry and the dismiss callback from context.
  const { tasks, dismissTask } = useTaskProgress();

  // Render nothing when no tasks are active — avoids empty DOM / z-index
  // artefacts while the app is idle.
  if (tasks.size === 0) return null;

  // Convert the Map to an array for rendering.  Map iteration order is
  // insertion order, so the most recently added task appears at the bottom.
  const taskList = Array.from(tasks.values());

  return (
    <div className="global-task-bar" role="status" aria-live="polite">
      {taskList.map((entry) => (
        <TaskCard
          key={entry.id}
          entry={entry}
          onDismiss={() => dismissTask(entry.id)}
        />
      ))}
    </div>
  );
};

export default GlobalTaskBar;
