/**
 * TaskProgressContext.tsx — Global task registry for multi-phase pipeline progress
 * ─────────────────────────────────────────────────────────────────────────────────
 *
 * Provides a single Context that tracks every async neuroimaging task
 * (segmentation, anomaly detection, longitudinal delta, neural decoding)
 * through four phases:
 *
 *   uploading → bytes are being sent to the server
 *   running   → server is processing (upload complete, awaiting response)
 *   done      → pipeline completed successfully (auto-dismissed after 3 s)
 *   error     → pipeline failed (user must manually dismiss)
 *
 * The Map<string, TaskEntry> state stores tasks by a caller-supplied string ID
 * so multiple concurrent tasks (e.g. Subject A and Subject B segmentation) can
 * coexist without collision.
 *
 * REACT STATE REQUIREMENT
 * ─────────────────────────
 * React compares state by reference.  A Map mutation (map.set()) does NOT
 * trigger a re-render because the Map reference is unchanged.  We work around
 * this by always spreading the old Map into a new Map on each update:
 *   new Map([...prev, [id, entry]])
 * which creates a new Map reference, causing React to re-render correctly.
 *
 * USAGE
 * ──────
 *   // In your root component:
 *   <TaskProgressProvider>
 *     <App />
 *     <GlobalTaskBar />
 *   </TaskProgressProvider>
 *
 *   // In any descendant:
 *   const { registerTask, setUploadProgress, setRunning, completeTask, failTask } = useTaskProgress();
 */

import {
  createContext,
  useContext,
  useState,
  useCallback,
  type FC,
  type ReactNode,
} from 'react';

// ── Task phase discriminated union ────────────────────────────────────────────

/**
 * The four lifecycle phases of an async pipeline task.
 *   uploading — bytes are being transferred to the server
 *   running   — server is computing (all bytes received)
 *   done      — pipeline finished successfully
 *   error     — pipeline failed; message contains the reason
 */
export type TaskPhase = 'uploading' | 'running' | 'done' | 'error';

// ── Task entry shape ──────────────────────────────────────────────────────────

/**
 * A single task tracked by the global task registry.
 *
 * id        — unique caller-supplied string (e.g. 'seg-a', 'anomaly')
 * label     — human-readable task name shown in the GlobalTaskBar card
 * phase     — current lifecycle phase (drives bar appearance and icons)
 * uploadPct — byte-level upload progress 0–100 (meaningful only in 'uploading')
 * message   — optional summary on 'done', or error description on 'error'
 */
export interface TaskEntry {
  id:        string;
  label:     string;
  phase:     TaskPhase;
  uploadPct: number;
  message?:  string;
}

// ── Context value shape ───────────────────────────────────────────────────────

/**
 * Functions and data exposed by TaskProgressContext.
 *
 * tasks           — live snapshot of all tasks (Map by id)
 * registerTask    — create a new task entry in 'uploading' phase
 * setUploadProgress — update uploadPct for a task in 'uploading' phase
 * setRunning      — advance a task from 'uploading' → 'running'
 * completeTask    — advance a task to 'done' + schedule auto-dismiss in 3 s
 * failTask        — advance a task to 'error' with a message
 * dismissTask     — remove a task from the registry immediately
 */
interface TaskProgressContextValue {
  tasks:             Map<string, TaskEntry>;
  registerTask:      (id: string, label: string) => void;
  setUploadProgress: (id: string, pct: number) => void;
  setRunning:        (id: string) => void;
  completeTask:      (id: string, summary?: string) => void;
  failTask:          (id: string, message: string) => void;
  dismissTask:       (id: string) => void;
}

// ── Context creation ──────────────────────────────────────────────────────────

/**
 * The actual React context object.  Default value is null — callers must
 * be wrapped in <TaskProgressProvider> or useTaskProgress() will throw.
 */
const TaskProgressContext = createContext<TaskProgressContextValue | null>(null);

// ── Provider component ────────────────────────────────────────────────────────

/**
 * Wraps children with the task registry.  Place this at the root of the app
 * (or at least above VolumetricViewer and GlobalTaskBar).
 */
export const TaskProgressProvider: FC<{ children: ReactNode }> = ({ children }) => {
  /**
   * Primary state: a Map from task id to TaskEntry.
   *
   * We store it as a React state value so that mutations (via the updater
   * functions below) trigger re-renders.  Each update creates a NEW Map
   * instance so React's reference-equality check fires correctly.
   */
  const [tasks, setTasks] = useState<Map<string, TaskEntry>>(new Map());

  // ── Internal helper: produce an updated Map from the previous one ──────────

  /**
   * Apply a patch to an existing task entry, creating a new Map so React
   * can detect the change.  If the task id does not exist, does nothing.
   */
  const patchTask = useCallback(
    (id: string, patch: Partial<TaskEntry>) => {
      setTasks((prev) => {
        const existing = prev.get(id);
        // Guard: ignore updates for tasks that no longer exist (e.g. after dismiss).
        if (!existing) return prev;
        // Build the updated entry by merging the patch into the existing entry.
        const updated: TaskEntry = { ...existing, ...patch };
        // Create a new Map with the updated entry so React detects the change.
        return new Map([...prev, [id, updated]]);
      });
    },
    [],
  );

  // ── registerTask ──────────────────────────────────────────────────────────

  /**
   * Create a new task entry in the 'uploading' phase.
   * If a task with the same id already exists it is replaced (idempotent for
   * re-runs — e.g. the user clicks "Run Segmentation" a second time).
   *
   * @param id    Unique task identifier string (e.g. 'seg-a')
   * @param label Human-readable task name for the GlobalTaskBar card
   */
  const registerTask = useCallback((id: string, label: string) => {
    // Build the initial entry.
    const entry: TaskEntry = {
      id,
      label,
      phase:     'uploading',
      uploadPct: 0,
    };
    // Insert (or replace) the entry; create a new Map to trigger re-render.
    setTasks((prev) => new Map([...prev, [id, entry]]));
  }, []);

  // ── setUploadProgress ─────────────────────────────────────────────────────

  /**
   * Update the upload byte-progress for a task that is in 'uploading' phase.
   *
   * @param id  Task identifier
   * @param pct Upload percentage 0–100
   */
  const setUploadProgress = useCallback((id: string, pct: number) => {
    // Clamp pct to [0, 100] defensively.
    patchTask(id, { uploadPct: Math.min(100, Math.max(0, pct)) });
  }, [patchTask]);

  // ── setRunning ────────────────────────────────────────────────────────────

  /**
   * Advance a task from 'uploading' → 'running'.
   * Called when the last upload byte has been sent and the server is now
   * processing (i.e. onUploadComplete callback from xhrUpload).
   *
   * @param id  Task identifier
   */
  const setRunning = useCallback((id: string) => {
    patchTask(id, { phase: 'running', uploadPct: 100 });
  }, [patchTask]);

  // ── completeTask ──────────────────────────────────────────────────────────

  /**
   * Advance a task to 'done' and schedule auto-dismiss after 3000 ms.
   *
   * @param id      Task identifier
   * @param summary Optional human-readable result summary (e.g. "42 structures")
   */
  const completeTask = useCallback((id: string, summary?: string) => {
    // Patch the task to 'done' with the optional summary message.
    patchTask(id, { phase: 'done', uploadPct: 100, message: summary });

    // Auto-dismiss after 3 seconds so the GlobalTaskBar does not accumulate
    // stale completed entries indefinitely.
    setTimeout(() => {
      // Remove the task from the Map entirely.
      setTasks((prev) => {
        const next = new Map(prev);
        next.delete(id);
        return next;
      });
    }, 3000);
  }, [patchTask]);

  // ── failTask ──────────────────────────────────────────────────────────────

  /**
   * Advance a task to 'error' with a human-readable message.
   * Error tasks are NOT auto-dismissed — the user must click the dismiss button
   * in GlobalTaskBar to acknowledge and remove the error card.
   *
   * @param id      Task identifier
   * @param message Error message (backend detail string or Error.message)
   */
  const failTask = useCallback((id: string, message: string) => {
    patchTask(id, { phase: 'error', message });
  }, [patchTask]);

  // ── dismissTask ───────────────────────────────────────────────────────────

  /**
   * Remove a task from the registry immediately.
   * Used by the dismiss button on error cards in GlobalTaskBar.
   *
   * @param id  Task identifier
   */
  const dismissTask = useCallback((id: string) => {
    setTasks((prev) => {
      const next = new Map(prev);
      next.delete(id);
      return next;
    });
  }, []);

  // ── Context value ─────────────────────────────────────────────────────────

  const value: TaskProgressContextValue = {
    tasks,
    registerTask,
    setUploadProgress,
    setRunning,
    completeTask,
    failTask,
    dismissTask,
  };

  return (
    <TaskProgressContext.Provider value={value}>
      {children}
    </TaskProgressContext.Provider>
  );
};

// ── Consumer hook ─────────────────────────────────────────────────────────────

/**
 * Returns the TaskProgressContextValue for the nearest TaskProgressProvider
 * ancestor.  Throws a descriptive error if called outside a provider so
 * misconfigured component trees are caught early.
 */
export function useTaskProgress(): TaskProgressContextValue {
  const ctx = useContext(TaskProgressContext);
  if (!ctx) {
    throw new Error('useTaskProgress must be called inside <TaskProgressProvider>');
  }
  return ctx;
}