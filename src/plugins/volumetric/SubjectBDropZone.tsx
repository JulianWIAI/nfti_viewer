/**
 * SubjectBDropZone.tsx — Drag-and-drop target for Subject B NIfTI file.
 *
 * Renders a bordered dashed zone inside the controls sidebar when Subject A
 * is loaded but Subject B has not yet been added.  Accepts .nii and .nii.gz
 * files via:
 *   1. Drag-and-drop directly onto the zone.
 *   2. Click → hidden <input type="file"> picker.
 *
 * On a valid file selection it calls onFile(file) — the parent (VolumetricViewer)
 * then parses the NIfTI in a worker and initialises the second VTK pane.
 *
 * Props
 * ─────
 *   onFile   (file: File) => void — fired with the raw File object.
 */

import { useState, useRef, useCallback, type FC, type DragEvent } from 'react';

interface Props {
  /** Called with the accepted File when the user completes a valid drop or pick. */
  onFile: (file: File) => void;
}

const SubjectBDropZone: FC<Props> = ({ onFile }) => {
  const [dragging, setDragging] = useState(false);
  const [error,    setError]    = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  /** Validate and forward a File; clear any previous error on success. */
  const accept = useCallback((file: File) => {
    const n = file.name.toLowerCase();
    if (!n.endsWith('.nii') && !n.endsWith('.nii.gz')) {
      setError('Please drop a .nii or .nii.gz file.');
      return;
    }
    setError(null);
    onFile(file);
  }, [onFile]);

  const onDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  const onDragLeave = useCallback(() => {
    setDragging(false);
  }, []);

  const onDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) accept(file);
  }, [accept]);

  const onInputChange = useCallback(() => {
    const file = inputRef.current?.files?.[0];
    if (file) accept(file);
  }, [accept]);

  return (
    <section className="control-section">
      <h3 className="section-title">Compare — Subject B</h3>

      {/* Hidden file input — triggered by click on the drop zone */}
      <input
        ref={inputRef}
        type="file"
        accept=".nii,.nii.gz"
        style={{ display: 'none' }}
        onChange={onInputChange}
      />

      {/* Drop zone */}
      <div
        className={`subjectb-dropzone${dragging ? ' subjectb-dropzone--active' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        aria-label="Drop Subject B NIfTI file here or click to browse"
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click(); }}
      >
        <span className="subjectb-dropzone__icon">⊕</span>
        <span className="subjectb-dropzone__label">
          Drop Subject B scan here<br />
          <span className="subjectb-dropzone__hint">
            (.nii / .nii.gz) or click to browse
          </span>
        </span>
      </div>

      {/* Validation error */}
      {error && (
        <p style={{ color: 'var(--accent-red, #e05252)', fontSize: 10, marginTop: 4 }}>
          {error}
        </p>
      )}

      {/* Informational note */}
      <p style={{ fontSize: 10, color: '#555', marginTop: 6, lineHeight: 1.5 }}>
        Both brains will be segmented concurrently. No registration is applied —
        each is shown in its own native voxel space.
      </p>
    </section>
  );
};

export default SubjectBDropZone;
