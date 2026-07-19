/**
 * FileUpload.tsx — Drag-and-drop + click-to-browse file picker
 * ─────────────────────────────────────────────────────────────
 *
 * Accepts all neuroimaging formats:
 *   .nii / .nii.gz  — MRI structural / functional / PET
 *   .edf / .bdf     — EEG, iEEG, MEG (European Data Format / BioSemi)
 *   .snirf          — fNIRS (HDF5-based Shared NIRS Format)
 *
 * Calls `onFile` for any dropped or selected file; format validation is
 * handled downstream by BidsRouter and the plugin's processFile().
 */

import { useRef, useState, useCallback, type DragEvent, type ChangeEvent } from 'react';

// ── Props ─────────────────────────────────────────────────────────────────────

interface FileUploadProps {
  /** Called with the full set of dropped/selected files. */
  onFiles:          (files: File[]) => void;
  loading:          boolean;
  error:            string | null;
  loadedFileName?:  string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// ── Component ─────────────────────────────────────────────────────────────────

export default function FileUpload({ onFiles, loading, error, loadedFileName }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // ── Drag events ───────────────────────────────────────────────────────────

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      if (loading) return;
      const files = Array.from(e.dataTransfer.files);
      if (files.length > 0) onFiles(files);
    },
    [loading, onFiles],
  );

  // ── Click-to-browse ───────────────────────────────────────────────────────

  const handleInputChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files ?? []);
      if (files.length > 0) onFiles(files);
      e.target.value = '';
    },
    [onFiles],
  );

  const handleClick = useCallback(() => {
    if (!loading) inputRef.current?.click();
  }, [loading]);

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="file-upload-section">
      <div
        className={[
          'drop-zone',
          isDragging ? 'drop-zone--active' : '',
          loading    ? 'drop-zone--loading' : '',
        ].filter(Boolean).join(' ')}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        role="button"
        tabIndex={0}
        aria-label="Upload neuroimaging file"
        onKeyDown={(e) => e.key === 'Enter' && handleClick()}
      >
        {loading ? (
          <div className="drop-zone__loading">
            <div className="spinner" aria-label="Processing" />
            <span>Parsing file…</span>
          </div>
        ) : (
          <div className="drop-zone__idle">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
              <path d="M12 2C9.5 2 7.5 3.5 7 5.5C5.3 5.8 4 7.2 4 9c0 1.1.4 2.1 1.1 2.8C4.4 12.4 4 13.2 4 14c0 2.2 1.8 4 4 4h8c2.2 0 4-1.8 4-4 0-.8-.4-1.6-1.1-2.2C19.6 11.1 20 10.1 20 9c0-1.8-1.3-3.2-3-3.5C16.5 3.5 14.5 2 12 2z" />
            </svg>
            <p className="drop-zone__primary">Drop a neuroimaging file</p>
            <p className="drop-zone__secondary">
              or <span className="drop-zone__link">click to browse</span>
            </p>
            <p className="drop-zone__hint">.nii · .nii.gz · .edf · .bdf · .fif (drop all split parts together) · .snirf · .vhdr+.eeg+.vmrk · .con+.mrk+.pos</p>
          </div>
        )}
      </div>

      {/* Hidden file input */}
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".nii,.nii.gz,.edf,.bdf,.vhdr,.eeg,.vmrk,.fif,.meg4,.con,.mrk,.pos,.snirf,application/octet-stream"
        style={{ display: 'none' }}
        onChange={handleInputChange}
      />

      {/* Loaded file badge */}
      {loadedFileName && !loading && !error && (
        <div className="file-badge" title={loadedFileName}>
          <span className="file-badge__dot" />
          <span className="file-badge__name">{loadedFileName}</span>
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="upload-error" role="alert">
          <strong>Error:</strong> {error}
        </div>
      )}
    </div>
  );
}
