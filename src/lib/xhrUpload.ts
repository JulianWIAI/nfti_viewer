/**
 * xhrUpload.ts — XHR-based multipart POST with upload-progress callbacks
 * ─────────────────────────────────────────────────────────────────────────
 *
 * The browser Fetch API does not expose upload progress events; it only
 * resolves when the full response is available.  XMLHttpRequest, on the
 * other hand, fires `progress` events on its `.upload` object during the
 * outgoing byte transfer, enabling per-byte progress tracking.
 *
 * This module wraps XHR into a Promise<T> so callers can use async/await
 * while still receiving granular upload-progress callbacks.
 *
 * Compatible with all modern browsers including Edge on Windows.
 * No OS-specific code; no path manipulation — pure browser Web APIs only.
 */

// ── Public option types ────────────────────────────────────────────────────────

/**
 * Options accepted by xhrPost.
 */
export interface XhrPostOptions<T> {
  // T is used as the return type of the promise, not in the options shape.
  // The type parameter is intentionally placed here for co-location with
  // the function signature.
  _phantom?: T;
  /** Absolute or relative URL to POST to. */
  url: string;

  /** FormData payload (files + text fields). */
  form: FormData;

  /**
   * Called during the outgoing byte transfer with a percentage 0–100.
   * Fires multiple times as chunks are sent.  Reaches 100 when the last
   * byte has left the client — the server may still be processing at that
   * point (see onUploadComplete).
   */
  onUploadProgress?: (pct: number) => void;

  /**
   * Called exactly once when the XHR upload phase completes, i.e. when all
   * bytes have been transmitted and the server has acknowledged receipt.
   * The response has NOT yet arrived — this is the transition from
   * "uploading" to "server processing" phase.
   */
  onUploadComplete?: () => void;

  /**
   * Optional AbortSignal.  When aborted, the in-flight XHR is aborted and
   * the returned Promise rejects with a DOMException named 'AbortError'.
   */
  signal?: AbortSignal;
}

// ── Implementation ─────────────────────────────────────────────────────────────

/**
 * POST a FormData payload via XMLHttpRequest and return the parsed JSON
 * response body as type T.
 *
 * Upload progress is reported via onUploadProgress(pct) during the outgoing
 * byte transfer.  onUploadComplete() fires once upload is fully done.
 *
 * On non-2xx HTTP status the Promise rejects with an Error whose message
 * matches the FastAPI `detail` field when the response body is JSON, or
 * falls back to "HTTP <status>" otherwise.
 *
 * @template T  Expected shape of the successful JSON response.
 */
export function xhrPost<T>(options: XhrPostOptions<T>): Promise<T> {
  // Destructure options for clarity inside the executor.
  const { url, form, onUploadProgress, onUploadComplete, signal } = options;

  return new Promise<T>((resolve, reject) => {
    // Create the XHR object.
    const xhr = new XMLHttpRequest();

    // ── Upload progress ──────────────────────────────────────────────────────
    // xhr.upload is a XMLHttpRequestUpload object that fires ProgressEvent
    // instances during the outgoing byte transfer.
    if (onUploadProgress) {
      xhr.upload.addEventListener('progress', (event: ProgressEvent) => {
        // event.lengthComputable is true when the browser knows the total size.
        if (event.lengthComputable && event.total > 0) {
          // Calculate percentage; clamp to [0, 100] to be defensive.
          const pct = Math.min(100, Math.round((event.loaded / event.total) * 100));
          onUploadProgress(pct);
        }
      });
    }

    // ── Upload complete (last byte sent, server now processing) ──────────────
    if (onUploadComplete) {
      // The 'load' event on xhr.upload fires when all bytes have been sent
      // AND the server has acknowledged receipt.  The response body is NOT
      // yet available at this point.
      xhr.upload.addEventListener('load', () => {
        onUploadComplete();
      });
    }

    // ── Response received ────────────────────────────────────────────────────
    xhr.addEventListener('load', () => {
      // status === 0 means the request was blocked (CORS, network error).
      if (xhr.status === 0) {
        reject(new Error('Network error: request was blocked or CORS failed'));
        return;
      }

      // HTTP 2xx = success.
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          // Parse the JSON response body.
          const data = JSON.parse(xhr.responseText) as T;
          resolve(data);
        } catch {
          // The server returned a non-JSON body on a 2xx status — unusual but
          // possible if the backend sends plain text on success.
          reject(new Error(`Response parse error: expected JSON but got: ${xhr.responseText.slice(0, 200)}`));
        }
        return;
      }

      // HTTP error (4xx / 5xx): try to surface the FastAPI `detail` message.
      let detail = `HTTP ${xhr.status} ${xhr.statusText}`;
      try {
        const errJson = JSON.parse(xhr.responseText) as { detail?: unknown };
        if (errJson.detail !== undefined) {
          detail = typeof errJson.detail === 'string'
            ? errJson.detail
            : JSON.stringify(errJson.detail);
        }
      } catch {
        // Response is not JSON — use the status text we already have.
      }
      reject(new Error(detail));
    });

    // ── Network / CORS errors ────────────────────────────────────────────────
    // The 'error' event fires when no HTTP response was received at all
    // (e.g. DNS failure, refused connection, offline).
    xhr.addEventListener('error', () => {
      reject(new Error('Network error: failed to reach the server'));
    });

    // ── Abort ────────────────────────────────────────────────────────────────
    xhr.addEventListener('abort', () => {
      reject(new DOMException('XHR upload aborted', 'AbortError'));
    });

    // ── Open and send ────────────────────────────────────────────────────────
    // open() must be called before setting headers or sending.
    xhr.open('POST', url);

    // Do NOT set Content-Type manually — the browser must set it to
    // multipart/form-data with the correct boundary string automatically.

    // Wire the AbortSignal AFTER open() so abort() works correctly.
    if (signal) {
      // If the signal is already aborted before we start, abort immediately.
      if (signal.aborted) {
        xhr.abort();
        reject(new DOMException('XHR upload aborted', 'AbortError'));
        return;
      }
      // Otherwise listen for the abort event.
      signal.addEventListener('abort', () => xhr.abort(), { once: true });
    }

    // Send the FormData payload.  The browser will set the correct
    // multipart/form-data Content-Type header with boundary automatically.
    xhr.send(form);
  });
}