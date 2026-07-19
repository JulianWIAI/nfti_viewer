/**
 * main.tsx — Application entry point
 * ─────────────────────────────────────
 * This is the ONLY entry point for the SPA. Vite reads this file through the
 * `<script type="module" src="/src/main.tsx">` tag in index.html.
 *
 * Nothing complex lives here; the purpose of keeping this file minimal is to
 * make the module graph easy to follow: index.html → main.tsx → App.tsx → …
 *
 * StrictMode is intentionally left ON for development. It double-invokes
 * effects and renders to catch side-effect bugs early. vtk.js is robust to
 * this because we guard all vtk.js setup with `if (!ctx)` checks.
 */

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';

// Mount the React tree onto the <div id="root"> defined in index.html.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
