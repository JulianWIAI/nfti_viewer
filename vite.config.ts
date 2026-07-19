/**
 * vite.config.ts — Build configuration for the NIfTI Viewer SPA
 * ─────────────────────────────────────────────────────────────────
 *
 * Three non-trivial concerns handled here:
 *
 * 1. Web Workers
 *    Vite supports ES-module workers natively. Setting `worker.format: 'es'`
 *    means the worker bundle is a true ES module — no legacy importScripts(),
 *    full tree-shaking, proper dynamic imports. The worker is instantiated in
 *    useNiftiWorker.ts with `new Worker(new URL(...), { type: 'module' })`.
 *
 * 2. WebAssembly (for ONNX Runtime Web)
 *    onnxruntime-web ships multi-threaded and single-threaded WASM blobs. If
 *    Vite pre-bundles the package with esbuild, the dynamic `new URL(...)` calls
 *    that load the .wasm files get rewritten and break. We exclude it from
 *    optimizeDeps so Vite serves it as raw ESM and the URLs stay intact.
 *    `viteStaticCopy` copies the WASM blobs from node_modules → public/wasm/
 *    at build time so they are reachable at runtime.
 *
 * 3. GitHub Pages deployment
 *    GitHub Pages serves the repo at /<repo-name>/. The `base` option prefixes
 *    all generated asset paths. Change this to match your repository name.
 */

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import fs from 'node:fs';
import path from 'node:path';

export default defineConfig({
  plugins: [
    react(),

    // Copy ONNX Runtime WASM binaries and their .mjs companion modules.
    // The .mjs files (e.g. ort-wasm-simd-threaded.jsep.mjs) are dynamically
    // imported at runtime by the WASM backend even when numThreads=1; they
    // must sit next to the .wasm files at BASE_URL + 'wasm/'.
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*.wasm',
          dest: 'wasm',
        },
        {
          src: 'node_modules/onnxruntime-web/dist/ort-wasm*.mjs',
          dest: 'wasm',
        },
      ],
    }),

    // Dev-server middleware: serve ort WASM companion .mjs files directly from
    // node_modules, BEFORE Vite's transform middleware runs.
    //
    // WHY: When onnxruntime-web (excluded from optimizeDeps) does a dynamic
    // import() of e.g. 'ort-wasm-simd-threaded.jsep.mjs', Vite appends
    // '?import' to the URL for module-graph tracking.  The static-copy
    // plugin's sirv middleware runs AFTER Vite's transform step, so Vite
    // tries to resolve the file through its own graph first — and fails
    // because the file is not in src/.
    //
    // This pre-middleware intercepts those requests, strips '?import', and
    // streams the raw file directly from node_modules with the correct MIME
    // type, bypassing Vite's transform entirely.
    {
      name: 'ort-wasm-mjs-dev',
      configureServer(server) {
        const ortDist = path.join(process.cwd(), 'node_modules/onnxruntime-web/dist');
        const wasmPrefix = '/nfti_viewer/wasm/';

        server.middlewares.use((req, res, next) => {
          const cleanUrl = (req.url ?? '').split('?')[0];
          if (cleanUrl.startsWith(wasmPrefix) && cleanUrl.endsWith('.mjs')) {
            const filename = cleanUrl.slice(wasmPrefix.length);
            const fullPath = path.join(ortDist, filename);
            if (fs.existsSync(fullPath)) {
              res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
              res.setHeader('Cache-Control', 'no-cache');
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              fs.createReadStream(fullPath).pipe(res as any);
              return;
            }
          }
          next();
        });
      },
    },
  ],

  // ── GitHub Pages base path ──────────────────────────────────────────────
  // Change '/nfti_viewer/' to '/' for a custom domain or local testing.
  base: '/nfti_viewer/',

  // ── Web Worker bundling ─────────────────────────────────────────────────
  worker: {
    // 'es' keeps tree-shaking and dynamic imports working inside the worker.
    format: 'es',
  },

  // ── Dependency pre-bundling ─────────────────────────────────────────────
  optimizeDeps: {
    // onnxruntime-web must NOT be pre-bundled: it uses dynamic new URL(...)
    // calls to locate its .wasm files, which esbuild rewrites and breaks.
    // h5wasm embeds a ~2 MB WASM blob inline in hdf5_util.js; esbuild cannot
    // process a file that large without hitting memory limits, and the embedded
    // binary must be kept intact.
    exclude: ['onnxruntime-web', 'h5wasm'],

    // vtk.js has 400+ ESM modules. If we exclude it, Vite dev serves them as
    // individual HTTP requests, which hits browser connection limits and causes
    // silent load failures (white page). Pre-bundling with esbuild collapses
    // them into one fast-loading chunk. esbuild preserves side-effect imports
    // (like Rendering/Profiles/All) because vtk.js has no "sideEffects: false"
    // in its package.json, so the profile factory registrations are safe.
    //
    // nifti-reader-js and pako are CommonJS; pre-bundling converts them to ESM.
    include: ['@kitware/vtk.js', 'nifti-reader-js', 'pako'],
  },

  // ── Production build ────────────────────────────────────────────────────
  build: {
    // esnext allows native BigInt, optional chaining, etc. — no transpilation.
    target: 'esnext',

    // vtk.js + ONNX Runtime push well past the default 500 kB warning level.
    chunkSizeWarningLimit: 12_000,

    rollupOptions: {
      output: {
        // Split into named async chunks so the browser can cache them
        // independently across deploys. The viewer code is tiny; the libs
        // are large but change infrequently.
        //
        // Vite 8 / rolldown requires manualChunks to be a FUNCTION, not an
        // object. The function receives each module id and returns the desired
        // chunk name, or undefined to keep the module in the default chunk.
        manualChunks(id: string) {
          if (id.includes('@kitware/vtk.js')) return 'vendor-vtk';
          if (id.includes('onnxruntime-web'))  return 'vendor-onnx';
          if (id.includes('node_modules/react') || id.includes('node_modules/react-dom')) return 'vendor-react';
          return undefined;
        },
      },
    },
  },

  // ── Development server ──────────────────────────────────────────────────
  server: {
    // Proxy /api/* to the FastAPI backend so the browser never makes a
    // cross-origin fetch.  This eliminates CORS and COEP (require-corp)
    // issues for all API calls — the browser sees only one origin (5173).
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
    headers: {
      // SharedArrayBuffer (used by ONNX multi-thread WASM) requires both of
      // these headers. They're injected by the dev server only; for GitHub
      // Pages production you need the coi-serviceworker workaround.
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
