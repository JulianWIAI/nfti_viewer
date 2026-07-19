/// <reference types="vite/client" />

/**
 * Augment Vite's ImportMeta so TypeScript knows about the custom env variables
 * we use throughout the app (BASE_URL is provided by Vite out of the box).
 *
 * Add project-specific VITE_* variables here as you introduce them, e.g.:
 *   readonly VITE_ONNX_MODEL_PATH: string;
 */
interface ImportMetaEnv {
  readonly BASE_URL: string;
  readonly MODE: string;
  readonly DEV: boolean;
  readonly PROD: boolean;
  readonly SSR: boolean;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

// jsfive has no published @types package
declare module 'jsfive';
