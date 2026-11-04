import { defineConfig } from 'vite';

export default defineConfig({
  // GitHub Pages serves this project at /<repo>/, not at the domain root.
  base: '/AI-RESUME-ANALYZER/',
  // onnxruntime-web ships .wasm binaries that must not be inlined or renamed.
  assetsInclude: ['**/*.wasm'],
  optimizeDeps: {
    exclude: ['@huggingface/transformers'],
  },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 1500,
  },
});
