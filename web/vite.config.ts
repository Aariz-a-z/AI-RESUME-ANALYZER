import { defineConfig } from 'vite';

export default defineConfig({
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
