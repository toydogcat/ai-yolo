import { defineConfig } from 'vite';

export default defineConfig({
  base: '/ai-yolo/',
  server: {
    port: 5173,
    strictPort: false,
    host: true,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    }
  },
  worker: {
    format: 'es'
  },
  build: {
    target: 'esnext',
    rollupOptions: {
      input: {
        main: 'index.html',
        ocr: 'ocr.html',
        hand: 'hand.html'
      }
    }
  }
});

