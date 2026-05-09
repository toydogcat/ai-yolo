import { defineConfig } from 'vite';

export default defineConfig({
  base: '/ai-yolo/',
  server: {
    port: 5173,
    strictPort: false,
    host: true
  },
  worker: {
    format: 'es'
  }
});

