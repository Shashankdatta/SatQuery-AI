import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Backend URL bound by Vercel Services or local fallback
const backendTarget = process.env.BACKEND_URL || 'http://127.0.0.1:8000';

// https://vitejs.dev/config/
export default defineConfig({
  base: '/',
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/health': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/analyze': {
        target: backendTarget,
        changeOrigin: true,
      },
    },
  },
})
