import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/health': {
        target: 'https://satquery-ai-5njv.onrender.com',
        changeOrigin: true,
        secure: false,
      },
      '/analyze': {
        target: 'https://satquery-ai-5njv.onrender.com',
        changeOrigin: true,
        secure: false,
      },
    },
  },
})
