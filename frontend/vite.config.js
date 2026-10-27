import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/health': {
        target: 'https://honor-proceed-olympus-eleven.trycloudflare.com',
        changeOrigin: true,
        secure: false,
      },
      '/analyze': {
        target: 'https://honor-proceed-olympus-eleven.trycloudflare.com',
        changeOrigin: true,
        secure: false,
      },
    },
  },
})
