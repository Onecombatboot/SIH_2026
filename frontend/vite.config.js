import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const BACKEND = 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    // Expose the dev server on the LAN so a phone can open it.
    host: true,
    proxy: {
      '^/(visioncane|neuromap)/(ws|config)': { target: BACKEND, changeOrigin: true, ws: true },
    },
  },
})
