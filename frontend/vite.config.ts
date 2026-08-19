import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
// Proxy target for the Python voice-agent backend (Supertonic-100m/app.py).
// Keeps the frontend's API calls same-origin in dev so no CORS setup is needed
// locally; override with VITE_API_PROXY_TARGET if the backend runs elsewhere.
const BACKEND_TARGET = process.env.VITE_API_PROXY_TARGET ?? "http://127.0.0.1:7860"

export default defineConfig({
  // The production build is served by the Python backend under /app (see
  // Supertonic-100m/app.py's serve_frontend route), not from the domain root, so
  // built asset URLs must be prefixed accordingly or the JS/CSS 404 and the page
  // renders blank. The Vite dev server itself still runs at "/".
  base: process.env.VITE_BASE_PATH ?? "/app/",
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: BACKEND_TARGET,
        changeOrigin: true,
      },
    },
  },
})
