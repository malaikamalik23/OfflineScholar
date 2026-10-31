import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// In development the UI runs on :5173 and forwards /api to the FastAPI backend on :8000.
// In the desktop app FastAPI serves the built UI itself, so /api is the same origin.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
});
