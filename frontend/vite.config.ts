import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy /api to the FastAPI backend so the SPA can use same-origin relative URLs.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    coverage: {
      provider: "v8",
      reporter: ["text", "html", "lcov"],
      include: ["src/**/*.{ts,tsx}"],
      // poseDetector.ts is the MediaPipe/WASM/WebGL boundary — it needs a real camera
      // and GPU, neither of which exist under jsdom, so it can't be meaningfully
      // unit-tested. The game's logic lives in src/lib/game/* and is fully covered there.
      exclude: ["src/main.tsx", "src/test/**", "src/components/game/poseDetector.ts"],
      thresholds: {
        lines: 70,
        functions: 70,
        branches: 60,
        statements: 68,
      },
    },
  },
});
