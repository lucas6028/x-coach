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
      // Impure edges that can't run under jsdom (kept in sync with codecov.yml `ignore`):
      // sixSevenDetector.ts is the MediaPipe/WASM/WebGL/canvas glue, and SixSeven.tsx is the
      // thin camera + requestAnimationFrame page shell. Every decision branch (visibility gate,
      // gesture reasoning, rep counting) lives in src/lib/sixseven/* and is fully covered there.
      exclude: [
        "src/main.tsx",
        "src/test/**",
        "src/components/sixseven/sixSevenDetector.ts",
        "src/pages/SixSeven.tsx",
      ],
      thresholds: {
        lines: 70,
        functions: 70,
        branches: 60,
        statements: 68,
      },
    },
  },
});
