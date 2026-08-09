import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy /api to the FastAPI backend so the SPA can use same-origin relative URLs.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Escape hatch for Docker: bind mounts on Docker Desktop/WSL2 often swallow the
    // inotify events HMR relies on. docker-compose.dev.yml sets VITE_USE_POLLING=true;
    // outside Docker this stays undefined and native watching is used.
    watch: process.env.VITE_USE_POLLING
      ? { usePolling: true, interval: 300 }
      : undefined,
    // LIFF endpoints must be HTTPS: in dev the app is exposed through an ngrok tunnel
    // (see docs/line-login-liff-setup.md). Vite blocks unknown Host headers by default;
    // allow every domain family ngrok hands out (free-tier tunnels come from both
    // ngrok-free.app AND ngrok-free.dev, legacy ones from ngrok.io) so the LIFF browser
    // can reach the dev server. A leading dot matches all subdomains.
    allowedHosts: [
      ".ngrok-free.app",
      ".ngrok-free.dev",
      ".ngrok.app",
      ".ngrok.dev",
      ".ngrok.io",
    ],
    proxy: {
      "/api": {
        // In Docker the API is another service on the compose network, not localhost —
        // docker-compose.dev.yml sets VITE_PROXY_TARGET=http://backend:8000.
        target: process.env.VITE_PROXY_TARGET || "http://localhost:8000",
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
        "src/components/ninja/ninjaDetector.ts",
        "src/pages/FruitNinja.tsx",
        "src/components/webslinger/webSlingerDetector.ts",
        "src/pages/WebSlinger.tsx",
        // Same impure boundary: the LIFF diag's camera→MediaPipe chain probe needs a real
        // camera + WASM + WebGL.
        "src/lib/poseProbe.ts",
        // The worker entry itself needs browser ImageBitmap/OffscreenCanvas plus MediaPipe GPU.
        // The runner is unit-tested with a mock worker and remains included in coverage.
        "src/workers/poseInference.worker.ts",
        "src/components/poseLandmarker.ts",
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
