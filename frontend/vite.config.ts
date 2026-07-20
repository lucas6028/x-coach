import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy /api to the FastAPI backend so the SPA can use same-origin relative URLs.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
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
      // the *Detector.ts files are the MediaPipe/WASM/WebGL/canvas glue, and SixSeven.tsx is a
      // thin camera + requestAnimationFrame page shell. Every decision branch lives in the
      // matching src/lib/* modules and is fully covered there.
      exclude: [
        "src/main.tsx",
        "src/test/**",
        "src/components/blast/blastDetector.ts",
        "src/pages/MemeBlast.tsx",
        "src/components/sixseven/sixSevenDetector.ts",
        "src/pages/SixSeven.tsx",
        "src/components/ninja/ninjaDetector.ts",
        "src/pages/FruitNinja.tsx",
        // Same impure boundary: the LIFF diag's camera→MediaPipe chain probe needs a real
        // camera + WASM + WebGL.
        "src/lib/poseProbe.ts",
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
