# x-coach frontend (React + Vite + TypeScript)

The X-Coach dashboard — a real implementation of the `demo/index.html` mock-up, wired to the
FastAPI backend. Package manager is **yarn**.

## Run

Start the backend first (see `backend/README.md`), then:

```bash
cd frontend
yarn install      # first time
yarn dev          # http://localhost:5173  (proxies /api -> http://localhost:8000)
```

Build / type-check: `yarn build`.

## Test

Unit/component tests run on **Vitest** + **React Testing Library** in a `jsdom`
environment (config lives in `vite.config.ts`; global stubs for
`matchMedia`/`localStorage`/`canvas`/`IntersectionObserver`/media APIs live in
`src/test/setup.ts`).

```bash
yarn test            # run once
yarn test:watch      # watch mode
yarn test:coverage   # run with a V8 coverage report (text + html + lcov)
```

Coverage thresholds are enforced in `vite.config.ts` (statements 68%, lines/functions 70%,
branches 60%); the HTML report is written to `frontend/coverage/` (git-ignored) and `lcov.info`
is uploaded by CI. Tests cover the API client, i18n/theme/pose/format helpers, every component,
the `App` shell, and the landing page.

## What it does

- **Upload** a squat video (primary flow) or pick from the **sample library** (instant demo).
- **Skeleton overlay**: a `<canvas>` draws the 33-landmark MediaPipe skeleton synced to the video
  via `requestAnimationFrame`; limbs implicated by a fault active at the current frame turn red.
- **Timeline**: red segments mark fault windows (opacity ∝ severity); click to seek.
- **Coaching feedback**: one card per detected fault with the rule evidence plus GraphRAG context
  (likely cause / injury risk / correction cue) and a RAG snippet — all deterministic, no LLM.
- **Knowledge-graph widget**: radial view of the active fault and its KG neighbours.
- Responsive: three-pane on desktop, single column with Coaching/Knowledge tabs on mobile (PWA-ready).

The chat box is intentionally disabled — conversational coaching arrives with the LLM layer.

## Layout

```
src/
  api.ts                 typed fetch client + interfaces mirroring the backend
  lib/pose.ts            33-landmark topology + fault→landmark groups
  lib/format.ts          time/severity formatting helpers
  lib/i18n.tsx           en / zh-Hant dictionaries + t() provider and label helpers
  lib/theme.ts           light/dark/system theme controller
  App.tsx                state + responsive layout
  main.tsx               router entry (landing "/" + app "/app")
  components/            Sidebar, Header, VideoPanel, SkeletonOverlay, Timeline,
                         MetricsCards, ReasoningLog, KnowledgeGraphWidget,
                         LibraryPicker, UploadDropzone, ChatInput,
                         ThemeToggle, LanguageToggle, DemoIntro, ResizeHandle
  landing/               marketing landing page (Landing, Reveal, PosePreview, …)
  test/                  Vitest setup, shared fixtures/render helpers, *.test.ts(x)
```
