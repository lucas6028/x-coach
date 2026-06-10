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
  App.tsx                state + responsive layout
  components/            Sidebar, Header, VideoPanel, SkeletonOverlay, Timeline,
                         MetricsCards, ReasoningLog, KnowledgeGraphWidget,
                         LibraryPicker, UploadDropzone, ChatInput
```
