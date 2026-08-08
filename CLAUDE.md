# CLAUDE.md

Guidance for Claude Code in this repository. This file is a ROUTING HUB: core facts
live here; everything long lives in the linked files. Machine-wide working rules
(delegation, verification, escalation) live in `~/.claude/harness/` — read
`~/.claude/harness/README.md` before any task that spans more than 3 steps or 3 files.

## What this is

x-coach is a research prototype for **explainable exercise coaching feedback**, now with
a working web app on top of the research pipelines:

- **ML/perception library** in `src/` (pose, video, knowledge, rehab24, fit3d, egoexo)
  with thin CLI entry points in `scripts/` — the research foundation.
- **Backend** in `backend/` — FastAPI app (`backend/app/main.py`); routers: analyze,
  analyses, chat, conversations, knowledge, videos; Supabase for auth/history;
  chat via OpenRouter (spec: `specs/llm-chat-spec.md`).
- **Frontend** in `frontend/` — React 18 + Vite + TypeScript + Tailwind, Supabase client.

Research framing: `project-overview.md` / `研究計畫.md`. Experiment results: `notes/`.
KG schema docs: `docs/kg-schema-generalization.md`, `docs/movement-kg-expansion-plan.md`.

## Environment & commands (Windows — this machine has NO `python` on PATH)

- **Python interpreter:** always `.venv\Scripts\python.exe` (from repo root).
  NEVER `source .venv/bin/activate` (POSIX-only, fails here), never bare `python`/`pip`.
  Use `.venv\Scripts\python.exe -m pip install ...` for deps.
  `.venv-mmpose\` is a second venv ONLY for the `--runtime mmpose` pose path.
- **Backend/ML tests:** `.venv\Scripts\python.exe -m pytest tests/` (always scope to
  `tests/`; never bare `pytest`).
  Single case: append `tests/test_x.py::Class::test_name`.
- **Backend coverage gate (CI enforces 95%):**
  `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
- **Frontend:** ALL yarn/vitest commands must run with cwd = `frontend/` (the Bash and
  PowerShell tools share one cwd — a stray `cd` elsewhere mass-fails vitest).
  `yarn dev` / `yarn test` (vitest run) / `yarn test:coverage` / `yarn build`.
- **CI** (`.github/workflows/ci.yml`): backend pytest (py 3.11/3.12) + coverage gate;
  frontend `yarn test:coverage`. Match it locally before claiming tests pass.
- **Docker** (web app only, not the research pipelines): `docker compose up --build` →
  app :8080, API :8000. Add `-f docker-compose.dev.yml` for hot reload (Vite :5173).
  `requirements-docker.txt` is the web subset of `requirements.txt` (no torch/transformers)
  and must be updated by hand alongside it. Full notes: `docs/docker.md`.
- `GOOGLE_API_KEY` is only needed for Gemini KG extraction (`src/knowledge/extract_kg.py`).
  Everything else, including RAG, runs fully offline.
- **Kaggle:** drive the Kaggle CLI via `uv` in the shell. Do NOT use the `kaggle` MCP
  tools — most are broken vs. the SDK (see memory `kaggle-mcp-buggy-use-cli`).
- Always launch Claude Code sessions from the repo root (not `frontend/`), otherwise the
  session gets a separate, empty memory directory.

## Architecture: scripts vs src

**`scripts/` are thin CLI entry points; all logic lives in `src/`.** Each script
bootstraps the repo root onto `sys.path` and calls into the matching `src/` module.
Modules import by absolute package path (`from src.pose... import ...`), so **run
everything from the repository root**. Workflow commands per area:
`scripts/{pose,video,knowledge,rehab24,egoexo}/README.md`.

Cross-pipeline link: `pose_rule_detector` enriches detected faults with retrieved
knowledge via `src/knowledge/graph_retrieval.py` (stores under `data/rag/vector_db` and
`data/kg`); pass `--no-retrieval` to run rules standalone.

## Data layout

Pipelines read/write under `data/`, resolved relative to the repo root:
`data/Squat/{Unlabeled,Labeled}_Dataset/` (videos, pose JSON, features, splits, labels),
`data/REHAB24-6/`, `data/kg/` (graphml + canonical mapping), `data/rag/{docs,vector_db}/`.

## Style & conventions

- ML modules favor a local-first, dependency-light style (stdlib + numpy/networkx),
  pure-function helpers unit-tested in isolation.
- Tests are `unittest.TestCase` classes under `tests/` (backend + ML) and vitest files
  under `frontend/src/test/` (frontend). New code gets tests in the matching suite.
- `AGENTS.md` is a pointer to this file — never duplicate content into it.

## graphify

Project knowledge graph at `graphify-out/` (graph.json + GRAPH_REPORT.md; no wiki/).

- Use `graphify query "<question>"` for ARCHITECTURE questions (how subsystems relate,
  what calls what across files). For a known symbol or string, use Grep directly — it is
  cheaper and faster.
- `graphify path "<A>" "<B>"` for relationships; `graphify explain "<concept>"` for one
  concept; `GRAPH_REPORT.md` only for broad architecture review.
- After modifying code, run `graphify update .` (AST-only, no API cost). Note the graph
  is scoped to the project proper (memory `graphify-graph-scoped`).
