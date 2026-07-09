# x-coach backend (FastAPI)

A thin web service that wraps the existing `src/` pipeline (pose extraction → rule-based
biomechanics → KG/RAG retrieval). It contains **no biomechanics logic of its own** — every
endpoint delegates to functions under `src/`.

## Run

From the **repository root** (so `from src... import` resolves), with the venv active:

```bash
source .venv/bin/activate
pip install -r requirements.txt          # first time (adds fastapi/uvicorn/python-multipart + supabase/PyJWT)
cp .env.example .env                      # then fill in the Supabase keys (see Auth below)
uvicorn backend.app.main:app --reload --port 8000
```

## Auth & persistence (Supabase)

Auth and history are **optional**: with no Supabase env set the server still runs — uploads are
analyzed but nothing is saved (public demo), and the history endpoints return 503/401.

Configure via `.env` at the repo root (gitignored; see `.env.example`):
`SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_JWT_SECRET`. The frontend authenticates with
Supabase Auth and sends `Authorization: Bearer <access_token>`; the backend verifies it locally
(HS256) and forwards it to Postgres so **RLS** scopes every row to its owner. Apply the schema
first — see `db/migrations/` (and its README for why migrations don't live under `supabase/`).

## User-video object storage & quotas (Cloudflare R2)

Uploaded videos need durable storage for a real deployment — the app container's disk is ephemeral,
so a restart would drop every upload and history replay would 404. When the `R2_*` vars are set, a
signed-in user's upload is pushed to a **private** R2 bucket (S3-compatible) and streamed back via a
short-lived **presigned GET** URL (`/api/video-file/{id}` 307-redirects to it); with them unset the
backend keeps videos on the local runtime disk (fine for dev and the anonymous demo). R2 is chosen
for **zero egress fees** — video streaming is egress-heavy. `boto3` is the client (lazily imported).

| Var | Purpose |
|-----|---------|
| `R2_ACCOUNT_ID` | Cloudflare account id (derives the endpoint) |
| `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` | R2 API token (Object Read & Write) |
| `R2_BUCKET` | private bucket name (do **not** make it public) |
| `R2_ENDPOINT` | optional explicit endpoint override |
| `R2_URL_TTL_S` | presigned-URL lifetime, seconds (default 300) |

**Guardrails** (env-overridable; see `config.py`) cap abuse and bound spend for a small deployment —
single upload **≤ 100 MB / 60 s**, and **≤ 30 videos & 1 GB per user** (a 40-user demo is ≤ 40 GB,
~$0.60/mo on R2). The frontend pre-checks size/duration before uploading; the backend re-checks both
(413 for size, 422 for duration, 507 when a user is over quota) — the client checks are UX only.

| Var | Default |
|-----|---------|
| `XCOACH_MAX_UPLOAD_BYTES` | `104857600` (100 MB) |
| `XCOACH_MAX_UPLOAD_DURATION_S` | `60` |
| `XCOACH_USER_VIDEO_QUOTA_COUNT` | `30` |
| `XCOACH_USER_STORAGE_QUOTA_BYTES` | `1073741824` (1 GB) |

## Conversational coaching (LLM)

`/api/chat` is the one endpoint that calls an LLM (the analysis pipeline itself is fully offline).
It answers **only** from the analysis's detected faults + retrieved KG/RAG knowledge — grounding is
enforced in the server-built system prompt — and the key never reaches the browser. Without
`LLM_API_KEY` the endpoint returns 503 and the frontend shows a disabled chat.

The transport speaks the plain **OpenAI-compatible** chat-completions dialect, so four env vars
(named `LLM_*` rather than after any one provider) point it at any compatible provider — no code
change. OpenRouter-only request extras (attribution headers, latency provider-routing for the
follow-up chips) are auto-suppressed when the base URL isn't OpenRouter's.

| Var | Purpose |
|-----|---------|
| `LLM_API_KEY` | provider key; unset ⇒ `/api/chat` disabled (503) |
| `LLM_MODELS` | comma-separated picker shown in Settings; **first = default** |
| `LLM_BASE_URL` | provider endpoint (default OpenRouter) |
| `LLM_FOLLOWUP_MODEL` | fast model for the follow-up chips (a separate call); blank ⇒ reuse default |

**OpenRouter (default)** — key at <https://openrouter.ai/keys>, `vendor/model` slugs:

```
LLM_MODELS=deepseek/deepseek-v4-flash,xiaomi/mimo-v2.5,minimax/minimax-m3,tencent/hy3-preview
```

**NVIDIA NIM (build.nvidia.com)** — OpenAI-compatible, so switching is a pure `.env` change; key
(prefix `nvapi-`) at <https://build.nvidia.com>:

```
LLM_API_KEY=nvapi-...
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODELS=deepseek-ai/deepseek-v3.2,meta/llama-3.3-70b-instruct,qwen/qwen3-235b-a22b,nvidia/llama-3.3-nemotron-super-49b-v1
LLM_FOLLOWUP_MODEL=openai/gpt-oss-120b
```

NIM caveats: the free tier is a dev/trial tier (~40 req/min, credit-capped, no throughput SLA) —
fine for a research prototype/demo, not open public traffic. Prefer **instruct** models — a
reasoning model that emits its chain-of-thought in `reasoning_content` (not `delta.content`) reads
as a long silence before the answer streams. If replies get truncated, NIM's default `max_tokens`
can be low; there's no env for it yet, so add one if you hit it.

## Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET  | `/api/health` | — | liveness + which data stores are present + `auth_configured` |
| POST | `/api/analyze` | optional | upload a squat video → extract → rule detection (+retrieval); persists + returns `analysis_id` when authenticated |
| GET  | `/api/analyses?limit=&offset=` | required | the caller's analysis history (newest first) |
| GET  | `/api/analyses/{analysis_id}` | required | one of the caller's analyses (full `result`) |
| GET  | `/api/videos?limit=&offset=&fault=` | — | list precomputed labeled clips (faulty clips first) |
| GET  | `/api/analysis/{video_id}` | — | precomputed analysis for a library clip (retrieval enriched on demand) |
| GET  | `/api/pose/{video_id}` | — | slim 33-landmark overlay block |
| GET  | `/api/video-file/{video_id}` | — | stream the source mp4 (supports HTTP Range / seeking) |
| GET  | `/api/knowledge/graph?query=` | — | knowledge-graph subgraph for the KG widget |
| GET  | `/api/knowledge/rag?query=` | — | ranked RAG snippets |
| POST | `/api/chat` | required | grounded LLM follow-up chat over an analysis (503 if `LLM_API_KEY` unset, 502 on upstream error) |

Interactive docs: <http://localhost:8000/docs>.

## Layout

```
backend/app/
  main.py            FastAPI app, CORS, router wiring, /api/health
  config.py          repo-root paths + runtime/upload dirs
  settings.py        env-driven secrets (Supabase URL / anon key / JWT secret)
  auth.py            Supabase JWT verification + get_current_user / get_optional_user
  services/
    analysis.py      live upload: process_video + detect_pose_rules_from_json + slim pose block
    library.py       list/load precomputed labeled videos + ground-truth labels
    knowledge.py     wrappers over retrieve_graph_context / query_vector_db
    store.py         user-scoped Supabase persistence (videos + analyses, RLS-enforced)
  routers/           analyze.py, analyses.py, videos.py, knowledge.py
```

Uploaded videos and their derived pose JSON land in `data/runtime/` (gitignored).

## Tests

The backend has a self-contained suite at `tests/test_backend.py` that covers every section
(`config`, `settings`, `auth`, the four `services/`, the four `routers/`, and `main`) at
**100% line + branch coverage**. The heavy ML pipeline (`src.pose.*`), knowledge retrieval
(`src.knowledge.*`), and the Supabase client are mocked, so the suite runs fast and needs no
MediaPipe/torch, `data/` fixtures, or a live Supabase project.

Run from the **repository root** (scope to `tests/` — a bare `pytest` collects the stale
root-level `test_metadata.py`):

```bash
source .venv/bin/activate
python -m pytest tests/test_backend.py            # backend API suite (106 tests)
python -m pytest tests/test_backend.py -k Library # one class
```

Two further backend suites complement it:

- `tests/test_analyze_endpoint.py` — upload-endpoint concurrency / semaphore internals.
- `tests/test_backend_analysis.py` — **integration** against the real `data/` tree + KG/RAG
  stores; it is skipped in CI and only passes when that data is present locally.

### Coverage

`scripts/run_backend_coverage.py` runs the suite under coverage.py and prints the percentage
for `backend/app`:

```bash
pip install coverage                              # or: pip install -r requirements.txt
python scripts/run_backend_coverage.py            # terminal report + TOTAL %
python scripts/run_backend_coverage.py --html     # also writes htmlcov/index.html
python scripts/run_backend_coverage.py --fail-under 95   # non-zero exit if below 95%
```

### CI

`.github/workflows/ci.yml` runs on every push/PR to `main` (Python 3.11 + 3.12). It installs
the lean `requirements-ci.txt` (no torch/opencv/mediapipe — they're mocked), runs the
deterministic test subset, and enforces backend coverage with `--fail-under 95`. The heavy
(`test_mediapipe_smoothing`, `test_rtmpose_pose_extraction`, `test_videomae_video_classifier`)
and data-dependent (`test_backend_analysis`) modules are skipped there.

## Notes

- Live `/api/analyze` runs MediaPipe (`model_complexity=2`) on CPU — expect ~15–25s for a ~3s clip.
- The analysis response is the `detect_pose_rules_from_json` dict with `frame_metrics` dropped and
  a compact `pose` block (x, y, visibility only) attached for the skeleton overlay.
- The analysis pipeline is fully offline (no API key needed). The **conversational-coaching**
  layer (`/api/chat`) is the one exception — it calls an LLM provider; see § Conversational coaching.
