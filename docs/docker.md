# Running x-coach in Docker

Two containers: `backend` (FastAPI + the pose/rules/retrieval pipeline) and `frontend`
(the React SPA — nginx in the default stack, the Vite dev server in the dev override).
Postgres/auth is Supabase-hosted, so there is no database container.

Requires **Docker Compose v2.24+** (the stack uses the `env_file: {path, required}` long
form and the `!override` merge tag). Check with `docker compose version`.

## Quick start

```bash
cp .env.example .env                    # backend secrets — every one is optional
docker compose up --build
```

- App: <http://localhost:8080>
- API + OpenAPI docs: <http://localhost:8000/docs>
- Health/config readout: <http://localhost:8000/api/health>

Nothing in `.env` is required to boot. Each unconfigured subsystem degrades to a clearly
reported state instead of failing — `/api/health` tells you which: `auth_configured`,
`chat_configured`, `line_login_configured`, `storage_configured`, plus a `stores` map for
the labeled-video, detection, KG and RAG directories.

### Hot reload while developing

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

- App (Vite, HMR): <http://localhost:5173>
- API (uvicorn `--reload`): <http://localhost:8000>

`backend/` and `src/` are bind-mounted read-only and watched by the reloader; `frontend/`
is bind-mounted with an anonymous volume shadowing `node_modules`, so the container keeps
its Linux-native install (esbuild and rollup ship platform-specific binaries — a host
`node_modules` from Windows or macOS will not run).

### File watching uses polling

Both watchers are set to poll rather than rely on inotify, because inotify events do not
reliably cross a bind mount.

`WATCHFILES_FORCE_POLLING=true` (backend) is what gets `uvicorn --reload` firing at all.
Without it the reloader picked up none of several host edits. The mount is live either way
(a host append is visible to `stat` inside the container immediately); watchfiles' inotify
backend simply never receives the event. The failure is silent — it logs `Will watch for
changes in these directories` and then does nothing, which reads like a code problem rather
than a watcher problem. It is not the `:ro` flag: re-tested read-write, same result.

**Backend reload is not fully reliable in a sandboxed Docker.** Verified where bind mounts
surface inside the container as a `fakeowner` FUSE mount: with polling on, `watchfiles`
itself detected every edit, but the uvicorn reloader layered on it acted only
intermittently — some reloads landed ~11s after the write, other edits were missed
entirely across several minutes. If an edit does not take, fall back to:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml restart backend
```

A stock Docker Engine on Linux may not have this problem; if inotify works on your host,
set the flag to `"false"` for instant reloads. Polling is the default because it is the
setting that works in the widest range of environments, and its cost is negligible here —
the watched trees are ~140 files, and `--reload-dir backend --reload-dir src` keeps the
sweep off the mounted `data/`.

The frontend side held up throughout: `VITE_USE_POLLING=true` is insurance rather than a
fix, since HMR was measured to fire on the same mounts *without* it — chokidar copes where
watchfiles did not, and no HMR update was ever missed in testing. It stays on
because bind mounts on Docker Desktop and WSL2 routinely drop inotify events and the
failure mode is silent. Turn it off for lower idle CPU once HMR is confirmed on your host.
Note `CHOKIDAR_USEPOLLING` would do nothing — chokidar reads no env vars of its own, so it
has to go through `server.watch` in `vite.config.ts`, which is what `VITE_USE_POLLING`
drives.

## Frontend env vars are baked at build time

Vite inlines `VITE_*` into the bundle when it builds, so they are **build args, not runtime
env**. Compose reads them from the repo-root `.env` (its variable-substitution file), not
from `frontend/.env`:

```bash
# in .env at the repo root
VITE_SUPABASE_URL=https://xxxx.supabase.co
VITE_SUPABASE_ANON_KEY=eyJ...
VITE_LIFF_ID=1234567890-Abcdefgh
```

Then `docker compose build frontend` to pick up a change. Leaving them blank runs the app
as a public demo with no sign-in, which is a supported mode.

The anon key is safe to ship — row access is governed by Postgres RLS. The backend-side
`SUPABASE_SERVICE_ROLE_KEY` is **not**, and never reaches an image layer: it is read from
`.env` at container start via `env_file`.

## The `data/` directory is mounted, not baked

`.dockerignore` excludes `data/` from the build context. It is gitignored, several GB in
places, and its KG/RAG stores are produced by the pipelines rather than shipped — so the
image stays lean and compose bind-mounts the host tree instead:

```yaml
- ./data:/app/data:ro                 # datasets, KG, RAG — read-only
- xcoach-runtime:/app/data/runtime    # named volume: the only path the API writes
```

The read-only mount is deliberate. The backend only ever *reads* under `data/`; the sole
write target is `data/runtime/objects/`, where `LocalObjectStore` puts uploads when
Cloudflare R2 is unconfigured. Keeping that on a named volume avoids the uid mismatch
between the host user and the container's non-root `xcoach` (uid 10001), which would
otherwise make writes to a bind-mounted host directory fail with `EACCES`.

Consequence: a plain `docker run` of the backend image, with nothing mounted, still serves
the API — analysis and the knowledge endpoints just report their stores missing.

In production, set the four `R2_*` variables. Uploads then go to Cloudflare R2 and the
`xcoach-runtime` volume goes unused; `/api/health` reports `"storage_configured": true`
and startup logs `Object storage: Cloudflare R2 (bucket=...)`. Leaving them blank logs a
WARNING and activates the unauthenticated dev endpoint `GET /api/local-object/{key}` —
fine locally, a misconfiguration anywhere real.

## What is in the backend image

`requirements-docker.txt` is the **web subset** of `requirements.txt`: FastAPI, Supabase,
boto3, httpx, plus what the analysis path actually imports (numpy, MediaPipe, OpenCV,
networkx, `langchain-text-splitters`, pypdf). `opencv-python` is swapped for
`opencv-python-headless` — no X11 in a container.

Excluded on purpose, because nothing reachable from `backend/` or `src/{pose,knowledge}`
imports them: `torch`, `transformers` (VideoMAE, `src/{video,rehab24}`), the `langchain*`
and `openrouter` packages (Gemini KG extraction; the chat service speaks raw HTTP to the
OpenAI-compatible endpoint), `beautifulsoup4`, and the test tooling. Including them would
add several GB for code the API never calls.

**Keep `requirements-docker.txt` in sync with `requirements.txt` by hand** when a backend
dependency changes. If you add a research-pipeline dependency, it does not belong here.

The image also installs `ffmpeg` so OpenCV can decode what phones actually upload (MOV,
HEVC), and `libgl1`/`libglib2.0-0`/`libsm6`/`libxext6` because MediaPipe pulls in the
non-headless `opencv-contrib-python` wheel, which links `libGL` even when nothing is drawn.

Running the research pipelines under `scripts/` in Docker is **not** covered by this setup
— they need the full `requirements.txt` (torch, transformers) and the datasets. Use the
local `.venv` for those.

## Operational notes

- **Concurrency.** `XCOACH_MAX_CONCURRENT_ANALYSES` (default 2) caps in-process analyses
  *per uvicorn worker*. The image runs a single worker, so the container's ceiling is that
  number. Scale with replicas rather than `--workers`: each analysis is CPU- and
  RAM-heavy, and workers multiply the cap.
- **Timeouts.** nginx allows 900s on `/api/` — a cold analysis (MediaPipe + rules + RAG)
  can run for minutes, and the 60s default would 504 mid-run.
- **Streaming.** `proxy_buffering off` on `/api/` keeps `/api/chat` SSE tokens flowing;
  with buffering on, the whole reply is held until the stream closes.
- **Uploads.** nginx caps request bodies at 256 MB (`client_max_body_size`).
- **Health.** Both services define a `HEALTHCHECK`; `frontend` waits on
  `backend: service_healthy` in the default stack. Watch with `docker compose ps`.
- **Ports** are overridable without editing compose: `BACKEND_PORT`, `FRONTEND_PORT`,
  `VITE_PORT`.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `/api/health` shows `kg_graph: false`, `rag_db: false` | `data/kg/sports_kg_v3.graphml` or `data/rag/vector_db/` is missing on the host. Build them with the `scripts/knowledge/` pipelines; the mount is read-only, so it must happen outside the container. |
| Sign-in missing after editing `frontend/.env` | Frontend vars come from the **repo-root** `.env` and are baked in. Set them there and `docker compose build frontend`. |
| `EACCES` writing an upload | Something is writing outside `data/runtime/`. The rest of `/app/data` is mounted read-only by design. |
| A backend edit doesn't reload | Known and intermittent over bind mounts — see [File watching uses polling](#file-watching-uses-polling). Fall back to `docker compose ... restart backend`. |
| `frontend` build fails on a native module | A host `node_modules` leaked into the build context — `frontend/.dockerignore` excludes it; check for a stale bind mount. |
