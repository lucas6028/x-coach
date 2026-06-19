# x-coach backend (FastAPI)

A thin web service that wraps the existing `src/` pipeline (pose extraction → rule-based
biomechanics → KG/RAG retrieval). It contains **no biomechanics logic of its own** — every
endpoint delegates to functions under `src/`.

## Run

From the **repository root** (so `from src... import` resolves), with the venv active:

```bash
source .venv/bin/activate
pip install -r requirements.txt          # first time (adds fastapi/uvicorn/python-multipart)
uvicorn backend.app.main:app --reload --port 8000
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/api/health` | liveness + which data stores are present |
| POST | `/api/analyze` | upload a squat video → MediaPipe extract → rule detection (+retrieval) |
| GET  | `/api/videos?limit=&offset=&fault=` | list precomputed labeled clips (faulty clips first) |
| GET  | `/api/analysis/{video_id}` | precomputed analysis for a library clip (retrieval enriched on demand) |
| GET  | `/api/pose/{video_id}` | slim 33-landmark overlay block |
| GET  | `/api/video-file/{video_id}` | stream the source mp4 (supports HTTP Range / seeking) |
| GET  | `/api/knowledge/graph?query=` | knowledge-graph subgraph for the KG widget |
| GET  | `/api/knowledge/rag?query=` | ranked RAG snippets |

Interactive docs: <http://localhost:8000/docs>.

## Layout

```
backend/app/
  main.py            FastAPI app, CORS, router wiring, /api/health
  config.py          repo-root paths + runtime/upload dirs
  services/
    analysis.py      live upload: process_video + detect_pose_rules_from_json + slim pose block
    library.py       list/load precomputed labeled videos + ground-truth labels
    knowledge.py     wrappers over retrieve_graph_context / query_vector_db
  routers/           analyze.py, videos.py, knowledge.py
```

Uploaded videos and their derived pose JSON land in `data/runtime/` (gitignored).

## Tests

The backend has a self-contained suite at `tests/test_backend.py` that covers every section
(`config`, the three `services/`, the three `routers/`, and `main`) at **100% line + branch
coverage**. The heavy ML pipeline (`src.pose.*`) and knowledge retrieval (`src.knowledge.*`)
are mocked, so the suite runs fast and needs no MediaPipe/torch or `data/` fixtures.

Run from the **repository root** (scope to `tests/` — a bare `pytest` collects the stale
root-level `test_metadata.py`):

```bash
source .venv/bin/activate
python -m pytest tests/test_backend.py            # backend API suite (69 tests)
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
(`test_mediapipe_smoothing`, `test_mmpose_pose_extraction`, `test_videomae_video_classifier`)
and data-dependent (`test_backend_analysis`) modules are skipped there.

## Notes

- Live `/api/analyze` runs MediaPipe (`model_complexity=2`) on CPU — expect ~15–25s for a ~3s clip.
- The analysis response is the `detect_pose_rules_from_json` dict with `frame_metrics` dropped and
  a compact `pose` block (x, y, visibility only) attached for the skeleton overlay.
- Fully offline: no API key needed (the LLM reasoning layer is deferred).
