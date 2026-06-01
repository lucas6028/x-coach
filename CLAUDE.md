# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

x-coach is a research prototype for **explainable squat-coaching feedback**. It links four pipelines so visual signals become grounded, interpretable advice: pose perception, video (VideoMAE) perception, rule-based biomechanics, and knowledge retrieval (RAG + knowledge graph). See `project-overview.md` / `研究計畫.md` for the full research framing (perception → GraphRAG → reasoning/generation → frontend); much of the reasoning/generation and frontend layers are still aspirational — the implemented code is the perception, rules, and retrieval foundation.

## Environment & commands

- The virtualenv is checked out at `.venv/`. Activate it first: `source .venv/bin/activate`. There is no `python` on PATH otherwise.
- Install deps: `pip install -r requirements.txt`.
- **Tests use `unittest`, not pytest** (pytest is not installed):
  - All tests: `python -m unittest discover -s tests -v`
  - Single test: `python -m unittest tests.test_pose_rule_detector -v`
  - Single case: `python -m unittest tests.test_pose_rule_detector.PoseRuleDetectorTests.test_depth_rule_distinguishes_above_and_below_parallel`
- `GOOGLE_API_KEY` is only needed for Gemini-backed knowledge-graph extraction (`src/knowledge/extract_kg.py`). Everything else, including RAG, runs fully offline.

## Architecture: scripts vs src

The most important structural convention: **`scripts/` are thin CLI entry points; all logic lives in `src/`.** Each script bootstraps the repo root onto `sys.path` (`PROJECT_ROOT = Path(__file__).resolve().parents[2]`) and then calls a `main()` (or function) in the matching `src/` module. Modules import each other by absolute package path from the repo root (`from src.pose.pose_rule_detector import ...`), so **run everything from the repository root**, not from inside `scripts/`.

`src/` and `scripts/` mirror each other in four workflow areas:

- **`pose/`** — MediaPipe / MMPose landmark extraction (`process_videos.py`, `mmpose_pose_extraction.py` adapts COCO-WholeBody to the MediaPipe 33-landmark layout), pose features, camera-view estimation, and `pose_rule_detector.py` (interpretable squat-fault rules over 33 landmarks with tunable thresholds).
- **`video/`** — VideoMAE spatio-temporal feature extraction and lightweight video-level error classifiers (`videomae_video_classifier.py`), plus experiment grids and error analysis.
- **`knowledge/`** — local RAG (`rag_vector_db.py`) and the squat knowledge graph (`extract_kg.py`, `graph_retrieval.py`, `perception_to_graph.py`). The RAG store uses a built-in offline `HashEmbeddingBackend`, so the vector DB builds/queries with no external API.
- **`rehab24/`** — REHAB24-6 dataset pipeline: repetition-level manifest + subject-wise splits, skeleton features (local), VideoMAE features (GPU/Colab), feature fusion, and a correctness classifier.

**Cross-pipeline link:** `pose_rule_detector` can enrich detected faults with retrieved knowledge via `src/knowledge/graph_retrieval.py` (the RAG/KG store under `data/rag/vector_db` and `data/kg`). This retrieval is optional — pass `--no-retrieval` to run rules standalone.

Common workflow commands live in the per-directory READMEs: `scripts/{pose,video,knowledge,rehab24}/README.md`.

## Data layout

Pipelines read/write under `data/` and paths are resolved relative to the repo root inside each module (`REPO_ROOT = Path(__file__).resolve().parents[2]`):

- `data/Squat/{Unlabeled,Labeled}_Dataset/` — videos, processed pose JSON, pose features, VideoMAE features, `Splits/`, `Labels/`.
- `data/REHAB24-6/` — REHAB24-6 source data and derived features.
- `data/kg/` — knowledge-graph `.graphml` files and canonical mapping JSON under `data/kg/docs/`.
- `data/rag/{docs,vector_db}/` — RAG source documents and the built index (chunks, hash embeddings, manifest).

## Notes for working here

- Modules favor a local-first, dependency-light style (stdlib + numpy/networkx; pure-function helpers that are unit-tested in isolation — e.g. `compute_frame_metrics`, feature-vector builders, normalization payloads).
- `test_metadata.py` at the repo root is a stale ad-hoc script (imports the old `src.rag_vector_db` path); prefer the modules under `src/knowledge/`. Don't model new code on it.
- `notes/` holds experiment summaries and results; `docs/` has longer walkthroughs (KG LLM extraction, MediaPipe processing).
