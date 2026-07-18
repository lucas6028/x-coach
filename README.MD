# x-coach

x-coach is an AI coaching research prototype. It analyzes squat form over a 16-movement biomechanics knowledge graph, connecting video perception, rule-based biomechanics checks, lightweight classifiers, and local retrieval over project knowledge so visual signals can be turned into explainable coaching feedback. (Video analysis is squat-only today; the knowledge graph and in-app Explore browser already span all 16 movements.)

## What This Project Does

- Extracts pose landmarks from squat videos with MediaPipe or RTMPose-compatible outputs.
- Builds pose-only and VideoMAE feature representations for labeled squat clips.
- Trains lightweight video-level classifiers for squat error labels.
- Builds REHAB24-6 repetition-level skeleton features and correctness classifiers.
- Detects common squat technique faults with interpretable pose rules.
- Builds and queries a local RAG index from project notes, documents, and knowledge-graph content.
- Provides knowledge-graph utilities for extracting, cleaning, auditing, and querying multi-movement biomechanics concepts (16 movements across 5 flagships + 11 general stubs).

## Project Layout

- `src/pose/` - pose extraction, pose features, view estimation, and rule detection.
- `src/video/` - VideoMAE feature extraction and video-level classifiers.
- `src/knowledge/` - knowledge graph, retrieval, and local RAG utilities.
- `src/rehab24/` - REHAB24-6 manifest, feature extraction, fusion, and correctness classification.
- `scripts/pose/` - pose pipeline and pose-analysis entry points.
- `scripts/video/` - VideoMAE and classifier experiment entry points.
- `scripts/knowledge/` - knowledge graph and RAG entry points.
- `scripts/rehab24/` - REHAB24-6 experiment entry points.
- `backend/` - FastAPI web service wrapping the pose/rules/retrieval pipeline (see `backend/README.md`).
- `frontend/` - React + Vite (yarn) dashboard: skeleton overlay, fault timeline, GraphRAG feedback (see `frontend/README.md`).
- `data/` - datasets, labels, processed poses, features, and cached RAG assets.
- `demo/` - browser demo assets and pose-estimation prototype code.
- `docs/` - longer walkthroughs.
- `notes/` - research notes and experiment summaries.
- `tests/` - Python unit tests (pytest) for core behavior and analysis helpers. Frontend tests are Vitest, under `frontend/` (`yarn test`).

## Data Areas

- `data/Squat/Unlabeled_Dataset/` - raw unlabeled videos and extracted pose JSON.
- `data/Squat/Labeled_Dataset/` - labeled clips, split files, labels, pose JSON, pose features, and VideoMAE features.
- `data/kg/` - knowledge graph files and canonical mappings.
- `data/rag/vector_db/` - local RAG chunks, embeddings, and manifest.

## Setup

Create and activate a Python virtual environment, then install the project requirements:

```bash
pip install -r requirements.txt
```

Set `GOOGLE_API_KEY` only if you plan to run Gemini-backed knowledge-graph extraction.

## Running Workflows

Command details live with the script directories:

- [Pose scripts](scripts/pose/README.md)
- [Video scripts](scripts/video/README.md)
- [REHAB24-6 scripts](scripts/rehab24/README.md)
- [Knowledge and RAG scripts](scripts/knowledge/README.md)

## Documentation

- [Knowledge graph extraction walkthrough](docs/kg_llm_extraction_walkthrough.md)
- [MediaPipe processing walkthrough](docs/media_pipe_processing_walkthrough.md)
- `notes/` contains experiment summaries and research notes.
- `tests/` contains executable examples for several core modules.

## Project Goal

Support explainable coaching feedback — squat analysis today, over a multi-movement biomechanics knowledge graph — by linking visual signals, structured biomechanics knowledge, and retrieval-based guidance.
