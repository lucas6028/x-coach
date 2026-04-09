# x-coach

An AI coaching research prototype for squat analysis. The repo combines pose extraction, VideoMAE-based video features, a lightweight classifier, and a local RAG knowledge base built from project notes and domain documents.

## What is in this repo

- MediaPipe pose extraction for unlabeled squat videos.
- VideoMAE feature extraction for labeled squat clips.
- A small classifier trained on cached VideoMAE embeddings.
- A local, offline RAG vector database built from project text, notes, and knowledge-graph content.
- Knowledge-graph extraction from PDFs, HTML, or text files using Gemini.
- A simple demo folder with HTML and pose-estimation code.

## Repository Layout

- `src/` core implementation modules.
- `scripts/` runnable entry points.
- `data/` datasets, labels, processed poses, features, and cached RAG assets.
- `demo/` browser demo assets and pose-estimation prototype code.
- `docs/` walkthroughs for knowledge-graph extraction and MediaPipe processing.
- `notes/` project notes used by the RAG builder.

## Data Overview

The current project is organized around the squat task.

- `data/Squat/Unlabeled_Dataset/` contains raw videos and extracted pose JSON.
- `data/Squat/Labeled_Dataset/` contains split files, labels, and cached VideoMAE features.
- `data/kg/sports_kg.graphml` stores the knowledge graph.
- `data/rag/vector_db/` stores the generated chunk index, embeddings, and manifest.

## Setup

This project is Python-based. Create and activate a virtual environment, then install the packages required by the scripts you want to run.

Typical dependencies used across the repo include:

- `opencv-python`
- `mediapipe`
- `numpy`
- `tqdm`
- `networkx`
- `pydantic`
- `langchain`
- `langchain-core`
- `langchain-community`
- `langchain-google-genai`
- `langchain-text-splitters`
- `pypdf`
- `beautifulsoup4`
- `torch`
- `transformers`

If you want to run knowledge-graph extraction with Gemini, set `GOOGLE_API_KEY` first.

## Common Commands

Build the local RAG index:

```bash
python scripts/build_rag_vector_db.py build
```

Query the local RAG index:

```bash
python scripts/query_rag.py query "knees inward correction"
```

Extract MediaPipe pose landmarks from unlabeled videos:

```bash
python scripts/run_pose_extraction.py --limit 5
```

Extract VideoMAE features for labeled squat clips:

```bash
python scripts/run_videomae_feature_extraction.py
```

Train the lightweight classifier on cached features:

```bash
python scripts/train_videomae_classifier.py
```

Extract a knowledge graph from a document:

```bash
python src/extract_kg.py path/to/document.pdf
```

## Notes on the Current Implementation

- The RAG builder is fully local and uses a hash-based embedding backend, so it does not require an external embedding API.
- The pose pipeline writes per-frame JSON and can optionally emit annotated videos.
- The VideoMAE pipeline expects cached clip features under `data/Squat/Labeled_Dataset/videomae_features/`.
- The classifier uses the split files and error-label JSON files under `data/Squat/Labeled_Dataset/`.

## Docs

- [Knowledge graph extraction walkthrough](docs/kg_llm_extraction_walkthrough.md)
- [MediaPipe processing walkthrough](docs/media_pipe_processing_walkthrough.md)

## Project Goal

The project aims to support explainable coaching feedback for squat analysis by linking visual signals, structured biomechanics knowledge, and retrieval-based guidance.
