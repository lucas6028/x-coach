# x-coach

An AI coaching research prototype for squat analysis. The repo combines pose extraction, VideoMAE-based video features, a lightweight classifier, a rule-based pose-error detector, and a local RAG knowledge base built from project notes and domain documents.

## Overview

This repository provides tools for extracting pose landmarks, generating VideoMAE features, running lightweight classifiers, and performing rule-based pose error detection for squat analysis. It also includes utilities to build and query a local RAG vector database and to extract/clean knowledge graphs.

## Key Features

- MediaPipe pose extraction for unlabeled squat videos.
- VideoMAE feature extraction for labeled squat clips.
- Lightweight classifier trained on cached VideoMAE embeddings.
- Pose Rule Detection: a rule-based detector for common squat technique errors using pose landmarks.
- Local, offline RAG vector database built from project text, notes, and knowledge-graph content.
- Knowledge-graph extraction from PDFs, HTML, or text files.
- A simple demo folder with HTML and pose-estimation code.

## Repository Layout

- `src/` — core implementation modules.
- `scripts/` — runnable entry points (data processing, model training, RAG builder, detectors).
- `data/` — datasets, labels, processed poses, features, and cached RAG assets.
- `demo/` — browser demo assets and pose-estimation prototype code.
- `docs/` — walkthroughs for knowledge-graph extraction and MediaPipe processing.
- `notes/` — project notes used by the RAG builder.

## Data Overview

The project is organized around a squat analysis dataset. Key locations:

- `data/Squat/Unlabeled_Dataset/` — raw videos and extracted pose JSON.
- `data/Squat/Labeled_Dataset/` — split files, labels, cached VideoMAE features, and pose features.
- `data/kg/` — knowledge-graph files and canonical mappings.
- `data/rag/vector_db/` — chunk index, embeddings, and manifest for local RAG queries.

## Setup

Create and activate a Python virtual environment, then install required packages:

```bash
pip install -r requirements.txt
```

If you plan to run knowledge-graph extraction with Gemini, set `GOOGLE_API_KEY` beforehand.

## Usage / Common Commands

### RAG: Add documents and build index

```bash
# Add a generic document
python scripts/add_document.py path/to/article.txt --type health_blog --ref "Blog: Top 10 Squat Tips"

# Rebuild local vector DB
python scripts/build_rag_vector_db.py build
```

### Query RAG

```bash
python scripts/query_rag.py query "knees inward correction"
```

### Pose extraction & feature pipelines

```bash
# Extract MediaPipe pose landmarks from unlabeled videos
python scripts/run_pose_extraction.py --dataset unlabeled --limit 5

# Extract VideoMAE features for labeled clips
python scripts/run_videomae_feature_extraction.py

# Extract pose-only features for labeled clips
python scripts/run_pose_extraction.py --dataset labeled --no-video
python scripts/run_pose_feature_extraction.py

# Estimate view/camera metadata
python scripts/run_view_estimation.py
```

### Pose Rule Detection

The pose rule detector is a lightweight, rule-based system that inspects pose landmarks to flag common squat technique errors (e.g., knees valgus/inward, excessive forward trunk lean, depth issues).

Run the detector over a dataset or single pose JSON:

```bash
python scripts/run_pose_rule_detection.py \
  --no-retrieval \
  --summary-output data/Squat/Labeled_Dataset/pose_rule_detections_summary.csv

# Run for a single pose JSON file
python scripts/run_pose_rule_detection.py --input data/Squat/Unlabeled_Dataset/processed_poses/25195_3.json --output results/single_detection.json
```

Unit tests and example outputs live in `tests/test_pose_rule_detector.py` and the `scripts/` logging outputs.

### Evaluate Pose Rule Detection

Use the evaluation script to compare rule-based detections against labeled ground-truth and produce per-class metrics and segment IoU. The script defaults to reading detections from `data/Squat/Labeled_Dataset/pose_rule_detections` and writing metrics to `data/Squat/Labeled_Dataset/pose_rule_validation_metrics.csv`.

```bash
# Basic evaluation (uses repo defaults)
python scripts/evaluate_pose_rule_detection.py

# Specify detections directory, view metadata CSV, and output path
python scripts/evaluate_pose_rule_detection.py \
  --detections-dir data/Squat/Labeled_Dataset/pose_rule_detections \
  --view-metadata data/Squat/Labeled_Dataset/view_metadata.csv \
  --output data/Squat/Labeled_Dataset/pose_rule_validation_metrics.csv
```

The script prints a Markdown-formatted summary to stdout and saves a CSV with per-class metrics (`precision`, `recall`, `f1`, `mean_segment_iou`, etc.). See `scripts/evaluate_pose_rule_detection.py` for the exact metric definitions and the default class/label mappings.

### Training and experiments

```bash
# Run classifier experiment grid (pose or VideoMAE features)
python scripts/run_videomae_experiment_grid.py \
  --feature-dir data/Squat/Labeled_Dataset/pose_features \
  --train-keys data/Squat/Labeled_Dataset/Splits/train_keys.json \
  --val-keys data/Squat/Labeled_Dataset/Splits/val_keys.json \
  --test-keys data/Squat/Labeled_Dataset/Splits/test_keys.json \
  --forward-labels data/Squat/Labeled_Dataset/Labels/error_knees_forward.json \
  --inward-labels data/Squat/Labeled_Dataset/Labels/error_knees_inward.json \
  --output-root data/Squat/pose_classifier_experiments

python scripts/train_videomae_classifier.py
```

### Knowledge Graph utilities

```bash
python src/extract_kg.py path/to/document.pdf
python scripts/clean_kg.py --graph-file data/kg/squat_kg_v2.graphml --mapping-file data/kg/docs/squat_canonical_mapping_v1.json --output-file data/kg/squat_kg_v2_cleaned.graphml
```

## Notes on Implementation

- The RAG builder is fully local and uses a hash-based embedding backend — no external embedding API required by default.
- The pose pipeline writes per-frame JSON and can optionally emit annotated videos for visualization.
- The VideoMAE pipeline expects cached clip features under `data/Squat/Labeled_Dataset/videomae_features/`.
- The pose rule detector reads per-frame landmarks and emits per-frame and per-clip rule detections as JSON.

## Docs

- [Knowledge graph extraction walkthrough](docs/kg_llm_extraction_walkthrough.md)
- [MediaPipe processing walkthrough](docs/media_pipe_processing_walkthrough.md)
- See `tests/test_pose_rule_detector.py` for detector examples and expected outputs.

## Project Goal

Support explainable coaching feedback for squat analysis by linking visual signals, structured biomechanics knowledge, and retrieval-based guidance.
