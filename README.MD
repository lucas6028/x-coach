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

```bash
pip install -r requirements.txt
```

If you want to run knowledge-graph extraction with Gemini, set `GOOGLE_API_KEY` first.

## Common Commands

### Adding New Documents to RAG

You can easily add new documents (e.g., text files, PDFs) to the RAG database while automatically generating the required metadata and citation references:

```bash
# Add a generic blog article
python scripts/add_document.py path/to/article.txt --type health_blog --ref "Blog: Top 10 Squat Tips"

# Add an academic paper with detailed metadata
python scripts/add_document.py path/to/paper.pdf \
    --type academic_paper \
    --title "Knee Joint Forces During Squat" \
    --author "John Doe" \
    --year 2024 \
    --ref "Doe, J. (2024). Knee Joint Forces During Squat. Journal of Sports Biomechanics."
```
Once added, simply rebuild the vector DB to index the new files:

```bash
python scripts/build_rag_vector_db.py build
```

### Build the local RAG index

```bash
python scripts/build_rag_vector_db.py build
```

Query the local RAG index:

```bash
python scripts/query_rag.py query "knees inward correction"
```

Extract MediaPipe pose landmarks from unlabeled videos:

```bash
python scripts/run_pose_extraction.py --dataset unlabeled --limit 5
```

Extract VideoMAE features for labeled squat clips:

```bash
python scripts/run_videomae_feature_extraction.py
```

Extract pose-only features for labeled squat clips:

```bash
python scripts/run_pose_extraction.py --dataset labeled --no-video
python scripts/run_pose_feature_extraction.py
```

Estimate coarse camera/view metadata from labeled pose JSON:

```bash
python scripts/run_view_estimation.py
```

By default, this squat-dataset estimator suppresses `front` and `front_oblique`
outputs because face landmark visibility is noisy in rear-view videos. Use
`--allow-front` only when the dataset contains verified front-view clips.

Train the existing classifier on pose-only features:

```bash
python scripts/run_videomae_experiment_grid.py \
  --feature-dir data/Squat/Labeled_Dataset/pose_features \
  --train-keys data/Squat/Labeled_Dataset/Splits/train_keys.json \
  --val-keys data/Squat/Labeled_Dataset/Splits/val_keys.json \
  --test-keys data/Squat/Labeled_Dataset/Splits/test_keys.json \
  --forward-labels data/Squat/Labeled_Dataset/Labels/error_knees_forward.json \
  --inward-labels data/Squat/Labeled_Dataset/Labels/error_knees_inward.json \
  --output-root data/Squat/pose_classifier_experiments
```

Train the lightweight classifier on cached features:

```bash
python scripts/train_videomae_classifier.py
```

Extract a knowledge graph from a document:

```bash
python src/extract_kg.py path/to/document.pdf
```

Query from Knowledge Graph
 ```bash
python.exe scripts/query_graph.py "knee valgus" --hops 1
python.exe scripts/query_graph.py "foot pronation" --hops 1
python.exe scripts/query_graph.py "anterior knee pain" --hops 1
python.exe scripts/query_graph.py --pose-json data\Squat\Unlabeled_Dataset\processed_poses\25195_3.json --hops 1
```

Clean Knowledge Graph
```bash
python scripts/clean_kg.py --graph-file data\kg\squat_kg_v2.graphml --mapping-file data\kg\docs\squat_canonical_mapping_v1.json --output-file data\kg\squat_kg_v2_cleaned.graphml
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
