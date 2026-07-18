"""Path configuration for the x-coach web backend.

All paths resolve relative to the repository root, mirroring the convention used by
modules under ``src/`` (``REPO_ROOT = Path(__file__).resolve().parents[N]``). The backend
is meant to be launched from the repo root so that ``from src.pose ... import`` resolves.
"""

from __future__ import annotations

import os
from pathlib import Path

# backend/app/config.py -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
LABELED_DIR = DATA_DIR / "Fitness-AQA" / "Squat" / "Labeled_Dataset"
VIDEOS_DIR = LABELED_DIR / "videos"
POSE_JSON_DIR = LABELED_DIR / "pose_json"
DETECTIONS_DIR = LABELED_DIR / "pose_rule_detections"
LABELS_DIR = LABELED_DIR / "Labels"

# Knowledge stores (defaults match the src/ modules).
KG_GRAPH_FILE = DATA_DIR / "kg" / "sports_kg_v3.graphml"
RAG_DB_DIR = DATA_DIR / "rag" / "vector_db"

# Transitional default for the analyze/library retrieval scope: the pose rule detector is
# squat-only today, so retrieval seeds are scoped to Squat until per-movement detectors exist.
DEFAULT_ANALYSIS_MOVEMENT = "Squat"

# Runtime scratch space for uploaded videos and their derived pose JSON (gitignored).
RUNTIME_DIR = DATA_DIR / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
UPLOAD_POSE_DIR = RUNTIME_DIR / "pose_json"

SPLIT_NAMES = ("train", "val", "test")

# Cap on concurrent in-process analyses (P0 stop-gap until the Celery/Redis worker queue lands).
# Each analysis is CPU/RAM-heavy (MediaPipe + rules + RAG), so excess uploads wait for a slot
# rather than all piling onto worker threads and exhausting the box. This is *per process*: with
# ``uvicorn --workers N`` the effective ceiling is N * MAX_CONCURRENT_ANALYSES.
MAX_CONCURRENT_ANALYSES = max(1, int(os.getenv("XCOACH_MAX_CONCURRENT_ANALYSES", "2")))

# Allowed origins: the Vite dev server, plus any extra comma-separated origins from the
# environment — e.g. the ngrok tunnel or production host serving the LINE LIFF frontend
# (XCOACH_CORS_ORIGINS=https://xxxx.ngrok-free.app). Note that when the SPA is served
# through Vite's /api proxy (including via ngrok), requests are same-origin and CORS never
# applies; this matters only when the frontend calls the API cross-origin directly.
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    *[o.strip() for o in os.getenv("XCOACH_CORS_ORIGINS", "").split(",") if o.strip()],
]


def ensure_runtime_dirs() -> None:
    """Create the runtime upload directories if they do not yet exist."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_POSE_DIR.mkdir(parents=True, exist_ok=True)
