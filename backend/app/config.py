"""Path configuration for the x-coach web backend.

All paths resolve relative to the repository root, mirroring the convention used by
modules under ``src/`` (``REPO_ROOT = Path(__file__).resolve().parents[N]``). The backend
is meant to be launched from the repo root so that ``from src.pose ... import`` resolves.
"""

from __future__ import annotations

from pathlib import Path

# backend/app/config.py -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
LABELED_DIR = DATA_DIR / "Squat" / "Labeled_Dataset"
VIDEOS_DIR = LABELED_DIR / "videos"
POSE_JSON_DIR = LABELED_DIR / "pose_json"
DETECTIONS_DIR = LABELED_DIR / "pose_rule_detections"
LABELS_DIR = LABELED_DIR / "Labels"

# Knowledge stores (defaults match the src/ modules).
KG_GRAPH_FILE = DATA_DIR / "kg" / "squat_kg_v2.graphml"
RAG_DB_DIR = DATA_DIR / "rag" / "vector_db"

# Runtime scratch space for uploaded videos and their derived pose JSON (gitignored).
RUNTIME_DIR = DATA_DIR / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
UPLOAD_POSE_DIR = RUNTIME_DIR / "pose_json"

SPLIT_NAMES = ("train", "val", "test")

# Allowed origins for the Vite dev server.
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def ensure_runtime_dirs() -> None:
    """Create the runtime upload directories if they do not yet exist."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_POSE_DIR.mkdir(parents=True, exist_ok=True)
