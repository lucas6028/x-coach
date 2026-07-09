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

# Cap on concurrent in-process analyses (P0 stop-gap until the Celery/Redis worker queue lands).
# Each analysis is CPU/RAM-heavy (MediaPipe + rules + RAG), so excess uploads wait for a slot
# rather than all piling onto worker threads and exhausting the box. This is *per process*: with
# ``uvicorn --workers N`` the effective ceiling is N * MAX_CONCURRENT_ANALYSES.
MAX_CONCURRENT_ANALYSES = max(1, int(os.getenv("XCOACH_MAX_CONCURRENT_ANALYSES", "2")))

# ---------------------------------------------------------------------------
# Upload limits & per-user storage quota (public-deployment guardrails).
#
# A squat-coaching clip is a few seconds long; a 1080p phone recording of one is well under
# 100 MB. Capping single uploads at 100 MB / 60 s keeps a stray 4K/multi-minute file from
# eating RAM (the file is read whole before the pose pipeline runs) or ballooning object
# storage, and the per-user quota bounds total spend for a 30–40 person demo:
#   40 users x 1 GB  = 40 GB worst case (~$0.60/mo on Cloudflare R2, egress free).
# All four are env-overridable so a bigger deployment can loosen them without a code change.
MAX_UPLOAD_BYTES = max(1, int(os.getenv("XCOACH_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024))))
MAX_UPLOAD_DURATION_S = max(1, int(os.getenv("XCOACH_MAX_UPLOAD_DURATION_S", "60")))
USER_VIDEO_QUOTA_COUNT = max(1, int(os.getenv("XCOACH_USER_VIDEO_QUOTA_COUNT", "30")))
USER_STORAGE_QUOTA_BYTES = max(1, int(os.getenv("XCOACH_USER_STORAGE_QUOTA_BYTES", str(1024**3))))

# Allowed origins for the Vite dev server.
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def ensure_runtime_dirs() -> None:
    """Create the runtime upload directories if they do not yet exist."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_POSE_DIR.mkdir(parents=True, exist_ok=True)
