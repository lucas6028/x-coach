"""Analysis orchestration: wrap the existing ``src/`` pose pipeline for the web API.

This module performs *no* biomechanics logic of its own. It calls:
  - ``src.pose.process_videos.process_video`` (video -> pose JSON), and
  - ``src.pose.pose_rule_detector.detect_pose_rules_from_json`` (pose JSON -> faults + retrieval).

It then attaches a slimmed ``pose`` block (x, y, visibility only) so the frontend can draw a
skeleton overlay synced to the video without re-downloading the heavy world-landmark data.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from backend.app import config

# NOTE: ``src.pose.process_videos`` / ``src.pose.pose_rule_detector`` are imported lazily inside
# ``analyze_video_file`` (not at module load) so importing this service does not drag in MediaPipe
# / OpenCV / torch. That keeps the web process import-light and lets the API layer be tested
# without the heavy ML stack installed; the upload path imports them on first use.

# Indices to keep from each 33-landmark frame: x, y, visibility (drop z / world landmarks).
_LANDMARK_COUNT = 33


def build_pose_block(pose_json_path: Path) -> dict[str, Any]:
    """Read a pose JSON file and return a compact overlay payload.

    Returns ``{"fps": float, "width": int, "height": int, "frames": [{"i", "lm"}]}`` where
    ``lm`` is a list of ``[x, y, visibility]`` triples (or ``None`` for frames with no pose).
    """
    payload = json.loads(pose_json_path.read_text(encoding="utf-8"))
    return build_pose_block_from_payload(payload)


def build_pose_block_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata", {}) or {}
    frames_out: list[dict[str, Any]] = []
    for frame in payload.get("frames", []) or []:
        landmarks = frame.get("landmarks")
        if not landmarks:
            slim = None
        else:
            slim = [
                [round(float(lm["x"]), 5), round(float(lm["y"]), 5), round(float(lm["visibility"]), 4)]
                for lm in landmarks
            ]
        frames_out.append({"i": int(frame.get("frame_index", len(frames_out))), "lm": slim})
    return {
        "fps": float(metadata.get("fps", 30.0) or 30.0),
        "width": int(metadata.get("width", 0) or 0),
        "height": int(metadata.get("height", 0) or 0),
        "frames": frames_out,
    }


def _strip_frame_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Drop the verbose per-frame metrics block; the slim pose block replaces it for the UI."""
    result.pop("frame_metrics", None)
    return result


def analyze_video_file(source_path: Path, *, video_id: str | None = None) -> dict[str, Any]:
    """Run the full pipeline on an arbitrary video file (the live-upload flow).

    Extracts pose to a runtime JSON path, runs rule detection with retrieval enrichment, and
    returns the detector result with a slimmed ``pose`` block attached.
    """
    # Deferred imports: pull in MediaPipe/OpenCV (process_videos) and the detector only when an
    # upload is actually analyzed, keeping module import (and server startup) lightweight.
    from src.pose.pose_rule_detector import detect_pose_rules_from_json
    from src.pose.process_videos import process_video

    config.ensure_runtime_dirs()
    vid = video_id or source_path.stem
    pose_json_path = config.UPLOAD_POSE_DIR / f"{vid}.json"

    ok = process_video(str(source_path), str(pose_json_path))
    if not ok or not pose_json_path.exists():
        raise RuntimeError(f"Pose extraction failed for {source_path.name}")

    result = detect_pose_rules_from_json(
        pose_json_path,
        video_id=vid,
        include_retrieval=True,
        graph_file=config.KG_GRAPH_FILE,
        rag_db_dir=config.RAG_DB_DIR,
        movement=config.DEFAULT_ANALYSIS_MOVEMENT,
    )
    result = _strip_frame_metrics(result)
    result["pose"] = build_pose_block(pose_json_path)
    result["source"] = "upload"
    return result


def save_upload(file_bytes: bytes, suffix: str = ".mp4") -> tuple[str, Path]:
    """Persist uploaded bytes under the runtime upload dir, returning ``(video_id, path)``."""
    config.ensure_runtime_dirs()
    video_id = f"upload_{uuid.uuid4().hex[:12]}"
    dest = config.UPLOAD_DIR / f"{video_id}{suffix}"
    dest.write_bytes(file_bytes)
    return video_id, dest
