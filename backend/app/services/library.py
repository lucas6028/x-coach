"""Library service: expose the already-processed labeled squat videos for instant demos.

Each labeled clip has a video (``videos/{id}.mp4``), pose JSON (``pose_json/{split}/{id}.json``)
and a precomputed rule-detection file (``pose_rule_detections/{split}/{id}.json``). This service
lists them and loads the precomputed analysis, enriching with knowledge retrieval on demand.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from src.pose.pose_rule_detector import retrieve_contexts_for_detections

from backend.app import config
from backend.app.services.analysis import build_pose_block

# Time-segment ground-truth labels keyed by video id (start/end seconds per fault file).
_LABEL_FILES = {
    "knees_forward": config.LABELS_DIR / "error_knees_forward.json",
    "knees_inward": config.LABELS_DIR / "error_knees_inward.json",
}

# A ``video_id`` is a filename stem (a dataset slug or ``upload_<hex>``). It is interpolated
# into filesystem paths and, previously, into a glob pattern — so it must never carry path
# separators, parent-dir hops, NUL, or glob metacharacters. We deny exactly those rather than
# allow-list a charset, so legitimate dataset slugs (which may use varied punctuation) are not
# rejected while traversal / wildcard injection is shut out.
_UNSAFE_VIDEO_ID_CHARS = frozenset("*?[]/\\\x00")


def is_safe_video_id(video_id: str) -> bool:
    """True if ``video_id`` is safe to interpolate into a data-directory path lookup."""
    if not video_id or ".." in video_id:
        return False
    return not any(ch in _UNSAFE_VIDEO_ID_CHARS for ch in video_id)


@functools.lru_cache(maxsize=1)
def _labels() -> dict[str, dict[str, list]]:
    out: dict[str, dict[str, list]] = {}
    for fault, path in _LABEL_FILES.items():
        if path.exists():
            out[fault] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _split_of(video_id: str) -> str | None:
    """Find which split a video's precomputed detection lives in."""
    for split in config.SPLIT_NAMES:
        if (config.DETECTIONS_DIR / split / f"{video_id}.json").exists():
            return split
    return None


def detection_path(video_id: str) -> Path | None:
    if not is_safe_video_id(video_id):
        return None
    split = _split_of(video_id)
    return config.DETECTIONS_DIR / split / f"{video_id}.json" if split else None


def pose_json_path(video_id: str) -> Path | None:
    if not is_safe_video_id(video_id):
        return None
    for split in config.SPLIT_NAMES:
        candidate = config.POSE_JSON_DIR / split / f"{video_id}.json"
        if candidate.exists():
            return candidate
    return None


def video_path(video_id: str) -> Path | None:
    if not is_safe_video_id(video_id):
        return None
    candidate = config.VIDEOS_DIR / f"{video_id}.mp4"
    return candidate if candidate.exists() else None


def list_videos(*, limit: int = 50, offset: int = 0, fault: str | None = None) -> dict[str, Any]:
    """List precomputed library entries with summary metadata.

    Returns ``{"total": int, "items": [{video_id, split, view_type, fault_count, faults[]}]}``.
    """
    items: list[dict[str, Any]] = []
    for split in config.SPLIT_NAMES:
        split_dir = config.DETECTIONS_DIR / split
        if not split_dir.exists():
            continue
        for det_file in sorted(split_dir.glob("*.json")):
            video_id = det_file.stem
            if not video_path(video_id):
                continue
            data = json.loads(det_file.read_text(encoding="utf-8"))
            faults = sorted({d.get("fault_id", "") for d in data.get("detections", [])})
            if fault and fault not in faults:
                continue
            items.append(
                {
                    "video_id": video_id,
                    "split": split,
                    "view_type": (data.get("view") or {}).get("view_type", "unknown"),
                    "fault_count": len(data.get("detections", [])),
                    "faults": faults,
                }
            )

    total = len(items)
    # Surface clips that actually contain faults first — better demo material.
    items.sort(key=lambda it: (it["fault_count"] == 0, it["video_id"]))
    return {"total": total, "items": items[offset : offset + limit]}


def ground_truth_labels(video_id: str) -> dict[str, list]:
    """Return human-annotated fault time segments for a library video, if any."""
    return {fault: mapping.get(video_id, []) for fault, mapping in _labels().items() if mapping.get(video_id)}


def load_analysis(video_id: str) -> dict[str, Any]:
    """Load the precomputed analysis for a library video, enriching retrieval if missing.

    Attaches the slimmed ``pose`` overlay block and ground-truth labels.
    """
    det_path = detection_path(video_id)
    if det_path is None:
        raise FileNotFoundError(f"No precomputed analysis for video '{video_id}'.")

    result = json.loads(det_path.read_text(encoding="utf-8"))
    result.pop("frame_metrics", None)

    if not result.get("retrievals") and result.get("detections"):
        result["retrievals"] = retrieve_contexts_for_detections(
            result["detections"],
            graph_file=config.KG_GRAPH_FILE,
            rag_db_dir=config.RAG_DB_DIR,
            movement=config.DEFAULT_ANALYSIS_MOVEMENT,
        )

    pose_path = pose_json_path(video_id)
    if pose_path is not None:
        result["pose"] = build_pose_block(pose_path)

    result["ground_truth"] = ground_truth_labels(video_id)
    result["source"] = "library"
    return result
