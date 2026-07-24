"""Fitness-AQA Squat video-level fault labels: knees-forward and knees-inward.

The frame-level shallow-depth verdict lives in ``shallow_dataset``; this is the
video-level side. ``Labels/error_{knees_forward,knees_inward}.json`` map a video id to a
list of ``[start, end]`` time spans where the fault occurs (empty list = clean). We
reduce that to a binary "does this rep contain the fault" per video.

CRITICAL: the spans mark *where the fault is* -- they must never drive frame selection,
or the classifier would be told the answer. Frame aggregation is over the whole clip or
a pose-derived phase (see ``video_features``), never the labelled interval.

Splits are the release's own video-level ``Splits/{train,val,test}_keys.json`` (1,136 /
243 / 244 videos, disjoint by construction).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQUAT_ROOT = REPO_ROOT / "data" / "Fitness-AQA" / "Squat" / "Labeled_Dataset"
SPLITS = ("train", "val", "test")
FAULTS = ("knees_forward", "knees_inward")


def label_file(fault: str, root: Path = DEFAULT_SQUAT_ROOT) -> Path:
    return root / "Labels" / f"error_{fault}.json"


def split_file(split: str, root: Path = DEFAULT_SQUAT_ROOT) -> Path:
    return root / "Splits" / f"{split}_keys.json"


def load_spans(fault: str, root: Path = DEFAULT_SQUAT_ROOT) -> dict[str, list]:
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; expected one of {FAULTS}")
    with open(label_file(fault, root), encoding="utf-8") as f:
        return {str(k): v for k, v in json.load(f).items()}


def load_binary_labels(fault: str, root: Path = DEFAULT_SQUAT_ROOT) -> dict[str, int]:
    """Video id -> 1 if the fault appears anywhere in the clip, else 0."""
    return {k: (1 if len(v) > 0 else 0) for k, v in load_spans(fault, root).items()}


def load_combined_labels(root: Path = DEFAULT_SQUAT_ROOT) -> dict[str, int]:
    """Video id -> 1 if *either* knees fault is present (union)."""
    kf = load_binary_labels("knees_forward", root)
    ki = load_binary_labels("knees_inward", root)
    keys = set(kf) | set(ki)
    return {k: (1 if (kf.get(k, 0) or ki.get(k, 0)) else 0) for k in keys}


def load_split(split: str, root: Path = DEFAULT_SQUAT_ROOT) -> list[str]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    with open(split_file(split, root), encoding="utf-8") as f:
        return [str(s) for s in json.load(f)]


def split_of(root: Path = DEFAULT_SQUAT_ROOT) -> dict[str, str]:
    """Video id -> split name, for the union of the three split key lists."""
    out: dict[str, str] = {}
    for split in SPLITS:
        for vid in load_split(split, root):
            out[vid] = split
    return out


def all_labels(root: Path = DEFAULT_SQUAT_ROOT) -> dict[str, dict[str, int]]:
    """{'knees_forward': {...}, 'knees_inward': {...}, 'combined': {...}}."""
    return {
        "knees_forward": load_binary_labels("knees_forward", root),
        "knees_inward": load_binary_labels("knees_inward", root),
        "combined": load_combined_labels(root),
    }
