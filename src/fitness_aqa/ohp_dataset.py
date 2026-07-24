"""Fitness-AQA OHP (overhead press) video-level fault labels: knees and elbows.

Same shape as ``squat_dataset``: ``Labels/error_{knees,elbows}.json`` map a video id to a
list of ``[start, end]`` fault spans (empty = clean), reduced to a binary per video.
Splits are the release's video-level ``Splits/{train,val,test}_keys.json`` (1,582 / 339 /
339). Spans mark *where* the fault is and must never drive frame selection.

Faults by plane, for the cross-movement depth test:
* ``knees``  -- knees bending during the press (should stay locked): sagittal / vertical.
* ``elbows`` -- elbow path / flare: has a mediolateral component that foreshortens into
  depth in oblique views (the OHP analogue of squat valgus).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OHP_ROOT = REPO_ROOT / "data" / "Fitness-AQA" / "OHP" / "Labeled_Dataset"
SPLITS = ("train", "val", "test")
FAULTS = ("knees", "elbows")


def load_spans(fault: str, root: Path = DEFAULT_OHP_ROOT) -> dict[str, list]:
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; expected one of {FAULTS}")
    with open(root / "Labels" / f"error_{fault}.json", encoding="utf-8") as f:
        return {str(k): v for k, v in json.load(f).items()}


def load_binary_labels(fault: str, root: Path = DEFAULT_OHP_ROOT) -> dict[str, int]:
    return {k: (1 if len(v) > 0 else 0) for k, v in load_spans(fault, root).items()}


def load_combined_labels(root: Path = DEFAULT_OHP_ROOT) -> dict[str, int]:
    kn = load_binary_labels("knees", root)
    el = load_binary_labels("elbows", root)
    keys = set(kn) | set(el)
    return {k: (1 if (kn.get(k, 0) or el.get(k, 0)) else 0) for k in keys}


def load_split(split: str, root: Path = DEFAULT_OHP_ROOT) -> list[str]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    with open(root / "Splits" / f"{split}_keys.json", encoding="utf-8") as f:
        return [str(s) for s in json.load(f)]


def split_of(root: Path = DEFAULT_OHP_ROOT) -> dict[str, str]:
    out: dict[str, str] = {}
    for split in SPLITS:
        for vid in load_split(split, root):
            out[vid] = split
    return out


def all_labels(root: Path = DEFAULT_OHP_ROOT) -> dict[str, dict[str, int]]:
    return {
        "knees": load_binary_labels("knees", root),
        "elbows": load_binary_labels("elbows", root),
        "combined": load_combined_labels(root),
    }
