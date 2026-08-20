"""Paths and split/label readers for the Fitness-AQA squat dataset.

Deliberately dependency-light (stdlib only): the extractor needs torch and cv2, but
the materialize, audit and analysis steps do not, and they all need these paths.

The dataset moved from ``data/Squat`` to ``data/Fitness-AQA/Squat`` when the other
Fitness-AQA movements landed; every default here points at the current location.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SQUAT_ROOT = REPO_ROOT / "data" / "Fitness-AQA" / "Squat"
SQUAT_LABELED_ROOT = SQUAT_ROOT / "Labeled_Dataset"
SQUAT_UNLABELED_ROOT = SQUAT_ROOT / "Unlabeled_Dataset"
SPLIT_NAMES = ("train", "val", "test")


def load_json_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}.")
    return [str(item) for item in data]


def load_split_map(split_dir: Path, split_names: tuple[str, ...] = SPLIT_NAMES) -> dict[str, str]:
    """Map every id in ``<split_dir>/<split>_keys.json`` to its split name."""
    split_map: dict[str, str] = {}
    for split_name in split_names:
        for video_id in load_json_list(split_dir / f"{split_name}_keys.json"):
            split_map[video_id] = split_name
    return split_map


def load_labeled_ids(labels_dir: Path) -> set[str]:
    """Ids carrying an entry in at least one squat error-label file.

    ``videomae_video_classifier.build_labels`` reads both files with ``.get``, so an
    id absent from both becomes a *negative* rather than an error -- a silent
    relabeling the audit's ``all_labeled`` check exists to catch.
    """
    labeled: set[str] = set()
    for name in ("error_knees_forward.json", "error_knees_inward.json"):
        path = labels_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            labeled.update(str(key) for key in json.load(f))
    return labeled
