"""Fitness-AQA Shallow_Squat_Error_Dataset: labels, splits and person crops.

The dataset labels single bottom-of-squat frames with a binary shallow-depth verdict
(``0`` = no error, ``1`` = erroneous, per the release ReadMe), which makes it the
in-the-wild counterpart of the squat-depth verdict measured against mocap on Fit3D.
3,611 labelled crops over 623 videos; the official train/val/test id lists are
video-disjoint, so a split leak cannot inflate the comparison.

Sample ids are ``<video_id>_<frame_index>`` (e.g. ``37803_2_44`` = video ``37803_2``,
frame 44). Images live in ``images.zip`` under ``crops_unaligned/<id>.jpg`` as
aspect-preserving letterboxed person crops. We read those crops rather than
re-decoding the source videos: matching a crop back to its source frame is ambiguous
to +/-1 frame at the bottom of a squat (adjacent frames are near-identical), and every
arm must see byte-identical input for the 2D-vs-3D contrast to mean anything.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHALLOW_ROOT = (
    REPO_ROOT / "data" / "Fitness-AQA" / "Squat" / "Labeled_Dataset" / "Shallow_Squat_Error_Dataset"
)
CROP_PREFIX = "crops_unaligned/"
SPLITS = ("train", "val", "test")


def label_file(root: Path = DEFAULT_SHALLOW_ROOT) -> Path:
    return root / "labels_shallow_depth.json"


def split_file(split: str, root: Path = DEFAULT_SHALLOW_ROOT) -> Path:
    return root / "splits" / f"{split}_ids.json"


def images_zip(root: Path = DEFAULT_SHALLOW_ROOT) -> Path:
    return root / "images.zip"


def load_labels(root: Path = DEFAULT_SHALLOW_ROOT) -> dict[str, int]:
    """Sample id -> 0 (no error) / 1 (shallow squat)."""
    with open(label_file(root), encoding="utf-8") as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def load_split(split: str, root: Path = DEFAULT_SHALLOW_ROOT) -> list[str]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    with open(split_file(split, root), encoding="utf-8") as f:
        return [str(s) for s in json.load(f)]


def video_id(sample_id: str) -> str:
    """``37803_2_44`` -> ``37803_2`` (the clip the frame was cut from)."""
    return sample_id.rsplit("_", 1)[0]


def frame_index(sample_id: str) -> int:
    return int(sample_id.rsplit("_", 1)[1])


def load_manifest(root: Path = DEFAULT_SHALLOW_ROOT) -> list[dict]:
    """Flat manifest of every labelled sample: id, split, label, video_id."""
    labels = load_labels(root)
    rows: list[dict] = []
    for split in SPLITS:
        for sid in load_split(split, root):
            if sid not in labels:
                continue
            rows.append({"id": sid, "split": split, "label": labels[sid], "video_id": video_id(sid)})
    return rows


def iter_crops(sample_ids: list[str], root: Path = DEFAULT_SHALLOW_ROOT) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(sample_id, bgr_image)`` for each id present in ``images.zip``.

    Ids whose crop is missing from the archive are skipped silently; callers track
    coverage through the NaN rows their extractor writes.
    """
    import cv2  # local: heavy optional dep, and this module is imported by pure-numpy tests

    with zipfile.ZipFile(images_zip(root)) as z:
        available = set(z.namelist())
        for sid in sample_ids:
            name = f"{CROP_PREFIX}{sid}.jpg"
            if name not in available:
                continue
            buf = np.frombuffer(z.read(name), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue
            yield sid, img
