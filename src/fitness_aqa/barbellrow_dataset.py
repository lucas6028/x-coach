"""Fitness-AQA BarbellRow frame-level fault labels: lumbar and torso-angle.

Frame-level, like the squat shallow set: ids are ``<video>_<seg>_<frame>`` with a binary
label, images packed in ``barbellrow_images_raw.zip`` under ``barbellrow_images_raw/<id>.jpg``.
Each fault has its OWN split directory (the two faults label overlapping but not identical
frame sets), so labels and splits are always requested per fault.

Faults by plane:
* ``lumbar``       -- lower-back rounding: sagittal spine curvature.
* ``torso_angle``  -- torso too upright / too horizontal: sagittal torso tilt.

Both are sagittal, so on a side-filmed row the depth axis carries little of either --
2D is expected to suffice, mirroring squat's shallow/knees_forward (sagittal) result.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BROW_ROOT = REPO_ROOT / "data" / "Fitness-AQA" / "BarbellRow" / "Labeled_Dataset"
CROP_PREFIX = "barbellrow_images_raw/"
SPLITS = ("train", "val", "test")
FAULTS = ("lumbar", "torso_angle")
_LABEL_FILE = {"lumbar": "labels_lumbar_error.json", "torso_angle": "labels_torso_angle_error.json"}
_SPLIT_DIR = {"lumbar": "Splits_Lumbar_Error", "torso_angle": "Splits_TorsoAngle_Error"}


def load_labels(fault: str, root: Path = DEFAULT_BROW_ROOT) -> dict[str, int]:
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; expected one of {FAULTS}")
    with open(root / "Labels" / _LABEL_FILE[fault], encoding="utf-8") as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def load_split(fault: str, split: str, root: Path = DEFAULT_BROW_ROOT) -> list[str]:
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    with open(root / "Splits" / _SPLIT_DIR[fault] / f"{split}_ids.json", encoding="utf-8") as f:
        return [str(s) for s in json.load(f)]


def video_id(sample_id: str) -> str:
    """``56067_3_51`` -> ``56067_3`` (the clip); used for video-level cluster bootstrap."""
    return sample_id.rsplit("_", 1)[0]


def load_manifest(fault: str, root: Path = DEFAULT_BROW_ROOT) -> list[dict]:
    labels = load_labels(fault, root)
    rows: list[dict] = []
    for split in SPLITS:
        for sid in load_split(fault, split, root):
            if sid not in labels:
                continue
            rows.append({"id": sid, "split": split, "label": labels[sid], "video_id": video_id(sid)})
    return rows


def all_sample_ids(root: Path = DEFAULT_BROW_ROOT) -> list[str]:
    """Union of every labelled id across both faults (what the pose extractor must cover)."""
    ids: set[str] = set()
    for fault in FAULTS:
        for split in SPLITS:
            ids |= set(load_split(fault, split, root))
    return sorted(ids)


def images_zip(root: Path = DEFAULT_BROW_ROOT) -> Path:
    return root / "barbellrow_images_raw.zip"


def iter_crops(sample_ids: list[str], root: Path = DEFAULT_BROW_ROOT) -> Iterator[tuple[str, np.ndarray]]:
    import cv2

    with zipfile.ZipFile(images_zip(root)) as z:
        available = set(z.namelist())
        for sid in sample_ids:
            name = f"{CROP_PREFIX}{sid}.jpg"
            if name not in available:
                continue
            img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                yield sid, img
