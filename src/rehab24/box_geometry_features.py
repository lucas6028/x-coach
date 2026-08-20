"""Build the zero-parameter box-geometry control for REHAB24-6.

On Fitness-AQA squats this descriptor -- twelve numbers, no pixels -- reached 65.6% of
``full_frame``'s above-chance signal, and the whole gym scene added nothing measurable
on top of it (``notes/videomae_b1_repeated_splits_results.md``). Two things make
REHAB24-6 the right place to ask whether that generalises:

*It ships ``person_id``.* Ten subjects, and the repo already scores every other feature
set leave-one-subject-out. The subject leakage that Fitness-AQA cannot exclude -- no
participant mapping, so the same athlete may sit on both sides of a split -- simply
does not arise.

*The scene is constant.* One lab, two fixed cameras, the same lighting. In Fitness-AQA
"scene" and "box" are confounded because gyms differ; here there is only one scene, so
whatever this control scores cannot be scene recognition.

The box comes from the dataset's own 2D skeletons (``skeleton_2d_path``), which are
mocap-derived and already expressed in each camera's pixel frame -- including cam18's
portrait transposition -- so no pose estimator is involved and no re-extraction is
needed. Points are taken over the repetition's own frame range, matching how every
other REHAB24-6 feature is segmented.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.video.box_geometry import box_from_points, box_geometry_feature

#: Non-finite joints do occur; they are dropped rather than allowed to poison min/max.
REQUIRED_COLUMNS = ("sample_id", "split", "skeleton_2d_path", "video_path", "first_frame", "last_frame")


def read_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{manifest_path} has no rows.")
    missing = [column for column in REQUIRED_COLUMNS if column not in rows[0]]
    if missing:
        raise ValueError(f"{manifest_path} lacks required columns {missing}.")
    return rows


def frame_size(video_path: Path) -> tuple[int, int]:
    """``(width, height)`` of a video, read from the container.

    cam18 is stored transposed (1080x1920) and its skeleton is expressed in that same
    portrait frame, so the size must come from the file rather than being assumed.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read frame size from {video_path}.")
    return width, height


def segment_points(skeleton: np.ndarray, first_frame: int, last_frame: int) -> tuple[np.ndarray, np.ndarray]:
    """Finite (x, y) joint positions inside one repetition's frame range.

    The range is inclusive of ``last_frame``, matching the dataset's own segmentation
    and every other REHAB24-6 feature builder. Out-of-range indices are clipped rather
    than raising: a few segments end on the final frame.
    """
    if skeleton.ndim != 3 or skeleton.shape[-1] < 2:
        raise ValueError(f"Expected a (frames, joints, 2+) skeleton, got {skeleton.shape}.")

    start = max(int(first_frame), 0)
    stop = min(int(last_frame) + 1, skeleton.shape[0])
    if stop <= start:
        raise ValueError(f"Empty frame range [{first_frame}, {last_frame}] for a {skeleton.shape[0]}-frame skeleton.")

    window = skeleton[start:stop, :, :2]
    xs = window[..., 0].reshape(-1)
    ys = window[..., 1].reshape(-1)
    finite = np.isfinite(xs) & np.isfinite(ys)
    return xs[finite], ys[finite]


def build_feature(row: dict, dataset_root: Path, size_cache: dict[str, tuple[int, int]]) -> np.ndarray:
    skeleton = np.load(dataset_root / row["skeleton_2d_path"])
    xs, ys = segment_points(skeleton, int(row["first_frame"]), int(row["last_frame"]))

    video_path = row["video_path"]
    if video_path not in size_cache:
        size_cache[video_path] = frame_size(dataset_root / video_path)
    width, height = size_cache[video_path]

    box = box_from_points(xs, ys, width, height)
    if box is None:
        raise ValueError(f"{row['sample_id']}: no finite 2D joints in its frame range.")

    n_frames = int(row["last_frame"]) - int(row["first_frame"]) + 1
    return box_geometry_feature(box, frame_width=width, frame_height=height, n_frames=n_frames)


def save_feature(output_dir: Path, row: dict, feature: np.ndarray) -> Path:
    path = output_dir / row["split"] / f"{row['sample_id']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        video_feature=feature,
        sample_id=np.asarray(row["sample_id"]),
        video_id=np.asarray(row["video_id"]),
        exercise_id=np.asarray(row["exercise_id"]),
        person_id=np.asarray(row["person_id"]),
        camera=np.asarray(row["camera"]),
        correctness=np.asarray(int(row["correctness"])),
        first_frame=np.asarray(int(row["first_frame"])),
        last_frame=np.asarray(int(row["last_frame"])),
    )
    return path
