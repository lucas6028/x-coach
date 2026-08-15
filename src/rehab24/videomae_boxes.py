"""One fixed person box per REHAB24-6 source video, from the dataset's own mocap 2D.

``notes/rehab24_videomae_framing_validation_plan.md`` §4.1 makes this a design
requirement rather than an implementation detail. A REHAB24-6 video holds ~16
repetitions, and a box computed over a single repetition's frame range encodes how far
that repetition travelled -- which is a function of its correctness. ``person_crop`` and
``background_only`` would then see label-correlated geometry *in the pixel transform
itself*, before any model ran. So the union is taken over the WHOLE video and the same
rectangle is applied to every repetition and every sampled frame of it.

The points come from ``skeleton_2d_path``: mocap-derived joints already expressed in
each camera's pixel frame, including cam18's portrait transposition. No pose estimator
is involved, which is what makes this a research control -- and also what stops it from
being a deployment story: nothing here says a detector would find the same box.

Rounding is delegated to ``box_from_points`` and expansion to ``expand_box`` so a box
built here is the same object as the one ``box_geometry_features`` and the Fitness-AQA
variants build; only the frame range differs.

The index is written once and read by the extractor. A missing entry is fatal there,
never a silent fall back to the untouched frame: that fallthrough is the failure that
already cost this study half a control arm (``squat_video_variants`` docstring).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from src.rehab24.dataset import DEFAULT_DATA_ROOT, DEFAULT_PROCESSED_ROOT, load_manifest, resolve_data_path
from src.video.box_geometry import box_from_points
from src.video.squat_video_variants import expand_box
from src.video.variant_geometry import DEFAULT_MARGIN, Box

#: Stamped into the index and into every bundle's provenance, so a feature dir states
#: where its rectangle came from instead of leaving it to a note.
BOX_SOURCE = "rehab24_mocap_skeleton_2d_full_video"


def video_box(skeleton: np.ndarray, frame_width: int, frame_height: int) -> Box | None:
    """Union box over EVERY frame of one video's 2D skeleton.

    Non-finite joints occur in this dataset and are dropped rather than allowed to
    poison the min/max -- the same treatment ``box_geometry_features`` gives them.
    """
    if skeleton.ndim != 3 or skeleton.shape[-1] < 2:
        raise ValueError(f"Expected a (frames, joints, 2+) skeleton, got {skeleton.shape}.")

    xs = skeleton[..., 0].reshape(-1)
    ys = skeleton[..., 1].reshape(-1)
    finite = np.isfinite(xs) & np.isfinite(ys)
    return box_from_points(xs[finite], ys[finite], frame_width, frame_height)


def video_rows(manifest_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """One representative manifest row per source video, ordered by path.

    The box is a property of the video, not of the repetition, so the ~16 rows sharing
    a ``video_path`` collapse to one piece of work.
    """
    seen: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        seen.setdefault(row["video_path"], row)
    return [seen[path] for path in sorted(seen)]


def describe_video(
    row: dict[str, str],
    data_root: Path,
    margin: float = DEFAULT_MARGIN,
) -> dict:
    """The index entry for one source video, including the consistency checks.

    Frame size comes from the video container and the skeleton length from the ``.npy``
    so the two can be compared: plan §6.1 requires that a camera's skeleton and its
    video actually describe the same footage before any framing claim rests on them. A
    box built against the wrong orientation would silently crop the wrong region.
    """
    import cv2  # imported here so box arithmetic stays importable without OpenCV

    video_path = resolve_data_path(data_root, row["video_path"])
    capture = cv2.VideoCapture(str(video_path))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read frame size from {video_path}.")

    skeleton = np.load(resolve_data_path(data_root, row["skeleton_2d_path"]))
    landmark_box = video_box(skeleton, width, height)
    if landmark_box is None:
        raise ValueError(f"{row['video_path']}: no finite 2D joints anywhere in the video.")

    expanded = expand_box(landmark_box, width, height, margin)
    xs = skeleton[..., 0].reshape(-1)
    ys = skeleton[..., 1].reshape(-1)
    finite = int(np.count_nonzero(np.isfinite(xs) & np.isfinite(ys)))

    return {
        "video_path": row["video_path"],
        "camera": row["camera"],
        "frame_size": [width, height],
        "video_frames": video_frames,
        "skeleton_frames": int(skeleton.shape[0]),
        "n_finite_joint_points": finite,
        "landmark_box": list(landmark_box.as_tuple()),
        "box": list(expanded.as_tuple()),
    }


def build_index(
    manifest_rows: Sequence[dict[str, str]],
    data_root: Path,
    margin: float = DEFAULT_MARGIN,
) -> dict:
    entries = {}
    for row in video_rows(manifest_rows):
        entry = describe_video(row, data_root, margin)
        entries[row["video_path"]] = entry
    return {
        "box_source": BOX_SOURCE,
        "margin": margin,
        "n_videos": len(entries),
        "videos": entries,
    }


def write_index(index: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def load_index(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def box_for_video(index: dict, video_path: str) -> Box:
    """The fixed expanded box for one video. Raises rather than returning ``None``.

    Fail closed, per plan §5.1: a box variant that cannot find its rectangle must stop
    the extraction. Returning ``None`` here would reach ``apply_variant``, which treats
    a null box as "leave the video untouched" and would write full-frame features into
    a control arm's directory under that arm's name.
    """
    entry = index.get("videos", {}).get(video_path)
    if entry is None or not entry.get("box"):
        raise KeyError(
            f"No fixed person box for {video_path!r} in the box index. "
            "Rebuild it with scripts/rehab24/build_videomae_boxes.py; do NOT fall back to the full frame."
        )
    return Box(*(int(value) for value in entry["box"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fixed per-video person boxes for REHAB24-6 framing arms.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_PROCESSED_ROOT / "videomae_boxes.json")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    index = build_index(rows, args.data_root, args.margin)
    write_index(index, args.output)

    print(f"Wrote {index['n_videos']} fixed video boxes to {args.output} (margin {args.margin})")
    for camera in sorted({entry["camera"] for entry in index["videos"].values()}):
        entries = [entry for entry in index["videos"].values() if entry["camera"] == camera]
        sizes = sorted({tuple(entry["frame_size"]) for entry in entries})
        mismatched = [e["video_path"] for e in entries if e["video_frames"] != e["skeleton_frames"]]
        print(f"  {camera}: {len(entries)} videos, frame sizes {sizes}")
        if mismatched:
            print(f"    WARNING: {len(mismatched)} videos where skeleton and video frame counts disagree")
            for path in mismatched[:5]:
                entry = index["videos"][path]
                print(f"      {path}: video {entry['video_frames']} vs skeleton {entry['skeleton_frames']}")


if __name__ == "__main__":
    main()
