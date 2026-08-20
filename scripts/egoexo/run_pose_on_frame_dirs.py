"""Run MediaPipe Pose over directories of extracted EgoExo frames.

EgoExo-Fitness distributes preprocessed FRAMES, not videos, so `src/pose/process_videos.py`
(which opens a `cv2.VideoCapture`) cannot be pointed at them. This writes the identical JSON
schema -- `{"metadata": {...}, "frames": [{"frame_index", "landmarks", "world_landmarks"}]}` --
one file per input directory, so everything downstream of pose extraction is unchanged.

`static_image_mode=False` matches `process_videos.py`: the frames ARE a video, in order, and the
tracker's temporal prior is part of what production sees.

    .venv\\Scripts\\python.exe scripts/egoexo/run_pose_on_frame_dirs.py ^
      --frames-root <dir of {sample}__{view}/ dirs> --out <dir> [--shard 0 --shards 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EGOEXO_FPS = 30.0  # stated by the dataset README ("Preprocessed video frames in 30 fps")


def landmarks_to_list(landmarks) -> list[dict]:
    return [
        {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
        for lm in landmarks.landmark
    ]


def process_directory(directory: Path, output_path: Path, model_complexity: int) -> int:
    images = sorted(directory.glob("*.jpg"))
    if not images:
        return 0
    first = cv2.imread(str(images[0]))
    if first is None:
        return 0
    height, width = first.shape[:2]

    payload = {
        "metadata": {
            "fps": EGOEXO_FPS,
            "width": width,
            "height": height,
            "total_frames": len(images),
            "source": str(directory.name),
        },
        "frames": [],
    }

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for index, image_path in enumerate(images):
            image = cv2.imread(str(image_path))
            if image is None:
                payload["frames"].append(
                    {"frame_index": index, "landmarks": None, "world_landmarks": None}
                )
                continue
            image.flags.writeable = False
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frame = {"frame_index": index, "landmarks": None, "world_landmarks": None}
            if results.pose_landmarks:
                frame["landmarks"] = landmarks_to_list(results.pose_landmarks)
                if results.pose_world_landmarks:
                    frame["world_landmarks"] = landmarks_to_list(results.pose_world_landmarks)
            payload["frames"].append(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return len(images)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model-complexity", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    directories = sorted(p for p in args.frames_root.iterdir() if p.is_dir())
    mine = [d for i, d in enumerate(directories) if i % args.shards == args.shard]
    print(f"{len(mine)} of {len(directories)} directories in shard {args.shard}", flush=True)

    for directory in mine:
        output_path = args.out / f"{directory.name}.json"
        if output_path.exists():
            print(f"skip {directory.name} (exists)", flush=True)
            continue
        count = process_directory(directory, output_path, args.model_complexity)
        print(f"{directory.name}: {count} frames", flush=True)


if __name__ == "__main__":
    main()
