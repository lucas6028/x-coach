"""Extract REHAB24-6 repetition-level skeleton features from RGB video via MMPose.

A third estimated-skeleton comparison group alongside the Vicon mocap baseline
and the MediaPipe pseudo-3D features. MMPose / RTMPose (run through the
``src.pose.mmpose_pose_extraction`` helpers, defaulting to the Colab-GPU-friendly
``rtmlib`` runtime) produces **2D image keypoints only** — there is no learned
depth, unlike MediaPipe's BlazePose world landmarks. We map COCO-WholeBody-133 to
the MediaPipe-33 layout and run the *same* geometric pipeline as
:mod:`src.rehab24.mediapipe_skeleton_features`, but on the 2D image branch alone.

The resulting ``.npz`` bundles are drop-in compatible with the correctness
classifier and LOSO driver (they carry the ``video_feature`` key); train with
``--feature-dir .../mmpose_skeleton_features``. Because the feature is 2D-only,
its dimension is half of the MediaPipe bundle — the classifier reads the feature
dimension dynamically, so this is fine; just interpret the comparison as
"more-accurate 2D keypoints, no depth" vs MediaPipe's "pseudo-3D".

GPU note: unlike MediaPipe (CPU-only TFLite), MMPose/RTMPose genuinely uses the
GPU, so a single Colab GPU process is the right way to run this — no
multiprocessing needed (the GPU serializes the work anyway).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.pose.mmpose_pose_extraction import (
    coco_wholebody_to_mediapipe_landmarks,
    import_mmpose_inferencer,
    import_rtmlib_wholebody,
    instance_keypoints,
    prediction_instances,
    rtmlib_device,
    rtmlib_mode,
    rtmlib_primary_pose,
    select_primary_instance,
)
from src.rehab24.dataset import DEFAULT_DATA_ROOT, DEFAULT_PROCESSED_ROOT, load_manifest, resolve_data_path
from src.rehab24.mediapipe_skeleton_features import (
    MP_LEFT_HIP,
    MP_NUM_LANDMARKS,
    MP_RIGHT_HIP,
    SCALE_PAIRS,
    interpolate_missing,
    midpoint,
    normalize_points_root,
)
from src.rehab24.skeleton_features import (
    add_velocity,
    feature_output_path,
    frame_bounds,
    save_feature,
    summarize_time_series,
)


def build_inferencer(runtime: str, model: str, device: str) -> Any:
    """Construct the pose inferencer once and reuse it across all videos."""
    if runtime == "mmpose":
        MMPoseInferencer = import_mmpose_inferencer()
        return MMPoseInferencer(pose2d=model, device=device, show_progress=False)
    Wholebody = import_rtmlib_wholebody()
    return Wholebody(
        to_openpose=False,
        mode=rtmlib_mode(model),
        backend="onnxruntime",
        device=rtmlib_device(device),
    )


def _landmarks_to_image_row(landmark_dicts: list[dict[str, float]]) -> np.ndarray:
    """Convert the 33 normalized-landmark dicts to an (33, 2) array; blanks -> NaN.

    ``coco_wholebody_to_mediapipe_landmarks`` returns visibility 0 for unmapped /
    undetected joints (the blank sentinel). We mark those NaN so the time-axis
    interpolation can fill them, matching the MediaPipe extractor's handling.
    """
    row = np.full((MP_NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
    for joint, landmark in enumerate(landmark_dicts):
        if landmark["visibility"] > 0.0:
            row[joint] = (landmark["x"], landmark["y"])
    return row


def landmarks_from_video(
    video_path: Path,
    inferencer: Any,
    runtime: str,
    *,
    bbox_thr: float = 0.3,
) -> np.ndarray:
    """Run MMPose/RTMPose over a video, returning normalized image landmarks.

    Shape: image (T, 33, 2) with x/y normalized to the frame size. Missing-detection
    frames are NaN-filled then linearly interpolated along the time axis so the
    manifest ``first_frame``/``last_frame`` indexing stays valid.
    """
    import cv2  # imported lazily so feature math stays importable without OpenCV

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    nan_image = np.full((MP_NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
    image_frames: list[np.ndarray] = []
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if runtime == "mmpose":
                try:
                    result = next(inferencer(frame, bbox_thr=bbox_thr, return_vis=False))
                except StopIteration:
                    result = {"predictions": []}
                instance = select_primary_instance(prediction_instances(result))
                if instance is None:
                    image_frames.append(nan_image.copy())
                    continue
                keypoints, scores = instance_keypoints(instance)
            else:  # rtmlib
                keypoints, scores, _pose_score = rtmlib_primary_pose(*inferencer(frame))

            if keypoints is None:
                image_frames.append(nan_image.copy())
                continue
            landmark_dicts = coco_wholebody_to_mediapipe_landmarks(keypoints, scores, width, height)
            image_frames.append(_landmarks_to_image_row(landmark_dicts))
    finally:
        cap.release()

    if not image_frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return interpolate_missing(np.stack(image_frames, axis=0))


def extract_feature_vector(image: np.ndarray, first_frame: int, last_frame: int) -> np.ndarray:
    """Build a repetition feature vector from MMPose 2D image landmarks only."""
    total_frames = int(image.shape[0])
    start, stop = frame_bounds(first_frame, last_frame, total_frames)
    segment_image = image[start:stop, :, :2]
    norm_image = add_velocity(
        normalize_points_root(segment_image, midpoint(segment_image, MP_LEFT_HIP, MP_RIGHT_HIP), SCALE_PAIRS)
    )
    return summarize_time_series(norm_image)


def process_one_video(
    inferencer: Any,
    runtime: str,
    data_root: Path,
    output_dir: Path,
    video_path: str,
    rows: list[dict[str, str]],
    bbox_thr: float,
) -> int:
    image = landmarks_from_video(
        resolve_data_path(data_root, video_path), inferencer, runtime, bbox_thr=bbox_thr
    )
    written = 0
    for row in rows:
        feature = extract_feature_vector(image, int(row["first_frame"]), int(row["last_frame"]))
        save_feature(feature_output_path(output_dir, row["split"], row["sample_id"]), row, feature)
        written += 1
    return written


def extract_features_for_manifest(
    data_root: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    runtime: str = "rtmlib",
    model: str = "balanced",
    device: str = "cuda:0",
    bbox_thr: float = 0.3,
    limit: int | None = None,
    video_limit: int | None = None,
    overwrite: bool = False,
) -> int:
    """Extract MMPose 2D skeleton features for every manifest row.

    The pose model runs once per source video (videos are shared across
    repetitions); each repetition is then sliced and saved as an ``.npz`` bundle.
    Resumable: reps whose bundle already exists are skipped, fully-done videos are
    not re-run.
    """
    rows = load_manifest(manifest_path)
    if limit is not None:
        rows = rows[:limit]

    rows_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_video[row["video_path"]].append(row)

    video_paths = list(rows_by_video)
    if video_limit is not None:
        video_paths = video_paths[:video_limit]

    pending_by_video: list[tuple[str, list[dict[str, str]]]] = []
    for video_path in video_paths:
        pending = [
            row
            for row in rows_by_video[video_path]
            if overwrite or not feature_output_path(output_dir, row["split"], row["sample_id"]).exists()
        ]
        if pending:
            pending_by_video.append((video_path, pending))

    total = len(pending_by_video)
    if total == 0:
        return 0

    inferencer = build_inferencer(runtime, model, device)
    written = 0
    for index, (video_path, pending) in enumerate(pending_by_video, start=1):
        print(f"[{index}/{total}] {runtime} on {video_path} ({len(pending)} reps)...", flush=True)
        written += process_one_video(inferencer, runtime, data_root, output_dir, video_path, pending, bbox_thr)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract REHAB24-6 repetition-level 2D skeleton features from RGB video using MMPose/RTMPose."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "mmpose_skeleton_features")
    parser.add_argument(
        "--runtime",
        choices=("rtmlib", "mmpose"),
        default="rtmlib",
        help="Pose runtime. rtmlib is the Colab Python 3.12 + GPU friendly default.",
    )
    parser.add_argument(
        "--model",
        default="balanced",
        help="rtmlib: performance/balanced/lightweight. mmpose: model alias/config.",
    )
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu.")
    parser.add_argument("--bbox-thr", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N manifest rows.")
    parser.add_argument("--video-limit", type=int, default=None, help="Process at most N source videos (smoke test).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    written = extract_features_for_manifest(
        data_root=args.data_root,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        runtime=args.runtime,
        model=args.model,
        device=args.device,
        bbox_thr=args.bbox_thr,
        limit=args.limit,
        video_limit=args.video_limit,
        overwrite=args.overwrite,
    )
    print(f"Wrote {written} feature files under {args.output_dir}")


if __name__ == "__main__":
    main()
