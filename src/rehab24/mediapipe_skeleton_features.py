"""Extract REHAB24-6 repetition-level skeleton features from RGB video via MediaPipe.

This mirrors :mod:`src.rehab24.skeleton_features` but, instead of reading the
dataset's Vicon mocap `.npy` skeletons, it estimates joints from the RGB videos
with MediaPipe Pose (BlazePose 33-landmark layout). The same geometric pipeline
(root-centering, scale normalization, velocity, time-series summary) is reused so
the resulting `.npz` bundles are drop-in compatible with the correctness
classifier: train with `--feature-dir .../mediapipe_skeleton_features`.

The goal is to quantify how close monocular estimated skeletons get to the
high-fidelity mocap baseline, and to put RGB back into joint space (the
representation that generalizes) rather than relying on opaque VideoMAE
embeddings.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from src.rehab24.dataset import DEFAULT_DATA_ROOT, DEFAULT_PROCESSED_ROOT, load_manifest, resolve_data_path
from src.rehab24.skeleton_features import (
    add_velocity,
    distance,
    feature_output_path,
    frame_bounds,
    save_feature,
    summarize_time_series,
)

# BlazePose / MediaPipe Pose 33-landmark indices.
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_NUM_LANDMARKS = 33

SCALE_PAIRS = ((MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER), (MP_LEFT_HIP, MP_RIGHT_HIP))


def interpolate_missing(series: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN gaps (missing detections) along the time axis.

    `series` has shape (T, J, C). Each joint/coordinate channel is interpolated
    independently; leading/trailing gaps are filled with the nearest value.
    Channels that are NaN for every frame are left as NaN (the summary step maps
    those to zero).
    """
    coords = series.astype(np.float32, copy=True)
    total = coords.shape[0]
    if total == 0:
        return coords
    times = np.arange(total, dtype=np.float32)
    flat = coords.reshape(total, -1)
    for channel in range(flat.shape[1]):
        column = flat[:, channel]
        valid = np.isfinite(column)
        if valid.all() or not valid.any():
            continue
        flat[:, channel] = np.interp(times, times[valid], column[valid])
    return flat.reshape(coords.shape)


def smooth_series(series: np.ndarray, window_length: int, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smooth each joint/channel along the time axis (axis 0).

    `series` has shape (T, J, C). This is the R1 temporal-smoothing step: it runs
    after :func:`interpolate_missing` (so gaps are already filled) and before
    `add_velocity` (so the differentiated velocity channels — most sensitive to
    per-frame jitter — see the denoised signal). It directly targets the
    estimation jitter that monocular pose introduces.

    Behaviour:
    - `window_length <= 1` is a no-op (smoothing disabled).
    - The window is forced odd (Savitzky-Golay requires it) and clamped to the
      series length, so short repetitions degrade gracefully instead of erroring.
    - Channels that are NaN for every frame (joints the summary maps to zero) are
      passed through untouched.

    Only the monocular path uses this; Vicon mocap is already clean and smoothing
    would erase the squat-bottom turn that carries the correctness signal.
    """
    coords = np.array(series, dtype=np.float32, copy=True)
    total = coords.shape[0]
    win = int(window_length)
    if win <= 1 or total < 3:
        return coords
    if win % 2 == 0:
        win += 1
    if win > total:
        win = total if total % 2 == 1 else total - 1
    if win <= polyorder or win < 3:
        return coords

    from scipy.signal import savgol_filter  # lazy: keeps feature math importable without scipy

    # interpolate_missing leaves each channel either fully finite or fully NaN
    # (never-detected joints). Savitzky-Golay rejects NaNs, so smooth only the
    # finite channels and pass the all-NaN ones through unchanged.
    flat = coords.reshape(total, -1)
    finite_cols = np.isfinite(flat).all(axis=0)
    if finite_cols.any():
        flat[:, finite_cols] = savgol_filter(flat[:, finite_cols], window_length=win, polyorder=polyorder, axis=0)
    return flat.reshape(coords.shape).astype(np.float32)


def midpoint(points: np.ndarray, a: int, b: int) -> np.ndarray:
    return (points[:, a, :] + points[:, b, :]) * 0.5


def normalize_points_root(points: np.ndarray, root: np.ndarray, scale_pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """Center on an explicit per-frame root point and scale by joint spans.

    Mirrors :func:`src.rehab24.skeleton_features.normalize_points` but accepts a
    root coordinate array (MediaPipe has no single hip joint, so we use the
    mid-hip) instead of a root joint index.
    """
    coords = points.astype(np.float32, copy=False)
    centered = coords - root[:, None, :]

    scales = []
    for a, b in scale_pairs:
        pair_scale = distance(coords, a, b)
        pair_scale = np.where(np.isfinite(pair_scale) & (pair_scale > 1e-6), pair_scale, np.nan)
        scales.append(pair_scale)
    stacked = np.stack(scales, axis=1)
    frame_scale = np.full(stacked.shape[0], np.nan, dtype=np.float32)
    valid_rows = np.isfinite(stacked).any(axis=1)
    frame_scale[valid_rows] = np.nanmedian(stacked[valid_rows], axis=1)
    finite_frame_scale = frame_scale[np.isfinite(frame_scale)]
    fallback = float(np.median(finite_frame_scale)) if finite_frame_scale.size else 1.0
    if not np.isfinite(fallback) or fallback <= 1e-6:
        fallback = 1.0
    frame_scale = np.where(np.isfinite(frame_scale) & (frame_scale > 1e-6), frame_scale, fallback)
    return centered / frame_scale[:, None, None]


def extract_feature_vector(world: np.ndarray, image: np.ndarray, first_frame: int, last_frame: int) -> np.ndarray:
    """Build a repetition feature vector from MediaPipe world (3D) + image (2D) landmarks."""
    total_frames = min(int(world.shape[0]), int(image.shape[0]))
    start, stop = frame_bounds(first_frame, last_frame, total_frames)
    segment_world = world[start:stop, :, :3]
    segment_image = image[start:stop, :, :2]

    norm_world = add_velocity(
        normalize_points_root(segment_world, midpoint(segment_world, MP_LEFT_HIP, MP_RIGHT_HIP), SCALE_PAIRS)
    )
    norm_image = add_velocity(
        normalize_points_root(segment_image, midpoint(segment_image, MP_LEFT_HIP, MP_RIGHT_HIP), SCALE_PAIRS)
    )
    return np.concatenate([summarize_time_series(norm_world), summarize_time_series(norm_image)], axis=0)


def landmarks_from_video(
    video_path: Path, model_complexity: int = 2, frame_stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Run MediaPipe Pose over a video, returning (world_xyz, image_xy) arrays.

    Shapes: world (T, 33, 3) metric world landmarks, image (T, 33, 2) normalized
    image landmarks. Missing-detection frames are NaN-filled then interpolated.

    `frame_stride > 1` only runs pose inference on every Nth frame (the expensive
    step) and leaves the rest NaN for interpolation. The time axis keeps full
    length T so manifest `first_frame`/`last_frame` indexing stays valid.
    """
    import cv2  # imported lazily so feature math stays importable without OpenCV
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    world_frames: list[np.ndarray] = []
    image_frames: list[np.ndarray] = []
    nan_world = np.full((MP_NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
    nan_image = np.full((MP_NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
    stride = max(int(frame_stride), 1)

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_idx % stride != 0:
                world_frames.append(nan_world.copy())
                image_frames.append(nan_image.copy())
                frame_idx += 1
                continue

            frame.flags.writeable = False
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_world_landmarks:
                world_frames.append(
                    np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark], dtype=np.float32)
                )
            else:
                world_frames.append(nan_world.copy())

            if results.pose_landmarks:
                image_frames.append(
                    np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark], dtype=np.float32)
                )
            else:
                image_frames.append(nan_image.copy())
            frame_idx += 1

    cap.release()
    if not world_frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    world = interpolate_missing(np.stack(world_frames, axis=0))
    image = interpolate_missing(np.stack(image_frames, axis=0))
    return world, image


def landmark_cache_path(cache_dir: Path, video_path: str) -> Path:
    return cache_dir / f"{Path(video_path).stem}.npz"


def load_or_extract_landmarks(
    data_root: Path,
    video_path: str,
    model_complexity: int,
    frame_stride: int,
    cache_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return interpolated (world, image) landmarks, using a per-video cache.

    MediaPipe inference is the expensive step (~minutes/video on CPU) and is
    independent of the cheap smoothing/feature math downstream. Caching the raw
    interpolated landmarks lets us run MediaPipe once and then regenerate many
    smoothing variants (the R1 cutoff sweep) in seconds. The cache stores the
    exact output of :func:`landmarks_from_video`, so no-smoothing features built
    from the cache are bit-for-bit identical to the direct path.
    """
    if cache_dir is not None:
        cache_path = landmark_cache_path(cache_dir, video_path)
        if cache_path.exists():
            with np.load(cache_path) as data:
                return data["world"], data["image"]

    world, image = landmarks_from_video(
        resolve_data_path(data_root, video_path), model_complexity=model_complexity, frame_stride=frame_stride
    )

    if cache_dir is not None:
        cache_path = landmark_cache_path(cache_dir, video_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, world=world, image=image)
    return world, image


def process_one_video(
    data_root: Path,
    output_dir: Path,
    video_path: str,
    rows: list[dict[str, str]],
    model_complexity: int,
    frame_stride: int,
    smooth_window: int = 0,
    smooth_polyorder: int = 2,
    cache_dir: Path | None = None,
) -> int:
    """Run MediaPipe once on a source video and write a feature bundle per rep.

    Top-level (picklable) so it can be dispatched to a process pool. Videos are
    independent, so workers never contend on the same output file. Smoothing (R1)
    is applied once at the full-video level — before slicing into repetitions — so
    the Savitzky-Golay window has full temporal context and edge effects land only
    at the clip boundaries, not at every rep boundary.
    """
    world, image = load_or_extract_landmarks(data_root, video_path, model_complexity, frame_stride, cache_dir)
    if smooth_window and smooth_window > 1:
        world = smooth_series(world, smooth_window, smooth_polyorder)
        image = smooth_series(image, smooth_window, smooth_polyorder)
    written = 0
    for row in rows:
        feature = extract_feature_vector(
            world=world,
            image=image,
            first_frame=int(row["first_frame"]),
            last_frame=int(row["last_frame"]),
        )
        save_feature(feature_output_path(output_dir, row["split"], row["sample_id"]), row, feature)
        written += 1
    return written


def extract_features_for_manifest(
    data_root: Path,
    manifest_path: Path,
    output_dir: Path,
    limit: int | None = None,
    video_limit: int | None = None,
    overwrite: bool = False,
    model_complexity: int = 2,
    frame_stride: int = 1,
    num_workers: int = 1,
    smooth_window: int = 0,
    smooth_polyorder: int = 2,
    cache_dir: Path | None = None,
) -> int:
    """Extract MediaPipe skeleton features for every manifest row.

    MediaPipe is run once per source video (videos are shared across repetitions),
    then each repetition segment is sliced and saved as an `.npz` bundle matching
    the mocap skeleton-feature format. With `num_workers > 1`, videos are processed
    in parallel across processes — this is accuracy-neutral (each video's result is
    identical to the serial run); only `frame_stride`/`model_complexity` trade
    fidelity for speed.
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

    # Resume support: drop reps whose bundle already exists, skip fully-done videos.
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

    written = 0
    if num_workers <= 1:
        for index, (video_path, pending) in enumerate(pending_by_video, start=1):
            print(f"[{index}/{total}] MediaPipe on {video_path} ({len(pending)} reps)...", flush=True)
            written += process_one_video(
                data_root, output_dir, video_path, pending, model_complexity, frame_stride,
                smooth_window, smooth_polyorder, cache_dir,
            )
        return written

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_one_video, data_root, output_dir, video_path, pending, model_complexity, frame_stride,
                smooth_window, smooth_polyorder, cache_dir,
            ): video_path
            for video_path, pending in pending_by_video
        }
        for done, future in enumerate(as_completed(futures), start=1):
            video_path = futures[future]
            written += future.result()
            print(f"[{done}/{total}] done {video_path}", flush=True)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract REHAB24-6 repetition-level skeleton features from RGB video using MediaPipe Pose."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "mediapipe_skeleton_features")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N manifest rows.")
    parser.add_argument("--video-limit", type=int, default=None, help="Process at most N source videos (for smoke tests).")
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=2,
        choices=(0, 1, 2),
        help="BlazePose model complexity (2 = most accurate, default).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Run pose inference every Nth frame and interpolate the rest. >1 trades accuracy for speed (default 1 = full fidelity).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel worker processes across videos. Accuracy-neutral; set to CPU core count for ~N x speedup.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=0,
        help="R1 temporal smoothing: Savitzky-Golay window in frames (odd; <=1 disables). "
        "Applied per joint/channel after interpolation, before velocity.",
    )
    parser.add_argument(
        "--smooth-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order for --smooth-window (default 2).",
    )
    parser.add_argument(
        "--landmark-cache",
        type=Path,
        default=None,
        help="Directory to cache/reuse raw interpolated per-video landmarks. "
        "MediaPipe runs once to fill it; later smoothing variants read the cache (no re-inference).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    written = extract_features_for_manifest(
        data_root=args.data_root,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        limit=args.limit,
        video_limit=args.video_limit,
        overwrite=args.overwrite,
        model_complexity=args.model_complexity,
        frame_stride=args.frame_stride,
        num_workers=args.num_workers,
        smooth_window=args.smooth_window,
        smooth_polyorder=args.smooth_polyorder,
        cache_dir=args.landmark_cache,
    )
    print(f"Wrote {written} feature files under {args.output_dir}")


if __name__ == "__main__":
    main()
