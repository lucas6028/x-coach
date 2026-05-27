from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_NAMES = ("train", "val", "test")
SUMMARY_STATS = ("mean", "std", "min", "max", "p10", "p25", "p50", "p75", "p90")

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

LANDMARK_COUNT = 33
VISIBILITY_THRESHOLD = 0.50
LOWER_BODY_LANDMARKS = (
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_HEEL,
    RIGHT_HEEL,
    LEFT_FOOT_INDEX,
    RIGHT_FOOT_INDEX,
)
REQUIRED_LOWER_BODY_LANDMARKS = (
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_FOOT_INDEX,
    RIGHT_FOOT_INDEX,
)

FRAME_FEATURE_NAMES = (
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_ankle_angle",
    "right_ankle_angle",
    "knee_angle_asymmetry",
    "hip_angle_asymmetry",
    "ankle_angle_asymmetry",
    "left_knee_line_deviation",
    "right_knee_line_deviation",
    "left_knee_line_abs_deviation",
    "right_knee_line_abs_deviation",
    "knee_width_to_hip_width",
    "knee_width_to_ankle_width",
    "left_knee_to_ankle_x",
    "right_knee_to_ankle_x",
    "left_knee_to_toe_x",
    "right_knee_to_toe_x",
    "left_knee_to_ankle_abs_x",
    "right_knee_to_ankle_abs_x",
    "left_knee_to_toe_abs_x",
    "right_knee_to_toe_abs_x",
    "avg_hip_y",
    "avg_knee_y",
    "avg_ankle_y",
    "hip_minus_knee_y",
    "hip_minus_ankle_y",
    "left_hip_knee_y_gap",
    "right_hip_knee_y_gap",
    "left_right_hip_y_diff",
    "left_right_knee_y_diff",
    "left_right_ankle_y_diff",
    "lower_body_visibility_mean",
    "lower_body_visibility_min",
    "valid_lower_body",
)


@dataclass(frozen=True)
class PoseFeatureRequest:
    split_name: str
    video_id: str
    pose_json_path: Path
    output_path: Path


def load_json_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}.")
    return [str(item) for item in data]


def parse_split_names(value: str) -> list[str]:
    split_names = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(split_names) - set(SPLIT_NAMES))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported splits: {', '.join(invalid)}")
    return split_names


def load_pose_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}.")
    return data


def landmarks_to_array(landmarks: object) -> np.ndarray | None:
    if not isinstance(landmarks, list) or len(landmarks) < LANDMARK_COUNT:
        return None

    array = np.full((LANDMARK_COUNT, 4), np.nan, dtype=np.float32)
    for index, landmark in enumerate(landmarks[:LANDMARK_COUNT]):
        if not isinstance(landmark, dict):
            continue
        try:
            array[index, 0] = float(landmark.get("x", np.nan))
            array[index, 1] = float(landmark.get("y", np.nan))
            array[index, 2] = float(landmark.get("z", np.nan))
            array[index, 3] = float(landmark.get("visibility", np.nan))
        except (TypeError, ValueError):
            continue
    return array


def has_valid_point(points: np.ndarray | None, index: int, min_visibility: float = VISIBILITY_THRESHOLD) -> bool:
    if points is None:
        return False
    xyz = points[index, :3]
    visibility = points[index, 3]
    return bool(np.all(np.isfinite(xyz)) and np.isfinite(visibility) and visibility >= min_visibility)


def valid_lower_body_count(points: np.ndarray | None) -> int:
    return sum(has_valid_point(points, index) for index in REQUIRED_LOWER_BODY_LANDMARKS)


def choose_geometry_points(world_points: np.ndarray | None, image_points: np.ndarray | None) -> np.ndarray | None:
    if valid_lower_body_count(world_points) >= 6:
        return world_points
    if valid_lower_body_count(image_points) >= 6:
        return image_points
    return world_points if world_points is not None else image_points


def point(points: np.ndarray | None, index: int, dims: int = 3) -> np.ndarray | None:
    if not has_valid_point(points, index):
        return None
    values = points[index, :dims].astype(np.float32, copy=False)
    if not np.all(np.isfinite(values)):
        return None
    return values


def distance(points: np.ndarray | None, a: int, b: int, dims: int = 2) -> float:
    pa = point(points, a, dims=dims)
    pb = point(points, b, dims=dims)
    if pa is None or pb is None:
        return np.nan
    return float(np.linalg.norm(pa - pb))


def angle_degrees(points: np.ndarray | None, a: int, b: int, c: int) -> float:
    pa = point(points, a, dims=3)
    pb = point(points, b, dims=3)
    pc = point(points, c, dims=3)
    if pa is None or pb is None or pc is None:
        return np.nan

    ba = pa - pb
    bc = pc - pb
    denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denominator <= 1e-8:
        return np.nan
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def signed_line_distance_2d(points: np.ndarray | None, p: int, a: int, b: int, scale: float) -> float:
    pp = point(points, p, dims=2)
    pa = point(points, a, dims=2)
    pb = point(points, b, dims=2)
    if pp is None or pa is None or pb is None or not np.isfinite(scale) or scale <= 1e-8:
        return np.nan

    line = pb - pa
    offset = pp - pa
    denominator = float(np.linalg.norm(line))
    if denominator <= 1e-8:
        return np.nan
    cross = float(line[0] * offset[1] - line[1] * offset[0])
    return cross / denominator / scale


def normalized_x_offset(points: np.ndarray | None, a: int, b: int, scale: float) -> float:
    pa = point(points, a, dims=2)
    pb = point(points, b, dims=2)
    if pa is None or pb is None or not np.isfinite(scale) or scale <= 1e-8:
        return np.nan
    return float((pa[0] - pb[0]) / scale)


def average_coordinate(points: np.ndarray | None, indices: Sequence[int], axis: int) -> float:
    values: list[float] = []
    for index in indices:
        landmark = point(points, index, dims=3)
        if landmark is not None:
            values.append(float(landmark[axis]))
    if not values:
        return np.nan
    return float(np.mean(values))


def absolute_difference(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return np.nan
    return float(abs(left - right))


def safe_scale(image_points: np.ndarray | None) -> float:
    hip_width = distance(image_points, LEFT_HIP, RIGHT_HIP, dims=2)
    if np.isfinite(hip_width) and hip_width > 1e-6:
        return hip_width
    shoulder_width = distance(image_points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
    if np.isfinite(shoulder_width) and shoulder_width > 1e-6:
        return shoulder_width
    return 1.0


def lower_body_visibility(points: np.ndarray | None) -> tuple[float, float]:
    if points is None:
        return 0.0, 0.0
    visibility = points[list(LOWER_BODY_LANDMARKS), 3]
    finite_visibility = visibility[np.isfinite(visibility)]
    if finite_visibility.size == 0:
        return 0.0, 0.0
    return float(np.mean(finite_visibility)), float(np.min(finite_visibility))


def frame_has_pose(frame: object) -> bool:
    if not isinstance(frame, dict):
        return False
    return landmarks_to_array(frame.get("landmarks")) is not None or landmarks_to_array(frame.get("world_landmarks")) is not None


def compute_frame_features(frame: object) -> np.ndarray:
    if not isinstance(frame, dict):
        return np.full(len(FRAME_FEATURE_NAMES), np.nan, dtype=np.float32)

    image_points = landmarks_to_array(frame.get("landmarks"))
    world_points = landmarks_to_array(frame.get("world_landmarks"))
    geometry_points = choose_geometry_points(world_points=world_points, image_points=image_points)
    scale = safe_scale(image_points)

    left_knee_angle = angle_degrees(geometry_points, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    right_knee_angle = angle_degrees(geometry_points, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    left_hip_angle = angle_degrees(geometry_points, LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
    right_hip_angle = angle_degrees(geometry_points, RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
    left_ankle_angle = angle_degrees(geometry_points, LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX)
    right_ankle_angle = angle_degrees(geometry_points, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX)

    left_knee_line_deviation = signed_line_distance_2d(
        image_points,
        LEFT_KNEE,
        LEFT_HIP,
        LEFT_ANKLE,
        scale=scale,
    )
    right_knee_line_deviation = signed_line_distance_2d(
        image_points,
        RIGHT_KNEE,
        RIGHT_HIP,
        RIGHT_ANKLE,
        scale=scale,
    )

    hip_width = distance(image_points, LEFT_HIP, RIGHT_HIP, dims=2)
    knee_width = distance(image_points, LEFT_KNEE, RIGHT_KNEE, dims=2)
    ankle_width = distance(image_points, LEFT_ANKLE, RIGHT_ANKLE, dims=2)
    knee_width_to_hip_width = (
        knee_width / hip_width
        if np.isfinite(knee_width) and np.isfinite(hip_width) and hip_width > 1e-8
        else np.nan
    )
    knee_width_to_ankle_width = (
        knee_width / ankle_width
        if np.isfinite(knee_width) and np.isfinite(ankle_width) and ankle_width > 1e-8
        else np.nan
    )

    left_knee_to_ankle_x = normalized_x_offset(image_points, LEFT_KNEE, LEFT_ANKLE, scale=scale)
    right_knee_to_ankle_x = normalized_x_offset(image_points, RIGHT_KNEE, RIGHT_ANKLE, scale=scale)
    left_knee_to_toe_x = normalized_x_offset(image_points, LEFT_KNEE, LEFT_FOOT_INDEX, scale=scale)
    right_knee_to_toe_x = normalized_x_offset(image_points, RIGHT_KNEE, RIGHT_FOOT_INDEX, scale=scale)

    avg_hip_y = average_coordinate(image_points, (LEFT_HIP, RIGHT_HIP), axis=1)
    avg_knee_y = average_coordinate(image_points, (LEFT_KNEE, RIGHT_KNEE), axis=1)
    avg_ankle_y = average_coordinate(image_points, (LEFT_ANKLE, RIGHT_ANKLE), axis=1)
    left_hip_y = average_coordinate(image_points, (LEFT_HIP,), axis=1)
    right_hip_y = average_coordinate(image_points, (RIGHT_HIP,), axis=1)
    left_knee_y = average_coordinate(image_points, (LEFT_KNEE,), axis=1)
    right_knee_y = average_coordinate(image_points, (RIGHT_KNEE,), axis=1)
    left_ankle_y = average_coordinate(image_points, (LEFT_ANKLE,), axis=1)
    right_ankle_y = average_coordinate(image_points, (RIGHT_ANKLE,), axis=1)

    visibility_mean, visibility_min = lower_body_visibility(image_points)
    valid_lower_body = float(valid_lower_body_count(image_points) >= 6 or valid_lower_body_count(world_points) >= 6)

    values = [
        left_knee_angle,
        right_knee_angle,
        left_hip_angle,
        right_hip_angle,
        left_ankle_angle,
        right_ankle_angle,
        absolute_difference(left_knee_angle, right_knee_angle),
        absolute_difference(left_hip_angle, right_hip_angle),
        absolute_difference(left_ankle_angle, right_ankle_angle),
        left_knee_line_deviation,
        right_knee_line_deviation,
        abs(left_knee_line_deviation) if np.isfinite(left_knee_line_deviation) else np.nan,
        abs(right_knee_line_deviation) if np.isfinite(right_knee_line_deviation) else np.nan,
        knee_width_to_hip_width,
        knee_width_to_ankle_width,
        left_knee_to_ankle_x,
        right_knee_to_ankle_x,
        left_knee_to_toe_x,
        right_knee_to_toe_x,
        abs(left_knee_to_ankle_x) if np.isfinite(left_knee_to_ankle_x) else np.nan,
        abs(right_knee_to_ankle_x) if np.isfinite(right_knee_to_ankle_x) else np.nan,
        abs(left_knee_to_toe_x) if np.isfinite(left_knee_to_toe_x) else np.nan,
        abs(right_knee_to_toe_x) if np.isfinite(right_knee_to_toe_x) else np.nan,
        avg_hip_y,
        avg_knee_y,
        avg_ankle_y,
        avg_hip_y - avg_knee_y if np.isfinite(avg_hip_y) and np.isfinite(avg_knee_y) else np.nan,
        avg_hip_y - avg_ankle_y if np.isfinite(avg_hip_y) and np.isfinite(avg_ankle_y) else np.nan,
        left_hip_y - left_knee_y if np.isfinite(left_hip_y) and np.isfinite(left_knee_y) else np.nan,
        right_hip_y - right_knee_y if np.isfinite(right_hip_y) and np.isfinite(right_knee_y) else np.nan,
        absolute_difference(left_hip_y, right_hip_y),
        absolute_difference(left_knee_y, right_knee_y),
        absolute_difference(left_ankle_y, right_ankle_y),
        visibility_mean,
        visibility_min,
        valid_lower_body,
    ]
    return np.asarray(values, dtype=np.float32)


def summarize_values(values: np.ndarray) -> list[float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return [0.0 for _ in SUMMARY_STATS]
    return [
        float(np.mean(finite_values)),
        float(np.std(finite_values)),
        float(np.min(finite_values)),
        float(np.max(finite_values)),
        float(np.percentile(finite_values, 10)),
        float(np.percentile(finite_values, 25)),
        float(np.percentile(finite_values, 50)),
        float(np.percentile(finite_values, 75)),
        float(np.percentile(finite_values, 90)),
    ]


def aggregate_frame_features(matrix: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    values: list[float] = []
    names: list[str] = []
    for column_index, feature_name in enumerate(FRAME_FEATURE_NAMES):
        summary_values = summarize_values(matrix[:, column_index])
        values.extend(summary_values)
        names.extend(f"{prefix}_{feature_name}_{stat}" for stat in SUMMARY_STATS)
    return values, names


def bottom_phase_mask(matrix: np.ndarray) -> np.ndarray:
    hip_y_index = FRAME_FEATURE_NAMES.index("avg_hip_y")
    hip_y = matrix[:, hip_y_index]
    finite_mask = np.isfinite(hip_y)
    if not finite_mask.any():
        return np.zeros(matrix.shape[0], dtype=bool)
    threshold = float(np.percentile(hip_y[finite_mask], 70))
    bottom_mask = finite_mask & (hip_y >= threshold)
    if bottom_mask.any():
        return bottom_mask
    return finite_mask


def finite_feature_vector(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def extract_pose_feature_bundle(pose_json_path: Path) -> dict[str, np.ndarray]:
    payload = load_pose_json(pose_json_path)
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        frames = []

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    if frames:
        matrix = np.stack([compute_frame_features(frame) for frame in frames], axis=0)
    else:
        matrix = np.full((1, len(FRAME_FEATURE_NAMES)), np.nan, dtype=np.float32)

    full_values, full_names = aggregate_frame_features(matrix, prefix="full")

    bottom_mask = bottom_phase_mask(matrix)
    bottom_matrix = matrix[bottom_mask] if bottom_mask.any() else matrix[:0]
    bottom_values, bottom_names = aggregate_frame_features(bottom_matrix, prefix="bottom")

    valid_index = FRAME_FEATURE_NAMES.index("valid_lower_body")
    valid_values = matrix[:, valid_index]
    valid_frame_ratio = float(np.mean(valid_values[np.isfinite(valid_values)])) if np.isfinite(valid_values).any() else 0.0
    pose_detected_ratio = float(np.mean([frame_has_pose(frame) for frame in frames])) if frames else 0.0
    bottom_frame_ratio = float(bottom_mask.mean()) if bottom_mask.size else 0.0
    fps = float(metadata.get("fps", 0.0) or 0.0)
    width = float(metadata.get("width", 0.0) or 0.0)
    height = float(metadata.get("height", 0.0) or 0.0)
    aspect_ratio = width / height if height > 0.0 else 0.0

    quality_values = [
        float(np.log1p(len(frames))),
        fps / 60.0,
        aspect_ratio,
        pose_detected_ratio,
        valid_frame_ratio,
        bottom_frame_ratio,
    ]
    quality_names = [
        "quality_log_total_frames",
        "quality_fps_over_60",
        "quality_aspect_ratio",
        "quality_pose_detected_ratio",
        "quality_valid_lower_body_ratio",
        "quality_bottom_frame_ratio",
    ]

    feature_names = [*full_names, *bottom_names, *quality_names]
    video_feature = finite_feature_vector([*full_values, *bottom_values, *quality_values])

    return {
        "video_feature": video_feature,
        "feature_names": np.asarray(feature_names),
        "valid_frame_ratio": np.asarray([valid_frame_ratio], dtype=np.float32),
    }


def save_feature_bundle(output_path: Path, video_id: str, pose_json_path: Path, bundle: dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        video_id=np.asarray(video_id),
        pose_json_path=np.asarray(str(pose_json_path)),
        **bundle,
    )


def build_requests(
    pose_json_dir: Path,
    split_dir: Path,
    output_dir: Path,
    split_names: Sequence[str],
) -> list[PoseFeatureRequest]:
    requests: list[PoseFeatureRequest] = []
    missing: list[str] = []

    for split_name in split_names:
        video_ids = load_json_list(split_dir / f"{split_name}_keys.json")
        for video_id in video_ids:
            pose_json_path = pose_json_dir / split_name / f"{video_id}.json"
            if not pose_json_path.exists():
                missing.append(f"{split_name}/{video_id}")
                continue
            requests.append(
                PoseFeatureRequest(
                    split_name=split_name,
                    video_id=video_id,
                    pose_json_path=pose_json_path,
                    output_path=output_dir / split_name / f"{video_id}.npz",
                )
            )

    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        print(f"Warning: {len(missing)} pose JSON files were not found: {preview}{suffix}")

    return requests


def iter_requests(requests: Sequence[PoseFeatureRequest], limit: int | None) -> Iterable[PoseFeatureRequest]:
    if limit is None:
        yield from requests
        return
    for request in requests[:limit]:
        yield request


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MediaPipe Pose JSON files into pose-only feature bundles.")
    parser.add_argument(
        "--pose-json-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "pose_json",
        help="Directory containing split subdirectories with pose JSON files.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "Splits",
        help="Directory containing train_keys.json, val_keys.json, and test_keys.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "pose_features",
        help="Directory to write pose feature .npz bundles.",
    )
    parser.add_argument("--splits", type=parse_split_names, default=list(SPLIT_NAMES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    requests = build_requests(
        pose_json_dir=args.pose_json_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        split_names=args.splits,
    )
    if not requests:
        raise SystemExit("No pose JSON files were found to process.")

    print(f"Processing {len(requests)} pose JSON files from {args.pose_json_dir}.")
    print(f"Writing feature bundles under {args.output_dir}.")

    processed = 0
    skipped = 0
    for index, request in enumerate(iter_requests(requests, args.limit), start=1):
        if request.output_path.exists() and not args.overwrite:
            print(f"[{index}] Skipping {request.video_id}, already exists.")
            skipped += 1
            continue

        print(f"[{index}] Extracting pose features for {request.split_name}/{request.video_id}...")
        bundle = extract_pose_feature_bundle(request.pose_json_path)
        save_feature_bundle(
            output_path=request.output_path,
            video_id=request.video_id,
            pose_json_path=request.pose_json_path,
            bundle=bundle,
        )
        processed += 1

    print(f"Done. processed={processed} skipped={skipped}")


if __name__ == "__main__":
    main()
