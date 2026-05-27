from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_NAMES = ("train", "val", "test")
VIEW_TYPES = ("front", "front_oblique", "side", "rear_oblique", "rear", "unknown")
OBLIQUE_THRESHOLD = 0.4
SIDE_THRESHOLD = 0.36

NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

LANDMARK_COUNT = 33
FACE_LANDMARKS = (
    NOSE,
    LEFT_EYE_INNER,
    LEFT_EYE,
    LEFT_EYE_OUTER,
    RIGHT_EYE_INNER,
    RIGHT_EYE,
    RIGHT_EYE_OUTER,
    LEFT_EAR,
    RIGHT_EAR,
)
TORSO_LANDMARKS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
BODY_LANDMARKS = (
    *FACE_LANDMARKS,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
)


@dataclass(frozen=True)
class ViewRequest:
    split_name: str
    video_id: str
    pose_json_path: Path


@dataclass(frozen=True)
class ViewEstimate:
    split_name: str
    video_id: str
    view_type: str
    view_confidence: float
    front_score: float
    rear_score: float
    side_score: float
    oblique_score: float
    face_visibility_mean: float
    torso_width_ratio_mean: float
    orientation_score_mean: float
    z_asymmetry_mean: float
    valid_frame_ratio: float
    valid_frame_count: int
    total_frames: int


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


def finite_visibility(points: np.ndarray | None, indices: Sequence[int]) -> np.ndarray:
    if points is None:
        return np.asarray([], dtype=np.float32)
    visibility = points[list(indices), 3]
    return visibility[np.isfinite(visibility)]


def visible_point(points: np.ndarray | None, index: int, min_visibility: float = 0.45) -> np.ndarray | None:
    if points is None:
        return None
    values = points[index, :3]
    visibility = points[index, 3]
    if not np.all(np.isfinite(values)) or not np.isfinite(visibility) or visibility < min_visibility:
        return None
    return values.astype(np.float32, copy=False)


def xy_distance(points: np.ndarray | None, left_index: int, right_index: int) -> float:
    left = visible_point(points, left_index)
    right = visible_point(points, right_index)
    if left is None or right is None:
        return np.nan
    return float(np.linalg.norm(left[:2] - right[:2]))


def body_height(points: np.ndarray | None) -> float:
    if points is None:
        return np.nan
    candidates: list[float] = []
    for index in BODY_LANDMARKS:
        point = visible_point(points, index, min_visibility=0.35)
        if point is not None:
            candidates.append(float(point[1]))
    if len(candidates) < 4:
        return np.nan
    return float(max(candidates) - min(candidates))


def signed_orientation(points: np.ndarray | None, left_index: int, right_index: int) -> float:
    left = visible_point(points, left_index)
    right = visible_point(points, right_index)
    if left is None or right is None:
        return np.nan
    return float(np.sign(left[0] - right[0]))


def z_asymmetry(world_points: np.ndarray | None, left_index: int, right_index: int) -> float:
    left = visible_point(world_points, left_index)
    right = visible_point(world_points, right_index)
    if left is None or right is None:
        return np.nan
    return float(abs(left[2] - right[2]))


def clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def mean_finite(values: Sequence[float], default: float = 0.0) -> float:
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return default
    return float(np.mean(finite))


def frame_view_signals(frame: object) -> dict[str, float] | None:
    if not isinstance(frame, dict):
        return None

    image_points = landmarks_to_array(frame.get("landmarks"))
    world_points = landmarks_to_array(frame.get("world_landmarks"))
    if image_points is None and world_points is None:
        return None

    shoulder_width = xy_distance(image_points, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_width = xy_distance(image_points, LEFT_HIP, RIGHT_HIP)
    height = body_height(image_points)
    torso_width = mean_finite([shoulder_width, hip_width], default=np.nan)
    torso_width_ratio = torso_width / height if np.isfinite(torso_width) and np.isfinite(height) and height > 1e-8 else np.nan

    shoulder_orientation = signed_orientation(image_points, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_orientation = signed_orientation(image_points, LEFT_HIP, RIGHT_HIP)
    orientation_score = mean_finite([shoulder_orientation, hip_orientation], default=np.nan)

    face_visibility = finite_visibility(image_points, FACE_LANDMARKS)
    face_visibility_mean = float(np.mean(face_visibility)) if face_visibility.size else 0.0

    shoulder_z = z_asymmetry(world_points, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_z = z_asymmetry(world_points, LEFT_HIP, RIGHT_HIP)
    z_asymmetry_mean = mean_finite([shoulder_z, hip_z], default=0.0)

    if not np.isfinite(orientation_score) and not np.isfinite(torso_width_ratio):
        return None

    return {
        "orientation_score": orientation_score,
        "torso_width_ratio": torso_width_ratio,
        "face_visibility": face_visibility_mean,
        "z_asymmetry": z_asymmetry_mean,
    }


def score_view(
    orientation_score: float,
    face_visibility: float,
    torso_width_ratio: float,
    z_asymmetry_value: float,
    valid_frame_ratio: float,
    allow_front: bool = False,
) -> tuple[str, float, float, float, float, float]:
    orientation_strength = abs(orientation_score) if np.isfinite(orientation_score) else 0.0
    front_direction = max(orientation_score, 0.0) if np.isfinite(orientation_score) else 0.0
    rear_direction = max(-orientation_score, 0.0) if np.isfinite(orientation_score) else 0.0

    narrow_body_signal = clip01((0.24 - torso_width_ratio) / 0.16) if np.isfinite(torso_width_ratio) else 0.0
    broad_body_signal = clip01((torso_width_ratio - 0.18) / 0.18) if np.isfinite(torso_width_ratio) else 0.0
    depth_signal = clip01(z_asymmetry_value / 0.28)

    side_score = clip01(0.65 * narrow_body_signal + 0.25 * (1.0 - orientation_strength) + 0.10 * depth_signal)
    oblique_score = clip01(0.45 * side_score + 0.35 * depth_signal + 0.20 * (1.0 - broad_body_signal))
    # MediaPipe pose face landmarks can remain highly visible for rear-view squat videos,
    # especially when background faces or pose hallucinations are present. Keep face
    # visibility as diagnostic metadata, but do not use it as front/rear evidence.
    front_score = clip01(front_direction * (0.72 + 0.28 * broad_body_signal))
    rear_score = clip01(rear_direction * (0.72 + 0.28 * broad_body_signal))

    if valid_frame_ratio < 0.15 or max(front_score, rear_score, side_score) < 0.20:
        return "unknown", 0.0, front_score, rear_score, side_score, oblique_score

    if side_score >= 0.62 and side_score >= max(front_score, rear_score) * 0.90:
        view_type = "side"
        confidence = side_score
    elif front_score >= rear_score and allow_front:
        view_type = "front_oblique" if oblique_score >= OBLIQUE_THRESHOLD or side_score >= SIDE_THRESHOLD else "front"
        confidence = front_score * (0.82 if view_type == "front_oblique" else 1.0)
    elif front_score >= rear_score:
        view_type = "rear_oblique"
        confidence = max(front_score, side_score, oblique_score) * 0.70
    else:
        view_type = "rear_oblique" if oblique_score >= OBLIQUE_THRESHOLD or side_score >= SIDE_THRESHOLD else "rear"
        confidence = rear_score * (0.82 if view_type == "rear_oblique" else 1.0)

    confidence = clip01(confidence * (0.55 + 0.45 * clip01(valid_frame_ratio)))
    return view_type, confidence, front_score, rear_score, side_score, oblique_score


def estimate_view_for_pose(
    pose_json_path: Path,
    split_name: str = "",
    video_id: str | None = None,
    allow_front: bool = False,
) -> ViewEstimate:
    payload = load_pose_json(pose_json_path)
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        frames = []

    signals = [frame_view_signals(frame) for frame in frames]
    valid_signals = [signal for signal in signals if signal is not None]
    total_frames = len(frames)
    valid_frame_count = len(valid_signals)
    valid_frame_ratio = valid_frame_count / total_frames if total_frames else 0.0

    orientation_score = mean_finite([signal["orientation_score"] for signal in valid_signals], default=0.0)
    face_visibility = mean_finite([signal["face_visibility"] for signal in valid_signals], default=0.0)
    torso_width_ratio = mean_finite([signal["torso_width_ratio"] for signal in valid_signals], default=0.0)
    z_asymmetry_value = mean_finite([signal["z_asymmetry"] for signal in valid_signals], default=0.0)

    view_type, confidence, front_score, rear_score, side_score, oblique_score = score_view(
        orientation_score=orientation_score,
        face_visibility=face_visibility,
        torso_width_ratio=torso_width_ratio,
        z_asymmetry_value=z_asymmetry_value,
        valid_frame_ratio=valid_frame_ratio,
        allow_front=allow_front,
    )

    return ViewEstimate(
        split_name=split_name,
        video_id=video_id or pose_json_path.stem,
        view_type=view_type,
        view_confidence=confidence,
        front_score=front_score,
        rear_score=rear_score,
        side_score=side_score,
        oblique_score=oblique_score,
        face_visibility_mean=face_visibility,
        torso_width_ratio_mean=torso_width_ratio,
        orientation_score_mean=orientation_score,
        z_asymmetry_mean=z_asymmetry_value,
        valid_frame_ratio=valid_frame_ratio,
        valid_frame_count=valid_frame_count,
        total_frames=total_frames,
    )


def build_requests(pose_json_dir: Path, split_dir: Path, split_names: Sequence[str]) -> list[ViewRequest]:
    requests: list[ViewRequest] = []
    missing: list[str] = []
    for split_name in split_names:
        video_ids = load_json_list(split_dir / f"{split_name}_keys.json")
        for video_id in video_ids:
            pose_json_path = pose_json_dir / split_name / f"{video_id}.json"
            if not pose_json_path.exists():
                missing.append(f"{split_name}/{video_id}")
                continue
            requests.append(ViewRequest(split_name=split_name, video_id=video_id, pose_json_path=pose_json_path))

    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        print(f"Warning: {len(missing)} pose JSON files were not found: {preview}{suffix}")
    return requests


def iter_requests(requests: Sequence[ViewRequest], limit: int | None) -> Iterable[ViewRequest]:
    if limit is None:
        yield from requests
        return
    for request in requests[:limit]:
        yield request


def write_view_metadata(path: Path, estimates: Sequence[ViewEstimate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "video_id",
        "view_type",
        "view_confidence",
        "front_score",
        "rear_score",
        "side_score",
        "oblique_score",
        "face_visibility_mean",
        "torso_width_ratio_mean",
        "orientation_score_mean",
        "z_asymmetry_mean",
        "valid_frame_ratio",
        "valid_frame_count",
        "total_frames",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for estimate in estimates:
            writer.writerow(
                {
                    "split": estimate.split_name,
                    "video_id": estimate.video_id,
                    "view_type": estimate.view_type,
                    "view_confidence": f"{estimate.view_confidence:.6f}",
                    "front_score": f"{estimate.front_score:.6f}",
                    "rear_score": f"{estimate.rear_score:.6f}",
                    "side_score": f"{estimate.side_score:.6f}",
                    "oblique_score": f"{estimate.oblique_score:.6f}",
                    "face_visibility_mean": f"{estimate.face_visibility_mean:.6f}",
                    "torso_width_ratio_mean": f"{estimate.torso_width_ratio_mean:.6f}",
                    "orientation_score_mean": f"{estimate.orientation_score_mean:.6f}",
                    "z_asymmetry_mean": f"{estimate.z_asymmetry_mean:.6f}",
                    "valid_frame_ratio": f"{estimate.valid_frame_ratio:.6f}",
                    "valid_frame_count": estimate.valid_frame_count,
                    "total_frames": estimate.total_frames,
                }
            )


def print_summary(estimates: Sequence[ViewEstimate]) -> None:
    counts = {view_type: 0 for view_type in VIEW_TYPES}
    for estimate in estimates:
        counts[estimate.view_type] = counts.get(estimate.view_type, 0) + 1
    print("View type summary:")
    for view_type in VIEW_TYPES:
        print(f"  {view_type}: {counts.get(view_type, 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate coarse person view type from MediaPipe pose JSON files.")
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
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "view_metadata.csv",
    )
    parser.add_argument("--splits", type=parse_split_names, default=list(SPLIT_NAMES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--allow-front",
        action="store_true",
        help=(
            "Allow front/front_oblique outputs. By default, front-like signals are treated "
            "as rear_oblique because this squat dataset is dominated by rear-view videos "
            "and face landmark visibility is noisy."
        ),
    )
    args = parser.parse_args()

    requests = build_requests(args.pose_json_dir, args.split_dir, args.splits)
    if not requests:
        raise SystemExit("No pose JSON files were found to process.")

    estimates: list[ViewEstimate] = []
    for index, request in enumerate(iter_requests(requests, args.limit), start=1):
        print(f"[{index}] Estimating view for {request.split_name}/{request.video_id}...")
        estimates.append(
            estimate_view_for_pose(
                request.pose_json_path,
                split_name=request.split_name,
                video_id=request.video_id,
                allow_front=args.allow_front,
            )
        )

    write_view_metadata(args.output, estimates)
    print(f"Saved view metadata to {args.output}")
    print_summary(estimates)


if __name__ == "__main__":
    main()
