from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import pkgutil
import sys
import types
import zipimport
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_NAMES = ("train", "val", "test")
LANDMARK_COUNT = 33

# COCO-WholeBody has 17 body keypoints followed by 6 foot keypoints.
# The existing downstream pipeline expects MediaPipe's 33-landmark indexing.
MEDIAPIPE_TO_COCO_WHOLEBODY = {
    0: 0,  # nose
    2: 1,  # left_eye
    5: 2,  # right_eye
    7: 3,  # left_ear
    8: 4,  # right_ear
    11: 5,  # left_shoulder
    12: 6,  # right_shoulder
    13: 7,  # left_elbow
    14: 8,  # right_elbow
    15: 9,  # left_wrist
    16: 10,  # right_wrist
    23: 11,  # left_hip
    24: 12,  # right_hip
    25: 13,  # left_knee
    26: 14,  # right_knee
    27: 15,  # left_ankle
    28: 16,  # right_ankle
    29: 19,  # left_heel
    30: 22,  # right_heel
}
LEFT_TOE_CANDIDATES = (17, 18)
RIGHT_TOE_CANDIDATES = (20, 21)


@dataclass(frozen=True)
class MMPoseRequest:
    split_name: str
    video_id: str
    video_path: Path
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


def find_video_path(video_dir: Path, video_id: str) -> Path | None:
    direct = video_dir / f"{video_id}.mp4"
    if direct.exists():
        return direct
    matches = list(video_dir.rglob(f"{video_id}.mp4"))
    if matches:
        return matches[0]
    return None


def build_requests(
    video_dir: Path,
    split_dir: Path,
    output_dir: Path,
    split_names: Sequence[str],
) -> list[MMPoseRequest]:
    requests: list[MMPoseRequest] = []
    missing: list[str] = []
    for split_name in split_names:
        for video_id in load_json_list(split_dir / f"{split_name}_keys.json"):
            video_path = find_video_path(video_dir, video_id)
            if video_path is None:
                missing.append(f"{split_name}/{video_id}")
                continue
            requests.append(
                MMPoseRequest(
                    split_name=split_name,
                    video_id=video_id,
                    video_path=video_path,
                    output_path=output_dir / split_name / f"{video_id}.json",
                )
            )

    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        print(f"Warning: {len(missing)} split videos were not found: {preview}{suffix}")

    return requests


def iter_requests(requests: Sequence[MMPoseRequest], limit: int | None) -> Iterable[MMPoseRequest]:
    yield from requests if limit is None else requests[:limit]


def blank_landmark() -> dict[str, float]:
    return {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0}


def normalized_landmark(x: float, y: float, score: float, width: int, height: int) -> dict[str, float]:
    if width <= 0 or height <= 0 or not np.isfinite(x) or not np.isfinite(y):
        return blank_landmark()
    visibility = float(np.clip(score if np.isfinite(score) else 0.0, 0.0, 1.0))
    return {
        "x": float(x / width),
        "y": float(y / height),
        "z": 0.0,
        "visibility": visibility,
    }


def point_score(scores: np.ndarray, index: int) -> float:
    if index >= scores.shape[0]:
        return 0.0
    value = float(scores[index])
    return value if np.isfinite(value) else 0.0


def assign_direct_landmark(
    landmarks: list[dict[str, float]],
    mediapipe_index: int,
    coco_index: int,
    keypoints: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
) -> None:
    if coco_index >= keypoints.shape[0]:
        return
    x, y = keypoints[coco_index, :2]
    landmarks[mediapipe_index] = normalized_landmark(
        float(x),
        float(y),
        point_score(scores, coco_index),
        width,
        height,
    )


def assign_average_landmark(
    landmarks: list[dict[str, float]],
    mediapipe_index: int,
    coco_indices: Sequence[int],
    keypoints: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
) -> None:
    candidates: list[tuple[float, float, float]] = []
    for coco_index in coco_indices:
        if coco_index >= keypoints.shape[0]:
            continue
        x, y = keypoints[coco_index, :2]
        score = point_score(scores, coco_index)
        if np.isfinite(x) and np.isfinite(y):
            candidates.append((float(x), float(y), score))
    if not candidates:
        return

    weights = np.asarray([max(score, 0.0) for *_xy, score in candidates], dtype=np.float32)
    if float(weights.sum()) <= 1e-8:
        weights = np.ones(len(candidates), dtype=np.float32)
    xy = np.asarray([[x, y] for x, y, _score in candidates], dtype=np.float32)
    mean_xy = np.average(xy, axis=0, weights=weights)
    landmarks[mediapipe_index] = normalized_landmark(
        float(mean_xy[0]),
        float(mean_xy[1]),
        max(score for *_xy, score in candidates),
        width,
        height,
    )


def coco_wholebody_to_mediapipe_landmarks(
    keypoints: np.ndarray,
    scores: np.ndarray | None,
    width: int,
    height: int,
) -> list[dict[str, float]]:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim == 3 and keypoints.shape[0] == 1:
        keypoints = keypoints[0]
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        return [blank_landmark() for _ in range(LANDMARK_COUNT)]

    if scores is None:
        score_values = np.ones(keypoints.shape[0], dtype=np.float32)
    else:
        score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
        if score_values.shape[0] < keypoints.shape[0]:
            score_values = np.pad(score_values, (0, keypoints.shape[0] - score_values.shape[0]))

    landmarks = [blank_landmark() for _ in range(LANDMARK_COUNT)]
    for mediapipe_index, coco_index in MEDIAPIPE_TO_COCO_WHOLEBODY.items():
        assign_direct_landmark(
            landmarks,
            mediapipe_index,
            coco_index,
            keypoints,
            score_values,
            width,
            height,
        )
    assign_average_landmark(landmarks, 31, LEFT_TOE_CANDIDATES, keypoints, score_values, width, height)
    assign_average_landmark(landmarks, 32, RIGHT_TOE_CANDIDATES, keypoints, score_values, width, height)
    return landmarks


def prediction_instances(result: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = result.get("predictions", [])
    if not isinstance(predictions, list) or not predictions:
        return []
    if all(isinstance(item, dict) for item in predictions):
        return predictions
    if len(predictions) == 1 and isinstance(predictions[0], list):
        return [item for item in predictions[0] if isinstance(item, dict)]
    flattened: list[dict[str, Any]] = []
    for item in predictions:
        if isinstance(item, list):
            flattened.extend(child for child in item if isinstance(child, dict))
    return flattened


def instance_keypoints(instance: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    keypoints = instance.get("keypoints")
    if keypoints is None:
        return None, None
    keypoint_array = np.asarray(keypoints, dtype=np.float32)
    if keypoint_array.ndim == 3 and keypoint_array.shape[0] == 1:
        keypoint_array = keypoint_array[0]
    if keypoint_array.ndim != 2 or keypoint_array.shape[1] < 2:
        return None, None

    scores = instance.get("keypoint_scores", instance.get("keypoints_visible"))
    score_array = None if scores is None else np.asarray(scores, dtype=np.float32).reshape(-1)
    return keypoint_array, score_array


def instance_score(instance: dict[str, Any]) -> float:
    bbox_score = instance.get("bbox_score")
    if bbox_score is not None:
        values = np.asarray(bbox_score, dtype=np.float32).reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size:
            return float(finite[0])

    _keypoints, scores = instance_keypoints(instance)
    if scores is None:
        return 0.0
    finite_scores = scores[np.isfinite(scores)]
    return float(np.mean(finite_scores)) if finite_scores.size else 0.0


def select_primary_instance(instances: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not instances:
        return None
    return max(instances, key=instance_score)


def install_python312_pkg_resources_shim() -> None:
    if not hasattr(pkgutil, "ImpImporter"):
        pkgutil.ImpImporter = zipimport.zipimporter  # type: ignore[attr-defined]
    if "pkg_resources" in sys.modules:
        return

    class DistributionNotFound(Exception):
        pass

    class Distribution:
        def __init__(self, package: str):
            spec = importlib.util.find_spec(package)
            if spec is None:
                raise DistributionNotFound(package)
            if spec.submodule_search_locations:
                package_dir = Path(next(iter(spec.submodule_search_locations)))
                self.location = str(package_dir.parent)
            elif spec.origin:
                self.location = str(Path(spec.origin).parent)
            else:
                raise DistributionNotFound(package)
            try:
                self.version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self.version = "0.0.0"

    def get_distribution(package: str) -> Distribution:
        return Distribution(package)

    pkg_resources = types.ModuleType("pkg_resources")
    pkg_resources.DistributionNotFound = DistributionNotFound
    pkg_resources.get_distribution = get_distribution
    sys.modules["pkg_resources"] = pkg_resources


def install_xtcocotools_shim() -> None:
    if "xtcocotools.coco" in sys.modules:
        return
    try:
        import pycocotools.coco as coco
        import pycocotools.cocoeval as cocoeval
        import pycocotools.mask as mask
    except ImportError:
        return

    xtcocotools = types.ModuleType("xtcocotools")
    xtcocotools.coco = coco
    xtcocotools.cocoeval = cocoeval
    xtcocotools.mask = mask
    sys.modules.setdefault("xtcocotools", xtcocotools)
    sys.modules.setdefault("xtcocotools.coco", coco)
    sys.modules.setdefault("xtcocotools.cocoeval", cocoeval)
    sys.modules.setdefault("xtcocotools.mask", mask)


def import_mmpose_inferencer():
    install_python312_pkg_resources_shim()
    install_xtcocotools_shim()
    try:
        from mmpose.apis import MMPoseInferencer
    except ImportError as exc:  # pragma: no cover - exercised in Colab/runtime only
        raise SystemExit(
            "Could not import `MMPoseInferencer` from `mmpose.apis`.\n"
            "In Colab, rerun the notebook install cell, restart the runtime if needed, then verify with:\n"
            "  from mmpose.apis import MMPoseInferencer\n"
            f"Original import error: {exc}"
        ) from exc
    return MMPoseInferencer


def import_rtmlib_wholebody():
    try:
        from rtmlib import Wholebody
    except ImportError as exc:  # pragma: no cover - exercised in Colab/runtime only
        raise SystemExit(
            "Could not import `Wholebody` from `rtmlib`.\n"
            "In Colab, install it with:\n"
            "  pip install rtmlib onnxruntime-gpu\n"
            f"Original import error: {exc}"
        ) from exc
    return Wholebody


def rtmlib_device(value: str) -> str:
    if value.startswith("cuda"):
        return "cuda"
    return value


def rtmlib_mode(value: str) -> str:
    if value in {"performance", "lightweight", "balanced"}:
        return value
    return "balanced"


def rtmlib_primary_pose(keypoints: object, scores: object) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    keypoint_array = np.asarray(keypoints, dtype=np.float32)
    score_array = np.asarray(scores, dtype=np.float32)
    if keypoint_array.ndim == 2:
        keypoint_array = keypoint_array[None, ...]
    if score_array.ndim == 1:
        score_array = score_array[None, ...]
    if keypoint_array.ndim != 3 or keypoint_array.shape[0] == 0:
        return None, None, 0.0

    pose_scores: list[float] = []
    for index in range(keypoint_array.shape[0]):
        current_scores = score_array[index] if index < score_array.shape[0] else np.asarray([], dtype=np.float32)
        finite_scores = current_scores[np.isfinite(current_scores)]
        pose_scores.append(float(np.mean(finite_scores)) if finite_scores.size else 0.0)
    primary_index = int(np.argmax(pose_scores))
    primary_scores = score_array[primary_index] if primary_index < score_array.shape[0] else None
    return keypoint_array[primary_index], primary_scores, pose_scores[primary_index]


def process_video(
    input_path: Path,
    output_json_path: Path,
    *,
    model: str = "wholebody",
    device: str = "cuda:0",
    bbox_thr: float = 0.3,
    runtime: str = "rtmlib",
) -> bool:
    if not input_path.exists():
        print(f"Error: File not found at {input_path}")
        return False

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if runtime == "mmpose":
        MMPoseInferencer = import_mmpose_inferencer()
        inferencer = MMPoseInferencer(pose2d=model, device=device, show_progress=False)
    else:
        Wholebody = import_rtmlib_wholebody()
        inferencer = Wholebody(
            to_openpose=False,
            mode=rtmlib_mode(model),
            backend="onnxruntime",
            device=rtmlib_device(device),
        )

    pose_data: dict[str, Any] = {
        "metadata": {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "backend": "mmpose" if runtime == "mmpose" else "rtmlib",
            "pose_runtime": runtime,
            "mmpose_model": model,
            "keypoint_schema": "coco_wholebody_133_to_mediapipe_33",
            "world_landmarks": False,
        },
        "frames": [],
    }

    frame_index = 0
    progress = tqdm(total=total_frames if total_frames > 0 else None, desc=input_path.name, leave=False)
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            frame_data: dict[str, Any] = {
                "frame_index": frame_index,
                "landmarks": None,
                "world_landmarks": None,
            }
            if runtime == "mmpose":
                try:
                    result = next(inferencer(image, bbox_thr=bbox_thr, return_vis=False))
                except StopIteration:
                    result = {"predictions": []}
                instance = select_primary_instance(prediction_instances(result))
                if instance is not None:
                    keypoints, scores = instance_keypoints(instance)
                    if keypoints is not None:
                        frame_data["landmarks"] = coco_wholebody_to_mediapipe_landmarks(
                            keypoints,
                            scores,
                            width,
                            height,
                        )
                        frame_data["backend_pose_score"] = instance_score(instance)
            else:
                keypoints, scores, pose_score = rtmlib_primary_pose(*inferencer(image))
                if keypoints is not None:
                    frame_data["landmarks"] = coco_wholebody_to_mediapipe_landmarks(
                        keypoints,
                        scores,
                        width,
                        height,
                    )
                    frame_data["backend_pose_score"] = pose_score

            pose_data["frames"].append(frame_data)
            frame_index += 1
            progress.update(1)
    finally:
        progress.close()
        cap.release()

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(pose_data, f, separators=(",", ":"))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch process squat videos with MMPose whole-body pose estimation.")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "videos",
        help="Directory containing labeled .mp4 videos.",
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
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "mmpose_pose_json",
        help="Directory for MMPose pose JSON outputs.",
    )
    parser.add_argument("--splits", type=parse_split_names, default=list(SPLIT_NAMES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--runtime",
        choices=("rtmlib", "mmpose"),
        default="rtmlib",
        help="Pose runtime. rtmlib is the Colab Python 3.12-friendly default.",
    )
    parser.add_argument(
        "--model",
        default="balanced",
        help="For --runtime rtmlib: performance, lightweight, or balanced. For --runtime mmpose: model alias/config.",
    )
    parser.add_argument("--device", default="cuda:0", help="MMPose device, e.g. cuda:0 or cpu.")
    parser.add_argument("--bbox-thr", type=float, default=0.3)
    args = parser.parse_args()

    requests = build_requests(
        video_dir=args.video_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        split_names=args.splits,
    )
    if not requests:
        raise SystemExit("No videos were found to process.")

    print(f"Found {len(requests)} videos to process from {args.video_dir}.")
    print(f"Writing MMPose pose outputs under {args.output_dir}.")

    processed = 0
    skipped = 0
    failed = 0
    for index, request in enumerate(iter_requests(requests, args.limit), start=1):
        if request.output_path.exists() and not args.overwrite:
            print(f"[{index}] Skipping {request.split_name}/{request.video_id}, already processed.")
            skipped += 1
            continue

        print(f"[{index}] Processing {request.split_name}/{request.video_id} from {request.video_path.name}...")
        if process_video(
            request.video_path,
            request.output_path,
            model=args.model,
            device=args.device,
            bbox_thr=args.bbox_thr,
            runtime=args.runtime,
        ):
            processed += 1
        else:
            failed += 1

    print(f"Done. processed={processed} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
