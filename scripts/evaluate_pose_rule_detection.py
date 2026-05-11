from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ClassSpec:
    class_id: str
    fault_id: str
    label_kind: str
    label_path: Path


@dataclass(frozen=True)
class VideoPrediction:
    split: str
    video_id: str
    view_type: str
    view_confidence: float
    intervals_by_fault: dict[str, list[tuple[float, float]]]
    fps: float


DEFAULT_CLASSES = (
    ClassSpec(
        class_id="knees_forward",
        fault_id="knees_forward",
        label_kind="intervals",
        label_path=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "Labels" / "error_knees_forward.json",
    ),
    ClassSpec(
        class_id="knees_inward",
        fault_id="knees_inward",
        label_kind="intervals",
        label_path=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "Labels" / "error_knees_inward.json",
    ),
    ClassSpec(
        class_id="shallow_depth",
        fault_id="shallow_depth",
        label_kind="frame_labels",
        label_path=REPO_ROOT
        / "data"
        / "Squat"
        / "Labeled_Dataset"
        / "Shallow_Squat_Error_Dataset"
        / "labels_shallow_depth.json",
    ),
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_view_metadata(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {(row["split"], row["video_id"]): row for row in reader}


def iter_detection_paths(detections_dir: Path) -> Iterable[Path]:
    yield from sorted(path for path in detections_dir.rglob("*.json") if path.is_file())


def load_predictions(detections_dir: Path, view_metadata_path: Path) -> dict[str, VideoPrediction]:
    view_metadata = load_view_metadata(view_metadata_path)
    predictions: dict[str, VideoPrediction] = {}
    for path in iter_detection_paths(detections_dir):
        payload = load_json(path)
        video_id = str(payload.get("video_id") or path.stem)
        split = path.parent.name if path.parent != detections_dir else str(payload.get("split", ""))
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
        fps = float(metadata.get("fps", 30.0) or 30.0)
        view_row = view_metadata.get((split, video_id), {})
        view_payload = payload.get("view", {}) if isinstance(payload.get("view"), dict) else {}
        view_type = str(view_row.get("view_type") or view_payload.get("view_type") or "unknown")
        view_confidence = float(view_row.get("view_confidence") or view_payload.get("view_confidence") or 0.0)

        intervals_by_fault: dict[str, list[tuple[float, float]]] = {}
        for detection in payload.get("detections", []):
            if not isinstance(detection, dict):
                continue
            if str(detection.get("observability", "")) == "low":
                continue
            if float(detection.get("severity", 0.0) or 0.0) <= 0.0:
                continue
            fault_id = str(detection.get("fault_id") or "")
            start_time = float(detection.get("start_time", 0.0) or 0.0)
            end_time = float(detection.get("end_time", start_time) or start_time)
            intervals_by_fault.setdefault(fault_id, []).append((start_time, max(start_time, end_time)))

        predictions[video_id] = VideoPrediction(
            split=split,
            video_id=video_id,
            view_type=view_type,
            view_confidence=view_confidence,
            intervals_by_fault=intervals_by_fault,
            fps=fps,
        )
    return predictions


def load_interval_labels(path: Path) -> dict[str, list[tuple[float, float]]]:
    payload = load_json(path)
    labels: dict[str, list[tuple[float, float]]] = {}
    for video_id, intervals in payload.items():
        parsed: list[tuple[float, float]] = []
        if isinstance(intervals, list):
            for interval in intervals:
                if isinstance(interval, list) and len(interval) >= 2:
                    start = float(interval[0])
                    end = float(interval[1])
                    parsed.append((start, max(start, end)))
        labels[str(video_id)] = parsed
    return labels


def split_shallow_key(key: str) -> tuple[str, int] | None:
    prefix, sep, frame_text = key.rpartition("_")
    if not sep:
        return None
    try:
        return prefix, int(frame_text)
    except ValueError:
        return None


def group_frame_indices(indices: Sequence[int], max_gap: int = 1) -> list[tuple[int, int]]:
    if not indices:
        return []
    sorted_indices = sorted(set(indices))
    groups: list[tuple[int, int]] = []
    start = sorted_indices[0]
    previous = sorted_indices[0]
    for index in sorted_indices[1:]:
        if index - previous <= max_gap:
            previous = index
            continue
        groups.append((start, previous))
        start = previous = index
    groups.append((start, previous))
    return groups


def load_frame_labels(path: Path, predictions: dict[str, VideoPrediction]) -> dict[str, list[tuple[float, float]]]:
    payload = load_json(path)
    all_video_ids: set[str] = set()
    positive_frames: dict[str, list[int]] = {}
    for key, value in payload.items():
        parsed = split_shallow_key(str(key))
        if parsed is None:
            continue
        video_id, frame_index = parsed
        all_video_ids.add(video_id)
        if int(value) != 1:
            continue
        positive_frames.setdefault(video_id, []).append(frame_index)

    labels: dict[str, list[tuple[float, float]]] = {video_id: [] for video_id in all_video_ids}
    for video_id, frame_indices in positive_frames.items():
        fps = predictions.get(video_id).fps if video_id in predictions else 30.0
        intervals = [
            (start / fps, (end + 1) / fps)
            for start, end in group_frame_indices(frame_indices, max_gap=1)
        ]
        labels[video_id] = intervals
    return labels


def interval_union_length(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return sum(end - start for start, end in merged)


def interval_intersection_length(a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    for start_a, end_a in a:
        for start_b, end_b in b:
            total += max(0.0, min(end_a, end_b) - max(start_a, start_b))
    return total


def interval_iou(predicted: Sequence[tuple[float, float]], truth: Sequence[tuple[float, float]]) -> float:
    intersection = interval_intersection_length(predicted, truth)
    union = interval_union_length([*predicted, *truth])
    return intersection / union if union > 0 else 0.0


def binary_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    return {
        "n": tp + fp + tn + fn,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
    }


def evaluate_group(
    video_ids: Sequence[str],
    predictions: dict[str, VideoPrediction],
    labels: dict[str, list[tuple[float, float]]],
    fault_id: str,
) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    ious: list[float] = []
    for video_id in video_ids:
        predicted_intervals = predictions[video_id].intervals_by_fault.get(fault_id, [])
        truth_intervals = labels.get(video_id, [])
        predicted_positive = bool(predicted_intervals)
        truth_positive = bool(truth_intervals)
        if predicted_positive and truth_positive:
            tp += 1
        elif predicted_positive and not truth_positive:
            fp += 1
        elif not predicted_positive and truth_positive:
            fn += 1
        else:
            tn += 1
        if predicted_positive or truth_positive:
            ious.append(interval_iou(predicted_intervals, truth_intervals))
    metrics = binary_metrics(tp, fp, tn, fn)
    metrics["mean_segment_iou"] = sum(ious) / len(ious) if ious else 0.0
    return metrics


def format_float(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class_id",
        "view_type",
        "n",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
        "mean_segment_iou",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_float(row.get(field, "")) for field in fieldnames})


def print_markdown(rows: Sequence[dict[str, Any]]) -> None:
    headers = ["class_id", "view_type", "n", "precision", "recall", "f1", "mean_segment_iou", "tp", "fp", "tn", "fn"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(format_float(row.get(header, "")) for header in headers) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rule-based pose detections against squat labels.")
    parser.add_argument(
        "--detections-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "pose_rule_detections",
    )
    parser.add_argument(
        "--view-metadata",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "view_metadata.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "pose_rule_validation_metrics.csv",
    )
    args = parser.parse_args()

    predictions = load_predictions(args.detections_dir, args.view_metadata)
    if not predictions:
        raise SystemExit(f"No detection JSON files found under {args.detections_dir}")

    rows: list[dict[str, Any]] = []
    for spec in DEFAULT_CLASSES:
        if spec.label_kind == "intervals":
            labels = load_interval_labels(spec.label_path)
        else:
            labels = load_frame_labels(spec.label_path, predictions)

        video_ids = sorted(set(predictions) & set(labels))
        overall = evaluate_group(video_ids, predictions, labels, spec.fault_id)
        rows.append({"class_id": spec.class_id, "view_type": "ALL", **overall})

        view_types = sorted({predictions[video_id].view_type for video_id in video_ids})
        for view_type in view_types:
            group_ids = [video_id for video_id in video_ids if predictions[video_id].view_type == view_type]
            if not group_ids:
                continue
            rows.append({"class_id": spec.class_id, "view_type": view_type, **evaluate_group(group_ids, predictions, labels, spec.fault_id)})

    write_csv(args.output, rows)
    print(f"Saved validation metrics to {args.output}")
    print_markdown(rows)


if __name__ == "__main__":
    main()
