from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.pose_feature_extraction import landmarks_to_array, valid_lower_body_count


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_METRICS = ("balanced_accuracy", "macro_f1", "recall", "specificity", "f1")
RULE_METRICS = ("precision", "recall", "f1", "specificity", "balanced_accuracy", "mean_segment_iou")


@dataclass(frozen=True)
class MetricRow:
    backend: str
    metric_group: str
    label: str
    metric: str
    value: float
    detail: str
    source: str


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def mean_or_zero(values: Sequence[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else 0.0


def iter_pose_json_paths(pose_json_dir: Path) -> Iterable[Path]:
    if not pose_json_dir.exists():
        return
    yield from sorted(path for path in pose_json_dir.rglob("*.json") if path.is_file())


def frame_has_pose(frame: object) -> bool:
    if not isinstance(frame, dict):
        return False
    return landmarks_to_array(frame.get("landmarks")) is not None or landmarks_to_array(frame.get("world_landmarks")) is not None


def frame_has_valid_lower_body(frame: object) -> bool:
    if not isinstance(frame, dict):
        return False
    image_points = landmarks_to_array(frame.get("landmarks"))
    world_points = landmarks_to_array(frame.get("world_landmarks"))
    return valid_lower_body_count(image_points) >= 6 or valid_lower_body_count(world_points) >= 6


def summarize_pose_json(path: Path) -> dict[str, float]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        return {"total_frames": 0.0, "pose_detected_ratio": 0.0, "valid_lower_body_ratio": 0.0}
    frames = payload.get("frames", [])
    if not isinstance(frames, list) or not frames:
        return {"total_frames": 0.0, "pose_detected_ratio": 0.0, "valid_lower_body_ratio": 0.0}
    return {
        "total_frames": float(len(frames)),
        "pose_detected_ratio": float(np.mean([frame_has_pose(frame) for frame in frames])),
        "valid_lower_body_ratio": float(np.mean([frame_has_valid_lower_body(frame) for frame in frames])),
    }


def pose_quality_rows(backend: str, pose_json_dir: Path) -> list[MetricRow]:
    if not pose_json_dir.exists():
        print(f"Warning: missing {backend} pose JSON directory: {pose_json_dir}")
        return []

    summaries = [summarize_pose_json(path) for path in iter_pose_json_paths(pose_json_dir)]
    if not summaries:
        print(f"Warning: no {backend} pose JSON files found under {pose_json_dir}")
        return []

    source = str(pose_json_dir)
    return [
        MetricRow(backend, "extraction", "ALL", "processed_videos", float(len(summaries)), "", source),
        MetricRow(
            backend,
            "extraction",
            "ALL",
            "mean_total_frames",
            mean_or_zero([item["total_frames"] for item in summaries]),
            "",
            source,
        ),
        MetricRow(
            backend,
            "extraction",
            "ALL",
            "mean_pose_detected_ratio",
            mean_or_zero([item["pose_detected_ratio"] for item in summaries]),
            "",
            source,
        ),
        MetricRow(
            backend,
            "extraction",
            "ALL",
            "mean_valid_lower_body_ratio",
            mean_or_zero([item["valid_lower_body_ratio"] for item in summaries]),
            "",
            source,
        ),
    ]


def rule_metric_rows(backend: str, path: Path) -> list[MetricRow]:
    if not path.exists():
        print(f"Warning: missing {backend} rule metrics CSV: {path}")
        return []

    rows: list[MetricRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("view_type") != "ALL":
                continue
            class_id = str(row.get("class_id") or "")
            for metric in RULE_METRICS:
                rows.append(
                    MetricRow(
                        backend=backend,
                        metric_group="rule",
                        label=class_id,
                        metric=metric,
                        value=safe_float(row.get(metric)),
                        detail=f"n={row.get('n', '')}",
                        source=str(path),
                    )
                )
    return rows


def classifier_metric_rows(backend: str, path: Path) -> list[MetricRow]:
    if not path.exists():
        print(f"Warning: missing {backend} classifier summary CSV: {path}")
        return []

    grouped: dict[tuple[str, str], list[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != "test" or row.get("threshold_kind") != "selected_threshold":
                continue
            label_mode = str(row.get("label_mode") or "")
            for metric in CLASSIFIER_METRICS:
                grouped.setdefault((label_mode, metric), []).append(safe_float(row.get(metric)))

    rows: list[MetricRow] = []
    for (label_mode, metric), values in sorted(grouped.items()):
        rows.append(
            MetricRow(
                backend=backend,
                metric_group="classifier",
                label=label_mode,
                metric=metric,
                value=mean_or_zero(values),
                detail=f"mean over {len(values)} seed(s)",
                source=str(path),
            )
        )
        if len(values) > 1:
            rows.append(
                MetricRow(
                    backend=backend,
                    metric_group="classifier",
                    label=label_mode,
                    metric=f"{metric}_std",
                    value=float(np.std(values)),
                    detail=f"std over {len(values)} seed(s)",
                    source=str(path),
                )
            )
    return rows


def first_existing(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def write_rows_csv(path: Path, rows: Sequence[MetricRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["backend", "metric_group", "label", "metric", "value", "detail", "source"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["value"] = f"{row.value:.8f}"
            writer.writerow(payload)


def format_value(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9 and abs(value) >= 10:
        return str(int(round(value)))
    return f"{value:.4f}"


def markdown_table(rows: Sequence[MetricRow], left_backend: str, right_backend: str) -> str:
    values: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        values[(row.backend, row.metric_group, row.label, row.metric)] = row.value

    keys = sorted({(row.metric_group, row.label, row.metric) for row in rows})
    lines = [
        "| metric_group | label | metric | "
        f"{left_backend} | {right_backend} | {right_backend}_minus_{left_backend} |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for metric_group, label, metric in keys:
        left_value = values.get((left_backend, metric_group, label, metric))
        right_value = values.get((right_backend, metric_group, label, metric))
        delta = right_value - left_value if left_value is not None and right_value is not None else None
        lines.append(
            "| "
            + " | ".join(
                [
                    metric_group,
                    label,
                    metric,
                    format_value(left_value),
                    format_value(right_value),
                    format_value(delta),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def collect_rows(args: argparse.Namespace) -> list[MetricRow]:
    media_classifier_summary = args.mediapipe_classifier_summary or first_existing(
        [
            REPO_ROOT / "data" / "Squat" / "pose_classifier_experiments_normalize" / "metrics" / "experiment_summary.csv",
            REPO_ROOT / "data" / "Squat" / "pose_classifier_experiments" / "metrics" / "experiment_summary.csv",
        ]
    )

    rows: list[MetricRow] = []
    rows.extend(pose_quality_rows("mediapipe", args.mediapipe_pose_json_dir))
    rows.extend(pose_quality_rows("mmpose", args.mmpose_pose_json_dir))
    rows.extend(rule_metric_rows("mediapipe", args.mediapipe_rule_metrics))
    rows.extend(rule_metric_rows("mmpose", args.mmpose_rule_metrics))
    rows.extend(classifier_metric_rows("mediapipe", media_classifier_summary))
    rows.extend(classifier_metric_rows("mmpose", args.mmpose_classifier_summary))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MediaPipe and MMPose squat-analysis artifacts.")
    parser.add_argument(
        "--mediapipe-pose-json-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "pose_json",
    )
    parser.add_argument(
        "--mmpose-pose-json-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "mmpose_pose_json",
    )
    parser.add_argument(
        "--mediapipe-rule-metrics",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "pose_rule_validation_metrics.csv",
    )
    parser.add_argument(
        "--mmpose-rule-metrics",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "mmpose_pose_rule_validation_metrics.csv",
    )
    parser.add_argument("--mediapipe-classifier-summary", type=Path, default=None)
    parser.add_argument(
        "--mmpose-classifier-summary",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "mmpose_pose_classifier_experiments" / "metrics" / "experiment_summary.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "mmpose_mediapipe_comparison" / "backend_comparison.csv",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "mmpose_mediapipe_comparison" / "backend_comparison.md",
    )
    args = parser.parse_args()

    rows = collect_rows(args)
    if not rows:
        raise SystemExit("No comparison rows were collected. Check the artifact paths.")

    write_rows_csv(args.output_csv, rows)
    markdown = markdown_table(rows, left_backend="mediapipe", right_backend="mmpose")
    write_markdown(args.output_markdown, markdown)
    print(f"Saved comparison CSV to {args.output_csv}")
    print(f"Saved comparison Markdown to {args.output_markdown}")


if __name__ == "__main__":
    main()
