from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_COLUMNS = {
    "fixed_0_5": "fixed_0_5_prediction",
    "selected_threshold": "selected_threshold_prediction",
}
QUALITY_FIELDS = (
    "quality_pose_detected_ratio",
    "quality_valid_lower_body_ratio",
    "quality_bottom_frame_ratio",
    "valid_frame_ratio",
)


@dataclass(frozen=True)
class PredictionRow:
    label_mode: str
    seed: str
    split: str
    video_id: str
    label: int
    probability: float
    prediction: int
    selected_threshold: float
    view_type: str
    view_confidence: float
    quality_pose_detected_ratio: float
    quality_valid_lower_body_ratio: float
    quality_bottom_frame_ratio: float
    valid_frame_ratio: float


@dataclass(frozen=True)
class ErrorRow:
    label_mode: str
    seed: str
    split: str
    video_id: str
    error_type: str
    label: int
    prediction: int
    probability: float
    selected_threshold: float
    confidence_margin: float
    view_type: str
    view_confidence: float
    quality_pose_detected_ratio: float
    quality_valid_lower_body_ratio: float
    quality_bottom_frame_ratio: float
    valid_frame_ratio: float


def parse_int(value: str, field_name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Expected integer for {field_name}, got {value!r}.") from exc


def parse_float(value: str, field_name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Expected float for {field_name}, got {value!r}.") from exc


def infer_seed(path: Path) -> str:
    match = re.search(r"_seed(\d+)_predictions\.csv$", path.name)
    return match.group(1) if match else ""


def prediction_paths(predictions_dir: Path) -> list[Path]:
    paths = sorted(predictions_dir.glob("*_predictions.csv"))
    if not paths:
        raise ValueError(f"No *_predictions.csv files found in {predictions_dir}")
    return paths


def load_view_metadata(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"split", "video_id", "view_type", "view_confidence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
        return {(row["split"], row["video_id"]): row for row in reader}


def load_pose_quality(feature_path: Path) -> dict[str, float]:
    with np.load(feature_path, allow_pickle=False) as data:
        feature_names = [str(value) for value in data["feature_names"].tolist()]
        video_feature = data["video_feature"].astype(np.float32)
        quality: dict[str, float] = {}
        for field in QUALITY_FIELDS:
            if field == "valid_frame_ratio" and field in data.files:
                quality[field] = float(data[field][0])
                continue
            index = feature_names.index(field)
            quality[field] = float(video_feature[index])
        return quality


def load_pose_quality_index(pose_feature_dir: Path) -> dict[tuple[str, str], dict[str, float]]:
    quality_index: dict[tuple[str, str], dict[str, float]] = {}
    for feature_path in sorted(pose_feature_dir.glob("*/*.npz")):
        split = feature_path.parent.name
        video_id = feature_path.stem
        quality_index[(split, video_id)] = load_pose_quality(feature_path)
    if not quality_index:
        raise ValueError(f"No pose feature .npz files found under {pose_feature_dir}")
    return quality_index


def iter_predictions(
    predictions_path: Path,
    view_metadata: dict[tuple[str, str], dict[str, str]],
    quality_index: dict[tuple[str, str], dict[str, float]],
    split_filter: str,
    threshold_kind: str,
) -> Iterable[PredictionRow]:
    prediction_column = THRESHOLD_COLUMNS[threshold_kind]
    seed = infer_seed(predictions_path)
    with predictions_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "label_mode",
            "split",
            "video_id",
            "label",
            "probability",
            prediction_column,
            "selected_threshold",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {predictions_path}: {', '.join(sorted(missing))}")

        missing_views: list[str] = []
        missing_features: list[str] = []
        for row in reader:
            split = row["split"]
            if split != split_filter:
                continue
            video_id = row["video_id"]
            key = (split, video_id)
            view_row = view_metadata.get(key)
            if view_row is None:
                missing_views.append(f"{split}/{video_id}")
                continue
            quality = quality_index.get(key)
            if quality is None:
                missing_features.append(f"{split}/{video_id}")
                continue

            yield PredictionRow(
                label_mode=row["label_mode"],
                seed=seed,
                split=split,
                video_id=video_id,
                label=parse_int(row["label"], "label"),
                probability=parse_float(row["probability"], "probability"),
                prediction=parse_int(row[prediction_column], prediction_column),
                selected_threshold=parse_float(row["selected_threshold"], "selected_threshold"),
                view_type=view_row["view_type"],
                view_confidence=parse_float(view_row["view_confidence"], "view_confidence"),
                quality_pose_detected_ratio=quality["quality_pose_detected_ratio"],
                quality_valid_lower_body_ratio=quality["quality_valid_lower_body_ratio"],
                quality_bottom_frame_ratio=quality["quality_bottom_frame_ratio"],
                valid_frame_ratio=quality["valid_frame_ratio"],
            )

        if missing_views:
            print_warning("view metadata", predictions_path, missing_views)
        if missing_features:
            print_warning("pose features", predictions_path, missing_features)


def print_warning(kind: str, predictions_path: Path, missing_items: list[str]) -> None:
    preview = ", ".join(missing_items[:10])
    suffix = "" if len(missing_items) <= 10 else f" ... (+{len(missing_items) - 10} more)"
    print(f"Warning: {len(missing_items)} rows in {predictions_path.name} had no {kind}: {preview}{suffix}")


def build_error_rows(predictions: Iterable[PredictionRow]) -> list[ErrorRow]:
    rows: list[ErrorRow] = []
    for row in predictions:
        if row.label == 0 and row.prediction == 1:
            error_type = "false_positive"
            confidence_margin = row.probability - row.selected_threshold
        elif row.label == 1 and row.prediction == 0:
            error_type = "false_negative"
            confidence_margin = row.selected_threshold - row.probability
        else:
            continue

        rows.append(
            ErrorRow(
                label_mode=row.label_mode,
                seed=row.seed,
                split=row.split,
                video_id=row.video_id,
                error_type=error_type,
                label=row.label,
                prediction=row.prediction,
                probability=row.probability,
                selected_threshold=row.selected_threshold,
                confidence_margin=confidence_margin,
                view_type=row.view_type,
                view_confidence=row.view_confidence,
                quality_pose_detected_ratio=row.quality_pose_detected_ratio,
                quality_valid_lower_body_ratio=row.quality_valid_lower_body_ratio,
                quality_bottom_frame_ratio=row.quality_bottom_frame_ratio,
                valid_frame_ratio=row.valid_frame_ratio,
            )
        )
    return rows


def top_error_rows(error_rows: list[ErrorRow], top_n: int) -> list[ErrorRow]:
    groups: dict[tuple[str, str, str], list[ErrorRow]] = defaultdict(list)
    for row in error_rows:
        groups[(row.label_mode, row.seed, row.error_type)].append(row)

    selected: list[ErrorRow] = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda row: (-row.confidence_margin, row.video_id))
        selected.extend(rows[:top_n])
    return selected


def summarize_errors(predictions: list[PredictionRow], error_rows: list[ErrorRow]) -> list[dict[str, object]]:
    prediction_groups: dict[tuple[str, str, str], list[PredictionRow]] = defaultdict(list)
    for row in predictions:
        prediction_groups[(row.label_mode, row.seed, row.view_type)].append(row)
        prediction_groups[(row.label_mode, row.seed, "ALL")].append(row)

    error_groups: dict[tuple[str, str, str, str], list[ErrorRow]] = defaultdict(list)
    for row in error_rows:
        error_groups[(row.label_mode, row.seed, row.view_type, row.error_type)].append(row)
        error_groups[(row.label_mode, row.seed, "ALL", row.error_type)].append(row)

    summary_rows: list[dict[str, object]] = []
    for (label_mode, seed, view_type), grouped_predictions in sorted(prediction_groups.items()):
        for error_type in ("false_positive", "false_negative"):
            grouped_errors = error_groups.get((label_mode, seed, view_type, error_type), [])
            if error_type == "false_positive":
                eligible_count = sum(1 for row in grouped_predictions if row.label == 0)
            else:
                eligible_count = sum(1 for row in grouped_predictions if row.label == 1)
            summary_rows.append(build_summary_row(label_mode, seed, view_type, error_type, eligible_count, grouped_errors))
    return summary_rows


def build_summary_row(
    label_mode: str,
    seed: str,
    view_type: str,
    error_type: str,
    eligible_count: int,
    error_rows: list[ErrorRow],
) -> dict[str, object]:
    n_errors = len(error_rows)
    row: dict[str, object] = {
        "label_mode": label_mode,
        "seed": seed,
        "view_type": view_type,
        "error_type": error_type,
        "eligible_count": float(eligible_count),
        "n_errors": float(n_errors),
        "error_rate": n_errors / eligible_count if eligible_count else 0.0,
    }
    for field in (
        "probability",
        "confidence_margin",
        "view_confidence",
        "quality_pose_detected_ratio",
        "quality_valid_lower_body_ratio",
        "quality_bottom_frame_ratio",
        "valid_frame_ratio",
    ):
        values = [float(getattr(error_row, field)) for error_row in error_rows]
        row[f"mean_{field}"] = sum(values) / len(values) if values else 0.0
    return row


def error_row_dict(row: ErrorRow) -> dict[str, object]:
    return {
        "label_mode": row.label_mode,
        "seed": row.seed,
        "split": row.split,
        "video_id": row.video_id,
        "error_type": row.error_type,
        "label": row.label,
        "prediction": row.prediction,
        "probability": row.probability,
        "selected_threshold": row.selected_threshold,
        "confidence_margin": row.confidence_margin,
        "view_type": row.view_type,
        "view_confidence": row.view_confidence,
        "quality_pose_detected_ratio": row.quality_pose_detected_ratio,
        "quality_valid_lower_body_ratio": row.quality_valid_lower_body_ratio,
        "quality_bottom_frame_ratio": row.quality_bottom_frame_ratio,
        "valid_frame_ratio": row.valid_frame_ratio,
    }


def write_dict_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for field in fieldnames:
                value = row[field]
                formatted[field] = f"{value:.8f}" if isinstance(value, float) else value
            writer.writerow(formatted)


def collect_predictions(
    predictions_dir: Path,
    view_metadata_path: Path,
    pose_feature_dir: Path,
    split: str,
    threshold_kind: str,
) -> list[PredictionRow]:
    view_metadata = load_view_metadata(view_metadata_path)
    quality_index = load_pose_quality_index(pose_feature_dir)
    rows: list[PredictionRow] = []
    for path in prediction_paths(predictions_dir):
        rows.extend(iter_predictions(path, view_metadata, quality_index, split, threshold_kind))
    if not rows:
        raise ValueError("No predictions could be joined with view metadata and pose quality.")
    return rows


def print_summary(error_rows: list[ErrorRow], top_rows: list[ErrorRow]) -> None:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in error_rows:
        counts[(row.label_mode, row.error_type)] += 1
    print("Error counts:")
    for (label_mode, error_type), count in sorted(counts.items()):
        print(f"  {label_mode} {error_type}: {count}")
    print(f"Top error rows: {len(top_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze normalized pose-only classifier errors.")
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "pose_classifier_experiments_normalize" / "predictions",
    )
    parser.add_argument(
        "--pose-feature-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "pose_features",
    )
    parser.add_argument(
        "--view-metadata",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "view_metadata.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "pose_classifier_experiments_normalize" / "error_analysis",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold-kind", choices=tuple(THRESHOLD_COLUMNS), default="selected_threshold")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    predictions = collect_predictions(
        predictions_dir=args.predictions_dir,
        view_metadata_path=args.view_metadata,
        pose_feature_dir=args.pose_feature_dir,
        split=args.split,
        threshold_kind=args.threshold_kind,
    )
    error_rows = build_error_rows(predictions)
    top_rows = top_error_rows(error_rows, top_n=args.top_n)
    summary_rows = summarize_errors(predictions, error_rows)

    prefix = f"{args.split}_{args.threshold_kind}"
    error_fieldnames = list(error_row_dict(error_rows[0]).keys()) if error_rows else list(error_row_dict_placeholder())
    write_dict_rows(
        args.output_dir / f"{prefix}_errors.csv",
        [error_row_dict(row) for row in error_rows],
        error_fieldnames,
    )
    write_dict_rows(
        args.output_dir / f"{prefix}_top_errors.csv",
        [error_row_dict(row) for row in top_rows],
        error_fieldnames,
    )
    write_dict_rows(
        args.output_dir / f"{prefix}_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()) if summary_rows else list(summary_row_placeholder()),
    )

    print(f"Saved errors to {args.output_dir / f'{prefix}_errors.csv'}")
    print(f"Saved top errors to {args.output_dir / f'{prefix}_top_errors.csv'}")
    print(f"Saved summary to {args.output_dir / f'{prefix}_summary.csv'}")
    print_summary(error_rows, top_rows)


def error_row_dict_placeholder() -> dict[str, object]:
    return {
        "label_mode": "",
        "seed": "",
        "split": "",
        "video_id": "",
        "error_type": "",
        "label": 0,
        "prediction": 0,
        "probability": 0.0,
        "selected_threshold": 0.0,
        "confidence_margin": 0.0,
        "view_type": "",
        "view_confidence": 0.0,
        "quality_pose_detected_ratio": 0.0,
        "quality_valid_lower_body_ratio": 0.0,
        "quality_bottom_frame_ratio": 0.0,
        "valid_frame_ratio": 0.0,
    }


def summary_row_placeholder() -> dict[str, object]:
    return {
        "label_mode": "",
        "seed": "",
        "view_type": "",
        "error_type": "",
        "eligible_count": 0.0,
        "n_errors": 0.0,
        "error_rate": 0.0,
        "mean_probability": 0.0,
        "mean_confidence_margin": 0.0,
        "mean_view_confidence": 0.0,
        "mean_quality_pose_detected_ratio": 0.0,
        "mean_quality_valid_lower_body_ratio": 0.0,
        "mean_quality_bottom_frame_ratio": 0.0,
        "mean_valid_frame_ratio": 0.0,
    }


if __name__ == "__main__":
    main()
