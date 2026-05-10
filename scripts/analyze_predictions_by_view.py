from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


THRESHOLD_COLUMNS = {
    "fixed_0_5": "fixed_0_5_prediction",
    "selected_threshold": "selected_threshold_prediction",
}


@dataclass(frozen=True)
class JoinedPrediction:
    label_mode: str
    seed: str
    split: str
    video_id: str
    label: int
    probability: float
    fixed_0_5_prediction: int
    selected_threshold_prediction: int
    selected_threshold: float
    view_type: str
    view_confidence: float


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


def load_view_metadata(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"split", "video_id", "view_type", "view_confidence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
        return {(row["split"], row["video_id"]): row for row in reader}


def infer_seed(path: Path) -> str:
    match = re.search(r"_seed(\d+)_predictions\.csv$", path.name)
    return match.group(1) if match else ""


def iter_prediction_paths(predictions: Path | None, predictions_dir: Path | None) -> list[Path]:
    if predictions and predictions_dir:
        raise ValueError("Use either --predictions or --predictions-dir, not both.")
    if predictions:
        return [predictions]
    if predictions_dir:
        paths = sorted(predictions_dir.glob("*_predictions.csv"))
        if not paths:
            raise ValueError(f"No *_predictions.csv files found in {predictions_dir}")
        return paths
    raise ValueError("One of --predictions or --predictions-dir is required.")


def iter_joined_predictions(
    predictions_path: Path,
    view_metadata: dict[tuple[str, str], dict[str, str]],
) -> Iterable[JoinedPrediction]:
    seed = infer_seed(predictions_path)
    with predictions_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "label_mode",
            "split",
            "video_id",
            "label",
            "probability",
            "fixed_0_5_prediction",
            "selected_threshold_prediction",
            "selected_threshold",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {predictions_path}: {', '.join(sorted(missing))}")

        missing_views: list[str] = []
        for row in reader:
            video_id = row["video_id"]
            split = row["split"]
            view_row = view_metadata.get((split, video_id))
            if view_row is None:
                missing_views.append(f"{split}/{video_id}")
                continue

            yield JoinedPrediction(
                label_mode=row["label_mode"],
                seed=seed,
                split=split,
                video_id=video_id,
                label=parse_int(row["label"], "label"),
                probability=parse_float(row["probability"], "probability"),
                fixed_0_5_prediction=parse_int(row["fixed_0_5_prediction"], "fixed_0_5_prediction"),
                selected_threshold_prediction=parse_int(
                    row["selected_threshold_prediction"],
                    "selected_threshold_prediction",
                ),
                selected_threshold=parse_float(row["selected_threshold"], "selected_threshold"),
                view_type=view_row["view_type"],
                view_confidence=parse_float(view_row["view_confidence"], "view_confidence"),
            )

        if missing_views:
            preview = ", ".join(missing_views[:10])
            suffix = "" if len(missing_views) <= 10 else f" ... (+{len(missing_views) - 10} more)"
            print(f"Warning: {len(missing_views)} predictions had no view metadata: {preview}{suffix}")


def compute_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    tp = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 1)
    fp = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 1)
    tn = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 0)
    fn = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 0)

    total = len(labels)
    positives = sum(labels)
    negatives = total - positives
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    false_negative_rate = fn / (fn + tp) if fn + tp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    negative_recall = specificity
    negative_f1 = (
        2 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if negative_precision + negative_recall
        else 0.0
    )
    macro_f1 = (f1 + negative_f1) / 2
    balanced_accuracy = (recall + specificity) / 2

    return {
        "n": float(total),
        "positive_count": float(positives),
        "negative_count": float(negatives),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def build_rows(joined_predictions: list[JoinedPrediction]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups: dict[tuple[str, str, str, str], list[JoinedPrediction]] = defaultdict(list)
    for prediction in joined_predictions:
        groups[(prediction.label_mode, prediction.seed, prediction.split, prediction.view_type)].append(prediction)
        groups[(prediction.label_mode, prediction.seed, prediction.split, "ALL")].append(prediction)

    for (label_mode, seed, split, view_type), predictions in sorted(groups.items()):
        labels = [prediction.label for prediction in predictions]
        mean_view_confidence = sum(prediction.view_confidence for prediction in predictions) / len(predictions)
        mean_probability = sum(prediction.probability for prediction in predictions) / len(predictions)
        selected_threshold = predictions[0].selected_threshold if predictions else 0.0

        for threshold_kind, prediction_column in THRESHOLD_COLUMNS.items():
            predicted_labels = [getattr(prediction, prediction_column) for prediction in predictions]
            metrics = compute_metrics(labels, predicted_labels)
            row: dict[str, object] = {
                "label_mode": label_mode,
                "seed": seed,
                "split": split,
                "view_type": view_type,
                "threshold_kind": threshold_kind,
                "selected_threshold": selected_threshold,
                "mean_probability": mean_probability,
                "mean_view_confidence": mean_view_confidence,
            }
            row.update(metrics)
            rows.append(row)

    return rows


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    numeric_fields = [
        "selected_threshold",
        "mean_probability",
        "mean_view_confidence",
        "n",
        "positive_count",
        "negative_count",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "false_positive_rate",
        "false_negative_rate",
        "balanced_accuracy",
        "macro_f1",
        "f1",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(row["label_mode"], row["split"], row["view_type"], row["threshold_kind"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for (label_mode, split, view_type, threshold_kind), grouped_rows in sorted(groups.items()):
        summary_row: dict[str, object] = {
            "label_mode": label_mode,
            "split": split,
            "view_type": view_type,
            "threshold_kind": threshold_kind,
            "seed_count": float(len(grouped_rows)),
        }
        for field in numeric_fields:
            values = [float(row[field]) for row in grouped_rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            summary_row[f"{field}_mean"] = mean
            summary_row[f"{field}_std"] = variance ** 0.5
        summary_rows.append(summary_row)

    return summary_rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label_mode",
        "seed",
        "split",
        "view_type",
        "threshold_kind",
        "selected_threshold",
        "mean_probability",
        "mean_view_confidence",
        "n",
        "positive_count",
        "negative_count",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "false_positive_rate",
        "false_negative_rate",
        "balanced_accuracy",
        "macro_f1",
        "f1",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key in fieldnames:
                value = row[key]
                formatted[key] = f"{value:.6f}" if isinstance(value, float) else value
            writer.writerow(formatted)


def write_summary_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key in fieldnames:
                value = row[key]
                formatted[key] = f"{value:.6f}" if isinstance(value, float) else value
            writer.writerow(formatted)


def print_test_summary(rows: list[dict[str, object]]) -> None:
    print("Test selected-threshold metrics by view:")
    for row in rows:
        if row["split"] != "test" or row["threshold_kind"] != "selected_threshold":
            continue
        print(
            f"  {row['label_mode']} seed={row['seed'] or 'NA'} {row['view_type']}: n={int(row['n'])} "
            f"pos={int(row['positive_count'])} neg={int(row['negative_count'])} "
            f"bal_acc={row['balanced_accuracy']:.3f} recall={row['recall']:.3f} "
            f"specificity={row['specificity']:.3f} macro_f1={row['macro_f1']:.3f} "
            f"fp={int(row['fp'])} fn={int(row['fn'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze classifier predictions grouped by estimated view type.")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument(
        "--view-metadata",
        type=Path,
        default=Path("data/Squat/Labeled_Dataset/view_metadata.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    view_metadata = load_view_metadata(args.view_metadata)
    joined_predictions = []
    for predictions_path in iter_prediction_paths(args.predictions, args.predictions_dir):
        joined_predictions.extend(iter_joined_predictions(predictions_path, view_metadata))
    if not joined_predictions:
        raise SystemExit("No predictions could be joined with view metadata.")

    rows = build_rows(joined_predictions)
    write_rows(args.output, rows)
    print(f"Saved view analysis to {args.output}")
    if args.summary_output:
        summary_rows = summarize_rows(rows)
        write_summary_rows(args.summary_output, summary_rows)
        print(f"Saved view analysis summary to {args.summary_output}")
    print_test_summary(rows)


if __name__ == "__main__":
    main()
