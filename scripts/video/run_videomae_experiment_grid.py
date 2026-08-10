from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_MODES = ("combined", "knees_forward", "knees_inward")
THRESHOLD_KINDS = (
    "fixed_0_5",
    "selected_threshold",
    "f1_selected_threshold",
    "always_positive",
    "always_negative",
)
METRIC_FIELDS = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "false_positive_rate",
    "balanced_accuracy",
    "macro_f1",
    "f1",
    "tp",
    "fp",
    "tn",
    "fn",
)


def parse_csv_list(value: str, choices: tuple[str, ...] | None = None) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if choices is not None:
        invalid = sorted(set(items) - set(choices))
        if invalid:
            raise argparse.ArgumentTypeError(f"Unsupported values: {', '.join(invalid)}")
    return items


def parse_seed_list(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers.") from exc


def run_classifier(args: argparse.Namespace, label_mode: str, seed: int) -> Path:
    checkpoint_path = args.output_root / "checkpoints" / f"{label_mode}_seed{seed}.pt"
    predictions_path = args.output_root / "predictions" / f"{label_mode}_seed{seed}_predictions.csv"
    metrics_path = args.output_root / "metrics" / f"{label_mode}_seed{seed}_metrics.json"

    command = [
        sys.executable,
        str(args.classifier_script),
        "--feature-dir",
        str(args.feature_dir),
        "--train-keys",
        str(args.train_keys),
        "--test-keys",
        str(args.test_keys),
        "--val-keys",
        str(args.val_keys),
        "--forward-labels",
        str(args.forward_labels),
        "--inward-labels",
        str(args.inward_labels),
        "--output-path",
        str(checkpoint_path),
        "--predictions-output",
        str(predictions_path),
        "--summary-output",
        str(metrics_path),
        "--label-mode",
        label_mode,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--hidden-dim",
        str(args.hidden_dim),
        "--dropout",
        str(args.dropout),
        "--weight-decay",
        str(args.weight_decay),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--seed",
        str(seed),
        "--threshold-objective",
        args.threshold_objective,
    ]
    if args.device:
        command.extend(["--device", args.device])
    if args.normalize_features:
        command.append("--normalize-features")

    print(f"Running label_mode={label_mode} seed={seed}")
    subprocess.run(command, check=True)
    return metrics_path


def summary_rows(metrics_path: Path) -> list[dict[str, Any]]:
    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    config = payload["config"]
    rows: list[dict[str, Any]] = []
    for split_name, split_metrics in payload["metrics"].items():
        for threshold_kind in THRESHOLD_KINDS:
            metrics = split_metrics[threshold_kind]
            row: dict[str, Any] = {
                "label_mode": config["label_mode"],
                "seed": config["seed"],
                "split": split_name,
                "threshold_kind": threshold_kind,
                "threshold": metrics["threshold"],
            }
            for field in METRIC_FIELDS:
                row[field] = metrics[field]
            rows.append(row)
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label_mode",
        "seed",
        "split",
        "threshold_kind",
        "threshold",
        *METRIC_FIELDS,
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VideoMAE classifier experiments over label modes and seeds.")
    # The thin entry point, not the module file: `src/video/videomae_video_classifier.py`
    # imports `src.video.classification_metrics`, which needs the repo root on sys.path.
    # Running the module file directly puts `src/video/` there instead and fails to import.
    parser.add_argument("--classifier-script", type=Path, default=REPO_ROOT / "scripts" / "video" / "train_videomae_classifier.py")
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--train-keys", type=Path, required=True)
    parser.add_argument("--val-keys", type=Path, required=True)
    parser.add_argument("--test-keys", type=Path, required=True)
    parser.add_argument("--forward-labels", type=Path, required=True)
    parser.add_argument("--inward-labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label-modes", type=lambda value: parse_csv_list(value, LABEL_MODES), default=list(LABEL_MODES))
    parser.add_argument("--seeds", type=parse_seed_list, default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--threshold-objective", choices=("f1", "balanced_accuracy", "macro_f1", "youden_j"), default="balanced_accuracy")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="Pass --normalize-features to each classifier run.",
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()

    args.classifier_script = args.classifier_script.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    metrics_paths: list[Path] = []
    for label_mode in args.label_modes:
        for seed in args.seeds:
            metrics_paths.append(run_classifier(args, label_mode=label_mode, seed=seed))

    rows: list[dict[str, Any]] = []
    for metrics_path in metrics_paths:
        rows.extend(summary_rows(metrics_path))

    summary_path = args.summary_output or args.output_root / "metrics" / "experiment_summary.csv"
    write_summary_csv(summary_path, rows)
    print(f"Saved experiment summary to {summary_path}")


if __name__ == "__main__":
    main()
