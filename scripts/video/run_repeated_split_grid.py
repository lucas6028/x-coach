"""Train one arm across every repeated-split fold.

    .venv\\Scripts\\python.exe scripts/video/run_repeated_split_grid.py \\
        --feature-dir data/Fitness-AQA/Squat/Labeled_Dataset/pose_features \\
        --output-root tmp/repeated_splits/pose

Folds are generated once into ``--fold-root`` and reused by every arm, so all arms are
scored on identical partitions and the comparison stays paired. Re-running with the
same seed reproduces them; the fold manifest records the sizes actually used.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.repeated_splits import (
    DEFAULT_FOLDS,
    DEFAULT_REPEATS,
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    make_folds,
    write_all_folds,
)
from src.video.squat_dataset import SQUAT_LABELED_ROOT, load_json_list


def build_labels(forward_path: Path, inward_path: Path, video_ids: list[str]) -> dict[str, int]:
    """Combined-mode labels: an id with any logged error interval is a positive.

    Mirrors ``videomae_video_classifier.build_labels`` for label_mode='combined'. It is
    needed here only to stratify the folds; the classifier still builds its own labels
    from the same files at training time.
    """
    with forward_path.open("r", encoding="utf-8") as f:
        forward = json.load(f)
    with inward_path.open("r", encoding="utf-8") as f:
        inward = json.load(f)
    return {
        video_id: int(bool(forward.get(video_id)) or bool(inward.get(video_id)))
        for video_id in video_ids
    }


def run_fold(args: argparse.Namespace, fold_dir: Path, fold_name: str) -> Path:
    metrics_path = args.output_root / "metrics" / f"{args.label_mode}_{fold_name}_metrics.json"
    predictions_path = args.output_root / "predictions" / f"{args.label_mode}_{fold_name}_predictions.csv"
    checkpoint_path = args.output_root / "checkpoints" / f"{args.label_mode}_{fold_name}.pt"

    if predictions_path.exists() and not args.overwrite:
        return metrics_path

    command = [
        sys.executable,
        str(args.classifier_script),
        "--feature-dir", str(args.feature_dir),
        "--train-keys", str(fold_dir / "train_keys.json"),
        "--val-keys", str(fold_dir / "val_keys.json"),
        "--test-keys", str(fold_dir / "test_keys.json"),
        "--forward-labels", str(args.forward_labels),
        "--inward-labels", str(args.inward_labels),
        "--output-path", str(checkpoint_path),
        "--predictions-output", str(predictions_path),
        "--summary-output", str(metrics_path),
        "--label-mode", args.label_mode,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--hidden-dim", str(args.hidden_dim),
        "--dropout", str(args.dropout),
        "--weight-decay", str(args.weight_decay),
        "--early-stopping-patience", str(args.early_stopping_patience),
        # The fold IS the randomisation, so the training seed is held fixed. Varying
        # both would confound split noise with initialisation noise in one number.
        "--seed", str(args.seed),
        "--threshold-objective", args.threshold_objective,
    ]
    if args.device:
        command.extend(["--device", args.device])
    if args.normalize_features:
        command.append("--normalize-features")

    print(f"  {fold_name}")
    subprocess.run(command, check=True, capture_output=True)
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one arm over repeated stratified splits.")
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, default=ROOT / "tmp" / "repeated_splits" / "folds")
    parser.add_argument("--classifier-script", type=Path, default=ROOT / "scripts" / "video" / "train_videomae_classifier.py")
    parser.add_argument("--split-dir", type=Path, default=SQUAT_LABELED_ROOT / "Splits")
    parser.add_argument("--forward-labels", type=Path, default=SQUAT_LABELED_ROOT / "Labels" / "error_knees_forward.json")
    parser.add_argument("--inward-labels", type=Path, default=SQUAT_LABELED_ROOT / "Labels" / "error_knees_inward.json")
    parser.add_argument("--label-mode", default="combined")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--fold-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seed", type=int, default=1, help="Training seed, held fixed across folds.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--threshold-objective", default="balanced_accuracy")
    parser.add_argument("--normalize-features", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    video_ids = sorted(
        video_id
        for split_name in ("train", "val", "test")
        for video_id in load_json_list(args.split_dir / f"{split_name}_keys.json")
    )
    labels = build_labels(args.forward_labels, args.inward_labels, video_ids)
    positives = sum(labels.values())
    print(f"{len(video_ids)} videos, {positives} positive ({positives / len(video_ids):.1%})")

    folds = make_folds(
        video_ids,
        labels,
        n_repeats=args.repeats,
        n_folds=args.folds,
        val_fraction=args.val_fraction,
        seed=args.fold_seed,
    )
    manifest = write_all_folds(folds, args.fold_root)
    train_size, val_size, test_size = folds[0].sizes()
    print(f"{manifest['n_folds']} folds under {args.fold_root} (train/val/test = {train_size}/{val_size}/{test_size})")

    print(f"Training {args.feature_dir.name} on every fold:")
    for fold in folds:
        run_fold(args, args.fold_root / fold.name, fold.name)

    print(f"Done. Predictions under {args.output_root / 'predictions'}")


if __name__ == "__main__":
    main()
