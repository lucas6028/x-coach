from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np

from src.rehab24.dataset import DEFAULT_PROCESSED_ROOT, load_manifest
from src.video.videomae_video_classifier import (
    VideoFeatureClassifier,
    build_samples,
    build_threshold_metrics,
    collect_sample_predictions,
    compute_feature_normalization,
    compute_metrics,
    feature_normalization_payload,
    find_best_threshold,
    format_metrics,
    label_counts,
    load_json_list,
    load_json_mapping,
    make_loader,
    print_threshold_report,
    set_seed,
)

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("REHAB24-6 classifier training requires `torch`.") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_metadata_by_sample(manifest_path: Path) -> dict[str, dict[str, str]]:
    return {row["sample_id"]: row for row in load_manifest(manifest_path)}


def selected_metrics_by_exercise(
    video_ids: list[str],
    probabilities: np.ndarray,
    labels: np.ndarray,
    metadata_by_sample: dict[str, dict[str, str]],
    threshold: float,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    exercise_ids = sorted({metadata_by_sample[sample_id]["exercise_id"] for sample_id in video_ids}, key=int)
    for exercise_id in exercise_ids:
        indices = [i for i, sample_id in enumerate(video_ids) if metadata_by_sample[sample_id]["exercise_id"] == exercise_id]
        exercise_probabilities = probabilities[indices]
        exercise_labels = labels[indices]
        result[f"Ex{exercise_id}"] = compute_metrics(exercise_probabilities, exercise_labels, threshold=threshold)
    return result


def write_rehab24_predictions_csv(
    path: Path,
    rows: list[tuple[str, list[str], np.ndarray, np.ndarray]],
    metadata_by_sample: dict[str, dict[str, str]],
    selected_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "sample_id",
                "exercise_id",
                "person_id",
                "camera",
                "label",
                "probability",
                "fixed_0_5_prediction",
                "selected_threshold_prediction",
                "selected_threshold",
            ],
        )
        writer.writeheader()
        for split_name, sample_ids, probabilities, labels in rows:
            for sample_id, probability, label in zip(sample_ids, probabilities, labels):
                metadata = metadata_by_sample[sample_id]
                writer.writerow(
                    {
                        "split": split_name,
                        "sample_id": sample_id,
                        "exercise_id": metadata["exercise_id"],
                        "person_id": metadata["person_id"],
                        "camera": metadata["camera"],
                        "label": int(label),
                        "probability": f"{float(probability):.8f}",
                        "fixed_0_5_prediction": int(float(probability) >= 0.5),
                        "selected_threshold_prediction": int(float(probability) >= selected_threshold),
                        "selected_threshold": f"{selected_threshold:.8f}",
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight REHAB24-6 correctness classifier.")
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "skeleton_features")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--train-keys", type=Path, default=DEFAULT_PROCESSED_ROOT / "splits" / "train_keys.json")
    parser.add_argument("--val-keys", type=Path, default=DEFAULT_PROCESSED_ROOT / "splits" / "val_keys.json")
    parser.add_argument("--test-keys", type=Path, default=DEFAULT_PROCESSED_ROOT / "splits" / "test_keys.json")
    parser.add_argument("--labels", type=Path, default=DEFAULT_PROCESSED_ROOT / "labels" / "correctness.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--normalize-features", action="store_true", default=True)
    parser.add_argument("--no-normalize-features", action="store_false", dest="normalize_features")
    parser.add_argument("--threshold-objective", choices=("f1", "balanced_accuracy", "macro_f1", "youden_j"), default="balanced_accuracy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or auto.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_PROCESSED_ROOT / "correctness_classifier.pt")
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PROCESSED_ROOT / "correctness_predictions.csv")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_PROCESSED_ROOT / "correctness_metrics.json")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ids = load_json_list(args.train_keys)
    val_ids = load_json_list(args.val_keys)
    test_ids = load_json_list(args.test_keys)
    labels = {key: int(value) for key, value in load_json_mapping(args.labels).items()}
    metadata_by_sample = load_metadata_by_sample(args.manifest)

    train_samples = build_samples(args.feature_dir, train_ids, labels)
    val_samples = build_samples(args.feature_dir, val_ids, labels)
    test_samples = build_samples(args.feature_dir, test_ids, labels)
    if not train_samples:
        raise SystemExit("No training feature files were found. Run REHAB24-6 skeleton feature extraction first.")

    with np.load(train_samples[0].feature_path, allow_pickle=False) as data:
        feature_dim = int(data["video_feature"].shape[0])

    feature_normalization = compute_feature_normalization(train_samples) if args.normalize_features else None
    model = VideoFeatureClassifier(feature_dim=feature_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = make_loader(train_samples, args.batch_size, shuffle=True, normalization=feature_normalization, generator=train_generator)
    train_eval_loader = make_loader(train_samples, args.batch_size, shuffle=False, normalization=feature_normalization)
    val_loader = make_loader(val_samples, args.batch_size, shuffle=False, normalization=feature_normalization)

    train_positives, train_negatives = label_counts(train_samples)
    val_positives, val_negatives = label_counts(val_samples)
    test_positives, test_negatives = label_counts(test_samples)
    pos_weight = torch.tensor([train_negatives / max(train_positives, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_score = -1.0
    epochs_without_improvement = 0
    print(f"Training REHAB24-6 correctness classifier on {device}.")
    print(f"Samples: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)} feature_dim={feature_dim}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for features, labels_batch in train_loader:
            features = features.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * features.size(0)

        train_ids_epoch, train_prob_epoch, train_labels_epoch = collect_sample_predictions(
            model, train_samples, args.batch_size, device, normalization=feature_normalization
        )
        del train_ids_epoch
        val_ids_epoch, val_prob_epoch, val_labels_epoch = collect_sample_predictions(
            model, val_samples, args.batch_size, device, normalization=feature_normalization
        )
        del val_ids_epoch
        val_threshold, val_metrics = find_best_threshold(val_prob_epoch, val_labels_epoch, objective=args.threshold_objective)
        train_metrics = compute_metrics(train_prob_epoch, train_labels_epoch, threshold=0.5)
        avg_loss = running_loss / max(len(train_samples), 1)
        print(
            f"Epoch {epoch:02d} | loss={avg_loss:.4f} train_f1={train_metrics['f1']:.3f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.3f} val_f1={val_metrics['f1']:.3f} "
            f"val_specificity={val_metrics['specificity']:.3f} val_threshold={val_threshold:.3f}"
        )

        if val_metrics[args.threshold_objective] > best_val_score:
            best_val_score = val_metrics[args.threshold_objective]
            epochs_without_improvement = 0
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "feature_dim": feature_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "threshold": float(val_threshold),
                "threshold_objective": args.threshold_objective,
                "task": "rehab24_correctness",
                "seed": args.seed,
                "feature_normalization": feature_normalization_payload(feature_normalization),
                "epoch": epoch,
                "best_val_metrics": val_metrics,
            }
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch:02d}.")
                break

    if best_state is None:
        raise SystemExit("No best checkpoint was selected. Use --epochs greater than 0.")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output_path)
    print(f"Saved best checkpoint to {args.output_path}")

    model.load_state_dict(best_state["model_state_dict"])
    best_threshold = float(best_state["threshold"])
    train_video_ids, train_probabilities, train_labels = collect_sample_predictions(
        model, train_samples, args.batch_size, device, normalization=feature_normalization
    )
    val_video_ids, val_probabilities, val_labels = collect_sample_predictions(
        model, val_samples, args.batch_size, device, normalization=feature_normalization
    )
    test_video_ids, test_probabilities, test_labels = collect_sample_predictions(
        model, test_samples, args.batch_size, device, normalization=feature_normalization
    )
    f1_threshold, _ = find_best_threshold(val_probabilities, val_labels, objective="f1")

    split_predictions = [
        ("train", train_video_ids, train_probabilities, train_labels),
        ("val", val_video_ids, val_probabilities, val_labels),
        ("test", test_video_ids, test_probabilities, test_labels),
    ]
    split_metrics = {
        split_name: build_threshold_metrics(probabilities, labels_array, best_threshold, f1_threshold)
        for split_name, _, probabilities, labels_array in split_predictions
    }
    per_exercise_metrics = {
        split_name: selected_metrics_by_exercise(sample_ids, probabilities, labels_array, metadata_by_sample, best_threshold)
        for split_name, sample_ids, probabilities, labels_array in split_predictions
    }

    print(f"Selected threshold objective: {args.threshold_objective}")
    print(f"Validation F1 comparison threshold: {f1_threshold:.3f}")
    for split_name, _, probabilities, labels_array in split_predictions:
        print_threshold_report(split_name.title(), probabilities, labels_array, best_threshold, f1_threshold)
    print(format_metrics("Test selected-threshold", split_metrics["test"]["selected_threshold"]))

    write_rehab24_predictions_csv(args.predictions_output, split_predictions, metadata_by_sample, best_threshold)
    print(f"Saved predictions CSV to {args.predictions_output}")

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "task": "rehab24_correctness",
            "feature_dir": str(args.feature_dir),
            "manifest": str(args.manifest),
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "normalize_features": args.normalize_features,
            "threshold_objective": args.threshold_objective,
        },
        "class_balance": {
            "train": {"positives": train_positives, "negatives": train_negatives},
            "val": {"positives": val_positives, "negatives": val_negatives},
            "test": {"positives": test_positives, "negatives": test_negatives},
        },
        "best_checkpoint": {
            "epoch": best_state["epoch"],
            "threshold": best_state["threshold"],
            "threshold_objective": best_state["threshold_objective"],
            "best_val_metrics": best_state["best_val_metrics"],
        },
        "metrics": split_metrics,
        "per_exercise_metrics": per_exercise_metrics,
    }
    with args.summary_output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Saved summary JSON to {args.summary_output}")


if __name__ == "__main__":
    main()

