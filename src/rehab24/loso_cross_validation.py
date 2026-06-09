"""Leave-One-Subject-Out cross-validation for the REHAB24-6 correctness classifier.

The fixed REHAB24-6 split tests on only two subjects (8, 9), so a single test
balanced-accuracy number is high-variance and tied to those two people. This
driver rotates every subject through the test position once, training a fresh
classifier per fold, and reports mean +/- std across folds plus a pooled metric
(every subject held out exactly once, predictions concatenated). It reuses the
exact training/evaluation primitives of ``train_correctness_classifier`` so the
per-fold model matches the single-split baseline.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.rehab24.dataset import DEFAULT_PROCESSED_ROOT, load_manifest
from src.video.videomae_video_classifier import (
    VideoFeatureClassifier,
    build_samples,
    collect_sample_predictions,
    compute_feature_normalization,
    compute_metrics,
    find_best_threshold,
    label_counts,
    make_loader,
    set_seed,
)

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("REHAB24-6 LOSO cross-validation requires `torch`.") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]

# Subjects with at least this many samples are eligible to serve as the per-fold
# validation subject (P10 has only 16 samples, too few for threshold selection).
MIN_VAL_SUBJECT_SAMPLES = 100


@dataclass
class FoldConfig:
    epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-4
    hidden_dim: int = 128
    dropout: float = 0.4
    weight_decay: float = 0.01
    early_stopping_patience: int = 5
    normalize_features: bool = True
    threshold_objective: str = "balanced_accuracy"


def subjects_to_samples(manifest_path: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in load_manifest(manifest_path):
        grouped[row["person_id"]].append(row["sample_id"])
    return dict(grouped)


def pick_val_subject(test_subject: str, ordered_subjects: list[str], sample_counts: dict[str, int]) -> str:
    """Cyclically pick the next subject after ``test_subject`` with enough samples."""
    start = ordered_subjects.index(test_subject)
    n = len(ordered_subjects)
    for offset in range(1, n):
        candidate = ordered_subjects[(start + offset) % n]
        if candidate != test_subject and sample_counts[candidate] >= MIN_VAL_SUBJECT_SAMPLES:
            return candidate
    raise ValueError("No eligible validation subject found.")


def train_one_fold(
    feature_dir: Path,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    labels: dict[str, int],
    config: FoldConfig,
    device: torch.device,
    seed: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Train one fold; return (selected_threshold, test_probabilities, test_labels)."""
    set_seed(seed)
    train_samples = build_samples(feature_dir, train_ids, labels)
    val_samples = build_samples(feature_dir, val_ids, labels)
    test_samples = build_samples(feature_dir, test_ids, labels)
    if not train_samples or not val_samples or not test_samples:
        raise SystemExit("A fold is missing feature files. Run feature extraction first.")

    with np.load(train_samples[0].feature_path, allow_pickle=False) as data:
        feature_dim = int(data["video_feature"].shape[0])

    normalization = compute_feature_normalization(train_samples) if config.normalize_features else None
    model = VideoFeatureClassifier(feature_dim=feature_dim, hidden_dim=config.hidden_dim, dropout=config.dropout).to(device)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = make_loader(train_samples, config.batch_size, shuffle=True, normalization=normalization, generator=generator)

    train_positives, train_negatives = label_counts(train_samples)
    pos_weight = torch.tensor([train_negatives / max(train_positives, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_state = None
    best_val_score = -1.0
    best_threshold = 0.5
    epochs_without_improvement = 0

    for _ in range(1, config.epochs + 1):
        model.train()
        for features, labels_batch in train_loader:
            features = features.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels_batch)
            loss.backward()
            optimizer.step()

        _, val_prob, val_labels = collect_sample_predictions(model, val_samples, config.batch_size, device, normalization=normalization)
        val_threshold, val_metrics = find_best_threshold(val_prob, val_labels, objective=config.threshold_objective)

        if val_metrics[config.threshold_objective] > best_val_score:
            best_val_score = val_metrics[config.threshold_objective]
            best_threshold = float(val_threshold)
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if config.early_stopping_patience > 0 and epochs_without_improvement >= config.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_prob, test_labels = collect_sample_predictions(model, test_samples, config.batch_size, device, normalization=normalization)
    return best_threshold, test_prob, test_labels


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "min": float(arr.min()), "max": float(arr.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-One-Subject-Out CV for the REHAB24-6 correctness classifier.")
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "skeleton_features")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--labels", type=Path, default=DEFAULT_PROCESSED_ROOT / "labels" / "correctness.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--no-normalize-features", action="store_false", dest="normalize_features", default=True)
    parser.add_argument("--threshold-objective", choices=("f1", "balanced_accuracy", "macro_f1", "youden_j"), default="balanced_accuracy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or auto.")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_PROCESSED_ROOT / "correctness_loso_metrics.json")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    config = FoldConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        normalize_features=args.normalize_features,
        threshold_objective=args.threshold_objective,
    )

    labels = {key: int(value) for key, value in json.load(args.labels.open()).items()}
    subject_samples = subjects_to_samples(args.manifest)
    sample_counts = {p: len(ids) for p, ids in subject_samples.items()}
    ordered_subjects = sorted(subject_samples, key=int)

    print(f"LOSO cross-validation on {device} | feature_dir={args.feature_dir.name}")
    print(f"Subjects: {', '.join('P' + s + f'(n={sample_counts[s]})' for s in ordered_subjects)}")
    print(f"{'fold(test)':<12}{'val':>5}{'n_test':>8}{'pos%':>7}{'thr':>7}{'bal_acc':>9}{'macro_f1':>10}{'acc':>7}")

    fold_records = []
    pooled_prob: list[float] = []
    pooled_labels: list[int] = []
    for test_subject in ordered_subjects:
        val_subject = pick_val_subject(test_subject, ordered_subjects, sample_counts)
        test_ids = subject_samples[test_subject]
        val_ids = subject_samples[val_subject]
        train_ids = [sid for s in ordered_subjects if s not in {test_subject, val_subject} for sid in subject_samples[s]]

        threshold, test_prob, test_labels = train_one_fold(
            args.feature_dir, train_ids, val_ids, test_ids, labels, config, device, args.seed
        )
        metrics = compute_metrics(test_prob, test_labels, threshold=threshold)
        pos_rate = float(np.mean(test_labels)) * 100
        fold_records.append(
            {
                "test_subject": test_subject,
                "val_subject": val_subject,
                "n_test": int(len(test_labels)),
                "pos_rate": pos_rate,
                "threshold": threshold,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
            }
        )
        pooled_prob.extend(test_prob.tolist())
        pooled_labels.extend(int(x) for x in test_labels.tolist())
        print(
            f"P{test_subject:<11}{('P' + val_subject):>5}{len(test_labels):>8}{pos_rate:>6.0f}%"
            f"{threshold:>7.2f}{metrics['balanced_accuracy']:>9.3f}{metrics['macro_f1']:>10.3f}{metrics['accuracy']:>7.3f}"
        )

    # Macro across folds: every subject weighted equally (incl. tiny P10).
    bal_all = [r["balanced_accuracy"] for r in fold_records]
    f1_all = [r["macro_f1"] for r in fold_records]
    # Excluding the under-powered P10 fold (n=16) for a more stable read.
    big = [r for r in fold_records if r["n_test"] >= MIN_VAL_SUBJECT_SAMPLES]
    bal_big = [r["balanced_accuracy"] for r in big]
    f1_big = [r["macro_f1"] for r in big]
    pooled_metrics = compute_metrics(np.asarray(pooled_prob), np.asarray(pooled_labels), threshold=0.5)

    print("\n=== LOSO summary (test balanced_accuracy) ===")
    print(f"all 10 folds   : mean={np.mean(bal_all):.3f} +/- {np.std(bal_all):.3f}  (min {np.min(bal_all):.3f}, max {np.max(bal_all):.3f})")
    print(f"9 folds (no P10): mean={np.mean(bal_big):.3f} +/- {np.std(bal_big):.3f}  (min {np.min(bal_big):.3f}, max {np.max(bal_big):.3f})")
    print(f"macro_f1 (9 folds): mean={np.mean(f1_big):.3f} +/- {np.std(f1_big):.3f}")
    print(f"pooled @0.5 (all subjects held out once): bal_acc={pooled_metrics['balanced_accuracy']:.3f} macro_f1={pooled_metrics['macro_f1']:.3f} acc={pooled_metrics['accuracy']:.3f}")

    payload = {
        "feature_dir": str(args.feature_dir),
        "config": vars(config) if hasattr(config, "__dict__") else config.__dict__,
        "folds": fold_records,
        "summary": {
            "balanced_accuracy_all": summarize(bal_all),
            "balanced_accuracy_no_p10": summarize(bal_big),
            "macro_f1_no_p10": summarize(f1_big),
            "pooled": pooled_metrics,
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"\nSaved LOSO summary to {args.summary_output}")


if __name__ == "__main__":
    main()
