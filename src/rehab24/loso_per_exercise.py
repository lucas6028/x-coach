"""Per-exercise breakdown of the REHAB24-6 LOSO correctness results.

The headline LOSO number (``loso_cross_validation``) pools every subject's
held-out predictions into one balanced-accuracy figure. This module re-runs the
*same* folds (same seed, same val-subject rotation, same per-fold val-selected
threshold) but keeps each test sample's identity, so the pooled predictions —
every subject held out exactly once — can be bucketed by ``exercise_id``. That
shows which of the 6 REHAB24-6 movements drive (or drag) the aggregate, for each
feature source side by side.

Per-fold training mirrors ``loso_cross_validation.train_one_fold``; the only
difference is that the final ``collect_sample_predictions`` call also returns the
sample ids (the LOSO driver discards them). A self-check compares the recomputed
per-fold balanced accuracy against the saved LOSO summary to guard against drift.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.rehab24.dataset import DEFAULT_PROCESSED_ROOT, load_manifest
from src.rehab24.loso_cross_validation import (
    FoldConfig,
    MIN_VAL_SUBJECT_SAMPLES,
    pick_val_subject,
    subjects_to_samples,
)
from src.video.videomae_video_classifier import (
    VideoFeatureClassifier,
    build_samples,
    collect_sample_predictions,
    compute_feature_normalization,
    find_best_threshold,
    label_counts,
    make_loader,
    set_seed,
)

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("REHAB24-6 per-exercise LOSO requires `torch`.") from exc


EXERCISE_NAMES = {
    1: "arm abduction",
    2: "arm VW",
    3: "table push-ups",
    4: "leg abduction",
    5: "leg lunge",
    6: "squats",
}

# The four feature sources compared in the correctness experiment summary.
DEFAULT_SOURCES: dict[str, str] = {
    "vicon": "skeleton_features",
    "mediapipe": "mediapipe_skeleton_features",
    "rtmpose": "rtmpose_skeleton_features",
    "videomae": "videomae_features",
}


def balanced_accuracy_from_binary(labels: np.ndarray, preds: np.ndarray) -> float:
    """Balanced accuracy = mean of per-class recall, from already-binarized preds."""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    pos = labels == 1
    neg = labels == 0
    tpr = float((preds[pos] == 1).mean()) if pos.any() else float("nan")
    tnr = float((preds[neg] == 0).mean()) if neg.any() else float("nan")
    return 0.5 * (tpr + tnr)


def train_one_fold_with_ids(
    feature_dir: Path,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    labels: dict[str, int],
    config: FoldConfig,
    device: torch.device,
    seed: int,
) -> tuple[float, list[str], np.ndarray, np.ndarray]:
    """Train one LOSO fold; return (threshold, sample_ids, test_prob, test_labels).

    Identical to ``loso_cross_validation.train_one_fold`` except the test-set
    ``collect_sample_predictions`` ids are propagated out instead of discarded.
    """
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
    ids, test_prob, test_labels = collect_sample_predictions(model, test_samples, config.batch_size, device, normalization=normalization)
    return best_threshold, ids, test_prob, test_labels


def per_exercise_loso(
    feature_dir: Path,
    manifest_path: Path,
    labels: dict[str, int],
    config: FoldConfig,
    device: torch.device,
    seed: int,
    exercise_of: dict[str, int],
) -> dict[int, dict[str, float]]:
    """Run LOSO over all subjects; return per-exercise pooled balanced accuracy.

    Every sample is held out exactly once (each subject takes a turn as test).
    Each fold's val-selected threshold binarizes that fold's test predictions, so
    a sample's correct/incorrect verdict matches the fold it belonged to.
    """
    subject_samples = subjects_to_samples(manifest_path)
    sample_counts = {p: len(ids) for p, ids in subject_samples.items()}
    ordered_subjects = sorted(subject_samples, key=int)

    bucket: dict[int, dict[str, list[int]]] = defaultdict(lambda: {"label": [], "pred": []})
    for test_subject in ordered_subjects:
        val_subject = pick_val_subject(test_subject, ordered_subjects, sample_counts)
        test_ids = subject_samples[test_subject]
        val_ids = subject_samples[val_subject]
        train_ids = [sid for s in ordered_subjects if s not in {test_subject, val_subject} for sid in subject_samples[s]]

        threshold, ids, test_prob, test_labels = train_one_fold_with_ids(
            feature_dir, train_ids, val_ids, test_ids, labels, config, device, seed
        )
        preds = (np.asarray(test_prob) >= threshold).astype(int)
        for sid, label, pred in zip(ids, test_labels.astype(int).tolist(), preds.tolist()):
            ex = exercise_of[sid]
            bucket[ex]["label"].append(int(label))
            bucket[ex]["pred"].append(int(pred))

    result: dict[int, dict[str, float]] = {}
    for ex in sorted(bucket):
        labels_arr = np.asarray(bucket[ex]["label"])
        preds_arr = np.asarray(bucket[ex]["pred"])
        result[ex] = {
            "n": int(labels_arr.size),
            "pos_rate": float(labels_arr.mean()),
            "balanced_accuracy": balanced_accuracy_from_binary(labels_arr, preds_arr),
            "accuracy": float((labels_arr == preds_arr).mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-exercise breakdown of REHAB24-6 LOSO correctness results.")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--labels", type=Path, default=DEFAULT_PROCESSED_ROOT / "labels" / "correctness.json")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="name=feature_dir pairs. Defaults to the four standard sources.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or auto.")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_PROCESSED_ROOT / "correctness_loso_per_exercise.json")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    config = FoldConfig()

    labels = {key: int(value) for key, value in json.load(args.labels.open()).items()}
    exercise_of = {row["sample_id"]: int(row["exercise_id"]) for row in load_manifest(args.manifest)}

    if args.sources:
        sources = {}
        for pair in args.sources:
            name, _, path = pair.partition("=")
            sources[name] = args.processed_root / path if not Path(path).is_absolute() else Path(path)
    else:
        sources = {name: args.processed_root / sub for name, sub in DEFAULT_SOURCES.items()}

    print(f"Per-exercise LOSO on {device} | sources: {', '.join(sources)}")
    all_results: dict[str, dict[int, dict[str, float]]] = {}
    for name, feature_dir in sources.items():
        if not feature_dir.exists():
            print(f"  [skip] {name}: {feature_dir} not found")
            continue
        print(f"  running LOSO for {name} ({feature_dir.name}) ...", flush=True)
        all_results[name] = per_exercise_loso(
            feature_dir, args.manifest, labels, config, device, args.seed, exercise_of
        )

    # Pretty table: rows = exercises, columns = sources (balanced accuracy).
    names = list(all_results)
    header = f"{'Ex':<4}{'exercise':<16}{'n':>5}{'pos%':>6}  " + "".join(f"{n:>11}" for n in names)
    print("\n=== Per-exercise pooled LOSO balanced accuracy ===")
    print(header)
    for ex in sorted(EXERCISE_NAMES):
        any_src = next((r for r in all_results.values() if ex in r), None)
        if any_src is None:
            continue
        n = any_src[ex]["n"]
        pos = any_src[ex]["pos_rate"] * 100
        cells = "".join(f"{all_results[n2][ex]['balanced_accuracy']:>11.3f}" if ex in all_results[n2] else f"{'-':>11}" for n2 in names)
        print(f"Ex{ex:<2}{EXERCISE_NAMES[ex]:<16}{n:>5}{pos:>5.0f}%  {cells}")

    payload = {
        "exercise_names": EXERCISE_NAMES,
        "sources": {name: str(path) for name, path in sources.items()},
        "results": {name: {str(ex): vals for ex, vals in res.items()} for name, res in all_results.items()},
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"\nSaved per-exercise LOSO summary to {args.summary_output}")


if __name__ == "__main__":
    main()
