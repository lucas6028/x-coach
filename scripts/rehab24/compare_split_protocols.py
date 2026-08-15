"""Score one REHAB24-6 feature set under subject-wise AND subject-blind splits.

The control that separates "this dataset is different" from "the split protocol was
leaking". Fitness-AQA has no participant mapping, so its arms are scored on random
video-level splits and same-athlete leakage cannot be excluded. REHAB24-6 has
``person_id``, so the same features can be scored both ways on the SAME data:

* ``subject``  -- leave-one-subject-out, the protocol every other REHAB24-6 number uses
* ``random``   -- repeated stratified splits ignoring ``person_id``, the Fitness-AQA
                  protocol, which lets a subject's other repetitions sit in training

A feature set that scores near chance under ``subject`` and well under ``random`` is
reading subject identity, not the movement. Everything else -- features, classifier,
hyperparameters, seed -- is held fixed between the two.

    .venv\\Scripts\\python.exe scripts/rehab24/compare_split_protocols.py \\
        --feature-dir data/REHAB24-6/processed/box_geometry_features
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.rehab24.loso_cross_validation import FoldConfig, train_one_fold
from src.video.classification_metrics import compute_metrics
from src.video.repeated_splits import make_folds

DEFAULT_PROCESSED_ROOT = ROOT / "data" / "REHAB24-6" / "processed"


def load_labels(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def load_subjects(manifest_path: Path) -> dict[str, str]:
    import csv

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        return {row["sample_id"]: row["person_id"] for row in csv.DictReader(f)}


def subject_folds(sample_ids: list[str], subjects: dict[str, str]) -> list[tuple[str, list[str], list[str], list[str]]]:
    """Leave-one-subject-out, with the next subject held out for validation."""
    ordered = sorted({subjects[sid] for sid in sample_ids}, key=lambda s: int(s))
    folds = []
    for index, test_subject in enumerate(ordered):
        val_subject = ordered[(index + 1) % len(ordered)]
        test = [sid for sid in sample_ids if subjects[sid] == test_subject]
        val = [sid for sid in sample_ids if subjects[sid] == val_subject]
        train = [sid for sid in sample_ids if subjects[sid] not in (test_subject, val_subject)]
        folds.append((f"P{test_subject}", train, val, test))
    return folds


def random_folds(sample_ids: list[str], labels: dict[str, int], repeats: int, seed: int):
    """Repeated stratified splits that ignore person_id -- the Fitness-AQA protocol."""
    for fold in make_folds(sample_ids, labels, n_repeats=repeats, n_folds=5, seed=seed):
        yield (fold.name, fold.train, fold.val, fold.test)


def run(protocol_folds, feature_dir: Path, labels: dict[str, int], config: FoldConfig, device, seed: int) -> list[float]:
    scores = []
    for name, train, val, test in protocol_folds:
        threshold, probabilities, test_labels = train_one_fold(
            feature_dir, train, val, test, labels, config, device, seed
        )
        score = compute_metrics(probabilities, test_labels, threshold=threshold)["balanced_accuracy"]
        scores.append(score)
        print(f"  {name:<8} n_test={len(test):<5} bal_acc={score:.4f}")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare subject-wise and subject-blind splits on one feature set.")
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--labels", type=Path, default=DEFAULT_PROCESSED_ROOT / "labels" / "correctness.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    subjects = load_subjects(args.manifest)
    sample_ids = sorted(sid for sid in subjects if sid in labels)
    print(f"{len(sample_ids)} samples, {len(set(subjects.values()))} subjects, feature_dir={args.feature_dir.name}")

    config = FoldConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nsubject-wise (leave-one-subject-out):")
    subject_scores = run(subject_folds(sample_ids, subjects), args.feature_dir, labels, config, device, args.seed)

    print("\nsubject-blind (repeated stratified, person_id ignored):")
    random_scores = run(
        random_folds(sample_ids, labels, args.repeats, args.seed), args.feature_dir, labels, config, device, args.seed
    )

    summary = {
        "feature_dir": str(args.feature_dir),
        "subject_wise": {"mean": float(np.mean(subject_scores)), "std": float(np.std(subject_scores, ddof=1)), "n_folds": len(subject_scores)},
        "subject_blind": {"mean": float(np.mean(random_scores)), "std": float(np.std(random_scores, ddof=1)), "n_folds": len(random_scores)},
    }
    summary["leakage_gap"] = summary["subject_blind"]["mean"] - summary["subject_wise"]["mean"]

    print("\n=== summary ===")
    print(f"  subject-wise  : {summary['subject_wise']['mean']:.4f} +/- {summary['subject_wise']['std']:.4f} ({len(subject_scores)} folds)")
    print(f"  subject-blind : {summary['subject_blind']['mean']:.4f} +/- {summary['subject_blind']['std']:.4f} ({len(random_scores)} folds)")
    print(f"  gap           : {summary['leakage_gap']:+.4f}  <- what ignoring person_id buys")

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved {args.summary_output}")


if __name__ == "__main__":
    main()
