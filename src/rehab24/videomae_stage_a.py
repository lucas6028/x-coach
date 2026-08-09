"""Stage A evidence for the VideoMAE dataset-validation plan (REHAB24-6).

Produces, in one pass, the four things Stage A's pass conditions ask for:

1. Paired per-subject balanced-accuracy deltas between token-pooling modes.
2. The direction of that delta on each held-out subject (not just the mean).
3. A null control: the same pipeline on labels permuted *within subject*, which
   preserves each fold's positive rate so the null moves only if real signal is
   gone -- a globally shuffled label set would also change class balance and make
   the null easy to beat for the wrong reason.
4. Stratification by camera and by exercise, so an effect confined to one view or
   one movement cannot pass as a general one.

Every arm trains through the same ``train_one_fold`` as the committed LOSO
baselines, on identical folds and seeds, so deltas are properly paired.
"""

from __future__ import annotations

import argparse
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
    summarize,
    train_one_fold,
)
from src.video.videomae_video_classifier import build_samples, compute_metrics

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("REHAB24-6 Stage A evaluation requires `torch`.") from exc

#: A stratum smaller than this makes balanced accuracy meaningless (a single
#: negative sample turns specificity into 0 or 1).
MIN_STRATUM_SAMPLES = 20


def load_metadata(manifest_path: Path) -> dict[str, dict[str, str]]:
    return {
        row["sample_id"]: {
            "camera": row["camera"],
            "exercise": row["exercise_name"],
            "person_id": row["person_id"],
            # Both camera rows of one repetition share this key -- correctness is a
            # property of the repetition, not of the view.
            "repetition": f"{row['exercise_id']}_{row['video_id']}_rep{row['repetition_number']}",
        }
        for row in load_manifest(manifest_path)
    }


def permute_labels_within_subject(
    labels: dict[str, int],
    subject_samples: dict[str, list[str]],
    metadata: dict[str, dict[str, str]],
    seed: int,
) -> dict[str, int]:
    """Shuffle labels among the *repetitions* of each subject, keeping balance.

    Two things must hold for the null to be fair, and both drive this design.

    Per-subject balance is preserved because every LOSO fold tests exactly one
    subject, so holding that subject's positive rate fixed keeps the null's chance
    level identical to the real run's. Any remaining gap is attributable to signal.

    Permutation is at repetition level, not sample level, because REHAB24-6 records
    each repetition from two cameras (verified: 1072 reps, all with exactly two rows
    and never with disagreeing labels). Shuffling per sample would give cam17 and
    cam18 of the same repetition opposite labels -- a contradiction the real training
    set never contains, which makes the null artificially *harder* and inflates the
    real-minus-null gap that pass condition 3 rests on.
    """
    rng = np.random.default_rng(seed)
    permuted = dict(labels)
    for sample_ids in subject_samples.values():
        repetitions: dict[str, list[str]] = defaultdict(list)
        for sample_id in sample_ids:
            if sample_id in labels:
                repetitions[metadata.get(sample_id, {}).get("repetition", sample_id)].append(sample_id)

        keys = sorted(repetitions)
        values = [labels[repetitions[key][0]] for key in keys]
        rng.shuffle(values)
        for key, value in zip(keys, values):
            for sample_id in repetitions[key]:
                permuted[sample_id] = int(value)
    return permuted


def ordered_test_ids(feature_dir: Path, test_ids: list[str], labels: dict[str, int]) -> list[str]:
    """Sample ids in the order ``train_one_fold`` scores them.

    ``train_one_fold`` builds its test set with ``build_samples`` and predicts with
    ``shuffle=False``, so rebuilding the same list reproduces the probability order.
    Callers assert the length against the probability array so this ordering
    contract is checked rather than assumed.
    """
    return [sample.video_id for sample in build_samples(feature_dir, test_ids, labels)]


def stratified_metrics(
    sample_ids: list[str],
    probabilities: np.ndarray,
    fold_labels: np.ndarray,
    threshold: float,
    metadata: dict[str, dict[str, str]],
    key: str,
) -> dict[str, dict[str, float]]:
    """Balanced accuracy per stratum, using the fold's own selected threshold."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        stratum = metadata.get(sample_id, {}).get(key)
        if stratum is not None:
            buckets[stratum].append(index)

    results: dict[str, dict[str, float]] = {}
    for stratum, indices in buckets.items():
        selected = np.asarray(indices, dtype=int)
        stratum_labels = fold_labels[selected]
        if len(selected) < MIN_STRATUM_SAMPLES or len(np.unique(stratum_labels)) < 2:
            continue
        metrics = compute_metrics(probabilities[selected], stratum_labels, threshold=threshold)
        results[stratum] = {
            "n": int(len(selected)),
            "balanced_accuracy": metrics["balanced_accuracy"],
        }
    return results


def run_arm(
    feature_dir: Path,
    labels: dict[str, int],
    subject_samples: dict[str, list[str]],
    ordered_subjects: list[str],
    sample_counts: dict[str, int],
    metadata: dict[str, dict[str, str]],
    config: FoldConfig,
    device: torch.device,
    seed: int,
) -> list[dict]:
    """Full LOSO for one feature dir, retaining per-sample predictions for strata."""
    folds: list[dict] = []
    for test_subject in ordered_subjects:
        val_subject = pick_val_subject(test_subject, ordered_subjects, sample_counts)
        test_ids = subject_samples[test_subject]
        val_ids = subject_samples[val_subject]
        train_ids = [sid for s in ordered_subjects if s not in {test_subject, val_subject} for sid in subject_samples[s]]

        threshold, probabilities, fold_labels = train_one_fold(
            feature_dir, train_ids, val_ids, test_ids, labels, config, device, seed
        )
        sample_ids = ordered_test_ids(feature_dir, test_ids, labels)
        if len(sample_ids) != len(probabilities):
            raise SystemExit(
                f"Prediction/id misalignment for subject P{test_subject} in {feature_dir.name}: "
                f"{len(probabilities)} predictions vs {len(sample_ids)} ids. Run the feature audit."
            )

        metrics = compute_metrics(probabilities, fold_labels, threshold=threshold)
        folds.append(
            {
                "test_subject": test_subject,
                "val_subject": val_subject,
                "n_test": int(len(fold_labels)),
                "threshold": float(threshold),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "by_camera": stratified_metrics(sample_ids, probabilities, fold_labels, threshold, metadata, "camera"),
                "by_exercise": stratified_metrics(sample_ids, probabilities, fold_labels, threshold, metadata, "exercise"),
            }
        )
    return folds


def paired_delta(candidate: list[dict], baseline: list[dict], drop_p10: bool = True) -> dict:
    """Per-fold paired delta between two arms evaluated on identical folds."""
    rows = []
    for cand, base in zip(candidate, baseline):
        if cand["test_subject"] != base["test_subject"]:
            raise ValueError("Arms were not evaluated on the same fold order.")
        if drop_p10 and cand["n_test"] < MIN_VAL_SUBJECT_SAMPLES:
            continue
        rows.append(
            {
                "test_subject": cand["test_subject"],
                "baseline": base["balanced_accuracy"],
                "candidate": cand["balanced_accuracy"],
                "delta": cand["balanced_accuracy"] - base["balanced_accuracy"],
            }
        )

    deltas = [row["delta"] for row in rows]
    positive = sum(delta > 0 for delta in deltas)
    result = {
        "folds": rows,
        "delta": summarize(deltas),
        "n_folds": len(rows),
        "n_positive": positive,
        "majority_positive": positive > len(rows) / 2,
    }
    if any(delta != 0 for delta in deltas):
        try:
            from scipy.stats import wilcoxon

            stat, p_value = wilcoxon(deltas)
            result["wilcoxon"] = {"stat": float(stat), "p_value": float(p_value), "significant": bool(p_value < 0.05)}
        except ImportError:
            result["wilcoxon"] = None
    return result


def stratum_deltas(candidate: list[dict], baseline: list[dict], key: str) -> dict[str, dict]:
    """Paired delta within each stratum, to test whether an effect is view- or
    exercise-specific rather than general."""
    per_stratum: dict[str, list[float]] = defaultdict(list)
    for cand, base in zip(candidate, baseline):
        if cand["test_subject"] != base["test_subject"]:
            raise ValueError("Arms were not evaluated on the same fold order.")
        for stratum, cand_metrics in cand[key].items():
            if stratum in base[key]:
                per_stratum[stratum].append(cand_metrics["balanced_accuracy"] - base[key][stratum]["balanced_accuracy"])

    return {
        stratum: {
            "n_folds": len(deltas),
            "mean_delta": float(np.mean(deltas)),
            "n_positive": int(sum(d > 0 for d in deltas)),
        }
        for stratum, deltas in sorted(per_stratum.items())
    }


def arm_summary(folds: list[dict]) -> dict:
    big = [f for f in folds if f["n_test"] >= MIN_VAL_SUBJECT_SAMPLES]
    return {
        "balanced_accuracy_all": summarize([f["balanced_accuracy"] for f in folds]),
        "balanced_accuracy_no_p10": summarize([f["balanced_accuracy"] for f in big]),
        "macro_f1_no_p10": summarize([f["macro_f1"] for f in big]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage A evidence for corrected-pooling VideoMAE on REHAB24-6.")
    parser.add_argument("--feature-parent", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[
            "videomae_legacy_first_token_max",
            "videomae_legacy_first_token_mean",
            "videomae_mean_pool_fc_norm_max",
            "videomae_mean_pool_fc_norm_mean",
        ],
        help="Feature dir names under --feature-parent, one per arm.",
    )
    parser.add_argument(
        "--baseline-arm",
        default="videomae_legacy_first_token_max",
        help="The historical extraction, reproduced. Paired deltas are measured against this.",
    )
    parser.add_argument(
        "--null-arm",
        default="videomae_mean_pool_fc_norm_mean",
        help="Arm to run the within-subject permuted-label null control on.",
    )
    parser.add_argument("--null-seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--labels", type=Path, default=DEFAULT_PROCESSED_ROOT / "labels" / "correctness.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or auto.")
    parser.add_argument("--output", type=Path, default=DEFAULT_PROCESSED_ROOT / "videomae_stage_a.json")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    config = FoldConfig()  # identical hyperparameters to the committed LOSO baselines

    labels = {key: int(value) for key, value in json.load(args.labels.open()).items()}
    metadata = load_metadata(args.manifest)
    subject_samples = subjects_to_samples(args.manifest)
    sample_counts = {person: len(ids) for person, ids in subject_samples.items()}
    ordered_subjects = sorted(subject_samples, key=int)

    print(f"Stage A on {device} | {len(args.arms)} arms | seed={args.seed}")

    arms: dict[str, list[dict]] = {}
    for arm in args.arms:
        feature_dir = args.feature_parent / arm
        if not feature_dir.exists():
            raise SystemExit(f"Missing feature dir {feature_dir}. Run videomae_materialize first.")
        print(f"\n--- arm: {arm} ---")
        folds = run_arm(
            feature_dir, labels, subject_samples, ordered_subjects, sample_counts, metadata, config, device, args.seed
        )
        arms[arm] = folds
        summary = arm_summary(folds)
        print(
            f"  bal_acc (9 folds, no P10): {summary['balanced_accuracy_no_p10']['mean']:.3f} "
            f"+/- {summary['balanced_accuracy_no_p10']['std']:.3f}"
        )

    # --- null control ----------------------------------------------------
    null_runs = []
    if args.null_arm in arms:
        for null_seed in args.null_seeds:
            permuted = permute_labels_within_subject(labels, subject_samples, metadata, null_seed)
            print(f"\n--- null control: {args.null_arm}, within-subject permutation seed {null_seed} ---")
            folds = run_arm(
                args.feature_parent / args.null_arm,
                permuted,
                subject_samples,
                ordered_subjects,
                sample_counts,
                metadata,
                config,
                device,
                args.seed,
            )
            summary = arm_summary(folds)
            print(f"  null bal_acc (9 folds): {summary['balanced_accuracy_no_p10']['mean']:.3f}")
            null_runs.append({"permutation_seed": null_seed, "folds": folds, "summary": summary})

    # --- paired comparisons ----------------------------------------------
    baseline_folds = arms[args.baseline_arm]
    comparisons = {}
    for arm, folds in arms.items():
        if arm == args.baseline_arm:
            continue
        comparisons[arm] = {
            "overall": paired_delta(folds, baseline_folds),
            "by_camera": stratum_deltas(folds, baseline_folds, "by_camera"),
            "by_exercise": stratum_deltas(folds, baseline_folds, "by_exercise"),
        }

    print(f"\n=== paired deltas vs {args.baseline_arm} (9 folds, no P10) ===")
    for arm, comparison in comparisons.items():
        overall = comparison["overall"]
        wilcoxon = overall.get("wilcoxon")
        p_text = f"  p={wilcoxon['p_value']:.3f}" if wilcoxon else ""
        print(
            f"  {arm:<40} d={overall['delta']['mean']:+.3f} +/- {overall['delta']['std']:.3f}"
            f"  ({overall['n_positive']}/{overall['n_folds']} folds positive){p_text}"
        )

    if null_runs:
        null_means = [run["summary"]["balanced_accuracy_no_p10"]["mean"] for run in null_runs]
        real_mean = arm_summary(arms[args.null_arm])["balanced_accuracy_no_p10"]["mean"]
        print(f"\n=== null control ({args.null_arm}) ===")
        print(f"  real labels     : {real_mean:.3f}")
        print(f"  permuted labels : {np.mean(null_means):.3f}  (seeds {args.null_seeds}: {[round(m, 3) for m in null_means]})")
        print(f"  gap             : {real_mean - float(np.mean(null_means)):+.3f}")

    payload = {
        "seed": args.seed,
        "config": vars(config),
        "baseline_arm": args.baseline_arm,
        "arms": {arm: {"folds": folds, "summary": arm_summary(folds)} for arm, folds in arms.items()},
        "comparisons": comparisons,
        "null_control": {"arm": args.null_arm, "runs": null_runs},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"\nSaved Stage A report to {args.output}")


if __name__ == "__main__":
    main()
