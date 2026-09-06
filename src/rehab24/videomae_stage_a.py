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
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

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


def materialize_transformed_features(
    source_dir: Path,
    sample_ids: Sequence[str],
    transform: Callable[[np.ndarray], np.ndarray],
    destination: Path,
) -> int:
    """Write ``transform``-ed copies of ``sample_ids``' features into ``destination``.

    Materialising to disk instead of patching the loader is what keeps
    ``train_one_fold`` untouched: it re-reads ``video_feature`` from an npz either way,
    so a fold trained on a transformed directory travels the identical code path as one
    trained on the originals.

    Only the 0-d metadata entries are copied. The per-clip arrays are dropped because
    the classifier never reads them and copying them would multiply the disk cost of a
    per-fold materialisation by an order of magnitude.

    A transform may expose a ``digest`` string; when it does, the sidecar written here
    lets a later call with the same transform and the same ids skip the rewrite, which
    is what makes three seeds of one arm cost one materialisation instead of three. A
    transform without a digest is always rewritten -- silently reusing a directory
    whose provenance cannot be checked is the failure this guards against.
    """
    from src.video.videomae_video_classifier import clear_feature_cache, feature_path_index

    ordered = list(sample_ids)
    digest = getattr(transform, "digest", None)
    signature = {
        "n_samples": len(ordered),
        "sample_id_digest": hashlib.sha256("\n".join(sorted(ordered)).encode("utf-8")).hexdigest(),
        "transform_digest": digest,
    }
    sidecar = destination / "_materialization.json"
    if digest is not None and sidecar.exists():
        try:
            if json.load(sidecar.open(encoding="utf-8")) == signature:
                if len(list(destination.glob("*.npz"))) == len(ordered):
                    return len(ordered)
        except json.JSONDecodeError:  # pragma: no cover - a truncated sidecar just rewrites
            pass

    index = feature_path_index(source_dir)
    missing = [sample_id for sample_id in ordered if sample_id not in index]
    if missing:
        raise SystemExit(
            f"{len(missing)} feature files missing from {source_dir} (first: {missing[:5]}); "
            "cannot materialise a transformed fold directory."
        )

    destination.mkdir(parents=True, exist_ok=True)
    for path in destination.glob("*.npz"):
        path.unlink()
    for sample_id in ordered:
        with np.load(index[sample_id], allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files if data[key].ndim == 0}
            payload["video_feature"] = np.asarray(transform(data["video_feature"]), dtype=np.float32)
        np.savez_compressed(destination / f"{sample_id}.npz", **payload)

    written = len(list(destination.glob("*.npz")))
    if written != len(ordered):
        raise SystemExit(
            f"Materialised {written} feature files into {destination} but expected {len(ordered)}. "
            "build_samples skips missing ids silently, so a short directory would train on fewer samples."
        )
    with sidecar.open("w", encoding="utf-8") as handle:
        json.dump(signature, handle, indent=2, sort_keys=True)
    # Paths were just rewritten in place, so any memoised feature or dir index is stale.
    clear_feature_cache()
    return written


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
    retain_predictions: bool = False,
    feature_transform_factory: Callable[[list[str]], Callable[[np.ndarray], np.ndarray]] | None = None,
    materialize_root: Path | None = None,
) -> list[dict]:
    """Full LOSO for one feature dir, retaining per-sample predictions for strata.

    ``retain_predictions`` additionally stores each fold's ``sample_ids`` and
    ``probabilities`` on the fold record. It defaults to False so the committed
    Stage A / framing artifacts keep their existing schema byte-for-byte; the
    identity control turns it on to save out-of-fold probabilities WITHOUT
    reimplementing the fold loop, which is the only way its OOF can be shown to
    reproduce the framing folds exactly.

    ``feature_transform_factory`` is the position control's hook. It is called once per
    fold with *that fold's training ids* and must return a callable applied to every
    sample's feature vector; the fold then trains on a materialised copy under
    ``materialize_root``. Calling it per fold from here is what makes a train-only fit
    structural rather than a convention: the factory is never handed the test ids.
    Left at ``None`` (the default) not one byte of the original path changes.
    """
    if feature_transform_factory is not None and materialize_root is None:
        raise SystemExit("feature_transform_factory needs materialize_root: the caller decides where fold dirs live.")

    folds: list[dict] = []
    for test_subject in ordered_subjects:
        val_subject = pick_val_subject(test_subject, ordered_subjects, sample_counts)
        test_ids = subject_samples[test_subject]
        val_ids = subject_samples[val_subject]
        train_ids = [sid for s in ordered_subjects if s not in {test_subject, val_subject} for sid in subject_samples[s]]

        fold_dir = feature_dir
        if feature_transform_factory is not None:
            fold_dir = materialize_root / f"fold_P{test_subject}"
            materialize_transformed_features(
                feature_dir,
                [*train_ids, *val_ids, *test_ids],
                feature_transform_factory(list(train_ids)),
                fold_dir,
            )

        threshold, probabilities, fold_labels = train_one_fold(
            fold_dir, train_ids, val_ids, test_ids, labels, config, device, seed
        )
        sample_ids = ordered_test_ids(fold_dir, test_ids, labels)
        if len(sample_ids) != len(probabilities):
            raise SystemExit(
                f"Prediction/id misalignment for subject P{test_subject} in {fold_dir.name}: "
                f"{len(probabilities)} predictions vs {len(sample_ids)} ids. Run the feature audit."
            )

        metrics = compute_metrics(probabilities, fold_labels, threshold=threshold)
        predictions = (
            {"sample_ids": list(sample_ids), "probabilities": [float(p) for p in probabilities]}
            if retain_predictions
            else {}
        )
        folds.append(
            {
                **predictions,
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
