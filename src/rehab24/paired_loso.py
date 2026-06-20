"""Paired Leave-One-Subject-Out comparison of two REHAB24-6 feature backbones.

Ports the ``rehab24_hrnet_colab`` notebook (cells 9-11): run LOSO once per fold and
train *both* feature sets on the identical train/val/test split and seed, so the
per-fold deltas are properly paired. Pairing cancels each subject's intrinsic
difficulty -- the dominant source of fold-to-fold variance -- which a single mean
LOSO number per backbone cannot.

Default comparison is HRNet-2D (candidate) vs RTMPose-2D (baseline): the honest
control that isolates *backbone accuracy* from the 2D-vs-pseudo-3D depth factor
(both stay 2D-only). Expected effect is small (~+0.01-0.03) and may sit inside the
per-fold std (~+/-0.05), so we report per-fold paired deltas + Wilcoxon, not a
single number. Reuses the exact primitives of ``loso_cross_validation`` so each
per-fold model matches the committed single-dir baselines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.rehab24.dataset import DEFAULT_PROCESSED_ROOT
from src.rehab24.loso_cross_validation import (
    FoldConfig,
    MIN_VAL_SUBJECT_SAMPLES,
    pick_val_subject,
    subjects_to_samples,
    summarize,
    train_one_fold,
)
from src.video.videomae_video_classifier import compute_metrics

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("REHAB24-6 paired LOSO requires `torch`.") from exc


def run_paired_loso(
    baseline_dir: Path,
    candidate_dir: Path,
    manifest_path: Path,
    labels_path: Path,
    config: FoldConfig,
    device: torch.device,
    seed: int,
    baseline_name: str = "rtmpose",
    candidate_name: str = "hrnet",
) -> list[dict]:
    """Run LOSO once per fold, scoring both feature dirs on the same split/seed."""
    labels = {key: int(value) for key, value in json.load(labels_path.open()).items()}
    subject_samples = subjects_to_samples(manifest_path)
    sample_counts = {person: len(ids) for person, ids in subject_samples.items()}
    ordered = sorted(subject_samples, key=int)
    feature_dirs = {baseline_name: baseline_dir, candidate_name: candidate_dir}

    print(f"Paired LOSO on {device} | {candidate_name}={candidate_dir.name} vs {baseline_name}={baseline_dir.name}")
    folds: list[dict] = []
    for test_subject in ordered:
        val_subject = pick_val_subject(test_subject, ordered, sample_counts)
        test_ids = subject_samples[test_subject]
        val_ids = subject_samples[val_subject]
        train_ids = [sid for s in ordered if s not in {test_subject, val_subject} for sid in subject_samples[s]]

        record = {"test": test_subject, "val": val_subject, "n_test": len(test_ids)}
        for name, fdir in feature_dirs.items():
            threshold, prob, lab = train_one_fold(fdir, train_ids, val_ids, test_ids, labels, config, device, seed)
            record[name] = compute_metrics(prob, lab, threshold=threshold)["balanced_accuracy"]
        record["delta"] = record[candidate_name] - record[baseline_name]
        folds.append(record)
        print(
            f"P{test_subject:<3} (val P{val_subject}, n={record['n_test']:>4})  "
            f"{baseline_name}={record[baseline_name]:.3f}  "
            f"{candidate_name}={record[candidate_name]:.3f}  d={record['delta']:+.3f}"
        )
    return folds


def report(rows: list[dict], title: str, baseline_name: str, candidate_name: str) -> dict:
    """Print + return mean/std per backbone, paired delta, and Wilcoxon over folds."""
    base = summarize([r[baseline_name] for r in rows])
    cand = summarize([r[candidate_name] for r in rows])
    delta = summarize([r["delta"] for r in rows])
    positive = sum(r["delta"] > 0 for r in rows)
    print(f"\n=== {title} ({len(rows)} folds) ===")
    print(f"  {baseline_name:<8} bal_acc : {base['mean']:.3f} +/- {base['std']:.3f}  (min {base['min']:.3f}, max {base['max']:.3f})")
    print(f"  {candidate_name:<8} bal_acc : {cand['mean']:.3f} +/- {cand['std']:.3f}  (min {cand['min']:.3f}, max {cand['max']:.3f})")
    print(f"  paired d ({candidate_name}-{baseline_name}): {delta['mean']:+.3f} +/- {delta['std']:.3f}  ({positive}/{len(rows)} folds positive)")

    wilcoxon_result = None
    deltas = [r["delta"] for r in rows]
    if any(d != 0 for d in deltas):
        try:
            from scipy.stats import wilcoxon

            stat, p_value = wilcoxon(deltas)
            significant = bool(p_value < 0.05)
            verdict = "significant" if significant else f"UNDETERMINED at n={len(rows)}"
            print(f"  Wilcoxon signed-rank: stat={stat:.1f}  p={p_value:.3f}  ({verdict})")
            wilcoxon_result = {"stat": float(stat), "p_value": float(p_value), "significant": significant}
        except ImportError:
            print("  (scipy not available; install for Wilcoxon)")
    else:
        print("  Wilcoxon: all deltas zero -- no difference.")

    return {baseline_name: base, candidate_name: cand, "delta": delta, "wilcoxon": wilcoxon_result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired LOSO comparison of two REHAB24-6 feature backbones.")
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "rtmpose_skeleton_features")
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "hrnet_w48_skeleton_features")
    parser.add_argument("--baseline-name", type=str, default="rtmpose")
    parser.add_argument("--candidate-name", type=str, default="hrnet")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--labels", type=Path, default=DEFAULT_PROCESSED_ROOT / "labels" / "correctness.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or auto.")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT / "correctness_loso_hrnet_vs_rtmpose.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    config = FoldConfig()  # same hyperparameters as the committed LOSO baselines

    folds = run_paired_loso(
        baseline_dir=args.baseline_dir,
        candidate_dir=args.candidate_dir,
        manifest_path=args.manifest,
        labels_path=args.labels,
        config=config,
        device=device,
        seed=args.seed,
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
    )

    big = [r for r in folds if r["n_test"] >= MIN_VAL_SUBJECT_SAMPLES]  # drop the under-powered P10 (n=16)
    no_p5 = [r for r in big if r["test"] != "5"]  # P5 is a data ceiling (near-random for every feature)
    summary = {
        "all_10_folds": report(folds, "all 10 folds", args.baseline_name, args.candidate_name),
        "no_p10_9_folds": report(big, "9 folds (no P10)", args.baseline_name, args.candidate_name),
        "no_p10_no_p5": report(no_p5, "9 folds minus P5 (data-ceiling subject)", args.baseline_name, args.candidate_name),
    }

    payload = {
        "seed": args.seed,
        "baseline_dir": str(args.baseline_dir),
        "candidate_dir": str(args.candidate_dir),
        "config": vars(config),
        "folds": folds,
        "summary": summary,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"\nSaved paired LOSO summary to {args.summary_output}")


if __name__ == "__main__":
    main()
