"""Build the Stage B evidence table and evaluate the pre-registered decision rules.

Example (after every arm has been trained over seeds 1-5):

    python scripts/video/run_stage_b_report.py \
        --pose-predictions   data/Fitness-AQA/Squat/experiments/pose_only/predictions \
        --videomae-predictions data/Fitness-AQA/Squat/experiments/videomae_corrected/predictions \
        --arm early_fusion=data/Fitness-AQA/Squat/experiments/early_fusion/predictions \
        --arm videomae_legacy=data/Fitness-AQA/Squat/experiments/videomae_legacy/predictions \
        --output data/Fitness-AQA/Squat/experiments/stage_b_report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.late_fusion import paired_bootstrap_delta
from src.video.stage_b_report import (
    denominator_gate,
    format_summary_table,
    load_late_fusion_arm,
    load_single_arm,
    retention_conditions,
    write_report,
)


def parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--arm expects name=path, got {value!r}")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the Stage B evidence table.")
    parser.add_argument("--pose-predictions", type=Path, required=True, help="Normalized pose-only predictions dir.")
    parser.add_argument("--videomae-predictions", type=Path, required=True, help="Corrected VideoMAE predictions dir.")
    parser.add_argument("--arm", type=parse_arm, action="append", default=[], help="Extra arm as name=predictions_dir.")
    parser.add_argument("--label-mode", default="combined")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--fusion-weight", type=float, default=0.5, help="Pose branch's share; 0.5 is pre-registered.")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pose = load_single_arm("pose_only_normalized", args.pose_predictions, args.label_mode, args.seeds)
    videomae = load_single_arm("videomae_corrected", args.videomae_predictions, args.label_mode, args.seeds)
    late = load_late_fusion_arm(
        "late_fusion_calibrated",
        args.pose_predictions,
        args.videomae_predictions,
        args.label_mode,
        args.seeds,
        weight=args.fusion_weight,
    )
    extras = [load_single_arm(name, path, args.label_mode, args.seeds) for name, path in args.arm]
    arms = [pose, videomae, late, *extras]

    gate = denominator_gate(pose)
    print("\n=== denominator gate (pose-only vs the published 0.635 +/- 0.010) ===")
    print(f"  re-derived : {gate['re_derived']:.4f}   delta {gate['delta_vs_published']:+.4f}")
    print(f"  => {'PASS' if gate['passed'] else 'FAIL -- report both numbers, do not absorb this into any delta'}")

    print("\n=== test metrics, mean +/- std over seeds " + str(args.seeds) + " ===")
    print(format_summary_table(arms))

    comparisons = {}
    for candidate in [late, videomae, *extras]:
        bootstrap = paired_bootstrap_delta(
            baseline_probabilities=pose.probabilities,
            baseline_thresholds=pose.thresholds,
            candidate_probabilities=candidate.probabilities,
            candidate_thresholds=candidate.thresholds,
            labels=pose.labels,
            resamples=args.resamples,
        )
        conditions = retention_conditions(pose, candidate, bootstrap)
        comparisons[candidate.name] = {"bootstrap": bootstrap, "conditions": conditions}

        primary = " (PRIMARY)" if candidate is late else ""
        print(f"\n=== {candidate.name} vs pose_only_normalized{primary} ===")
        print(
            f"  delta balanced accuracy : {bootstrap['observed_delta']:+.4f}"
            f"  95% CI [{bootstrap['ci_low']:+.4f}, {bootstrap['ci_high']:+.4f}]"
        )
        for name, condition in conditions.items():
            mark = "PASS" if condition["passed"] else "FAIL"
            print(f"  [{mark}] {name}: {condition['value']}")

    payload = {
        "label_mode": args.label_mode,
        "seeds": args.seeds,
        "fusion_weight": args.fusion_weight,
        "denominator_gate": gate,
        "arms": {arm.name: {"summary": arm.summary(), "per_seed": arm.metrics, "thresholds": arm.thresholds} for arm in arms},
        "comparisons": comparisons,
        "n_test_videos": int(pose.labels.size),
    }
    write_report(args.output, payload)
    print(f"\nSaved {args.output}")

    if not gate["passed"]:
        raise SystemExit(
            "Denominator gate FAILED: the re-derived pose-only baseline is outside 0.635 +/- 0.010. "
            "Every delta above is against a moved denominator and must be reported as such."
        )


if __name__ == "__main__":
    main()
