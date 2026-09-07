"""Compute the checkpoint-robustness primary statistic (see
`notes/videomae_ssv2_checkpoint_validation_plan.md`).

Kin-only mode (also gate G4 -- must pass before any SSv2 dir is read):

    python scripts/video/run_checkpoint_robustness_report.py \
        --kin-full-frame       data/Fitness-AQA/Squat/experiments/videomae_corrected/predictions \
        --kin-background-only  data/Fitness-AQA/Squat/experiments/videomae_background_only/predictions \
        --kin-person-crop      data/Fitness-AQA/Squat/experiments/videomae_person_crop/predictions \
        --output data/Fitness-AQA/Squat/experiments/checkpoint_robustness_kin_only.json

Full run, once the SSv2 arms exist:

    python scripts/video/run_checkpoint_robustness_report.py \
        --kin-full-frame  ... --kin-background-only ... --kin-person-crop ... \
        --ssv2-full-frame data/Fitness-AQA/Squat/experiments/videomae_ssv2_full_frame/predictions \
        --ssv2-background-only data/Fitness-AQA/Squat/experiments/videomae_ssv2_background_only/predictions \
        --ssv2-person-crop data/Fitness-AQA/Squat/experiments/videomae_ssv2_person_crop/predictions \
        --output data/Fitness-AQA/Squat/experiments/checkpoint_robustness_report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.checkpoint_robustness_report import (
    ARM_KINDS,
    READING_DETAIL,
    align_arms,
    arm_name,
    athlete_contribution,
    g4_gate,
    paired_multi_arm_bootstrap,
    primary_reading,
    retention,
    safe_ratio,
)
from src.video.stage_b_report import ArmRuns, format_summary_table, load_single_arm, write_report

#: The seed set fixed in the pre-registration; every published number uses it.
PRE_REGISTERED_SEEDS = [1, 2, 3, 4, 5]


def build_statistics(checkpoints: list[str], have_ssv2: bool) -> dict[str, Callable[[dict[str, float]], float]]:
    """One statistic callable per bootstrapped quantity, all sharing the same draws."""
    statistics: dict[str, Callable[[dict[str, float]], float]] = {}
    for checkpoint in checkpoints:
        full = arm_name(checkpoint, "full_frame")
        background = arm_name(checkpoint, "background_only")
        crop = arm_name(checkpoint, "person_crop")
        statistics[f"R_{checkpoint}"] = lambda vec, full=full, background=background: safe_ratio(
            vec[background] - 0.5, vec[full] - 0.5
        )
        statistics[f"person_crop_minus_full_frame_{checkpoint}"] = lambda vec, full=full, crop=crop: (
            vec[crop] - vec[full]
        )

    if have_ssv2:
        kin_full, kin_background = arm_name("kin", "full_frame"), arm_name("kin", "background_only")
        ssv2_full, ssv2_background = arm_name("ssv2", "full_frame"), arm_name("ssv2", "background_only")
        statistics["delta_A"] = lambda vec: (vec[ssv2_full] - vec[ssv2_background]) - (
            vec[kin_full] - vec[kin_background]
        )

    return statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the checkpoint-robustness primary statistic.")
    parser.add_argument("--kin-full-frame", type=Path, required=True)
    parser.add_argument("--kin-background-only", type=Path, required=True)
    parser.add_argument("--kin-person-crop", type=Path, required=True)
    parser.add_argument("--ssv2-full-frame", type=Path, default=None)
    parser.add_argument("--ssv2-background-only", type=Path, default=None)
    parser.add_argument("--ssv2-person-crop", type=Path, default=None)
    parser.add_argument("--label-mode", default="combined")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(PRE_REGISTERED_SEEDS))
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260907)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.seeds != PRE_REGISTERED_SEEDS:
        print(
            f"WARNING: seeds {args.seeds} are not the pre-registered {PRE_REGISTERED_SEEDS}. "
            "Treat this run as diagnostic -- gate G4 and the primary statistic are defined "
            "over the pre-registered set."
        )

    ssv2_dirs = {
        "full_frame": args.ssv2_full_frame,
        "background_only": args.ssv2_background_only,
        "person_crop": args.ssv2_person_crop,
    }
    have_ssv2 = any(value is not None for value in ssv2_dirs.values())
    if have_ssv2 and not all(value is not None for value in ssv2_dirs.values()):
        missing = [f"--ssv2-{kind.replace('_', '-')}" for kind, value in ssv2_dirs.items() if value is None]
        raise SystemExit(
            "SSv2 arms are partially specified; also pass "
            f"{', '.join(missing)} (or omit all three to run kin-only / gate G4)."
        )

    kin_dirs = {
        "full_frame": args.kin_full_frame,
        "background_only": args.kin_background_only,
        "person_crop": args.kin_person_crop,
    }

    arms: dict[str, ArmRuns] = {}
    for kind, directory in kin_dirs.items():
        name = arm_name("kin", kind)
        arms[name] = load_single_arm(name, directory, args.label_mode, args.seeds)
    if have_ssv2:
        for kind, directory in ssv2_dirs.items():
            name = arm_name("ssv2", kind)
            arms[name] = load_single_arm(name, directory, args.label_mode, args.seeds)

    gate = g4_gate(arms, checkpoint="kin")
    print("\n=== gate G4 (kin reproduction vs the plan's published numbers) ===")
    for kind in ARM_KINDS:
        print(
            f"  {kind:<16}: {gate['balanced_accuracy'][kind]:.4f}"
            f"  (target {gate['targets'][kind]:.4f}, delta {gate['deltas_vs_target'][kind]:+.4f})"
        )
    print(f"  R_kin           : {gate['R']:.4f}  (target {gate['R_target']:.4f}, delta {gate['R_delta_vs_target']:+.4f})")
    print(f"  => {'PASS' if gate['passed'] else 'FAIL -- everything below is against an unreproduced baseline'}")

    print("\n=== test metrics, mean +/- std over seeds " + str(args.seeds) + " ===")
    print(format_summary_table(list(arms.values())))

    checkpoints = ["kin", "ssv2"] if have_ssv2 else ["kin"]
    aligned = align_arms(arms)
    statistics = build_statistics(checkpoints, have_ssv2)
    bootstrap = paired_multi_arm_bootstrap(
        aligned, statistics, resamples=args.resamples, seed=args.bootstrap_seed
    )

    per_checkpoint: dict[str, dict[str, object]] = {}
    print("\n=== per checkpoint: athlete contribution A and retention R ===")
    for checkpoint in checkpoints:
        full = arms[arm_name(checkpoint, "full_frame")]
        background = arms[arm_name(checkpoint, "background_only")]
        crop = arms[arm_name(checkpoint, "person_crop")]
        person_crop_delta = crop.summary()["balanced_accuracy"]["mean"] - full.summary()["balanced_accuracy"]["mean"]
        r_ci = bootstrap[f"R_{checkpoint}"]
        secondary_ci = bootstrap[f"person_crop_minus_full_frame_{checkpoint}"]
        per_checkpoint[checkpoint] = {
            "A": athlete_contribution(full, background),
            "R": retention(full, background),
            "R_ci": r_ci,
            "person_crop_minus_full_frame": person_crop_delta,
            "person_crop_minus_full_frame_ci": secondary_ci,
        }
        print(
            f"  {checkpoint}: A={per_checkpoint[checkpoint]['A']:+.4f}"
            f"  R={per_checkpoint[checkpoint]['R']:.4f} [{r_ci['ci_low']:.4f}, {r_ci['ci_high']:.4f}]"
            f"  person_crop-full_frame={person_crop_delta:+.4f}"
            f" [{secondary_ci['ci_low']:+.4f}, {secondary_ci['ci_high']:+.4f}]"
        )

    payload: dict[str, object] = {
        "label_mode": args.label_mode,
        "seeds": args.seeds,
        "g4_gate": gate,
        "arms": {
            name: {"summary": arm.summary(), "per_seed": arm.metrics, "thresholds": arm.thresholds}
            for name, arm in arms.items()
        },
        "per_checkpoint": per_checkpoint,
    }

    if have_ssv2:
        delta = bootstrap["delta_A"]
        reading = primary_reading(delta["ci_low"], delta["ci_high"])
        payload["primary"] = {
            "delta_A": delta["observed"],
            "ci_low": delta["ci_low"],
            "ci_high": delta["ci_high"],
            "fraction_positive": delta["fraction_positive"],
            "resamples": delta["resamples"],
            "seed": args.bootstrap_seed,
            "reading": reading,
            "reading_detail": READING_DETAIL[reading],
        }
        print("\n=== primary: delta A = A_ssv2 - A_kin ===")
        print(
            f"  delta A = {delta['observed']:+.4f}"
            f"  95% CI [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]"
            f"  fraction > 0: {delta['fraction_positive']:.3f}"
        )
        print(f"  => {reading}")

    write_report(args.output, payload)
    print(f"\nSaved {args.output}")

    if not gate["passed"]:
        raise SystemExit(
            "Gate G4 FAILED: the re-derived Kinetics numbers do not match the plan's published "
            "0.6401 / 0.6271 / 0.6662 (R_kin 0.907) to tolerance. No SSv2 number may be read until "
            "this passes."
        )


if __name__ == "__main__":
    main()
