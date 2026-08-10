"""Replay EgoExo-Fitness's judged Jumping Jacks actions through the shipped detector.

    .venv\\Scripts\\python.exe scripts/egoexo/run_jumping_jacks_validation.py \\
        --pose-dir <dir of {sample_id}__{view}.json> [--json out.json]

The pose JSON is NOT in the repository: `frames_open` is a 3 GiB-split download whose `.ac` part
is missing, so only the records inside the `.aa`+`.ab` gzip prefix are recoverable. The recipe
for producing the input -- streaming the prefix through `tarfile` in `r|gz` mode and running
MediaPipe over the manifest's frame ranges -- is recorded in
`notes/jumping-jacks-rule-validation.md`.

Thin entry point per this repository's convention: all logic lives in
`src/egoexo/jumping_jacks_validation.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.egoexo.jumping_jacks_validation import evaluate_dataset, summarize  # noqa: E402

DEFAULT_LABELS = REPO_ROOT / "data" / "EgoExo-Fitness" / "processed" / "labels" / "tkv.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", required=True, type=Path)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--view-type",
        default="unknown",
        help="what to hand the rules as `ctx.view_type`. Default `unknown` because the view "
             "estimator was measured systematically inverted on an upright subject (Leg "
             "Abduction section 1.3); pass `front` to see what the discount would be worth.",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = evaluate_dataset(args.pose_dir, args.labels, view_type=args.view_type)
    summary = summarize(payload)

    print(f"actions                          {summary['actions']}")
    print(f"action x camera pairs            {summary['action_camera_pairs']}")
    print(f"judged CORRECT on foot split     {summary['actions_judged_correct_on_foot_split']}")
    print(f"median validity rate             {summary['median_validity_rate']:.3f}")
    print(f"pairs on the whole-clip fallback {summary['pairs_on_fallback']}")
    print(f"median cadence                   {summary['median_cadence_hz']:.2f} Hz")
    print(f"fastest cadence                  {summary['max_cadence_hz']:.2f} Hz "
          f"({summary['min_seconds_per_rep']:.2f} s/rep)")
    print(f"reps found / lost to the floor   {summary['reps_found_total']} / "
          f"{summary['reps_lost_to_the_floor']}")
    print(f"detections actually emitted      {summary['detections_emitted']} "
          f"(every rule is silent or withdrawn)")
    print("--- what the PARENT SPEC's cuts would have said on the same windows ---")
    print(f"jj_incomplete_leg_rom pair rate   {summary['leg_rom_fire_rate']:.3f}")
    print(f"jj_knee_valgus_landing pair rate  {summary['valgus_fire_rate']:.3f}")
    print(f"scored repetitions               {summary['scored_reps']}")
    print(f"  per-REP leg-rom fire rate      {summary['leg_rom_rep_fire_rate']:.3f}"
          f"   median widest stance {summary['median_per_rep_widest']:.3f}")
    print(f"  per-REP valgus  fire rate      {summary['valgus_rep_fire_rate']:.3f}"
          f"   median tightest knee/ankle {summary['median_per_rep_tightest']:.3f}")
    print(f"WITHDRAWN valgus cut, open frames {summary['open_frames']}")
    print(f"  observed knees below 0.82      {summary['valgus_observed_frame_rate']:.3f}"
          f"   median {summary['median_open_observed']:.3f}")
    print(f"  PERFECTLY ALIGNED knees below  {summary['valgus_aligned_frame_rate']:.3f}"
          f"   median {summary['median_open_aligned']:.3f}   <- the confound")
    print(f"  DECOMPOSITION: stance alone {summary['valgus_both_frame_rate']:.3f}"
          f"  +  needed real deviation {summary['valgus_observed_only_frame_rate']:.3f}"
          f"  (aligned-only {summary['valgus_aligned_only_frame_rate']:.3f})")
    print(f"cross-camera agreement, ROM      {summary['agreement_leg_rom']}")
    print(f"cross-camera agreement, valgus   {summary['agreement_valgus']}")
    print(f"median cross-camera stance spread {summary['median_stance_spread']:.3f}")
    print(f"median cross-camera valgus spread {summary['median_valgus_spread']:.3f}")

    for action in payload["actions"]:
        judged = {0: "correct", 1: "FAULT", None: "?"}[action["foot_split_fault"]]
        views = " ".join(
            f"{view}:reps={record['reps_found']},val={record['validity_rate']:.2f},"
            f"stance={record['max_stance_width_ratio']:.2f},"
            f"knee={record['min_knee_ankle_ratio']:.2f},"
            f"fired={'+'.join(record['fired']) or '-'}"
            for view, record in sorted(action["views"].items())
        )
        print(f"  {action['sample_id']:20s} foot_split={judged:7s} {views}")

    if args.json:
        args.json.write_text(
            json.dumps({"summary": summary, "detail": payload}, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
