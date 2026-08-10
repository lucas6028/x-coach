"""Replay the recovered EgoExo-Fitness High Knee actions through the shipped detector.

The pose JSON is NOT in the repository: `frames_open` is a 3 GiB-split download whose `.ac` part
is missing, so only six judged High Knee actions are reachable. Reproduce with

    .venv\\Scripts\\python.exe scripts/egoexo/extract_action_frames.py ^
      --movement "High Knee" --out <frames-dir>
    .venv\\Scripts\\python.exe scripts/egoexo/run_pose_on_frame_dirs.py ^
      --frames-root <frames-dir> --out <pose-dir>          [--shard i --shards n]
    .venv\\Scripts\\python.exe scripts/egoexo/run_high_knee_validation.py ^
      --pose-dir <pose-dir> --json <out.json>

Every figure in `notes/high-knee-rule-validation.md` and in design spec sections 5-7 comes from
here.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.egoexo.high_knee_validation import (  # noqa: E402
    BACK_STRAIGHT_CRITERION, EXO_VIEWS, STABILITY_CRITERION, WITHDRAWN_PELVIC_DROP_CUT_DEG,
    cross_camera_spread, evaluate_view, floor_discarded, load_judgements, load_pose_frames,
    pearson,
)

# The two cameras whose `anterior_axis_length` says they can see a sagittal quantity at all. Which
# these are is DISCOVERED (the gate separates them with no overlap), not assumed from their names.
GATED_VIEWS = ("exo_l", "exo_r")


def _median(values):
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return statistics.median(finite) if finite else math.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose-dir", required=True, type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument(
        "--judgements",
        type=Path,
        default=REPO_ROOT / "data/EgoExo-Fitness/raw_annotations/interpretable_action_judgement.json",
    )
    args = ap.parse_args()

    judgements = load_judgements(args.judgements)
    actions: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(args.pose_dir.glob("*.json")):
        sample_id, view = path.stem.rsplit("__", 1)
        frames, fps = load_pose_frames(path)
        if not frames:
            continue
        actions[sample_id][view] = evaluate_view(frames, fps, view)

    all_views = [v for views in actions.values() for v in views.values()]
    gated = [v for views in actions.values() for name, v in views.items() if name in GATED_VIEWS]
    if not all_views:
        raise SystemExit(f"no pose JSON found under {args.pose_dir}")

    print(f"{'pair':34} {'val':>5} {'rep':>4} {'r0.4':>5} {'Hz':>5} {'peak':>7} {'cited%':>7} "
          f"{'impl%':>6} {'gate':>6} {'lean':>7} {'leanSD':>7} {'back%':>6} {'fwd%':>5} "
          f"{'axErr':>6} {'obl':>7}")
    for sample_id in sorted(actions):
        for view in EXO_VIEWS:
            row = actions[sample_id].get(view)
            if row is None:
                continue
            print(f"{sample_id + '__' + view:34} {row['validity']:5.2f} "
                  f"{row['reps_shipped_floor']:4d} {row['reps_default_floor']:5d} "
                  f"{row['cadence_hz']:5.2f} {row['peak_elevation_median']:7.3f} "
                  f"{row['fire_rate_cited_cut'] * 100:6.1f} "
                  f"{row['fire_rate_implemented_cut'] * 100:5.1f} "
                  f"{row['anterior_axis_median']:6.3f} {row['trunk_lean_median_deg']:7.2f} "
                  f"{row['trunk_lean_sd_deg']:7.2f} {row['back_lean_fire_rate'] * 100:5.1f} "
                  f"{row['forward_lean_fire_rate'] * 100:4.1f} "
                  f"{row['axis_error_median_deg']:6.2f} {row['obliquity_median_deg']:7.2f}")

    shipped = sum(v["reps_shipped_floor"] for v in all_views)
    default = sum(v["reps_default_floor"] for v in all_views)
    summary = {
        "actions": len(actions),
        "pairs": len(all_views),
        "frames": sum(v["frames"] for v in all_views),
        "median_validity": _median([v["validity"] for v in all_views]),
        "pairs_on_whole_clip_fallback": sum(1 for v in all_views if v["fallback"]),
        "reps_shipped_floor": shipped,
        "reps_default_floor": default,
        "reps_discarded_by_default_floor": floor_discarded(shipped, default),
        "median_cadence_hz": _median([v["cadence_hz"] for v in all_views]),
        "max_cadence_hz": max(v["cadence_hz"] for v in all_views if math.isfinite(v["cadence_hz"])),
        "gate_side_cameras": [
            min(v["anterior_axis_median"] for v in gated),
            max(v["anterior_axis_median"] for v in gated),
        ],
        "gate_front_camera": [
            min(v["anterior_axis_median"] for views in actions.values()
                for name, v in views.items() if name == "exo_m"),
            max(v["anterior_axis_median"] for views in actions.values()
                for name, v in views.items() if name == "exo_m"),
        ],
        "knee_lift_fire_cited_cut_gated": _median([v["fire_rate_cited_cut"] for v in gated]),
        "knee_lift_fire_implemented_cut_gated": _median(
            [v["fire_rate_implemented_cut"] for v in gated]
        ),
        "back_lean_fire_rate_gated": _median([v["back_lean_fire_rate"] for v in gated]),
        "forward_lean_fire_rate_gated": _median([v["forward_lean_fire_rate"] for v in gated]),
        "trunk_to_support_limb_deg_gated": [
            min(v["axis_error_median_deg"] for v in gated),
            max(v["axis_error_median_deg"] for v in gated),
            _median([v["axis_error_median_deg"] for v in gated]),
        ],
        "pelvic_drop_threshold_deg": WITHDRAWN_PELVIC_DROP_CUT_DEG,
    }

    print("\n=== pelvic obliquity: two SIMULTANEOUS gated cameras, same instant ===")
    spreads, correlations = [], []
    for sample_id in sorted(actions):
        spread = cross_camera_spread(actions[sample_id], "obliquity_median_deg", GATED_VIEWS)
        left = actions[sample_id].get("exo_l", {}).get("obliquity_series", [])
        right = actions[sample_id].get("exo_r", {}).get("obliquity_series", [])
        correlation = pearson(left, right) if left and right else math.nan
        spreads.append(spread)
        correlations.append(correlation)
        print(f"  {sample_id:24} spread {spread:6.2f} deg   frame-by-frame r {correlation:+.3f}")
    summary["obliquity_camera_spread_deg"] = sorted(s for s in spreads if math.isfinite(s))
    summary["obliquity_camera_correlation"] = sorted(c for c in correlations if math.isfinite(c))

    print("\n=== stability criterion vs the trunk quantity, gated cameras ===")
    groups = defaultdict(list)
    for sample_id, views in actions.items():
        label = "FALSE" if judgements.get(sample_id, {}).get(STABILITY_CRITERION) else "TRUE"
        for name, row in views.items():
            if name in GATED_VIEWS:
                groups[label].append(row)
    for label in ("FALSE", "TRUE"):
        rows = groups[label]
        if not rows:
            continue
        print(f"  judged {label:5} on stability: n={len(rows):2d} pairs   "
              f"median lean {_median([r['trunk_lean_median_deg'] for r in rows]):6.2f} deg   "
              f"median sd {_median([r['trunk_lean_sd_deg'] for r in rows]):5.2f} deg")
        summary[f"stability_{label.lower()}_lean_median_deg"] = _median(
            [r["trunk_lean_median_deg"] for r in rows]
        )
        summary[f"stability_{label.lower()}_lean_sd_deg"] = _median(
            [r["trunk_lean_sd_deg"] for r in rows]
        )
    summary["actions_failing_back_straight"] = sum(
        1 for sample_id in actions if judgements.get(sample_id, {}).get(BACK_STRAIGHT_CRITERION)
    )

    print("\n" + json.dumps(summary, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "per_action": {
                        sample_id: {
                            view: {k: v for k, v in row.items() if k != "obliquity_series"}
                            for view, row in views.items()
                        }
                        for sample_id, views in actions.items()
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
