"""Consolidated Squat experiment: does depth change classification, per fault type?

Runs the same shared-feature, same-classifier, same-bootstrap machinery as the shallow
experiment, but at video level and across all squat faults -- knees_forward,
knees_inward, and their union (combined) -- to test the paper's "route sensing by fault
type" thesis in the wild:

* knees_inward is a frontal-plane (mediolateral) cue -> image-plane readable, depth ~ redundant.
* shallow depth is a vertical cue -> image-plane readable (measured separately: redundant).
* knees_forward is anterior knee travel -> projects onto the depth axis in the rear /
  rear-oblique views that dominate this data, so it is the one fault where true 3D can help.

Per fault, the payload is the within-model depth delta (nlf_3d - nlf_2d, same rows) --
clean per fault and comparable across faults. Absolute balanced accuracy is NOT compared
against the frame-level shallow number (different granularity).

    .venv\\Scripts\\python.exe scripts/fitness_aqa/run_squat_depth_classification.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fit3d.biomech import IMAGE2D  # noqa: E402
from src.fitness_aqa import depth_classify as dc  # noqa: E402
from src.fitness_aqa import squat_dataset as sq  # noqa: E402
from src.fitness_aqa import video_features as vf  # noqa: E402
from src.fitness_aqa.cue_features import CAM3D  # noqa: E402

# (arm name, pose dir, which joints key, coordinate space)
ARM_SPECS = [
    ("mediapipe_2d", "video_pose", "joints_2d", IMAGE2D),
    ("mediapipe_3d", "video_pose", "joints_3d", CAM3D),
    ("nlf_2d", "video_pose_nlf", "joints_2d", IMAGE2D),
    ("nlf_3d", "video_pose_nlf", "joints_3d", CAM3D),
]
FAULTS = ("knees_forward", "knees_inward", "combined")
PAIRS = [("nlf_3d", "nlf_2d", "same detector, depth on vs off (primary)"),
         ("mediapipe_3d", "mediapipe_2d", "same detector, heuristic depth on vs off"),
         ("nlf_2d", "mediapipe_2d", "2D detector quality only")]


def load_arm_features(spec, root: Path, video_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    name, subdir, key, space = spec
    pose_dir = root / "derived" / subdir
    per_video = {}
    for vid in video_ids:
        f = pose_dir / f"{vid}.npz"
        if not f.exists():
            continue
        with np.load(f, allow_pickle=False) as d:
            per_video[vid] = (d[key], d["detected"])
    return vf.build_matrix(per_video, space, video_ids)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=sq.DEFAULT_SQUAT_ROOT)
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=sq.DEFAULT_SQUAT_ROOT / "derived" / "squat_depth_classification.json")
    args = p.parse_args()

    split_of = sq.split_of(args.root)
    labels_by_fault = sq.all_labels(args.root)
    video_ids = sorted(split_of)

    specs = ARM_SPECS if not args.arms else [s for s in ARM_SPECS if s[0] in set(args.arms)]
    arms = {}
    for spec in specs:
        feats, ok = load_arm_features(spec, args.root, video_ids)
        arms[spec[0]] = {"features": feats, "valid": ok, "space": spec[3]}
        print(f"{spec[0]:>14} space={spec[3]:<8} valid {ok.sum()}/{len(video_ids)} ({ok.mean():.3f})", flush=True)

    common = np.ones(len(video_ids), bool)
    for a in arms.values():
        common &= a["valid"]
    splits = np.array([split_of[v] for v in video_ids])
    print(f"\ncommon videos across {len(arms)} arms: {common.sum()}/{len(video_ids)} ({common.mean():.3f})",
          flush=True)

    report = {"n_common": int(common.sum()), "faults": {},
              "config": {"l2": args.l2, "n_boot": args.n_boot, "seed": args.seed,
                         "feature_dim": len(vf.FEATURE_NAMES)}}

    for fault in FAULTS:
        lab = labels_by_fault[fault]
        y_all = np.array([lab.get(v, -1) for v in video_ids])
        usable = common & (y_all >= 0)
        idx = {s: np.flatnonzero(usable & (splits == s)) for s in sq.SPLITS}
        y = {s: y_all[idx[s]].astype(float) for s in sq.SPLITS}
        groups = {s: np.array(video_ids)[idx[s]] for s in sq.SPLITS}  # each video is its own cluster
        reps = dc.cluster_bootstrap_indices(groups["test"], args.n_boot, args.seed)

        print(f"\n===== fault={fault} =====")
        for s in sq.SPLITS:
            print(f"  {s}: n={len(idx[s])} pos={int(y[s].sum())} neg={int((1-y[s]).sum())}")

        results, arm_report = {}, {}
        for name, arm in arms.items():
            x = {s: arm["features"][idx[s]] for s in sq.SPLITS}
            res = dc.run_arm(name, x, y, groups, l2=args.l2)
            boot = dc.bootstrap_metric(res, reps)
            res.metrics["ba_ci_low"], res.metrics["ba_ci_high"] = (
                float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
            results[name] = res
            arm_report[name] = {"metrics": res.metrics,
                                "top_coef": _top_coef(res.coef)}
        for name, res in sorted(results.items(), key=lambda kv: -kv[1].metrics["balanced_accuracy"]):
            m = res.metrics
            print(f"  {name:>14} BA {m['balanced_accuracy']:.3f} "
                  f"[{m['ba_ci_low']:.3f},{m['ba_ci_high']:.3f}] AUC {m['auc']:.3f} "
                  f"recall {m['recall']:.3f} spec {m['specificity']:.3f}")

        deltas, overlaps = {}, {}
        for a, b, why in PAIRS:
            if a in results and b in results:
                deltas[f"{a}-vs-{b}"] = {**dc.paired_delta(results[a], results[b], reps), "isolates": why}
        for a, b in (("nlf_3d", "nlf_2d"), ("mediapipe_3d", "mediapipe_2d")):
            if a in results and b in results:
                overlaps[f"{a}-vs-{b}"] = _overlap(results[a], results[b])
        print("  -- depth deltas --")
        for k, d in deltas.items():
            print(f"    {k}: {d['delta']:+.3f} CI[{d['ci_low']:+.3f},{d['ci_high']:+.3f}] p={d['p_two_sided']:.3f}")

        report["faults"][fault] = {
            "splits": {s: {"n": int(len(idx[s])), "pos": int(y[s].sum())} for s in sq.SPLITS},
            "arms": arm_report, "paired_deltas": deltas, "depth_redundancy": overlaps,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


def _top_coef(coef: np.ndarray, k: int = 8) -> dict:
    order = np.argsort(-np.abs(coef))[:k]
    return {vf.FEATURE_NAMES[i]: float(coef[i]) for i in order}


def _overlap(a, b) -> dict:
    yt = a.test_labels
    pa = (a.test_scores >= a.metrics["threshold"]).astype(int)
    pb = (b.test_scores >= b.metrics["threshold"]).astype(int)
    wa, wb = pa != yt, pb != yt
    return {"three_d_fixes": int((wb & ~wa).sum()), "three_d_breaks": int((~wb & wa).sum()),
            "pred_agreement": float(np.mean(pa == pb)),
            "score_pearson_r": float(np.corrcoef(a.test_scores, b.test_scores)[0, 1])}


if __name__ == "__main__":
    main()
