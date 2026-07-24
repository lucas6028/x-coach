"""Video-level depth-vs-detector classification for squat OR ohp, any fault.

Generalises ``run_squat_depth_classification`` across movements: pick the dataset
(``--dataset squat|ohp``), it selects the matching label loader, cue module, derived pose
dirs, and fault list, then runs the same shared-feature / same-classifier / video-cluster
bootstrap machinery. Per fault the payload is the within-model depth delta (nlf_3d -
nlf_2d), reported AUC-first (threshold-free) with balanced accuracy secondary.

    .venv\\Scripts\\python.exe scripts/fitness_aqa/run_video_depth_classification.py --dataset ohp
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
from src.fitness_aqa import video_features as vf  # noqa: E402
from src.fitness_aqa.cue_features import CAM3D  # noqa: E402

PAIRS = [("nlf_3d", "nlf_2d", "same detector, depth on vs off (primary)"),
         ("mediapipe_3d", "mediapipe_2d", "same detector, heuristic depth on vs off"),
         ("nlf_2d", "mediapipe_2d", "2D detector quality only")]


def dataset_config(name: str):
    if name == "squat":
        from src.fitness_aqa import cue_features as cue
        from src.fitness_aqa import squat_dataset as ds
        return ds, cue, ds.DEFAULT_SQUAT_ROOT, list(ds.FAULTS) + ["combined"]
    if name == "ohp":
        from src.fitness_aqa import movement_cues as cue
        from src.fitness_aqa import ohp_dataset as ds
        return ds, cue, ds.DEFAULT_OHP_ROOT, list(ds.FAULTS) + ["combined"]
    raise SystemExit(f"unknown dataset {name!r}")


def load_arm(subdir: str, key: str, space: str, cue, root: Path, video_ids: list[str]):
    pose_dir = root / "derived" / subdir
    per_video = {}
    for vid in video_ids:
        f = pose_dir / f"{vid}.npz"
        if not f.exists():
            continue
        with np.load(f, allow_pickle=False) as d:
            per_video[vid] = (d[key], d["detected"])
    return vf.build_matrix(per_video, space, video_ids, cue_module=cue)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["squat", "ohp"])
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    ds, cue, root, faults = dataset_config(args.dataset)
    out = args.out or root / "derived" / f"{args.dataset}_depth_classification.json"

    split_of = ds.split_of(root)
    labels_by_fault = ds.all_labels(root)
    video_ids = sorted(split_of)
    splits = np.array([split_of[v] for v in video_ids])

    arm_specs = [("mediapipe_2d", "video_pose", "joints_2d", IMAGE2D),
                 ("mediapipe_3d", "video_pose", "joints_3d", CAM3D),
                 ("nlf_2d", "video_pose_nlf", "joints_2d", IMAGE2D),
                 ("nlf_3d", "video_pose_nlf", "joints_3d", CAM3D)]
    arms = {}
    for name, subdir, key, space in arm_specs:
        feats, ok = load_arm(subdir, key, space, cue, root, video_ids)
        arms[name] = {"features": feats, "valid": ok, "space": space}
        print(f"{name:>14} valid {ok.sum()}/{len(video_ids)} ({ok.mean():.3f})", flush=True)

    common = np.ones(len(video_ids), bool)
    for a in arms.values():
        common &= a["valid"]
    print(f"\ncommon videos: {common.sum()}/{len(video_ids)} ({common.mean():.3f})", flush=True)

    names = vf.feature_names(cue)
    report = {"dataset": args.dataset, "n_common": int(common.sum()), "faults": {},
              "config": {"l2": args.l2, "n_boot": args.n_boot, "feature_dim": len(names)}}

    for fault in faults:
        lab = labels_by_fault[fault]
        y_all = np.array([lab.get(v, -1) for v in video_ids])
        usable = common & (y_all >= 0)
        idx = {s: np.flatnonzero(usable & (splits == s)) for s in ds.SPLITS}
        y = {s: y_all[idx[s]].astype(float) for s in ds.SPLITS}
        groups = {s: np.array(video_ids)[idx[s]] for s in ds.SPLITS}
        reps = dc.cluster_bootstrap_indices(groups["test"], args.n_boot, args.seed)

        print(f"\n===== {args.dataset}/{fault} =====")
        for s in ds.SPLITS:
            print(f"  {s}: n={len(idx[s])} pos={int(y[s].sum())} neg={int((1-y[s]).sum())}")

        results, arm_report = {}, {}
        for name, arm in arms.items():
            x = {s: arm["features"][idx[s]] for s in ds.SPLITS}
            res = dc.run_arm(name, x, y, groups, l2=args.l2)
            boot = dc.bootstrap_metric(res, reps)
            res.metrics["ba_ci_low"], res.metrics["ba_ci_high"] = (
                float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
            results[name] = res
            order = np.argsort(-np.abs(res.coef))[:8]
            arm_report[name] = {"metrics": res.metrics,
                                "top_coef": {names[i]: float(res.coef[i]) for i in order}}
        for name, res in sorted(results.items(), key=lambda kv: -kv[1].metrics["auc"]):
            m = res.metrics
            print(f"  {name:>14} AUC {m['auc']:.3f} BA {m['balanced_accuracy']:.3f} "
                  f"[{m['ba_ci_low']:.3f},{m['ba_ci_high']:.3f}] recall {m['recall']:.3f} "
                  f"spec {m['specificity']:.3f}")

        deltas = {}
        for a, b, why in PAIRS:
            if a in results and b in results:
                d = dc.paired_delta(results[a], results[b], reps)
                d["auc_delta"] = results[a].metrics["auc"] - results[b].metrics["auc"]
                deltas[f"{a}-vs-{b}"] = {**d, "isolates": why}
        print("  -- depth deltas (ΔAUC | ΔBA) --")
        for k, d in deltas.items():
            print(f"    {k}: ΔAUC {d['auc_delta']:+.3f} | ΔBA {d['delta']:+.3f} "
                  f"CI[{d['ci_low']:+.3f},{d['ci_high']:+.3f}] p={d['p_two_sided']:.3f}")

        report["faults"][fault] = {
            "splits": {s: {"n": int(len(idx[s])), "pos": int(y[s].sum())} for s in ds.SPLITS},
            "arms": arm_report, "paired_deltas": deltas}

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
