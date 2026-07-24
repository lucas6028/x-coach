"""Frame-level depth-vs-detector classification for shallow-squat OR barbell-row.

Generalises the shallow experiment to any frame-level Fitness-AQA fault. For BarbellRow
it runs both faults (``lumbar``, ``torso_angle``), each with its own split and its own
video-cluster bootstrap; cues come from ``movement_cues`` (spine / torso / upper-body),
not the squat set. Same five-arm intersection, same logistic + threshold-on-val, AUC-first.

    .venv\\Scripts\\python.exe scripts/fitness_aqa/run_frame_depth_classification.py --dataset brow
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
from src.fitness_aqa.cue_features import CAM3D  # noqa: E402

PAIRS = [("nlf_3d", "nlf_2d", "same detector, depth on vs off (primary)"),
         ("mediapipe_3d", "mediapipe_2d", "same detector, heuristic depth on vs off"),
         ("nlf_2d", "mediapipe_2d", "2D detector quality only")]
ARM_SPECS = [("mediapipe_2d", IMAGE2D), ("mediapipe_3d", CAM3D),
             ("nlf_2d", IMAGE2D), ("nlf_3d", CAM3D)]
SPLITS = ("train", "val", "test")


def dataset_config(name: str):
    if name == "brow":
        from src.fitness_aqa import barbellrow_dataset as ds
        from src.fitness_aqa import movement_cues as cue
        return ds, cue, ds.DEFAULT_BROW_ROOT, list(ds.FAULTS)
    if name == "shallow":
        from src.fitness_aqa import cue_features as cue
        from src.fitness_aqa import shallow_dataset as ds
        return ds, cue, ds.DEFAULT_SHALLOW_ROOT, ["shallow"]
    raise SystemExit(f"unknown dataset {name!r}")


def load_arm_features(pose_dir: Path, name: str, space: str, cue, order: list[str], row_of: dict):
    """Return (features (N,D), valid (N,)) aligned to manifest order for arm ``name``."""
    f = pose_dir / f"{name}.npz"
    joints = np.full((len(order), 17, 3 if space == CAM3D else 2), np.nan)
    detected = np.zeros(len(order), bool)
    if f.exists():
        with np.load(f, allow_pickle=False) as d:
            for k, sid in enumerate([str(s) for s in d["sample_ids"]]):
                if sid in row_of:
                    joints[row_of[sid]] = d["joints"][k]
                    detected[row_of[sid]] = d["detected"][k]
    feats = cue.compute_features(joints, space)
    return feats, detected & np.isfinite(feats).all(axis=1)


def manifest_for(ds, fault: str, root: Path):
    if fault == "shallow":
        return ds.load_manifest(root)
    return ds.load_manifest(fault, root)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["shallow", "brow"])
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    ds, cue, root, faults = dataset_config(args.dataset)
    pose_dir = root / "derived" / "pose"
    out = args.out or root / "derived" / f"{args.dataset}_depth_classification.json"
    report = {"dataset": args.dataset, "faults": {},
              "config": {"l2": args.l2, "n_boot": args.n_boot}}

    for fault in faults:
        man = manifest_for(ds, fault, root)
        order = [r["id"] for r in man]
        row_of = {sid: i for i, sid in enumerate(order)}
        labels = np.array([r["label"] for r in man], float)
        splits = np.array([r["split"] for r in man])
        groups = np.array([r["video_id"] for r in man])

        arms = {}
        for name, space in ARM_SPECS:
            feats, valid = load_arm_features(pose_dir, name, space, cue, order, row_of)
            arms[name] = {"features": feats, "valid": valid}
        common = np.ones(len(order), bool)
        for a in arms.values():
            common &= a["valid"]
        idx = {s: np.flatnonzero(common & (splits == s)) for s in SPLITS}
        y = {s: labels[idx[s]] for s in SPLITS}
        g = {s: groups[idx[s]] for s in SPLITS}
        reps = dc.cluster_bootstrap_indices(g["test"], args.n_boot, args.seed)

        print(f"\n===== {args.dataset}/{fault}  common {common.sum()}/{len(order)} =====")
        for s in SPLITS:
            print(f"  {s}: n={len(idx[s])} pos={int(y[s].sum())} neg={int((1-y[s]).sum())} "
                  f"videos={len(set(g[s]))}")

        results, arm_report = {}, {}
        for name, arm in arms.items():
            x = {s: arm["features"][idx[s]] for s in SPLITS}
            res = dc.run_arm(name, x, y, g, l2=args.l2)
            boot = dc.bootstrap_metric(res, reps)
            res.metrics["ba_ci_low"], res.metrics["ba_ci_high"] = (
                float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
            results[name] = res
            o = np.argsort(-np.abs(res.coef))[:8]
            arm_report[name] = {"metrics": res.metrics,
                                "top_coef": {cue.FEATURE_NAMES[i]: float(res.coef[i]) for i in o}}
        for name, res in sorted(results.items(), key=lambda kv: -kv[1].metrics["auc"]):
            m = res.metrics
            print(f"  {name:>14} AUC {m['auc']:.3f} BA {m['balanced_accuracy']:.3f} "
                  f"recall {m['recall']:.3f} spec {m['specificity']:.3f}")

        deltas = {}
        for a, b, why in PAIRS:
            if a in results and b in results:
                d = dc.paired_delta(results[a], results[b], reps)
                d["auc_delta"] = results[a].metrics["auc"] - results[b].metrics["auc"]
                deltas[f"{a}-vs-{b}"] = {**d, "isolates": why}
        print("  -- depth deltas (ΔAUC | ΔBA) --")
        for k, d in deltas.items():
            print(f"    {k}: ΔAUC {d['auc_delta']:+.3f} | ΔBA {d['delta']:+.3f} p={d['p_two_sided']:.3f}")

        report["faults"][fault] = {
            "splits": {s: {"n": int(len(idx[s])), "pos": int(y[s].sum())} for s in SPLITS},
            "arms": arm_report, "paired_deltas": deltas}

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
