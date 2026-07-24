"""Does the depth channel change the shallow-squat verdict on Fitness-AQA?

Loads every extracted arm, builds identical cue features, restricts all arms to the
**same** rows (a sample any arm failed on is dropped from all of them), fits the same
logistic regression on the official video-disjoint splits, and reports test balanced
accuracy with a video-level cluster bootstrap.

The comparison that answers the question is the *within-model* pair -- ``nlf_3d`` vs
``nlf_2d``, same detector with and without depth. Cross-model pairs (nlf vs mediapipe)
confound depth with detector quality and are reported only as context.

    .venv\\Scripts\\python.exe scripts/fitness_aqa/run_shallow_depth_classification.py
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

from src.fitness_aqa import cue_features as cf  # noqa: E402
from src.fitness_aqa import depth_classify as dc  # noqa: E402
from src.fitness_aqa import shallow_dataset as sd  # noqa: E402

DEFAULT_POSE_DIR = sd.DEFAULT_SHALLOW_ROOT / "derived" / "pose"
DEFAULT_OUT = sd.DEFAULT_SHALLOW_ROOT / "derived" / "shallow_depth_classification.json"

# Pairs worth naming in the report: (arm_a, arm_b, what the delta isolates).
PAIRS = [
    ("nlf_3d", "nlf_2d", "same detector, depth on vs off (primary)"),
    ("mediapipe_3d", "mediapipe_2d", "same detector, heuristic depth on vs off"),
    ("nlf_2d", "mediapipe_2d", "2D detector quality only"),
    ("nlf_2d", "rtmpose_2d", "2D detector quality only"),
    ("nlf_3d", "mediapipe_3d", "depth quality: regressed vs heuristic"),
]


def load_arm(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {
            "name": path.stem,
            "sample_ids": [str(s) for s in d["sample_ids"]],
            "joints": d["joints"].astype(np.float64),
            "detected": d["detected"],
            "space": str(d["space"]) if "space" in d else cf.IMAGE2D,
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose-dir", type=Path, default=DEFAULT_POSE_DIR)
    p.add_argument("--arms", nargs="*", default=None, help="arm npz stems; default = all found")
    p.add_argument("--root", type=Path, default=sd.DEFAULT_SHALLOW_ROOT)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mlp-seeds", type=int, default=0, help="also run an MLP with this many seeds (0 = skip)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    manifest = sd.load_manifest(args.root)
    order = [r["id"] for r in manifest]
    row_of = {sid: i for i, sid in enumerate(order)}
    labels = np.array([r["label"] for r in manifest], dtype=np.float64)
    splits = np.array([r["split"] for r in manifest])
    groups = np.array([r["video_id"] for r in manifest])

    paths = sorted(args.pose_dir.glob("*.npz"))
    if args.arms:
        paths = [p for p in paths if p.stem in set(args.arms)]
    if not paths:
        raise SystemExit(f"no arm npz found in {args.pose_dir}")

    arms: dict[str, dict] = {}
    for path in paths:
        arm = load_arm(path)
        # Re-index onto manifest order; an arm may have been run on a subset.
        joints = np.full((len(order), arm["joints"].shape[1], arm["joints"].shape[2]), np.nan)
        detected = np.zeros(len(order), bool)
        for k, sid in enumerate(arm["sample_ids"]):
            if sid in row_of:
                joints[row_of[sid]] = arm["joints"][k]
                detected[row_of[sid]] = arm["detected"][k]
        feats = cf.compute_features(joints, arm["space"])
        valid = detected & np.isfinite(feats).all(axis=1)
        arms[arm["name"]] = {"features": feats, "valid": valid, "space": arm["space"]}
        print(f"{arm['name']:>14}  space={arm['space']:<8} valid {valid.sum()}/{len(order)} "
              f"({valid.mean():.3f})", flush=True)

    common = np.ones(len(order), bool)
    for arm in arms.values():
        common &= arm["valid"]
    print(f"\ncommon rows across {len(arms)} arms: {common.sum()}/{len(order)} ({common.mean():.3f})", flush=True)

    idx = {s: np.flatnonzero(common & (splits == s)) for s in sd.SPLITS}
    for s in sd.SPLITS:
        y = labels[idx[s]]
        print(f"  {s:>5}: n={len(y):>5}  pos={int(y.sum()):>4}  neg={int((1 - y).sum()):>4}  "
              f"videos={len(set(groups[idx[s]]))}")

    y = {s: labels[idx[s]] for s in sd.SPLITS}
    g = {s: groups[idx[s]] for s in sd.SPLITS}
    reps = dc.cluster_bootstrap_indices(g["test"], args.n_boot, args.seed)

    results: dict[str, dc.ArmResult] = {}
    report: dict[str, dict] = {}
    for name, arm in arms.items():
        x = {s: arm["features"][idx[s]] for s in sd.SPLITS}
        res = dc.run_arm(name, x, y, g, l2=args.l2)
        boot = dc.bootstrap_metric(res, reps)
        res.metrics["ba_ci_low"] = float(np.percentile(boot, 2.5))
        res.metrics["ba_ci_high"] = float(np.percentile(boot, 97.5))
        results[name] = res
        report[name] = {"space": arm["space"], "metrics": res.metrics,
                        "coef": dict(zip(cf.FEATURE_NAMES, res.coef.tolist()))}

    print("\n=== test metrics (threshold chosen on val for balanced accuracy) ===")
    print(f"{'arm':>14} {'space':>8} {'BA':>6} {'95% CI':>16} {'AUC':>6} {'recall':>7} {'spec':>7} {'trainBA':>8}")
    for name, res in sorted(results.items(), key=lambda kv: -kv[1].metrics["balanced_accuracy"]):
        m = res.metrics
        print(f"{name:>14} {report[name]['space']:>8} {m['balanced_accuracy']:>6.3f} "
              f"[{m['ba_ci_low']:.3f},{m['ba_ci_high']:.3f}] {m['auc']:>6.3f} {m['recall']:>7.3f} "
              f"{m['specificity']:>7.3f} {m['train_balanced_accuracy']:>8.3f}")

    print("\n=== paired deltas (cluster bootstrap over test videos) ===")
    deltas = {}
    for a, b, why in PAIRS:
        if a not in results or b not in results:
            continue
        d = dc.paired_delta(results[a], results[b], reps)
        deltas[f"{a}-vs-{b}"] = {**d, "isolates": why}
        print(f"{a:>14} - {b:<14} {d['delta']:+.3f}  95% CI [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]  "
              f"p={d['p_two_sided']:.3f}   {why}")

    # Depth-redundancy check on the primary pair: is +0.003 a ceiling artifact, or do the
    # two arms genuinely read the same signal? If 3D fixes ~as many as it breaks and the
    # scores correlate, depth is redundant here, not merely headroom-limited.
    overlaps = {}
    for a, b in (("nlf_3d", "nlf_2d"), ("mediapipe_3d", "mediapipe_2d")):
        if a not in results or b not in results:
            continue
        ra, rb = results[a], results[b]
        yt = ra.test_labels
        pa = (ra.test_scores >= ra.metrics["threshold"]).astype(int)
        pb = (rb.test_scores >= rb.metrics["threshold"]).astype(int)
        wa, wb = pa != yt, pb != yt
        overlaps[f"{a}-vs-{b}"] = {
            "n": int(len(yt)),
            "wrong_3d": int(wa.sum()), "wrong_2d": int(wb.sum()),
            "three_d_fixes": int((wb & ~wa).sum()), "three_d_breaks": int((~wb & wa).sum()),
            "both_wrong": int((wa & wb).sum()),
            "pred_agreement": float(np.mean(pa == pb)),
            "score_pearson_r": float(np.corrcoef(ra.test_scores, rb.test_scores)[0, 1]),
        }
    print("\n=== depth-redundancy (3D vs 2D error overlap) ===")
    for k, o in overlaps.items():
        print(f"{k}: 3D fixes {o['three_d_fixes']} / breaks {o['three_d_breaks']}  "
              f"agree {o['pred_agreement']:.3f}  score r {o['score_pearson_r']:.3f}")

    mlp = {}
    if args.mlp_seeds:
        print("\n=== MLP capacity check (same features, same splits) ===")
        for name, arm in arms.items():
            x = {s: arm["features"][idx[s]] for s in sd.SPLITS}
            bas = []
            for seed in range(1, args.mlp_seeds + 1):
                val_scores, test_scores = dc.fit_mlp(x, y, seed=seed)
                t = dc.select_threshold(y["val"], val_scores)
                bas.append(dc.binary_metrics(y["test"], test_scores, t)["balanced_accuracy"])
            mlp[name] = {"mean": float(np.mean(bas)), "std": float(np.std(bas)), "seeds": bas}
            print(f"{name:>14}  BA {np.mean(bas):.3f} +/- {np.std(bas):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_common": int(common.sum()),
        "n_total": len(order),
        "splits": {s: {"n": int(len(idx[s])), "pos": int(y[s].sum()),
                       "videos": len(set(g[s]))} for s in sd.SPLITS},
        "arms": report,
        "paired_deltas": deltas,
        "depth_redundancy": overlaps,
        "mlp": mlp,
        "config": {"l2": args.l2, "n_boot": args.n_boot, "seed": args.seed},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
