"""CLI for Fit3D experiment 0 (is segment axial rotation observable from joint centres?).

Run from the repository root:

    python scripts/fit3d/run_axial_rotation.py --action squat
    python scripts/fit3d/run_axial_rotation.py --action squat --side R --frame gt
    python scripts/fit3d/run_axial_rotation.py --action deadlift --json data/Fit3D/derived/axial_rotation_deadlift.json

``--frame gt`` re-expresses the skeleton in the true SMPLX pelvis frame. That is an
ORACLE (it consumes ground truth and is not deployable); it exists to separate "this
joint set carries hip information" from "this joint set helps pin down the reference
frame". The default keypoint frame builds its lateral axis from ``L_HIP - R_HIP``, which
leaks bilateral information into every joint set -- including single-leg ones.
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

from src.fit3d import axial_rotation as ax  # noqa: E402
from src.fit3d import dataset as ds  # noqa: E402

RIDGE_LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
KRR_LAMBDAS = (0.1, 1.0, 10.0)


def build_dataset(split, action, side, subjects, frame, subsample, root):
    """Return (X (N, 25, 3) canonical, y (N,) twist degrees, groups (N,) subject ids)."""
    xs, ys, groups = [], [], []
    for subj in subjects:
        smplx_path = root / split / subj / "smplx" / f"{action}.json"
        joints_path = root / split / subj / "joints3d_25" / f"{action}.json"
        if not (smplx_path.exists() and joints_path.exists()):
            continue
        smplx = ax.load_smplx(split, subj, action, root)
        joints = ds.load_joints3d(split, subj, action, root)
        n = min(len(smplx["body_pose"]), len(joints))
        twist = ax.hip_twist_series(smplx["body_pose"][:n], side)
        if frame == "gt":
            canon = ax.canonicalize_gt(joints[:n], smplx["global_orient"][:n, 0])
        else:
            canon = ax.canonicalize(joints[:n])
        keep = np.arange(0, n, subsample)
        ok = np.isfinite(twist[keep]) & np.isfinite(canon[keep]).all(axis=(1, 2))
        xs.append(canon[keep][ok])
        ys.append(twist[keep][ok])
        groups.append(np.full(int(ok.sum()), subj))
    if not xs:
        raise SystemExit(f"no sequences found for action={action!r} (train split has the GT)")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)


def best_over_lambdas(x, y, groups):
    """Tune lambda per joint set -- being generous to keypoints keeps the bar honest."""
    best = None
    for lam in RIDGE_LAMBDAS:
        scores = ax.loso_scores(x, y, groups, lambda a, b, L=lam: ax.ridge_predictor(a, b, L))
        rw = float(np.mean([s.r2_within for s in scores]))
        if best is None or rw > best[1]:
            best = ("ridge", rw, lam, scores)
    for lam in KRR_LAMBDAS:
        scores = ax.loso_scores(x, y, groups, lambda a, b, L=lam: ax.rbf_krr_predictor(a, b, L))
        rw = float(np.mean([s.r2_within for s in scores]))
        if rw > best[1]:
            best = ("rbf_krr", rw, lam, scores)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--action", default="squat", help="Fit3D action name (default: squat)")
    parser.add_argument("--split", default="train", help="dataset split (train has SMPLX GT)")
    parser.add_argument("--side", default="L", choices=("L", "R"), help="which hip")
    parser.add_argument("--frame", default="keypoint", choices=("keypoint", "gt"),
                        help="canonicalisation frame; 'gt' is an oracle (see module docstring)")
    parser.add_argument("--subjects", nargs="*", default=None, help="restrict to these subjects")
    parser.add_argument("--subsample", type=int, default=5, help="keep every Nth frame")
    parser.add_argument("--root", type=Path, default=ds.DEFAULT_FIT3D_ROOT)
    parser.add_argument("--json", type=Path, default=None, help="write the full result here")
    args = parser.parse_args()

    subjects = args.subjects or ds.subjects(args.split, args.root)
    x_all, y, groups = build_dataset(args.split, args.action, args.side, subjects,
                                     args.frame, args.subsample, args.root)

    header = (f"{args.action} | {args.side} hip | frame={args.frame} | "
              f"n={len(y)} | target sd {np.std(y):.2f} deg")
    print(f"=== axial-rotation observability from perfect 3D keypoints ===\n{header}\n")
    print(f"{'joint set':12} {'dims':>5} {'model':>8} {'lambda':>8} {'R2_within':>10} {'MAE deg':>9}")

    result = {"action": args.action, "side": args.side, "frame": args.frame,
              "n": int(len(y)), "target_sd_deg": float(np.std(y)),
              "sign_convention": ("UNVERIFIED -- consistent within a side, mirrored between "
                                  "L and R; do NOT read the sign as internal/external rotation "
                                  "without checking it first"),
              "twist_axis": "hip-centre -> knee-centre (proxy for the femoral mechanical axis)",
              "subjects": sorted(set(groups.tolist())), "sets": {}}
    for name, idx in ax.KEYPOINT_SETS.items():
        x = x_all[:, idx].reshape(len(x_all), -1)
        model, rw, lam, scores = best_over_lambdas(x, y, groups)
        mae = float(np.mean([s.mae for s in scores]))
        print(f"{name:12} {x.shape[1]:5d} {model:>8} {lam:8g} {rw:+10.3f} {mae:9.2f}")
        result["sets"][name] = {
            "dims": int(x.shape[1]), "model": model, "lambda": lam,
            "r2_within": rw, "mae_deg": mae,
            "r2_global": float(np.mean([s.r2_global for s in scores])),
            "per_subject": [{"subject": s.subject, "r2_within": s.r2_within,
                             "mae_deg": s.mae, "target_sd_deg": s.target_sd} for s in scores],
        }

    best_set = min(result["sets"].items(), key=lambda kv: kv[1]["mae_deg"])
    print(f"\nBAR for a monocular rotation estimator: it must beat "
          f"{best_set[1]['mae_deg']:.2f} deg MAE (set '{best_set[0]}') to add anything.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
