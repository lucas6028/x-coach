"""Fit3D experiment 0d -- can a monocular model ESTIMATE femoral axial rotation well enough?

Experiment 0 (``notes/fit3d_axial_rotation_summary.md``) established that a *perfect* keypoint
skeleton already predicts femoral twist to 2.6-4.0 deg MAE. An explicit rotation channel therefore
only earns its place if a monocular model estimates the rotation *better than that bar*. This
script reads the rotation kernel's output and answers that.

    estimate = HMR2.0 ``body_pose`` rotmats (Kaggle kernel ``fit3d-hmr2-rotation``)
    truth    = Fit3D SMPLX ``body_pose`` rotmats

Both are **parent-relative** local rotations, so HMR2.0's crop-frame global orientation -- the
caveat that dogged the depth experiments -- does not enter here at all.

Run from the repository root, after downloading the kernel output:

    .venv\\Scripts\\kaggle.exe kernels output haoping6028/fit3d-hmr2-rotation -p .kaggle_tmp/fit3d_hmr2_rot_out
    python scripts/fit3d/run_rotation_estimate_eval.py --pred-dir .kaggle_tmp/fit3d_hmr2_rot_out/fit3d_hmr2_rot
    python scripts/fit3d/run_rotation_estimate_eval.py --pred-dir ... --bar 2.57 --action deadlift
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

#: Best keypoint MAE per action x side from experiment 0 -- the bar the estimator must beat.
KEYPOINT_BAR_DEG = {("squat", "L"): 3.98, ("squat", "R"): 3.72,
                    ("deadlift", "L"): 2.57, ("deadlift", "R"): 3.23}


def load_predictions(pred_dir: Path, action: str) -> list[dict]:
    files = sorted(pred_dir.glob(f"*__{action}__*.npz"))
    if not files:
        raise SystemExit(
            f"no '*__{action}__*.npz' under {pred_dir}.\n"
            "Download the kernel output first:\n"
            "  .venv\\Scripts\\kaggle.exe kernels output haoping6028/fit3d-hmr2-rotation "
            "-p .kaggle_tmp/fit3d_hmr2_rot_out"
        )
    out = []
    for path in files:
        with np.load(path, allow_pickle=True) as npz:
            if "body_pose" not in npz.files:
                raise SystemExit(
                    f"{path.name} has no 'body_pose' -- this is an output of the ORIGINAL "
                    "fit3d-hmr2-extract kernel, which discarded the rotation params. Run "
                    "fit3d-hmr2-rotation instead."
                )
            out.append({
                "subject": str(npz["subject"]), "camera": str(npz["camera"]),
                "action": str(npz["action"]), "body_pose": npz["body_pose"],
                "rest": npz["rest_j_smpl"] if "rest_j_smpl" in npz.files else None,
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=Path, required=True, help="kernel output npz directory")
    parser.add_argument("--action", default="squat")
    parser.add_argument("--side", default="L", choices=("L", "R"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--bar", type=float, default=None,
                        help="keypoint MAE to beat (default: experiment 0's value for this cell)")
    parser.add_argument("--root", type=Path, default=ds.DEFAULT_FIT3D_ROOT)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    bar = args.bar if args.bar is not None else KEYPOINT_BAR_DEG.get((args.action, args.side))
    preds = load_predictions(args.pred_dir, args.action)

    gt_cache: dict[str, np.ndarray] = {}
    gt_all, est_all, grp_all, cam_all = [], [], [], []
    for rec in preds:
        subj = rec["subject"]
        if subj not in gt_cache:
            smplx = ax.load_smplx(args.split, subj, args.action, args.root)
            gt_cache[subj] = ax.hip_twist_series(smplx["body_pose"], args.side)
        gt = gt_cache[subj]
        est = ax.hip_twist_series(rec["body_pose"], args.side, rest=rec["rest"])
        n = min(len(gt), len(est))
        gt_all.append(gt[:n]); est_all.append(est[:n])
        grp_all.append(np.full(n, subj)); cam_all.append(np.full(n, rec["camera"]))

    gt = np.concatenate(gt_all); est = np.concatenate(est_all)
    grp = np.concatenate(grp_all); cam = np.concatenate(cam_all)
    agree = ax.compare_twist(gt, est, grp)

    print(f"=== HMR2.0 estimated vs SMPLX ground-truth femoral twist ===")
    print(f"{args.action} | {args.side} hip | {len(set(grp.tolist()))} subjects | "
          f"{len(preds)} videos | {agree.n} paired frames\n")
    print(f"  MAE raw            {agree.mae_raw:6.2f} deg   (carries the SMPL-vs-SMPLX offset)")
    print(f"  MAE debiased(LOSO) {agree.mae_debiased:6.2f} deg   <- compare against the bar")
    print(f"  MAE oracle         {agree.mae_oracle:6.2f} deg   (per-subject offset; upper bound)")
    if agree.loso_is_degenerate:
        print(f"  !! only {agree.n_groups} subject: there are no other subjects to fit the offset on,")
        print( "     so 'debiased(LOSO)' above IS the oracle number and is OPTIMISTIC. The bar was")
        print( "     computed under real LOSO over 8 subjects, so this is not yet a fair comparison.")
    print(f"  bias               {agree.bias:+6.2f} deg")
    print(f"  pearson r          {agree.pearson:+6.3f}   (near 0 => tracks nothing; "
          f"strongly negative => sign-convention bug)")

    print("\n  per camera (view-dependence of rotation estimation):")
    for c in sorted(set(cam.tolist())):
        m = cam == c
        try:
            sub = ax.compare_twist(gt[m], est[m], grp[m])
            print(f"    {c:12} n={sub.n:6d}  MAE deb {sub.mae_debiased:5.2f}  r {sub.pearson:+.3f}")
        except ValueError:
            print(f"    {c:12} (too few paired finite frames)")

    verdict = None
    if bar is not None:
        margin = bar - agree.mae_debiased
        verdict = "BEATS" if margin > 0 else "does NOT beat"
        print(f"\nVERDICT: estimator {verdict} the keypoint bar "
              f"({agree.mae_debiased:.2f} vs {bar:.2f} deg, margin {margin:+.2f}).")
        if margin <= 0:
            print("  => an explicit rotation channel adds nothing here; a full-body keypoint")
            print("     skeleton already estimates this rotation more accurately.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "action": args.action, "side": args.side, "n_videos": len(preds),
            "bar_deg": bar, "verdict": verdict,
            "agreement": {k: getattr(agree, k) for k in
                          ("n", "mae_raw", "mae_debiased", "mae_oracle", "bias", "pearson")},
        }, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
