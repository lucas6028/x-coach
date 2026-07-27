"""Does NLF's per-joint uncertainty carry information its keypoints do not?

    python scripts/fit3d/run_uncertainty_eval.py --action squat
    python scripts/fit3d/run_uncertainty_eval.py --action deadlift --json out.json

Four questions, in the order that can kill the line earliest:

1. **Calibrated?**  within-joint Spearman between ``unc`` and true error.
2. **Better than nothing?**  vs a per-joint constant lookup (zero features).
3. **Redundant?**  does it add anything on top of the predicted pose itself?
4. **Useful for routing?**  can it pick which of 4 synchronised views to trust?

All comparisons are WITHIN joint. Comparing uncertainty across joints is invalid here: the
SMPL/H36M joint conventions differ by up to 176 mm, which swamps real localisation error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fit3d import uncertainty_eval as ue  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--action", default="squat")
    ap.add_argument("--pred-root", type=Path, default=ue.DEFAULT_PRED_ROOT)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    sequences = ue.load_sequences(args.action, args.pred_root)
    if not sequences:
        print(f"no NLF predictions for action={args.action} under {args.pred_root}")
        return 1
    subjects = sorted({s.subject for s in sequences})
    cameras = sorted({s.camera for s in sequences})
    print(f"action={args.action}  sequences={len(sequences)}  subjects={len(subjects)}  "
          f"cameras={len(cameras)}  swap_lr={sorted({s.swap_lr for s in sequences})}")

    error = np.concatenate([ue.corrected_error(s.delta) for s in sequences])
    uncertainty = np.concatenate([s.uncertainty for s in sequences])
    bias = ue.convention_bias(np.concatenate([s.delta for s in sequences]))
    print(f"joint-convention offset (NOT error): max {np.linalg.norm(bias[0], axis=1).max():.0f} mm "
          f"at {ue.H36M17_NAMES[int(np.argmax(np.linalg.norm(bias[0], axis=1)))]} "
          f"-- this is why nothing here compares across joints")

    print("\n--- 1. CALIBRATION (within joint; pelvis is the root so its error is 0 by "
          "construction and it reports nan) ---")
    calibration = ue.within_joint_calibration(error, uncertainty)
    for j in np.argsort(-np.nan_to_num(calibration, nan=-9)):
        if np.isnan(calibration[j]):
            continue
        print(f"   {ue.H36M17_NAMES[j]:<12} rho {calibration[j]:+.3f}   "
              f"median err {np.median(error[:, j]):6.1f} mm   median unc {np.median(uncertainty[:, j]):5.1f} mm")
    print(f"   MEAN rho {np.nanmean(calibration):+.3f} over "
          f"{int(np.isfinite(calibration).sum())} non-degenerate joints")

    print("\n--- 2/3. IS IT REDUNDANT?  (LOSO, MAE mm on bias-corrected error, lower better) ---")
    red = ue.redundancy_test(sequences)
    print(f"   lookup   (per-joint constant,  0 feats): {red['lookup']:6.2f}")
    print(f"   unc      (uncertainty only,    1 feat ): {red['unc']:6.2f}  vs lookup {red['unc']-red['lookup']:+.2f}")
    print(f"   pose     (predicted pose,     51 feats): {red['pose']:6.2f}  vs lookup {red['pose']-red['lookup']:+.2f}")
    print(f"   pose+unc (both,               52 feats): {red['pose_unc']:6.2f}  vs pose   {red['pose_unc']-red['pose']:+.2f}")
    print("   ^ the last line is the verdict: if adding unc to pose is ~0, the channel is redundant.")

    print("\n--- 4. CROSS-VIEW ROUTING (synchronised cameras; which view to trust) ---")
    agree = ue.cross_view_agreement(sequences)
    print(f"   unc-vs-true-error agreement, standardised within camera x joint: "
          f"rho {agree['mean_rho']:+.3f} over {agree['n_subjects']} subjects")
    route = ue.cross_view_routing(sequences)
    print(f"   rank agreement with true error:  unc {route['unc_rho']:+.3f}   pose {route['pose_rho']:+.3f}")
    print(f"   top-1 'pick the best view':      unc {route['unc_top1']*100:5.1f}%  "
          f"pose {route['pose_top1']*100:5.1f}%  chance {route['chance_top1']*100:5.1f}%")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "action": args.action,
            "n_sequences": len(sequences), "subjects": subjects, "cameras": cameras,
            "max_convention_offset_mm": float(np.linalg.norm(bias[0], axis=1).max()),
            "calibration_within_joint": {ue.H36M17_NAMES[j]: (None if np.isnan(calibration[j])
                                                              else float(calibration[j]))
                                         for j in range(ue.N_JOINTS)},
            "calibration_mean_rho": float(np.nanmean(calibration)),
            "redundancy_mae_mm": red,
            "cross_view_agreement": agree,
            "cross_view_routing": route,
        }, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
