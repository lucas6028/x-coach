"""CLI for Fit3D experiment 1 (monocular-3D depth recovery vs mocap GT).

Compares a monocular 3D method's predictions against Fit3D ground truth: the in-plane
vs depth error decomposition, plus the squat-cue errors set side by side with the
single-view 2D-projection baseline from experiment 2 (apples-to-apples: both are
"readings" of the same true 3D cue).

Run from the repository root, after the NLF kernel output is under --pred-root:

    python scripts/fit3d/run_depth_eval.py --pred-root data/Fit3D/derived/preds/nlf \
        --json data/Fit3D/derived/depth_eval_squat_nlf.json
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

from src.fit3d import dataset as ds  # noqa: E402
from src.fit3d import depth_eval as de  # noqa: E402
from src.fit3d.biomech import IMAGE2D, WORLD3D, frame_metrics  # noqa: E402


def projection_biomech_error(action: str, split: str, subjs, root) -> dict:
    """Single-view 2D-projection cue error vs 3D truth, per-frame mean-abs, matched to NLF."""
    acc = {k: [] for k in de.BIOMECH_KEYS}
    for subj in subjs or ds.subjects(split, root):
        if action not in ds.actions(split, subj, root):
            continue
        j3d_m = ds.load_joints3d(split, subj, action, root)  # metres (correct for projection)
        gm = frame_metrics(j3d_m, WORLD3D)
        for cam in ds.cameras(split, subj, root):
            cp = ds.read_cam_params(split, subj, cam, action, root)
            pm = frame_metrics(ds.project_world_to_image(j3d_m, cp), IMAGE2D)
            for k in de.BIOMECH_KEYS:
                acc[k].append(float(np.nanmean(np.abs(pm[k] - gm[k]))))
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-root", type=Path, required=True, help="dir of <subj>__<action>__<cam>.npz")
    p.add_argument("--action", default="squat")
    p.add_argument("--split", default="train")
    p.add_argument("--pred-units", default="mm", choices=["mm", "m"])
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    proj = projection_biomech_error(args.action, args.split, args.subjects, ds.DEFAULT_FIT3D_ROOT)
    results = {}
    for source in ("smpl3d", "smpl3d_np"):
        r = de.evaluate(args.pred_root, action=args.action, split=args.split,
                        pred_units=args.pred_units, source=source, subjs=args.subjects)
        if r["n"] == 0:
            print(f"no predictions found under {args.pred_root}"); return
        results[source] = r

    r = results["smpl3d"]
    print(f"Fit3D depth recovery -- action={args.action}  n={r['n']} sequences  "
          f"(L/R resolved: swap={r['swap_lr']})")
    print("\n  position error vs mocap GT (mm, root-relative):")
    print(f"  {'readout':<12}{'MPJPE':>8}{'PA-MPJPE':>10}{'err_x':>8}{'err_y':>8}{'err_z(depth)':>14}{'ez/exy':>8}")
    for source, res in results.items():
        a = res["aggregate"]
        exy = 0.5 * (a["ex"] + a["ey"])
        print(f"  {source:<12}{a['mpjpe']:>8.1f}{a['pa_mpjpe']:>10.1f}{a['ex']:>8.1f}{a['ey']:>8.1f}"
              f"{a['ez']:>14.1f}{a['ez']/exy:>8.2f}")
    print("  (ez/exy ~1 => per-axis depth error on par with in-plane; the 2D-lifting failure mode is ez >> exy)")

    units = {"knee_angle": "deg", "hip_angle": "deg", "torso_lean_deg": "deg", "depth_ratio": ""}
    print("\n  squat-cue error vs 3D truth -- NLF monocular-3D vs single-view 2D projection:")
    print(f"  {'cue':<16}{'2D-view':>10}{'NLF param':>12}{'NLF nonpar':>12}   recovered?")
    for k in de.BIOMECH_KEYS:
        twod = proj[k]
        nlf = results["smpl3d"]["aggregate"][f"err_{k}"]
        nlf_np = results["smpl3d_np"]["aggregate"][f"err_{k}"]
        verdict = "YES" if nlf < 0.6 * twod else ("partial" if nlf < twod else "no")
        print(f"  {k:<16}{twod:>10.2f}{nlf:>12.2f}{nlf_np:>12.2f}   {verdict} ({units[k] or 'ratio'})")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"projection_baseline": proj,
                   "nlf": {s: results[s]["aggregate"] for s in results},
                   "swap_lr": r["swap_lr"], "n": r["n"]}
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
