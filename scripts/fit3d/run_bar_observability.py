"""Can body keypoints predict where the barbell is?

Measures the 3D bar axis from Fit3D's calibrated 4-camera rig (no manual annotation, no
GPU), then asks how well a PERFECT keypoint detector -- the ``joints3d_25`` ground truth --
predicts it, LOSO over the 8 training subjects.

    python scripts/fit3d/run_bar_observability.py --action deadlift
    python scripts/fit3d/run_bar_observability.py --action deadlift --json out.json

Extraction is the slow part (video decode), so tracks are cached under
``data/Fit3D/derived/bar_tracks/``; pass ``--refresh`` to recompute.

NOTE ``--action squat`` is refused by default: the extractor is not reliable there. See the
module docstring of ``src.fit3d.bar_geometry``. ``--force-unreliable`` overrides it, but the
numbers should not be reported.
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

from src.fit3d import bar_geometry as bg  # noqa: E402
from src.fit3d import dataset as ds  # noqa: E402
from src.fit3d.axial_rotation import (  # noqa: E402
    loso_scores,
    rbf_krr_predictor,
    ridge_predictor,
)

UNRELIABLE_ACTIONS = {"squat"}
LAMBDAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
CACHE_DIR = ds.DEFAULT_FIT3D_ROOT / "derived" / "bar_tracks"

# The project's own withdrawn OHP bar-path threshold: "anterior offset > ~0.3 shoulder-widths"
# (src/pose/movements/overhead_press.py). Using it means the comparison is against a number
# this codebase already committed to, not one invented for the experiment.
COACHING_THRESHOLD_SHOULDER_WIDTHS = 0.30


def load_track(subject: str, action: str, stride: int, refresh: bool) -> bg.BarTrack:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{subject}_{action}_s{stride}.npz"
    if cache.exists() and not refresh:
        d = np.load(cache)
        return bg.BarTrack(subject, action, d["frames"], d["point"], d["direction"],
                           d["plane_residual"], d["n_views"])
    track = bg.extract_bar_track(subject, action, stride=stride)
    np.savez(cache, frames=track.frames, point=track.point, direction=track.direction,
             plane_residual=track.plane_residual, n_views=track.n_views)
    return track


def best_model(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, list, str]:
    """Best mean ``r2_within`` over ridge and RBF-KRR, lambda tuned.

    Lambda is selected on the same LOSO folds it is scored on, so every number here is an
    UPPER BOUND on keypoint performance. That direction is deliberate: it makes a "keypoints
    cannot see the bar" verdict conservative. It also means a *good* score must be read as
    "at best this good", not as a deployable accuracy.
    """
    best = (-np.inf, None, "")
    for name, maker in (("ridge", ridge_predictor), ("rbf_krr", rbf_krr_predictor)):
        for lam in LAMBDAS:
            folds = loso_scores(x, y, groups, lambda a, b, m=maker, l=lam: m(a, b, l))
            score = float(np.mean([f.r2_within for f in folds]))
            if score > best[0]:
                best = (score, folds, f"{name}(lam={lam:g})")
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--action", default="deadlift")
    ap.add_argument("--stride", type=int, default=bg.DEFAULT_STRIDE)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--force-unreliable", action="store_true")
    ap.add_argument("--max-plane-residual", type=float, default=0.005,
                    help="drop frames whose 4-view triangulation disagrees by more than this (m)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if args.action in UNRELIABLE_ACTIONS and not args.force_unreliable:
        print(f"REFUSED: bar extraction is not reliable for '{args.action}'.")
        print("  See src/fit3d/bar_geometry.py module docstring. Use --force-unreliable to override.")
        return 2

    subjects = [s for s in ds.subjects("train")
                if args.action in ds.actions("train", s)]
    print(f"action={args.action}  subjects={len(subjects)}  stride={args.stride}")

    offsets, directions, feats, groups, widths, residuals = [], [], {}, [], [], []
    retained, wrist_gap, midhands, heights, drop_heights = [], [], [], [], []
    for subject in subjects:
        track = load_track(subject, args.action, args.stride, args.refresh)
        if len(track) == 0:
            print(f"  {subject}: no frames extracted -- skipped")
            continue
        raw_n = len(track)
        span = np.ptp(track.point[:, 2])
        normalised = (track.point[:, 2] - track.point[:, 2].min()) / max(span, 1e-9)
        # Quality gate on the ONLY keypoint-independent signal available: whether the four
        # views agree. Extraction is bimodal -- median plane residual ~1 mm with a heavy tail
        # of frames where RANSAC locked onto the wrong structure -- so this drops failures
        # rather than trimming a smooth distribution. It must NOT be a keypoint-derived
        # criterion: gating on "frames where the wrists explain the bar" would manufacture
        # the very result the experiment is testing for.
        keep = track.plane_residual <= args.max_plane_residual
        track = bg.BarTrack(track.subject, track.action, track.frames[keep], track.point[keep],
                            track.direction[keep], track.plane_residual[keep], track.n_views[keep])
        if len(track) < 20:
            print(f"  {subject}: only {len(track)}/{raw_n} frames survived the quality gate -- skipped")
            continue
        retained.append(len(track) / raw_n)
        heights.append(normalised[keep]); drop_heights.append(normalised[~keep])
        joints = ds.load_joints3d("train", subject, args.action)
        offset, direction = bg.axis_offset_in_body_frame(track, joints)
        offsets.append(offset)
        directions.append(direction)
        widths.append(bg.shoulder_width(joints, track.frames))
        residuals.append(track.plane_residual)
        groups.append(np.full(len(track), subject))
        midhands.append(bg.mid_hand_in_body_frame(joints, track.frames))
        for name, subset in bg.BAR_KEYPOINT_SETS.items():
            feats.setdefault(name, []).append(bg.keypoint_features(joints, track.frames, subset))
        # Physical sanity check, REPORTED not filtered on: in a bar-in-hands lift the bar is
        # gripped, so the wrist-to-axis distance is a near-constant grip geometry. A large sd
        # means the extraction is wandering. Using it as a filter would be circular.
        picked = joints[track.frames]
        mid_wrist = (picked[:, ds.R_WRIST] + picked[:, ds.L_WRIST]) / 2 - picked[:, ds.ROOT]
        _, basis, _ = bg.body_frame(picked)
        wrist_body = np.einsum("fij,fj->fi", basis, mid_wrist)
        gap = np.linalg.norm(offset[:, 1:] - wrist_body[:, 1:], axis=1)
        wrist_gap.append(gap)
        print(f"  {subject}: {len(track):4d}/{raw_n} frames kept  plane residual "
              f"{track.plane_residual.mean()*1000:4.1f} mm  wrist->bar {gap.mean()*100:4.1f} "
              f"+- {gap.std()*100:4.1f} cm")

    if not offsets:
        print("no data")
        return 1

    offset = np.concatenate(offsets)
    direction = np.concatenate(directions)
    groups = np.concatenate(groups)
    width = np.concatenate(widths)
    residual = np.concatenate(residuals)
    feats = {k: np.concatenate(v) for k, v in feats.items()}
    midhand = np.concatenate(midhands)

    gaps = np.concatenate(wrist_gap)
    print(f"\nGT quality: plane residual mean {residual.mean()*1000:.2f} mm  "
          f"p95 {np.percentile(residual,95)*1000:.2f} mm  |  frames retained "
          f"{np.mean(retained)*100:.0f}%")
    print(f"physical check (reported, NOT filtered on): wrist->bar axis "
          f"{gaps.mean()*100:.1f} +- {gaps.std()*100:.1f} cm")
    print(f"shoulder width {width.mean()*100:.1f} cm -> coaching threshold "
          f"{COACHING_THRESHOLD_SHOULDER_WIDTHS} sw = "
          f"{COACHING_THRESHOLD_SHOULDER_WIDTHS*width.mean()*100:.1f} cm")
    tilt = np.degrees(np.arcsin(np.clip(np.abs(direction[:, 1]), 0, 1)))
    print(f"bar axis tilt from body-lateral: {tilt.mean():.1f} deg (sd {tilt.std():.1f})")

    kept_h, drop_h = np.concatenate(heights), np.concatenate(drop_heights)
    if len(drop_h):
        print(f"gate phase bias: kept bar-height {kept_h.mean():.2f} vs dropped {drop_h.mean():.2f} "
              f"(top quintile kept {int((kept_h>0.8).sum())} / dropped {int((drop_h>0.8).sum())})")

    targets = {"anterior": offset[:, 2], "vertical": offset[:, 1]}
    axis_index = {"anterior": 2, "vertical": 1}
    for name, values in targets.items():
        print(f"\n--- target: bar {name} offset from root  "
              f"(n={len(values)}, sd {values.std()*100:.1f} cm) ---")
        base = bg.constant_offset_baseline(values, midhand[:, axis_index[name]], groups)
        print(f"  ZERO-PARAMETER baseline (mid-hand + constant offset): MAE {base*100:.2f} cm "
              f"= {base/width.mean():.3f} sw   <-- a learned model must beat THIS, not the threshold")
        print(f"{'keypoint set':<12} {'dims':>4}  {'R2_within':>9}  {'MAE cm':>7}  "
              f"{'MAE / sw':>8}  model")
        for kp_name, matrix in feats.items():
            score, folds, model = best_model(matrix, values, groups)
            mae_m = float(np.mean([f.mae for f in folds]))
            print(f"{kp_name:<12} {matrix.shape[1]:>4}  {score:>9.3f}  {mae_m*100:>7.2f}  "
                  f"{mae_m/width.mean():>8.3f}  {model}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "action": args.action,
            "stride": args.stride,
            "n_frames": int(len(offset)),
            "subjects": sorted(set(groups.tolist())),
            "plane_residual_mm": {"mean": float(residual.mean() * 1000),
                                  "p95": float(np.percentile(residual, 95) * 1000)},
            "shoulder_width_cm": float(width.mean() * 100),
            "coaching_threshold_cm": float(COACHING_THRESHOLD_SHOULDER_WIDTHS * width.mean() * 100),
            "lambda_selected_on_test_folds": True,
            "results": {},
        }
        for name, values in targets.items():
            out["results"][name] = {}
            for kp_name, matrix in feats.items():
                score, folds, model = best_model(matrix, values, groups)
                mae_m = float(np.mean([f.mae for f in folds]))
                out["results"][name][kp_name] = {
                    "dims": int(matrix.shape[1]), "r2_within": score,
                    "mae_cm": mae_m * 100, "mae_shoulder_widths": mae_m / float(width.mean()),
                    "model": model,
                }
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
