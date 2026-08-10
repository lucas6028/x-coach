"""CLI for the Torso Twist rotation-proxy sensing-fidelity pass.

Reproduces every number quoted in
`docs/superpowers/specs/2026-08-10-torso-twist-detector-design.md` sections 8.1 and 8.2, and in
the parent spec's Group F update block. Run it before editing either of those, and re-run it
after any change to `src/pose/movements/torso_twist.py`'s brace construction.

    .venv\\Scripts\\python.exe scripts/fit3d/run_rotation_proxy_fidelity.py \
        --json data/Fit3D/derived/torso_twist_rotation_fidelity.json

The 2-D side is MOCAP-2D -- ground truth projected through the real calibration, i.e. a PERFECT
detector -- so every error printed is projection alone. The corpus is `standing_ab_twists`, which
is a DIFFERENT VARIANT from the seated Russian twist the detector models: the projection geometry
transfers, the distributions do not, and no threshold may be taken from this output.

`--jitter` additionally measures MediaPipe's own frame-to-frame width movement on REHAB24-6's
cached landmarks, which is the noise floor the sensitivity figures are compared against.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fit3d import rotation_proxy_fidelity as rp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", default="standing_ab_twists")
    parser.add_argument("--split", default="train")
    parser.add_argument("--jitter", action="store_true", help="Also measure the MediaPipe noise floor.")
    parser.add_argument("--json", type=Path, default=None, help="Write the full result here.")
    args = parser.parse_args()

    result = rp.run(action=args.action, split=args.split)
    if args.jitter:
        result["mediapipe_width_jitter"] = rp.mediapipe_width_jitter()

    print(f"=== rotation proxy fidelity: {args.action} ({result['records']} line records) ===\n")
    for line, stats in result["lines"].items():
        print(f"--- {line} line ---")
        print(f"  true peak |rotation| deg : median {stats['true_peak_deg']['median']:6.1f}")
        print(f"  proxy peak estimate deg  : median {stats['proxy_peak_deg']['median']:6.1f}")
        print(f"  per-frame MAE deg        : median {stats['per_frame_mae_deg']['median']:6.1f}")
        print(f"  per-rep correlation      : median {stats['per_rep_correlation']['median']:6.2f}")
        print(f"  fraction anticorrelated  : {stats['fraction_anticorrelated']:6.2f}\n")

    twist = result["true_relative_trunk_twist_peak_deg"]
    print(f"true relative trunk twist peak deg: median {twist['median']:.1f}  p90 {twist['p90']:.1f}  max {twist['max']:.1f}")
    print("  (the spec's left-right x-ordering flip needs >90 deg, so it never occurs)\n")

    ratio = result["ratio"]
    decision = ratio["decision"]
    print(f"TRUE  hip/shoulder ratio : median {ratio['true']['median']:.2f}")
    print(f"PROXY hip/shoulder ratio : median {ratio['proxy']['median']:.2f}")
    print(f"rank correlation         : {ratio['rank_correlation']:.3f}  (high => BIASED, not noisy)")
    print(f"at the spec's {decision['cut']} cut: truth {decision['truth_fires']}/{decision['n']}, "
          f"proxy {decision['proxy_fires']}/{decision['n']}, "
          f"DISAGREE {decision['disagree']}/{decision['n']} ({decision['disagree_fraction']:.1%}) "
          f"-- {decision['proxy_only']} proxy-only, {decision['truth_only']} truth-only\n")

    for band, value in sorted(result["shoulder_width_change_per_degree_of_image_width"].items()):
        print(f"shoulder-width change per 1 deg of rotation, {band:9s}: {value:.5f} of image width")
    if args.jitter:
        for name, stats in result["mediapipe_width_jitter"].items():
            print(f"MediaPipe {name:9s} frame-to-frame step: {stats['frame_to_frame_step']:.6f} "
                  f"of image width ({stats['videos']} videos)")

    brace = result["brace_angle"]
    print("\n--- brace angle, four SIMULTANEOUS cameras (disagreement = pure projection) ---")
    print(f"  absolute angle, cross-camera spread deg : median "
          f"{brace['absolute_cross_camera_spread_deg']['median']:.1f}  "
          f"p90 {brace['absolute_cross_camera_spread_deg']['p90']:.1f}")
    print(f"  SAG the rule scores, value deg          : median {brace['sag_value_deg']['median']:.1f}")
    print(f"  SAG, cross-camera spread deg            : median "
          f"{brace['sag_cross_camera_spread_deg']['median']:.1f}  "
          f"p90 {brace['sag_cross_camera_spread_deg']['p90']:.1f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
