"""CLI for the Fit3D 2D-vs-3D-vs-mocap cue-error decomposition.

Shows, per squat cue, the reading error vs mocap-3D truth for real-2D (RTMPose), mocap-2D (GT
projected = perfect detector), and the direct image->3D models -- decomposing the 2D error into
its detector and projection components. Run after RTMPose extraction (run_rtmpose_fit3d.py) and
the 3D preds are in place.

    python scripts/fit3d/run_twod_vs_threed.py --action squat \
        --rtmpose-root data/Fit3D/derived/preds/rtmpose \
        --model NLF=data/Fit3D/derived/preds/nlf \
        --model HMR2=data/Fit3D/derived/preds/hmr2 \
        --model MultiHMR=data/Fit3D/derived/preds/multihmr \
        --json data/Fit3D/derived/twod_vs_threed_squat.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fit3d import dataset as ds  # noqa: E402
from src.fit3d import twod_vs_threed as tt  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", default="squat")
    p.add_argument("--split", default="train")
    p.add_argument("--rtmpose-root", type=Path, default=PROJECT_ROOT / "data" / "Fit3D" / "derived" / "preds" / "rtmpose")
    p.add_argument("--model", action="append", default=[], metavar="NAME=PRED_ROOT",
                   help="repeatable 3D model; e.g. --model NLF=data/Fit3D/derived/preds/nlf")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    models: dict[str, str] = {}
    for spec in args.model:
        name, root = spec.split("=", 1)
        models[name] = root

    result = tt.compare(args.action, args.rtmpose_root, models, split=args.split,
                        subjs=args.subjects, root=ds.DEFAULT_FIT3D_ROOT)
    if result["twod"]["n_seq"] == 0:
        print(f"no RTMPose npz found under {args.rtmpose_root} for action={args.action}")
        return
    print(tt.format_comparison(result))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
