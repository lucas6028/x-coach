"""CLI for Fit3D experiment 3 (coaching-verdict fidelity of 2D vs NLF 3D).

Translates the cue findings (exp 1/2) into the pass/fault verdict a coaching app emits, and
asks whether a single 2D camera flips that verdict vs the mocap truth -- and whether NLF 3D
fixes it. The 2D arm is reported raw AND after oracle per-view debiasing (the calibration
upper bound), with the same debiasing applied to NLF, so the deciding comparison is fair:
after calibration, is the knee/depth verdict-flip still worse for 2D than NLF?

Run from the repository root, after the NLF kernel output is under --pred-root:

    python scripts/fit3d/run_decision_eval.py --pred-root data/Fit3D/derived/preds/nlf \
        --action squat --json data/Fit3D/derived/decision_eval_squat_nlf.json
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
from src.fit3d import decision_eval as dec  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-root", type=Path, required=True, help="dir of <subj>__<action>__<cam>.npz")
    p.add_argument("--action", default="squat")
    p.add_argument("--split", default="train")
    p.add_argument("--source", default="smpl3d", choices=["smpl3d", "smpl3d_np"])
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    result = dec.run(args.pred_root, action=args.action, split=args.split,
                     source=args.source, subjs=args.subjects, root=ds.DEFAULT_FIT3D_ROOT)
    if result["n_pairs"] == 0:
        print(f"no predictions found under {args.pred_root} for action={args.action}")
        return
    print(dec.format_report(result))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dec.to_json(result), indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
