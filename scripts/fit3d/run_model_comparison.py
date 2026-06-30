"""CLI for the Fit3D direct image->3D model comparison (NLF vs HMR2.0 vs ...).

Each model is a --pred-root of <subj>__<action>__<cam>.npz (SMPL-24 camera-frame 'smpl3d', mm).
Headlines the *mechanism* question with bias-tolerant metrics (pa_mpjpe, ez/exy pattern,
rotation-invariant cues, debiased verdict-flip), since the body conventions differ per model.

    python scripts/fit3d/run_model_comparison.py --action squat \
        --model NLF=data/Fit3D/derived/preds/nlf \
        --model HMR2=data/Fit3D/derived/preds/hmr2 \
        --json data/Fit3D/derived/model_comparison_squat.json
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
from src.fit3d import model_comparison as mc  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", action="append", required=True, metavar="NAME=PRED_ROOT",
                   help="repeatable; e.g. --model NLF=data/Fit3D/derived/preds/nlf")
    p.add_argument("--action", default="squat")
    p.add_argument("--split", default="train")
    p.add_argument("--source", default="smpl3d")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    models: dict[str, str] = {}
    for spec in args.model:
        if "=" not in spec:
            p.error(f"--model expects NAME=PRED_ROOT, got {spec!r}")
        name, root = spec.split("=", 1)
        models[name] = root

    result = mc.compare(models, action=args.action, split=args.split, source=args.source,
                        root=ds.DEFAULT_FIT3D_ROOT)
    print(mc.format_comparison(result))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
