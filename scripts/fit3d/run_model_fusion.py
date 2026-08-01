"""CLI for the Fit3D model-fusion GATE experiment.

Answers, before any fusion is built: do the direct image->3D models fail on the SAME frames?
If they do, no combination rule (averaging, weighting, routing) can help. Uses predictions
already on disk -- no new inference. Only rotation-invariant cues are compared, because the
models' body conventions / crop frames / assumed intrinsics differ and would otherwise be
mistaken for a fusion effect.

    python scripts/fit3d/run_model_fusion.py --action squat \
        --model NLF=data/Fit3D/derived/preds/nlf \
        --model HMR2=data/Fit3D/derived/preds/hmr2 \
        --model MeTRAbs=data/Fit3D/derived/preds/metrabs \
        --json data/Fit3D/derived/model_fusion_squat.json
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
from src.fit3d import model_fusion as mf  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", action="append", required=True, metavar="NAME=PRED_ROOT",
                   help="repeatable; e.g. --model NLF=data/Fit3D/derived/preds/nlf")
    p.add_argument("--action", default="squat")
    p.add_argument("--split", default="train")
    p.add_argument("--source", default="smpl3d")
    p.add_argument("--no-proj2d", action="store_true",
                   help="drop the GT-projected single-view 2D arm (kept by default as the "
                        "perfect-detector 2D reference)")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    models: dict[str, str] = {}
    for spec in args.model:
        if "=" not in spec:
            p.error(f"--model expects NAME=PRED_ROOT, got {spec!r}")
        name, root = spec.split("=", 1)
        models[name] = root

    result = mf.run(models, action=args.action, split=args.split, source=args.source,
                    root=ds.DEFAULT_FIT3D_ROOT, include_proj2d=not args.no_proj2d,
                    subjs=args.subjects)
    print(mf.format_report(result))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(mf.to_json(result), indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
