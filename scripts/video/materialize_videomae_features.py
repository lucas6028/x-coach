"""Thin CLI entry point for deriving classifier-ready Fitness-AQA VideoMAE dirs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.squat_dataset import SQUAT_LABELED_ROOT
from src.video.videomae_materialize import materialize_all
from src.video.videomae_pooling import CLIP_AGGREGATIONS, TOKEN_POOLING_MODES

#: Fitness-AQA raw bundles carry no per-subject metadata -- the video id and its
#: split are all that ride along into the materialized dirs.
CARRIED_KEYS = ("video_id", "split", "video_path")


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive Fitness-AQA VideoMAE feature dirs from raw per-clip bundles.")
    parser.add_argument("--raw-dir", type=Path, default=SQUAT_LABELED_ROOT / "videomae_raw")
    parser.add_argument("--output-parent", type=Path, default=SQUAT_LABELED_ROOT)
    parser.add_argument(
        "--token-pooling",
        nargs="+",
        choices=TOKEN_POOLING_MODES,
        default=list(TOKEN_POOLING_MODES),
    )
    parser.add_argument("--aggregation", nargs="+", choices=CLIP_AGGREGATIONS, default=list(CLIP_AGGREGATIONS))
    args = parser.parse_args()

    counts = materialize_all(
        raw_dir=args.raw_dir,
        output_parent=args.output_parent,
        carried_keys=CARRIED_KEYS,
        token_poolings=tuple(args.token_pooling),
        aggregations=tuple(args.aggregation),
    )
    print(f"\nMaterialized {len(counts)} feature dirs from {args.raw_dir}")


if __name__ == "__main__":
    main()
