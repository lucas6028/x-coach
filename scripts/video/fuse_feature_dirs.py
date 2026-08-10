"""Thin CLI entry point for the early-fusion feature concat (Fitness-AQA squat)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.early_fusion import fuse_feature_dirs
from src.video.squat_dataset import SQUAT_LABELED_ROOT, load_split_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate two feature dirs for the early-fusion arm.")
    parser.add_argument("--first-feature-dir", type=Path, required=True)
    parser.add_argument("--second-feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=SQUAT_LABELED_ROOT / "Splits")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split_map = load_split_map(args.split_dir)
    result = fuse_feature_dirs(
        first_dir=args.first_feature_dir,
        second_dir=args.second_feature_dir,
        output_dir=args.output_dir,
        split_map=split_map,
        overwrite=args.overwrite,
    )

    print(f"wrote {result['written']} fused bundles under {args.output_dir}")
    if result["missing"]:
        preview = ", ".join(result["missing_ids"][:10])
        raise SystemExit(
            f"{result['missing']} of {len(split_map)} ids were missing from one of the two dirs "
            f"({preview}). A fusion arm covering fewer videos than its baselines is not a paired "
            "comparison -- fix the coverage before training."
        )


if __name__ == "__main__":
    main()
