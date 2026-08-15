"""Split the box-geometry control into its clip-length term and its geometry terms.

    .venv\\Scripts\\python.exe scripts/video/build_box_geometry_ablations.py

The 12-number control was assumed to work because it encodes the athlete's coarse
geometry. It does not. One of the twelve is ``n_frames``, the clip's length, and it
carries the whole thing:

  n_frames alone (1 number)      0.6139   66.8% of full_frame's above-chance signal
  box_geometry (all 12)          0.6120   65.6%
  box_geometry minus n_frames    0.5447   26.2%

Adding the eleven geometry terms to clip length is worth -0.0020, CI [-0.0227,
+0.0194]. This script writes the two ablation feature dirs so that result can be
re-run. See ``notes/videomae_b1_repeated_splits_results.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.box_geometry import FEATURE_NAMES
from src.video.squat_dataset import SQUAT_LABELED_ROOT

LENGTH_TERM = "n_frames"


def write_subset(source: Path, output: Path, keep: list[int]) -> int:
    written = 0
    for path in sorted(source.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            feature = np.asarray(data["video_feature"])[keep].astype(np.float32)
            destination = output / path.parent.name / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination,
                video_feature=feature,
                video_id=data["video_id"],
                split=data["split"],
            )
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the clip-length / geometry ablations.")
    parser.add_argument("--source", type=Path, default=SQUAT_LABELED_ROOT / "box_geometry_features")
    parser.add_argument("--output-parent", type=Path, default=SQUAT_LABELED_ROOT)
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"{args.source} does not exist.")

    length_index = FEATURE_NAMES.index(LENGTH_TERM)
    others = [index for index in range(len(FEATURE_NAMES)) if index != length_index]

    for name, keep in (
        ("box_geometry_nframes_only", [length_index]),
        ("box_geometry_no_nframes", others),
    ):
        count = write_subset(args.source, args.output_parent / name, keep)
        print(f"{name}: {count} features, dim {len(keep)}")


if __name__ == "__main__":
    main()
