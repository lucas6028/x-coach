"""Check that a variant arm's features actually differ from the arm it modifies.

    .venv\\Scripts\\python.exe scripts/video/verify_variant_took_effect.py \\
        --arm data/Fitness-AQA/Squat/Labeled_Dataset/full_frame_letterbox/videomae_mean_pool_fc_norm_mean \\
        --reference data/Fitness-AQA/Squat/Labeled_Dataset/videomae_mean_pool_fc_norm_mean \\
        --expect-unchanged square

This exists because of a measured failure: half of one Stage B control arm was
extracted from UNTRANSFORMED video and nothing failed -- the numbers just meant
something other than what they claimed. The check is cheap and it is not optional.

Two subtleties, both learned the hard way:

*Tolerance, not equality.* Identical pixels do NOT give bit-identical features across
machines or library versions -- a square video re-extracted locally matched its Kaggle
features at cos 1.0000 with ``allclose`` false. Demanding equality would condemn a
correct arm.

*Zero is the wrong expectation for some arms.* ``full_frame_letterbox`` pads a frame to
square, which is a legitimate no-op on the 768 of 1623 Fitness-AQA squat videos that
are already square. Their ``full_frame`` input was never centre-cropped, so there is
nothing to restore. ``--expect-unchanged square`` states that up front so the run is
checked against the number it should produce, not against zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.squat_dataset import SQUAT_LABELED_ROOT

#: Below this the two features are the same vector to within cross-machine noise.
DEFAULT_TOLERANCE = 1e-3


def load_feature(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["video_feature"], dtype=np.float64)


def index_features(feature_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(feature_dir.rglob("*.npz")):
        index.setdefault(path.stem, path)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a variant arm's transform reached its features.")
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--expect-unchanged",
        choices=("none", "square"),
        default="none",
        help="'square': the arm is a no-op on already-square videos, so they must be unchanged.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SQUAT_LABELED_ROOT / "videos_person_crop" / "manifest.json",
        help="Supplies each video's frame size for the 'square' expectation.",
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    arm = index_features(args.arm)
    reference = index_features(args.reference)
    shared = sorted(set(arm) & set(reference))
    if not shared:
        raise SystemExit(f"No video ids are shared between {args.arm} and {args.reference}.")
    if len(shared) != len(arm) or len(shared) != len(reference):
        raise SystemExit(
            f"Coverage differs: arm has {len(arm)}, reference has {len(reference)}, {len(shared)} shared. "
            "A partial comparison cannot verify the transform."
        )

    square: set[str] = set()
    if args.expect_unchanged == "square":
        with args.manifest.open("r", encoding="utf-8") as f:
            for row in json.load(f)["rows"]:
                width, height = row["frame_size"]
                if width == height:
                    square.add(str(row["video_id"]))

    unchanged: list[str] = []
    changed: list[str] = []
    worst_unchanged = 0.0
    for video_id in shared:
        difference = float(np.abs(load_feature(arm[video_id]) - load_feature(reference[video_id])).max())
        if difference <= args.tolerance:
            unchanged.append(video_id)
            worst_unchanged = max(worst_unchanged, difference)
        else:
            changed.append(video_id)

    print(f"{len(shared)} videos compared, tolerance {args.tolerance:g}")
    print(f"  changed  : {len(changed)}")
    print(f"  unchanged: {len(unchanged)} (largest difference among them {worst_unchanged:.2e})")

    if args.expect_unchanged == "none":
        if unchanged:
            raise SystemExit(
                f"!! {len(unchanged)} videos are unchanged ({unchanged[:5]}). This arm transforms every "
                "video, so an unchanged one was extracted from untransformed input."
            )
        print("PASS: every video was transformed.")
        return

    unexpected_unchanged = sorted(set(unchanged) - square)
    unexpected_changed = sorted(set(changed) & square)
    print(f"  expected unchanged (already square): {len(square)}")

    if unexpected_unchanged or unexpected_changed:
        if unexpected_unchanged:
            print(f"!! {len(unexpected_unchanged)} non-square videos were NOT transformed ({unexpected_unchanged[:5]})")
        if unexpected_changed:
            print(f"!! {len(unexpected_changed)} square videos WERE transformed ({unexpected_changed[:5]})")
        raise SystemExit(1)

    print(f"PASS: unchanged set is exactly the {len(square)} already-square videos.")


if __name__ == "__main__":
    main()
