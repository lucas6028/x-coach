"""The duration-only floor: one number per repetition, no pixels and no geometry.

``notes/rehab24_videomae_framing_validation_plan.md`` §2 makes this a precondition for
reading ``background_only`` rather than an optional extra. A background arm that scores
above chance means nothing on its own -- the question is whether it beats what you get
from knowing *how long the repetition is*, and nothing else. On Fitness-AQA that floor
turned out to be most of the story: clip length alone reached 0.6139 balanced accuracy,
66.8% of ``full_frame``'s above-chance signal, and it is what eventually explained most
of that dataset's background-only score (``notes/videomae_b1_repeated_splits_results.md``).

REHAB24-6 has a reason to differ -- one lab, two fixed cameras, so "scene" cannot be
scene *recognition* here -- which is exactly why the floor has to be measured rather
than assumed to transfer.

The feature is *sliced out of* the existing ``box_geometry`` control rather than
recomputed from the manifest. The plan allows either ("新增一維 duration-only control,
或從既有 box control 拆出"), and slicing is the safer of the two: the two floors are then
provably the same number, so ``background_only − n_frames`` and
``background_only − box_geometry`` differ only by the eleven geometry terms. Recomputing
would leave any discrepancy between the floors ambiguous -- a different frame-range
convention would look like a finding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.rehab24.dataset import DEFAULT_PROCESSED_ROOT
from src.video.box_geometry import FEATURE_NAMES

#: The single term kept. Index rather than name because the archived .npz files store a
#: bare array whose column order is fixed by ``FEATURE_NAMES``.
FEATURE_NAME = "n_frames"
FEATURE_INDEX = FEATURE_NAMES.index(FEATURE_NAME)

#: Everything except ``video_feature`` is copied through, so the derived dir stays
#: self-describing for the same stratified analysis as every other arm.
CARRIED_KEYS = (
    "sample_id",
    "video_id",
    "exercise_id",
    "person_id",
    "camera",
    "correctness",
    "first_frame",
    "last_frame",
)


def derive_bundle(source_path: Path, output_dir: Path) -> Path:
    """Write the 1-dim duration bundle for one sample, preserving its split."""
    with np.load(source_path, allow_pickle=False) as data:
        feature = np.asarray(data["video_feature"])
        if feature.shape != (len(FEATURE_NAMES),):
            raise ValueError(
                f"{source_path.name}: expected a {len(FEATURE_NAMES)}-dim box-geometry feature, got {feature.shape}."
            )
        carried = {key: data[key] for key in CARRIED_KEYS if key in data.files}

    destination = output_dir / source_path.parent.name / source_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(
            handle,
            video_feature=feature[FEATURE_INDEX : FEATURE_INDEX + 1].astype(np.float32),
            **carried,
            provenance_variant=np.asarray("n_frames"),
            provenance_derived_from=np.asarray("box_geometry_features"),
            provenance_feature_names=np.asarray(FEATURE_NAME),
        )
    partial.replace(destination)
    return destination


def derive_all(box_dir: Path, output_dir: Path) -> int:
    sources = sorted(box_dir.rglob("*.npz"))
    if not sources:
        raise SystemExit(f"No box-geometry bundles under {box_dir}. Run build_box_geometry_features.py first.")
    for source in sources:
        derive_bundle(source, output_dir)
    return len(sources)


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive the duration-only (n_frames) control from the box control.")
    parser.add_argument("--box-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "box_geometry_features")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "n_frames_features")
    args = parser.parse_args()

    count = derive_all(args.box_dir, args.output_dir)
    print(f"Derived {count} duration-only bundles ({FEATURE_NAME}, dim 1) under {args.output_dir}")

    values = np.array(
        [float(np.load(p, allow_pickle=False)["video_feature"][0]) for p in sorted(args.output_dir.rglob("*.npz"))]
    )
    print(f"  n_frames: min {values.min():.0f}  median {np.median(values):.0f}  max {values.max():.0f}")
    if float(values.std()) == 0.0:
        raise SystemExit("Every repetition has the same length; a duration control would be a constant feature.")


if __name__ == "__main__":
    main()
