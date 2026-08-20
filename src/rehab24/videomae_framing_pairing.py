"""Check that two REHAB24-6 framing arms are actually paired, sample for sample.

Plan §6.2. A paired LOSO delta is only a measurement of framing if the two arms differ
in framing and in nothing else, and every way that can fail here fails quietly:

* an arm short of 2144 bundles trains on a smaller set and still prints a number;
* a repetition whose clip starts drifted compares different frames under one label;
* an arm that silently fell through to the untouched video produces features
  bit-identical to its baseline, which reads as "the manipulation did nothing";
* two arms extracted with different clip lengths or models are simply not comparable,
  and only the provenance stamp records it.

The last check is the one with teeth. Both REHAB24-6 cameras are non-square, so
``full_frame_letterbox`` must change every frame it is given; a single sample whose
features match ``full_frame`` bit-for-bit means the transform did not run for it. On
Fitness-AQA the same check has to tolerate 768 legitimately-identical square videos --
here the tolerance is zero, which is why this module is REHAB24-specific.

Comparison is on the raw per-clip bundles rather than the materialized dirs, because
that is where clip starts and provenance live; the materialized dirs are a pure
function of them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from src.rehab24.dataset import DEFAULT_PROCESSED_ROOT, load_manifest
from src.video.videomae_pooling import MEAN_POOL_FC_NORM

#: Fields that describe the repetition rather than the pixels. Every one must survive a
#: framing change untouched -- a variant that renames or re-splits samples has broken
#: the pairing regardless of what its features look like.
PAIRED_KEYS = (
    "sample_id",
    "video_id",
    "exercise_id",
    "person_id",
    "camera",
    "correctness",
    "clip_starts",
    "first_frame",
    "last_frame",
    "total_frames",
)


def index_bundles(feature_dir: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted(feature_dir.rglob("*.npz"))}


def read_bundle(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        record = {key: data[key] for key in PAIRED_KEYS if key in data.files}
        record["_features"] = data[f"clip_features_{MEAN_POOL_FC_NORM}"]
        record["_split"] = path.parent.name
        record["_provenance"] = json.dumps(
            {key[len("provenance_") :]: str(data[key]) for key in data.files if key.startswith("provenance_")},
            sort_keys=True,
        )
    return record


def compare_raw_dirs(
    baseline_dir: Path,
    candidate_dir: Path,
    expected_split: Mapping[str, str] | None = None,
    require_different_features: bool = True,
) -> dict:
    """Every §6.2 finding for one candidate arm against its paired baseline."""
    baseline_paths = index_bundles(baseline_dir)
    candidate_paths = index_bundles(candidate_dir)

    shared = sorted(set(baseline_paths) & set(candidate_paths))
    missing_from_candidate = sorted(set(baseline_paths) - set(candidate_paths))
    extra_in_candidate = sorted(set(candidate_paths) - set(baseline_paths))

    metadata_mismatch: list[str] = []
    clip_start_mismatch: list[str] = []
    split_mismatch: list[str] = []
    identical_features: list[str] = []
    nonfinite: list[str] = []
    dims: set[tuple[int, ...]] = set()
    dtypes: set[str] = set()
    provenances: dict[str, set[str]] = {"baseline": set(), "candidate": set()}

    for stem in shared:
        base = read_bundle(baseline_paths[stem])
        cand = read_bundle(candidate_paths[stem])
        provenances["baseline"].add(base["_provenance"])
        provenances["candidate"].add(cand["_provenance"])

        if base["_split"] != cand["_split"]:
            split_mismatch.append(f"{stem}: {base['_split']} vs {cand['_split']}")
        for key in PAIRED_KEYS:
            if key not in base or key not in cand:
                continue
            if not np.array_equal(base[key], cand[key]):
                target = clip_start_mismatch if key == "clip_starts" else metadata_mismatch
                target.append(f"{stem}.{key}: {base[key]} vs {cand[key]}")

        features = cand["_features"]
        dims.add(tuple(features.shape))
        dtypes.add(str(features.dtype))
        if not np.all(np.isfinite(features)):
            nonfinite.append(stem)
        if features.shape == base["_features"].shape and np.array_equal(features, base["_features"]):
            identical_features.append(stem)

    expected_ids = set(expected_split) if expected_split is not None else set(baseline_paths)
    checks = {
        "baseline_is_complete": not (expected_ids - set(baseline_paths)),
        "candidate_is_complete": not (expected_ids - set(candidate_paths)),
        "same_sample_ids": not missing_from_candidate and not extra_in_candidate,
        "same_splits": not split_mismatch,
        "same_clip_starts": not clip_start_mismatch,
        "same_repetition_metadata": not metadata_mismatch,
        "single_feature_shape": len(dims) <= 1,
        "single_dtype": len(dtypes) <= 1,
        "all_finite": not nonfinite,
        "single_provenance_per_arm": all(len(values) <= 1 for values in provenances.values()),
        "arms_declare_different_variants": provenances["baseline"] != provenances["candidate"],
    }
    if require_different_features:
        # Zero tolerance: both REHAB24-6 cameras are non-square, so there is no video
        # for which a framing transform is legitimately a no-op.
        checks["features_actually_differ"] = not identical_features

    return {
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "n_expected": len(expected_ids),
        "n_baseline": len(baseline_paths),
        "n_candidate": len(candidate_paths),
        "n_compared": len(shared),
        "feature_shapes": sorted(str(dim) for dim in dims),
        "dtypes": sorted(dtypes),
        "provenance": {name: sorted(values) for name, values in provenances.items()},
        "missing_from_candidate": missing_from_candidate,
        "extra_in_candidate": extra_in_candidate,
        "split_mismatch": split_mismatch,
        "clip_start_mismatch": clip_start_mismatch,
        "metadata_mismatch": metadata_mismatch,
        "identical_features": identical_features,
        "nonfinite": nonfinite,
        "checks": checks,
        "passed": all(checks.values()),
    }


def print_report(report: dict) -> None:
    print(f"\n=== pairing: {Path(report['candidate_dir']).name} vs {Path(report['baseline_dir']).name} ===")
    print(f"  coverage    : baseline {report['n_baseline']}, candidate {report['n_candidate']}, expected {report['n_expected']}")
    print(f"  compared    : {report['n_compared']} samples, shapes {report['feature_shapes']}, dtypes {report['dtypes']}")
    for name, values in report["provenance"].items():
        for value in values:
            print(f"  provenance  : {name}: {value}")

    for name, ok in report["checks"].items():
        if ok:
            continue
        detail = report.get(
            {
                "same_sample_ids": "missing_from_candidate",
                "same_splits": "split_mismatch",
                "same_clip_starts": "clip_start_mismatch",
                "same_repetition_metadata": "metadata_mismatch",
                "features_actually_differ": "identical_features",
                "all_finite": "nonfinite",
            }.get(name, ""),
            [],
        )
        preview = ", ".join(map(str, detail[:5]))
        suffix = f" ... (+{len(detail) - 5} more)" if len(detail) > 5 else ""
        print(f"  FAIL {name}" + (f": {preview}{suffix}" if detail else ""))

    print(f"  => {'PASS' if report['passed'] else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that two REHAB24-6 VideoMAE framing arms are paired.")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--allow-identical-features",
        action="store_true",
        help="Only for an identity arm. Never for a framing arm on REHAB24-6.",
    )
    args = parser.parse_args()

    expected_split = {row["sample_id"]: row["split"] for row in load_manifest(args.manifest)}
    report = compare_raw_dirs(
        args.baseline_dir,
        args.candidate_dir,
        expected_split,
        require_different_features=not args.allow_identical_features,
    )
    print_report(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"\nSaved pairing report to {args.output}")

    if not report["passed"]:
        failed = [name for name, ok in report["checks"].items() if not ok]
        raise SystemExit(f"Pairing gate FAILED: {', '.join(failed)}. Do not train a classifier on these arms.")


if __name__ == "__main__":
    main()
