"""Audit a VideoMAE feature dir before it is trusted by a training run.

Every failure mode listed here has a matching way of quietly corrupting a result
rather than crashing: partial coverage silently shrinks a split, a duplicate stem
makes ``build_samples`` pick an arbitrary split's copy, a stale bundle mixed into a
fresh dir fuses two extractions, and NaN/Inf turn a run's metrics into nonsense that
still prints as a number.

The core takes an ``expected_split`` mapping rather than a dataset file, so REHAB24-6
(manifest-driven) and Fitness-AQA (split-key JSONs) share one implementation.

Exits non-zero when any check fails, so it can gate the training step.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np

from src.video.squat_dataset import SQUAT_LABELED_ROOT, load_labeled_ids, load_split_map


def audit_feature_dir(
    feature_dir: Path,
    expected_split: Mapping[str, str],
    labeled_ids: set[str] | None = None,
) -> dict:
    """Collect coverage, integrity and provenance findings for one feature dir."""
    paths = sorted(feature_dir.rglob("*.npz"))
    stems_to_paths: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        stems_to_paths[path.stem].append(path)

    dims: Counter[int] = Counter()
    dtypes: Counter[str] = Counter()
    provenances: Counter[str] = Counter()
    clip_counts: Counter[int] = Counter()
    nonfinite: list[str] = []
    split_mismatch: list[str] = []
    unreadable: list[str] = []
    constant_features: list[str] = []

    for stem, stem_paths in stems_to_paths.items():
        path = stem_paths[0]
        try:
            with np.load(path, allow_pickle=False) as data:
                if "video_feature" not in data.files:
                    unreadable.append(f"{stem}: no `video_feature` key")
                    continue
                feature = data["video_feature"]
                dims[int(feature.shape[0])] += 1
                dtypes[str(feature.dtype)] += 1
                if not np.all(np.isfinite(feature)):
                    nonfinite.append(stem)
                if float(np.std(feature)) == 0.0:
                    constant_features.append(stem)
                if "clip_features" in data.files:
                    clip_counts[int(data["clip_features"].shape[0])] += 1
                provenance = {
                    key[len("provenance_") :]: str(data[key]) for key in data.files if key.startswith("provenance_")
                }
                provenances[json.dumps(provenance, sort_keys=True)] += 1
        except Exception as exc:  # noqa: BLE001 - a corrupt npz must be reported, not raised
            unreadable.append(f"{stem}: {type(exc).__name__}: {exc}")
            continue

        if stem in expected_split and path.parent.name != expected_split[stem]:
            split_mismatch.append(f"{stem}: on disk in {path.parent.name!r}, expected {expected_split[stem]!r}")

    found = set(stems_to_paths)
    expected = set(expected_split)
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    duplicates = sorted(stem for stem, stem_paths in stems_to_paths.items() if len(stem_paths) > 1)

    unlabeled: list[str] = []
    if labeled_ids is not None:
        unlabeled = sorted(found - labeled_ids)

    checks = {
        "coverage_complete": not missing,
        "no_unexpected_samples": not unexpected,
        "no_duplicate_stems": not duplicates,
        "single_feature_dim": len(dims) <= 1,
        "single_dtype": len(dtypes) <= 1,
        "single_clip_count": len(clip_counts) <= 1,
        "all_finite": not nonfinite,
        "no_constant_features": not constant_features,
        "splits_match_manifest": not split_mismatch,
        "all_readable": not unreadable,
        "single_provenance": len(provenances) <= 1,
        "all_labeled": not unlabeled,
    }

    return {
        "feature_dir": str(feature_dir),
        "n_expected": len(expected),
        "n_found": len(found),
        "feature_dims": dict(dims),
        "dtypes": dict(dtypes),
        "clip_counts": dict(clip_counts),
        "provenance_variants": [json.loads(key) for key in provenances],
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": duplicates,
        "nonfinite": nonfinite,
        "constant_features": constant_features,
        "split_mismatch": split_mismatch,
        "unreadable": unreadable,
        "unlabeled": unlabeled,
        "checks": checks,
        "passed": all(checks.values()),
    }


def print_report(report: dict) -> None:
    print(f"\n=== audit: {Path(report['feature_dir']).name} ===")
    print(f"  coverage       : {report['n_found']}/{report['n_expected']} expected samples")
    print(f"  feature dims   : {report['feature_dims']}")
    print(f"  dtypes         : {report['dtypes']}")
    print(f"  clips/sample   : {report['clip_counts']}")
    for variant in report["provenance_variants"]:
        print(f"  provenance     : {variant}")

    for name, ok in report["checks"].items():
        if not ok:
            detail_key = {
                "coverage_complete": "missing",
                "no_unexpected_samples": "unexpected",
                "no_duplicate_stems": "duplicates",
                "all_finite": "nonfinite",
                "no_constant_features": "constant_features",
                "splits_match_manifest": "split_mismatch",
                "all_readable": "unreadable",
                "all_labeled": "unlabeled",
            }.get(name)
            detail = report.get(detail_key, []) if detail_key else []
            preview = ", ".join(map(str, detail[:5]))
            suffix = f" ... (+{len(detail) - 5} more)" if len(detail) > 5 else ""
            print(f"  FAIL {name}" + (f": {preview}{suffix}" if detail else ""))

    print(f"  => {'PASS' if report['passed'] else 'FAIL'}")


def write_reports(reports: list[dict], report_output: Path | None) -> None:
    if report_output is None:
        return
    report_output.parent.mkdir(parents=True, exist_ok=True)
    with report_output.open("w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, sort_keys=True)
    print(f"\nSaved audit report to {report_output}")


def raise_on_failure(reports: list[dict]) -> None:
    failed = [r["feature_dir"] for r in reports if not r["passed"]]
    if failed:
        raise SystemExit(f"\n{len(failed)} feature dir(s) FAILED audit: {', '.join(failed)}")
    print(f"\nAll {len(reports)} feature dir(s) passed audit.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit VideoMAE feature dirs against split-key JSONs (Fitness-AQA layout)."
    )
    parser.add_argument("feature_dirs", nargs="+", type=Path, help="One or more feature dirs to audit.")
    parser.add_argument("--split-dir", type=Path, default=SQUAT_LABELED_ROOT / "Splits")
    parser.add_argument("--labels-dir", type=Path, default=SQUAT_LABELED_ROOT / "Labels")
    parser.add_argument("--report-output", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    expected_split = load_split_map(args.split_dir)
    labeled_ids = load_labeled_ids(args.labels_dir)

    reports = []
    for feature_dir in args.feature_dirs:
        report = audit_feature_dir(feature_dir, expected_split, labeled_ids or None)
        print_report(report)
        reports.append(report)

    write_reports(reports, args.report_output)
    raise_on_failure(reports)


if __name__ == "__main__":
    main()
