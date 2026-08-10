"""Audit a VideoMAE feature dir before it is trusted by a LOSO run.

Every failure mode listed here has a matching way of quietly corrupting a result
rather than crashing: partial coverage silently shrinks a fold, a duplicate stem
makes ``build_samples`` (which takes the first ``rglob`` hit) pick an arbitrary
split's copy, a stale bundle mixed into a fresh dir fuses two extractions, and
NaN/Inf turn a fold's metrics into nonsense that still prints as a number.

Exits non-zero when any check fails, so it can gate the LOSO step.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.rehab24.dataset import DEFAULT_PROCESSED_ROOT, load_manifest


def audit_feature_dir(feature_dir: Path, manifest_path: Path, labels_path: Path | None = None) -> dict:
    """Collect coverage, integrity and provenance findings for one feature dir."""
    rows = load_manifest(manifest_path)
    expected_split = {row["sample_id"]: row["split"] for row in rows}

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
            split_mismatch.append(f"{stem}: on disk in {path.parent.name!r}, manifest says {expected_split[stem]!r}")

    found = set(stems_to_paths)
    expected = set(expected_split)
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    duplicates = sorted(stem for stem, stem_paths in stems_to_paths.items() if len(stem_paths) > 1)

    unlabeled: list[str] = []
    if labels_path is not None and labels_path.exists():
        labels = json.load(labels_path.open())
        unlabeled = sorted(found - set(labels))

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
    print(f"  coverage       : {report['n_found']}/{report['n_expected']} manifest samples")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit VideoMAE feature dirs for coverage and integrity.")
    parser.add_argument("feature_dirs", nargs="+", type=Path, help="One or more feature dirs to audit.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--labels", type=Path, default=DEFAULT_PROCESSED_ROOT / "labels" / "correctness.json")
    parser.add_argument("--report-output", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    reports = []
    for feature_dir in args.feature_dirs:
        report = audit_feature_dir(feature_dir, args.manifest, args.labels)
        print_report(report)
        reports.append(report)

    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        with args.report_output.open("w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2, sort_keys=True)
        print(f"\nSaved audit report to {args.report_output}")

    failed = [r["feature_dir"] for r in reports if not r["passed"]]
    if failed:
        raise SystemExit(f"\n{len(failed)} feature dir(s) FAILED audit: {', '.join(failed)}")
    print(f"\nAll {len(reports)} feature dir(s) passed audit.")


if __name__ == "__main__":
    main()
