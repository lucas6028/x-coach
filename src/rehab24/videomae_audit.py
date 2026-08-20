"""REHAB24-6 CLI over the shared VideoMAE feature audit.

The checks live in ``src.video.videomae_audit`` because Fitness-AQA runs the same
gate; REHAB24-6 differs only in where the expected sample->split mapping and the
label set come from (a manifest CSV and a correctness JSON).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rehab24.dataset import DEFAULT_PROCESSED_ROOT, load_manifest
from src.video.videomae_audit import (
    audit_feature_dir as _audit_feature_dir,
    print_report,
    raise_on_failure,
    write_reports,
)

__all__ = ["audit_feature_dir", "print_report", "main"]


def audit_feature_dir(feature_dir: Path, manifest_path: Path, labels_path: Path | None = None) -> dict:
    """Audit one feature dir against the REHAB24-6 manifest and correctness labels."""
    rows = load_manifest(manifest_path)
    expected_split = {row["sample_id"]: row["split"] for row in rows}

    labeled_ids = None
    if labels_path is not None and labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
            labeled_ids = {str(key) for key in json.load(f)}

    return _audit_feature_dir(feature_dir, expected_split, labeled_ids)


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

    write_reports(reports, args.report_output)
    raise_on_failure(reports)


if __name__ == "__main__":
    main()
