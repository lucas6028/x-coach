"""Build REHAB24-6's zero-parameter box-geometry control features.

    .venv\\Scripts\\python.exe scripts/rehab24/build_box_geometry_features.py

No pose estimator and no GPU: the boxes come from the dataset's own mocap-derived 2D
skeletons. See ``src/rehab24/box_geometry_features.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rehab24.box_geometry_features import build_feature, read_manifest, save_feature

DEFAULT_DATASET_ROOT = ROOT / "data" / "REHAB24-6"
DEFAULT_PROCESSED_ROOT = DEFAULT_DATASET_ROOT / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build REHAB24-6 box-geometry control features.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "box_geometry_features")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    print(f"{len(rows)} samples in {args.manifest}")

    size_cache: dict[str, tuple[int, int]] = {}
    written = 0
    skipped = 0
    failures: list[str] = []

    for index, row in enumerate(rows, start=1):
        destination = args.output_dir / row["split"] / f"{row['sample_id']}.npz"
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            feature = build_feature(row, args.dataset_root, size_cache)
        except Exception as exc:  # noqa: BLE001 - report every failure, do not abort the set
            failures.append(f"{row['sample_id']}: {type(exc).__name__}: {exc}")
            continue
        save_feature(args.output_dir, row, feature)
        written += 1
        if index % 250 == 0:
            print(f"  [{index}/{len(rows)}] wrote {written}, skipped {skipped}")

    print(f"Wrote {written} features ({skipped} already present) under {args.output_dir}")
    print(f"Frame sizes seen: {sorted(set(size_cache.values()))}")

    if failures:
        preview = "\n  ".join(failures[:10])
        raise SystemExit(
            f"{len(failures)} samples failed:\n  {preview}\n"
            "A control arm covering fewer samples than the arms it is compared against "
            "is not a paired comparison."
        )


if __name__ == "__main__":
    main()
