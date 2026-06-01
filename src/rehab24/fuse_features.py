from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.rehab24.dataset import DEFAULT_PROCESSED_ROOT, load_manifest


def iter_rows(rows: Sequence[dict[str, str]], limit: int | None) -> Iterable[dict[str, str]]:
    if limit is None:
        yield from rows
        return
    yield from rows[:limit]


def feature_path(feature_dir: Path, row: dict[str, str]) -> Path:
    return feature_dir / row["split"] / f"{row['sample_id']}.npz"


def fuse_feature_files(first_path: Path, second_path: Path) -> np.ndarray:
    with np.load(first_path, allow_pickle=False) as first, np.load(second_path, allow_pickle=False) as second:
        return np.concatenate(
            [first["video_feature"].astype(np.float32), second["video_feature"].astype(np.float32)],
            axis=0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate two REHAB24-6 feature directories for fusion baselines.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--first-feature-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "skeleton_features")
    parser.add_argument("--second-feature-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "videomae_features")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "fused_features")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    written = 0
    missing = 0
    for row in iter_rows(rows, args.limit):
        output_path = feature_path(args.output_dir, row)
        if output_path.exists() and not args.overwrite:
            continue
        first_path = feature_path(args.first_feature_dir, row)
        second_path = feature_path(args.second_feature_dir, row)
        if not first_path.exists() or not second_path.exists():
            missing += 1
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            video_feature=fuse_feature_files(first_path, second_path),
            sample_id=np.asarray(row["sample_id"]),
            exercise_id=np.asarray(row["exercise_id"]),
            person_id=np.asarray(row["person_id"]),
            camera=np.asarray(row["camera"]),
            correctness=np.asarray(int(row["correctness"]), dtype=np.int64),
        )
        written += 1

    if missing:
        print(f"Skipped {missing} rows with missing feature files.")
    print(f"Wrote {written} fused feature files under {args.output_dir}")


if __name__ == "__main__":
    main()

