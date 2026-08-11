"""Concatenate two feature dirs into one, for the early-fusion (secondary) arm.

Stage B's diagnostic fusion: the plain feature concat that Stage A already measured
on REHAB24-6, where it converged to the stronger branch on 3/3 skeleton backbones
without ever beating it. It is run here to see whether that reproduces on a different
dataset and task, not to decide the retention question -- `late_fusion` is primary.

Scale is handled downstream by the classifier's ``--normalize-features``, which fits
per-dimension statistics on the training fold only. Capacity is not: whichever branch
brings more dimensions brings more parameters, and that imbalance is a property of
concat itself, which is exactly what the primary arm avoids.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def concat_features(first_path: Path, second_path: Path) -> np.ndarray:
    with np.load(first_path, allow_pickle=False) as first, np.load(second_path, allow_pickle=False) as second:
        return np.concatenate(
            [first["video_feature"].astype(np.float32), second["video_feature"].astype(np.float32)],
            axis=0,
        )


def fuse_feature_dirs(
    first_dir: Path,
    second_dir: Path,
    output_dir: Path,
    split_map: dict[str, str],
    overwrite: bool = False,
) -> dict[str, int]:
    """Write ``<output_dir>/<split>/<video_id>.npz`` for every id present in both dirs.

    Ids missing from either side are counted and returned rather than skipped
    silently: a fusion arm quietly covering fewer videos than its baselines is not a
    paired comparison, and the caller is expected to refuse a nonzero count.
    """
    written = 0
    missing: list[str] = []
    for video_id, split_name in sorted(split_map.items()):
        first_path = first_dir / split_name / f"{video_id}.npz"
        second_path = second_dir / split_name / f"{video_id}.npz"
        if not first_path.exists() or not second_path.exists():
            missing.append(video_id)
            continue

        output_path = output_dir / split_name / f"{video_id}.npz"
        if output_path.exists() and not overwrite:
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            video_feature=concat_features(first_path, second_path),
            video_id=np.asarray(video_id),
            split=np.asarray(split_name),
            provenance_first_dir=np.asarray(str(first_dir)),
            provenance_second_dir=np.asarray(str(second_dir)),
        )
        written += 1

    return {"written": written, "missing": len(missing), "missing_ids": missing}
