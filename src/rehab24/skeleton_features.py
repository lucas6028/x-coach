from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.rehab24.dataset import DEFAULT_DATA_ROOT, DEFAULT_PROCESSED_ROOT, load_manifest, resolve_data_path


SUMMARY_PERCENTILES = (10, 25, 50, 75, 90)
SUMMARY_SIZE = 9
LEFT_SHOULDER = 6
RIGHT_SHOULDER = 11
LEFT_UP_LEG = 16
RIGHT_UP_LEG = 21
HIPS = 0


def frame_bounds(first_frame: int, last_frame: int, total_frames: int) -> tuple[int, int]:
    start = max(first_frame - 1, 0)
    stop = min(last_frame, total_frames)
    if stop <= start:
        raise ValueError(f"Invalid frame interval first_frame={first_frame}, last_frame={last_frame}, total={total_frames}")
    return start, stop


def distance(points: np.ndarray, a: int, b: int) -> np.ndarray:
    return np.linalg.norm(points[:, a, :] - points[:, b, :], axis=1)


def normalize_points(points: np.ndarray, root_index: int, scale_pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    coords = points.astype(np.float32, copy=False)
    centered = coords - coords[:, root_index : root_index + 1, :]

    scales = []
    for a, b in scale_pairs:
        pair_scale = distance(coords, a, b)
        pair_scale = np.where(np.isfinite(pair_scale) & (pair_scale > 1e-6), pair_scale, np.nan)
        scales.append(pair_scale)
    stacked = np.stack(scales, axis=1)
    frame_scale = np.full(stacked.shape[0], np.nan, dtype=np.float32)
    valid_rows = np.isfinite(stacked).any(axis=1)
    frame_scale[valid_rows] = np.nanmedian(stacked[valid_rows], axis=1)
    finite_frame_scale = frame_scale[np.isfinite(frame_scale)]
    fallback = float(np.median(finite_frame_scale)) if finite_frame_scale.size else 1.0
    if not np.isfinite(fallback) or fallback <= 1e-6:
        fallback = 1.0
    frame_scale = np.where(np.isfinite(frame_scale) & (frame_scale > 1e-6), frame_scale, fallback)
    return centered / frame_scale[:, None, None]


def add_velocity(points: np.ndarray) -> np.ndarray:
    velocity = np.diff(points, axis=0, prepend=points[:1])
    return np.concatenate([points, velocity], axis=2)


def summarize_time_series(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1)
    summaries = [
        np.nanmean(flat, axis=0),
        np.nanstd(flat, axis=0),
        np.nanmin(flat, axis=0),
        np.nanmax(flat, axis=0),
    ]
    summaries.extend(np.nanpercentile(flat, percentile, axis=0) for percentile in SUMMARY_PERCENTILES)
    summary = np.concatenate(summaries, axis=0).astype(np.float32, copy=False)
    return np.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0)


def extract_feature_vector(skeleton_3d: np.ndarray, skeleton_2d: np.ndarray, first_frame: int, last_frame: int) -> np.ndarray:
    total_frames = min(int(skeleton_3d.shape[0]), int(skeleton_2d.shape[0]))
    start, stop = frame_bounds(first_frame, last_frame, total_frames)
    segment_3d = skeleton_3d[start:stop, :, :3]
    segment_2d = skeleton_2d[start:stop, :, :2]

    norm_3d = add_velocity(
        normalize_points(
            segment_3d,
            root_index=HIPS,
            scale_pairs=((LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_UP_LEG, RIGHT_UP_LEG)),
        )
    )
    norm_2d = add_velocity(
        normalize_points(
            segment_2d,
            root_index=HIPS,
            scale_pairs=((LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_UP_LEG, RIGHT_UP_LEG)),
        )
    )
    return np.concatenate([summarize_time_series(norm_3d), summarize_time_series(norm_2d)], axis=0)


def feature_output_path(output_dir: Path, split: str, sample_id: str) -> Path:
    return output_dir / split / f"{sample_id}.npz"


def save_feature(path: Path, row: dict[str, str], feature: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        video_feature=feature.astype(np.float32, copy=False),
        sample_id=np.asarray(row["sample_id"]),
        video_id=np.asarray(row["video_id"]),
        exercise_id=np.asarray(row["exercise_id"]),
        person_id=np.asarray(row["person_id"]),
        camera=np.asarray(row["camera"]),
        correctness=np.asarray(int(row["correctness"]), dtype=np.int64),
        first_frame=np.asarray(int(row["first_frame"]), dtype=np.int64),
        last_frame=np.asarray(int(row["last_frame"]), dtype=np.int64),
    )


def iter_rows(rows: Sequence[dict[str, str]], limit: int | None) -> Iterable[dict[str, str]]:
    if limit is None:
        yield from rows
        return
    yield from rows[:limit]


def extract_features_for_manifest(
    data_root: Path,
    manifest_path: Path,
    output_dir: Path,
    limit: int | None = None,
    overwrite: bool = False,
) -> int:
    rows = load_manifest(manifest_path)
    written = 0
    for index, row in enumerate(iter_rows(rows, limit), start=1):
        output_path = feature_output_path(output_dir, row["split"], row["sample_id"])
        if output_path.exists() and not overwrite:
            continue

        skeleton_3d = np.load(resolve_data_path(data_root, row["skeleton_3d_path"]), mmap_mode="r")
        skeleton_2d = np.load(resolve_data_path(data_root, row["skeleton_2d_path"]), mmap_mode="r")
        feature = extract_feature_vector(
            skeleton_3d=skeleton_3d,
            skeleton_2d=skeleton_2d,
            first_frame=int(row["first_frame"]),
            last_frame=int(row["last_frame"]),
        )
        save_feature(output_path, row, feature)
        written += 1
        if index % 100 == 0:
            print(f"Processed {index} manifest rows...")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract REHAB24-6 repetition-level skeleton features.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "skeleton_features")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    written = extract_features_for_manifest(
        data_root=args.data_root,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"Wrote {written} feature files under {args.output_dir}")


if __name__ == "__main__":
    main()
