"""Build REHAB24-6 repetition features from NLF direct image->3D joints.

Stage 3 of the depth experiment (see ``notes/rehab24_correctness_experiment_summary.md``
and the ``lift_2d_to_3d*`` modules): monocular 2D->3D *lifting* could not recover the
out-of-plane depth that separates the best 2D-only path (~0.63) from true Vicon mocap
(0.702). This asks the orthogonal question: does a **direct image->3D** model -- NLF
(Neural Localizer Fields, NeurIPS'24), which regresses metric 3D straight from pixels --
capture real depth that lifting could not, and so beat MediaPipe pseudo-3D (0.633)?

The expensive NLF inference runs once on Kaggle GPU (``.kaggle_tmp/nlf_extract`` ->
``detect_smpl_batched`` at half resolution, ~104 ms/frame) and is saved as RAW per-video
SMPL-24 joints under ``--raw-dir``. This module is the cheap, local, label-blind step that
turns those joints into the same repetition-feature format the correctness LOSO consumes,
so the only thing that varies versus the lifting/mocap stages is the 3D *source*.

Per-video raw ``.npz`` (written by the kernel, one per ``video_path``, name =
``video_path`` with ``/`` -> ``__`` and the extension dropped) holds, NaN where a frame
was outside every rep range or had no detection:

* ``smpl3d``     (F, 24, 3)  parametric SMPL-24 joints, millimetres, camera frame.
* ``smpl3d_np``  (F, 24, 3)  nonparametric (localizer-field) readout at the same joints.
* ``smpl2d``     (F, 24, 2)  image-space joints (half-res pixels; scale-invariant here).
* ``unc``        (F, 24)     per-joint uncertainty;  ``ndet`` (F,) detections/frame.

SMPL-24 order: pelv, lhip, rhip, spi1, lkne, rkne, spi2, lank, rank, spi3, ltoe, rtoe,
neck, lcla, rcla, head, lsho, rsho, lelb, relb, lwri, rwri, lhan, rhan. The features are a
different joint layout than the H36M-17 lifting features (a documented caveat -- the
scientific quantity is the cross-stage LOSO ranking, not a same-dim swap).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.rehab24.dataset import DEFAULT_PROCESSED_ROOT, load_manifest
from src.rehab24.mediapipe_skeleton_features import interpolate_missing
from src.rehab24.skeleton_features import (
    add_velocity,
    feature_output_path,
    frame_bounds,
    normalize_points,
    save_feature,
    summarize_time_series,
)

# SMPL-24 indices for normalization (mirrors the shoulder-span / hip-span convention used
# across the other skeleton feature builders).
SMPL_ROOT = 0            # pelvis
SMPL_LSHO, SMPL_RSHO = 16, 17
SMPL_LHIP, SMPL_RHIP = 1, 2
SMPL_SCALE_PAIRS = ((SMPL_LSHO, SMPL_RSHO), (SMPL_LHIP, SMPL_RHIP))
SMPL_NECK = 12           # for diagnostics only (pelvis-neck bone length)

SOURCE_KEY = {"parametric": "smpl3d", "nonparam": "smpl3d_np"}


def raw_npz_name(video_path: str) -> str:
    """Per-video raw filename written by the extraction kernel (must match it exactly)."""
    return video_path.replace("/", "__").replace("\\", "__").rsplit(".", 1)[0] + ".npz"


def load_video_joints(raw_dir: Path, video_path: str, source: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (joints3d (F,24,3), joints2d (F,24,2)) for one source video."""
    with np.load(raw_dir / raw_npz_name(video_path)) as data:
        j3 = data[SOURCE_KEY[source]].astype(np.float32)
        j2 = data["smpl2d"].astype(np.float32)
    return j3, j2


def repetition_feature(j3: np.ndarray, j2: np.ndarray, first_frame: int, last_frame: int, mode: str) -> np.ndarray:
    """3D (+optional 2D) repetition feature: slice -> interpolate -> normalize -> velocity -> summary."""
    total = min(int(j3.shape[0]), int(j2.shape[0]))
    start, stop = frame_bounds(first_frame, last_frame, total)
    seg3 = interpolate_missing(j3[start:stop, :, :3])
    block3d = summarize_time_series(add_velocity(normalize_points(seg3, SMPL_ROOT, SMPL_SCALE_PAIRS)))
    if mode == "3d":
        return block3d
    seg2 = interpolate_missing(j2[start:stop, :, :2])
    block2d = summarize_time_series(add_velocity(normalize_points(seg2, SMPL_ROOT, SMPL_SCALE_PAIRS)))
    return np.concatenate([block3d, block2d], axis=0)


def build_features(raw_dir: Path, manifest_path: Path, output_dir: Path, source: str, mode: str, overwrite: bool) -> tuple[int, list[str]]:
    rows = load_manifest(manifest_path)
    rows_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_video[row["video_path"]].append(row)

    written = 0
    missing_videos: list[str] = []
    for video_path, vid_rows in rows_by_video.items():
        npz_path = raw_dir / raw_npz_name(video_path)
        if not npz_path.exists():
            missing_videos.append(video_path)
            continue
        # only load (decode) the video once if any of its reps still need writing
        pending = [r for r in vid_rows if overwrite or not feature_output_path(output_dir, r["split"], r["sample_id"]).exists()]
        if not pending:
            continue
        j3, j2 = load_video_joints(raw_dir, video_path, source)
        for row in pending:
            feat = repetition_feature(j3, j2, int(row["first_frame"]), int(row["last_frame"]), mode)
            save_feature(feature_output_path(output_dir, row["split"], row["sample_id"]), row, feat)
            written += 1
    return written, missing_videos


def main() -> None:
    parser = argparse.ArgumentParser(description="Build REHAB24-6 repetition features from NLF SMPL-24 image->3D joints.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "nlf_raw3d",
                        help="Directory of per-video raw NLF npz from the Kaggle extraction kernel.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to nlf_<source>_<mode>_skeleton_features under the processed root.")
    parser.add_argument("--source", choices=("parametric", "nonparam"), default="parametric",
                        help="parametric = SMPL-fit joints3d (stable); nonparam = localizer-field readout.")
    parser.add_argument("--mode", choices=("3d2d", "3d"), default="3d2d",
                        help="3d2d = 3D + image-2D blocks (default); 3d = depth-bearing 3D block only.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or DEFAULT_PROCESSED_ROOT / f"nlf_{args.source}_{args.mode}_skeleton_features"
    if not args.raw_dir.exists():
        raise SystemExit(f"raw-dir not found: {args.raw_dir}. Download + merge the Kaggle nlf_raw3d chunks first.")

    written, missing = build_features(args.raw_dir, args.manifest, output_dir, args.source, args.mode, args.overwrite)
    print(f"Wrote {written} NLF feature files ({args.source}/{args.mode}) -> {output_dir}")
    n_videos = len({r['video_path'] for r in load_manifest(args.manifest)})
    print(f"Source videos: {n_videos - len(missing)}/{n_videos} present" + (f" | MISSING {len(missing)}" if missing else ""))
    if missing:
        preview = ", ".join(missing[:5])
        print(f"  missing raw npz (e.g. {preview}{'...' if len(missing) > 5 else ''}) -- chunk(s) not yet downloaded?")

    meta = {"raw_dir": str(args.raw_dir), "output_dir": str(output_dir), "source": args.source,
            "mode": args.mode, "n_written": written, "n_missing_videos": len(missing)}
    (output_dir / "_build_meta.json").parent.mkdir(parents=True, exist_ok=True)
    with (output_dir / "_build_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
