"""Pretrained 2D->3D lifting (VideoPose3D) for REHAB24-6 squat-correctness.

Stages 1-2 trained a small dataset-specific TCN lifter. Stage 1 (MediaPipe) showed a
lifter can mimic BlazePose's *pseudo*-depth (which is itself 2D-derivable); Stage 2
(Vicon) showed the same lifter cannot recover Vicon's *true* mocap depth from one view
(``lifted3d_vicon`` 0.566 ~ ``vicon2d`` 0.583 << real Vicon 0.702). This stage asks the
natural follow-up: does a **strong lifter with a learned 3D prior**, pretrained on
massive H36M mocap (VideoPose3D), recover more genuine depth than our small TCN?

Pipeline (offline, CPU): cached MediaPipe image landmarks ``(T,33,2)`` -> remap to
COCO-17 (the layout VideoPose3D's ``pretrained_h36m_detectron_coco`` expects) -> screen-
normalize with each video's real resolution -> run VideoPose3D (243-frame TCN) -> H36M-17
3D ``(T,17,3)`` -> rebuild skeleton features and run the correctness LOSO.

Controlled comparison (same MediaPipe 2D source, only the 3D block varies):

* ``vp3d_2d``     (612)  — COCO-17 image 2D only.                 [lower bound]
* ``vp3d_lifted`` (1530) — same 2D + VideoPose3D-lifted H36M 3D.  [the experiment]

Compare the ``vp3d_lifted`` LOSO against our own TCN ``lifted3d`` (0.621), MediaPipe
pseudo-3D (0.633), and the Vicon results. As with stages 1-2 the lifter is label-blind,
so the downstream LOSO stays a clean generalization estimate. Feature dims differ across
joint layouts (documented caveat); the scientific quantity is the lifted-vs-2D delta
*within* this stage plus the cross-stage ranking.

Requires the cloned VideoPose3D repo + pretrained weights under ``third_party/VideoPose3D``
(see the Stage 3 setup in the rehab24 README / notes).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.rehab24.dataset import DEFAULT_DATA_ROOT, DEFAULT_PROCESSED_ROOT, load_manifest, resolve_data_path
from src.rehab24.mediapipe_skeleton_features import interpolate_missing
from src.rehab24.skeleton_features import (
    add_velocity,
    distance,
    feature_output_path,
    frame_bounds,
    normalize_points,
    save_feature,
    summarize_time_series,
)

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Stage 3 pretrained lifting needs `opencv-python` to read video resolutions.") from exc
try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Stage 3 pretrained lifting requires `torch`.") from exc


# MediaPipe-33 -> COCO-17 (Detectron order: nose, eyes, ears, shoulders, elbows,
# wrists, hips, knees, ankles). VideoPose3D's detectron_coco model takes this order.
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# COCO-17 indices used for the 2D normalization root/scale.
COCO_LSHO, COCO_RSHO, COCO_LHIP, COCO_RHIP = 5, 6, 11, 12
# H36M-17 indices for the lifted-3D normalization (0 = pelvis root).
H36M_ROOT = 0
H36M_SCALE_PAIRS = ((11, 14), (1, 4))  # (LShoulder,RShoulder), (RHip,LHip)
VP3D_FILTER_WIDTHS = (3, 3, 3, 3, 3)


def load_videopose3d(repo_dir: Path, weights: Path, device: "torch.device"):
    """Import the cloned VideoPose3D, build the model and load pretrained weights."""
    import sys

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from common.model import TemporalModel  # type: ignore

    model = TemporalModel(
        17, 2, 17, filter_widths=list(VP3D_FILTER_WIDTHS), causal=False, dropout=0.25, channels=1024
    ).to(device)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_pos"])
    model.eval()
    return model


def video_resolution(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        raise SystemExit(f"Could not read resolution for {path}")
    return w, h


def normalize_screen(coco_frac: np.ndarray, w: int, h: int) -> np.ndarray:
    """Fractional [0,1] image coords -> VideoPose3D screen coords (X/w*2 - [1, h/w])."""
    px = coco_frac * np.array([w, h], dtype=np.float32)
    return (px / w * 2 - np.array([1.0, h / w], dtype=np.float32)).astype(np.float32)


def lift_clip(model, coco_screen: np.ndarray, device: "torch.device") -> np.ndarray:
    """Run VideoPose3D on one clip's screen-normalized COCO-17 2D -> H36M-17 3D (T,17,3)."""
    pad = (model.receptive_field() - 1) // 2
    padded = np.pad(coco_screen, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    with torch.no_grad():
        out = model(torch.from_numpy(padded[None]).to(device))
    return out.squeeze(0).cpu().numpy().astype(np.float32)  # (T, 17, 3)


def normalize_coco2d(points: np.ndarray) -> np.ndarray:
    """Mid-hip-centered, shoulder/hip-span-scaled COCO-17 2D (mirrors normalize_points)."""
    coords = points.astype(np.float32, copy=False)
    root = (coords[:, COCO_LHIP] + coords[:, COCO_RHIP]) / 2.0  # (T, 2)
    centered = coords - root[:, None, :]
    sho = distance(coords, COCO_LSHO, COCO_RSHO)
    hip = distance(coords, COCO_LHIP, COCO_RHIP)
    stacked = np.stack([sho, hip], axis=1)
    stacked = np.where(np.isfinite(stacked) & (stacked > 1e-6), stacked, np.nan)
    scale = np.full(stacked.shape[0], np.nan, dtype=np.float32)
    valid = np.isfinite(stacked).any(axis=1)
    scale[valid] = np.nanmedian(stacked[valid], axis=1)
    finite = scale[np.isfinite(scale)]
    fallback = float(np.median(finite)) if finite.size else 1.0
    if not np.isfinite(fallback) or fallback <= 1e-6:
        fallback = 1.0
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fallback)
    return centered / scale[:, None, None]


class ClipLift:
    """One source video's lifted H36M-17 3D + normalized COCO-17 2D, ready for features."""

    __slots__ = ("video_path", "lifted3d", "coco2d")

    def __init__(self, video_path: str, lifted3d: np.ndarray, coco2d: np.ndarray) -> None:
        self.video_path = video_path
        self.lifted3d = lifted3d  # (T, 17, 3) H36M, lifted
        self.coco2d = coco2d  # (T, 17, 2) frac, COCO order


def landmark_cache_path(cache_dir: Path, video_path: str) -> Path:
    return cache_dir / (Path(video_path).stem + ".npz")


def build_clip_lifts(manifest_path: Path, cache_dir: Path, data_root: Path, model, device) -> dict[str, ClipLift]:
    rows = load_manifest(manifest_path)
    video_of_clip: dict[str, str] = {}
    for row in rows:
        video_of_clip.setdefault(row["video_path"], row["video_path"])

    lifts: dict[str, ClipLift] = {}
    missing: list[str] = []
    for i, video_path in enumerate(dict.fromkeys(video_of_clip), start=1):
        cache_path = landmark_cache_path(cache_dir, video_path)
        if not cache_path.exists():
            missing.append(video_path)
            continue
        with np.load(cache_path) as data:
            image = data["image"].astype(np.float32)  # (T, 33, 2) frac
        coco = image[:, MP_TO_COCO, :]  # (T, 17, 2)
        coco = interpolate_missing(coco)  # fill missed detections along time
        coco = np.nan_to_num(coco, nan=0.5)  # any all-NaN channel -> image center
        w, h = video_resolution(resolve_data_path(data_root, video_path))
        screen = normalize_screen(coco, w, h)
        lifted = lift_clip(model, screen, device)
        lifts[video_path] = ClipLift(video_path, lifted, coco)
        if i % 25 == 0:
            print(f"  lifted {i} videos...", flush=True)
    if missing:
        raise SystemExit(
            f"{len(missing)} videos have no MediaPipe landmark cache (e.g. {missing[0]}). "
            f"Build it first via extract_mediapipe_skeleton_features --landmark-cache {cache_dir}"
        )
    return lifts


def _summarize_segment(points_norm: np.ndarray) -> np.ndarray:
    return summarize_time_series(add_velocity(points_norm))


def vp3d_lifted_feature(lift: ClipLift, first_frame: int, last_frame: int) -> np.ndarray:
    """1530-d: H36M-17 lifted-3D block (918) + COCO-17 2D block (612)."""
    total = min(lift.lifted3d.shape[0], lift.coco2d.shape[0])
    start, stop = frame_bounds(first_frame, last_frame, total)
    block3d = _summarize_segment(
        normalize_points(lift.lifted3d[start:stop, :, :3], root_index=H36M_ROOT, scale_pairs=H36M_SCALE_PAIRS)
    )
    block2d = _summarize_segment(normalize_coco2d(lift.coco2d[start:stop, :, :2]))
    return np.concatenate([block3d, block2d], axis=0)


def vp3d_2d_feature(lift: ClipLift, first_frame: int, last_frame: int) -> np.ndarray:
    """612-d COCO-17 2D-only feature (depth-free lower bound, identical 2D source)."""
    total = lift.coco2d.shape[0]
    start, stop = frame_bounds(first_frame, last_frame, total)
    return _summarize_segment(normalize_coco2d(lift.coco2d[start:stop, :, :2]))


def write_features(manifest_path: Path, lifts: dict[str, ClipLift], lifted_dir: Path, twod_dir: Path, overwrite: bool) -> tuple[int, int]:
    rows = load_manifest(manifest_path)
    rows_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_video[row["video_path"]].append(row)

    n_lift = n_2d = 0
    for video_path, vid_rows in rows_by_video.items():
        lift = lifts[video_path]
        for row in vid_rows:
            first, last = int(row["first_frame"]), int(row["last_frame"])
            lp = feature_output_path(lifted_dir, row["split"], row["sample_id"])
            if overwrite or not lp.exists():
                save_feature(lp, row, vp3d_lifted_feature(lift, first, last))
                n_lift += 1
            tp = feature_output_path(twod_dir, row["split"], row["sample_id"])
            if overwrite or not tp.exists():
                save_feature(tp, row, vp3d_2d_feature(lift, first, last))
                n_2d += 1
    return n_lift, n_2d


def main() -> None:
    parser = argparse.ArgumentParser(description="Lift REHAB24-6 MediaPipe 2D to 3D with pretrained VideoPose3D.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "mediapipe_landmarks_cache")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--repo-dir", type=Path, default=Path("third_party/VideoPose3D"))
    parser.add_argument(
        "--weights", type=Path, default=Path("third_party/VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin")
    )
    parser.add_argument("--lifted-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "vp3d_lifted_skeleton_features")
    parser.add_argument("--twod-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "vp3d_2d_skeleton_features")
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_PROCESSED_ROOT / "lift_2d_to_3d_vp3d_metrics.json")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or auto.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    print(f"VideoPose3D lifting on {device}")

    model = load_videopose3d(args.repo_dir, args.weights, device)
    print(f"Loaded VideoPose3D (receptive field {model.receptive_field()})")

    lifts = build_clip_lifts(args.manifest, args.cache_dir, args.data_root, model, device)
    print(f"Lifted {len(lifts)} source videos")

    n_lift, n_2d = write_features(args.manifest, lifts, args.lifted_dir, args.twod_dir, args.overwrite)
    print(f"Wrote {n_lift} vp3d_lifted features -> {args.lifted_dir}")
    print(f"Wrote {n_2d} vp3d_2d features -> {args.twod_dir}")

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", encoding="utf-8") as f:
        json.dump(
            {"n_videos": len(lifts), "n_lifted_features": n_lift, "n_2d_features": n_2d, "model": "videopose3d_h36m_detectron_coco"},
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Saved metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()
