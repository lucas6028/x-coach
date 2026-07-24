"""Extract per-video H36M-17 joints over Fitness-AQA Squat clips (video-level faults).

Companion to ``run_shallow_pose_extraction.py`` (which works on single crops). Here each
labelled video is decoded at a stride, posed frame-by-frame, and saved as one npz per
video holding ``frame_idx`` (F,), ``joints`` (F, 17, C) and ``detected`` (F,). The
video-level aggregator (``src/fitness_aqa/video_features.py``) turns these into features.

MediaPipe writes two arms per pass (``mediapipe_2d`` image px + ``mediapipe_3d`` world mm)
with tracking ON, since consecutive frames of one clip are a genuine sequence. NLF (the
strong-depth arm) runs on Kaggle; its per-video output is folded in by
``ingest_squat_nlf.py``. RTMPose over full videos is ~28h on CPU and is deliberately
skipped here -- ``nlf_2d`` already serves as the strong-2D arm; run it on a GPU box if a
second 2D reference is wanted.

    .venv\\Scripts\\python.exe scripts/fitness_aqa/run_squat_video_pose.py --backend mediapipe --stride 3
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fitness_aqa import squat_dataset as sq  # noqa: E402

DEFAULT_OUT = sq.DEFAULT_SQUAT_ROOT / "derived" / "video_pose"
VIDEOS_ZIP = sq.DEFAULT_SQUAT_ROOT / "videos.zip"
LOWER_BODY_LANDMARKS = [23, 24, 25, 26, 27, 28]


def sampled_frames(cap, stride: int, max_frames: int) -> list[int]:
    import cv2

    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = list(range(0, nf, stride))
    if len(idxs) > max_frames:
        sel = np.linspace(0, len(idxs) - 1, max_frames).round().astype(int)
        idxs = [idxs[i] for i in sorted(set(sel))]
    return idxs


def iter_videos(video_ids: list[str], root: Path):
    """Yield (video_id, cv2.VideoCapture) reading straight from videos.zip via a temp file."""
    import cv2
    import tempfile

    with zipfile.ZipFile(VIDEOS_ZIP) as z:
        names = {Path(n).stem: n for n in z.namelist() if n.endswith(".mp4")}
        for vid in video_ids:
            if vid not in names:
                continue
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                tf.write(z.read(names[vid]))
                tmp = tf.name
            cap = cv2.VideoCapture(tmp)
            try:
                yield vid, cap
            finally:
                cap.release()
                Path(tmp).unlink(missing_ok=True)


def run_mediapipe(video_ids: list[str], root: Path, out_dir: Path, stride: int, max_frames: int,
                  model_complexity: int, detection_conf: float) -> None:
    import cv2
    from mediapipe.python.solutions import pose as mp_pose

    from src.fit3d import mediapipe_baseline as mpb

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for n, (vid, cap) in enumerate(iter_videos(video_ids, root), 1):
        op = out_dir / f"{vid}.npz"
        if op.exists() and op.stat().st_size > 500:
            continue
        want = set(sampled_frames(cap, stride, max_frames))
        # Fresh tracker per clip; tracking ON within the clip (it is a real sequence).
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=model_complexity,
                            smooth_landmarks=True, min_detection_confidence=detection_conf,
                            min_tracking_confidence=0.5)
        keep, j2d, j3d, det = [], [], [], []
        i = -1
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            i += 1
            if i not in want:
                continue
            res = pose.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            wl = getattr(res, "pose_world_landmarks", None)
            il = getattr(res, "pose_landmarks", None)
            keep.append(i)
            if wl is None or il is None:
                j2d.append(np.full((17, 2), np.nan)); j3d.append(np.full((17, 3), np.nan))
                det.append(False)
                continue
            h, w = fr.shape[:2]
            vis = np.array([p.visibility for p in wl.landmark], dtype=np.float64)
            world = np.array([[p.x, p.y, p.z] for p in wl.landmark], dtype=np.float64)
            img = np.array([[p.x * w, p.y * h, 0.0] for p in il.landmark], dtype=np.float64)
            j3d.append(mpb.blazepose33_to_h36m17(world[None])[0] * 1000.0)
            j2d.append(mpb.blazepose33_to_h36m17(img[None])[0][:, :2])
            det.append(True)
        pose.close()
        keep = np.asarray(keep, "int32")
        det = np.asarray(det, bool)
        np.savez_compressed(op, frame_idx=keep,
                            joints_2d=np.asarray(j2d, "float32"),
                            joints_3d=np.asarray(j3d, "float32"), detected=det)
        if n % 50 == 0:
            el = time.time() - t0
            print(f"[{n}/{len(video_ids)}] {vid} frames={len(keep)} det%={det.mean() if len(det) else 0:.2f} "
                  f"{el/60:.1f}m elapsed eta {el/n*(len(video_ids)-n)/60:.0f}m", flush=True)
    print(f"=== mediapipe video pose done ({time.time()-t0:.0f}s) ===", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="mediapipe", choices=["mediapipe"])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--detection-conf", type=float, default=0.2)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    video_ids = sorted(set(sum((sq.load_split(s) for s in sq.SPLITS), [])))
    if args.limit:
        video_ids = video_ids[:args.limit]
    print(f"{len(video_ids)} labelled videos | backend={args.backend} stride={args.stride} "
          f"-> {args.out}", flush=True)
    run_mediapipe(video_ids, sq.DEFAULT_SQUAT_ROOT, args.out, args.stride, args.max_frames,
                  args.model_complexity, args.detection_conf)


if __name__ == "__main__":
    main()
