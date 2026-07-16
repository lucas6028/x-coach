"""Extract MediaPipe (BlazePose GHUM) 3D on Fit3D videos -> H36M-17 camera-frame preds.

The **sparse + weak-depth** arm of the depth-bottleneck study, the lightweight counterpart
to MeTRAbs (sparse + metric depth) and the dense SMPL regressors (NLF/HMR2.0/Multi-HMR).
MediaPipe Pose runs on CPU per frame and emits ``pose_world_landmarks`` (metric metres,
mid-hip origin, roughly image-aligned axes). We map BlazePose-33 -> H36M-17
(``src/fit3d/mediapipe_baseline.py``), convert metres -> mm, and save per video as
``joints_cam`` (F, 17, 3) so it drops straight into ``depth_eval`` / ``decision_eval`` /
``model_comparison`` (the eval resolves L/R against the GT via ``resolve_lr_h36m17``).

MediaPipe is fast enough to run **every** frame, keeping pred frame i aligned with GT frame i
(the eval matches by index). Undetected frames stay NaN and are ignored.

    .venv\\Scripts\\python.exe -m pip install mediapipe
    .venv\\Scripts\\python.exe scripts/fit3d/run_mediapipe_fit3d.py \
        --actions squat deadlift overhead_extension_thruster \
        --out data/Fit3D/derived/preds/mediapipe
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from src.fit3d import dataset as ds  # noqa: E402
from src.fit3d import mediapipe_baseline as mpb  # noqa: E402


def video_path(subj: str, action: str, cam: str, split: str, root: Path) -> Path:
    return root / split / subj / "videos" / cam / f"{action}.mp4"


def process_video(vpath: Path, pose, subsample: int, min_vis: float) -> tuple[np.ndarray, int, int]:
    """Run MediaPipe over a video -> (F, 17, 3) H36M-17 camera-frame mm (NaN where no/low-conf).

    ``pose_world_landmarks`` are metric metres (mid-hip origin); we scale to mm. A landmark
    below ``min_vis`` visibility is NaN'd so it does not feed the biomech cues as a phantom.
    """
    cap = cv2.VideoCapture(str(vpath))
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = np.full((nf, 17, 3), np.nan, np.float32)
    i, ndet = -1, 0
    while True:
        ok, fr = cap.read()
        if not ok or i + 1 >= nf:
            break
        i += 1
        if i % subsample != 0:
            continue
        res = pose.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        wl = getattr(res, "pose_world_landmarks", None)
        if wl is None:
            continue
        lm = np.array([[p.x, p.y, p.z] for p in wl.landmark], dtype=np.float64)  # metres
        vis = np.array([p.visibility for p in wl.landmark], dtype=np.float64)
        lm[vis < min_vis] = np.nan
        out[i] = (mpb.blazepose33_to_h36m17(lm[None])[0] * 1000.0).astype(np.float32)  # m -> mm
        ndet += 1
    cap.release()
    return out, ndet, nf


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actions", nargs="+", default=["squat", "deadlift", "overhead_extension_thruster"])
    p.add_argument("--split", default="train")
    p.add_argument("--subsample", type=int, default=1, help="process every k-th frame (1 = all)")
    p.add_argument("--model-complexity", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--min-detection-confidence", type=float, default=0.5)
    p.add_argument("--min-visibility", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "Fit3D" / "derived" / "preds" / "mediapipe")
    p.add_argument("--subjects", nargs="*", default=None)
    args = p.parse_args()

    from mediapipe.python.solutions import pose as mp_pose  # lazy: heavy optional dep

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for subj in args.subjects or ds.subjects(args.split, ds.DEFAULT_FIT3D_ROOT):
        avail = ds.actions(args.split, subj, ds.DEFAULT_FIT3D_ROOT)
        for action in args.actions:
            if action not in avail:
                continue
            for cam in ds.cameras(args.split, subj, ds.DEFAULT_FIT3D_ROOT):
                jobs.append((subj, action, cam))
    print(f"{len(jobs)} videos | complexity={args.model_complexity} subsample={args.subsample} "
          f"-> {args.out}", flush=True)

    # static_image_mode=False lets MediaPipe track across frames (smoother 3D on video).
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=args.model_complexity,
                        smooth_landmarks=True, enable_segmentation=False,
                        min_detection_confidence=args.min_detection_confidence,
                        min_tracking_confidence=0.5)
    t0 = 0.0
    for n, (subj, action, cam) in enumerate(jobs, 1):
        op = args.out / mpb.pred_npz_name(subj, action, cam)
        if op.exists() and op.stat().st_size > 1000:
            print(f"[{n}/{len(jobs)}] skip {op.name}", flush=True)
            continue
        vpath = video_path(subj, action, cam, args.split, ds.DEFAULT_FIT3D_ROOT)
        if not vpath.exists():
            print(f"[{n}/{len(jobs)}] MISSING {vpath}", flush=True)
            continue
        t = time.time()
        if t0 == 0.0:
            t0 = t
        joints_cam, ndet, nf = process_video(vpath, pose, args.subsample, args.min_visibility)
        np.savez_compressed(op, joints_cam=joints_cam, ndet=np.asarray(ndet),
                            subject=np.asarray(subj), action=np.asarray(action), camera=np.asarray(cam),
                            subsample=np.asarray(args.subsample),
                            model_complexity=np.asarray(args.model_complexity), units=np.asarray("mm"))
        el = (time.time() - t0) / 60
        det = float(ndet) / max(1, nf // max(1, args.subsample))
        print(f"[{n}/{len(jobs)}] {op.name} frames{nf} det%={det:.2f} {time.time()-t:.0f}s "
              f"(elapsed {el:.0f}m eta {el/n*(len(jobs)-n):.0f}m)", flush=True)
    pose.close()
    print("=== MediaPipe Fit3D extraction done ===", flush=True)


if __name__ == "__main__":
    main()
