"""Extract RTMPose (real 2D-detector) keypoints on Fit3D videos -> Fit3D-25 pixel keypoints.

The *real-2D* arm of the 2D-vs-3D comparison. RTMPose (rtmlib Wholebody, COCO-WholeBody 133) runs
on each Fit3D video; we keep the COCO-17 body subset mapped into the Fit3D-25 biomech slots (pixels)
via ``src/fit3d/twod_baseline.py``. Per-frame cue error is unbiased under subsampling, so we infer
every ``--subsample`` frame to keep the CPU run tractable (rtmlib balanced ~1.5 s/frame on CPU).

    source .venv/bin/activate  # needs: pip install rtmlib onnxruntime
    python scripts/fit3d/run_rtmpose_fit3d.py --actions squat deadlift overhead_extension_thruster \
        --mode balanced --subsample 15 --out data/Fit3D/derived/preds/rtmpose
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
from src.fit3d import twod_baseline as tb  # noqa: E402


def video_path(subj: str, action: str, cam: str, split: str, root: Path) -> Path:
    return root / split / subj / "videos" / cam / f"{action}.mp4"


def process_video(vpath: Path, detector, subsample: int, score_thr: float) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    cap = cv2.VideoCapture(str(vpath))
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    kp2d = np.full((nf, ds.NUM_JOINTS, 2), np.nan, np.float32)
    ndet = np.zeros(nf, np.int16)
    i, nseen = -1, 0
    while True:
        ok, fr = cap.read()
        if not ok or i + 1 >= nf:
            break
        i += 1
        if i % subsample != 0:
            continue
        nseen += 1
        kps, scs = detector(fr)
        if kps is None or len(kps) == 0:
            continue
        best = int(np.argmax([float(np.nanmean(s)) if np.size(s) else 0.0 for s in scs]))
        ndet[i] = 1
        kp2d[i] = tb.coco_wholebody_to_fit3d25(np.asarray(kps[best]), np.asarray(scs[best]), score_thr)
    cap.release()
    return kp2d, ndet, nseen, w, h


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actions", nargs="+", default=["squat", "deadlift", "overhead_extension_thruster"])
    p.add_argument("--split", default="train")
    p.add_argument("--mode", default="balanced", choices=["performance", "balanced", "lightweight"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--subsample", type=int, default=15)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "Fit3D" / "derived" / "preds" / "rtmpose")
    p.add_argument("--subjects", nargs="*", default=None)
    args = p.parse_args()

    from rtmlib import Wholebody  # lazy: heavy optional dep
    detector = Wholebody(to_openpose=False, mode=args.mode, backend="onnxruntime",
                         device="cuda" if args.device.startswith("cuda") else args.device)

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for subj in args.subjects or ds.subjects(args.split, ds.DEFAULT_FIT3D_ROOT):
        avail = ds.actions(args.split, subj, ds.DEFAULT_FIT3D_ROOT)
        for action in args.actions:
            if action not in avail:
                continue
            for cam in ds.cameras(args.split, subj, ds.DEFAULT_FIT3D_ROOT):
                jobs.append((subj, action, cam))
    print(f"{len(jobs)} videos | mode={args.mode} subsample={args.subsample} -> {args.out}", flush=True)

    t0 = 0.0
    for n, (subj, action, cam) in enumerate(jobs, 1):
        op = args.out / tb.pred_npz_name(subj, action, cam)
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
        kp2d, ndet, nseen, w, h = process_video(vpath, detector, args.subsample, args.score_thr)
        np.savez_compressed(op, kp2d=kp2d, ndet=ndet, width=np.asarray(w), height=np.asarray(h),
                            subject=np.asarray(subj), action=np.asarray(action), camera=np.asarray(cam),
                            subsample=np.asarray(args.subsample), mode=np.asarray(args.mode))
        el = (time.time() - t0) / 60
        det = float(ndet.sum()) / max(1, nseen)
        print(f"[{n}/{len(jobs)}] {op.name} inf{nseen} det%={det:.2f} {time.time()-t:.0f}s "
              f"(elapsed {el:.0f}m eta {el/n*(len(jobs)-n):.0f}m)", flush=True)
    print("=== RTMPose Fit3D extraction done ===", flush=True)


if __name__ == "__main__":
    main()
