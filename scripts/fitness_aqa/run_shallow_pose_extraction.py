"""Extract per-arm H36M-17 joints for the Fitness-AQA shallow-squat crops.

One arm = one pose source in one coordinate space. Every arm is written as an npz with
``sample_ids`` (N,), ``joints`` (N, 17, C) and ``detected`` (N,), aligned row-for-row on
the manifest order, so the classifier can intersect valid rows across arms and give all
of them byte-identical inputs.

    .venv\\Scripts\\python.exe scripts/fitness_aqa/run_shallow_pose_extraction.py --backend mediapipe
    .venv\\Scripts\\python.exe scripts/fitness_aqa/run_shallow_pose_extraction.py --backend rtmpose

``mediapipe`` writes two arms in one pass -- ``mediapipe_2d`` (image pixels, depth gone)
and ``mediapipe_3d`` (BlazePose world landmarks, heuristic z) -- which is the
same-detector +/- depth pair for the weak-depth model. ``rtmpose`` writes a real
2D-detector reference arm. NLF (the strong-depth pair) needs a GPU and lives in
``run_shallow_nlf_extraction.py``.
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

from src.fitness_aqa import shallow_dataset as sd  # noqa: E402

DEFAULT_OUT = sd.DEFAULT_SHALLOW_ROOT / "derived" / "pose"


def save_arm(out_dir: Path, name: str, sample_ids: list[str], joints: np.ndarray,
             detected: np.ndarray, **extra) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npz"
    np.savez_compressed(path, sample_ids=np.asarray(sample_ids), joints=joints.astype(np.float32),
                        detected=detected.astype(bool), **extra)
    det = float(detected.mean()) if len(detected) else 0.0
    print(f"wrote {path}  {joints.shape}  detected={det:.3f}", flush=True)
    return path


LOWER_BODY_LANDMARKS = [23, 24, 25, 26, 27, 28]  # BlazePose hips, knees, ankles


def run_mediapipe(sample_ids: list[str], root: Path, out_dir: Path, min_visibility: float,
                  model_complexity: int, detection_conf: float) -> None:
    from mediapipe.python.solutions import pose as mp_pose  # lazy: heavy optional dep

    from src.fit3d import mediapipe_baseline as mpb

    n = len(sample_ids)
    index = {sid: i for i, sid in enumerate(sample_ids)}
    j2d = np.full((n, 17, 2), np.nan, np.float32)
    j3d = np.full((n, 17, 3), np.nan, np.float32)
    detected = np.zeros(n, bool)
    lower_vis = np.full(n, np.nan, np.float32)

    # static_image_mode=True: the crops are independent frames from different clips, so
    # cross-frame tracking would carry a pose from one athlete onto the next.
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=model_complexity,
                        enable_segmentation=False, min_detection_confidence=detection_conf)
    import cv2

    t0 = time.time()
    for k, (sid, img) in enumerate(sd.iter_crops(sample_ids, root), 1):
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        wl = getattr(res, "pose_world_landmarks", None)
        il = getattr(res, "pose_landmarks", None)
        if wl is None or il is None:
            continue
        i = index[sid]
        h, w = img.shape[:2]
        vis = np.array([p.visibility for p in wl.landmark], dtype=np.float64)
        # Gym crops are cluttered (racks, plates, bystanders): MediaPipe reports
        # visibility < 0.5 on the knees/ankles for about half of them. Masking on that
        # would drop half the dataset non-randomly, so visibility is stored as a
        # diagnostic (see --min-visibility for the stricter sensitivity run) and the
        # default keeps every landmark the detector emitted -- which is also what a
        # deployed pipeline would act on.
        lower_vis[i] = float(np.min(vis[LOWER_BODY_LANDMARKS]))

        world = np.array([[p.x, p.y, p.z] for p in wl.landmark], dtype=np.float64)
        world[vis < min_visibility] = np.nan
        j3d[i] = (mpb.blazepose33_to_h36m17(world[None])[0] * 1000.0).astype(np.float32)  # m -> mm

        # Pixels, not normalised coords: x/w and y/h scale the axes anisotropically and
        # would distort every angle the cue features read.
        img_lm = np.array([[p.x * w, p.y * h, 0.0] for p in il.landmark], dtype=np.float64)
        img_lm[vis < min_visibility] = np.nan
        j2d[i] = mpb.blazepose33_to_h36m17(img_lm[None])[0][:, :2].astype(np.float32)
        detected[i] = True
        if k % 500 == 0:
            print(f"  {k}/{n} crops  {time.time() - t0:.0f}s", flush=True)
    pose.close()

    save_arm(out_dir, "mediapipe_2d", sample_ids, j2d, detected,
             space=np.asarray("image2d"), units=np.asarray("pixels"), lower_body_visibility=lower_vis)
    save_arm(out_dir, "mediapipe_3d", sample_ids, j3d, detected,
             space=np.asarray("cam3d"), units=np.asarray("mm"), lower_body_visibility=lower_vis)


def run_rtmpose(sample_ids: list[str], root: Path, out_dir: Path, mode: str, device: str,
                score_thr: float) -> None:
    from rtmlib import Wholebody  # lazy: heavy optional dep

    from src.fit3d import twod_baseline as tb

    detector = Wholebody(to_openpose=False, mode=mode, backend="onnxruntime",
                         device="cuda" if device.startswith("cuda") else device)
    n = len(sample_ids)
    index = {sid: i for i, sid in enumerate(sample_ids)}
    j2d = np.full((n, 17, 2), np.nan, np.float32)
    detected = np.zeros(n, bool)

    t0 = time.time()
    for k, (sid, img) in enumerate(sd.iter_crops(sample_ids, root), 1):
        keypoints, scores = detector(img)
        if keypoints is None or len(keypoints) == 0:
            continue
        # Largest bounding box = the lifter, not a bystander or a mirror reflection.
        areas = [np.ptp(kp[:17, 0]) * np.ptp(kp[:17, 1]) for kp in keypoints]
        best = int(np.argmax(areas))
        j2d[index[sid]] = tb.coco_wholebody_to_fit3d25(keypoints[best], scores[best], score_thr)[:17, :2]
        detected[index[sid]] = True
        if k % 200 == 0:
            print(f"  {k}/{n} crops  {time.time() - t0:.0f}s", flush=True)

    save_arm(out_dir, "rtmpose_2d", sample_ids, j2d, detected,
             space=np.asarray("image2d"), units=np.asarray("pixels"), mode=np.asarray(mode))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True, choices=["mediapipe", "rtmpose"])
    p.add_argument("--root", type=Path, default=sd.DEFAULT_SHALLOW_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=0, help="process only the first N samples (smoke test)")
    p.add_argument("--min-visibility", type=float, default=0.0,
                   help="mediapipe landmark visibility floor (0 = keep every emitted landmark)")
    p.add_argument("--detection-conf", type=float, default=0.2,
                   help="mediapipe min_detection_confidence (0.2 detects ~94%% of these crops vs 71%% at 0.5)")
    p.add_argument("--model-complexity", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--mode", default="balanced", choices=["performance", "balanced", "lightweight"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--score-thr", type=float, default=0.3)
    args = p.parse_args()

    manifest = sd.load_manifest(args.root)
    sample_ids = [r["id"] for r in manifest]
    if args.limit:
        sample_ids = sample_ids[:args.limit]
    print(f"{len(sample_ids)} labelled crops | backend={args.backend} -> {args.out}", flush=True)

    if args.backend == "mediapipe":
        run_mediapipe(sample_ids, args.root, args.out, args.min_visibility, args.model_complexity,
                      args.detection_conf)
    else:
        run_rtmpose(sample_ids, args.root, args.out, args.mode, args.device, args.score_thr)
    print("=== extraction done ===", flush=True)


if __name__ == "__main__":
    main()
