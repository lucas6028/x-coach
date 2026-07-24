"""Fold the Kaggle NLF per-video output into local ``nlf_3d`` / ``nlf_2d`` video-pose arms.

The Squat NLF kernel writes one npz per video (``frame_idx``, ``joints3d_smpl24`` mm,
``joints2d_smpl24`` px, ``detected``). This maps SMPL-24 -> H36M-17 and re-saves each
video in the same ``joints_2d`` / ``joints_3d`` / ``detected`` layout the MediaPipe arm
uses, so ``run_squat_depth_classification.py`` treats every arm identically.

    .venv\\Scripts\\python.exe scripts/fitness_aqa/ingest_squat_nlf.py --nlf-dir .kaggle_tmp/squat_nlf/out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fit3d.depth_eval import map_smpl24_to_h36m17  # noqa: E402
from src.fitness_aqa import squat_dataset as sq  # noqa: E402

DEFAULT_OUT = sq.DEFAULT_SQUAT_ROOT / "derived" / "video_pose_nlf"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nlf-dir", type=Path, default=PROJECT_ROOT / ".kaggle_tmp" / "squat_nlf" / "out")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(args.nlf_dir.glob("*.npz"))
    npz_files = [f for f in npz_files if f.name != "summary.json"]
    print(f"{len(npz_files)} per-video NLF npz in {args.nlf_dir}", flush=True)

    n_done = 0
    det_rates = []
    for f in npz_files:
        with np.load(f, allow_pickle=False) as d:
            if "joints3d_smpl24" not in d:
                continue
            j3 = d["joints3d_smpl24"].astype(np.float64)     # (F, 24, 3)
            j2 = d["joints2d_smpl24"].astype(np.float64)     # (F, 24, 2)
            frame_idx = d["frame_idx"]
            detected = d["detected"].astype(bool)
        if j3.shape[0] == 0:
            joints_3d = np.zeros((0, 17, 3), "float32")
            joints_2d = np.zeros((0, 17, 2), "float32")
        else:
            joints_3d = map_smpl24_to_h36m17(j3).astype("float32")
            joints_2d = map_smpl24_to_h36m17(j2).astype("float32")
        np.savez_compressed(args.out / f.name, frame_idx=frame_idx,
                            joints_2d=joints_2d, joints_3d=joints_3d, detected=detected)
        n_done += 1
        if len(detected):
            det_rates.append(float(detected.mean()))
    print(f"wrote {n_done} videos -> {args.out}  mean det%={np.mean(det_rates):.3f}" if det_rates
          else f"wrote {n_done} videos", flush=True)
    print("=== ingest done ===", flush=True)


if __name__ == "__main__":
    main()
