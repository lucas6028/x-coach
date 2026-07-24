"""Fold the Kaggle NLF output into local ``nlf_3d`` / ``nlf_2d`` video-pose arms.

Two input shapes are accepted, because Kaggle caps a kernel at 500 output files:

* **loose** -- one npz per video (``<vid>.npz`` with ``frame_idx``, ``joints3d_smpl24``
  mm, ``joints2d_smpl24`` px, ``detected``). The first kernel run used this and hit the
  cap at 500 videos.
* **consolidated** -- a single npz packing many videos under per-video keys
  ``<vid>__idx / __j3 / __j2 / __det``. The re-run for the remaining videos uses this to
  stay under the file cap.

Either way each video is mapped SMPL-24 -> H36M-17 and re-saved in the same
``joints_2d`` / ``joints_3d`` / ``detected`` layout the MediaPipe arm uses, so
``run_squat_depth_classification.py`` treats every arm identically. Pass ``--nlf-dir``
once per source; both write into the same ``--out`` dir (idempotent, so re-running is
safe).

    .venv\\Scripts\\python.exe scripts/fitness_aqa/ingest_squat_nlf.py --nlf-dir .kaggle_tmp/squat_nlf/out/nlf_squat
    .venv\\Scripts\\python.exe scripts/fitness_aqa/ingest_squat_nlf.py --nlf-dir .kaggle_tmp/squat_nlf2/out
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


def _save_video(out: Path, vid: str, j3: np.ndarray, j2: np.ndarray,
                frame_idx: np.ndarray, detected: np.ndarray) -> float:
    if j3.shape[0] == 0:
        joints_3d = np.zeros((0, 17, 3), "float32")
        joints_2d = np.zeros((0, 17, 2), "float32")
    else:
        joints_3d = map_smpl24_to_h36m17(j3.astype(np.float64)).astype("float32")
        joints_2d = map_smpl24_to_h36m17(j2.astype(np.float64)).astype("float32")
    np.savez_compressed(out / f"{vid}.npz", frame_idx=frame_idx,
                        joints_2d=joints_2d, joints_3d=joints_3d, detected=detected.astype(bool))
    return float(detected.mean()) if len(detected) else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nlf-dir", type=Path, default=PROJECT_ROOT / ".kaggle_tmp" / "squat_nlf" / "out" / "nlf_squat")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    n_done = 0
    det_rates = []

    # Loose: one npz per video.
    for f in sorted(args.nlf_dir.glob("*.npz")):
        with np.load(f, allow_pickle=False) as d:
            if "joints3d_smpl24" not in d:
                continue  # consolidated file handled below
            det_rates.append(_save_video(args.out, f.stem, d["joints3d_smpl24"], d["joints2d_smpl24"],
                                         d["frame_idx"], d["detected"]))
        n_done += 1

    # Consolidated: one npz packing many videos under <vid>__j3/__j2/__idx/__det.
    for f in sorted(args.nlf_dir.glob("*.npz")):
        with np.load(f, allow_pickle=False) as d:
            keys = list(d.keys())
            if "joints3d_smpl24" in keys:
                continue  # loose file, already handled
            vids = sorted({k[:-4] for k in keys if k.endswith("__j3")})
            for vid in vids:
                det_rates.append(_save_video(args.out, vid, d[f"{vid}__j3"], d[f"{vid}__j2"],
                                             d[f"{vid}__idx"], d[f"{vid}__det"]))
                n_done += 1

    msg = f"wrote {n_done} videos -> {args.out}"
    if det_rates:
        msg += f"  mean det%={np.mean(det_rates):.3f}"
    print(msg, flush=True)
    print("=== ingest done ===", flush=True)


if __name__ == "__main__":
    main()
