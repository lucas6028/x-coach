"""Convert the Kaggle NLF output into the local ``nlf_3d`` / ``nlf_2d`` arms.

The Kaggle kernel (``.kaggle_tmp/fitaqa_nlf/fitness-aqa-shallow-nlf.py``) saves raw
SMPL-24 ``joints3d`` (mm) and ``joints2d`` (px) per crop. This maps both to H36M-17 with
the same ``SMPL24_TO_H36M17`` used everywhere else and writes them into the pose dir
alongside the MediaPipe/RTMPose arms, so ``run_shallow_depth_classification.py`` picks
them up unchanged.

    .venv\\Scripts\\python.exe scripts/fitness_aqa/ingest_shallow_nlf.py \
        --nlf .kaggle_tmp/fitaqa_nlf/out/nlf_shallow.npz
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
from src.fitness_aqa import shallow_dataset as sd  # noqa: E402
from scripts.fitness_aqa.run_shallow_pose_extraction import save_arm  # noqa: E402

DEFAULT_OUT = sd.DEFAULT_SHALLOW_ROOT / "derived" / "pose"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nlf", type=Path, default=PROJECT_ROOT / ".kaggle_tmp" / "fitaqa_nlf" / "out" / "nlf_shallow.npz")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    with np.load(args.nlf, allow_pickle=False) as d:
        sample_ids = [str(s) for s in d["sample_ids"]]
        smpl3 = d["joints3d_smpl24"].astype(np.float64)   # (N, 24, 3) mm
        smpl2 = d["joints2d_smpl24"].astype(np.float64)   # (N, 24, 2) px
        detected = d["detected"].astype(bool)

    n = len(sample_ids)
    print(f"{n} crops from {args.nlf.name}  detected={detected.mean():.3f}", flush=True)

    j3d = map_smpl24_to_h36m17(smpl3).astype(np.float32)   # (N, 17, 3)
    j2d = map_smpl24_to_h36m17(smpl2).astype(np.float32)   # (N, 17, 2)

    save_arm(args.out, "nlf_3d", sample_ids, j3d, detected,
             space=np.asarray("cam3d"), units=np.asarray("mm"))
    save_arm(args.out, "nlf_2d", sample_ids, j2d, detected,
             space=np.asarray("image2d"), units=np.asarray("pixels"))
    print("=== ingest done ===", flush=True)


if __name__ == "__main__":
    main()
