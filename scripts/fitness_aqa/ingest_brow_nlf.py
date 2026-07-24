"""Fold the Kaggle BarbellRow NLF chunks into local ``nlf_3d`` / ``nlf_2d`` frame arms.

The BRow kernel writes chunked npz (``nlf_brow_XXX.npz``) packing crops under per-id keys
``<id>__j3 / __j2 / __det`` (SMPL-24). This maps each to H36M-17 and writes the two arm
npz in the same ``sample_ids`` / ``joints`` / ``detected`` layout the MediaPipe/RTMPose
frame arms use, so ``run_frame_depth_classification.py --dataset brow`` treats every arm
identically.

    .venv\\Scripts\\python.exe scripts/fitness_aqa/ingest_brow_nlf.py --nlf-dir .kaggle_tmp/brow_nlf/out
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
from src.fitness_aqa import barbellrow_dataset as bd  # noqa: E402

DEFAULT_OUT = bd.DEFAULT_BROW_ROOT / "derived" / "pose"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nlf-dir", type=Path, default=PROJECT_ROOT / ".kaggle_tmp" / "brow_nlf" / "out")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    sample_ids: list[str] = []
    j3_list: list[np.ndarray] = []
    j2_list: list[np.ndarray] = []
    det_list: list[bool] = []

    chunks = sorted(args.nlf_dir.glob("nlf_brow_*.npz")) or sorted(args.nlf_dir.glob("*.npz"))
    for f in chunks:
        with np.load(f, allow_pickle=False) as d:
            keys = list(d.keys())
            if "joints3d_smpl24" in keys:  # single-npz legacy shape
                for k, sid in enumerate([str(s) for s in d["sample_ids"]]):
                    sample_ids.append(sid)
                    j3_list.append(d["joints3d_smpl24"][k])
                    j2_list.append(d["joints2d_smpl24"][k])
                    det_list.append(bool(d["detected"][k]))
                continue
            for vid in sorted({k[:-4] for k in keys if k.endswith("__j3")}):
                sample_ids.append(vid)
                j3_list.append(d[f"{vid}__j3"])
                j2_list.append(d[f"{vid}__j2"])
                det_list.append(bool(d[f"{vid}__det"]))

    if not sample_ids:
        raise SystemExit(f"no NLF crops found in {args.nlf_dir}")

    smpl3 = np.stack(j3_list).astype(np.float64)   # (N, 24, 3)
    smpl2 = np.stack(j2_list).astype(np.float64)   # (N, 24, 2)
    detected = np.asarray(det_list, bool)
    j3d = map_smpl24_to_h36m17(smpl3).astype(np.float32)
    j2d = map_smpl24_to_h36m17(smpl2).astype(np.float32)

    args.out.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(sample_ids)
    np.savez_compressed(args.out / "nlf_3d.npz", sample_ids=ids, joints=j3d, detected=detected,
                        space=np.asarray("cam3d"))
    np.savez_compressed(args.out / "nlf_2d.npz", sample_ids=ids, joints=j2d, detected=detected,
                        space=np.asarray("image2d"))
    print(f"wrote {len(ids)} crops -> {args.out}  det%={detected.mean():.3f}", flush=True)
    print("=== ingest done ===", flush=True)


if __name__ == "__main__":
    main()
