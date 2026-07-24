"""NLF (direct image->3D) on the Fitness-AQA shallow-squat crops -- the strong-depth arm.

Writes the pair the experiment turns on, both from the *same* forward pass:

``nlf_3d``  camera-frame SMPL-24 -> H36M-17 in mm  (depth present)
``nlf_2d``  the model's own image-plane joints2d  (depth gone, detector identical)

Because both arms come from one inference, a gap between them cannot be detector
quality, dataset filtering, or feature code -- only the depth channel.

    .venv\\Scripts\\python.exe scripts/fitness_aqa/run_shallow_nlf_extraction.py --device cpu
    .venv\\Scripts\\python.exe scripts/fitness_aqa/run_shallow_nlf_extraction.py --device cuda --batch 8

The torchscript is the same ``nlf_l_multi`` build used for the REHAB24-6 extraction.
torchvision must be imported before ``torch.jit.load`` -- it registers the
``torchvision::nms`` op the bundled detector calls.
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

from src.fit3d.depth_eval import map_smpl24_to_h36m17  # noqa: E402
from src.fitness_aqa import shallow_dataset as sd  # noqa: E402

DEFAULT_MODEL = PROJECT_ROOT / ".kaggle_tmp" / "nlf_smoke_out" / "models" / "nlf_l_multi.torchscript"
DEFAULT_OUT = sd.DEFAULT_SHALLOW_ROOT / "derived" / "pose"


def largest_box(boxes) -> int:
    """Index of the biggest detection -- the lifter, not a bystander or a mirror image."""
    if boxes is None or boxes.ndim != 2 or boxes.shape[0] == 0:
        return -1
    return int((boxes[:, 2] * boxes[:, 3]).argmax())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--root", type=Path, default=sd.DEFAULT_SHALLOW_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--num-aug", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="reuse an existing partial nlf_3d.npz")
    args = p.parse_args()

    import cv2
    import torch
    import torchvision  # noqa: F401  registers torchvision::nms for the bundled detector

    from scripts.fitness_aqa.run_shallow_pose_extraction import save_arm

    manifest = sd.load_manifest(args.root)
    sample_ids = [r["id"] for r in manifest]
    if args.limit:
        sample_ids = sample_ids[:args.limit]
    n = len(sample_ids)
    index = {sid: i for i, sid in enumerate(sample_ids)}

    j3d = np.full((n, 17, 3), np.nan, np.float32)
    j2d = np.full((n, 17, 2), np.nan, np.float32)
    detected = np.zeros(n, bool)
    if args.resume and (args.out / "nlf_3d.npz").exists():
        with np.load(args.out / "nlf_3d.npz", allow_pickle=False) as d:
            prev = {str(s): k for k, s in enumerate(d["sample_ids"])}
            for sid, k in prev.items():
                if sid in index and d["detected"][k]:
                    j3d[index[sid]] = d["joints"][k]
                    detected[index[sid]] = True
        with np.load(args.out / "nlf_2d.npz", allow_pickle=False) as d:
            prev = {str(s): k for k, s in enumerate(d["sample_ids"])}
            for sid, k in prev.items():
                if sid in index and d["detected"][k]:
                    j2d[index[sid]] = d["joints"][k]
        print(f"resumed {int(detected.sum())} already-inferred crops", flush=True)

    print(f"loading {args.model} on {args.device}", flush=True)
    t = time.time()
    model = torch.jit.load(str(args.model), map_location=args.device).eval()
    if args.device.startswith("cuda"):
        model = model.cuda()
    print(f"model loaded in {time.time() - t:.0f}s", flush=True)

    todo = [sid for sid in sample_ids if not detected[index[sid]]]
    print(f"{len(todo)}/{n} crops to infer | batch={args.batch} num_aug={args.num_aug}", flush=True)

    buf: list[np.ndarray] = []
    ids: list[str] = []
    t0 = time.time()
    done = 0

    def flush() -> None:
        nonlocal done
        if not buf:
            return
        bt = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).contiguous()
        if args.device.startswith("cuda"):
            bt = bt.cuda()
        with torch.inference_mode():
            pred = model.detect_smpl_batched(bt, num_aug=args.num_aug)
        for k, sid in enumerate(ids):
            boxes = pred["boxes"][k].detach().cpu().numpy()
            bi = largest_box(boxes)
            if bi < 0:
                continue
            i = index[sid]
            smpl3 = pred["joints3d"][k][bi].detach().cpu().numpy()[None]      # (1, 24, 3) mm
            smpl2 = pred["joints2d"][k][bi].detach().cpu().numpy()[None]      # (1, 24, 2) px
            j3d[i] = map_smpl24_to_h36m17(smpl3)[0].astype(np.float32)
            j2d[i] = map_smpl24_to_h36m17(smpl2)[0].astype(np.float32)
            detected[i] = True
        done += len(ids)
        buf.clear()
        ids.clear()
        el = time.time() - t0
        print(f"  {done}/{len(todo)}  {el / 60:.1f}m elapsed  {1000 * el / max(1, done):.0f} ms/crop  "
              f"eta {el / max(1, done) * (len(todo) - done) / 60:.0f}m", flush=True)

    for sid, img in sd.iter_crops(todo, args.root):
        buf.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ids.append(sid)
        if len(buf) >= args.batch:
            flush()
    flush()

    save_arm(args.out, "nlf_3d", sample_ids, j3d, detected,
             space=np.asarray("cam3d"), units=np.asarray("mm"))
    save_arm(args.out, "nlf_2d", sample_ids, j2d, detected,
             space=np.asarray("image2d"), units=np.asarray("pixels"))
    print("=== NLF extraction done ===", flush=True)


if __name__ == "__main__":
    main()
