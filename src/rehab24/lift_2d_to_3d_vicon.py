"""Vicon-native 2D->3D lifting experiment for REHAB24-6 squat-correctness.

This is the Vicon counterpart of :mod:`src.rehab24.lift_2d_to_3d`. Stage 1 lifted
MediaPipe image landmarks toward MediaPipe's *own* BlazePose world (a learned
pseudo-3D prior). Here we lift in the native REHAB24-6 26-joint mocap space: the
per-camera Vicon 2D skeleton ``(T,26,2)`` -> Vicon 3D ``(T,26,3)``, supervised by the
**real Vicon mocap** 3D (high-fidelity ground truth, not a learned prior). Then we
rebuild the standard 2340-d skeleton feature from the **lifted** 3D + the same 2D and
run the correctness LOSO on it.

The question this isolates: *how much of Vicon's 3D discriminative power (LOSO 0.702)
is recoverable from a single camera's 2D alone?* Because the lifter never sees
correctness labels, the downstream LOSO stays a clean generalization estimate.

Controlled three-way ablation (identical 2D source, only the 3D block varies):

* ``vicon2d``  (936)  — 2D-only, no 3D block.                 [lower bound]
* ``lifted3d`` (2340) — same 2D + lifted-from-2D 3D block.    [the experiment]
* ``vicon``    (2340) — same 2D + real Vicon mocap 3D.        [upper bound, LOSO 0.702]

Data source: the dataset's own ``skeleton_3d_path`` ``(T,26,4)`` (xyz + likelihood) and
per-camera ``skeleton_2d_path`` ``(T,26,2)`` arrays, so the experiment runs offline on
CPU with no GPU and no external weights. The 3D mocap is camera-independent, so the two
camera views of a video become two training samples that share one 3D target.

Caveat (same as Stage 1): the lifter trains on a subject-disjoint split for early
stopping but is then applied to all subjects — a generically-pretrained label-blind
lifter applied to unseen subjects. A fully nested per-fold lifter is a follow-up.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.rehab24.dataset import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PROCESSED_ROOT,
    load_manifest,
    resolve_data_path,
)
from src.rehab24.skeleton_features import (
    HIPS,
    LEFT_SHOULDER,
    LEFT_UP_LEG,
    RIGHT_SHOULDER,
    RIGHT_UP_LEG,
    add_velocity,
    feature_output_path,
    frame_bounds,
    normalize_points,
    save_feature,
    summarize_time_series,
)
from src.rehab24.lift_2d_to_3d import TemporalLifter, _masked_mse, set_seed

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("REHAB24-6 Vicon 2D->3D lifting requires `torch`.") from exc


NUM_JOINTS = 26
SCALE_PAIRS = ((LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_UP_LEG, RIGHT_UP_LEG))


def normalize_root_scale(points: np.ndarray) -> np.ndarray:
    """Root-center on the hips and scale by shoulder/hip span (mirror the feature builder)."""
    return normalize_points(points, root_index=HIPS, scale_pairs=SCALE_PAIRS)


# ---------------------------------------------------------------------------
# Per-source-clip data: normalized 2D input, normalized 3D target, validity mask
# ---------------------------------------------------------------------------
class ClipSample:
    """One (video, camera) clip's normalized Vicon landmarks, ready for the lifter."""

    __slots__ = ("skeleton_2d_path", "person_id", "norm_image", "norm_world", "mask")

    def __init__(self, skeleton_2d_path: str, person_id: int, norm_image: np.ndarray, norm_world: np.ndarray) -> None:
        self.skeleton_2d_path = skeleton_2d_path
        self.person_id = person_id
        self.norm_image = norm_image  # (T, 26, 2)
        self.norm_world = norm_world  # (T, 26, 3)
        self.mask = np.isfinite(norm_world).all(axis=2)  # (T, 26) — joints with real 3D


def load_clip_samples(manifest_path: Path, data_root: Path) -> list[ClipSample]:
    """Load + normalize every (video, camera) clip's Vicon 2D/3D arrays once.

    Keyed by ``skeleton_2d_path`` (unique per video+camera); the 3D mocap is shared
    across the two cameras of a video, so each camera view is a separate sample with
    its own 2D but the same 3D target.
    """
    rows = load_manifest(manifest_path)
    meta: dict[str, tuple[str, int]] = {}  # skeleton_2d_path -> (skeleton_3d_path, person_id)
    for row in rows:
        meta.setdefault(row["skeleton_2d_path"], (row["skeleton_3d_path"], int(row["person_id"])))

    samples: list[ClipSample] = []
    for skel2d, (skel3d, person_id) in meta.items():
        world = np.load(resolve_data_path(data_root, skel3d), mmap_mode="r")[:, :, :3].astype(np.float32)
        image = np.load(resolve_data_path(data_root, skel2d), mmap_mode="r")[:, :, :2].astype(np.float32)
        total = min(world.shape[0], image.shape[0])
        samples.append(
            ClipSample(
                skel2d,
                person_id,
                normalize_root_scale(image[:total]),
                normalize_root_scale(world[:total]),
            )
        )
    return samples


def _to_tensor(sample: ClipSample, device: "torch.device") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    t = sample.norm_image.shape[0]
    x = np.nan_to_num(sample.norm_image.reshape(t, -1), nan=0.0)
    y = np.nan_to_num(sample.norm_world.reshape(t, -1), nan=0.0)
    m = np.repeat(sample.mask, 3, axis=1).astype(np.float32)  # (T, 26*3)
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
    y_t = torch.from_numpy(y).unsqueeze(0).to(device)
    m_t = torch.from_numpy(m).unsqueeze(0).to(device)
    return x_t, y_t, m_t


# ---------------------------------------------------------------------------
# Train the lifter (subject-disjoint val split for early stopping)
# ---------------------------------------------------------------------------
def train_lifter(
    samples: list[ClipSample],
    device: "torch.device",
    seed: int,
    epochs: int = 120,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    patience: int = 12,
    hidden: int = 256,
    dropout: float = 0.2,
) -> TemporalLifter:
    set_seed(seed)
    subjects = sorted({s.person_id for s in samples})
    rng = np.random.default_rng(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_subjects = set(shuffled[:n_val])
    train_samples = [s for s in samples if s.person_id not in val_subjects]
    val_samples = [s for s in samples if s.person_id in val_subjects]
    print(
        f"  lifter: {len(train_samples)} train / {len(val_samples)} val clips "
        f"(val subjects {sorted(val_subjects)})",
        flush=True,
    )

    model = TemporalLifter(in_dim=NUM_JOINTS * 2, out_dim=NUM_JOINTS * 3, hidden=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_tensors = [_to_tensor(s, device) for s in train_samples]
    val_tensors = [_to_tensor(s, device) for s in val_samples]

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    epochs_no_improve = 0
    order = np.arange(len(train_tensors))

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(order)
        train_loss = 0.0
        for idx in order:
            x_t, y_t, m_t = train_tensors[idx]
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_mse(model(x_t), y_t, m_t)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach())
        train_loss /= max(len(train_tensors), 1)

        model.eval()
        with torch.no_grad():
            val_loss = (
                float(np.mean([float(_masked_mse(model(x), y, m)) for x, y, m in val_tensors]))
                if val_tensors
                else train_loss
            )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epoch == 1 or epoch % 10 == 0 or epochs_no_improve >= patience:
            print(f"  epoch {epoch:3d}  train_mse {train_loss:.5f}  val_mse {val_loss:.5f}  best {best_val:.5f}", flush=True)
        if epochs_no_improve >= patience:
            print(f"  early stop at epoch {epoch} (best val_mse {best_val:.5f})", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_world(model: TemporalLifter, sample: ClipSample, device: "torch.device") -> np.ndarray:
    """Lift one clip's 2D to normalized 3D; re-impose NaN on never-tracked joints."""
    t = sample.norm_image.shape[0]
    x = torch.from_numpy(np.nan_to_num(sample.norm_image.reshape(t, -1), nan=0.0)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).squeeze(0).cpu().numpy().reshape(t, NUM_JOINTS, 3)
    pred[~sample.mask] = np.nan
    return pred.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature builders (lifted 3D, and the matched 2D-only baseline)
# ---------------------------------------------------------------------------
def _summarize_segment(points_norm: np.ndarray) -> np.ndarray:
    return summarize_time_series(add_velocity(points_norm))


def lifted_feature_vector(world_norm: np.ndarray, image_norm: np.ndarray, first_frame: int, last_frame: int) -> np.ndarray:
    """2340-dim feature: lifted (already-normalized) 3D block + normalized 2D block.

    Structurally identical to the Vicon builder (3D block 26*6*9=1404, then 2D block
    26*4*9=936). Both branches are pre-normalized, so we summarize directly.
    """
    total = min(int(world_norm.shape[0]), int(image_norm.shape[0]))
    start, stop = frame_bounds(first_frame, last_frame, total)
    world_block = _summarize_segment(world_norm[start:stop, :, :3])
    image_block = _summarize_segment(image_norm[start:stop, :, :2])
    return np.concatenate([world_block, image_block], axis=0)


def vicon2d_feature_vector(image_norm: np.ndarray, first_frame: int, last_frame: int) -> np.ndarray:
    """936-dim 2D-only feature (the depth-free lower bound, identical 2D source)."""
    total = int(image_norm.shape[0])
    start, stop = frame_bounds(first_frame, last_frame, total)
    return _summarize_segment(image_norm[start:stop, :, :2])


def write_features(
    samples: list[ClipSample],
    lifted_world: dict[str, np.ndarray],
    manifest_path: Path,
    lifted_dir: Path,
    vicon2d_dir: Path,
    overwrite: bool,
) -> tuple[int, int]:
    """Slice each manifest rep and write the lifted3d + vicon2d feature bundles."""
    rows = load_manifest(manifest_path)
    image_norm_of = {s.skeleton_2d_path: s.norm_image for s in samples}

    n_lifted = n_2d = 0
    for row in rows:
        skel2d = row["skeleton_2d_path"]
        image_norm = image_norm_of[skel2d]
        world_norm = lifted_world[skel2d]
        first, last = int(row["first_frame"]), int(row["last_frame"])

        lifted_path = feature_output_path(lifted_dir, row["split"], row["sample_id"])
        if overwrite or not lifted_path.exists():
            save_feature(lifted_path, row, lifted_feature_vector(world_norm, image_norm, first, last))
            n_lifted += 1
        v2d_path = feature_output_path(vicon2d_dir, row["split"], row["sample_id"])
        if overwrite or not v2d_path.exists():
            save_feature(v2d_path, row, vicon2d_feature_vector(image_norm, first, last))
            n_2d += 1
    return n_lifted, n_2d


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Vicon 2D->3D lifter and write lifted REHAB24-6 features.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--lifted-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "lifted3d_vicon_skeleton_features")
    parser.add_argument("--vicon2d-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "vicon2d_skeleton_features")
    parser.add_argument("--model-output", type=Path, default=DEFAULT_PROCESSED_ROOT / "lift_2d_to_3d_vicon_model.pt")
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_PROCESSED_ROOT / "lift_2d_to_3d_vicon_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or auto.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    print(f"Vicon 2D->3D lifting on {device}")

    samples = load_clip_samples(args.manifest, args.data_root)
    print(f"Loaded {len(samples)} source clips (unique video+camera)")

    model = train_lifter(
        samples, device, seed=args.seed, epochs=args.epochs, lr=args.lr,
        patience=args.patience, hidden=args.hidden, dropout=args.dropout,
    )

    lifted_world: dict[str, np.ndarray] = {}
    sq_err = 0.0
    n_terms = 0
    for sample in samples:
        pred = predict_world(model, sample, device)
        lifted_world[sample.skeleton_2d_path] = pred
        diff = (pred - sample.norm_world) ** 2
        finite = np.isfinite(diff)
        sq_err += float(diff[finite].sum())
        n_terms += int(finite.sum())
    overall_mse = sq_err / max(n_terms, 1)
    print(f"Lifted-vs-real normalized-Vicon-3D MSE (all clips): {overall_mse:.5f}")

    n_lifted, n_2d = write_features(
        samples, lifted_world, args.manifest, args.lifted_dir, args.vicon2d_dir, args.overwrite
    )
    print(f"Wrote {n_lifted} lifted3d-vicon features -> {args.lifted_dir}")
    print(f"Wrote {n_2d} vicon2d features -> {args.vicon2d_dir}")

    torch.save(model.state_dict(), args.model_output)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_lifted_world_mse": overall_mse,
                "n_clips": len(samples),
                "n_lifted_features": n_lifted,
                "n_vicon2d_features": n_2d,
                "seed": args.seed,
                "epochs": args.epochs,
                "hidden": args.hidden,
                "num_joints": NUM_JOINTS,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Saved lifter to {args.model_output} and metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()
