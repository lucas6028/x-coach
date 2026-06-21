"""2D->3D lifting experiment for REHAB24-6 squat-correctness.

Motivation
----------
The paired-LOSO study found that a stronger *2D* backbone (HRNet-w48) does not beat
RTMPose-2D on squat correctness (within noise), while both 2D-only paths sit below
MediaPipe's pseudo-3D. The working thesis (see notes / memory
``hrnet-vs-rtmpose-loso-undetermined``): the bottleneck is the missing **depth
channel**, not 2D keypoint accuracy.

This module tests whether that depth signal is *recoverable from 2D alone*. It trains
a temporal 2D->3D lifter — MediaPipe image landmarks ``(T,33,2)`` -> world landmarks
``(T,33,3)`` — using MediaPipe's own world landmarks as supervision, then rebuilds
skeleton features from the **lifted** world. Because the lifter never sees correctness
labels, the downstream correctness LOSO remains a clean generalization estimate.

The controlled three-way ablation (identical 2D source, only the world block varies):

* ``mp2d``     (1188) — MediaPipe image-only 2D, no depth.            [lower bound]
* ``lifted3d`` (2970) — same image block + lifted-from-2D world block. [the experiment]
* ``mediapipe``(2970) — same image block + BlazePose real world.      [upper bound]

If ``lifted3d`` climbs from ``mp2d`` toward ``mediapipe``, depth is 2D-recoverable and
the lifting path is validated (motivating the heavier HRNet-2D -> lift pipeline). If it
stalls at ``mp2d``, MediaPipe's depth advantage comes from BlazePose's learned 3D prior,
not from 2D geometry — which would reframe the project.

Data source: the on-disk MediaPipe landmark cache
(``data/REHAB24-6/processed/mediapipe_landmarks_cache/{video_stem}.npz`` with keys
``world (T,33,3)`` and ``image (T,33,2)``), so the whole experiment runs offline on CPU
with no GPU, no Vicon mocap, and no external lifter weights.

Caveat (documented, not a leak of the measured quantity): the lifter is trained on a
subject-disjoint split for early stopping but is then applied to all subjects — the
standard "generically pretrained lifter applied to unseen subjects" setup. It learns a
label-blind geometric map, so the correctness LOSO it feeds is unaffected; a fully
nested per-fold lifter is a possible follow-up.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.rehab24.dataset import DEFAULT_DATA_ROOT, DEFAULT_PROCESSED_ROOT, load_manifest
from src.rehab24.mediapipe_skeleton_features import (
    MP_LEFT_HIP,
    MP_RIGHT_HIP,
    MP_NUM_LANDMARKS,
    SCALE_PAIRS,
    landmark_cache_path,
    midpoint,
    normalize_points_root,
)
from src.rehab24.skeleton_features import (
    add_velocity,
    feature_output_path,
    frame_bounds,
    save_feature,
    summarize_time_series,
)

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("REHAB24-6 2D->3D lifting requires `torch`.") from exc


# ---------------------------------------------------------------------------
# Normalization helpers (mirror the MediaPipe feature builder exactly)
# ---------------------------------------------------------------------------
def normalize_root_scale(points: np.ndarray) -> np.ndarray:
    """Root-center on the mid-hip and scale by shoulder/hip span (per frame).

    Same normalization the MediaPipe feature builder applies to both the world and
    image branches, so the lifter trains in — and predicts into — the exact space the
    downstream summary consumes.
    """
    return normalize_points_root(points, midpoint(points, MP_LEFT_HIP, MP_RIGHT_HIP), SCALE_PAIRS)


# ---------------------------------------------------------------------------
# Temporal 2D->3D lifter (dilated residual TCN; CPU-friendly)
# ---------------------------------------------------------------------------
class _ResidualBlock(nn.Module):
    """Two dilated 1D convs over time with a residual skip (length-preserving)."""

    def __init__(self, channels: int, dilation: int, kernel: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        pad = dilation * (kernel - 1) // 2  # 'same' padding keeps T fixed
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        y = self.drop(self.act(self.conv1(x)))
        y = self.drop(self.act(self.conv2(y)))
        return x + y


class TemporalLifter(nn.Module):
    """Lift normalized 2D joints to normalized 3D joints with temporal context.

    Depth from a single view is ambiguous frame-by-frame; motion disambiguates it.
    The stacked dilated convolutions give a wide temporal receptive field (~120
    frames / ~4s at 30fps) so the lifter can use how a joint moves to infer how far
    it is — the squat depth/lean cues a per-frame MLP cannot see.
    """

    def __init__(
        self,
        in_dim: int = MP_NUM_LANDMARKS * 2,
        out_dim: int = MP_NUM_LANDMARKS * 3,
        hidden: int = 256,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.inp = nn.Conv1d(in_dim, hidden, 1)
        self.blocks = nn.ModuleList(_ResidualBlock(hidden, d, dropout=dropout) for d in dilations)
        self.out = nn.Conv1d(hidden, out_dim, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # x: (B, T, in_dim)
        h = x.transpose(1, 2)  # -> (B, in_dim, T) for Conv1d
        h = torch.relu(self.inp(h))
        for block in self.blocks:
            h = block(h)
        y = self.out(h)
        return y.transpose(1, 2)  # -> (B, T, out_dim)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Per-video data: normalized 2D input, normalized 3D target, validity mask
# ---------------------------------------------------------------------------
class VideoSample:
    """One source video's normalized landmarks, ready for the lifter."""

    __slots__ = ("video_path", "person_id", "norm_image", "norm_world", "mask")

    def __init__(self, video_path: str, person_id: int, norm_image: np.ndarray, norm_world: np.ndarray) -> None:
        self.video_path = video_path
        self.person_id = person_id
        self.norm_image = norm_image  # (T, 33, 2)
        self.norm_world = norm_world  # (T, 33, 3)
        self.mask = np.isfinite(norm_world).all(axis=2)  # (T, 33) — joints with real depth


def load_video_samples(manifest_path: Path, cache_dir: Path) -> list[VideoSample]:
    """Load + normalize every source video's cached MediaPipe landmarks once."""
    rows = load_manifest(manifest_path)
    person_of: dict[str, int] = {}
    for row in rows:
        person_of.setdefault(row["video_path"], int(row["person_id"]))

    samples: list[VideoSample] = []
    missing: list[str] = []
    for video_path in dict.fromkeys(person_of):  # unique, manifest order
        cache_path = landmark_cache_path(cache_dir, video_path)
        if not cache_path.exists():
            missing.append(video_path)
            continue
        with np.load(cache_path) as data:
            world = data["world"].astype(np.float32)  # (T, 33, 3)
            image = data["image"].astype(np.float32)  # (T, 33, 2)
        samples.append(
            VideoSample(video_path, person_of[video_path], normalize_root_scale(image), normalize_root_scale(world))
        )
    if missing:
        raise SystemExit(
            f"{len(missing)} videos have no MediaPipe landmark cache (e.g. {missing[0]}). "
            f"Build it first: extract_mediapipe_skeleton_features --landmark-cache {cache_dir}"
        )
    return samples


def _to_tensor(sample: VideoSample, device: "torch.device") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    t = sample.norm_image.shape[0]
    x = np.nan_to_num(sample.norm_image.reshape(t, -1), nan=0.0)
    y = np.nan_to_num(sample.norm_world.reshape(t, -1), nan=0.0)
    m = np.repeat(sample.mask, 3, axis=1).astype(np.float32)  # (T, 33*3)
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
    y_t = torch.from_numpy(y).unsqueeze(0).to(device)
    m_t = torch.from_numpy(m).unsqueeze(0).to(device)
    return x_t, y_t, m_t


def _masked_mse(pred: "torch.Tensor", target: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    denom = mask.sum().clamp_min(1.0)
    return ((pred - target) ** 2 * mask).sum() / denom


# ---------------------------------------------------------------------------
# Train the lifter
# ---------------------------------------------------------------------------
def train_lifter(
    samples: list[VideoSample],
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
    """Train the TCN lifter with a subject-disjoint val split for early stopping."""
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
        f"  lifter: {len(train_samples)} train / {len(val_samples)} val videos "
        f"(val subjects {sorted(val_subjects)})",
        flush=True,
    )

    model = TemporalLifter(hidden=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_tensors = [_to_tensor(s, device) for s in train_samples]
    val_tensors = [_to_tensor(s, device) for s in val_samples]

    import copy

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
            val_loss = float(np.mean([float(_masked_mse(model(x), y, m)) for x, y, m in val_tensors])) if val_tensors else train_loss

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


def predict_world(model: TemporalLifter, sample: VideoSample, device: "torch.device") -> np.ndarray:
    """Lift one video's 2D to normalized 3D; re-impose NaN on never-detected joints.

    Returning NaN for the structurally-absent joints (mouth, fingers — never detected
    by the whole-body remap) makes the downstream time-series summary zero exactly the
    same channels as the real-MediaPipe path, so ``lifted3d`` and ``mediapipe`` differ
    *only* at the joints that actually carry depth.
    """
    t = sample.norm_image.shape[0]
    x = torch.from_numpy(np.nan_to_num(sample.norm_image.reshape(t, -1), nan=0.0)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).squeeze(0).cpu().numpy().reshape(t, MP_NUM_LANDMARKS, 3)
    pred[~sample.mask] = np.nan
    return pred.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature builders (lifted 3D, and the matched 2D-only baseline)
# ---------------------------------------------------------------------------
def _summarize_segment(points_norm: np.ndarray) -> np.ndarray:
    """add_velocity + time-series summary on an already-normalized segment."""
    return summarize_time_series(add_velocity(points_norm))


def lifted_feature_vector(world_norm: np.ndarray, image_norm: np.ndarray, first_frame: int, last_frame: int) -> np.ndarray:
    """2970-dim feature from lifted (already-normalized) world + normalized image.

    Structurally identical to the MediaPipe builder: world block (33*6*9=1782) followed
    by image block (33*4*9=1188). The world is pre-normalized by the lifter, so we skip
    the redundant re-normalization and summarize directly; the image block is byte-for-
    byte the same as the MediaPipe / mp2d path.
    """
    total = min(int(world_norm.shape[0]), int(image_norm.shape[0]))
    start, stop = frame_bounds(first_frame, last_frame, total)
    world_block = _summarize_segment(world_norm[start:stop, :, :3])
    image_block = _summarize_segment(image_norm[start:stop, :, :2])
    return np.concatenate([world_block, image_block], axis=0)


def mp2d_feature_vector(image_norm: np.ndarray, first_frame: int, last_frame: int) -> np.ndarray:
    """1188-dim image-only feature (the depth-free lower bound, identical 2D source)."""
    total = int(image_norm.shape[0])
    start, stop = frame_bounds(first_frame, last_frame, total)
    return _summarize_segment(image_norm[start:stop, :, :2])


def write_features(
    samples: list[VideoSample],
    lifted_world: dict[str, np.ndarray],
    manifest_path: Path,
    lifted_dir: Path,
    mp2d_dir: Path,
    overwrite: bool,
) -> tuple[int, int]:
    """Slice each manifest rep and write the lifted3d + mp2d feature bundles."""
    rows = load_manifest(manifest_path)
    rows_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_video[row["video_path"]].append(row)
    image_norm_of = {s.video_path: s.norm_image for s in samples}

    n_lifted = n_mp2d = 0
    for video_path, vid_rows in rows_by_video.items():
        image_norm = image_norm_of[video_path]
        world_norm = lifted_world[video_path]
        for row in vid_rows:
            first, last = int(row["first_frame"]), int(row["last_frame"])
            lifted_path = feature_output_path(lifted_dir, row["split"], row["sample_id"])
            if overwrite or not lifted_path.exists():
                save_feature(lifted_path, row, lifted_feature_vector(world_norm, image_norm, first, last))
                n_lifted += 1
            mp2d_path = feature_output_path(mp2d_dir, row["split"], row["sample_id"])
            if overwrite or not mp2d_path.exists():
                save_feature(mp2d_path, row, mp2d_feature_vector(image_norm, first, last))
                n_mp2d += 1
    return n_lifted, n_mp2d


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 2D->3D lifter and write lifted REHAB24-6 skeleton features.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "mediapipe_landmarks_cache")
    parser.add_argument("--lifted-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "lifted3d_skeleton_features")
    parser.add_argument("--mp2d-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "mp2d_skeleton_features")
    parser.add_argument("--model-output", type=Path, default=DEFAULT_PROCESSED_ROOT / "lift_2d_to_3d_model.pt")
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_PROCESSED_ROOT / "lift_2d_to_3d_metrics.json")
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
    print(f"2D->3D lifting on {device}")

    samples = load_video_samples(args.manifest, args.cache_dir)
    print(f"Loaded {len(samples)} source videos from {args.cache_dir.name}")

    model = train_lifter(
        samples, device, seed=args.seed, epochs=args.epochs, lr=args.lr,
        patience=args.patience, hidden=args.hidden, dropout=args.dropout,
    )

    # Per-joint 3D reconstruction error on held data is reported via train; here we also
    # log the overall lifted-vs-real world MSE on every video for a quick sanity figure.
    lifted_world: dict[str, np.ndarray] = {}
    sq_err = 0.0
    n_terms = 0
    for sample in samples:
        pred = predict_world(model, sample, device)
        lifted_world[sample.video_path] = pred
        diff = (pred - sample.norm_world) ** 2
        finite = np.isfinite(diff)
        sq_err += float(diff[finite].sum())
        n_terms += int(finite.sum())
    overall_mse = sq_err / max(n_terms, 1)
    print(f"Lifted-vs-real normalized-world MSE (all videos): {overall_mse:.5f}")

    n_lifted, n_mp2d = write_features(
        samples, lifted_world, args.manifest, args.lifted_dir, args.mp2d_dir, args.overwrite
    )
    print(f"Wrote {n_lifted} lifted3d features -> {args.lifted_dir}")
    print(f"Wrote {n_mp2d} mp2d features -> {args.mp2d_dir}")

    torch.save(model.state_dict(), args.model_output)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_lifted_world_mse": overall_mse,
                "n_videos": len(samples),
                "n_lifted_features": n_lifted,
                "n_mp2d_features": n_mp2d,
                "seed": args.seed,
                "epochs": args.epochs,
                "hidden": args.hidden,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Saved lifter to {args.model_output} and metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()
