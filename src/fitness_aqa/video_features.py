"""Aggregate per-frame squat cues into one fixed-length video-level feature vector.

The knees-forward / knees-inward faults are labelled per *clip*, so each arm's per-frame
cues (from ``cue_features.compute_features``) must be reduced to a per-video vector before
classification. The reducer is identical across arms -- the only thing that differs
between the 2D and 3D arm of a model is still just the depth channel inside the cues.

Two frame pools are summarised and concatenated:

* **all** -- every detected frame in the clip.
* **bottom** -- the deepest portion of the clip, selected purely by *pose* (lowest-hip
  frames), never by the fault label. Knees-forward/inward are worst near the bottom, so
  a bottom-phase summary sharpens the signal without leaking the answer.

Each pool contributes mean / std / min / max / p10 / p50 / p90 per cue. NaN-robust: a cue
that is missing on some frames is summarised over the frames where it exists.
"""

from __future__ import annotations

import numpy as np

from src.fit3d.biomech import IMAGE2D
from src.fitness_aqa import cue_features as cf
from src.fitness_aqa.cue_features import CAM3D

STATS = ("mean", "std", "min", "max", "p10", "p50", "p90")
POOLS = ("all", "bottom")
FEATURE_NAMES = tuple(
    f"{pool}__{cue}__{stat}" for pool in POOLS for cue in cf.FEATURE_NAMES for stat in STATS
)


def _hip_height(joints: np.ndarray, mode: str) -> np.ndarray:
    """Mean hip vertical coordinate per frame, oriented so 'up' increases. (F,)."""
    pts, bmode = cf.to_biomech_space(joints, mode)
    from src.fit3d.dataset import L_HIP, R_HIP

    h = pts[..., 2] if bmode == "world3d" else -pts[..., 1]
    return 0.5 * (h[:, R_HIP] + h[:, L_HIP])


def _summarise(values: np.ndarray) -> np.ndarray:
    """(F, C) frame cues -> (C * len(STATS),) NaN-robust stats, flattened by cue then stat."""
    n_cue = values.shape[1]
    out = np.full((n_cue, len(STATS)), np.nan)
    for c in range(n_cue):
        col = values[:, c]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        out[c] = [col.mean(), col.std(), col.min(), col.max(),
                  np.percentile(col, 10), np.percentile(col, 50), np.percentile(col, 90)]
    return out.reshape(-1)


def feature_names(cue_module=cf) -> tuple[str, ...]:
    """Video-level feature names for a given per-frame cue module."""
    return tuple(f"{pool}__{cue}__{stat}"
                 for pool in POOLS for cue in cue_module.FEATURE_NAMES for stat in STATS)


def video_feature(joints: np.ndarray, detected: np.ndarray, mode: str,
                  bottom_fraction: float = 0.4, cue_module=cf) -> np.ndarray:
    """(F, 17, C) joints + (F,) detected mask -> video feature for ``cue_module``.

    ``bottom_fraction`` of the detected frames with the lowest hips form the bottom pool
    (the squat dip -- also where OHP knee-bend shows, since the hips drop). Returns all-NaN
    if no frame was detected (the caller drops such videos from every arm). ``cue_module``
    defaults to the squat cues; pass ``movement_cues`` for OHP.
    """
    joints = np.asarray(joints, dtype=np.float64)
    detected = np.asarray(detected, dtype=bool)
    n_out = len(feature_names(cue_module))
    valid = detected & np.isfinite(joints).all(axis=(1, 2))
    if not valid.any():
        return np.full(n_out, np.nan)

    cues = cue_module.compute_features(joints[valid], mode)   # (V, n_cue)
    hip_h = _hip_height(joints[valid], mode)                  # (V,)

    n_bottom = max(1, int(round(bottom_fraction * cues.shape[0])))
    order = np.argsort(hip_h)                                 # ascending = lowest hips first
    bottom_idx = order[:n_bottom]

    return np.concatenate([_summarise(cues), _summarise(cues[bottom_idx])])


def build_matrix(per_video: dict[str, tuple[np.ndarray, np.ndarray]], mode: str,
                 video_ids: list[str], cue_module=cf) -> tuple[np.ndarray, np.ndarray]:
    """video_ids -> (N, D) features and (N,) detected mask, in the given id order.

    ``per_video[vid] = (joints (F,17,C), detected (F,))``. A missing or fully-undetected
    video yields an all-NaN row and ``detected=False``.
    """
    n_out = len(feature_names(cue_module))
    feats = np.full((len(video_ids), n_out), np.nan)
    ok = np.zeros(len(video_ids), bool)
    for i, vid in enumerate(video_ids):
        if vid not in per_video:
            continue
        joints, detected = per_video[vid]
        if joints.shape[0] == 0 or not np.asarray(detected, bool).any():
            continue
        feats[i] = video_feature(joints, detected, mode, cue_module=cue_module)
        ok[i] = np.isfinite(feats[i]).all()
    return feats, ok
