"""One feature function, two coordinate spaces -- the whole point of the experiment.

Every arm (NLF 3D, NLF 2D, MediaPipe 3D, MediaPipe 2D, RTMPose 2D) is reduced to the
*same* squat cues by the *same* formulas from ``src/fit3d/biomech.py``. The only thing
that differs is the space the joints live in:

``IMAGE2D``  (F, 17, 2) image pixels, y grows downward -- the depth channel is gone.
``CAM3D``    (F, 17, 3) camera frame (x right, y down, z away) -- depth present.

``CAM3D`` is rotated into the biomech world convention (z up, xy the ground plane) by
mapping (x, y, z)_cam -> (x, z, -y)_world. That keeps "up" defined the same way in both
spaces -- image ``-y`` and camera ``-y`` are the same axis -- so a 3D-vs-2D gap is the
depth channel and not a different notion of vertical. Neither space assumes a level
camera; both inherit whatever tilt the shot has, which is exactly the deployment
condition being tested.

Angles and ratios only (no raw coordinates): they are scale-free, so an arm in
millimetres and an arm in pixels produce comparable numbers, and the feature dimension
is identical across arms -- a 3D win cannot come from simply having more columns.
"""

from __future__ import annotations

import numpy as np

from src.fit3d.biomech import IMAGE2D, WORLD3D, joint_angle, lean_from_vertical
from src.fit3d.dataset import (
    L_ANKLE,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    R_ANKLE,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    ROOT,
    THORAX,
)

CAM3D = "cam3d"
MODES = (IMAGE2D, CAM3D)

FEATURE_NAMES = (
    "knee_angle_r",
    "knee_angle_l",
    "hip_angle_r",
    "hip_angle_l",
    "thigh_incline_r",
    "thigh_incline_l",
    "shank_incline_r",
    "shank_incline_l",
    "torso_lean",
    "depth_ratio",
    "hip_drop_ratio",
    "knee_width_ratio",
    "knee_over_ankle_r",
    "knee_over_ankle_l",
)


def to_biomech_space(points: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    """Return ``(points, biomech_mode)`` in the convention ``src.fit3d.biomech`` expects.

    ``CAM3D`` -> world convention (z up) via (x, y, z)_cam -> (x, z, -y)_world.
    ``IMAGE2D`` passes through unchanged (biomech already treats image y as down).
    """
    pts = np.asarray(points, dtype=np.float64)
    if mode == IMAGE2D:
        if pts.shape[-1] != 2:
            raise ValueError(f"{IMAGE2D} expects (..., 2) points, got {pts.shape}")
        return pts, IMAGE2D
    if mode == CAM3D:
        if pts.shape[-1] != 3:
            raise ValueError(f"{CAM3D} expects (..., 3) points, got {pts.shape}")
        return np.stack([pts[..., 0], pts[..., 2], -pts[..., 1]], axis=-1), WORLD3D
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")


def _height(points: np.ndarray, biomech_mode: str) -> np.ndarray:
    return points[..., 2] if biomech_mode == WORLD3D else -points[..., 1]


def _horizontal(points: np.ndarray, biomech_mode: str) -> np.ndarray:
    return points[..., :2] if biomech_mode == WORLD3D else points[..., :1]


def _seg_len(points: np.ndarray, a: int, b: int) -> np.ndarray:
    return np.linalg.norm(points[:, a, :] - points[:, b, :], axis=1)


def _h_span(points: np.ndarray, a: int, b: int, biomech_mode: str) -> np.ndarray:
    h = _horizontal(points, biomech_mode)
    return np.linalg.norm(h[:, a, :] - h[:, b, :], axis=1)


def _safe(x: np.ndarray) -> np.ndarray:
    return np.where(np.abs(x) < 1e-9, np.nan, x)


def compute_features(points: np.ndarray, mode: str) -> np.ndarray:
    """(N, 17, C) H36M-17 joints -> (N, 14) cue features. NaN propagates for missing joints.

    Cues, all defined identically in both spaces:

    * ``knee_angle`` / ``hip_angle``   -- interior joint angles; smaller knee = deeper.
    * ``thigh_incline``                -- hip->knee segment away from vertical; 90 deg is
      a thigh parallel to the ground, which *is* the squat-depth verdict.
    * ``shank_incline``                -- ankle->knee tilt (knees-forward travel).
    * ``torso_lean``                   -- pelvis->thorax tilt from vertical.
    * ``depth_ratio``                  -- (hip - knee) height / femur; < 0 = below parallel.
    * ``hip_drop_ratio``               -- (hip - ankle) height / leg length; how far the
      hips actually descended, normalised out of body size.
    * ``knee_width_ratio``             -- knee vs ankle lateral separation (valgus proxy).
    * ``knee_over_ankle``              -- horizontal knee-ankle offset / femur.
    """
    pts, bmode = to_biomech_space(points, mode)
    if pts.ndim != 3 or pts.shape[1] < 17:
        raise ValueError(f"expected (N, >=17, C) H36M-17 joints, got {pts.shape}")

    height = _height(pts, bmode)
    femur_r = _seg_len(pts, R_HIP, R_KNEE)
    femur_l = _seg_len(pts, L_HIP, L_KNEE)
    femur = _safe(0.5 * (femur_r + femur_l))
    tibia = 0.5 * (_seg_len(pts, R_KNEE, R_ANKLE) + _seg_len(pts, L_KNEE, L_ANKLE))
    leg = _safe(femur + tibia)

    hip_h = 0.5 * (height[:, R_HIP] + height[:, L_HIP])
    knee_h = 0.5 * (height[:, R_KNEE] + height[:, L_KNEE])
    ankle_h = 0.5 * (height[:, R_ANKLE] + height[:, L_ANKLE])

    cols = {
        "knee_angle_r": joint_angle(pts, R_HIP, R_KNEE, R_ANKLE),
        "knee_angle_l": joint_angle(pts, L_HIP, L_KNEE, L_ANKLE),
        "hip_angle_r": joint_angle(pts, R_SHOULDER, R_HIP, R_KNEE),
        "hip_angle_l": joint_angle(pts, L_SHOULDER, L_HIP, L_KNEE),
        "thigh_incline_r": lean_from_vertical(pts, R_HIP, R_KNEE, bmode),
        "thigh_incline_l": lean_from_vertical(pts, L_HIP, L_KNEE, bmode),
        "shank_incline_r": lean_from_vertical(pts, R_ANKLE, R_KNEE, bmode),
        "shank_incline_l": lean_from_vertical(pts, L_ANKLE, L_KNEE, bmode),
        "torso_lean": lean_from_vertical(pts, ROOT, THORAX, bmode),
        "depth_ratio": (hip_h - knee_h) / femur,
        "hip_drop_ratio": (hip_h - ankle_h) / leg,
        "knee_width_ratio": _h_span(pts, R_KNEE, L_KNEE, bmode) / _safe(_h_span(pts, R_ANKLE, L_ANKLE, bmode)),
        "knee_over_ankle_r": _h_span(pts, R_KNEE, R_ANKLE, bmode) / femur,
        "knee_over_ankle_l": _h_span(pts, L_KNEE, L_ANKLE, bmode) / femur,
    }
    return np.stack([cols[name] for name in FEATURE_NAMES], axis=1)
