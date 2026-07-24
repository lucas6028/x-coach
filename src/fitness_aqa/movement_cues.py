"""Movement-general biomech cues for OHP and BarbellRow, sharing squat's depth-isolation.

``cue_features`` is squat-specific (knee/hip/thigh/depth). OHP faults are elbow and knee;
BarbellRow faults are lumbar rounding and torso angle -- these need upper-body and spine
cues. This module computes a broad, fault-relevant set from the *same* H36M-17 joints and
the *same* ``to_biomech_space`` mapping, so the 2D-vs-3D contrast stays clean: the only
thing that differs between an arm's 2D and 3D reading is still the depth channel.

The cues split by plane on purpose, to test the squat finding across movements:

* **sagittal / angular** -- ``knee_angle``, ``hip_hinge``, ``elbow_angle``,
  ``shoulder_flexion``, ``torso_lean``, ``spine_flexion``. Rotation-invariant interior
  angles or tilt-from-vertical; readable in the image plane, so 2D is expected to suffice.
* **mediolateral** -- ``elbow_flare``, ``wrist_drift``. Horizontal offset of the elbow /
  wrist from directly under the shoulder, normalised by the limb length. This is the
  OHP analogue of squat valgus: it lives in the horizontal (ground) plane, so in an
  oblique view it foreshortens partly onto the camera depth axis and only true 3D reads
  it faithfully. If depth helps *these* cues but not the sagittal ones, the squat
  "cue-axis x dominant-viewpoint" story generalises.
"""

from __future__ import annotations

import numpy as np

from src.fit3d.biomech import joint_angle, lean_from_vertical
from src.fit3d.dataset import (
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    ROOT,
    SPINE,
    THORAX,
)
from src.fitness_aqa.cue_features import CAM3D, IMAGE2D, to_biomech_space

FEATURE_NAMES = (
    "knee_angle_r",
    "knee_angle_l",
    "hip_hinge_r",
    "hip_hinge_l",
    "elbow_angle_r",
    "elbow_angle_l",
    "shoulder_flexion_r",
    "shoulder_flexion_l",
    "torso_lean",
    "spine_flexion",
    "elbow_flare_r",
    "elbow_flare_l",
    "wrist_drift_r",
    "wrist_drift_l",
)

MODES = (IMAGE2D, CAM3D)


def _horizontal(points: np.ndarray, biomech_mode: str) -> np.ndarray:
    return points[..., :2] if biomech_mode == "world3d" else points[..., :1]


def _seg_len(points: np.ndarray, a: int, b: int) -> np.ndarray:
    return np.linalg.norm(points[:, a, :] - points[:, b, :], axis=1)


def _h_span(points: np.ndarray, a: int, b: int, biomech_mode: str) -> np.ndarray:
    h = _horizontal(points, biomech_mode)
    return np.linalg.norm(h[:, a, :] - h[:, b, :], axis=1)


def _safe(x: np.ndarray) -> np.ndarray:
    return np.where(np.abs(x) < 1e-9, np.nan, x)


def compute_features(points: np.ndarray, mode: str) -> np.ndarray:
    """(N, 17, C) H36M-17 joints -> (N, 14) movement cues. NaN propagates for missing joints."""
    pts, bmode = to_biomech_space(points, mode)
    if pts.ndim != 3 or pts.shape[1] < 17:
        raise ValueError(f"expected (N, >=17, C) H36M-17 joints, got {pts.shape}")

    upper_r = _safe(_seg_len(pts, R_SHOULDER, R_ELBOW))
    upper_l = _safe(_seg_len(pts, L_SHOULDER, L_ELBOW))
    arm_r = _safe(_seg_len(pts, R_SHOULDER, R_WRIST))
    arm_l = _safe(_seg_len(pts, L_SHOULDER, L_WRIST))

    cols = {
        # sagittal / angular (rotation-invariant, image-plane readable)
        "knee_angle_r": joint_angle(pts, R_HIP, R_KNEE, R_ANKLE),
        "knee_angle_l": joint_angle(pts, L_HIP, L_KNEE, L_ANKLE),
        "hip_hinge_r": joint_angle(pts, R_SHOULDER, R_HIP, R_KNEE),
        "hip_hinge_l": joint_angle(pts, L_SHOULDER, L_HIP, L_KNEE),
        "elbow_angle_r": joint_angle(pts, R_SHOULDER, R_ELBOW, R_WRIST),
        "elbow_angle_l": joint_angle(pts, L_SHOULDER, L_ELBOW, L_WRIST),
        "shoulder_flexion_r": joint_angle(pts, R_HIP, R_SHOULDER, R_ELBOW),
        "shoulder_flexion_l": joint_angle(pts, L_HIP, L_SHOULDER, L_ELBOW),
        "torso_lean": lean_from_vertical(pts, ROOT, THORAX, bmode),
        "spine_flexion": joint_angle(pts, ROOT, SPINE, THORAX),
        # mediolateral (horizontal-plane; foreshortens into depth in oblique views)
        "elbow_flare_r": _h_span(pts, R_SHOULDER, R_ELBOW, bmode) / upper_r,
        "elbow_flare_l": _h_span(pts, L_SHOULDER, L_ELBOW, bmode) / upper_l,
        "wrist_drift_r": _h_span(pts, R_SHOULDER, R_WRIST, bmode) / arm_r,
        "wrist_drift_l": _h_span(pts, L_SHOULDER, L_WRIST, bmode) / arm_l,
    }
    return np.stack([cols[name] for name in FEATURE_NAMES], axis=1)
