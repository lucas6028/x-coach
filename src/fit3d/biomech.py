"""Squat biomechanics computed identically from 3D world truth or 2D image points.

Every metric is a pure function of a joint array ``points`` with shape (F, 25, C):
``C == 3`` for world-frame ground truth (Z up, metres) or ``C == 2`` for a camera's
image projection (Y down, pixels). The *only* thing that changes between the two is
``mode``; the formulas are shared. That is what makes experiment 2 clean -- any gap
between the 3D-truth reading and a 2D-view reading is pure projection distortion, not
a different metric.

Angles are scale-free (no unit conversion needed between metres and pixels). The
length-based ratios are normalised by a body segment *within the same space*, so they
are dimensionless and comparable across 3D and each 2D view.
"""

from __future__ import annotations

import numpy as np

from src.fit3d.dataset import (
    HEAD, L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER,
    ROOT, THORAX,
)

WORLD3D = "world3d"
IMAGE2D = "image2d"


def _height(points: np.ndarray, mode: str) -> np.ndarray:
    """Vertical coordinate per joint, oriented so 'up' increases. (F, J)."""
    if mode == WORLD3D:
        return points[..., 2]
    if mode == IMAGE2D:
        return -points[..., 1]  # image y grows downward
    raise ValueError(f"unknown mode {mode!r}")


def _horizontal(points: np.ndarray, mode: str) -> np.ndarray:
    """Coordinates spanning the horizontal (ground) plane, perpendicular to gravity. (F, J, h)."""
    if mode == WORLD3D:
        return points[..., :2]      # (x, y); z is up
    if mode == IMAGE2D:
        return points[..., :1]      # image x; image y is up/down
    raise ValueError(f"unknown mode {mode!r}")


def joint_angle(points: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    """Interior angle at joint ``b`` (degrees), formed by a-b-c. Works in 2D or 3D."""
    v1 = points[:, a, :] - points[:, b, :]
    v2 = points[:, c, :] - points[:, b, :]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    cos = np.sum(v1 * v2, axis=1) / np.where(denom < 1e-9, np.nan, denom)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def lean_from_vertical(points: np.ndarray, lower: int, upper: int, mode: str) -> np.ndarray:
    """Angle (degrees) of the ``lower``->``upper`` segment away from the up axis."""
    vec = points[:, upper, :] - points[:, lower, :]
    up = np.zeros(points.shape[-1])
    if mode == WORLD3D:
        up[2] = 1.0
    elif mode == IMAGE2D:
        up[1] = -1.0
    else:
        raise ValueError(f"unknown mode {mode!r}")
    n = np.linalg.norm(vec, axis=1)
    cos = (vec @ up) / np.where(n < 1e-9, np.nan, n)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _segment_len(points: np.ndarray, a: int, b: int) -> np.ndarray:
    return np.linalg.norm(points[:, a, :] - points[:, b, :], axis=1)


def _horizontal_span(points: np.ndarray, a: int, b: int, mode: str) -> np.ndarray:
    h = _horizontal(points, mode)
    return np.linalg.norm(h[:, a, :] - h[:, b, :], axis=1)


def frame_metrics(points: np.ndarray, mode: str) -> dict[str, np.ndarray]:
    """Per-frame squat metrics. Returns dict of (F,) arrays.

    knee_angle / hip_angle  -- mean of left & right (smaller knee angle = deeper).
    torso_lean_deg          -- pelvis->thorax tilt from vertical (sagittal lean is
                               invisible to a front camera: a key view-dependent cue).
    depth_ratio             -- (hip_height - knee_height) / femur_len, mean L/R.
                               > 0 hips above knees, ~0 parallel, < 0 below parallel.
    knee_width_ratio        -- knee lateral separation / ankle lateral separation
                               (valgus/"knees caving" proxy; collapses in a side view).
    """
    points = np.asarray(points, dtype=np.float64)

    knee = 0.5 * (joint_angle(points, R_HIP, R_KNEE, R_ANKLE) + joint_angle(points, L_HIP, L_KNEE, L_ANKLE))
    hip = 0.5 * (joint_angle(points, R_SHOULDER, R_HIP, R_KNEE) + joint_angle(points, L_SHOULDER, L_HIP, L_KNEE))
    torso = lean_from_vertical(points, ROOT, THORAX, mode)

    height = _height(points, mode)
    femur = 0.5 * (_segment_len(points, R_HIP, R_KNEE) + _segment_len(points, L_HIP, L_KNEE))
    femur = np.where(femur < 1e-9, np.nan, femur)
    hip_h = 0.5 * (height[:, R_HIP] + height[:, L_HIP])
    knee_h = 0.5 * (height[:, R_KNEE] + height[:, L_KNEE])
    depth_ratio = (hip_h - knee_h) / femur

    ankle_span = _horizontal_span(points, R_ANKLE, L_ANKLE, mode)
    knee_span = _horizontal_span(points, R_KNEE, L_KNEE, mode)
    knee_width_ratio = knee_span / np.where(ankle_span < 1e-9, np.nan, ankle_span)

    return {
        "knee_angle": knee,
        "hip_angle": hip,
        "torso_lean_deg": torso,
        "depth_ratio": depth_ratio,
        "knee_width_ratio": knee_width_ratio,
    }


# Per-rep reduction: which extreme of each metric is the coaching-relevant reading.
# "min" picks the bottom-of-squat value; "max" the peak.
REP_REDUCERS: dict[str, str] = {
    "knee_angle": "min",        # deepest flexion
    "hip_angle": "min",         # most closed hip
    "torso_lean_deg": "max",    # worst forward lean
    "depth_ratio": "min",       # lowest hip relative to knee
    "knee_width_ratio": "min",  # worst inward collapse
}


def rep_summary(points: np.ndarray, mode: str, start: int, end: int) -> dict[str, float]:
    """Reduce each metric over a rep window [start, end) to its coaching-relevant extreme."""
    seg = points[max(start, 0):end]
    metrics = frame_metrics(seg, mode)
    out: dict[str, float] = {}
    for name, series in metrics.items():
        reducer = np.nanmin if REP_REDUCERS[name] == "min" else np.nanmax
        out[name] = float(reducer(series)) if np.isfinite(series).any() else float("nan")
    return out
