"""SMPL-24 joint geometry: knee angle, the camera's "up" axis, and the body frame used to
render turntables. Pure numpy; works on a single frame (24, 3) or a batch (T, 24, 3), and
tolerates NaN rows for undetected frames throughout.

Camera coordinates follow NLF's convention: y points DOWN the image. So for an upright person
the "up" direction in camera coordinates is approximately (0, -1, 0), not (0, 1, 0).
"""
from __future__ import annotations

import warnings

import numpy as np

# SMPL-24 joint indices used by this module (the rest of the 24 are unused here).
PELVIS = 0
L_HIP = 1
R_HIP = 2
L_KNEE = 4
R_KNEE = 5
L_ANKLE = 7
R_ANKLE = 8
NECK = 12
L_SHOULDER = 16
R_SHOULDER = 17

# The camera looks down its own +z axis (into the scene), in camera coordinates.
_CAMERA_FORWARD = np.array([0.0, 0.0, 1.0])


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees, where a/b/c are (..., 3) arrays. NaN in, NaN out."""
    u, v = a - b, c - b
    denom = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(u * v, axis=-1) / denom
        cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def knee_angle(joints: np.ndarray) -> np.ndarray | float:
    """Mean of the left and right hip-knee-ankle angle, in degrees.

    ``joints`` is (24, 3) for a single frame or (T, 24, 3) for a batch. A side missing any of
    its three joints (NaN) drops out of that frame's mean; a frame missing both sides is NaN.
    """
    joints = np.asarray(joints, dtype=float)
    left = _angle_deg(joints[..., L_HIP, :], joints[..., L_KNEE, :], joints[..., L_ANKLE, :])
    right = _angle_deg(joints[..., R_HIP, :], joints[..., R_KNEE, :], joints[..., R_ANKLE, :])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.stack([left, right], axis=0), axis=0)


def estimate_up_axis(joints: np.ndarray, knee: np.ndarray) -> np.ndarray:
    """Normalised "up" direction (camera coordinates), from frames the person is standing in.

    ``joints`` is (T, 24, 3); ``knee`` is (T,) degrees (see ``knee_angle``). Averages
    mid-shoulder minus mid-ankle over the frames at or above the 80th percentile knee angle
    (i.e. the straightest-legged frames), then normalises. For an upright, untilted camera
    this points close to (0, -1, 0).
    """
    joints = np.asarray(joints, dtype=float)
    knee = np.asarray(knee, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        threshold = np.nanpercentile(knee, 80)
        standing = knee >= threshold
        mid_shoulder = (joints[standing, L_SHOULDER] + joints[standing, R_SHOULDER]) / 2.0
        mid_ankle = (joints[standing, L_ANKLE] + joints[standing, R_ANKLE]) / 2.0
        up = np.nanmean(mid_shoulder - mid_ankle, axis=0)
    norm = np.linalg.norm(up)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("estimate_up_axis: no standing frames with finite shoulder/ankle joints")
    return up / norm


def body_basis(up: np.ndarray) -> np.ndarray:
    """3x3 matrix whose rows are [right, up, back], a right-handed basis in camera coordinates.

    ``back`` is the camera's forward direction (0, 0, 1), negated, with its component along
    ``up`` removed and renormalised -- i.e. the body's own backward direction, projected to be
    perpendicular to "up". ``right`` completes the right-handed frame as ``cross(up, back)``.
    """
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)
    forward_flat = _CAMERA_FORWARD - np.dot(_CAMERA_FORWARD, up) * up
    back = -forward_flat
    back = back / np.linalg.norm(back)
    right = np.cross(up, back)
    right = right / np.linalg.norm(right)
    return np.stack([right, up, back])


def to_body_frame(points_mm: np.ndarray, pelvis_mm: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Camera-coordinate points (mm), pelvis-centred and re-expressed in the body frame, metres."""
    points_mm = np.asarray(points_mm, dtype=float)
    pelvis_mm = np.asarray(pelvis_mm, dtype=float)
    basis = np.asarray(basis, dtype=float)
    return (points_mm - pelvis_mm) @ basis.T / 1000.0


def turntable_framing(
    bodies_m: list[np.ndarray],
    *,
    height_margin: float = 1.12,
    radius_margin: float = 1.05,
) -> dict[str, float]:
    """One orthographic framing shared by every strip of a job.

    ``bodies_m`` holds each key frame's mesh vertices in body-frame metres (``to_body_frame``
    output: pelvis at the origin, y up). Framing uses the UNION of their extents, so every strip
    is drawn at the same scale and body sizes compare across reps. Each body spins about the
    vertical axis through its pelvis, so the square view of half-extent ``mag`` must fit it at
    EVERY angle: ``mag`` is the larger of the padded half-height (the Phase 0 framing) and the
    padded largest horizontal distance of any vertex from that axis (hands or knees reaching
    forward in a squat would otherwise be clipped side-on).

    Returns ``cy`` (camera centre height) and ``mag``.
    """
    if not bodies_m:
        raise ValueError("turntable_framing: no meshes to frame")
    v = np.concatenate([np.asarray(b, dtype=float).reshape(-1, 3) for b in bodies_m])
    y_lo, y_hi = float(v[:, 1].min()), float(v[:, 1].max())
    radius = float(np.max(np.hypot(v[:, 0], v[:, 2])))
    return {
        "cy": (y_lo + y_hi) / 2.0,
        "mag": max((y_hi - y_lo) / 2.0 * height_margin, radius * radius_margin),
    }


def guide_line_heights(joints_m: np.ndarray) -> dict[str, float]:
    """Mean hip and knee JOINT-CENTRE heights (body-frame metres) for one key frame's guide
    lines. Joint centres, not the hip crease or the top of the knee, so their crossing is not
    the coaching "parallel" cutoff and must never be labelled as one."""
    j = np.asarray(joints_m, dtype=float)
    return {
        "hip_y": float((j[L_HIP, 1] + j[R_HIP, 1]) / 2.0),
        "knee_y": float((j[L_KNEE, 1] + j[R_KNEE, 1]) / 2.0),
    }


def ortho_pixel_y(y: float, cy: float, ymag: float, size: int) -> int:
    """Map a body-frame y (metres) to a pixel row under an orthographic camera centred at ``cy``
    with half-height ``ymag``, in a ``size``x``size`` render."""
    return int(round(size / 2 - (y - cy) / ymag * size / 2))
