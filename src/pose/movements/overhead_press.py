# Thresholds in this module are spec-derived (docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md), NOT validated against labeled OHP data (spec §8.4).
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, angle_degrees, midpoint, mean_visibility, mean_finite,
)

# MediaPipe indices not already exported by src.pose.geometry.
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_EAR = 7
RIGHT_EAR = 8

# Same generic "lower body" landmark set used across movements (src.pose.pose_rule_detector's
# LOWER_BODY_LANDMARKS) for the framework-level lower_body_visibility quality field; OHP's own
# rules do not consume it.
LOWER_BODY_LANDMARKS = (
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_HEEL,
    RIGHT_HEEL,
    LEFT_FOOT_INDEX,
    RIGHT_FOOT_INDEX,
)

OHP_METRIC_KEYS: tuple[str, ...] = (
    "left_elbow_angle",
    "right_elbow_angle",
    "avg_elbow_angle",
    "wrist_above_shoulder",
    "torso_lean_signed_deg",
    "elbow_height_asymmetry",
    "shoulder_ear_gap",
)


def _y(points: np.ndarray | None, index: int) -> float:
    point = visible_point(points, index, dims=2)
    return float(point[1]) if point is not None else np.nan


def ohp_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
    raw: list[dict] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raw.append({"valid": False})
            continue

        points = landmarks_to_array(frame.get("landmarks"))
        frame_index = int(frame.get("frame_index", 0) or 0)
        time = frame_index / fps if fps > 0 else 0.0
        required = (
            LEFT_SHOULDER, RIGHT_SHOULDER,
            LEFT_ELBOW, RIGHT_ELBOW,
            LEFT_WRIST, RIGHT_WRIST,
            LEFT_HIP, RIGHT_HIP,
        )
        valid = all(visible_point(points, index, dims=2) is not None for index in required)
        if not valid:
            raw.append(
                {
                    "frame_index": frame_index,
                    "time": time,
                    "valid": False,
                    "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                }
            )
            continue

        left_elbow_angle = angle_degrees(points, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        right_elbow_angle = angle_degrees(points, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        avg_elbow_angle = mean_finite([left_elbow_angle, right_elbow_angle])

        left_wrist_y = _y(points, LEFT_WRIST)
        right_wrist_y = _y(points, RIGHT_WRIST)
        left_shoulder_y = _y(points, LEFT_SHOULDER)
        right_shoulder_y = _y(points, RIGHT_SHOULDER)
        wrist_mean_y = mean_finite([left_wrist_y, right_wrist_y])
        shoulder_mean_y = mean_finite([left_shoulder_y, right_shoulder_y])
        wrist_above_shoulder = wrist_mean_y - shoulder_mean_y

        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        if shoulder_mid is not None and hip_mid is not None:
            torso_lean_signed_deg = float(
                np.degrees(
                    np.arctan2(
                        shoulder_mid[0] - hip_mid[0],
                        hip_mid[1] - shoulder_mid[1],
                    )
                )
            )
        else:
            torso_lean_signed_deg = np.nan

        left_elbow_y = _y(points, LEFT_ELBOW)
        right_elbow_y = _y(points, RIGHT_ELBOW)
        elbow_height_asymmetry = (
            abs(left_elbow_y - right_elbow_y)
            if np.isfinite(left_elbow_y) and np.isfinite(right_elbow_y)
            else np.nan
        )

        left_ear_y = _y(points, LEFT_EAR)
        right_ear_y = _y(points, RIGHT_EAR)
        left_gap = left_shoulder_y - left_ear_y if np.isfinite(left_ear_y) else np.nan
        right_gap = right_shoulder_y - right_ear_y if np.isfinite(right_ear_y) else np.nan
        shoulder_ear_gap = mean_finite([left_gap, right_gap])

        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "left_elbow_angle": left_elbow_angle,
                "right_elbow_angle": right_elbow_angle,
                "avg_elbow_angle": avg_elbow_angle,
                "wrist_above_shoulder": wrist_above_shoulder,
                "torso_lean_signed_deg": torso_lean_signed_deg,
                "elbow_height_asymmetry": elbow_height_asymmetry,
                "shoulder_ear_gap": shoulder_ear_gap,
            }
        )
    return raw


def ohp_assign_phases(raw: list[dict]) -> list[str]:
    frame_count = len(raw)
    if frame_count == 0:
        return []

    avg_elbow_values = np.asarray(
        [float(item.get("avg_elbow_angle", np.nan)) for item in raw], dtype=np.float32
    )
    wrist_values = np.asarray(
        [float(item.get("wrist_above_shoulder", np.nan)) for item in raw], dtype=np.float32
    )
    valid_elbow = avg_elbow_values[np.isfinite(avg_elbow_values)]
    valid_wrist = wrist_values[np.isfinite(wrist_values)]
    if valid_elbow.size == 0 and valid_wrist.size == 0:
        return ["unknown" for _ in raw]

    elbow_threshold = float(np.percentile(valid_elbow, 70)) if valid_elbow.size else np.inf
    wrist_threshold = float(np.percentile(valid_wrist, 30)) if valid_wrist.size else -np.inf

    if valid_wrist.size:
        wrist_highest_index = int(np.nanargmin(np.where(np.isfinite(wrist_values), wrist_values, np.inf)))
    else:
        wrist_highest_index = -1

    setup_cutoff = max(1, int(frame_count * 0.15))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append("setup")
            continue

        is_lockout = (
            np.isfinite(avg_elbow_values[index])
            and avg_elbow_values[index] >= elbow_threshold
            and np.isfinite(wrist_values[index])
            and wrist_values[index] <= wrist_threshold
        )
        if is_lockout:
            phases.append("lockout")
        elif index < wrist_highest_index:
            phases.append("press")
        else:
            phases.append("lower")
    return phases
