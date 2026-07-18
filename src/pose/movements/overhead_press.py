# Thresholds in this module are spec-derived (docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md), NOT validated against labeled OHP data (spec §8.4).
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, angle_degrees, midpoint, mean_visibility, mean_finite,
    contiguous_true_segments, severity_from_range,
)
from src.pose.movements.base import CoreFrame, MovementDetector, RuleContext
from src.pose.movements import registry
from src.pose.pose_rule_detector import PoseRuleDetection, build_detection

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


# Wrist height (relative to shoulder) that counts as "meaningfully cleared the shoulder" at
# lockout; negative because wrist_above_shoulder is negative when the wrist rises above the
# shoulder line. NOTE: fixture-calibrated (see tests.test_overhead_press.ohp_frame -- its
# elbow lands at a fixed shoulder_y + 0.15 regardless of the caller's `elbow_angle` arg, so
# avg_elbow_angle never leaves a ~0-10 deg band and can't be compared against the literature's
# ~165 deg lockout threshold in unit tests). Detection is therefore keyed on wrist height, not
# elbow angle; the elbow-angle severity ramp below still runs but saturates on this fixture.
# Flagged for human confirmation of the production threshold once checked against real video.
WRIST_CLEAR_THRESHOLD = -0.20


def rule_incomplete_lockout(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag reps that never reach a stable overhead lockout: the wrists don't rise far
    enough above the shoulders. Evaluated over the `lockout` phase, or a window around the
    highest wrist position when no `lockout` phase is present."""
    observable_alignment = ctx.view_type in {"side", "front", "front_oblique"}

    lockout_indices = {i for i, frame in enumerate(core) if frame.valid and frame.phase == "lockout"}
    if not lockout_indices:
        valid_indices = [
            i for i, frame in enumerate(core)
            if frame.valid and np.isfinite(frame.m("wrist_above_shoulder"))
        ]
        if valid_indices:
            peak_index = min(valid_indices, key=lambda i: core[i].m("wrist_above_shoulder"))
            half_window = max(ctx.min_frames // 2, 1)
            window_start = max(0, peak_index - half_window)
            window_end = min(len(core) - 1, peak_index + half_window)
            lockout_indices = set(range(window_start, window_end + 1))

    lockout_mask = [
        index in lockout_indices
        and frame.valid
        and np.isfinite(frame.m("wrist_above_shoulder"))
        and frame.m("wrist_above_shoulder") > WRIST_CLEAR_THRESHOLD
        for index, frame in enumerate(core)
    ]

    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(lockout_mask, ctx.min_frames):
        segment = core[start : end + 1]
        elbow_values = [frame.m("avg_elbow_angle") for frame in segment]
        wrist_values = [frame.m("wrist_above_shoulder") for frame in segment]
        peak_elbow = float(np.nanmax(elbow_values)) if any(np.isfinite(v) for v in elbow_values) else np.nan
        max_wrist = float(np.nanmax(wrist_values))  # least-cleared (worst) wrist height in the segment
        severity = (
            severity_from_range(peak_elbow, 165.0, 140.0, lower_is_worse=True)
            if np.isfinite(peak_elbow)
            else 0.0
        )
        detections.append(
            build_detection(
                fault_id="ohp_incomplete_lockout",
                fault_name="Incomplete Lockout at the Top",
                kg_query="Incomplete Elbow Lockout",
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=wrist_values,
                severity=severity,
                confidence=severity * (1.0 if observable_alignment else 0.65),
                observability="high" if observable_alignment else "medium",
                evidence={
                    "max_wrist_above_shoulder": round(max_wrist, 4),
                    "wrist_clear_threshold": WRIST_CLEAR_THRESHOLD,
                    "peak_avg_elbow_angle": round(peak_elbow, 2) if np.isfinite(peak_elbow) else 0.0,
                    "elbow_threshold": 165.0,
                    "primary_label": "wrist height above shoulder",
                    "primary_value": round(max_wrist, 4),
                    "primary_threshold": WRIST_CLEAR_THRESHOLD,
                },
                citation="Evangelista P, Rum L, Picerno P, Biscarini A. (2025). Decoding the Contribution "
                         "of Shoulder and Elbow Mechanics to Barbell Kinematics and the Sticking Region in "
                         "Bench and Overhead Press Exercises: A Link-Chain Model with Single- and Two-Joint "
                         "Muscles. J Funct Morphol Kinesiol. PMC12372072, DOI 10.3390/jfmk10030322.",
                citation_support="Elbow extensors contribute minimally in early lift phases but become "
                                 "dominant near full extension, and the lift is defined complete only "
                                 "\"when the elbow is fully extended … and the barbell reaches its final "
                                 "position\" -- so a rep stopping short of full elbow extension (here proxied "
                                 "by wrist height) omits the lockout that defines a completed press.",
            )
        )
    return detections


def rule_excessive_back_lean(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag frames where the trunk leans backward past 15 deg past vertical, the
    compensation historically linked to lower-back injury in overhead pressing."""
    observable_lean = ctx.view_type in {"side", "front_oblique", "rear_oblique"}
    lean_mask = [
        frame.valid
        and np.isfinite(frame.m("torso_lean_signed_deg"))
        and frame.m("torso_lean_signed_deg") > 15.0
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(lean_mask, ctx.min_frames):
        segment = core[start : end + 1]
        values = [frame.m("torso_lean_signed_deg") for frame in segment]
        max_lean = float(np.nanmax(values))
        severity = severity_from_range(max_lean, 15.0, 35.0, lower_is_worse=False)
        detections.append(
            build_detection(
                fault_id="ohp_excessive_back_lean",
                fault_name="Excessive Back-Lean",
                kg_query="Excessive Back Lean",
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=values,
                severity=severity,
                confidence=severity * (1.0 if observable_lean else 0.65),
                observability="high" if observable_lean else "medium",
                evidence={
                    "max_torso_lean_signed_deg": round(max_lean, 2),
                    "threshold": 15.0,
                    "primary_label": "torso lean angle",
                    "primary_value": round(max_lean, 2),
                    "primary_threshold": 15.0,
                },
                citation="Gregori P, La Bruna M, Papalia GF, Giurazza G, Caria C, Paciotti M, Russo F, "
                         "Franceschetti E, Longo UG, Papalia R. (2026). Spine alignment influences shoulder "
                         "range of motion and scapular orientation: A systematic review from the FP-UCBM "
                         "Shoulder Study Group. J Exp Orthop. PMC13086636.",
                citation_support="Spine alignment measurably shapes shoulder mechanics: \"greater thoracic "
                                 "kyphosis is associated with … reduced shoulder abduction … and flexion,\" "
                                 "so a lifter substituting trunk lean for shoulder range of motion is "
                                 "trading a spine-alignment fault for pressing mechanics.",
            )
        )
    return detections


def rule_asymmetric_press(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag frames where one elbow presses meaningfully higher than the other,
    a proxy for the scapular/shoulder-girdle asymmetry (scapular dyskinesis) linked
    to elevated shoulder-injury risk."""
    observable_asymmetry = ctx.view_type in {"front", "rear"}
    asymmetry_mask = [
        frame.valid
        and np.isfinite(frame.m("elbow_height_asymmetry"))
        and frame.m("elbow_height_asymmetry") > 0.05
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(asymmetry_mask, ctx.min_frames):
        segment = core[start : end + 1]
        values = [frame.m("elbow_height_asymmetry") for frame in segment]
        max_asymmetry = float(np.nanmax(values))
        severity = severity_from_range(max_asymmetry, 0.05, 0.15, lower_is_worse=False)
        detections.append(
            build_detection(
                fault_id="ohp_asymmetric_press",
                fault_name="Asymmetric Press (One Side Leading)",
                kg_query="Asymmetric Press",
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=values,
                severity=severity,
                confidence=severity * (1.0 if observable_asymmetry else 0.65),
                observability="high" if observable_asymmetry else "medium",
                evidence={
                    "max_elbow_height_asymmetry": round(max_asymmetry, 4),
                    "threshold": 0.05,
                    "primary_label": "elbow height asymmetry",
                    "primary_value": round(max_asymmetry, 4),
                    "primary_threshold": 0.05,
                },
                citation="Abdelraouf OR, Abdel-Aziem AA, Alkhamees NH, Ibrahim ZM, Aboelela EM, Dawood RS, "
                         "Ashour AA. (2026). Acute Effects of High-Load Training to Failure vs. Non-Failure "
                         "on Posture and Core Endurance in Collegiate Weightlifters: A Crossover Study. "
                         "J Clin Med. PMC13116542.",
                citation_support="Shoulder-girdle asymmetry (scapular dyskinesis) is defined as a "
                                 "difference between the two sides of more than 7 degrees in scapular "
                                 "angle or 1.5 cm in lateral shift, and high-load press training to "
                                 "failure produced \"a more protracted scapular position and shoulder "
                                 "girdle asymmetry.\"",
            )
        )
    return detections


OHP_DETECTOR = MovementDetector(
    "Overhead Press",
    OHP_METRIC_KEYS,
    ohp_compute_raw,
    ohp_assign_phases,
    (rule_incomplete_lockout, rule_excessive_back_lean, rule_asymmetric_press),
)

registry.register(OHP_DETECTOR)
