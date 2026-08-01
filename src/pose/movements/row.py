# Row (bent-over barbell row) raw metrics and phase segmentation. Fault rules land in
# Tasks 2-5.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `row_compute_raw` / `row_assign_phases` compute
# per-frame quantities and a phase label only. Every number that decides anything belongs in a
# `rule_*` function in a later task. The only constant this module defines, `_DEGENERATE_LENGTH`,
# is a division-by-zero guard, never a tunable threshold.
#
# ---------------------------------------------------------------------------------------
# THE PARENT SPEC'S FIFTH ROW RULE CANNOT BE IMPLEMENTED, AND THIS IS THE PROOF.
# ---------------------------------------------------------------------------------------
# The parent spec (docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md,
# §Row) lists FIVE faults. Four are implemented here. `rounded_thoracolumbar_spine` is not,
# because its detection heuristic is geometrically degenerate under BOTH constructions it
# offers:
#
#   1. "three-point angle at mid-spine using shoulder-midpoint(11,12), a synthesized mid-trunk
#      point = 0.5*(shoulder_mid + hip_mid), and hip-midpoint(23,24)" -- the middle point is BY
#      CONSTRUCTION the midpoint of the segment joining the other two. Three collinear points
#      subtend exactly 180 degrees on every frame of every video. The metric is a constant.
#   2. "Flag flexion if the shoulder-midpoint drops below the straight shoulder-hip line by a
#      normalized sag > 0.04" -- shoulder_mid is an ENDPOINT of that line. Its distance to a
#      line passing through itself is identically zero. The threshold can never be crossed.
#
# The root cause is not a wording slip: MediaPipe Pose has NO thoracic or lumbar landmark, so
# there is no measured point anywhere between the shoulders and the hips, and no sag,
# curvature or three-point spinal angle is computable from this detection model by any
# construction. The spec wrote a proxy requiring a landmark its own detection model (§3) does
# not provide.
#
# NOT SUBSTITUTED, DELIBERATELY. Two monocular signals do carry some trunk-shape information --
# trunk-length foreshortening (dist(shoulder_mid, hip_mid) shrinking as the spine flexes) and
# ear-drop relative to the trunk line. Both are confounded by camera distance and by the hinge
# angle itself, and NEITHER is what the rule's citation (Saeterbakken PMID 26134664, an
# erector-spinae EMG MAGNITUDE result) supports. Shipping either under the spec's fault_id
# would attach a real citation to a metric that citation says nothing about, which is exactly
# the fabrication this project's anti-hallucination rule forbids. Precedent for carrying the
# gap instead: `pushup.rule_scapular_winging`, permanently silent for a weaker reason (a
# view-gate accident rather than a geometric impossibility).
#
# The knowledge graph is NOT the gap: `Row:Trunk Flexion` resolves with a non-empty
# `corrections` bucket ("Maintain Neutral Spine"). The metric is the gap.
#
# ---------------------------------------------------------------------------------------
# ONE DROPPED LANDMARK SILENCES EVERY ROW RULE FOR THAT FRAME.
# ---------------------------------------------------------------------------------------
# `required` below lists both shoulders, both elbows, both wrists and both hips. If
# `visible_point` drops any ONE of them the frame is marked `valid=False` and carries no
# metric keys at all, so every rule that masks on `frame.valid` goes silent for that frame,
# not just the one whose input landmark went missing. This mirrors `pushup_compute_raw`,
# `ohp_compute_raw` and `lunge_compute_raw`: an unmeasurable frame is refused wholesale rather
# than degraded, because a silently-wrong verdict is worse than no verdict.
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, angle_degrees, midpoint, mean_visibility, distance,
)

# Defined locally, matching overhead_press.py: geometry.py exports only the lower-body and
# shoulder/hip constants.
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

# The generic "lower body" set every movement module uses for the framework-level
# `lower_body_visibility` quality field. The name is squat-centric and carries awkwardly for a
# hinged upper-body pull, exactly as it does for OHP and push-up; Row's own rules never consume
# it.
LOWER_BODY_LANDMARKS = (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

ROW_METRIC_KEYS: tuple[str, ...] = (
    "left_elbow_angle",
    "right_elbow_angle",
    "min_elbow_angle",
    "max_elbow_angle",
    "trunk_angle_from_horizontal_deg",
    "left_wrist_hip_dist",
    "right_wrist_hip_dist",
    "mean_wrist_hip_dist",
    "wrist_hip_dist_shoulder_norm",
    "elbow_height_asymmetry",
    "elbow_height_delta_signed",
    "shoulder_tilt",
    "wrist_travel_asymmetry",
    "wrist_accel_norm",
    "trunk_angle_speed_deg_s",
    "shoulder_width",
)

# Below this a length/normalizer is degenerate and the dependent metric is NaN. Same guard
# value pushup.py, overhead_press.py and lunge.py use; not a tunable threshold.
_DEGENERATE_LENGTH = 1e-6


def _derivative(values: Sequence[float], fps: float) -> list[float]:
    """Central-difference time derivative, NaN at both boundaries.

    ONE-SIDED BOUNDARY ESTIMATES ARE REFUSED ON PURPOSE. A forward difference at frame 0 and a
    central difference at frame 1 have different biases; mixing them into one series makes the
    first samples systematically unlike the rest, and `rule_momentum_jerk` compares a PEAK
    against a MEDIAN of exactly this series. NaN propagates through the mask and the frame is
    simply not scored.

    A NaN input (an invalid frame) poisons its two neighbours' derivatives, which is correct:
    a derivative across a hole in the data is not measured, it is guessed.
    """
    count = len(values)
    out = [float(np.nan)] * count
    if fps <= 0 or count < 3:
        return out
    arr = np.asarray(values, dtype=np.float64)
    for index in range(1, count - 1):
        before, after = arr[index - 1], arr[index + 1]
        if np.isfinite(before) and np.isfinite(after):
            out[index] = float((after - before) * fps / 2.0)
    return out


def row_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
    raw: list[dict] = []
    wrist_mid_x: list[float] = []
    wrist_mid_y: list[float] = []
    trunk_angles: list[float] = []

    for frame in frames:
        if not isinstance(frame, dict):
            raw.append({"valid": False})
            wrist_mid_x.append(np.nan)
            wrist_mid_y.append(np.nan)
            trunk_angles.append(np.nan)
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
            wrist_mid_x.append(np.nan)
            wrist_mid_y.append(np.nan)
            trunk_angles.append(np.nan)
            continue

        left_elbow_angle = angle_degrees(points, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        right_elbow_angle = angle_degrees(points, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        finite_elbows = [v for v in (left_elbow_angle, right_elbow_angle) if np.isfinite(v)]
        min_elbow_angle = float(min(finite_elbows)) if finite_elbows else np.nan
        max_elbow_angle = float(max(finite_elbows)) if finite_elbows else np.nan

        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        # BOTH COMPONENTS ABSOLUTE, and that is the whole point: |dx| makes the angle
        # independent of which way the subject faces, |dy| of which point is higher in the
        # image. A signed form would flip by 180 degrees when the lifter turns around, and the
        # torso-rising test would then mean the opposite thing for the other facing. In a
        # bent-over row the shoulders stay above the hips throughout, so no real sign
        # information is discarded. Same reasoning `lunge_compute_raw` applies to its |dx|.
        if shoulder_mid is not None and hip_mid is not None:
            trunk_dx = abs(float(hip_mid[0] - shoulder_mid[0]))
            trunk_dy = abs(float(hip_mid[1] - shoulder_mid[1]))
            trunk_angle = (
                float(np.degrees(np.arctan2(trunk_dy, trunk_dx)))
                if trunk_dx > _DEGENERATE_LENGTH or trunk_dy > _DEGENERATE_LENGTH
                else np.nan
            )
        else:
            trunk_angle = np.nan

        left_wrist_hip = distance(points, LEFT_WRIST, LEFT_HIP)
        right_wrist_hip = distance(points, RIGHT_WRIST, RIGHT_HIP)
        finite_dists = [v for v in (left_wrist_hip, right_wrist_hip) if np.isfinite(v)]
        mean_wrist_hip = float(np.mean(finite_dists)) if finite_dists else np.nan
        wrist_travel_asymmetry = (
            abs(left_wrist_hip - right_wrist_hip)
            if np.isfinite(left_wrist_hip) and np.isfinite(right_wrist_hip)
            else np.nan
        )

        shoulder_width = distance(points, LEFT_SHOULDER, RIGHT_SHOULDER)
        shoulder_norm = (
            mean_wrist_hip / shoulder_width
            if np.isfinite(mean_wrist_hip)
            and np.isfinite(shoulder_width)
            and shoulder_width > _DEGENERATE_LENGTH
            else np.nan
        )

        left_elbow = visible_point(points, LEFT_ELBOW, dims=2)
        right_elbow = visible_point(points, RIGHT_ELBOW, dims=2)
        left_shoulder = visible_point(points, LEFT_SHOULDER, dims=2)
        right_shoulder = visible_point(points, RIGHT_SHOULDER, dims=2)
        # SIGNED companion, positive when the LEFT elbow sits LOWER in the image (larger y).
        # `rule_asymmetric_pull` needs the DIRECTION for its coaching cue and an absolute value
        # cannot supply it; the absolute one stays because that is the quantity the spec states
        # its 0.05 threshold on.
        elbow_height_delta_signed = float(left_elbow[1] - right_elbow[1])
        elbow_height_asymmetry = abs(elbow_height_delta_signed)
        shoulder_tilt = abs(float(left_shoulder[1] - right_shoulder[1]))

        left_wrist = visible_point(points, LEFT_WRIST, dims=2)
        right_wrist = visible_point(points, RIGHT_WRIST, dims=2)
        wrist_mid_x.append(float((left_wrist[0] + right_wrist[0]) / 2.0))
        wrist_mid_y.append(float((left_wrist[1] + right_wrist[1]) / 2.0))
        trunk_angles.append(trunk_angle)

        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "left_elbow_angle": left_elbow_angle,
                "right_elbow_angle": right_elbow_angle,
                "min_elbow_angle": min_elbow_angle,
                "max_elbow_angle": max_elbow_angle,
                "trunk_angle_from_horizontal_deg": trunk_angle,
                "left_wrist_hip_dist": left_wrist_hip,
                "right_wrist_hip_dist": right_wrist_hip,
                "mean_wrist_hip_dist": mean_wrist_hip,
                "wrist_hip_dist_shoulder_norm": shoulder_norm,
                "elbow_height_asymmetry": elbow_height_asymmetry,
                "elbow_height_delta_signed": elbow_height_delta_signed,
                "shoulder_tilt": shoulder_tilt,
                "wrist_travel_asymmetry": wrist_travel_asymmetry,
                "shoulder_width": shoulder_width,
            }
        )

    # DERIVATIVES ARE COMPUTED HERE, IN THE METRIC LAYER, AND THAT IS LOAD-BEARING.
    # `run_detector` median-filters EVERY key in `metric_keys` with a 5-frame window. A median
    # over a POSITION series flattens the acceleration transient `rule_momentum_jerk` exists to
    # find, before the rule ever sees it. Emitting the derivative as the metric means the
    # framework's filter acts on the acceleration -- a defensible low-pass on the quantity of
    # interest instead of an erasure of it. Task 4 pins that a 1-3 frame spike survives.
    accel_x = _derivative(_derivative(wrist_mid_x, fps), fps)
    accel_y = _derivative(_derivative(wrist_mid_y, fps), fps)
    trunk_speed = _derivative(trunk_angles, fps)
    for index, item in enumerate(raw):
        if not item.get("valid"):
            continue
        ax, ay = accel_x[index], accel_y[index]
        item["wrist_accel_norm"] = (
            float(np.hypot(ax, ay)) if np.isfinite(ax) and np.isfinite(ay) else float(np.nan)
        )
        speed = trunk_speed[index]
        item["trunk_angle_speed_deg_s"] = abs(float(speed)) if np.isfinite(speed) else float(np.nan)
    return raw


def row_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> pull -> peak -> lower, segmented on `min_elbow_angle`.

    Mirrors `ohp_assign_phases` and `lunge_assign_phases`, substituting the row's pull depth
    signal. "Return" is not a separate label: after the peak the arms extend and those frames
    are `lower`, the same reduction OHP makes for the press's return. Same fallbacks: an empty
    clip returns an empty list, a clip with no finite signal is entirely `unknown`, and an
    invalid frame is `unknown` regardless of where it sits (the validity check precedes the
    setup cutoff, so an occluded frame in the opening 15% is NOT labelled `setup`).
    """
    frame_count = len(raw)
    if frame_count == 0:
        return []

    elbow_values = np.asarray(
        [float(item.get("min_elbow_angle", np.nan)) for item in raw], dtype=np.float32
    )
    valid_elbow = elbow_values[np.isfinite(elbow_values)]
    if valid_elbow.size == 0:
        return ["unknown" for _ in raw]

    # The most-flexed 30% of the rep is the peak hold.
    peak_threshold = float(np.percentile(valid_elbow, 30))
    deepest_index = int(np.nanargmin(np.where(np.isfinite(elbow_values), elbow_values, np.inf)))
    setup_cutoff = max(1, int(frame_count * 0.15))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append("setup")
            continue

        value = elbow_values[index]
        if np.isfinite(value) and value <= peak_threshold:
            phases.append("peak")
        elif index < deepest_index:
            phases.append("pull")
        else:
            phases.append("lower")
    return phases
