# Lunge raw metrics and phase segmentation. Fault rules land in Tasks 3-5.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `lunge_compute_raw` / `lunge_assign_phases`
# compute scale-free per-frame metrics and a phase label only. Every number that decides
# anything (a fire threshold, a severity ramp endpoint, a measurability gate) belongs in a
# `rule_*` function in a later task, not here. The only constant this module defines,
# `_DEGENERATE_LENGTH`, is a division-by-zero guard, never a tunable threshold -- see its
# docstring.
#
# ---------------------------------------------------------------------------------------
# BOTH LEGS, SYMMETRICALLY, RESOLVES NOTHING -- the design decision this module exists to
# encode.
# ---------------------------------------------------------------------------------------
# Unlike push-up's or OHP's raw metrics, which are already single-valued because the body is
# bilaterally symmetric in the movement they score, a lunge is a SPLIT STANCE: one leg is
# genuinely the "lead" (forward, loaded, the one every fault rule is about) and one is the
# "trailing" leg. It would be tempting for `lunge_compute_raw` to resolve that here and emit
# `lead_knee_angle`, `lead_knee_medial_offset_ratio`, etc. It deliberately does not.
#
# `run_detector` (src/pose/movements/base.py) calls `compute_raw` over the WHOLE CLIP, before
# `segment_reps` has split it into per-rep slices and before any rep's bottom frame is known.
# At metric time there is therefore no rep boundary to resolve "which leg is loaded THIS rep"
# against. A per-frame "whichever knee is more flexed right now" heuristic would:
#
#   1. Flicker through setup and recovery, where both knees sit near full extension and the
#      difference between them is landmark noise, not signal -- the "lead" leg would swap
#      randomly frame to frame in exactly the phases where it is least meaningful to ask.
#   2. Corrupt `centered_median` and any other frame-to-frame smoothing: a metric that means
#      "the left knee's offset" on frame 40 and "the right knee's offset" on frame 41 is not
#      one time series, it is two interleaved ones, and averaging across the swap produces a
#      number that describes neither leg.
#
# Lead-side resolution therefore happens in the RULES (Task 3's `resolve_lead_side`), which
# receive a PER-REP slice of `CoreFrame`s and can legitimately ask "which knee was most
# flexed at THIS rep's bottom frame" -- a question that has an answer only once a rep
# boundary exists. Until then, every side-specific metric here is emitted for BOTH legs,
# under `left_*` / `right_*` keys, and nothing chooses between them.
#
# ---------------------------------------------------------------------------------------
# ONE DROPPED LANDMARK SILENCES EVERY LUNGE RULE FOR THAT FRAME.
# ---------------------------------------------------------------------------------------
# `required` below lists both hips, both knees, both ankles, both foot indices and both
# shoulders. If `visible_point` drops any ONE of them the frame is marked `valid=False` and
# carries no metric keys at all -- so every rule that masks on `frame.valid` goes silent for
# that frame, not just the one whose input landmark went missing. This mirrors
# `pushup_compute_raw`'s and `ohp_compute_raw`'s validity gate (see pushup.py's MODULE-WIDE
# SILENCE RISK note): an unmeasurable frame is refused wholesale rather than degraded,
# because a silently-wrong verdict is worse than no verdict. Foot indices are required
# because `knee_forward_ratio` needs the toe-ankle vector for BOTH legs (Task 4 needs both
# sides even though only the lead leg's ratio ends up cited); shoulders because the trunk
# lean does.
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, angle_degrees, midpoint, mean_visibility,
    knee_forward_ratio, distance,
)

# Same generic "lower body" landmark set used across movements for the framework-level
# lower_body_visibility quality field. The NAME is squat-centric (inherited across every
# movement module, including the upper-body ones that carry it awkwardly), but the field it
# feeds (`CoreFrame.lower_body_visibility`) is genuinely lower-body FOR THIS MOVEMENT, unlike
# push-up's or OHP's use of the same name for an upper-body-dominant exercise.
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

LUNGE_METRIC_KEYS: tuple[str, ...] = (
    "left_knee_angle",
    "right_knee_angle",
    "min_knee_angle",
    "left_knee_forward_ratio",
    "right_knee_forward_ratio",
    "left_knee_medial_offset_ratio",
    "right_knee_medial_offset_ratio",
    "pelvis_tilt_signed_deg",
    "trunk_lateral_lean_deg",
    "hip_width",
)

# Below this a length/normalizer is degenerate and the dependent metric is NaN. Same guard
# value pushup.py and overhead_press.py use; not a tunable threshold.
_DEGENERATE_LENGTH = 1e-6


def _medial_offset_ratio(
    points, hip_index: int, knee_index: int, ankle_index: int, mid_hip, hip_width: float
) -> float:
    """Signed offset of one knee from its own hip->ankle line, POSITIVE = toward the mid-hip.

    The frontal-plane knee-abduction proxy the spec asks for ("signed medial offset of the
    knee from the hip-ankle line, normalised by hip width"). No true 3-D abduction angle is
    recoverable from monocular pose, and none is claimed.

    WHY THIS IS FACING-INDEPENDENT, which is what lets the rule avoid gating on `front` /
    `front_oblique` (unreachable in production under allow_front=False): "medial" is defined
    as "toward the mid-hip", and the mid-hip is the midline whether the camera is in front of
    or behind the subject. Nothing here consults `signed_orientation`.
    """
    hip = visible_point(points, hip_index, dims=2)
    knee = visible_point(points, knee_index, dims=2)
    ankle = visible_point(points, ankle_index, dims=2)
    if hip is None or knee is None or ankle is None or mid_hip is None:
        return np.nan
    if not np.isfinite(hip_width) or hip_width <= _DEGENERATE_LENGTH:
        return np.nan

    leg = np.asarray(ankle, dtype=np.float64) - np.asarray(hip, dtype=np.float64)
    leg_length = float(np.linalg.norm(leg))
    if leg_length <= _DEGENERATE_LENGTH:
        return np.nan
    normal = np.asarray([-leg[1], leg[0]], dtype=np.float64) / leg_length

    # Orient the normal toward the midline, so a positive projection means "medial" for
    # whichever leg this is -- the left and right legs point in opposite image-x directions.
    toward_midline = np.asarray(mid_hip, dtype=np.float64) - np.asarray(hip, dtype=np.float64)
    if float(np.dot(normal, toward_midline)) < 0.0:
        normal = -normal

    offset = float(np.dot(np.asarray(knee, dtype=np.float64) - np.asarray(hip, dtype=np.float64), normal))
    return offset / float(hip_width)


def lunge_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
    raw: list[dict] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raw.append({"valid": False})
            continue

        points = landmarks_to_array(frame.get("landmarks"))
        frame_index = int(frame.get("frame_index", 0) or 0)
        time = frame_index / fps if fps > 0 else 0.0
        # Foot indices are required because `knee_forward_ratio` needs the toe-ankle vector;
        # shoulders because the trunk lean does. See the module docstring: one dropped
        # landmark silences EVERY lunge rule for this frame, not just the dependent one.
        required = (
            LEFT_SHOULDER, RIGHT_SHOULDER,
            LEFT_HIP, RIGHT_HIP,
            LEFT_KNEE, RIGHT_KNEE,
            LEFT_ANKLE, RIGHT_ANKLE,
            LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
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

        left_knee_angle = angle_degrees(points, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
        right_knee_angle = angle_degrees(points, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
        finite_knees = [v for v in (left_knee_angle, right_knee_angle) if np.isfinite(v)]
        min_knee_angle = float(min(finite_knees)) if finite_knees else np.nan

        hip_width = distance(points, LEFT_HIP, RIGHT_HIP)
        mid_hip = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)

        left_hip = visible_point(points, LEFT_HIP, dims=2)
        right_hip = visible_point(points, RIGHT_HIP, dims=2)
        # atan2 over |dx|: the magnitude of the horizontal hip separation, never its sign.
        # Using signed dx would flip the whole angle by 180 degrees when the subject turns
        # around, making the metric mean "which hip is lower" only for one facing.
        if left_hip is not None and right_hip is not None:
            dx = abs(float(right_hip[0] - left_hip[0]))
            dy = float(right_hip[1] - left_hip[1])
            pelvis_tilt_signed_deg = (
                float(np.degrees(np.arctan2(dy, dx))) if dx > _DEGENERATE_LENGTH else np.nan
            )
        else:
            pelvis_tilt_signed_deg = np.nan

        if shoulder_mid is not None and mid_hip is not None:
            lean_dy = abs(float(shoulder_mid[1] - mid_hip[1]))
            lean_dx = float(shoulder_mid[0] - mid_hip[0])
            trunk_lateral_lean_deg = (
                float(np.degrees(np.arctan2(lean_dx, lean_dy))) if lean_dy > _DEGENERATE_LENGTH else np.nan
            )
        else:
            trunk_lateral_lean_deg = np.nan

        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "left_knee_angle": left_knee_angle,
                "right_knee_angle": right_knee_angle,
                "min_knee_angle": min_knee_angle,
                "left_knee_forward_ratio": knee_forward_ratio(
                    points, LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX
                ),
                "right_knee_forward_ratio": knee_forward_ratio(
                    points, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX
                ),
                "left_knee_medial_offset_ratio": _medial_offset_ratio(
                    points, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, mid_hip, hip_width
                ),
                "right_knee_medial_offset_ratio": _medial_offset_ratio(
                    points, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, mid_hip, hip_width
                ),
                "pelvis_tilt_signed_deg": pelvis_tilt_signed_deg,
                "trunk_lateral_lean_deg": trunk_lateral_lean_deg,
                "hip_width": hip_width,
            }
        )
    return raw


def lunge_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> descent -> bottom -> ascent, segmented on `min_knee_angle`.

    Mirrors `pushup_assign_phases` (src/pose/movements/pushup.py) and `ohp_assign_phases`
    (src/pose/movements/overhead_press.py), substituting `min_knee_angle` for
    `min_elbow_angle` as the depth signal -- the more-flexed (smaller) of the two knee
    angles is the lunge's depth analogue of the more-flexed elbow. Same fallbacks: an empty
    clip returns an empty list, a clip with no finite depth signal is entirely `unknown`, and
    an invalid frame is `unknown` regardless of where it sits (the validity check precedes
    the setup cutoff, so an occluded frame in the opening 15% is NOT labelled `setup`)."""
    frame_count = len(raw)
    if frame_count == 0:
        return []

    knee_values = np.asarray(
        [float(item.get("min_knee_angle", np.nan)) for item in raw], dtype=np.float32
    )
    valid_knee = knee_values[np.isfinite(knee_values)]
    if valid_knee.size == 0:
        return ["unknown" for _ in raw]

    # The deepest 30% of the rep by knee flexion is the bottom.
    bottom_threshold = float(np.percentile(valid_knee, 30))
    deepest_index = int(np.nanargmin(np.where(np.isfinite(knee_values), knee_values, np.inf)))
    setup_cutoff = max(1, int(frame_count * 0.15))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append("setup")
            continue

        value = knee_values[index]
        if np.isfinite(value) and value <= bottom_threshold:
            phases.append("bottom")
        elif index < deepest_index:
            phases.append("descent")
        else:
            phases.append("ascent")
    return phases
