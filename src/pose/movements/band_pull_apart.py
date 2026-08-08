# Band Pull Apart (standing, resistance band) raw metrics and phase segmentation. Fault rules
# land in Tasks 2-5.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `band_pull_apart_compute_raw` /
# `band_pull_apart_assign_phases` compute per-frame quantities and a phase label only. Every
# number that decides anything belongs in a `rule_*` function. The only constant this module
# defines, `_DEGENERATE_LENGTH`, is a division-by-zero guard, never a tunable threshold. This is
# also why `trunk_lean_image_signed_deg` is emitted RAW rather than facing-corrected: the facing
# derivation needs a floor, and a floor is a threshold. Rule 4 does that correction itself.
#
# ---------------------------------------------------------------------------------------
# THIS IS THE FIRST MOVEMENT WHOSE DEFINING EXCURSION IS FRONTAL, NOT SAGITTAL.
# ---------------------------------------------------------------------------------------
# Squat, Lunge, Deadlift, Push-up, OHP and Row all excurse in the sagittal plane -- a knee angle,
# an elbow angle, a hip height, a trunk pitch. The band pull apart's excursion is the hands
# travelling APART in the image plane, which makes the REP SIGNAL ITSELF view-bound rather than
# only the rules: from a pure `side` view the hands overlap, the excursion vanishes, and
# `segment_reps` returns nothing before a single rule runs.
#
# That is safe in production only because of a reachability fact, not by luck:
# `estimate_view_for_pose` is called with `allow_front=False` (src/pose/view_estimation.py:14-16),
# so the reachable labels are {side, rear, rear_oblique, unknown}, and across the 45 real pose
# JSONs in this repository the estimator emitted `rear_oblique` 30 times, `rear` 13, `unknown` 2,
# and `side` effectively never. Wrist spread survives `rear_oblique` foreshortened but present.
#
# ---------------------------------------------------------------------------------------
# ONE DROPPED LANDMARK SILENCES EVERY BAND PULL APART RULE FOR THAT FRAME.
# ---------------------------------------------------------------------------------------
# `required` below lists both ears, both shoulders, both elbows, both wrists and both hips. If
# `visible_point` drops any ONE of them the frame is marked `valid=False` and carries no metric
# keys at all, so every rule that masks on `frame.valid` goes silent for that frame, not just the
# one whose input landmark went missing. This mirrors `pushup_compute_raw`, `ohp_compute_raw`,
# `lunge_compute_raw` and `row_compute_raw`: an unmeasurable frame is refused wholesale rather
# than degraded, because a silently-wrong verdict is worse than no verdict.
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, angle_degrees, midpoint, mean_visibility, distance,
)
from src.pose.movements.base import CoreFrame

# Defined locally, matching row.py and overhead_press.py: geometry.py exports only the
# lower-body and shoulder/hip constants.
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

# The generic "lower body" set every movement module uses for the framework-level
# `lower_body_visibility` quality field. The name is squat-centric and carries awkwardly for a
# standing upper-body band exercise, exactly as it does for OHP, push-up and Row; this module's
# own rules never consume it.
LOWER_BODY_LANDMARKS = (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

BAND_PULL_APART_METRIC_KEYS: tuple[str, ...] = (
    "wrist_spread",
    "shoulder_width",
    "wrist_spread_shoulder_norm",
    "left_shoulder_ear_gap",
    "right_shoulder_ear_gap",
    "shoulder_ear_gap_shoulder_norm",
    "left_elbow_angle",
    "right_elbow_angle",
    "min_elbow_angle",
    "trunk_lean_image_signed_deg",
    "trunk_angle_speed_deg_s",
    "wrist_depth_offset",
)

# Below this a length/normalizer is degenerate and the dependent metric is NaN. Same guard value
# pushup.py, overhead_press.py, lunge.py and row.py use; not a tunable threshold.
_DEGENERATE_LENGTH = 1e-6


def _derivative(values: Sequence[float], fps: float) -> list[float]:
    """Central-difference time derivative, NaN at both boundaries.

    ONE-SIDED BOUNDARY ESTIMATES ARE REFUSED ON PURPOSE. A forward difference at frame 0 and a
    central difference at frame 1 have different biases; mixing them into one series makes the
    first samples systematically unlike the rest. NaN propagates through the mask and the frame
    is simply not scored. A NaN input (an invalid frame) poisons its two neighbours' derivatives,
    which is correct: a derivative across a hole in the data is not measured, it is guessed.
    Copied from `row._derivative`, whose momentum rule needs the identical property.
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


def band_pull_apart_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
    raw: list[dict] = []
    trunk_leans: list[float] = []

    for frame in frames:
        if not isinstance(frame, dict):
            raw.append({"valid": False})
            trunk_leans.append(np.nan)
            continue

        points = landmarks_to_array(frame.get("landmarks"))
        frame_index = int(frame.get("frame_index", 0) or 0)
        time = frame_index / fps if fps > 0 else 0.0
        required = (
            LEFT_EAR, RIGHT_EAR,
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
            trunk_leans.append(np.nan)
            continue

        wrist_spread = distance(points, LEFT_WRIST, RIGHT_WRIST)
        shoulder_width = distance(points, LEFT_SHOULDER, RIGHT_SHOULDER)
        normalizer_ok = np.isfinite(shoulder_width) and shoulder_width > _DEGENERATE_LENGTH
        wrist_spread_shoulder_norm = (
            wrist_spread / shoulder_width
            if np.isfinite(wrist_spread) and normalizer_ok
            else np.nan
        )

        left_shoulder = visible_point(points, LEFT_SHOULDER, dims=2)
        right_shoulder = visible_point(points, RIGHT_SHOULDER, dims=2)
        left_ear = visible_point(points, LEFT_EAR, dims=2)
        right_ear = visible_point(points, RIGHT_EAR, dims=2)
        # Image y grows DOWNWARD, so shoulder_y - ear_y is POSITIVE when the ear sits above the
        # shoulder, and SHRINKS as the shoulder rises toward the ear. The spec states its shrug
        # threshold on exactly this quantity ("gap_peak < gap_setup - 0.03").
        left_shoulder_ear_gap = float(left_shoulder[1] - left_ear[1])
        right_shoulder_ear_gap = float(right_shoulder[1] - right_ear[1])
        mean_gap = float(np.mean([left_shoulder_ear_gap, right_shoulder_ear_gap]))
        # SCALE-FREE DIAGNOSTIC THAT NO RULE FIRES ON. The spec's 0.03 shrug threshold carries no
        # normalizer, so it is raw image units and therefore camera-distance dependent (design
        # spec 4.5). Emitting the normalized companion lets a future validation compare the two
        # WITHOUT any threshold having been moved in the meantime.
        shoulder_ear_gap_shoulder_norm = mean_gap / shoulder_width if normalizer_ok else np.nan

        left_elbow_angle = angle_degrees(points, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        right_elbow_angle = angle_degrees(points, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        finite_elbows = [v for v in (left_elbow_angle, right_elbow_angle) if np.isfinite(v)]
        min_elbow_angle = float(min(finite_elbows)) if finite_elbows else np.nan

        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        # SIGNED, and deliberately NOT facing-corrected here. Positive = the shoulders sit toward
        # +x relative to the hips, in IMAGE coordinates. Which physical direction that is depends
        # on which way the lifter faces, which this layer cannot know and must not guess -- see
        # `_clip_facing_sign` in Task 4.
        if shoulder_mid is not None and hip_mid is not None:
            trunk_lean = float(
                np.degrees(
                    np.arctan2(
                        float(shoulder_mid[0] - hip_mid[0]),
                        float(hip_mid[1] - shoulder_mid[1]),
                    )
                )
            )
        else:
            trunk_lean = np.nan

        left_wrist3 = visible_point(points, LEFT_WRIST, dims=3)
        right_wrist3 = visible_point(points, RIGHT_WRIST, dims=3)
        left_shoulder3 = visible_point(points, LEFT_SHOULDER, dims=3)
        right_shoulder3 = visible_point(points, RIGHT_SHOULDER, dims=3)
        # MediaPipe z is depth relative to the hip midpoint, NEGATIVE toward the camera. A band
        # pull apart holds the band in FRONT of the torso by definition, so the SIGN of this
        # offset identifies which way the lifter faces. Rule 4 reduces it; nothing is decided
        # here. NaN when any z is missing, and identically 0.0 under the RTMPose extraction path
        # (src/pose/rtmpose_pose_extraction.py writes z=0.0 for every landmark) -- both cases are
        # handled by rule 4's floor, not by a branch here.
        if all(p is not None for p in (left_wrist3, right_wrist3, left_shoulder3, right_shoulder3)):
            wrist_depth_offset = float(
                np.mean([left_wrist3[2], right_wrist3[2]])
                - np.mean([left_shoulder3[2], right_shoulder3[2]])
            )
        else:
            wrist_depth_offset = np.nan

        trunk_leans.append(trunk_lean)
        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "wrist_spread": wrist_spread,
                "shoulder_width": shoulder_width,
                "wrist_spread_shoulder_norm": wrist_spread_shoulder_norm,
                "left_shoulder_ear_gap": left_shoulder_ear_gap,
                "right_shoulder_ear_gap": right_shoulder_ear_gap,
                "shoulder_ear_gap_shoulder_norm": shoulder_ear_gap_shoulder_norm,
                "left_elbow_angle": left_elbow_angle,
                "right_elbow_angle": right_elbow_angle,
                "min_elbow_angle": min_elbow_angle,
                "trunk_lean_image_signed_deg": trunk_lean,
                "wrist_depth_offset": wrist_depth_offset,
            }
        )

    # THE DERIVATIVE IS COMPUTED HERE, IN THE METRIC LAYER, AND THAT IS LOAD-BEARING.
    # `run_detector` median-filters EVERY key in `metric_keys` with a 5-frame window. A median
    # over a POSITION/ANGLE series flattens the velocity transient rule 4's whip evidence exists
    # to find, before the rule ever sees it. Emitting the derivative AS the metric means the
    # framework's filter acts on the velocity -- a defensible low-pass on the quantity of
    # interest instead of an erasure of it. Same argument row.py makes for `wrist_accel_norm`.
    trunk_speed = _derivative(trunk_leans, fps)
    for index, item in enumerate(raw):
        if not item.get("valid"):
            continue
        speed = trunk_speed[index]
        item["trunk_angle_speed_deg_s"] = abs(float(speed)) if np.isfinite(speed) else float(np.nan)
    return raw


def band_pull_apart_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> pull -> peak -> return, segmented on `wrist_spread_shoulder_norm`.

    Mirrors `row_assign_phases`, substituting the pull-apart's spread signal and inverting the
    polarity: the row's peak is the MOST-FLEXED 30% of the rep, this movement's peak is the
    WIDEST 30%. Same fallbacks: an empty clip returns an empty list, a clip with no finite signal
    is entirely `unknown`, and an invalid frame is `unknown` regardless of where it sits (the
    validity check precedes the setup cutoff, so an occluded frame in the opening 15% is NOT
    labelled `setup`, which matters because `_setup_baseline` reduces over exactly those frames).
    """
    frame_count = len(raw)
    if frame_count == 0:
        return []

    spread_values = np.asarray(
        [float(item.get("wrist_spread_shoulder_norm", np.nan)) for item in raw], dtype=np.float32
    )
    valid_spread = spread_values[np.isfinite(spread_values)]
    if valid_spread.size == 0:
        return ["unknown" for _ in raw]

    # The widest 30% of the rep is the peak hold.
    peak_threshold = float(np.percentile(valid_spread, 70))
    widest_index = int(np.nanargmax(np.where(np.isfinite(spread_values), spread_values, -np.inf)))
    setup_cutoff = max(1, int(frame_count * 0.15))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append("setup")
            continue

        value = spread_values[index]
        if np.isfinite(value) and value >= peak_threshold:
            phases.append("peak")
        elif index < widest_index:
            phases.append("pull")
        else:
            phases.append("return")
    return phases
