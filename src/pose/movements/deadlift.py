# Deadlift raw metrics and phase segmentation. Fault rules land in Tasks 2-4.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `deadlift_compute_raw` / `deadlift_assign_phases`
# compute scale-free per-frame metrics and a phase label only. Every number that decides
# anything belongs in a `rule_*` function, not here. `_DEGENERATE_LENGTH` is a
# division-by-zero guard, never a tunable threshold.
#
# ---------------------------------------------------------------------------------------
# THE REP STARTS FLEXED, AND THAT IS WHY A SETUP BASELINE MEANS ANYTHING HERE.
# ---------------------------------------------------------------------------------------
# `DEADLIFT_DETECTOR` sets `rep_start="flexed"` (Task 5) -- the hook `base.py:55` names
# deadlift as the motivating case for. A rep therefore runs floor -> lockout -> floor, so
# the window's OPENING frames are genuinely the bar-on-the-floor setup. Two rules
# (`rule_hips_shoot_up`, `rule_lumbar_flexion`) reference a per-rep setup baseline, which is
# only meaningful because of this. For a movement whose rep starts standing, the same
# baseline would be measuring the wrong end of the lift.
#
# The window also contains the ECCENTRIC. The parent spec's four phases cover only the
# concentric, so a fifth phase `lowering` exists here; without it, return-to-floor frames
# would be labelled `lockout` and `rule_incomplete_lockout` would score the descent.
# `lowering` is excluded from `DEADLIFT_ACTIVE_PHASES`: no rule has literature backing for a
# claim about the eccentric.
#
# ---------------------------------------------------------------------------------------
# EVERY METRIC IS BUILT FROM MIDPOINTS, AND EVERY RULE WANTS A SAGITTAL VIEW.
# ---------------------------------------------------------------------------------------
# Parent spec section 7 item 3 records that `_visible_midpoint` needs BOTH landmarks of a
# pair above 0.35 visibility, and that one occluded shoulder silently reverts body-extent
# measurement to a vertical fallback -- "exactly in the view most likely to trigger it: a
# sagittal (side) view is precisely where far-side landmarks are most often occluded." This
# detector sits squarely in that failure mode. `required` below therefore refuses the frame
# wholesale when any input landmark is missing, matching lunge/pushup/OHP: an unmeasurable
# frame is refused rather than degraded, because a silently-wrong verdict is worse than none.
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, midpoint, mean_visibility,
    line_angle_from_vertical,
)

LOWER_BODY_LANDMARKS = (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

DEADLIFT_METRIC_KEYS: tuple[str, ...] = (
    "hip_angle_deg",
    "knee_angle_deg",
    "torso_pitch_deg",
    "hip_y",
    "torso_len",
)

# `shoulder_y` is deliberately absent. An earlier design emitted it for a hip-vs-shoulder
# rise differential in `rule_hips_shoot_up`; that term was shown to be algebraically
# identical to a trunk-pitch change (see the rule's docstring in Task 3), so nothing consumes
# it.

_DEGENERATE_LENGTH = 1e-6

# Phases in which the deadlift is under load. `lowering` and `setup` are excluded.
DEADLIFT_ACTIVE_PHASES = {"lift_off", "mid_pull", "lockout"}


def deadlift_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
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
            LEFT_HIP, RIGHT_HIP,
            LEFT_KNEE, RIGHT_KNEE,
            LEFT_ANKLE, RIGHT_ANKLE,
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

        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        knee_mid = midpoint(points, LEFT_KNEE, RIGHT_KNEE, dims=2)
        ankle_mid = midpoint(points, LEFT_ANKLE, RIGHT_ANKLE, dims=2)

        hip_angle_deg = _angle_between(shoulder_mid, hip_mid, knee_mid)
        knee_angle_deg = _angle_between(hip_mid, knee_mid, ankle_mid)
        # `line_angle_from_vertical(top, bottom)` takes abs() of both deltas, so this is an
        # UNSIGNED angle in [0, 90] -- it cannot distinguish a forward from a backward lean.
        # Correct for the deadlift, where the trunk only ever pitches forward, and it is why
        # `rule_hips_shoot_up` can compare magnitudes without resolving the subject's facing.
        torso_pitch_deg = line_angle_from_vertical(shoulder_mid, hip_mid)
        torso_len = (
            float(np.linalg.norm(shoulder_mid - hip_mid))
            if shoulder_mid is not None and hip_mid is not None
            else np.nan
        )

        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "hip_angle_deg": hip_angle_deg,
                "knee_angle_deg": knee_angle_deg,
                "torso_pitch_deg": torso_pitch_deg,
                "hip_y": float(hip_mid[1]) if hip_mid is not None else np.nan,
                "torso_len": torso_len,
            }
        )
    return raw


def _angle_between(a: np.ndarray | None, b: np.ndarray | None, c: np.ndarray | None) -> float:
    """Interior angle at `b`, in degrees. NaN when any point is missing or degenerate.

    `geometry.angle_degrees` takes LANDMARK INDICES, not points; these vertices are computed
    midpoints with no index, so the arithmetic is done here rather than reaching for it.
    """
    if a is None or b is None or c is None:
        return float(np.nan)
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    na = float(np.linalg.norm(ba))
    nc = float(np.linalg.norm(bc))
    if na < _DEGENERATE_LENGTH or nc < _DEGENERATE_LENGTH:
        return float(np.nan)
    cosine = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def deadlift_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> lift_off -> mid_pull -> lockout -> lowering, segmented on `hip_angle_deg`.

    Mirrors `lunge_assign_phases`, substituting hip angle for knee angle and inverting the
    sense: a lunge rep is deepest in the middle, a deadlift rep is most EXTENDED in the
    middle.

    THE PHASE CUTOFFS ARE PERCENTILES OF THIS REP'S OWN EXCURSION, NOT ABSOLUTE ANGLES, and
    that is load-bearing rather than stylistic. `rule_incomplete_lockout` scores the `lockout`
    phase, and the fault it detects IS failing to reach extension. An absolute cutoff (say
    "lockout = hip angle above 165 degrees") would give a shallow-finishing rep NO lockout
    frames at all, so the rule would go silent on precisely the reps it exists to catch. A
    percentile guarantees the phase exists for every rep, however badly performed. Same
    reasoning as lunge's `bottom_threshold = np.percentile(valid_knee, 30)`.

    The lockout test precedes the post-peak test deliberately: a lifter standing at lockout
    produces high-angle frames on BOTH sides of the peak frame, and those are lockout, not
    lowering. Checking `index > peak` first would discard half the lockout plateau.
    """
    frame_count = len(raw)
    if frame_count == 0:
        return []

    hip_values = np.asarray(
        [float(item.get("hip_angle_deg", np.nan)) for item in raw], dtype=np.float32
    )
    finite = hip_values[np.isfinite(hip_values)]
    if finite.size == 0:
        return ["unknown" for _ in raw]

    lockout_threshold = float(np.percentile(finite, 75))
    mid_pull_threshold = float(np.percentile(finite, 40))
    peak_index = int(np.nanargmax(np.where(np.isfinite(hip_values), hip_values, -np.inf)))
    setup_cutoff = max(1, int(frame_count * 0.10))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append("setup")
            continue

        value = hip_values[index]
        if np.isfinite(value) and value >= lockout_threshold:
            phases.append("lockout")
        elif index > peak_index:
            phases.append("lowering")
        elif np.isfinite(value) and value >= mid_pull_threshold:
            phases.append("mid_pull")
        else:
            phases.append("lift_off")
    return phases
