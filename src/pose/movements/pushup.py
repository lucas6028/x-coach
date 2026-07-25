# Push-up raw metrics and phase segmentation.
#
# THIS MODULE CONTAINS NO THRESHOLDS. It computes scale-free per-frame metrics and a phase
# label only; the cited fault rules that consume them arrive in later tasks. Nothing here may
# be read as a validated criterion.
#
# ---------------------------------------------------------------------------------------
# MODULE-WIDE SILENCE RISK -- one dropped landmark silences EVERY push-up rule, not one.
# ---------------------------------------------------------------------------------------
# `pushup_compute_raw` puts both shoulders, both elbows, both wrists, both hips AND BOTH
# ANKLES in its `required` tuple. If `visible_point` drops any one of them the frame is
# marked `valid=False` and carries NO metric keys at all, so every rule that masks on
# `frame.valid` goes silent for that frame. Two concrete consequences worth stating plainly:
#
#   1. ANKLES ARE REQUIRED because the plank line is shoulder-mid -> ankle-mid; without them
#      there is no line to measure hip offset against. A clip framed from the knees up
#      therefore silences ALL push-up rules -- hip sag, depth, hand width, head drop alike --
#      rather than silencing only the plank rules. This is the same validity-gate effect
#      documented for OHP in src/pose/movements/overhead_press.py, widened by two landmarks.
#
#   2. THE PRIMARY PUSH-UP VIEW IS THE ONE MOST LIKELY TO TRIP IT. The spec calls `side`
#      (sagittal) the primary useful view, and sagittal is exactly where the far-side
#      shoulder/hip/ankle get occluded. So the view in which these metrics are most
#      meaningful is also the view in which they are most likely to be refused outright.
#      That trade is deliberate -- see the next note for why.
#
# WHY REFUSE RATHER THAN DEGRADE. `src/pose/view_estimation.py`'s `_visible_midpoint` takes
# the opposite approach for its own purposes, and it was measured to produce a
# plausible-but-wrong answer: with one shoulder occluded the body-axis extent read 0.070
# instead of ~0.60 (8.6x low) with no NaN and no other signal. Every plank metric here
# (hip_offset_ratio, plank_angle_deviation_deg, body_axis_tilt_deg) is built on left/right
# MIDPOINTS and would fail the same silent way. A silently-wrong push-up verdict is worse
# than no verdict, so this module refuses: an unmeasurable frame is `valid=False` with no
# metrics, and an unmeasurable metric inside a valid frame is NaN. Note the two gates are
# NOT the same number -- `geometry.VISIBILITY_THRESHOLD` is 0.50, `view_estimation`'s
# midpoint gate is 0.35. Do not conflate them.
#
# SCOPE OF THAT GUARANTEE -- READ BEFORE QUOTING IT. "NaN, never a degraded number" covers
# LANDMARK DROP-OUT ONLY. `geometry.visible_point` gates on `visibility >= 0.50` and then
# trusts the coordinates completely, so a landmark that MediaPipe reports confidently but
# places wrongly sails straight through and produces a finite, wrong metric with no signal:
#
#     R_shoulder vis=0.49, displaced +0.20 in x  ->  valid=False (no metrics)
#     R_shoulder vis=0.55, displaced +0.20 in x  ->  valid=True,
#                                                    hand_width_ratio 0.447 (truth 1.0),
#                                                    hip_offset_ratio +0.120 (truth +0.100)
#
# That is inherited from the shared `visible_point` gate, not introduced here, and this
# module has no way to second-guess it without inventing a plausibility threshold (out of
# scope). It matters because MediaPipe routinely assigns mid-range visibility to HALLUCINATED
# far-side landmarks in exactly the sagittal view this module calls primary -- the failure
# mode is "confidently wrong", which the validity gate does not catch, and which
# `test_visible_but_misplaced_landmark_is_trusted` pins as a known gap rather than a claim.
#
# The real-world rate at which this silences side-view push-up video is UNMEASURED; the unit
# fixtures here keep every landmark visible except where occlusion is the thing under test.
#
# ---------------------------------------------------------------------------------------
# SIGN CONVENTION for `hip_offset_ratio` (positive = sag, negative = pike).
# ---------------------------------------------------------------------------------------
# The offset is a TRUE PERPENDICULAR deviation from the shoulder-mid -> ankle-mid plank line
# (projected onto the body-axis normal), not a vertical image-y distance. That distinction is
# not cosmetic: on an incline/decline push-up, or with the camera tilted, a raw
# `hip.y - line_y_at_hip_x` over-reads the offset by 1/cos(body tilt), while the perpendicular
# projection is exact.
#
# Image y is used for ONE thing only: ORIENTING that normal (the normal is flipped so its
# y-component is non-negative, i.e. it points groundward). It never MEASURES the offset. This
# is what makes the metric independent of which way the subject faces -- the raw 2D cross
# product's sign flips when the subject turns around, the y-oriented normal's does not -- and
# it matches the spec, which defines sag as "the hip sits toward the ground (larger `y`)"
# (docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md, `pushup_hip_sag`).
#
# HONEST RESIDUAL: the sign is valid only while the camera's +y actually points groundward.
# A rolled or inverted camera inverts it, turning every sag into a pike and vice versa.
#
# DETECTING THAT INVERSION -- and what does NOT detect it:
#
#   `body_axis_tilt_deg` CANNOT. It is folded to [0, 90], so tilt t and 180-t are
#   indistinguishable, which is precisely where the sign flips. Measured:
#       tilt= 89.999  hip_offset_ratio=+0.100  body_axis_tilt_deg=89.999
#       tilt= 90.001  hip_offset_ratio=-0.100  body_axis_tilt_deg=89.999
#       tilt=180.000  hip_offset_ratio=-0.100  body_axis_tilt_deg= 0.000   <-- reads "ideal"
#   A 180-degree-rotated clip therefore reports the PERFECT plank tilt while every sign is
#   backwards. Do not use this metric as an inversion guard; it is a "is the subject
#   horizontal" diagnostic and nothing more.
#
#   A SIGNED BODY-AXIS ANGLE CANNOT EITHER, which is why one is not emitted. It confounds
#   camera inversion with the subject simply facing the other way -- both give +180 deg,
#   with OPPOSITE correctness:
#       faces-other-way, camera upright : signed_axis=+180.0  hip_offset_ratio=+0.100 (right)
#       camera rotated 180 deg          : signed_axis=+180.0  hip_offset_ratio=-0.100 (wrong)
#
#   `hand_offset_ratio` CAN, and is emitted for exactly this purpose. The hands are on the
#   floor, so in ANY genuine push-up the wrists lie on the groundward side of the plank line
#   and the metric is POSITIVE. A negative value means the direction this module believes is
#   groundward is not, so `hip_offset_ratio`'s SIGN (not its magnitude) must not be trusted;
#   a value near zero means the hands are too close to the body axis to arbitrate. It is an
#   anatomical ground reference rather than a gravity assumption, which is why it survives
#   the facing ambiguity above.
#
# No tilt or agreement THRESHOLD is defined here -- thresholds are out of scope for this
# module. The metric is emitted so Tasks 6-7 can gate on it; deciding where to cut is theirs.
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, angle_degrees, midpoint, mean_visibility, mean_finite,
    distance,
)

# MediaPipe indices not already exported by src.pose.geometry.
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_EAR = 7
RIGHT_EAR = 8

# Same generic "lower body" landmark set used across movements for the framework-level
# lower_body_visibility quality field (cf. src.pose.pose_rule_detector.LOWER_BODY_LANDMARKS).
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

PUSHUP_METRIC_KEYS: tuple[str, ...] = (
    "left_elbow_angle",
    "right_elbow_angle",
    "min_elbow_angle",
    "hip_offset_ratio",
    "hand_offset_ratio",
    "plank_angle_deviation_deg",
    "hand_width_ratio",
    "neck_line_angle_deg",
    "body_axis_tilt_deg",
)

# Below this, a length/normalizer is treated as degenerate and the dependent metric is NaN.
# Same guard value OHP uses for its shoulder-width normalizer; not a tunable threshold.
_DEGENERATE_LENGTH = 1e-6


def _interior_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at `b` of the path a -> b -> c, in degrees. Point-based counterpart to
    `geometry.angle_degrees`, which only accepts landmark indices; needed because the plank
    and neck angles are taken at MIDPOINTS, which are not landmarks. NaN if either arm of the
    angle is degenerate."""
    ba = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    bc = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if not np.isfinite(denominator) or denominator <= 1e-8:
        return np.nan
    cosine = float(np.clip(float(np.dot(ba, bc)) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _groundward_normal(axis: np.ndarray) -> np.ndarray | None:
    """Unit normal to a 2D body-axis vector, oriented so it points groundward (non-negative
    image y). Returns None when the axis is degenerate. See the SIGN CONVENTION note at the
    top of this module for why image y orients but never measures."""
    length = float(np.linalg.norm(axis))
    if not np.isfinite(length) or length <= _DEGENERATE_LENGTH:
        return None
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64) / length
    if normal[1] < 0:
        normal = -normal
    return normal


def _neck_line_angle(
    points: np.ndarray | None, ear: int, shoulder: int, axis: np.ndarray | None
) -> float:
    """Angle between the ear->shoulder vector and the BODY AXIS (shoulder-mid -> ankle-mid),
    in degrees. 0 = the head sits exactly on the line the body is supposed to form; the value
    grows as the head leaves it.

    REFERENCED TO THE BODY AXIS, NOT THE SHOULDER->HIP CHORD -- deliberate deviation from the
    spec's wording, because the spec's version double-counts hip sag. The chord rotates when
    the hips drop, so referencing to it manufactures neck deviation out of a perfectly
    on-line head. Measured against the chord, with ear_offset = 0 throughout:

        hip_offset_ratio 0.000 -> neck  0.000
        hip_offset_ratio 0.067 -> neck  7.595
        hip_offset_ratio 0.100 -> neck 11.310      (55% of a full head-drop signal)
        hip_offset_ratio 0.150 -> neck 16.699
        genuine head drop, straight back          -> neck 20.556

    That is not a constant offset Task 7's per-clip baseline could subtract: sag GROWS through
    the descent, so a sagging rep would earn `pushup_hip_sag` AND a spurious
    `pushup_head_drop`, phase-correlated so the two look like they corroborate each other.

    The body axis is the sound reference because a hip sag moves neither endpoint of it: the
    hands and feet are planted, so the shoulders and ankles hold still while the hips drop
    between them. It is also what the spec's own rationale is after -- "the head, spine and
    pelvis ... in a straight line". In a correct plank the chord and the axis ARE the same
    line; they diverge only when the plank is already broken, which is exactly when the neck
    must be measured against the intact reference rather than the broken one. No correction
    factor is applied and no constant is invented -- the reference vector is simply swapped.

    ---- THE MODELING ASSUMPTION THIS RESTS ON. READ BEFORE TRUSTING THE DECOUPLING. ----

    Swapping the reference does not make the metric assumption-free; it TRADES one assumption
    for another, and the trade is only a win if this one holds:

        ASSUMED: as the hips sag, the head stays neutral relative to the BODY AXIS
                 (shoulder-mid -> ankle-mid) -- i.e. the head does not rotate with the
                 dropping torso segment.

    If the truth is the opposite -- the head staying neutral relative to the THORACIC
    (shoulder->hip) chord, rotating with the torso as the hips drop -- then this metric reads,
    with no head fault present at all:

        sag 0.06 -> 11.310 deg     sag 0.09 -> 16.699 deg     sag 0.15 -> 26.565 deg

    which is bit-identical in magnitude to the chord-reference contamination this change
    removed, merely inverted in which posture it penalises. And because the metric is
    UNSIGNED (see below), that reading is indistinguishable from a genuine head drop.
    Symmetrically, a bent body whose ANKLES move (axis rotates, chord does not) reads ~5.7 deg
    at an ankle displacement of 0.06 with the head on the chord.

    Why the axis assumption is nonetheless the better model, stated as reasoning and not as
    fact: in a real sag the hands and feet are planted and the LUMBAR spine hyperextends, so
    the pelvis drops while the thorax -- braced by the arms -- stays put. The head rides on
    the cervical spine atop that thorax, so it moves with the thorax, which has NOT rotated;
    the shoulder->hip chord meanwhile spans both thorax and lumbar and therefore rotates by
    more than any segment the head is attached to. That is an argument from segment
    kinematics, not a measurement: NEITHER reference has been validated against labeled
    push-up video, and a real head presumably lands somewhere between the two.

    This is deliberately NOT corrected for, blended, or interpolated -- doing any of those
    would require a constant nobody has measured. It is stated so Task 7 can decide with the
    trade in view. `test_neck_reference_assumption_is_visible` makes the alternative concrete:
    the fixture's `head_follows="chord"` knob rotates the head with the chord instead of
    translating it, and pins the three numbers above, so the assumption is exercised by the
    suite rather than only described here.

    Computed per side (same-side ear and shoulder) so a sagittal clip showing only ONE ear
    still yields a reading -- an ear midpoint would go NaN there, which is exactly the
    primary view for this cue.

    UNSIGNED (it is an angle between two vectors): a head DROPPED toward the floor and a head
    LIFTED away from it give the same number. Task 7 baselines this per clip; recovering the
    direction is an open item for that task, not something this module invents."""
    if axis is None:
        return np.nan
    ear_point = visible_point(points, ear, dims=2)
    shoulder_point = visible_point(points, shoulder, dims=2)
    if ear_point is None or shoulder_point is None:
        return np.nan
    head_vector = np.asarray(shoulder_point, dtype=np.float64) - np.asarray(
        ear_point, dtype=np.float64
    )
    denominator = float(np.linalg.norm(head_vector) * np.linalg.norm(axis))
    if not np.isfinite(denominator) or denominator <= 1e-8:
        return np.nan
    cosine = float(np.clip(float(np.dot(head_vector, axis)) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def pushup_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
    raw: list[dict] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raw.append({"valid": False})
            continue

        points = landmarks_to_array(frame.get("landmarks"))
        frame_index = int(frame.get("frame_index", 0) or 0)
        time = frame_index / fps if fps > 0 else 0.0
        # Ankles are required: without them there is no plank line at all. See the
        # MODULE-WIDE SILENCE RISK note at the top of this file. Ears are deliberately NOT
        # required -- a briefly occluded head must not invalidate the elbow/plank metrics --
        # so `neck_line_angle_deg` carries its own NaN path instead.
        required = (
            LEFT_SHOULDER, RIGHT_SHOULDER,
            LEFT_ELBOW, RIGHT_ELBOW,
            LEFT_WRIST, RIGHT_WRIST,
            LEFT_HIP, RIGHT_HIP,
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

        left_elbow_angle = angle_degrees(points, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        right_elbow_angle = angle_degrees(points, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        # The MORE-FLEXED (smaller) of the two arms drives depth; NaN if neither is finite.
        finite_elbows = [
            value for value in (left_elbow_angle, right_elbow_angle) if np.isfinite(value)
        ]
        min_elbow_angle = float(min(finite_elbows)) if finite_elbows else np.nan

        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        ankle_mid = midpoint(points, LEFT_ANKLE, RIGHT_ANKLE, dims=2)

        wrist_mid = midpoint(points, LEFT_WRIST, RIGHT_WRIST, dims=2)

        hip_offset_ratio = np.nan
        hand_offset_ratio = np.nan
        plank_angle_deviation_deg = np.nan
        body_axis_tilt_deg = np.nan
        # `axis` stays None unless it is genuinely measurable, so every consumer below --
        # including the neck angle, which references it -- inherits the NaN path for free.
        axis: np.ndarray | None = None
        axis_length = np.nan
        if shoulder_mid is not None and ankle_mid is not None:
            candidate = np.asarray(ankle_mid, dtype=np.float64) - np.asarray(
                shoulder_mid, dtype=np.float64
            )
            candidate_length = float(np.linalg.norm(candidate))
            if np.isfinite(candidate_length) and candidate_length > _DEGENERATE_LENGTH:
                axis, axis_length = candidate, candidate_length

        if axis is not None and shoulder_mid is not None:
            # Diagnostic: 0 deg = body axis lies along image horizontal (a subject lying
            # flat, filmed upright), 90 deg = it projects vertically. Folded to [0, 90] so
            # it does not depend on which way the subject faces -- which ALSO means it
            # cannot see a 180-degree inversion; `hand_offset_ratio` is what does that.
            body_axis_tilt_deg = float(
                np.degrees(np.arctan2(abs(float(axis[1])), abs(float(axis[0]))))
            )
            normal = _groundward_normal(axis)
            if normal is not None:

                def _offset_ratio(point: np.ndarray | None) -> float:
                    """Signed perpendicular offset of `point` from the plank line, in units
                    of the plank line's own length. Positive = groundward."""
                    if point is None:
                        return np.nan
                    vector = np.asarray(point, dtype=np.float64) - np.asarray(
                        shoulder_mid, dtype=np.float64
                    )
                    return float(np.dot(vector, normal) / axis_length)

                hip_offset_ratio = _offset_ratio(hip_mid)
                # Same normal, same normalizer: where the HANDS sit relative to the plank
                # line. The hands are on the floor, so this must be positive in any genuine
                # push-up; a negative value means "groundward" has been resolved backwards
                # and hip_offset_ratio's SIGN is not trustworthy. See the inversion note at
                # the top of this module for why neither body_axis_tilt_deg nor a signed
                # axis angle can substitute for it.
                hand_offset_ratio = _offset_ratio(wrist_mid)

            if hip_mid is not None:
                # The spec's stated equivalent of the offset criterion: how far the
                # shoulder-hip-ankle chain departs from straight.
                plank_angle = _interior_angle(shoulder_mid, hip_mid, ankle_mid)
                plank_angle_deviation_deg = (
                    abs(180.0 - plank_angle) if np.isfinite(plank_angle) else np.nan
                )

        shoulder_width = distance(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        wrist_span = distance(points, LEFT_WRIST, RIGHT_WRIST, dims=2)
        hand_width_ratio = (
            wrist_span / shoulder_width
            if np.isfinite(wrist_span)
            and np.isfinite(shoulder_width)
            and shoulder_width > _DEGENERATE_LENGTH
            else np.nan
        )

        neck_line_angle_deg = mean_finite(
            [
                _neck_line_angle(points, LEFT_EAR, LEFT_SHOULDER, axis),
                _neck_line_angle(points, RIGHT_EAR, RIGHT_SHOULDER, axis),
            ]
        )

        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "left_elbow_angle": left_elbow_angle,
                "right_elbow_angle": right_elbow_angle,
                "min_elbow_angle": min_elbow_angle,
                "hip_offset_ratio": hip_offset_ratio,
                "hand_offset_ratio": hand_offset_ratio,
                "plank_angle_deviation_deg": plank_angle_deviation_deg,
                "hand_width_ratio": hand_width_ratio,
                "neck_line_angle_deg": neck_line_angle_deg,
                "body_axis_tilt_deg": body_axis_tilt_deg,
            }
        )
    return raw


def pushup_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> descent -> bottom -> ascent, segmented on `min_elbow_angle`.

    Mirrors `ohp_assign_phases` (src/pose/movements/overhead_press.py), including its
    fallbacks: an empty clip returns an empty list, a clip with no finite depth signal is
    entirely `unknown`, and an invalid frame is `unknown` regardless of where it sits (the
    validity check precedes the setup cutoff, so an occluded frame in the opening 15% is
    NOT labelled `setup`)."""
    frame_count = len(raw)
    if frame_count == 0:
        return []

    elbow_values = np.asarray(
        [float(item.get("min_elbow_angle", np.nan)) for item in raw], dtype=np.float32
    )
    valid_elbow = elbow_values[np.isfinite(elbow_values)]
    if valid_elbow.size == 0:
        return ["unknown" for _ in raw]

    # The deepest 30% of the rep by elbow flexion is the bottom.
    bottom_threshold = float(np.percentile(valid_elbow, 30))
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
        if np.isfinite(value) and value <= bottom_threshold:
            phases.append("bottom")
        elif index < deepest_index:
            phases.append("descent")
        else:
            phases.append("ascent")
    return phases
