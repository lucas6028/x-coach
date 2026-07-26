# Push-up raw metrics, phase segmentation, and the cited fault rules built on them.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `pushup_compute_raw` / `pushup_assign_phases`
# compute scale-free per-frame metrics and a phase label only. Every number that decides
# anything lives in a `rule_*` function, and those numbers come in TWO CATEGORIES that must
# not be conflated:
#
#   FIRE THRESHOLDS COPIED FROM THE SPEC
#   (docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md, Push-up section):
#   +/-0.06 hip offset, 100 deg elbow, 15 deg neck deviation, 1.6 hand-width ratio.
#
#   FIRE THRESHOLDS THAT ARE *NOT* THE SPEC'S -- exactly one, and it is called out here because
#   it is the easiest kind of number to mistake for a copied one: `rule_head_drop`'s nose-ahead
#   cut, 0.06 of body length. The spec ORs in that cue but quantifies it only as "clearly
#   ahead". The magnitude is borrowed BY ANALOGY from `pushup_hip_sag`'s 0.06, which is a
#   different fault. See that rule's docstring; it is a rule-level choice with reasoning.
#
#   SEVERITY RAMP ENDPOINTS (0.15 hip offset, 140 deg elbow, 35 deg neck, 0.15 nose, 2.2 hand
#   width) are RULE-LEVEL CHOICES MADE HERE. The spec states no severity ramp for ANY push-up
#   fault -- its Push-up section carries no `Severity ramp` line at all, while the
#   Squat/Lunge/Deadlift sections do (e.g. "Severity ramp 0.82 -> 0.70"), so the absence is
#   meaningful rather than a formatting quirk. Verified by FIXED-STRING grep over the section:
#   "0.15", "140", "2.2" and "0.25" do not occur in it, and "35" occurs ONLY inside the DOI
#   string "10.1249/01.mss.0000189317.08635.1b", never as a quantity. Each rule's docstring
#   gives the reasoning for the endpoint it picked. They are ranking curves, not cited
#   quantities.
#
#   A THIRD CATEGORY, used by `rule_elbow_flare`: MEASURABILITY GATES (0.15 shoulder/axis ratio,
#   0.25 wrist separation). Not fault thresholds at all -- they ask whether the metric means
#   anything in this camera geometry rather than whether the lifter did something wrong, and
#   they can only ever SILENCE. Both are rule-level. The 0.25 one is honestly labelled INERT in
#   that rule's docstring; the 0.15 one is the live guard against a sagittal noise/noise ratio.
#
# UNITS, because one collision here has already produced a wrong claim: every `hip_offset*`,
# `nose_ahead*` and `shoulder_axis*` figure quoted anywhere in this file is a RATIO IN BODY
# LENGTHS (shoulder-mid -> ankle-mid), the units the spec's +/-0.06 is in -- never the unit
# fixture's raw landmark displacement, which is 0.6x smaller.
#
# Like OHP's, NONE of these numbers -- any category -- has been validated against labeled
# push-up video (spec §8.4). All five of the spec's push-up rules are now present:
# `rule_hip_sag`, `rule_shallow_depth`, `rule_head_drop`, `rule_elbow_flare` and
# `rule_scapular_winging` -- the last of which is PERMANENTLY SILENT by design (see its
# docstring: MediaPipe has no scapular landmarks, so the spec rates it observability `none`).
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
# `rule_hip_sag` is the consumer of that guard: see its docstring for where it cuts and why
# the cut is the sign boundary itself rather than an invented magnitude.
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, angle_degrees, midpoint, mean_visibility, mean_finite,
    contiguous_true_segments, severity_from_range, distance,
)
from src.pose.movements.base import CoreFrame, MovementDetector, RuleContext
from src.pose.movements import registry
from src.pose.pose_rule_detector import (
    SIDE_VIEW_CONF_THRESHOLD,
    PoseRuleDetection,
    build_detection,
)

# MediaPipe indices not already exported by src.pose.geometry.
NOSE = 0
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
    "shoulder_axis_ratio",
    "neck_line_angle_deg",
    "neck_line_signed_deg",
    "nose_ahead_ratio",
    "body_axis_tilt_deg",
)

# Below this, a length/normalizer is treated as degenerate and the dependent metric is NaN.
# Same guard value OHP uses for its shoulder-width normalizer; not a tunable threshold.
_DEGENERATE_LENGTH = 1e-6

# Phases in which the plank is supposed to be held rigid. `setup` is excluded: the lifter is
# still getting into position there, so a hip off the line is not yet a fault. The spec does
# NOT scope `pushup_hip_sag` to a phase, so this is a RULE-LEVEL call, made to match the
# squat detector's ACTIVE_PHASES precedent (src/pose/movements/squat.py) rather than a
# spec requirement.
PUSHUP_ACTIVE_PHASES = {"descent", "bottom", "ascent"}

# Views in which the spec rates the shallow-depth elbow angle `high`. Defined locally rather
# than imported from overhead_press: the two modules happen to agree today, but they are
# answering different spec lines and must be free to diverge.
DEPTH_OBSERVABLE_VIEWS = {"side", "front_oblique"}

# Views in which the spec rates the neck-line angle `medium` (its TOP tier for that fault --
# it is never rated `high` from any view). Deliberately a SEPARATE constant from
# DEPTH_OBSERVABLE_VIEWS despite holding the same members today: it answers a different spec
# line ("observability: medium -- `side` / `front_oblique`; low from `front`/`rear`") and must
# be free to diverge from it.
NECK_OBSERVABLE_VIEWS = {"side", "front_oblique"}

# Phases in which `rule_head_drop` judges the neck. NARROWER than PUSHUP_ACTIVE_PHASES, and a
# RULE-LEVEL CALL: the spec scopes `pushup_head_drop` to no phase at all. `setup` is excluded
# structurally (it DEFINES the per-clip baseline, so it would read ~0 by construction), and
# `ascent` is excluded because the spec describes the fault as the chin "reach[ing] for the
# floor ahead of the chest" -- a descent/bottom event. A head that drops only on the way back
# up is therefore missed; that is the cost of the narrowing, stated rather than hidden.
HEAD_DROP_PHASES = {"descent", "bottom"}

# Views from which the spec rates hip sag near-`none`. See `rule_hip_sag` for why these are
# hard-gated to silence rather than emitted at reduced confidence.
HEAD_ON_VIEWS = {"front", "rear"}

# Confidence multiplier applied when a rule fires from a view the spec does not rate `high`.
# Same 0.65 already used across squat and OHP -- not a new number.
_OFF_VIEW_CONFIDENCE = 0.65


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
    on-line head. Measured against the chord, with ear_offset = 0 throughout. UNITS: every
    `hip_offset*` figure quoted anywhere in this file is `hip_offset_ratio`, i.e. BODY LENGTHS
    (the units the spec's +/-0.06 threshold is in), NOT the fixture's raw displacement, which is
    0.6x smaller:

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

        sag 0.10 -> 11.310 deg     sag 0.15 -> 16.699 deg     sag 0.25 -> 26.565 deg

    which is bit-identical in magnitude to the chord-reference contamination this change
    removed, merely inverted in which posture it penalises. And because the metric is
    UNSIGNED (see below), that reading is indistinguishable from a genuine head drop.
    Symmetrically, a bent body whose ANKLES move (axis rotates, chord does not) reads ~5.7 deg
    at an ankle displacement of 0.10 of body length with the head on the chord.

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
    LIFTED away from it give the same number. `_signed_neck_line_angle` is the directional
    companion added for `rule_head_drop` -- same magnitude, plus the side of the axis the head
    sits on. This unsigned form is kept because it needs no groundward reference and therefore
    survives a rolled/inverted camera, which the signed form does not."""
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


def _unclipped(value: float, mild: float, severe: float) -> float:
    """`severity_from_range`'s ramp WITHOUT the clip to [0, 1]: 0.0 at the fire threshold, 1.0 at
    the ramp's severe end, and free to exceed 1.0 beyond it. NaN in, NaN out.

    Exists so a per-frame RANKING series can stay monotonic past saturation. Severities cannot
    do that job: `clip01` makes every frame past the severe end read exactly 1.0, and
    `build_detection` picks the peak with `nanargmax`, which returns the FIRST maximum -- so a
    ranking built on severities nominates the first severe frame rather than the worst one. Used
    only for `score_values` (which feeds `peak_frame` / `evidence["peak_time"]`); the reported
    SEVERITY still goes through `severity_from_range` and is still clipped.

    The NaN-in/NaN-out guard is DEFENSIVE rather than load-bearing at today's only call site, and
    saying which is which matters: `contiguous_true_segments` only yields frames where the mask
    was True, so every frame in a segment has at least one axis above its fire threshold, i.e. an
    unclipped value > 0 -- a spurious 0.0 from an unmeasurable axis could never win that max.
    (Verified: a mutant returning 0.0 here is EQUIVALENT and survives the suite by construction.
    The two forms diverge only on all-negative input, which the mask excludes.) The guard is kept
    so the function stays correct if it is ever called outside that invariant."""
    if not np.isfinite(value):
        return np.nan
    span = severe - mild
    if not np.isfinite(span) or span == 0.0:
        return np.nan
    return (value - mild) / span


def _worst_axis(*values: float) -> float:
    """Largest of the given per-axis scores, ignoring NaN; NaN only if every axis is NaN. Plain
    `max` would propagate a NaN from an axis that merely happens to be unmeasurable on this
    frame, which would hand `nanargmax` a hole exactly where one cue is occluded."""
    finite = [value for value in values if np.isfinite(value)]
    return max(finite) if finite else np.nan


def _signed_neck_line_angle(
    points: np.ndarray | None,
    ear: int,
    shoulder: int,
    axis: np.ndarray | None,
    normal: np.ndarray | None,
) -> float:
    """DIRECTIONAL companion to `_neck_line_angle`: identical magnitude, plus the sign that says
    WHICH SIDE of the body axis the head sits on.

        POSITIVE = the ear sits GROUNDWARD of the shoulder    -> head DROPPED toward the floor
        NEGATIVE = the ear sits skyward of the shoulder       -> head LIFTED off the line

    WHY THIS EXISTS. `rule_head_drop` flags a deviation from a per-clip setup baseline, and on
    the UNSIGNED metric that construction is not merely non-directional, it is ACTIVELY
    INVERTED. Worked example: a lifter whose setup baseline is +5 deg (head very slightly
    dropped) then LIFTS the head to -15 deg. Unsigned, that reads 15; deviation from the
    unsigned baseline is 15 - 5 = +10 and grows with the lift, so a big enough head LIFT fires
    as a head DROP. The spec's fault is directional ("the head ... drops so the neck leaves the
    straight line"), so the direction has to be recoverable at the metric layer; it is not
    reconstructible at the rule layer, which sees only `CoreFrame.m(key)` scalars.

    HOW THE SIGN IS OBTAINED. Decompose the ear->shoulder vector in the frame formed by the
    body axis and `_groundward_normal(axis)` -- the SAME normal `hip_offset_ratio` uses -- and
    take `atan2` of the two components. Because the normal is a unit vector orthogonal to the
    axis, `|signed| == unsigned` exactly, per side. (Across sides that identity can break:
    both metrics average their two sides with `mean_finite`, so a clip whose left and right
    readings disagree in SIGN averages toward 0 signed while the unsigned mean stays large.
    Anatomically the two ears are close together, so this is a landmark-error regime, not a
    posture regime -- averaging toward "no directional evidence" is the desired failure.)

    INHERITS THE GROUNDWARD ASSUMPTION, AND THEREFORE THE INVERSION GUARD. The sign is only
    meaningful while the module's idea of groundward really is groundward; a rolled or
    180-degree-rotated clip flips it, turning a head lift into a confident head drop. That is
    the same exposure `hip_offset_ratio` has, so it takes the same remedy: `rule_head_drop`
    requires `hand_offset_ratio > 0.0` before it will emit. See the SIGN CONVENTION note at the
    top of this module.

    Every other property is inherited from `_neck_line_angle` unchanged -- body-axis reference
    (not the shoulder->hip chord), the head-neutral-to-the-axis modeling assumption, per-side
    computation so a sagittal single-ear clip still reads, and NaN whenever the axis, the
    normal or either landmark is unavailable."""
    if axis is None or normal is None:
        return np.nan
    ear_point = visible_point(points, ear, dims=2)
    shoulder_point = visible_point(points, shoulder, dims=2)
    if ear_point is None or shoulder_point is None:
        return np.nan
    head_vector = np.asarray(shoulder_point, dtype=np.float64) - np.asarray(
        ear_point, dtype=np.float64
    )
    axis_length = float(np.linalg.norm(axis))
    denominator = float(np.linalg.norm(head_vector)) * axis_length
    if not np.isfinite(denominator) or denominator <= 1e-8:
        return np.nan
    along = float(np.dot(head_vector, axis)) / axis_length
    # `normal` is already a unit vector, so this IS the perpendicular component. Negated
    # because `head_vector` runs ear -> shoulder: a groundward EAR makes it point skyward.
    across = float(np.dot(head_vector, normal))
    return float(np.degrees(np.arctan2(-across, along)))


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
        # `normal` is hoisted out of the block below for the same reason: the SIGNED neck angle
        # needs the very same groundward normal `hip_offset_ratio` uses, and must go NaN
        # wherever that normal is unavailable rather than fall back to some other reference.
        axis: np.ndarray | None = None
        normal: np.ndarray | None = None
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

        # AXIAL position of the nose relative to the shoulders, in units of the plank line's
        # own length. POSITIVE = the nose sits AHEAD of the shoulder line, toward the head end
        # (the axis runs shoulder-mid -> ankle-mid, so "ahead" is the NEGATIVE axis direction,
        # hence the sign flip). This is the spec's second, OR-ed firing cue for
        # `pushup_head_drop` -- "when nose 0 sits clearly ahead of the shoulder along the body
        # axis" -- and it is a genuinely INDEPENDENT axis, not a restatement of the neck angle:
        # `neck_line_angle_deg` measures the head's departure PERPENDICULAR to the body axis and
        # is exactly blind to translation ALONG it (a head jutted straight forward keeps the
        # ear->shoulder vector parallel to the axis, so the angle stays 0.0 at any jut).
        #
        # ROLL-INVARIANT, unlike the signed neck angle: it is a dot product of two vectors that
        # both rotate with the camera, so a rolled or 180-degree-inverted clip leaves it
        # unchanged. It needs no groundward reference and therefore NO inversion guard -- see
        # `rule_head_drop`, which applies the `hand_offset_ratio` guard to the neck term only.
        nose_ahead_ratio = np.nan
        if axis is not None and shoulder_mid is not None:
            nose_point = visible_point(points, NOSE, dims=2)
            if nose_point is not None:
                nose_vector = np.asarray(nose_point, dtype=np.float64) - np.asarray(
                    shoulder_mid, dtype=np.float64
                )
                nose_ahead_ratio = -float(np.dot(nose_vector, axis)) / (axis_length**2)

        shoulder_width = distance(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        wrist_span = distance(points, LEFT_WRIST, RIGHT_WRIST, dims=2)
        hand_width_ratio = (
            wrist_span / shoulder_width
            if np.isfinite(wrist_span)
            and np.isfinite(shoulder_width)
            and shoulder_width > _DEGENERATE_LENGTH
            else np.nan
        )
        # HOW MUCH TRANSVERSE EXTENT SURVIVED THE PROJECTION: shoulder width in units of the
        # body's own length. This is the normalizer `hand_width_ratio` lacks -- the one that does
        # NOT collapse in a sagittal view. Sagittally the shoulders overlap while the body axis
        # is at its longest, so this goes toward 0 exactly where `hand_width_ratio` degenerates
        # into noise/noise; looking down the body's long axis it does the opposite (the axis
        # foreshortens, the shoulders do not). `rule_elbow_flare` gates on it. It is a
        # measurability quantity, not a fault quantity: nothing about the lifter is wrong when
        # it is small, the camera simply cannot answer the hand-width question.
        shoulder_axis_ratio = (
            shoulder_width / axis_length
            if np.isfinite(shoulder_width)
            and np.isfinite(axis_length)
            and axis_length > _DEGENERATE_LENGTH
            else np.nan
        )

        neck_line_angle_deg = mean_finite(
            [
                _neck_line_angle(points, LEFT_EAR, LEFT_SHOULDER, axis),
                _neck_line_angle(points, RIGHT_EAR, RIGHT_SHOULDER, axis),
            ]
        )
        neck_line_signed_deg = mean_finite(
            [
                _signed_neck_line_angle(points, LEFT_EAR, LEFT_SHOULDER, axis, normal),
                _signed_neck_line_angle(points, RIGHT_EAR, RIGHT_SHOULDER, axis, normal),
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
                "shoulder_axis_ratio": shoulder_axis_ratio,
                "neck_line_angle_deg": neck_line_angle_deg,
                "neck_line_signed_deg": neck_line_signed_deg,
                "nose_ahead_ratio": nose_ahead_ratio,
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


def rule_hip_sag(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag a broken plank line: the hips drop toward the floor (SAG) or rise above the line
    (PIKE) by more than 0.06 of the shoulder-to-ankle length.

    THRESHOLD PROVENANCE -- TWO DIFFERENT CATEGORIES, DO NOT CONFLATE THEM.

      FIRE THRESHOLDS +/-0.06: FROM THE SPEC. `pushup_hip_sag` flags sag "by offset > ~0.06
      of body length" and pike "by the same margin". These are the spec's numbers.

      SEVERITY RAMP 0.06 -> 0.15: A RULE-LEVEL CHOICE MADE HERE. The spec states NO severity
      ramp for this fault -- it has no `Severity ramp` line, and the strings "0.15" and "140"
      appear nowhere in its entire Push-up section. (The Squat section DOES state ramps
      explicitly, e.g. "Severity ramp 0.82 -> 0.70", so the absence is meaningful rather than a
      formatting quirk.) 0.15 is chosen as ~2.5x the fire threshold, so severity reaches 1.0
      at an offset two and a half times the one that merely earns a flag; nothing in the
      literature or the spec fixes that multiple. Treat it as a display/ranking curve, not as a
      cited quantity.

    Neither category has been validated against labeled push-up video (spec section 8.4), but
    only the first can claim spec provenance at all.

    DIRECTION IS PART OF THE VERDICT. Sag and pike share one `fault_id` per the spec, but the
    coaching cue is the exact opposite ("brace, lift the hips" vs "drop the hips"), so
    `evidence["direction"]` records which of the two fired. It is read off the SIGN of the
    frame with the largest ABSOLUTE offset, and `score_values` is the absolute series, so
    `build_detection`'s `peak_frame` is the worst frame in either direction -- passing the
    signed series would make a pike segment nominate its LEAST-piked frame as the peak.

    INVERSION GUARD -- WHY THIS RULE CAN REFUSE TO ANSWER. `hip_offset_ratio`'s sign is only
    meaningful while the module's idea of "groundward" is really groundward; a rolled or
    180-degree-rotated clip inverts it and turns every sag into a confident, full-severity
    PIKE (see the SIGN CONVENTION note at the top of this module, and
    `test_camera_inversion_flips_hip_sign_and_hand_offset_catches_it`). The hands are planted
    on the floor, so `hand_offset_ratio` is POSITIVE in any genuine push-up; the mask therefore
    requires `hand_offset_ratio > 0.0` and stays silent otherwise.

      * The cut is at ZERO -- the sign boundary the guard is defined by -- and not at some
        margin around it, because no margin has been measured and this project does not invent
        one. The honest cost of that: a hand offset arbitrarily close to zero (hands nearly on
        the body axis, e.g. a very steep decline push-up) passes the guard while carrying
        almost no evidential weight. That gap is documented, not closed.
      * A NaN guard value refuses for free (`nan > 0.0` is False). That is deliberate, and
        pinned by `test_nan_hand_offset_guard_refuses_rather_than_assuming`.
      * Refusing matches how this module already handles unmeasurable input: an unmeasurable
        frame gets no metrics rather than a degraded number. A non-directional "your plank is
        broken, direction unknown" detection was considered and rejected -- it would still
        assert the MAGNITUDE, which is exactly as inflated/valid as the sign is, and downstream
        feedback has no way to render a directionless plank fault.

    THE PLANK-ANGLE FORM DOES NOT GATE FIRING. The spec offers "Equivalent: hip angle
    (shoulder-hip-ankle) departs from 180 deg by > ~12 deg" as a RESTATEMENT of the offset
    criterion, not a second way in. It is deliberately not OR-ed into the mask because
    `plank_angle_deviation_deg` is `abs(180 - angle)` and therefore UNSIGNED -- Task 5's
    `test_plank_angle_deviation_grows_with_the_offset_and_is_unsigned` pins sag and pike to
    identical values -- so a frame firing on the angle alone would carry no recoverable
    direction, reintroducing the inverted-feedback failure the guard above exists to prevent.
    The angle is reported in the evidence dict as corroboration only.

    VIEW HANDLING. `high` on `side`, per the spec. `front`/`rear` are HARD-GATED to silence
    rather than emitted at reduced confidence: the spec rates them near-`none` because the
    offset is in-plane, and in a head-on view down the body the shoulder-mid -> ankle-mid
    projection SHORTENS, so `axis_length` shrinks and the normalized offset INFLATES. That is a
    false-positive amplifier, not merely a weak signal, and `_DEGENERATE_LENGTH` guards against
    division by zero, not against inflation. Following `rule_forward_head`'s hard gate in
    src/pose/movements/overhead_press.py. Oblique views keep the module's standard reduced
    confidence.

    THE VIEW GATE KEYS OFF THE CLASSIFIED VIEW LABEL, NOT THE TRUE CAMERA ANGLE, so a WEAKLY
    CLASSIFIED `side` label must not buy the full-confidence treatment. `ctx.view_confidence`
    is therefore required to reach `SIDE_VIEW_CONF_THRESHOLD` (0.20, the existing shared
    constant in src.pose.pose_rule_detector -- no new number) before `side` counts as
    `high`/undiscounted.

    This follows the two OTHER rules that refuse rather than degrade -- `rule_forward_head`
    (overhead_press.py) and squat's `rule_knees_forward` -- both of which pair their hard gate
    with exactly this floor. That is the apt reference class, because like them this rule makes
    a DIRECTIONAL claim (sag vs pike) that a bad view can invert.

    It is applied as a TIER DOWNGRADE, not as silence, and that is where this rule legitimately
    differs from those two: they are sagittal-only, so outside sagittal they have no reading at
    all and must return []. Hip sag additionally has a readable middle tier (obliques), so the
    faithful analogue of "this side label is not trustworthy" is "treat it as an unclassified
    view" -- medium observability, 0.65 confidence -- rather than discarding a measurement that
    the oblique branch would have accepted anyway. The floor can only ever LOWER a confidence;
    it can never create a detection.

    Inert in production today: `view_estimation.score_view` only emits `"side"` at
    `side_score >= 0.62`, well above 0.20, so no current clip can be affected. It is adopted
    for consistency with the sibling hard-gated rules and to close the misclassification gap,
    not to change today's behaviour, and `test_a_weakly_classified_side_view_is_downgraded`
    exercises it directly."""
    if ctx.view_type in HEAD_ON_VIEWS:
        return []
    observable_sag = (
        ctx.view_type == "side" and ctx.view_confidence >= SIDE_VIEW_CONF_THRESHOLD
    )

    sag_mask = [
        frame.valid
        and frame.phase in PUSHUP_ACTIVE_PHASES
        and frame.m("hand_offset_ratio") > 0.0
        and np.isfinite(frame.m("hip_offset_ratio"))
        and (frame.m("hip_offset_ratio") > 0.06 or frame.m("hip_offset_ratio") < -0.06)
        for frame in core
    ]

    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(sag_mask, ctx.min_frames):
        segment = core[start : end + 1]
        # Every frame in the segment passed the mask, which already required a FINITE
        # hip_offset_ratio -- so no NaN can reach here and no NaN-tolerant argmax is needed.
        signed_values = [frame.m("hip_offset_ratio") for frame in segment]
        abs_values = [abs(value) for value in signed_values]
        peak_offset = int(np.argmax(abs_values))
        peak_signed = float(signed_values[peak_offset])
        max_abs = float(abs_values[peak_offset])
        direction = "sag" if peak_signed > 0.0 else "pike"
        severity = severity_from_range(max_abs, 0.06, 0.15, lower_is_worse=False)

        plank_values = [frame.m("plank_angle_deviation_deg") for frame in segment]
        max_plank = (
            float(np.nanmax(plank_values)) if any(np.isfinite(v) for v in plank_values) else 0.0
        )

        detections.append(
            build_detection(
                fault_id="pushup_hip_sag",
                fault_name="Hip sag / broken plank line",
                # Verified to resolve: graph_retrieval.resolve_nodes(..., movement="Push-up")
                # returns "Push-up:Trunk Sagging" for this string.
                kg_query="Trunk Sagging",
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=abs_values,
                severity=severity,
                confidence=severity * (1.0 if observable_sag else _OFF_VIEW_CONFIDENCE),
                observability="high" if observable_sag else "medium",
                evidence={
                    "direction": direction,
                    "peak_hip_offset_ratio": round(peak_signed, 4),
                    "max_abs_hip_offset_ratio": round(max_abs, 4),
                    "threshold": 0.06,
                    # Corroboration only -- unsigned, and does not gate firing. See docstring.
                    "max_plank_angle_deviation_deg": round(max_plank, 2),
                    "plank_angle_deviation_threshold_deg": 12.0,
                    "primary_label": f"hip offset from plank line ({direction})",
                    "primary_value": round(peak_signed, 4),
                    "primary_threshold": 0.06 if direction == "sag" else -0.06,
                },
                citation="Freeman S, Karpowicz A, Gray J, McGill S. Med Sci Sports Exerc (2006). "
                         "DOI 10.1249/01.mss.0000189317.08635.1b.",
                citation_support="The study \"quantify[ied] the normalized amplitudes of the "
                                 "abdominal wall and back extensor musculature\" and \"their impact "
                                 "on spinal loading by calculating spinal compression and torque "
                                 "generation in the L4-5 area,\" finding push-up form drives large "
                                 "differences in L4-L5 spine compression (the one-arm push-up produced "
                                 "\"the highest spine compression\"). This establishes that push-up "
                                 "trunk posture governs lumbar load; a sagging (hyperextended) trunk "
                                 "is the posture that raises passive lumbar loading. Note: the paper "
                                 "measured spine load by variant, not sag angle directly, so the "
                                 "sag->load link is inferred from the loading mechanism it quantifies.",
            )
        )
    return detections


def rule_shallow_depth(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag a partial rep: at the bottom of the descent the elbows never bend past 100 deg.

    THRESHOLD PROVENANCE -- TWO DIFFERENT CATEGORIES, DO NOT CONFLATE THEM.

      FIRE THRESHOLD 100 deg: FROM THE SPEC, taking the conservative end of its band. The spec
      (`pushup_shallow_depth`) flags "min elbow flexion angle > ~100-110 deg (a full rep
      reaches roughly <= 90 deg)". 100 is the LOW end of that band and therefore the strictest
      criterion for firing -- a rep must be shallower than 100 deg to be flagged at all, so
      fewer borderline reps are called faults. Picking 110 instead would flag every rep between
      100 and 110 as well. No number outside the spec's band is used.

      SEVERITY RAMP 100 -> 140 deg: A RULE-LEVEL CHOICE MADE HERE. The spec states NO severity
      ramp for this fault; "140" appears nowhere in its Push-up section. 140 deg is chosen
      because it is far past the spec's own description of a complete rep ("a full rep reaches
      roughly <= 90 deg") -- an elbow that never bends below 140 deg has travelled well under
      half the useful range, which is where "maximally shallow" reasonably saturates. That
      reasoning is an argument, not a measurement, and no source fixes the number.

    Neither category has been validated against labeled push-up video (spec section 8.4), but
    only the first can claim spec provenance at all.

    Scoped to the `bottom` phase, per the spec's "at the bottom frame", and following the
    squat detector's `rule_shallow_depth`, which gates on `phase == "bottom"` the same way.

    METRIC DEVIATION INHERITED FROM TASK 5 (stated, not corrected here): the spec says to take
    the elbow "whichever is more visible"; `pushup_compute_raw` emits `min_elbow_angle`, the
    more FLEXED of the two arms. On a symmetric rep they agree. On an asymmetric one the
    more-flexed arm is the more generous reading, so this rule under-reports depth faults
    rather than over-reporting them -- consistent with the conservative threshold choice above.

    VIEW HANDLING follows the spec exactly: `high` on `side`/`front_oblique`, downgraded (and
    confidence-discounted) elsewhere because a head-on view foreshortens the elbow angle.
    Unlike `rule_hip_sag` there is no hard gate, because a foreshortened elbow angle reads
    LARGER-or-equal, i.e. it degrades toward the un-faulted direction, and the rule makes no
    directional claim that an unknown facing could invert."""
    observable_depth = ctx.view_type in DEPTH_OBSERVABLE_VIEWS

    depth_mask = [
        frame.valid
        and frame.phase == "bottom"
        and np.isfinite(frame.m("min_elbow_angle"))
        and frame.m("min_elbow_angle") > 100.0
        for frame in core
    ]

    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(depth_mask, ctx.min_frames):
        segment = core[start : end + 1]
        values = [frame.m("min_elbow_angle") for frame in segment]
        # Larger = less flexed = shallower = worse.
        shallowest = float(np.nanmax(values))
        severity = severity_from_range(shallowest, 100.0, 140.0, lower_is_worse=False)
        detections.append(
            build_detection(
                fault_id="pushup_shallow_depth",
                fault_name="Shallow depth (partial rep)",
                # Verified to resolve: graph_retrieval.resolve_nodes(..., movement="Push-up")
                # returns "Push-up:Limited Range Of Motion" for this string. (The literal fault
                # name "Shallow depth" resolves only to the generic node "Depth".)
                kg_query="Limited Range Of Motion",
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=values,
                severity=severity,
                confidence=severity * (1.0 if observable_depth else _OFF_VIEW_CONFIDENCE),
                observability="high" if observable_depth else "medium",
                evidence={
                    "max_min_elbow_angle": round(shallowest, 2),
                    "threshold": 100.0,
                    "primary_label": "elbow angle at the bottom",
                    "primary_value": round(shallowest, 2),
                    "primary_threshold": 100.0,
                },
                citation="San Juan JG, Suprak DN, Roach SM, Lyda M. BMC Musculoskelet Disord (2015) "
                         "PMC4327800.",
                citation_support="Measuring elbow kinematics in 5 deg increments across the push-up "
                                 "range, vertical ground-reaction force \"displayed a significant "
                                 "linear decrease across the ROM\" and was \"highest during the "
                                 "traditional PUP at 90 deg ... of elbow flexion and lowest at 20 "
                                 "deg,\" while serratus anterior and other muscle EMG rose across "
                                 "elbow extension. Deeper elbow flexion = higher force/demand, so a "
                                 "shallow rep that never reaches the deep-flexion positions forfeits "
                                 "the largest portion of the stimulus.",
            )
        )
    return detections


def rule_head_drop(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag the head leaving the plank line during the descent/bottom, on EITHER of the spec's
    two OR-ed cues: the neck deviates GROUNDWARD from the lifter's own setup posture by more
    than 15 deg, OR the nose juts AHEAD along the body axis by more than 0.06 of body length
    from that same setup posture.

    BOTH CUES ARE FIRING CRITERIA, NOT ONE CUE PLUS CORROBORATION. The spec's heuristic reads
    "flag `head_drop` when the head deviates below the torso line ... by > ~15 deg, OR when
    nose 0 sits clearly ahead of the shoulder along the body axis". Round 1 of this rule
    implemented only the first half and recorded the omission nowhere, which silently dropped
    the FORWARD-HEAD half of a fault whose own name is "Forward head / neck drop" and whose
    citation (PMC12514857) is about forward head posture specifically. The two axes are
    genuinely independent, not two views of one quantity: `neck_line_signed_deg` measures the
    head's departure PERPENDICULAR to the body axis and is exactly blind to translation ALONG
    it. Measured through the real metric layer, on a head jutted straight forward with the neck
    left on the line:

        jut 0.00 of body length -> signed neck  0.0000 deg   nose_ahead deviation 0.000
        jut 0.10 of body length -> signed neck  0.0000 deg   nose_ahead deviation 0.100
        jut 0.25 of body length -> signed neck  0.0000 deg   nose_ahead deviation 0.250

    i.e. the neck criterion reads exactly 0.0 at ANY jut magnitude and can never fire on it.
    `pushup_compute_raw` emits `nose_ahead_ratio` (landmark 0, which was available and unused)
    for this second axis. NOTE the spec's parenthetical for the FIRST cue mentions "nose/ear
    `y`" as well as the ear; this rule reads the ear only there, which is Task 5's choice and
    unchanged -- the nose enters only through the AXIAL cue, where the spec names it explicitly.

    THRESHOLD PROVENANCE -- THREE DIFFERENT CATEGORIES, DO NOT CONFLATE THEM.

      FIRE THRESHOLD 15 deg (neck axis): FROM THE SPEC. `pushup_head_drop` flags the head
      deviating from the torso line "by > ~15 deg". That is the spec's number, used unmodified.

      FIRE THRESHOLD 0.06 of body length (nose axis): A RULE-LEVEL CHOICE MADE HERE. The spec
      says only "clearly ahead" and gives NO number for it -- there is nothing to copy, and
      inventing one and calling it spec-backed is exactly what this module must not do. The
      magnitude is borrowed BY ANALOGY from the one positional criterion the spec does quantify
      for this movement in these units: `pushup_hip_sag`'s "> ~0.06 of body length". The
      argument is that the spec is willing to call 0.06 body lengths a visible positional
      departure elsewhere in the same movement, so it is a defensible reading of "clearly". That
      is an ANALOGY, not provenance: the spec never applies 0.06 to the head, and no source
      fixes it. It is labelled here so nobody can later cite it as the spec's.

      SEVERITY RAMPS 15 -> 35 deg and 0.06 -> 0.15 of body length: RULE-LEVEL CHOICES MADE HERE.
      The spec states NO severity ramp for this fault -- its Push-up section has no
      `Severity ramp` line, and a FIXED-STRING grep finds "35" in that section only inside the
      DOI "10.1249/01.mss.0000189317.08635.1b", never as a quantity, while "0.15" does not occur
      at all. The module has no ramp convention to appeal to either (hip sag runs 2.5x its fire
      threshold, shallow depth 1.4x), so the multiple is genuinely unfixed. 35 deg was picked
      because it is a bit over 2x the flag threshold -- a head more than a third of a right
      angle off the line the lifter themselves set is plainly gross rather than marginal -- and,
      stated plainly rather than dressed up, because it puts the ramp midpoint at exactly
      25.0 deg so one exact-severity test pins BOTH endpoints. The nose ramp 0.06 -> 0.15
      continues the same analogy as its fire threshold (it is hip sag's ramp verbatim). Neither
      reason is a measurement. Treat both as display/ranking curves, not cited quantities.

    TWO AXES, ONE VERDICT: the mask ORs them and the severity is the MAX of the two axes'
    severities, following the spec's OWN two-axis idiom for the squat's shallow depth
    ("Severity ramps: hip axis ... knee-angle axis ...; take the max") and its implementation in
    `rule_shallow_depth` (src/pose/movements/squat.py). `evidence["criterion"]` records which
    cue fired -- `neck_angle`, `nose_ahead` or `both` -- because the coaching cue differs ("stop
    reaching your chin at the floor" vs "pull your head back over your shoulders"), and the
    `primary_*` display fields follow whichever axis drove the severity. On an exact tie the neck
    axis wins the display (`>=`), pinned by `test_the_display_tie_break_favours_the_neck_axis` --
    an arbitrary but fixed choice, so the label cannot silently flip.

      ONE IMPROVEMENT ON THE SQUAT PRECEDENT, AND THE TRAP IN IT. `score_values` is a PER-FRAME
      series rather than the constant squat's `rule_shallow_depth` passes, so
      `build_detection`'s `peak_frame` is the genuinely worst frame instead of always frame 0.
      But the series must be UNCLIPPED to deliver that, and a first cut of this rule used
      per-frame SEVERITIES, which are clipped: every frame past the ramp's severe end reads
      exactly 1.0, and `nanargmax` returns the FIRST maximum, so the peak silently regressed to
      "first severe frame" precisely in the severe regime where the field matters. Measured on
      deviations [16, 40, 36, 60, 17, 16, 16], it nominated the 40-degree frame over the
      60-degree one. `_unclipped` is what fixes it, and
      `test_peak_frame_is_the_worst_frame_even_when_the_ramp_saturates` pins the saturated
      regime specifically -- a sub-saturation fixture passes either way and proves nothing.

    SPEC DEVIATION 1 -- PER-CLIP BASELINE INSTEAD OF AN ABSOLUTE READING, ON BOTH AXES. The
    spec's heuristic reads as absolute on both cues ("deviates below the torso line ... by
    > ~15 deg", "nose 0 sits clearly ahead of the shoulder") but supplies no absolute reference
    for what "neutral" is on either. It cannot: the ear-to-shoulder vector's resting angle
    varies with neck length, ear position, hairline and camera height, so an absolute cut would
    fault some anatomies at rest and excuse others at extremes.

      The NOSE axis needs the baseline even more badly, and this is measurable rather than
      arguable: in a perfectly neutral plank the nose is ALREADY well ahead of the shoulder line
      -- that is simply where a head sits on a horizontal body. Measured on the neutral fixture,
      `nose_ahead_ratio = 0.1833` with no fault present, i.e. 3x an absolute 0.06 cut. Read
      absolutely, "nose sits ahead of the shoulder" fires on every correct push-up ever filmed.
      Only the CHANGE from the lifter's own setup posture carries information.

    This rule therefore measures each axis's deviation from its MEAN over the clip's own `setup`
    frames. That is not an invention of this task -- it mirrors the spec's OWN construction for
    the squat's heel rise ("Establish a `setup` baseline (mean over setup frames)") and the
    implementation of it already in this repo, `rule_heel_rise` in src/pose/movements/squat.py,
    right down to using the mean rather than the median.

      HONEST COST, which is the direct consequence of baselining: the rule measures CHANGE, not
      POSTURE. A lifter who sets up with their head already dropped and simply holds it there
      reads a deviation of ~0 and is never flagged. The spec's absolute phrasing would have
      caught that lifter; the price of catching them is faulting long-necked lifters at rest,
      and this rule chooses the false-negative side. It is pinned by
      `test_a_head_drop_held_from_setup_is_invisible`, so it stays a known limitation rather
      than a surprise.

      SECOND HONEST COST, inherited from the framework: the baseline reads the SMOOTHED metric.
      `run_detector` runs a centred median (window 5) over every metric key before the rules
      see it, so near the clip's start the setup-phase values are blended with the first
      descent frames -- the baseline is slightly contaminated by the very motion it is meant to
      be independent of. `rule_heel_rise` has exactly this property and shipped with it; it is
      documented here rather than fixed, because fixing it means rules reading unsmoothed
      metrics, which is a framework change, not a rule change.

      If the clip has NO usable setup frames the baseline is NaN and that AXIS emits nothing.
      Refusing matches how the whole module handles unmeasurable input; a fallback baseline of 0
      would silently reinstate the absolute criterion this deviation exists to avoid -- and on
      the nose axis a 0 fallback would fire on every clip, per the 0.1833 measurement above. The
      two baselines are independent: an occluded nose through the setup window silences the nose
      axis while leaving the neck axis working, and vice versa.

    SPEC DEVIATION 2 -- A SIGNED METRIC WAS ADDED TO SUPPORT THIS RULE. Task 5's
    `neck_line_angle_deg` is UNSIGNED, and a baseline on an unsigned angle is not merely
    non-directional, it is ACTIVELY INVERTED: with a baseline of +5 deg, a head LIFTED to
    -15 deg reads unsigned 15 and deviation +10, i.e. a head lift reported as a head drop, more
    severely the more the lifter lifts. `neck_line_signed_deg` (see `_signed_neck_line_angle`)
    was therefore added to the metric layer, positive = groundward = dropped. The direction was
    not reconstructible at the rule layer: rules see only `CoreFrame.m(key)` scalars, and the
    groundward normal that carries the sign lives inside `pushup_compute_raw`.

    INVERSION GUARD, SHARED WITH `rule_hip_sag` -- BUT ON THE NECK TERM ONLY. The neck sign
    comes from the same groundward normal `hip_offset_ratio` uses, so it has the same exposure:
    a rolled or 180-degree-rotated clip inverts it and would report every head LIFT as a
    confident head DROP. That term therefore carries the identical `hand_offset_ratio > 0.0`
    guard -- the hands are planted on the floor, so that ratio is positive in any genuine
    push-up -- and stays silent otherwise, NaN guard included (`nan > 0.0` is False).

      The NOSE term is deliberately NOT guarded, because it does not need to be:
      `nose_ahead_ratio` is a dot product of two vectors that both rotate with the camera, so it
      is invariant to roll and to a 180-degree inversion (measured: identical to 6 decimals at
      tilt 0, 37 and 180 deg). Guarding it would silence a working measurement for a hazard it
      does not have. The practical effect is that an inverted clip loses the neck cue and keeps
      the forward-head cue, which is the correct partial answer rather than a blanket refusal.
      See the SIGN CONVENTION note at the top of this module.

      WHAT THE ROLL TESTS DO *NOT* PROVE, because they look stronger than they are: an in-plane
      rotation is a similarity transform, and this metric is a ratio of two lengths measured
      along the same rotating axis, so invariance under it is close to arithmetically trivial.
      The interesting question is a NON-similarity transform -- an oblique camera, which
      foreshortens along one direction only. Probed during review:

        * A CONSTANT oblique view cannot forge a deviation at all -- exactly 0.000000 at
          x-compressions 0.70 / 0.40 / 0.20 and tilts 0 / 30 / 55 deg. The per-clip baseline
          cancels any FIXED linear map, whatever it is, because both the baseline and the
          reading pass through the same map.
        * A spurious fire therefore needs BOTH an intra-rep CHANGE in foreshortening AND the
          nose sitting off the body axis on the sky side. At a 20% intra-rep change the worst
          drift measured is +0.034, still under the 0.06 cut even with the head 0.12 off-axis;
          crossing the cut needs roughly a 40% foreshortening change plus a nose ~0.12 above the
          axis. For a rigid body under a fixed camera that combination is not plausible.

      So the invariance this rule actually relies on is the baseline's cancellation of a fixed
      projection, not the roll tests -- stated here so nobody cites the easy result for the hard
      claim.

    THE MODELING ASSUMPTION THIS RULE INHERITS, AND WHICH FAULT IT CONTAMINATES. The neck angle
    is referenced to the BODY AXIS, which assumes the head stays neutral to that axis rather
    than rotating with the shoulder->hip torso segment as the hips move. If a given lifter's
    head in fact rides the torso segment, a plank fault with no head fault at all produces a
    reading. MEASURED on the `head_follows="chord"` fixture, with the SIGN this rule reads.

    UNITS: `hip_offset_ratio`, i.e. BODY LENGTHS -- the same units the spec's +/-0.06 sag/pike
    threshold is in, and the same units `_neck_line_angle`'s table above uses. (Round 1 quoted
    this table in the FIXTURE's raw displacement instead, which is 0.6x smaller, and drew a
    wrong conclusion from it. The two unit systems are now unified across this file; anything
    quoted as `hip_offset_*` anywhere here is a body-length ratio.)

        hip_offset_ratio +0.10 (sag) -> signed neck -11.31    -0.10 (pike) -> +11.31
        hip_offset_ratio +0.15 (sag) -> signed neck -16.70    -0.15 (pike) -> +16.70
        hip_offset_ratio +0.25 (sag) -> signed neck -26.57    -0.25 (pike) -> +26.57

    So the contaminated direction is the OPPOSITE of the intuitive one, and the sign is what
    reveals it: under the chord model a SAG rotates the head skyward relative to the axis, which
    this drop-only rule reads as a LIFT and does not fire on -- it instead MASKS a genuine head
    drop occurring at the same time (a false negative). The false POSITIVE belongs to a PIKE,
    which must reach `hip_offset_ratio` -0.1340 to forge the 15 deg threshold unaided -- 2.23x
    the spec's own -0.06 pike threshold, so a pike is already being reported as a plank fault
    well before it can forge a head verdict. Task 5's unsigned metric could not tell those two
    cases apart at all, so going signed removed one contamination path and made the surviving
    one legible; it did not remove both. The per-clip baseline does not cancel either, because
    the plank fault grows through the descent while the baseline is fixed at setup. Pinned by
    `test_a_chord_following_head_contaminates_the_pike_direction`, and stated as a limitation of
    the metric rather than corrected -- correcting it needs a constant nobody has measured.

    VIEW HANDLING follows the spec exactly, including its ceiling: `medium` on
    `side`/`front_oblique` -- the spec never rates this fault `high` from ANY view -- and `low`
    (with the module's standard 0.65 confidence discount) from head-on views, which the spec
    rates `low` rather than `none`. Note `run_detector` sorts `low`-observability detections
    behind everything else, so a head-on head-drop lands at the bottom of the list.

      NO HARD GATE, unlike `rule_hip_sag`: a head-on view foreshortens the neck angle toward
      zero, which degrades toward the un-faulted direction, so it is a weak signal rather than a
      false-positive amplifier.

      NO `SIDE_VIEW_CONF_THRESHOLD` CONFIDENCE FLOOR either, and that omission is deliberate
      rather than an oversight. `rule_hip_sag` carries the floor because a mislabelled `side`
      could be a head-on view in which its normalized offset INFLATES. Here the directional
      hazard is camera ROLL, which the `hand_offset_ratio` guard already covers and which no
      view label can see; view misclassification only moves the magnitude, and `side` and
      `front_oblique` are treated identically anyway, so a floor would change nothing but a
      cosmetic confidence number on a weakly-labelled `side`. An unjustified knob is worse than
      none.

    Neither threshold category has been validated against labeled push-up video (spec
    section 8.4)."""
    observable_neck = ctx.view_type in NECK_OBSERVABLE_VIEWS

    def _setup_baseline(key: str) -> float:
        """Mean of `key` over the clip's own valid `setup` frames; NaN if there are none.
        Per axis, so one unmeasurable cue does not silence the other."""
        values = [
            frame.m(key)
            for frame in core
            if frame.valid and frame.phase == "setup" and np.isfinite(frame.m(key))
        ]
        return float(np.mean(values)) if values else np.nan

    neck_baseline = _setup_baseline("neck_line_signed_deg")
    nose_baseline = _setup_baseline("nose_ahead_ratio")

    def _neck_deviation(frame: CoreFrame) -> float:
        """Neck deviation from the setup baseline, or NaN when this frame's neck reading is not
        ADMISSIBLE. The `hand_offset_ratio` guard belongs to THIS axis only -- the neck sign
        rides on the groundward normal, which a rolled camera inverts (see the docstring) -- and
        it is applied here, at the single point where the deviation is produced, rather than
        only in the mask. That matters: an inadmissible reading must not leak into the severity,
        the reported maximum or `evidence["criterion"]` either. Doing it in the mask alone let a
        camera-inverted clip report `criterion="both"` while the neck term had in fact been
        refused -- i.e. claim a verdict the rule had deliberately declined to make."""
        if not (
            frame.m("hand_offset_ratio") > 0.0
            and np.isfinite(neck_baseline)
            and np.isfinite(frame.m("neck_line_signed_deg"))
        ):
            return np.nan
        return frame.m("neck_line_signed_deg") - neck_baseline

    def _nose_deviation(frame: CoreFrame) -> float:
        """Nose deviation from the setup baseline, or NaN when unmeasurable. NOT guarded on
        `hand_offset_ratio`: `nose_ahead_ratio` is roll-invariant, so it has no inversion
        exposure to guard against."""
        if not (np.isfinite(nose_baseline) and np.isfinite(frame.m("nose_ahead_ratio"))):
            return np.nan
        return frame.m("nose_ahead_ratio") - nose_baseline

    def _neck_fires(frame: CoreFrame) -> bool:
        return _neck_deviation(frame) > 15.0

    def _nose_fires(frame: CoreFrame) -> bool:
        return _nose_deviation(frame) > 0.06

    head_mask = [
        frame.valid
        and frame.phase in HEAD_DROP_PHASES
        and (_neck_fires(frame) or _nose_fires(frame))
        for frame in core
    ]

    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(head_mask, ctx.min_frames):
        segment = core[start : end + 1]
        # Per-axis deviation series. A frame can be finite on one axis and NaN on the other (an
        # occluded nose, say), and `severity_from_range` scores a non-finite value 0.0, so the
        # per-frame max below degrades to the axis that is actually measurable.
        neck_values = [_neck_deviation(frame) for frame in segment]
        nose_values = [_nose_deviation(frame) for frame in segment]
        max_neck = float(np.nanmax(neck_values)) if any(np.isfinite(v) for v in neck_values) else np.nan
        max_nose = float(np.nanmax(nose_values)) if any(np.isfinite(v) for v in nose_values) else np.nan
        neck_severity = severity_from_range(max_neck, 15.0, 35.0, lower_is_worse=False)
        nose_severity = severity_from_range(max_nose, 0.06, 0.15, lower_is_worse=False)
        severity = max(neck_severity, nose_severity)

        # Which cue actually crossed its threshold, for the coaching cue. Read off the same
        # predicates the mask used -- NOT off the severities -- so an axis that fired but scores
        # lower is still reported, and an axis that was refused as inadmissible is not.
        # `severity > 0.0` happens to give the identical answer TODAY, purely because each ramp's
        # mild endpoint equals its fire threshold (15/15 and 0.06/0.06); it stops being identical
        # the moment either ramp is retuned, so the coupling is deliberately not relied on. A
        # mutant swapping these two forms is therefore EQUIVALENT under the current constants and
        # is reported as such rather than as a coverage gap.
        neck_fired = any(_neck_fires(frame) for frame in segment)
        nose_fired = any(_nose_fires(frame) for frame in segment)
        if neck_fired and nose_fired:
            criterion = "both"
        elif nose_fired:
            criterion = "nose_ahead"
        else:
            criterion = "neck_angle"

        # The display axis is whichever DROVE the severity (squat's rule_shallow_depth idiom),
        # which can differ from `criterion` when both fired.
        if neck_severity >= nose_severity:
            primary_label = "neck deviation from the setup baseline"
            primary_value, primary_threshold = round(float(max_neck), 2), 15.0
        else:
            primary_label = "nose jut ahead of the setup baseline"
            primary_value, primary_threshold = round(float(max_nose), 4), 0.06

        # Per-frame ranking series for `build_detection`'s `peak_frame`. UNCLIPPED on purpose:
        # `severity_from_range` clips to 1.0, so in any severe segment several frames tie at
        # 1.0 and `nanargmax` returns the FIRST of them rather than the worst -- measured, a
        # segment with deviations [16, 40, 36, 60, 17, ...] nominated the 40-degree frame over
        # the 60-degree one. The two axes are in different units (degrees vs body lengths), so
        # they are put on a common scale by their own ramps and only then compared.
        values = [
            _worst_axis(_unclipped(neck, 15.0, 35.0), _unclipped(nose, 0.06, 0.15))
            for neck, nose in zip(neck_values, nose_values)
        ]
        detections.append(
            build_detection(
                fault_id="pushup_head_drop",
                fault_name="Forward head / neck drop",
                # Verified to resolve: graph_retrieval.resolve_nodes(..., movement="Push-up")
                # returns "Forward Head Posture" for this string. NOTE that is the SHARED node
                # (movement="shared"), not a `Push-up:`-scoped one -- sports_kg_v3 has no
                # push-up-scoped head or neck node at all. Its one edge comes from
                # "Overhead Press:Subacromial Impingement Syndrome", which is precisely the
                # mechanism the spec's citation invokes. "Head Drop" and "Neck Drop" resolve to
                # NOTHING, so neither is usable.
                kg_query="Forward Head Posture",
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=values,
                severity=severity,
                confidence=severity * (1.0 if observable_neck else _OFF_VIEW_CONFIDENCE),
                observability="medium" if observable_neck else "low",
                evidence={
                    "criterion": criterion,
                    "max_neck_deviation_deg": (
                        round(float(max_neck), 2) if np.isfinite(max_neck) else None
                    ),
                    "neck_setup_baseline_deg": (
                        round(neck_baseline, 2) if np.isfinite(neck_baseline) else None
                    ),
                    "neck_threshold_deg": 15.0,
                    "max_nose_ahead_deviation": (
                        round(float(max_nose), 4) if np.isfinite(max_nose) else None
                    ),
                    "nose_setup_baseline": (
                        round(nose_baseline, 4) if np.isfinite(nose_baseline) else None
                    ),
                    "nose_ahead_threshold": 0.06,
                    "primary_label": primary_label,
                    "primary_value": primary_value,
                    "primary_threshold": primary_threshold,
                },
                citation="Lee S et al. J Phys Ther Sci (2013) PMC3820220 "
                         "(form/neutral-alignment standard); mechanism corroborated by "
                         "Al Hammadi MI et al. Cureus (2025) PMC12514857.",
                citation_support="PMC3820220's protocol required that \"the head, spine, and "
                                 "pelvis were positioned in a straight line, in a neutral "
                                 "state\" with \"the cervical vertebrae in a neutral "
                                 "position,\" defining neutral cervical alignment as correct "
                                 "form; PMC12514857 lists \"forward head posture\" among the "
                                 "postural factors that \"interfere with scapular movement ... "
                                 "leading to a reduction in subacromial space,\" supplying the "
                                 "injury rationale. Direct push-up-specific cervical-injury "
                                 "evidence is thin, so this rule leans on the alignment "
                                 "standard plus the general forward-posture->impingement "
                                 "mechanism.",
            )
        )
    return detections


def rule_elbow_flare(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag hands planted well outside the shoulders: wrist-to-wrist distance exceeds 1.6x
    shoulder width.

    THRESHOLD PROVENANCE -- THREE DIFFERENT CATEGORIES HERE, NOT TWO.

      FIRE THRESHOLD 1.6: FROM THE SPEC. `pushup_elbow_flare` defines hand-width ratio =
      wrist-to-wrist (15<->16) / shoulder width (11<->12) and says "flag when ratio > ~1.6".

      SEVERITY RAMP 1.6 -> 2.2: A RULE-LEVEL CHOICE MADE HERE. The spec states no severity ramp
      for this fault, and a FIXED-STRING grep confirms "2.2" does not occur anywhere in its
      Push-up section. (It DOES occur in tests/test_pushup.py as a fixture value predating this
      rule -- that is not provenance and must not be cited as such.) 2.2 is chosen as the point
      where the hands are more than twice shoulder width apart, i.e. the upper arm is closer to
      straight out to the side than to tracking back; nothing in the literature fixes it.

      MEASURABILITY GATE 0.25 (a third category, and the one most easily misread as a fault
      threshold): "0.25" likewise does not occur in the spec's Push-up section. It does not ask
      whether the lifter did something wrong -- it asks whether this camera geometry can answer
      the question at all. See the next block, INCLUDING the part where it does not work.

    WHY THE GATE IS ON MEASURABILITY RATHER THAN ON THE SPEC'S VIEW LABELS. The spec asks for a
    `front`/`rear` view down the body's long axis. Two findings from Task 4
    (src/pose/view_estimation.py's header) make gating on those labels unusable:
    `signed_orientation` is an image-space left/right ordering whose front/rear meaning is
    validated only for UPRIGHT subjects, so for a horizontal body the `front`/`rear`/`*_oblique`
    labels carry no validated meaning ("Do not gate a horizontal-movement rule on them"); and
    the production path calls `estimate_view_for_pose(allow_front=False)`, so `front` is never
    emitted downstream at all. A positive gate on those labels would therefore be either
    meaningless or permanently false. The measurable condition is used instead: the wrists must
    be genuinely separated in the image, `distance(15, 16) > 0.25 * shoulder_width`, which --
    since `hand_width_ratio` IS that quotient -- is written directly as
    `hand_width_ratio > 0.25`.

    THE GATE IS ARITHMETICALLY INERT TODAY, AND SAYING SO IS THE POINT. `hand_width_ratio > 1.6`
    already implies `> 0.25`, so the term can never change an outcome while the fire threshold
    sits at 1.6, and no fixture can distinguish it from any other value below 1.6. It is kept as
    an explicit mask term so the measurability CONDITION stays legible, and so it becomes live
    rather than forgotten if anyone ever lowers the fire threshold. It is listed as a knowingly
    surviving mutant rather than pretended to be tested.

    THE GATE THAT DOES THE REAL WORK: `shoulder_axis_ratio > 0.15`. The ratio's real failure mode
    is not the one the 0.25 gate imagines. From a true sagittal view the wrists do overlap, but
    SO DO THE SHOULDERS: numerator and denominator collapse together, `hand_width_ratio` becomes
    noise/noise, and it can land anywhere, including far above 1.6. `_DEGENERATE_LENGTH` guards
    division by zero, not this. Measured on real landmark arrays, a shoulder separation of 0.0020
    and a wrist separation of 0.0050 -- both sub-pixel, both meaningless -- yield
    `hand_width_ratio = 2.500`, i.e. a full-severity flare verdict out of pure noise.

      Closing it needs the shoulder width normalized by something that does NOT collapse
      sagittally, so `pushup_compute_raw` now emits `shoulder_axis_ratio` = shoulder width /
      shoulder-to-ankle length. Sagittally the numerator collapses while the denominator is at
      its LONGEST, so the ratio goes toward 0 exactly where `hand_width_ratio` degenerates;
      looking down the body's long axis it does the opposite, because the axis foreshortens and
      the shoulders do not. On the forged case above it reads 0.0033.

      THE 0.15 IS A RULE-LEVEL MEASURABILITY THRESHOLD -- the third category in this module's
      header, not a fault threshold, and NOT in the spec (fixed-string grep: "0.15" does not
      occur in the Push-up section). Reasoning: on rough human proportions the shoulders span on
      the order of 0.3 of the shoulder-to-ankle length, so requiring 0.15 asks for roughly half
      of that transverse extent to have survived the projection before the hand-width question
      is treated as answerable. That is an argument from approximate anatomy, not a measurement,
      and no source in this repo fixes it. The margins are wide in both directions: the forged
      noise case sits ~45x below the cut, and a genuine down-the-axis view sits above the
      anatomical 0.3 rather than below it. (For transparency, the unit fixture's deliberately
      non-physical geometry reads 0.1667, just above the cut -- the threshold was derived from
      the proportion above, not fitted to that fixture, and the fixture's value is stated here
      so a reader can see the proximity rather than discover it.)

      ROUND 1 REFUSED TO INVENT THIS NUMBER and shipped the hole as a documented known gap. That
      call is reversed here, deliberately and for a stated reason: review established that the
      hole was not a corner case but the rule's ENTIRE live firing envelope (see the next
      block), so refusing to invent left the rule loud and wrong instead of quiet and honest. A
      labelled rule-level measurability guard that can only ever SILENCE is the lesser evil; the
      original objection -- do not fabricate an anthropometric constant and call it spec-backed
      -- is still honoured, since nothing here claims spec provenance. The forged case is pinned
      by `test_sagittal_collapse_can_forge_a_wide_ratio` (kept, now asserting silence) in the
      same idiom as `test_visible_but_misplaced_landmark_is_trusted`.

    WHAT DOES NARROW IT -- A NEGATIVE VIEW GATE, WHICH IS A DIFFERENT CLAIM FROM THE ONE TASK 4
    FORBIDS. Task 4 discredits the front/rear/oblique DISTINCTION for a horizontal body; it does
    not discredit the `side` verdict, which rests on `body_axis_extent` and which Task 3
    explicitly validated FOR horizontal subjects. So while "is this a front view?" is
    unanswerable, "is the camera confidently perpendicular to the body's long axis?" is
    answerable -- and if it is, the wrists overlap and every reading here is the noise/noise case
    above. This rule therefore stays SILENT on a confident `side` label, using the existing
    shared `SIDE_VIEW_CONF_THRESHOLD` (0.20) and no new number. It is the exact mirror of
    `rule_hip_sag`, which hard-gates `front`/`rear` because ITS metric inflates there; this one
    inflates on `side`. The error direction is safe: a mislabelled clip is silenced, not falsely
    fired, which is this module's stated preference.

      THE GATE IS PARTIAL, and the docstring must not imply otherwise. Task 4's limit 3 records
      that a genuine sagittal clip often FAILS to earn the `side` label -- one occluded far-side
      shoulder silently reverts `body_axis_extent` to the vertical fallback -- so a real
      sagittal clip labelled `unknown` still reaches this rule carrying a noise/noise ratio.
      The gate narrows the hazard; the known-gap test above is what stands for the remainder.

      IN PRODUCTION this makes the rule near-dead in the TRUE-POSITIVE direction, and review
      established the sharper form of that statement, which round 1 understated: with
      `allow_front=False` the reachable labels are {`side`, `rear`, `rear_oblique`, `unknown`};
      `side` always clears the 0.20 floor (a `side` verdict requires `side_score >= 0.62`), so
      the gate above always bites there -- leaving a LIVE SET of exactly {`rear`,
      `rear_oblique`, `unknown`}, every one of which Task 4 limit 1 says carries no validated
      meaning for a horizontal body. Worse, limit 4 records that an evidence-FREE clip resolves
      to `rear_oblique`. So the live firing envelope is precisely the regime in which the label
      means nothing.

    WHY THAT NO LONGER PRODUCES A LOUD WRONG ANSWER -- two independent changes, either of which
    would help, both of which are needed:

      1. The `shoulder_axis_ratio` guard above is a VIEW-INDEPENDENT measurability test. It does
         not care what the label says, so it also covers the residual the view gate cannot
         reach: a genuine sagittal clip that fails to earn the `side` label (Task 4 limit 3 --
         one occluded far-side shoulder reverts `body_axis_extent` to the vertical fallback)
         still has collapsed shoulders and is refused on the geometry.

      2. WHATEVER IS LEFT IS EMITTED AT `low` OBSERVABILITY WITH THE 0.65 CONFIDENCE DISCOUNT,
         never at 1.0/1.0. The spec's ceiling for this fault is `medium` on `front`/`rear`, but
         for a HORIZONTAL body no label the pipeline can emit carries validated meaning, so
         nothing here can honestly claim that ceiling. A detection is still emitted -- the
         geometry passed a real measurability test and the fault is cited -- but `run_detector`
         sorts `low` last, so it can never outrank a fault that was observed from a view the
         pipeline actually validated. No new constant: `_OFF_VIEW_CONFIDENCE` (0.65) is the same
         discount squat, OHP and the other push-up rules already use.

      Task 8 SHOULD register this rule. It is measurable when the geometry says so, cited, and
      now incapable of shouting; the decision is stated here rather than left to be inferred.

    PHASE SCOPE: `PUSHUP_ACTIVE_PHASES`, the same rule-level call `rule_hip_sag` makes for the
    same reason -- during `setup` the lifter is still placing their hands, so a wide reading
    there is not yet a fault. The spec scopes this fault to no phase.

    OBSERVABILITY is a flat `low` with the standard 0.65 confidence discount, for the reason in
    point 2 above: the spec's `medium` ceiling is attached to `front`/`rear`, and those labels
    are not validated for a horizontal body, so no clip this rule can fire on has earned it. The
    tier does not vary by label, because varying it would mean tiering on exactly the labels Task
    4 discredited -- theatre dressed as precision.

    THE SPEC'S CORROBORATING SIGNAL IS NOT IMPLEMENTED: "If the upper arm is visible,
    corroborate with the trunk-to-upper-arm angle ... exceeding ~65 deg". It is offered as
    optional corroboration, not as a gate, and the metric layer emits no trunk-to-upper-arm
    angle; adding one is out of this task's scope. Stated so its absence is a recorded choice.

    Neither threshold category has been validated against labeled push-up video (spec
    section 8.4)."""
    # NEGATIVE view gate -- see the docstring. "Is this confidently a side view?" is answerable
    # for a horizontal body (Task 3/4); "is this a front view?" is not.
    if ctx.view_type == "side" and ctx.view_confidence >= SIDE_VIEW_CONF_THRESHOLD:
        return []

    flare_mask = [
        frame.valid
        and frame.phase in PUSHUP_ACTIVE_PHASES
        and np.isfinite(frame.m("hand_width_ratio"))
        # MEASURABILITY, not fault: enough transverse extent survived the projection for the
        # hand-width question to mean anything. This is the gate that does the work -- it is
        # view-independent, and NaN refuses for free. See the docstring.
        and frame.m("shoulder_axis_ratio") > 0.15
        # MEASURABILITY, not fault: the wrists must actually be separated in the image.
        # Arithmetically implied by the 1.6 test below -- kept explicit, see the docstring.
        and frame.m("hand_width_ratio") > 0.25
        and frame.m("hand_width_ratio") > 1.6
        for frame in core
    ]

    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(flare_mask, ctx.min_frames):
        segment = core[start : end + 1]
        values = [frame.m("hand_width_ratio") for frame in segment]
        widest = float(np.nanmax(values))
        severity = severity_from_range(widest, 1.6, 2.2, lower_is_worse=False)
        detections.append(
            build_detection(
                fault_id="pushup_elbow_flare",
                fault_name="Flared elbows / excessive hand width",
                # Verified to resolve: graph_retrieval.resolve_nodes(..., movement="Push-up")
                # returns "Push-up:Elbow Valgus Torque" for this string, which carries an edge
                # IN from "Hand Position" and OUT to "Elbow Injury" -- exactly the mechanism the
                # spec's citation quantifies. The fault's own wording resolves badly: "Elbow
                # Flare" and "Flared Elbows" resolve to NOTHING, and "Hand Positioning" resolves
                # to "Push-up:Hand Positioning", a `Phase`-labelled node with no outgoing edges
                # for retrieval to walk.
                kg_query="Elbow Valgus Torque",
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=values,
                severity=severity,
                # Never 1.0/1.0 from an unvalidated label: `low` + the standard 0.65 discount.
                confidence=severity * _OFF_VIEW_CONFIDENCE,
                observability="low",
                evidence={
                    "max_hand_width_ratio": round(widest, 4),
                    "min_shoulder_axis_ratio": round(
                        float(np.nanmin([frame.m("shoulder_axis_ratio") for frame in segment])), 4
                    ),
                    "shoulder_axis_ratio_threshold": 0.15,
                    "threshold": 1.6,
                    "primary_label": "hand width / shoulder width",
                    "primary_value": round(widest, 4),
                    "primary_threshold": 1.6,
                },
                citation="Donkers MJ, An KN, Chao EY, Morrey BF. J Biomech (1993). "
                         "DOI 10.1016/0021-9290(93)90026-b.",
                citation_support="Recording elbow forces in six hand positions, \"peak forces "
                                 "exerted on the elbow joint along the forearm axis averaged "
                                 "45% of the body weight for the 'normal' hand position and "
                                 "were significantly decreased if hands were positioned either "
                                 "'apart' or 'superior',\" while \"the maximum valgus torque at "
                                 "the elbow opposed by the medial ligamentous structure ... was "
                                 "significantly increased if the hand was positioned "
                                 "superiorly\" (and rose 42% one-handed). Hand position "
                                 "therefore strongly modulates elbow joint loading, justifying "
                                 "a rule that flags hand placement deviating from a "
                                 "shoulder-width baseline.",
            )
        )
    return detections


def rule_scapular_winging(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Registered but PERMANENTLY SILENT -- always returns [].

    Scapular winging is a real, well-cited push-up fault (serratus anterior weakness lets the
    scapula wing and over-internally-rotate, reducing subacromial space), but MediaPipe's 33
    landmarks contain NO scapular border points -- no medial border, no inferior angle -- so it
    cannot be measured from any view. The spec rates it observability `low`/`none`, calls the
    one indirect proxy it can think of (gross upper-back rounding from a `rear` view) "not
    trustworthy", and says in as many words: "Recommend NOT emitting a confident verdict".

    IT IS REGISTERED RATHER THAN OMITTED so the spec and the code stay in 1:1 correspondence.
    Anyone auditing "are all five push-up rules present?" finds it here, with its citation and
    this explanation, instead of finding a gap and closing it by inventing an unvalidated proxy.
    Silence is the verdict, and it is a cited one.

    Note this is the ONE push-up fault for which the spec authorises silence. `rule_elbow_flare`
    is also close to silent in production, but for a different and weaker reason (no view the
    pipeline can confirm), so it is gated rather than stubbed -- do not merge the two patterns.

    The signature takes `core`/`ctx` and ignores them so the function satisfies `RuleFn` and can
    sit in a `MovementDetector.rules` tuple unchanged if the landmark model ever gains scapular
    points.

    Citation: Lee S, Lee D, Park J. J Phys Ther Sci (2013) PMC3820220; corroborated by
    Abdollahi S et al. J Orthop Surg Res (2025) PMC12366113.

    Citation support: PMC3820220 states "Weakening of the serratus anterior muscle leads to
    excessive activation of the upper trapezius ... reducing the dynamic stability of the
    scapula," which drives "a clash between the subacromion and the head of the humerus";
    PMC12366113 similarly notes fatigue of the serratus anterior yields "increased internal
    rotation and decreased posterior tilt of the scapula." The fault is biomechanically real and
    important, but honestly not monocular-observable, hence observability `none`."""
    return []


# All FIVE of the spec's push-up rules are listed, deliberately: four can fire, and
# `rule_scapular_winging` is registered while being permanently silent so the spec and the code
# stay in 1:1 correspondence (see its docstring). Registering it costs one no-op call per clip
# and buys an auditor the answer "yes, it is accounted for, and here is why it says nothing".
#
# `rule_elbow_flare` IS registered, per the explicit decision recorded in its docstring: it is
# measurable when the geometry says so, cited, hard-gated to silence on a confident `side` label,
# and emitted at `low` observability with the 0.65 confidence discount. `run_detector` sorts
# `low` detections behind every other detection regardless of severity -- its key is
# `(observability == "low", -severity, start_frame)` and `False < True` -- so a flare verdict can
# never outrank a fault observed from a view the pipeline actually validated.
#
# PUSHUP_METRIC_KEYS must stay a two-way match with what `pushup_compute_raw` emits: keys the
# tuple omits are dropped by `run_detector` (which builds each CoreFrame's metrics dict FROM this
# tuple) and read back as NaN. That failure is silent and, for `hand_offset_ratio`, total -- it is
# the camera-inversion guard in both `rule_hip_sag` and `rule_head_drop`, and `nan > 0.0` is
# False, so dropping it would permanently silence both rules with no error anywhere.
# `test_pushup_metric_keys_match_the_emitted_metrics` pins the correspondence in both directions.
PUSHUP_DETECTOR = MovementDetector(
    "Push-up",
    PUSHUP_METRIC_KEYS,
    pushup_compute_raw,
    pushup_assign_phases,
    (
        rule_hip_sag,
        rule_shallow_depth,
        rule_elbow_flare,
        rule_head_drop,
        rule_scapular_winging,
    ),
    rep_signal="min_elbow_angle",
    rep_polarity="min",
)

registry.register(PUSHUP_DETECTOR)
