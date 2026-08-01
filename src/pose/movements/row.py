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
    contiguous_true_segments, severity_from_range,
)
from src.pose.movements.base import CoreFrame, RuleContext
from src.pose.pose_rule_detector import (
    VIEW_UNAVAILABLE_CONFIDENCE_SCALE,
    PoseRuleDetection,
    build_detection,
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


# ---------------------------------------------------------------------------------------
# STEP 0 -- KG QUERY RESOLUTION, recorded before any rule was written. Each string below was
# checked against data/kg/sports_kg_v3.graphml with `retrieve_graph_context(query,
# movement="Row")` -- the function PRODUCTION calls, not just `resolve_nodes` -- and returned a
# `Row:`-scoped seed with at least one NON-EMPTY bucket. OHP shipped three dangling queries
# because only `resolve_nodes` was checked; this is the check that would have caught them.
#
#   Trunk Extension            -> Row:Trunk Extension (Fault)
#                                 phases; corrections=[Maintain Neutral Spine];
#                                 quality_impacts=[Core Stability]
#   Scapular Protraction       -> Row:Scapular Protraction (Fault)
#                                 evidence=[Anterior Translation Of Scapulae]; related_actions
#   Loss Of Neutral Body       -> Row:Loss Of Neutral Body Position (Fault)
#     Position                    phases; evidence=[Head/Trunk/Hip Not Aligned ...];
#                                 corrections; quality_impacts; related_actions
#   Asymmetry                  -> Row:Asymmetry (Fault)
#                                 phases; risks=[Shoulder Injury, Injury Risk]; related_actions
#
# TWO DELIBERATE DEVIATIONS from the obvious name, load-bearing for later tasks that import
# these constants blind:
#   - Momentum: "Compensatory Movements" is a real `Row:`-scoped Fault node whose buckets are
#     ENTIRELY EMPTY -- precisely the OHP failure mode. "Loss Of Neutral Body Position" is the
#     richest on-topic node, and its three evidence signals ("Head Not Aligned With Trunk And
#     Hip", "Trunk Not Aligned With Head And Hip", "Hip Not Aligned With Head And Trunk") are a
#     direct description of the whole-body heave this fault is about.
#   - Asymmetry: "Interlimb Asymmetry" resolves but is scoped to `Unilateral Cable Row`, and
#     "Muscle Strength Asymmetry" carries only a generic `Injury Risk`. `Row:Asymmetry` is the
#     one whose buckets name both the phases the fault occurs in and a specific Shoulder Injury
#     risk.
ROW_TORSO_RISING_KG_QUERY = "Trunk Extension"
ROW_INCOMPLETE_ROM_KG_QUERY = "Scapular Protraction"
ROW_MOMENTUM_KG_QUERY = "Loss Of Neutral Body Position"
ROW_ASYMMETRY_KG_QUERY = "Asymmetry"

# Confidence multiplier applied when a rule fires from a view the spec does not rate `high`.
# Not a new number: aliases the shared constant rather than re-typing its value, so a future
# change to it cannot silently skip this module.
_OFF_VIEW_CONFIDENCE = VIEW_UNAVAILABLE_CONFIDENCE_SCALE

# Views with a lateral component, in which the parent spec rates trunk pitch and pull depth
# `high` ("side / front_oblique / rear_oblique ... Low from pure front/rear").
TRUNK_OBSERVABLE_VIEWS = {"side", "front_oblique", "rear_oblique"}

# FROM THE SPEC: "Flag if `trunk_angle_peak - trunk_angle_setup > 15deg`".
TRUNK_RISE_MILD_DEG = 15.0
# RULE-LEVEL CHOICE MADE HERE. The parent spec states NO severity ramp for ANY Row fault (the
# Lunge section states its ramps explicitly, so the absence is meaningful rather than a
# formatting quirk). 37.5 is 2.5x the fire threshold, the convention `pushup.rule_hip_sag`
# already uses for exactly this situation (ramp 0.06 -> 0.15). Treat it as a display/ranking
# curve, not a cited quantity.
TRUNK_RISE_SEVERE_DEG = 37.5


def _setup_baseline(core: list[CoreFrame], key: str) -> float:
    """Median of `key` over this window's valid `setup` frames; NaN when there are none.

    WHY THE BASELINE LIVES IN THE RULES AND NOT IN `row_compute_raw` -- the Row analogue of
    lunge's lead-leg problem. Three of the parent spec's five Row heuristics are deltas from a
    setup baseline, and a baseline is a PER-REP reduction. `run_detector` calls `compute_raw`
    over the WHOLE CLIP before `segment_reps`, so at metric time no rep boundary exists and
    there is no "this rep's setup" to reduce against. Rules receive a per-rep slice, which is
    the first place the question is answerable.

    MEDIAN, NOT MEAN, so one bad frame in a six-frame setup cannot move the reference every
    later comparison is made against.

    NEVER A GUESSED BASELINE -- but what a caller does with a NaN one is conditional on the
    caller's own fire condition, NOT a universal "return []" contract:
      - A rule whose fire condition depends ONLY on this baseline (e.g. `rule_torso_rising`'s
        `peak - baseline > threshold`) has nothing left to evaluate once the baseline is NaN,
        and must return `[]`, same as `rule_torso_rising` does.
      - A rule whose fire condition is a DISJUNCTION with a non-baseline term (e.g. Task 5's
        `rule_asymmetric_pull`, which fires on `elbow_height_asymmetry > 0.05` OR
        `shoulder_tilt - baseline > 0.04`) must NOT return `[]` outright on a NaN baseline --
        that would silence the elbow-only branch, which the spec states should still fire when
        the baseline term is unmeasurable. Such a rule drops only the baseline-dependent
        disjunct (treat that comparison as `False`, which `NaN > threshold` already evaluates
        to) and keeps evaluating the rest.
    Both branches serve the same principle: an occluded setup must never be papered over with a
    guessed number, only either silence the whole rule (when the guess was the only signal) or
    silently drop just the guessed term (when other signal remains). Stated cost of the per-rep
    scope either way: a lifter who is ALREADY rounded or rotated at this rep's setup reads as
    clean on the baseline-dependent term. A clip-level baseline would catch that but would make
    rep N's verdict depend on rep 1's frames, which this architecture deliberately does not do.

    STATED LIMITATION, MEASURED NOT HYPOTHETICAL: `setup` is the first 15% of the REP WINDOW
    (`row_assign_phases`'s `setup_cutoff`), and the window handed to a rule has already been
    trimmed by `segment_reps` to the rep's excursion -- so on a short rep `setup` can be as
    thin as 1-2 frames. Measured case: a 22-frame rep's 2-frame setup slice already overlapped
    a loaded peak frame, pulling the baseline to 37.5 degrees when the true resting angle was
    20 degrees. Because every comparison below is `peak - baseline`, a baseline biased UPWARD
    by an intruding loaded frame makes the MEASURED rise smaller than the true one -- the
    failure mode is a missed fault, never a false one. This is not corrected here (there is no
    principled way to detect "this setup frame is actually already loaded" without a second
    threshold this spec does not supply); it is simply the accuracy cost of a per-rep baseline
    on short reps, same category of tradeoff as the "already rounded at setup" cost above.
    """
    values = [
        frame.m(key)
        for frame in core
        if frame.valid and frame.phase == "setup" and np.isfinite(frame.m(key))
    ]
    if not values:
        return float(np.nan)
    return float(np.median(values))


def rule_torso_rising(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag the trunk drifting from its hinged setup angle toward upright across the pull.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM.
      FIRE THRESHOLD 15 deg: FROM THE SPEC ("Flag if trunk_angle_peak - trunk_angle_setup >
      15deg").
      SEVERITY RAMP 15 -> 37.5 deg: A RULE-LEVEL CHOICE (see TRUNK_RISE_SEVERE_DEG).

    PHASE SCOPE `peak`, FROM THE SPEC's own wording ("at setup baseline and at peak pull") --
    not a rule-level call and not a shared ACTIVE_PHASES set, of which this module defines
    none: every Row heuristic names its own phase, so a shared set would be a constant every
    rule overrides.

    OBSERVABILITY DOWNGRADE, NOT A GATE. The spec rates this `high` on side/oblique and low
    from pure front/rear, but a hard gate would likely ship this rule SILENT: the production
    path calls `estimate_view_for_pose(allow_front=False)`, so the reachable labels are
    {side, rear, rear_oblique, unknown}, and across the 45 real pose JSONs in this repository
    the estimator emitted `side` exactly ONCE (from a fixture since removed) against 30
    `rear_oblique` and 13 `rear`. `rear_oblique` supplies the lateral component this rule
    needs, so it earns the spec's `high`; everything else downgrades to `medium` and takes the
    x0.65 discount, following `squat.rule_knees_inward` rather than `rule_knees_forward`.
    """
    baseline = _setup_baseline(core, "trunk_angle_from_horizontal_deg")
    if not np.isfinite(baseline):
        return []
    observable = ctx.view_type in TRUNK_OBSERVABLE_VIEWS

    mask = [
        frame.valid
        and frame.phase == "peak"
        and np.isfinite(frame.m("trunk_angle_from_horizontal_deg"))
        and (frame.m("trunk_angle_from_horizontal_deg") - baseline) > TRUNK_RISE_MILD_DEG
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        rises = [frame.m("trunk_angle_from_horizontal_deg") - baseline for frame in segment]
        max_rise = float(np.nanmax(rises))
        severity = severity_from_range(
            max_rise, TRUNK_RISE_MILD_DEG, TRUNK_RISE_SEVERE_DEG, lower_is_worse=False
        )
        detections.append(
            build_detection(
                fault_id="row_torso_rising",
                fault_name="Torso Rising (Loss of Hip-Hinge)",
                kg_query=ROW_TORSO_RISING_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=rises,
                severity=severity,
                confidence=severity * (1.0 if observable else _OFF_VIEW_CONFIDENCE),
                observability="high" if observable else "medium",
                evidence={
                    "setup_trunk_angle_deg": round(baseline, 2),
                    "max_trunk_rise_deg": round(max_rise, 2),
                    "threshold": TRUNK_RISE_MILD_DEG,
                    "primary_label": "torso rise vs setup",
                    "primary_value": round(max_rise, 2),
                    "primary_threshold": TRUNK_RISE_MILD_DEG,
                },
                citation="Saeterbakken A, et al. Int J Sports Med (2015), PMID 26134664. "
                         "Supplemented by Owens LP, et al. Int J Sports Phys Ther (2026), "
                         "PMC13232157.",
                citation_support="Saeterbakken: the free-weight bent-over row produced greater "
                                 "erector spinae EMG than the machine row both bilaterally and "
                                 "unilaterally — the hinged free-weight row imposes a high, "
                                 "sustained trunk-extensor stabilizing demand that a rising "
                                 "torso abandons. Owens: breaks in efficient kinetic-chain "
                                 "sequencing \"require distal segments to increase functional "
                                 "capacity … described as the 'catch-up' phenomenon,\" and the "
                                 "protocol uses a trunk-parallel-to-floor position specifically "
                                 "to control trunk posture during rowing.",
            )
        )
    return detections


# FROM THE SPEC: "(a) Pull depth: minimum normalized distance from wrist(15/16) to hip(23/24)
# … flag if `min_wrist_to_torso_dist > 0.12`. (b) Elbow flexion at peak: `elbow_angle … > 100deg`
# at the top = pull not completed."
PULL_DEPTH_MILD = 0.12
PEAK_ELBOW_MILD_DEG = 100.0
# RULE-LEVEL CHOICE MADE HERE, both of them. The spec states no ramp for this fault.
#   0.30 is 2.5x the fire threshold -- `pushup.rule_hip_sag`'s convention.
#   140 deg is NOT re-derived: it is taken verbatim from `pushup.rule_shallow_depth`, whose
#   ramp is also 100 -> 140 on the very same quantity (an elbow angle whose fire threshold is
#   100). Copying it keeps the codebase's two elbow-ROM ramps from drifting apart, and inherits
#   that rule's stated argument: an elbow that never bends past 140 has travelled well under
#   half the useful range, which is where "maximally incomplete" reasonably saturates. That is
#   an argument, not a measurement.
PULL_DEPTH_SEVERE = 0.30
PEAK_ELBOW_SEVERE_DEG = 140.0


def rule_incomplete_rom(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag a pull that stops short -- the hands never reach the torso, or the elbows never bend.

    THRESHOLD PROVENANCE: fire thresholds 0.12 and 100 deg are FROM THE SPEC; both severity
    ramps are RULE-LEVEL (see the constants above).

    TWO OR'd CONDITIONS, ONE FAULT, per the spec's own (a)/(b) structure. A frame qualifies if
    either holds; the segment's severity is the WORSE of the two sub-severities, and
    `evidence["fired_on"]` records which one(s) drove it, because the coaching cue differs
    ("pull the bar all the way to the abdomen" vs "finish the elbow bend").

    `score_values` is the per-frame MAXIMUM of the two sub-severities rather than either raw
    metric, so `build_detection` nominates the frame that was worst OVERALL. Passing one raw
    series would let a frame that was fine on that metric but terrible on the other be named
    the peak.

    IT READS `max_elbow_angle`, THE LESS-FLEXED ARM, AND THAT IS A RULE-LEVEL READING OF AN
    UNDER-SPECIFIED SPEC LINE. The spec's condition (b) names no side. Taking the less-flexed
    arm is conservative -- a rep is incomplete if EITHER arm fell short -- and is the deliberate
    opposite of `pushup_shallow_depth`, whose docstring already flags its inherited more-flexed
    reading as the generous one.

    THE SPEC'S THRESHOLDS ARE IN RAW IMAGE UNITS, WHICH IS CAMERA-DISTANCE DEPENDENT. 0.12
    carries no stated normalizer, and the same spec says "normalized by shoulder width
    dist(11,12)" explicitly where it means that (Band Pull Apart), so the absence here is
    meaningful. Implemented as written; the same rep filmed further away yields a smaller body,
    smaller distances, and less firing. `wrist_hip_dist_shoulder_norm` is emitted as a
    SCALE-FREE DIAGNOSTIC that nothing fires on, so a future validation can compare the two
    readings without any threshold having been moved in the meantime.

    PHASE SCOPE `peak`, from the spec ("at the top"), and the same downgrade-not-gate view
    handling `rule_torso_rising` documents.

    KNOWN LIMITATION, NOT CORRECTED HERE: `score_values` are CLIPPED severities (`clip01`
    inside `severity_from_range`), so in a segment that saturates both sub-severities to 1.0
    on more than one frame, `build_detection`'s `nanargmax` nominates the FIRST such frame as
    `peak_frame`, not the worst one. `pushup.rule_head_drop` hit exactly this and fixed it with
    an unclipped ranking series (`_unclipped`, its docstring has the measured example). Not
    adopted here because this rule's two axes are in different units (a distance and an angle)
    and are only comparable at all once put on the SAME 0-1 scale by their own ramps -- an
    unclipped `_worst_axis` would need its own cross-axis calibration this task does not
    introduce. The blast radius is narrow: severity, confidence, evidence and `fired_on` are
    all computed independently of this ordering and are unaffected; only which frame's
    timestamp gets reported as `peak_frame` inside an already-saturated segment can be
    suboptimal.
    """
    observable = ctx.view_type in TRUNK_OBSERVABLE_VIEWS

    def _sub_severities(frame: CoreFrame) -> tuple[float, float]:
        distance_value = frame.m("mean_wrist_hip_dist")
        elbow_value = frame.m("max_elbow_angle")
        distance_severity = (
            severity_from_range(distance_value, PULL_DEPTH_MILD, PULL_DEPTH_SEVERE, lower_is_worse=False)
            if np.isfinite(distance_value) and distance_value > PULL_DEPTH_MILD
            else 0.0
        )
        elbow_severity = (
            severity_from_range(elbow_value, PEAK_ELBOW_MILD_DEG, PEAK_ELBOW_SEVERE_DEG, lower_is_worse=False)
            if np.isfinite(elbow_value) and elbow_value > PEAK_ELBOW_MILD_DEG
            else 0.0
        )
        return distance_severity, elbow_severity

    mask = [
        frame.valid and frame.phase == "peak" and max(_sub_severities(frame)) > 0.0
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        pairs = [_sub_severities(frame) for frame in segment]
        scores = [max(pair) for pair in pairs]
        severity = float(np.nanmax(scores))
        distance_fired = any(pair[0] > 0.0 for pair in pairs)
        elbow_fired = any(pair[1] > 0.0 for pair in pairs)
        fired_on = (
            "both" if distance_fired and elbow_fired else "pull_distance" if distance_fired else "elbow_angle"
        )
        max_distance = float(np.nanmax([frame.m("mean_wrist_hip_dist") for frame in segment]))
        max_elbow = float(np.nanmax([frame.m("max_elbow_angle") for frame in segment]))
        detections.append(
            build_detection(
                fault_id="row_incomplete_rom",
                fault_name="Incomplete ROM (Pull Not Completed)",
                kg_query=ROW_INCOMPLETE_ROM_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=scores,
                severity=severity,
                confidence=severity * (1.0 if observable else _OFF_VIEW_CONFIDENCE),
                observability="high" if observable else "medium",
                evidence={
                    "fired_on": fired_on,
                    "max_wrist_hip_dist": round(max_distance, 4),
                    "max_peak_elbow_angle_deg": round(max_elbow, 2),
                    "wrist_hip_dist_shoulder_norm": round(
                        float(np.nanmax([frame.m("wrist_hip_dist_shoulder_norm") for frame in segment])), 4
                    ),
                    "distance_threshold": PULL_DEPTH_MILD,
                    "elbow_threshold": PEAK_ELBOW_MILD_DEG,
                    "primary_label": "wrist-to-hip distance at peak"
                    if fired_on != "elbow_angle"
                    else "elbow angle at peak",
                    "primary_value": round(max_distance, 4) if fired_on != "elbow_angle" else round(max_elbow, 2),
                    "primary_threshold": PULL_DEPTH_MILD if fired_on != "elbow_angle" else PEAK_ELBOW_MILD_DEG,
                },
                citation="Fischer J, et al. J Electromyogr Kinesiol (2025), PMID 40513198. "
                         "Supplemented by Padovan R, et al. J Funct Morphol Kinesiol (2025), "
                         "PMC12821611.",
                citation_support="Fischer (prone barbell row, 3 ROMs): \"The LD showed "
                                 "significantly higher mean muscle excitation in the upper-half "
                                 "ROM compared to both the lower-half ROM (p < 0.001) and full "
                                 "ROM (p < 0.001)\" — the top of the pull drives peak lat "
                                 "excitation. Padovan: the row is driven by \"scapular "
                                 "retraction, external rotation, and posterior tilt [which] "
                                 "contributes to optimizing glenohumeral alignment and force "
                                 "transmission,\" with the concentric endpoint \"defined when "
                                 "the handle reached the abdominal target.\"",
            )
        )
    return detections
