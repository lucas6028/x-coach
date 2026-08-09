# Arm VW (scapular V-to-W protraction/retraction) raw metrics, phase segmentation and fault rules.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `arm_vw_compute_raw` / `arm_vw_assign_phases` compute
# per-frame quantities and a phase label only. Every number that decides anything belongs in a
# `rule_*` function. The only constant this module defines outside a rule, `_DEGENERATE_LENGTH`,
# is a division-by-zero guard, never a tunable threshold.
#
# ---------------------------------------------------------------------------------------
# THE LABELED DATA FINALLY MATCHES THE VARIANT THE APP MODELS.
# ---------------------------------------------------------------------------------------
# REHAB24-6 `Ex2` IS arm VW (src/rehab24/dataset.py EXERCISE_NAMES["2"]): 208 repetitions, 94
# correct / 114 incorrect, 9 subjects, 12 videos, 0 flagged mocap-erroneous, with marker-driven
# 3-D alongside the video and cached MediaPipe landmarks for every video. At 208 reps it is the
# LARGEST labeled set of any non-squat movement so far (Lunge 174, Arm Abduction 178).
#
# AND IT IS BILATERAL -- measured, not inferred from the blank `exercise_subtype` field: per-rep
# left/right excursion ratio median 0.954 (min 0.791) and within-rep r(L,R) elevation median
# 0.9977 (min 0.9628). Arm Abduction had to reach for Fit3D `side_lateral_raise` because its
# labeled set (Ex1) was UNILATERAL on 178/178 reps and could not speak to a two-arm rule at all.
# This module needs no second dataset for that purpose. `validated` is still False for one reason
# only: NOTHING HAS RUN THE CHECK. See the registration site at the bottom. Design spec section 2.
#
# ---------------------------------------------------------------------------------------
# THREE RULES SHIP, ONE IS PERMANENTLY SILENT, AND TWO SUB-CRITERIA ARE ABSENT.
# ---------------------------------------------------------------------------------------
#   rule_incomplete_excursion   ships (one disjunct dropped -- see EXCURSION_MILD_DEG)
#   rule_shrug_substitution     REGISTERED, PERMANENTLY SILENT -- real fault, cited mechanism,
#                               MEASURED sensing failure (its docstring)
#   rule_loss_of_elevation      ships; its W-abduction disjunct is WITHDRAWN, not implemented
#   rule_lr_asymmetry           ships, and is THE FIRST ASYMMETRY RULE IN THIS PROJECT TO GATE
#
# Registered-but-silent (pushup.rule_scapular_winging, band_pull_apart
# .rule_loss_of_scapular_retraction, arm_abduction.rule_shoulder_shrug) says "real, well-cited
# fault, the sensor cannot see it". Withdrawn (OHP bar-path, deadlift bar-drift, curl
# wrist-flexion, arm-abduction impingement arc) says "no citation supports the rule as written".
# Design spec sections 4, 5 and 6.
#
# ---------------------------------------------------------------------------------------
# ALL FOUR CITED SOURCES STUDY A DIFFERENT EXERCISE THAN THIS ONE, AND ALL FOUR ARE EMG.
# ---------------------------------------------------------------------------------------
# Jung PMC12734928 is quadruped / single-leg PUSH-UP-PLUS and sternum-drop. Abiara PMC12335237 is
# prone cobra / wall slide / scapula setting / prone trapezius exercise. Mun PMC12029123 is a
# PILATES REFORMER "arm work" movement. Terre PMC12110944 is bilateral scapular retraction at 45
# and 90 degrees. None reports a kinematic threshold in any landmark unit, so EVERY number in the
# parent spec's Arm VW section is the spec author's rather than a source's -- each constant below
# says so. This is the generalised form of the lesson the impingement-arc withdrawal drew:
# verifying that a source contains a quoted string is not verifying that it supports the claim the
# quote is attached to. Design spec section 3.
#
# ---------------------------------------------------------------------------------------
# ONE DROPPED LANDMARK SILENCES EVERY ARM VW RULE FOR THAT FRAME.
# ---------------------------------------------------------------------------------------
# `required` below lists both shoulders, both elbows and both hips -- and deliberately NOT the
# ears (see `rule_shrug_substitution`) and NOT the wrists (the only cue that would have read them
# is not implemented; see ASYMMETRY_MILD_DEG). If `visible_point` drops any ONE of the required
# points the frame is marked `valid=False` and carries no metric keys at all, so every rule that
# masks on `frame.valid` goes silent for that frame, not just the one whose input landmark went
# missing. This mirrors every movement module since Push-up: an unmeasurable frame is refused
# wholesale rather than degraded.
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, angle_degrees, mean_visibility, distance,
    contiguous_true_segments, severity_from_range,
)
from src.pose.movements.base import CoreFrame, MovementDetector, RuleContext
from src.pose.movements import registry
from src.pose.pose_rule_detector import (
    VIEW_UNAVAILABLE_CONFIDENCE_SCALE,
    PoseRuleDetection,
    build_detection,
)

# Defined locally, matching row.py, overhead_press.py, band_pull_apart.py, bicep_curl.py and
# arm_abduction.py: geometry.py exports only the lower-body and shoulder/hip constants.
LEFT_ELBOW = 13
RIGHT_ELBOW = 14

# The generic "lower body" set every movement module uses for the framework-level
# `lower_body_visibility` quality field. The name is squat-centric and carries awkwardly for a
# standing scapular drill, exactly as it does for OHP, Push-up, Row, Band Pull Apart, Bicep Curl
# and Arm Abduction; this module's own rules never consume it.
LOWER_BODY_LANDMARKS = (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

ARM_VW_METRIC_KEYS: tuple[str, ...] = (
    "left_arm_elevation_deg",
    "right_arm_elevation_deg",
    "avg_arm_elevation_deg",
    "arm_elevation_asymmetry_deg",
    "shoulder_width",
)

# Below this a length/normalizer is degenerate and the dependent metric is NaN. Same guard value
# every other movement module uses; not a tunable threshold.
_DEGENERATE_LENGTH = 1e-6


def arm_vw_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
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

        # The parent spec's own definition, shared with Arm Abduction: angle between the torso
        # vector (shoulder->hip) and the upper-arm vector (shoulder->elbow). ~0 deg = arm at the
        # side, ~90 deg = horizontal, ~180 deg = fully overhead. Same-side hip, so a trunk lean
        # does not silently inflate both arms together.
        #
        # `angle_degrees` consumes dims=3, i.e. MediaPipe's estimated z as well as x/y -- the
        # shared helper every movement module uses, not changed here. Under the RTMPose path
        # (src/pose/rtmpose_pose_extraction.py writes z=0.0) these ARE pure image-plane
        # projections.
        #
        # MEASURED MAGNITUDE ERROR, AND IT RUNS TOWARD SILENCE. Against REHAB24-6 Ex2's markers on
        # the MediaPipe path, the per-rep mean absolute error of this angle is 25.4 deg (p90 33.8;
        # 20.5 on `front` clips, 30.9 on `half-profile`), and it systematically OVER-READS the
        # excursion: median peak 166.4 vs the markers' 143.8, median trough 24.6 vs 58.4, median
        # swing 143.5 vs 87.7. Both live magnitude rules here fire on a value being TOO SMALL, so
        # an over-read swing and an over-read V both push them toward SILENCE -- missed faults,
        # never false ones. Design spec section 8.6.
        left_arm_elevation = angle_degrees(points, LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW)
        right_arm_elevation = angle_degrees(points, RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW)
        finite_elevations = [
            v for v in (left_arm_elevation, right_arm_elevation) if np.isfinite(v)
        ]
        avg_arm_elevation = float(np.mean(finite_elevations)) if finite_elevations else np.nan
        # NaN unless BOTH sides are finite: an asymmetry between one measured arm and one missing
        # arm is not a small asymmetry, it is no measurement. Contrast `avg_arm_elevation_deg`
        # above, which is a rep SIGNAL and degrades gracefully to whichever arm was seen.
        if np.isfinite(left_arm_elevation) and np.isfinite(right_arm_elevation):
            asymmetry = float(abs(left_arm_elevation - right_arm_elevation))
        else:
            asymmetry = np.nan

        shoulder_width = distance(points, LEFT_SHOULDER, RIGHT_SHOULDER)

        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "left_arm_elevation_deg": left_arm_elevation,
                "right_arm_elevation_deg": right_arm_elevation,
                "avg_arm_elevation_deg": avg_arm_elevation,
                "arm_elevation_asymmetry_deg": asymmetry,
                # DIAGNOSTIC ONLY -- no rule reads this. It is emitted so that the TWO
                # unimplemented `0.05 normalized units` cues (the elbow-to-shoulder-line disjunct
                # of rule 1 and the wrist-height disjunct of rule 4) stay CHECKABLE without
                # re-deriving the measurement that shows them ill-defined: see EXCURSION_MILD_DEG.
                "shoulder_width": shoulder_width if shoulder_width > _DEGENERATE_LENGTH else np.nan,
            }
        )

    return raw


def arm_vw_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> concentric -> peak -> eccentric, segmented on `avg_arm_elevation_deg`.

    Mirrors `row_assign_phases` and `bicep_curl_assign_phases`, and is the POLARITY INVERSE of
    `arm_abduction_assign_phases`: this movement's effort peak is the signal's MINIMUM (the W,
    arms pulled down and back), so the peak hold is the LEAST-ELEVATED 30% of the rep, i.e. the
    30th percentile of the elevation and below. Measured on REHAB24-6 Ex2's 208 annotated reps,
    the average of the two arms starts at a median 140.4 deg, bottoms at a median 54.7 deg at
    position 0.508 of the rep, and returns to 141.1 -- V -> W -> V. Same fallbacks as every other
    module: an empty clip returns an empty list, a clip with no finite signal is entirely
    `unknown`, and an invalid frame is `unknown` regardless of where it sits (the validity check
    precedes the setup cutoff, so an occluded frame in the opening 15% is NOT labelled `setup`).

    `setup` IS THE OPENING V and `peak` IS THE W HOLD -- the two scopes the parent spec's rules
    name ("V-phase peak", "at the top... at the W hold"). Note what that costs: with
    rep_start="extended" the rep runs V -> W -> V, so the CLOSING V falls in `eccentric` and no
    rule reads it. Measured, the rep's global maximum sits near the END on most reps (median
    argmax position 0.918), so reading only the opening V UNDER-READS the movement's best moment.
    That is the conservative direction -- a missed fault, never a false one -- and it is stated
    rather than repaired.

    THIS IS THE FIRST DETECTOR IN THE PROJECT WITH A SHIPPED RULE SCOPED TO `setup`, AND THE
    MARGIN IS 1.25x. Bicep Curl section 4.3 found the arithmetic that silences a phase-scoped
    rule: `phase_fraction * T >= min_frames / fps` with `min_frames = max(3, ceil(0.20 * fps))`
    (base.py:197), so a 15% `setup` window needs T >= 1.333 s while a 30% `peak` window needs only
    T >= 0.667 s. Arm Abduction dodged this by scoping nothing to `setup`; `rule_loss_of_elevation`
    cannot, because the V IS the opening of the rep.

    Measured on the REAL segmenter rather than on annotation windows -- `segment_reps` trims each
    window to the excursion and is therefore TIGHTER than the annotations. Running
    `segment_reps(smoothed avg_arm_elevation, fps=30, polarity="min", rep_start="extended",
    min_rep_seconds=0.4)` over all 12 REHAB24-6 Ex2 videos yields 234 reps of 50-752 frames =
    1.67-25.07 s (median 4.65): the `setup` window is 7-113 frames (min 7) against min_frames=6,
    failing on 0/234, and `peak` is 15-225 (min 15), failing on 0/234. The shortest segmented rep
    sits at 1.25x the `setup` requirement, against 2.5x for `peak`. At Fit3D's 50 fps
    (min_frames=10) the same requirement is 1.333 s against a 2.12 s shortest rep, 1.59x. It
    clears, and it is the tightest clearance any detector here has shipped with -- which is why
    `EndToEndSegmentationTest` pins it. Design spec section 8.3.
    """
    frame_count = len(raw)
    if frame_count == 0:
        return []

    elevation_values = np.asarray(
        [float(item.get("avg_arm_elevation_deg", np.nan)) for item in raw], dtype=np.float32
    )
    valid_elevation = elevation_values[np.isfinite(elevation_values)]
    if valid_elevation.size == 0:
        return ["unknown" for _ in raw]

    # The LEAST-elevated 30% of the rep is the W hold.
    peak_threshold = float(np.percentile(valid_elevation, 30))
    lowest_index = int(
        np.nanargmin(np.where(np.isfinite(elevation_values), elevation_values, np.inf))
    )
    setup_cutoff = max(1, int(frame_count * 0.15))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append("setup")
            continue

        value = elevation_values[index]
        if np.isfinite(value) and value <= peak_threshold:
            phases.append("peak")
        elif index < lowest_index:
            phases.append("concentric")
        else:
            phases.append("eccentric")
    return phases


# ---------------------------------------------------------------------------------------
# STEP 0 -- KG QUERY RESOLUTION, recorded before any rule was written. Each string below was
# checked against data/kg/sports_kg_v3.graphml with `retrieve_graph_context(query, movement="Arm
# VW")` -- the function PRODUCTION calls, not just `resolve_nodes`. Observed results, not
# predicted ones:
#
#   "Insufficient Scapular Retraction" -> Arm VW:Insufficient Scapular Retraction
#       causes: Limited Scapular Retraction                                        NON-EMPTY
#   "Shoulder Shrug"                   -> Arm VW:Compensatory Shoulder Shrug
#       quality_impacts: Shoulder Depression                                       NON-EMPTY
#   "Muscle Imbalance"                 -> Muscle Imbalance (generic Cause node)
#       ZERO buckets, 1 edge                                                       DANGLING
#
# RULES 1 AND 3 SHARE A QUERY, DELIBERATELY. The graph has exactly THREE Arm VW fault nodes --
# Insufficient Scapular Retraction, Compensatory Shoulder Shrug, Trunk Lean Compensation -- and no
# incomplete-elevation node at all. An under-swung excursion and an under-elevated V are the same
# thing in the graph's vocabulary. Substituting the generic `Range Of Motion` node for one of them
# was rejected for the reason Band Pull Apart, Bicep Curl and Arm Abduction all rejected it: it
# returns full buckets whose `corrections` entry is "Wrapping Surface Adjustment", meaningless
# here. A semantically correct shared card beats a semantically wrong distinct one, and the two
# rules stay distinguishable by fault_id, fault_name, citation and evidence.
#
# THE THIRD ROW IS THE `Row:Compensatory Movements` / `arm_abd_lr_asymmetry` CASE, ACCEPTED THE
# SAME WAY. The graph has NO Arm VW asymmetry fault node; "Asymmetry" and "Left Right Asymmetry"
# resolve to the generic `Symmetry` QualityDimension, which is ALSO zero-bucket. The user-visible
# failure the OHP KG fix (PR #48) targeted DOES NOT OCCUR, verified by reading the frontend rather
# than inferring: `FaultCard.tsx:55-57` pushes a causes/risks/cue rung only `if (...).length` and
# wraps the whole block in `rungs.length > 0`, so a zero-bucket seed renders a THINNER card --
# fault name, severity, evidence -- never an empty "Causes:" heading with nothing under it.
#
# RECORDED AND NOT ACTED ON: the graph carries `Arm VW:Trunk Lean Compensation` and the parent
# spec gives Arm VW NO trunk-lean rule -- the exact mirror of Arm Abduction, where the graph
# carried `Incomplete Elevation` and the spec had no ROM rule. Two movements, two unused nodes, in
# opposite directions. Filling either needs a source that puts a number on it; no rule is invented
# here. The graphml is gitignored, so authoring an Arm VW asymmetry node is a deploy step, logged
# against TODO.md's existing "many faults have no KG node" item. Design spec section 9.
VW_EXCURSION_KG_QUERY = "Insufficient Scapular Retraction"
VW_SHRUG_KG_QUERY = "Shoulder Shrug"
VW_ELEVATION_KG_QUERY = "Insufficient Scapular Retraction"
VW_ASYMMETRY_KG_QUERY = "Muscle Imbalance"

# Imported rather than re-typed, so a change to the shared constant cannot silently skip this
# module.
_OFF_VIEW_CONFIDENCE = VIEW_UNAVAILABLE_CONFIDENCE_SCALE

# THE FULLY-OBSERVABLE SET, WRITTEN IN THE POSITIVE -- as Arm Abduction writes it, and for the
# same reason: all three live rules here measure FRONTAL-PLANE quantities, where the
# fully-observable views are the small set (the parent spec rates every Arm VW rule `high` on
# front/rear specifically). `front` is listed even though `estimate_view_for_pose(allow_front=
# False)` can never emit it, because this is the spec's observability rating transcribed and
# `run_detector` is called with whatever view label its caller supplies -- the REHAB24-6 replay
# harness (src/rehab24/lunge_rule_validation.py ORACLE_VIEWS) deliberately feeds the literal
# "front".
#
# PRODUCTION VIEW CENSUS, RE-MEASURED FOR THIS MODULE RATHER THAN INHERITED (the Bicep Curl doc
# warns that an inherited view figure once stopped reproducing): running
# `estimate_view_for_pose(path, allow_front=False).view_type` over all 49 files under
# data/runtime/pose_json on 2026-08-09 gives rear_oblique 37, rear 9, unknown 3, side 0. `front`
# and `front_oblique` are unreachable under allow_front=False (src/pose/view_estimation.py:14-16).
# Reproduces the Arm Abduction census exactly.
#
# THIS SET IS USED TWO DIFFERENT WAYS AND THE DIFFERENCE IS THE POINT OF THIS MODULE:
# `rule_incomplete_excursion` and `rule_loss_of_elevation` DISCOUNT outside it (x0.65, still
# live); `rule_lr_asymmetry` GATES on it and is silent outside. See that rule's docstring for the
# measurement that forces the distinction.
FRONTAL_OBSERVABLE_VIEWS = {"front", "rear"}


# FROM THE SPEC: "Flag if `arm_elevation_angle` swing between V and W phases `< 40deg`".
#
# ITS PROVENANCE, STATED: the 40 is the parent spec author's. Jung PMC12734928 supplies the
# MECHANISM -- "STD variations elicited higher trapezius activation, especially during large
# scapular excursions" and "greater scapular excursion is known to increase muscle activation",
# both read verbatim in the RAG doc -- but the study is a QUADRUPED / SINGLE-LEG PUSH-UP-PLUS with
# sternum drop, not a standing V-to-W drill, and it is EMG throughout. No kinematic number appears
# anywhere in it.
#
# WHERE 40 SITS IN THE OBSERVED DISTRIBUTION, RECORDED RATHER THAN REPAIRED: it fires on 0/208
# REHAB24-6 Ex2 reps on the marker 3-D, 0/208 on the same reps read through MediaPipe, and 0/41 on
# Fit3D `overhead_trap_raises`. The smallest swing observed anywhere is 47.0 deg. As shipped this
# rule will almost never fire, and when it does the rep really was truncated.
#
# IT IS NOT LOGICALLY DOMINATED BY `rule_loss_of_elevation`, AND THE TEMPTING CLAIM THAT IT IS
# WOULD BE WRONG. That rule is silent when V >= 120 and W >= 75; a rep with V = 120 and W = 85
# satisfies both and still swings only 35 deg. So this is NOT the vacuous-branch defect that
# killed row.rule_momentum_jerk's second condition, Bicep Curl's elbow-displacement disjunct and
# the impingement arc's first conjunct -- it is a live branch that simply never fires on anything
# measured.
EXCURSION_MILD_DEG = 40.0
# RULE-LEVEL CHOICE MADE HERE. The parent spec states NO severity ramp for ANY Arm VW fault (the
# Lunge section states its ramps explicitly, so the absence is meaningful). 16 is 0.4x the fire
# threshold; `lower_is_worse=True` because this is a "not enough" quantity, matching
# band_pull_apart.rule_incomplete_rom and deadlift.rule_incomplete_lockout. A display/ranking
# curve, not a cited quantity.
EXCURSION_SEVERE_DEG = 16.0

# THE PARENT SPEC'S SECOND EXCURSION CUE IS NOT IMPLEMENTED, for the reason Arm Abduction section
# 6.7 established and pinned. "or elbow fails to descend to within `0.05` (normalized y) of the
# shoulder line at the W" is dropped because `0.05` IN RAW NORMALIZED IMAGE UNITS IS NOT A
# WELL-DEFINED CRITERION: normalized coordinates scale with how much of the frame the subject
# occupies. Measured across the 43 production pose JSONs under data/runtime/pose_json that carry a
# usable shoulder width, the per-clip median `shoulder_width` runs 0.0591 to 0.4923 normalized
# units -- an 8.3x spread -- so 0.05 units is 0.102 shoulder-widths on the widest-framed clip and
# 0.846 on the narrowest. The same physical shortfall fires or does not depending on how far the
# phone was from the lifter. `ohp_asymmetric_press` avoids this by normalizing by shoulder width
# explicitly; this spec line does not, and renormalizing it here would be inventing a threshold.
# `shoulder_width` is emitted so the spread stays checkable; the arithmetic is pinned by
# test_the_dropped_normalized_disjuncts_are_frame_scale_dependent.


def rule_incomplete_excursion(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag a rep whose arms barely travel between the V and the W.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM.
      FIRE THRESHOLD 40 deg of swing: FROM THE SPEC (whose citation supplies the mechanism and no
        number at all -- read EXCURSION_MILD_DEG's comment before quoting it).
      SEVERITY RAMP 40 -> 16 deg: A RULE-LEVEL CHOICE.

    SCOPE IS THE WHOLE REP, NOT A PHASE, because an excursion is a property of the rep. The
    aggregate-inside-the-loop shape follows `deadlift.rule_incomplete_lockout`, which likewise
    moved its test off the individual frame and onto the segment. The mask is validity alone and
    the segment is taken over ALL valid frames of the window at once rather than per contiguous
    run: splitting on an occlusion gap would hand each half a partial excursion and FIRE ON A
    GOOD REP, which is the opposite of this rule's intended failure direction.

    THIS RULE SHIPS ON SEMANTIC CORRECTNESS AND A BACKGROUND-CITED MECHANISM, NOT ON MEASURED
    DISCRIMINATION, AND THAT DIFFERENCE FROM `arm_abd_contralateral_trunk_lean` MUST NOT BE
    ELIDED. Trunk lean shipped past an UNVERIFIED citation because the cue scored a per-subject
    median AUC of 0.800 at ranking incorrect reps above correct ones. This cue scores 0.452
    (pooled 0.476) across REHAB24-6 Ex2's 9 subjects and 0.494 (pooled 0.502) across the eight
    non-degenerate ones -- EXACTLY AT CHANCE. It ships anyway, on three other grounds:

      1. SEMANTIC CORRECTNESS. A rep whose arms swing less than 40 deg between the V and the W
         really is an incomplete V-to-W excursion. Firing on one is never wrong.
      2. A CITED MECHANISM. Jung PMC12734928 states that greater scapular excursion increases
         trapezius activation. The mechanism is cited; the threshold is not (EXCURSION_MILD_DEG).
      3. THE AUC IS EVIDENCE ABOUT Ex2'S ERROR TYPE, NOT ABOUT THE RULE. REHAB24-6 does not record
         WHICH error each incorrect rep contains. That excursion magnitude fails to separate its
         two classes says its incorrect reps are wrong some other way; it does not say a truncated
         rep is fine.

    AN ARM METRIC UNDER A SCAPULAR FAULT NODE IS NOT THE SUBSTITUTION THIS PROJECT FORBIDS, AND A
    READER WILL TRIP ON THIS UNLESS IT IS WRITTEN DOWN. The KG seed is `Arm VW:Insufficient
    Scapular Retraction` and the metric is arm elevation. The parent spec DECLARES this rule a
    proxy -- "Use the visible arm-excursion proxy for the (non-observable) scapular travel ... True
    A-P scapular retraction is not directly measured" -- and rates it `medium` for the arm
    excursion, `low` for true scapular protraction/retraction. The forbidden move is shipping
    metric B under a fault_id whose citation is about metric A WITHOUT SAYING SO. This says so:
    here, in the spec, and in the observability the rule emits.

    NO VIEW GATE, ONLY A DISCOUNT. An arm-elevation excursion is the right quantity from every
    reachable view; obliquity foreshortens it, so a real shortfall reads as a DEEPER shortfall and
    the rule errs toward firing -- in the opposite direction to the threshold placement, which
    errs away from it. Contrast `rule_lr_asymmetry`, which gates.
    """
    scale = 1.0 if ctx.view_type in FRONTAL_OBSERVABLE_VIEWS else _OFF_VIEW_CONFIDENCE

    segment = [
        frame
        for frame in core
        if frame.valid and np.isfinite(frame.m("avg_arm_elevation_deg"))
    ]
    if len(segment) < ctx.min_frames:
        return []

    values = [frame.m("avg_arm_elevation_deg") for frame in segment]
    highest = float(np.nanmax(values))
    lowest = float(np.nanmin(values))
    excursion = highest - lowest
    if not excursion < EXCURSION_MILD_DEG:
        return []

    severity = severity_from_range(
        excursion, EXCURSION_MILD_DEG, EXCURSION_SEVERE_DEG, lower_is_worse=True
    )
    # NEGATED so `build_detection`'s argmax lands on the LEAST-elevated frame -- the bottom of the
    # W, which is the frame the evidence below is quoting. Same intent as
    # `deadlift.rule_incomplete_lockout` feeding its driver axis's raw angles.
    score_values = [-value for value in values]
    return [
        build_detection(
            fault_id="vw_incomplete_excursion",
            fault_name="Incomplete V-to-W Excursion",
            kg_query=VW_EXCURSION_KG_QUERY,
            retrieval_mode="kg",
            segment_metrics=segment,
            score_values=score_values,
            severity=severity,
            confidence=severity * scale,
            observability="medium" if ctx.view_type in FRONTAL_OBSERVABLE_VIEWS else "low",
            evidence={
                "arm_elevation_excursion_deg": round(excursion, 2),
                "v_elevation_deg": round(highest, 2),
                "w_elevation_deg": round(lowest, 2),
                "threshold_deg": EXCURSION_MILD_DEG,
                "primary_label": "V-to-W arm elevation swing",
                "primary_value": round(excursion, 2),
                "primary_threshold": EXCURSION_MILD_DEG,
            },
            citation=(
                "Jung EY, Roh SY, Mun WL, Life (2025), PMC12734928, DOI 10.3390/life15121840."
            ),
            citation_support=(
                "The study found the larger-excursion variation (sternum-drop) \"elicited higher "
                "trapezius activation, especially during large scapular excursions,\" and states "
                "that \"greater scapular excursion is known to increase muscle activation.\" "
                "NOTE: it studies QUADRUPED and SINGLE-LEG PUSH-UP-PLUS / sternum-drop, not a "
                "standing V-to-W drill, and it is an EMG study containing NO kinematic threshold "
                "— the 40° cut applied here is the parent spec's. The metric is the visible "
                "ARM-excursion proxy the parent spec prescribes for the non-observable scapular "
                "travel, not a measurement of scapular retraction itself."
            ),
        )
    ]


def rule_shrug_substitution(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Registered but PERMANENTLY SILENT -- always returns [].

    Upper-trapezius substitution during the pull-down and W hold is a real, cited fault. Abiara S
    et al., PeerJ (2025) PMC12335237, read in the RAG doc: shoulder pain is "characterized by
    increased activation of the upper trapezius and decreased activation of the lower trapezius
    and serratus anterior", and "ratios lower than 1.0 for the UT/LT ratio are preferred
    (suggesting the LT is more active than the UT), although lower than 0.6 are ideal ... ratios
    >1.0 are considered non-optimal for rehabilitation interventions". Jung PMC12734928 supplies
    the scapular-dyskinesis framing. The fault is genuine. What fails is the SENSING, and -- as in
    `arm_abduction.rule_shoulder_shrug` -- that failure was MEASURED rather than argued.

    THIS IS THE SECOND MOVEMENT ON WHICH THE `neck_gap = ear_y - shoulder_y` CONSTRUCTION HAS BEEN
    MEASURED, AND IT FAILS FOR A DIFFERENT REASON HERE. Arm Abduction's excursion runs the arm UP
    and the 18% shrink threshold fired on 96.6% of MediaPipe reps. Arm VW's pull-down runs the arm
    DOWN, and the spec's own mitigation ("restrict this flag to the pull-down/W-hold phases where
    depression is expected") is structurally sounder as a result -- so this rule was re-measured
    rather than silenced by inheritance. Measured on REHAB24-6 Ex2's 208 reps, with each candidate
    "shoulder" taken as height above the mid-hip and the gap referenced to the rep's opening frame:

        point                          rho(gap, elevation)   18% shrink   gap travel
                                       over the pull-down    fires        % of baseline
        marker CLAVICLE (acromion)     med -0.305            0/208        1.2%
        marker GLENOHUMERAL            med -0.998            0/208        36.3%
        MediaPipe |ear - shoulder|     med -0.957            0/208        --

    Shoulder-height travel as a fraction of its own baseline: marker clavicle 0.6%, marker
    glenohumeral 9.8%. MEDIAPIPE REPORTS THE GLENOHUMERAL JOINT, NOT THE ACROMION -- the Arm
    Abduction finding reproduced under a REVERSED elevation direction, which is what makes it a
    property of the landmark rather than of that movement.

    TWO INDEPENDENT FAILURES, AND THE SECOND IS NEW:

    (a) THE METRIC IS AN ARM-ELEVATION READOUT, NOT A SHRUG READOUT. rho = -0.957 on MediaPipe
        against the arm's own elevation. Whatever it flags, it flags because the arm moved.
    (b) THE 18% THRESHOLD CAN NEVER FIRE ON THIS MOVEMENT'S BASELINE CONVENTION. The rep OPENS at
        the V -- arms overhead, shoulders legitimately at their most elevated point in the whole
        movement. Every later frame has a LARGER gap, so "shrink below baseline" is negative
        throughout the pull-down and the W hold. 0/208 on all three instruments. That is the exact
        inverse of Arm Abduction's 96.6%, and just as unusable.

    And the cue carries no information about the labels: the clavicle-gap shrink scores pooled AUC
    0.484 / per-subject median 0.549 at ranking Ex2's incorrect reps above its correct ones.

    SEPARATELY, THE 18% CARRIES NO CITATION. Abiara reports EMG ratios and no landmark
    displacement in any units. That would matter a great deal for a rule that fires and does not
    for one that never does -- recorded so a future reader who repairs the sensing does not
    inherit the number as though it were cited.

    ONE HONESTY NOTE ON THE CITATION THAT DOES NOT CHANGE THE TREATMENT. Abiara's Exercise C --
    "Participants stood against the wall and began with their arm abducted to 90 degrees, their
    elbows bent to 90 degrees, and their palms facing forward" -- is the closest thing in any cited
    source to the W position, and the paper reports its UT/LT ratio as OVER 1.0, concluding "only
    the Modified Prone Cobra (Exercise B) can be recommended". The cited literature is lukewarm
    about the exercise this rule set is built around. Out of scope: the rule set is the parent
    spec's.

    SILENT, NOT WITHDRAWN. Abiara genuinely backs the fault, so this is a sensing failure, not a
    citation failure. Contrast the W-abduction disjunct of `rule_loss_of_elevation`, which is
    ABSENT from this module entirely.

    NOT SUBSTITUTED, AND ITS METRIC IS NOT EMITTED. Shipping a different quantity under this
    fault_id would attach Abiara's UT/LT citation to something Abiara says nothing about. And
    `shoulder_ear_gap` is absent from ARM_VW_METRIC_KEYS: no live rule reads it, and emitting it
    would force landmarks 7/8 into `required`, where the all-or-nothing gate would let one lost
    ear silence the three rules that DO fire.

    WHAT THIS DOES AND DOES NOT SAY ABOUT `band_pull_apart.rule_shrugging`, WHICH SHIPS LIVE ON
    THIS SAME CONSTRUCTION. It NARROWS the open item without discharging it. The gap now measurably
    tracks arm elevation on two movements whose elevation runs in OPPOSITE directions (abduction
    rho = -0.957 rising, VW rho = -0.957 falling), so the confound scales with the MAGNITUDE of the
    elevation excursion rather than its sign. Band Pull Apart's excursion is horizontal abduction
    at roughly fixed elevation, so its confound should be SMALL -- an argument FOR that rule being
    sound, not against it. It is still not measured on Band Pull Apart's own data and no claim is
    made here. TODO.md, unchanged in status, better informed.

    OPEN, RECORDED, NOT RESOLVED: a working shrug rule needs shoulder height read AT MATCHED ARM
    ELEVATION -- comparing like with like across the rep rather than against a baseline taken at a
    different arm position. Arm Abduction recorded the same requirement. Novel construction, no
    citation, no validation; inventing it here is the fabrication this project's honesty rules
    forbid.

    THE KG IS NOT THE GAP: `Shoulder Shrug` resolves to `Arm VW:Compensatory Shoulder Shrug` with a
    non-empty `quality_impacts` bucket (`Shoulder Depression`). The metric is the gap.
    """
    return []


# FROM THE SPEC: "Flag if V-phase peak `< 120deg` (arms not raised high enough)".
#
# ITS PROVENANCE, STATED PRECISELY: 120 is the LOW END OF A CITED OPTIMUM, never a stated fault
# threshold. Mun WL et al., Medicina (2025) PMC12029123 was re-read in the RAG doc. Its OWN finding
# is "The LT showed the highest muscle activity at the shoulder abduction angle of 135 degrees
# (p < 0.001)", measured at 0/90/135/160 degrees during a PILATES REFORMER arm-work movement. Its
# DISCUSSION cites other work -- "Researchers recommend shoulder abduction near 145 degrees,
# aligning with the muscle fiber direction, for maximum LT activation" and "In a previous study,
# the LT activation was the highest at 120 degrees of shoulder abduction compared to at 30, 60,
# and 90". Abiara PMC12335237 adds that its LT-targeting prone exercise uses "arms abducted above
# 90 degrees, thumbs up".
#
# So the literature gives an LT-OPTIMAL BAND of roughly 120-145 degrees. Reading 120 as a FLOOR is
# a defensible rendering of "stay in the band", but NO SOURCE STATES IT AS A FAILURE THRESHOLD and
# the parent spec never says where it came from. Not moved.
#
# WHERE IT SITS, AND WHAT IT DISCRIMINATES. REHAB24-6 Ex2's median V peak on the marker 3-D is
# 143.8 deg -- inside the cited optimum -- so this is the `lunge_insufficient_depth` shape, a real
# cue whose cited cut lives in the tail. It is ALSO THE BEST-DISCRIMINATING CUE MEASURED ON THIS
# MOVEMENT: ranking incorrect reps above correct ones on the V peak gives pooled AUC 0.596 /
# per-subject median 0.660 over all 9 subjects, and pooled 0.713 / per-subject median 0.735 over
# the eight non-degenerate ones (person 8 contributes 2 correct against 20 incorrect and was
# suppressing it). 0.735 is comparable to the 0.800 that carried `arm_abd_contralateral_trunk_lean`.
#
# FIRE RATES, AND ONE SEMANTIC NOTE ABOUT THE READING. The parent spec says "V-phase PEAK < 120",
# which strictly means the MAXIMUM over the V window is below 120. The codebase idiom is a
# per-frame mask plus `contiguous_true_segments`, which fires on any SUSTAINED RUN below 120 in the
# window -- a strictly weaker condition, so it fires more. Both are recorded rather than one being
# silently chosen:
#
#     reading                                         markers    MediaPipe
#     max over `setup` < 120 (spec-literal)             6/208        0/208
#     sustained run below 120 in `setup` (SHIPPED)     31/208        9/208
#
# The shipped reading is the codebase idiom and the more sensitive of the two; 15% on 3-D truth and
# 4.3% through the estimator are both plausible fault rates rather than a false-positive machine.
ELEVATION_MILD_DEG = 120.0
# RULE-LEVEL CHOICE MADE HERE. 60 is 0.5x the fire threshold; `lower_is_worse=True` because this is
# a "not enough" quantity. A display/ranking curve, not a cited quantity.
ELEVATION_SEVERE_DEG = 60.0

# THE PARENT SPEC'S SECOND ELEVATION CUE IS WITHDRAWN -- ABSENT, NOT SILENT -- AND IT FAILS TWO
# WAYS, EITHER OF WHICH WOULD BE SUFFICIENT. "or W-phase abduction < 75deg (elbows collapsed
# toward the body)".
#
#   1. THE 75 APPEARS IN NO CITED SOURCE. Mun measures 0/90/135/160. Abiara's wall slide begins
#      "abducted to 90 degrees" and its prone exercise is "above 90 degrees". Terre tests 45 and
#      90. There is no 75 anywhere, and no source read describes a FLOOR on the W position at all.
#   2. THE QUANTITY THE DETECTOR CAN COMPUTE PUTS THE ENTIRE OBSERVED DISTRIBUTION BELOW THE CUT.
#      `angle(hip, shoulder, elbow)` is a frontal-plane reading, and in the W the elbow travels
#      down AND BACK -- an anterior-posterior component the parent spec itself rates non-observable
#      from a monocular frontal view. Measured: median W elevation 58.4 deg on REHAB24-6 Ex2's
#      markers (fires 187/208), 24.6 deg through MediaPipe (fires 206/208), 67.9 deg on Fit3D
#      `overhead_trap_raises` (fires 39/41). A criterion that fires on 90-99% of reps in a dataset
#      that is 45% correct is not measuring a fault, and its discrimination confirms it: per-subject
#      AUC 0.360 over 9 subjects, 0.510 over the eight non-degenerate ones -- at chance, and the
#      apparent inversion was a person-8 artifact.
#
# WITHDRAWN, NOT REGISTERED-SILENT, AND THE DISTINCTION IS LOAD-BEARING. A silent stub asserts
# "real fault, the sensor cannot see it". The sensor reads frontal-plane elevation angles perfectly
# well; it is the NUMBER that has no source and the QUANTITY that does not capture what the spec
# meant by the W. Same treatment as the impingement arc in arm_abduction.py. Whether a real
# W-position rule is possible needs either a source that puts a number on the W, or a metric that
# captures the A-P component a frontal reading loses. Neither is invented here.


def rule_loss_of_elevation(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag a V that never gets high enough to sit in the lower-trapezius-optimal band.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM, AND SEE ELEVATION_MILD_DEG FOR
    THE THIRD THING THIS ONE NEEDS SAID.
      FIRE THRESHOLD 120 deg: FROM THE SPEC (the low end of Mun's cited 120-145 deg LT-optimal
        band -- read the constant's comment before quoting this number as a cited threshold).
      SEVERITY RAMP 120 -> 60 deg: A RULE-LEVEL CHOICE.

    PHASE SCOPE `setup`, WHICH IS THE OPENING V. This is the first shipped rule in the project
    scoped to the 15% `setup` window -- the exact arithmetic trap that silenced Bicep Curl's
    extension term -- and it clears with a 1.25x margin measured on the REAL segmenter. See
    `arm_vw_assign_phases` for the numbers, and note that the CLOSING V (in `eccentric`) is not
    read, which under-reads the movement and errs toward silence.

    THE SPEC'S SECOND DISJUNCT (W-phase abduction < 75 deg) IS WITHDRAWN AND ABSENT -- see the
    block comment above this function for the two independent reasons.

    THE MEDIAPIPE MAGNITUDE ERROR RUNS TOWARD SILENCE HERE, WHICH IS WHY THIS RULE MAY READ A
    MAGNITUDE AT ALL. On Ex2 the estimator OVER-reads the V (median peak 166.4 deg against the
    markers' 143.8), so a rep that truly fell short reads as though it did not: fire rate 9/208
    through MediaPipe against 31/208 on the markers. Missed faults, never false ones.
    `arm_abduction.py` refused to let ANY rule read an elevation magnitude for the mirror-image
    reason -- there the error had no established direction relative to its thresholds.

    NO VIEW GATE, ONLY A DISCOUNT. An arm-elevation magnitude is the right quantity from every
    reachable view; obliquity foreshortens it, so a real shortfall reads as a DEEPER shortfall and
    the rule errs toward firing. Contrast `rule_lr_asymmetry`, which gates.
    """
    scale = 1.0 if ctx.view_type in FRONTAL_OBSERVABLE_VIEWS else _OFF_VIEW_CONFIDENCE

    def elevation(frame: CoreFrame) -> float:
        return frame.m("avg_arm_elevation_deg")

    mask = [
        frame.valid
        and frame.phase == "setup"
        and np.isfinite(elevation(frame))
        and elevation(frame) < ELEVATION_MILD_DEG
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        values = [elevation(frame) for frame in segment]
        min_elevation = float(np.nanmin(values))
        severity = severity_from_range(
            min_elevation, ELEVATION_MILD_DEG, ELEVATION_SEVERE_DEG, lower_is_worse=True
        )
        detections.append(
            build_detection(
                fault_id="vw_loss_of_elevation",
                fault_name="V Position Too Low (Off the Lower-Trap Band)",
                kg_query=VW_ELEVATION_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                # NEGATED so the reported peak frame is the LOWEST V, the frame the evidence quotes.
                score_values=[-value for value in values],
                severity=severity,
                confidence=severity * scale,
                observability=(
                    "high" if ctx.view_type in FRONTAL_OBSERVABLE_VIEWS else "medium"
                ),
                evidence={
                    "min_v_elevation_deg": round(min_elevation, 2),
                    "threshold_deg": ELEVATION_MILD_DEG,
                    "primary_label": "arm elevation in the V",
                    "primary_value": round(min_elevation, 2),
                    "primary_threshold": ELEVATION_MILD_DEG,
                },
                citation=(
                    "Mun WL, Jung EY, Lei S, Roh SY, Medicina (2025), PMC12029123, "
                    "DOI 10.3390/medicina61040645; supported by Abiara S et al., PeerJ (2025), "
                    "PMC12335237, DOI 10.7717/peerj.19861."
                ),
                citation_support=(
                    "Mun: \"The LT showed the highest muscle activity at the shoulder abduction "
                    "angle of 135° (p < 0.001)\"; its discussion cites \"shoulder abduction near "
                    "145°, aligning with the muscle fiber direction, for maximum LT activation\" "
                    "and a previous study where \"the LT activation was the highest at 120° of "
                    "shoulder abduction.\" Abiara's LT-targeting exercise uses \"arms abducted "
                    "above 90°, thumbs up.\" NOTE: Mun measures EMG during a PILATES REFORMER arm "
                    "movement at 0/90/135/160°, so the literature supplies an LT-OPTIMAL BAND of "
                    "~120–145° and NO failure threshold; the 120° floor applied here is the "
                    "parent spec's rendering of the band's low end."
                ),
            )
        )
    return detections


# FROM THE SPEC: "Flag if `asym > 12deg`" at the V peak and the W hold.
#
# ITS PROVENANCE NEEDS STATING PRECISELY, BECAUSE THE CITATION IS A DIFFERENT QUANTITY IN DIFFERENT
# UNITS -- the same non-provenance `arm_abd_lr_asymmetry` carries. Terre M & Solana-Tramunt M,
# Healthcare (Basel) 2025;13(10):1153 (PMC12110944) measures MIDDLE- AND LOWER-TRAPEZIUS EMG
# SYMMETRY during bilateral scapular retraction at 45 and 90 degrees of shoulder abduction, and
# every threshold in it is a PERCENTAGE -- "asymmetries between 10% and 15% are often associated
# with a higher risk of injury and reduced performance", on a limb-symmetry scale of 0-79% /
# 80-89% / 90-100%. NO ANGULAR THRESHOLD APPEARS ANYWHERE IN THE PAPER. Shipped unchanged anyway,
# following `ohp_asymmetric_press` and `arm_abd_lr_asymmetry`: re-expressing the rule as a
# percentage was considered and REJECTED, because changing units changes what fires (which the
# no-tuning rule covers) and it would still transfer an EMG figure to a kinematic quantity.
ASYMMETRY_MILD_DEG = 12.0
# RULE-LEVEL CHOICE MADE HERE. 2.5x the fire threshold, the `pushup.rule_hip_sag` convention.
ASYMMETRY_SEVERE_DEG = 30.0

# THE PARENT SPEC'S SECOND ASYMMETRY CUE IS NOT IMPLEMENTED, for the same frame-scale reason as
# rule 1's dropped disjunct: "or if |wrist_y_L - wrist_y_R| > 0.05 (normalized)" is not a
# well-defined criterion when the per-clip median `shoulder_width` spans 0.0591-0.4923 normalized
# units across the production corpus (8.3x). See EXCURSION_MILD_DEG's block comment for the full
# measurement; `shoulder_width` is emitted so it stays checkable.
#
# THE SPEC'S TRAILING "sustained across reps" IS ALSO NOT IMPLEMENTED: no rule in this codebase
# carries cross-rep state. `run_detector` scores one rep at a time and `merge_by_fault` reports the
# rep count afterwards, which is the framework's answer to the same question.


def rule_lr_asymmetry(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag one arm lagging the other at the V or at the W.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM, AND SEE ASYMMETRY_MILD_DEG FOR
    THE THIRD THING THIS ONE NEEDS SAID.
      FIRE THRESHOLD 12 deg: FROM THE SPEC (whose citation gives 10-15% of EMG symmetry, not
        degrees of joint angle).
      SEVERITY RAMP 12 -> 30 deg: A RULE-LEVEL CHOICE.

    PHASE SCOPE `setup` AND `peak`, FROM THE SPEC's own wording ("at the V peak and at the W
    hold"). The two windows are not adjacent, so `contiguous_true_segments` cannot weld a V
    detection to a W one.

    THIS IS THE FIRST ASYMMETRY RULE IN THIS PROJECT TO GATE ON VIEW, AND THE MEASUREMENT THAT
    FORCES IT ALSO REFUTES THE REASON `arm_abd_lr_asymmetry` GIVES FOR NOT GATING. That module
    ships live on every view on the argument that "obliquity foreshortens both arms together, so a
    real asymmetry reads smaller -- a missed fault, never a false one". On REHAB24-6 Ex2, split by
    camera orientation, taking max |L - R| over each window against the 12 deg cut:

        cam17          instrument            V window median (fires)   W window median (fires)
        front (109)    marker 3-D                   4.6 ( 3/109)              6.4 (12/109)
        front (109)    MediaPipe image 2-D          5.9 (13/109)              7.4 (20/109)
        front (109)    MediaPipe `world` 3-D       27.0 (107/109)            27.8 (104/109)
        half-prof (99) marker 3-D                   4.1 ( 0/99)               5.8 ( 5/99)
        half-prof (99) MediaPipe image 2-D         16.0 (66/99)              22.2 (88/99)
        half-prof (99) MediaPipe `world` 3-D       28.8 (96/99)              20.4 (86/99)

    FROM A TRUE FRONTAL VIEW THE DIFFERENCE METRIC BEHAVES -- 5.9 against the markers' 4.6, and the
    common-mode-cancellation argument holds. FROM AN OBLIQUE VIEW IT IS FABRICATED -- 16.0 against
    4.1, with the shipped threshold firing on 66 of 99 reps the 3-D truth calls symmetric. The near
    arm and the far arm foreshorten by DIFFERENT amounts, so obliquity does not shrink the
    asymmetry; it manufactures one. MediaPipe's own 3-D does not rescue it either (`world` is worse
    on both views), though `world` is a metric hip-centred output and NOT the image-z that
    `angle_degrees(dims=3)` consumes, so that row is a proxy in both directions.

    SO THIS RULE GATES, joining `band_pull_apart` and `bicep_curl` (which gate out the views where
    their sagittal metrics read the wrong plane) rather than `arm_abduction` (which discounts and
    stays live everywhere). The gate is not an invention: the parent spec rates this rule `high` on
    FRONT/REAR specifically, and gating to that set is implementing the rating.

    STATE THE CEILING, BECAUSE IT IS SEVERE. Production is rear_oblique 37, rear 9, unknown 3,
    side 0 over 49 pose JSONs, and `front` is unreachable under allow_front=False. So this rule is
    LIVE ON 9 OF 49 CLIPS and silent on the other 40. That is the price of not firing falsely on
    two thirds of them.

    AND TWO INFERENTIAL STEPS SIT UNDERNEATH THE GATE. Ex2's cameras are `front` and
    `half-profile`, both front-hemisphere. The gate EXCLUDES the views where fabrication was
    MEASURED (obliquity) and ADMITS one view where it was NOT (`rear`). Geometrically a
    frontal-plane difference reads the same from behind, mirrored, and |L - R| is sign-invariant --
    but MediaPipe's landmark regime on rear views is untested here, so the 9 clips that earn `high`
    earn it on a geometric argument rather than on a measurement.

    `arm_abduction.py` IS DELIBERATELY NOT EDITED ON THIS BRANCH, and the reason is evidence rather
    than scope. The measurement above is on Ex2 (arm VW) not Ex1 (arm abduction, whose unilateral
    variant makes its own false-positive rate unmeasurable in this metric); on the `image` 2-D
    cache while production runs dims=3; and on front-hemisphere obliquity while production is
    rear-hemisphere. Changing a shipped rule's firing behaviour across two inferential steps is
    exactly the move this project's honesty rules exist to prevent. The parent spec's Arm Abduction
    asymmetry NOTE is annotated and TODO.md carries a scoped check to run against abduction's own
    data.

    WHAT Ex2 SAYS ABOUT THIS RULE IS VERY LITTLE, AND IT IS ENTITLED TO SAY IT. |L - R| on the
    marker 3-D scores per-subject AUC 0.378 over the V window and 0.536 over the W window (0.375 /
    0.513 without person 8), and exceeds 12 deg on 3/208 and 17/208 reps. Ex2's incorrect reps are
    not asymmetric ones. UNLIKE Arm Abduction -- where Ex1's unilateral variant made the rule
    unvalidatable in either direction -- Ex2 is bilateral, so this is a real if uninformative
    reading rather than a variant artifact.
    """
    if ctx.view_type not in FRONTAL_OBSERVABLE_VIEWS:
        return []

    def asymmetry(frame: CoreFrame) -> float:
        return frame.m("arm_elevation_asymmetry_deg")

    mask = [
        frame.valid
        and frame.phase in {"setup", "peak"}
        and np.isfinite(asymmetry(frame))
        and asymmetry(frame) > ASYMMETRY_MILD_DEG
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        values = [asymmetry(frame) for frame in segment]
        max_asymmetry = float(np.nanmax(values))
        severity = severity_from_range(
            max_asymmetry, ASYMMETRY_MILD_DEG, ASYMMETRY_SEVERE_DEG, lower_is_worse=False
        )
        detections.append(
            build_detection(
                fault_id="vw_lr_asymmetry",
                fault_name="Left/Right Asymmetry (One Arm Lagging)",
                kg_query=VW_ASYMMETRY_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=values,
                severity=severity,
                confidence=severity,
                observability="high",
                evidence={
                    "max_arm_elevation_asymmetry_deg": round(max_asymmetry, 2),
                    "threshold_deg": ASYMMETRY_MILD_DEG,
                    "primary_label": "left/right arm elevation difference",
                    "primary_value": round(max_asymmetry, 2),
                    "primary_threshold": ASYMMETRY_MILD_DEG,
                },
                citation=(
                    "Terré M, Solana-Tramunt M, Healthcare (Basel) (2025), 13(10):1153, "
                    "PMC12110944, DOI 10.3390/healthcare13101153; scapular-dyskinesis context "
                    "from Jung EY et al., Life (2025), PMC12734928."
                ),
                citation_support=(
                    "The paper states \"asymmetries between 10% and 15% are often associated with "
                    "a higher risk of injury and reduced performance,\" on a limb-symmetry scale "
                    "of asymmetry 0–79% / limit 80–89% / normal 90–100%. IT MEASURES MIDDLE- AND "
                    "LOWER-TRAPEZIUS EMG SYMMETRY during bilateral scapular retraction at 45° and "
                    "90° of shoulder abduction, and contains NO ANGULAR THRESHOLD: the 12° cut "
                    "applied here is the parent spec's."
                ),
            )
        )
    return detections


# ALL FOUR of the parent spec's Arm VW rules are accounted for. `rule_shrug_substitution` is
# listed and permanently silent so the spec and the code stay in 1:1 correspondence -- registering
# it costs one no-op call per clip and buys an auditor the answer "yes, it is accounted for, and
# here is why it says nothing", the same trade `pushup.rule_scapular_winging`,
# `band_pull_apart.rule_loss_of_scapular_retraction` and `arm_abduction.rule_shoulder_shrug` make.
# Three of the four SUB-criteria the spec offers are absent rather than approximated: the two
# `0.05 normalized units` cues (frame-scale dependent) and the W-abduction floor (withdrawn, see
# the block comment above `rule_loss_of_elevation`).
#
# `ARM_VW_METRIC_KEYS` must stay a two-way match with what `arm_vw_compute_raw` emits (pinned by
# `test_metric_keys_match_the_emitted_metrics_exactly`): a key the tuple omits is dropped by
# `run_detector`, which builds each CoreFrame's metrics dict FROM this tuple, and read back as NaN
# by every rule.
ARM_VW_DETECTOR = MovementDetector(
    "Arm VW",
    ARM_VW_METRIC_KEYS,
    arm_vw_compute_raw,
    arm_vw_assign_phases,
    (
        rule_incomplete_excursion,
        rule_shrug_substitution,
        rule_loss_of_elevation,
        rule_lr_asymmetry,
    ),
    # `validated` stays at its default False, and for the SECOND time in this registry that is not
    # because labeled data is unavailable -- but this is the first movement whose labeled data
    # matches the variant the app models. REHAB24-6 `Ex2` IS arm VW: 208 repetitions (the LARGEST
    # labeled set of any non-squat movement -- Lunge 174, Arm Abduction 178), 94 correct / 114
    # incorrect, 9 subjects each contributing both classes, 0 flagged mocap-erroneous, marker 3-D
    # and cached MediaPipe landmarks for all 12 videos, and BILATERAL on measurement (per-rep L/R
    # excursion ratio median 0.954, within-rep r(L,R) median 0.9977). Arm Abduction had to caveat
    # that Ex1 was unilateral on 178/178 reps and therefore could not validate its own asymmetry
    # rule; NO SUCH CAVEAT APPLIES HERE.
    #
    # NOTHING HAS RUN THE CHECK. That is the only reason this is False. What a validation looks
    # like is notes/lunge-rule-validation.md, and it is scoped in TODO.md. Three things bound it in
    # advance: `rule_shrug_substitution` is silent so there is nothing to validate;
    # `rule_incomplete_excursion` and `rule_lr_asymmetry` score at chance against Ex2's labels
    # (0.494 and 0.375-0.513 per-subject), which is evidence about Ex2's error type rather than
    # about the rules; and `rule_loss_of_elevation` is the one rule Ex2 speaks to clearly, at a
    # per-subject AUC of 0.735 over the eight non-degenerate subjects while the shipped threshold
    # fires on 31/208 reps of 3-D truth and 9/208 through the estimator. Beta is the factual label
    # until the replay harness exists. Design spec section 2.
    rep_signal="avg_arm_elevation_deg",
    # `min`, matching Row and Bicep Curl and INVERSE to Arm Abduction: this movement's effort peak
    # is the W, where arm elevation is at its LOWEST. Measured on REHAB24-6 Ex2's 208 annotated
    # reps, the mean of the two arms starts at a median 140.4 deg, bottoms at 54.7 at position
    # 0.508 of the rep, and returns to 141.1.
    rep_polarity="min",
    # `extended` -- the rep opens away from the effort peak, in the V. Only Deadlift uses `flexed`.
    rep_start="extended",
    # `avg`, not an extremum of the two arms, and the choice was MEASURED rather than preferred: on
    # REHAB24-6 Ex2 left and right arm elevation correlate r = 0.9977 within-rep (min 0.9628) and
    # on Fit3D `overhead_trap_raises` 0.9989 (min 0.9962), so the arms are in phase and the mean is
    # the same excursion with per-arm landmark noise halved. Per-rep excursion of the mean on the
    # Ex2 markers: median 87.7 deg, minimum 47.0. UNLIKE Arm Abduction there is NO unilateral
    # variant to degrade on -- Ex2 is bilateral on 208/208 and movements.ts offers one Arm VW -- so
    # the stated-limitation paragraph that module needed does not apply here.
    #
    # `min_rep_seconds` stays at DEFAULT_MIN_REP_SECONDS (0.4s). Running the real segmenter over
    # all 12 Ex2 videos yields 234 reps of 1.67-25.07 s, so the tightest real rep is 4.2x the
    # floor. The TIGHTER constraint -- the phase-fraction x min_frames interaction Bicep Curl
    # section 4.3 found -- BINDS HERE, because `rule_loss_of_elevation` is scoped to the 15%
    # `setup` window: see `arm_vw_assign_phases` for the 1.25x margin and the test that pins it.
)

registry.register(ARM_VW_DETECTOR)
