# Torso Twist (seated Russian twist) raw metrics, phase segmentation and fault rules.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `torso_twist_compute_raw` /
# `torso_twist_assign_phases` compute per-frame quantities and a phase label only. Every number
# that decides anything belongs in a `rule_*` function.
#
# ---------------------------------------------------------------------------------------
# ONE RULE SHIPS, ONE IS PERMANENTLY SILENT, AND TWO ARE WITHDRAWN.
# ---------------------------------------------------------------------------------------
#   rule_trunk_not_braced           ships -- the only Torso Twist rule with a PRIMARY sentence
#                                   naming this exercise and a knowledge-graph node that means
#                                   what the rule means
#   rule_insufficient_rotation_rom  REGISTERED, PERMANENTLY SILENT -- clean roll- and
#                                   mirror-invariant metric, real fault, and NO SOURCE STATES A
#                                   RANGE; the graph's only ROM-adjacent node for this movement
#                                   is about LATERAL FLEXION, a different axis of a different
#                                   exercise (see the KG block)
#   tt_lumbar_rotation_dominant     WITHDRAWN, absent -- four independent failures, one of them
#                                   measured: the 2-D proxy the parent spec proposes disagrees
#                                   with 3-D ground truth on 16.7% of repetitions AT THE SPEC'S
#                                   OWN CUT, with a PERFECT detector
#   tt_momentum_over_control        WITHDRAWN, absent -- the cited source PRESCRIBES the
#                                   behaviour half the heuristic flags, and no number exists for
#                                   the other half
#
# One live rule ties Sit-up for the thinnest detector in this registry. Design spec
# `docs/superpowers/specs/2026-08-10-torso-twist-detector-design.md`, sections 5-7.
#
# ---------------------------------------------------------------------------------------
# FOUR ARTIFACTS IN THIS PROJECT NAME "TORSO TWIST" AND THEY DESCRIBE FOUR DIFFERENT EXERCISES.
# ---------------------------------------------------------------------------------------
# This module models the SEATED RUSSIAN TWIST, which is what three of the four say:
#
#   parent spec        rep phases written in seated geometry -- "hips fixed on floor, knees bent,
#                      torso held ~45 deg off the ground"                        SEATED
#   RAG doc            data/rag/docs/torso_twist_russian_wiki.txt, the Russian
#                      twist, "one sits on the floor and bends both knees"       SEATED
#   app card art       frontend/public/movements/torso-twist.png -- a subject
#                      seated on the floor, knees bent, hands clasped, rotated   SEATED
#   app icon           frontend/src/components/movements/MovementIcon.tsx:148 --
#                      comment AND strokes both draw a STANDING figure           standing
#
# The icon is the outlier and it is an ASSET DEFECT, recorded and NOT fixed here: changing it is
# a frontend change on a movement this branch is not about. Design spec section 2.1.
#
# AND THE TWO CORPORA THAT CONTAIN A "TORSO TWIST" ADD TWO MORE VARIANTS, WHICH IS WHY
# `validated` IS FALSE. Fit3D's `standing_ab_twists` is a STANDING CROSS-BODY KNEE-TO-ELBOW
# twist -- looked at, not inferred from the name, across four subjects and three cameras.
# EgoExo-Fitness's `Kneeling Side Torso Twist` is, by its own criteria text, a PRONE/KNEELING
# LATERAL FLEXION ("lie prone on a yoga mat ... lower your body towards the ground by bending at
# the right elbow"). Neither is a seated Russian twist. This is Sit-up's reason for
# `validated=False` -- the labeled data describes a different variant -- and NOT a new one; what
# is new is that it holds against THREE corpora at once, each modelling a different exercise.
#
# ---------------------------------------------------------------------------------------
# THE KNOWLEDGE GRAPH'S THREE TORSO TWIST FAULTS ARE SEEDED FROM THE WRONG EXERCISE, AND THE
# SEEDING SCRIPT SAYS SO IN ITS OWN WORDS.
# ---------------------------------------------------------------------------------------
# `scripts/knowledge/stub_general_movements_v3.py:152-160` records this movement's provenance as
#
#     "grounding": "EgoExo-Fitness TKV (Kneeling Side Torso Twist: pause-at-bottom 23%,
#                   lateral-flexion depth 21%, base 13%, abs)"
#
# so the graph's `Torso Twist:Insufficient Lateral Flexion Depth` is not a mis-naming: it is a
# faithful stub of a LATERAL FLEXION exercise, sitting under a movement whose four spec rules are
# all about AXIAL ROTATION. This is PRIMARY provenance -- the seeding script, not an inference
# from the node names.
#
# Leg Abduction section 7.3 established that a MISSING node reliably predicts a rule should not
# exist while a PRESENT node predicts nothing. This movement adds the sharper case: a present
# node can be ACTIVELY MISLEADING, because it describes a different movement pattern. Sit-up
# refused an INVERTED seed; this module refuses a WRONG-AXIS one, on the same reasoning.
#
# ---------------------------------------------------------------------------------------
# NO VIEW GATE AND NO VIEW DISCOUNT, FOR THE FOURTH TIME AND WITH A NEW REASON.
# ---------------------------------------------------------------------------------------
# `view_estimation.py`'s limit 1 voids the front/rear/oblique labels for a HORIZONTAL subject,
# and Leg Abduction then measured the same labels systematically INVERTED on an UPRIGHT subject
# in the exercise's own plane (0 of 210 repetitions carried a frontal-observable label when 116
# of them were filmed frontally). A seated Russian twist is neither posture -- the trunk is held
# at roughly 45 deg -- so it sits between two regimes in BOTH of which the labels have been
# measured wrong, and there is no seated-twist footage anywhere in this repository on which the
# question could be settled. Gating or discounting would dress an unmeasured label as evidence.
# `ctx.view_type` is deliberately unread. Design spec section 8.3.
#
# ---------------------------------------------------------------------------------------
# ONE DROPPED LANDMARK SILENCES EVERY RULE FOR THAT FRAME, AND THE HANDS ARE THE RISK.
# ---------------------------------------------------------------------------------------
# `required` lists both shoulders, both hips, both knees and both WRISTS -- eight, matching Leg
# Abduction and two more than Sit-up. The wrists are the price of the rep signal: this movement's
# repetition is defined by the hands swinging across the body, and there is no other landmark
# pair that tracks it. It bites harder here than the ankle requirement did on Leg Abduction,
# because the Russian twist is performed with the HANDS CLASPED TOGETHER in front of the torso --
# the configuration in which one wrist is most likely to be swallowed by the other, and by the
# forearms, exactly at the centre of the swing. Stated rather than relaxed: dropping the wrist
# requirement would leave the rep signal NaN on those frames anyway, and would additionally hand
# the brace rule a window that segmentation could not have found.
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, midpoint, distance, mean_visibility,
    severity_from_range,
)
from src.pose.movements.base import CoreFrame, MovementDetector, RuleContext
from src.pose.movements import registry
from src.pose.pose_rule_detector import PoseRuleDetection, build_detection

# `src/pose/geometry.py` exports the landmark indices the SQUAT pipeline needed and no others;
# every upper-body module since Band Pull Apart has defined the wrist pair locally rather than
# widening that module for one movement. Same convention here.
LEFT_WRIST = 15
RIGHT_WRIST = 16

# The generic "lower body" set every movement module uses for the framework-level
# `lower_body_visibility` quality field. This module's own rules never read it.
LOWER_BODY_LANDMARKS = (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

TORSO_TWIST_METRIC_KEYS: tuple[str, ...] = (
    # The brace quantity: how far the trunk has opened away from the thighs.
    "trunk_thigh_angle_deg",
    # The rotation quantity, SIGNED and therefore bipolar -- this is the rep signal.
    "twist_offset_ratio",
)


def _unit(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    return None if norm <= 0.0 else vector / norm


def _angle_between(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Unsigned angle in degrees between two 2-D vectors, NaN if either is degenerate."""
    ua, ub = _unit(a), _unit(b)
    if ua is None or ub is None:
        return math.nan
    return float(np.degrees(np.arccos(float(np.clip(np.dot(ua, ub), -1.0, 1.0)))))


def torso_twist_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
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
            LEFT_WRIST, RIGHT_WRIST,
            LEFT_HIP, RIGHT_HIP,
            LEFT_KNEE, RIGHT_KNEE,
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
        wrist_mid = midpoint(points, LEFT_WRIST, RIGHT_WRIST, dims=2)
        shoulder_width = distance(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)

        # THE BRACE QUANTITY, AND THE MIDPOINTS ARE THE POINT.
        #
        # The parent spec measures the brace as "the trunk vector hip-midpoint -> shoulder-midpoint
        # angle relative to VERTICAL". Vertical is the image's, not the world's -- the reference
        # Group E spent three movements establishing is not recoverable from a frame. Referencing
        # the trunk to the THIGHS instead makes the quantity a pure angle between two body
        # segments, so it is invariant under camera roll AND under mirroring, and it still
        # measures what the source describes: as a seated twister sags back toward the floor, the
        # trunk opens away from the thighs.
        #
        # SHOULDER MIDPOINT, NOT A SAME-SIDE SHOULDER, AND THIS IS THE ONE PLACE THIS MODULE
        # DIFFERS FROM SIT-UP'S OTHERWISE IDENTICAL CONSTRUCTION. `situp_compute_raw` uses
        # same-side `angle(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)` so that a rolled subject does not
        # blend one side's shoulder with the other side's knee. That is right for a sit-up and
        # WRONG here: this movement's whole content is the shoulder line rotating about the trunk
        # axis, which swings a same-side shoulder forward and back and injects the rotation
        # straight into the brace angle. The shoulder MIDPOINT sits ON the rotation axis, so it
        # does not move when the subject twists -- pinned by
        # `MetricLayerTest::test_the_brace_angle_does_not_move_when_the_subject_only_twists`.
        trunk_thigh_angle = _angle_between(
            None if (shoulder_mid is None or hip_mid is None) else shoulder_mid - hip_mid,
            None if (knee_mid is None or hip_mid is None) else knee_mid - hip_mid,
        )

        # THE ROTATION QUANTITY. How far the clasped hands have travelled across the body,
        # measured ALONG THE SHOULDER LINE and normalized by shoulder width.
        #
        # A DOT PRODUCT ONTO A BODY AXIS, following Leg Abduction section 1.2: invariant under
        # camera roll, and its MAGNITUDE is invariant under mirroring. The SIGN flips under
        # mirroring -- it names which way the hands went in the image, and no monocular pipeline
        # can map that onto the subject's own left and right. Nothing here claims a body side:
        # the rep signal is rectified and the only rule that reads this metric reads |.|.
        #
        # WHY NOT THE PARENT SPEC'S PROJECTED-WIDTH PROXY: see
        # LUMBAR_ROTATION_DOMINANT_WITHDRAWN. A hand TRANSLATING across the body is a
        # first-order signal; a shoulder line ROTATING about the vertical projects as
        # `width * |cos theta|`, whose derivative is ZERO at the braced centre and whose value
        # is EVEN in theta. This metric has neither defect.
        twist_offset = math.nan
        if (
            wrist_mid is not None
            and hip_mid is not None
            and shoulder_mid is not None
            and math.isfinite(shoulder_width)
            and shoulder_width > 0.0
        ):
            shoulder_axis = _unit(
                visible_point(points, LEFT_SHOULDER, dims=2)
                - visible_point(points, RIGHT_SHOULDER, dims=2)
            )
            if shoulder_axis is not None:
                twist_offset = float(np.dot(wrist_mid - hip_mid, shoulder_axis)) / shoulder_width

        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "trunk_thigh_angle_deg": trunk_thigh_angle,
                "twist_offset_ratio": twist_offset,
            }
        )

    return raw


def torso_twist_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> rotate -> peak -> return, on |twist_offset_ratio|.

    THE SIGNAL IS BIPOLAR AND THE PHASES READ ITS MAGNITUDE. The source's own repetition is one
    swing -- "each swing to a side counting as one repetition" -- so a rep runs centre -> side
    peak -> centre and the effort peak is the LARGEST |offset|. `segment_reps` is handed the
    signed metric with `rep_rectify=True`, which is the framework hook `base.py` was written
    with this movement named in the comment; the phase labels below rectify the same way so the
    two agree about where the peak is.

    Same fallbacks as every other module: an empty clip returns an empty list, a clip with no
    finite signal is entirely `unknown`, and an invalid frame is `unknown` regardless of where it
    sits (the validity check precedes the setup cutoff, so an occluded frame in the opening 15%
    is NOT labelled `setup`).

    THE SHIPPED RULE IS SCOPED TO NO PHASE BUT DOES READ `setup`, WHICH IS THE ONLY PLACE THE
    BICEP CURL PHASE-FRACTION TRAP COULD BITE. It is not a scope, so the trap's
    `phase_fraction * T >= min_frames / fps` form does not apply; what applies instead is that
    the baseline is a MEDIAN OVER A SLICE WHOSE LENGTH `segment_reps` CHOOSES, which is Row's
    setup-baseline defect. `rule_trunk_not_braced` documents the measured cost.
    """
    frame_count = len(raw)
    if frame_count == 0:
        return []

    offsets = np.asarray(
        [abs(float(item.get("twist_offset_ratio", np.nan))) for item in raw], dtype=np.float32
    )
    finite = offsets[np.isfinite(offsets)]
    if finite.size == 0:
        return ["unknown" for _ in raw]

    peak_threshold = float(np.percentile(finite, 70))
    highest_index = int(np.nanargmax(np.where(np.isfinite(offsets), offsets, -np.inf)))
    setup_cutoff = max(1, int(frame_count * 0.15))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append("setup")
            continue

        value = offsets[index]
        if np.isfinite(value) and value >= peak_threshold:
            phases.append("peak")
        elif index < highest_index:
            phases.append("rotate")
        else:
            phases.append("return")
    return phases


# ---------------------------------------------------------------------------------------
# STEP 0 -- KG QUERY RESOLUTION, recorded before any rule was written. Each string below was
# checked against data/kg/sports_kg_v3.graphml with `retrieve_graph_context(query, movement=
# "Torso Twist")` -- the function PRODUCTION calls, not just `resolve_nodes`. Observed results:
#
#   "Poor Abdominal Engagement" -> Torso Twist:Poor Abdominal Engagement
#       quality_impacts: Core Stability; causes: Weak Core Stability            NON-EMPTY (two)
#   "Unstable Base"            -> ['Unstable Base', 'Torso Twist:Unstable Base']
#       the SHARED node has zero buckets, the scoped one only `related_actions` DANGLING, and
#       AMBIGUOUS: the query matches two nodes, so a rule seeded on it would surface a
#       cross-movement node alongside its own
#   "Insufficient Lateral Flexion Depth" -> Torso Twist:Insufficient Lateral Flexion Depth
#       quality_impacts: Range Of Motion                                        NON-EMPTY (thin)
#       -- and about the WRONG AXIS; see the module header for the primary provenance
#   "Lumbar Rotation"          -> []                                            NO NODE
#   "Insufficient Rotation Range" -> []                                         NO NODE
#   "Momentum"                 -> ['Anterior Momentum Generation', 'Forward Momentum'], both
#       reached from other movements' subgraphs, both zero-bucket -- the SAME result Leg
#       Abduction section 7.2 recorded for `abd_momentum`, verbatim
#
# THE NEGATIVE FILTER HOLDS FOR A SECOND MOVEMENT AND THE POSITIVE ONE STILL DOES NOTHING. The
# two rules with no node (`tt_lumbar_rotation_dominant`, `tt_momentum_over_control`) are exactly
# the two withdrawn on citation grounds. Of the two rules that DO have a node, one ships and one
# is permanently silent -- so, as on Leg Abduction, presence predicted nothing.
TORSO_TWIST_BRACE_KG_QUERY = "Poor Abdominal Engagement"
TORSO_TWIST_ROM_KG_QUERY = "Insufficient Lateral Flexion Depth"


# FROM THE SPEC: "Flag when the trunk angle deviates from baseline by > ~15 deg".
#
# ITS PROVENANCE, STATED: the 15 is the parent spec author's. NO SOURCE STATES IT. What the RAG
# doc states -- primarily, in its own words, naming this exercise -- is the TARGET: "the torso is
# kept straight with the back kept off the ground at a 45-degree angle". A target is not a
# tolerance, and 45 deg is measured against the GROUND, which this module deliberately does not
# use as a reference (module header). So the 45 CANNOT be transferred and the 15 IS NOT DERIVED
# FROM IT; the fire threshold is a deviation from the repetition's own opening posture, in
# degrees, which is reference-frame-free in a way an absolute target is not.
#
# THIS IS THE SAME TREATMENT SIT-UP GAVE ITS 20 deg: the spec's number, shipped with its
# provenance on the record, not moved and not dressed up as cited.
BRACE_MILD_DEG = 15.0
# RULE-LEVEL CHOICE MADE HERE. The parent spec states no severity ramp for any Torso Twist fault
# (the Lunge section states its ramps explicitly, so the absence is meaningful). 40 is 2.7x the
# fire threshold; `lower_is_worse=False` because this is a "too much" quantity, matching
# `squat`'s lean rules and `leg_abduction.rule_trunk_lean`. A display/ranking curve, not a cited
# quantity.
BRACE_SEVERE_DEG = 40.0


def rule_trunk_not_braced(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag a repetition in which the trunk SAGS BACK away from its opening posture.

    DIRECTIONAL, and the parent spec's wording is not: it says "deviates from baseline by
    > ~15 deg". See the signed-deviation comment in the body for why an unsigned reading fires on
    the opposite of the fault, and for the measurement that showed it doing so.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM:
      FIRE THRESHOLD 15 deg of SAG from the repetition's opening posture: FROM THE SPEC
        (whose own source gives a 45 deg TARGET against the ground and no tolerance at all).
      SEVERITY RAMP 15 -> 40 deg: A RULE-LEVEL CHOICE.

    THIS IS THE ONLY TORSO TWIST RULE WITH A PRIMARY SENTENCE NAMING THIS EXERCISE AND A GRAPH
    NODE THAT MEANS WHAT THE RULE MEANS, WHICH IS WHY IT IS THE ONLY ONE THAT SHIPS. The RAG
    doc's "the torso is kept straight with the back kept off the ground at a 45-degree angle" is
    primary, descriptive and about the Russian twist; `Torso Twist:Poor Abdominal Engagement`
    carries two non-empty buckets and means losing exactly this. McGill 1991 supplies the
    mechanism -- during axial-torque efforts the trunk musculature co-activates to STABILIZE
    rather than to produce torque -- which is why dropping the brace is the fault and not merely
    untidy. Every other rule in the section fails on at least one of those three.

    THE SPEC'S SPINAL-ROUNDING DISJUNCT IS WITHDRAWN. Its second clause is "shoulder-midpoint
    moving forward of the hip-midpoint in x on a side view": an IMAGE-X offset, so not
    roll-invariant, whose direction ("forward") is unresolvable without knowing which way the
    subject faces -- the mirroring ambiguity no monocular pipeline resolves. `ohp_forward_head`
    withdrew its bar-path sub-criterion for the same reason and `arm_vw.rule_loss_of_elevation`
    kept its id through the loss of a branch; both precedents are followed. The `fault_id` is
    unchanged because it is the join key between the spec, the registry and stored analyses.

    SCOPE IS THE WHOLE REP, NOT A PHASE, because a deviation-from-opening-posture is a property
    of the rep. The mask is validity alone.

    THE BASELINE IS THE ROW DEFECT AND ITS COST IS MEASURED, NOT ASSUMED -- AND THE MEASUREMENT
    SEPARATES THE TWO MECHANISMS. The baseline is the median over this window's `setup` frames.
    Row derived an exactly-2x inflation of its own effective threshold from `segment_reps`
    TRIMMING the window to the excursion and leaving a `setup` slice that is already loaded.
    Measured here through the real `run_detector` on a three-swing clip, this rule's effective
    threshold is 18.0 deg against a nominal 15.0 -- an inflation of 1.20x -- and NONE of it is
    trimming: the segmenter returns the windows untrimmed (0-23, 24-47, 48-71 on a 72-frame
    clip), so the whole residual is the `setup` slice carrying part of the ramp. Fed the same
    window without the framework's median-5 smoothing the cut measures 17.5 deg, which is the
    smallest sweep step past the algebraic value 15 / (1 - f) = 17.36 for the fraction `f` of the
    ramp the 3-frame `setup` median already contains. `tests/test_torso_twist.py::
    EffectiveThresholdTest` pins all three numbers. THIS IS A STATEMENT ABOUT THAT FIXTURE: a
    swing that does not begin from rest would be trimmed, and Row's mechanism would then apply on
    top of this one.

    AND IT CARRIES PUSH-UP'S BLINDNESS, WHICH MATTERS MORE HERE THAN IT DID THERE. A baseline
    measures CHANGE, not POSTURE: a twister who sets up already collapsed and holds that
    position for the whole repetition is never flagged. On a push-up that was one rule among
    four; here it is the detector's only live rule, so the blind spot is the entire verdict for
    that user. Pinned by `test_a_brace_lost_before_the_rep_opens_is_invisible`.

    NO VIEW GATE AND NO VIEW DISCOUNT -- see the module header. `ctx.view_type` is unread.

    WHAT CAMERA PLACEMENT ALONE DOES TO THIS NUMBER, MEASURED. Fit3D films four cameras
    simultaneously, so any disagreement between them on the same repetition is pure projection
    error, and the joints are mocap ground truth projected through the real calibration -- a
    PERFECT detector. Harness: `scripts/fit3d/run_rotation_proxy_fidelity.py`. On 45 repetitions
    of `standing_ab_twists`: the ABSOLUTE trunk-thigh angle is robust, cross-camera spread of the
    per-rep median 4.5 deg (p90 10.6). The SIGNED SAG this rule scores has a median value of only
    6.3 deg there and a cross-camera spread of 5.1 deg (p90 15.7) -- taking a maximum over a
    window picks up the worst projection excursion, so the derived quantity is LESS camera-robust
    than the angle it is built from, and its p90 disagreement is the size of the 15 deg cut.
    TWO CAVEATS, AND THE SECOND BINDS HARDER. (i) The variant does not match (module header).
    (ii) `standing_ab_twists` moves the trunk mostly in FORWARD FLEXION, which is the direction
    this rule does NOT score, which is why the sag there is small. So these figures bound the sag
    direction only weakly and must not be read as "the rule barely moves on real footage". Design
    spec section 8.2.
    """
    segment = [
        frame for frame in core if frame.valid and np.isfinite(frame.m("trunk_thigh_angle_deg"))
    ]
    if len(segment) < ctx.min_frames:
        return []

    setup = [frame for frame in segment if frame.phase == "setup"]
    if not setup:
        return []
    baseline = float(np.nanmedian([frame.m("trunk_thigh_angle_deg") for frame in setup]))
    if not np.isfinite(baseline):
        return []

    # SIGNED, NOT `abs`, AND THAT IS PUSH-UP'S FINDING ARRIVING BY A THIRD ROUTE. The parent
    # spec says "deviates from baseline by > ~15 deg", and an unsigned deviation is not merely
    # non-directional but ACTIVELY INVERTED here: `trunk_thigh_angle_deg` is monotone in sag --
    # larger means the torso has laid further back toward the floor, smaller means the subject is
    # sitting UP -- so an `abs` fires on the OPPOSITE of the fault. Measured before the fix, on
    # the shipped path: a twister who sets up loose at 95 deg and then TIGHTENS to 50 deg for the
    # swing was reported "Braced Torso Lost" at severity 1.0, quoting a 45 deg deviation.
    #
    # AND THE BASELINE MAKES THAT THE COMMON CASE RATHER THAN AN EDGE ONE, because `setup` is the
    # window's first 15% -- the frames BEFORE the subject braces. Set up loose, brace, swing is an
    # ordinary way to perform the movement and produces exactly a large positive tightening.
    #
    # No new number: the same 15 deg cut on a strictly smaller set of firings. `pushup_head_drop`
    # had to add a signed metric to the metric layer for this; here the sign is already in the
    # comparison, so nothing about roll- or mirror-invariance changes.
    deviations = [frame.m("trunk_thigh_angle_deg") - baseline for frame in segment]
    peak = float(np.nanmax(deviations))
    if not peak > BRACE_MILD_DEG:
        return []

    severity = severity_from_range(peak, BRACE_MILD_DEG, BRACE_SEVERE_DEG, lower_is_worse=False)
    return [
        build_detection(
            fault_id="tt_trunk_not_braced",
            fault_name="Braced Torso Lost",
            kg_query=TORSO_TWIST_BRACE_KG_QUERY,
            retrieval_mode="kg",
            segment_metrics=segment,
            score_values=deviations,
            severity=severity,
            confidence=severity,
            # The parent spec's own rating for this fault, transcribed. It attaches that rating
            # to a `side` view; see the module header for why no view term is applied.
            observability="medium",
            evidence={
                "max_trunk_sag_deg": round(peak, 2),
                "setup_trunk_thigh_angle_deg": round(baseline, 2),
                "threshold_deg": BRACE_MILD_DEG,
                "primary_label": "trunk sag from the braced opening posture",
                "primary_value": round(peak, 2),
                "primary_threshold": BRACE_MILD_DEG,
            },
            citation=(
                "McGill SM, J Orthop Res (1991) 9(1):91-103, PMID 1824571; exercise geometry "
                "from Wikipedia, \"Russian twist\" (CC BY-SA), "
                "data/rag/docs/torso_twist_russian_wiki.txt."
            ),
            citation_support=(
                "The Wikipedia doc states the technique target primarily and in its own words: "
                "\"the torso is kept straight with the back kept off the ground at a 45-degree "
                "angle\". McGill supplies the mechanism, as his own result rather than a "
                "borrowed one: during maximal axial-torque efforts the obliques were the "
                "dominant abdominal actors (external oblique 52%, internal oblique 55% MVC vs "
                "rectus abdominis 22%) and he concludes that \"stabilization of the joints "
                "during twisting is far more important to the lumbar spine than production of "
                "large levels of axial torque\". NOTE, because the parent spec marks this "
                "\"VERIFIED\": McGill never mentions this exercise, measured isometric and "
                "30/60 deg-per-second dynamic twists in a laboratory, and states no range, no "
                "tolerance and no fault threshold. The Wikipedia 45 deg is a TARGET against the "
                "ground, not a tolerance, and is not the number applied here — the 15 deg "
                "deviation cut is the parent spec's own, and it is measured against the "
                "repetition's opening posture because the image carries no recoverable ground "
                "reference."
            ),
        )
    ]


def rule_insufficient_rotation_rom(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Registered but PERMANENTLY SILENT -- always returns [].

    A shallow twist is a real fault and this module can measure it. `twist_offset_ratio` is the
    hands' travel across the body along the shoulder axis, normalized by shoulder width: a dot
    product onto a body axis, so roll-invariant, mirror-invariant in magnitude, and free of the
    two defects that withdrew the lumbar-rotation rule (it is a first-order translation, not a
    `|cos theta|` projection). Nothing about the sensing fails. What fails is the threshold.

    NO SOURCE STATES A ROTATION RANGE. McGill 1991 -- the only peer-reviewed source this movement
    has -- reports EMG amplitudes and torque, and its two angular figures (30 and 60 deg per
    second) are the velocities of his own protocol conditions, performed by healthy subjects with
    no fault attached. The RAG doc defines the swing ("the arms should be swung from one side to
    another in a twisting motion, with each swing to a side counting as one repetition") and
    states no depth. The parent spec's own cut -- the wrist midpoint failing to pass the hip
    midline by more than "~0.08 of shoulder width" -- appears in no source, and the spec does not
    claim otherwise.

    THE GRAPH CANNOT SUPPLY THE MISSING MEANING EITHER, AND ITS NODE IS ABOUT A DIFFERENT AXIS.
    `Torso Twist:Insufficient Lateral Flexion Depth` is the only ROM-adjacent node this movement
    has, it has one non-empty bucket (`Range Of Motion`), and the seeding script records it as a
    stub of EgoExo-Fitness's KNEELING SIDE TORSO TWIST -- a prone lateral-flexion exercise
    (module header). Seeding a rotation-depth card from a lateral-flexion node would put the
    wrong movement's explanation on the user's screen. Sit-up refused an INVERTED seed on the
    same reasoning; this is the wrong-AXIS case.

    SILENT, NOT WITHDRAWN, AND THE DISTINCTION IS LOAD-BEARING. The fault is real, the metric
    works, and the sensor can see it -- what is missing is one number nobody has published. That
    is `abd_insufficient_rom`'s situation exactly, and it is registered the same way. Contrast
    `tt_lumbar_rotation_dominant` and `tt_momentum_over_control`, which are ABSENT because no
    citation supports them as written.

    THE UPGRADE PATH, RECORDED AND NOT TAKEN: a per-user baseline -- "this swing is shorter than
    your own usual" -- needs no literature threshold. The architecture has no cross-clip state.
    That is the THIRD time this wall has been hit, after `situp_excessive_speed` and
    `abd_momentum`, and it is now the most common single reason a Group E/F rule cannot ship.

    `test_the_rom_rule_never_fires_even_on_a_repetition_that_trips_the_specs_cut` pins the
    silence on the exact case the spec says to flag.
    """
    return []


# BOTH of the parent spec's remaining Torso Twist rules are ABSENT rather than silent, and the
# distinction is deliberate. A silent stub asserts "real fault, and the number or the sensor is
# missing"; an absent rule asserts "no citation supports this as written".
#
# `tt_lumbar_rotation_dominant` -- WITHDRAWN, four independent failures, any one sufficient:
#
#   1. THE CITATION SAYS NOTHING ABOUT WHERE THE ROTATION COMES FROM. The rule's claim is that
#      the twist should be driven by the thoracic spine and not the lumbar. McGill 1991 measured
#      EMG, kinematics and torque during axial-torque efforts and concluded that the musculature
#      STABILIZES rather than produces torque; he makes no thoracic-versus-lumbar
#      contribution claim at all. The parent spec's own rationale sentence is a mechanistic
#      inference layered on top of that conclusion, and the spec's own heuristic concedes the
#      point -- "true thoracic-vs-lumbar segmentation is not resolvable from 33 sparse
#      landmarks".
#
#   2. THE 0.6 RATIO IS INVENTED. No source states a hip-to-shoulder rotation ratio, in this
#      exercise or any other.
#
#   3. NO KG NODE. Both `"Lumbar Rotation"` and `"Insufficient Rotation Range"` return zero
#      matches under `movement="Torso Twist"` (KG block above).
#
#   4. AND THE PROXY IS MEASURED TO BE UNFIT, WHICH IS THE PART THAT COULD NOT HAVE BEEN ARGUED
#      FROM THE SOURCES. The spec reads axial rotation as the change in the PROJECTED HORIZONTAL
#      SEPARATION of a paired landmark line, `|x11-x12|` and `|x23-x24|`. That quantity is
#      `width * |cos theta|`, which has two structural defects and one measured one:
#
#        (a) ITS DERIVATIVE IS ZERO AT THE BRACED CENTRE, so it has least resolution exactly
#            where the rule must separate "square" from "slightly turned". Measured against a
#            real noise floor: on the Fit3D projections one degree of true rotation moves the
#            shoulder width by 0.00016 of the image width in the 0-15 deg band and 0.00109 in
#            the 45-75 deg band, while the frame-to-frame movement of MediaPipe's own shoulder
#            width over all 130 REHAB24-6 cached-landmark videos is 0.000323 of the image width.
#            One frame of that is therefore worth about 2.0 deg of rotation near the centre and
#            0.30 deg near the peak.
#        (b) IT IS EVEN IN theta, so it cannot tell a twist to one side from a twist to the
#            other. The spec's remedy is the "left-right x-ordering flip", which only occurs past
#            90 deg of rotation; the true relative trunk twist measured on Fit3D peaks at a
#            median of 44.9 deg per repetition (p90 54.1, max 58.8), so the flip never happens.
#        (c) MEASURED END TO END, WITH A PERFECT DETECTOR. Fit3D ships mocap ground truth and
#            real camera calibration, so the 2-D side can be built by projecting the truth --
#            zero landmark error, every error below is projection alone. Over 8 subjects x 4
#            cameras x 45 repetitions of `standing_ab_twists`: the proxy's per-frame MAE against
#            true rotation is 20.4 deg on the shoulder line and 17.2 deg on the hip line, the
#            latter against a true peak of only 19.7 deg. On the HIP line -- the smaller, noisier
#            and decisive term of the ratio -- the proxy is ANTI-correlated with the truth on
#            35% of repetitions. Carried through to the decision the rule actually makes: the
#            true hip/shoulder ratio fires at the spec's 0.6 cut on 64/180 records and the proxy
#            fires on 86/180, DISAGREEING ON 30/180 = 16.7%, of which 26 are the proxy firing
#            where the truth does not. (Honest qualifier: the rank correlation between the true
#            and proxy ratios is 0.876 -- the proxy is not noise, it is biased. And the corpus is
#            `standing_ab_twists`, a DIFFERENT VARIANT with a free pelvis, so the TRUTH
#            distribution of the ratio does not transfer to a seated twist with the hips pinned;
#            what transfers is the projection geometry.) Harness, so these numbers are
#            re-runnable rather than a scratch probe someone has to trust:
#            `scripts/fit3d/run_rotation_proxy_fidelity.py --jitter`.
#
#   NOT SAID BY THIS WITHDRAWAL: that rotating through the lumbar spine is fine. It is the
#   torsional-injury pathway McGill's stabilization finding implies. What is missing is a source
#   that states the fault, a number for the ratio, a graph node, and a proxy that survives
#   projection. Design spec section 7.1 and section 8.1.
#
# `tt_momentum_over_control` -- WITHDRAWN, three independent failures:
#
#   1. THE CITED SOURCE PRESCRIBES THE BEHAVIOUR THE RULE FLAGS. The heuristic flags repetitions
#      "that show no near-zero-velocity dwell at the side-peaks (no control pause)". The RAG doc
#      says: "When moving one's arms during the exercise, it is crucial to NOT STOP between
#      repetitions or else one will lose the effect of working the abdomen." The parent spec's
#      `citation_support` paraphrases that sentence as a warning "not to rely on between-rep
#      momentum", which is not what it says. Read in place, the source instructs continuous
#      movement, and the rule would fault a user for obeying it. This is a SEVENTH distinct
#      citation failure mode for the programme -- the paraphrase inverts the source's
#      instruction -- and it is a sharper case than Leg Abduction's citation/observation sign
#      disagreement, because here the contradiction is inside the quoted document.
#
#   2. THE OTHER DISJUNCT HAS NO NUMBER. "Flag reps whose peak angular speed exceeds a tempo
#      threshold" and "rep cadence above a set ceiling" -- neither ceiling exists anywhere. The
#      RAG doc's only tempo statement is directional and unquantified ("The slower one moves the
#      arms from side to side, the harder the exercise becomes"). McGill's 30 and 60 deg/s are
#      his protocol's imposed velocities, performed by healthy subjects; adopting either as a
#      fault cut would invert a condition into a fault.
#
#   3. NO KG NODE. `"Momentum"` under `movement="Torso Twist"` returns `Anterior Momentum
#      Generation` and `Forward Momentum`, both reached from other movements' subgraphs and both
#      zero-bucket -- the identical result Leg Abduction recorded for `abd_momentum`.
#
#   NOT SAID BY THIS WITHDRAWAL: that flinging the weight is fine. Design spec section 7.2.
#
# `TORSO_TWIST_METRIC_KEYS` must stay a two-way match with what `torso_twist_compute_raw` emits
# (pinned by `test_metric_keys_match_the_emitted_metrics_exactly`): a key the tuple omits is
# dropped by `run_detector`, which builds each CoreFrame's metrics dict FROM this tuple, and read
# back as NaN by every rule.
TORSO_TWIST_DETECTOR = MovementDetector(
    "Torso Twist",
    TORSO_TWIST_METRIC_KEYS,
    torso_twist_compute_raw,
    torso_twist_assign_phases,
    (
        rule_trunk_not_braced,
        rule_insufficient_rotation_rom,
    ),
    # `validated` stays at its default False, and the reason is SIT-UP'S -- the labeled data
    # describes a different variant -- not a new one. What is new is that it holds three times
    # over: REHAB24-6 has no twist at all (Ex1 arm abduction, Ex2 arm VW, Ex3 table push-ups,
    # Ex4 leg abduction, Ex5 leg lunge, Ex6 squats); Fit3D's `standing_ab_twists` is a standing
    # cross-body knee-to-elbow twist; EgoExo-Fitness's 95 judged `Kneeling Side Torso Twist`
    # actions are a prone lateral-flexion exercise. Fit3D's twist data was used here for a
    # SENSING-FIDELITY measurement -- how much true 3-D rotation survives projection -- which is
    # projection geometry and transfers across variants; it was NOT used to validate a
    # threshold, which would not. Design spec sections 2 and 8.
    rep_signal="twist_offset_ratio",
    # `max` WITH `rep_rectify`: rectifying makes each swing its own excursion from zero and `max`
    # then orients it so the effort peak -- the largest |offset| -- is the low value
    # `segment_reps` looks for. This is the framework hook `base.py:55` was written with this
    # movement named in its comment, and this module is its first user.
    rep_polarity="max",
    rep_rectify=True,
    # `extended` -- the rep opens away from the effort peak, at the braced centre with the hands
    # in front of the body. Only Deadlift uses `flexed`.
    rep_start="extended",
    # `min_rep_seconds` stays at DEFAULT_MIN_REP_SECONDS (0.4 s). The shipped rule is not
    # phase-SCOPED, so the Bicep Curl phase-fraction trap does not bind; it does read the `setup`
    # slice, which is a different interaction and is measured in the tests rather than guessed.
)

registry.register(TORSO_TWIST_DETECTOR)
