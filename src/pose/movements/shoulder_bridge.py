# Shoulder Bridge (supine bridge) raw metrics, phase segmentation and fault rules.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `shoulder_bridge_compute_raw` /
# `shoulder_bridge_assign_phases` compute per-frame quantities and a phase label only. Every
# number that decides anything belongs in a `rule_*` function.
#
# ---------------------------------------------------------------------------------------
# ONE RULE SHIPS, ONE IS PERMANENTLY SILENT, AND TWO ARE WITHDRAWN.
# ---------------------------------------------------------------------------------------
#   rule_incomplete_hip_extension  ships -- the endpoint it measures is stated in the OWN WORDS of
#                                  BOTH cited sources, its knowledge-graph seed is exact, and the
#                                  criterion is human-judged on 77 actions of the matching variant
#   rule_lumbar_hyperextension     REGISTERED, PERMANENTLY SILENT -- real fault, cited FOR THIS
#                                  EXERCISE, and the arc it needs cannot be signed from monocular
#                                  landmarks; TWO body-relative sign constructions were built and
#                                  both were MEASURED to fail (see the rule's docstring)
#   asymmetric_pelvic_drop         WITHDRAWN, absent -- its citation describes GAIT, its metric is
#                                  specified against the image horizontal, and it is a fault of the
#                                  SINGLE-LEG variant, which neither the app nor the labeled data
#                                  performs
#   knee_valgus                    WITHDRAWN, absent -- its citation describes landing and
#                                  patellofemoral pain rather than bridging, and MEASURED on
#                                  repetitions human annotators judged CORRECT the spec's own
#                                  ratio already sits at or below its own fire threshold
#
# ---------------------------------------------------------------------------------------
# THE LABELED DATA MATCHES THE VARIANT -- AND THE PIXELS IT LABELS ARE MOSTLY UNREACHABLE.
# ---------------------------------------------------------------------------------------
# THIS IS THE FIRST GROUP E MOVEMENT WHOSE GROUND TRUTH DESCRIBES THE EXERCISE THE SPEC MODELS.
# EgoExo-Fitness carries 77 human-judged `Shoulder Bridge` actions, 130 annotator records, and its
# canonical guidance -- identical across all 77 -- names this rule's endpoint verbatim: "raise your
# spine from the tailbone section by section to roll off the mat until your knees and hips are
# raised in a straight line with the shoulders". One of its twelve technical-keypoint criteria is
# "Progressively raise your body until your knees, hips, and shoulders align in a straight line",
# marked FALSE on 16 of 77 actions (21 of 130 votes). Sit-up's variant mismatch -- curl-up in the
# spec, full sit-up everywhere else -- does not recur here.
#
# WHAT BLOCKS VALIDATION IS NOT THE LABELS, IT IS THE ARCHIVE. `frames_open` downloads in 3 GiB
# parts and `.ac` is missing, so only records that fall inside `.aa` decode. Exactly TWO of the 77
# judged actions land there (`z8RAua_action_4`, `z8RAua_action_11`) -- 2.6%. That is a FOURTH
# distinct reason for `validated=False` in this registry, and the only one that is a DOWNLOAD away
# from being fixed: see the registration site at the bottom. Design spec section 2.
#
# ---------------------------------------------------------------------------------------
# `angle_degrees` IS UNSIGNED, AND THAT BREAKS TWO OF THE PARENT SPEC'S FOUR RULES AT ONCE.
# ---------------------------------------------------------------------------------------
# `src/pose/geometry.py:73` returns `degrees(arccos(...))`, whose range is [0, 180]. Two
# consequences, and only the first was anticipated by the parent spec's own reviewers:
#
#   1. `lumbar_hyperextension`'s test -- "flag if peak hip angle ... > ~190deg" -- CAN NEVER FIRE.
#      This is the FIFTH vacuous-branch defect in this registry, after `row.rule_momentum_jerk`'s
#      second condition, Bicep Curl's elbow-displacement disjunct, the arm-abduction impingement
#      arc's first conjunct and `situp_hip_flexor_dominance`'s self-comparison. It is the second
#      one caught BEFORE implementation rather than after.
#
#   2. The function is EXACTLY SYMMETRIC ABOUT 180 deg, so the SHIPPED rule inherits the same
#      defect in the opposite direction. Measured on a synthetic fixture: a pelvis 20 deg SHORT of
#      the straight line and a pelvis 20 deg PAST it both read 140.00 deg. `incomplete_hip
#      _extension`'s "< 160 deg" therefore fires on a sufficiently ARCHED bridge and reports it as
#      a bridge that was not lifted far enough -- the opposite cue. That is not a false alarm (both
#      are faults); it is a MISLABEL, and it is stated in the shipped rule's docstring rather than
#      hidden. Design spec sections 3 and 5.
#
# AND THE SIGN IS NOT RECOVERABLE -- TWO CONSTRUCTIONS BUILT, BOTH MEASURED, BOTH REFUTED.
# Both are body-relative and therefore roll-invariant in principle, which is what the Group E
# re-anchoring mandate asks for; the failure is the ESTIMATOR, not the reference frame, exactly as
# it was for Sit-up.
#   (A) hip vs ANKLE about the shoulder->knee line. Invariant under rotation AND mirroring by
#       construction (both cross products flip together, so their product does not) -- verified on
#       a synthetic fixture at 0/17/90/180/-90 degrees x mirrored, byte-identical, recovering a
#       signed angle of 120/160/180/200/240 where the unsigned one gives 120/160/180/160/120.
#       ON REAL FOOTAGE IT COLLAPSES: it reads "arched" on 57.0% and 62.3% of detected frames of
#       the two `exo_l` clips -- both of which are repetitions every annotator marked CORRECT on
#       the straight-line criterion, and on which the pelvis is BELOW the line for most of every
#       repetition by construction.
#   (B) hip vs KNEE about the shoulder->ankle line (the mat, since in a two-leg floor bridge the
#       shoulders and the feet are the two contact points). It disagrees with the subject's OWN
#       OTHER SIDE on nearly every frame: the per-frame mean of the left and right signs is 0.0 --
#       i.e. exactly opposite -- on 21 of 24 sampled frames of `action_4_exo_r`.
# Two independent constructions, two different failure modes, one conclusion: the sag/arch arc is
# not measurable here. Design spec section 4.
#
# ---------------------------------------------------------------------------------------
# NO VIEW GATE AND NO VIEW DISCOUNT -- FOLLOWING SIT-UP, AND ON FRESH EVIDENCE.
# ---------------------------------------------------------------------------------------
# `src/pose/view_estimation.py`'s module docstring, limit 1, forbids gating a horizontal-movement
# rule on `front`/`rear`/`*_oblique`, and Sit-up measured those labels to be INVERTED on supine
# subjects. This module re-measured on a different record and a different movement and found
# something slightly worse: not merely inverted but UNSTABLE. Across the six clip-views the
# estimator returns `rear` three times and `rear_oblique` three times, `side` and `unknown` never
# -- and the SAME CAMERA disagrees with itself between the two clips (`exo_l` -> `rear` on
# action_11, `rear_oblique` on action_4), at confidences from 0.02 to 0.72. A gate on {"side"}
# would silence this rule on 6/6; a discount outside {"front", "rear"} would discount half the
# clips at random. `ctx.view_type` is deliberately unread. Design spec section 7.
#
# ---------------------------------------------------------------------------------------
# ONE DROPPED LANDMARK SILENCES EVERY SHOULDER BRIDGE RULE FOR THAT FRAME.
# ---------------------------------------------------------------------------------------
# `required` below lists both shoulders, both hips and both knees -- the same six as Sit-up, and
# deliberately NOT the ankles: the only quantities that would have read them belong to the two
# withdrawn rules and to sign construction (A), which does not ship. If `visible_point` drops any
# ONE of the six the frame is marked `valid=False` and carries no metric keys, so every rule
# masking on `frame.valid` goes silent for that frame. Measured cost on the six real clip-views,
# read off `run_detector` rather than estimated: 78.2%, 89.1%, 91.7%, 99.2%, 100.0%, 100.0%. The
# two near-sagittal `exo_r` views lose nothing at all; the expensive ones are `action_4_exo_m`
# (78.2%) and the two rolled `exo_l` views (89.1%, 91.7%), which lose the far-side hip and knee.
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, angle_degrees, mean_visibility,
    severity_from_range,
)
from src.pose.movements.base import CoreFrame, MovementDetector, RuleContext
from src.pose.movements import registry
from src.pose.pose_rule_detector import PoseRuleDetection, build_detection

# The generic "lower body" set every movement module uses for the framework-level
# `lower_body_visibility` quality field. Squat-centric name; this module's own rules never read it.
LOWER_BODY_LANDMARKS = (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

SHOULDER_BRIDGE_METRIC_KEYS: tuple[str, ...] = (
    "left_hip_angle_deg",
    "right_hip_angle_deg",
    "hip_angle_deg",
)


def shoulder_bridge_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
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

        # THE PARENT SPEC'S OWN "hip angle", and for once the spec's quantity needs no
        # re-anchoring: `angle(shoulder, hip, knee)` is joint-relative, so it is invariant under
        # the camera roll that made every OTHER Group E quantity unusable. ~180 deg is the
        # straight line both sources define as the endpoint; smaller means the hip is off that
        # line. See the module header for the direction it cannot tell apart.
        #
        # Same-side throughout, so a subject rolled toward the camera does not silently blend one
        # side's shoulder with the other side's knee.
        left_hip_angle = angle_degrees(points, LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
        right_hip_angle = angle_degrees(points, RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
        finite = [v for v in (left_hip_angle, right_hip_angle) if np.isfinite(v)]
        # Degrades gracefully to whichever side was measured: this is the REP SIGNAL, and refusing
        # it when one side is occluded would disable segmentation on exactly the near-sagittal
        # geometry this movement is filmed in. Contrast an asymmetry metric, which must be NaN
        # unless both sides are finite -- the only rule that wanted one is withdrawn, so the case
        # does not arise. Measured on the six real clip-views: the two sides disagree by a median
        # 5.9-11.1 deg (p90 27.6-32.2), which is why the MEAN is taken rather than either side.
        hip_angle = float(np.mean(finite)) if finite else np.nan

        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "left_hip_angle_deg": left_hip_angle,
                "right_hip_angle_deg": right_hip_angle,
                "hip_angle_deg": hip_angle,
            }
        )

    return raw


def shoulder_bridge_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> concentric -> top -> eccentric, segmented on `hip_angle_deg`.

    POLARITY IS THE INVERSE OF SIT-UP'S, on the same signal and in the same body position. Both
    movements are supine and both read `angle(shoulder, hip, knee)`, but a sit-up's effort peak is
    the hip CLOSING (trunk curls toward the thighs, angle at its MINIMUM) and a bridge's is the hip
    OPENING (pelvis rises to the straight line, angle at its MAXIMUM). The `top` hold is therefore
    the MOST-EXTENDED 30% of the repetition -- the 70th percentile of the hip angle and above.

    Same fallbacks as every other module: an empty clip returns an empty list, a clip with no
    finite signal is entirely `unknown`, and an invalid frame is `unknown` regardless of where it
    sits (the validity check precedes the setup cutoff, so an occluded frame in the opening 15% is
    NOT labelled `setup`).

    THE SHIPPED RULE IS PHASE-SCOPED TO `top`, WHICH SIT-UP'S WAS NOT, so the Bicep Curl
    phase-fraction trap DOES bind here and was checked rather than assumed. `base.py:197` sets
    `min_frames = max(3, ceil(0.20 * fps))`, and a rule scoped to a phase covering a fraction `f`
    of a rep of `T` seconds needs `f * T * fps >= min_frames`. With `f = 0.30` and the default
    `min_rep_seconds = 0.4` the requirement is `0.30 * T >= 0.20`, i.e. `T >= 0.67 s` -- 1.67x the
    segmentation floor, so the shortest rep segmentation will emit is one this rule cannot score.
    That gap is REAL and is not closed by raising `min_rep_seconds`, because a bridge shorter than
    0.67 s is not a bridge. Measured on the six real clip-views, no analyzed repetition fell in it.
    Design spec section 9, test 6.
    """
    frame_count = len(raw)
    if frame_count == 0:
        return []

    hip_values = np.asarray(
        [float(item.get("hip_angle_deg", np.nan)) for item in raw], dtype=np.float32
    )
    valid_hip = hip_values[np.isfinite(hip_values)]
    if valid_hip.size == 0:
        return ["unknown" for _ in raw]

    top_threshold = float(np.percentile(valid_hip, 70))
    highest_index = int(np.nanargmax(np.where(np.isfinite(hip_values), hip_values, -np.inf)))
    setup_cutoff = max(1, int(frame_count * 0.15))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append("setup")
            continue

        value = hip_values[index]
        if np.isfinite(value) and value >= top_threshold:
            phases.append("top")
        elif index < highest_index:
            phases.append("concentric")
        else:
            phases.append("eccentric")
    return phases


TOP_PHASE = "top"


# ---------------------------------------------------------------------------------------
# STEP 0 -- KG QUERY RESOLUTION, recorded before any rule was written. Each string below was
# checked against data/kg/sports_kg_v3.graphml with `retrieve_graph_context(query, movement=
# "Shoulder Bridge")` -- the function PRODUCTION calls, not just `resolve_nodes`. Observed:
#
#   "Incomplete Hip Extension" -> Shoulder Bridge:Incomplete Hip Extension
#       causes: Poor Hip Extension; Weak Gluteus Maximus
#       corrections: Squeeze Glutes
#       related_actions: Shoulder Bridge                                    THREE BUCKETS, EXACT
#   "No Segmental Spinal Articulation" -> Shoulder Bridge:No Segmental Spinal Articulation
#       related_actions only                                                DANGLING
#   "Loss Of Core Engagement" -> Shoulder Bridge:Loss Of Core Engagement
#       quality_impacts: Core Stability                                     THIN
#   "Pelvic Drop" -> NO MATCH AT ALL, zero nodes
#   "Knee Valgus" -> Knee Valgus Load (Cause), Knee Valgus Control (QualityDimension); no
#       `Shoulder Bridge:` node exists
#
# THIS IS THE FIRST SEED IN THE WHOLE 16-MOVEMENT PROGRAMME THAT IS NEITHER THIN, SHARED, NOR
# INVERTED. Band Pull Apart, Bicep Curl, Arm Abduction and Sit-up all accepted ONE-bucket seeds;
# Arm VW accepted a SHARED one; Sit-up REFUSED an inverted one and withdrew the rule. This one
# resolves to a movement-scoped fault node carrying two causes AND a corrective cue, so the card
# the user sees is rich by the graph's own content rather than by the rule padding it -- and the
# cue it carries, "Squeeze Glutes", is exactly what both sources say the endpoint trains.
SHOULDER_BRIDGE_EXTENSION_KG_QUERY = "Incomplete Hip Extension"


# FROM THE SPEC: "Target is a straight line (~170-180deg, 0deg hip flexion); flag
# `incomplete_extension` if peak hip angle < ~160deg (hips remain visibly flexed / pelvis low)."
#
# ITS PROVENANCE, STATED, AND IT IS THE STRONGEST IN GROUP E BY A MARGIN.
#   THE ENDPOINT IS PRIMARY IN BOTH SOURCES, IN THEIR OWN WORDS, WITH NO REFERENCE MARKER.
#     Escamilla PMC11048684, Methods: the subject "pushed through the feet and hands, lifting the
#       buttocks upwards until the hips were in a neutral position with 0 deg hip flexion, with the
#       knees, hips, and shoulders approximately in a straight line."
#     Colonna PMC11981018, describing the exercise: "the pelvis is lifted from the floor until it
#       reaches the neutral angular position of the hip."
#   AND ESCAMILLA ALSO GIVES THE START, WHICH NO OTHER RULE IN THIS PROGRAMME HAS HAD: "the supine
#     hook-lying position with the hips flexed approximately 50 deg". So the source states BOTH
#     ends of the repetition -- roughly 130 deg of hip angle rising to 180 -- and therefore an
#     expected excursion near 50 deg. Measured on the six real clip-views the observed central
#     tendency lands on it: median hip angle 128-134 deg on the two cleanest views against
#     Escamilla's ~130 start.
#
# WHAT IS *NOT* PRIMARY, SAID PLAINLY BECAUSE THE PARENT SPEC MARKS IT "VERIFIED". The RATIONALE
# sentences the parent spec quotes are Colonna citing other people: "The greatest hip extension
# torque during the SBE occurs when the hip is nearly fully extended [ ]" and "In this position,
# the GM is recruited more than at any other angle within the range of motion [ , , ]" both carry
# reference markers. They ARE in the document verbatim, so the parent spec's verification claim is
# true as far as it goes -- but this is the SECONDARY-SOURCING failure mode Sit-up named, recurring
# on a different movement. Here it lands on the rule's WHY rather than on its WHAT, which is why
# the rule ships: the endpoint that defines the threshold is primary twice over.
#
# THE 160 IS STILL THE PARENT SPEC AUTHOR'S NUMBER. Neither source states a failure threshold. What
# the sources state is the TARGET (0 deg hip flexion, straight line), and 160 renders "20 deg of
# hip flexion still remaining" -- a defensible reading of "the hips stay flexed and sagging". It is
# not moved. Design spec section 5.
EXTENSION_MILD_DEG = 160.0
# RULE-LEVEL CHOICE MADE HERE. The parent spec states no severity ramp for any Shoulder Bridge
# fault (the Lunge section states its ramps explicitly, so the absence is meaningful). 130 deg is
# Escamilla's stated STARTING position -- a repetition whose peak never left the start has achieved
# nothing, which is the natural floor for "worst possible". This is the first severity ramp in the
# registry whose severe end is a source-stated quantity rather than a round number; it is still a
# display/ranking curve, not a cited threshold.
EXTENSION_SEVERE_DEG = 130.0


def rule_incomplete_hip_extension(
    core: list[CoreFrame], ctx: RuleContext
) -> list[PoseRuleDetection]:
    """Flag a repetition whose pelvis never reaches the shoulder-hip-knee straight line.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM.
      FIRE THRESHOLD 160 deg of peak hip angle: FROM THE SPEC. Both sources state the TARGET
        (0 deg hip flexion / straight line) in their own words; neither states a failure cut.
      SEVERITY RAMP 160 -> 130 deg: A RULE-LEVEL CHOICE, with a source-stated severe end
        (Escamilla's ~50 deg of starting hip flexion). See EXTENSION_SEVERE_DEG.

    THE FAULT IT NAMES IS THE ONE HUMAN ANNOTATORS ACTUALLY SEE IN THIS MOVEMENT. Of the twelve
    EgoExo-Fitness technical-keypoint criteria for Shoulder Bridge, "Progressively raise your body
    until your knees, hips, and shoulders align in a straight line" is marked FALSE on 16 of 77
    judged actions -- the second most-faulted criterion, and the most-faulted one that is about the
    movement rather than about hand placement or breathing. No criterion in the list describes
    over-arching. That distribution is the empirical answer to the conflation below.

    IT CANNOT TELL A SAG FROM AN ARCH, AND THAT IS A PROPERTY OF `angle_degrees`, NOT OF THIS
    THRESHOLD. The function is unsigned and exactly symmetric about 180 deg (module header, and
    `test_the_metric_cannot_distinguish_a_sag_from_an_arch` pins it): a pelvis 20 deg short of the
    line and a pelvis 20 deg past it both read 140 deg, so a sufficiently arched bridge is reported
    under this fault_id and the user is told to lift HIGHER when the correct cue is the opposite.
    Two body-relative sign constructions were built to repair it and BOTH were measured to fail on
    real footage (module header, design spec section 4). What makes shipping it anyway defensible
    rather than negligent is the paragraph above: in the only labeled data for this movement, the
    direction this rule assumes is the direction annotators fault, 16/77 times, and the other
    direction is not among the criteria at all. Stated, not hidden, and not repaired by inventing
    a sign the estimator cannot support.

    SCOPE IS THE `top` PHASE, NOT THE WHOLE REP, and that follows the spec ("peak taken over the
    top-hold frames") rather than the module next door. `situp.rule_incomplete_rom` reads the whole
    rep because an EXCURSION is a property of a rep; a PEAK POSITION is a property of the moment
    the position is held, and scoping it to `top` keeps a transient overshoot during the concentric
    from standing in for a hold that never got there. The phase-fraction cost of that choice is
    computed in `shoulder_bridge_assign_phases` rather than left implicit.

    IT DOES NOT FAIL OPEN ON A MOTIONLESS CLIP, WHICH IS A FIRST FOR A "NOT ENOUGH" RULE HERE.
    `situp.rule_incomplete_rom`, `arm_vw.rule_incomplete_excursion` and every other whole-rep
    excursion rule fire at severity 1.0 on a subject holding still, because `segment_reps`
    thresholds on PERCENTILES of the signal and is therefore scale-free -- jitter segments into
    reps and a tiny excursion reads as a tiny range. This rule reads an ABSOLUTE POSITION instead
    of a range, so a motionless subject is judged on where they actually are: someone lying flat
    (~130 deg) is correctly told they never bridged, and someone holding a good bridge (~175 deg)
    is correctly left alone. `test_a_motionless_clip_is_judged_on_position_not_range` pins both
    directions. The trap is not absent from the framework; this rule simply does not read the
    quantity that springs it.

    NO VIEW GATE AND NO VIEW DISCOUNT -- the second rule in this registry with neither, following
    `situp.rule_incomplete_rom`. See the module header: measured here, the estimator's labels are
    not merely unvalidated for a supine subject but unstable, with one camera returning different
    labels for two clips of the same person in the same room. `ctx.view_type` is deliberately
    unread.

    IT FIRES ON 5 OF THE 6 REAL CLIP-VIEWS, ALL OF WHICH ARE REPETITIONS ANNOTATORS JUDGED CORRECT
    ON THIS EXACT CRITERION. That is the single most important fact about this rule and it is
    stated first rather than buried. Measured through `run_detector`:

        clip-view                view          peak hip angle (deg)   severity
        z8RAua_action_11_exo_l   rear(0.38)    160.3 163.8 161.5      SILENT
        z8RAua_action_4_exo_r    rear(0.71)    159.5                  0.02
        z8RAua_action_11_exo_r   rear(0.72)    (whole-clip fallback)  0.08
        z8RAua_action_4_exo_l    rear_obl(0.02) 167.9 149.8 155.5     0.15
        z8RAua_action_11_exo_m   rear_obl(0.55) 131.4 139.3 143.9     0.95
        z8RAua_action_4_exo_m    rear_obl(0.17) 110.6 112.3 124.9     1.00

    THE CENSUS SPLITS CLEANLY BY CAMERA GEOMETRY, NOT BY REPETITION. On the four NEAR-SAGITTAL
    clip-views the rule is silent once and otherwise fires at 0.02, 0.08 and 0.15 -- a low-severity
    card saying a repetition landed a few degrees short of the geometric target, on footage whose
    annotators scored both actions 3/5 with the comment "the movement was not completed according
    to the instructional text". A binary technical-keypoint `True` is not ground truth for a
    continuous quantity, and these three are not obviously wrong. On the two AXIAL `exo_m`
    clip-views it fires at 0.95 and 1.00, and those are simply wrong: viewed down the body's long
    axis the sagittal hip angle is foreshortened into meaninglessness (median 90 deg on exo_m
    against 128-134 deg on exo_r, for the SAME repetitions). This rule has no way to tell those
    cameras apart, because the view estimator cannot -- it labels the axial views `rear_oblique`
    and one near-sagittal view `rear_oblique` too.

    WHAT THAT ESTABLISHES AND WHAT IT DOES NOT. It establishes the MAGNITUDE of the measurement
    error: the same repetitions read 110.6 to 167.9 deg depending only on which of three
    simultaneous cameras is used, a spread of ~50-58 deg against the 20 deg margin between this
    threshold and the straight line it renders. It does NOT establish a fire rate (n = 2 actions,
    1 subject) and it does NOT establish a bias direction. The tempting claim -- "MediaPipe
    systematically under-reads a straight-line bridge by about 20 deg, which is exactly the margin
    this threshold relies on" -- is consistent with the data and is NOT made here, because one
    subject whose reps the annotators say were not completed as instructed cannot support it.
    `situp.rule_incomplete_rom` had a first draft that claimed a sign for its residual error and
    the measurement refuted it; the lesson is applied in advance this time.

    NOT REPAIRED, AND EVERY AVAILABLE REPAIR IS FORBIDDEN. Moving 160 is tuning a cited endpoint to
    flatter the estimator. Gating on "is this view sagittal enough" needs a threshold no source
    states. Gating on the view LABEL is what the module header measures to be unstable. The rule
    ships live with the census above pinned by
    `test_the_axial_view_fires_this_rule_at_near_full_severity_on_a_correct_rep`, so the next
    reader meets this instead of rediscovering it. The 77-action validation that would settle it is
    one completed download away -- see the registration site. Design spec section 8.

    ONE MORE THING THE RUN SURFACED, RECORDED NOT FIXED. On `action_11_exo_r` -- the CLEANEST
    footage available, 100% landmark detection -- `segment_reps` found 2 repetitions and marked
    BOTH partial, so `run_detector` took the `only_partial_reps` fallback and handed this rule the
    entire 16-second clip as one window. The rule then scored a whole clip as though it were one
    repetition. That is the same gap the Deadlift setup-baseline defect recorded: `RunResult
    .fallback` is not threaded into `RuleContext`, so a rule cannot decline a window it was handed
    by the whole-clip path. It bites hardest on the best footage, which is worth knowing.
    """
    segment = [
        frame
        for frame in core
        if frame.valid and frame.phase == TOP_PHASE and np.isfinite(frame.m("hip_angle_deg"))
    ]
    if len(segment) < ctx.min_frames:
        return []

    values = [frame.m("hip_angle_deg") for frame in segment]
    peak = float(np.nanmax(values))
    if not peak < EXTENSION_MILD_DEG:
        return []

    severity = severity_from_range(
        peak, EXTENSION_MILD_DEG, EXTENSION_SEVERE_DEG, lower_is_worse=True
    )
    # NEGATED so `build_detection`'s argmax lands on the LEAST-extended frame of the hold -- the
    # worst moment of the fault, which is what the evidence below is quoting. Same intent as
    # `situp.rule_incomplete_rom` and `deadlift.rule_incomplete_lockout`.
    score_values = [-value for value in values]
    return [
        build_detection(
            fault_id="bridge_incomplete_hip_extension",
            fault_name="Incomplete Hip Extension at the Top",
            kg_query=SHOULDER_BRIDGE_EXTENSION_KG_QUERY,
            retrieval_mode="kg",
            segment_metrics=segment,
            score_values=score_values,
            severity=severity,
            confidence=severity,
            # The parent spec's own rating, transcribed. It rates this cue `high` on `side`, and
            # `side` is unreachable in production -- but see the module header: the honest
            # qualifier belongs in prose, not in a fabricated confidence scale factor.
            observability="high",
            evidence={
                "peak_hip_angle_deg": round(peak, 2),
                "target_hip_angle_deg": 180.0,
                "threshold_deg": EXTENSION_MILD_DEG,
                "residual_hip_flexion_deg": round(180.0 - peak, 2),
                "primary_label": "peak hip angle at the top",
                "primary_value": round(peak, 2),
                "primary_threshold": EXTENSION_MILD_DEG,
            },
            citation=(
                "Escamilla RF, Lewis C, Fukuda D, et al., Bioengineering (2024), PMC11048684, "
                "DOI 10.3390/bioengineering11040356; endpoint corroborated by Colonna S, "
                "D'Alessandro A, Tarozzi R, Casacci F, Cureus (2025), PMC11981018, "
                "DOI 10.7759/cureus.80349."
            ),
            citation_support=(
                "Escamilla defines both ends of the repetition in his own Methods, with no "
                "reference marker: the start is \"the supine hook-lying position with the hips "
                "flexed approximately 50°\", and the subject \"pushed through the feet and hands, "
                "lifting the buttocks upwards until the hips were in a neutral position with 0° "
                "hip flexion, with the knees, hips, and shoulders approximately in a straight "
                "line.\" Colonna describes the same endpoint in his own words: \"the pelvis is "
                "lifted from the floor until it reaches the neutral angular position of the "
                "hip.\" NOTE: neither source states a failure threshold — the 160° cut applied "
                "here is the parent spec's rendering of \"20° of hip flexion still remaining\" — "
                "and the RATIONALE sentences the parent spec quotes (greatest hip-extension "
                "torque near full extension; peak gluteus-maximus recruitment there) are Colonna "
                "citing other authors behind reference markers, not his own result. The quantity "
                "measured is unsigned, so it cannot distinguish a pelvis short of the straight "
                "line from one arched past it."
            ),
        )
    ]


def rule_lumbar_hyperextension(
    core: list[CoreFrame], ctx: RuleContext
) -> list[PoseRuleDetection]:
    """Registered but PERMANENTLY SILENT -- always returns [].

    Lumbar hyperextension -- driving the pelvis so high that the low back arches into lordosis and
    anterior pelvic tilt, so back extension replaces hip extension -- is a real fault and, unlike
    the two withdrawn rules below, it is cited FOR THIS EXERCISE rather than for gait or landing.
    Colonna PMC11981018: "In patients performing bridging exercises, excessive and uncontrolled
    lumbar lordosis and anterior pelvic tilt (APT) are frequently observed due to the dominant
    hyperactivity of the ES", and "Others recommend maintaining a straight alignment of the
    shoulders, hips, and thighs during bridging to prevent excessive APT". Both sentences carry
    reference markers, so the support is SECONDARY -- but it is secondary about the right exercise,
    which is the distinction the two withdrawals below fail.

    THE SPEC'S TEST CANNOT FIRE, AND THAT ALONE WOULD NOT BE ENOUGH TO SILENCE IT. "Flag if peak
    hip angle overshoots the straight line into extension (> ~190 deg)" is unreachable because
    `angle_degrees` returns `arccos`, range [0, 180] (module header, fifth vacuous-branch defect in
    this registry, second caught before implementation). That is a defect in the SPEC'S ARITHMETIC,
    repairable in principle by signing the angle -- which is exactly what was attempted.

    WHAT SILENCES IT IS THAT THE SIGN IS NOT RECOVERABLE, MEASURED TWICE. Both constructions are
    body-relative and roll-invariant by design, so this is not the Group E reference-frame problem
    recurring; it is the estimator.
      (A) Hip vs ANKLE about the shoulder->knee line. Correct on a synthetic fixture and invariant
          under rotation and mirroring, recovering 120/160/180/200/240 deg where the unsigned angle
          gives 120/160/180/160/120. On real footage it reads "arched" on 57.0% and 62.3% of
          detected frames of two repetitions every annotator marked CORRECT -- repetitions on which
          the pelvis is below the line for most of their duration by construction.
      (B) Hip vs KNEE about the shoulder->ankle line, i.e. taking the mat to be the line joining
          the two contact points of a two-leg floor bridge. It disagrees with the subject's own
          other side on nearly every frame: the mean of the left and right signs is 0.0 -- exactly
          opposite -- on 21 of 24 sampled frames of `action_4_exo_r`.
    Near the straight line, where this rule would have to decide, both cross products go to zero
    and the sign becomes noise. That is not a tuning problem; it is the geometry of asking which
    side of a line a point lies on when the point is ON the line.

    AND EVEN WITH A PERFECT SIGN, THE SPEC'S THRESHOLD WOULD FLAG A NORMAL POSITION. Colonna states
    that "the range of motion during maximum physiological hip extension is approximately 20 deg
    beyond the neutral position", so 190 deg -- 10 deg beyond neutral -- sits squarely INSIDE
    normal hip extension. A rule firing there would fault healthy movement. This is the same class
    of objection that helped withdraw `situp_excessive_rom`: the source that grounds the fault
    contradicts the number chosen to detect it.

    THE PELVIS-HEIGHT PROXY IS NOT SUBSTITUTED, AND ITS METRIC IS NOT EMITTED. The spec offers
    "hip-midpoint y at top rises above the straight line interpolated between shoulder-midpoint and
    knee-midpoint". Image `y` is the quantity the parent spec's own Group E update block names as
    unrecoverable -- EgoExo ships these frames rolled, with no EXIF tag -- and re-expressing it as
    a perpendicular distance would need a threshold no source states. Worse, Colonna makes such a
    threshold non-transferable in principle: hip torque and pelvis height during the bridge depend
    on the foot-to-pelvis distance and the knee flexion angle, both of which the performer chooses.
    A single normalized lift height means different things at different setups.

    SILENT, NOT WITHDRAWN, AND THE DISTINCTION IS LOAD-BEARING. Colonna genuinely backs this fault
    for this exercise, so this is a SENSING failure. Contrast `asymmetric_pelvic_drop` and
    `knee_valgus`, which are ABSENT from this module entirely because their citations are about
    other activities (design spec sections 6 and 7).

    OPEN, RECORDED, NOT RESOLVED: a working arch rule needs either a lumbar landmark (MediaPipe has
    none between shoulders and hips) or a depth-bearing 3-D estimator that can place the pelvis off
    the shoulder-knee line with a reliable sign. This project has measured such estimators
    elsewhere -- see the NLF and Multi-HMR notes -- and that is the honest upgrade path. Inventing
    a sign here is the fabrication this project's rules forbid.
    """
    return []


# BOTH of the parent spec's remaining Shoulder Bridge rules are ABSENT rather than silent, and the
# distinction is deliberate. A silent stub asserts "real fault, the sensor cannot see it"; an
# absent rule asserts "no citation supports this as written".
#
# `asymmetric_pelvic_drop` -- WITHDRAWN, four independent failures, any one sufficient:
#   1. THE CITATION IS ABOUT GAIT, NOT ABOUT BRIDGING. The parent spec's citation_support quotes
#      Colonna's Trendelenburg passage, and that passage says what it is about in its own first
#      words: "In a Trendelenburg gait, the Gmed is unable to maintain the pelvis on the opposite
#      side during single-leg support, causing the pelvis to drop when the swing leg is in the
#      air." It sits in Colonna's section on Gmed weakness and its consequences for walking,
#      running and landing. The other bridge source, Escamilla PMC11048684, studies UNIPEDAL
#      bridging directly and never mentions pelvic drop, pelvic level, or Trendelenburg at all
#      (checked, not assumed). This is the EXERCISE-IDENTITY failure mode that withdrew the
#      arm-abduction impingement arc and put all four Arm VW sources on notice.
#   2. THE METRIC IS SPECIFIED AGAINST THE IMAGE HORIZONTAL. "Frontal-plane pelvic-tilt angle =
#      angle of the left-hip(23)->right-hip(24) line vs horizontal" is precisely what the parent
#      spec's own Group E update block says is not recoverable from an image.
#   3. IT IS A FAULT OF A VARIANT NOTHING HERE PERFORMS. The spec itself scopes it "esp. single-leg
#      bridge" and says the pelvis "typically drops on the unsupported (swing-leg) side". The app
#      models the two-leg bridge, and EgoExo-Fitness's canonical guidance -- the only labeled data
#      -- is two-leg throughout ("keep legs hip-width apart"). Its twelve criteria contain nothing
#      about a level pelvis.
#   4. THE KNOWLEDGE GRAPH HAS NO HOME FOR IT. `retrieve_graph_context("Pelvic Drop", movement=
#      "Shoulder Bridge")` matches ZERO nodes -- not a thin seed, not an inverted one, none.
#   A BODY-RELATIVE RE-ANCHORING IS POSSIBLE AND IS NOT BUILT HERE. Pelvic tilt against the
#   SHOULDER line rather than the image horizontal would be roll-invariant and is the obvious
#   repair for failure 2. It does not touch failures 1, 3 or 4, so it would ship a metric with no
#   source, no variant and no graph node behind it.
#
# `knee_valgus` -- WITHDRAWN, three independent failures:
#   1. THE CITATION IS ABOUT LANDING AND PATELLOFEMORAL PAIN, NOT ABOUT BRIDGING. "Powers [ ]
#      theorized that hip abductor and external rotator weakness may lead to excessive hip
#      adduction and internal rotation, resulting in increased knee valgus" is explicitly
#      attributed to Powers and sits in Colonna's passage on hip dysfunction and knee pathology,
#      whose surrounding sentences are about ACL injury during LANDING and lateral patellar
#      tracking. Nothing in either bridge source reports knee valgus during a bridge.
#   2. MEASURED ON CORRECT REPETITIONS, THE SPEC'S OWN RATIO IS ALREADY AT OR BELOW ITS OWN CUT.
#      Across the six real clip-views the median `knee_width/ankle_width` is 0.726, 0.895, 0.911,
#      0.927, 1.020 and 1.027, and the per-clip minimum reaches 0.043. The spec's fire threshold is
#      0.85 (squat's shipped one is 0.82). Two of the six clip-views sit below 0.85 on their MEDIAN
#      frame -- on repetitions every annotator judged correct on every alignment criterion. The
#      cut is inside the noise, not above it. Viewed near-sagittally a supine subject's two knees
#      and two ankles project onto nearly the same points, so the ratio is a quotient of two small,
#      noisy numbers.
#   3. THE SPEC'S FORM IS NOT EVEN ROLL-INVARIANT. It specifies `knee_width = |x(25)-x(26)|`, an
#      image-x projection, which a rolled camera collapses. The codebase's own shipped precedent
#      (`pose_feature_extraction.py:296`, feeding `squat.rule_knees_inward`) uses the full 2-D
#      distance, which is invariant. Failure 2 was measured with the INVARIANT form, so fixing this
#      does not rescue the rule.
#   NOT SAID BY THIS WITHDRAWAL: that knees collapsing inward during a bridge is fine. Colonna's
#   Gmed material is a genuine mechanism and the fault is plausible. What is missing is a source
#   observing it in this exercise, and a metric whose noise floor is below its own threshold.
#
# `SHOULDER_BRIDGE_METRIC_KEYS` must stay a two-way match with what `shoulder_bridge_compute_raw`
# emits (pinned by `test_metric_keys_match_the_emitted_metrics_exactly`): a key the tuple omits is
# dropped by `run_detector`, which builds each CoreFrame's metrics dict FROM this tuple, and read
# back as NaN by every rule.
SHOULDER_BRIDGE_DETECTOR = MovementDetector(
    "Shoulder Bridge",
    SHOULDER_BRIDGE_METRIC_KEYS,
    shoulder_bridge_compute_raw,
    shoulder_bridge_assign_phases,
    (
        rule_incomplete_hip_extension,
        rule_lumbar_hyperextension,
    ),
    # `validated` stays at its default False, and the reason is a FOURTH one this registry has not
    # recorded before -- and the first that is fixable by a download rather than by research.
    #
    # Deadlift, Row, Band Pull Apart and Bicep Curl are False because NO LABELED DATA EXISTS.
    # Arm Abduction and Arm VW are False because NOBODY RAN THE CHECK against data that does.
    # Sit-up is False because THE LABELED DATA DESCRIBES A DIFFERENT VARIANT.
    # Shoulder Bridge is False because THE LABELS EXIST AND MATCH, AND THE PIXELS ARE MISSING.
    #
    # EgoExo-Fitness has 77 human-judged Shoulder Bridge actions across 130 annotator records, and
    # its canonical guidance names this detector's endpoint verbatim -- no variant mismatch of the
    # kind that stopped Sit-up. But `frames_open` downloads in 3 GiB parts, part `.ac` is missing,
    # and only records inside `.aa` decode: exactly 2 of the 77 judged actions (2.6%) are
    # recoverable. REHAB24-6 has no bridge (Ex1 arm abduction, Ex2 arm VW, Ex3 table push-ups, Ex4
    # leg abduction, Ex5 leg lunge, Ex6 squats) and Fit3D has no supine action among its 47
    # activity types, so there is no substitute corpus.
    #
    # A 77-action validation run against the exact criterion this rule implements is therefore ONE
    # COMPLETED DOWNLOAD away, and that is recorded in TODO.md as an action rather than as a
    # limitation. Design spec section 2.
    rep_signal="hip_angle_deg",
    # `max`, INVERSE to Sit-up on the identical signal in the identical body position: a bridge's
    # effort peak is the hip OPENING to the straight line, so it is the signal's MAXIMUM. Row,
    # Bicep Curl, Arm VW and Sit-up use `min`; Arm Abduction and this module use `max`.
    rep_polarity="max",
    # `extended` names the end of the signal AWAY FROM the effort peak, which for this movement is
    # the hips FLEXED on the mat -- the framework's word and the anatomy's word point opposite ways
    # here, and the framework's is what this flag selects. It puts repetition boundaries at the
    # supine rest position (`_windows_from_plateaus`), which is where a bridge starts and ends.
    # Only Deadlift uses `flexed`.
    rep_start="extended",
    # `min_rep_seconds` stays at DEFAULT_MIN_REP_SECONDS (0.4 s). Unlike Sit-up, the shipped rule
    # here IS phase-scoped, so the Bicep Curl phase-fraction interaction binds: a repetition
    # shorter than 0.67 s segments but cannot be scored. Raising this to close the gap would be
    # tuning a framework constant to flatter a rule; the gap is documented in
    # `shoulder_bridge_assign_phases` instead, and no analyzed repetition on the real footage fell
    # inside it.
)

registry.register(SHOULDER_BRIDGE_DETECTOR)
