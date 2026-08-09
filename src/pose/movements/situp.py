# Sit-up (curl-up) raw metrics, phase segmentation and fault rules.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `situp_compute_raw` / `situp_assign_phases` compute
# per-frame quantities and a phase label only. Every number that decides anything belongs in a
# `rule_*` function.
#
# ---------------------------------------------------------------------------------------
# ONE RULE SHIPS, ONE IS PERMANENTLY SILENT, AND TWO ARE WITHDRAWN.
# ---------------------------------------------------------------------------------------
#   rule_incomplete_rom             ships -- the only rule in the parent spec's Sit-up section
#                                   whose citation is PRIMARY and whose claim survives the
#                                   curl-up / full-sit-up variant question
#   rule_hip_flexor_dominance       REGISTERED, PERMANENTLY SILENT -- real, cited, and
#                                   human-observable fault; MediaPipe has no landmark between the
#                                   shoulders and the hips, so the trunk is a rigid segment BY
#                                   CONSTRUCTION and there is no candidate metric at all
#   excessive_speed                 WITHDRAWN, absent -- the cited source MEASURED the exact proxy
#                                   the spec proposes and found no effect (see below)
#   excessive_rom                   WITHDRAWN, absent -- no source states the threshold, the only
#                                   labeled data performs the flagged range as CORRECT, and the
#                                   knowledge graph's only ROM node for this movement means the
#                                   OPPOSITE
#
# One live rule is the thinnest detector in this registry and it is the honest outcome. Design
# spec sections 5, 6 and 10.
#
# ---------------------------------------------------------------------------------------
# THIS IS THE FIRST MOVEMENT WHOSE SUBJECT IS HORIZONTAL AND WHOSE SPEC RULES NEED A WORLD
# REFERENCE -- AND THE IMAGE HAS NONE.
# ---------------------------------------------------------------------------------------
# The parent spec defines every Group E quantity against "the floor/horizontal". The image
# horizontal is not the floor. MEASURED, not argued: EgoExo-Fitness -- the only dataset on earth
# with labeled sit-ups -- stores its `exo_l` and `exo_r` frames ROTATED A QUARTER TURN, with the
# room's ceiling running down the side of the frame, and carries NO EXIF orientation tag (PIL
# `getexif()` returns empty on all three exo views of `zOfbr6`; `Orientation` is None). A
# trunk-flexion angle read against the image horizontal reads a supine subject as 90 degrees flexed.
#
# Every rule shipped in this project so far has been immune to this by accident -- they all read
# joint-relative angles via `angle_degrees(a, b, c)`, which are invariant under camera roll. This
# module re-anchors the shipped rule to the body for the same reason, deliberately rather than
# accidentally: see `rule_incomplete_rom`. That choice is a GROUP E-WIDE convention, because
# Shoulder Bridge and Leg Abduction are both specified "vs horizontal" too.
#
# WHAT RE-ANCHORING BUYS AND WHAT IT DOES NOT, MEASURED RATHER THAN ASSUMED. It makes the METRIC
# invariant under rotation of a landmark set -- pinned end-to-end by
# tests/test_situp.py::RollInvarianceTest, which asserts byte-identical detections at 0/17/90/180/
# -90 degrees. It does NOT make the PIPELINE invariant, because MediaPipe is not roll-equivariant:
# rotating 300 real `zOfbr6/exo_l` frames by 90 degrees and re-running the estimator moves the same
# frame's hip angle by a MEDIAN OF 9.8 deg (p90 18.6, max 32.5), with detection succeeding on
# 300/300 frames either way. That is HALF the 20-degree fire threshold, produced by camera roll
# alone. Re-anchoring removes an error that would have been 90 degrees; the residue is the
# estimator's, and no landmark convention can remove it. Design spec section 8.4.
#
# ---------------------------------------------------------------------------------------
# NO VIEW GATE AND NO VIEW DISCOUNT -- A FIRST IN THIS REGISTRY, AND DELIBERATE.
# ---------------------------------------------------------------------------------------
# Three of the parent spec's four Sit-up rules are rated on `side`, which the production estimator
# has emitted on 0 of 49 real clips (arm_vw.py's re-measured census). Worse than unreachable:
# UNDEFINED. `src/pose/view_estimation.py`'s module docstring, limit 1, was written for exactly
# this case -- "for a horizontal body the frontal axis no longer maps onto image x, so the
# front/rear/*_oblique labels carry no validated meaning there. Do not gate a horizontal-movement
# rule on them." Every previous module either gated on a view set or scaled confidence by
# VIEW_UNAVAILABLE_CONFIDENCE_SCALE outside it. Doing either here would encode a false claim that
# the label was informative.
#
# AND THE ESTIMATOR IS NOT MERELY SILENT ON A SUPINE SUBJECT -- IT IS INVERTED, MEASURED ON 18 REAL
# CLIPS. Running `estimate_view_for_pose(allow_front=False)` over the six EgoExo-Fitness sit-up
# actions in all three exocentric views: the two NEAR-SAGITTAL cameras (`exo_l`, `exo_r`) return
# `rear` on 6/6 each, and the HEAD-ON camera (`exo_m`) returns `rear_oblique` on 6/6. `side` and
# `unknown` never appear. The mapping is deterministic per camera, so this is systematic, not noise
# -- and it is backwards relative to what those labels mean for an upright subject. Gating on
# {"side"} would silence this rule on 18/18 real clips; discounting outside {"front", "rear"} would
# hand FULL confidence to the head-on camera and discount nothing on the sagittal ones. Neither is
# conservative. Design spec sections 7.3 and 8.1.
#
# ---------------------------------------------------------------------------------------
# ONE DROPPED LANDMARK SILENCES EVERY SIT-UP RULE FOR THAT FRAME.
# ---------------------------------------------------------------------------------------
# `required` below lists both shoulders, both hips and both knees. If `visible_point` drops any ONE
# of them the frame is marked `valid=False` and carries no metric keys, so every rule masking on
# `frame.valid` goes silent for that frame. This mirrors every movement module since Push-up: an
# unmeasurable frame is refused wholesale rather than degraded. It bites harder here than anywhere
# else -- a sagittal view of a supine subject is the geometry in which the far-side shoulder, hip
# and knee are MOST often occluded -- and that is stated rather than relaxed.
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

SITUP_METRIC_KEYS: tuple[str, ...] = (
    "left_hip_angle_deg",
    "right_hip_angle_deg",
    "hip_angle_deg",
)


def situp_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
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

        # THE PARENT SPEC'S OWN "hip angle": angle at the hip formed by shoulder -> hip -> knee.
        # ~180 deg = trunk and thigh in a straight line, smaller = hip flexed. For a hook-lying
        # sit-up the feet are planted and the thigh is approximately stationary, so CLOSURE of this
        # angle IS rotation of the trunk -- in the same unit (degrees) as the spec's
        # trunk-flexion-vs-floor quantity, but measured against the body instead of an image axis
        # that is provably rotated 90 degrees on the only footage that exists (module header).
        #
        # Same-side throughout, so a subject rolled toward the camera does not silently blend one
        # side's shoulder with the other side's knee.
        left_hip_angle = angle_degrees(points, LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
        right_hip_angle = angle_degrees(points, RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
        finite = [v for v in (left_hip_angle, right_hip_angle) if np.isfinite(v)]
        # Degrades gracefully to whichever side was measured: this is the REP SIGNAL, and refusing
        # it when one side is occluded would disable segmentation on exactly the sagittal geometry
        # this movement is filmed in. Contrast an asymmetry metric, which must be NaN unless both
        # sides are finite -- there is no asymmetry rule in this module, so the case does not arise.
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


def situp_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> concentric -> top -> eccentric, segmented on `hip_angle_deg`.

    POLARITY MATCHES ARM VW AND IS THE INVERSE OF ARM ABDUCTION: the effort peak is the signal's
    MINIMUM (curled up, hip most flexed), so the `top` hold is the LEAST-EXTENDED 30% of the rep --
    the 30th percentile of the hip angle and below.

    Same fallbacks as every other module: an empty clip returns an empty list, a clip with no
    finite signal is entirely `unknown`, and an invalid frame is `unknown` regardless of where it
    sits (the validity check precedes the setup cutoff, so an occluded frame in the opening 15% is
    NOT labelled `setup`).

    NO SHIPPED RULE IN THIS MODULE IS PHASE-SCOPED, which is why the Bicep Curl phase-fraction trap
    (`phase_fraction * T >= min_frames / fps`, with `min_frames = max(3, ceil(0.20 * fps))`,
    base.py:197) does not bind here: `rule_incomplete_rom` reads the WHOLE rep, so its requirement
    collapses to `T >= min_frames / fps` -- 0.20 s at any frame rate. Phases are still assigned
    because `build_detection` reports a `dominant_phase` and because a future rule scoped to the
    `top` hold (the natural home for any revived ROM cue) needs them to exist. Design spec
    section 9, test 5.
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

    top_threshold = float(np.percentile(valid_hip, 30))
    lowest_index = int(np.nanargmin(np.where(np.isfinite(hip_values), hip_values, np.inf)))
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
        if np.isfinite(value) and value <= top_threshold:
            phases.append("top")
        elif index < lowest_index:
            phases.append("concentric")
        else:
            phases.append("eccentric")
    return phases


# ---------------------------------------------------------------------------------------
# STEP 0 -- KG QUERY RESOLUTION, recorded before any rule was written. Each string below was
# checked against data/kg/sports_kg_v3.graphml with `retrieve_graph_context(query, movement=
# "Sit-up")` -- the function PRODUCTION calls, not just `resolve_nodes`. Observed results:
#
#   "Incomplete Forward Reach" -> Sit-up:Incomplete Forward Reach
#       quality_impacts: Range Of Motion                                       NON-EMPTY (thin)
#   "Abdominal Disengagement" -> Sit-up:Abdominal Disengagement
#       quality_impacts: Core Stability; causes: Weak Core Stability           NON-EMPTY
#
# THE GRAPH HAS EXACTLY FOUR `Sit-up:` FAULT NODES AND THEY ARE THE EgoExo-Fitness TKV CRITERIA --
# Incomplete Forward Reach, Abdominal Disengagement, Feet Not Together, Arms Not Extended Overhead
# (the last two DANGLING, zero buckets). The general-tier stubs were EgoExo-TKV-grounded, so the
# graph models a FULL SIT-UP while the parent spec models a CURL-UP. That mismatch is what
# withdrew `excessive_rom`: the graph's ONLY ROM-adjacent Sit-up node means INCOMPLETE reach, and
# seeding an "you went too far" card from a node meaning "you didn't go far enough" would put a
# contradiction on the user's screen. Band Pull Apart, Bicep Curl, Arm Abduction and Arm VW all
# accepted THIN seeds and Arm VW accepted a SHARED one; none accepted an INVERTED one, and neither
# does this module.
#
# THE GENERIC `Range Of Motion` FALLBACK WAS REJECTED for the reason four previous modules
# rejected it, verified rather than inherited: its `corrections` bucket is "Wrapping Surface
# Adjustment" and its ten `quality_impacts` are scapular/arm-activation nodes (Scapular Upward
# Rotation, Serratus Anterior Activation, Biceps Brachii Activation, ...) -- meaningless for a
# curl-up. A semantically correct thin card beats a semantically wrong rich one.
#
# THE THIN SEED DOES NOT PRODUCE AN EMPTY HEADING, verified by reading the frontend rather than
# inferring: `FaultCard.tsx:55-57` pushes a causes/risks/cue rung only `if (...).length` and wraps
# the whole block in `rungs.length > 0`, so a one-bucket seed renders a THINNER card -- fault name,
# severity, evidence -- never a "Causes:" heading with nothing under it. Design spec section 7.1.
SITUP_ROM_KG_QUERY = "Incomplete Forward Reach"
SITUP_HIP_FLEXOR_KG_QUERY = "Abdominal Disengagement"


# FROM THE SPEC: "Flag `incomplete_rom` if peak trunk-flexion angle < ~20deg (shoulder midpoint y
# barely rises relative to hip; scapula not lifted)."
#
# ITS PROVENANCE, STATED: the 20 is the parent spec author's. NO SOURCE STATES IT.
#   Barbado PMC4519219 supplies the ENDPOINT IN KIND, not in degrees, and it is his own Methods
#     text with no reference marker: curl-ups "consisted of a head, arms and upper trunk lift to
#     the point where the scapula was lifted from the force plate, then returning to the starting
#     position."
#   Mandroukas PMC9505236 supplies the TARGET, 35-40 deg from the floor -- but SECONDARILY: the
#     sentence that states it as a limit carries a reference marker ("The stress placed on the
#     lumbar spine decreases by limiting the amount of trunk flexion to 35-40 deg [ ]") and the
#     Introduction presents it as what "is generally accepted". Mandroukas's OWN result is EMG:
#     "Rectus abdominis muscle activity was greatest in the early stages of trunk flexion and
#     decreased as the range of motion became greater, more than 35-40 deg."
#
# A floor at 20 deg is a defensible rendering of "did not get anywhere near the 35-40 deg target".
# It is not a cited threshold and it is not moved. This is the SECONDARY-SOURCING failure mode --
# right paper, right exercise, but the paper is quoting someone else -- recorded here as the fourth
# distinct way a citation_support string can be true while the rule is unsupported, after inference
# (arm-abduction impingement arc), absence (curl wrist flexion) and exercise identity (all four Arm
# VW sources). Design spec section 3.
ROM_MILD_DEG = 20.0
# RULE-LEVEL CHOICE MADE HERE. The parent spec states no severity ramp for any Sit-up fault (the
# Lunge section states its ramps explicitly, so the absence is meaningful). 6 is 0.3x the fire
# threshold; `lower_is_worse=True` because this is a "not enough" quantity, matching
# `band_pull_apart.rule_incomplete_rom`, `deadlift.rule_incomplete_lockout` and
# `arm_vw.rule_incomplete_excursion`. A display/ranking curve, not a cited quantity.
ROM_SEVERE_DEG = 6.0


def rule_incomplete_rom(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag a repetition in which the trunk barely rotates -- the scapulae never clear the mat.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM, AND SEE ROM_MILD_DEG FOR THE
    THIRD THING THIS ONE NEEDS SAID.
      FIRE THRESHOLD 20 deg of hip-angle excursion: FROM THE SPEC (whose sources give the endpoint
        in kind and the target secondarily -- neither gives this number).
      SEVERITY RAMP 20 -> 6 deg: A RULE-LEVEL CHOICE.

    THIS IS THE ONLY RULE IN THE PARENT SPEC'S SIT-UP SECTION WHOSE CLAIM SURVIVES THE VARIANT
    QUESTION, AND THAT IS WHY IT IS THE ONLY ONE THAT SHIPS. The parent spec models a CURL-UP
    (Mandroukas 35-40 deg, Barbado's scapula-lift endpoint). EgoExo-Fitness's canonical guidance
    and all four `Sit-up:` KG nodes model a FULL SIT-UP ("touch your feet with your hands"), and
    its annotators FAULT the failure to reach the feet on 28 of 82 judged actions. `excessive_rom`
    dies on that disagreement -- it would flag the correctly-performed repetitions of the only
    labeled sit-up data in existence. A repetition that never lifts the shoulders off the mat,
    however, fails the curl-up and the full sit-up ALIKE. Design spec sections 2.2 and 6.

    THE REFERENCE FRAME IS CHANGED AND THE THRESHOLD IS NOT, AND THE DIFFERENCE MATTERS UNDER THE
    NO-TUNING RULE. The spec's quantity is trunk flexion against the floor; the shipped quantity is
    the excursion of `angle(shoulder, hip, knee)` over the repetition. Same unit (degrees of trunk
    rotation), same number (20), different reference -- the body rather than an image axis that is
    provably rotated a quarter turn on the only footage available (module header). `arm_vw
    .rule_lr_asymmetry` REFUSED to re-express its threshold in different units, because changing
    units changes what fires. Nothing is re-expressed here; the alternative reference is not less
    accurate, it is meaningless.

    SCOPE IS THE WHOLE REP, NOT A PHASE, because an excursion is a property of the rep. Follows
    `arm_vw.rule_incomplete_excursion` and `deadlift.rule_incomplete_lockout`. The mask is validity
    alone and the segment is taken over ALL valid frames of the window at once rather than per
    contiguous run: splitting on an occlusion gap would hand each half a partial excursion and FIRE
    ON A GOOD REP, the opposite of this rule's intended failure direction.

    SEGMENTATION CANNOT STRUCTURALLY HIDE THE FAULT FROM THIS RULE, which is the trap worth
    checking whenever a rule reads the same signal that defines a rep. `segment_reps` thresholds on
    PERCENTILES of the signal (`low` = 5th, `high` = 95th, `enter = low + ENTER_FRACTION * span`,
    rep_segmentation.py:186-194), so it is SCALE-FREE: a shallow curl still segments as a rep and is
    still handed to this rule with its small excursion intact. Contrast the Bicep Curl extension
    term, which was silenced by an absolute interaction (`phase_fraction * T` against `min_frames`).

    AND THAT SAME SCALE-FREEDOM MAKES THIS RULE FAIL OPEN ON A CLIP CONTAINING NO MOVEMENT AT ALL.
    Stated here rather than in a follow-up, because the property that protects the rule from one
    failure is the property that causes the other, and only one of them was noticed first. Measured
    through the real `run_detector`: 60 frames of a subject holding still, with the hip angle
    oscillating by 0.4 deg of jitter, segments into THREE reps and fires this rule at
    `severity=1.0, confidence=1.0, observability="high"`, quoting an excursion of 0.74 deg.

    IT IS NOT SPECIFIC TO THIS MODULE AND WAS NOT INTRODUCED HERE -- the identical probe against
    `arm_vw.rule_incomplete_excursion`, already shipped and merged, gives 3 reps and
    `severity=1.0, confidence=1.0`. Every whole-rep "not enough travel" rule in the registry
    inherits it, because `segment_reps`'s hysteresis has no noise floor by design (that design is
    what lets a genuinely shallow rep be found at all). `band_pull_apart` escapes only when the
    signal is bit-exactly constant, which real footage never is.

    NOT REPAIRED HERE, AND THE REASON IS THE NO-INVENTED-NUMBERS RULE. Any in-rule guard is a
    minimum-excursion floor, i.e. a threshold no source states and no measurement here places. The
    honest repairs are both framework-level: a noise floor in `segment_reps`, or threading
    `RunResult.fallback` into `RuleContext` so a rule can decline a window it was handed by the
    whole-clip path (the upgrade path already recorded for the Deadlift setup-baseline defect).
    `test_a_motionless_clip_fires_this_rule_at_full_severity` PINS the current behaviour so the next
    reader meets it instead of rediscovering it. IT MATTERS MORE HERE THAN ON ARM VW: this is the
    detector's ONLY live rule, so a false positive is the entire verdict. Design spec section 8.6.

    NO VIEW GATE AND NO VIEW DISCOUNT -- THE FIRST RULE IN THIS REGISTRY WITH NEITHER. See the
    module header: for a horizontal subject `view_estimation`'s own docstring says its labels carry
    no validated meaning, so both a gate and a discount would dress an unvalidated label as
    evidence. `ctx.view_type` is deliberately unread.

    THE RESIDUAL ERROR HAS NO ESTABLISHED DIRECTION, AND SAYING OTHERWISE WOULD BE A GUESS THE
    DATA REFUTES. The tempting claim -- obliquity foreshortens an in-plane angle, so an off-axis
    curl reads SMALLER and the rule errs toward firing -- was written first and then measured, and
    the measurement contradicts its sign. On the six EgoExo-Fitness sit-up clips filmed by THREE
    SIMULTANEOUS cameras -- so every disagreement between them is pure measurement error -- the
    head-on `exo_m` view reads a LARGER median excursion than the near-sagittal `exo_l` on three
    clips and a SMALLER one on the other three. There is no direction, because `angle_degrees`
    consumes dims=3 and therefore MediaPipe's estimated z as well as x/y (under the RTMPose path,
    src/pose/rtmpose_pose_extraction.py writes z=0.0, and those ARE pure image-plane projections).
    See `arm_vw.rule_incomplete_excursion`, which could point to an estimator error running toward
    silence; this rule cannot, and does not pretend to.

    WHAT IS ESTABLISHED IS THE MAGNITUDE, AND IT IS LARGER THAN THE THRESHOLD. Per-clip cross-camera
    spread of the median excursion: 15.1, 16.3, 22.4, 34.1, 34.4, 43.2 deg -- MEDIAN 28.2 deg
    against a 20 deg fire threshold.

    THE THRESHOLD SITS LOW (20 deg, not near the sources' 35-40 deg target) FOR EXACTLY THAT REASON.
    A cut placed inside a distribution whose measurement spread is ~28 deg would fire on camera
    placement. At 20 deg the smallest per-rep excursion measured on a complete repetition, from any
    of the three cameras, is 54.6 deg -- 2.7x the cut, and the rule fires on 0 of the 18 clips.
    Design spec section 8.
    """
    segment = [
        frame for frame in core if frame.valid and np.isfinite(frame.m("hip_angle_deg"))
    ]
    if len(segment) < ctx.min_frames:
        return []

    values = [frame.m("hip_angle_deg") for frame in segment]
    highest = float(np.nanmax(values))
    lowest = float(np.nanmin(values))
    excursion = highest - lowest
    if not excursion < ROM_MILD_DEG:
        return []

    severity = severity_from_range(excursion, ROM_MILD_DEG, ROM_SEVERE_DEG, lower_is_worse=True)
    # NEGATED so `build_detection`'s argmax lands on the MOST-FLEXED frame -- the top of the curl,
    # which is the moment the evidence below is quoting. Same intent as
    # `arm_vw.rule_incomplete_excursion` and `deadlift.rule_incomplete_lockout`.
    score_values = [-value for value in values]
    return [
        build_detection(
            fault_id="situp_incomplete_rom",
            fault_name="Incomplete Range (Scapulae Never Clear the Mat)",
            kg_query=SITUP_ROM_KG_QUERY,
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
                "hip_angle_excursion_deg": round(excursion, 2),
                "supine_hip_angle_deg": round(highest, 2),
                "curled_hip_angle_deg": round(lowest, 2),
                "threshold_deg": ROM_MILD_DEG,
                "primary_label": "trunk rotation over the repetition",
                "primary_value": round(excursion, 2),
                "primary_threshold": ROM_MILD_DEG,
            },
            citation=(
                "Barbado D, Moreno-Navarro P, Vera-Garcia FJ, et al., J Hum Kinet (2015), "
                "PMC4519219, DOI 10.1515/hukin-2015-0031; endpoint corroborated by Mandroukas A, "
                "Michailidis Y, Metaxas T, J Funct Morphol Kinesiol (2022), PMC9505236, "
                "DOI 10.3390/jfmk7030067."
            ),
            citation_support=(
                "Barbado defines the exercise in his own Methods, with no reference marker: "
                "curl-ups \"consisted of a head, arms and upper trunk lift to the point where the "
                "scapula was lifted from the force plate, then returning to the starting "
                "position.\" Mandroukas describes the same endpoint as a lift \"with a rounded "
                "back to approximately 35–40° from the floor.\" NOTE: neither source states a "
                "failure threshold — the 20° floor applied here is the parent spec's — and "
                "Mandroukas's 35–40° is itself SECONDARILY sourced (the sentence stating it as a "
                "limit carries a reference marker; his own result is the EMG finding that rectus "
                "abdominis activity \"was greatest in the early stages of trunk flexion and "
                "decreased as the range of motion became greater, more than 35–40°\"). The "
                "quantity measured is hip-angle excursion, not trunk flexion against the floor, "
                "because the image carries no recoverable floor reference."
            ),
        )
    ]


def rule_hip_flexor_dominance(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Registered but PERMANENTLY SILENT -- always returns [].

    Hip-flexor dominance -- lifting the trunk as a rigid bar rotating about a fixed pelvis, feet
    anchored, instead of curling the spine segmentally -- is a real, well-cited fault. Mandroukas
    PMC9505236, read in the RAG doc, states it directly and it is HIS OWN framing rather than a
    borrowed number: "support on the feet activates the hip flexors and reduces the activity of the
    abdominal muscles"; the curl-up should be performed "with flexed unsupported knees, without
    holding the knees or feet ... (1) to avoid the uneven loading on the lumbar spine, and (2) to
    isolate the activity of the hip flexors"; and the movements "are performed by the hip flexors,
    particularly by the iliopsoas, rectus femoris, and sartorius ... [which] increases lordosis in
    the lumbar spine."

    IT IS ALSO HUMAN-OBSERVABLE, WHICH IS THE STRONGEST AVAILABLE PROOF THAT THIS IS A SENSING
    FAILURE AND NOT A FAULT-REALISM FAILURE. EgoExo-Fitness's technical-keypoint criterion "Lift
    your head, shoulders, upper back, and then lower back off the ground in turn" is marked faulted
    on 4 of 82 judged sit-up actions, with 12 individual annotator `false` votes. People watching
    the video can see a rigid trunk. The 33-landmark skeleton cannot.

    THE SENSING FAILURE IS STRUCTURAL, NOT MARGINAL -- THE TRUNK IS RIGID BY CONSTRUCTION.
    MediaPipe Pose has NO landmark between the shoulders (11/12) and the hips (23/24): no
    mid-thoracic point, no lumbar point, no sacrum. The trunk is represented as a single segment
    joining two midpoints. "Does the spine curl segmentally or rotate as a rigid bar?" is a question
    about the INTERIOR of that segment, and the interior does not exist in the landmark set.

    THAT MAKES THIS A STRONGER SILENCE THAN THE SHRUG RULES, NOT A WEAKER ONE. `arm_abduction
    .rule_shoulder_shrug` and `arm_vw.rule_shrug_substitution` each had a CANDIDATE metric
    (`ear_y - shoulder_y`) that had to be measured on two datasets before it could be rejected.
    There is no candidate metric for intra-trunk curvature at all, so the landmark-set argument --
    the one `pushup.rule_scapular_winging` and `band_pull_apart.rule_loss_of_scapular_retraction`
    make -- is sufficient here in a way it was not there.

    THE PARENT SPEC'S HEURISTIC COMPARES ONE QUANTITY AGAINST ITSELF, AND A FUTURE READER WILL
    OTHERWISE TRY TO IMPLEMENT IT. It reads: flag if "shoulder-hip-knee remain close to collinear,
    trunk_curl change < ~10-15deg" WHILE "the trunk-thigh (hip) angle closes rapidly". The spec's
    own convention block defines "Hip angle = angle at the hip landmark formed by shoulder->hip->
    knee", so BOTH clauses name THE SAME ANGLE: stay near 180 degrees while closing rapidly. The
    conjuncts are mutually exclusive and the rule as written can never fire. This is the
    vacuous-branch defect that killed `row.rule_momentum_jerk`'s second condition, Bicep Curl's
    elbow-displacement disjunct and the impingement arc's first conjunct -- a fourth instance, and
    the first found BEFORE implementation rather than after.

    THE HEEL PROXY IS NOT SUBSTITUTED, AND ITS METRIC IS NOT EMITTED. The spec offers "feet/heels
    (29/30, 31/32) remain fixed (low displacement) and knees do not lift, indicating anchored feet."
    Anchored feet is the CONDITION UNDER WHICH Mandroukas expects hip-flexor dominance, not the
    dominance itself: a trained lifter with anchored feet can still curl segmentally, and an
    untrained one with free feet can still pull rigidly. Shipping heel displacement under this
    fault_id would attach Mandroukas's abdominal-versus-iliopsoas EMG citation to a quantity he says
    nothing about -- the substitution this project forbids. The heel landmarks are correspondingly
    absent from `required` in `situp_compute_raw`, so a lost heel cannot silence the rule that DOES
    fire.

    SILENT, NOT WITHDRAWN, AND THE DISTINCTION IS LOAD-BEARING. Mandroukas genuinely backs the
    fault, so this is a sensing failure. Contrast `excessive_speed` and `excessive_rom`, which are
    ABSENT from this module entirely because no citation supports them as written (design spec
    sections 5 and 6).

    THE KG IS NOT THE GAP: "Abdominal Disengagement" resolves to `Sit-up:Abdominal Disengagement`
    with a non-empty `causes` bucket (`Weak Core Stability`) and a non-empty `quality_impacts`
    bucket (`Core Stability`). The graph has a good home for this rule. The metric is the gap.

    OPEN, RECORDED, NOT RESOLVED: a working segmental-curl rule needs either a denser body model
    (a mesh or a spine-aware 3-D estimator, which this project has measured elsewhere -- see the
    NLF and Multi-HMR notes) or a landmark set with a thoracic point. Inventing a proxy here is the
    fabrication this project's honesty rules forbid.
    """
    return []


# BOTH of the parent spec's remaining Sit-up rules are ABSENT rather than silent, and the
# distinction is deliberate. A silent stub asserts "real fault, the sensor cannot see it"; an
# absent rule asserts "no citation supports this as written".
#
# `excessive_speed` -- WITHDRAWN, three independent failures, any one sufficient:
#   1. THE CITED SOURCE MEASURED THE PROPOSED PROXY AND FOUND NOTHING. The spec's secondary signal
#      is "increased medial-lateral wobble of the shoulder midpoint (x-variance about the sagittal
#      path)". That IS Barbado PMC4519219's `SG_ML`, and Barbado's headline result is a null: "Our
#      main finding was that linear variability of SG_ML did not change significantly as speed
#      increased" and participants "were able to constrain their upper trunk motion to the sagittal
#      plane without significant changes between cadences." The quantity that DID rise, `COP_ML`,
#      is force-plate centre of pressure -- not observable from video under any camera placement.
#      This is a FIFTH distinct citation failure mode, and the only one that survives checking what
#      the source SAYS and falls only to checking what the source FOUND.
#   2. THE ~1.0 s THRESHOLD IS A PROTOCOL VALUE. The parent spec's own parenthetical gives it away
#      -- "(roughly the fastest cadence tested)". Barbado's four metronome cadences are 1 rep per
#      4 s / 2 s / 1.5 s / 1 s; C1 is the fastest setting, never nominated as a fault threshold, and
#      the paper's conclusion is that sagittal trunk control was MAINTAINED across all four.
#   3. THE PRIMARY SIGNAL DOES NOT EXIST IN THIS ARCHITECTURE. "exceeds a PER-USER BASELINE by a
#      large margin" -- no rule in this codebase carries state across clips. `run_detector` sees one
#      clip and `merge_by_fault` aggregates within it.
#   AN ABSOLUTE-CADENCE RULE IS POSSIBLE AND IS NOT BUILT HERE. EgoExo-Fitness's guidance prescribes
#   "about one repetition every 2 seconds" and its annotators fault deviation on 7/82 actions -- a
#   real number attached to this movement. It is DATASET GUIDANCE TEXT, not a peer-reviewed
#   threshold, and it belongs to the full-sit-up variant. Adopting it would ship an annotation
#   instruction as a threshold under Barbado's citation. Recorded for whoever finds a source that
#   states a cadence.
#
# `excessive_rom` -- WITHDRAWN, three independent failures:
#   1. NO SOURCE STATES 50-60 deg, and the nearest number (Mandroukas 35-40) is secondarily sourced
#      (see ROM_MILD_DEG), as is the Nachemson L3 intradiscal-pressure line the spec leans on
#      ("Nachemson [ ] reported ...", explicitly attributed).
#   2. IT CONTRADICTS THE ONLY LABELED DATA. EgoExo-Fitness's canonical sit-up guidance instructs
#      the performer to lift the lower back off the ground and touch the feet, and its annotators
#      fault the FAILURE to reach the feet on 28/82 actions. The rule would fire on the
#      correctly-performed repetitions of the only dataset in which this movement is labeled.
#   3. THE KG SEED WOULD BE SEMANTICALLY INVERTED, and that one has no workaround. See the KG block
#      above.
#   NOT SAID BY THIS WITHDRAWAL: that over-ranging a curl-up is fine. Mandroukas's own EMG result
#   (RA activity falls off beyond 35-40 deg) is a genuine argument that the extra range buys little.
#   What is missing is a source stating a threshold, a graph node that does not mean the opposite,
#   and -- before either -- a decision about WHICH SIT-UP THE APP SHIPS. Design spec section 10.
#
# `SITUP_METRIC_KEYS` must stay a two-way match with what `situp_compute_raw` emits (pinned by
# `test_metric_keys_match_the_emitted_metrics_exactly`): a key the tuple omits is dropped by
# `run_detector`, which builds each CoreFrame's metrics dict FROM this tuple, and read back as NaN
# by every rule.
SITUP_DETECTOR = MovementDetector(
    "Sit-up",
    SITUP_METRIC_KEYS,
    situp_compute_raw,
    situp_assign_phases,
    (
        rule_incomplete_rom,
        rule_hip_flexor_dominance,
    ),
    # `validated` stays at its default False, and the reason is a THIRD one this registry has not
    # recorded before. Deadlift, Row, Band Pull Apart and Bicep Curl are False because no labeled
    # data exists; Arm Abduction and Arm VW are False because nobody ran the check against data
    # that does. Sit-up is False because THE LABELED DATA THAT EXISTS DESCRIBES A DIFFERENT VARIANT.
    #
    # REHAB24-6 has no sit-up (Ex1 arm abduction, Ex2 arm VW, Ex3 table push-ups, Ex4 leg abduction,
    # Ex5 leg lunge, Ex6 squats) and Fit3D has no supine action among its 47 activity types.
    # EgoExo-Fitness has 82 human-judged sit-up actions with per-criterion fault labels -- and its
    # canonical guidance is a FULL sit-up ("touch your feet with your hands"), while the parent spec
    # specifies a CURL-UP. A validation run against it would be measuring a different exercise.
    # Design spec section 2.
    rep_signal="hip_angle_deg",
    # `min`, matching Row, Bicep Curl and Arm VW and INVERSE to Arm Abduction: this movement's
    # effort peak is the top of the curl, where the shoulder-hip-knee angle is at its SMALLEST.
    rep_polarity="min",
    # `extended` -- the rep opens away from the effort peak, lying supine with the trunk and thigh
    # at their most open. Only Deadlift uses `flexed`.
    rep_start="extended",
    # `min_rep_seconds` stays at DEFAULT_MIN_REP_SECONDS (0.4 s). The tighter constraint elsewhere
    # in this registry -- the phase-fraction interaction Bicep Curl found and Arm VW had to clear at
    # 1.25x -- DOES NOT BIND HERE, because no shipped rule in this module is phase-scoped:
    # `rule_incomplete_rom` reads the whole rep, so its requirement collapses to
    # `T >= min_frames / fps` = 0.20 s. See `situp_assign_phases`.
)

registry.register(SITUP_DETECTOR)
