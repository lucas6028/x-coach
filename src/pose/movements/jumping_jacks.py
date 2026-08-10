# Jumping Jacks (side-straddle hop) raw metrics, phase segmentation and fault rules.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `jumping_jacks_compute_raw` /
# `jumping_jacks_assign_phases` compute per-frame quantities and a phase label only. Every number
# that decides anything belongs in a `rule_*` function.
#
# ---------------------------------------------------------------------------------------
# NO RULE SHIPS LIVE. TWO ARE PERMANENTLY SILENT, THREE ARE WITHDRAWN, AND THE DETECTOR IS
# DELIBERATELY NOT REGISTERED. THE LABELED DATA DECIDED IT.
# ---------------------------------------------------------------------------------------
#   rule_incomplete_leg_rom       PERMANENTLY SILENT -- the fault is real and HUMAN-JUDGED (the
#                                 most-failed of EgoExo-Fitness's eight criteria, 9.9% of 121
#                                 actions), the metric is the movement's own definition of a
#                                 repetition, and the KG node is the only one this movement has
#                                 that is grounded in this exercise. What fails is the number:
#                                 the parent spec's 1.3 cut FIRES ON 79% OF THE REPETITIONS
#                                 HUMANS JUDGED CORRECT. That is `abd_insufficient_rom`'s
#                                 situation exactly.
#   rule_incomplete_arm_rom       PERMANENTLY SILENT -- needs NO threshold at all (its criterion
#                                 is a landmark comparison, not a number) and its metric is
#                                 clean; silent because the only source that states its target
#                                 attributes an injury association, in the same document, to the
#                                 range of motion the rule would coach users toward
#   jj_knee_valgus_landing        WITHDRAWN, absent -- THE METRIC IS CONFOUNDED BY THE VERY
#                                 STANCE THIS MOVEMENT IS DEFINED BY, measured with a
#                                 zero-parameter control: replacing both knees with PERFECTLY
#                                 STRAIGHT-LIMB positions still puts the ratio below the 0.82 cut
#                                 on 68.5% of open-phase frames, against 79.4% for the real
#                                 knees. The rule reads stance geometry, not knee alignment.
#   jj_stiff_landing              WITHDRAWN, absent -- the cited paper's OWN STIFF CONDITION
#                                 WOULD NOT FIRE THIS RULE (77 deg of flexion, against a cut
#                                 below 20 deg), and the cue carries a measured projection bias
#                                 pointing in the firing direction
#   jj_landing_asymmetry          WITHDRAWN, absent -- no KG node, a three-quantity disjunction
#                                 that would put an arm, a foot or a knee behind one fault_id,
#                                 and cross-rep state this architecture has never had
#
# THIS IS THE FIRST DETECTOR IN THE PROGRAMME THAT IS NOT REGISTERED. See the block above
# `JUMPING_JACKS_DETECTOR` for why, and for what does work and is kept.
#
# Design spec `docs/superpowers/specs/2026-08-10-jumping-jacks-detector-design.md`. Measurements:
# `notes/jumping-jacks-rule-validation.md`, harness `src/egoexo/jumping_jacks_validation.py`.
#
# ---------------------------------------------------------------------------------------
# THE VARIANT MATCHES, THE LABELS ARE THE RICHEST IN THE PROGRAMME, AND THEY ARE ABOUT
# DIFFERENT FAULTS. THAT IS A NEW REASON, NOT AN EXISTING ONE.
# ---------------------------------------------------------------------------------------
# EgoExo-Fitness carries 121 human-judged `Jumping Jacks` actions -- the LARGEST judged class in
# that dataset -- with per-criterion True/False verification by 2+ annotators. And unlike Torso
# Twist, THE EXERCISE MATCHES: feet open and close, arms drive overhead.
#
# Its eight criteria, with the fraction of the 121 actions each was judged failed on:
#
#     Perform the jump by opening and closing your feet.                fault  9.9% (12/121)
#     Keep your arms tense and ready for movement.                      fault  8.3% (10/121)
#     Press your arms down using the strength of your back.             fault  8.3% (10/121)
#     Use the movement of your arms to help drive your body to jump.    fault  6.6% (8/121)
#     Lift your arms using shoulder strength.                           fault  4.1% (5/121)
#     Relax your calves as much as possible during the jump.            fault  4.1% (5/121)
#     Maintain a steady head position, avoiding lowering or raising     fault  2.5% (3/121)
#       your head.
#     Tighten your waist and abdominal muscles for stability.           fault  0.8% (1/121)
#
# THE PARENT SPEC'S FIVE RULES ARE KNEE VALGUS, STIFF LANDING, ARM ROM, LEG ROM AND LANDING
# ASYMMETRY. EXACTLY ONE PAIR OVERLAPS -- "Perform the jump by opening and closing your feet" and
# `jj_incomplete_leg_rom`. Nothing in the labeled corpus judges valgus, landing stiffness,
# overhead arm reach or left-right asymmetry, and nothing in the parent spec judges arm tension,
# back-driven arm return, calf relaxation or head steadiness.
#
# `validated` stays False and THE REASON IS NEW. Sit-up's reason -- used again by Shoulder Bridge
# and three times over by Torso Twist -- is that the labeled data describes a DIFFERENT VARIANT.
# Here the variant is right and the labels are richer than anything this programme has met; what
# they do not do is judge the faults the spec wrote rules for. Torso Twist's update block records
# that the count of distinct reasons "stays at five"; this is the sixth. Design spec section 2.
#
# ---------------------------------------------------------------------------------------
# THE KNOWLEDGE GRAPH'S THREE JUMPING JACKS FAULTS ARE SEEDED FROM TWO EXERCISES BLENDED
# TOGETHER, AND THE BLEND IS REPRODUCIBLE TO THE DECIMAL.
# ---------------------------------------------------------------------------------------
# `scripts/knowledge/stub_general_movements_v3.py:133-141` records this movement's provenance as
#
#     "grounding": "EgoExo-Fitness TKV (Jumping/Clap Jacks: arm tension 8-27%, foot split 10%,
#                   arm-leg coordination)"
#
# `Clap Jacks` is a SEPARATE EgoExo action class with 74 judged actions and its own guidance --
# "clap your hands while jumping back and forth with ALTERNATING feet", pectoral-driven, no
# side-straddle at all. Recomputing the seed's own statistic over the two classes separately:
#
#     "Keep your arms tense ..."            Jumping Jacks   8.3% (10/121)
#     "Keep your arms tense."               Clap Jacks     27.0% (20/74)
#     "... opening and closing your feet."  Jumping Jacks   9.9% (12/121)
#
# so "arm tension 8-27%" is the two ENDS OF A RANGE SPANNING TWO DIFFERENT EXERCISES, and "foot
# split 10%" is this exercise alone. Torso Twist established that a KG node can be actively
# misleading because it faithfully describes a different movement; this is the milder cousin -- a
# node seeded from a BLEND, of which one component is correct, and the correct component
# (`Jumping Jacks:Incomplete Foot Split`) is the one the silent ROM rule would seed from.
#
# ---------------------------------------------------------------------------------------
# NINE LANDMARKS ARE READ AND ONLY EIGHT ARE REQUIRED, WHICH NO EARLIER MODULE HAS DONE.
# ---------------------------------------------------------------------------------------
# `required` is both shoulders, both hips, both knees and both ankles -- what the rep signal and
# the leg rule need. The wrists and the nose are read for `hands_above_head_ratio` and are
# deliberately NOT required: they are the fastest-moving landmarks in the movement (the hands
# sweep a half-circle every repetition and motion-blur at the top), and requiring them would let a
# blurred hand mark the frame invalid and silence a rule that never reads it. Torso Twist required
# its wrists because its REP SIGNAL was built on them; here the rep signal is the feet. The
# principle is unchanged and only its application moved: REQUIRE WHAT THE REP SIGNAL AND THE RULES
# NEED. Design spec section 4.3.
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, midpoint, distance, mean_visibility,
)
from src.pose.movements.base import CoreFrame, MovementDetector, RuleContext
from src.pose.pose_rule_detector import PoseRuleDetection

# `src/pose/geometry.py` exports the landmark indices the SQUAT pipeline needed and no others;
# every module since Band Pull Apart has defined the extra ones locally rather than widening that
# module for one movement. Same convention here.
NOSE = 0
LEFT_WRIST = 15
RIGHT_WRIST = 16

# The generic "lower body" set every movement module uses for the framework-level
# `lower_body_visibility` quality field. This module's own rules never read it.
LOWER_BODY_LANDMARKS = (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

JUMPING_JACKS_METRIC_KEYS: tuple[str, ...] = (
    # How far apart the feet are, in shoulder widths. THE REP SIGNAL, and the quantity the silent
    # leg-ROM rule would read.
    "stance_width_ratio",
    # How far the hands are above the head ALONG THE TRUNK AXIS, in shoulder widths. Signed;
    # positive means overhead. NaN whenever a wrist or the nose is missing -- see the module
    # header for why that is a per-metric gap rather than an invalid frame.
    "hands_above_head_ratio",
    #
    # NOTE WHAT IS NOT HERE. `knee_width_to_ankle_width` (the withdrawn valgus rule's quantity)
    # and any knee FLEXION angle (the withdrawn stiff-landing rule's) are deliberately absent: a
    # withdrawn rule leaves no metric behind for something to quietly start reading. Both are
    # recomputed from landmarks inside `src/egoexo/jumping_jacks_validation.py` so the evidence
    # for the withdrawals stays re-runnable without the module carrying a refuted quantity.
)


def _unit(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    return None if norm <= 0.0 else vector / norm


def jumping_jacks_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
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

        shoulder_width = distance(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        ankle_width = distance(points, LEFT_ANKLE, RIGHT_ANKLE, dims=2)

        # THE REP SIGNAL, AND IT IS THE MOVEMENT'S OWN DEFINITION. The RAG doc defines a jumping
        # jack as "jumping to a position with the legs spread wide ... and then return to a
        # position with the feet together", so the feet ARE the repetition. A RATIO OF TWO
        # EUCLIDEAN DISTANCES is invariant under camera roll and under mirroring, and it is
        # scale-free, so it survives the 8.3x span of normalized `shoulder_width` that Arm VW
        # measured across the production corpus. The parent spec writes this as
        # `|x27-x28| / |x11-x12|`, an IMAGE-X difference, which is neither roll-invariant nor well
        # defined for a rolled camera; the distance form agrees with it exactly when the camera is
        # upright and degrades gracefully when it is not.
        #
        # AND BOTH TERMS ARE FRONTAL-PLANE WIDTHS, WHICH IS WHY THE RATIO SURVIVES OBLIQUITY. An
        # azimuthally oblique camera compresses the shoulder line and the ankle line by nearly the
        # same factor, so the factor cancels to first order. That is what makes the measurement in
        # `notes/jumping-jacks-rule-validation.md` a statement about the performers rather than
        # about the camera placement.
        stance_width_ratio = (
            ankle_width / shoulder_width
            if math.isfinite(shoulder_width) and shoulder_width > 1e-8
            else math.nan
        )

        # THE ARM QUANTITY, AND IT IS A COMPARISON RATHER THAN A NUMBER. The parent spec's
        # criterion is "both wrists fail to rise above the nose (y15 and y16 > y0, remembering y
        # increases downward)" -- a comparison between two body landmarks. Read in IMAGE Y it
        # needs the image vertical to be the world vertical, which Group E spent three movements
        # establishing is not recoverable from a frame. Projected onto the TRUNK AXIS
        # (`hip_mid -> shoulder_mid`) the identical comparison becomes a dot product onto a body
        # axis: roll-invariant, mirror-invariant, and still exactly the spec's criterion. Leg
        # Abduction section 1.2's rule, applied to an upright subject whose trunk IS the axis.
        #
        # NaN unless BOTH wrists and the nose are present: an "overhead" reading from one arm is
        # not a small reading, it is no reading -- `arm_vw`'s construction for the same problem.
        hands_above_head = math.nan
        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        wrist_mid = midpoint(points, LEFT_WRIST, RIGHT_WRIST, dims=2)
        nose = visible_point(points, NOSE, dims=2)
        if (
            shoulder_mid is not None
            and hip_mid is not None
            and wrist_mid is not None
            and nose is not None
            and math.isfinite(shoulder_width)
            and shoulder_width > 1e-8
        ):
            trunk_up = _unit(shoulder_mid - hip_mid)
            if trunk_up is not None:
                hands_above_head = float(np.dot(wrist_mid - nose, trunk_up)) / shoulder_width

        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "stance_width_ratio": stance_width_ratio,
                "hands_above_head_ratio": hands_above_head,
            }
        )

    return raw


def jumping_jacks_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> opening -> open -> closing, on `stance_width_ratio`.

    The RAG doc's repetition is "jumping to a position with the legs spread wide ... and then
    return to a position with the feet together", so one repetition is one open-and-close cycle
    and the effort peak is the WIDEST stance. `segment_reps` is handed the same metric with
    `rep_polarity="max"` and no rectification -- the signal is unipolar, because the feet never
    cross -- so the phase labels and the segmenter agree about where the peak is.

    `open` IS THIS MODULE'S LANDING WINDOW, AND THAT IS A DELIBERATE SUBSTITUTION. The parent spec
    keys three of its five rules to "the landing frame" -- a single-frame impact event. An impact
    instant is not identifiable from landmarks: there is no ground plane in the image, no force
    plate, and Group E established that the image vertical is not the world vertical, so "the
    lowest point of the hips" is not available either. What IS identifiable, and roll-invariantly,
    is the WIDE-STANCE PLATEAU: the feet reach maximum separation at touchdown and stay there
    through ground contact until push-off, so the open-phase frames contain the landing by
    construction. Design spec section 4.4.

    Same fallbacks as every other module: an empty clip returns an empty list, a clip with no
    finite signal is entirely `unknown`, and an invalid frame is `unknown` wherever it sits (the
    validity check precedes the setup cutoff, so an occluded frame in the opening 15% is NOT
    labelled `setup`).
    """
    frame_count = len(raw)
    if frame_count == 0:
        return []

    widths = np.asarray(
        [float(item.get("stance_width_ratio", np.nan)) for item in raw], dtype=np.float32
    )
    finite = widths[np.isfinite(widths)]
    if finite.size == 0:
        return ["unknown" for _ in raw]

    peak_threshold = float(np.percentile(finite, 70))
    widest_index = int(np.nanargmax(np.where(np.isfinite(widths), widths, -np.inf)))
    setup_cutoff = max(1, int(frame_count * 0.15))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append("setup")
            continue

        value = widths[index]
        if np.isfinite(value) and value >= peak_threshold:
            phases.append("open")
        elif index < widest_index:
            phases.append("opening")
        else:
            phases.append("closing")
    return phases


# ---------------------------------------------------------------------------------------
# STEP 0 -- KG QUERY RESOLUTION, recorded before any rule was written. Each string below was
# checked against data/kg/sports_kg_v3.graphml with `retrieve_graph_context(query, movement=
# "Jumping Jacks")` -- the function PRODUCTION calls, not just `resolve_nodes`. Observed:
#
#   "Incomplete Foot Split"     -> Jumping Jacks:Incomplete Foot Split
#       only `related_actions`                                              DANGLING
#   "Insufficient Arm Tension"  -> Jumping Jacks:Insufficient Arm Tension
#       only `related_actions`                                              DANGLING
#   "Poor Arm-Leg Coordination" -> Jumping Jacks:Poor Arm-Leg Coordination
#       causes: Poor Neuromuscular Control                                  NON-EMPTY (one)
#   "Knee Valgus"               -> ['Knee Valgus Load', 'Knee Valgus Control']
#       Knee Valgus Load    risks: ACL Injury
#       Knee Valgus Control quality_impacts: Frontal Plane Stability, Joint Stiffness
#       BOTH NON-EMPTY, BOTH SHARED -- neither is a `Jumping Jacks:` node; they are reached from
#       the Squat flagship's subgraph, and the query matches TWO of them
#   "Stiff Landing"             -> []                                       NO NODE
#   "Landing"                   -> []                                       NO NODE
#   "Asymmetry"                 -> ['Symmetry'], shared, reached from Squat  NO SCOPED NODE
#
# THE NEGATIVE FILTER HOLDS FOR A THIRD MOVEMENT. The two rules with NO node at all
# (`jj_stiff_landing`, `jj_landing_asymmetry`) are exactly two of the three withdrawn -- Leg
# Abduction section 7.3's finding, reproduced again. The positive signal again predicts nothing on
# its own: of the three scoped fault nodes, the one the ROM rule would seed from is DANGLING, the
# one with a real bucket corresponds to NO rule in the parent spec, and the third is about arm
# TENSION where the spec's arm rule is about arm RANGE.
JUMPING_JACKS_LEG_ROM_KG_QUERY = "Incomplete Foot Split"
# Recorded because it is the REASON the arm rule has no graph support, not because it is used:
# the only arm node this movement has is about tension, not range.
JUMPING_JACKS_ARM_KG_QUERY = "Insufficient Arm Tension"


# FROM THE SPEC: "Flag when the ratio stays below ~1.3 (feet barely wider than the shoulders)".
#
# ITS PROVENANCE, STATED: the 1.3 is the parent spec author's and NO SOURCE STATES IT. The RAG doc
# (Wikipedia, "Jumping jack", CC BY-SA) states the TARGET -- "jumping to a position with the legs
# spread wide" -- and a target in words is not a tolerance in shoulder widths.
#
# IT IS KEPT AT 1.3 AND THE RULE IS SILENCED INSTEAD, WHICH IS THE WHOLE POINT. On the 11
# reachable judged actions the widest stance of a repetition has a median of about 1.15, so a cut
# anywhere near the observed distribution could be manufactured -- and manufacturing one is
# exactly what this programme forbids. `abd_insufficient_rom` was silenced rather than moved for
# the same reason. See `rule_incomplete_leg_rom`.
LEG_ROM_MILD_RATIO = 1.3


def rule_incomplete_leg_rom(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """PERMANENTLY SILENT -- always returns [].

    THIS RULE HAD MORE GOING FOR IT THAN ANY OTHER IN THE SECTION AND THE DATA STILL SILENCED IT,
    which is why the reasoning is worth reading in full.

    WHAT IT HAS:
      A PRIMARY SENTENCE NAMING THIS EXERCISE. The RAG doc defines the movement as "jumping to a
      position with the legs spread wide ... and then return to a position with the feet
      together".
      A KNOWLEDGE-GRAPH NODE GROUNDED IN THIS EXERCISE. `Jumping Jacks:Incomplete Foot Split`,
      and the seeding script's own grounding figure ("foot split 10%") reproduces from the labels
      as 9.9% (12/121) -- the only one of this movement's three nodes not contaminated by the
      Clap Jacks blend (module header).
      HUMAN CORROBORATION THAT THE FAULT IS REAL. It is the MOST-FAILED of EgoExo-Fitness's eight
      criteria for this exercise.
      A CLEAN METRIC. `stance_width_ratio` is a ratio of two frontal-plane distances: roll-,
      mirror- and scale-invariant, and first-order invariant to azimuthal obliquity because both
      terms foreshorten together (`jumping_jacks_compute_raw`).

    WHAT FAILS IS THE NUMBER, AND IT FAILS MEASURABLY. Replayed through the real `run_detector`
    over the 11 judged actions recoverable from the truncated EgoExo archive -- 31 (action,
    camera) pairs and 91 scored repetitions, three simultaneous exo cameras each -- the parent
    spec's 1.3 cut fires on 79.1% OF SCORED REPETITIONS (90.3% of pairs). Every one of those
    repetitions belongs to an action a human judged TRUE on "Perform the jump by opening and
    closing your feet", so every firing is a false positive by the only judgement available. The
    median widest stance of a repetition is 1.163 shoulder widths, i.e. THE CORRECT POPULATION
    SITS BELOW THE CUT.
    `notes/jumping-jacks-rule-validation.md` carries the exact figures and the harness that
    produced them.

    THE ALTERNATIVE READING IS STATED RATHER THAN DISMISSED: the criterion may simply be laxer
    than the rule -- an annotator asked whether the feet opened and closed may be answering "did
    they open at all", not "did they open wide enough". That would make the 79% a disagreement
    about strictness rather than an error. It does not change the conclusion, because a rule that
    fires on 79% of what the only available human judgement accepts cannot be shown to a user.

    THREE CONFOUNDS THAT DO NOT EXPLAIN IT, CHECKED RATHER THAN ASSUMED:
      RESOLUTION. EgoExo ships preprocessed 456x256 frames, so landmark error in normalized units
      is roughly 2.8x production's. That inflates VARIANCE; it does not move a median by 12%.
      OBLIQUITY. Both terms of the ratio are frontal-plane widths and foreshorten together, so a
      common-mode azimuth error cancels to first order -- which is exactly why this metric was
      chosen over the spec's image-x form.
      SEGMENTATION. Median validity 1.00, no action on the whole-clip fallback, and the cadence
      measurement below shows repetitions were found cleanly.

    THE UPGRADE PATH IS CONCRETE, AND THAT IS NEW FOR A SILENT RULE IN THIS PROGRAMME. Every
    earlier silent rule needed either a paper nobody has written (`abd_insufficient_rom`,
    `tt_insufficient_rotation_rom`) or a per-user baseline this architecture does not have. This
    one needs neither: EgoExo-Fitness judges this exact criterion on 121 actions, 12 of them
    FAILED, so a cut separating them could be READ OFF HUMAN JUDGEMENT rather than authored. What
    blocks it is that the `frames_open` download is missing its `.ac` part, leaving 11 of the 121
    reachable and all 11 judged correct -- so there is no positive class. That is a DOWNLOAD, not
    a research programme.

    SCOPE, RECORDED FOR WHOEVER WAKES IT UP: the `open` phase (the wide-stance plateau that
    contains the landing), reading the MAXIMUM stance width over that window, with `min_frames`
    tested against the WHOLE repetition rather than the phase -- the Bicep Curl phase-fraction
    trap, which would otherwise silence this rule structurally on any jack faster than about
    1.3 Hz (design spec section 4.4).
    """
    return []


def rule_incomplete_arm_rom(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """PERMANENTLY SILENT -- always returns [].

    AND THE REASON IS NOT THE USUAL ONE. Every other silent rule in this registry
    (`abd_insufficient_rom`, `tt_insufficient_rotation_rom`, `bridge_lumbar_hyperextension`, and
    `rule_incomplete_leg_rom` above) is silent because a NUMBER or a SENSOR is missing. Neither is
    missing here:

      THE CRITERION NEEDS NO NUMBER. The parent spec's own test is "both wrists fail to rise above
      the nose" -- a comparison between two body landmarks. `hands_above_head_ratio` is that
      comparison projected onto the trunk axis, so it fires at zero. There is no threshold to
      lack, which makes this the only rule in the section immune to the failure that silenced the
      leg rule.

      THE SENSOR IS FINE. A dot product onto a body axis: roll-invariant, mirror-invariant,
      scale-free. Pinned by `InvarianceTest` and by
      `test_the_metric_it_would_have_used_is_computed_and_correct`.

    IT IS SILENT BECAUSE THE ONLY SOURCE THAT STATES ITS TARGET CAUTIONS, IN THE SAME DOCUMENT,
    AGAINST THE RANGE OF MOTION IT WOULD COACH USERS TOWARD. `data/rag/docs/jumping_jacks_wiki.txt`
    supplies the target -- "the hands go overhead, sometimes in a clap" -- and four sections later
    records: half-jacks "were created to prevent rotator cuff injuries, which have been linked to
    the repetitive movements of the exercise. They are like regular jumping jacks, but the arms go
    halfway above the head instead of all the way above it." A rule telling a user their arms did
    not go high enough is pushing them toward the range that paragraph attaches an injury
    association to. The parent spec noticed this and wrote it into its own rationale while still
    proposing the rule.
    THIS IS AN EIGHTH DISTINCT CITATION FAILURE MODE for the programme, after inference, absence,
    exercise identity, secondary sourcing, a source-measured null on the proposed proxy, a
    citation/observation sign disagreement, and a paraphrase that inverts the source. Nothing here
    is misquoted: it is a COUNTER-INDICATION inside the supporting source.

    TWO FURTHER FAILURES, EITHER SUFFICIENT ALONE:
      NO KG NODE MEANS WHAT THIS RULE MEANS. The only arm node this movement has is
      `Insufficient Arm Tension` -- tension, not range -- and it is dangling. Seeding an arm-range
      card from an arm-tension node is Torso Twist's wrong-axis mistake in different clothing.
      THE LABELED DATA DOES NOT JUDGE IT. Three of the eight criteria concern the arms -- tension,
      shoulder-driven lift, back-driven press-down -- and none concerns how high the hands travel.

    `test_the_arm_rule_never_fires_even_when_the_hands_stay_at_shoulder_height` pins the silence
    on the exact case the parent spec says to flag.
    """
    return []


# THREE of the parent spec's five Jumping Jacks rules are ABSENT rather than silent, and the
# distinction is the one this registry has always drawn: a silent stub asserts "real fault, and
# the number, the sensor or the corroboration is missing"; an absent rule asserts "no citation
# supports this as written, or the quantity it reads does not measure it".
#
# `jj_knee_valgus_landing` -- WITHDRAWN, three failures, and the first is the one that could not
# have been argued from the sources:
#
#   1. THE METRIC IS CONFOUNDED BY THE VERY STANCE THIS MOVEMENT IS DEFINED BY, AND THE CONFOUND
#      IS MEASURED WITH A ZERO-PARAMETER CONTROL. The rule reads `knee_width / ankle_width` and
#      fires below 0.82. In a SQUAT -- feet about shoulder width, shanks near vertical -- that
#      ratio is near 1.0 when the knees track the feet, which is what makes 0.82 meaningful there.
#      In a WIDE SIDE-STRADDLE the legs splay from a pelvis that does not widen, so a knee sits
#      partway along the hip->ankle line and its separation is NECESSARILY smaller than the
#      ankles' -- with no valgus whatsoever.
#      Measured on 2353 open-phase frames of the 11 judged actions, replacing both knees with
#      their projections onto the same-side hip->ankle line (a PERFECTLY straight limb, zero
#      valgus by construction):
#
#          observed knee/ankle    median 0.769,  below the 0.82 cut on 79.4% of frames
#          ALIGNED  knee/ankle    median 0.810,  below the 0.82 cut on 68.5% of frames
#
#      So of the 79.4 points of firing, 68.5 are STANCE GEOMETRY and about 11 are any inward
#      deviation at all -- on a population every one of whose actions a human judged correct.
#      THE RULE READS THE MOVEMENT, NOT THE FAULT. Harness:
#      `src/egoexo/jumping_jacks_validation.py::aligned_knee_ratio`; this is the same shape as the
#      zero-parameter controls that refuted the keypoint blind-spot claims elsewhere in this
#      project -- run the control before believing the cue.
#
#      WHAT WOULD WORK, RECORDED AND NOT BUILT: the deviation of each knee from its own
#      hip->ankle line, normalized by limb length -- a per-side quantity that is zero for a
#      straight limb at ANY stance width. It is not built here because no source states a
#      threshold for it and inventing one is what this programme forbids; the parent spec's rule
#      is withdrawn rather than replaced by an uncited construction.
#
#   2. NO KG NODE SCOPED TO THIS MOVEMENT. "Knee Valgus" under movement="Jumping Jacks" matches
#      TWO SHARED nodes carried by the Squat flagship (`Knee Valgus Load`, `Knee Valgus Control`).
#      Both are non-empty and both are about valgus, so this is the weakest of the three failures
#      -- it is recorded for completeness, not as the deciding one.
#
#   3. THE CITATION MEASURES A DIFFERENT AND MUCH HARDER TASK, AND SUPPLIES NO RATIO. Tamura et al.
#      (2017), PMC5478135, re-fetched: a SINGLE-LEG drop vertical jump from a 40 cm box, with
#      valgus classified from 3-D knee abduction angle at peak vertical GRF by 8-camera motion
#      capture (valgus group 4.4 deg, varus -5.3 deg). Knee angular impulse 0.093 vs 0.045
#      Nms/kg.m and hip 0.019 vs 0.067, both p<0.01. NO KNEE-TO-ANKLE WIDTH RATIO APPEARS IN THE
#      PAPER, so the 0.82 came from this codebase's squat rule, not from the citation the parent
#      spec attaches -- and failure 1 is why that transfer does not hold.
#
#   NOT SAID BY THIS WITHDRAWAL: that knees collapsing inward on landing is fine. Tamura is a real
#   result about a real mechanism. What is missing is a quantity that measures it in a movement
#   whose stance is wide by definition. Design spec section 7.1.
#
# `jj_stiff_landing` -- WITHDRAWN, three failures, and the first is self-contained:
#
#   1. THE CITED PAPER'S OWN STIFF CONDITION WOULD NOT FIRE THIS RULE. DeVita & Skelly (1992),
#      PMID 1548984, re-fetched: subjects landed from a 59 cm vertical fall, and the soft and
#      stiff conditions "averaged 117 and 77 degrees of knee flexion"; "the stiff landing had
#      larger GRFs". The parent spec's heuristic flags a landing whose knee angle stays above
#      ~160 deg, i.e. FEWER THAN 20 DEGREES OF BEND. The paper's stiff landing is 77 degrees of
#      bend -- a knee angle of about 103 deg -- nowhere near the cut, and the number 160 appears
#      in the paper nowhere. The spec also claims "the >=/<90 deg knee-flexion soft/stiff
#      convention originates here"; the abstract states 117 and 77 and states no convention.
#
#   2. THE CUE IS MEASURED VIEW-CORRUPTED, AND THE PROJECTION BIAS RUNS TOWARD THE FIRING
#      DIRECTION -- WITH A BOUND THAT IS STATED RATHER THAN GLOSSED. On Fit3D (4 calibrated
#      cameras, mocap 3-D truth, 160 paired readings) the projected 2-D knee angle carries MAE
#      42.4 deg with a SYSTEMATIC +41.2 deg bias and noise/sig 1.21 -- which camera you used
#      matters more than what the athlete did. At one verified squat bottom the true knee angle is
#      78 deg and the four cameras report 108 / 118 / 119 / 133.
#      THE DIRECTION IS GEOMETRIC AND TRANSFERS: thigh and shank straddle the limb's long axis and
#      an oblique camera compresses the fore-aft component of both, so a projected knee angle errs
#      TOWARD 180 deg, which is the direction this rule fires in.
#      THE MAGNITUDE DOES NOT, AND THE BOUND MATTERS. The bias cannot exceed 180 - theta_true, so
#      it shrinks to nothing at full extension: a genuinely stiff landing at 170 deg cannot be made
#      to look 41 deg stiffer. What it CAN do is open a moderately absorbed landing -- 140 deg,
#      i.e. 40 deg of flexion -- past the 160 deg cut, and 40 deg of room is exactly the size of
#      the measured bias. So the corruption bites PRECISELY IN THE BAND THE RULE MUST DISCRIMINATE
#      IN, and nowhere else. `notes/fit3d_view_dependence_summary.md`.
#
#   3. NO KG NODE. "Stiff Landing" and "Landing" both return zero matches under
#      movement="Jumping Jacks".
#
#   NOT SAID BY THIS WITHDRAWAL: that landing stiff-legged is fine. What is missing is a threshold
#   that survives being read from a monocular camera. Design spec section 7.2.
#
# `jj_landing_asymmetry` -- WITHDRAWN, three failures:
#
#   1. NO KG NODE. "Asymmetry" under movement="Jumping Jacks" reaches only the shared `Symmetry`
#      quality dimension carried by the Squat flagship.
#
#      THIS IS THE FIRST ASYMMETRY RULE THIS PROGRAMME HAS WITHDRAWN, and the reason is NOT the
#      missing number: `ohp_asymmetric_press`, `arm_abd_lr_asymmetry` and
#      `arm_vw.rule_lr_asymmetry` all ship on spec-authored thresholds their citations do not
#      state, and that precedent is not being reversed.
#
#   2. IT IS A DISJUNCTION OF THREE UNRELATED QUANTITIES. The heuristic compares "wrist peak
#      height, ankle lateral excursion from hip-midline, and per-side knee-valgus ratio" and fires
#      if any differs by 15-20%. One `fault_id` whose evidence might be an arm, a foot or a knee
#      cannot produce a coherent explanation card, and `fault_id` is the join key between the
#      spec, the registry and every stored analysis. Arm VW kept its id through the loss of ONE
#      branch; keeping this one would mean choosing which of three faults it is. And one of the
#      three -- the per-side valgus ratio -- is the quantity withdrawal 1 above just refuted.
#
#   3. "CONSISTENTLY ACROSS REPS" IS CROSS-REP STATE THIS ARCHITECTURE DOES NOT HAVE.
#      `run_detector` scores one repetition at a time and `merge_by_fault` reports the rep count
#      afterwards; `arm_vw` recorded the same limit for the same spec wording.
#
#   NOT SAID BY THIS WITHDRAWAL: that landing harder on one leg than the other is fine. Design
#   spec section 7.3.
#
# `JUMPING_JACKS_METRIC_KEYS` must stay a two-way match with what `jumping_jacks_compute_raw`
# emits (pinned by `test_metric_keys_match_the_emitted_metrics_exactly`): a key the tuple omits is
# dropped by `run_detector`, which builds each CoreFrame's metrics dict FROM this tuple, and read
# back as NaN by every rule.
JUMPING_JACKS_DETECTOR = MovementDetector(
    "Jumping Jacks",
    JUMPING_JACKS_METRIC_KEYS,
    jumping_jacks_compute_raw,
    jumping_jacks_assign_phases,
    (
        rule_incomplete_leg_rom,
        rule_incomplete_arm_rom,
    ),
    # `validated` stays at its default False, and the reason is a NEW one -- module header.
    rep_signal="stance_width_ratio",
    # `max`: the effort peak is the WIDEST stance. No rectification -- the signal is unipolar,
    # because the feet never cross, unlike Torso Twist's bipolar swing.
    rep_polarity="max",
    # `extended` -- the repetition opens away from the effort peak, standing with the feet
    # together. Only Deadlift uses `flexed`.
    rep_start="extended",
    # `min_rep_seconds` STAYS AT THE DEFAULT 0.4 s, AND THAT CONTRADICTS `base.py:55`, WHICH NAMES
    # THIS MOVEMENT AS ONE THAT "MUST LOWER IT".
    #
    # The RS-SP1 audit put jumping jacks at "~1-2 Hz" and prescribed the knob on that basis. At
    # 2 Hz a repetition lasts 0.5 s, which CLEARS the 0.4 s floor; the floor would only bite above
    # 2.5 Hz. The fastest cadence with a citable number in this project's sources is the RAG doc's
    # Guinness record -- "the most jumping jacks performed in one minute is 136", i.e. 2.27 Hz or
    # 0.44 s per repetition -- still above the floor.
    #
    # AND IT IS MEASURED RATHER THAN ARGUED. Re-segmenting all 31 (action, camera) pairs of the
    # recovered EgoExo footage at a 0.15 s floor finds EXACTLY THE SAME 255 REPETITIONS as the
    # shipped 0.4 s floor -- nothing discarded -- at a median observed cadence of 0.93 Hz and a
    # fastest of 1.14 Hz (0.88 s per repetition), half the floor's reach. So the knob the
    # framework reserved for this movement by name is not needed by it. The framework comment is
    # left alone, because it also names High Knee -- the ~3 Hz movement it was really written for,
    # and the sixteenth detector's problem. Design spec section 4.5.
)

# ---------------------------------------------------------------------------------------
# THE DETECTOR IS DELIBERATELY NOT REGISTERED, AND THIS IS THE FIRST TIME IN THE PROGRAMME.
# ---------------------------------------------------------------------------------------
# There is no `registry.register(JUMPING_JACKS_DETECTOR)` call here, and its absence is the
# considered outcome rather than an oversight.
#
# Registration is what makes a movement ANALYZABLE in the web app: `registry.list_detectors()`
# backs GET /api/movements, and `analyze_pose_payload` routes to a detector when one exists and
# returns `analysis_pending` ("coming soon") when one does not. With every rule silent or
# withdrawn, registering would offer users an analysis that CANNOT EVER REPORT A FAULT while
# wearing the Beta tag that says faults are possible. "Coming soon" is the truthful state of this
# movement, so that is what the app says.
#
# WHAT WORKS AND IS KEPT, because none of it is what failed:
#   - the metric layer (roll-, mirror- and scale-invariant; obliquity-cancelling by construction),
#   - the phase assignment and the `open` landing-window substitution,
#   - the repetition segmentation, measured on real footage of this exercise: median validity
#     1.00 over 31 (action, camera) pairs, not one on the whole-clip fallback, 255 repetitions
#     found, and nothing lost to the duration floor.
# All of it is exercised by `tests/test_jumping_jacks.py` and by the validation harness, so
# whoever obtains the missing `.ac` archive part -- or any corpus with judged-FAULTY jumping jacks
# -- can read a threshold off human judgement, wake `rule_incomplete_leg_rom`, add one line here
# and ship. That is the concrete upgrade path this file exists to preserve.
