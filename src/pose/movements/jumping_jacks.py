# Jumping Jacks (side-straddle hop) raw metrics, phase segmentation and fault rules.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `jumping_jacks_compute_raw` /
# `jumping_jacks_assign_phases` compute per-frame quantities and a phase label only. Every number
# that decides anything belongs in a `rule_*` function.
#
# ---------------------------------------------------------------------------------------
# ONE RULE IS LIVE (SINCE 2026-09-26), ONE IS PERMANENTLY SILENT, THREE ARE WITHDRAWN, AND THE
# DETECTOR IS STILL NOT REGISTERED. THE LABELED DATA DECIDED BOTH HALVES OF THAT.
# ---------------------------------------------------------------------------------------
#   rule_incomplete_leg_rom       LIVE -- the fault is real and HUMAN-JUDGED (the most-failed of
#                                 EgoExo-Fitness's eight criteria, 9.9% of 121 actions), the
#                                 metric is the movement's own definition of a repetition, and on
#                                 the full archive it ranks the 12 flagged actions at AUC 0.730
#                                 [0.591, 0.879]; the cut read off the labels is 1.321, the spec's
#                                 1.3. It was silent while only 11 actions were reachable (79% of
#                                 their reps fired -- a sampling artefact; 37.1% corpus-wide). It
#                                 still carries a card on 51.4% of correctly judged clips; see its
#                                 docstring for the costs the user accepted.
#   rule_incomplete_arm_rom       PERMANENTLY SILENT -- needs NO threshold at all (its criterion
#                                 is a landmark comparison, not a number) and its metric is
#                                 clean; silent because the only source that states its target
#                                 attributes an injury association, in the same document, to the
#                                 range of motion the rule would coach users toward
#   jj_knee_valgus_landing        WITHDRAWN, absent -- THE METRIC IS CONFOUNDED BY THE VERY
#                                 STANCE THIS MOVEMENT IS DEFINED BY, measured with a
#                                 zero-parameter control: of the 79.4% of open-phase frames the
#                                 0.82 cut fires on, 63.2 points ALSO fire with both knees
#                                 replaced by PERFECTLY STRAIGHT-LIMB positions -- four firings in
#                                 five need no valgus at all. It reads stance, not alignment.
#   jj_stiff_landing              WITHDRAWN, absent -- the cited paper's OWN STIFF CONDITION
#                                 WOULD NOT FIRE THIS RULE (77 deg of flexion, against a cut
#                                 below 20 deg), and the cue carries a measured projection bias
#                                 pointing in the firing direction
#   jj_landing_asymmetry          WITHDRAWN, absent -- no KG node, a three-quantity disjunction
#                                 that would put an arm, a foot or a knee behind one fault_id,
#                                 and cross-rep state this architecture has never had
#
# THIS IS THE FIRST DETECTOR IN THE PROGRAMME THAT IS NOT REGISTERED. See the block above
# `JUMPING_JACKS_DETECTOR` for why, and for what registering it now would mean.
#
# Design spec `docs/superpowers/specs/2026-08-10-jumping-jacks-detector-design.md`. Measurements:
# `notes/jumping-jacks-rule-validation.md` (11 actions, 2026-08) and
# `notes/egoexo-silent-rules-full-archive.md` (all 121, 2026-09-26), harness
# `src/egoexo/jumping_jacks_validation.py`.
# Re-search 2026-09-25: all three stay withdrawn; every jumping-jack valgus/landing hit is a warm-up
# before a drop jump, and no cadence source states a fault. docs/superpowers/specs/2026-09-25-withdrawn-rules-literature-research.md section 5.
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
# (`Jumping Jacks:Incomplete Foot Split`) is the one the live ROM rule seeds from.
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
    severity_from_range,
)
from src.pose.movements.base import CoreFrame, MovementDetector, RuleContext
from src.pose.pose_rule_detector import PoseRuleDetection, build_detection

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
    # How far apart the feet are, in shoulder widths. THE REP SIGNAL, and the quantity the live
    # leg-ROM rule reads.
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
# IT WAS KEPT AT 1.3 WHILE THE RULE WAS SILENT, AND IT SHIPS AT 1.3 UNMOVED. The silence rested on
# 11 reachable actions whose widest stance had a median of about 1.16, so any cut near that
# distribution could have been manufactured. On the full archive (2026-09-26) the cut that best
# separates the 12 human-flagged actions from the 109 judged correct, READ OFF THE LABELS rather
# than authored, is 1.321 -- the spec's own number to within 0.02. So the constant never moved:
# the labels landed on it. `notes/egoexo-silent-rules-full-archive.md`.
LEG_ROM_MILD_RATIO = 1.3

# SEVERITY RAMP 1.3 -> 1.0: A RULE-LEVEL CHOICE, NOT A SOURCED NUMBER. 1.0 is the ankles exactly
# one shoulder width apart -- a zero-parameter anatomical anchor, not fitted to anything. It shapes
# how strongly a firing rep is worded; it does not decide whether the rule fires.
LEG_ROM_SEVERE_RATIO = 1.0


def rule_incomplete_leg_rom(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag a repetition whose feet never spread past 1.3 shoulder widths -- a narrow, shuffled jack.

    LIVE SINCE 2026-09-26, BY THE USER'S DECISION ON A PRE-REGISTERED PROPOSAL. It was permanently
    silent because its number looked wrong on 11 actions; the full EgoExo-Fitness archive settled
    the question the silence had left open. The analysis, fixed before the archive was decoded, is
    `docs/superpowers/specs/2026-09-26-egoexo-silent-rule-separability-prereg.md`; the result is
    `notes/egoexo-silent-rules-full-archive.md`.

    WHAT IT HAS:
      A PRIMARY SENTENCE NAMING THIS EXERCISE. The RAG doc defines the movement as "jumping to a
      position with the legs spread wide ... and then return to a position with the feet
      together".
      A KNOWLEDGE-GRAPH NODE GROUNDED IN THIS EXERCISE. `Jumping Jacks:Incomplete Foot Split`,
      and the seeding script's own grounding figure ("foot split 10%") reproduces from the labels
      as 9.9% (12/121) -- the only one of this movement's three nodes not contaminated by the
      Clap Jacks blend (module header). The node is DANGLING (only `related_actions`), so the card
      it seeds is thin.
      A HUMAN-JUDGED POSITIVE CLASS THAT THE METRIC RANKS. Per action, the median over cameras of
      the median over reps of each rep's widest `stance_width_ratio` ranks the 12 actions judged
      FALSE on "Perform the jump by opening and closing your feet" below the 109 judged TRUE at
      AUC 0.730, 95% CI [0.591, 0.879] by participant-clustered bootstrap. It holds with the
      front-back scissor jumps dropped (0.723 [0.526, 0.905]) and within single-annotator actions
      (0.752 [0.579, 0.917]).
      A CLEAN METRIC. `stance_width_ratio` is a ratio of two frontal-plane distances: roll-,
      mirror- and scale-invariant, and first-order invariant to azimuthal obliquity because both
      terms foreshorten together (`jumping_jacks_compute_raw`).

    WHAT IT COSTS -- KNOWN WHEN THE USER DECIDED:
      IT FIRES OFTEN ON CORRECT JACKS. Per rep, 1.3 fires on 37.1% of 941 reps in actions judged
      correct (71.3% of 101 in flagged ones). Per (action, camera) clip, which is what a user sees
      because `merge_by_fault` shows one card if any rep fires: 164 of 319 correct clips (51.4%)
      against 28 of 35 flagged (80.0%). Held-out-by-participant specificity at the action level is
      0.633.
      THE LABEL IS BROADER THAN THE FAULT. Of the 12 flagged actions only 2 comments describe
      insufficient width; 3 are front-back scissor jumps, which this frontal width ratio also
      reads as narrow, and the width + direction-free subset (5 actions) is undetermined. On a
      scissor jump the card's advice is half-right at best.

    FOUND AFTER THE DECISION (exploratory, 2026-09-26) -- NO SINGLE CAMERA IS ESTABLISHED ON ITS
    OWN. The pooled result uses three simultaneous cameras; production sees ONE. Per camera:
    `exo_l` 0.803 [0.678, 0.936], frontal `exo_m` (the view the parent spec names) 0.654
    [0.441, 0.853], `exo_r` 0.647 [0.416, 0.836]. All three point estimates are above 0.5 and
    every interval is wide. `exo_l` and `exo_r` are the two mirror-image side cameras, so their
    gap at 11-12 positives is not evidence of a view effect.
    The earlier "79% of correct reps" figure was a sampling artefact of the 11 reachable actions:
    the same 11 give 77.3% under the current pipeline, the whole corpus 37.1%.

    SCOPE IS THE WHOLE REPETITION, NOT THE `open` PHASE -- A DELIBERATE DEPARTURE FROM THE SCOPE
    THIS DOCSTRING USED TO RECORD. `open` is `stance_width_ratio >= the CLIP's 70th percentile`
    (`jumping_jacks_assign_phases`), so in a clip mixing wide and narrow reps a narrow rep may hold
    NO `open` frame at all, and an open-scoped rule would be silent on exactly the reps it exists
    to flag. The maximum over the whole rep window equals the maximum over its `open` frames
    whenever it has any, and it is the quantity the validation measured
    (`src/egoexo/jumping_jacks_validation.py::evaluate_view`, `per_rep_widest`). `min_frames` is
    tested against the whole rep's valid frames, which also sidesteps the Bicep Curl
    phase-fraction trap `PhaseFractionTest` records.

    NO VIEW GATE, because no camera-specific effect is established (above) and the view
    estimator is unreliable on this footage. Observability is `medium`, not the spec's `high`,
    because no single camera's separation is established.

    INHERITED, NOT INTRODUCED HERE: like every whole-rep "not enough travel" rule in the registry,
    a motionless clip can segment into reps and fire this at full severity
    (`situp.rule_incomplete_rom` documents the mechanism). The framework-level repairs recorded
    there apply unchanged.
    """
    segment = [
        frame for frame in core if frame.valid and np.isfinite(frame.m("stance_width_ratio"))
    ]
    if len(segment) < ctx.min_frames:
        return []

    values = [frame.m("stance_width_ratio") for frame in segment]
    widest = float(np.nanmax(values))
    if not widest < LEG_ROM_MILD_RATIO:
        return []

    severity = severity_from_range(
        widest, LEG_ROM_MILD_RATIO, LEG_ROM_SEVERE_RATIO, lower_is_worse=True
    )
    return [
        build_detection(
            fault_id="jj_incomplete_leg_rom",
            fault_name="Incomplete Leg Spread (Narrow Stance)",
            kg_query=JUMPING_JACKS_LEG_ROM_KG_QUERY,
            retrieval_mode="kg",
            segment_metrics=segment,
            # The peak is the WIDEST frame: the moment the stance got as wide as it would, still
            # short of the cut. Unclipped, per the registry's `keyEvidence` convention.
            score_values=values,
            severity=severity,
            confidence=severity,
            observability="medium",
            evidence={
                "widest_stance_width_ratio": round(widest, 3),
                "threshold_ratio": LEG_ROM_MILD_RATIO,
                "primary_label": "widest foot spread (shoulder widths)",
                "primary_value": round(widest, 3),
                "primary_threshold": LEG_ROM_MILD_RATIO,
            },
            citation=(
                "Wikipedia, \"Jumping jack\" (CC BY-SA), data/rag/docs/jumping_jacks_wiki.txt; "
                "threshold checked against human judgement on EgoExo-Fitness "
                "(notes/egoexo-silent-rules-full-archive.md)."
            ),
            citation_support=(
                "The source defines the movement as \"jumping to a position with the legs spread "
                "wide\" and back \"with the feet together\" -- descriptive support for the target, "
                "no tolerance. The 1.3 shoulder-width cut is the parent spec's; on 121 "
                "human-judged EgoExo-Fitness actions the cut that best separates the 12 judged "
                "FALSE on \"Perform the jump by opening and closing your feet\" is 1.321, "
                "action-level AUC 0.730 [0.591, 0.879]. It still fires on 51.4% of correctly "
                "judged clips, so one card is a prompt to check, not a verdict."
            ),
        )
    ]


def rule_incomplete_arm_rom(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """PERMANENTLY SILENT -- always returns [].

    AND THE REASON IS NOT THE USUAL ONE. Every other silent rule in this registry
    (`abd_insufficient_rom`, `tt_insufficient_rotation_rom`, `bridge_lumbar_hyperextension`, and
    `rule_incomplete_leg_rom` above until the full archive woke it on 2026-09-26) is or was silent
    because a NUMBER or a SENSOR is missing. Neither is missing here:

      THE CRITERION NEEDS NO NUMBER. The parent spec's own test is "both wrists fail to rise above
      the nose" -- a comparison between two body landmarks. `hands_above_head_ratio` is that
      comparison projected onto the trunk axis, so it fires at zero. There is no threshold to
      lack, which makes this the only rule in the section immune to the failure that once
      silenced the leg rule.

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
#      AND THE JOINT COUNTS, BECAUSE TWO MARGINAL RATES ARE NOT A DECOMPOSITION -- the aligned
#      firings are NOT a clean subset of the observed ones, which had to be counted rather than
#      inferred from the two medians:
#
#          fires with a PERFECTLY STRAIGHT LIMB too   63.2% of frames   <- stance alone
#          fires only with the REAL knees             16.2%             <- needed real deviation
#          would fire straight-limb but does NOT       5.2%             <- knees BOWED OUT
#
#      So FOUR FIRINGS IN FIVE (63.2 of 79.4) need no inward deviation whatsoever, on a population
#      every one of whose actions a human judged correct. THE RULE READS THE MOVEMENT, NOT THE
#      FAULT. (The 16.2% is not nothing and is not claimed to be: on about one open-phase frame in
#      six there IS measurable inward deviation. That is a reason to want a metric that isolates
#      it -- see below -- not a reason to keep one that cannot.) Harness:
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
# returns `analysis_pending` ("coming soon") when one does not. While every rule was silent or
# withdrawn, registering would have offered users an analysis that CANNOT EVER REPORT A FAULT
# while wearing the Beta tag that says faults are possible.
#
# SINCE 2026-09-26 ONE RULE IS LIVE, AND REGISTRATION IS STILL NOT DONE HERE -- IT IS A SEPARATE
# DECISION. The pre-registered plan that licensed waking `rule_incomplete_leg_rom` fixed that the
# movement stays unregistered either way: registering would ship a movement whose ONLY live rule
# shows a card on 51.4% of correctly judged clips (the rule's docstring). Registering is one line
# below plus the test updates in `tests/test_jumping_jacks.py::NotRegisteredTest` and
# `test_unknown_movement_returns_coming_soon_without_detector`, which needs a listed-but-
# unregistered movement.
#
# WHAT WORKS AND IS KEPT, because none of it is what failed:
#   - the metric layer (roll-, mirror- and scale-invariant; obliquity-cancelling by construction),
#   - the phase assignment and the `open` landing-window substitution,
#   - the repetition segmentation, measured on real footage of this exercise: on all 121 judged
#     actions (356 action x camera pairs), median validity 1.00, 2 pairs on the whole-clip
#     fallback, 2701 repetitions found and 27 lost to the duration floor.
# All of it is exercised by `tests/test_jumping_jacks.py` and by the validation harness. The
# upgrade path this block used to describe -- obtain the full archive, read a threshold off human
# judgement, wake `rule_incomplete_leg_rom` -- was taken on 2026-09-26; the threshold read off the
# labels was the spec's own 1.3.
