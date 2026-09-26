# High Knee (running-in-place / marching drill) raw metrics, phase segmentation and fault rules.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `high_knee_compute_raw` / `high_knee_assign_phases`
# compute per-frame quantities and a phase label only. Every number that decides anything belongs
# in a `rule_*` function.
#
# ---------------------------------------------------------------------------------------
# ONE RULE IS LIVE, FOUR ARE WITHDRAWN, AND THE DETECTOR IS REGISTERED -- BOTH SINCE 2026-09-26,
# BY THE USER'S DECISION. THIS IS THE SIXTEENTH AND LAST MOVEMENT.
# ---------------------------------------------------------------------------------------
#   rule_insufficient_knee_lift   LIVE at 65.6 deg of hip flexion, a cut READ OFF DATA by the
#                                 user's decision -- neither of the spec's two numbers (it CITES
#                                 the A-skip's 45 deg and IMPLEMENTS the B-skip's 90 deg; on the
#                                 full archive 90 deg cards 89% of clips without a complaint and
#                                 45 deg misses 59% of those with one). The metric ranks the
#                                 comment labels at held-out AUC 0.730 [0.586, 0.864]; the cut
#                                 still cards 40% of side-camera clips without a complaint. See
#                                 its docstring.
#   hk_trunk_lean_back            WITHDRAWN, absent -- ITS REFERENCE AXIS IS AS BIG AS THE FAULT.
#                                 A trunk lean is an angle from the VERTICAL, and this drill has
#                                 no vertical: the support limb, which Leg Abduction established
#                                 as Group E's substitute, sits 8.6-23.6 deg off the trunk in
#                                 normal marching, against a rule threshold of 10-15 deg. The
#                                 spec's 10 deg cut fires on a median 69.7% of scored frames, and
#                                 on 56-83% of the two actions judged CORRECT on every criterion.
#   hk_forward_trunk_collapse     WITHDRAWN, absent -- the same scalar with the opposite sign, so
#                                 the same axis failure; and measured, it fires on 0.0% of frames,
#                                 because the axis error runs entirely into the other rule.
#   hk_contralateral_pelvic_drop  WITHDRAWN, absent -- refuted by three SIMULTANEOUS cameras:
#                                 filming the same instant they disagree by 0.97-13.68 deg
#                                 (median 6.7) against a 5-8 deg threshold, and frame by frame the
#                                 two side cameras are ANTI-correlated on four of six actions
#                                 (r = -0.48 to +0.12).
#   hk_stride_asymmetry           WITHDRAWN, absent -- no scoped KG node, a disjunction of two
#                                 quantities behind one fault_id (one of which is the quantity the
#                                 withdrawal above just refuted), and "consistently across reps"
#                                 is cross-rep state this architecture does not have.
#
# IT WAS THE SECOND DETECTOR LEFT UNREGISTERED, after Jumping Jacks, until 2026-09-26. See the
# block below `HIGH_KNEE_DETECTOR`.
#
# Design spec `docs/superpowers/specs/2026-08-10-high-knee-detector-design.md`. Measurements:
# `notes/high-knee-rule-validation.md` (6 actions, 2026-08) and
# `notes/egoexo-silent-rules-full-archive.md` (all 68, 2026-09-26), harness
# `src/egoexo/high_knee_validation.py`.
# Re-search 2026-09-25: all four withdrawals stand (three are measured refutations). Alberton 2015
# (PMC4723158) instructs 90 deg hip/knee flexion for running in place -- a protocol TARGET with no
# tolerance. It is now the live knee-lift rule's citation for the target; the cut is data-derived.
# docs/superpowers/specs/2026-09-25-withdrawn-rules-literature-research.md section 5.
#
# ---------------------------------------------------------------------------------------
# THE CORPUS JUDGES THIS EXERCISE RICHLY AND JUDGES DIFFERENT FAULTS. THAT IS JUMPING JACKS'
# REASON, AND IT IS SHARPER HERE.
# ---------------------------------------------------------------------------------------
# EgoExo-Fitness carries 68 judged `High Knee` actions -- the right exercise, the right variant,
# 120 annotations by 1-3 annotators each. Its seven criteria, with the fraction of the 68 actions
# on which a strict majority of annotators judged each FAILED:
#
#     Aim to maintain the fastest speed possible while performing the leg lifts.   44.1% (30/68)
#     Swing your arms in rhythm with the leg lifts.                                26.5% (18/68)
#     Maintain a stable upper body throughout the exercise.                        14.7% (10/68)
#     Lift your legs alternately and quickly.                                      10.3%  (7/68)
#     Look straight ahead.                                                          1.5%  (1/68)
#     Keep your back straight.                                                      0.0%  (0/68)
#     Keep the balls of your feet in contact with the ground during the lifts.      0.0%  (0/68)
#
# THE PARENT SPEC'S FIVE RULES ARE KNEE-LIFT HEIGHT, BACKWARD TRUNK LEAN, FORWARD TRUNK COLLAPSE,
# CONTRALATERAL PELVIC DROP AND STRIDE ASYMMETRY. NOT ONE OF THE SEVEN CRITERIA JUDGES ANY OF
# THEM. The corpus's two largest faults -- cadence and arm rhythm -- are unmodelled by the spec,
# and the spec's whole roster is unjudged by the corpus. Jumping Jacks had ONE overlapping pair;
# this has ZERO.
#
# THE NEAR MISSES ARE NEAR MISSES, AND SAYING SO IS THE POINT:
#   "Maintain a stable upper body" is the criterion with a real positive class, and the annotator
#   comments say what it means -- "upper body lacks stability", "excessive upper body sway",
#   "slightly shaking", "unstable center of gravity". That is VARIANCE. Both trunk rules read a
#   signed MEAN offset. Measured on the six reachable actions, the TWO judged FALSE on this
#   criterion and the FOUR judged TRUE separate on NEITHER the mean nor the variance (design
#   spec section 6.3).
#   "Keep your back straight" is the criterion the two trunk rules would model, and NO action in
#   the corpus fails it by majority.
#
# `validated` stays False, and the reason is Jumping Jacks' sixth reason rather than a seventh.
#
# ---------------------------------------------------------------------------------------
# THE KNOWLEDGE GRAPH'S FOUR HIGH KNEE FAULTS OVERLAP THE SPEC IN ONE NODE, AND THAT NODE'S
# STATED GROUNDING IS BORROWED FROM A CRITERION ABOUT SOMETHING ELSE.
# ---------------------------------------------------------------------------------------
# `scripts/knowledge/stub_general_movements_v3.py:142-151` records this movement's provenance as
#
#     "grounding": "EgoExo-Fitness TKV (top-failed: cadence/speed 44%, arm rhythm 26%,
#                   upper-body stability 15%, knee lift 10%)"
#
# The first three figures reproduce exactly from the labels (44.1%, 26.5%, 14.7% above). SO DOES
# THE FOURTH -- 10.3% -- AND THAT IS THE PROBLEM: it is the failure rate of "Lift your legs
# alternately and quickly", a criterion about ALTERNATION AND SPEED. There is no knee-height
# criterion in the checklist at all. So `High Knee:Insufficient Knee Lift` is a plausible node
# whose only stated evidence measures a different fault.
#
# Torso Twist found a node that faithfully describes a DIFFERENT MOVEMENT; Jumping Jacks found one
# seeded from a BLEND of two. This is the third variety: a node seeded from THE WRONG CRITERION OF
# THE RIGHT MOVEMENT. In all three the node is present, resolvable, and not evidence.
#
# ---------------------------------------------------------------------------------------
# EVERY QUANTITY HERE IS ROLL-INVARIANT, AND ON THIS CORPUS THAT IS NOT A LUXURY.
# ---------------------------------------------------------------------------------------
# EgoExo's two SIDE cameras ship their frames ROLLED 90 DEGREES with no EXIF -- a standing subject
# lies horizontally across a 456x256 landscape frame. Sit-up found this for a supine movement and
# it was tempting to read it as a quirk of filming someone on the floor; it is not, it is how
# these cameras ship. Every rule the parent spec writes for this movement is phrased in IMAGE Y
# ("y_knee - y_hip", "shoulder-midpoint x moves behind the hip-midpoint x") and every one of them
# is meaningless on those frames. The metrics below are cosines and ratios between BODY vectors,
# so they are unaffected -- which is the only reason there are numbers to report at all.
#
# THE COROLLARY IS A CAVEAT, NOT A REASSURANCE: MediaPipe is not roll-equivariant (this project
# measured a median 9.8 deg landmark shift under rotation), so the side-camera landmarks are
# degraded even though the metrics computed from them are well defined.
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

# The generic "lower body" set every movement module uses for the framework-level
# `lower_body_visibility` quality field. This module's own rules never read it.
LOWER_BODY_LANDMARKS = (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

HIGH_KNEE_METRIC_KEYS: tuple[str, ...] = (
    # Per-side thigh elevation: the COSINE of the angle between the hip->knee vector and the
    # trunk-up axis, in the image plane. -1 is a thigh hanging straight down along the trunk,
    # 0 is a thigh perpendicular to the trunk (the knee at hip height), +1 a knee straight up.
    "thigh_elevation_left",
    "thigh_elevation_right",
    # left minus right. THE REP SIGNAL, rectified -- see `HIGH_KNEE_DETECTOR`.
    "thigh_elevation_difference",
    # |anterior axis| / shoulder width: how much of the subject's own fore-aft direction survives
    # projection into this image. The knee-lift rule's OWN test of whether the camera can see a
    # sagittal quantity -- not a `view_estimation.py` call, which this programme has now measured
    # inverted once (Sit-up) and outside its stated regime once (Leg Abduction).
    "anterior_axis_length",
    #
    # NOTE WHAT IS NOT HERE. `trunk_lean_forward` -- the signed fore-aft trunk angle both withdrawn
    # trunk rules would read -- is deliberately absent: a withdrawn rule leaves no metric behind
    # for something to quietly start reading. It is recomputed inside
    # `src/egoexo/high_knee_validation.py` so the evidence for the withdrawal stays re-runnable
    # without this module carrying a refuted quantity. Pelvic obliquity is absent for the same
    # reason.
)


def _unit(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    return None if norm <= 0.0 else vector / norm


def _thigh_elevation(
    points: np.ndarray, hip_index: int, knee_index: int, trunk_up: np.ndarray
) -> float:
    """cos(angle between the hip->knee vector and trunk-up), in the image plane.

    TRUNK-RELATIVE ON PURPOSE, AND THE SOURCE'S TARGET IS GROUND-RELATIVE. Matijasevic's criterion
    is "the thigh of the swinging leg reaches 45 degrees relative to the ground"; hip flexion, the
    thing the drill trains, is by anatomical definition an angle between the femur and the
    trunk/pelvis. The two differ by exactly the trunk's own lean -- which is what makes the
    trunk-relative form the RIGHT one for this rule: an athlete who throws the torso backward to
    hoist the knee gains ground-relative thigh angle and gains NO hip flexion, so a trunk-relative
    rule cannot be cheated by the very fault `hk_trunk_lean_back` describes. Stated as the
    deliberate substitution it is, not as an equivalence.

    Normalizing by the PROJECTED thigh length rather than by a body width makes this a cosine
    rather than a length ratio, bounding it to [-1, 1] on every frame, so no threshold on it can
    be dominated by a scale error.
    """
    hip = visible_point(points, hip_index, dims=2)
    knee = visible_point(points, knee_index, dims=2)
    thigh = _unit(knee - hip) if hip is not None and knee is not None else None
    if thigh is None:
        return math.nan
    return float(np.dot(thigh, trunk_up))


def _anterior_axis(points: np.ndarray, trunk_up: np.ndarray) -> np.ndarray | None:
    """The subject's ANTERIOR direction in the image, from their own feet.

    "In front" is not recoverable from image x: a camera on the subject's left and one on their
    right disagree about which way +x points, and nothing in a frame says which is looking. The
    subject's feet DO say it -- heel->toe points anterior -- so the axis comes from the body,
    mirror-invariantly. Leg Abduction's dot-product construction; a cross product would be
    anti-invariant under mirroring, which is why Shoulder Bridge's two attempts failed.

    Averaged over both feet because one foot is airborne at any instant in this drill and a
    plantar-flexed airborne foot points somewhere other than anterior. The `trunk_up` component is
    removed so what remains is the horizontal-anterior direction.

    ONLY ITS LENGTH IS USED, AND THAT DISTINCTION IS LOAD-BEARING. Projecting the trunk onto an
    axis built by removing the trunk direction returns zero on every frame by construction -- the
    first draft of this module did exactly that and reported a trunk lean of identically 0.000 for
    all 18 (action, camera) pairs. A length is still meaningful: it says how much fore-aft
    direction this camera can see at all.
    """
    vectors = []
    for heel_index, toe_index in ((LEFT_HEEL, LEFT_FOOT_INDEX), (RIGHT_HEEL, RIGHT_FOOT_INDEX)):
        heel = visible_point(points, heel_index, dims=2)
        toe = visible_point(points, toe_index, dims=2)
        if heel is not None and toe is not None:
            vectors.append(toe - heel)
    if not vectors:
        return None
    mean = np.mean(np.stack(vectors), axis=0)
    return mean - float(np.dot(mean, trunk_up)) * trunk_up


def high_knee_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
    raw: list[dict] = []

    for frame in frames:
        if not isinstance(frame, dict):
            raw.append({"valid": False})
            continue

        points = landmarks_to_array(frame.get("landmarks"))
        frame_index = int(frame.get("frame_index", 0) or 0)
        time = frame_index / fps if fps > 0 else 0.0
        # REQUIRE WHAT THE REP SIGNAL AND THE RULES NEED (the jumping_jacks principle): the
        # shoulders and hips define the trunk axis every metric is expressed in, and the knees are
        # the rep signal. The heels and toes are read for `_anterior_axis` and NOT required -- they
        # are the landmarks this drill occludes and motion-blurs most, and a missing foot must
        # cost the view gate, not the whole frame.
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

        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        shoulder_width = distance(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        trunk_up = _unit(shoulder_mid - hip_mid)

        left = right = difference = axis_length = math.nan
        if trunk_up is not None:
            left = _thigh_elevation(points, LEFT_HIP, LEFT_KNEE, trunk_up)
            right = _thigh_elevation(points, RIGHT_HIP, RIGHT_KNEE, trunk_up)
            if math.isfinite(left) and math.isfinite(right):
                difference = left - right

            anterior = _anterior_axis(points, trunk_up)
            if anterior is not None and math.isfinite(shoulder_width) and shoulder_width > 1e-8:
                axis_length = float(np.linalg.norm(anterior)) / shoulder_width

        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "thigh_elevation_left": left,
                "thigh_elevation_right": right,
                "thigh_elevation_difference": difference,
                "anterior_axis_length": axis_length,
            }
        )

    return raw


def high_knee_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> drive -> peak -> recovery, on |thigh_elevation_difference|.

    ONE REPETITION IS ONE KNEE DRIVE, not one left-right cycle, and the choice is forced rather
    than stylistic: the two legs alternate, so the signed difference swings symmetrically about
    zero and its MAGNITUDE peaks once per drive. Rectifying gives a rep per drive; not rectifying
    would give a rep per cycle, with one side's peak landing on the window BOUNDARY where the
    Bicep Curl trimming finding says it is least reliable.

    Which leg is driving is NOT encoded in the phase: it is the SIGN of the difference at the
    peak, which any rule can read off the metric tuple it is handed. Phases stay side-agnostic so
    a rule about the driving limb and one about the support limb can share them.

    Same fallbacks as every other module: an empty clip returns an empty list, a clip with no
    finite signal is entirely `unknown`, and an invalid frame is `unknown` wherever it sits (the
    validity check precedes the setup cutoff, so an occluded frame in the opening 15% is NOT
    labelled `setup`).
    """
    frame_count = len(raw)
    if frame_count == 0:
        return []

    magnitudes = np.asarray(
        [abs(float(item.get("thigh_elevation_difference", np.nan))) for item in raw],
        dtype=np.float32,
    )
    finite = magnitudes[np.isfinite(magnitudes)]
    if finite.size == 0:
        return ["unknown" for _ in raw]

    peak_threshold = float(np.percentile(finite, 70))
    peak_index = int(np.nanargmax(np.where(np.isfinite(magnitudes), magnitudes, -np.inf)))
    setup_cutoff = max(1, int(frame_count * 0.15))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append("setup")
            continue

        value = magnitudes[index]
        if np.isfinite(value) and value >= peak_threshold:
            phases.append("peak")
        elif index < peak_index:
            phases.append("drive")
        else:
            phases.append("recovery")
    return phases


# ---------------------------------------------------------------------------------------
# STEP 0 -- KG QUERY RESOLUTION, recorded before any rule was written. Each string below was
# checked against data/kg/sports_kg_v3.graphml with `retrieve_graph_context(query, movement=
# "High Knee")` -- the function PRODUCTION calls, not just `resolve_nodes`. Observed:
#
#   "Insufficient Knee Lift"    -> High Knee:Insufficient Knee Lift
#       only `related_actions`                                              DANGLING
#   "Knee Lift"                 -> High Knee:Insufficient Knee Lift          (same node)
#   "Unstable Upper Body"       -> High Knee:Unstable Upper Body
#       quality_impacts: Trunk Stability, Core Stability                    NON-EMPTY
#   "Slow Cadence"              -> High Knee:Slow Cadence
#       corrections: Maintain Even Tempo                                    NON-EMPTY
#   "Poor Arm-Leg Rhythm"       -> High Knee:Poor Arm-Leg Rhythm
#       causes: Poor Neuromuscular Control                                  NON-EMPTY
#   "Trunk Lean"                -> []                                       NO NODE
#   "Trunk Lean Compensation"   -> []                                       NO NODE
#   "Forward Trunk Lean"        -> []                                       NO NODE
#   "Lumbar Hyperextension"     -> []                                       NO NODE
#   "Pelvic Drop"               -> []                                       NO NODE
#   "Contralateral Pelvic Drop" -> []                                       NO NODE
#   "Pelvic Control"            -> Pelvic Control, shared QualityDimension  NO SCOPED NODE
#   "Asymmetry" / "Stride Asymmetry"
#                               -> Symmetry, shared, reached from Squat     NO SCOPED NODE
#
# THE NEGATIVE FILTER HOLDS FOR A FOURTH MOVEMENT, AND HERE IT IS PERFECT. The four rules with no
# scoped node (`hk_trunk_lean_back`, `hk_forward_trunk_collapse`, `hk_contralateral_pelvic_drop`,
# `hk_stride_asymmetry`) are EXACTLY the four withdrawn, and the one rule with a scoped node is
# exactly the one kept as silent. Leg Abduction section 7.3's finding, reproduced with no
# exceptions in either direction for the first time.
#
# The positive signal again predicts nothing on its own, which is the other half of that finding:
# the movement's one scoped node matching a spec rule is DANGLING, and the three nodes with real
# buckets correspond to no rule in the parent spec.
HIGH_KNEE_KNEE_LIFT_KG_QUERY = "Insufficient Knee Lift"


# FROM THE SPEC: "flag when the knee never rises to near hip height, e.g. (y_knee - y_hip) stays
# > +0.05". The knee at hip height is the thigh perpendicular to the trunk, i.e. a
# `thigh_elevation` of 0.0.
#
# AND FROM THE SPEC'S OWN CITATION, A DIFFERENT NUMBER. Matijasevic's Table 1 scores the A-skip on
# "the thigh of the swinging leg reaches 45 degrees relative to the ground" and Table 2 scores the
# B-skip at "90 degrees" -- a graded pair, easier drill then harder one. 90 degrees is the thigh
# parallel to the ground, which is the knee at hip height; 45 degrees is HALFWAY THERE, a
# `thigh_elevation` of -cos(45 deg) = -0.7071.
#
# NEITHER SHIPS, AND BOTH ARE KEPT: the validation harness replays them as the spec's two readings.
KNEE_LIFT_CITED_A_SKIP = -math.cos(math.radians(45.0))  # -0.7071, the number the spec CITES
KNEE_LIFT_IMPLEMENTED_B_SKIP = 0.0                      # the number the spec's heuristic USES

# THE SHIPPED CUT, AND ITS PROVENANCE IS DATA, NOT A PAPER -- THE USER'S DECISION, 2026-09-26.
# 65.6 deg of hip flexion is the cut that best separates the EgoExo-Fitness High Knee actions whose
# annotator comments complain about leg height from those whose comments do not (Youden's J on the
# 59 held-out actions, gated side cameras). Refit without each participant in turn it lands at
# 64.5-65.8 deg (23 of 25 folds at 65.6), so it does not hinge on one person; with 12 positives,
# read it as "about 65", not as a precise value. It sits between the spec's two sourced grades
# (45 and 90 deg), where real performers are: flagged actions peak at a median 50.6 deg, unflagged
# at 77.7. No paper states it. `notes/egoexo-silent-rules-full-archive.md`.
KNEE_LIFT_HIP_FLEXION_DEG = 65.6
KNEE_LIFT_THRESHOLD = -math.cos(math.radians(KNEE_LIFT_HIP_FLEXION_DEG))  # -0.4131

# SEVERITY RAMP 65.6 -> 45 deg: A RULE-LEVEL CHOICE. 45 deg is the A-skip grade, the EASIER of the
# spec source's two targets, so a thigh that falls short even of that reads at full severity. It
# shapes how strongly a firing rep is worded; it does not decide whether the rule fires.
KNEE_LIFT_SEVERE_HIP_FLEXION_DEG = 45.0
KNEE_LIFT_SEVERE_THRESHOLD = -math.cos(math.radians(KNEE_LIFT_SEVERE_HIP_FLEXION_DEG))

# THE VIEW GATE'S NUMBER, AND IT IS ALSO READ OFF DATA. A frontal camera cannot see thigh elevation
# (the thigh swings toward the lens), and on EgoExo's frontal `exo_m` the metric reads nonsense in
# both directions. `anterior_axis_length` is this module's own measure of how much fore-aft
# direction a camera sees. 0.170 is the LARGEST per-clip median it takes on the frontal camera
# over all 68 judged actions, so every frontal clip observed falls at or below it; 17 of the 130
# side-camera clips do too and are silenced with them. With this gate applied the held-out
# separation is AUC 0.754 [0.620, 0.869] (exploratory; the pre-registered primary gated by camera
# name). At n=6 the two cameras separated with no overlap (side 0.156-0.318, frontal 0.027-0.044);
# at 68 actions they overlap (side 0.066-0.441, frontal 0.019-0.170), which is why the gate sits at
# the frontal maximum rather than in a gap that no longer exists.
#
# CHECKED PER REPETITION, WHICH IS WHAT THE RULE GATES ON (the 0.170 is a per-clip maximum): of
# 177 analysed frontal reps on the 59 held-out actions, 3 pass the gate, and no frontal clip is
# carded. With the gate removed the rule would card 28 of 47 no-complaint and 11 of 12 complaint
# frontal clips -- so the gate, not the metric, is what keeps the frontal camera silent. On the side
# cameras it silences 3 clips the ungated analysis would card (2 no-complaint, 1 complaint).
KNEE_LIFT_VIEW_GATE = 0.170


def _hip_flexion_deg(thigh_elevation: float) -> float:
    """`thigh_elevation` is -cos(hip flexion): -1 hanging, 0 at hip height. Degrees for the card."""
    return math.degrees(math.acos(max(-1.0, min(1.0, -thigh_elevation))))


def rule_insufficient_knee_lift(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag a knee drive whose thigh never reaches 65.6 deg of hip flexion.

    LIVE SINCE 2026-09-26, AT A DATA-DERIVED CUT THE USER CHOSE OVER BOTH OF THE SPEC'S NUMBERS.
    It was permanently silent because the spec cites 45 deg and implements 90 deg, and on 6 actions
    the one fired on everything while the other appeared to sort the labels backwards. On the full
    EgoExo-Fitness archive the metric ranks the comment labels (pre-registered primary: held-out
    AUC 0.730 [0.586, 0.864]), and the user decided to ship the cut read off those labels. The
    pre-registered plan had fixed that a comment label cannot license a number; this is the user's
    explicit override of that, recorded as such.

    WHAT IT HAS:
      THE ONLY SCOPED KG NODE. `High Knee:Insufficient Knee Lift` -- DANGLING (only
      `related_actions`), so the card it seeds is thin, and its stated grounding figure measures
      a different criterion (module header).
      A TARGET FROM A SOURCE THAT STUDIES THIS EXERCISE. Alberton et al. 2015 instruct dry-land
      running in place as "the right hip and knee flexion to 90 deg starting the swing phase".
      That is the target the user falls short of. It is not the tolerance.
      A HUMAN SIGNAL THE METRIC RANKS. 15 of 68 actions carry a free-text leg-height complaint
      ("the leg raising range is too small, should be lifted higher") under a rule fixed before the
      comments were read; 12 of them are in the 59 held-out actions the primary used.
      A CLEAN METRIC. `thigh_elevation` is a cosine between two body vectors: roll-, mirror- and
      scale-invariant, which on a corpus whose side cameras are rolled 90 degrees is what makes it
      measurable at all. Trunk-relative on purpose (`_thigh_elevation`).

    WHAT IT COSTS, THIS RULE AS SHIPPED (gate included), THROUGH `run_detector`, SIDE CAMERAS OF
    THE 59 ACTIONS HELD OUT FROM BUILDING THE COMMENT RULE -- NOT FROM FITTING THE CUT:
      A card appears on 36 of 90 clips whose comments did not complain (40.0%) and on 19 of 22
      that did (86.4%). These rates are IN-SAMPLE: the cut was fitted on the same actions, which
      biases the false-card rate DOWN, while "no complaint" is not a judgement that the lift was
      high enough, which biases it UP. The two run in opposite directions, so neither is a bound.
      The out-of-sample figure is the cut refit without each participant: sensitivity 0.833
      (10/12), specificity 0.660 (31/47). Fold cuts of 64.5-65.8 deg suggest the fitting bias is
      small.
      THE LABEL IS SECONDARY: free text, 12 positives from 9 people, and any-annotator (a
      multi-annotated action gets more chances to be flagged; both strata point the same way).

    ALBERTON'S +/-5 DEG IS NOT A TOLERANCE, AND IT IS NOT USED. The paper kept for analysis only
    repetitions "within a range of +/-5 deg from the target angle (90 deg)", "considered to be an
    adequate amplitude of motion" -- a rep-SELECTION window for a laboratory protocol. Applied as a
    fault cut it would fire on the median EgoExo performer judged fine (77.7 deg).

    SCOPE: the whole repetition, one knee drive (`high_knee_assign_phases`), reading the MAXIMUM of
    the two thighs' elevation -- the driving leg is whichever is higher -- which is the quantity the
    validation measured (`src/egoexo/high_knee_validation.py::evaluate_view`). `min_frames` is
    tested against the whole repetition's usable frames.

    THE VIEW GATE: the median `anterior_axis_length` over the repetition must exceed
    `KNEE_LIFT_VIEW_GATE`, and a repetition with no finite gate reading (no heel or toe visible) is
    silent, because nothing then says the camera can see the lift. A frontal camera is silent BY
    DESIGN, not by accident: it cannot see this quantity. See the gate constant for its provenance.

    INHERITED, NOT INTRODUCED HERE: like every "not enough travel" rule, a clip of someone standing
    still can segment into reps and fire (`situp.rule_incomplete_rom` documents the mechanism).
    """
    segment = [
        frame
        for frame in core
        if frame.valid
        and (
            np.isfinite(frame.m("thigh_elevation_left"))
            or np.isfinite(frame.m("thigh_elevation_right"))
        )
    ]
    if len(segment) < ctx.min_frames:
        return []

    gate = [frame.m("anterior_axis_length") for frame in segment]
    gate = [value for value in gate if np.isfinite(value)]
    if not gate or not float(np.median(gate)) > KNEE_LIFT_VIEW_GATE:
        return []

    values = [
        float(np.nanmax([frame.m("thigh_elevation_left"), frame.m("thigh_elevation_right")]))
        for frame in segment
    ]
    peak = float(np.max(values))
    if not peak < KNEE_LIFT_THRESHOLD:
        return []

    severity = severity_from_range(
        peak, KNEE_LIFT_THRESHOLD, KNEE_LIFT_SEVERE_THRESHOLD, lower_is_worse=True
    )
    peak_deg = _hip_flexion_deg(peak)
    return [
        build_detection(
            fault_id="hk_insufficient_knee_lift",
            fault_name="Insufficient Knee Lift",
            kg_query=HIGH_KNEE_KNEE_LIFT_KG_QUERY,
            retrieval_mode="kg",
            segment_metrics=segment,
            # The peak is the HIGHEST thigh of the drive, still short of the cut. Unclipped.
            score_values=values,
            severity=severity,
            confidence=severity,
            observability="medium",
            evidence={
                "peak_hip_flexion_deg": round(peak_deg, 1),
                "threshold_hip_flexion_deg": KNEE_LIFT_HIP_FLEXION_DEG,
                "anterior_axis_length_median": round(float(np.median(gate)), 3),
                "primary_label": "peak hip flexion of the driving leg (deg)",
                "primary_value": round(peak_deg, 1),
                "primary_threshold": KNEE_LIFT_HIP_FLEXION_DEG,
            },
            citation=(
                "Alberton CL, Pinto SS, da Silva Azenha NA, Cadore EL, Tartaruga MP, Brasil B, "
                "Kruel LF. Kinesiological Analysis of Stationary Running Performed in Aquatic and "
                "Dry Land Environments. J Hum Kinet (2015) 49:5-14, PMC4723158, "
                "DOI 10.1515/hukin-2015-0103."
            ),
            citation_support=(
                "Alberton gives the TARGET for running in place: \"the right hip and knee flexion "
                "to 90° starting the swing phase\". It states no fault tolerance. The 65.6 "
                "degree cut is not from any paper: this project derived it from annotators' "
                "free-text comments in the EgoExo-Fitness dataset (Li YM et al., ECCV 2024) -- the "
                "cut that best separates actions whose comments complain about leg height, "
                "held-out AUC 0.730 [0.586, 0.864] (notes/egoexo-silent-rules-full-archive.md). "
                "It still carries a card on 40% of side-camera clips without such a complaint, "
                "so one card is "
                "a prompt to check, not a verdict."
            ),
        )
    ]


# FOUR of the parent spec's five High Knee rules are ABSENT rather than silent, and the
# distinction is the one this registry has always drawn: a silent stub asserts "real fault, and
# the number, the sensor or the corroboration is missing"; an absent rule asserts "no citation
# supports this as written, or the quantity it reads does not measure it".
#
# `hk_trunk_lean_back` and `hk_forward_trunk_collapse` -- WITHDRAWN TOGETHER, because they are the
# two signs of ONE scalar and they fail on that scalar's reference axis:
#
#   1. THE MOVEMENT HAS NO VERTICAL, AND THE SUBSTITUTE IS THE SIZE OF THE FAULT. A trunk lean is
#      an angle from the WORLD VERTICAL. Group E established across three movements that the image
#      vertical is not the world vertical, and this corpus proves it twice over by shipping its
#      side cameras rolled 90 degrees. Leg Abduction's answer -- take the vertical from the
#      SUPPORT LIMB -- is the only construction available, and in a marching drill it does not
#      hold: measured over the 12 side-camera pairs, the angle between the trunk and the support
#      limb is 6.4-14.2 deg (median 9.3) DURING NORMAL MARCHING, against rule thresholds of
#      10-15 deg (backward) and 15-20 deg (forward). The reference axis is as uncertain as the
#      quantity being measured, and the rule attributes the whole of it to the trunk.
#
#      AND THE ERROR RUNS TOWARD ONE RULE'S FIRING DIRECTION. The spec's 10 deg backward cut fires
#      on a median 47.0% OF FRAMES across the side cameras -- 46-56% on the two actions humans
#      judged faultless on every criterion -- while the 15 deg forward cut fires on 0.0%. An
#      unsigned or unvalidated baseline offset that runs toward the fault is `pushup_head_drop`'s
#      finding and Torso Twist's brace finding for the THIRD time; what is new is that here it
#      sinks both signs at once, one by false-firing and one by never firing at all.
#
#   2. NEITHER HAS A SCOPED KG NODE. "Trunk Lean", "Forward Trunk Lean" and "Lumbar
#      Hyperextension" all return zero matches under movement="High Knee".
#
#   3. THE CITATION IS ABOUT RUNNING, NOT ABOUT THIS DRILL. Bramah et al. (2018), PMID 30193080,
#      re-fetched: a controlled laboratory study of 72 INJURED RUNNERS and 36 healthy controls, in
#      which "the injured runners demonstrated greater contralateral pelvic drop and forward trunk
#      lean at midstance". Midstance of overground running is not an instant of a stationary
#      marching drill, and the finding is an injured-vs-healthy CONTRAST, not a technique
#      criterion. The parent spec marks this citation VERIFIED, and it is -- for its own claim.
#
#   4. AND THE CRITERION WITH THE POSITIVE CLASS MEASURES SOMETHING ELSE. EgoExo's "Maintain a
#      stable upper body" fails on 10 of 68 actions and its comments describe SWAY -- variance --
#      where both rules read a signed mean. On the six reachable actions, the three judged FALSE
#      and the three judged TRUE separate on neither: median trunk lean of -8.02 deg (FALSE) vs
#      -10.38 deg (TRUE), i.e. the WRONG WAY, and median standard deviation 8.69 vs 8.52 deg, i.e.
#      not at all. The corpus cannot discriminate these rules, and it is the reading of the criterion
#      -- not the n of 6 -- that is decisive.
#
#   NOT SAID BY THIS WITHDRAWAL: that throwing the torso backward to hoist a knee is fine. What is
#   missing is a vertical. Design spec section 7.1.
#
# `hk_contralateral_pelvic_drop` -- WITHDRAWN, and a control refutes it before any threshold is
# argued:
#
#   1. THREE SIMULTANEOUS CAMERAS DISAGREE BY MORE THAN THE THRESHOLD. The three exo views film
#      the SAME instant of the SAME performance, so any disagreement between them is pure
#      projection, with no performance variation in it. Median pelvic obliquity per (action,
#      camera), restricted to the two cameras whose `anterior_axis_length` says they can see
#      anything, over the six actions:
#
#          spread between exo_l and exo_r    1.9, 4.0, 7.2, 8.9, 9.3, 12.9 deg
#
#      against the parent spec's "> ~5-8 deg" threshold. THE CAMERA MOVES THE QUANTITY BY MORE
#      THAN THE FAULT DOES, on four of six actions.
#
#      AND FRAME BY FRAME THE TWO CAMERAS ARE ANTI-CORRELATED: r = -0.554, -0.426, -0.166, -0.145,
#      +0.122, +0.217. They do not merely offset each other, they largely disagree about which way
#      the pelvis is tilting at any instant. A quantity two simultaneous views of one pelvis
#      report in opposite directions is not measuring the pelvis.
#
#   2. NO SCOPED KG NODE. "Pelvic Drop" and "Contralateral Pelvic Drop" both return zero under
#      movement="High Knee"; "Pelvic Control" reaches only a shared, dangling QualityDimension.
#
#   3. THE CITATION'S OWN CAVEAT IS ALREADY IN THE PARENT SPEC and is not the reason: Bramah's
#      pelvic-drop-to-injury association is strong (an 80% increase in the odds of being
#      classified injured per degree), and McCarney 2020 contests only the drop-to-abductor-
#      weakness step. The withdrawal is about MEASURABILITY, not about whether the fault is real.
#
#   NOT SAID BY THIS WITHDRAWAL: that a dropping pelvis is fine. Bramah's is the strongest single
#   result any citation in this section carries. What is missing is a monocular quantity that
#   survives the camera. Design spec section 7.2.
#
# `hk_stride_asymmetry` -- WITHDRAWN, three failures, and they are `jj_landing_asymmetry`'s:
#
#   1. NO SCOPED KG NODE. "Asymmetry" and "Stride Asymmetry" under movement="High Knee" reach only
#      the shared `Symmetry` quality dimension carried by the Squat flagship, and it is dangling.
#
#   2. IT IS A DISJUNCTION OF TWO QUANTITIES. The heuristic compares "peak knee-lift height AND
#      per-side pelvic-drop angle", firing if either differs by 15-20%. One `fault_id` whose
#      evidence might be a knee or a pelvis cannot produce a coherent explanation card, and
#      `fault_id` is the join key between the spec, the registry and every stored analysis. One of
#      the two quantities is the one withdrawal 1 above just refuted.
#
#   3. "CONSISTENTLY ACROSS REPS" IS CROSS-REP STATE THIS ARCHITECTURE DOES NOT HAVE.
#      `run_detector` scores one repetition at a time and `merge_by_fault` reports the rep count
#      afterwards. `arm_vw` and `jj_landing_asymmetry` recorded the same limit for the same spec
#      wording; the rep semantics chosen here (one drive per repetition, so a single repetition
#      contains ONE side's drive) make it structural rather than incidental.
#
#   NOT SAID BY THIS WITHDRAWAL: that a habitually under-driving side is fine. Design spec
#   section 7.3.
#
# `HIGH_KNEE_METRIC_KEYS` must stay a two-way match with what `high_knee_compute_raw` emits
# (pinned by `test_metric_keys_match_the_emitted_metrics_exactly`): a key the tuple omits is
# dropped by `run_detector`, which builds each CoreFrame's metrics dict FROM this tuple, and read
# back as NaN by every rule.
HIGH_KNEE_DETECTOR = MovementDetector(
    "High Knee",
    HIGH_KNEE_METRIC_KEYS,
    high_knee_compute_raw,
    high_knee_assign_phases,
    (
        rule_insufficient_knee_lift,
    ),
    # `validated` stays at its default False -- module header.
    rep_signal="thigh_elevation_difference",
    # `max` with rectification: the two legs alternate, so the signed difference is BIPOLAR and
    # its magnitude peaks once per knee drive. Torso Twist is the precedent for `rep_rectify`.
    rep_polarity="max",
    rep_rectify=True,
    # `extended` -- the repetition opens away from the effort peak, with the thighs level.
    rep_start="extended",
    # `min_rep_seconds` IS LOWERED, AND THIS IS THE MOVEMENT `base.py:55` RESERVED THE KNOB FOR BY
    # NAME. Jumping Jacks measured that it did not need it; this one does, and the measurement is
    # the same non-circular one: every window `segment_reps` returns is at least
    # `min_rep_seconds` long by construction, so only re-segmenting at a lower floor and
    # DIFFERENCING THE COUNTS can show the floor biting.
    #
    # Over the 18 recovered (action, camera) pairs, the default 0.4 s floor finds 52 repetitions
    # and a 0.15 s floor finds 150 -- THE DEFAULT DISCARDS 65.3% OF THEM. The surviving cadence
    # spans 0.70-2.20 Hz of single knee drives (median 1.31), i.e. 0.45-1.42 s per repetition, all
    # physically ordinary; the low floor is not manufacturing noise repetitions.
    #
    # THE CORPUS MAKES THE RESULT STRONGER, NOT WEAKER. 30 of its 68 actions are judged FAILED on
    # "maintain the fastest speed possible", so this is a population humans considered TOO SLOW --
    # and the default floor still throws away two repetitions in three. The knob was reserved for
    # a ~3 Hz regime this corpus never reaches.
    #
    # THE VALUE IS THE FRAMEWORK'S OWN ARITHMETIC, NOT A FITTED ONE. `base.py:55` states "high
    # knees run ~3Hz, about 10 frames per rep at 30fps", i.e. 0.33 s; 0.15 s is half of that,
    # leaving headroom below the fastest cadence the framework itself anticipated. It is not
    # tuned to the 1.31 Hz median this corpus happens to show. Design spec section 5.3.
    min_rep_seconds=0.15,
)

# ---------------------------------------------------------------------------------------
# REGISTERED 2026-09-26, THE SIXTEENTH AND LAST MOVEMENT.
# ---------------------------------------------------------------------------------------
# While its one rule was silent and four withdrawn the movement stayed "coming soon", because an
# analysis that can never report a fault should not wear the Beta tag. Once
# `rule_insufficient_knee_lift` went live at the data-derived 65.6 deg the user decided to register
# it, knowing its costs on the side cameras of the 59 held-out EgoExo actions:
#   - per rep (server-side segmentation): 40.0% of clips without a leg-height complaint carded,
#     86.4% of those with one;
#   - on the BROWSER path (`segmentation_disabled`, whole clip scored as one window): 23.3% and
#     50.0%. A frontal camera is silent on both paths, by the view gate.
# `validated` stays False, so the app shows it as Beta.
#
# WHAT WORKS AND IS KEPT from the unregistered period:
#   - the metric layer: cosines and ratios between body vectors, roll-, mirror- and
#     scale-invariant, which is the only reason a corpus of 90-degree-rolled frames produced
#     numbers at all;
#   - the phase assignment and the rectified per-drive repetition definition;
#   - `min_rep_seconds=0.15`, the first use of a framework knob reserved fifteen movements ago,
#     measured to recover 65% of this movement's repetitions.
#
# AND THE MOST PROMISING RULE THIS MOVEMENT COULD HAVE IS ONE THE PARENT SPEC NEVER WROTE. The
# corpus's largest fault by a wide margin is CADENCE -- 30 of 68 actions judged too slow -- the KG
# carries `High Knee:Slow Cadence` with a real correction bucket (`Maintain Even Tempo`), and
# cadence is the one quantity here that is fully roll-, view- and scale-invariant, since it is
# counted in time rather than measured in space. It is not built, because this programme
# implements the parent spec's roster and does not author new rules; it is recorded because it is
# the obvious next rule. Design spec section 7.4.
registry.register(HIGH_KNEE_DETECTOR)
