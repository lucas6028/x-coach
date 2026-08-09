# Arm Abduction (standing lateral / shoulder-abduction raise) raw metrics, phase segmentation
# and fault rules.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `arm_abduction_compute_raw` /
# `arm_abduction_assign_phases` compute per-frame quantities and a phase label only. Every number
# that decides anything belongs in a `rule_*` function. The only constant this module defines
# outside a rule, `_DEGENERATE_LENGTH`, is a division-by-zero guard, never a tunable threshold.
#
# ---------------------------------------------------------------------------------------
# LABELED CORRECT/INCORRECT GROUND TRUTH EXISTS FOR THIS MOVEMENT. IT IS THE FIRST.
# ---------------------------------------------------------------------------------------
# Every detector since Push-up carries the sentence "no labeled repetition of this movement
# exists anywhere in this repository". For Arm Abduction that sentence is FALSE and must not be
# copied forward: REHAB24-6 `Ex1` IS arm abduction (src/rehab24/dataset.py EXERCISE_NAMES["1"]),
# 178 repetitions, 9 subjects, 90 correct / 88 incorrect, with marker-driven 3-D alongside the
# video and cached MediaPipe landmarks for all 13 videos. `validated` is still False because
# NOTHING RAN THE CHECK -- see the registration site at the bottom of this module for what Ex1
# can and cannot decide. Design spec section 2.
#
# ---------------------------------------------------------------------------------------
# TWO RULES SHIP, ONE IS PERMANENTLY SILENT, ONE IS ABSENT. THE THREE ARE NOT THE SAME THING.
# ---------------------------------------------------------------------------------------
#   rule_shoulder_shrug              REGISTERED, PERMANENTLY SILENT -- real fault, cited
#                                    mechanism, MEASURED sensing failure (its docstring)
#   excessive_elevation_impingement  WITHDRAWN from the parent spec -- no function here at all
#   rule_contralateral_trunk_lean    ships
#   rule_lr_asymmetry                ships
#
# Registered-but-silent (pushup.rule_scapular_winging, band_pull_apart
# .rule_loss_of_scapular_retraction) says "real, well-cited fault, the sensor cannot see it".
# Withdrawn (OHP bar-path 2026-07-25, deadlift bar-drift 2026-08-01, curl wrist-flexion
# 2026-08-09) says "no citation supports the rule as written". Design spec sections 3 and 4.
#
# ---------------------------------------------------------------------------------------
# THE FIRST DETECTOR WHOSE SPEC-RATED `high` VIEWS ARE REACHABLE, AND THEREFORE THE FIRST
# SINCE LUNGE WITH NO VIEW GATE ANYWHERE.
# ---------------------------------------------------------------------------------------
# Every previous movement's best rules want `side`, and `side` NEVER occurs: re-measured
# 2026-08-09 by running `estimate_view_for_pose(path, allow_front=False)` over all 49 files under
# data/runtime/pose_json -- rear_oblique 37, rear 9, unknown 3, side 0. `front`/`front_oblique`
# are unreachable under allow_front=False (src/pose/view_estimation.py:14-16).
#
# Both live rules here are rated `high` on **front/rear** and both metrics are FRONTAL-PLANE and
# UNSIGNED by construction, so a rear view reads the same plane as a front one (mirrored, which
# an unsigned metric cannot tell apart) and earns full confidence with no facing determination.
# That is 9 of 49 real clips at full confidence. Nothing here gates: an elevation DIFFERENCE and
# a lateral trunk lean are the RIGHT quantity from every reachable view, and obliquity makes them
# noisier rather than different -- so `rear_oblique` and `unknown` take the x0.65 discount and
# stay live. Design spec section 6.6.
#
# ---------------------------------------------------------------------------------------
# ONE DROPPED LANDMARK SILENCES EVERY ARM ABDUCTION RULE FOR THAT FRAME.
# ---------------------------------------------------------------------------------------
# `required` below lists both shoulders, both elbows and both hips -- and deliberately NOT the
# ears (see `rule_shoulder_shrug`) and NOT the wrists (the only cue that would have read them is
# not implemented; see `ASYMMETRY_MILD_DEG`). If `visible_point` drops any ONE of the required
# points the frame is marked `valid=False` and carries no metric keys at all, so every rule that
# masks on `frame.valid` goes silent for that frame, not just the one whose input landmark went
# missing. This mirrors `pushup_compute_raw`, `ohp_compute_raw`, `lunge_compute_raw`,
# `row_compute_raw`, `band_pull_apart_compute_raw` and `bicep_curl_compute_raw`: an unmeasurable
# frame is refused wholesale rather than degraded.
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, angle_degrees, midpoint, mean_visibility, distance,
    line_angle_from_vertical, contiguous_true_segments, severity_from_range,
)
from src.pose.movements.base import CoreFrame, MovementDetector, RuleContext
from src.pose.movements import registry
from src.pose.pose_rule_detector import (
    VIEW_UNAVAILABLE_CONFIDENCE_SCALE,
    PoseRuleDetection,
    build_detection,
)

# Defined locally, matching row.py, overhead_press.py, band_pull_apart.py and bicep_curl.py:
# geometry.py exports only the lower-body and shoulder/hip constants.
LEFT_ELBOW = 13
RIGHT_ELBOW = 14

# The generic "lower body" set every movement module uses for the framework-level
# `lower_body_visibility` quality field. The name is squat-centric and carries awkwardly for a
# standing arm-isolation exercise, exactly as it does for OHP, push-up, Row, Band Pull Apart and
# Bicep Curl; this module's own rules never consume it.
LOWER_BODY_LANDMARKS = (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

ARM_ABDUCTION_METRIC_KEYS: tuple[str, ...] = (
    "left_arm_elevation_deg",
    "right_arm_elevation_deg",
    "avg_arm_elevation_deg",
    "arm_elevation_asymmetry_deg",
    "lateral_trunk_lean_deg",
    "shoulder_width",
)

# Below this a length/normalizer is degenerate and the dependent metric is NaN. Same guard value
# pushup.py, overhead_press.py, lunge.py, row.py, band_pull_apart.py and bicep_curl.py use; not a
# tunable threshold.
_DEGENERATE_LENGTH = 1e-6


def arm_abduction_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
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

        # The parent spec's own definition: angle between the torso vector (shoulder->hip) and
        # the upper-arm vector (shoulder->elbow). ~0 deg = arm at the side, ~90 deg = horizontal,
        # ~180 deg = fully overhead. Same-side hip, so a trunk lean does not silently inflate the
        # elevation of both arms together.
        #
        # `angle_degrees` consumes dims=3, i.e. MediaPipe's estimated z as well as x/y -- the
        # shared helper every movement module uses, not changed here. Under the RTMPose path
        # (src/pose/rtmpose_pose_extraction.py writes z=0.0) these ARE pure image-plane
        # projections. Measured against REHAB24-6 Ex1's markers on the MediaPipe path, the
        # per-rep mean absolute error of this angle is 20.6 deg (p90 43.8), which is why NO rule
        # in this module reads an elevation MAGNITUDE -- `rule_lr_asymmetry` reads a DIFFERENCE
        # of two like-measured quantities, where the common-mode error cancels. Design spec
        # section 2.4.
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

        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        # UNSIGNED angle of the hip->shoulder trunk line from image vertical.
        # `line_angle_from_vertical` takes abs() of both components, so an upright trunk reads 0
        # regardless of which side it side-bends toward. That is the intended reading, not a
        # limitation of the helper: the parent spec's "away from the raising arm" qualifier is
        # dropped (design spec section 5.1) because on a BILATERAL raise -- the variant this app
        # models -- there is no raising arm, so the qualifier is undefined, and on the unilateral
        # variant recovering it would need a working-side determination this layer cannot make.
        # The cost is that a lean TOWARD the working arm also fires: a wider net, in the direction
        # the verified injury mechanism (compensation during elevation) supports.
        lateral_trunk_lean = line_angle_from_vertical(shoulder_mid, hip_mid)

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
                "lateral_trunk_lean_deg": lateral_trunk_lean,
                # DIAGNOSTIC ONLY -- no rule reads this. It is emitted so that the parent spec's
                # unimplemented second asymmetry cue ("peak wrist heights differ by > 0.05
                # normalized units") stays CHECKABLE without re-deriving the measurement that
                # shows it ill-defined: see ASYMMETRY_MILD_DEG.
                "shoulder_width": shoulder_width if shoulder_width > _DEGENERATE_LENGTH else np.nan,
            }
        )

    return raw


def arm_abduction_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> concentric -> peak -> eccentric, segmented on `avg_arm_elevation_deg`.

    Mirrors `band_pull_apart_assign_phases`: this movement's signal PEAKS AT ITS MAXIMUM (arms
    overhead-most), so the peak hold is the MOST-ELEVATED 30% of the rep, i.e. the 70th
    percentile of the elevation and above -- the polarity inverse of `row_assign_phases` and
    `bicep_curl_assign_phases`, which take the 30th percentile and below. Same fallbacks: an
    empty clip returns an empty list, a clip with no finite signal is entirely `unknown`, and an
    invalid frame is `unknown` regardless of where it sits (the validity check precedes the setup
    cutoff, so an occluded frame in the opening 15% is NOT labelled `setup`).

    `peak` IS THE PARENT SPEC'S "top-hold", which is the scope `rule_lr_asymmetry` needs.

    NO SHIPPED RULE MASKS ON `setup`, AND THAT IS WHAT KEEPS THIS MOVEMENT CLEAR OF THE TRAP
    THAT SILENCED BICEP CURL'S EXTENSION TERM. That trap is arithmetic, not luck: a phase-scoped
    rule needs `phase_fraction * T >= min_frames / fps` with `min_frames = max(3, ceil(0.20 *
    fps))` (base.py:197), so a 15% `setup` window needs T >= 1.333 s per rep while a 30% `peak`
    window needs only T >= 0.667 s. Measured rep durations are 1.40-4.96 s (Fit3D
    `side_lateral_raise`, 8 subjects x 5 reps, 50 fps ffprobe-verified) and 2.77-10.53 s
    (REHAB24-6 Ex1, 178 reps, 30 fps), so the tightest real rep sits at 2.1x the `peak`
    requirement and would have sat at only 1.05x a `setup` one. Design spec section 6.2.
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

    # The most-ELEVATED 30% of the rep is the peak hold.
    peak_threshold = float(np.percentile(valid_elevation, 70))
    highest_index = int(
        np.nanargmax(np.where(np.isfinite(elevation_values), elevation_values, -np.inf))
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
        if np.isfinite(value) and value >= peak_threshold:
            phases.append("peak")
        elif index < highest_index:
            phases.append("concentric")
        else:
            phases.append("eccentric")
    return phases


# ---------------------------------------------------------------------------------------
# STEP 0 -- KG QUERY RESOLUTION, recorded before any rule was written. Each string below was
# checked against data/kg/sports_kg_v3.graphml with `retrieve_graph_context(query,
# movement="Arm Abduction")` -- the function PRODUCTION calls, not just `resolve_nodes`. Observed
# results, not predicted ones:
#
#   "Shoulder Shrug"           -> Arm Abduction:Compensatory Shoulder Shrug
#       quality_impacts: Shoulder Depression                                          NON-EMPTY
#   "Trunk Lean Compensation"  -> Arm Abduction:Trunk Lean Compensation
#       quality_impacts: No Compensatory Trunk Movement                               NON-EMPTY
#   "Muscle Imbalance"         -> Muscle Imbalance (generic Cause node)
#       NO buckets under this movement                                                THIN
#
# The one gap is recorded rather than masked: the graph has NO Arm Abduction asymmetry fault
# node. "Asymmetry" and "Left Right Asymmetry" both resolve to the generic `Symmetry`
# QualityDimension whose inbound edges are all Squat and Overhead Press, and "Muscle Imbalance"
# returns only the bare generic node here -- under movement="Overhead Press" that same query
# ALSO returns the OHP-scoped fault that carries the content, which is why
# `ohp_asymmetric_press` uses it. It is kept anyway because it is the semantically correct thin
# card and matches the sibling rule; pointing at the shared `Range Of Motion` QualityDimension
# WOULD return a rich bucket set, and was rejected for the same reason Band Pull Apart and Bicep
# Curl rejected it -- that node's `corrections` bucket is "Wrapping Surface Adjustment",
# meaningless here. A semantically correct thin card beats a semantically wrong full one. The
# gap is logged against TODO.md's existing "many faults have no KG node" item; the graphml is
# gitignored, so adding the node is a deploy step.
#
# NOTE, recorded and NOT acted on: the graph's third Arm Abduction fault is `Arm Abduction:
# Incomplete Elevation` -- the OPPOSITE fault to the parent spec's withdrawn "raised too high"
# rule, and the richest of the three (quality_impacts: Humerus Abduction; causes: Limited
# Shoulder ROM). Every other movement in the parent spec got an incomplete-ROM rule; this one
# did not, and no rule here points at that node. Inventing one would need a source that puts a
# number on insufficient abduction. Design spec section 4.
ARM_ABD_SHRUG_KG_QUERY = "Shoulder Shrug"
ARM_ABD_TRUNK_LEAN_KG_QUERY = "Trunk Lean Compensation"
ARM_ABD_ASYMMETRY_KG_QUERY = "Muscle Imbalance"

# Imported rather than re-typed, so a change to the shared constant cannot silently skip this
# module.
_OFF_VIEW_CONFIDENCE = VIEW_UNAVAILABLE_CONFIDENCE_SCALE

# THE FULLY-OBSERVABLE SET, WRITTEN IN THE POSITIVE -- the inverse of the negative form Band
# Pull Apart and Bicep Curl use, and the difference is which set is the small one. Those
# movements measure SAGITTAL quantities, where the blind views are few and naming them is the
# compact statement. Both rules here measure FRONTAL-plane quantities, where the fully-observable
# views are few (the parent spec rates both `high` on front/rear specifically), so the whitelist
# is the compact statement AND the one that says what is novel: `rear` is reachable, `front` is
# not (allow_front=False), and everything else -- `rear_oblique`, `unknown`, `side` -- is
# foreshortened rather than blind and takes the x0.65 discount instead of being gated.
#
# `front` is listed even though `estimate_view_for_pose(allow_front=False)` can never emit it.
# That is NOT the dead-weight-that-reads-as-coverage case Bicep Curl's negative gate avoids: this
# is the spec's own observability rating transcribed, the whitelist is not a gate (nothing is
# excluded by it, only discounted), and `run_detector` is called with whatever view label its
# caller supplies -- the REHAB24-6 replay harness (src/rehab24/lunge_rule_validation.py
# ORACLE_VIEWS) deliberately feeds the literal "front" for exactly this reason.
FRONTAL_OBSERVABLE_VIEWS = {"front", "rear"}


def rule_shoulder_shrug(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Registered but PERMANENTLY SILENT -- always returns [].

    Shrugging (upper-trapezius dominance) during abduction is a real, cited fault. Mun WL et al.,
    Medicina (2025) PMC12029123, read in the RAG doc: "overactivation of the muscles that elevate
    the scapula, such as the upper trapezius and levator scapulae, and low activation of the
    muscles that stabilize the scapula ... can lead to increased shoulder instability, which can
    increase the risk of musculoskeletal conditions such as impingement syndrome and rotator cuff
    injury", and "Persistent overactivity of the UT can lead to scapular dysfunction (or
    dyskinesia), such as subacromial impingement or glenohumeral instability". The fault is
    genuine. What fails is the SENSING -- and unlike every other silent rule in this repository,
    that failure was MEASURED rather than argued.

    (a) THE METRIC COLLAPSES DURING ABDUCTION AS A MATTER OF ANATOMY, IN 3-D GROUND TRUTH, ON THE
        BILATERAL VARIANT, WITH NO POSE ESTIMATOR ANYWHERE IN THE PATH. The spec's proxy is
        `neck_gap = ear_y - shoulder_y` against a setup baseline. On Fit3D `side_lateral_raise`
        (mocap `joints3d_25`, 8 subjects x 5 reps), the within-clip Spearman correlation between
        the head-to-shoulder vertical gap and the arm's own elevation is -0.699 to -0.954 across
        ALL 8 subjects, the gap travels 27%-94% of its own baseline within a clip, and the spec's
        18% shrink threshold fires on 34 of 40 reps performed deliberately for a mocap capture.
        Decomposed, both endpoints move the same way -- the shoulder joint rises (per-subject rho
        +0.00 to +0.94) AND the head drops (rho -0.32 to -0.86). The glenohumeral joint rising
        during abduction is scapulohumeral rhythm: it is the movement, not a fault.

    (b) MEDIAPIPE REPORTS THE GLENOHUMERAL JOINT, NOT THE ACROMION, so the one reading that could
        rescue the metric is unavailable. On REHAB24-6 Ex1, each candidate "shoulder" measured as
        height above the mid-hip and expressed as a fraction of its own baseline: the marker
        CLAVICLE (acromion -- true scapular elevation) travels 1.0%, the marker GLENOHUMERAL
        joint 13.9%, and MediaPipe's landmark 11/12 travels 11.2%. MediaPipe tracks the humerus,
        an order of magnitude away from the point the rule needs. The fire rates split on the
        same line: 18% fires on 1/178 reps read off the marker clavicle and 172/178 = 96.6% read
        off MediaPipe, with the working-side gap correlating rho = -0.957 against true arm
        elevation ON CORRECT REPS ALONE. The control that makes this a statement about the ARM
        rather than about the head or the framing: on the RESTING arm of the same unilateral
        reps, the gap is uncorrelated with the working arm's elevation (rho = +0.068).

    This measurement also converts `band_pull_apart.rule_loss_of_scapular_retraction`'s asserted
    premise -- "MediaPipe's shoulder landmark is a GLENOHUMERAL point ... and it moves with the
    humerus" -- into a measured one. Whether the same confound reaches `band_pull_apart
    .rule_shrugging`, which SHIPS LIVE on this same construction, is NOT claimed here: that
    movement's excursion is horizontal abduction at roughly fixed elevation, so the confound
    plausibly differs in kind and nothing measured here bears on it. Logged in TODO.md as a check
    to run, not as a defect found.

    THE CONFOUND IS VARIANT-INDEPENDENT, which is what licenses silencing rather than gating: on
    a bilateral raise both shoulders ride their own humerus, so there is nothing left to read the
    shrug against. Component (a) is measured on the bilateral variant precisely to establish
    this. The parent spec's own mitigation was measured too and does not rescue it -- restricting
    to frames below 90 deg of elevation (its "early or disproportionate shrug") takes the
    MediaPipe fire rate only from 96.6% to 49.4%, which is half of every rep in a dataset that is
    half correct.

    SEPARATELY, THE 18% CARRIES NO CITATION. Mun measures EMG at four abduction angles during a
    Pilates Reformer arm-work movement and supplies no landmark-displacement magnitude in any
    units. That would matter a great deal for a rule that fires, and does not for one that never
    does -- it is recorded so a future reader who repairs the sensing does not inherit the number
    as though it were cited.

    SILENT, NOT WITHDRAWN, AND THE DISTINCTION IS LOAD-BEARING. Mun genuinely backs the fault, so
    this is a sensing failure, not a citation failure. The parent spec carries a NOTE, not a
    WITHDRAWN blockquote. Contrast the arc rule, which is ABSENT from this module entirely.

    NOT SUBSTITUTED, DELIBERATELY, AND ITS METRIC IS NOT EVEN EMITTED. Shipping a different
    quantity under this fault_id would attach Mun's citation to something Mun says nothing about.
    And unlike `bicep_curl`'s `upper_arm_length` -- emitted as a live diagnostic so an unreachable
    disjunct's arithmetic stays checkable -- `shoulder_ear_gap` is absent from
    ARM_ABDUCTION_METRIC_KEYS: no live rule reads it, and emitting it would force landmarks 7/8
    into `required`, where the all-or-nothing gate would let one lost ear silence the two rules
    that DO fire. Measured on the 49-file production corpus the ears are never lost (0.00% of
    9426 frames) so the cost today is zero, but paying a live cost for a dead metric is the wrong
    default regardless.

    OPEN, RECORDED, NOT RESOLVED: a working shrug rule would need shoulder height read AT MATCHED
    ARM ELEVATION -- comparing like with like across the rep rather than against a setup baseline
    taken at a different arm position. That is a novel construction with no citation and no
    validation, and inventing it here is the fabrication this project's honesty rules forbid.

    THE KG IS NOT THE GAP: `Shoulder Shrug` resolves to `Arm Abduction:Compensatory Shoulder
    Shrug` with a non-empty `quality_impacts` bucket (`Shoulder Depression`). The metric is the
    gap.
    """
    return []


# FROM THE SPEC: "Flag if lateral lean `> 12deg` away from the raising arm during concentric".
# Taken UNSIGNED -- see `lateral_trunk_lean_deg` in `arm_abduction_compute_raw` for why the
# direction qualifier is undefined for the variant this app models.
TRUNK_LEAN_MILD_DEG = 12.0
# RULE-LEVEL CHOICE MADE HERE. The parent spec states NO severity ramp for ANY Arm Abduction
# fault (the Lunge section states its ramps explicitly, so the absence is meaningful). 30 is 2.5x
# the fire threshold, the convention `pushup.rule_hip_sag` uses. A display/ranking curve, not a
# cited quantity.
TRUNK_LEAN_SEVERE_DEG = 30.0


def rule_contralateral_trunk_lean(
    core: list[CoreFrame], ctx: RuleContext
) -> list[PoseRuleDetection]:
    """Flag the torso side-bending to help hoist the arm -- frontal-plane compensation.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM.
      FIRE THRESHOLD 12 deg: FROM THE SPEC.
      SEVERITY RAMP 12 -> 30 deg: A RULE-LEVEL CHOICE (see TRUNK_LEAN_SEVERE_DEG).

    PHASE SCOPE `concentric`, FROM THE SPEC's own wording ("during concentric").

    THIS RULE SHIPS DESPITE AN `UNVERIFIED` CITATION LINE, AND THE DISCRIMINATOR IS MEASUREMENT
    RATHER THAN PARAPHRASE. The parent spec's own citation_support ends "the specific
    frontal-plane trunk-lean substitution during abduction is UNVERIFIED in a peer-reviewed
    source", and fetching StatPearls NBK554518 confirms it: asked for any mention of trunk lean,
    lateral trunk flexion, side-bending or contralateral compensation during abduction, the
    source yields nothing. That is the same starting point as the WITHDRAWN curl wrist-flexion
    rule -- and it lands differently for three measured reasons:

      1. THE CUE CARRIES INFORMATION ABOUT REP CORRECTNESS. On REHAB24-6 Ex1's 178 human-labeled
         reps, ranking incorrect reps above correct ones on this quantity (marker 3-D) gives a
         per-subject median AUC of 0.800 across 9 subjects (pooled 0.647); 0.760 per-subject when
         measured against the rep's own setup baseline instead of vertical. Curl wrist-flexion
         had no such measurement and no way to obtain one.
      2. OBSERVABILITY IS `high` ON front/rear, and unlike every previous movement those views
         are reachable (module header). Wrist flexion was rated `low` on EVERY view.
      3. THE INJURY MECHANISM IS VERIFIED. StatPearls attributes impingement in part to
         "inadequate scapular upward rotation and posterior tilt", and gross trunk compensation
         during elevation is a coarse form of that. What is unverified is the SPECIFIC
         frontal-plane substitution finding, not the mechanism.

    So this is the `lunge_insufficient_depth` shape -- a real cue whose cited cut sits in the
    tail of the observed distribution -- and notes/lunge-rule-validation.md section 5.4 settled
    that treatment: "That is still not evidence the rules are wrong ... Neither threshold moves."

    THE THRESHOLD'S PLACEMENT, RECORDED RATHER THAN REPAIRED. 12 deg fires on 0/178 REHAB24-6 Ex1
    reps (max lean observed: 7.6 deg) and 1/40 Fit3D `side_lateral_raise` reps (max 14.1 deg).
    Read plainly: as shipped this rule will almost never fire, and when it does the lean is gross.
    Every one of those numbers is 3-D ground truth; in the image plane obliquity foreshortens a
    frontal-plane lean, so a real lean reads SMALLER than it is -- a missed fault, never a false
    one, in the same direction the threshold placement already errs.

    THE SPEC'S "or if it grows with load across a set" IS NOT IMPLEMENTED, and is not an
    oversight: this pipeline has no load, and `run_detector` scores one rep at a time with no
    cross-rep state anywhere. Absent rather than approximated.

    NO VIEW GATE, ONLY A DISCOUNT -- a lateral trunk lean is the right quantity from every
    reachable view and obliquity makes it noisier, not different. Module header for why the
    whitelist is written in the positive.
    """
    scale = 1.0 if ctx.view_type in FRONTAL_OBSERVABLE_VIEWS else _OFF_VIEW_CONFIDENCE

    def lean(frame: CoreFrame) -> float:
        return frame.m("lateral_trunk_lean_deg")

    mask = [
        frame.valid
        and frame.phase == "concentric"
        and np.isfinite(lean(frame))
        and lean(frame) > TRUNK_LEAN_MILD_DEG
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        leans = [lean(frame) for frame in segment]
        max_lean = float(np.nanmax(leans))
        severity = severity_from_range(
            max_lean, TRUNK_LEAN_MILD_DEG, TRUNK_LEAN_SEVERE_DEG, lower_is_worse=False
        )
        detections.append(
            build_detection(
                fault_id="arm_abd_contralateral_trunk_lean",
                fault_name="Trunk Lean (Frontal-Plane Compensation)",
                kg_query=ARM_ABD_TRUNK_LEAN_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=leans,
                severity=severity,
                confidence=severity * scale,
                observability=(
                    "high" if ctx.view_type in FRONTAL_OBSERVABLE_VIEWS else "medium"
                ),
                evidence={
                    "max_lateral_trunk_lean_deg": round(max_lean, 2),
                    "threshold_deg": TRUNK_LEAN_MILD_DEG,
                    "primary_label": "lateral trunk lean from vertical",
                    "primary_value": round(max_lean, 2),
                    "primary_threshold": TRUNK_LEAN_MILD_DEG,
                },
                citation=(
                    "Creech JA, Busse A, Li D, et al. Shoulder Impingement Syndrome, StatPearls "
                    "(NCBI Bookshelf NBK554518, updated 2026)."
                ),
                citation_support=(
                    "StatPearls attributes impingement in part to \"inadequate scapular upward "
                    "rotation and posterior tilt\" — i.e., compensation that fails to control the "
                    "scapula during elevation, which contralateral trunk lean is a gross form of. "
                    "The injury MECHANISM is verified; the specific frontal-plane trunk-lean "
                    "substitution during abduction is UNVERIFIED — no peer-reviewed source read "
                    "isolates trunk lateral flexion during abduction, and the source contains no "
                    "trunk-lean statement and no 12-degree figure."
                ),
            )
        )
    return detections


# FROM THE SPEC: "Flag if `asym > 12deg` at the top-hold".
#
# ITS PROVENANCE NEEDS STATING PRECISELY, BECAUSE THE CITATION IS A DIFFERENT QUANTITY IN
# DIFFERENT UNITS. Terre M & Solana-Tramunt M, Healthcare (Basel) 2025;13(10):1153 (PMC12110944)
# was fetched and read: it measures MIDDLE- AND LOWER-TRAPEZIUS EMG SYMMETRY during bilateral
# scapular retraction at 45 and 90 degrees of shoulder abduction, and every threshold in it is a
# PERCENTAGE -- "asymmetries between 10% and 15% are often associated with a higher risk of
# injury and reduced performance", on a limb-symmetry scale of asymmetry 0-79% / limit 80-89% /
# normal 90-100%. NO ANGULAR THRESHOLD APPEARS ANYWHERE IN THE PAPER. A 12-degree difference on a
# 90-degree raise is ~13%, which lands inside the cited band -- but that correspondence is a
# RECONSTRUCTION, NOT A PROVENANCE, it silently assumes a 90-degree target this pipeline does not
# have (the same missing referent that withdrew the impingement-arc rule), and the parent spec
# never states it.
#
# This is the `ohp_asymmetric_press` situation exactly -- cited at 7 degrees of scapular angle /
# 1.5 cm of lateral shift, shipped as 0.15 of normalized wrist height -- and it takes the same
# treatment: ship the spec's number unchanged, with the mismatch written here. Re-expressing the
# rule as a percentage was considered and REJECTED: changing units changes what fires, which the
# no-tuning rule covers, and it would still be transferring an EMG figure to a kinematic quantity.
ASYMMETRY_MILD_DEG = 12.0
# RULE-LEVEL CHOICE MADE HERE. 2.5x the fire threshold, the same convention as the trunk-lean
# ramp above.
ASYMMETRY_SEVERE_DEG = 30.0

# THE PARENT SPEC'S SECOND ASYMMETRY CUE IS NOT IMPLEMENTED, and the reason is new to this
# codebase. "or if peak wrist heights differ by > 0.05 normalized units" is NOT redundant with
# the angular cue the way Bicep Curl's elbow-displacement disjunct was -- a wrist-height
# difference and an elevation-angle difference genuinely differ when arm lengths or elbow bends
# differ. It is dropped because `0.05` IN RAW NORMALIZED IMAGE UNITS IS NOT A WELL-DEFINED
# CRITERION: normalized coordinates scale with how much of the frame the subject occupies.
# Measured across the 43 production pose JSONs under data/runtime/pose_json that carry a usable
# shoulder width, the per-clip median `shoulder_width` runs 0.0591 to 0.4923 normalized units --
# an 8.3x spread (p90/p10 = 1.54x). So 0.05 normalized units is 0.102 shoulder-widths on the
# widest-framed clip and 0.846 on the narrowest: the same physical asymmetry fires or does not
# depending on how far the phone was from the lifter. `ohp_asymmetric_press` avoids this by
# normalizing its own asymmetry by shoulder width explicitly; this spec line does not, and
# renormalizing it here would be inventing a threshold. `shoulder_width` is still emitted so the
# spread stays checkable; the arithmetic is pinned by
# test_the_wrist_height_disjunct_is_frame_scale_dependent.


def rule_lr_asymmetry(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag one arm lagging the other at the top of the raise.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM, AND SEE ASYMMETRY_MILD_DEG FOR
    THE THIRD THING THIS ONE NEEDS SAID.
      FIRE THRESHOLD 12 deg: FROM THE SPEC (whose citation gives 10-15% of EMG symmetry, not
        degrees of joint angle -- read the constant's comment before quoting this number).
      SEVERITY RAMP 12 -> 30 deg: A RULE-LEVEL CHOICE.

    PHASE SCOPE `peak`, FROM THE SPEC's own wording ("at the top-hold").

    THIS IS THE ONE RULE IN THE MODULE THE 20.6-DEGREE MEDIAPIPE ELEVATION ERROR DOES NOT BOUND,
    and that is a property of the metric rather than a hope. `arm_elevation_asymmetry_deg` is a
    DIFFERENCE of two quantities measured the same way by the same estimator on the same frame,
    so a common-mode projection or estimator error cancels; an elevation MAGNITUDE would inherit
    it in full. That is also why no shipped rule here reads a magnitude.

    THE ONLY LABELED DATASET FOR THIS MOVEMENT CANNOT VALIDATE THIS RULE, IN EITHER DIRECTION,
    AND THE REASON IS A VARIANT MISMATCH RATHER THAN A DEFECT. REHAB24-6 Ex1 is UNILATERAL on
    178/178 reps (`exercise_subtype == "right arm"`), and on a one-armed raise the two arms'
    elevations differ by 64.3-132.2 degrees (median 104.2), so this threshold is exceeded on
    every rep of both classes. That 178/178 is NOT a false-positive rate: a rule reporting "your
    two arms did completely different things" is CORRECT about a one-armed raise, and what the
    measurement establishes is that Ex1 is not performing the movement this app calls Arm
    Abduction. The bilateral variant comes from Fit3D `side_lateral_raise` instead (8 subjects x
    5 reps of 3-D mocap, no correctness label), where the same threshold at the same phase fires
    on 2/40 reps -- median asymmetry at the peak 4.4 deg, max 16.8 deg. Design spec sections 2.2
    and 2.3.

    NO BILATERAL PRECONDITION IS IMPLEMENTED. Gating this rule on "both arms are actually
    raising" would need an elevation floor no cited source supplies, which is the construct the
    OHP bar-path and deadlift bar-drift withdrawals both rejected. Stated, not corrected.

    THE SPEC'S SECOND CUE IS NOT IMPLEMENTED BECAUSE ITS THRESHOLD IS FRAME-SCALE DEPENDENT --
    see the block comment above ASYMMETRY_MILD_DEG for the measurement.

    THE SPEC'S TRAILING "sustained across reps" IS ALSO NOT IMPLEMENTED: no rule in this codebase
    carries cross-rep state. `run_detector` scores one rep at a time and `merge_by_fault` reports
    the rep count afterwards, which is the framework's answer to the same question.

    NO VIEW GATE, ONLY A DISCOUNT. `|L - R|` is sign-invariant, so a rear view reads it
    identically to a front one; obliquity foreshortens both arms together, so a real asymmetry
    reads smaller -- a missed fault, never a false one.
    """
    scale = 1.0 if ctx.view_type in FRONTAL_OBSERVABLE_VIEWS else _OFF_VIEW_CONFIDENCE

    def asymmetry(frame: CoreFrame) -> float:
        return frame.m("arm_elevation_asymmetry_deg")

    mask = [
        frame.valid
        and frame.phase == "peak"
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
                fault_id="arm_abd_lr_asymmetry",
                fault_name="Left/Right Asymmetry (One Arm Lagging)",
                kg_query=ARM_ABD_ASYMMETRY_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=values,
                severity=severity,
                confidence=severity * scale,
                observability=(
                    "high" if ctx.view_type in FRONTAL_OBSERVABLE_VIEWS else "medium"
                ),
                evidence={
                    "max_arm_elevation_asymmetry_deg": round(max_asymmetry, 2),
                    "threshold_deg": ASYMMETRY_MILD_DEG,
                    "primary_label": "left/right arm elevation difference",
                    "primary_value": round(max_asymmetry, 2),
                    "primary_threshold": ASYMMETRY_MILD_DEG,
                },
                citation=(
                    "Terré M, Solana-Tramunt M, Healthcare (Basel) (2025), 13(10):1153, "
                    "PMC12110944, DOI 10.3390/healthcare13101153."
                ),
                citation_support=(
                    "The paper states \"asymmetries between 10% and 15% are often associated with "
                    "a higher risk of injury and reduced performance,\" on a limb-symmetry scale "
                    "of asymmetry 0–79% / limit 80–89% / normal 90–100%. IT MEASURES MIDDLE- AND "
                    "LOWER-TRAPEZIUS EMG SYMMETRY during bilateral scapular retraction at 45° and "
                    "90° of shoulder abduction, and contains NO ANGULAR THRESHOLD: the 12° cut "
                    "applied here is the parent spec's, and its correspondence to the cited band "
                    "(~13% of a 90° raise) is a reconstruction, not a provenance."
                ),
            )
        )
    return detections


# ALL FOUR of the parent spec's Arm Abduction rules are accounted for, and the three treatments
# are deliberately different. `rule_shoulder_shrug` is listed and permanently silent so the spec
# and the code stay in 1:1 correspondence -- registering it costs one no-op call per clip and
# buys an auditor the answer "yes, it is accounted for, and here is why it says nothing", the
# same trade `pushup.rule_scapular_winging` and `band_pull_apart.rule_loss_of_scapular_retraction`
# make. `excessive_elevation_impingement_arc` is ABSENT rather than silent because its problem is
# the citation and the arithmetic, not the sensor: StatPearls describes the 70-120 degree painful
# arc as a DIAGNOSTIC SIGN of existing subacromial pathology and never says raising through it is
# a fault, all 178 REHAB24-6 Ex1 reps enter that band (so its first disjunct is vacuous and
# reduces to "the silent rule fired"), and its second disjunct needs a prescribed target that
# exists nowhere in this pipeline. A silent stub would assert that elevation is a real fault the
# sensor cannot see; the sensor sees elevation angles perfectly well. Design spec section 4.
#
# `ARM_ABDUCTION_METRIC_KEYS` must stay a two-way match with what `arm_abduction_compute_raw`
# emits (pinned by `test_metric_keys_match_the_emitted_metrics_exactly`): a key the tuple omits
# is dropped by `run_detector`, which builds each CoreFrame's metrics dict FROM this tuple, and
# read back as NaN by every rule.
ARM_ABDUCTION_DETECTOR = MovementDetector(
    "Arm Abduction",
    ARM_ABDUCTION_METRIC_KEYS,
    arm_abduction_compute_raw,
    arm_abduction_assign_phases,
    (
        rule_shoulder_shrug,
        rule_contralateral_trunk_lean,
        rule_lr_asymmetry,
    ),
    # `validated` stays at its default False -- and for the FIRST TIME in this registry that is
    # not because labeled data is unavailable. REHAB24-6 `Ex1` IS arm abduction: 178 repetitions,
    # 9 subjects (every one contributing both classes), 90 correct / 88 incorrect, 0 flagged
    # mocap-erroneous, with marker 3-D and cached MediaPipe landmarks for all 13 videos. NOTHING
    # HAS RUN THE CHECK. What a validation would look like is notes/lunge-rule-validation.md, and
    # it is scoped in TODO.md. Three things bound it in advance: `rule_lr_asymmetry` is
    # unvalidatable on Ex1 because Ex1 is unilateral on 178/178 reps (its docstring),
    # `rule_shoulder_shrug` is silent so there is nothing to validate, and
    # `rule_contralateral_trunk_lean` is the one rule Ex1 could genuinely speak to -- where its
    # cue already scores a per-subject median AUC of 0.800 on the marker 3-D while the shipped
    # threshold fires on 0/178. Beta is the factual label until the replay harness exists.
    rep_signal="avg_arm_elevation_deg",
    # `max`, matching Overhead Press and Band Pull Apart and inverse to Row/Bicep Curl: this
    # movement's excursion is arms-down -> raised -> down, so the rep peaks at the signal's
    # MAXIMUM.
    rep_polarity="max",
    # `extended` -- the rep opens away from the peak, with the arms at the sides. Only Deadlift
    # uses `flexed` (it starts AT the extremum, bar on the floor).
    rep_start="extended",
    # `avg`, not an extremum of the two arms, and the choice was MEASURED rather than preferred:
    # on Fit3D `side_lateral_raise` left and right arm elevation correlate r = 0.9896-0.9964
    # across all 8 subjects, so the arms are in phase and the mean is the same excursion with
    # per-arm landmark noise halved. Per-clip excursion of the mean is 61.7-109.4 deg against the
    # `max`-of-both alternative's 64.6-109.7, i.e. no excursion advantage to offset the noise
    # cost. STATED LIMITATION: on the unilateral variant the resting arm contributes a
    # near-constant ~27 deg, so the mean still excurses monotonically at roughly half amplitude
    # -- degraded signal-to-noise, not a cancelled signal (contrast Bicep Curl, where alternating
    # curls cancel outright). Not corrected: choosing `max` to serve a variant this app does not
    # model, at a measured cost on the one it does, trades a measured decision for a guessed one.
    #
    # `min_rep_seconds` stays at DEFAULT_MIN_REP_SECONDS (0.4s). Measured rep durations are
    # 1.40-4.96 s (Fit3D, 40 reps, 50 fps ffprobe-verified) and 2.77-10.53 s (REHAB24-6 Ex1, 178
    # reps, 30 fps), so the tightest real rep is 3.5x the floor. The TIGHTER constraint -- the
    # phase-fraction x min_frames interaction Bicep Curl section 4.3 found -- is handled by
    # scoping no rule to `setup`; see `arm_abduction_assign_phases`.
)

registry.register(ARM_ABDUCTION_DETECTOR)
