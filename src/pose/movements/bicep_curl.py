# Bicep Curl (standing, a dumbbell in each hand) raw metrics, phase segmentation and fault rules.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `bicep_curl_compute_raw` /
# `bicep_curl_assign_phases` compute per-frame quantities and a phase label only. Every number
# that decides anything belongs in a `rule_*` function. The only constant this module defines
# outside a rule, `_DEGENERATE_LENGTH`, is a division-by-zero guard, never a tunable threshold.
#
# ---------------------------------------------------------------------------------------
# THREE RULES SHIP AND THE PARENT SPEC'S FOURTH IS ABSENT, NOT SILENT.
# ---------------------------------------------------------------------------------------
# `wrist_flexion_curl` is WITHDRAWN from the parent spec (2026-08-09) and has no function here.
# This project has two treatments for a rule it will not fire, and they are not
# interchangeable: registered-but-permanently-silent (pushup.rule_scapular_winging,
# band_pull_apart.rule_loss_of_scapular_retraction) says "real, well-cited fault, the sensor
# cannot see it"; withdrawn (OHP bar-path 2026-07-25, deadlift bar-drift 2026-08-01) says "no
# citation supports the rule as written". Parpa PMC12550948 was read in full: every wrist
# statement in it is about forearm ROTATION (supination/pronation) or grip type, never flexion.
# The parent spec's wrist-strain mechanism and its 30-degree threshold appear nowhere in the
# source. That is a citation failure, so the rule is ABSENT -- adding a silent stub here would
# assert the opposite diagnosis. Design spec section 3.
#
# ---------------------------------------------------------------------------------------
# EVERY CITATION IN THIS MODULE IS A PROTOCOL QUOTE, NOT A FAULT FINDING.
# ---------------------------------------------------------------------------------------
# Parpa K et al. (2025) is an EMG comparison of dumbbell vs Bayesian cable curls. It backs
# these rules only through its PROPER-EXECUTION PROTOCOL, which defines correct form and
# therefore defines deviation from it; it never studies any of these faults AS faults. That is
# weaker support than Fukunaga gives Band Pull Apart or Ford gives the squat. Each rule's
# docstring says so at its own site rather than leaving it to be discovered.
#
# ---------------------------------------------------------------------------------------
# ONE DROPPED LANDMARK SILENCES EVERY BICEP CURL RULE FOR THAT FRAME.
# ---------------------------------------------------------------------------------------
# `required` below lists both shoulders, both elbows, both wrists and both hips. If
# `visible_point` drops any ONE of them the frame is marked `valid=False` and carries no metric
# keys at all, so every rule that masks on `frame.valid` goes silent for that frame, not just
# the one whose input landmark went missing. This mirrors `pushup_compute_raw`,
# `ohp_compute_raw`, `lunge_compute_raw`, `row_compute_raw` and `band_pull_apart_compute_raw`:
# an unmeasurable frame is refused wholesale rather than degraded, because a silently-wrong
# verdict is worse than no verdict.
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

# Defined locally, matching row.py, overhead_press.py and band_pull_apart.py: geometry.py
# exports only the lower-body and shoulder/hip constants.
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

# The generic "lower body" set every movement module uses for the framework-level
# `lower_body_visibility` quality field. The name is squat-centric and carries awkwardly for a
# standing arm-isolation exercise, exactly as it does for OHP, push-up, Row and Band Pull Apart;
# this module's own rules never consume it.
LOWER_BODY_LANDMARKS = (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

BICEP_CURL_METRIC_KEYS: tuple[str, ...] = (
    "left_elbow_angle",
    "right_elbow_angle",
    "avg_elbow_angle",
    "min_elbow_angle",
    "max_elbow_angle",
    "left_upper_arm_lean_deg",
    "right_upper_arm_lean_deg",
    "max_upper_arm_lean_deg",
    "trunk_lean_image_signed_deg",
    "shoulder_width",
    "upper_arm_length",
)

# Below this a length/normalizer is degenerate and the dependent metric is NaN. Same guard value
# pushup.py, overhead_press.py, lunge.py, row.py and band_pull_apart.py use; not a tunable
# threshold.
_DEGENERATE_LENGTH = 1e-6


def bicep_curl_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
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
            continue

        # `angle_degrees` consumes dims=3, i.e. MediaPipe's estimated z as well as x/y. That is
        # the shared helper every movement module uses and is not changed here, but it means
        # these angles are NOT pure image-plane projections under the MediaPipe path -- they
        # carry whatever weak depth that estimator supplies. Under the RTMPose extraction path
        # (src/pose/rtmpose_pose_extraction.py writes z=0.0 for every landmark) they ARE pure
        # image-plane projections. Design spec section 2 states the consequence for
        # `rule_incomplete_rom`'s two thresholds, which differs between the two paths.
        left_elbow_angle = angle_degrees(points, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        right_elbow_angle = angle_degrees(points, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        finite_elbows = [v for v in (left_elbow_angle, right_elbow_angle) if np.isfinite(v)]
        avg_elbow_angle = float(np.mean(finite_elbows)) if finite_elbows else np.nan
        min_elbow_angle = float(min(finite_elbows)) if finite_elbows else np.nan
        max_elbow_angle = float(max(finite_elbows)) if finite_elbows else np.nan

        # UNSIGNED angle of the upper arm from image-vertical-down. `line_angle_from_vertical`
        # takes abs() of both components, so a shoulder-to-elbow segment hanging straight down
        # reads 0 regardless of which side it deviates toward. That is the intended reading, not
        # a limitation of the helper: the parent spec's "toward the anterior side" qualifier is
        # deliberately dropped (design spec section 4.8) because Parpa's protocol says "elbows
        # kept close to the torso" with no direction, and recovering "anterior" would need a
        # facing proxy whose threshold no cited source supplies.
        left_shoulder = visible_point(points, LEFT_SHOULDER, dims=2)
        right_shoulder = visible_point(points, RIGHT_SHOULDER, dims=2)
        left_elbow = visible_point(points, LEFT_ELBOW, dims=2)
        right_elbow = visible_point(points, RIGHT_ELBOW, dims=2)
        left_upper_arm_lean = line_angle_from_vertical(left_shoulder, left_elbow)
        right_upper_arm_lean = line_angle_from_vertical(right_shoulder, right_elbow)
        finite_leans = [v for v in (left_upper_arm_lean, right_upper_arm_lean) if np.isfinite(v)]
        max_upper_arm_lean = float(max(finite_leans)) if finite_leans else np.nan

        shoulder_width = distance(points, LEFT_SHOULDER, RIGHT_SHOULDER)
        left_upper_arm = distance(points, LEFT_SHOULDER, LEFT_ELBOW)
        right_upper_arm = distance(points, RIGHT_SHOULDER, RIGHT_ELBOW)
        finite_arms = [v for v in (left_upper_arm, right_upper_arm) if np.isfinite(v)]
        # DIAGNOSTIC ONLY -- no rule reads this. It is emitted so that the parent spec's
        # withdrawn second drift cue ("elbow displacement > 0.5 x upper_arm_length") stays
        # CHECKABLE without re-deriving the arithmetic that shows it unreachable: displacement =
        # upper_arm_length * sin(lean), so its 0.5 threshold IS lean > 30 degrees, strictly
        # inside the 25-degree angular threshold `rule_elbow_drift_forward` already applies.
        # Design spec section 4.9; pinned by test_the_displacement_disjunct_is_unreachable.
        upper_arm_length = float(np.mean(finite_arms)) if finite_arms else np.nan

        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        # SIGNED, and deliberately NOT facing-corrected. Positive = the shoulders sit toward +x
        # relative to the hips, in IMAGE coordinates. Which physical direction that is depends on
        # which way the lifter faces, which this layer cannot know and must not guess. Unlike
        # Band Pull Apart -- which pins facing from a wrist-depth sign, available only because
        # that movement holds the band in front of the torso by definition -- a curl's wrists
        # travel from hip height to shoulder height and their depth offset changes sign WITHIN
        # the rep, so that construction does not transfer. `rule_trunk_swing_momentum` therefore
        # reduces this signed series to facing-free quantities (a range, and an absolute
        # deviation) rather than reading its sign. Design spec section 4.8.
        if shoulder_mid is not None and hip_mid is not None:
            trunk_lean = float(
                np.degrees(
                    np.arctan2(
                        float(shoulder_mid[0] - hip_mid[0]),
                        float(hip_mid[1] - shoulder_mid[1]),
                    )
                )
            )
        else:
            trunk_lean = np.nan

        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "left_elbow_angle": left_elbow_angle,
                "right_elbow_angle": right_elbow_angle,
                "avg_elbow_angle": avg_elbow_angle,
                "min_elbow_angle": min_elbow_angle,
                "max_elbow_angle": max_elbow_angle,
                "left_upper_arm_lean_deg": left_upper_arm_lean,
                "right_upper_arm_lean_deg": right_upper_arm_lean,
                "max_upper_arm_lean_deg": max_upper_arm_lean,
                "trunk_lean_image_signed_deg": trunk_lean,
                "shoulder_width": shoulder_width if shoulder_width > _DEGENERATE_LENGTH else np.nan,
                "upper_arm_length": upper_arm_length,
            }
        )

    return raw


def bicep_curl_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> concentric -> peak -> eccentric, segmented on `avg_elbow_angle`.

    Mirrors `band_pull_apart_assign_phases` with the polarity inverted: that movement's peak is
    the WIDEST 30% of the rep, this one's is the MOST-FLEXED 30%, i.e. the 30th percentile of
    the elbow angle and below. Same fallbacks: an empty clip returns an empty list, a clip with
    no finite signal is entirely `unknown`, and an invalid frame is `unknown` regardless of
    where it sits (the validity check precedes the setup cutoff, so an occluded frame in the
    opening 15% is NOT labelled `setup`, which matters because `_setup_baseline` reduces over
    exactly those frames).

    `setup` IS THE ARMS-EXTENDED END OF THE REP, and that is what makes it the right scope for
    `rule_incomplete_rom`'s extension term. `rep_start="extended"` means `segment_reps` opens
    each window at the bottom of the curl, so the first 15% of the window is the hanging-arm
    position. A lifter who fails to LOWER all the way finishes rep N short -- and because reps
    are contiguous, that same short position IS rep N+1's `setup`, where the rule catches it.
    The only rep this misses is the last one in the clip. Not corrected: the alternative is
    scoping the extension term over `eccentric` too, which spans the whole mid-range where a
    90-degree elbow is CORRECT, and would fire on every rep.
    """
    frame_count = len(raw)
    if frame_count == 0:
        return []

    elbow_values = np.asarray(
        [float(item.get("avg_elbow_angle", np.nan)) for item in raw], dtype=np.float32
    )
    valid_elbow = elbow_values[np.isfinite(elbow_values)]
    if valid_elbow.size == 0:
        return ["unknown" for _ in raw]

    # The most-flexed 30% of the rep is the peak hold.
    peak_threshold = float(np.percentile(valid_elbow, 30))
    flexed_index = int(np.nanargmin(np.where(np.isfinite(elbow_values), elbow_values, np.inf)))
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
        elif index < flexed_index:
            phases.append("concentric")
        else:
            phases.append("eccentric")
    return phases


# ---------------------------------------------------------------------------------------
# STEP 0 -- KG QUERY RESOLUTION, recorded before any rule was written. Each string below was
# checked against data/kg/sports_kg_v3.graphml with `retrieve_graph_context(query,
# movement="Bicep Curl")` -- the function PRODUCTION calls, not just `resolve_nodes`. Observed
# results, not predicted ones:
#
#   "Elbow Drift Forward"        -> Bicep Curl:Elbow Drift Forward
#       NO buckets -- only the HAS_FAULT backlink                                      THIN
#   "Using Momentum"             -> Bicep Curl:Using Momentum
#       quality_impacts: Forward Momentum                                              NON-EMPTY
#   "Incomplete Range Of Motion" -> Bicep Curl:Incomplete Range Of Motion
#       quality_impacts: Range Of Motion                                               NON-EMPTY
#
# The one gap is recorded rather than masked. Pointing rule 1 at the shared `Range Of Motion`
# QualityDimension WOULD return a rich bucket set, and was rejected for the same reason Band
# Pull Apart rejected it: that node's `corrections` bucket is "Wrapping Surface Adjustment",
# meaningless for this movement. A semantically correct thin card beats a semantically wrong
# full one. The gap is a one-line fix in scripts/knowledge/stub_general_movements_v3.py:71-78
# (the node's list is `[]` there) and is logged against TODO.md's existing "many faults have no
# KG node" item; the graphml is gitignored, so regenerating it is a deploy step.
CURL_DRIFT_KG_QUERY = "Elbow Drift Forward"
CURL_MOMENTUM_KG_QUERY = "Using Momentum"
CURL_ROM_KG_QUERY = "Incomplete Range Of Motion"

# Imported rather than re-typed, so a change to the shared constant cannot silently skip this
# module.
_OFF_VIEW_CONFIDENCE = VIEW_UNAVAILABLE_CONFIDENCE_SCALE

# HARD GATE, WRITTEN IN THE NEGATIVE, shared by the two SAGITTAL rules (drift, trunk swing).
#
# WHY A GATE AT ALL, when Row's design doc argues "downgrade, never gate": both rules measure a
# sagittal quantity. From a pure `front`/`rear` view the sagittal axis is perpendicular to the
# image plane, so `max_upper_arm_lean_deg` computed there reads LATERAL ELBOW FLARE and
# `trunk_lean_image_signed_deg` reads FRONTAL-PLANE SWAY -- different faults, or none. That is
# not a low-confidence reading of the right quantity (the case the x0.65 discount exists for);
# it is a confident reading of the WRONG PLANE. Row's objection was that gated rules ship
# silent, and that does not apply: the gate leaves `rear_oblique` standing, which is 37 of the
# 49 real pose JSONs under data/runtime/pose_json (re-measured 2026-08-09 for this movement:
# rear_oblique 37, rear 9, unknown 3, side NEVER).
#
# WHY NEGATIVE rather than a {side, rear_oblique, front_oblique} whitelist: `front_oblique` is
# unreachable under `allow_front=False` (src/pose/view_estimation.py:14-16), so a whitelist
# containing it is dead weight that READS as coverage, and `unknown` -- 3 of the 49 -- would be
# excluded by a whitelist without anyone having decided to exclude it. The negative form needs
# no edit if `allow_front` is ever enabled.
#
# THE UNCOMFORTABLE PART, STATED RATHER THAN HIDDEN: `side` -- the view the parent spec rates
# `high` for all three of this movement's rules -- does not occur in production at all. No rule
# here will earn its spec-rated observability on a real clip. What this gate protects is
# narrower than it looks: that a rule reads the right PLANE, not that it reads it well. On the
# obliques that survive, the sagittal axis is foreshortened, so a real drift or lean reads
# SMALLER than it is -- a missed fault, never a false one.
SAGITTAL_BLIND_VIEWS = {"front", "rear"}

# FROM THE SPEC: "Flag if `upper_arm_lean > 25deg` ... at any frame during concentric."
DRIFT_MILD_DEG = 25.0
# RULE-LEVEL CHOICE MADE HERE. The parent spec states NO severity ramp for ANY Bicep Curl fault
# (the Lunge section states its ramps explicitly, so the absence is meaningful). 62.5 is 2.5x
# the fire threshold, the convention `pushup.rule_hip_sag` uses for exactly this situation. A
# display/ranking curve, not a cited quantity.
DRIFT_SEVERE_DEG = 62.5


def _setup_baseline(core: list[CoreFrame], key: str) -> float:
    """Median of `key` over this window's valid `setup` frames; NaN when there are none.

    WHY THE BASELINE LIVES IN THE RULES AND NOT IN `bicep_curl_compute_raw`. A baseline is a
    PER-REP reduction. `run_detector` calls `compute_raw` over the WHOLE CLIP before
    `segment_reps`, so at metric time no rep boundary exists and there is no "this rep's setup"
    to reduce against. Rules receive a per-rep slice, which is the first place the question is
    answerable.

    MEDIAN, NOT MEAN, so one bad frame in a short setup cannot move the reference every later
    comparison is made against.

    NEVER A GUESSED BASELINE. Only `rule_trunk_swing_momentum`'s SECOND term consults this; its
    first term (a within-rep range) and both of `rule_incomplete_rom`'s terms and
    `rule_elbow_drift_forward` are absolute readings, so a NaN baseline silences one term and
    nothing else. Contrast `band_pull_apart._setup_baseline`, where two of three firing rules
    depend on it wholesale.

    STATED LIMITATION, inherited from `row._setup_baseline` where it was measured: `setup` is
    the first 15% of the REP WINDOW, and the window has already been trimmed by `segment_reps`
    to the rep's excursion -- so on a short rep `setup` can be 1-2 frames and may already
    overlap loaded frames. Because the comparison here is `frame - baseline`, a baseline biased
    toward the loaded state makes the MEASURED change smaller than the true one: the failure
    mode is a MISSED fault, never a false one. Not corrected -- there is no principled way to
    detect "this setup frame is already loaded" without a second threshold the parent spec does
    not supply.
    """
    values = [
        frame.m(key)
        for frame in core
        if frame.valid and frame.phase == "setup" and np.isfinite(frame.m(key))
    ]
    if not values:
        return float(np.nan)
    return float(np.median(values))


def rule_elbow_drift_forward(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag the upper arm leaving its vertical hang -- the elbow stops being a fixed pivot.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM.
      FIRE THRESHOLD 25 deg: FROM THE SPEC.
      SEVERITY RAMP 25 -> 62.5 deg: A RULE-LEVEL CHOICE (see DRIFT_SEVERE_DEG).

    PHASE SCOPE `concentric`, FROM THE SPEC's own wording ("at any frame during concentric").
    `peak` is deliberately NOT added even though drift is largest there: widening a phase scope
    is a rule-level change that would make this rule fire on reps the spec's wording exempts,
    and no citation distinguishes the two scopes.

    UNSIGNED, NOT "ANTERIOR". The parent spec qualifies the drift as "toward the anterior
    (wrist) side"; that qualifier is dropped and the metric taken unsigned. Parpa's protocol is
    "the elbows kept close to the torso throughout the whole movement" -- no direction -- so an
    undirected departure is EXACTLY what the source prescribes against, while a signed one
    asserts more than the source does. Recovering "anterior" would need a facing proxy whose
    threshold no cited source supplies, which is the construct the OHP bar-path and deadlift
    bar-drift withdrawals both rejected. The cost is that a BACKWARD drift (the drag-curl
    position) also fires -- a wider net than the spec describes, in the direction the citation
    supports. Design spec section 4.8.

    THE PARENT SPEC'S SECOND CUE IS NOT IMPLEMENTED BECAUSE IT IS UNREACHABLE, not because it
    was overlooked. "elbow x-displacement ... exceeds 0.5 x upper_arm_length" is this same cue
    in different units: displacement = upper_arm_length * sin(lean), so 0.5 IS lean > 30 deg,
    strictly inside the 25 deg applied here. Any frame it would catch, this one has caught. Same
    defect `row.rule_momentum_jerk`'s second condition had -- a strict subset of its first, and
    therefore dead code that READ as coverage. `upper_arm_length` is still emitted so the
    equivalence stays checkable. Design spec section 4.9.

    THE WORSE ARM, and that is a RULE-LEVEL reading of a spec line that names no side. A rep is
    faulty if EITHER elbow drifts, so `max_upper_arm_lean_deg` is the conservative choice --
    deliberately the OPPOSITE of `rule_incomplete_rom` below, which takes the generous reading
    for a reason stated at its own constants.

    CITATION IS A PROTOCOL QUOTE. Parpa defines correct execution and had investigators monitor
    it; the paper does not study elbow drift as a fault.
    """
    if ctx.view_type in SAGITTAL_BLIND_VIEWS:
        return []
    scale = 1.0 if ctx.view_type == "side" else _OFF_VIEW_CONFIDENCE

    def lean(frame: CoreFrame) -> float:
        return frame.m("max_upper_arm_lean_deg")

    mask = [
        frame.valid
        and frame.phase == "concentric"
        and np.isfinite(lean(frame))
        and lean(frame) > DRIFT_MILD_DEG
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        leans = [lean(frame) for frame in segment]
        max_lean = float(np.nanmax(leans))
        severity = severity_from_range(
            max_lean, DRIFT_MILD_DEG, DRIFT_SEVERE_DEG, lower_is_worse=False
        )
        detections.append(
            build_detection(
                fault_id="curl_elbow_drift_forward",
                fault_name="Elbow Drift (Loss of Elbow Fixation)",
                kg_query=CURL_DRIFT_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=leans,
                severity=severity,
                confidence=severity * scale,
                observability="high" if ctx.view_type == "side" else "medium",
                evidence={
                    "max_upper_arm_lean_deg": round(max_lean, 2),
                    "threshold_deg": DRIFT_MILD_DEG,
                    "primary_label": "upper-arm lean from vertical",
                    "primary_value": round(max_lean, 2),
                    "primary_threshold": DRIFT_MILD_DEG,
                },
                citation="Parpa K et al., Muscles (2025), PMC12550948, DOI 10.3390/muscles4040045.",
                citation_support=(
                    "The paper's validated proper-execution protocol states the arms were "
                    "\"fully extended at the sides, with the elbows kept close to the torso "
                    "throughout the whole movement,\" with two investigators visually monitoring "
                    "execution — i.e., the elbow staying fixed at the torso is the defined "
                    "correct form, so drift away from it is a deviation from that form. This is "
                    "a PROTOCOL definition of correct execution, not a study of elbow drift as a "
                    "fault."
                ),
            )
        )
    return detections


# FROM THE SPEC: "Flag if within-rep oscillation `max(torso_lean_deg) - min(torso_lean_deg) >
# 12deg`".
SWING_RANGE_MILD_DEG = 12.0
# RULE-LEVEL CHOICE MADE HERE. 2.5x the fire threshold, the `pushup.rule_hip_sag` convention.
SWING_RANGE_SEVERE_DEG = 30.0

# FROM THE SPEC: "or backward lean during concentric exceeds the setup baseline by `> 10deg`".
# Taken as an ABSOLUTE deviation rather than a signed backward one, for the reason
# `rule_elbow_drift_forward` documents at length and design spec section 4.8 states: Parpa's
# protocol is "avoiding trunk movements and jerky motions" with no direction attached.
SWING_BASELINE_MILD_DEG = 10.0
# RULE-LEVEL CHOICE MADE HERE. 2.5x the fire threshold, same convention.
SWING_BASELINE_SEVERE_DEG = 25.0


def rule_trunk_swing_momentum(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag the trunk swinging or leaning to heave the load instead of isolating the elbow.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM.
      FIRE THRESHOLDS 12 deg (within-rep range) and 10 deg (deviation from setup): FROM THE SPEC.
      SEVERITY RAMPS 12 -> 30 and 10 -> 25 deg: RULE-LEVEL CHOICES.

    A GENUINE DISJUNCTION, and the check that it is one was run rather than assumed -- unlike
    `row.rule_momentum_jerk`'s second condition, which turned out to be a strict SUBSET of its
    first and therefore unreachable. Neither term here nests inside the other: a rep that leans
    11 degrees one way and holds fires (b) but not (a); a rep that oscillates +/-7 degrees about
    its baseline fires (a) but not (b). Both are real failure modes and the coaching cue differs
    ("stop rocking" vs "stand up straight and stay there").

    TERM (a) IS A PER-REP REDUCTION AND TERM (b) IS PER-FRAME, which is why the mask below
    carries a rep-level scalar into a per-frame expression. When (a) alone fires it marks every
    `concentric` frame -- the fault is "this rep used trunk swing" and the concentric is where
    the swing is spent -- so the reported span is the concentric, which is the honest answer to
    "when did this happen".

    `score_values` IS TERM (b)'s PER-FRAME DEVIATION IN BOTH CASES, and that is deliberate.
    `build_detection` nominates `peak_frame` by `nanargmax` over this series; term (a) is a
    single scalar identical on every frame, so feeding it in would tie every frame and nominate
    the FIRST one -- the trap `row.rule_incomplete_rom`'s docstring documents hitting with a
    clipped series. The frame of largest trunk deviation is the right frame to show a user under
    either term. When the setup baseline is NaN (no valid setup frames, so term (b) cannot fire
    at all) the deviation is measured against the rep's MEDIAN lean instead, purely so a term-(a)
    detection can still nominate a frame; that fallback reference decides nothing about whether
    the rule fires and is not a threshold.

    PHASE SCOPE `concentric` for term (b), FROM THE SPEC ("during concentric"). Term (a)'s range
    is reduced over the whole rep, also from the spec ("within-rep oscillation").

    FACING-FREE BY CONSTRUCTION. `trunk_lean_image_signed_deg` is signed in IMAGE coordinates,
    and neither term reads its sign -- (a) is a range and (b) an absolute deviation. Both are
    invariant to which way the lifter faces, which is why this rule needs no facing proxy even
    though the underlying metric is signed.

    CITATION IS A PROTOCOL QUOTE. Parpa excluded trunk movement as a compensation; the paper
    does not study trunk swing as a fault.
    """
    if ctx.view_type in SAGITTAL_BLIND_VIEWS:
        return []
    scale = 1.0 if ctx.view_type == "side" else _OFF_VIEW_CONFIDENCE

    leans = [
        frame.m("trunk_lean_image_signed_deg")
        for frame in core
        if frame.valid and np.isfinite(frame.m("trunk_lean_image_signed_deg"))
    ]
    swing_range = float(max(leans) - min(leans)) if len(leans) >= 2 else float(np.nan)
    range_severity = (
        severity_from_range(
            swing_range, SWING_RANGE_MILD_DEG, SWING_RANGE_SEVERE_DEG, lower_is_worse=False
        )
        if np.isfinite(swing_range) and swing_range > SWING_RANGE_MILD_DEG
        else 0.0
    )

    baseline = _setup_baseline(core, "trunk_lean_image_signed_deg")
    # Display-only reference (see docstring); never consulted by a fire condition.
    display_reference = baseline if np.isfinite(baseline) else (
        float(np.median(leans)) if leans else float(np.nan)
    )

    def deviation(frame: CoreFrame) -> float:
        value = frame.m("trunk_lean_image_signed_deg")
        if not np.isfinite(value) or not np.isfinite(display_reference):
            return float(np.nan)
        return abs(value - display_reference)

    def baseline_severity(frame: CoreFrame) -> float:
        if not np.isfinite(baseline):
            return 0.0
        value = frame.m("trunk_lean_image_signed_deg")
        if not np.isfinite(value):
            return 0.0
        offset = abs(value - baseline)
        return (
            severity_from_range(
                offset, SWING_BASELINE_MILD_DEG, SWING_BASELINE_SEVERE_DEG, lower_is_worse=False
            )
            if offset > SWING_BASELINE_MILD_DEG
            else 0.0
        )

    mask = [
        frame.valid
        and frame.phase == "concentric"
        and (range_severity > 0.0 or baseline_severity(frame) > 0.0)
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        worst_baseline = max((baseline_severity(frame) for frame in segment), default=0.0)
        severity = max(range_severity, worst_baseline)
        # The display axis is whichever term drove the SEVERITY, compared directly rather than
        # by branching on which term(s) crossed their fire threshold -- the precise mistake
        # `row.rule_incomplete_rom`'s docstring records, where a "both fired" segment reported
        # the wrong axis as primary. On an exact tie the range axis wins, an arbitrary but fixed
        # choice matching that rule's own tie-break direction for its first axis.
        range_drove = range_severity >= worst_baseline
        max_deviation = float(np.nanmax([deviation(frame) for frame in segment]))
        detections.append(
            build_detection(
                fault_id="curl_trunk_swing_momentum",
                fault_name="Trunk Swing / Momentum",
                kg_query=CURL_MOMENTUM_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=[deviation(frame) for frame in segment],
                severity=severity,
                confidence=severity * scale,
                observability="high" if ctx.view_type == "side" else "medium",
                evidence={
                    "within_rep_swing_range_deg": (
                        round(swing_range, 2) if np.isfinite(swing_range) else None
                    ),
                    "swing_range_threshold_deg": SWING_RANGE_MILD_DEG,
                    "setup_baseline_lean_deg": (
                        round(baseline, 2) if np.isfinite(baseline) else None
                    ),
                    "max_deviation_from_setup_deg": (
                        round(max_deviation, 2) if np.isfinite(max_deviation) else None
                    ),
                    "baseline_threshold_deg": SWING_BASELINE_MILD_DEG,
                    "primary_label": (
                        "within-rep trunk swing range" if range_drove
                        else "trunk lean vs setup baseline"
                    ),
                    "primary_value": (
                        round(swing_range, 2) if range_drove and np.isfinite(swing_range)
                        else (round(max_deviation, 2) if np.isfinite(max_deviation) else None)
                    ),
                    "primary_threshold": (
                        SWING_RANGE_MILD_DEG if range_drove else SWING_BASELINE_MILD_DEG
                    ),
                },
                citation="Parpa K et al., Muscles (2025), PMC12550948, DOI 10.3390/muscles4040045.",
                citation_support=(
                    "Participants performed the curl \"avoiding trunk movements and jerky "
                    "motions,\" and \"two experienced investigators visually monitored trunk "
                    "movements and knee flexion to ensure the proper execution\" — trunk "
                    "movement is explicitly treated as a cheating/compensation deviation to be "
                    "excluded. This is a PROTOCOL exclusion criterion, not a study of trunk "
                    "swing as a fault."
                ),
            )
        )
    return detections


# FROM THE SPEC: "flag incomplete extension if `max(elbow_angle)` over the rep `< 150deg`".
EXTENSION_MILD_DEG = 150.0
# RULE-LEVEL CHOICE MADE HERE. 40 degrees of ramp width, taken from `pushup.rule_shallow_depth`
# (100 -> 140) and matched by `band_pull_apart`'s elbow ramp (150 -> 110), so the three elbow
# ramps in this codebase cannot drift apart. DESCENDING: severity grows as the arm stays more
# bent at the bottom.
EXTENSION_SEVERE_DEG = 110.0

# FROM THE SPEC: "flag incomplete flexion if `min(elbow_angle) > 60deg`".
FLEXION_MILD_DEG = 60.0
# RULE-LEVEL CHOICE MADE HERE. The same 40-degree ramp width, ASCENDING: severity grows as the
# arm closes less at the top.
FLEXION_SEVERE_DEG = 100.0


def rule_incomplete_rom(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag a partial curl -- the elbow never straightens at the bottom, or never closes at the top.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM.
      FIRE THRESHOLDS 150 deg (extension) and 60 deg (flexion): FROM THE SPEC.
      SEVERITY RAMPS 150 -> 110 and 60 -> 100 deg: RULE-LEVEL CHOICES.

    THE FIRST RULE IN THIS CODEBASE WHOSE TWO TERMS LIVE IN DIFFERENT PHASES. Row's and Band
    Pull Apart's ROM cues both sit at the peak; a curl's two ROM failures sit at OPPOSITE ends
    of the rep by definition -- extension at the bottom (`setup`), flexion at the top (`peak`).
    One mask with a phase-conditional score function keeps a single `contiguous_true_segments`
    pass, and the two phases are disjoint and non-adjacent (`concentric` separates them), so no
    segment can span both terms and no detection can mix their evidence.

    EACH TERM READS THE ARM THAT MAKES IT HARDER TO FIRE, and that is a RULE-LEVEL reading of a
    spec line naming no side. Extension reads `max_elbow_angle` (the LESS-flexed, i.e.
    straighter arm) and flexion reads `min_elbow_angle` (the MORE-flexed arm): the rep is called
    incomplete only if BOTH arms fell short at that end. This is deliberately the opposite of
    `row.rule_incomplete_rom`'s conservative reading. The reason is measured, not stylistic --
    design spec section 2 found that on 40 Fit3D reps of 3D mocap ground truth both of these
    thresholds sit within ~1 degree of the edge of the real-rep distribution (1/40 reps below
    150, 0/40 above 60, worst 59.0). Pairing an already-sensitive threshold with a conservative
    side-selection would compound two independent pushes toward false firing.

    THE THRESHOLDS ARE SENSITIVE AND THAT IS RECORDED, NOT REPAIRED. No number here is moved;
    the finding above is the honest characterisation a future validation should start from. The
    Fit3D evidence is ambiguous by construction -- that dataset carries no correctness label, so
    the 149.7-degree rep may genuinely have been a short one.

    PROJECTION ERROR IS DIRECTIONAL, AND OPPOSITE FOR THE TWO TERMS. Under the RTMPose
    extraction path (z == 0.0 for every landmark) these angles are pure image-plane
    projections; under MediaPipe they carry weak estimated depth. Where projection dominates, an
    oblique bend plane AMPLIFIES the apparent bend: a true 160 degrees can read lower, pushing
    the extension term TOWARD firing on a threshold already shown to be sensitive, while the
    same amplification reads a flexed arm as MORE flexed, pushing the flexion term toward
    MISSING faults. Perfectly collinear points project collinearly, so a genuinely straight arm
    still reads ~180 from any view and the extension term is safe at its limit.

    THE EXTENSION TERM IS FRAGILE, AND THE MEASUREMENT IS RECORDED RATHER THAN REPAIRED. Two
    framework interactions narrow the `setup` window it fires in:

      1. `setup` is 15% of the rep window and `contiguous_true_segments` needs
         `min_frames = max(3, ceil(0.20 * fps))`, so the term needs `0.15 * fps * T >= 0.20 *
         fps`, i.e. **T >= 1.333 s per rep** -- fps-independent above 15 fps. Measured Fit3D
         cadence is 1.92-3.68 s/rep, so the fastest real rep sits at 1.44x this floor. That is
         a TIGHTER constraint than DEFAULT_MIN_REP_SECONDS (0.4 s), which is what the design
         spec's cadence figure was checked against.
      2. Worse, `segment_reps` trims each window to the signal's EXCURSION. A lifter who pauses
         with the arms extended between reps has that hold cut away, so `setup` covers
         mid-range frames rather than the bottom -- and the trimmed window is shorter, which
         can push `setup` back under `min_frames` on its own. Measured on a 63-frame-per-rep
         fixture with a between-reps hold: windows came out 37 frames, `setup` 5 frames (one
         short of 6), and the frames it covered read 84-110 degrees instead of the true
         130-degree bottom. Pinned by
         `tests/test_bicep_curl.py::EndToEndSegmentationTest::test_rep_trimming_can_silence_the_extension_term`.

    So whether this term reports depends on the SHAPE of the rep, not only its duration. NOT
    repaired: the 15% setup fraction and `min_frames` are shared framework constants, and
    neither has a cited basis to move for one movement. The failure mode is a MISSED fault,
    never a false one. `_setup_baseline`'s docstring records the same trimming caveat for the
    baseline rules, where it degrades more gently.

    DOWNGRADE, NOT GATE, unlike the two sagittal rules above. An elbow angle is the RIGHT
    quantity from every view; obliquity makes it noisier, not different in kind. So this rule
    fires everywhere with the standard discount off `side` -- matching
    `band_pull_apart.rule_incomplete_rom`.

    CITATION: Havers supplies the full-ROM strength finding, Parpa the protocol prescription.
    """
    observable = ctx.view_type == "side"
    scale = 1.0 if observable else _OFF_VIEW_CONFIDENCE

    def scores(frame: CoreFrame) -> tuple[float, float]:
        """(extension severity, flexion severity) for one frame; 0.0 where the term is out of
        phase or does not fire."""
        if frame.phase == "setup":
            value = frame.m("max_elbow_angle")
            extension = (
                severity_from_range(
                    value, EXTENSION_MILD_DEG, EXTENSION_SEVERE_DEG, lower_is_worse=True
                )
                if np.isfinite(value) and value < EXTENSION_MILD_DEG
                else 0.0
            )
            return extension, 0.0
        if frame.phase == "peak":
            value = frame.m("min_elbow_angle")
            flexion = (
                severity_from_range(
                    value, FLEXION_MILD_DEG, FLEXION_SEVERE_DEG, lower_is_worse=False
                )
                if np.isfinite(value) and value > FLEXION_MILD_DEG
                else 0.0
            )
            return 0.0, flexion
        return 0.0, 0.0

    mask = [frame.valid and max(scores(frame)) > 0.0 for frame in core]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        pairs = [scores(frame) for frame in segment]
        combined = [max(pair) for pair in pairs]
        severity = float(np.nanmax(combined))
        worst = int(np.nanargmax(combined))
        # A segment lies entirely inside ONE phase (see docstring), so exactly one term can be
        # non-zero across it and this comparison cannot mix two axes in different units the way
        # `row.rule_incomplete_rom`'s does.
        extension_drove = pairs[worst][0] >= pairs[worst][1]
        if extension_drove:
            reported = float(np.nanmin([frame.m("max_elbow_angle") for frame in segment]))
            label, threshold = "elbow extension at the bottom", EXTENSION_MILD_DEG
        else:
            reported = float(np.nanmax([frame.m("min_elbow_angle") for frame in segment]))
            label, threshold = "elbow flexion at the top", FLEXION_MILD_DEG
        detections.append(
            build_detection(
                fault_id="curl_incomplete_rom",
                fault_name="Incomplete Range of Motion (Partial Curl)",
                kg_query=CURL_ROM_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=combined,
                severity=severity,
                confidence=severity * scale,
                observability="high" if observable else "medium",
                evidence={
                    "worst_elbow_angle_deg": round(reported, 2),
                    "extension_threshold_deg": EXTENSION_MILD_DEG,
                    "flexion_threshold_deg": FLEXION_MILD_DEG,
                    "fired_on": "extension" if extension_drove else "flexion",
                    "primary_label": label,
                    "primary_value": round(reported, 2),
                    "primary_threshold": threshold,
                },
                citation=(
                    "Havers et al., European Journal of Sport Science (2025), DOI "
                    "10.1002/ejsc.70087 (PubMed 41247250); supported by Parpa K et al., Muscles "
                    "(2025), PMC12550948."
                ),
                citation_support=(
                    "Havers et al. found full ROM (0–140°) produced greater strength gains than "
                    "initial partial ROM — larger 1RM (SMD≈0.17) and greater MVC at the 100° "
                    "elbow angle (SMD≈0.24). The RAG doc (Parpa) prescribes \"a slow, controlled "
                    "lowering of the dumbbells back to the starting position through the full "
                    "range of motion.\""
                ),
            )
        )
    return detections


# THREE RULES, NOT FOUR, and the missing one is ABSENT rather than silent -- see this module's
# header. `deadlift`'s withdrawn bar-drift rule is the precedent: a citation failure leaves no
# stub behind, because a registered-but-silent stub is this codebase's way of saying "the
# citation holds, the sensor does not", which would be the wrong diagnosis here.
#
# `BICEP_CURL_METRIC_KEYS` must stay a two-way match with what `bicep_curl_compute_raw` emits
# (pinned by `test_metric_keys_match_the_emitted_metrics_exactly`): a key the tuple omits is
# dropped by `run_detector`, which builds each CoreFrame's metrics dict FROM this tuple, and
# read back as NaN by every rule.
BICEP_CURL_DETECTOR = MovementDetector(
    "Bicep Curl",
    BICEP_CURL_METRIC_KEYS,
    bicep_curl_compute_raw,
    bicep_curl_assign_phases,
    (
        rule_elbow_drift_forward,
        rule_trunk_swing_momentum,
        rule_incomplete_rom,
    ),
    # `validated` stays at its default False, and that is not a formality. REHAB24-6 holds arm
    # abduction, arm VW, table push-ups, leg abduction, lunge and squats -- no bicep curl. Fit3D
    # DOES ship `dumbbell_biceps_curls` with 3D mocap ground truth and rep boundaries, and this
    # detector's design spec used those 40 reps for a cadence measurement and a
    # does-it-fire-on-ordinary-reps check -- but Fit3D carries NO binary correct/incorrect label
    # on any rep, so no REHAB24-6-style fire-rate/AUC-against-correctness check is possible. No
    # labeled-CORRECTNESS bicep curl repetition exists anywhere in this repository, so no
    # threshold here has ever been checked against a rep judged correct or incorrect by a human.
    # Beta is the factual label.
    rep_signal="avg_elbow_angle",
    # `avg`, not the `min` Push-up/Row/Band Pull Apart use, and the choice was MEASURED rather
    # than preferred: left and right elbow angles correlate r = 0.992-0.996 across all 8 Fit3D
    # subjects (`joints3d_25`, `dumbbell_biceps_curls`), so the arms are in phase and the mean
    # is the same excursion with per-arm landmark noise halved, while `min` would inherit
    # whichever arm was noisier on each frame. Matches `overhead_press`, which averages for the
    # same reason. STATED LIMITATION: alternating curls would cancel in the mean and fall back
    # to whole-clip analysis -- degradation to the pre-existing path, not a wrong verdict.
    # Design spec section 4.2.
    rep_polarity="min",
    rep_start="extended",
    # `min_rep_seconds` stays at DEFAULT_MIN_REP_SECONDS (0.4s). Measured from Fit3D
    # `rep_ann.json` across 8 subjects / 40 reps (50 fps, ffprobe-verified): 1.92-3.68 s/rep,
    # mean 2.54 -- 4.8x to 9.2x the floor. This closes for this movement the gap Band Pull
    # Apart's Task 6 had to leave open. The residual risk is unchanged by the larger n: these
    # are subjects performing deliberately for a mocap capture, and a real user can be faster.
)

registry.register(BICEP_CURL_DETECTOR)
