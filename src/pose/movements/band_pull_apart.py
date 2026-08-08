# Band Pull Apart (standing, resistance band) raw metrics and phase segmentation. Fault rules
# land in Tasks 2-5.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `band_pull_apart_compute_raw` /
# `band_pull_apart_assign_phases` compute per-frame quantities and a phase label only. Every
# number that decides anything belongs in a `rule_*` function. The only constant this module
# defines, `_DEGENERATE_LENGTH`, is a division-by-zero guard, never a tunable threshold. This is
# also why `trunk_lean_image_signed_deg` is emitted RAW rather than facing-corrected: the facing
# derivation needs a floor, and a floor is a threshold. Rule 4 does that correction itself.
#
# ---------------------------------------------------------------------------------------
# THIS IS THE FIRST MOVEMENT WHOSE DEFINING EXCURSION IS FRONTAL, NOT SAGITTAL.
# ---------------------------------------------------------------------------------------
# Squat, Lunge, Deadlift, Push-up, OHP and Row all excurse in the sagittal plane -- a knee angle,
# an elbow angle, a hip height, a trunk pitch. The band pull apart's excursion is the hands
# travelling APART in the image plane, which makes the REP SIGNAL ITSELF view-bound rather than
# only the rules: from a pure `side` view the hands overlap, the excursion vanishes, and
# `segment_reps` returns nothing before a single rule runs.
#
# That is safe in production only because of a reachability fact, not by luck:
# `estimate_view_for_pose` is called with `allow_front=False` (src/pose/view_estimation.py:14-16),
# so the reachable labels are {side, rear, rear_oblique, unknown}, and across the 45 real pose
# JSONs in this repository the estimator emitted `rear_oblique` 30 times, `rear` 13, `unknown` 2,
# and `side` effectively never. Wrist spread survives `rear_oblique` foreshortened but present.
#
# ---------------------------------------------------------------------------------------
# ONE DROPPED LANDMARK SILENCES EVERY BAND PULL APART RULE FOR THAT FRAME.
# ---------------------------------------------------------------------------------------
# `required` below lists both ears, both shoulders, both elbows, both wrists and both hips. If
# `visible_point` drops any ONE of them the frame is marked `valid=False` and carries no metric
# keys at all, so every rule that masks on `frame.valid` goes silent for that frame, not just the
# one whose input landmark went missing. This mirrors `pushup_compute_raw`, `ohp_compute_raw`,
# `lunge_compute_raw` and `row_compute_raw`: an unmeasurable frame is refused wholesale rather
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

# Defined locally, matching row.py and overhead_press.py: geometry.py exports only the
# lower-body and shoulder/hip constants.
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

# The generic "lower body" set every movement module uses for the framework-level
# `lower_body_visibility` quality field. The name is squat-centric and carries awkwardly for a
# standing upper-body band exercise, exactly as it does for OHP, push-up and Row; this module's
# own rules never consume it.
LOWER_BODY_LANDMARKS = (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

BAND_PULL_APART_METRIC_KEYS: tuple[str, ...] = (
    "wrist_spread",
    "shoulder_width",
    "wrist_spread_shoulder_norm",
    "left_shoulder_ear_gap",
    "right_shoulder_ear_gap",
    "shoulder_ear_gap_shoulder_norm",
    "left_elbow_angle",
    "right_elbow_angle",
    "min_elbow_angle",
    "trunk_lean_image_signed_deg",
    "trunk_angle_speed_deg_s",
    "wrist_depth_offset",
)

# Below this a length/normalizer is degenerate and the dependent metric is NaN. Same guard value
# pushup.py, overhead_press.py, lunge.py and row.py use; not a tunable threshold.
_DEGENERATE_LENGTH = 1e-6


def _derivative(values: Sequence[float], fps: float) -> list[float]:
    """Central-difference time derivative, NaN at both boundaries.

    ONE-SIDED BOUNDARY ESTIMATES ARE REFUSED ON PURPOSE. A forward difference at frame 0 and a
    central difference at frame 1 have different biases; mixing them into one series makes the
    first samples systematically unlike the rest. NaN propagates through the mask and the frame
    is simply not scored. A NaN input (an invalid frame) poisons its two neighbours' derivatives,
    which is correct: a derivative across a hole in the data is not measured, it is guessed.
    Copied from `row._derivative`, whose momentum rule needs the identical property.
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


def band_pull_apart_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
    raw: list[dict] = []
    trunk_leans: list[float] = []

    for frame in frames:
        if not isinstance(frame, dict):
            raw.append({"valid": False})
            trunk_leans.append(np.nan)
            continue

        points = landmarks_to_array(frame.get("landmarks"))
        frame_index = int(frame.get("frame_index", 0) or 0)
        time = frame_index / fps if fps > 0 else 0.0
        required = (
            LEFT_EAR, RIGHT_EAR,
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
            trunk_leans.append(np.nan)
            continue

        wrist_spread = distance(points, LEFT_WRIST, RIGHT_WRIST)
        shoulder_width = distance(points, LEFT_SHOULDER, RIGHT_SHOULDER)
        normalizer_ok = np.isfinite(shoulder_width) and shoulder_width > _DEGENERATE_LENGTH
        wrist_spread_shoulder_norm = (
            wrist_spread / shoulder_width
            if np.isfinite(wrist_spread) and normalizer_ok
            else np.nan
        )

        left_shoulder = visible_point(points, LEFT_SHOULDER, dims=2)
        right_shoulder = visible_point(points, RIGHT_SHOULDER, dims=2)
        left_ear = visible_point(points, LEFT_EAR, dims=2)
        right_ear = visible_point(points, RIGHT_EAR, dims=2)
        # Image y grows DOWNWARD, so shoulder_y - ear_y is POSITIVE when the ear sits above the
        # shoulder, and SHRINKS as the shoulder rises toward the ear. The spec states its shrug
        # threshold on exactly this quantity ("gap_peak < gap_setup - 0.03").
        left_shoulder_ear_gap = float(left_shoulder[1] - left_ear[1])
        right_shoulder_ear_gap = float(right_shoulder[1] - right_ear[1])
        mean_gap = float(np.mean([left_shoulder_ear_gap, right_shoulder_ear_gap]))
        # SCALE-FREE DIAGNOSTIC THAT NO RULE FIRES ON. The spec's 0.03 shrug threshold carries no
        # normalizer, so it is raw image units and therefore camera-distance dependent (design
        # spec 4.5). Emitting the normalized companion lets a future validation compare the two
        # WITHOUT any threshold having been moved in the meantime.
        shoulder_ear_gap_shoulder_norm = mean_gap / shoulder_width if normalizer_ok else np.nan

        left_elbow_angle = angle_degrees(points, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        right_elbow_angle = angle_degrees(points, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        finite_elbows = [v for v in (left_elbow_angle, right_elbow_angle) if np.isfinite(v)]
        min_elbow_angle = float(min(finite_elbows)) if finite_elbows else np.nan

        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        # SIGNED, and deliberately NOT facing-corrected here. Positive = the shoulders sit toward
        # +x relative to the hips, in IMAGE coordinates. Which physical direction that is depends
        # on which way the lifter faces, which this layer cannot know and must not guess -- see
        # `_clip_facing_sign` in Task 4.
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

        left_wrist3 = visible_point(points, LEFT_WRIST, dims=3)
        right_wrist3 = visible_point(points, RIGHT_WRIST, dims=3)
        left_shoulder3 = visible_point(points, LEFT_SHOULDER, dims=3)
        right_shoulder3 = visible_point(points, RIGHT_SHOULDER, dims=3)
        # MediaPipe z is depth relative to the hip midpoint, NEGATIVE toward the camera. A band
        # pull apart holds the band in FRONT of the torso by definition, so the SIGN of this
        # offset identifies which way the lifter faces. Rule 4 reduces it; nothing is decided
        # here. NaN when any z is missing, and identically 0.0 under the RTMPose extraction path
        # (src/pose/rtmpose_pose_extraction.py writes z=0.0 for every landmark) -- both cases are
        # handled by rule 4's floor, not by a branch here.
        if all(p is not None for p in (left_wrist3, right_wrist3, left_shoulder3, right_shoulder3)):
            wrist_depth_offset = float(
                np.mean([left_wrist3[2], right_wrist3[2]])
                - np.mean([left_shoulder3[2], right_shoulder3[2]])
            )
        else:
            wrist_depth_offset = np.nan

        trunk_leans.append(trunk_lean)
        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "wrist_spread": wrist_spread,
                "shoulder_width": shoulder_width,
                "wrist_spread_shoulder_norm": wrist_spread_shoulder_norm,
                "left_shoulder_ear_gap": left_shoulder_ear_gap,
                "right_shoulder_ear_gap": right_shoulder_ear_gap,
                "shoulder_ear_gap_shoulder_norm": shoulder_ear_gap_shoulder_norm,
                "left_elbow_angle": left_elbow_angle,
                "right_elbow_angle": right_elbow_angle,
                "min_elbow_angle": min_elbow_angle,
                "trunk_lean_image_signed_deg": trunk_lean,
                "wrist_depth_offset": wrist_depth_offset,
            }
        )

    # THE DERIVATIVE IS COMPUTED HERE, IN THE METRIC LAYER, AND THAT IS LOAD-BEARING.
    # `run_detector` median-filters EVERY key in `metric_keys` with a 5-frame window. A median
    # over a POSITION/ANGLE series flattens the velocity transient rule 4's whip evidence exists
    # to find, before the rule ever sees it. Emitting the derivative AS the metric means the
    # framework's filter acts on the velocity -- a defensible low-pass on the quantity of
    # interest instead of an erasure of it. Same argument row.py makes for `wrist_accel_norm`.
    trunk_speed = _derivative(trunk_leans, fps)
    for index, item in enumerate(raw):
        if not item.get("valid"):
            continue
        speed = trunk_speed[index]
        item["trunk_angle_speed_deg_s"] = abs(float(speed)) if np.isfinite(speed) else float(np.nan)
    return raw


def band_pull_apart_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> pull -> peak -> return, segmented on `wrist_spread_shoulder_norm`.

    Mirrors `row_assign_phases`, substituting the pull-apart's spread signal and inverting the
    polarity: the row's peak is the MOST-FLEXED 30% of the rep, this movement's peak is the
    WIDEST 30%. Same fallbacks: an empty clip returns an empty list, a clip with no finite signal
    is entirely `unknown`, and an invalid frame is `unknown` regardless of where it sits (the
    validity check precedes the setup cutoff, so an occluded frame in the opening 15% is NOT
    labelled `setup`, which matters because `_setup_baseline` reduces over exactly those frames).
    """
    frame_count = len(raw)
    if frame_count == 0:
        return []

    spread_values = np.asarray(
        [float(item.get("wrist_spread_shoulder_norm", np.nan)) for item in raw], dtype=np.float32
    )
    valid_spread = spread_values[np.isfinite(spread_values)]
    if valid_spread.size == 0:
        return ["unknown" for _ in raw]

    # The widest 30% of the rep is the peak hold.
    peak_threshold = float(np.percentile(valid_spread, 70))
    widest_index = int(np.nanargmax(np.where(np.isfinite(spread_values), spread_values, -np.inf)))
    setup_cutoff = max(1, int(frame_count * 0.15))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append("setup")
            continue

        value = spread_values[index]
        if np.isfinite(value) and value >= peak_threshold:
            phases.append("peak")
        elif index < widest_index:
            phases.append("pull")
        else:
            phases.append("return")
    return phases


# ---------------------------------------------------------------------------------------
# STEP 0 -- KG QUERY RESOLUTION, recorded before any rule was written. Each string below was
# checked against data/kg/sports_kg_v3.graphml with `retrieve_graph_context(query,
# movement="Band Pull Apart")` -- the function PRODUCTION calls, not just `resolve_nodes`.
# Observed results, not predicted ones:
#
#   "Shoulder Shrugging" -> Band Pull Apart:Shoulder Shrugging
#       causes: Weak Scapular Stabilizers | quality_impacts: Shoulder Depression      NON-EMPTY
#   "Bent Elbows"        -> Band Pull Apart:Bent Elbows
#       NO buckets -- only the HAS_FAULT backlink                                     THIN
#   rule 4               -> NO Band-Pull-Apart-scoped node exists at all
#       "Trunk Extension" / "Loss Of Neutral Body Position" (Row's queries) do not resolve
#       under this movement's scoping; the shared nodes that do resolve are bare.
#
# The two gaps are recorded rather than masked. Pointing rule 2 at the shared `Range Of Motion`
# QualityDimension WOULD return a rich bucket set, and was rejected: its `corrections` bucket is
# "Wrapping Surface Adjustment", meaningless for this movement. A semantically correct thin card
# beats a semantically wrong full one. Both gaps are one-line fixes in
# scripts/knowledge/stub_general_movements_v3.py:80-87 and are logged against TODO.md's existing
# "many faults have no KG node" item.
BPA_SHRUG_KG_QUERY = "Shoulder Shrugging"

# Imported rather than re-typed, so a change to the shared constant cannot silently skip this
# module.
_OFF_VIEW_CONFIDENCE = VIEW_UNAVAILABLE_CONFIDENCE_SCALE

# FROM THE SPEC: "flag shrug if `gap_peak < gap_setup - 0.03` (shoulders elevate) on either
# side". RAW IMAGE UNITS -- the spec states no normalizer here, and it says "normalized by
# shoulder width" explicitly where it means that (the very next rule), so the absence is
# meaningful rather than an omission. The honest cost is camera-distance dependence:
# the same shrug filmed further away yields a smaller closure and fires less. Implemented as
# written; `shoulder_ear_gap_shoulder_norm` is emitted as the scale-free companion that no rule
# fires on, so a future validation can compare the two without moving this number.
SHRUG_MILD = 0.03
# RULE-LEVEL CHOICE MADE HERE. The parent spec states NO severity ramp for ANY Band Pull Apart
# fault (the Lunge section states its ramps explicitly, so the absence is meaningful). 0.075 is
# 2.5x the fire threshold, the convention `pushup.rule_hip_sag` uses for exactly this situation.
# A display/ranking curve, not a cited quantity.
SHRUG_SEVERE = 0.075


def _setup_baseline(core: list[CoreFrame], key: str) -> float:
    """Median of `key` over this window's valid `setup` frames; NaN when there are none.

    WHY THE BASELINE LIVES IN THE RULES AND NOT IN `band_pull_apart_compute_raw`. Two of this
    movement's three firing rules are deltas from a setup baseline, and a baseline is a PER-REP
    reduction. `run_detector` calls `compute_raw` over the WHOLE CLIP before `segment_reps`, so
    at metric time no rep boundary exists and there is no "this rep's setup" to reduce against.
    Rules receive a per-rep slice, which is the first place the question is answerable.

    MEDIAN, NOT MEAN, so one bad frame in a short setup cannot move the reference every later
    comparison is made against.

    NEVER A GUESSED BASELINE -- but what a caller does with a NaN one is conditional on the
    caller's own fire condition, NOT a universal "return []" contract:
      - A rule whose fire condition depends ONLY on this baseline (`rule_shrugging`,
        `rule_trunk_extension_compensation`) has nothing left to evaluate and returns [].
      - A rule whose fire condition is a DISJUNCTION with a non-baseline term
        (`rule_incomplete_rom`, which fires on spread ratio OR elbow angle, neither of which is a
        baseline delta) never consults this function at all.

    STATED LIMITATION, inherited from `row._setup_baseline` where it was measured: `setup` is the
    first 15% of the REP WINDOW, and the window has already been trimmed by `segment_reps` to the
    rep's excursion -- so on a short rep `setup` can be 1-2 frames and may already overlap loaded
    frames. Because every comparison here is `peak - baseline`, a baseline biased toward the
    loaded state makes the MEASURED change smaller than the true one: the failure mode is a
    MISSED fault, never a false one. Not corrected -- there is no principled way to detect "this
    setup frame is already loaded" without a second threshold the parent spec does not supply.
    """
    values = [
        frame.m(key)
        for frame in core
        if frame.valid and frame.phase == "setup" and np.isfinite(frame.m(key))
    ]
    if not values:
        return float(np.nan)
    return float(np.median(values))


def rule_shrugging(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag the shoulders rising toward the ears across the pull -- upper-trap dominance.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM.
      FIRE THRESHOLD 0.03: FROM THE SPEC ("flag shrug if gap_peak < gap_setup - 0.03").
      SEVERITY RAMP 0.03 -> 0.075: A RULE-LEVEL CHOICE (see SHRUG_SEVERE).

    PHASE SCOPE `peak`, FROM THE SPEC's own wording ("Compute at setup baseline and at peak").

    EITHER SIDE, NOT THE MEAN, also FROM THE SPEC ("on either side"). A unilateral shrug is the
    common presentation and averaging the two gaps would halve it toward the threshold.

    NO VIEW DISCOUNT, AND THAT IS AN ARGUMENT RATHER THAN AN OVERSIGHT. This rule's metric is a
    VERTICAL image-y difference between a shoulder and its own ear. A magnitude in image-y reads
    identically from in front of or behind the subject, so the rule is facing-free BY
    CONSTRUCTION -- the same argument `row.rule_asymmetric_pull` and `lunge.rule_knee_valgus`
    make for their own metrics. `rear` and `rear_oblique` (between them, 43 of the 45 real pose
    JSONs in this repository) therefore both earn the spec's `high` rating with no discount.
    """
    left_baseline = _setup_baseline(core, "left_shoulder_ear_gap")
    right_baseline = _setup_baseline(core, "right_shoulder_ear_gap")
    if not np.isfinite(left_baseline) and not np.isfinite(right_baseline):
        return []

    def closure(frame: CoreFrame) -> float:
        """Largest gap CLOSURE across the two sides; NaN-safe, so one occluded ear does not
        silence the other side."""
        options = [
            base - frame.m(key)
            for base, key in (
                (left_baseline, "left_shoulder_ear_gap"),
                (right_baseline, "right_shoulder_ear_gap"),
            )
            if np.isfinite(base) and np.isfinite(frame.m(key))
        ]
        return float(max(options)) if options else float(np.nan)

    mask = [
        frame.valid
        and frame.phase == "peak"
        and np.isfinite(closure(frame))
        and closure(frame) > SHRUG_MILD
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        closures = [closure(frame) for frame in segment]
        max_closure = float(np.nanmax(closures))
        severity = severity_from_range(max_closure, SHRUG_MILD, SHRUG_SEVERE, lower_is_worse=False)
        detections.append(
            build_detection(
                fault_id="bpa_shrugging",
                fault_name="Shrugging (Upper-Trap Dominance)",
                kg_query=BPA_SHRUG_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=closures,
                severity=severity,
                confidence=severity,
                observability="high",
                evidence={
                    "setup_left_gap": round(left_baseline, 4) if np.isfinite(left_baseline) else None,
                    "setup_right_gap": round(right_baseline, 4) if np.isfinite(right_baseline) else None,
                    "max_gap_closure": round(max_closure, 4),
                    "threshold": SHRUG_MILD,
                    "primary_label": "shoulder-ear gap closure vs setup",
                    "primary_value": round(max_closure, 4),
                    "primary_threshold": SHRUG_MILD,
                },
                citation=(
                    "Fukunaga T et al. Int J Sports Phys Ther (2022) PMC8975561, DOI "
                    "10.26603/001c.33026; Camargo PR & Neumann DA, Braz J Phys Ther (2019) "
                    "23(6):467–475, PMC6849087, DOI 10.1016/j.bjpt.2019.01.011."
                ),
                citation_support=(
                    "Fukunaga: \"it has been suggested that exercises should aim to "
                    "preferentially target the middle trapezius, lower trapezius, and posterior "
                    "RTC, with lower contributions from the upper trapezius and deltoid "
                    "muscles\" — a shrug inverts the intended UT-low pattern. Camargo & Neumann: "
                    "\"Exercises that increase the strength or relative activation of the upper "
                    "trapezius may be counterproductive in many patients with shoulder pain, "
                    "especially those with symptoms of impingement,\" because \"the upper "
                    "trapezius naturally causes an increased anterior tilt of the scapula, which "
                    "may compromise the volume within the subacromial space.\""
                ),
            )
        )
    return detections


BPA_ROM_KG_QUERY = "Bent Elbows"

# FROM THE SPEC: "Flag if `wrist_spread_peak / shoulder_width < 1.6`". Explicitly normalized by
# shoulder width by the spec's own wording, so scale-free -- unlike SHRUG_MILD above.
SPREAD_MILD = 1.6
# RULE-LEVEL CHOICE MADE HERE. A DESCENDING ramp: severity grows as the spread SHRINKS. 1.0 is
# the spread ratio at which the wrists sit exactly at shoulder width -- no horizontal abduction
# beyond the torso line at all, which is the natural floor of this movement rather than an
# arbitrary 2.5x. Display/ranking curve, not a cited quantity.
SPREAD_SEVERE = 1.0

# FROM THE SPEC, WITH ITS INEQUALITY CORRECTED. Parent spec line 739 reads "elbow-extension
# check `elbow_angle > ~150deg` maintained (bent-elbow curl-style cheat = fault)". Read
# literally, >150 degrees -- nearly STRAIGHT arms -- is the fault, which contradicts the
# parenthetical in the same sentence. The parenthetical is right and the inequality is a slip: a
# bent-elbow cheat means a SMALLER elbow angle. Corroboration rather than inference alone: the
# knowledge graph names this fault "Bent Elbows"
# (scripts/knowledge/stub_general_movements_v3.py:85), and Fukunaga's rationale -- more range
# covered against the band drives higher activation -- is a range argument that bending the
# elbows shortens. The NUMBER 150 is unchanged and stays FROM THE SPEC; only the comparison
# direction is corrected, and the correction is annotated in the parent spec so it cannot be
# silently re-flipped by someone reading line 739 alone.
ELBOW_MILD_DEG = 150.0
# RULE-LEVEL CHOICE MADE HERE. 40 degrees of ramp width, taken from `pushup.rule_shallow_depth`
# (100 -> 140) so this module's elbow ramp and push-up's cannot drift apart. DESCENDING.
ELBOW_SEVERE_DEG = 110.0

# Views in which the spec rates wrist spread `high` ("high -- front / rear"). `front` is listed
# knowing it is unreachable under `allow_front=False`: it is correct on the merits and costs
# nothing. `rear_oblique` foreshortens the frontal-plane spread, so it downgrades.
SPREAD_HIGH_VIEWS = {"front", "rear"}


def rule_incomplete_rom(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag a pull that stops short -- hands not fully spread, or elbows bent to cheat the range.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM.
      FIRE THRESHOLDS 1.6 and 150 deg: FROM THE SPEC (the latter with its inequality corrected,
      see ELBOW_MILD_DEG).
      SEVERITY RAMPS 1.6 -> 1.0 and 150 -> 110 deg: RULE-LEVEL CHOICES.

    A GENUINE DISJUNCTION, unlike `row.rule_momentum_jerk`'s second condition which was a strict
    SUBSET of its first and therefore unreachable. These two cues are independent failure modes:
    a lifter can reach full spread with bent elbows (short-lever cheat) or hold straight arms and
    stop short. `evidence["primary_label"]` records which term drove the verdict.

    PHASE SCOPE `peak`, FROM THE SPEC ("Peak wrist separation").

    NO SETUP BASELINE. Both terms are absolute thresholds on the peak, not deltas, so this rule
    never calls `_setup_baseline` and an occluded setup cannot silence it.
    """
    observable = ctx.view_type in SPREAD_HIGH_VIEWS
    scale = 1.0 if observable else _OFF_VIEW_CONFIDENCE

    def scores(frame: CoreFrame) -> tuple[float, float]:
        """(spread severity, elbow severity) for one frame; 0.0 where the term does not fire."""
        spread = frame.m("wrist_spread_shoulder_norm")
        elbow = frame.m("min_elbow_angle")
        spread_sev = (
            severity_from_range(spread, SPREAD_MILD, SPREAD_SEVERE, lower_is_worse=True)
            if np.isfinite(spread) and spread < SPREAD_MILD
            else 0.0
        )
        elbow_sev = (
            severity_from_range(elbow, ELBOW_MILD_DEG, ELBOW_SEVERE_DEG, lower_is_worse=True)
            if np.isfinite(elbow) and elbow < ELBOW_MILD_DEG
            else 0.0
        )
        return spread_sev, elbow_sev

    mask = [
        frame.valid and frame.phase == "peak" and max(scores(frame)) > 0.0 for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        pairs = [scores(frame) for frame in segment]
        combined = [max(p) for p in pairs]
        severity = float(np.nanmax(combined))
        worst = int(np.nanargmax(combined))
        spread_drove = pairs[worst][0] >= pairs[worst][1]
        min_spread = float(
            np.nanmin([frame.m("wrist_spread_shoulder_norm") for frame in segment])
        )
        min_elbow = float(np.nanmin([frame.m("min_elbow_angle") for frame in segment]))
        detections.append(
            build_detection(
                fault_id="bpa_incomplete_rom",
                fault_name="Incomplete ROM (Hands Not Fully Spread)",
                kg_query=BPA_ROM_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=combined,
                severity=severity,
                confidence=severity * scale,
                observability="high" if observable else "medium",
                evidence={
                    "min_spread_ratio": round(min_spread, 3),
                    "spread_threshold": SPREAD_MILD,
                    "min_elbow_angle_deg": round(min_elbow, 2),
                    "elbow_threshold_deg": ELBOW_MILD_DEG,
                    "primary_label": (
                        "wrist spread at peak" if spread_drove else "elbow flexion at peak"
                    ),
                    "primary_value": round(min_spread if spread_drove else min_elbow, 3),
                    "primary_threshold": SPREAD_MILD if spread_drove else ELBOW_MILD_DEG,
                },
                citation=(
                    "Fukunaga T et al. Int J Sports Phys Ther (2022) PMC8975561, DOI "
                    "10.26603/001c.33026."
                ),
                citation_support=(
                    "Fukunaga: peak muscle activity spanned \"15.3% to 72.6% of MVC across "
                    "muscles and exercise conditions,\" and the diagonal-up (largest-excursion, "
                    "against-gravity) direction produced the highest trapezius activity — "
                    "\"the diagonal up movement showing the highest shoulder-girdle muscle "
                    "activity is understandable as the arm is moving against gravity, resulting "
                    "in higher overall load\" — i.e. covering more range against the band "
                    "drives higher target activation, which a truncated pull loses."
                ),
            )
        )
    return detections
