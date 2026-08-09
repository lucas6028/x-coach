# Leg Abduction (standing unilateral hip abduction) raw metrics, phase segmentation and rules.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `leg_abduction_compute_raw` /
# `leg_abduction_assign_phases` compute per-frame quantities and a phase label only. Every
# number that decides anything belongs in a `rule_*` function.
#
# ---------------------------------------------------------------------------------------
# ONE RULE SHIPS, ONE IS PERMANENTLY SILENT, TWO ARE WITHDRAWN -- AND THE LABELED DATA DECIDED
# THAT, WHICH HAS NOT HAPPENED BEFORE IN THIS PROGRAMME.
# ---------------------------------------------------------------------------------------
#   rule_trunk_lean_compensation      ships -- exact KG node, the parent spec's own ratio
#                                     transfers to a body-relative frame by an identity, and its
#                                     signal orders correctness in EVERY subject of 210 labeled
#                                     repetitions of the matching variant
#   rule_insufficient_abduction_rom   REGISTERED, PERMANENTLY SILENT -- best KG seed of the three
#                                     and a clean metric, but no source states a range and the
#                                     spec's own cut fires PREFERENTIALLY ON REPETITIONS HUMANS
#                                     JUDGED CORRECT
#   abd_hip_flexion_er_substitution   WITHDRAWN, absent -- citation is band placement in a monster
#                                     walk; needs a sagittal view of a frontal-plane exercise;
#                                     no KG node
#   abd_momentum                      WITHDRAWN, absent -- half the heuristic is a per-user
#                                     baseline this architecture does not have, the other half is
#                                     uncited; no KG node
#
# AND ONE SUB-CLAUSE: the parent spec's shipped rule is a DISJUNCTION of pelvic tilt and trunk
# lean, and only the trunk-lean disjunct is implemented. See PELVIC_TILT_DISJUNCT_NOT_IMPLEMENTED.
#
# ---------------------------------------------------------------------------------------
# THE SUPPORT LIMB IS THIS MODULE'S VERTICAL, AND THAT IS WHAT UNBLOCKS GROUP E HERE.
# ---------------------------------------------------------------------------------------
# The parent spec's Group E update block records that every quantity that section defines "vs the
# floor/horizontal" is unrecoverable from a frame, and names what that leaves for this movement:
# `pelvic-tilt vs horizontal` and `trunk lateral-lean` are BOTH specified against the image
# horizontal, and `insufficient_rom`'s "thigh vector relative to the pelvis midline / vertical" is
# a mixed case whose vertical reading does not survive.
#
# In a STANDING unilateral movement the stance leg is planted and load-bearing, so
# `hip_stance -> ankle_stance` is a body-internal stand-in for the world vertical. Every metric
# here is measured against it and every metric is therefore roll-invariant. Neither other Group E
# movement could take this route: both are performed lying down and neither has a planted limb.
#
# ---------------------------------------------------------------------------------------
# THE SIGN IS RECOVERABLE HERE, AND THE REASON IS DOT PRODUCTS RATHER THAN CROSS PRODUCTS.
# ---------------------------------------------------------------------------------------
# Shoulder Bridge's central finding was that the arc it needed could not be signed: two
# body-relative constructions, both roll-invariant by design, both measured to fail on real
# footage. Both were CROSS products. A cross product against a body axis is invariant under camera
# roll but ANTI-invariant under mirroring -- its sign flips when the subject faces away from the
# camera rather than toward it, which no monocular pipeline can tell. A DOT product against a body
# axis is invariant under both, and every signed quantity here is one. `tests/test_leg_abduction
# .py::InvarianceTest` pins both invariances all the way through to a byte-identical detection.
#
# That is not a rebuttal of the Shoulder Bridge result -- its construction (A) was mirror-
# invariant too, by taking a product of two cross products, and still failed empirically. It is a
# narrower claim: when a signed body-relative quantity is available as a PROJECTION ONTO A BODY
# AXIS, prefer it, because it needs no argument about mirroring at all. Here that buys the ability
# to tell a pelvic HIKE from a pelvic DROP -- and that turns out to decide a sub-clause.
#
# ---------------------------------------------------------------------------------------
# STANDING UP IS NOT ENOUGH TO FIX THE VIEW ESTIMATOR.
# ---------------------------------------------------------------------------------------
# `src/pose/view_estimation.py`'s limit 1 voids the front/rear/oblique labels for HORIZONTAL
# subjects, which is what silenced the view logic in both other Group E modules. This subject is
# upright, so the limit does not apply and the labels could finally be checked against ground
# truth the dataset records per repetition. Measured on REHAB24-6 Ex4: the estimator is
# SYSTEMATICALLY INVERTED, returning an oblique label for the frontal camera and a sagittal label
# for the oblique one, and it emits a `FRONTAL_OBSERVABLE_VIEWS` label on essentially no
# repetition. So the confidence discount below is a CONSTANT on the only labeled corpus that
# exists -- it distinguishes nothing, and nothing here is evidence that view gating works. The
# rule ships anyway because it never GATES on a view label, only discounts. Design spec 1.3.
#
# ---------------------------------------------------------------------------------------
# EIGHT REQUIRED LANDMARKS, TWO MORE THAN ANY OTHER GROUP E MODULE, AND THE EXTRA TWO ARE ANKLES.
# ---------------------------------------------------------------------------------------
# Sit-up and Shoulder Bridge require six and neither requires an ankle. The ankles are the price of
# the support limb: without them this module has no vertical. If `visible_point` drops any ONE of
# the eight the frame is marked `valid=False` and carries no metric keys, so every rule masking on
# `frame.valid` goes silent for that frame. MEASURED ON 210 LABELED REPETITIONS, THE COST IS LARGE:
# median validity rate 0.600 and p10 0.000 -- at least a tenth of repetitions carry NO fully
# landmarked frame at all -- and 35 of 210 (17%) end up on a `segment_reps` fallback path as a
# result. The p10 is quoted alongside the median deliberately, because the median hides exactly the
# repetitions where the gate bites. notes/leg-abduction-rule-validation.md section 2.
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, midpoint, distance, mean_visibility,
    contiguous_true_segments, severity_from_range,
)
from src.pose.movements.base import CoreFrame, MovementDetector, RuleContext
from src.pose.movements import registry
from src.pose.pose_rule_detector import (
    VIEW_UNAVAILABLE_CONFIDENCE_SCALE,
    PoseRuleDetection,
    build_detection,
)

# The generic "lower body" set every movement module uses for the framework-level
# `lower_body_visibility` quality field. This module's own rules never read it.
LOWER_BODY_LANDMARKS = (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

LEG_ABDUCTION_METRIC_KEYS: tuple[str, ...] = (
    # Support-limb-referenced, and therefore computed UNDER THE HYPOTHESIS that the named side
    # is the working leg -- the reference is the OTHER leg, which is only the support limb if
    # the hypothesis holds. Read exactly one of the two, after `resolve_moving_side`.
    "left_abduction_deg",
    "right_abduction_deg",
    "left_trunk_tilt_deg",
    "right_trunk_tilt_deg",
    "left_pelvic_hike_ratio",
    "right_pelvic_hike_ratio",
    # Trunk-referenced, and therefore side-INDEPENDENT: both legs are measured against the same
    # thing, so the two are comparable and the larger one names the working leg. This pair
    # exists because the support-limb pair CANNOT do that job -- see `resolve_moving_side`.
    "left_thigh_trunk_deg",
    "right_thigh_trunk_deg",
    "max_thigh_trunk_deg",
)

_SIDE_HIP = {"left": LEFT_HIP, "right": RIGHT_HIP}
_SIDE_KNEE = {"left": LEFT_KNEE, "right": RIGHT_KNEE}
_SIDE_ANKLE = {"left": LEFT_ANKLE, "right": RIGHT_ANKLE}
_OTHER_SIDE = {"left": "right", "right": "left"}


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


def _side_metrics(points: np.ndarray, moving: str) -> dict[str, float]:
    """The three moving-side quantities, all measured against the SUPPORT LIMB.

    THE SUPPORT LIMB IS THIS MODULE'S VERTICAL, and that is the whole reason the parent spec's
    Group E quantities become computable here. `pelvic-tilt vs horizontal` and `trunk
    lateral-lean` are both specified against the IMAGE horizontal, which the parent spec's own
    Group E update block records as not recoverable from a frame. In a STANDING unilateral
    movement the stance leg is planted and load-bearing, so `hip_stance -> ankle_stance` is a
    body-internal stand-in for the world vertical -- available here and available in no other
    Group E movement, because the other two are performed lying down.

    EVERY SIGN COMES FROM A DOT PRODUCT, NEVER A CROSS PRODUCT, and that is deliberate. A cross
    product against a body axis is invariant under camera roll but ANTI-invariant under
    mirroring, so its sign flips when the subject faces away from the camera instead of toward
    it -- which no monocular pipeline can tell. A dot product is invariant under both. Shoulder
    Bridge built two cross-product sign constructions and measured both to fail; this module
    does not repeat that.
    """
    stance = _OTHER_SIDE[moving]
    hip_m = visible_point(points, _SIDE_HIP[moving], dims=2)
    hip_s = visible_point(points, _SIDE_HIP[stance], dims=2)
    knee_m = visible_point(points, _SIDE_KNEE[moving], dims=2)
    ankle_s = visible_point(points, _SIDE_ANKLE[stance], dims=2)
    shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
    hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
    hip_width = distance(points, LEFT_HIP, RIGHT_HIP, dims=2)

    out = {
        f"{moving}_abduction_deg": math.nan,
        f"{moving}_trunk_tilt_deg": math.nan,
        f"{moving}_pelvic_hike_ratio": math.nan,
    }
    if hip_m is None or hip_s is None or ankle_s is None:
        return out

    support = _unit(hip_s - ankle_s)  # points "up" along the planted limb
    if support is None:
        return out

    if knee_m is not None:
        # Abduction as the angle between the moving thigh and the DOWNWARD support direction:
        # 0 deg = the lifted thigh still hangs parallel to the stance leg, larger = carried out.
        out[f"{moving}_abduction_deg"] = _angle_between(knee_m - hip_m, -support)

    if shoulder_mid is not None and hip_mid is not None:
        # How far the trunk departs from the support-limb line. This is the parent spec's
        # "trunk lateral-lean ... normalized by trunk length" with the image horizontal replaced
        # by the support limb; `sin` of this angle IS that ratio, so the spec's own number
        # transfers without a unit conversion. UNSIGNED: see the rule for why the direction is
        # not claimed.
        out[f"{moving}_trunk_tilt_deg"] = _angle_between(shoulder_mid - hip_mid, support)

    if math.isfinite(hip_width) and hip_width > 0.0:
        # + => the moving-side hip sits HIGHER along the support limb than the stance hip
        #      (pelvic HIKE, the knowledge graph's `Leg Abduction:Pelvic Hiking`)
        # - => it sits LOWER (pelvic DROP, the parent spec's Trendelenburg framing)
        out[f"{moving}_pelvic_hike_ratio"] = float(np.dot(hip_m - hip_s, support)) / hip_width

    return out


def _thigh_trunk_angles(points: np.ndarray) -> dict[str, float]:
    """Each thigh against the DOWNWARD trunk direction -- the one side-independent pair here.

    THIS EXISTS BECAUSE THE SUPPORT-LIMB PAIR CANNOT ANSWER "WHICH LEG IS WORKING", AND THAT WAS
    MEASURED, NOT ASSUMED. `_side_metrics` references each thigh to the OTHER leg, so
    `left_abduction_deg` and `right_abduction_deg` are both, approximately, the angle BETWEEN the
    two legs -- near-equal by construction. Comparing them to pick the working leg scored 7
    correct / 14 wrong / 30 refused on 51 labeled repetitions whose working leg the dataset
    records: worse than a coin flip. Referencing both thighs to the same trunk axis fixes it.

    The trunk axis leans when the subject leans, so this quantity is contaminated by exactly the
    compensation the trunk-lean rule scores. That is tolerable for ranking two legs against each
    other -- the contamination is common to both -- and is why the RULES read the support-limb
    pair instead.
    """
    shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
    hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
    out = {"left_thigh_trunk_deg": math.nan, "right_thigh_trunk_deg": math.nan}
    if shoulder_mid is None or hip_mid is None:
        return out
    trunk_down = hip_mid - shoulder_mid
    for side in ("left", "right"):
        hip = visible_point(points, _SIDE_HIP[side], dims=2)
        knee = visible_point(points, _SIDE_KNEE[side], dims=2)
        if hip is None or knee is None:
            continue
        out[f"{side}_thigh_trunk_deg"] = _angle_between(knee - hip, trunk_down)
    return out


def leg_abduction_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
    raw: list[dict] = []

    for frame in frames:
        if not isinstance(frame, dict):
            raw.append({"valid": False})
            continue

        points = landmarks_to_array(frame.get("landmarks"))
        frame_index = int(frame.get("frame_index", 0) or 0)
        time = frame_index / fps if fps > 0 else 0.0
        # EIGHT required landmarks, two more than any other Group E module, and the two extra
        # ones are the ANKLES: without them there is no support limb and therefore no vertical.
        required = (
            LEFT_SHOULDER, RIGHT_SHOULDER,
            LEFT_HIP, RIGHT_HIP,
            LEFT_KNEE, RIGHT_KNEE,
            LEFT_ANKLE, RIGHT_ANKLE,
        )
        valid = points is not None and all(
            visible_point(points, index, dims=2) is not None for index in required
        )
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

        item: dict[str, float | int | bool] = {
            "frame_index": frame_index,
            "time": time,
            "valid": True,
            "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
        }
        item.update(_side_metrics(points, "left"))
        item.update(_side_metrics(points, "right"))
        item.update(_thigh_trunk_angles(points))
        # THE REP SIGNAL, and it is the TRUNK-referenced pair rather than the support-limb one.
        # `compute_raw` runs over the WHOLE CLIP before `segment_reps`, so there is no repetition
        # boundary yet and therefore no way to know which leg is working -- the signal must be
        # side-independent, and only the trunk-referenced pair is.
        finite = [
            value
            for value in (
                float(item["left_thigh_trunk_deg"]),  # type: ignore[arg-type]
                float(item["right_thigh_trunk_deg"]),  # type: ignore[arg-type]
            )
            if np.isfinite(value)
        ]
        item["max_thigh_trunk_deg"] = max(finite) if finite else math.nan
        raw.append(item)

    return raw


PEAK_PHASE = "peak"
CONCENTRIC_PHASE = "concentric"
ECCENTRIC_PHASE = "eccentric"
SETUP_PHASE = "setup"
# Every frame of the lift that is not the standing setup. The compensations this module scores
# happen WHILE the leg is being carried out and brought back, not while the subject stands.
ACTIVE_PHASES = (CONCENTRIC_PHASE, PEAK_PHASE, ECCENTRIC_PHASE)


def leg_abduction_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> concentric -> peak -> eccentric, segmented on `max_thigh_trunk_deg`.

    Polarity matches Arm Abduction and Shoulder Bridge and is the inverse of Sit-up's on the
    same body: the effort peak of an abduction is the limb carried FURTHEST from the support
    line, so `peak` is the most-abducted 30% of the repetition (the 70th percentile and above).

    Same fallbacks as every other module: an empty clip returns an empty list, a clip with no
    finite signal is entirely `unknown`, and an invalid frame is `unknown` regardless of where
    it sits (the validity check precedes the setup cutoff, so an occluded frame in the opening
    15% is NOT labelled `setup`).
    """
    frame_count = len(raw)
    if frame_count == 0:
        return []

    values = np.asarray(
        [float(item.get("max_thigh_trunk_deg", np.nan)) for item in raw], dtype=np.float32
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ["unknown" for _ in raw]

    peak_threshold = float(np.percentile(finite, 70))
    highest_index = int(np.nanargmax(np.where(np.isfinite(values), values, -np.inf)))
    setup_cutoff = max(1, int(frame_count * 0.15))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append(SETUP_PHASE)
            continue
        value = values[index]
        if np.isfinite(value) and value >= peak_threshold:
            phases.append(PEAK_PHASE)
        elif index < highest_index:
            phases.append(CONCENTRIC_PHASE)
        else:
            phases.append(ECCENTRIC_PHASE)
    return phases


# Minimum peak-abduction gap, in degrees, between the two legs before this module will name one
# of them the working leg. Below it the repetition is ambiguous and every side-relative quantity
# is refused rather than guessed -- the same shape as `lunge.LEAD_SIDE_MIN_SEPARATION_DEG`.
MOVING_SIDE_MIN_SEPARATION_DEG = 8.0


def resolve_moving_side(window: list[CoreFrame]) -> str | None:
    """Which leg was carried out in this repetition: "left", "right", or None if unresolvable.

    WHY THIS LIVES IN THE RULES AND NOT IN `leg_abduction_compute_raw`: `run_detector` calls
    `compute_raw` over the WHOLE CLIP before `segment_reps`, so at metric time there is no
    repetition boundary and therefore no question to answer. A per-frame "whichever leg is more
    abducted right now" flickers through `setup`, where both legs sit within landmark noise of
    the support line, and `centered_median` would then blend two legs into a number describing
    neither. Rules receive a per-rep slice, which is the first place the question is answerable.
    This is `lunge.resolve_lead_side`'s argument, restated because it applies unchanged.

    UNLIKE EVERY OTHER RESOLVER IN THIS REGISTRY, THIS ONE HAS GROUND TRUTH, AND IT WAS CHECKED.
    REHAB24-6 `Ex4` records `exercise_subtype` ("left leg" / "right leg") on all 210 labeled
    repetitions. Of the 175 that reached this function, it returned 163 correct, **1 wrong**, and
    declined 11 as ambiguous: accuracy 0.994 when it answers, coverage 0.937. The single error
    matters more than the rate, because every side-relative quantity in this module is read off
    the leg named here -- a wrong answer puts the rule on the STANCE leg's landmarks.
    notes/leg-abduction-rule-validation.md section 3.

    NOT PHASE-SCOPED: it scans every frame in `window`, because it is only choosing which leg
    goes furthest. `setup` frames simply lose that competition on a real repetition, and on a
    fallback whole-clip path the phase labels are the least trustworthy thing available.
    """
    best = {"left": -math.inf, "right": -math.inf}
    for frame in window:
        if not frame.valid:
            continue
        for side in ("left", "right"):
            value = frame.m(f"{side}_thigh_trunk_deg")
            if np.isfinite(value) and value > best[side]:
                best[side] = value
    if not all(math.isfinite(value) for value in best.values()):
        return None
    if abs(best["left"] - best["right"]) < MOVING_SIDE_MIN_SEPARATION_DEG:
        return None
    return "left" if best["left"] > best["right"] else "right"


# ---------------------------------------------------------------------------------------
# STEP 0 -- KG QUERY RESOLUTION, recorded before any rule was written. Each string below was
# checked against data/kg/sports_kg_v3.graphml with `retrieve_graph_context(query, movement=
# "Leg Abduction")` -- the function PRODUCTION calls, not just `resolve_nodes`. Observed:
#
#   "Trunk Lean"          -> Leg Abduction:Trunk Lean Compensation
#       quality_impacts: No Compensatory Trunk Movement                    THIN, EXACT
#   "Insufficient Abduction Range" -> Leg Abduction:Insufficient Abduction Range
#       causes: Weak Hip Abductors; quality_impacts: Hip Abduction         TWO BUCKETS, EXACT
#   "Pelvic Drop"         -> NO MATCH AT ALL, zero nodes
#   "Pelvic Hiking"       -> Leg Abduction:Pelvic Hiking
#       quality_impacts: Pelvic Control                                    THIN, EXACT
#   "Hip Flexion Substitution" -> Hip, Hip Flexion (generic anatomy nodes, no fault)
#   "Momentum"                 -> Anterior Momentum Generation, Forward Momentum (both from
#                                 OTHER movements' subgraphs)              NO `Leg Abduction:` NODE
#
# `"Insufficient Abduction ROM"` -- the obvious first phrasing -- resolves to ZERO nodes; the
# graph's own wording is `Range`, not `ROM`. Recorded because a rule shipped on the first
# phrasing would have carried an empty card and nothing would have failed.
#
# THE GRAPH HAS EXACTLY THREE FAULTS FOR THIS MOVEMENT, AND THE TWO RULES THIS MODULE DOES NOT
# IMPLEMENT ARE THE TWO WITH NO NODE. That is the first time in this programme the graph and the
# citation audit have independently selected the same subset.
#
# AND THE THIRD NODE IS THE PARENT SPEC'S FAULT WITH ITS SIGN REVERSED: the graph says `Pelvic
# Hiking`, the spec says pelvic DROP, and `Pelvic Drop` matches nothing. See
# `PELVIC_TILT_DISJUNCT_NOT_IMPLEMENTED` below for what the labeled data says about that.
LEG_ABDUCTION_TRUNK_LEAN_KG_QUERY = "Trunk Lean"
LEG_ABDUCTION_ROM_KG_QUERY = "Insufficient Abduction Range"


# Views in which a frontal-plane compensation is fully observable. Transcribed from
# `arm_abduction.FRONTAL_OBSERVABLE_VIEWS`, and kept rather than deleted even though the module
# header records that this set is satisfied on essentially NO repetition of the only labeled
# corpus. Two reasons: the discount is the honest thing to apply if the estimator is ever fixed,
# and deleting it would hide the measurement behind an absence.
FRONTAL_OBSERVABLE_VIEWS = {"front", "rear"}
_OFF_VIEW_CONFIDENCE = VIEW_UNAVAILABLE_CONFIDENCE_SCALE


# FROM THE SPEC: "trunk lateral-lean = horizontal offset of shoulder midpoint (11,12) from hip
# midpoint (23,24), normalized by trunk length ... Flag if ... lateral-lean exceeds ~0.10-0.15 of
# trunk length during the abduction phase."
#
# THE RATIO TRANSFERS WITHOUT A UNIT CONVERSION, WHICH IS WHY THIS RE-ANCHORING IS CHEAP. The
# spec's quantity is the component of the trunk vector PERPENDICULAR to the reference direction,
# over the trunk length -- i.e. exactly `sin` of the angle between the trunk and that reference.
# Swapping the image horizontal for the support limb changes the reference and nothing else, so
# the spec's own 0.10-0.15 becomes 5.74-8.63 deg with no number invented.
#
# THE CONSERVATIVE END OF THE SPEC'S RANGE IS TAKEN. 0.15 rather than 0.10, because the spec
# gives a band and this rule is the one that will fire in production; the measured cost of the
# choice is in the design spec's fire census rather than argued in the abstract.
TRUNK_LEAN_MILD_RATIO = 0.15
TRUNK_LEAN_MILD_DEG = math.degrees(math.asin(TRUNK_LEAN_MILD_RATIO))       # 8.63
# RULE-LEVEL CHOICE. The parent spec states no severity ramp for any Leg Abduction fault (the
# Lunge section states its ramps explicitly, so the absence is meaningful). Twice the fire ratio
# is the severe end: a display/ranking curve, not a cited threshold.
TRUNK_LEAN_SEVERE_RATIO = 0.30
TRUNK_LEAN_SEVERE_DEG = math.degrees(math.asin(TRUNK_LEAN_SEVERE_RATIO))   # 17.46


def rule_trunk_lean_compensation(
    core: list[CoreFrame], ctx: RuleContext
) -> list[PoseRuleDetection]:
    """Flag a repetition whose trunk leans off the support limb while the leg is carried out.

    THRESHOLD PROVENANCE -- TWO CATEGORIES, DO NOT CONFLATE THEM.
      FIRE THRESHOLD 0.15 of trunk length (8.63 deg off the support limb): FROM THE SPEC, whose
        range is 0.10-0.15; the conservative end is taken. Neither cited source states a pose
        threshold of any kind.
      SEVERITY RAMP 0.15 -> 0.30: A RULE-LEVEL CHOICE. See TRUNK_LEAN_SEVERE_RATIO.

    SCOPE IS THE ACTIVE PHASES, NOT THE WHOLE REP, and that follows the spec ("during the
    abduction phase"). `setup` is excluded because a subject who stands crooked before starting
    has not compensated for anything yet.

    IT USES `contiguous_true_segments`, so a single noisy frame over the cut cannot fire it --
    the exceedance has to be SUSTAINED for `ctx.min_frames`. That matters more here than in most
    modules: `ACTIVE_PHASES` covers about 85% of the repetition, so the Bicep Curl phase-fraction
    trap barely binds and the mask is long; what protects against a false alarm is the run-length
    requirement, not the phase scope.

    MEASURED ON 210 LABELED REPETITIONS OF THE MATCHING VARIANT, THROUGH THE REAL `run_detector`:

        fired 44/210    tp 39   fp 5   fn 51   tn 115
        precision 0.886    specificity 0.958    sensitivity 0.433

    The signal's AUC is 0.840 pooled and above chance in EVERY ONE of the 9 subjects (median
    0.833, min 0.690). The rule is deliberately conservative and was not tuned to be: 8.63 deg is
    `asin(0.15)`, the conservative end of the parent spec's own band, and it fires on 10% of
    repetitions judged correct against 59% judged incorrect.

    THE OBLIQUE CAMERA COSTS SENSITIVITY, NOT PRECISION, and that is the benign failure mode.
    Split by the orientation the dataset records: `front` 30 tp / 5 fp / 23 fn, `half-profile` 9
    tp / ZERO fp / 28 fn. A lateral lean projected obliquely reads smaller than it is, so the rule
    goes quiet rather than wrong. Shoulder Bridge's census went the other way -- its unfavourable
    camera produced near-full-severity FALSE ALARMS on correct repetitions.

    THE VIEW DISCOUNT BELOW IS A CONSTANT ON THAT CORPUS AND PROVES NOTHING. The estimator emitted
    a `FRONTAL_OBSERVABLE_VIEWS` label on 0 of 210 repetitions (module header), so every one of
    the numbers above was produced with `scale` at 0.65. The rule never GATES on a view label,
    which is why the census is still the rule's own behaviour.
    """
    moving = resolve_moving_side(core)
    if moving is None:
        return []

    scale = 1.0 if ctx.view_type in FRONTAL_OBSERVABLE_VIEWS else _OFF_VIEW_CONFIDENCE
    key = f"{moving}_trunk_tilt_deg"

    def tilt(frame: CoreFrame) -> float:
        return frame.m(key)

    mask = [
        frame.valid
        and frame.phase in ACTIVE_PHASES
        and np.isfinite(tilt(frame))
        and tilt(frame) > TRUNK_LEAN_MILD_DEG
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        tilts = [tilt(frame) for frame in segment]
        peak = float(np.nanmax(tilts))
        severity = severity_from_range(
            peak, TRUNK_LEAN_MILD_DEG, TRUNK_LEAN_SEVERE_DEG, lower_is_worse=False
        )
        detections.append(
            build_detection(
                # THE PARENT SPEC'S ID, KEPT DESPITE NAMING A DISJUNCT THIS RULE DOES NOT
                # IMPLEMENT. `arm_vw.rule_loss_of_elevation` set the precedent: an id survives
                # the withdrawal of one of its branches, because the id is the join key between
                # the spec, the registry and stored analyses. The user never sees it -- the card
                # shows `fault_name`, which says Trunk Lean and nothing about a pelvis. Read
                # PELVIC_TILT_DISJUNCT_NOT_IMPLEMENTED before treating the id as a claim.
                fault_id="abd_pelvic_drop_trunk_lean",
                fault_name="Trunk Lean Compensation",
                kg_query=LEG_ABDUCTION_TRUNK_LEAN_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=tilts,
                severity=severity,
                confidence=severity * scale,
                observability=(
                    "high" if ctx.view_type in FRONTAL_OBSERVABLE_VIEWS else "medium"
                ),
                evidence={
                    "max_trunk_lean_deg": round(peak, 2),
                    "max_trunk_lean_ratio": round(math.sin(math.radians(peak)), 3),
                    "threshold_deg": round(TRUNK_LEAN_MILD_DEG, 2),
                    "threshold_ratio": TRUNK_LEAN_MILD_RATIO,
                    "moving_side": moving,
                    "primary_label": "trunk lean off the support limb",
                    "primary_value": round(peak, 2),
                    "primary_threshold": round(TRUNK_LEAN_MILD_DEG, 2),
                },
                citation=(
                    "González-de-la-Flor Á, J Funct Morphol Kinesiol (2025), PMC12372021, "
                    "DOI 10.3390/jfmk10030294; corroborated by Rodrigues R et al., PLoS One "
                    "(2025), PMC12416692, DOI 10.1371/journal.pone.0331553."
                ),
                citation_support=(
                    "González-de-la-Flor describes standing hip abduction in his own words as "
                    "resisted movement \"in the frontal plane\", and states that hip-abductor "
                    "\"weakness leads to a characteristic Trendelenburg gait or compensatory "
                    "trunk lean\" and that \"excessive sway or lateral trunk lean may reduce "
                    "abductor demand by mechanically offloading the stance limb\", the optimal "
                    "technique being to maintain \"frontal plane neutrality\". Rodrigues "
                    "likewise reports that hip-abductor weakness is compensated \"by increasing "
                    "ipsilateral trunk lean\". NOTE, because the parent spec marks this "
                    "\"VERIFIED\": both trunk-lean sentences carry reference markers, so the "
                    "support is SECONDARY, and both describe gait or a band-walk stance limb "
                    "rather than a standing abduction repetition; Rodrigues explicitly collected "
                    "no trunk or pelvis kinematics at all. Neither source states any pose "
                    "threshold — the 0.15-of-trunk-length cut applied here is the parent spec's "
                    "own number, at the conservative end of its 0.10-0.15 band."
                ),
            )
        )
    return detections


# THE PARENT SPEC'S RULE IS A DISJUNCTION AND ONLY ONE DISJUNCT IS IMPLEMENTED. `abd_pelvic_drop
# _trunk_lean` names "two coupled signals": (1) frontal-plane PELVIC TILT and (2) TRUNK LEAN,
# flagged on either. The rule above implements (2). This constant records why (1) is absent, so
# the omission is a decision on the record rather than a gap.
#
#   1. THE METRIC IS COMPUTED AND EMITTED, AND IT DISCRIMINATES AS WELL AS THE SHIPPED ONE.
#      `{side}_pelvic_hike_ratio` is a real metric key, signed, roll- and mirror-invariant, and
#      measured over 210 labeled repetitions it scores AUC 0.848 pooled (per-subject median
#      0.800, min 0.690) against the shipped signal's 0.840. It is not absent for want of signal.
#   2. THE CITATION AND THE MEASUREMENT DISAGREE ABOUT THE SIGN. The spec's fault is pelvic
#      DROP -- "Trendelenburg-like", the pelvis falling on the unsupported side -- and its
#      citation_support quotes a sentence that announces its own subject: "weakness leads to a
#      characteristic Trendelenburg GAIT". Measured on the labeled repetitions, the direction
#      that separates incorrect from correct is the OPPOSITE one: the moving-side hip rides
#      HIGHER, not lower. Firing the spec's rule as written would fire on the sign the data says
#      is the CORRECT execution.
#   3. THE KNOWLEDGE GRAPH AGREES WITH THE DATA AND NOT WITH THE SPEC. `retrieve_graph_context
#      ("Pelvic Drop", movement="Leg Abduction")` matches ZERO nodes; `"Pelvic Hiking"` matches
#      `Leg Abduction:Pelvic Hiking`. So the graph names the observed direction and has no node
#      for the cited one.
#   4. SHIPPING THE OBSERVED DIRECTION INSTEAD WOULD BE A RULE WITH NO CITATION. Neither source
#      mentions pelvic hiking, in this exercise or any other. Sit-up withdrew a rule for a KG
#      seed that was semantically inverted; this is the mirror case -- the graph and the data
#      agree with each other and the citation points the other way -- and it is resolved the
#      same way, by not shipping.
#   5. AND THE OMISSION IS NOT FREE, WHICH IS SAID RATHER THAN GLOSSED. The rank correlation
#      between the shipped trunk-lean signal and this one is rho = 0.713 over 163 scored
#      repetitions: related, but not redundant. Had it come back above 0.9 the omission would
#      have cost nothing measurable. It did not, so a real detection opportunity is being
#      declined here and failure 2 has to carry that weight on its own.
PELVIC_TILT_DISJUNCT_NOT_IMPLEMENTED = True


# FROM THE SPEC: "Flag `insufficient_rom` if peak abduction angle < ~25-30deg for
# standing/side-lying abduction (target range commonly ~30-45deg)." Kept as a named constant even
# though the rule below never fires, because the design spec quotes the number it was measured
# against and a reader must be able to find it.
ROM_MILD_DEG = 30.0


def rule_insufficient_abduction_rom(
    core: list[CoreFrame], ctx: RuleContext
) -> list[PoseRuleDetection]:
    """Registered but PERMANENTLY SILENT -- always returns [].

    Insufficient abduction range is a real fault and this module has the best knowledge-graph
    seed of the three: `Leg Abduction:Insufficient Abduction Range` resolves with TWO non-empty
    buckets (causes: Weak Hip Abductors; quality_impacts: Hip Abduction). It is also the only one
    of the four parent-spec rules whose quantity is measured cleanly here -- the peak angle
    between the working thigh and the support limb, roll- and mirror-invariant, emitted as
    `{side}_abduction_deg`. Nothing about the SENSING fails.

    WHAT SILENCES IT IS THE THRESHOLD, AND THE LABELED DATA DECIDED IT RATHER THAN A JUDGEMENT
    CALL. Scored the way this rule would score -- lower peak abduction is worse -- the cue's AUC
    over 210 labeled repetitions is 0.206 pooled, and EVERY ONE of the 9 subjects lands below
    chance (0.000, 0.000, 0.200, 0.200, 0.218, 0.238, 0.306, 0.333, 0.347). It is not a weak
    signal; it is a signal pointing the wrong way, in every subject. In fire-rate terms, at the
    parent spec's own ~30 deg cut:

        fires on 39/93 (42%) of repetitions humans judged CORRECT
        fires on  8/70 (11%) of repetitions humans judged INCORRECT

    WHAT THAT DOES AND DOES NOT SAY. REHAB24-6's incorrect repetitions are performed incorrectly
    on request, so the SCARCITY of short repetitions among them is a fact about the protocol, not
    about the world. The finding used here is the other direction -- the cut fires preferentially
    on the CORRECT class -- which is a fact about the rule.

    AND NO SOURCE STATES A NUMBER TO REPLACE IT WITH. The parent spec says so itself -- "the
    specific degree threshold is a practical target, not a value stated in the source" -- and
    reading the source confirms it: González-de-la-Flor's only quantities for this exercise are
    EMG amplitudes (side-lying hip abduction "approximately 80% of maximal voluntary isometric
    contraction", standing hip abduction "high (60% MVIC)"), never a range of motion. Moving the
    cut to fit the data would be tuning, and this project's rules do not tune to labels.

    SILENT, NOT WITHDRAWN, AND THE DISTINCTION IS LOAD-BEARING HERE. The fault has an exact graph
    node, a primary sentence naming the exercise ("standing hip abduction ... allows resisted
    movement in the frontal plane"), and a working metric. What is missing is a cited threshold,
    which is a smaller gap than the two withdrawn rules below have. Contrast
    `bridge_lumbar_hyperextension`, which is silent because the SENSOR cannot see the quantity.

    OPEN, RECORDED, NOT RESOLVED: a per-user baseline -- "this repetition is shorter than your
    own usual" -- would need no literature threshold at all, and is the natural upgrade. The
    architecture has no cross-clip state, which is the same wall `situp_excessive_speed` hit.
    """
    return []


# BOTH of the parent spec's remaining Leg Abduction rules are ABSENT rather than silent, and the
# distinction is deliberate. A silent stub asserts "real fault, cited, and something else blocks
# it"; an absent rule asserts "no citation supports this as written".
#
# `abd_hip_flexion_er_substitution` -- WITHDRAWN, four independent failures, any one sufficient:
#   1. THE CITATION IS ABOUT BAND PLACEMENT IN A MONSTER WALK, NOT ABOUT A SUBSTITUTION PATTERN.
#      The parent spec's support for the external-rotation half quotes "distal band placement
#      introduces a slight external rotation torque"; read in place, that sentence is about where
#      to loop the band in a monster walk -- "The monster walk often specifically uses an ankle or
#      forefoot placement, which not only provides lateral resistance but also introduces a slight
#      external rotation torque (especially if the band is around the forefoot, tending to pull
#      the toes inward)". It describes a deliberate feature of a DIFFERENT exercise, not a fault.
#      The parent spec already grades this one MODERATE and calls the toes-up cue "inferred
#      clinical description"; reading the source downgrades it further.
#   2. THE HIP-FLEXION HALF RESTS ON BAND-WALK TECHNIQUE. "Maintaining frontal plane neutrality"
#      appears in the review's lateral-band-walk biomechanics section, one sentence after "a
#      slight forward trunk lean increases gluteus medius and maximus activation" and alongside
#      "optimal squat depth" -- band-walk posture, not a standing abduction repetition.
#   3. IT NEEDS A SAGITTAL VIEW OF A FRONTAL-PLANE EXERCISE. The parent spec rates it low/medium
#      and says the forward-drift component "needs a **side** view ... Not reliably separable from
#      true abduction on a **front** view alone". The app films one camera, and the view the other
#      shipped rule here needs is the frontal one.
#   4. THE KNOWLEDGE GRAPH HAS NO HOME FOR IT. `retrieve_graph_context("Hip Flexion
#      Substitution", movement="Leg Abduction")` returns the generic anatomy nodes `Hip` and
#      `Hip Flexion`, not a fault -- and this movement has exactly three fault nodes, none of them
#      this.
#   NOT SAID BY THIS WITHDRAWAL: that the leg drifting forward is fine. It is a real clinical
#   substitution. What is missing is a source that observes it in this exercise and a view that
#   can see it.
#
# `abd_momentum` -- WITHDRAWN, three independent failures:
#   1. HALF THE HEURISTIC DOES NOT EXIST IN THIS ARCHITECTURE. "Flag `momentum` if peak angular
#      velocity greatly exceeds a per-user baseline" -- there is no per-user baseline anywhere in
#      this pipeline; `run_detector` scores one clip with no cross-clip state. This is the same
#      wall `situp_excessive_speed` hit, and it is the SECOND disjunct-level defect of that kind.
#   2. THE SURVIVING DISJUNCT IS COMPUTABLE AND UNCITED. "The eccentric (return) phase is much
#      faster than the concentric" is measurable from this module's own phase labels, and it is
#      the one thing here that could have shipped. What backs it is "Proper execution requires
#      control of the trunk and pelvis, optimal squat depth, and consistent band tension" and
#      "Proper form (minimal pelvic sway, controlled steps) ensures the targeted muscles are
#      effectively engaged" -- both about band WALKS, and neither stating a ratio, a velocity or a
#      duration. The parent spec grades this MODERATE and admits "the specific velocity thresholds
#      are practical proxies".
#   3. NO GRAPH NODE. `retrieve_graph_context("Momentum", movement="Leg Abduction")` matches
#      `Anterior Momentum Generation` and `Forward Momentum`, both reached from OTHER movements'
#      subgraphs. There is no `Leg Abduction:` momentum fault.
#
# `LEG_ABDUCTION_METRIC_KEYS` must stay a two-way match with what `leg_abduction_compute_raw`
# emits (pinned by `test_metric_keys_match_the_emitted_metrics_exactly`): a key the tuple omits
# is dropped by `run_detector`, which builds each CoreFrame's metrics dict FROM this tuple, and
# read back as NaN by every rule.
LEG_ABDUCTION_DETECTOR = MovementDetector(
    "Leg Abduction",
    LEG_ABDUCTION_METRIC_KEYS,
    leg_abduction_compute_raw,
    leg_abduction_assign_phases,
    (
        rule_trunk_lean_compensation,
        rule_insufficient_abduction_rom,
    ),
    rep_signal="max_thigh_trunk_deg",
    # `max`: the effort peak of an abduction is the limb carried FURTHEST from the trunk axis,
    # so the signal peaks at its maximum. Shares polarity with Arm Abduction and Shoulder
    # Bridge; Row, Bicep Curl, Arm VW and Sit-up use `min`.
    rep_polarity="max",
    # `extended` names the end of the signal AWAY FROM the effort peak, which here is the leg
    # hanging under the hip in neutral stance -- where a repetition starts and ends. Only
    # Deadlift uses `flexed`.
    rep_start="extended",
)

registry.register(LEG_ABDUCTION_DETECTOR)
