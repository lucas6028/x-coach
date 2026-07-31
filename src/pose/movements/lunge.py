# Lunge raw metrics and phase segmentation. Fault rules land in Tasks 3-5.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `lunge_compute_raw` / `lunge_assign_phases`
# compute scale-free per-frame metrics and a phase label only. Every number that decides
# anything (a fire threshold, a severity ramp endpoint, a measurability gate) belongs in a
# `rule_*` function in a later task, not here. The only constant this module defines,
# `_DEGENERATE_LENGTH`, is a division-by-zero guard, never a tunable threshold -- see its
# docstring.
#
# ---------------------------------------------------------------------------------------
# BOTH LEGS, SYMMETRICALLY, RESOLVES NOTHING -- the design decision this module exists to
# encode.
# ---------------------------------------------------------------------------------------
# Unlike push-up's or OHP's raw metrics, which are already single-valued because the body is
# bilaterally symmetric in the movement they score, a lunge is a SPLIT STANCE: one leg is
# genuinely the "lead" (forward, loaded, the one every fault rule is about) and one is the
# "trailing" leg. It would be tempting for `lunge_compute_raw` to resolve that here and emit
# `lead_knee_angle`, `lead_knee_medial_offset_ratio`, etc. It deliberately does not.
#
# `run_detector` (src/pose/movements/base.py) calls `compute_raw` over the WHOLE CLIP, before
# `segment_reps` has split it into per-rep slices and before any rep's bottom frame is known.
# At metric time there is therefore no rep boundary to resolve "which leg is loaded THIS rep"
# against. A per-frame "whichever knee is more flexed right now" heuristic would:
#
#   1. Flicker through setup and recovery, where both knees sit near full extension and the
#      difference between them is landmark noise, not signal -- the "lead" leg would swap
#      randomly frame to frame in exactly the phases where it is least meaningful to ask.
#   2. Corrupt `centered_median` and any other frame-to-frame smoothing: a metric that means
#      "the left knee's offset" on frame 40 and "the right knee's offset" on frame 41 is not
#      one time series, it is two interleaved ones, and averaging across the swap produces a
#      number that describes neither leg.
#
# Lead-side resolution therefore happens in the RULES (Task 3's `resolve_lead_side`), which
# receive a PER-REP slice of `CoreFrame`s and can legitimately ask "which knee was most
# flexed at THIS rep's bottom frame" -- a question that has an answer only once a rep
# boundary exists. Until then, every side-specific metric here is emitted for BOTH legs,
# under `left_*` / `right_*` keys, and nothing chooses between them.
#
# ---------------------------------------------------------------------------------------
# ONE DROPPED LANDMARK SILENCES EVERY LUNGE RULE FOR THAT FRAME.
# ---------------------------------------------------------------------------------------
# `required` below lists both hips, both knees, both ankles, both foot indices and both
# shoulders. If `visible_point` drops any ONE of them the frame is marked `valid=False` and
# carries no metric keys at all -- so every rule that masks on `frame.valid` goes silent for
# that frame, not just the one whose input landmark went missing. This mirrors
# `pushup_compute_raw`'s and `ohp_compute_raw`'s validity gate (see pushup.py's MODULE-WIDE
# SILENCE RISK note): an unmeasurable frame is refused wholesale rather than degraded,
# because a silently-wrong verdict is worse than no verdict. Foot indices are required
# because `knee_forward_ratio` needs the toe-ankle vector for BOTH legs (Task 4 needs both
# sides even though only the lead leg's ratio ends up cited); shoulders because the trunk
# lean does.
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, angle_degrees, midpoint, mean_visibility,
    knee_forward_ratio, distance, contiguous_true_segments, severity_from_range,
)
from src.pose.movements.base import CoreFrame, RuleContext
from src.pose.pose_rule_detector import (
    KNEE_FORWARD_MILD,
    KNEE_FORWARD_SEVERE,
    SIDE_VIEW_CONF_THRESHOLD,
    VIEW_UNAVAILABLE_CONFIDENCE_SCALE,
    PoseRuleDetection,
    build_detection,
)

# Same generic "lower body" landmark set used across movements for the framework-level
# lower_body_visibility quality field. The NAME is squat-centric (inherited across every
# movement module, including the upper-body ones that carry it awkwardly), but the field it
# feeds (`CoreFrame.lower_body_visibility`) is genuinely lower-body FOR THIS MOVEMENT, unlike
# push-up's or OHP's use of the same name for an upper-body-dominant exercise.
LOWER_BODY_LANDMARKS = (
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_HEEL,
    RIGHT_HEEL,
    LEFT_FOOT_INDEX,
    RIGHT_FOOT_INDEX,
)

LUNGE_METRIC_KEYS: tuple[str, ...] = (
    "left_knee_angle",
    "right_knee_angle",
    "min_knee_angle",
    "left_knee_forward_ratio",
    "right_knee_forward_ratio",
    "left_knee_medial_offset_ratio",
    "right_knee_medial_offset_ratio",
    "pelvis_tilt_signed_deg",
    "trunk_lateral_lean_deg",
    "hip_width",
)

# Below this a length/normalizer is degenerate and the dependent metric is NaN. Same guard
# value pushup.py and overhead_press.py use; not a tunable threshold.
_DEGENERATE_LENGTH = 1e-6

# ---------------------------------------------------------------------------------------
# STEP 0 -- KG QUERY RESOLUTION, recorded before any rule was written (task-3-report.md has
# the full transcript). Each string below was checked against data/kg/sports_kg_v3.graphml via
# `src.knowledge.graph_retrieval.resolve_nodes` AND `retrieve_graph_context` (the latter is
# what production actually calls, and is what OHP's three-blank-queries defect would have
# been caught by) BEFORE being written here. All four resolve to exactly one `Lunge:`-scoped
# node with a non-empty causes/risks/evidence bucket -- none of the four faults has a gap to
# record.
#
# THESE ARE NOT THE BRIEF'S EXAMPLE STRINGS VERBATIM for two of the four -- both deviations
# are deliberate, not typos, and are load-bearing for Tasks 4/5 which import these constants
# blind:
#   - Depth: "Excessive Knee Flexion" (the brief's example candidate) resolves to a real node,
#     but that node's only edge is `INCREASES_RISK_OF -> Achilles Tendon Injury` -- the WRONG
#     direction of fault (too much flexion, not too little) and the wrong injury. The spec's
#     own citation_support for `lunge_insufficient_depth` says "reduced knee flexion/extensor
#     moment marks impaired (non-coper) function" -- i.e. the fault is REDUCED flexion.
#     "Decreased Knee Flexion" resolves to the one node whose edges actually match that
#     sentence: `CAUSED_BY <- Weak Quadriceps`, `INCREASES_RISK_OF -> ACL Injury`.
#   - Pelvic drop: none of the brief's four candidates for this fault ("Anterior Trunk Tilt",
#     "Poor Dynamic Stability", "Knee Anterior Displacement", "Compensatory Trunk Lean") is the
#     best match once the graph is actually read. "Trendelenburg Posture" is literally named in
#     the spec's fault_name, and its citation_support quotes "observed as a Trendelenburg
#     posture, with the contralateral pelvis dropping" almost verbatim against the node's own
#     `INDICATED_BY -> Contralateral Pelvis Drop` and `CAUSED_BY -> Weak Hip Abductors` edges.
#   - Past-toes and valgus: "Knee Anterior To Toes" and "Knee Valgus" both matched a brief
#     candidate AND carried rich, on-topic buckets (patellar tendon stress / tendinopathy risk;
#     ACL/cartilage/patellofemoral risk), so no deviation was needed for either.
LUNGE_PAST_TOES_KG_QUERY = "Knee Anterior To Toes"
LUNGE_VALGUS_KG_QUERY = "Knee Valgus"
LUNGE_DEPTH_KG_QUERY = "Decreased Knee Flexion"
LUNGE_PELVIC_DROP_KG_QUERY = "Trendelenburg Posture"

# Phases in which a lunge is under load. RULE-LEVEL CHOICE, not a spec quantity: the parent
# spec scopes only `lunge_knee_past_toes` to phases ("during descent/bottom/ascent") and
# scopes the other three to none. Applying that same set to Tasks 4/5's rules follows the squat
# detector's ACTIVE_PHASES precedent (src/pose/movements/squat.py) rather than a spec
# requirement. Cost, stated: a fault that appears only during `setup` or `recovery` is missed.
#
# `rule_insufficient_depth` (this task) does NOT use this set -- see its docstring. Its
# predicate is "the rep's MINIMUM lead-knee angle", a single per-rep number, and masking on
# `descent`/`ascent` as well as `bottom` would catch the ordinary >100-degree transit every
# rep makes on the way down and back up, firing on reps that bottom out perfectly deep. That
# is a real defect this task found empirically (a synthetic 170->85->170 trajectory produced
# two severity-1.0 detections before the fix) and corrected by narrowing the depth mask to
# `phase == "bottom"` alone, matching squat.rule_shallow_depth's identical narrowing for the
# identical reason.
LUNGE_ACTIVE_PHASES = {"descent", "bottom", "ascent"}

# Minimum left/right knee-angle difference at the bottom before a lead leg is claimed.
# RULE-LEVEL CHOICE -- the parent spec defines the lead leg ("the more flexed / more anterior
# foot") but names no separation below which the answer is unsafe. 5 degrees is chosen as the
# scale at which a landmark-noise-driven difference could flip the answer; below it the two
# legs are doing the same thing, which is not a lunge. This constant can ONLY SILENCE: an
# unresolved lead side emits no detections at all, never a guessed one. A coin-flip here would
# mis-attribute every fault in the rep to the wrong leg, which is worse than saying nothing.
LEAD_SIDE_MIN_SEPARATION_DEG = 5.0

# Views in which the parent spec rates lunge depth `high` ("high on side / front_oblique;
# medium head-on"). Defined locally rather than imported from pushup.py: the two modules
# happen to agree today but answer different spec lines and must be free to diverge.
DEPTH_OBSERVABLE_VIEWS = {"side", "front_oblique"}

# Views in which the parent spec rates the frontal-plane cues `high`. Matches the set
# `squat.rule_knees_inward` already uses for the same fault family.
ALIGNMENT_OBSERVABLE_VIEWS = {"front", "front_oblique", "rear", "rear_oblique"}

# Confidence multiplier applied when a rule fires from a view the spec does not rate `high`.
# Not a new number: aliases the same constant squat/OHP/push-up already share
# (`pose_rule_detector.VIEW_UNAVAILABLE_CONFIDENCE_SCALE`, currently 0.65) rather than
# re-typing its value, so a future change to that shared constant does not silently diverge
# from this module.
_OFF_VIEW_CONFIDENCE = VIEW_UNAVAILABLE_CONFIDENCE_SCALE

# FROM THE SPEC: "Flag when the minimum lead-knee angle across the rep > 100 degrees.
# Severity ramp 100 degrees -> 130 degrees (more extended = worse)."
LUNGE_DEPTH_MILD_DEG = 100.0
LUNGE_DEPTH_SEVERE_DEG = 130.0


def _medial_offset_ratio(
    points, hip_index: int, knee_index: int, ankle_index: int, mid_hip, hip_width: float
) -> float:
    """Signed offset of one knee from its own hip->ankle line, POSITIVE = toward the mid-hip.

    The frontal-plane knee-abduction proxy the spec asks for ("signed medial offset of the
    knee from the hip-ankle line, normalised by hip width"). No true 3-D abduction angle is
    recoverable from monocular pose, and none is claimed.

    WHY THIS IS FACING-INDEPENDENT, which is what lets the rule avoid gating on `front` /
    `front_oblique` (unreachable in production under allow_front=False): "medial" is defined
    as "toward the mid-hip", and the mid-hip is the midline whether the camera is in front of
    or behind the subject. Nothing here consults `signed_orientation`.
    """
    hip = visible_point(points, hip_index, dims=2)
    knee = visible_point(points, knee_index, dims=2)
    ankle = visible_point(points, ankle_index, dims=2)
    if hip is None or knee is None or ankle is None or mid_hip is None:
        return np.nan
    if not np.isfinite(hip_width) or hip_width <= _DEGENERATE_LENGTH:
        return np.nan

    leg = np.asarray(ankle, dtype=np.float64) - np.asarray(hip, dtype=np.float64)
    leg_length = float(np.linalg.norm(leg))
    if leg_length <= _DEGENERATE_LENGTH:
        return np.nan
    normal = np.asarray([-leg[1], leg[0]], dtype=np.float64) / leg_length

    # Orient the normal toward the midline, so a positive projection means "medial" for
    # whichever leg this is -- the left and right legs point in opposite image-x directions.
    toward_midline = np.asarray(mid_hip, dtype=np.float64) - np.asarray(hip, dtype=np.float64)
    if float(np.dot(normal, toward_midline)) < 0.0:
        normal = -normal

    offset = float(np.dot(np.asarray(knee, dtype=np.float64) - np.asarray(hip, dtype=np.float64), normal))
    return offset / float(hip_width)


def lunge_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
    raw: list[dict] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raw.append({"valid": False})
            continue

        points = landmarks_to_array(frame.get("landmarks"))
        frame_index = int(frame.get("frame_index", 0) or 0)
        time = frame_index / fps if fps > 0 else 0.0
        # Foot indices are required because `knee_forward_ratio` needs the toe-ankle vector;
        # shoulders because the trunk lean does. See the module docstring: one dropped
        # landmark silences EVERY lunge rule for this frame, not just the dependent one.
        required = (
            LEFT_SHOULDER, RIGHT_SHOULDER,
            LEFT_HIP, RIGHT_HIP,
            LEFT_KNEE, RIGHT_KNEE,
            LEFT_ANKLE, RIGHT_ANKLE,
            LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
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

        left_knee_angle = angle_degrees(points, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
        right_knee_angle = angle_degrees(points, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
        finite_knees = [v for v in (left_knee_angle, right_knee_angle) if np.isfinite(v)]
        min_knee_angle = float(min(finite_knees)) if finite_knees else np.nan

        hip_width = distance(points, LEFT_HIP, RIGHT_HIP)
        mid_hip = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)

        left_hip = visible_point(points, LEFT_HIP, dims=2)
        right_hip = visible_point(points, RIGHT_HIP, dims=2)
        # atan2 over |dx|: the magnitude of the horizontal hip separation, never its sign.
        # Using signed dx would flip the whole angle by 180 degrees when the subject turns
        # around, making the metric mean "which hip is lower" only for one facing.
        if left_hip is not None and right_hip is not None:
            dx = abs(float(right_hip[0] - left_hip[0]))
            dy = float(right_hip[1] - left_hip[1])
            pelvis_tilt_signed_deg = (
                float(np.degrees(np.arctan2(dy, dx))) if dx > _DEGENERATE_LENGTH else np.nan
            )
        else:
            pelvis_tilt_signed_deg = np.nan

        if shoulder_mid is not None and mid_hip is not None:
            lean_dy = abs(float(shoulder_mid[1] - mid_hip[1]))
            lean_dx = float(shoulder_mid[0] - mid_hip[0])
            trunk_lateral_lean_deg = (
                float(np.degrees(np.arctan2(lean_dx, lean_dy))) if lean_dy > _DEGENERATE_LENGTH else np.nan
            )
        else:
            trunk_lateral_lean_deg = np.nan

        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "left_knee_angle": left_knee_angle,
                "right_knee_angle": right_knee_angle,
                "min_knee_angle": min_knee_angle,
                "left_knee_forward_ratio": knee_forward_ratio(
                    points, LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX
                ),
                "right_knee_forward_ratio": knee_forward_ratio(
                    points, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX
                ),
                "left_knee_medial_offset_ratio": _medial_offset_ratio(
                    points, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, mid_hip, hip_width
                ),
                "right_knee_medial_offset_ratio": _medial_offset_ratio(
                    points, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, mid_hip, hip_width
                ),
                "pelvis_tilt_signed_deg": pelvis_tilt_signed_deg,
                "trunk_lateral_lean_deg": trunk_lateral_lean_deg,
                "hip_width": hip_width,
            }
        )
    return raw


def lunge_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> descent -> bottom -> ascent, segmented on `min_knee_angle`.

    Mirrors `pushup_assign_phases` (src/pose/movements/pushup.py) and `ohp_assign_phases`
    (src/pose/movements/overhead_press.py), substituting `min_knee_angle` for
    `min_elbow_angle` as the depth signal -- the more-flexed (smaller) of the two knee
    angles is the lunge's depth analogue of the more-flexed elbow. Same fallbacks: an empty
    clip returns an empty list, a clip with no finite depth signal is entirely `unknown`, and
    an invalid frame is `unknown` regardless of where it sits (the validity check precedes
    the setup cutoff, so an occluded frame in the opening 15% is NOT labelled `setup`)."""
    frame_count = len(raw)
    if frame_count == 0:
        return []

    knee_values = np.asarray(
        [float(item.get("min_knee_angle", np.nan)) for item in raw], dtype=np.float32
    )
    valid_knee = knee_values[np.isfinite(knee_values)]
    if valid_knee.size == 0:
        return ["unknown" for _ in raw]

    # The deepest 30% of the rep by knee flexion is the bottom.
    bottom_threshold = float(np.percentile(valid_knee, 30))
    deepest_index = int(np.nanargmin(np.where(np.isfinite(knee_values), knee_values, np.inf)))
    setup_cutoff = max(1, int(frame_count * 0.15))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append("setup")
            continue

        value = knee_values[index]
        if np.isfinite(value) and value <= bottom_threshold:
            phases.append("bottom")
        elif index < deepest_index:
            phases.append("descent")
        else:
            phases.append("ascent")
    return phases


def resolve_lead_side(window: list[CoreFrame]) -> str | None:
    """Which leg led this repetition: "left", "right", or None when unresolvable.

    SUBSTITUTION, NOT A RESTATEMENT -- record it as one. The parent spec defines the lead leg
    as "the more flexed / more anterior foot". `more anterior` is exactly the axis that
    collapses in a frontal view, which is where two of the four lunge rules live, so the
    anterior half of that definition is unusable where it is most needed. This uses the
    more-flexed half only, evaluated at the window's bottom frame.

    WHY THIS LIVES IN THE RULES AND NOT IN `lunge_compute_raw`: `run_detector` calls
    `compute_raw` over the WHOLE CLIP before `segment_reps`, so at metric time there is no rep
    boundary and therefore no bottom frame. A per-frame "whichever knee is more flexed right
    now" flickers through `setup` and `recovery`, where both knees sit near extension within
    landmark noise of each other; every lead-relative quantity would then swap legs mid-clip
    and `centered_median` would blend two legs into a number describing neither. Rules receive
    a per-rep slice (`run_detector` slices `core[rep.start:rep.end + 1]`), which is the first
    place the question is answerable.

    On a fallback path (`no_reps_detected`, `only_partial_reps`, `segmentation_disabled`) the
    rules receive the whole clip, so `window` is the whole clip and this resolves once for it.
    That degrades exactly as everything else on the fallback path does; it is stated, not
    hidden.

    NOT PHASE-SCOPED: this scans every frame in `window`, including `setup`/`recovery` ones,
    because it is only choosing which frame is deepest (via `min_knee_angle`) -- setup/recovery
    frames simply lose that competition to the true bottom in a real rep. Scoping the scan to
    `LUNGE_ACTIVE_PHASES` would be redundant on a real rep and actively wrong on the fallback
    whole-clip path, where phase labels are far less reliable across multiple reps.
    """
    bottom: CoreFrame | None = None
    bottom_value = np.inf
    for frame in window:
        if not frame.valid:
            continue
        value = frame.m("min_knee_angle")
        if np.isfinite(value) and value < bottom_value:
            bottom_value, bottom = value, frame
    if bottom is None:
        return None

    left = bottom.m("left_knee_angle")
    right = bottom.m("right_knee_angle")
    if not np.isfinite(left) or not np.isfinite(right):
        return None
    if abs(left - right) < LEAD_SIDE_MIN_SEPARATION_DEG:
        return None
    return "left" if left < right else "right"


def rule_insufficient_depth(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag a lunge whose lead knee never reaches roughly a right angle.

    THRESHOLD PROVENANCE -- BOTH FROM THE SPEC, unlike every push-up rule. The parent spec's
    Lunge entry states the fire threshold ("Flag when the minimum lead-knee angle across the
    rep > 100 degrees") AND the ramp ("Severity ramp 100 degrees -> 130 degrees (more extended
    = worse)"). Neither number is chosen here.

    The lead side is resolved over THIS window (see `resolve_lead_side`); an unresolved side
    emits nothing.

    WHY THE MASK IS `phase == "bottom"`, NOT `phase in LUNGE_ACTIVE_PHASES` (unlike every other
    lunge rule): the spec's predicate is "the MINIMUM lead-knee angle ACROSS THE REP" -- one
    number per rep, not a per-frame gate. On a real rep the lead knee travels roughly
    170 -> 85 -> 170, so an `ACTIVE_PHASES` mask (descent/bottom/ascent) would catch the long
    transit through >100 degrees on BOTH the way down and the way back up, even for a rep that
    bottoms out at a perfectly good 85 degrees -- `contiguous_true_segments` would then emit
    two detections at severity 1.0 (max_angle near full extension) for a clean rep. `bottom` is
    exactly the deepest 30% of the rep (`lunge_assign_phases`), so "the angle is above 100
    during `bottom`" is the frame-level statement equivalent to "the rep's minimum exceeds
    100" -- the same substitution `squat.rule_shallow_depth` makes for the identical reason.
    """
    lead = resolve_lead_side(core)
    if lead is None:
        return []
    lead_key = f"{lead}_knee_angle"
    observable = ctx.view_type in DEPTH_OBSERVABLE_VIEWS

    mask = [
        frame.valid
        and frame.phase == "bottom"
        and np.isfinite(frame.m(lead_key))
        and frame.m(lead_key) > LUNGE_DEPTH_MILD_DEG
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        angles = [frame.m(lead_key) for frame in segment]
        max_angle = float(np.nanmax(angles))
        severity = severity_from_range(
            max_angle, LUNGE_DEPTH_MILD_DEG, LUNGE_DEPTH_SEVERE_DEG, lower_is_worse=False
        )
        detections.append(
            build_detection(
                fault_id="lunge_insufficient_depth",
                fault_name="Insufficient Depth",
                kg_query=LUNGE_DEPTH_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=angles,
                severity=severity,
                confidence=severity * (1.0 if observable else _OFF_VIEW_CONFIDENCE),
                observability="high" if observable else "medium",
                evidence={
                    "lead_side": lead,
                    "max_lead_knee_angle_deg": round(max_angle, 2),
                    "threshold": LUNGE_DEPTH_MILD_DEG,
                    "primary_label": "lead knee angle",
                    "primary_value": round(max_angle, 2),
                    "primary_threshold": LUNGE_DEPTH_MILD_DEG,
                },
                citation="Alkjær T, et al. \"Forward lunge before and after anterior cruciate "
                         "ligament reconstruction.\" PLoS One (2020), PMC6980669. Supplemented by "
                         "Escamilla R, et al. \"Patellofemoral Joint Loading During the "
                         "Performance of the Forward and Side Lunge with Step Height Variations.\" "
                         "IJSPT (2022), PMC8805090.",
                citation_support="PMC6980669 defines the protocol as \"flexing the knee to 90°\" "
                                 "as the target depth, and reduced knee flexion/extensor moment "
                                 "marks impaired (non-coper) function. PMC8805090: \"patellofemoral "
                                 "joint force and stress generally increased progressively as knee "
                                 "flexion increased during the descent phase\" — i.e., depth is "
                                 "what produces the loading/strengthening stimulus. Verified in "
                                 "RAG docs.",
            )
        )
    return detections


def rule_knee_past_toes(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag the lead knee translating well in front of the toes.

    THRESHOLD PROVENANCE: the parent spec's Lunge entry says "Flag when > 0.10 during
    descent/bottom/ascent; severe >= 0.30" -- word-for-word the Squat entry's wording, which
    this repo already reads as a 0.10 -> 0.30 ramp via KNEE_FORWARD_MILD / KNEE_FORWARD_SEVERE
    in src/pose/pose_rule_detector.py. Those constants are IMPORTED here rather than restated,
    so the two movements cannot drift apart. No new number is introduced.

    HARD VIEW GATE, not a downgrade. The spec rates this `high` on `side` and `low` head-on
    ("sagittal knee travel not resolvable"). `squat.rule_knees_forward` sets the precedent:
    outside a confidently-classified `side` view the rule emits NOTHING rather than a
    low-confidence claim, because the projection that produces the number is the thing that
    has failed. SIDE_VIEW_CONF_THRESHOLD is the same 0.20 floor squat already applies.
    """
    lead = resolve_lead_side(core)
    if lead is None:
        return []
    observable_side = (
        ctx.view_type == "side" and ctx.view_confidence >= SIDE_VIEW_CONF_THRESHOLD
    )
    lead_key = f"{lead}_knee_forward_ratio"

    mask = [
        frame.valid
        and frame.phase in LUNGE_ACTIVE_PHASES
        and observable_side
        and np.isfinite(frame.m(lead_key))
        and frame.m(lead_key) > KNEE_FORWARD_MILD
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        ratios = [frame.m(lead_key) for frame in segment]
        max_ratio = float(np.nanmax(ratios))
        severity = severity_from_range(
            max_ratio, KNEE_FORWARD_MILD, KNEE_FORWARD_SEVERE, lower_is_worse=False
        )
        detections.append(
            build_detection(
                fault_id="lunge_knee_past_toes",
                fault_name="Lead Knee Past Toes / Anterior Knee Translation",
                kg_query=LUNGE_PAST_TOES_KG_QUERY,   # resolved in Task 3 Step 0
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=ratios,
                severity=severity,
                confidence=severity,
                observability="high",
                evidence={
                    "lead_side": lead,
                    "max_knee_forward_ratio": round(max_ratio, 4),
                    "threshold": KNEE_FORWARD_MILD,
                    "primary_label": "lead knee past toes",
                    "primary_value": round(max_ratio, 4),
                    "primary_threshold": KNEE_FORWARD_MILD,
                },
                citation="Zellmer M, et al. \"Patellar tendon stress between two variations of "
                         "the forward step lunge.\" J Sport Health Sci (2019). PMC6523035.",
                citation_support="Knee-in-front-of-toes lunges (FSL-FT) vs knee-behind-toes "
                                 "(FSL-BT) gave \"peak patellar tendon stress … 11.1% greater,\" "
                                 "stress impulse \"18.8% greater,\" peak quadriceps force 12.6% "
                                 "greater, peak knee-extension moment 25.8% greater, and peak "
                                 "knee flexion 110.2°→124.7° (all p<0.001; Table 1). Verified in "
                                 "RAG doc.",
            )
        )
    return detections


# FROM THE SPEC: "Flag when medial offset > ~0.10 * hip_width toward the midline;
# ramp 0.10 -> 0.25."
LUNGE_VALGUS_MILD = 0.10
LUNGE_VALGUS_SEVERE = 0.25


def rule_knee_valgus(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag the lead knee caving medially relative to its own hip-ankle line.

    THRESHOLD PROVENANCE: both numbers FROM THE SPEC (fire > 0.10 of hip width, ramp
    0.10 -> 0.25). This is a frontal-plane knee-abduction PROXY; monocular pose yields no true
    3-D abduction angle and none is claimed.

    OBSERVABILITY DOWNGRADE, NOT A GATE -- and deliberately not gated on `front`. The
    production path calls estimate_view_for_pose(allow_front=False), so `front` and
    `front_oblique` are never emitted downstream; a rule gated positively on them would be
    PERMANENTLY SILENT, which is what happened to `pushup_elbow_flare`. This rule does not need
    them: `_medial_offset_ratio` defines medial as "toward the mid-hip", and the mid-hip is
    the midline from in front of the subject or behind. So `rear`/`rear_oblique` -- the labels
    production actually reaches -- earn the same `high` rating, matching
    `squat.rule_knees_inward`, which resolves the same fault family the same way.

    KNOWN CONTAMINATION, NOT CORRECTED HERE -- carried over verbatim from the projection facts
    in tests/test_lunge.py::lunge_frame, and it does not let the `high` rating above be read as
    a claim of cleanliness. A knee's perpendicular displacement from its hip-ankle line is the
    sum of its MEDIAL travel and its ANTERIOR travel projected into the image. In a true
    frontal view the anterior component projects onto the leg line and vanishes, leaving the
    proxy clean -- but `front` is exactly the label production can never emit. In the oblique
    views it does reach, a deep, perfectly-tracked lunge produces a positive reading with no
    valgus present (pinned by test_anterior_knee_travel_contaminates_the_valgus_proxy).
    Separating the two needs a depth estimate this pipeline does not have, so the limitation is
    documented rather than corrected, and Phase 2 checks whether firing tracks step depth
    rather than correctness.
    """
    lead = resolve_lead_side(core)
    if lead is None:
        return []
    observable = ctx.view_type in ALIGNMENT_OBSERVABLE_VIEWS
    lead_key = f"{lead}_knee_medial_offset_ratio"

    mask = [
        frame.valid
        and frame.phase in LUNGE_ACTIVE_PHASES
        and np.isfinite(frame.m(lead_key))
        and frame.m(lead_key) > LUNGE_VALGUS_MILD
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        offsets = [frame.m(lead_key) for frame in segment]
        max_offset = float(np.nanmax(offsets))
        severity = severity_from_range(
            max_offset, LUNGE_VALGUS_MILD, LUNGE_VALGUS_SEVERE, lower_is_worse=False
        )
        detections.append(
            build_detection(
                fault_id="lunge_knee_valgus",
                fault_name="Lead Knee Valgus / Medial Collapse",
                kg_query=LUNGE_VALGUS_KG_QUERY,   # resolved in Task 3 Step 0
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=offsets,
                severity=severity,
                confidence=severity * (1.0 if observable else _OFF_VIEW_CONFIDENCE),
                observability="high" if observable else "medium",
                evidence={
                    "lead_side": lead,
                    "max_medial_offset_ratio": round(max_offset, 4),
                    "threshold": LUNGE_VALGUS_MILD,
                    "primary_label": "lead knee medial offset",
                    "primary_value": round(max_offset, 4),
                    "primary_threshold": LUNGE_VALGUS_MILD,
                },
                citation="Ford KR, et al. \"An evidence-based review of hip-focused "
                         "neuromuscular exercise interventions to address dynamic lower "
                         "extremity valgus.\" PMC4556293 (2015).",
                citation_support="\"knee abduction moment … was a significant predictor for "
                                 "future ACL injury risk with 73% sensitivity and 78% "
                                 "specificity\"; \"the inability to eccentrically control hip "
                                 "adduction and internal rotation may lead to greater dynamic "
                                 "lower extremity valgus commonly seen during landing, "
                                 "squatting, and running.\" Verified in RAG doc.",
            )
        )
    return detections


# FROM THE SPEC, unlike the other three rules' scopes: the pelvic-drop entry says the tilt is
# flagged "sustained through bottom/ascent". `descent` is deliberately excluded -- this is the
# spec's own scoping, not the rule-level call that LUNGE_ACTIVE_PHASES represents.
PELVIC_DROP_PHASES = {"bottom", "ascent"}

# FROM THE SPEC: "Flag when pelvis_tilt_deg > 8 degrees (contralateral hip lower) sustained
# through bottom/ascent; ramp 8 -> 20."
LUNGE_PELVIC_TILT_MILD_DEG = 8.0
LUNGE_PELVIC_TILT_SEVERE_DEG = 20.0


def rule_pelvic_drop(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag the NON-lead-side pelvis dropping -- the Trendelenburg signature of hip-abductor
    insufficiency on the lead leg.

    THRESHOLD PROVENANCE: both numbers FROM THE SPEC (fire > 8 degrees, ramp 8 -> 20).

    PHASE SCOPE, FROM THE SPEC, NOT LUNGE_ACTIVE_PHASES. The spec's own words are "sustained
    through bottom/ascent" -- `descent` is not in that list, unlike every other lunge rule,
    which uses the rule-level `LUNGE_ACTIVE_PHASES` (descent/bottom/ascent). `PELVIC_DROP_PHASES`
    is defined next to it as its own set so the two cannot silently drift together.

    SIGN, and why it takes two facts to get right. `pelvis_tilt_signed_deg` is positive when
    the RIGHT hip is lower, in a fixed convention that does not depend on which way the
    subject faces (it is built on |dx|, never signed dx). "Contralateral" then depends on the
    lead leg: a LEFT-lead lunge drops the RIGHT hip (positive), a RIGHT-lead lunge drops the
    LEFT hip (negative). Reading the magnitude alone would report an IPSILATERAL drop -- a
    different postural fault -- as Trendelenburg and invert the coaching cue.

    OBSERVABILITY CEILING IS `medium`, NEVER `high` -- FROM THE SPEC ("observability: medium on
    front/rear"). This differs from `rule_insufficient_depth` and `rule_knee_valgus`, whose
    ceiling is `high`; this fault's spec entry names no rating above `medium` at all, so
    `observable` selects `medium` rather than `high`. Off the alignment-observable views the
    rule downgrades further to `low` -- a RULE-LEVEL rating, not a number the spec names for
    that case (the spec states no rating below `medium`, only that the fault is unavailable
    off-axis) -- mirroring `squat.rule_heel_rise`'s precedent of a rule-level `low` where the
    spec is silent on the off-view number. `low` is also the sort key `run_detector` demotes
    behind every other observability via `(observability == "low", -severity, start_frame)`;
    that demotion is the intended consequence of choosing `low` here, not a side effect.

    SILENT FROM `side`: the spec rates this "not observable from a pure side view". A
    frontal-plane tilt has no meaning in the sagittal projection, so this is silence rather
    than a discounted claim, following `rule_knee_past_toes`'s reasoning in the mirror image.

    KNOWN MEASUREMENT BIAS, NOT CORRECTED HERE. In a frontal view of a SPLIT STANCE the
    L_hip -> R_hip vector is rotated in the transverse plane, so its image projection shortens
    and atan2(dy, |dx|) INFLATES the apparent tilt -- the deeper the lunge, the worse. The
    expected failure mode is therefore FALSE POSITIVES on deep, correctly-performed reps, not
    silence. Correcting it would require a depth estimate this pipeline does not have, so it
    is documented rather than papered over; Phase 2 reads specificity on correct reps first
    for exactly this reason.
    """
    lead = resolve_lead_side(core)
    if lead is None:
        return []
    if ctx.view_type == "side":
        return []
    observable = ctx.view_type in ALIGNMENT_OBSERVABLE_VIEWS
    # Left lead -> the contralateral (right) hip dropping is a POSITIVE tilt; right lead -> negative.
    sign = 1.0 if lead == "left" else -1.0

    mask = [
        frame.valid
        and frame.phase in PELVIC_DROP_PHASES
        and np.isfinite(frame.m("pelvis_tilt_signed_deg"))
        and sign * frame.m("pelvis_tilt_signed_deg") > LUNGE_PELVIC_TILT_MILD_DEG
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        drops = [sign * frame.m("pelvis_tilt_signed_deg") for frame in segment]
        max_drop = float(np.nanmax(drops))
        severity = severity_from_range(
            max_drop, LUNGE_PELVIC_TILT_MILD_DEG, LUNGE_PELVIC_TILT_SEVERE_DEG, lower_is_worse=False
        )
        detections.append(
            build_detection(
                fault_id="lunge_pelvic_drop",
                fault_name="Pelvic Drop / Contralateral Trunk Lean (Trendelenburg)",
                kg_query=LUNGE_PELVIC_DROP_KG_QUERY,   # resolved in Task 3 Step 0
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=drops,
                severity=severity,
                confidence=severity * (1.0 if observable else _OFF_VIEW_CONFIDENCE),
                observability="medium" if observable else "low",
                evidence={
                    "lead_side": lead,
                    "max_contralateral_drop_deg": round(max_drop, 2),
                    "threshold": LUNGE_PELVIC_TILT_MILD_DEG,
                    "primary_label": "contralateral pelvic drop",
                    "primary_value": round(max_drop, 2),
                    "primary_threshold": LUNGE_PELVIC_TILT_MILD_DEG,
                },
                citation="Ford KR, et al. PMC4556293 (2015). Cross-support: Alkjær T, et al. "
                         "PMC6980669 (2020).",
                citation_support="PMC4556293: \"Failure to produce the abduction force is "
                                 "observed as a Trendelenburg posture, with the contralateral "
                                 "pelvis dropping,\" and hip-focused training reduced "
                                 "\"ipsilateral trunk inclination, and contralateral pelvis "
                                 "depression during a single leg squat.\" PMC6980669 found "
                                 "gluteus medius EMG \"significantly higher for the ACL injured "
                                 "participants … possibly a compensatory mechanism to control "
                                 "the trunk and pelvis in the frontal plane.\" Verified in RAG "
                                 "docs.",
            )
        )
    return detections
