# Deadlift raw metrics, phase segmentation, and the three surviving fault rules
# (`rule_hips_shoot_up`, `rule_incomplete_lockout`, `rule_lumbar_flexion`), registered as
# `DEADLIFT_DETECTOR` at the foot of the file.
#
# THE METRIC LAYER CONTAINS NO THRESHOLDS -- `deadlift_compute_raw` / `deadlift_assign_phases`
# compute scale-free per-frame metrics and a phase label only. Every number that decides
# anything belongs in a `rule_*` function, not here. `_DEGENERATE_LENGTH` is a
# division-by-zero guard, never a tunable threshold.
#
# ---------------------------------------------------------------------------------------
# THE REP STARTS FLEXED, AND THAT IS WHY A SETUP BASELINE MEANS ANYTHING HERE.
# ---------------------------------------------------------------------------------------
# `DEADLIFT_DETECTOR` sets `rep_start="flexed"` (Task 5) -- the hook `base.py:55` names
# deadlift as the motivating case for. A rep therefore runs floor -> lockout -> floor, so
# ON THE SEGMENTED PATH the window's OPENING frames are genuinely the bar-on-the-floor setup.
# Two rules (`rule_hips_shoot_up`, `rule_lumbar_flexion`) reference a per-rep setup baseline,
# which is only meaningful because of this. For a movement whose rep starts standing, the same
# baseline would be measuring the wrong end of the lift.
#
# THAT GUARANTEE HOLDS ONLY WHEN `segment_reps` SUCCEEDS, AND `run_detector` HAS THREE WAYS TO
# FALL BACK. On `segmentation_disabled`, `no_reps_detected` or `only_partial_reps`
# (`base.py:159`) it phases the WHOLE CLIP in one pass (`base.py:182`) and runs every rule over
# it (`base.py:214`). `deadlift_assign_phases` labels the first 10% of whatever it is handed
# `setup` POSITIONALLY, without ever inspecting `hip_angle_deg` -- so on a fallback run `setup`
# is the first 10% of the CLIP, which may be the lifter standing around before walking up to
# the bar, not the setup at all. `setup_baseline` then returns a STANDING torso.
#
# The consequence is concrete, not theoretical, and it is worst for `rule_hips_shoot_up`: with a
# baseline of ~7 degrees instead of ~60, its `torso_pitch_deg > baseline` clause is satisfied by
# every loaded frame and contributes nothing, so the rule degenerates to its bare 55-degree
# absolute gate -- and that clause is precisely what the design spec's section 4.1 identifies as
# THE discriminator between the sequencing fault and a lifter who merely sets up flat.
# Reproduced on the production path with `DEADLIFT_DETECTOR` unmodified: a trimmed clip yielding
# `fallback=only_partial_reps` fired `deadlift_hips_shoot_up` at severity 0.2821 on a
# well-executed rep whose trunk pitch decreased monotonically, with
# `setup_torso_pitch_deg: 6.84`. `rule_lumbar_flexion` escapes the same corrupted baseline only
# incidentally, because its `_hips_still` term happens to reject travelling hips.
#
# NOT FIXED HERE, deliberately. Threading `fallback` into `RuleContext` so setup-relative rules
# can abstain is a framework change that touches squat and push-up too, and a plausibility gate
# on the baseline would be exactly the unsourced threshold this module forbids. Disclosed rather
# than papered over -- the same convention this module already uses for the `0.0` evidence
# sentinel. Recorded in the parent spec's section 7.
#
# The window also contains the ECCENTRIC. The parent spec's four phases cover only the
# concentric, so a fifth phase `lowering` exists here; without it, return-to-floor frames
# would be labelled `lockout` and `rule_incomplete_lockout` would score the descent.
# `lowering` is excluded from `DEADLIFT_ACTIVE_PHASES`: no rule has literature backing for a
# claim about the eccentric.
#
# ---------------------------------------------------------------------------------------
# EVERY METRIC IS BUILT FROM MIDPOINTS, AND EVERY RULE WANTS A SAGITTAL VIEW.
# ---------------------------------------------------------------------------------------
# Parent spec section 7 item 3 records that `_visible_midpoint` needs BOTH landmarks of a
# pair above 0.35 visibility, and that one occluded shoulder silently reverts body-extent
# measurement to a vertical fallback -- "exactly in the view most likely to trigger it: a
# sagittal (side) view is precisely where far-side landmarks are most often occluded." This
# detector sits squarely in that failure mode. `required` below therefore refuses the frame
# wholesale when any input landmark is missing, matching lunge/pushup/OHP: an unmeasurable
# frame is refused rather than degraded, because a silently-wrong verdict is worse than none.
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, midpoint, mean_visibility,
    line_angle_from_vertical, contiguous_true_segments, severity_from_range,
)
from src.pose.movements.base import CoreFrame, MovementDetector, RuleContext
from src.pose.movements import registry
from src.pose.pose_rule_detector import (
    SIDE_VIEW_CONF_THRESHOLD,
    VIEW_UNAVAILABLE_CONFIDENCE_SCALE,
    PoseRuleDetection,
    build_detection,
)

_OFF_VIEW_CONFIDENCE = VIEW_UNAVAILABLE_CONFIDENCE_SCALE

# Views in which a sagittal angle reads at full confidence. Per parent spec section 7 item 2,
# `front_oblique` is unreachable in the production path (`allow_front=False`), so in practice
# this is `side`; it is listed because the spec names it and a test can reach it.
SAGITTAL_VIEWS = {"side", "front_oblique"}

LOWER_BODY_LANDMARKS = (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)

DEADLIFT_METRIC_KEYS: tuple[str, ...] = (
    "hip_angle_deg",
    "knee_angle_deg",
    "torso_pitch_deg",
    "hip_y",
    "torso_len",
)

# `shoulder_y` is deliberately absent. An earlier design emitted it for a hip-vs-shoulder
# rise differential in `rule_hips_shoot_up`; that term was shown to be algebraically
# identical to a trunk-pitch change (see the rule's docstring in Task 3), so nothing consumes
# it.

_DEGENERATE_LENGTH = 1e-6

# Phases in which the deadlift is under load. `lowering` and `setup` are excluded.
DEADLIFT_ACTIVE_PHASES = {"lift_off", "mid_pull", "lockout"}


def deadlift_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
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

        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        knee_mid = midpoint(points, LEFT_KNEE, RIGHT_KNEE, dims=2)
        ankle_mid = midpoint(points, LEFT_ANKLE, RIGHT_ANKLE, dims=2)

        hip_angle_deg = _angle_between(shoulder_mid, hip_mid, knee_mid)
        knee_angle_deg = _angle_between(hip_mid, knee_mid, ankle_mid)
        # `line_angle_from_vertical(top, bottom)` takes abs() of both deltas, so this is an
        # UNSIGNED angle in [0, 90] -- it cannot distinguish a forward from a backward lean.
        # Correct for the deadlift, where the trunk only ever pitches forward, and it is why
        # `rule_hips_shoot_up` can compare magnitudes without resolving the subject's facing.
        torso_pitch_deg = line_angle_from_vertical(shoulder_mid, hip_mid)
        torso_len = (
            float(np.linalg.norm(shoulder_mid - hip_mid))
            if shoulder_mid is not None and hip_mid is not None
            else np.nan
        )

        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "hip_angle_deg": hip_angle_deg,
                "knee_angle_deg": knee_angle_deg,
                "torso_pitch_deg": torso_pitch_deg,
                "hip_y": float(hip_mid[1]) if hip_mid is not None else np.nan,
                "torso_len": torso_len,
            }
        )
    return raw


def _angle_between(a: np.ndarray | None, b: np.ndarray | None, c: np.ndarray | None) -> float:
    """Interior angle at `b`, in degrees. NaN when any point is missing or degenerate.

    `geometry.angle_degrees` takes LANDMARK INDICES, not points; these vertices are computed
    midpoints with no index, so the arithmetic is done here rather than reaching for it.
    """
    if a is None or b is None or c is None:
        return float(np.nan)
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    na = float(np.linalg.norm(ba))
    nc = float(np.linalg.norm(bc))
    if na < _DEGENERATE_LENGTH or nc < _DEGENERATE_LENGTH:
        return float(np.nan)
    cosine = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def deadlift_assign_phases(raw: list[dict]) -> list[str]:
    """setup -> lift_off -> mid_pull -> lockout -> lowering, segmented on `hip_angle_deg`.

    Mirrors `lunge_assign_phases`, substituting hip angle for knee angle and inverting the
    sense: a lunge rep is deepest in the middle, a deadlift rep is most EXTENDED in the
    middle.

    THE PHASE CUTOFFS ARE PERCENTILES OF THIS REP'S OWN EXCURSION, NOT ABSOLUTE ANGLES, and
    that is load-bearing rather than stylistic. `rule_incomplete_lockout` scores the `lockout`
    phase, and the fault it detects IS failing to reach extension. An absolute cutoff (say
    "lockout = hip angle above 165 degrees") would give a shallow-finishing rep NO lockout
    frames at all, so the rule would go silent on precisely the reps it exists to catch. A
    percentile guarantees the phase exists for every rep, however badly performed. Same
    reasoning as lunge's `bottom_threshold = np.percentile(valid_knee, 30)`.

    The lockout test precedes the post-peak test deliberately: a lifter standing at lockout
    produces high-angle frames on BOTH sides of the peak frame, and those are lockout, not
    lowering. Checking `index > peak` first would discard half the lockout plateau.
    """
    frame_count = len(raw)
    if frame_count == 0:
        return []

    hip_values = np.asarray(
        [float(item.get("hip_angle_deg", np.nan)) for item in raw], dtype=np.float32
    )
    finite = hip_values[np.isfinite(hip_values)]
    if finite.size == 0:
        return ["unknown" for _ in raw]

    lockout_threshold = float(np.percentile(finite, 75))
    mid_pull_threshold = float(np.percentile(finite, 40))
    peak_index = int(np.nanargmax(np.where(np.isfinite(hip_values), hip_values, -np.inf)))
    setup_cutoff = max(1, int(frame_count * 0.10))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append("setup")
            continue

        value = hip_values[index]
        if np.isfinite(value) and value >= lockout_threshold:
            phases.append("lockout")
        elif index > peak_index:
            phases.append("lowering")
        elif np.isfinite(value) and value >= mid_pull_threshold:
            phases.append("mid_pull")
        else:
            phases.append("lift_off")
    return phases


def setup_baseline(core: list[CoreFrame], key: str) -> float:
    """Median of `key` over the window's `setup` frames; NaN when there are none.

    A per-rep baseline cannot live in `deadlift_compute_raw`, which `run_detector` calls over
    the WHOLE CLIP before any rep boundary exists. It belongs here, where the window IS one
    rep -- the same split lunge uses for lead-side resolution and squat uses for its heel
    baseline. Median rather than mean so one mis-tracked setup frame cannot move it.
    """
    values = [
        frame.m(key) for frame in core if frame.valid and frame.phase == "setup"
    ]
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float(np.nan)


# The ONLY Deadlift rule whose kg_query resolves. Verified through the production path:
# `retrieve_graph_context("Lumbar Flexion", movement="Deadlift")` returns the seed
# `Deadlift:Lumbar Flexion` with a NON-EMPTY bucket -- `INCREASES_RISK_OF -> Lumbar Spine
# Injury`, `CORRECTED_BY -> Maintain Neutral Spine`, `HAS_FAULT <- Deadlift`. Checking
# resolution alone was not enough: OHP shipped queries that resolved but returned nothing.
DEADLIFT_LUMBAR_KG_QUERY = "Lumbar Flexion"

# ---------------------------------------------------------------------------------------
# THESE THREE NUMBERS ARE UNSOURCED. The suffix is not decoration.
# ---------------------------------------------------------------------------------------
# No source anywhere gives a segment-shortening-to-lumbar-flexion figure. 0.95 says "5%
# shortening", chosen to sit above frame-to-frame landmark jitter WITHOUT ANY MEASUREMENT OF
# WHAT THAT JITTER IS; 0.85 is a doubling of it; 0.10 of a torso length is a loose "the hips
# have not really moved yet" band. The fault itself IS cited (see citation_support) -- what
# is unsupported is the detection, which is why this rule emits at observability `low` with
# the off-view discount and `run_detector` sorts it last. Calibrating the gate against a
# measured landmark-jitter floor is the known upgrade path; see the design spec section 4.3.
DEADLIFT_TORSO_SHORTENING_MILD_UNSOURCED = 0.95
DEADLIFT_TORSO_SHORTENING_SEVERE_UNSOURCED = 0.85
DEADLIFT_HIP_STATIONARY_BAND_UNSOURCED = 0.10


def rule_lumbar_flexion(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Lower back rounds under load -- PROXY ONLY, and the weakest rule in this module.

    MediaPipe has no lumbar landmarks, so true rounded-vs-neutral spine is not recoverable;
    the parent spec rates this `low` observability and says "Do NOT assert precision here".
    The proxy: in a sagittal view a rigid hip hinge holds the PROJECTED shoulder-to-hip length
    constant, because the trunk rotates within the image plane. Shortening against the rep's
    own setup baseline, while the hips are not themselves travelling, is consistent with the
    trunk curling.

    HARD VIEW GATE, unlike this module's other two rules. Off-view, trunk pitch alone shortens
    the projected segment, so the proxy produces FALSE POSITIVES rather than silence. Where the
    off-view failure mode is a wrong claim rather than a missed one, the OHP precedent
    (`ohp_forward_head`) gates instead of discounting. The `SIDE_VIEW_CONF_THRESHOLD` floor
    follows squat's `rule_knees_forward` and OHP -- no new number.
    """
    if ctx.view_type not in SAGITTAL_VIEWS or ctx.view_confidence < SIDE_VIEW_CONF_THRESHOLD:
        return []

    torso_0 = setup_baseline(core, "torso_len")
    hip_0 = setup_baseline(core, "hip_y")
    if not np.isfinite(torso_0) or not np.isfinite(hip_0) or torso_0 < _DEGENERATE_LENGTH:
        return []

    def _ratio(frame: CoreFrame) -> float:
        value = frame.m("torso_len")
        return value / torso_0 if np.isfinite(value) else float(np.nan)

    def _hips_still(frame: CoreFrame) -> bool:
        value = frame.m("hip_y")
        if not np.isfinite(value):
            return False
        return abs(value - hip_0) / torso_0 < DEADLIFT_HIP_STATIONARY_BAND_UNSOURCED

    mask = [
        frame.valid
        and frame.phase in DEADLIFT_ACTIVE_PHASES
        and np.isfinite(_ratio(frame))
        and _ratio(frame) < DEADLIFT_TORSO_SHORTENING_MILD_UNSOURCED
        and _hips_still(frame)
        for frame in core
    ]

    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        ratios = [_ratio(frame) for frame in segment]
        worst = float(np.nanmin(ratios))
        severity = severity_from_range(
            worst,
            DEADLIFT_TORSO_SHORTENING_MILD_UNSOURCED,
            DEADLIFT_TORSO_SHORTENING_SEVERE_UNSOURCED,
            lower_is_worse=True,
        )
        detections.append(
            build_detection(
                fault_id="deadlift_lumbar_flexion",
                fault_name="Rounded Lower Back / Lumbar Flexion",
                kg_query=DEADLIFT_LUMBAR_KG_QUERY,
                retrieval_mode="kg",
                segment_metrics=segment,
                # Severity rises as the ratio FALLS, so the peak frame is the smallest ratio;
                # negate so build_detection's argmax finds it.
                score_values=[-r for r in ratios],
                severity=severity,
                # ALWAYS discounted: this is a proxy, not a measurement, even in its own view.
                confidence=severity * _OFF_VIEW_CONFIDENCE,
                observability="low",
                evidence={
                    "min_torso_length_ratio": round(worst, 4),
                    "setup_torso_length": round(torso_0, 4),
                    "threshold": DEADLIFT_TORSO_SHORTENING_MILD_UNSOURCED,
                    "proxy": "projected torso shortening; MediaPipe has no lumbar landmarks",
                    "primary_label": "torso length vs setup",
                    "primary_value": round(worst, 4),
                    "primary_threshold": DEADLIFT_TORSO_SHORTENING_MILD_UNSOURCED,
                },
                citation="Moreira VM, et al. \"Analysis of Muscle Strength and Electromyographic "
                         "Activity during Different Deadlift Positions.\" Muscles (2023). "
                         "PMC12225233.",
                citation_support="PMC12225233: \"The lift-off position in DL, using the "
                                 "powerlift posture, generates greater lumbar spine shear "
                                 "force,\" and erector-spinae activation peaks at "
                                 "lift-off/mid-pull because \"ERE requires higher activation "
                                 "and higher strength to avoid trunk flexion, reducing shear.\" "
                                 "Verified in RAG doc. NOTE what this does and does not "
                                 "support: the FAULT is cited, loaded and mechanistically "
                                 "understood; the source says nothing about detecting it from "
                                 "pose, and the detection threshold here is unsourced.",
            )
        )
    return detections


# STEP 0 -- KG QUERY RESOLUTION, recorded before the rule was written. Checked against
# data/kg/sports_kg_v3.graphml via BOTH `resolve_nodes` and `retrieve_graph_context` (the
# latter is what production calls, and is what OHP's three-blank-queries defect would have
# been caught by).
#
# NO KG NODE EXISTS FOR THIS FAULT, so it takes the `rag` fallback. The 5-node Deadlift stub
# carries exactly one lockout node -- `Deadlift:Hyperextension At Lockout` -- which is the
# LITERAL OPPOSITE fault: too much extension, not too little. `Incomplete Lockout` resolves to
# nothing and `Incomplete Range Of Motion` resolves only to the generic shared-layer
# `Range Of Motion` concept node, which would ground a coaching explanation on an abstraction
# rather than on an error. Grounding this rule on any of them would retrieve advice for a
# different problem, so per the lunge Step-0 rule -- "do NOT invent a near-miss" -- it does not.
#
# IN `rag` MODE THIS STRING IS A VECTOR-DB SEARCH PHRASE, NOT A NODE NAME
# (`pose_rule_detector.py:756` passes it straight to `query_vector_db`), so it is written as
# one and was verified by running it. The node-style "Incomplete Lockout" retrieves a ROW
# suspension-EMG paper and a LEG ABDUCTION paper -- the wrong movement entirely. The phrasing
# below returns PMC12148905, this rule's cross-support citation, at ranks 1 and 3. Verified
# 2026-08-01; re-run before changing it.
DEADLIFT_LOCKOUT_KG_QUERY = "deadlift incomplete lockout hip and knee extension"

# Spec-derived, unvalidated. The 180-degree TARGET is measured -- Moreira PMC12225233 recorded
# the three key positions at lift-off 95 deg, mid-pull 126 deg and lock-out 180 deg, with
# "180 degrees ... equivalent to full extension" -- but the 165-degree tolerance below which a
# rep is called incomplete is the parent spec's number and no source states it.
DEADLIFT_LOCKOUT_MILD_DEG = 165.0
DEADLIFT_LOCKOUT_SEVERE_DEG = 140.0


def _peak_extension(segment: list[CoreFrame], key: str) -> float:
    """GREATEST finite value of `key` across the segment; NaN when the axis is wholly missing.

    "How far did this rep extend" is a `nanmax`, not a `nanmin`. The guard is not decoration:
    the rule below flags on either axis independently, so a segment can be flagged entirely by
    the knee while every hip reading in it is NaN (or vice versa) -- exactly the
    occluded-landmark failure mode this module's header describes. An unguarded `np.nanmax` over
    an all-NaN slice both emits a RuntimeWarning AND returns a bare NaN, which
    `dataclasses.asdict()` carries into a postgrest write with `allow_nan=False`, whose
    ValueError this codebase documents as silently swallowed -- the analysis vanishes from the
    user's history with no surfaced error. `overhead_press.rule_incomplete_lockout` (this rule's
    model) guards its analogous `peak_worse_elbow`/`max_wrist` the same way.
    """
    values = [frame.m(key) for frame in segment]
    return float(np.nanmax(values)) if any(np.isfinite(v) for v in values) else float(np.nan)


def rule_incomplete_lockout(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Rep ends without full hip AND knee extension.

    THIS SCORES THE REP'S PEAK EXTENSION, NOT A RUN OF INDIVIDUALLY-FAILING FRAMES, and that
    distinction is the whole rule. It fires when the BEST hip extension reached during `lockout`
    is under 165 degrees, OR the best knee extension is -- `nanmax` per axis, then the ramp.

    ---------------------------------------------------------------------------------------
    WHY, because the frame-window version shipped a false positive (fixed 2026-08-01)
    ---------------------------------------------------------------------------------------
    The first implementation built a per-frame mask (`phase == "lockout" and angle < 165`) and
    handed it to `contiguous_true_segments`, mirroring this module's other two rules. That is
    wrong HERE, and the reason is `deadlift_assign_phases`: `lockout` is the 75th PERCENTILE of
    the rep's own hip-angle excursion -- a RANK cutoff, not an angle. So if a lifter spends less
    than 25% of the rep's frames above 165 degrees, the `lockout` band necessarily extends BELOW
    165, and those frames satisfy a per-frame `< 165` test even though the rep locked out
    perfectly. Measured end to end on the segmented production path (`fallback=None`): a
    three-rep clip peaking at 178 degrees -- full extension by this rule's own cited target --
    produced `lockout` bands of 148.5-178.0 and fired at severity 0.66 / confidence 0.66 /
    observability "high", reporting "minimum hip angle 148.5". The trigger was purely a
    contiguity accident: it needed `min_frames` (0.20 s) spent between the percentile cutoff and
    165, so it appeared at 2.8 s/rep and vanished at 2.5 s/rep, and a 0.1-second pause at lockout
    suppressed it entirely.
    ---------------------------------------------------------------------------------------

    Peak-scoring removes the failure mode outright rather than tuning around it, and introduces
    NO new number: it reuses the same 165/140 ramp on a different aggregate. It is also what the
    parent spec always described -- it phrases this fault as measured "at the top phase ... at
    rep end", i.e. a property of the rep's MAXIMUM extension, not of any window of frames -- and
    it brings the rule into line with `overhead_press.rule_incomplete_lockout`, which this
    docstring already cited as its model and which has always aggregated with `np.nanmax` before
    scoring.

    The `lockout` phase gate and the `min_frames` floor both stay: the phase keeps its meaning,
    and a two-frame lockout phase is still too little to judge a rep on.

    BOTH RAMPS ARE SCORED UNCONDITIONALLY and the worse is taken. Selecting the ramp by "which
    reading is finite" is the OHP mis-attribution bug recorded in the parent spec's section 8
    status note; that discipline is unchanged by the peak rewrite. `severity_from_range` already
    returns 0.0 for a non-finite value, so a missing axis contributes nothing WITHOUT ever
    selecting which ramp is used.

    View policy is DEGRADE, not gate. An extension angle seen head-on is foreshortened, so it
    under-reads -- the off-view failure mode is a missed fault, not a false one. Contrast
    `rule_lumbar_flexion`, which inverts off-view and is therefore hard-gated.
    """
    observable = ctx.view_type in SAGITTAL_VIEWS

    # The mask is now the PHASE ALONE. The `< 165` test has moved off the individual frame and
    # onto the segment's peak, below -- see the docstring for why a per-frame test was unsound.
    mask = [frame.valid and frame.phase == "lockout" for frame in core]

    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        peak_hip = _peak_extension(segment, "hip_angle_deg")
        peak_knee = _peak_extension(segment, "knee_angle_deg")
        flagged = (np.isfinite(peak_hip) and peak_hip < DEADLIFT_LOCKOUT_MILD_DEG) or (
            np.isfinite(peak_knee) and peak_knee < DEADLIFT_LOCKOUT_MILD_DEG
        )
        if not flagged:
            continue

        hip_sev = severity_from_range(
            peak_hip, DEADLIFT_LOCKOUT_MILD_DEG, DEADLIFT_LOCKOUT_SEVERE_DEG, lower_is_worse=True
        )
        knee_sev = severity_from_range(
            peak_knee, DEADLIFT_LOCKOUT_MILD_DEG, DEADLIFT_LOCKOUT_SEVERE_DEG, lower_is_worse=True
        )
        severity = float(max(hip_sev, knee_sev))
        driver = "hip" if hip_sev >= knee_sev else "knee"
        driver_peak = peak_hip if driver == "hip" else peak_knee
        # `build_detection` takes `argmax(score_values)` as the peak frame, so feeding it the
        # DRIVER AXIS'S raw angles points `peak_frame` at the frame that actually achieved the
        # reported peak extension -- the frame the evidence is quoting. Same convention as
        # `overhead_press.rule_incomplete_lockout`, which passes its raw `wrist_values`.
        # `flagged` guarantees the driver axis has at least one finite reading, so this is never
        # an all-NaN argmax.
        score_values = [frame.m(f"{driver}_angle_deg") for frame in segment]

        detections.append(
            build_detection(
                fault_id="deadlift_incomplete_lockout",
                fault_name="Incomplete Lockout",
                kg_query=DEADLIFT_LOCKOUT_KG_QUERY,
                retrieval_mode="rag",
                segment_metrics=segment,
                score_values=score_values,
                severity=severity,
                confidence=severity * (1.0 if observable else _OFF_VIEW_CONFIDENCE),
                observability="high" if observable else "medium",
                evidence={
                    # Fall back to 0.0, never a bare NaN: `PoseRuleDetection.evidence` is
                    # serialized via `asdict()` with no NaN sanitizer downstream (unlike
                    # `json_safe_view_payload`), and postgrest's JSON encoder raises on NaN --
                    # matching `overhead_press.rule_incomplete_lockout`'s
                    # `round(x, 2) if np.isfinite(x) else 0.0` for the same evidence shape.
                    "peak_hip_angle_deg": round(peak_hip, 2) if np.isfinite(peak_hip) else 0.0,
                    "peak_knee_angle_deg": round(peak_knee, 2) if np.isfinite(peak_knee) else 0.0,
                    "threshold": DEADLIFT_LOCKOUT_MILD_DEG,
                    "driver": driver,
                    "primary_label": f"peak {driver} angle at lockout",
                    "primary_value": (
                        round(driver_peak, 2) if np.isfinite(driver_peak) else 0.0
                    ),
                    "primary_threshold": DEADLIFT_LOCKOUT_MILD_DEG,
                },
                citation="Moreira VM, et al. \"Analysis of Muscle Strength and Electromyographic "
                         "Activity during Different Deadlift Positions.\" Muscles (2023). "
                         "PMC12225233. Cross-support: Hanen NC, et al. PMC12148905 (2025).",
                citation_support="PMC12225233 measured the three key positions at "
                                 "\"approximately 95 deg, 126 deg, and 180 deg\" for lift-off, "
                                 "mid-pull and lock-out, with \"180 deg ... equivalent to full "
                                 "extension\" -- so full triple extension is a measured target, "
                                 "not an assumption. PMC12148905: \"lift completion[] is "
                                 "achieved when the athlete assumes a fully upright position "
                                 "with extended hips and knees, with scapular retraction.\" "
                                 "Verified in RAG docs. The 165 deg flag point is spec-derived.",
            )
        )
    return detections


# NO KG NODE EXISTS FOR THIS FAULT -- `rag` fallback, resolved before the rule was written.
# The nearest Deadlift-scoped candidate, `Deadlift:Insufficient Hip Hinge`, is a near-miss
# POINTING THE WRONG WAY: insufficient hinge means failing to push the hips back, a
# knee-dominant squat-like pull, whereas this fault is excessive hip dominance with the trunk
# flattening. Its only edge is `AFFECTS_QUALITY -> Hip Hinge` -- no risk, no correction. The
# other candidates (`Hips Rise Before Shoulders`, `Trunk Over Inclination`, `Anterior Trunk
# Tilt`, `Excessive Forward Lean`) resolve to nothing or to the bare `Hip` anatomy node.
#
# IN `rag` MODE THIS STRING IS A VECTOR-DB SEARCH PHRASE, NOT A NODE NAME, and it was chosen by
# running candidates rather than by writing something plausible. The corpus holds only 2
# deadlift documents among 85, so semantic search drifts badly: "Hips Rise Before Shoulders"
# returns a row EMG paper, and four different mechanism-keyword phrasings ("...erector spinae
# trunk flexion barbell shear force", "...lever arm lower back barbell", and two more) each
# returned 0/3 deadlift documents, mostly Overhead Press. The phrasing below returns
# PMC12225233 -- this rule's primary citation -- at ranks 1, 2 AND 3. Verified 2026-08-01;
# re-run before changing it, because near-miss phrasings silently ground this fault in the
# wrong movement's literature.
DEADLIFT_HIPS_KG_QUERY = "deadlift trunk position electromyographic activity lift-off mid-pull lockout"

# Spec-derived, UNVALIDATED AND UNSOURCED. Neither deadlift RAG document reports a trunk
# inclination in degrees -- the only degree value in PMC12148905 is an unrelated 8 deg knee
# adduction. What the citation backs is the MECHANISM and the DIRECTION (a flatter trunk means
# more spinal flexion torque), which is what the two-clause criterion encodes; these endpoints
# are the parent spec's numbers.
DEADLIFT_PITCH_MILD_DEG = 55.0
DEADLIFT_PITCH_SEVERE_DEG = 75.0


def rule_hips_shoot_up(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Hips out-run the shoulders off the floor, flattening the trunk into a back-dominant pull.

    THIS IS NOT WRITTEN AS A HIP-VS-SHOULDER RISE DIFFERENTIAL, and the omission is deliberate.
    The parent spec phrases the signal as "Delta(hip_y) rises faster than Delta(shoulder_y)",
    and an earlier draft implemented that literally as

        hip_lead_ratio = ((hip_y0 - hip_y) - (shoulder_y0 - shoulder_y)) / torso_len0 > 0

    That term was checked numerically before any code was written and is ALGEBRAICALLY
    IDENTICAL to a trunk-pitch change. Since `shoulder_y - hip_y = -torso_len*cos(pitch)`, a
    rigid torso gives

        hip_lead_ratio == cos(pitch_0) - cos(pitch_t)

    exact to machine precision on a sagittal stick model. It depends ONLY on pitch and carries
    no information about how far the hips actually travelled -- two landmarks dressing up a
    single-angle test. Writing it as a differential would have implied this rule corroborates
    trunk pitch with an independent kinematic signal, which is false. The parent spec's own
    "i.e." equating the two phrasings turns out to be correct, so stating the rule in pitch
    terms is faithful to it rather than a deviation.

    The relative-to-setup clause is kept because it is what separates the SEQUENCING fault the
    citation describes from a lifter who merely sets up flat and stays there; the absolute
    55-degree gate alone cannot tell those apart.

    KNOWN DEFECT ON `run_detector`'S WHOLE-CLIP FALLBACK, and this rule is the one that
    misbehaves, so it is repeated here rather than left in the module header. When `segment_reps`
    fails (`segmentation_disabled` / `no_reps_detected` / `only_partial_reps`, `base.py:159`)
    the rules run over the whole clip, and `setup` becomes the clip's first 10% POSITIONALLY --
    which can be the lifter standing around before approaching the bar. `setup_baseline` then
    returns a STANDING torso (~7 degrees measured, vs ~60 for a real setup), the
    `torso_pitch_deg > baseline` clause is satisfied by every loaded frame and so nullified, and
    this rule DEGENERATES TO ITS BARE 55-DEGREE ABSOLUTE GATE -- losing exactly the
    discriminator the paragraph above says it exists for, and firing on a clean rep. Measured on
    the production path: `fallback=only_partial_reps` fired this rule at severity 0.2821 with
    `setup_torso_pitch_deg: 6.84` on a well-executed rep. Not fixed here (threading `fallback`
    into `RuleContext` is a framework change touching squat and push-up too); see the module
    header and the parent spec's section 7.

    View policy is DEGRADE, not gate: head-on, a pitched trunk projects short and near-vertical
    so the angle UNDER-reads, making the off-view failure mode silence rather than a wrong
    claim.
    """
    baseline = setup_baseline(core, "torso_pitch_deg")
    if not np.isfinite(baseline):
        return []
    observable = ctx.view_type in SAGITTAL_VIEWS

    mask = [
        frame.valid
        and frame.phase in DEADLIFT_ACTIVE_PHASES
        and np.isfinite(frame.m("torso_pitch_deg"))
        and frame.m("torso_pitch_deg") > baseline
        and frame.m("torso_pitch_deg") > DEADLIFT_PITCH_MILD_DEG
        for frame in core
    ]

    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(mask, ctx.min_frames):
        segment = core[start : end + 1]
        pitches = [frame.m("torso_pitch_deg") for frame in segment]
        peak = float(np.nanmax(pitches))
        severity = severity_from_range(
            peak, DEADLIFT_PITCH_MILD_DEG, DEADLIFT_PITCH_SEVERE_DEG, lower_is_worse=False
        )
        detections.append(
            build_detection(
                fault_id="deadlift_hips_shoot_up",
                fault_name="Hips Rise Before Shoulders / Trunk Over-Inclination",
                kg_query=DEADLIFT_HIPS_KG_QUERY,
                retrieval_mode="rag",
                segment_metrics=segment,
                score_values=pitches,
                severity=severity,
                confidence=severity * (1.0 if observable else _OFF_VIEW_CONFIDENCE),
                observability="high" if observable else "medium",
                evidence={
                    "peak_torso_pitch_deg": round(peak, 2),
                    "setup_torso_pitch_deg": round(baseline, 2),
                    "threshold": DEADLIFT_PITCH_MILD_DEG,
                    "primary_label": "peak trunk pitch from vertical",
                    "primary_value": round(peak, 2),
                    "primary_threshold": DEADLIFT_PITCH_MILD_DEG,
                },
                citation="Moreira VM, et al. PMC12225233 (2023). Cross-support: Hanen NC, "
                         "et al. PMC12148905 (2025).",
                citation_support="PMC12225233: \"leaning the trunk forward results in higher "
                                 "spinal flexion torque generated by the barbell. Therefore, "
                                 "ERE [erector spinae] requires higher activation and higher "
                                 "strength to avoid trunk flexion, reducing shear.\" "
                                 "PMC12148905 frames \"a significantly reduced trunk "
                                 "inclination angle\" as the low-back-sparing state. Verified "
                                 "in RAG docs. Both ramp endpoints are spec-derived: neither "
                                 "source reports a trunk inclination in degrees.",
            )
        )
    return detections


# THREE of the parent spec's FOUR Deadlift rules are listed. `deadlift_bar_drift` is absent
# because it is WITHDRAWN, not because it was forgotten -- see the boxed note in the parent
# spec's Deadlift section. Briefly: its citation (Hanen PMC12148905) contains no bar-path
# measurement and explicitly defers one ("Analyzing the bar path would be valuable to validate
# this hypothesis"), and its `midfoot_x` reference is the invented construct that the OHP
# bar-path withdrawal already ruled out. Unlike push-up's `rule_scapular_winging`, it is not
# registered-but-silent: a silent rule says "real fault, unmeasurable", whereas this one says
# "no citation supports the rule as written", which is a spec problem, not a sensing problem.
#
# `DEADLIFT_METRIC_KEYS` must stay a two-way match with what `deadlift_compute_raw` emits --
# pinned by `test_metric_keys_match_the_emitted_metrics`.
DEADLIFT_DETECTOR = MovementDetector(
    "Deadlift",
    DEADLIFT_METRIC_KEYS,
    deadlift_compute_raw,
    deadlift_assign_phases,
    (rule_hips_shoot_up, rule_incomplete_lockout, rule_lumbar_flexion),
    # `validated` stays at its default False. No labeled deadlift data exists anywhere in this
    # repository, so unlike Lunge there is not even a validation pass to defer to; flipping
    # this would need evidence that cannot currently be obtained.
    rep_signal="hip_angle_deg",
    # The signal bottoms out at the floor and peaks at lockout, so a rep is an excursion in
    # hip EXTENSION that starts and ends flexed -- `rep_start="flexed"`, the case base.py:55
    # names deadlift for.
    rep_polarity="min",
    rep_start="flexed",
)

registry.register(DEADLIFT_DETECTOR)
