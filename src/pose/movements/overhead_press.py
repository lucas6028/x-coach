# Thresholds in this module are spec-derived (docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md), NOT validated against labeled OHP data (spec §8.4).
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, angle_degrees, midpoint, mean_visibility, mean_finite,
    contiguous_true_segments, severity_from_range, distance,
)
from src.pose.movements.base import CoreFrame, MovementDetector, RuleContext
from src.pose.movements import registry
from src.pose.pose_rule_detector import PoseRuleDetection, build_detection

# MediaPipe indices not already exported by src.pose.geometry.
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_EAR = 7
RIGHT_EAR = 8
NOSE = 0

# Same generic "lower body" landmark set used across movements (src.pose.pose_rule_detector's
# LOWER_BODY_LANDMARKS) for the framework-level lower_body_visibility quality field; OHP's own
# rules do not consume it.
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

OHP_METRIC_KEYS: tuple[str, ...] = (
    "left_elbow_angle",
    "right_elbow_angle",
    "avg_elbow_angle",
    "wrist_above_shoulder",
    "torso_lean_signed_deg",
    "elbow_height_asymmetry",
    "wrist_height_asymmetry",
    "shoulder_ear_gap",
    "wrist_above_nose",
    "ear_forward_offset",
    "wrist_forward_offset",
)

# Views in which a sagittal-plane (anterior/posterior) offset is actually resolvable.
# Same set `rule_excessive_back_lean` uses for its own sagittal cue.
SAGITTAL_VIEWS = {"side", "front_oblique"}


def _y(points: np.ndarray | None, index: int) -> float:
    point = visible_point(points, index, dims=2)
    return float(point[1]) if point is not None else np.nan


def _x(points: np.ndarray | None, index: int) -> float:
    point = visible_point(points, index, dims=2)
    return float(point[0]) if point is not None else np.nan


def ohp_compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
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

        left_elbow_angle = angle_degrees(points, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        right_elbow_angle = angle_degrees(points, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        avg_elbow_angle = mean_finite([left_elbow_angle, right_elbow_angle])

        left_wrist_y = _y(points, LEFT_WRIST)
        right_wrist_y = _y(points, RIGHT_WRIST)
        left_shoulder_y = _y(points, LEFT_SHOULDER)
        right_shoulder_y = _y(points, RIGHT_SHOULDER)
        wrist_mean_y = mean_finite([left_wrist_y, right_wrist_y])
        shoulder_mean_y = mean_finite([left_shoulder_y, right_shoulder_y])
        wrist_above_shoulder = wrist_mean_y - shoulder_mean_y

        shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
        # Facing assumption (SINGLE convention for the whole module): the subject's
        # POSTERIOR direction is +x, i.e. anterior is -x. That is what this formula
        # already encodes -- shoulders drifting toward +x relative to the hips yields a
        # positive torso_lean_signed_deg, which rule_excessive_back_lean reads as a
        # BACKWARD lean (shoulders behind the hips). Every other sagittal offset in this
        # module (see rule_forward_head_barpath) therefore measures "anterior" as
        # shoulder_x - point_x. A subject facing the opposite way inverts the sign of
        # every such reading. This is a known monocular limitation -- neither the sign
        # nor any of the sagittal thresholds has been validated on real data.
        if shoulder_mid is not None and hip_mid is not None:
            torso_lean_signed_deg = float(
                np.degrees(
                    np.arctan2(
                        shoulder_mid[0] - hip_mid[0],
                        hip_mid[1] - shoulder_mid[1],
                    )
                )
            )
        else:
            torso_lean_signed_deg = np.nan

        left_elbow_y = _y(points, LEFT_ELBOW)
        right_elbow_y = _y(points, RIGHT_ELBOW)
        elbow_height_asymmetry = (
            abs(left_elbow_y - right_elbow_y)
            if np.isfinite(left_elbow_y) and np.isfinite(right_elbow_y)
            else np.nan
        )

        shoulder_width = distance(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
        wrist_height_asymmetry = (
            abs(left_wrist_y - right_wrist_y) / shoulder_width
            if np.isfinite(left_wrist_y)
            and np.isfinite(right_wrist_y)
            and np.isfinite(shoulder_width)
            and shoulder_width > 1e-6
            else np.nan
        )

        left_ear_y = _y(points, LEFT_EAR)
        right_ear_y = _y(points, RIGHT_EAR)
        left_gap = left_shoulder_y - left_ear_y if np.isfinite(left_ear_y) else np.nan
        right_gap = right_shoulder_y - right_ear_y if np.isfinite(right_ear_y) else np.nan
        shoulder_ear_gap = mean_finite([left_gap, right_gap])

        normalizer_ok = np.isfinite(shoulder_width) and shoulder_width > 1e-6

        # Wrist height relative to the NOSE (landmark 0), normalized by shoulder width.
        # Negative => wrists are above the nose (MediaPipe y grows DOWNWARD).
        # NaN when the nose is not visible or the shoulder-width normalizer degenerates;
        # the nose is deliberately NOT added to `required` above, because that flag gates
        # frame validity for every OHP rule and a briefly occluded face must not
        # invalidate the elbow/asymmetry rules.
        nose_y = _y(points, NOSE)
        wrist_above_nose = (
            (wrist_mean_y - nose_y) / shoulder_width
            if np.isfinite(wrist_mean_y) and np.isfinite(nose_y) and normalizer_ok
            else np.nan
        )

        # Sagittal-plane horizontal offsets, normalized by shoulder width. Positive =
        # the point sits ANTERIOR to the shoulder line, per the single facing convention
        # documented above (anterior = -x), hence shoulder_x - point_x.
        shoulder_mean_x = mean_finite([_x(points, LEFT_SHOULDER), _x(points, RIGHT_SHOULDER)])
        ear_mean_x = mean_finite([_x(points, LEFT_EAR), _x(points, RIGHT_EAR)])
        wrist_mean_x = mean_finite([_x(points, LEFT_WRIST), _x(points, RIGHT_WRIST)])
        ear_forward_offset = (
            (shoulder_mean_x - ear_mean_x) / shoulder_width
            if np.isfinite(shoulder_mean_x) and np.isfinite(ear_mean_x) and normalizer_ok
            else np.nan
        )
        wrist_forward_offset = (
            (shoulder_mean_x - wrist_mean_x) / shoulder_width
            if np.isfinite(shoulder_mean_x) and np.isfinite(wrist_mean_x) and normalizer_ok
            else np.nan
        )

        raw.append(
            {
                "frame_index": frame_index,
                "time": time,
                "valid": True,
                "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
                "left_elbow_angle": left_elbow_angle,
                "right_elbow_angle": right_elbow_angle,
                "avg_elbow_angle": avg_elbow_angle,
                "wrist_above_shoulder": wrist_above_shoulder,
                "torso_lean_signed_deg": torso_lean_signed_deg,
                "elbow_height_asymmetry": elbow_height_asymmetry,
                "wrist_height_asymmetry": wrist_height_asymmetry,
                "shoulder_ear_gap": shoulder_ear_gap,
                "wrist_above_nose": wrist_above_nose,
                "ear_forward_offset": ear_forward_offset,
                "wrist_forward_offset": wrist_forward_offset,
            }
        )
    return raw


def ohp_assign_phases(raw: list[dict]) -> list[str]:
    frame_count = len(raw)
    if frame_count == 0:
        return []

    avg_elbow_values = np.asarray(
        [float(item.get("avg_elbow_angle", np.nan)) for item in raw], dtype=np.float32
    )
    wrist_values = np.asarray(
        [float(item.get("wrist_above_shoulder", np.nan)) for item in raw], dtype=np.float32
    )
    valid_elbow = avg_elbow_values[np.isfinite(avg_elbow_values)]
    valid_wrist = wrist_values[np.isfinite(wrist_values)]
    if valid_elbow.size == 0 and valid_wrist.size == 0:
        return ["unknown" for _ in raw]

    elbow_threshold = float(np.percentile(valid_elbow, 70)) if valid_elbow.size else np.inf
    wrist_threshold = float(np.percentile(valid_wrist, 30)) if valid_wrist.size else -np.inf

    if valid_wrist.size:
        wrist_highest_index = int(np.nanargmin(np.where(np.isfinite(wrist_values), wrist_values, np.inf)))
    else:
        wrist_highest_index = -1

    setup_cutoff = max(1, int(frame_count * 0.15))

    phases: list[str] = []
    for index, item in enumerate(raw):
        if not item.get("valid"):
            phases.append("unknown")
            continue
        if index < setup_cutoff:
            phases.append("setup")
            continue

        is_lockout = (
            np.isfinite(avg_elbow_values[index])
            and avg_elbow_values[index] >= elbow_threshold
            and np.isfinite(wrist_values[index])
            and wrist_values[index] <= wrist_threshold
        )
        if is_lockout:
            phases.append("lockout")
        elif index < wrist_highest_index:
            phases.append("press")
        else:
            phases.append("lower")
    return phases


def _worse_elbow_angle(frame: CoreFrame) -> float:
    """The more-limiting (smaller) of the two elbow angles for a frame, per spec's
    "take the worse of the two arms" rule; NaN if neither side is finite."""
    left = frame.m("left_elbow_angle")
    right = frame.m("right_elbow_angle")
    finite = [value for value in (left, right) if np.isfinite(value)]
    return min(finite) if finite else np.nan


def rule_incomplete_lockout(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag reps that never reach a stable overhead lockout: the worse (more bent) elbow's
    peak extension stays below ~160 deg, or the wrists never clear the shoulders at all.
    Evaluated over the `lockout` phase, or a window around the highest wrist position when
    no `lockout` phase is present."""
    observable_alignment = ctx.view_type in {"side", "front", "front_oblique"}

    lockout_indices = {i for i, frame in enumerate(core) if frame.valid and frame.phase == "lockout"}
    if not lockout_indices:
        valid_indices = [
            i for i, frame in enumerate(core)
            if frame.valid and np.isfinite(frame.m("wrist_above_shoulder"))
        ]
        if valid_indices:
            peak_index = min(valid_indices, key=lambda i: core[i].m("wrist_above_shoulder"))
            half_window = max(ctx.min_frames // 2, 1)
            window_start = max(0, peak_index - half_window)
            window_end = min(len(core) - 1, peak_index + half_window)
            lockout_indices = set(range(window_start, window_end + 1))

    lockout_mask = []
    for index, frame in enumerate(core):
        if index not in lockout_indices or not frame.valid:
            lockout_mask.append(False)
            continue
        worse_elbow = _worse_elbow_angle(frame)
        wrist = frame.m("wrist_above_shoulder")
        elbow_flag = np.isfinite(worse_elbow) and worse_elbow < 160.0
        wrist_flag = np.isfinite(wrist) and wrist > 0.0
        lockout_mask.append(elbow_flag or wrist_flag)

    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(lockout_mask, ctx.min_frames):
        segment = core[start : end + 1]
        worse_elbow_values = [_worse_elbow_angle(frame) for frame in segment]
        wrist_values = [frame.m("wrist_above_shoulder") for frame in segment]
        peak_worse_elbow = (
            float(np.nanmax(worse_elbow_values))
            if any(np.isfinite(v) for v in worse_elbow_values)
            else np.nan
        )
        max_wrist = (
            float(np.nanmax(wrist_values)) if any(np.isfinite(v) for v in wrist_values) else np.nan
        )

        if np.isfinite(peak_worse_elbow):
            severity = severity_from_range(peak_worse_elbow, 160.0, 140.0, lower_is_worse=True)
            primary_label = "peak worse-arm elbow angle"
            primary_value = round(peak_worse_elbow, 2)
            primary_threshold = 160.0
        elif np.isfinite(max_wrist):
            # Flagged only by the wrist-never-clears-shoulder condition (no finite elbow
            # reading in the segment) -- drive severity off how far below the shoulder line
            # the wrist stayed instead.
            severity = severity_from_range(max_wrist, 0.0, 0.15, lower_is_worse=False)
            primary_label = "wrist height above shoulder"
            primary_value = round(max_wrist, 4)
            primary_threshold = 0.0
        else:
            severity = 0.0
            primary_label = "peak worse-arm elbow angle"
            primary_value = 0.0
            primary_threshold = 160.0

        detections.append(
            build_detection(
                fault_id="ohp_incomplete_lockout",
                fault_name="Incomplete lockout at the top",
                kg_query="Incomplete Elbow Lockout",
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=wrist_values,
                severity=severity,
                confidence=severity * (1.0 if observable_alignment else 0.65),
                observability="high" if observable_alignment else "medium",
                evidence={
                    "peak_worse_elbow_angle": round(peak_worse_elbow, 2) if np.isfinite(peak_worse_elbow) else 0.0,
                    "elbow_threshold": 160.0,
                    "max_wrist_above_shoulder": round(max_wrist, 4) if np.isfinite(max_wrist) else 0.0,
                    "wrist_threshold": 0.0,
                    "primary_label": primary_label,
                    "primary_value": primary_value,
                    "primary_threshold": primary_threshold,
                },
                citation="Evangelista P, Rum L, Picerno P, Biscarini A. (2025). Decoding the Contribution "
                         "of Shoulder and Elbow Mechanics to Barbell Kinematics and the Sticking Region in "
                         "Bench and Overhead Press Exercises: A Link-Chain Model with Single- and Two-Joint "
                         "Muscles. J Funct Morphol Kinesiol. PMC12372072, DOI 10.3390/jfmk10030322.",
                citation_support="Elbow extensors contribute minimally in early lift phases but become "
                                 "dominant near full extension, and the lift is defined complete only "
                                 "\"when the elbow is fully extended … and the barbell reaches its final "
                                 "position\" -- so a rep stopping short of full elbow extension (peak worse-arm "
                                 "extension < ~160 deg, vs full lockout ~175-180 deg) omits the lockout that "
                                 "defines a completed press.",
            )
        )
    return detections


def rule_excessive_back_lean(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag frames where the trunk leans backward past 15 deg past vertical, the
    compensation historically linked to lower-back injury in overhead pressing.
    Per spec, back-lean is assessed at/after mid-press, so the mask is restricted to
    the press/lockout phases."""
    observable_lean = ctx.view_type in {"side", "front_oblique"}
    back_lean_phases = {"press", "lockout"}
    lean_mask = [
        frame.valid
        and frame.phase in back_lean_phases
        and np.isfinite(frame.m("torso_lean_signed_deg"))
        and frame.m("torso_lean_signed_deg") > 15.0
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(lean_mask, ctx.min_frames):
        segment = core[start : end + 1]
        values = [frame.m("torso_lean_signed_deg") for frame in segment]
        max_lean = float(np.nanmax(values))
        severity = severity_from_range(max_lean, 15.0, 35.0, lower_is_worse=False)
        detections.append(
            build_detection(
                fault_id="ohp_lumbar_hyperextension",
                fault_name="Excessive back-lean / lumbar hyperextension (rib flare)",
                kg_query="Lumbar Hyperextension",
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=values,
                severity=severity,
                confidence=severity * (1.0 if observable_lean else 0.65),
                observability="high" if observable_lean else "medium",
                evidence={
                    "max_torso_lean_signed_deg": round(max_lean, 2),
                    "threshold": 15.0,
                    "primary_label": "torso lean angle",
                    "primary_value": round(max_lean, 2),
                    "primary_threshold": 15.0,
                },
                citation="Soriano MA, Suchomel TJ, Comfort P. \"Weightlifting Overhead Pressing Derivatives: "
                         "A Review of the Literature.\" Sports Med (2019) PMC6548056.",
                citation_support="The review recounts the press degenerating into the \"continental press\" "
                                 "with a quick backbend, and that a long list of lower-back injuries from the "
                                 "accentuated backbend drove the IWF to eliminate the press -- naming lumbar "
                                 "hyperextension as the lower-back injury mechanism in overhead pressing.",
            )
        )
    return detections


def _lockout_phases(core: list[CoreFrame]) -> set[str]:
    """Phase set for rules the spec assesses "at/near lockout": the `lockout` phase when the
    rep has one, otherwise press+lockout so a rep with a weak/absent lockout window still
    gets evaluated instead of silently never firing."""
    return {"lockout"} if any(frame.phase == "lockout" for frame in core) else {"press", "lockout"}


def rule_asymmetric_press(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag frames where one wrist presses meaningfully higher than the other (normalized by
    shoulder width), a proxy for the scapular/shoulder-girdle asymmetry (scapular dyskinesis)
    linked to elevated shoulder-injury risk. Per spec, asymmetric press is assessed at/near
    lockout, so the mask is restricted to lockout-phase frames -- falling back to
    press+lockout when a rep has no lockout-phase frames at all (e.g. a weak/incomplete
    lockout window), so it still gets evaluated instead of silently never firing."""
    observable_asymmetry = ctx.view_type in {"front", "rear"}
    asymmetry_phases = _lockout_phases(core)
    asymmetry_mask = [
        frame.valid
        and frame.phase in asymmetry_phases
        and np.isfinite(frame.m("wrist_height_asymmetry"))
        and frame.m("wrist_height_asymmetry") > 0.15
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(asymmetry_mask, ctx.min_frames):
        segment = core[start : end + 1]
        values = [frame.m("wrist_height_asymmetry") for frame in segment]
        max_asymmetry = float(np.nanmax(values))
        severity = severity_from_range(max_asymmetry, 0.15, 0.30, lower_is_worse=False)
        detections.append(
            build_detection(
                fault_id="ohp_asymmetric_press",
                fault_name="Asymmetric Press (One Side Leading)",
                kg_query="Asymmetric Press",
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=values,
                severity=severity,
                confidence=severity * (1.0 if observable_asymmetry else 0.65),
                observability="high" if observable_asymmetry else "medium",
                evidence={
                    "max_wrist_height_asymmetry": round(max_asymmetry, 4),
                    "threshold": 0.15,
                    "primary_label": "wrist height asymmetry",
                    "primary_value": round(max_asymmetry, 4),
                    "primary_threshold": 0.15,
                },
                citation="Abdelraouf OR, Abdel-Aziem AA, Alkhamees NH, Ibrahim ZM, Aboelela EM, Dawood RS, "
                         "Ashour AA. (2026). Acute Effects of High-Load Training to Failure vs. Non-Failure "
                         "on Posture and Core Endurance in Collegiate Weightlifters: A Crossover Study. "
                         "J Clin Med. PMC13116542.",
                citation_support="Shoulder-girdle asymmetry (scapular dyskinesis) is defined as a "
                                 "difference between the two sides of more than 7 degrees in scapular "
                                 "angle or 1.5 cm in lateral shift, and high-load press training to "
                                 "failure produced \"a more protracted scapular position and shoulder "
                                 "girdle asymmetry.\"",
            )
        )
    return detections


def rule_insufficient_elevation(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag reps whose hands never travel to a true overhead position -- the press ends around
    forehead/eye level instead of with the wrists clearly above the head. Assessed at/near
    lockout (see `_lockout_phases`).

    SUBSTITUTED CRITERION -- read before touching the threshold:
    The spec (docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md,
    `ohp_insufficient_elevation`) expresses this as the wrist failing to clear the nose "by at
    least ~0.5 head-heights". That is NOT implementable here: MediaPipe's 33 landmarks contain
    no head-height measure at all -- nose, eyes, ears and mouth ALL lie within the face, so no
    pair of them spans the head. This rule therefore SUBSTITUTES a different criterion:
    nose clearance normalized by SHOULDER WIDTH (the module's existing normalizer, cf.
    `wrist_height_asymmetry`), firing when `wrist_above_nose > -0.15`. The -0.15 is a
    SUBSTITUTION, NOT A UNIT CONVERSION of the spec's 0.5 head-heights -- no anthropometric
    head-height-to-biacromial-width constant was assumed, invented, or applied. Like every
    other threshold in this module it is unvalidated against labeled OHP data (spec §8.4).

    Distinct from `ohp_incomplete_lockout`, which keys off elbow extension: a lifter can lock
    the elbows out at forehead level (this rule fires, that one does not) or press the hands
    well overhead with bent elbows (that one fires, this one does not)."""
    observable_elevation = ctx.view_type in {"side", "front", "front_oblique"}
    elevation_phases = _lockout_phases(core)
    elevation_mask = [
        frame.valid
        and frame.phase in elevation_phases
        and np.isfinite(frame.m("wrist_above_nose"))
        and frame.m("wrist_above_nose") > -0.15
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(elevation_mask, ctx.min_frames):
        segment = core[start : end + 1]
        values = [frame.m("wrist_above_nose") for frame in segment]
        # Larger = wrists sit LOWER relative to the nose = worse.
        worst = float(np.nanmax(values))
        severity = severity_from_range(worst, -0.15, 0.35, lower_is_worse=False)
        detections.append(
            build_detection(
                fault_id="ohp_insufficient_elevation",
                fault_name="Insufficient Overhead Elevation / Short Press",
                # Verified to resolve: graph_retrieval.resolve_nodes(..., movement="Overhead
                # Press") returns "Overhead Press:Limited Shoulder Elevation" for this string.
                # (The literal fault name "Insufficient Overhead Elevation" resolves to nothing.)
                kg_query="Limited Shoulder Elevation",
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=values,
                severity=severity,
                confidence=severity * (1.0 if observable_elevation else 0.65),
                observability="high" if observable_elevation else "medium",
                evidence={
                    "max_wrist_above_nose": round(worst, 4),
                    "threshold": -0.15,
                    "primary_label": "wrist height above nose (shoulder-widths)",
                    "primary_value": round(worst, 4),
                    "primary_threshold": -0.15,
                },
                citation="Coratella G, Tornatore G, Longo S, Esposito F, Cè E. (2022). Front vs Back "
                         "and Barbell vs Machine Overhead Press: An Electromyographic Analysis and "
                         "Implications For Resistance Training. Frontiers in Physiology. PMC9354811; "
                         "end position corroborated by Evangelista P, Rum L, Picerno P, Biscarini A. "
                         "(2025). Decoding the Contribution of Shoulder and Elbow Mechanics to Barbell "
                         "Kinematics and the Sticking Region in Bench and Overhead Press Exercises. "
                         "J Funct Morphol Kinesiol. PMC12372072.",
                citation_support="PMC9354811 defines the full overhead end position as \"the simultaneous "
                                 "scapular upward rotation …, together with the humerus abduction and "
                                 "elbow extension\" being what \"makes the overhead press suitable to "
                                 "stimulate upper trapezius, deltoids and triceps\"; PMC12372072 treats "
                                 "the lift as complete only when \"the barbell reaches its final "
                                 "position\". A press that stalls below overhead never reaches that end "
                                 "position, so the target motion and its musculature are never fully "
                                 "loaded. (Detection criterion is a shoulder-width-normalized "
                                 "substitution for the spec's head-height wording -- see docstring.)",
            )
        )
    return detections


def rule_forward_head_barpath(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Flag a forward-head / bar-ahead-of-midline finish: (a) the ear sitting anterior to the
    shoulder line by > 0.30 shoulder-widths, and/or (b) at lockout, the wrist not stacked over
    the shoulder but anterior by > 0.30 shoulder-widths.

    HARD VIEW GATE (deliberately not a confidence multiplier): both cues are PURE horizontal
    offsets, and their direction is meaningless unless the subject's facing is known. Outside a
    sagittal view the sign of `shoulder_x - point_x` is arbitrary, so firing there -- even at
    reduced confidence -- would emit a confidently-wrong DIRECTION claim ("head juts forward")
    on data that cannot support any direction at all. A wrong direction is worse than silence,
    so off-sagittal views return no detections rather than low-confidence ones.

    Sign convention: anterior = -x, the single module-wide convention documented in
    `ohp_compute_raw` and encoded by `rule_excessive_back_lean`. Do not introduce a second one.

    Known weakness -- the shoulder-width normalizer is worst exactly where this rule is gated:
    - In a true sagittal view the two shoulder landmarks project nearly on top of each other, so
      shoulder_width shrinks and the normalized offsets INFLATE. The > 1e-6 guard in
      `ohp_compute_raw` prevents a divide-by-zero, not this inflation.
    - Worse, if the far shoulder is fully occluded, `visible_point` drops it, `shoulder_width`
      becomes NaN, and every metric this rule reads goes NaN -- i.e. from a hard side view this
      rule can go SILENT rather than merely noisy. It has not been exercised on real sagittal
      video (the unit fixture keeps both shoulders fully visible), so its real-world hit rate
      is unknown on top of its threshold being unvalidated like the rest of this module."""
    if ctx.view_type not in SAGITTAL_VIEWS:
        return []

    lockout_phases = _lockout_phases(core)

    def worst_offset(frame: CoreFrame) -> float:
        """Larger = further anterior = worse. The bar-path cue only counts at lockout."""
        ear = frame.m("ear_forward_offset")
        wrist = frame.m("wrist_forward_offset") if frame.phase in lockout_phases else np.nan
        finite = [value for value in (ear, wrist) if np.isfinite(value)]
        return max(finite) if finite else np.nan

    forward_mask = [
        frame.valid and np.isfinite(worst_offset(frame)) and worst_offset(frame) > 0.30
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(forward_mask, ctx.min_frames):
        segment = core[start : end + 1]
        values = [worst_offset(frame) for frame in segment]
        max_offset = float(np.nanmax(values))
        ear_values = [frame.m("ear_forward_offset") for frame in segment]
        wrist_values = [
            frame.m("wrist_forward_offset") if frame.phase in lockout_phases else np.nan
            for frame in segment
        ]
        max_ear = float(np.nanmax(ear_values)) if any(np.isfinite(v) for v in ear_values) else np.nan
        max_wrist = (
            float(np.nanmax(wrist_values)) if any(np.isfinite(v) for v in wrist_values) else np.nan
        )
        severity = severity_from_range(max_offset, 0.30, 0.60, lower_is_worse=False)
        detections.append(
            build_detection(
                fault_id="ohp_forward_head_barpath",
                fault_name="Forward Head / Bar Path Forward of Midline",
                # Verified to resolve to "Forward Head Posture" / "Overhead Press:Forward Head
                # Posture" via graph_retrieval.resolve_nodes.
                kg_query="Forward Head Posture",
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=values,
                severity=severity,
                confidence=severity,
                # Spec rates this medium-high from `side`; reported as "medium" because the
                # rule only ever runs on sagittal views (so there is no unobservable branch to
                # downgrade) and its normalizer is weakly conditioned there -- see docstring.
                observability="medium",
                evidence={
                    "max_ear_forward_offset": round(max_ear, 4) if np.isfinite(max_ear) else 0.0,
                    "max_wrist_forward_offset": round(max_wrist, 4) if np.isfinite(max_wrist) else 0.0,
                    "threshold": 0.30,
                    "primary_label": "anterior offset from shoulder line (shoulder-widths)",
                    "primary_value": round(max_offset, 4),
                    "primary_threshold": 0.30,
                },
                citation="Abdelraouf OR, Abdel-Aziem AA, Alkhamees NH, Ibrahim ZM, Aboelela EM, "
                         "Dawood RS, Ashour AA. (2026). Acute Effects of High-Load Training to Failure "
                         "vs. Non-Failure on Posture and Core Endurance in Collegiate Weightlifters: A "
                         "Crossover Study. J Clin Med. PMC13116542; mechanism from Gregori P, La Bruna M, "
                         "Papalia GF, et al. (2026). Spine alignment influences shoulder range of motion "
                         "and scapular orientation: A systematic review. J Exp Orthop. PMC13086636, and "
                         "Al Hammadi MI, Shah ZA, Rathod RK, Seddik MA. (2025). Shoulder Impingement Pain "
                         "Syndrome: Pathophysiology, Diagnosis, and a Review of Current Treatment "
                         "Strategies. Cureus. PMC12514857.",
                citation_support="PMC13116542 found high-load overhead-press training to failure "
                                 "significantly reduced the craniovertebral angle (defining \"a "
                                 "craniovertebral angle … less than 48 degrees … as forward head "
                                 "posture\"), i.e. pressing measurably drives the head forward. "
                                 "PMC13086636 reports \"greater thoracic kyphosis is associated with … "
                                 "reduced shoulder abduction … and flexion,\" and PMC12514857 notes "
                                 "forward head / thoracic kyphosis \"reduce[s] the subacromial space\" -- "
                                 "giving both the ROM (performance) and impingement (injury) rationale "
                                 "for flagging a forward head and an un-stacked bar path.",
            )
        )
    return detections


OHP_DETECTOR = MovementDetector(
    "Overhead Press",
    OHP_METRIC_KEYS,
    ohp_compute_raw,
    ohp_assign_phases,
    (
        rule_incomplete_lockout,
        rule_excessive_back_lean,
        rule_asymmetric_press,
        rule_insufficient_elevation,
        rule_forward_head_barpath,
    ),
)

registry.register(OHP_DETECTOR)
