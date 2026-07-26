from __future__ import annotations

from typing import Sequence

import numpy as np

from src.pose.geometry import contiguous_true_segments, severity_from_range
from src.pose.movements.base import CoreFrame, MovementDetector, RuleContext
from src.pose.movements import registry
from src.pose.pose_rule_detector import (
    HEEL_OBSERVABLE_VIEWS,
    KNEE_FORWARD_MILD,
    KNEE_FORWARD_SEVERE,
    SIDE_VIEW_CONF_THRESHOLD,
    UNKNOWN_VIEW,
    VIEW_UNAVAILABLE_CONFIDENCE_SCALE,
    PoseRuleDetection,
    assign_phases,
    build_detection,
    raw_frame_metrics,
)

METRIC_KEYS: tuple[str, ...] = (
    "avg_knee_angle",
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_ankle_angle",
    "right_ankle_angle",
    "hip_minus_knee_y",
    "knee_width_to_ankle_width",
    "knee_forward_ratio",
    "torso_lean_deg",
    "heel_height_delta",
)

ACTIVE_PHASES = {"descent", "bottom", "ascent"}


def compute_raw(frames: Sequence[object], fps: float) -> list[dict]:
    return [raw_frame_metrics(frame, fps=fps) for frame in frames]


def rule_knees_inward(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    observable_alignment = ctx.view_type in {"rear", "rear_oblique", "front", "front_oblique"}
    inward_mask = [
        frame.valid
        and frame.phase in ACTIVE_PHASES
        and np.isfinite(frame.m("knee_width_to_ankle_width"))
        and frame.m("knee_width_to_ankle_width") < 0.82
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(inward_mask, ctx.min_frames):
        segment = core[start : end + 1]
        ratios = [frame.m("knee_width_to_ankle_width") for frame in segment]
        peak_score = [0.82 - value if np.isfinite(value) else np.nan for value in ratios]
        min_ratio = float(np.nanmin(ratios))
        severity = severity_from_range(min_ratio, 0.82, 0.70, lower_is_worse=True)
        detections.append(
            build_detection(
                fault_id="knees_inward",
                fault_name="Knees Inward / Knee Valgus",
                kg_query="Knee Valgus",
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=peak_score,
                severity=severity,
                confidence=severity * (1.0 if observable_alignment else 0.65),
                observability="high" if observable_alignment else "medium",
                evidence={
                    "min_knee_width_to_ankle_width": round(min_ratio, 4),
                    "threshold": 0.82,
                    # `primary_*` names the one metric to surface (with the threshold it breached)
                    # so the UI/chat never guess it from key order; see keyEvidence in retrieval.ts.
                    "primary_label": "knee/ankle width",
                    "primary_value": round(min_ratio, 4),
                    "primary_threshold": 0.82,
                },
                citation="Ford KR, Nguyen AD, Dischiavi SL, Hegedus EJ, Zuk EF, Taylor JB. (2015). "
                         "An evidence-based review of hip-focused neuromuscular exercise interventions "
                         "to address dynamic lower extremity valgus. Open Access J Sports Med. PMC4556293.",
                citation_support="Knee abduction moment predicted future ACL injury risk with 73% "
                                 "sensitivity and 78% specificity in young female athletes.",
            )
        )
    return detections


def rule_knees_forward(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    observable_side = ctx.view_type == "side" and ctx.view_confidence >= SIDE_VIEW_CONF_THRESHOLD
    forward_mask = [
        frame.valid
        and frame.phase in ACTIVE_PHASES
        and observable_side
        and np.isfinite(frame.m("knee_forward_ratio"))
        and frame.m("knee_forward_ratio") > KNEE_FORWARD_MILD
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(forward_mask, ctx.min_frames):
        segment = core[start : end + 1]
        ratios = [frame.m("knee_forward_ratio") for frame in segment]
        max_ratio = float(np.nanmax(ratios))
        severity = severity_from_range(max_ratio, KNEE_FORWARD_MILD, KNEE_FORWARD_SEVERE, lower_is_worse=False)
        detections.append(
            build_detection(
                fault_id="knees_forward",
                fault_name="Knees Forward / Anterior Knee Translation",
                kg_query="Anterior Knee Translation",
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=ratios,
                severity=severity,
                confidence=severity,
                observability="high",
                evidence={
                    "max_knee_forward_ratio": round(max_ratio, 4),
                    "threshold": KNEE_FORWARD_MILD,
                    "view_type": ctx.view_type,
                    "primary_label": "knee-forward ratio",
                    "primary_value": round(max_ratio, 4),
                    "primary_threshold": KNEE_FORWARD_MILD,
                },
                citation="Zellmer M, et al. (2019). Patellar tendon stress between two variations "
                         "of the forward step lunge. J Sport Health Sci. PMC6523035.",
                citation_support="Moving the knee in front of the toes increased peak patellar "
                                 "tendon stress by 11.1% and peak knee extension moment by 25.8% (p<0.001).",
            )
        )
    if not observable_side:
        side_candidates = [frame for frame in core if frame.valid and frame.phase in ACTIVE_PHASES]
        if side_candidates:
            detections.append(
                build_detection(
                    fault_id="knees_forward",
                    fault_name="Knees Forward / Anterior Knee Translation",
                    kg_query="Anterior Knee Translation",
                    retrieval_mode="kg",
                    segment_metrics=side_candidates[:1],
                    score_values=[0.0],
                    severity=0.0,
                    confidence=0.0,
                    observability="low",
                    evidence={
                        "reason": "side view required for reliable knee-to-toe projection",
                        "view_type": ctx.view_type,
                        "view_confidence": round(ctx.view_confidence, 4),
                    },
                    citation="Zellmer M, et al. (2019). Patellar tendon stress between two variations "
                             "of the forward step lunge. J Sport Health Sci. PMC6523035.",
                    citation_support="Moving the knee in front of the toes increased peak patellar "
                                     "tendon stress by 11.1% and peak knee extension moment by 25.8% (p<0.001).",
                )
            )
    return detections


def rule_shallow_depth(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    # Spec rates depth high on side/front/front_oblique and medium on rear/rear_oblique (the hip
    # crease is occluded), and says nothing about `unknown` -- which is the view estimator's
    # "no usable evidence" verdict, not a view. Folding it in with rear would still hand a
    # geometry-less clip full confidence, so it takes the same medium/x0.65 treatment
    # `rule_knees_inward` and `rule_forward_lean` already give it.
    view_unavailable = ctx.view_type == UNKNOWN_VIEW
    depth_mask = [
        frame.valid
        and frame.phase == "bottom"
        and (
            (np.isfinite(frame.m("hip_minus_knee_y")) and frame.m("hip_minus_knee_y") < -0.02)
            or (np.isfinite(frame.m("avg_knee_angle")) and frame.m("avg_knee_angle") > 105.0)
        )
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(depth_mask, ctx.min_frames):
        segment = core[start : end + 1]
        hip_values = [frame.m("hip_minus_knee_y") for frame in segment]
        knee_values = [frame.m("avg_knee_angle") for frame in segment]
        min_hip_depth = float(np.nanmin(hip_values))
        max_knee_angle = float(np.nanmax(knee_values))
        hip_severity = severity_from_range(min_hip_depth, -0.02, -0.10, lower_is_worse=True)
        knee_severity = severity_from_range(max_knee_angle, 105.0, 125.0, lower_is_worse=False)
        severity = max(hip_severity, knee_severity)
        # Depth has two candidate axes; the primary display metric is whichever drove the severity.
        if hip_severity >= knee_severity:
            primary_label, primary_value, primary_threshold = "hip-to-knee depth", round(min_hip_depth, 4), -0.02
        else:
            primary_label, primary_value, primary_threshold = "knee flexion angle", round(max_knee_angle, 2), 105.0
        detections.append(
            build_detection(
                fault_id="shallow_depth",
                fault_name="Shallow Depth",
                kg_query="Shallow Depth",
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=[max(hip_severity, knee_severity) for _ in segment],
                severity=severity,
                confidence=severity * (VIEW_UNAVAILABLE_CONFIDENCE_SCALE if view_unavailable else 1.0),
                observability="medium"
                if view_unavailable or ctx.view_type in {"rear", "rear_oblique"}
                else "high",
                evidence={
                    "min_hip_minus_knee_y": round(min_hip_depth, 4),
                    "max_avg_knee_angle": round(max_knee_angle, 2),
                    "hip_threshold": -0.02,
                    "knee_angle_threshold": 105.0,
                    "primary_label": primary_label,
                    "primary_value": primary_value,
                    "primary_threshold": primary_threshold,
                },
                citation="Hartmann H, Wirth K, Klusemann M. (2013). Analysis of the load on the knee "
                         "joint and vertebral column with changes in squatting depth and weight load. "
                         "Sports Medicine 43(10):993-1008. PMID 23821469.",
                citation_support="Half and quarter squat training with heavy loads favors long-term "
                                 "degenerative changes in the knee and spinal joints versus deep squats.",
            )
        )
    return detections


def rule_forward_lean(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    observable_lean = ctx.view_type in {"side", "rear_oblique", "front_oblique"}
    lean_mask = [
        frame.valid and np.isfinite(frame.m("torso_lean_deg")) and frame.m("torso_lean_deg") > 35.0
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(lean_mask, ctx.min_frames):
        segment = core[start : end + 1]
        values = [frame.m("torso_lean_deg") for frame in segment]
        max_lean = float(np.nanmax(values))
        severity = severity_from_range(max_lean, 35.0, 55.0, lower_is_worse=False)
        detections.append(
            build_detection(
                fault_id="excessive_forward_lean",
                fault_name="Excessive Forward Lean",
                kg_query="Excessive Forward Lean",
                retrieval_mode="kg",
                segment_metrics=segment,
                score_values=values,
                severity=severity,
                confidence=severity * (1.0 if observable_lean else 0.65),
                observability="high" if observable_lean else "medium",
                evidence={
                    "max_torso_lean_deg": round(max_lean, 2),
                    "threshold": 35.0,
                    "primary_label": "torso lean angle",
                    "primary_value": round(max_lean, 2),
                    "primary_threshold": 35.0,
                },
                citation="Moreira VM, et al. (2023). Analysis of Muscle Strength and Electromyographic "
                         "Activity during Different Deadlift Positions. Muscles. PMC12225233.",
                citation_support="Leaning the trunk forward raises spinal flexion torque, requiring "
                                 "higher erector spinae activation and strength to resist trunk flexion.",
            )
        )
    return detections


def rule_heel_rise(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    # Heel-vs-toe height only survives the projection from a lateral or oblique camera; head-on
    # (and on an `unknown` view, where no camera geometry was established at all) the fault is
    # "nearly invisible" per spec, so the verdict is downgraded to `low` and discounted rather
    # than emitted as confidently as a sagittal one. `low` also sorts it behind every observed
    # fault in `run_detector`, so a barely-seen heel cue cannot outrank one the camera did show.
    observable_heel = ctx.view_type in HEEL_OBSERVABLE_VIEWS
    setup_heel = [
        frame.m("heel_height_delta")
        for frame in core
        if frame.valid and frame.phase == "setup" and np.isfinite(frame.m("heel_height_delta"))
    ]
    baseline = float(np.mean(setup_heel)) if setup_heel else np.nan
    heel_mask = [
        frame.valid
        and frame.phase == "bottom"
        and np.isfinite(frame.m("heel_height_delta"))
        and np.isfinite(baseline)
        and frame.m("heel_height_delta") - baseline > 0.015
        for frame in core
    ]
    detections: list[PoseRuleDetection] = []
    for start, end in contiguous_true_segments(heel_mask, ctx.min_frames):
        segment = core[start : end + 1]
        values = [frame.m("heel_height_delta") - baseline for frame in segment]
        max_lift = float(np.nanmax(values))
        severity = severity_from_range(max_lift, 0.015, 0.055, lower_is_worse=False)
        detections.append(
            build_detection(
                fault_id="heel_rise",
                fault_name="Heel Rise",
                kg_query="heel rise squat ankle dorsiflexion",
                retrieval_mode="rag",
                segment_metrics=segment,
                score_values=values,
                severity=severity,
                confidence=severity * (1.0 if observable_heel else VIEW_UNAVAILABLE_CONFIDENCE_SCALE),
                observability="medium" if observable_heel else "low",
                evidence={
                    "max_heel_lift_delta": round(max_lift, 4),
                    "setup_baseline": round(baseline, 4),
                    "threshold": 0.015,
                    "view_type": ctx.view_type,
                    "primary_label": "heel lift",
                    "primary_value": round(max_lift, 4),
                    "primary_threshold": 0.015,
                },
                citation="Mata AJ, Hayashi H, Moreno PA, Dudley RI, Sorenson EA. (2021). Hip Flexion "
                         "Angles During Supine Range of Motion and Bodyweight Squats. Int J Exerc Sci "
                         "14(1):912-918.",
                citation_support="Heel elevation increased ankle excursion and squat depth (p<0.001); "
                                 "reduced dorsiflexion mobility can lead to compensatory joint moments "
                                 "up the kinetic chain, risking injury.",
            )
        )
    return detections


SQUAT_DETECTOR = MovementDetector(
    "Squat",
    METRIC_KEYS,
    compute_raw,
    assign_phases,
    (rule_knees_inward, rule_knees_forward, rule_shallow_depth, rule_forward_lean, rule_heel_rise),
    validated=True,
    rep_signal="avg_knee_angle",
    rep_polarity="min",
)

registry.register(SQUAT_DETECTOR)
