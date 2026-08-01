from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.knowledge.graph_retrieval import DEFAULT_GRAPH_FILE, retrieve_graph_context
from src.pose.geometry import (
    LANDMARK_COUNT, VISIBILITY_THRESHOLD,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    landmarks_to_array, visible_point, distance, angle_degrees, midpoint,
    line_angle_from_vertical, mean_visibility, mean_finite, centered_median,
    knee_forward_ratio, heel_height_delta, clip01, contiguous_true_segments,
    severity_from_range,
)
from src.pose.view_estimation import ViewEstimate, estimate_view_for_pose


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAG_DB_DIR = REPO_ROOT / "data" / "rag" / "vector_db"
SPLIT_NAMES = ("train", "val", "test")

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

# Thresholds and tunable constants for rule detection
KNEE_FORWARD_MILD = 0.10  # ratio above which knee-forward is considered present
KNEE_FORWARD_SEVERE = 0.30  # ratio at or above which knee-forward is severe
SIDE_VIEW_CONF_THRESHOLD = 0.20  # min view confidence to treat a 'side' view as reliable
# Spec convention (design doc §3): "Confidence is scaled down when the required view is
# unavailable (the coded squat detector multiplies by ~0.65)". Named here so every rule that
# applies it shares one number instead of re-typing the literal.
VIEW_UNAVAILABLE_CONFIDENCE_SCALE = 0.65
# `estimate_view_for_pose` returns "unknown" only when a clip FAILS its evidence floor
# (valid_frame_ratio < 0.15, or every view score < 0.20) -- i.e. the camera geometry could not
# be established at all. Rules must treat it as an unavailable view, never as a default-good
# one: resolving it to a rule's best branch hands the worst clips the most confident verdicts.
UNKNOWN_VIEW = "unknown"
# Heel-vs-toe height is a sagittal cue -- "medium on side / oblique ... nearly invisible
# head-on" (spec, Squat / Heel Rise).
HEEL_OBSERVABLE_VIEWS = frozenset({"side", "front_oblique", "rear_oblique"})



@dataclass(frozen=True)
class FrameMetrics:
    frame_index: int
    time: float
    phase: str
    valid: bool
    lower_body_visibility: float
    avg_knee_angle: float
    left_knee_angle: float
    right_knee_angle: float
    left_hip_angle: float
    right_hip_angle: float
    left_ankle_angle: float
    right_ankle_angle: float
    hip_minus_knee_y: float
    knee_width_to_ankle_width: float
    knee_forward_ratio: float
    torso_lean_deg: float
    heel_height_delta: float


@dataclass(frozen=True)
class PoseRuleDetection:
    fault_id: str
    fault_name: str
    kg_query: str
    retrieval_mode: str
    severity: float
    confidence: float
    observability: str
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    peak_frame: int
    phase: str
    evidence: dict[str, float | int | str]
    citation: str = ""
    citation_support: str = ""
    # Per-rep attribution, populated by `run_detector` when rules ran on a single rep's slice.
    # All three stay at their zero/empty default on the whole-clip fallback path, where there
    # are no repetitions to attribute a detection to. `rep_count`/`occurred_reps` are owned by
    # `merge_by_fault` (a later task); `run_detector` itself only ever sets `rep_count=1`.
    rep_index: int = 0
    occurred_reps: tuple[int, ...] = ()
    rep_count: int = 0


@dataclass(frozen=True)
class PoseRuleRequest:
    split_name: str
    video_id: str
    pose_json_path: Path
    output_path: Path


def load_pose_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}.")
    return data


def raw_frame_metrics(frame: object, fps: float) -> dict[str, float | int | bool]:
    if not isinstance(frame, dict):
        return {"valid": False}

    points = landmarks_to_array(frame.get("landmarks"))
    frame_index = int(frame.get("frame_index", 0) or 0)
    time = frame_index / fps if fps > 0 else 0.0
    required = (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE)
    valid = all(visible_point(points, index, dims=2) is not None for index in required)
    if not valid:
        return {
            "frame_index": frame_index,
            "time": time,
            "valid": False,
            "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
        }

    left_knee_angle = angle_degrees(points, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    right_knee_angle = angle_degrees(points, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    left_hip_angle = angle_degrees(points, LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
    right_hip_angle = angle_degrees(points, RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
    left_ankle_angle = angle_degrees(points, LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX)
    right_ankle_angle = angle_degrees(points, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX)

    hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP)
    knee_mid = midpoint(points, LEFT_KNEE, RIGHT_KNEE)
    shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER)
    ankle_width = distance(points, LEFT_ANKLE, RIGHT_ANKLE, dims=2)
    knee_width = distance(points, LEFT_KNEE, RIGHT_KNEE, dims=2)
    knee_width_to_ankle_width = knee_width / ankle_width if np.isfinite(ankle_width) and ankle_width > 1e-8 else np.nan

    left_forward = knee_forward_ratio(points, LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX)
    right_forward = knee_forward_ratio(points, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX)
    heel_delta_left = heel_height_delta(points, LEFT_HEEL, LEFT_FOOT_INDEX)
    heel_delta_right = heel_height_delta(points, RIGHT_HEEL, RIGHT_FOOT_INDEX)

    return {
        "frame_index": frame_index,
        "time": time,
        "valid": True,
        "lower_body_visibility": mean_visibility(points, LOWER_BODY_LANDMARKS),
        "avg_knee_angle": mean_finite([left_knee_angle, right_knee_angle]),
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "left_ankle_angle": left_ankle_angle,
        "right_ankle_angle": right_ankle_angle,
        "hip_minus_knee_y": float(hip_mid[1] - knee_mid[1]) if hip_mid is not None and knee_mid is not None else np.nan,
        "knee_width_to_ankle_width": knee_width_to_ankle_width,
        "knee_forward_ratio": mean_finite([left_forward, right_forward]),
        "torso_lean_deg": line_angle_from_vertical(shoulder_mid, hip_mid),
        "heel_height_delta": mean_finite([heel_delta_left, heel_delta_right]),
    }


def assign_phases(raw_metrics: list[dict[str, float | int | bool]]) -> list[str]:
    if not raw_metrics:
        return []
    hip_values = np.asarray([float(item.get("hip_minus_knee_y", np.nan)) for item in raw_metrics], dtype=np.float32)
    knee_angles = np.asarray([float(item.get("avg_knee_angle", np.nan)) for item in raw_metrics], dtype=np.float32)
    frame_count = len(raw_metrics)
    valid_hips = hip_values[np.isfinite(hip_values)]
    valid_knees = knee_angles[np.isfinite(knee_angles)]
    if valid_hips.size == 0 and valid_knees.size == 0:
        return ["unknown" for _ in raw_metrics]

    bottom_index = int(np.nanargmin(np.where(np.isfinite(knee_angles), knee_angles, np.inf)))
    hip_bottom_threshold = float(np.percentile(valid_hips, 70)) if valid_hips.size else np.inf
    knee_bottom_threshold = float(np.percentile(valid_knees, 30)) if valid_knees.size else -np.inf
    setup_cutoff = max(1, int(frame_count * 0.15))
    lockout_cutoff = max(setup_cutoff + 1, int(frame_count * 0.85))

    phases: list[str] = []
    for index, item in enumerate(raw_metrics):
        if not item.get("valid"):
            phases.append("unknown")
        elif index < setup_cutoff:
            phases.append("setup")
        elif index >= lockout_cutoff:
            phases.append("lockout")
        elif hip_values[index] >= hip_bottom_threshold or knee_angles[index] <= knee_bottom_threshold:
            phases.append("bottom")
        elif index < bottom_index:
            phases.append("descent")
        else:
            phases.append("ascent")
    return phases


def compute_frame_metrics(frames: Sequence[object], fps: float) -> list[FrameMetrics]:
    raw = [raw_frame_metrics(frame, fps=fps) for frame in frames]
    phases = assign_phases(raw)
    smooth_fields = {
        name: centered_median([float(item.get(name, np.nan)) for item in raw], window=5)
        for name in (
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
    }

    metrics: list[FrameMetrics] = []
    for index, item in enumerate(raw):
        metrics.append(
            FrameMetrics(
                frame_index=int(item.get("frame_index", index) or index),
                time=float(item.get("time", 0.0) or 0.0),
                phase=phases[index],
                valid=bool(item.get("valid", False)),
                lower_body_visibility=float(item.get("lower_body_visibility", 0.0) or 0.0),
                avg_knee_angle=float(smooth_fields["avg_knee_angle"][index]),
                left_knee_angle=float(smooth_fields["left_knee_angle"][index]),
                right_knee_angle=float(smooth_fields["right_knee_angle"][index]),
                left_hip_angle=float(smooth_fields["left_hip_angle"][index]),
                right_hip_angle=float(smooth_fields["right_hip_angle"][index]),
                left_ankle_angle=float(smooth_fields["left_ankle_angle"][index]),
                right_ankle_angle=float(smooth_fields["right_ankle_angle"][index]),
                hip_minus_knee_y=float(smooth_fields["hip_minus_knee_y"][index]),
                knee_width_to_ankle_width=float(smooth_fields["knee_width_to_ankle_width"][index]),
                knee_forward_ratio=float(smooth_fields["knee_forward_ratio"][index]),
                torso_lean_deg=float(smooth_fields["torso_lean_deg"][index]),
                heel_height_delta=float(smooth_fields["heel_height_delta"][index]),
            )
        )
    return metrics


def dominant_phase(metrics: Sequence[FrameMetrics]) -> str:
    counts: dict[str, int] = {}
    for metric in metrics:
        counts[metric.phase] = counts.get(metric.phase, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0] if counts else "unknown"


def build_detection(
    *,
    fault_id: str,
    fault_name: str,
    kg_query: str,
    retrieval_mode: str,
    segment_metrics: Sequence[FrameMetrics],
    score_values: Sequence[float],
    severity: float,
    confidence: float,
    observability: str,
    evidence: dict[str, float | int | str],
    citation: str = "",
    citation_support: str = "",
) -> PoseRuleDetection:
    finite_scores = np.asarray(score_values, dtype=np.float32)
    if finite_scores.size and np.isfinite(finite_scores).any():
        peak_offset = int(np.nanargmax(np.where(np.isfinite(finite_scores), finite_scores, -np.inf)))
    else:
        peak_offset = 0
    peak = segment_metrics[peak_offset]
    start = segment_metrics[0]
    end = segment_metrics[-1]
    evidence = dict(evidence)
    evidence["peak_time"] = round(peak.time, 3)
    return PoseRuleDetection(
        fault_id=fault_id,
        fault_name=fault_name,
        kg_query=kg_query,
        retrieval_mode=retrieval_mode,
        severity=round(clip01(severity), 4),
        confidence=round(clip01(confidence), 4),
        observability=observability,
        start_time=round(start.time, 3),
        end_time=round(end.time, 3),
        start_frame=start.frame_index,
        end_frame=end.frame_index,
        peak_frame=peak.frame_index,
        phase=dominant_phase(segment_metrics),
        evidence=evidence,
        citation=citation,
        citation_support=citation_support,
    )


def detect_rule_segments(metrics: Sequence[FrameMetrics], fps: float, view_type: str, view_confidence: float) -> list[PoseRuleDetection]:
    min_frames = max(3, int(math.ceil(max(fps, 1.0) * 0.20)))
    detections: list[PoseRuleDetection] = []
    observable_alignment = view_type in {"rear", "rear_oblique", "front", "front_oblique"}
    observable_side = view_type == "side" and view_confidence >= SIDE_VIEW_CONF_THRESHOLD
    observable_lean = view_type in {"side", "rear_oblique", "front_oblique"}
    observable_heel = view_type in HEEL_OBSERVABLE_VIEWS
    view_unavailable = view_type == UNKNOWN_VIEW

    active_phases = {"descent", "bottom", "ascent"}
    inward_mask = [
        metric.valid
        and metric.phase in active_phases
        and np.isfinite(metric.knee_width_to_ankle_width)
        and metric.knee_width_to_ankle_width < 0.82
        for metric in metrics
    ]
    for start, end in contiguous_true_segments(inward_mask, min_frames):
        segment = metrics[start : end + 1]
        ratios = [metric.knee_width_to_ankle_width for metric in segment]
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

    forward_mask = [
        metric.valid
        and metric.phase in active_phases
        and observable_side
        and np.isfinite(metric.knee_forward_ratio)
        and metric.knee_forward_ratio > KNEE_FORWARD_MILD
        for metric in metrics
    ]
    for start, end in contiguous_true_segments(forward_mask, min_frames):
        segment = metrics[start : end + 1]
        ratios = [metric.knee_forward_ratio for metric in segment]
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
                    "view_type": view_type,
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
        side_candidates = [metric for metric in metrics if metric.valid and metric.phase in active_phases]
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
                        "view_type": view_type,
                        "view_confidence": round(view_confidence, 4),
                    },
                    citation="Zellmer M, et al. (2019). Patellar tendon stress between two variations "
                             "of the forward step lunge. J Sport Health Sci. PMC6523035.",
                    citation_support="Moving the knee in front of the toes increased peak patellar "
                                     "tendon stress by 11.1% and peak knee extension moment by 25.8% (p<0.001).",
                )
            )

    depth_mask = [
        metric.valid
        and metric.phase == "bottom"
        and (
            (np.isfinite(metric.hip_minus_knee_y) and metric.hip_minus_knee_y < -0.02)
            or (np.isfinite(metric.avg_knee_angle) and metric.avg_knee_angle > 105.0)
        )
        for metric in metrics
    ]
    for start, end in contiguous_true_segments(depth_mask, min_frames):
        segment = metrics[start : end + 1]
        hip_values = [metric.hip_minus_knee_y for metric in segment]
        knee_values = [metric.avg_knee_angle for metric in segment]
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
                if view_unavailable or view_type in {"rear", "rear_oblique"}
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

    lean_mask = [
        metric.valid and np.isfinite(metric.torso_lean_deg) and metric.torso_lean_deg > 35.0
        for metric in metrics
    ]
    for start, end in contiguous_true_segments(lean_mask, min_frames):
        segment = metrics[start : end + 1]
        values = [metric.torso_lean_deg for metric in segment]
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

    setup_heel = [
        metric.heel_height_delta
        for metric in metrics
        if metric.valid and metric.phase == "setup" and np.isfinite(metric.heel_height_delta)
    ]
    baseline = float(np.mean(setup_heel)) if setup_heel else np.nan
    heel_mask = [
        metric.valid
        and metric.phase == "bottom"
        and np.isfinite(metric.heel_height_delta)
        and np.isfinite(baseline)
        and metric.heel_height_delta - baseline > 0.015
        for metric in metrics
    ]
    for start, end in contiguous_true_segments(heel_mask, min_frames):
        segment = metrics[start : end + 1]
        values = [metric.heel_height_delta - baseline for metric in segment]
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
                    "view_type": view_type,
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

    detections.sort(key=lambda item: (item.observability == "low", -item.severity, item.start_frame))
    return detections


def json_safe_view_payload(view: ViewEstimate) -> dict[str, Any]:
    """Serialize a ViewEstimate for the API/DB boundary, replacing non-finite
    floats with None.

    torso_width_ratio_mean is NaN whenever a clip carries no measurable-width
    evidence (e.g. degenerate/empty pose JSON) -- that is deliberate, honest
    "no evidence" signaling from estimate_view_for_pose, not a bug. But NaN
    survives dataclasses.asdict() untouched, and postgrest's httpx JSON
    encoder serializes with allow_nan=False: it raises ValueError before any
    network call, which the broad except in the analyze route swallows,
    silently dropping the analysis from the user's history with no error
    surfaced. None/null is the honest JSON representation of "no evidence";
    0.0 is not. (The HTTP response path is unaffected -- FastAPI/pydantic
    already maps NaN to null there.)
    """
    payload = asdict(view)
    for key, value in payload.items():
        if isinstance(value, float) and not math.isfinite(value):
            payload[key] = None
    return payload


def detect_pose_rules_from_payload(
    payload: dict[str, Any],
    *,
    pose_json_path: Path | None = None,
    video_id: str | None = None,
    include_retrieval: bool = False,
    graph_file: Path = DEFAULT_GRAPH_FILE,
    rag_db_dir: Path = DEFAULT_RAG_DB_DIR,
    movement: str | None = None,
    # -1 (not None) is the "caller said nothing" sentinel: None is a meaningful value here,
    # meaning "analyze every rep".
    max_reps: int | None = -1,
) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        frames = []

    fps = float(metadata.get("fps", 0.0) or 0.0)
    if pose_json_path is not None:
        view = estimate_view_for_pose(pose_json_path)
        view_payload = json_safe_view_payload(view)
        view_type = view.view_type
        view_confidence = view.view_confidence
    else:
        view_payload = {"view_type": "unknown", "view_confidence": 0.0}
        view_type = "unknown"
        view_confidence = 0.0

    from src.pose.movements import registry
    from src.pose.movements.base import DEFAULT_MAX_REPS, run_detector

    detector = registry.get_detector(movement)
    effective_max_reps = DEFAULT_MAX_REPS if max_reps == -1 else max_reps
    run = run_detector(
        detector,
        frames,
        fps if fps > 0 else 30.0,
        view_type,
        view_confidence,
        max_reps=effective_max_reps,
    )
    core, detections = run.core, run.detections

    analyzed_indices = [rep.index for rep in run.analyzed]
    analyzed_frames = sum(rep.end - rep.start + 1 for rep in run.analyzed) or len(core)
    valid_frames = [c for c in core if c.valid]
    result = {
        "video_id": video_id or (pose_json_path.stem if pose_json_path else ""),
        # The CANONICAL movement name, taken from the resolved detector rather than the caller's
        # string, so "push-up" normalises to "Push-up". That exact spelling is simultaneously the
        # KG `movement` scope and the frontend's movement.<Name> i18n key. Echoing it here (not in
        # the web layer) means the CLI's written JSON carries it too, and a stored analysis records
        # which rules produced it -- permanently, without depending on a database column.
        "movement": detector.name,
        "pose_json_path": str(pose_json_path) if pose_json_path else "",
        "metadata": metadata,
        "view": view_payload,
        "quality": {
            "total_frames": len(frames),
            "valid_frames": len(valid_frames),
            "valid_frame_ratio": round(len(valid_frames) / len(frames), 4) if frames else 0.0,
            "lower_body_visibility_mean": round(float(np.mean([c.lower_body_visibility for c in core])), 4)
            if core
            else 0.0,
            # ADDITIVE. The existing denominators above stay whole-clip on purpose -- they are a
            # compatibility surface for backend/app/services/analysis.py, the frontend, and
            # src/knowledge/perception_to_graph.py.
            "analyzed_frames": analyzed_frames if core else 0,
            "analyzed_frame_ratio": round(analyzed_frames / len(frames), 4) if frames else 0.0,
        },
        "detections": [asdict(detection) for detection in detections],
        "retrievals": [],
        "frame_metrics": [
            {
                "frame_index": c.frame_index,
                "time": c.time,
                "phase": c.phase,
                "valid": c.valid,
                "lower_body_visibility": c.lower_body_visibility,
                **c.metrics,
            }
            for c in core
        ],
        # Which repetitions were found and which were actually scored. `segments` exists so a UI
        # can show which spans were examined: when whole stretches of a clip are never looked
        # at, the interface must not imply they were clean.
        "reps": {
            "detected": len(run.reps),
            # Repetition indices scored PER-REPETITION. Empty on any fallback (`run.fallback is
            # not None`) -- not because nothing was examined, but because on fallback the whole
            # clip was scored as one unit instead of rep-by-rep. See `segments[].analyzed` below
            # for whether a given span was actually looked at.
            "analyzed": analyzed_indices,
            "max_reps": effective_max_reps,
            "fallback": run.fallback,
            "segments": [
                {
                    "index": rep.index,
                    "start_frame": core[rep.start].frame_index,
                    "end_frame": core[rep.end].frame_index,
                    "start_time": round(core[rep.start].time, 3),
                    "end_time": round(core[rep.end].time, 3),
                    # On a normal (non-fallback) run this mirrors `analyzed_indices`: only the
                    # sampled reps were scored. On any fallback the whole clip -- including every
                    # span listed here -- WAS examined, just as one unit rather than rep-by-rep,
                    # so every segment is genuinely analyzed=true. `reps.analyzed` staying `[]` on
                    # fallback must not be read as "these spans are unexamined" -- see the comment
                    # on `reps.analyzed` above.
                    "analyzed": True if run.fallback is not None else rep.index in set(analyzed_indices),
                    "partial": rep.partial,
                }
                for rep in run.reps
            ],
        },
    }
    if include_retrieval:
        result["retrievals"] = retrieve_contexts_for_detections(
            result["detections"],
            graph_file=graph_file,
            rag_db_dir=rag_db_dir,
            movement=movement,
        )
    return result


def detect_pose_rules_from_json(
    pose_json_path: str | Path,
    *,
    video_id: str | None = None,
    include_retrieval: bool = False,
    graph_file: Path = DEFAULT_GRAPH_FILE,
    rag_db_dir: Path = DEFAULT_RAG_DB_DIR,
    movement: str | None = None,
    max_reps: int | None = -1,
) -> dict[str, Any]:
    path = Path(pose_json_path)
    return detect_pose_rules_from_payload(
        load_pose_json(path),
        pose_json_path=path,
        video_id=video_id,
        include_retrieval=include_retrieval,
        graph_file=graph_file,
        rag_db_dir=rag_db_dir,
        movement=movement,
        max_reps=max_reps,
    )


def retrieve_contexts_for_detections(
    detections: Sequence[dict[str, Any]],
    *,
    graph_file: Path = DEFAULT_GRAPH_FILE,
    rag_db_dir: Path = DEFAULT_RAG_DB_DIR,
    top_k: int = 5,
    movement: str | None = None,
) -> list[dict[str, Any]]:
    retrievals: list[dict[str, Any]] = []
    for detection in detections:
        query = str(detection.get("kg_query") or detection.get("fault_name") or "")
        retrieval_mode = str(detection.get("retrieval_mode") or "kg")
        if not query:
            continue
        if retrieval_mode == "rag":
            from src.knowledge.rag_vector_db import query_vector_db

            context = {
                "query": query,
                "results": query_vector_db(query, db_dir=rag_db_dir, top_k=top_k),
            }
        else:
            context = retrieve_graph_context(query, graph_file=graph_file, hops=1, max_seeds=3, movement=movement)
        retrievals.append(
            {
                "fault_id": detection.get("fault_id", ""),
                "fault_name": detection.get("fault_name", ""),
                "query_text": query,
                "retrieval_mode": retrieval_mode,
                "context": context,
            }
        )
    return retrievals


def load_json_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}.")
    return [str(item) for item in data]


def parse_split_names(value: str) -> list[str]:
    split_names = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(split_names) - set(SPLIT_NAMES))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported splits: {', '.join(invalid)}")
    return split_names


def parse_max_reps(value: str) -> int | None:
    """Parse ``--max-reps``. ``all`` and ``0`` both mean every repetition."""
    text = (value or "").strip().lower()
    if text == "all":
        return None
    if not text.isdigit():
        raise argparse.ArgumentTypeError(
            f"--max-reps must be a non-negative integer or 'all', got {value!r}"
        )
    count = int(text)
    return None if count == 0 else count


def build_requests(
    pose_json_dir: Path,
    split_dir: Path,
    output_dir: Path,
    split_names: Sequence[str],
) -> list[PoseRuleRequest]:
    requests: list[PoseRuleRequest] = []
    for split_name in split_names:
        for video_id in load_json_list(split_dir / f"{split_name}_keys.json"):
            pose_json_path = pose_json_dir / split_name / f"{video_id}.json"
            if not pose_json_path.exists():
                continue
            requests.append(
                PoseRuleRequest(
                    split_name=split_name,
                    video_id=video_id,
                    pose_json_path=pose_json_path,
                    output_path=output_dir / split_name / f"{video_id}.json",
                )
            )
    return requests


def iter_limited(items: Sequence[PoseRuleRequest], limit: int | None) -> Iterable[PoseRuleRequest]:
    yield from items if limit is None else items[:limit]


def write_detection_json(path: Path, result: dict[str, Any], *, include_frames: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    if not include_frames:
        payload.pop("frame_metrics", None)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "video_id",
        "fault_id",
        "fault_name",
        "severity",
        "confidence",
        "observability",
        "start_time",
        "end_time",
        "peak_frame",
        "phase",
        "kg_query",
        "retrieval_mode",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rule-based MediaPipe squat form detection.")
    parser.add_argument("--pose-json", type=Path, default=None, help="Single pose JSON to process.")
    parser.add_argument(
        "--pose-json-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "pose_json",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "Splits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "pose_rule_detections",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Output path for single-file mode.")
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--splits", type=parse_split_names, default=list(SPLIT_NAMES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-frames", action="store_true")
    parser.add_argument("--no-retrieval", action="store_true", help="Skip KG/RAG retrieval context.")
    parser.add_argument(
        "--movement",
        default="Squat",
        help="Canonical movement name to detect (registered: 'Squat', 'Overhead Press', "
             "'Push-up', 'Lunge', 'Row'). Only Squat is validated against labeled data.",
    )
    # Local import, not module-level: `src.pose.movements.base` imports THIS module
    # (`PoseRuleDetection`) at module scope, so importing it back at module scope here would be
    # circular. `detect_pose_rules_from_payload` already defers the same import for the same
    # reason (see above). This is the one place `DEFAULT_MAX_REPS` is actually *defined*; the
    # argparse default below references it rather than repeating the literal `3`.
    from src.pose.movements.base import DEFAULT_MAX_REPS
    parser.add_argument(
        "--max-reps",
        type=parse_max_reps,
        default=DEFAULT_MAX_REPS,
        help="How many repetitions to analyze (first/middle/last are sampled). "
             "Use 0 or 'all' to analyze every repetition.",
    )
    args = parser.parse_args()

    summary_rows: list[dict[str, Any]] = []
    if args.pose_json is not None:
        result = detect_pose_rules_from_json(
            args.pose_json,
            include_retrieval=not args.no_retrieval,
            movement=args.movement,
            max_reps=args.max_reps,
        )
        output_path = args.output_json or args.output_dir / f"{args.pose_json.stem}.json"
        write_detection_json(output_path, result, include_frames=args.include_frames)
        for detection in result["detections"]:
            summary_rows.append({"split": "", "video_id": result["video_id"], **detection})
        print(f"Saved rule detections to {output_path}")
    else:
        requests = build_requests(args.pose_json_dir, args.split_dir, args.output_dir, args.splits)
        if not requests:
            raise SystemExit("No pose JSON files were found to process.")
        for index, request in enumerate(iter_limited(requests, args.limit), start=1):
            print(f"[{index}] Detecting squat rules for {request.split_name}/{request.video_id}...")
            result = detect_pose_rules_from_json(
                request.pose_json_path,
                video_id=request.video_id,
                include_retrieval=not args.no_retrieval,
                movement=args.movement,
                max_reps=args.max_reps,
            )
            write_detection_json(request.output_path, result, include_frames=args.include_frames)
            for detection in result["detections"]:
                summary_rows.append({"split": request.split_name, "video_id": request.video_id, **detection})
        print(f"Saved rule detections under {args.output_dir}")

    if args.summary_output is not None:
        write_summary_csv(args.summary_output, summary_rows)
        print(f"Saved summary CSV to {args.summary_output}")


if __name__ == "__main__":
    main()
