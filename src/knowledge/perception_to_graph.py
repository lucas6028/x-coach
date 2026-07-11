from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.knowledge.graph_retrieval import DEFAULT_GRAPH_FILE, retrieve_graph_context
from src.pose.pose_rule_detector import detect_pose_rules_from_json
from src.knowledge.rag_vector_db import DEFAULT_DB_DIR, query_vector_db


VISIBILITY_THRESHOLD = 0.5

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


@dataclass
class FaultDetection:
    fault: str
    query_text: str
    severity: float
    evidence: dict[str, float | int | str]


def _get_point(landmarks: list[dict[str, Any]], index: int) -> tuple[float, float] | None:
    if index >= len(landmarks):
        return None
    landmark = landmarks[index]
    if float(landmark.get("visibility", 0.0)) < VISIBILITY_THRESHOLD:
        return None
    return float(landmark["x"]), float(landmark["y"])


def _angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    denom = math.hypot(*ba) * math.hypot(*bc)
    if denom == 0:
        return 180.0
    cosine = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1]) / denom))
    return math.degrees(math.acos(cosine))


def _line_angle_from_vertical(top: tuple[float, float], bottom: tuple[float, float]) -> float:
    dx = top[0] - bottom[0]
    dy = top[1] - bottom[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-8))


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _safe_ratio(a: float, b: float) -> float | None:
    if abs(b) < 1e-8:
        return None
    return a / b


def _frame_metrics(frame: dict[str, Any]) -> dict[str, float] | None:
    landmarks = frame.get("landmarks")
    if not landmarks:
        return None

    left_hip = _get_point(landmarks, LEFT_HIP)
    right_hip = _get_point(landmarks, RIGHT_HIP)
    left_knee = _get_point(landmarks, LEFT_KNEE)
    right_knee = _get_point(landmarks, RIGHT_KNEE)
    left_ankle = _get_point(landmarks, LEFT_ANKLE)
    right_ankle = _get_point(landmarks, RIGHT_ANKLE)
    left_shoulder = _get_point(landmarks, LEFT_SHOULDER)
    right_shoulder = _get_point(landmarks, RIGHT_SHOULDER)
    left_heel = _get_point(landmarks, LEFT_HEEL)
    right_heel = _get_point(landmarks, RIGHT_HEEL)
    left_foot_index = _get_point(landmarks, LEFT_FOOT_INDEX)
    right_foot_index = _get_point(landmarks, RIGHT_FOOT_INDEX)

    required = [
        left_hip,
        right_hip,
        left_knee,
        right_knee,
        left_ankle,
        right_ankle,
        left_shoulder,
        right_shoulder,
    ]
    if any(point is None for point in required):
        return None

    left_knee_angle = _angle(left_hip, left_knee, left_ankle)
    right_knee_angle = _angle(right_hip, right_knee, right_ankle)
    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0

    hip_mid = _midpoint(left_hip, right_hip)
    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    knee_mid = _midpoint(left_knee, right_knee)

    ankle_width = abs(left_ankle[0] - right_ankle[0])
    knee_width = abs(left_knee[0] - right_knee[0])
    stance_ratio = _safe_ratio(knee_width, ankle_width)
    hip_to_knee_drop = hip_mid[1] - knee_mid[1]
    torso_lean_deg = _line_angle_from_vertical(shoulder_mid, hip_mid)

    heel_height = None
    if left_heel and right_heel and left_foot_index and right_foot_index:
        heel_height = ((left_heel[1] - left_foot_index[1]) + (right_heel[1] - right_foot_index[1])) / 2.0

    return {
        "frame_index": float(frame.get("frame_index", 0)),
        "avg_knee_angle": avg_knee_angle,
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "knee_to_ankle_width_ratio": stance_ratio if stance_ratio is not None else float("nan"),
        "hip_to_knee_drop": hip_to_knee_drop,
        "torso_lean_deg": torso_lean_deg,
        "heel_height_delta": heel_height if heel_height is not None else float("nan"),
    }


def detect_faults_from_pose_json(
    pose_json_path: str | Path,
    *,
    action: str = "Squat",
) -> dict[str, Any]:
    rule_result = detect_pose_rules_from_json(pose_json_path)
    return {
        "action": action,
        "pose_json_path": rule_result["pose_json_path"],
        "metadata": rule_result["metadata"],
        "view": rule_result["view"],
        "quality": rule_result["quality"],
        "detections": rule_result["detections"],
        "summary": {
            "valid_frames": rule_result["quality"]["valid_frames"],
            "total_frames": rule_result["quality"]["total_frames"],
            "valid_frame_ratio": rule_result["quality"]["valid_frame_ratio"],
        },
    }


def detect_faults_from_pose_json_legacy(
    pose_json_path: str | Path,
    *,
    action: str = "Squat",
) -> dict[str, Any]:
    pose_json_path = Path(pose_json_path)
    payload = json.loads(pose_json_path.read_text(encoding="utf-8"))
    frame_metrics = [metric for frame in payload.get("frames", []) if (metric := _frame_metrics(frame)) is not None]

    detections: list[FaultDetection] = []
    if not frame_metrics:
        return {
            "action": action,
            "pose_json_path": str(pose_json_path),
            "metadata": payload.get("metadata", {}),
            "detections": [],
            "summary": {"valid_frames": 0},
        }

    bottom_frame = min(frame_metrics, key=lambda item: item["avg_knee_angle"])
    narrowest_knee_frame = min(
        (item for item in frame_metrics if not math.isnan(item["knee_to_ankle_width_ratio"])),
        key=lambda item: item["knee_to_ankle_width_ratio"],
        default=None,
    )
    most_lean_frame = max(frame_metrics, key=lambda item: item["torso_lean_deg"])
    setup_window = frame_metrics[: max(3, min(10, len(frame_metrics) // 5 or 3))]
    baseline_heel = [
        item["heel_height_delta"]
        for item in setup_window
        if not math.isnan(item["heel_height_delta"])
    ]
    baseline_heel_mean = sum(baseline_heel) / len(baseline_heel) if baseline_heel else None

    width_ratio = (
        narrowest_knee_frame["knee_to_ankle_width_ratio"]
        if narrowest_knee_frame is not None
        else float("nan")
    )
    if not math.isnan(width_ratio) and width_ratio < 0.82:
        severity = min(1.0, max(0.0, (0.82 - width_ratio) / 0.32))
        detections.append(
            FaultDetection(
                fault="Knee Valgus",
                query_text="knee valgus",
                severity=severity,
                evidence={
                    "frame_index": int(narrowest_knee_frame["frame_index"]),
                    "knee_to_ankle_width_ratio": round(width_ratio, 3),
                    "avg_knee_angle": round(narrowest_knee_frame["avg_knee_angle"], 2),
                },
            )
        )

    if bottom_frame["avg_knee_angle"] > 105 or bottom_frame["hip_to_knee_drop"] < -0.02:
        angle_score = max(0.0, (bottom_frame["avg_knee_angle"] - 105) / 35.0)
        hip_score = max(0.0, (-0.02 - bottom_frame["hip_to_knee_drop"]) / 0.08)
        severity = min(1.0, max(angle_score, hip_score))
        detections.append(
            FaultDetection(
                fault="Shallow Depth",
                query_text="shallow depth",
                severity=severity,
                evidence={
                    "frame_index": int(bottom_frame["frame_index"]),
                    "avg_knee_angle": round(bottom_frame["avg_knee_angle"], 2),
                    "hip_to_knee_drop": round(bottom_frame["hip_to_knee_drop"], 3),
                },
            )
        )

    if most_lean_frame["torso_lean_deg"] > 35:
        severity = min(1.0, max(0.0, (most_lean_frame["torso_lean_deg"] - 35) / 20.0))
        detections.append(
            FaultDetection(
                fault="Excessive Forward Lean",
                query_text="excessive forward lean",
                severity=severity,
                evidence={
                    "frame_index": int(most_lean_frame["frame_index"]),
                    "torso_lean_deg": round(most_lean_frame["torso_lean_deg"], 2),
                },
            )
        )

    if baseline_heel_mean is not None and not math.isnan(bottom_frame["heel_height_delta"]):
        heel_lift = bottom_frame["heel_height_delta"] - baseline_heel_mean
        if heel_lift > 0.015:
            severity = min(1.0, max(0.0, (heel_lift - 0.015) / 0.04))
            detections.append(
                FaultDetection(
                    fault="Heel Rise",
                    query_text="heel rise",
                    severity=severity,
                    evidence={
                        "frame_index": int(bottom_frame["frame_index"]),
                        "heel_height_delta": round(heel_lift, 4),
                    },
                )
            )

    detections.sort(key=lambda item: item.severity, reverse=True)
    return {
        "action": action,
        "pose_json_path": str(pose_json_path),
        "metadata": payload.get("metadata", {}),
        "detections": [asdict(item) for item in detections],
        "summary": {
            "valid_frames": len(frame_metrics),
            "bottom_frame_index": int(bottom_frame["frame_index"]),
            "bottom_frame_avg_knee_angle": round(bottom_frame["avg_knee_angle"], 2),
            "bottom_frame_torso_lean_deg": round(bottom_frame["torso_lean_deg"], 2),
        },
    }


def retrieve_from_pose_faults(
    pose_json_path: str | Path,
    *,
    graph_file: Path = DEFAULT_GRAPH_FILE,
    rag_db_dir: Path = DEFAULT_DB_DIR,
    hops: int = 1,
    max_faults: int = 3,
    action: str = "Squat",
) -> dict[str, Any]:
    perception_result = detect_faults_from_pose_json(pose_json_path, action=action)
    detections = perception_result["detections"][:max_faults]

    retrievals = []
    for detection in detections:
        query_text = detection.get("kg_query") or detection.get("query_text") or detection.get("fault_name")
        retrieval_mode = detection.get("retrieval_mode", "kg")
        if retrieval_mode == "rag":
            retrieval = {
                "query": query_text,
                "results": query_vector_db(str(query_text), db_dir=rag_db_dir, top_k=5),
            }
        else:
            retrieval = retrieve_graph_context(
                str(query_text),
                graph_file=graph_file,
                hops=hops,
                max_seeds=3,
                movement=action,
            )
        retrievals.append(
            {
                "fault": detection.get("fault_name", detection.get("fault", "")),
                "fault_id": detection.get("fault_id", ""),
                "query_text": query_text,
                "retrieval_mode": retrieval_mode,
                "severity": detection["severity"],
                "evidence": detection["evidence"],
                "context": retrieval,
            }
        )

    return {
        "perception": perception_result,
        "retrievals": retrievals,
        "graph_retrievals": retrievals,
    }
