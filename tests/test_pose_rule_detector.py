from __future__ import annotations

import unittest
from dataclasses import replace

from src.pose_rule_detector import (
    LANDMARK_COUNT,
    FrameMetrics,
    compute_frame_metrics,
    detect_rule_segments,
    raw_frame_metrics,
)


def landmark(x: float, y: float, visibility: float = 1.0) -> dict[str, float]:
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility}


def frame(
    *,
    left_knee_x: float = 0.45,
    right_knee_x: float = 0.55,
    hip_y: float = 0.72,
    knee_y: float = 0.68,
    ankle_y: float = 0.90,
    toe_offset: float = 0.12,
    frame_index: int = 0,
) -> dict[str, object]:
    landmarks = [landmark(0.0, 0.0, visibility=0.0) for _ in range(LANDMARK_COUNT)]
    landmarks[11] = landmark(0.35, 0.25)
    landmarks[12] = landmark(0.65, 0.25)
    landmarks[23] = landmark(0.35, hip_y)
    landmarks[24] = landmark(0.65, hip_y)
    landmarks[25] = landmark(left_knee_x, knee_y)
    landmarks[26] = landmark(right_knee_x, knee_y)
    landmarks[27] = landmark(0.30, ankle_y)
    landmarks[28] = landmark(0.70, ankle_y)
    landmarks[29] = landmark(0.30, ankle_y)
    landmarks[30] = landmark(0.70, ankle_y)
    landmarks[31] = landmark(0.30 + toe_offset, ankle_y)
    landmarks[32] = landmark(0.70 + toe_offset, ankle_y)
    return {"frame_index": frame_index, "landmarks": landmarks, "world_landmarks": landmarks}


class PoseRuleDetectorTests(unittest.TestCase):
    def test_knee_width_ratio_detects_inward_collapse(self) -> None:
        metrics = raw_frame_metrics(frame(left_knee_x=0.47, right_knee_x=0.53), fps=30.0)
        self.assertLess(metrics["knee_width_to_ankle_width"], 0.82)

    def test_knee_forward_projection_requires_side_observability(self) -> None:
        frames = [
            frame(left_knee_x=0.48, right_knee_x=0.88, toe_offset=0.12, frame_index=index)
            for index in range(12)
        ]
        metrics = compute_frame_metrics(frames, fps=30.0)
        side_detections = detect_rule_segments(metrics, fps=30.0, view_type="side", view_confidence=0.8)
        rear_detections = detect_rule_segments(metrics, fps=30.0, view_type="rear_oblique", view_confidence=0.8)

        self.assertTrue(any(item.fault_id == "knees_forward" and item.severity > 0 for item in side_detections))
        self.assertTrue(
            any(
                item.fault_id == "knees_forward" and item.observability == "low" and item.severity == 0
                for item in rear_detections
            )
        )

    def test_depth_rule_distinguishes_above_and_below_parallel(self) -> None:
        shallow = compute_frame_metrics([frame(hip_y=0.45, knee_y=0.70, frame_index=index) for index in range(12)], fps=30.0)
        deep = compute_frame_metrics([frame(hip_y=0.92, knee_y=0.70, frame_index=index) for index in range(12)], fps=30.0)

        shallow_detections = detect_rule_segments(shallow, fps=30.0, view_type="rear", view_confidence=0.8)
        deep_detections = detect_rule_segments(deep, fps=30.0, view_type="rear", view_confidence=0.8)

        self.assertTrue(any(item.fault_id == "shallow_depth" for item in shallow_detections))
        self.assertFalse(any(item.fault_id == "shallow_depth" for item in deep_detections))

    def test_persistence_filter_suppresses_single_frame_spike(self) -> None:
        base = FrameMetrics(
            frame_index=0,
            time=0.0,
            phase="bottom",
            valid=True,
            lower_body_visibility=1.0,
            avg_knee_angle=90.0,
            left_knee_angle=90.0,
            right_knee_angle=90.0,
            left_hip_angle=90.0,
            right_hip_angle=90.0,
            left_ankle_angle=90.0,
            right_ankle_angle=90.0,
            hip_minus_knee_y=0.10,
            knee_width_to_ankle_width=1.0,
            knee_forward_ratio=0.0,
            torso_lean_deg=10.0,
            heel_height_delta=0.0,
        )
        metrics = [
            replace(
                base,
                frame_index=index,
                time=index / 30.0,
                knee_width_to_ankle_width=0.50 if index == 5 else 1.0,
            )
            for index in range(12)
        ]
        detections = detect_rule_segments(metrics, fps=30.0, view_type="rear", view_confidence=0.8)
        self.assertFalse(any(item.fault_id == "knees_inward" for item in detections))


if __name__ == "__main__":
    unittest.main()
