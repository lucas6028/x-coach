"""Unit tests for the perception->graph geometry helpers (src/knowledge/perception_to_graph.py).

Covers the pure angle/ratio/midpoint utilities, per-frame metric extraction, and the
self-contained legacy fault detector run over a synthetic pose JSON.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.knowledge.perception_to_graph import (
    LEFT_HIP,
    _angle,
    _frame_metrics,
    _get_point,
    _line_angle_from_vertical,
    _midpoint,
    _safe_ratio,
    detect_faults_from_pose_json_legacy,
)

# MediaPipe-style landmark indices used by the detector.
_INDICES = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


def _landmarks(overrides: dict[int, tuple[float, float, float]]) -> list[dict]:
    """Build 33 landmark dicts; overrides maps index -> (x, y, visibility)."""
    points = [{"x": 0.5, "y": 0.5, "visibility": 1.0} for _ in range(33)]
    for index, (x, y, visibility) in overrides.items():
        points[index] = {"x": x, "y": y, "visibility": visibility}
    return points


def _standing_overrides() -> dict[int, tuple[float, float, float]]:
    """A near-vertical standing skeleton: legs straight (knee angle ~180 -> shallow)."""
    return {
        _INDICES["left_shoulder"]: (0.45, 0.20, 1.0),
        _INDICES["right_shoulder"]: (0.55, 0.20, 1.0),
        _INDICES["left_hip"]: (0.45, 0.50, 1.0),
        _INDICES["right_hip"]: (0.55, 0.50, 1.0),
        _INDICES["left_knee"]: (0.45, 0.75, 1.0),
        _INDICES["right_knee"]: (0.55, 0.75, 1.0),
        _INDICES["left_ankle"]: (0.45, 0.95, 1.0),
        _INDICES["right_ankle"]: (0.55, 0.95, 1.0),
        _INDICES["left_heel"]: (0.44, 0.97, 1.0),
        _INDICES["right_heel"]: (0.56, 0.97, 1.0),
        _INDICES["left_foot_index"]: (0.47, 0.97, 1.0),
        _INDICES["right_foot_index"]: (0.53, 0.97, 1.0),
    }


class GetPointTests(unittest.TestCase):
    def test_returns_xy_for_visible_landmark(self):
        landmarks = _landmarks({5: (0.3, 0.7, 0.9)})
        self.assertEqual(_get_point(landmarks, 5), (0.3, 0.7))

    def test_returns_none_below_visibility_threshold(self):
        landmarks = _landmarks({5: (0.3, 0.7, 0.1)})
        self.assertIsNone(_get_point(landmarks, 5))

    def test_returns_none_for_out_of_range_index(self):
        self.assertIsNone(_get_point(_landmarks({}), 999))


class AngleTests(unittest.TestCase):
    def test_straight_line_is_180_degrees(self):
        self.assertAlmostEqual(_angle((0, 1), (0, 0), (0, -1)), 180.0, places=6)

    def test_right_angle_is_90_degrees(self):
        self.assertAlmostEqual(_angle((1, 0), (0, 0), (0, 1)), 90.0, places=6)

    def test_degenerate_zero_length_returns_180(self):
        self.assertEqual(_angle((0, 0), (0, 0), (1, 1)), 180.0)


class LineAngleFromVerticalTests(unittest.TestCase):
    def test_vertical_line_is_zero(self):
        self.assertAlmostEqual(_line_angle_from_vertical((0.5, 0.1), (0.5, 0.9)), 0.0, places=4)

    def test_horizontal_line_is_ninety(self):
        self.assertAlmostEqual(_line_angle_from_vertical((0.9, 0.5), (0.1, 0.5)), 90.0, places=4)

    def test_diagonal_is_forty_five(self):
        self.assertAlmostEqual(_line_angle_from_vertical((0.0, 0.0), (1.0, 1.0)), 45.0, places=4)


class MidpointAndRatioTests(unittest.TestCase):
    def test_midpoint_averages_coordinates(self):
        self.assertEqual(_midpoint((0.0, 0.0), (1.0, 2.0)), (0.5, 1.0))

    def test_safe_ratio_divides(self):
        self.assertAlmostEqual(_safe_ratio(1.0, 4.0), 0.25)

    def test_safe_ratio_guards_against_zero(self):
        self.assertIsNone(_safe_ratio(1.0, 0.0))


class FrameMetricsTests(unittest.TestCase):
    def test_returns_none_without_landmarks(self):
        self.assertIsNone(_frame_metrics({"frame_index": 0}))

    def test_returns_none_when_required_point_occluded(self):
        overrides = _standing_overrides()
        overrides[LEFT_HIP] = (0.45, 0.50, 0.0)  # hidden hip -> required point missing
        self.assertIsNone(_frame_metrics({"frame_index": 0, "landmarks": _landmarks(overrides)}))

    def test_standing_pose_reports_extended_knee_angle(self):
        metrics = _frame_metrics({"frame_index": 3, "landmarks": _landmarks(_standing_overrides())})
        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(metrics["avg_knee_angle"], 180.0, places=3)
        self.assertEqual(metrics["frame_index"], 3.0)
        self.assertAlmostEqual(metrics["torso_lean_deg"], 0.0, places=3)


class DetectFaultsLegacyTests(unittest.TestCase):
    def _write_pose_json(self, frames: list[dict]) -> Path:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        path = tmp / "pose.json"
        path.write_text(json.dumps({"frames": frames}), encoding="utf-8")
        return path

    def test_no_valid_frames_returns_empty_detections(self):
        path = self._write_pose_json([{"frame_index": 0}])  # no landmarks
        result = detect_faults_from_pose_json_legacy(path)
        self.assertEqual(result["detections"], [])
        self.assertEqual(result["summary"]["valid_frames"], 0)

    def test_extended_legs_flagged_as_shallow_depth(self):
        frames = [
            {"frame_index": i, "landmarks": _landmarks(_standing_overrides())}
            for i in range(5)
        ]
        path = self._write_pose_json(frames)
        result = detect_faults_from_pose_json_legacy(path)
        faults = {detection["fault"] for detection in result["detections"]}
        self.assertIn("Shallow Depth", faults)
        self.assertEqual(result["summary"]["valid_frames"], 5)
        for detection in result["detections"]:
            self.assertTrue(0.0 <= detection["severity"] <= 1.0)
            self.assertFalse(math.isnan(detection["severity"]))


if __name__ == "__main__":
    unittest.main()
