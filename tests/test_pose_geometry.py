import unittest
import numpy as np
from src.pose.geometry import angle_degrees, severity_from_range, contiguous_true_segments


class PoseGeometryTests(unittest.TestCase):
    def test_angle_degrees_right_angle(self) -> None:
        pts = np.zeros((33, 4), dtype=np.float32)
        pts[:, 3] = 1.0
        pts[0, :3] = [0, 1, 0]
        pts[1, :3] = [0, 0, 0]
        pts[2, :3] = [1, 0, 0]
        self.assertAlmostEqual(angle_degrees(pts, 0, 1, 2), 90.0, places=3)

    def test_severity_ramp_monotonic(self) -> None:
        self.assertEqual(severity_from_range(0.82, 0.82, 0.70, lower_is_worse=True), 0.0)
        self.assertEqual(severity_from_range(0.70, 0.82, 0.70, lower_is_worse=True), 1.0)

    def test_contiguous_segments_respects_min_frames(self) -> None:
        self.assertEqual(contiguous_true_segments([True, True, False, True], 2), [(0, 1)])
