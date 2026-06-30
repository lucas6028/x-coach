"""Biomechanical metric formulas on synthetic poses with known geometry."""

from __future__ import annotations

import unittest

import numpy as np

from src.fit3d import dataset as ds
from src.fit3d.biomech import IMAGE2D, WORLD3D, frame_metrics, joint_angle, lean_from_vertical


def _frame(joints: dict[int, tuple]) -> np.ndarray:
    """One synthetic frame (1, 25, 3); unspecified joints sit at the origin."""
    pts = np.zeros((1, ds.NUM_JOINTS, 3))
    for idx, xyz in joints.items():
        pts[0, idx] = xyz
    return pts


class JointAngleTests(unittest.TestCase):
    def test_straight_limb_is_180(self):
        pts = _frame({ds.R_HIP: (0, 0, 2), ds.R_KNEE: (0, 0, 1), ds.R_ANKLE: (0, 0, 0)})
        self.assertAlmostEqual(joint_angle(pts, ds.R_HIP, ds.R_KNEE, ds.R_ANKLE)[0], 180.0, places=4)

    def test_right_angle(self):
        pts = _frame({ds.R_HIP: (0, 0, 1), ds.R_KNEE: (0, 0, 0), ds.R_ANKLE: (1, 0, 0)})
        self.assertAlmostEqual(joint_angle(pts, ds.R_HIP, ds.R_KNEE, ds.R_ANKLE)[0], 90.0, places=4)


class LeanTests(unittest.TestCase):
    def test_vertical_torso_zero_lean_world(self):
        pts = _frame({ds.ROOT: (0, 0, 0), ds.THORAX: (0, 0, 1)})
        self.assertAlmostEqual(lean_from_vertical(pts, ds.ROOT, ds.THORAX, WORLD3D)[0], 0.0, places=4)

    def test_45_degree_lean_world(self):
        pts = _frame({ds.ROOT: (0, 0, 0), ds.THORAX: (1, 0, 1)})
        self.assertAlmostEqual(lean_from_vertical(pts, ds.ROOT, ds.THORAX, WORLD3D)[0], 45.0, places=4)

    def test_image_mode_uses_negative_y_as_up(self):
        # In image space, straight up the screen is -y; thorax above pelvis -> 0 lean.
        pts = np.zeros((1, ds.NUM_JOINTS, 2))
        pts[0, ds.ROOT] = (0, 10)
        pts[0, ds.THORAX] = (0, 5)  # smaller y == higher
        self.assertAlmostEqual(lean_from_vertical(pts, ds.ROOT, ds.THORAX, IMAGE2D)[0], 0.0, places=4)


class FrameMetricsTests(unittest.TestCase):
    def _deep_squat_frame(self) -> np.ndarray:
        # Hips dropped to knee height (parallel) with bent knees, both sides identical.
        return _frame({
            ds.R_HIP: (0.1, 0, 1.0), ds.R_KNEE: (0.15, 0, 1.0), ds.R_ANKLE: (0.1, 0, 0.5),
            ds.L_HIP: (-0.1, 0, 1.0), ds.L_KNEE: (-0.15, 0, 1.0), ds.L_ANKLE: (-0.1, 0, 0.5),
            ds.R_SHOULDER: (0.1, 0, 1.6), ds.L_SHOULDER: (-0.1, 0, 1.6),
            ds.ROOT: (0, 0, 1.0), ds.THORAX: (0, 0, 1.5),
        })

    def test_depth_ratio_zero_at_parallel(self):
        m = frame_metrics(self._deep_squat_frame(), WORLD3D)
        self.assertAlmostEqual(m["depth_ratio"][0], 0.0, places=4)  # hip height == knee height

    def test_depth_ratio_positive_when_standing(self):
        pts = _frame({
            ds.R_HIP: (0.1, 0, 1.0), ds.R_KNEE: (0.1, 0, 0.5), ds.R_ANKLE: (0.1, 0, 0.0),
            ds.L_HIP: (-0.1, 0, 1.0), ds.L_KNEE: (-0.1, 0, 0.5), ds.L_ANKLE: (-0.1, 0, 0.0),
        })
        self.assertGreater(frame_metrics(pts, WORLD3D)["depth_ratio"][0], 0.5)

    def test_all_metrics_present_and_finite(self):
        m = frame_metrics(self._deep_squat_frame(), WORLD3D)
        self.assertEqual(set(m), {"knee_angle", "hip_angle", "torso_lean_deg", "depth_ratio", "knee_width_ratio"})
        for name, series in m.items():
            self.assertTrue(np.isfinite(series[0]), f"{name} is not finite")


if __name__ == "__main__":
    unittest.main()
