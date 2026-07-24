"""Cue features must read the same in 3D regardless of viewing angle -- and must not in 2D."""

import unittest

import numpy as np

from src.fit3d.biomech import IMAGE2D
from src.fitness_aqa import cue_features as cf


def squat_skeleton(knee_deg: float = 90.0) -> np.ndarray:
    """A synthetic H36M-17 squat in camera coords (x right, y down, z away), mm.

    Built in a "world" frame (z up, subject facing +x) and then rotated into the camera
    convention, so the intended knee angle is exact by construction.
    """
    femur, tibia = 400.0, 400.0
    theta = np.radians(180.0 - knee_deg)  # thigh direction relative to the shank
    world = np.zeros((17, 3))
    for sign, hip, knee, ankle, shoulder in ((1, 1, 2, 3, 11), (-1, 4, 5, 6, 14)):
        y = sign * 100.0
        world[ankle] = [0.0, y, 0.0]
        world[knee] = [0.0, y, tibia]
        world[hip] = [-femur * np.sin(theta), y, tibia + femur * np.cos(theta)]
        world[shoulder] = world[hip] + [0.0, 0.0, 500.0]
    world[0] = 0.5 * (world[1] + world[4])          # pelvis
    world[8] = 0.5 * (world[11] + world[14])        # thorax
    world[7] = 0.5 * (world[0] + world[8])          # spine
    world[9] = world[8] + [0.0, 0.0, 100.0]         # neck
    world[10] = world[8] + [0.0, 0.0, 200.0]        # head
    # world (x fwd, y left, z up) -> camera (x right, y down, z away)
    return np.stack([world[:, 1], -world[:, 2], world[:, 0]], axis=-1)


def rotate_about_vertical(cam_points: np.ndarray, deg: float) -> np.ndarray:
    """Spin the subject around the camera's vertical axis (y), i.e. move the camera."""
    a = np.radians(deg)
    x, y, z = cam_points[..., 0], cam_points[..., 1], cam_points[..., 2]
    return np.stack([x * np.cos(a) + z * np.sin(a), y, -x * np.sin(a) + z * np.cos(a)], axis=-1)


class TestCueFeatures(unittest.TestCase):
    def test_feature_vector_shape_and_names(self):
        pts = squat_skeleton()[None]
        feats = cf.compute_features(pts, cf.CAM3D)
        self.assertEqual(feats.shape, (1, len(cf.FEATURE_NAMES)))
        self.assertEqual(feats.shape[1], cf.compute_features(pts[..., :2], IMAGE2D).shape[1])

    def test_known_knee_angle_recovered_in_3d(self):
        for target in (60.0, 90.0, 120.0):
            feats = cf.compute_features(squat_skeleton(target)[None], cf.CAM3D)
            k = cf.FEATURE_NAMES.index("knee_angle_r")
            self.assertAlmostEqual(feats[0, k], target, places=4)

    def test_3d_knee_angle_is_view_invariant_but_2d_is_not(self):
        base = squat_skeleton(70.0)
        k = cf.FEATURE_NAMES.index("knee_angle_r")
        readings_3d, readings_2d = [], []
        for deg in (0.0, 30.0, 60.0, 80.0):
            rot = rotate_about_vertical(base, deg)[None]
            readings_3d.append(cf.compute_features(rot, cf.CAM3D)[0, k])
            readings_2d.append(cf.compute_features(rot[..., :2], IMAGE2D)[0, k])
        self.assertLess(float(np.ptp(readings_3d)), 1e-6)
        # Same skeleton, same formula: only the missing depth channel moves the 2D reading.
        self.assertGreater(float(np.ptp(readings_2d)), 20.0)

    def test_2d_knee_angle_degenerates_when_the_femur_points_at_the_camera(self):
        # A 90-degree knee puts the femur horizontal; viewed head-on it projects to a
        # single point and the angle is not merely wrong but unmeasurable -- the extreme
        # of the projection loss the 2D arm carries everywhere else.
        pts = squat_skeleton(90.0)[None]
        k = cf.FEATURE_NAMES.index("knee_angle_r")
        self.assertTrue(np.isnan(cf.compute_features(pts[..., :2], IMAGE2D)[0, k]))
        self.assertFalse(np.isnan(cf.compute_features(pts, cf.CAM3D)[0, k]))

    def test_depth_ratio_sign_tracks_squat_depth(self):
        shallow = cf.compute_features(squat_skeleton(150.0)[None], cf.CAM3D)
        deep = cf.compute_features(squat_skeleton(40.0)[None], cf.CAM3D)
        d = cf.FEATURE_NAMES.index("depth_ratio")
        self.assertGreater(shallow[0, d], deep[0, d])

    def test_scale_invariance(self):
        pts = squat_skeleton()[None]
        a = cf.compute_features(pts, cf.CAM3D)
        b = cf.compute_features(pts * 7.5, cf.CAM3D)
        np.testing.assert_allclose(a, b, atol=1e-8)

    def test_nan_joint_propagates_only_to_affected_cues(self):
        pts = squat_skeleton()[None].copy()
        pts[0, 3] = np.nan  # right ankle
        feats = cf.compute_features(pts, cf.CAM3D)
        self.assertTrue(np.isnan(feats[0, cf.FEATURE_NAMES.index("knee_angle_r")]))
        self.assertFalse(np.isnan(feats[0, cf.FEATURE_NAMES.index("knee_angle_l")]))

    def test_mode_validation(self):
        pts = squat_skeleton()[None]
        with self.assertRaises(ValueError):
            cf.compute_features(pts, IMAGE2D)          # 3 channels into the 2D mode
        with self.assertRaises(ValueError):
            cf.compute_features(pts[..., :2], cf.CAM3D)
        with self.assertRaises(ValueError):
            cf.compute_features(pts, "nonsense")

    def test_cam3d_to_biomech_space_puts_up_on_z(self):
        cam = np.array([[[1.0, -2.0, 3.0]]])  # y is down, so -2 is *above* the origin
        out, mode = cf.to_biomech_space(cam, cf.CAM3D)
        np.testing.assert_allclose(out[0, 0], [1.0, 3.0, 2.0])
        self.assertEqual(mode, "world3d")


if __name__ == "__main__":
    unittest.main()
