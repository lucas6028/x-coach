"""Unit tests for src.fit3d.bar_geometry -- synthetic only, no dataset or OpenCV required.

``bar_geometry`` imports cv2 lazily (it is absent from requirements-ci.txt, and CI skips the
opencv-dependent suites), so the module must import and most of it must work without it.
Tests that genuinely need cv2 are skipped when it is missing.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.fit3d import bar_geometry as bg
from src.fit3d import dataset as ds

try:  # pragma: no cover - availability differs between CI and dev machines
    import cv2 as _cv2
except Exception:  # pragma: no cover
    _cv2 = None


def make_camera(position, look_at, focal=1000.0, centre=(450.0, 450.0), distortion=True):
    """Synthetic Fit3D-format calibration. Fit3D stores T as the camera position in world."""
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(look_at, dtype=np.float64) - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rot = np.stack([right, down, forward])          # rows map world -> camera axes
    intr = {"f": np.array([[focal, focal]]), "c": np.array([[centre[0], centre[1]]])}
    params = {
        "extrinsics": {"R": rot, "T": position.reshape(1, 3)},
        "intrinsics_wo_distortion": {"f": np.array([focal, focal]),
                                     "c": np.array([centre[0], centre[1]])},
        "intrinsics_w_distortion": dict(intr, k=np.array([[0.0, 0.0, 0.0]]),
                                        p=np.array([[0.0, 0.0]])),
    }
    if distortion:
        params["intrinsics_w_distortion"]["k"] = np.array([[-0.19, 0.07, 0.0025]])
        params["intrinsics_w_distortion"]["p"] = np.array([[-0.0032, 0.0024]])
    return params


def make_skeleton(n_frames=20, seed=0):
    """A plausible upright skeleton with per-frame jitter, world frame, Z up."""
    rng = np.random.default_rng(seed)
    base = np.zeros((ds.NUM_JOINTS, 3))
    base[ds.ROOT] = [0.0, 0.0, 1.0]
    base[ds.R_HIP] = [-0.10, 0.0, 1.0]
    base[ds.L_HIP] = [0.10, 0.0, 1.0]
    base[ds.THORAX] = [0.0, 0.0, 1.45]
    base[ds.NECK] = [0.0, 0.0, 1.55]
    base[ds.HEAD] = [0.0, 0.0, 1.70]
    base[ds.R_SHOULDER] = [-0.16, 0.0, 1.42]
    base[ds.L_SHOULDER] = [0.16, 0.0, 1.42]
    base[ds.R_ELBOW] = [-0.20, 0.02, 1.15]
    base[ds.L_ELBOW] = [0.20, 0.02, 1.15]
    base[ds.R_WRIST] = [-0.22, 0.05, 0.90]
    base[ds.L_WRIST] = [0.22, 0.05, 0.90]
    base[ds.R_KNEE] = [-0.11, 0.0, 0.55]
    base[ds.L_KNEE] = [0.11, 0.0, 0.55]
    base[ds.R_ANKLE] = [-0.11, 0.0, 0.10]
    base[ds.L_ANKLE] = [0.11, 0.0, 0.10]
    base[ds.SPINE] = [0.0, 0.0, 1.22]
    for j in range(17, 25):
        base[j] = base[ds.R_WRIST] + rng.normal(0, 0.02, 3)
    return base[None, :, :] + rng.normal(0, 0.01, (n_frames, ds.NUM_JOINTS, 3))


class RansacLineTests(unittest.TestCase):
    def test_recovers_a_planted_line(self):
        rng = np.random.default_rng(1)
        t = np.linspace(0, 250, 300)
        line = np.stack([100 + t, 400 + 0.05 * t], axis=1) + rng.normal(0, 0.6, (300, 2))
        result = bg.ransac_line_2d(line)
        self.assertIsNotNone(result)
        _, direction, inliers, span = result
        self.assertGreater(inliers.sum(), 250)
        self.assertGreater(span, 200)
        expected = np.array([1.0, 0.05]) / np.linalg.norm([1.0, 0.05])
        self.assertGreater(abs(float(direction @ expected)), 0.999)

    def test_prefers_the_long_structure_over_a_denser_blob(self):
        """The bar is out-numbered by skin pixels in real frames but never out-spanned."""
        rng = np.random.default_rng(2)
        t = np.linspace(0, 240, 160)
        bar = np.stack([200 + t, 300 + np.zeros_like(t)], axis=1) + rng.normal(0, 0.5, (160, 2))
        blob = rng.normal([300, 500], 6.0, (400, 2))     # denser, compact
        result = bg.ransac_line_2d(np.vstack([bar, blob]))
        self.assertIsNotNone(result)
        mean, direction, _, span = result
        self.assertGreater(span, 200)
        self.assertLess(abs(mean[1] - 300), 15)          # sits on the bar, not the blob
        self.assertGreater(abs(float(direction @ np.array([1.0, 0.0]))), 0.99)

    def test_returns_none_when_too_few_points(self):
        self.assertIsNone(bg.ransac_line_2d(np.zeros((10, 2))))

    def test_is_deterministic_for_a_fixed_seed(self):
        rng = np.random.default_rng(3)
        pts = np.vstack([np.stack([np.linspace(0, 200, 120), np.full(120, 50.0)], 1),
                         rng.normal([80, 120], 5.0, (60, 2))])
        a = bg.ransac_line_2d(pts, seed=7)
        b = bg.ransac_line_2d(pts, seed=7)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])


class IntrinsicsTests(unittest.TestCase):
    def test_coefficient_order_is_k1_k2_p1_p2_k3(self):
        cam = make_camera([3, 0, 1.5], [0, 0, 1.0])
        K, dist = bg.opencv_intrinsics(cam)
        self.assertEqual(dist.shape, (5,))
        self.assertAlmostEqual(dist[0], -0.19)      # k1
        self.assertAlmostEqual(dist[1], 0.07)       # k2
        self.assertAlmostEqual(dist[2], -0.0032)    # p1
        self.assertAlmostEqual(dist[3], 0.0024)     # p2
        self.assertAlmostEqual(dist[4], 0.0025)     # k3
        self.assertAlmostEqual(K[0, 0], 1000.0)
        self.assertAlmostEqual(K[0, 2], 450.0)

    @unittest.skipIf(_cv2 is None, "opencv not installed")
    def test_undistort_inverts_the_fit3d_projection(self):
        """The empirical claim in opencv_intrinsics' docstring, re-checked."""
        cam = make_camera([3.5, 1.0, 1.6], [0, 0, 1.0])
        rng = np.random.default_rng(4)
        world = np.column_stack([rng.uniform(-0.6, 0.6, 200), rng.uniform(-0.6, 0.6, 200),
                                 rng.uniform(0.4, 1.9, 200)])
        distorted = ds.project_world_to_image(world, cam, True)
        undistorted = ds.project_world_to_image(world, cam, False)
        K, dist = bg.opencv_intrinsics(cam)
        recovered = _cv2.undistortPoints(distorted.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
        f = cam["intrinsics_wo_distortion"]["f"]
        c = cam["intrinsics_wo_distortion"]["c"]
        self.assertLess(float(np.abs(f * recovered + c - undistorted).max()), 0.5)


class TriangulateAxisTests(unittest.TestCase):
    """Planes are built analytically, so this runs without OpenCV."""

    @staticmethod
    def plane_through(camera_position, point_a, point_b):
        normal = np.cross(point_a - camera_position, point_b - camera_position)
        normal = normal / np.linalg.norm(normal)
        return normal, float(normal @ camera_position)

    def test_recovers_a_known_3d_line_from_three_views(self):
        point = np.array([0.05, -0.12, 1.40])
        direction = np.array([1.0, 0.08, 0.02]); direction /= np.linalg.norm(direction)
        a, b = point - 0.6 * direction, point + 0.6 * direction
        cams = [np.array(p, dtype=float) for p in ([4, 0, 1.5], [0, 4, 1.6], [-3, -3, 1.4])]
        planes = [self.plane_through(c, a, b) for c in cams]
        got_point, got_dir, residual = bg.triangulate_axis(
            np.stack([p[0] for p in planes]), np.array([p[1] for p in planes]))
        self.assertLess(residual, 1e-9)
        self.assertGreater(abs(float(got_dir @ direction)), 1.0 - 1e-9)
        perpendicular = (got_point - point) - ((got_point - point) @ direction) * direction
        self.assertLess(float(np.linalg.norm(perpendicular)), 1e-9)

    def test_two_views_suffice(self):
        point = np.array([0.0, 0.0, 1.2])
        direction = np.array([1.0, 0.0, 0.0])
        a, b = point - 0.5 * direction, point + 0.5 * direction
        planes = [self.plane_through(np.array(c, dtype=float), a, b)
                  for c in ([3.0, 0.5, 1.5], [0.5, 3.0, 1.5])]
        _, got_dir, residual = bg.triangulate_axis(
            np.stack([p[0] for p in planes]), np.array([p[1] for p in planes]))
        self.assertLess(residual, 1e-9)
        self.assertGreater(abs(float(got_dir @ direction)), 1.0 - 1e-9)

    def test_rejects_a_single_view(self):
        with self.assertRaises(ValueError):
            bg.triangulate_axis(np.array([[1.0, 0.0, 0.0]]), np.array([0.0]))

    def test_residual_exposes_inconsistent_views(self):
        """A view that saw a different structure must show up as a large residual."""
        point = np.array([0.0, 0.0, 1.2])
        direction = np.array([1.0, 0.0, 0.0])
        a, b = point - 0.5 * direction, point + 0.5 * direction
        planes = [self.plane_through(np.array(c, dtype=float), a, b)
                  for c in ([3.0, 0.5, 1.5], [0.5, 3.0, 1.5], [-3.0, 0.2, 1.5])]
        bad_a, bad_b = a + np.array([0, 0, 0.4]), b + np.array([0, 0, 0.4])
        planes.append(self.plane_through(np.array([0.0, -3.0, 1.5]), bad_a, bad_b))
        _, _, residual = bg.triangulate_axis(
            np.stack([p[0] for p in planes]), np.array([p[1] for p in planes]))
        self.assertGreater(residual, 0.01)


class BodyFrameTests(unittest.TestCase):
    def test_basis_is_orthonormal_and_right_handed(self):
        joints = make_skeleton()
        _, basis, torso = bg.body_frame(joints)
        for frame in basis:
            np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-9)
            self.assertGreater(float(np.linalg.det(frame)), 0.0)
        self.assertTrue(np.all(torso > 0.2))

    def test_torso_length_is_root_to_thorax(self):
        joints = make_skeleton(n_frames=1, seed=5)
        _, _, torso = bg.body_frame(joints)
        expected = np.linalg.norm(joints[0, ds.THORAX] - joints[0, ds.ROOT])
        self.assertAlmostEqual(float(torso[0]), float(expected), places=2)


class AxisOffsetTests(unittest.TestCase):
    @staticmethod
    def track_for(joints, point, direction):
        n = len(joints)
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        return bg.BarTrack("s", "a", np.arange(n),
                           np.repeat(np.asarray(point, float)[None], n, 0),
                           np.repeat(direction[None], n, 0),
                           np.zeros(n), np.full(n, 4))

    def test_offset_is_perpendicular_to_the_bar_axis(self):
        joints = make_skeleton()
        track = self.track_for(joints, [0.0, 0.30, 1.45], [1.0, 0.0, 0.0])
        offset, direction = bg.axis_offset_in_body_frame(track, joints)
        self.assertLess(float(np.abs(np.sum(offset * direction, axis=1)).max()), 1e-9)

    def test_is_invariant_to_a_global_rigid_transform(self):
        """Rotating the whole scene must not change a body-frame quantity."""
        joints = make_skeleton()
        point, direction = np.array([0.0, 0.30, 1.45]), np.array([1.0, 0.0, 0.0])
        base, _ = bg.axis_offset_in_body_frame(self.track_for(joints, point, direction), joints)

        angle = 0.7
        rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]])
        shift = np.array([2.5, -1.0, 0.0])
        moved_joints = joints @ rot.T + shift
        moved = self.track_for(moved_joints, rot @ point + shift, rot @ direction)
        rotated, _ = bg.axis_offset_in_body_frame(moved, moved_joints)
        np.testing.assert_allclose(base, rotated, atol=1e-9)

    def test_anterior_component_tracks_a_bar_moving_forward(self):
        joints = make_skeleton(n_frames=1, seed=6)
        near, _ = bg.axis_offset_in_body_frame(
            self.track_for(joints, [0.0, 0.10, 1.45], [1.0, 0.0, 0.0]), joints)
        far, _ = bg.axis_offset_in_body_frame(
            self.track_for(joints, [0.0, 0.35, 1.45], [1.0, 0.0, 0.0]), joints)
        self.assertAlmostEqual(abs(float(far[0, 2] - near[0, 2])), 0.25, places=2)

    def test_direction_sign_is_canonicalised(self):
        joints = make_skeleton()
        forward = self.track_for(joints, [0.0, 0.3, 1.45], [1.0, 0.0, 0.0])
        backward = self.track_for(joints, [0.0, 0.3, 1.45], [-1.0, 0.0, 0.0])
        _, d1 = bg.axis_offset_in_body_frame(forward, joints)
        _, d2 = bg.axis_offset_in_body_frame(backward, joints)
        np.testing.assert_allclose(d1, d2, atol=1e-9)


class FeatureTests(unittest.TestCase):
    def test_shapes_match_the_declared_keypoint_sets(self):
        joints = make_skeleton()
        frames = np.arange(len(joints))
        for name, subset in bg.BAR_KEYPOINT_SETS.items():
            feats = bg.keypoint_features(joints, frames, subset)
            self.assertEqual(feats.shape, (len(frames), 3 * len(subset)), name)

    def test_keypoint_sets_are_nested_and_ordered(self):
        sets = bg.BAR_KEYPOINT_SETS
        self.assertTrue(set(sets["wrists"]).issubset(sets["hands"]))
        self.assertTrue(set(sets["hands"]).issubset(sets["arms"]))
        self.assertTrue(set(sets["arms"]).issubset(sets["full25"]))

    def test_features_are_scale_invariant(self):
        joints = make_skeleton()
        frames = np.arange(len(joints))
        subset = bg.BAR_KEYPOINT_SETS["arms"]
        base = bg.keypoint_features(joints, frames, subset)
        scaled = bg.keypoint_features(joints * 1.7, frames, subset)
        np.testing.assert_allclose(base, scaled, atol=1e-9)

    def test_shoulder_width_is_the_shoulder_distance(self):
        joints = make_skeleton(n_frames=3, seed=8)
        widths = bg.shoulder_width(joints, np.arange(3))
        expected = np.linalg.norm(joints[:, ds.R_SHOULDER] - joints[:, ds.L_SHOULDER], axis=1)
        np.testing.assert_allclose(widths, expected)


class ConstantOffsetBaselineTests(unittest.TestCase):
    """The zero-parameter control that decides how the regression results may be worded."""

    def test_is_exact_when_the_offset_really_is_constant(self):
        groups = np.repeat(["a", "b", "c"], 10)
        reference = np.linspace(0.0, 1.0, 30)
        target = reference + 0.12
        self.assertAlmostEqual(bg.constant_offset_baseline(target, reference, groups), 0.0, places=12)

    def test_reports_the_within_subject_scatter(self):
        rng = np.random.default_rng(11)
        groups = np.repeat(["a", "b"], 200)
        reference = rng.normal(0, 1, 400)
        target = reference + 0.2 + rng.normal(0, 0.05, 400)
        self.assertLess(abs(bg.constant_offset_baseline(target, reference, groups) - 0.04), 0.01)

    def test_offset_is_fitted_without_the_held_out_subject(self):
        """A per-subject offset would leak; the held-out subject's bias must survive."""
        groups = np.repeat(["a", "b"], 50)
        reference = np.zeros(100)
        target = np.where(groups == "a", 0.0, 1.0)
        # offsets differ by 1.0 between subjects, so a leak-free baseline cannot score 0
        self.assertGreater(bg.constant_offset_baseline(target, reference, groups), 0.9)

    def test_mid_hand_uses_the_four_hand_points(self):
        joints = make_skeleton(n_frames=4, seed=12)
        frames = np.arange(4)
        got = bg.mid_hand_in_body_frame(joints, frames)
        origin, basis, _ = bg.body_frame(joints)
        hands = joints[:, [ds.R_HAND_A, ds.R_HAND_B, ds.L_HAND_A, ds.L_HAND_B]].mean(axis=1)
        expected = np.einsum("fij,fj->fi", basis, hands - origin)
        np.testing.assert_allclose(got, expected, atol=1e-12)


class BarTrackTests(unittest.TestCase):
    def test_len_reports_frame_count(self):
        track = bg.BarTrack("s03", "deadlift", np.arange(5), np.zeros((5, 3)),
                            np.tile([1.0, 0, 0], (5, 1)), np.zeros(5), np.full(5, 4))
        self.assertEqual(len(track), 5)


if __name__ == "__main__":
    unittest.main()
