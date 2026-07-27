"""Self-tests for the axial-rotation harness (synthetic; no dataset or SMPLX model needed)."""

from __future__ import annotations

import unittest

import numpy as np

from src.fit3d import dataset as ds
from src.fit3d.axial_rotation import (
    FoldScore, canonicalize, canonicalize_gt, compare_twist, loso_scores,
    rbf_krr_predictor, ridge_predictor, rotation_matrices_from_rotvec, swing_twist,
)

AXIS = np.array([0.145, -0.989, -0.024])


def _rot(rotvec) -> np.ndarray:
    """Single rotation matrix from an axis-angle vector (numpy only, no scipy)."""
    return rotation_matrices_from_rotvec(np.asarray(rotvec, dtype=np.float64))


def _perpendicular_to(axis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    v -= (v @ axis) * axis
    return v / np.linalg.norm(v)


def _skeleton(frames: int = 24, seed: int = 0) -> np.ndarray:
    """A plausible (F, 25, 3) skeleton: hips apart on x, thorax above the root on y."""
    rng = np.random.default_rng(seed)
    base = rng.normal(scale=0.25, size=(ds.NUM_JOINTS, 3))
    base[ds.ROOT] = (0.0, 0.0, 0.0)
    base[ds.L_HIP] = (0.12, -0.02, 0.0)
    base[ds.R_HIP] = (-0.12, -0.02, 0.0)
    base[ds.THORAX] = (0.0, 0.50, 0.0)
    return base[None] + rng.normal(scale=0.01, size=(frames, ds.NUM_JOINTS, 3))


class SwingTwistTests(unittest.TestCase):
    def setUp(self):
        self.axis = AXIS / np.linalg.norm(AXIS)

    def test_recovers_known_swing_and_twist(self):
        rng = np.random.default_rng(0)
        for _ in range(8):
            twist_true = np.radians(rng.uniform(-40, 40))
            swing_true = np.radians(rng.uniform(0, 50))
            swing_axis = _perpendicular_to(self.axis, rng)
            composed = _rot(swing_true * swing_axis) @ _rot(twist_true * self.axis)
            twist, swing = swing_twist(composed[None], self.axis)
            self.assertAlmostEqual(float(twist[0]), twist_true, places=10)
            self.assertAlmostEqual(float(swing[0]), swing_true, places=10)

    def test_pure_twist_has_no_swing(self):
        mat = _rot(np.radians(25.0) * self.axis)
        twist, swing = swing_twist(mat[None], self.axis)
        self.assertAlmostEqual(float(np.degrees(twist[0])), 25.0, places=10)
        self.assertAlmostEqual(float(swing[0]), 0.0, places=10)

    def test_pure_swing_has_no_twist(self):
        perp = _perpendicular_to(self.axis, np.random.default_rng(3))
        mat = _rot(np.radians(30.0) * perp)
        twist, swing = swing_twist(mat[None], self.axis)
        self.assertAlmostEqual(float(twist[0]), 0.0, places=10)
        self.assertAlmostEqual(float(np.degrees(swing[0])), 30.0, places=10)

    def test_identity_is_zero(self):
        twist, swing = swing_twist(np.eye(3)[None], self.axis)
        self.assertAlmostEqual(float(twist[0]), 0.0, places=12)
        self.assertAlmostEqual(float(swing[0]), 0.0, places=12)

    def test_twist_sign_flips_with_axis(self):
        mat = _rot(np.radians(18.0) * self.axis)
        forward, _ = swing_twist(mat[None], self.axis)
        backward, _ = swing_twist(mat[None], -self.axis)
        self.assertAlmostEqual(float(forward[0]), -float(backward[0]), places=10)

    def test_axis_is_normalised_internally(self):
        mat = _rot(np.radians(12.0) * self.axis)
        a, _ = swing_twist(mat[None], self.axis)
        b, _ = swing_twist(mat[None], self.axis * 7.5)
        self.assertAlmostEqual(float(a[0]), float(b[0]), places=12)

    def test_leading_shape_is_preserved(self):
        mats = np.tile(np.eye(3), (4, 6, 1, 1))
        twist, swing = swing_twist(mats, self.axis)
        self.assertEqual(twist.shape, (4, 6))
        self.assertEqual(swing.shape, (4, 6))

    def test_non_finite_rotations_yield_nan_not_an_exception(self):
        mats = np.tile(np.eye(3), (3, 1, 1))
        mats[1] = np.nan
        twist, swing = swing_twist(mats, self.axis)
        self.assertTrue(np.isnan(twist[1]) and np.isnan(swing[1]))
        self.assertFalse(np.isnan(twist[0]) or np.isnan(twist[2]))

    def test_degenerate_axis_rejected(self):
        with self.assertRaises(ValueError):
            swing_twist(np.eye(3)[None], np.zeros(3))

    def test_bad_shape_rejected(self):
        with self.assertRaises(ValueError):
            swing_twist(np.zeros((5, 3)), self.axis)


class CanonicalizeTests(unittest.TestCase):
    def test_invariant_to_global_rotation_and_translation(self):
        joints = _skeleton()
        rot = _rot([0.3, -1.1, 0.45])
        moved = joints @ rot.T + np.array([2.0, -3.0, 7.0])
        np.testing.assert_allclose(canonicalize(joints), canonicalize(moved), atol=1e-9)

    def test_invariant_to_uniform_scale(self):
        joints = _skeleton(seed=1)
        np.testing.assert_allclose(canonicalize(joints), canonicalize(joints * 3.7), atol=1e-9)

    def test_root_is_at_the_origin(self):
        out = canonicalize(_skeleton(seed=2))
        np.testing.assert_allclose(out[:, ds.ROOT], 0.0, atol=1e-12)

    def test_hip_axis_lands_on_the_lateral_basis_vector(self):
        out = canonicalize(_skeleton(seed=4))
        lateral = out[:, ds.L_HIP] - out[:, ds.R_HIP]
        lateral /= np.linalg.norm(lateral, axis=1, keepdims=True)
        # by construction the lateral basis vector is axis 0, so y/z components vanish
        np.testing.assert_allclose(lateral[:, 1:], 0.0, atol=1e-9)

    def test_gt_frame_undoes_a_known_pelvis_rotation(self):
        joints = _skeleton(seed=5)
        rot = _rot([0.0, 0.8, 0.0])
        rotated = joints @ rot.T
        pelvis = np.tile(rot, (len(joints), 1, 1))
        identity = np.tile(np.eye(3), (len(joints), 1, 1))
        np.testing.assert_allclose(
            canonicalize_gt(rotated, pelvis), canonicalize_gt(joints, identity), atol=1e-9
        )


class RegressionHarnessTests(unittest.TestCase):
    def test_ridge_recovers_a_linear_target(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(400, 5))
        y = x @ np.array([2.0, -1.0, 0.5, 0.0, 3.0]) + 1.5
        pred = ridge_predictor(x, y, lam=1e-6)(x)
        self.assertLess(float(np.mean(np.abs(pred - y))), 1e-3)

    def test_ridge_shrinks_towards_the_mean_as_lambda_grows(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(200, 4))
        y = x @ np.array([1.0, 2.0, -1.0, 0.5])
        loose = ridge_predictor(x, y, lam=1e-6)(x)
        tight = ridge_predictor(x, y, lam=1e6)(x)
        self.assertLess(float(np.std(tight)), float(np.std(loose)))
        self.assertAlmostEqual(float(np.mean(tight)), float(np.mean(y)), places=3)

    def test_krr_fits_a_nonlinear_target_that_ridge_cannot(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(-2, 2, size=(300, 2))
        y = np.sin(3.0 * x[:, 0]) * np.cos(2.0 * x[:, 1])
        krr_err = float(np.mean(np.abs(rbf_krr_predictor(x, y, lam=1e-6)(x) - y)))
        ridge_err = float(np.mean(np.abs(ridge_predictor(x, y, lam=1e-6)(x) - y)))
        self.assertLess(krr_err, ridge_err / 2.0)

    def test_loso_holds_each_subject_out_once(self):
        rng = np.random.default_rng(3)
        groups = np.repeat(["a", "b", "c"], 40)
        x = rng.normal(size=(120, 3))
        y = x @ np.array([1.0, 0.0, -2.0])
        scores = loso_scores(x, y, groups, lambda a, b: ridge_predictor(a, b, 1e-6))
        self.assertEqual([s.subject for s in scores], ["a", "b", "c"])
        self.assertTrue(all(isinstance(s, FoldScore) for s in scores))
        self.assertTrue(all(s.r2_within > 0.99 for s in scores))

    def test_predicting_the_pooled_mean_gives_zero_global_and_negative_within(self):
        groups = np.repeat(["a", "b"], 50)
        y = np.concatenate([np.full(50, 5.0), np.full(50, -5.0)])
        y = y + np.random.default_rng(4).normal(scale=0.1, size=100)
        x = np.zeros((100, 2))  # carries no information, so ridge can only fit an intercept
        scores = loso_scores(x, y, groups, lambda a, b: ridge_predictor(a, b, 1.0))
        # each fold predicts the other subject's mean -> far worse than the fold's own mean
        self.assertTrue(all(s.r2_within < -1.0 for s in scores))


class CompareTwistTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.gt = rng.normal(scale=8.0, size=200)
        self.groups = np.repeat(["s1", "s2", "s3", "s4"], 50)

    def test_perfect_estimate_scores_zero(self):
        out = compare_twist(self.gt, self.gt.copy(), self.groups)
        self.assertEqual(out.n, 200)
        self.assertAlmostEqual(out.mae_raw, 0.0, places=10)
        self.assertAlmostEqual(out.mae_debiased, 0.0, places=10)
        self.assertAlmostEqual(out.bias, 0.0, places=10)
        self.assertAlmostEqual(out.pearson, 1.0, places=10)

    def test_constant_offset_is_removed_by_debiasing(self):
        out = compare_twist(self.gt, self.gt + 7.0, self.groups)
        self.assertAlmostEqual(out.mae_raw, 7.0, places=8)
        self.assertAlmostEqual(out.bias, 7.0, places=8)
        self.assertAlmostEqual(out.mae_debiased, 0.0, places=8)
        self.assertAlmostEqual(out.pearson, 1.0, places=8)

    def test_per_subject_offsets_favour_the_oracle(self):
        est = self.gt + np.repeat([10.0, -10.0, 4.0, -4.0], 50)
        out = compare_twist(self.gt, est, self.groups)
        self.assertAlmostEqual(out.mae_oracle, 0.0, places=8)
        self.assertGreater(out.mae_debiased, 1.0)  # a single offset cannot fix all four
        self.assertLess(out.mae_oracle, out.mae_debiased)

    def test_debias_offset_comes_from_other_subjects_only(self):
        """One subject's own error must not shrink its own debiased error."""
        est = self.gt.copy()
        est[:50] += 30.0  # only s1 is offset; the other three are perfect
        out = compare_twist(self.gt, est, self.groups)
        # s1's LOSO offset is estimated from s2-s4 (~0), so its 30 deg error survives
        self.assertGreater(out.mae_debiased, 7.0)
        self.assertAlmostEqual(out.mae_oracle, 0.0, places=8)

    def test_sign_flip_shows_up_as_negative_pearson(self):
        out = compare_twist(self.gt, -self.gt, self.groups)
        self.assertAlmostEqual(out.pearson, -1.0, places=8)

    def test_non_finite_frames_are_dropped(self):
        est = self.gt.copy()
        est[::2] = np.nan  # the kernel writes NaN off the subsample grid
        out = compare_twist(self.gt, est, self.groups)
        self.assertEqual(out.n, 100)
        self.assertAlmostEqual(out.mae_raw, 0.0, places=10)

    def test_single_subject_flags_the_degenerate_loso(self):
        """With one subject the LOSO offset has no other subjects, so it becomes the oracle."""
        one = np.full(200, "solo")
        out = compare_twist(self.gt, self.gt + 7.0, one)
        self.assertEqual(out.n_groups, 1)
        self.assertTrue(out.loso_is_degenerate)
        self.assertAlmostEqual(out.mae_debiased, out.mae_oracle, places=10)

    def test_multiple_subjects_are_not_flagged(self):
        out = compare_twist(self.gt, self.gt + 7.0, self.groups)
        self.assertEqual(out.n_groups, 4)
        self.assertFalse(out.loso_is_degenerate)

    def test_group_count_ignores_dropped_frames(self):
        """A subject whose frames are all NaN must not be counted."""
        est = self.gt.copy()
        est[:50] = np.nan  # s1 entirely missing
        out = compare_twist(self.gt, est, self.groups)
        self.assertEqual(out.n_groups, 3)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            compare_twist(self.gt, self.gt[:100], self.groups)

    def test_too_few_paired_frames_rejected(self):
        est = np.full_like(self.gt, np.nan)
        est[0] = self.gt[0]
        with self.assertRaises(ValueError):
            compare_twist(self.gt, est, self.groups)


class RestAxisOverrideTests(unittest.TestCase):
    def test_rest_override_changes_the_twist_axis(self):
        """A different rest skeleton must yield a different twist for the same rotation.

        This is why the kernel exports SMPL's rest joints: HMR2.0 predicts SMPL, the GT is
        SMPLX, and reading both about one shared axis would inject a spurious offset.
        """
        rest_a = np.zeros((25, 3))
        rest_a[1] = (0.06, -0.44, 0.0)   # L_Hip
        rest_a[4] = (0.12, -0.82, 0.0)   # L_Knee
        rest_b = rest_a.copy()
        rest_b[4] = (0.12, -0.82, 0.25)  # femur tilted out of the frontal plane

        rot = _rot(np.radians(20.0) * np.array([0.0, 1.0, 0.0]))
        body_pose = np.tile(np.eye(3), (4, 21, 1, 1))
        body_pose[:, 0] = rot  # body_pose[0] == joint 1 == L_Hip

        from src.fit3d.axial_rotation import hip_twist_series
        a = hip_twist_series(body_pose, "L", rest=rest_a)
        b = hip_twist_series(body_pose, "L", rest=rest_b)
        self.assertFalse(np.allclose(a, b))
        self.assertTrue(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))

    def test_rejects_a_bad_side(self):
        from src.fit3d.axial_rotation import hip_twist_series
        with self.assertRaises(ValueError):
            hip_twist_series(np.tile(np.eye(3), (2, 21, 1, 1)), "left")


try:  # scipy is OPTIONAL in this repo (absent from both requirements files)
    from scipy.spatial.transform import Rotation as _ScipyRotation
except ImportError:  # pragma: no cover - exercised only on the CI image
    _ScipyRotation = None


@unittest.skipIf(_ScipyRotation is None, "scipy not installed (optional dependency)")
class ScipyCrossCheckTests(unittest.TestCase):
    """Validate the numpy quaternion path against scipy's independent implementation.

    The module deliberately avoids scipy so the CI image does not need it, but where scipy
    IS available these confirm the hand-rolled Shepperd conversion and Rodrigues formula
    agree with a battle-tested implementation -- including the branch-selection edge cases
    (near-180-degree rotations) that a naive conversion gets wrong.
    """

    def test_rodrigues_matches_scipy(self):
        rng = np.random.default_rng(7)
        vecs = rng.normal(scale=1.5, size=(64, 3))
        np.testing.assert_allclose(
            rotation_matrices_from_rotvec(vecs),
            _ScipyRotation.from_rotvec(vecs).as_matrix(), atol=1e-12,
        )

    def test_rodrigues_handles_the_zero_rotation(self):
        np.testing.assert_allclose(rotation_matrices_from_rotvec(np.zeros((3, 3))),
                                   np.broadcast_to(np.eye(3), (3, 3, 3)), atol=1e-15)

    def test_swing_twist_matches_a_scipy_reference(self):
        """Same decomposition computed through scipy, over rotations from every quaternion branch."""
        rng = np.random.default_rng(11)
        axis = AXIS / np.linalg.norm(AXIS)
        vecs = rng.normal(size=(256, 3))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        # angles spanning all four Shepperd branches, including close to pi
        angles = np.concatenate([rng.uniform(0.01, np.pi - 0.01, 224), np.full(32, np.pi - 1e-4)])
        mats = rotation_matrices_from_rotvec(vecs * angles[:, None])

        twist, swing = swing_twist(mats, axis)

        q = _ScipyRotation.from_matrix(mats).as_quat()
        q[q[:, 3] < 0] *= -1.0
        ref_twist = 2.0 * np.arctan2(q[:, :3] @ axis, q[:, 3])
        q_tw = np.concatenate([(q[:, :3] @ axis)[:, None] * axis[None, :], q[:, 3:4]], axis=1)
        q_tw /= np.linalg.norm(q_tw, axis=1, keepdims=True)
        ref_swing = (_ScipyRotation.from_quat(q) * _ScipyRotation.from_quat(q_tw).inv()).magnitude()

        np.testing.assert_allclose(twist, ref_twist, atol=1e-9)
        np.testing.assert_allclose(swing, ref_swing, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
