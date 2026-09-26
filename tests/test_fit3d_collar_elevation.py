"""Self-tests for the collar-elevation harness (synthetic; no dataset or SMPL-X model needed)."""

from __future__ import annotations

import unittest

import numpy as np

from src.fit3d import collar_elevation as ce
from src.fit3d.axial_rotation import rotation_matrices_from_rotvec

MIRROR = np.diag([-1.0, 1.0, 1.0])


def _rest() -> np.ndarray:
    """A symmetric SMPL-X-like rest skeleton: collars near the midline, arms out along +/-x."""
    rest = np.zeros((55, 3))
    rest[ce.COLLAR["L"]] = (0.05, 0.03, 0.0)
    rest[ce.SHOULDER["L"]] = (0.16, 0.08, -0.02)
    rest[ce.ELBOW["L"]] = (0.42, 0.03, -0.05)
    for j in ("COLLAR", "SHOULDER", "ELBOW"):
        rest[getattr(ce, j)["R"]] = MIRROR @ rest[getattr(ce, j)["L"]]
    return rest


def _pose(frames: int = 1) -> np.ndarray:
    return np.tile(np.eye(3), (frames, 21, 1, 1))


def _set(pose: np.ndarray, joint: int, rotvec) -> None:
    pose[:, joint - ce.BODY_POSE_OFFSET] = rotation_matrices_from_rotvec(np.asarray(rotvec, float))


class AngleTests(unittest.TestCase):
    def test_identity_pose_reads_zero_collar_angles_and_a_t_pose_arm(self):
        for side in ("L", "R"):
            elev, prot = ce.collar_angles(_pose(), _rest(), side)
            self.assertAlmostEqual(float(elev[0]), 0.0, places=9)
            self.assertAlmostEqual(float(prot[0]), 0.0, places=9)
            arm_elev, horiz = ce.arm_angles(_pose(), _rest(), side)
            arm = _rest()[ce.ELBOW[side]] - _rest()[ce.SHOULDER[side]]
            expected = np.degrees(np.arccos(-arm[1] / np.linalg.norm(arm)))
            self.assertAlmostEqual(float(arm_elev[0]), expected, places=9)
            self.assertGreater(float(horiz[0]), 80.0)

    def test_collar_raise_about_the_forward_axis_reads_as_positive_elevation_on_both_sides(self):
        # Raising the shoulder = rotating the collar bone about +z (left) / -z (right).
        for side, sign in (("L", 1.0), ("R", -1.0)):
            pose = _pose()
            _set(pose, ce.COLLAR[side], (0.0, 0.0, sign * np.radians(20.0)))
            elev, _ = ce.collar_angles(pose, _rest(), side)
            self.assertAlmostEqual(float(elev[0]), 20.0, delta=1.0)

    def test_mirrored_rotation_gives_the_same_angles_on_the_other_side(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            pose_l = _pose()
            _set(pose_l, ce.COLLAR["L"], rng.normal(scale=0.4, size=3))
            _set(pose_l, ce.SHOULDER["L"], rng.normal(scale=0.8, size=3))
            pose_r = _pose()
            for j in ("COLLAR", "SHOULDER"):
                src = pose_l[:, getattr(ce, j)["L"] - 1]
                pose_r[:, getattr(ce, j)["R"] - 1] = MIRROR @ src @ MIRROR
            for fn in (ce.collar_angles, ce.arm_angles):
                left = fn(pose_l, _rest(), "L")
                right = fn(pose_r, _rest(), "R")
                for a, b in zip(left, right):
                    self.assertAlmostEqual(float(a[0]), float(b[0]), places=9)

    def test_shoulder_rotation_alone_leaves_collar_angles_unchanged(self):
        pose = _pose()
        _set(pose, ce.SHOULDER["L"], (0.3, -0.5, 0.9))
        elev, prot = ce.collar_angles(pose, _rest(), "L")
        self.assertAlmostEqual(float(elev[0]), 0.0, places=9)
        self.assertAlmostEqual(float(prot[0]), 0.0, places=9)


class RepAndFitTests(unittest.TestCase):
    def test_rep_deltas_use_the_drivers_extreme_frames(self):
        driver = np.array([0.0, 5.0, 10.0, 5.0, 0.0, 1.0, 8.0, 2.0])
        collar = driver * 0.3
        arm = driver * 2.0
        out = ce.rep_deltas(driver, collar, arm, [(0, 5), (5, 8)], "s")
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0].d_collar, 3.0)
        self.assertAlmostEqual(out[0].d_arm, 20.0)
        self.assertAlmostEqual(out[1].d_arm, 14.0)

    def test_a_slaved_collar_has_zero_residual_and_an_independent_one_does_not(self):
        rng = np.random.default_rng(0)
        slaved, free = [], []
        for s in range(4):
            for r in range(5):
                arm = 80.0 + 5 * s + rng.normal(scale=6.0)
                slaved.append(ce.RepDelta(f"s{s}", r, 0.35 * arm, arm))
                free.append(ce.RepDelta(f"s{s}", r, 0.35 * arm + rng.normal(scale=4.0), arm))
        fit_slaved = ce.within_subject_fit(slaved)
        self.assertAlmostEqual(fit_slaved.slope, 0.35, places=9)
        self.assertLess(fit_slaved.residual_sd, 1e-9)
        fit_free = ce.within_subject_fit(free)
        self.assertGreater(fit_free.residual_sd, 2.0)
        self.assertLess(fit_free.r2_within, 0.9)

    def test_subject_offsets_do_not_count_as_independent_signal(self):
        deltas = [ce.RepDelta(f"s{s}", r, 10.0 * s + 0.3 * (70 + r), 70 + r) for s in range(3) for r in range(5)]
        self.assertLess(ce.within_subject_fit(deltas).residual_sd, 1e-9)

    def test_jitter_residuals_are_zero_on_a_constant_and_small_on_a_slow_ramp(self):
        self.assertTrue(np.allclose(ce.jitter_residuals(np.full(50, 7.0)), 0.0))
        ramp = np.linspace(0, 10, 200)
        self.assertLess(np.abs(ce.jitter_residuals(ramp)[20:-20]).max(), 1e-9)

    def test_second_difference_sigma_ignores_smooth_motion_and_recovers_white_noise(self):
        t = np.linspace(0, 4, 400)
        self.assertLess(ce.second_difference_sigma(30 * np.sin(t)), 1e-3)
        noise = np.random.default_rng(1).normal(scale=0.5, size=20000)
        self.assertAlmostEqual(ce.second_difference_sigma(30 * np.sin(np.linspace(0, 4, 20000)) + noise), 0.5, delta=0.02)

    def test_verdict_rule(self):
        self.assertEqual(ce.verdict(residual_sd=1.0, noise_floor=0.6, shrug_amplitude=30.0), "STOP")
        self.assertEqual(ce.verdict(residual_sd=2.0, noise_floor=0.6, shrug_amplitude=5.0), "NOISY")
        self.assertEqual(ce.verdict(residual_sd=2.0, noise_floor=0.6, shrug_amplitude=30.0), "PASS")
        self.assertEqual(ce.verdict(residual_sd=float("nan"), noise_floor=0.6, shrug_amplitude=30.0), "STOP")


if __name__ == "__main__":
    unittest.main()
