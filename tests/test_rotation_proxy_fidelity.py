"""Pure-helper tests for the Torso Twist rotation-proxy fidelity harness.

Only the helpers ABOVE the corpus banner in `src/fit3d/rotation_proxy_fidelity.py` are tested
here, because everything below it reads `data/Fit3D`, which is gitignored and absent in CI. Same
split as `tests/test_lunge_rule_validation.py`.

These exist because the design spec quotes this harness's output as a citation of record, and a
number whose script nobody can re-run is a defect this project has already logged once.
"""

import math
import unittest

import numpy as np

from src.fit3d.rotation_proxy_fidelity import (
    SPEC_RATIO_CUT,
    decision_agreement,
    horizontal_azimuth,
    proxy_rotation_deg,
    rotation_about_own_median,
    sensitivity_per_degree,
    signed_axial_twist,
    smoothing_residual,
    spearman,
    summarize,
    trunk_thigh_deg,
    wrap_radians,
)


class ProxyInversionTest(unittest.TestCase):
    def test_it_recovers_the_rotation_that_produced_the_width(self) -> None:
        """`arccos(width / resting)` must invert `resting * cos(theta)` exactly when the window
        contains a square-on frame."""
        angles = np.array([0.0, 10.0, 30.0, 60.0, 85.0])
        widths = 0.2 * np.cos(np.radians(angles))
        recovered = proxy_rotation_deg(widths)
        np.testing.assert_allclose(recovered, angles, atol=1e-6)

    def test_it_is_EVEN_so_the_two_sides_of_a_twist_are_indistinguishable(self) -> None:
        """The structural defect that withdrew `tt_lumbar_rotation_dominant`, asserted rather
        than described: opposite rotations give the identical reading."""
        left = proxy_rotation_deg(0.2 * np.cos(np.radians([0.0, 40.0])))
        right = proxy_rotation_deg(0.2 * np.cos(np.radians([0.0, -40.0])))
        np.testing.assert_allclose(left, right, atol=1e-9)

    def test_a_window_that_never_passes_square_over_reads_the_rotation(self) -> None:
        """"Resting width" is taken as the window's own maximum, so a subject who is oblique
        throughout has their obliquity absorbed into the reference and every reading is
        understated relative to the true rotation -- the mechanism behind the measured bias."""
        true_angles = np.array([25.0, 45.0, 65.0])
        widths = 0.2 * np.cos(np.radians(true_angles))
        recovered = proxy_rotation_deg(widths)
        self.assertAlmostEqual(recovered[0], 0.0, places=6)
        self.assertLess(recovered[2], true_angles[2])

    def test_a_degenerate_window_is_nan_rather_than_a_number(self) -> None:
        self.assertTrue(np.all(np.isnan(proxy_rotation_deg(np.array([0.0, 0.0])))))
        self.assertTrue(np.all(np.isnan(proxy_rotation_deg(np.array([np.nan, np.nan])))))


class SensitivityTest(unittest.TestCase):
    def test_the_derivative_is_zero_at_the_braced_centre(self) -> None:
        self.assertAlmostEqual(sensitivity_per_degree(0.2, 0.0), 0.0, places=12)

    def test_it_grows_monotonically_across_the_useful_range(self) -> None:
        values = [sensitivity_per_degree(0.2, angle) for angle in (2.0, 7.5, 30.0, 60.0, 89.0)]
        self.assertEqual(values, sorted(values))

    def test_it_matches_the_analytic_form(self) -> None:
        self.assertAlmostEqual(
            sensitivity_per_degree(0.2, 60.0),
            0.2 * math.sin(math.radians(60.0)) * math.radians(1.0),
            places=12,
        )


class RotationReferenceTest(unittest.TestCase):
    def test_azimuth_reads_the_world_horizontal_plane(self) -> None:
        vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 5.0]])
        np.testing.assert_allclose(horizontal_azimuth(vectors), [0.0, math.pi / 2, math.pi])

    def test_wrap_folds_into_the_principal_branch(self) -> None:
        """The boundary lands on -pi rather than +pi, for BOTH signs of the input. They are the
        same angle, so nothing downstream depends on which representative is chosen -- pinned so
        a future reader meets the convention instead of assuming the other one."""
        np.testing.assert_allclose(
            wrap_radians(np.array([0.0, 0.5, -0.5, 3 * math.pi, -3 * math.pi])),
            [0.0, 0.5, -0.5, -math.pi, -math.pi],
            atol=1e-12,
        )

    def test_rotation_is_referenced_to_the_clips_own_median_not_its_first_frame(self) -> None:
        """A subject does not stand square to the camera on request. Referenced to the median,
        a clip that starts already turned still reports a symmetric excursion."""
        headings = np.radians(np.array([20.0, 30.0, 40.0, 50.0, 60.0]))
        vectors = np.stack([np.cos(headings), np.sin(headings), np.zeros_like(headings)], axis=1)
        rotation = np.degrees(rotation_about_own_median(vectors))
        np.testing.assert_allclose(rotation, [-20.0, -10.0, 0.0, 10.0, 20.0], atol=1e-9)


class AxialTwistTest(unittest.TestCase):
    def test_a_whole_body_turn_is_not_a_twist(self) -> None:
        """The reason the twist is measured in the pelvis frame: rotating the WHOLE subject moves
        both lines together and must read zero."""
        axis = np.tile([0.0, 0.0, 1.0], (3, 1))
        for degrees in (0.0, 35.0, 90.0):
            radians = math.radians(degrees)
            line = np.tile([math.cos(radians), math.sin(radians), 0.0], (3, 1))
            twist = signed_axial_twist(line, line, axis)
            np.testing.assert_allclose(twist, 0.0, atol=1e-9)

    def test_it_is_signed_so_the_two_directions_are_distinguishable(self) -> None:
        axis = np.tile([0.0, 0.0, 1.0], (1, 1))
        hip = np.array([[1.0, 0.0, 0.0]])
        left = signed_axial_twist(hip, np.array([[math.cos(0.5), math.sin(0.5), 0.0]]), axis)
        right = signed_axial_twist(hip, np.array([[math.cos(-0.5), math.sin(-0.5), 0.0]]), axis)
        self.assertAlmostEqual(float(left[0]), -float(right[0]), places=9)
        self.assertAlmostEqual(float(left[0]), 0.5, places=9)


class BraceAngleTest(unittest.TestCase):
    INDICES = {"shoulders": (0, 1), "hips": (2, 3), "knees": (4, 5)}

    def _points(self, trunk_deg: float, shoulder_half: float = 0.07) -> np.ndarray:
        hip_mid = np.array([0.5, 0.6])
        thigh = np.array([1.0, 0.0])
        trunk = np.array([math.cos(math.radians(trunk_deg)), math.sin(math.radians(trunk_deg))])
        across = np.array([-trunk[1], trunk[0]])
        shoulder_mid = hip_mid + 0.24 * trunk
        return np.array([[
            shoulder_mid + shoulder_half * across, shoulder_mid - shoulder_half * across,
            hip_mid + np.array([0.05, 0.0]), hip_mid - np.array([0.05, 0.0]),
            hip_mid + 0.2 * thigh + np.array([0.0, 0.05]), hip_mid + 0.2 * thigh - np.array([0.0, 0.05]),
        ]])

    def test_it_equals_the_trunk_to_thigh_angle(self) -> None:
        for angle in (60.0, 95.0, 130.0):
            measured = trunk_thigh_deg(self._points(angle), self.INDICES)
            self.assertAlmostEqual(float(measured[0]), angle, places=4)

    def test_it_is_unmoved_by_shoulder_line_foreshortening(self) -> None:
        """The midpoint construction the shipped rule uses: an axial twist projects as a narrower
        shoulder line and must not change the brace reading. A same-side construction would."""
        square = trunk_thigh_deg(self._points(95.0, shoulder_half=0.07), self.INDICES)
        turned = trunk_thigh_deg(self._points(95.0, shoulder_half=0.02), self.INDICES)
        self.assertAlmostEqual(float(square[0]), float(turned[0]), places=9)


class DecisionAgreementTest(unittest.TestCase):
    def test_it_counts_the_two_disagreement_directions_apart(self) -> None:
        truth = np.array([0.9, 0.2, 0.9, 0.2])
        proxy = np.array([0.9, 0.9, 0.2, 0.2])
        result = decision_agreement(truth, proxy)
        self.assertEqual(result["truth_fires"], 2)
        self.assertEqual(result["proxy_fires"], 2)
        self.assertEqual(result["disagree"], 2)
        self.assertEqual(result["proxy_only"], 1)
        self.assertEqual(result["truth_only"], 1)

    def test_it_uses_the_specs_own_cut_by_default(self) -> None:
        self.assertEqual(decision_agreement(np.array([0.6]), np.array([0.6]))["cut"], SPEC_RATIO_CUT)
        self.assertEqual(SPEC_RATIO_CUT, 0.6)

    def test_a_high_rank_correlation_does_not_imply_agreement(self) -> None:
        """The reason both are reported: a monotone but biased proxy scores a perfect rank
        correlation while flipping every decision."""
        truth = np.array([0.1, 0.2, 0.3, 0.4])
        proxy = truth + 0.5
        self.assertAlmostEqual(spearman(truth, proxy), 1.0, places=9)
        self.assertEqual(decision_agreement(truth, proxy)["disagree"], 4)


class JitterTest(unittest.TestCase):
    def test_a_smooth_series_has_no_residual(self) -> None:
        self.assertAlmostEqual(smoothing_residual(np.linspace(0.0, 1.0, 60), window=15), 0.0, places=6)

    def test_it_grows_with_the_noise_it_is_measuring(self) -> None:
        rng = np.random.default_rng(0)
        clean = np.linspace(0.0, 1.0, 400)
        quiet = smoothing_residual(clean + rng.normal(0.0, 0.001, 400), window=15)
        loud = smoothing_residual(clean + rng.normal(0.0, 0.010, 400), window=15)
        self.assertGreater(loud, quiet * 5.0)

    def test_a_period_two_alternation_returns_the_FULL_swing_not_half_of_it(self) -> None:
        """A quirk worth pinning rather than discovering later: an odd-width median window
        centred on one phase of a period-2 alternation contains 8 samples of the OTHER phase and
        7 of its own, so the median locks onto the opposite phase and the residual is the whole
        swing. Real landmark jitter is not period-2, so this does not affect the reported floor."""
        self.assertAlmostEqual(smoothing_residual(np.array([0.0, 0.02] * 40), window=15), 0.02, places=9)

    def test_a_series_shorter_than_the_window_is_nan_rather_than_a_number(self) -> None:
        self.assertTrue(math.isnan(smoothing_residual(np.zeros(5), window=15)))


class SummarizeTest(unittest.TestCase):
    def test_it_drops_non_finite_values_rather_than_propagating_them(self) -> None:
        result = summarize(np.array([1.0, np.nan, 3.0, np.inf]))
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["median"], 2.0)

    def test_an_empty_input_reports_a_count_and_no_statistics(self) -> None:
        self.assertEqual(summarize(np.array([])), {"n": 0})


if __name__ == "__main__":
    unittest.main()
