"""Mathematical contracts for the retrospective threshold audit."""
import unittest

import numpy as np

from src.fit3d.threshold_audit import integrate_decisions, threshold_losses, uniform_losses


class ThresholdIntegralTests(unittest.TestCase):
    def test_known_disagreement_interval_and_clipping(self):
        g, p = np.array([89.]), np.array([93.])
        self.assertAlmostEqual(integrate_decisions(g, p, 0., 180.), 4.)
        self.assertAlmostEqual(integrate_decisions(g, p, 90., 91.), 1.)
        self.assertAlmostEqual(integrate_decisions(g, p, 94., 100.), 0.)

    def test_full_integral_equals_mae_with_mixed_error_directions(self):
        g = np.array([-2., 1., 1., 10.])
        p = np.array([3., -1., 1., 7.])
        self.assertAlmostEqual(integrate_decisions(g, p, -5., 15.), 2.5)
        self.assertAlmostEqual(float(uniform_losses(g, p, -5., 15.).mean()), .125)

    def test_full_improvement_can_coexist_with_central_deterioration(self):
        g = np.array([60., 89.])
        before, after = np.array([70., 89.]), np.array([60., 91.])
        self.assertLess(integrate_decisions(g, after, 0., 180.),
                        integrate_decisions(g, before, 0., 180.))
        self.assertGreater(integrate_decisions(g, after, 88., 92.),
                           integrate_decisions(g, before, 88., 92.))

    def test_symmetry_and_strict_threshold_boundary(self):
        g, p = np.array([89., 90.]), np.array([90., 91.])
        thresholds = np.array([90.])
        np.testing.assert_array_equal(threshold_losses(g, p, thresholds), [0., 1.])
        np.testing.assert_array_equal(threshold_losses(g, p, thresholds),
                                      threshold_losses(p, g, thresholds))
