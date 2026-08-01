"""Unit tests for src.fit3d.uncertainty_eval -- synthetic only, no dataset required."""

from __future__ import annotations

import unittest

import numpy as np

from src.fit3d import uncertainty_eval as ue

try:  # pragma: no cover - scipy is optional in this repo and absent from requirements-ci
    from scipy.stats import spearmanr as _scipy_spearmanr
except Exception:  # pragma: no cover
    _scipy_spearmanr = None


def make_sequence(subject, camera, n=60, err_scale=1.0, noise=0.0, seed=0, action="squat"):
    """A sequence whose uncertainty is a noisy linear function of its true error."""
    rng = np.random.default_rng(seed)
    magnitude = rng.gamma(2.0, 5.0, (n, ue.N_JOINTS)) * err_scale
    direction = rng.normal(0, 1, (n, ue.N_JOINTS, 3))
    direction /= np.linalg.norm(direction, axis=2, keepdims=True)
    delta = direction * magnitude[:, :, None]
    delta[:, 0, :] = 0.0                                    # root: zero by construction
    uncertainty = magnitude + rng.normal(0, noise, magnitude.shape)
    pose = rng.normal(0, 300, (n, ue.N_JOINTS, 3))
    pose[:, 0, :] = 0.0
    return ue.Sequence(subject=subject, camera=camera, action=action, delta=delta,
                       uncertainty=uncertainty, pose=pose, swap_lr=True)


class AverageRanksTests(unittest.TestCase):
    def test_ties_get_the_average_rank(self):
        np.testing.assert_allclose(ue._average_ranks(np.array([1., 1., 1., 2., 3.])),
                                   [1.0, 1.0, 1.0, 3.0, 4.0])

    def test_distinct_values_give_plain_ranks(self):
        np.testing.assert_allclose(ue._average_ranks(np.array([30., 10., 20.])), [2.0, 0.0, 1.0])

    def test_a_constant_array_has_no_rank_variance(self):
        """argsort(argsort(x)) would return 0..n-1 here and fabricate a correlation."""
        self.assertAlmostEqual(float(ue._average_ranks(np.zeros(50)).std()), 0.0)


class SpearmanTests(unittest.TestCase):
    def test_monotone_relationship_is_one(self):
        x = np.array([1., 2., 3., 4., 5.])
        self.assertAlmostEqual(ue.spearman(x, np.exp(x)), 1.0, places=12)
        self.assertAlmostEqual(ue.spearman(x, -np.exp(x)), -1.0, places=12)

    def test_constant_input_returns_nan(self):
        self.assertTrue(np.isnan(ue.spearman(np.zeros(50), np.arange(50.))))

    def test_ignores_non_finite_pairs(self):
        a = np.array([1., 2., 3., np.nan, 5.])
        b = np.array([1., 2., 3., 100., 5.])
        self.assertAlmostEqual(ue.spearman(a, b), 1.0, places=12)

    def test_too_few_points_returns_nan(self):
        self.assertTrue(np.isnan(ue.spearman(np.array([1.0]), np.array([2.0]))))

    @unittest.skipIf(_scipy_spearmanr is None, "scipy not installed")
    def test_matches_scipy_including_heavy_ties(self):
        rng = np.random.default_rng(4)
        for _ in range(5):
            x = rng.integers(0, 5, 400).astype(float)
            y = rng.integers(0, 4, 400).astype(float)
            self.assertAlmostEqual(ue.spearman(x, y), float(_scipy_spearmanr(x, y).statistic),
                                   places=10)


class BiasCorrectionTests(unittest.TestCase):
    def test_recovers_a_planted_constant_offset(self):
        rng = np.random.default_rng(5)
        delta = rng.normal(0, 1.0, (500, ue.N_JOINTS, 3))
        offset = np.zeros((1, ue.N_JOINTS, 3))
        offset[0, 8] = [176.0, 0.0, 0.0]           # a thorax-sized convention mismatch
        bias = ue.convention_bias(delta + offset)
        np.testing.assert_allclose(bias[0, 8], [176.0, 0.0, 0.0], atol=0.3)

    def test_correction_removes_the_offset_from_the_error(self):
        rng = np.random.default_rng(6)
        delta = rng.normal(0, 1.0, (500, ue.N_JOINTS, 3))
        offset = np.zeros((1, ue.N_JOINTS, 3)); offset[0, 8] = [176.0, 0.0, 0.0]
        raw = ue.corrected_error(delta + offset, bias=np.zeros((1, ue.N_JOINTS, 3)))
        fixed = ue.corrected_error(delta + offset)
        self.assertGreater(np.median(raw[:, 8]), 170.0)
        self.assertLess(np.median(fixed[:, 8]), 3.0)

    def test_uses_a_median_so_outliers_do_not_move_it(self):
        delta = np.zeros((100, ue.N_JOINTS, 3))
        delta[:5, 3] = 5000.0                       # a handful of tracking blow-ups
        np.testing.assert_allclose(ue.convention_bias(delta)[0, 3], [0.0, 0.0, 0.0])

    def test_accepts_an_externally_supplied_bias(self):
        """LOSO must be able to pass a bias fitted on training folds only."""
        delta = np.ones((20, ue.N_JOINTS, 3))
        supplied = np.full((1, ue.N_JOINTS, 3), 1.0)
        np.testing.assert_allclose(ue.corrected_error(delta, bias=supplied), 0.0, atol=1e-12)


class CalibrationTests(unittest.TestCase):
    def test_detects_a_calibrated_channel(self):
        seq = make_sequence("s01", "c0", n=400, noise=0.5, seed=7)
        rho = ue.within_joint_calibration(ue.corrected_error(seq.delta), seq.uncertainty)
        self.assertTrue(np.all(rho[1:] > 0.7))

    def test_reports_nan_for_the_degenerate_root_joint(self):
        seq = make_sequence("s01", "c0", n=200, seed=8)
        rho = ue.within_joint_calibration(ue.corrected_error(seq.delta), seq.uncertainty)
        self.assertTrue(np.isnan(rho[0]))

    def test_an_uninformative_channel_scores_near_zero(self):
        rng = np.random.default_rng(9)
        seq = make_sequence("s01", "c0", n=400, seed=9)
        scrambled = ue.Sequence(seq.subject, seq.camera, seq.action, seq.delta,
                                rng.permutation(seq.uncertainty), seq.pose, seq.swap_lr)
        rho = ue.within_joint_calibration(ue.corrected_error(seq.delta), scrambled.uncertainty)
        self.assertLess(float(np.nanmax(np.abs(rho))), 0.25)


class CrossViewTests(unittest.TestCase):
    def test_agreement_is_high_when_uncertainty_tracks_error(self):
        seqs = [make_sequence("s01", f"c{i}", n=200, noise=0.5, seed=20 + i) for i in range(4)]
        self.assertGreater(ue.cross_view_agreement(seqs)["mean_rho"], 0.7)

    def test_subjects_with_too_few_views_are_skipped(self):
        seqs = [make_sequence("s01", "c0", n=100, seed=30),
                make_sequence("s01", "c1", n=100, seed=31)]
        self.assertEqual(ue.cross_view_agreement(seqs)["n_subjects"], 0)

    def test_a_uniformly_worse_camera_cannot_create_agreement(self):
        """Standardising within (view, joint) must remove a constant per-camera penalty."""
        rng = np.random.default_rng(40)
        seqs = []
        for i in range(4):
            s = make_sequence("s01", f"c{i}", n=300, err_scale=1.0 + i, seed=100)
            # uncertainty is pure noise, but camera i is genuinely worse overall
            seqs.append(ue.Sequence(s.subject, s.camera, s.action, s.delta,
                                    rng.normal(50 * (i + 1), 1.0, s.uncertainty.shape),
                                    s.pose, s.swap_lr))
        self.assertLess(abs(ue.cross_view_agreement(seqs)["mean_rho"]), 0.2)


class RedundancyTests(unittest.TestCase):
    def test_reports_every_predictor(self):
        seqs = [make_sequence(f"s{k}", f"c{i}", n=60, noise=1.0, seed=50 + 4 * k + i)
                for k in range(3) for i in range(2)]
        out = ue.redundancy_test(seqs, lambdas=(10.0,))
        self.assertEqual(out["n_subjects"], 3)
        for key in ("lookup", "unc", "pose", "pose_unc"):
            self.assertTrue(np.isfinite(out[key]), key)

    def test_a_perfectly_informative_channel_beats_the_lookup(self):
        seqs = [make_sequence(f"s{k}", "c0", n=300, noise=0.01, seed=70 + k) for k in range(4)]
        out = ue.redundancy_test(seqs, lambdas=(1.0, 10.0))
        self.assertLess(out["unc"], out["lookup"] * 0.6)

    def test_a_useless_channel_does_not_beat_the_lookup(self):
        rng = np.random.default_rng(80)
        seqs = []
        for k in range(4):
            s = make_sequence(f"s{k}", "c0", n=300, seed=80 + k)
            seqs.append(ue.Sequence(s.subject, s.camera, s.action, s.delta,
                                    rng.normal(20, 5, s.uncertainty.shape), s.pose, s.swap_lr))
        out = ue.redundancy_test(seqs, lambdas=(10.0, 100.0))
        self.assertGreater(out["unc"], out["lookup"] * 0.95)


class SequenceTests(unittest.TestCase):
    def test_len_reports_frame_count(self):
        self.assertEqual(len(make_sequence("s01", "c0", n=42)), 42)


if __name__ == "__main__":
    unittest.main()
