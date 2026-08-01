"""Tests for the Fit3D model-fusion gate (src/fit3d/model_fusion.py).

The analysis layer is pure numpy over a signed-error matrix, so it is tested on synthetic
error matrices with KNOWN complementarity structure -- that is where the claims live
("errors are correlated => fusion cannot help"), and a bug there would silently produce a
publishable-looking number.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.fit3d import decision_eval as dec
from src.fit3d import model_fusion as mf


class TestRankdata(unittest.TestCase):
    def test_matches_scipy_with_ties(self):
        try:
            from scipy.stats import rankdata as scipy_rankdata
        except ImportError:  # pragma: no cover - scipy optional
            self.skipTest("scipy not installed")
        rng = np.random.default_rng(0)
        x = rng.integers(0, 5, size=200).astype(np.float64)  # heavy ties
        np.testing.assert_allclose(mf.rankdata(x), scipy_rankdata(x))

    def test_ties_share_mean_rank(self):
        # argsort(argsort(x)) would return 0,1,2,3 here -- fabricating an ordering.
        np.testing.assert_allclose(mf.rankdata(np.array([7.0, 7.0, 7.0, 7.0])), [2.5] * 4)

    def test_constant_column_gives_nan_correlation(self):
        a = np.ones(50)
        self.assertTrue(np.isnan(mf._spearman(a, np.arange(50.0))))


class TestAnalyseCue(unittest.TestCase):
    def setUp(self):
        self.seq = np.zeros(400, dtype=np.int64)
        self.rng = np.random.default_rng(7)

    def test_identical_errors_leave_no_headroom(self):
        """Perfectly correlated arms: oracle == best single, nothing to route."""
        e = self.rng.normal(size=400)
        err = np.stack([e, e], axis=1)
        m = mf.analyse_cue(err, ["a", "b"], self.seq)
        self.assertAlmostEqual(m["headroom_frame"], 0.0, places=9)
        self.assertAlmostEqual(m["oracle_frame"], m["best_single"]["mae"], places=9)
        self.assertAlmostEqual(m["corr_signed"][0][1], 1.0, places=6)
        # Averaging identical errors changes nothing.
        self.assertAlmostEqual(m["mean_fusion_mae"], m["best_single"]["mae"], places=9)

    def test_independent_errors_expose_headroom_and_averaging_gain(self):
        """Decorrelated arms: oracle beats best single AND averaging cancels noise."""
        err = self.rng.normal(size=(4000, 3))
        m = mf.analyse_cue(err, ["a", "b", "c"], np.zeros(4000, dtype=np.int64))
        self.assertGreater(m["headroom_frame"], 0.2)
        self.assertLess(m["mean_fusion_mae"], m["best_single"]["mae"])
        self.assertGreater(m["switch_rate"], 0.4)
        self.assertAlmostEqual(m["corr_signed"][0][1], 0.0, places=1)

    def test_opposite_bias_is_cancelled_by_averaging_but_not_by_routing(self):
        """The signature case: two arms with equal and opposite constant bias.

        Averaging is exact (bias cancels); an oracle router cannot beat either arm, because
        each is uniformly wrong. This is why signed and absolute correlation are reported
        separately -- they answer different fusion questions.
        """
        n = 500
        err = np.stack([np.full(n, 4.0), np.full(n, -4.0)], axis=1)
        m = mf.analyse_cue(err, ["a", "b"], np.zeros(n, dtype=np.int64))
        self.assertAlmostEqual(m["mean_fusion_mae"], 0.0, places=9)
        self.assertAlmostEqual(m["oracle_frame"], 4.0, places=9)   # routing buys nothing
        self.assertAlmostEqual(m["headroom_frame"], 0.0, places=9)

    def test_frac_best_and_switch_rate_identify_a_dominant_arm(self):
        err = np.stack([self.rng.normal(scale=0.1, size=400),
                        self.rng.normal(scale=9.0, size=400)], axis=1)
        m = mf.analyse_cue(err, ["good", "bad"], self.seq)
        self.assertEqual(m["best_single"]["arm"], "good")
        self.assertGreater(m["per_arm"]["good"]["frac_best"], 0.9)
        self.assertLess(m["switch_rate"], 0.1)

    def test_oracle_per_sequence_is_between_best_single_and_per_frame(self):
        """Per-sequence routing is weaker than per-frame but no worse than one fixed model."""
        n = 300
        seq = np.repeat(np.arange(3), n // 3)
        err = np.zeros((n, 2))
        err[seq == 0] = np.stack([np.full(100, 1.0), np.full(100, 5.0)], axis=1)
        err[seq == 1] = np.stack([np.full(100, 5.0), np.full(100, 1.0)], axis=1)
        err[seq == 2] = np.stack([np.full(100, 3.0), np.full(100, 3.0)], axis=1)
        m = mf.analyse_cue(err, ["a", "b"], seq)
        self.assertAlmostEqual(m["best_single"]["mae"], 3.0, places=9)
        self.assertAlmostEqual(m["oracle_seq"], (1 + 1 + 3) / 3, places=9)
        self.assertLessEqual(m["oracle_frame"], m["oracle_seq"] + 1e-9)

    def test_median_fusion_resists_a_single_outlier_arm(self):
        n = 300
        err = np.stack([np.full(n, 1.0), np.full(n, -1.0), np.full(n, 50.0)], axis=1)
        m = mf.analyse_cue(err, ["a", "b", "outlier"], np.zeros(n, dtype=np.int64))
        self.assertAlmostEqual(m["median_fusion_mae"], 1.0, places=9)
        self.assertGreater(m["mean_fusion_mae"], 10.0)

    def test_empty_input_is_reported_not_crashed(self):
        self.assertEqual(mf.analyse_cue(np.zeros((0, 2)), ["a", "b"],
                                        np.zeros(0, dtype=np.int64)), {"n_frames": 0})


class TestDebiasControl(unittest.TestCase):
    """The control that separates a real fusion gain from bias cancellation."""

    def test_removes_each_arm_and_camera_offset(self):
        cam = np.array(["A"] * 100 + ["B"] * 100, dtype=object)
        err = np.zeros((200, 2))
        err[:100, 0], err[100:, 0] = 3.0, -7.0
        err[:100, 1], err[100:, 1] = 11.0, 2.0
        out = mf.debias_per_camera(err, cam)
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_opposite_bias_gain_disappears_after_debiasing(self):
        """The trap this control exists for.

        Two arms with equal/opposite constant bias: mean-fusion looks like it crushes the best
        single arm, but the entire gain is bias cancellation -- after one constant per arm both
        arms are perfect and fusion buys exactly nothing.
        """
        n, cam = 400, np.full(400, "A", dtype=object)
        err = np.stack([np.full(n, 6.0), np.full(n, -6.0)], axis=1)
        both = mf.analyse_cue_both(err, ["a", "b"], np.zeros(n, dtype=np.int64), cam)
        # RAW: fusion looks like a huge win over the best single arm.
        self.assertAlmostEqual(both["raw"]["best_single"]["mae"], 6.0, places=9)
        self.assertAlmostEqual(both["raw"]["mean_fusion_mae"], 0.0, places=9)
        # DEBIASED: the single arm is already perfect; the "gain" was calibration, not fusion.
        self.assertAlmostEqual(both["debiased"]["best_single"]["mae"], 0.0, places=9)
        self.assertAlmostEqual(both["debiased"]["headroom_frame"], 0.0, places=9)

    def test_bias_diversity_inflates_the_raw_per_frame_oracle(self):
        """Arms that make the IDENTICAL mistake, differing only by a constant.

        There is zero complementarity by construction -- after one constant per arm the
        residuals are the same number. Yet the raw per-frame oracle reports real headroom,
        because on any frame it can pick whichever constant happens to offset the shared
        error. This is the artifact the debiased track exists to expose, and it is exactly the
        shape of the real data (metric-3D arms biased ~-6 deg, projected-2D arm ~+18 deg).
        """
        rng = np.random.default_rng(11)
        n = 4000
        shared = rng.normal(scale=5.0, size=n)            # ONE error, shared by every arm
        err = shared[:, None] + np.array([-6.0, 0.0, 18.0])
        cam = np.full(n, "A", dtype=object)
        both = mf.analyse_cue_both(err, ["a", "b", "c"], np.zeros(n, dtype=np.int64), cam)
        self.assertGreater(both["raw"]["headroom_frame_pct"], 10.0)
        self.assertAlmostEqual(both["debiased"]["headroom_frame_pct"], 0.0, places=6)

    def test_genuine_complementarity_survives_debiasing(self):
        """Positive control: independent noise, no bias -> headroom is real and survives."""
        rng = np.random.default_rng(5)
        err = rng.normal(size=(3000, 3))
        cam = np.full(3000, "A", dtype=object)
        both = mf.analyse_cue_both(err, ["a", "b", "c"], np.zeros(3000, dtype=np.int64), cam)
        self.assertGreater(both["debiased"]["headroom_frame_pct"], 20.0)
        self.assertLess(both["debiased"]["mean_fusion_mae"], both["debiased"]["best_single"]["mae"])

    def test_empty_input_returns_both_tracks(self):
        out = mf.analyse_cue_both(np.zeros((0, 2)), ["a", "b"],
                                  np.zeros(0, dtype=np.int64), np.zeros(0, dtype=object))
        self.assertEqual(out, {"raw": {"n_frames": 0}, "debiased": {"n_frames": 0}})


class TestShuffledOracleControl(unittest.TestCase):
    """Order-statistic baseline for the per-frame oracle.

    NOTE the confound documented on :func:`mf.shuffled_oracle`: a global column shuffle also
    destroys SHARED per-frame difficulty, so ``shuffled < oracle_frame`` does not by itself
    imply routing is unexploitable. ``test_shared_difficulty_alone_beats_the_real_oracle``
    below is the counterexample that makes the confound concrete.
    """

    def test_shared_difficulty_alone_beats_the_real_oracle(self):
        """Zero cross-model dependence, yet shuffling still 'wins' -- the confound, exhibited.

        Errors factor as d_f * z_fm with z independent across arms, so no arm is ever
        predictably better than another. The shuffled oracle nonetheless comes out LOWER,
        purely because pooling mixes hard frames' magnitudes with easy frames'.
        """
        rng = np.random.default_rng(17)
        n = 6000
        d = np.where(rng.random(n) < 0.5, 1.0, 12.0)      # frame difficulty, shared by all arms
        err = d[:, None] * rng.normal(size=(n, 3))        # z independent across arms
        real = float(np.abs(err).min(axis=1).mean())
        self.assertLess(mf.shuffled_oracle(err), real)

    def test_independent_noise_has_no_learnable_structure(self):
        """Independent arms: the real oracle gain is ENTIRELY luck, so shuffling changes nothing."""
        rng = np.random.default_rng(2)
        err = rng.normal(size=(6000, 3))
        real = float(np.abs(err).min(axis=1).mean())
        self.assertAlmostEqual(mf.shuffled_oracle(err), real, delta=0.03)

    def test_real_structure_beats_the_shuffled_ceiling(self):
        """Arms that take turns being exactly right -- structure a router could actually learn."""
        n = 3000
        err = np.full((n, 3), 9.0)
        err[np.arange(n), np.arange(n) % 3] = 0.0     # one arm is perfect on each frame, in turn
        real = float(np.abs(err).min(axis=1).mean())
        self.assertAlmostEqual(real, 0.0, places=9)
        # Luck alone lands at (2/3)^3 * 9 = 2.67 -- nowhere near the structured 0.0.
        self.assertGreater(mf.shuffled_oracle(err), 2.0)

    def test_is_deterministic_for_a_fixed_seed(self):
        rng = np.random.default_rng(4)
        err = rng.normal(size=(500, 3))
        self.assertEqual(mf.shuffled_oracle(err, seed=1), mf.shuffled_oracle(err, seed=1))

    def test_reported_in_analyse_cue(self):
        rng = np.random.default_rng(6)
        m = mf.analyse_cue(rng.normal(size=(800, 3)), ["a", "b", "c"],
                           np.zeros(800, dtype=np.int64))
        self.assertIn("shuffled_oracle", m)
        self.assertTrue(np.isfinite(m["shuffled_oracle"]))


class TestRepExtremeDecomposition(unittest.TestCase):
    """Separates the two ways a model can lose the verdict despite a better mean cue error."""

    def test_constant_offset_shows_up_everywhere_and_picks_the_right_frame(self):
        gt = np.array([90.0, 80.0, 70.0, 85.0])
        d = mf.decompose_rep_extreme(gt + 3.0, gt, "min")
        self.assertAlmostEqual(d["point_err"], 3.0)
        self.assertAlmostEqual(d["extreme_err"], 3.0)
        self.assertAlmostEqual(d["frame_offset"], 0.0)     # same frame is still the minimum
        self.assertAlmostEqual(d["pooled_err"], 3.0)

    def test_accurate_everywhere_but_selects_the_wrong_extreme_frame(self):
        """Selection effect: tiny error at the GT extreme, yet the reported extreme is wrong."""
        gt = np.array([90.0, 70.0, 71.0, 90.0])
        pred = np.array([90.0, 70.0, 60.0, 90.0])          # spurious dip one frame later
        d = mf.decompose_rep_extreme(pred, gt, "min")
        self.assertAlmostEqual(d["point_err"], 0.0)        # perfect ON the GT extreme frame
        self.assertAlmostEqual(d["extreme_err"], 10.0)     # but the reported extreme is 10 off
        self.assertAlmostEqual(d["frame_offset"], 1.0)

    def test_bad_at_the_extreme_frame_specifically(self):
        """Accuracy effect: fine on average, wrong exactly where the coach looks."""
        gt = np.array([90.0, 90.0, 70.0, 90.0])
        pred = np.array([90.0, 90.0, 82.0, 90.0])
        d = mf.decompose_rep_extreme(pred, gt, "min")
        self.assertAlmostEqual(d["point_err"], 12.0)
        self.assertAlmostEqual(d["extreme_err"], 12.0)
        self.assertAlmostEqual(d["frame_offset"], 0.0)
        self.assertAlmostEqual(d["pooled_err"], 3.0)       # pooled hides it (12/4 frames)

    def test_max_reducer_uses_the_maximum(self):
        gt = np.array([10.0, 40.0, 20.0])
        d = mf.decompose_rep_extreme(np.array([10.0, 44.0, 20.0]), gt, "max")
        self.assertAlmostEqual(d["extreme_err"], 4.0)
        self.assertAlmostEqual(d["frame_offset"], 0.0)

    def test_unsampled_frames_are_excluded_from_the_extreme_search(self):
        """A subsampled model is judged on the frames it actually saw, not on ones it never had.

        Frame 1 is the GT extreme but the model never sampled it; the comparison therefore
        re-anchors on the deepest SAMPLED frame (index 3, GT 75) instead of reporting NaN.
        """
        gt = np.array([90.0, 70.0, 80.0, 75.0])
        pred = np.array([91.0, np.nan, 81.0, 76.0])
        d = mf.decompose_rep_extreme(pred, gt, "min")
        self.assertAlmostEqual(d["point_err"], 1.0)        # 76 vs 75 at the deepest sampled frame
        self.assertAlmostEqual(d["extreme_err"], 1.0)
        self.assertAlmostEqual(d["frame_offset"], 0.0)
        self.assertTrue(np.isfinite(d["pooled_err"]))

    def test_frame_offset_is_measured_in_original_frame_numbers(self):
        gt = np.array([90.0, 88.0, 70.0, 72.0, 95.0])
        pred = np.array([90.0, np.nan, 71.0, 60.0, 95.0])  # its min sits 1 frame later
        d = mf.decompose_rep_extreme(pred, gt, "min")
        self.assertAlmostEqual(d["frame_offset"], 1.0)
        self.assertAlmostEqual(d["point_err"], 1.0)        # accurate where GT is deepest
        self.assertAlmostEqual(d["extreme_err"], 10.0)     # but reports a 10-deg deeper extreme

    def test_all_nan_returns_none(self):
        self.assertIsNone(mf.decompose_rep_extreme(np.full(4, np.nan), np.arange(4.0), "min"))


class TestDiscrimination(unittest.TestCase):
    """Shrinkage lowers MAE while destroying the ordering a verdict threshold needs."""

    def test_faithful_reading_has_unit_slope_and_spread(self):
        rng = np.random.default_rng(1)
        g = rng.normal(70.0, 12.0, size=300)
        d = mf._discrimination(list(zip(g, g + 2.0)))          # constant offset only
        self.assertAlmostEqual(d["slope"], 1.0, places=6)
        self.assertAlmostEqual(d["spread_ratio"], 1.0, places=6)
        self.assertAlmostEqual(d["r"], 1.0, places=6)

    def test_shrunk_reading_wins_on_mae_but_loses_slope(self):
        """The exact trade the verdict punishes: half the spread, lower MAE, worse ordering."""
        rng = np.random.default_rng(2)
        g = rng.normal(70.0, 12.0, size=2000)
        faithful = g + rng.normal(0.0, 6.0, size=2000)
        shrunk = 70.0 + 0.7 * (g - 70.0) + rng.normal(0.0, 1.0, size=2000)
        self.assertLess(np.abs(shrunk - g).mean(), np.abs(faithful - g).mean())  # better MAE
        d_shrunk = mf._discrimination(list(zip(g, shrunk)))
        d_faithful = mf._discrimination(list(zip(g, faithful)))
        self.assertAlmostEqual(d_shrunk["slope"], 0.7, places=1)
        self.assertAlmostEqual(d_faithful["slope"], 1.0, places=1)
        self.assertLess(d_shrunk["spread_ratio"], d_faithful["spread_ratio"])

    def test_r_is_immune_to_offset_and_scale(self):
        rng = np.random.default_rng(3)
        g = rng.normal(70.0, 12.0, size=200)
        self.assertAlmostEqual(mf._discrimination(list(zip(g, 3.0 * g - 40.0)))["r"], 1.0, places=6)

    def test_too_few_or_degenerate_pairs_return_nan(self):
        self.assertEqual(mf._discrimination([(1.0, 2.0)])["n_rep"], 0)
        d = mf._discrimination([(5.0, 1.0), (5.0, 2.0), (5.0, 3.0)])   # zero-variance truth
        self.assertTrue(np.isnan(d["slope"]))


class TestShrinkageSweep(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9)
        g = rng.normal(70.0, 12.0, size=800)
        self.pairs = list(zip(g, g + rng.normal(0.0, 8.0, size=800)))

    def test_lambda_sets_the_slope(self):
        rows = mf.shrinkage_sweep(self.pairs, lambdas=(0.5, 1.0))
        self.assertAlmostEqual(rows[0]["slope"] / rows[1]["slope"], 0.5, places=6)

    def test_shrinking_buys_mae_while_costing_the_verdict(self):
        """The claim, made falsifiable: MAE bottoms out below lam=1, flip does not."""
        rows = mf.shrinkage_sweep(self.pairs)
        by_lam = {r["lam"]: r for r in rows}
        best_mae_lam = min(rows, key=lambda r: r["mae"])["lam"]
        self.assertLess(best_mae_lam, 1.0)                       # MAE prefers a shrunk reading
        self.assertLess(by_lam[1.0]["swept_flip"], by_lam[0.5]["swept_flip"])  # verdict does not

    def test_bias_is_held_fixed_so_only_spread_varies(self):
        rows = mf.shrinkage_sweep(self.pairs, lambdas=(0.4, 1.5))
        self.assertEqual(len(rows), 2)   # re-centring never drops rows
        self.assertGreater(rows[1]["slope"], rows[0]["slope"])

    def test_too_few_pairs_returns_empty(self):
        self.assertEqual(mf.shrinkage_sweep([(1.0, 2.0)]), [])


class TestMaskToStride(unittest.TestCase):
    """Experiment 0.5 -- putting an every-frame model on a subsampled model's grid."""

    def test_keeps_exactly_the_sparse_grid(self):
        pts = np.ones((45, 25, 3))
        out = dec.mask_to_stride(pts, 15)
        finite_rows = np.where(np.isfinite(out).all(axis=(1, 2)))[0]
        np.testing.assert_array_equal(finite_rows, [0, 15, 30])

    def test_stride_one_is_identity_and_does_not_copy_semantics(self):
        pts = np.ones((10, 25, 3))
        np.testing.assert_array_equal(dec.mask_to_stride(pts, 1), pts)

    def test_does_not_mutate_the_input(self):
        pts = np.ones((30, 25, 3))
        dec.mask_to_stride(pts, 15)
        self.assertTrue(np.isfinite(pts).all())

    def test_masked_frames_drop_out_of_the_rep_extreme(self):
        """The point of the control: nanmin over fewer frames can only be less extreme."""
        from src.fit3d.biomech import WORLD3D, rep_summary

        rng = np.random.default_rng(3)
        pts = rng.normal(size=(60, 25, 3))
        full = rep_summary(pts, WORLD3D, 0, 60)["knee_angle"]
        sparse = rep_summary(dec.mask_to_stride(pts, 15), WORLD3D, 0, 60)["knee_angle"]
        self.assertTrue(np.isfinite(full) and np.isfinite(sparse))
        self.assertGreaterEqual(sparse, full - 1e-9)  # min over a subset is >= min over all


if __name__ == "__main__":
    unittest.main()
