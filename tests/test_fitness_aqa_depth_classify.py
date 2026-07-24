"""The classification harness must be honest: correct metrics, video-level bootstrap."""

import unittest

import numpy as np

from src.fitness_aqa import depth_classify as dc


def separable_split(n=200, d=4, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name, size in (("train", n), ("val", n // 2), ("test", n // 2)):
        y = (rng.random(size) < 0.5).astype(np.float64)
        x = rng.normal(size=(size, d)) * 0.3 + y[:, None] * 3.0
        out[name] = (x, y)
    return out


class TestMetrics(unittest.TestCase):
    def test_balanced_accuracy_ignores_class_imbalance(self):
        y = np.array([1.0] * 90 + [0.0] * 10)
        all_positive = np.ones(100)
        self.assertAlmostEqual(float((all_positive == y).mean()), 0.9)
        self.assertAlmostEqual(dc.balanced_accuracy(y, all_positive), 0.5)

    def test_roc_auc_known_values(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(dc.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertAlmostEqual(dc.roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)
        self.assertAlmostEqual(dc.roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])), 0.5)

    def test_select_threshold_finds_the_perfect_cut(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        t = dc.select_threshold(y, np.array([0.1, 0.2, 0.7, 0.9]))
        self.assertAlmostEqual(dc.balanced_accuracy(y, (np.array([0.1, 0.2, 0.7, 0.9]) >= t).astype(float)), 1.0)

    def test_binary_metrics_counts(self):
        y = np.array([1.0, 1.0, 0.0, 0.0])
        m = dc.binary_metrics(y, np.array([0.9, 0.1, 0.8, 0.2]), 0.5)
        self.assertEqual((m["tp"], m["fp"], m["tn"], m["fn"]), (1.0, 1.0, 1.0, 1.0))
        self.assertAlmostEqual(m["balanced_accuracy"], 0.5)


class TestLogistic(unittest.TestCase):
    def test_learns_a_separable_problem(self):
        data = separable_split()
        x = {k: v[0] for k, v in data.items()}
        y = {k: v[1] for k, v in data.items()}
        g = {k: np.arange(len(v[1])) for k, v in data.items()}
        res = dc.run_arm("sep", x, y, g)
        self.assertGreater(res.metrics["balanced_accuracy"], 0.95)

    def test_pure_noise_features_stay_near_chance(self):
        rng = np.random.default_rng(1)
        x = {k: rng.normal(size=(n, 4)) for k, n in (("train", 400), ("val", 200), ("test", 200))}
        y = {k: (rng.random(v.shape[0]) < 0.5).astype(float) for k, v in x.items()}
        g = {k: np.arange(v.shape[0]) for k, v in x.items()}
        res = dc.run_arm("noise", x, y, g)
        self.assertLess(abs(res.metrics["balanced_accuracy"] - 0.5), 0.12)

    def test_nan_cells_are_imputed_not_propagated(self):
        data = separable_split()
        x = {k: v[0].copy() for k, v in data.items()}
        x["test"][0, 0] = np.nan
        y = {k: v[1] for k, v in data.items()}
        g = {k: np.arange(len(v[1])) for k, v in data.items()}
        res = dc.run_arm("nan", x, y, g)
        self.assertTrue(np.isfinite(res.test_scores).all())

    def test_standardizer_uses_train_statistics_only(self):
        s = dc.Standardizer.fit(np.array([[0.0], [2.0]]))
        np.testing.assert_allclose(s.transform(np.array([[1.0]])), [[0.0]])
        np.testing.assert_allclose(s.transform(np.array([[2.0]])), [[1.0]])

    def test_constant_column_does_not_divide_by_zero(self):
        s = dc.Standardizer.fit(np.full((5, 1), 3.0))
        self.assertTrue(np.isfinite(s.transform(np.full((2, 1), 3.0))).all())


class TestClusterBootstrap(unittest.TestCase):
    def test_resamples_whole_videos(self):
        groups = np.array(["a", "a", "a", "b", "b", "c"])
        reps = dc.cluster_bootstrap_indices(groups, n_boot=50, seed=0)
        for idx in reps:
            drawn = groups[idx]
            # every drawn video contributes all of its rows, so counts are multiples
            for name, size in (("a", 3), ("b", 2), ("c", 1)):
                self.assertEqual(int((drawn == name).sum()) % size, 0)
            self.assertEqual(len(np.unique(idx[np.argsort(idx)])) <= len(groups), True)

    def test_bootstrap_is_seed_reproducible(self):
        groups = np.array(["a", "a", "b", "c"])
        a = dc.cluster_bootstrap_indices(groups, 10, seed=7)
        b = dc.cluster_bootstrap_indices(groups, 10, seed=7)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_paired_delta_detects_a_real_gap(self):
        n = 200
        rng = np.random.default_rng(3)
        y = (rng.random(n) < 0.5).astype(float)
        groups = np.repeat(np.arange(n // 4), 4).astype(str)
        good = dc.ArmResult("good", {"threshold": 0.5, "balanced_accuracy": 0.0},
                            np.where(y == 1, 0.9, 0.1), y, groups)
        bad = dc.ArmResult("bad", {"threshold": 0.5, "balanced_accuracy": 0.0},
                           rng.random(n), y, groups)
        good.metrics["balanced_accuracy"] = dc.binary_metrics(y, good.test_scores, 0.5)["balanced_accuracy"]
        bad.metrics["balanced_accuracy"] = dc.binary_metrics(y, bad.test_scores, 0.5)["balanced_accuracy"]
        reps = dc.cluster_bootstrap_indices(groups, 500, seed=0)
        d = dc.paired_delta(good, bad, reps)
        self.assertGreater(d["delta"], 0.3)
        self.assertGreater(d["ci_low"], 0.0)
        self.assertLess(d["p_two_sided"], 0.05)

    def test_paired_delta_of_identical_arms_is_zero(self):
        y = np.array([1.0, 0.0, 1.0, 0.0])
        groups = np.array(["v1", "v1", "v2", "v2"])
        scores = np.array([0.9, 0.2, 0.7, 0.1])
        arm = dc.ArmResult("a", {"threshold": 0.5, "balanced_accuracy": 1.0}, scores, y, groups)
        reps = dc.cluster_bootstrap_indices(groups, 100, seed=0)
        d = dc.paired_delta(arm, arm, reps)
        self.assertAlmostEqual(d["delta"], 0.0)
        self.assertAlmostEqual(d["ci_low"], 0.0)
        self.assertAlmostEqual(d["ci_high"], 0.0)


if __name__ == "__main__":
    unittest.main()
