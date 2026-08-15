from __future__ import annotations

import unittest

from src.rehab24.videomae_framing_report import (
    PRACTICAL_EFFECT,
    exact_wilcoxon,
    holm_correct,
    paired_comparison,
    parse_arm,
    parse_pair,
    seed_averaged_accuracy,
    seed_averaged_strata,
    verdict,
)


def fold(subject: str, accuracy: float, n_test: int = 200, cameras: dict | None = None) -> dict:
    return {
        "test_subject": subject,
        "n_test": n_test,
        "balanced_accuracy": accuracy,
        "macro_f1": accuracy,
        "by_camera": cameras or {},
        "by_exercise": {},
    }


class SeedAveragingTests(unittest.TestCase):
    def test_averages_a_subject_over_seeds_before_anything_else(self) -> None:
        """§7.2 step 2. Averaging here rather than after the delta is what keeps the
        seed from entering the test as an independent observation."""
        averaged = seed_averaged_accuracy({42: [fold("1", 0.6)], 7: [fold("1", 0.8)], 1234: [fold("1", 0.7)]})
        self.assertAlmostEqual(averaged["1"], 0.7)

    def test_yields_one_value_per_subject_not_per_subject_seed(self) -> None:
        """9 subjects x 3 seeds is 9 observations, not 27. Treating it as 27 is
        pseudo-replication and would roughly halve every p-value in the report."""
        folds_by_seed = {seed: [fold(str(s), 0.6) for s in range(1, 10)] for seed in (42, 7, 1234)}
        self.assertEqual(len(seed_averaged_accuracy(folds_by_seed)), 9)

    def test_drops_the_undersized_subject_by_default(self) -> None:
        """P10 has 16 samples; balanced accuracy on it is not a usable fold."""
        averaged = seed_averaged_accuracy({42: [fold("1", 0.6), fold("10", 0.9, n_test=16)]})
        self.assertEqual(sorted(averaged), ["1"])

    def test_keeps_the_undersized_subject_for_the_sensitivity_analysis(self) -> None:
        averaged = seed_averaged_accuracy({42: [fold("1", 0.6), fold("10", 0.9, n_test=16)]}, drop_p10=False)
        self.assertEqual(sorted(averaged, key=int), ["1", "10"])

    def test_orders_subjects_numerically_not_lexically(self) -> None:
        folds_by_seed = {42: [fold(str(s), 0.6) for s in (1, 2, 10, 3)]}
        self.assertEqual(list(seed_averaged_accuracy(folds_by_seed, drop_p10=False)), ["1", "2", "3", "10"])

    def test_strata_are_seed_averaged_the_same_way(self) -> None:
        cameras = {"cam17": {"n": 100, "balanced_accuracy": 0.6}}
        other = {"cam17": {"n": 100, "balanced_accuracy": 0.8}}
        strata = seed_averaged_strata(
            {42: [fold("1", 0.6, cameras=cameras)], 7: [fold("1", 0.7, cameras=other)]}, "by_camera"
        )
        self.assertAlmostEqual(strata["cam17"]["1"], 0.7)


class PairedComparisonTests(unittest.TestCase):
    def accuracies(self, values: list[float]) -> dict[str, float]:
        return {str(index + 1): value for index, value in enumerate(values)}

    def test_reports_the_per_subject_deltas_and_their_direction(self) -> None:
        result = paired_comparison(self.accuracies([0.7, 0.6, 0.8]), self.accuracies([0.6, 0.6, 0.7]))
        self.assertEqual(result["n_subjects"], 3)
        self.assertEqual(result["n_positive"], 2)
        self.assertTrue(result["majority_positive"])
        self.assertAlmostEqual(result["delta"]["mean"], (0.1 + 0.0 + 0.1) / 3)

    def test_reports_the_range_because_n_is_nine(self) -> None:
        result = paired_comparison(self.accuracies([0.9, 0.5]), self.accuracies([0.6, 0.6]))
        self.assertAlmostEqual(result["delta_range"][0], -0.1)
        self.assertAlmostEqual(result["delta_range"][1], 0.3)

    def test_refuses_to_compare_arms_evaluated_on_different_subjects(self) -> None:
        with self.assertRaises(ValueError):
            paired_comparison({"1": 0.6, "2": 0.7}, {"1": 0.6})

    def test_a_uniform_delta_is_reported_without_a_test(self) -> None:
        """Wilcoxon on all-zero differences has no defined statistic."""
        result = paired_comparison(self.accuracies([0.6, 0.6]), self.accuracies([0.6, 0.6]))
        self.assertIsNone(result["wilcoxon"])


class ExactWilcoxonTests(unittest.TestCase):
    def test_uses_the_exact_method_not_the_normal_approximation(self) -> None:
        """n=9 is far below where the normal approximation is valid."""
        stats = exact_wilcoxon([0.05, 0.04, 0.03, 0.02, 0.01, 0.06, 0.07, 0.08, 0.09])
        if stats is None:
            self.skipTest("scipy is unavailable")
        self.assertEqual(stats["method"], "exact")
        self.assertAlmostEqual(stats["p_value"], 2 / 512, places=6)

    def test_returns_none_when_every_delta_is_zero(self) -> None:
        self.assertIsNone(exact_wilcoxon([0.0, 0.0, 0.0]))

    def test_returns_none_on_an_empty_input(self) -> None:
        self.assertIsNone(exact_wilcoxon([]))


class VerdictTests(unittest.TestCase):
    def test_a_large_consistent_gain_is_called_practical(self) -> None:
        self.assertEqual(verdict(0.05, 0.01), "practical_gain")

    def test_a_large_consistent_loss_is_called_practical(self) -> None:
        self.assertEqual(verdict(-0.05, 0.01), "practical_loss")

    def test_a_non_significant_result_is_never_called_no_difference(self) -> None:
        """§7.2: at n=9 this design cannot distinguish a null from an effect it is too
        small to see, so p>=0.05 is undetermined -- not equivalence."""
        self.assertIn("undetermined", verdict(0.04, 0.4))
        self.assertNotIn("no_difference", verdict(0.04, 0.4))

    def test_a_tiny_delta_is_a_small_point_estimate_not_equivalence(self) -> None:
        self.assertEqual(verdict(0.005, 0.9), "practically_small_point_estimate")

    def test_the_band_is_the_pre_registered_one(self) -> None:
        self.assertEqual(PRACTICAL_EFFECT, 0.02)
        self.assertEqual(verdict(0.019, None), "practically_small_point_estimate")
        self.assertIn("undetermined", verdict(0.021, None))


class HolmTests(unittest.TestCase):
    def test_corrects_the_smallest_p_value_by_the_full_family_size(self) -> None:
        corrected = holm_correct({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(corrected["a"]["holm"], 0.03)

    def test_is_monotone_so_a_later_test_is_never_more_significant(self) -> None:
        corrected = holm_correct({"a": 0.01, "b": 0.02, "c": 0.03})
        values = [corrected[name]["holm"] for name in ("a", "b", "c")]
        self.assertEqual(values, sorted(values))

    def test_keeps_the_raw_value_alongside_the_corrected_one(self) -> None:
        corrected = holm_correct({"a": 0.01})
        self.assertEqual(corrected["a"]["raw"], 0.01)

    def test_caps_at_one(self) -> None:
        corrected = holm_correct({"a": 0.6, "b": 0.7})
        self.assertLessEqual(corrected["b"]["holm"], 1.0)


class ArgumentParsingTests(unittest.TestCase):
    def test_parses_an_arm_spec(self) -> None:
        name, path = parse_arm("full_frame_letterbox=data/x/y")
        self.assertEqual(name, "full_frame_letterbox")
        self.assertEqual(path.as_posix(), "data/x/y")

    def test_parses_a_comparison_spec(self) -> None:
        self.assertEqual(parse_pair("letterbox:full_frame"), ("letterbox", "full_frame"))

    def test_rejects_a_malformed_spec(self) -> None:
        with self.assertRaises(ValueError):
            parse_arm("no-equals-sign")
        with self.assertRaises(ValueError):
            parse_pair("no-colon")


if __name__ == "__main__":
    unittest.main()
