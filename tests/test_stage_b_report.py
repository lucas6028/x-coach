from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.video.stage_b_report import (
    ArmRuns,
    denominator_gate,
    format_summary_table,
    load_late_fusion_arm,
    load_single_arm,
    read_selected_threshold,
    retention_conditions,
)


def write_predictions(
    path: Path,
    splits: dict[str, tuple[list[str], list[float], list[int]]],
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label_mode",
                "split",
                "video_id",
                "label",
                "probability",
                "fixed_0_5_prediction",
                "selected_threshold_prediction",
                "selected_threshold",
            ],
        )
        writer.writeheader()
        for split, (ids, probabilities, labels) in splits.items():
            for video_id, probability, label in zip(ids, probabilities, labels):
                writer.writerow(
                    {
                        "label_mode": "combined",
                        "split": split,
                        "video_id": video_id,
                        "label": label,
                        "probability": f"{probability:.8f}",
                        "fixed_0_5_prediction": int(probability >= 0.5),
                        "selected_threshold_prediction": int(probability >= threshold),
                        "selected_threshold": f"{threshold:.8f}",
                    }
                )


def arm_from(probabilities: list[list[float]], labels: list[int], thresholds: list[float]) -> ArmRuns:
    return ArmRuns(
        name="arm",
        seeds=list(range(1, len(probabilities) + 1)),
        probabilities=[np.asarray(values, dtype=np.float64) for values in probabilities],
        thresholds=thresholds,
        labels=np.asarray(labels, dtype=np.int32),
        video_ids=[f"v{index}" for index in range(len(labels))],
    )


class LoadArmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.ids = ["a", "b", "c", "d"]
        self.labels = [1, 0, 1, 0]

    def write_seed(self, seed: int, test_probabilities: list[float], threshold: float, ids: list[str] | None = None) -> None:
        write_predictions(
            self.dir / f"combined_seed{seed}_predictions.csv",
            {
                "val": (["x", "y"], [0.6, 0.3], [1, 0]),
                "test": (ids or self.ids, test_probabilities, self.labels),
            },
            threshold=threshold,
        )

    def test_each_seed_keeps_its_own_validation_selected_threshold(self) -> None:
        self.write_seed(1, [0.9, 0.2, 0.8, 0.1], threshold=0.4)
        self.write_seed(2, [0.7, 0.3, 0.6, 0.2], threshold=0.55)

        arm = load_single_arm("pose", self.dir, "combined", [1, 2])

        self.assertEqual(arm.thresholds, [0.4, 0.55])
        self.assertEqual(len(arm.metrics), 2)
        self.assertEqual(arm.metrics[0]["balanced_accuracy"], 1.0)

    def test_a_seed_scored_on_a_different_test_set_is_refused(self) -> None:
        """Seed means and the paired bootstrap both assume one fixed test set."""
        self.write_seed(1, [0.9, 0.2, 0.8, 0.1], threshold=0.5)
        self.write_seed(2, [0.9, 0.2, 0.8, 0.1], threshold=0.5, ids=["a", "b", "c", "zz"])

        with self.assertRaises(ValueError):
            load_single_arm("pose", self.dir, "combined", [1, 2])

    def test_seeds_written_in_a_different_video_order_are_realigned(self) -> None:
        self.write_seed(1, [0.9, 0.2, 0.8, 0.1], threshold=0.5)
        write_predictions(
            self.dir / "combined_seed2_predictions.csv",
            {
                "val": (["x", "y"], [0.6, 0.3], [1, 0]),
                "test": (["d", "c", "b", "a"], [0.1, 0.8, 0.2, 0.9], [0, 1, 0, 1]),
            },
            threshold=0.5,
        )

        arm = load_single_arm("pose", self.dir, "combined", [1, 2])

        np.testing.assert_allclose(arm.probabilities[0], arm.probabilities[1])

    def test_a_missing_seed_is_an_error_not_a_shorter_mean(self) -> None:
        self.write_seed(1, [0.9, 0.2, 0.8, 0.1], threshold=0.5)
        with self.assertRaises(FileNotFoundError):
            load_single_arm("pose", self.dir, "combined", [1, 2])

    def test_threshold_column_must_be_constant_within_a_run(self) -> None:
        path = self.dir / "combined_seed1_predictions.csv"
        self.write_seed(1, [0.9, 0.2, 0.8, 0.1], threshold=0.5)
        text = path.read_text(encoding="utf-8").replace("0.50000000\n", "0.90000000\n", 1)
        path.write_text(text, encoding="utf-8")
        with self.assertRaises(ValueError):
            read_selected_threshold(path)


class LateFusionArmTests(unittest.TestCase):
    def test_fusion_arm_is_built_seed_for_seed(self) -> None:
        with TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            for seed in (1, 2):
                write_predictions(
                    first / f"combined_seed{seed}_predictions.csv",
                    {
                        "val": (["x", "y", "z", "w"], [0.8, 0.2, 0.7, 0.3], [1, 0, 1, 0]),
                        "test": (["a", "b"], [0.9, 0.1], [1, 0]),
                    },
                    threshold=0.5,
                )
                write_predictions(
                    second / f"combined_seed{seed}_predictions.csv",
                    {
                        "val": (["x", "y", "z", "w"], [0.6, 0.4, 0.55, 0.45], [1, 0, 1, 0]),
                        "test": (["a", "b"], [0.6, 0.4], [1, 0]),
                    },
                    threshold=0.5,
                )

            arm = load_late_fusion_arm("late", first, second, "combined", [1, 2])

            self.assertEqual(len(arm.probabilities), 2)
            self.assertEqual(arm.video_ids, ["a", "b"])
            self.assertEqual(arm.metrics[0]["balanced_accuracy"], 1.0)


class DenominatorGateTests(unittest.TestCase):
    def test_gate_passes_inside_the_published_band(self) -> None:
        labels = [1] * 50 + [0] * 50
        # 0.64 balanced accuracy: 64 of 100 correct, symmetric.
        probabilities = [0.9] * 32 + [0.1] * 18 + [0.1] * 32 + [0.9] * 18
        gate = denominator_gate(arm_from([probabilities], labels, [0.5]))
        self.assertAlmostEqual(gate["re_derived"], 0.64, places=6)
        self.assertTrue(gate["passed"])

    def test_gate_fails_outside_the_band(self) -> None:
        labels = [1] * 50 + [0] * 50
        probabilities = [0.9] * 25 + [0.1] * 25 + [0.1] * 25 + [0.9] * 25
        gate = denominator_gate(arm_from([probabilities], labels, [0.5]))
        self.assertAlmostEqual(gate["re_derived"], 0.5, places=6)
        self.assertFalse(gate["passed"])


class RetentionConditionTests(unittest.TestCase):
    def make_pair(self, candidate_correct: int) -> tuple[ArmRuns, ArmRuns]:
        labels = [1] * 50 + [0] * 50
        baseline = [0.9] * 30 + [0.1] * 20 + [0.1] * 30 + [0.9] * 20
        candidate = (
            [0.9] * candidate_correct
            + [0.1] * (50 - candidate_correct)
            + [0.1] * candidate_correct
            + [0.9] * (50 - candidate_correct)
        )
        return arm_from([baseline], labels, [0.5]), arm_from([candidate], labels, [0.5])

    def test_a_delta_below_the_plan_threshold_fails_condition_one(self) -> None:
        baseline, candidate = self.make_pair(31)  # +0.02 exactly on 50/50... just under
        bootstrap = {"ci_low": 0.001, "ci_high": 0.05, "observed_delta": 0.02}
        conditions = retention_conditions(baseline, candidate, bootstrap)
        self.assertAlmostEqual(conditions["delta_at_least_0.02"]["value"], 0.02, places=6)
        self.assertTrue(conditions["delta_at_least_0.02"]["passed"])

    def test_a_ci_touching_zero_fails_condition_two(self) -> None:
        baseline, candidate = self.make_pair(40)
        conditions = retention_conditions(baseline, candidate, {"ci_low": -0.001, "ci_high": 0.1})
        self.assertFalse(conditions["ci_lower_bound_above_zero"]["passed"])

    def test_a_recall_collapse_fails_the_guardrail_even_when_balanced_accuracy_rises(self) -> None:
        """The guardrail exists because balanced accuracy can rise while one side
        of the confusion matrix falls apart."""
        labels = [1] * 50 + [0] * 50
        baseline = arm_from([[0.9] * 40 + [0.1] * 10 + [0.1] * 30 + [0.9] * 20], labels, [0.5])
        candidate = arm_from([[0.9] * 30 + [0.1] * 20 + [0.1] * 45 + [0.9] * 5], labels, [0.5])

        conditions = retention_conditions(baseline, candidate, {"ci_low": 0.01, "ci_high": 0.1})

        self.assertGreater(conditions["delta_at_least_0.02"]["value"], 0.0)
        self.assertFalse(conditions["no_guardrail_drop_over_0.03"]["passed"])
        self.assertLess(conditions["no_guardrail_drop_over_0.03"]["detail"]["recall"], -0.03)

    def test_seed_consistency_counts_positive_seeds(self) -> None:
        labels = [1] * 20 + [0] * 20
        good = [0.9] * 20 + [0.1] * 20
        bad = [0.1] * 20 + [0.9] * 20
        baseline = ArmRuns("b", [1, 2], [np.asarray(good), np.asarray(bad)], [0.5, 0.5], np.asarray(labels), ["v"] * 40)
        candidate = ArmRuns("c", [1, 2], [np.asarray(bad), np.asarray(good)], [0.5, 0.5], np.asarray(labels), ["v"] * 40)

        conditions = retention_conditions(baseline, candidate, {"ci_low": 0.0, "ci_high": 0.0})

        self.assertEqual(conditions["consistent_across_seeds"]["value"], "1/2")
        self.assertFalse(conditions["consistent_across_seeds"]["passed"])


class FormatTests(unittest.TestCase):
    def test_table_lists_one_row_per_arm(self) -> None:
        labels = [1, 0, 1, 0]
        arm = arm_from([[0.9, 0.1, 0.8, 0.2]], labels, [0.5])
        table = format_summary_table([arm])
        self.assertIn("arm", table)
        self.assertIn("1.000", table)


if __name__ == "__main__":
    unittest.main()
