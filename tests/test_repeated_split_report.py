from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.video.repeated_split_report import (
    RepeatedArm,
    align_arms,
    denominator_gate,
    format_arm_table,
    load_repeated_arm,
    paired_bootstrap_delta,
    read_fold_decisions,
)


def write_fold_csv(path: Path, rows: list[tuple[str, int, int]], threshold: float = 0.4) -> None:
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
        # A validation row the pooling must ignore.
        writer.writerow(
            {
                "label_mode": "combined",
                "split": "val",
                "video_id": "ignored",
                "label": 1,
                "probability": "0.9",
                "fixed_0_5_prediction": 1,
                "selected_threshold_prediction": 1,
                "selected_threshold": f"{threshold}",
            }
        )
        for video_id, decision, label in rows:
            writer.writerow(
                {
                    "label_mode": "combined",
                    "split": "test",
                    "video_id": video_id,
                    "label": label,
                    "probability": "0.5",
                    "fixed_0_5_prediction": decision,
                    "selected_threshold_prediction": decision,
                    "selected_threshold": f"{threshold}",
                }
            )


def arm_from(name: str, decisions: dict[int, list[int]], labels: list[int]) -> RepeatedArm:
    return RepeatedArm(
        name=name,
        repeats=sorted(decisions),
        decisions={repeat: np.asarray(values, dtype=np.int32) for repeat, values in decisions.items()},
        labels=np.asarray(labels, dtype=np.int32),
        video_ids=[f"v{index}" for index in range(len(labels))],
    )


class ReadFoldTests(unittest.TestCase):
    def test_only_test_rows_are_read(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "combined_r1f0_predictions.csv"
            write_fold_csv(path, [("a", 1, 1), ("b", 0, 0)])
            video_ids, decisions, labels = read_fold_decisions(path)
            self.assertEqual(video_ids, ["a", "b"])
            self.assertEqual(decisions, [1, 0])
            self.assertEqual(labels, [1, 0])

    def test_a_fold_with_no_test_rows_is_an_error(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "combined_r1f0_predictions.csv"
            write_fold_csv(path, [])
            with self.assertRaises(ValueError):
                read_fold_decisions(path)


class LoadRepeatedArmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def write_repeat(self, repeat: int, split_a: list[tuple[str, int, int]], split_b: list[tuple[str, int, int]]) -> None:
        write_fold_csv(self.dir / f"combined_r{repeat}f0_predictions.csv", split_a)
        write_fold_csv(self.dir / f"combined_r{repeat}f1_predictions.csv", split_b)

    def test_folds_pool_into_one_out_of_fold_vector_per_repeat(self) -> None:
        self.write_repeat(1, [("a", 1, 1), ("b", 0, 0)], [("c", 1, 1), ("d", 0, 1)])
        self.write_repeat(2, [("c", 1, 1), ("a", 0, 1)], [("b", 0, 0), ("d", 1, 1)])

        arm = load_repeated_arm("pose", self.dir, "combined", {1: ["r1f0", "r1f1"], 2: ["r2f0", "r2f1"]})

        self.assertEqual(arm.video_ids, ["a", "b", "c", "d"])
        self.assertEqual(arm.repeats, [1, 2])
        np.testing.assert_array_equal(arm.decisions[1], [1, 0, 1, 0])
        np.testing.assert_array_equal(arm.decisions[2], [0, 0, 1, 1])

    def test_a_video_scored_by_two_folds_of_one_repeat_is_refused(self) -> None:
        """The out-of-fold vector must hold exactly one decision per video, or a
        video silently votes twice in the pooled score."""
        self.write_repeat(1, [("a", 1, 1), ("b", 0, 0)], [("a", 0, 1), ("c", 1, 1)])
        with self.assertRaises(ValueError) as ctx:
            load_repeated_arm("pose", self.dir, "combined", {1: ["r1f0", "r1f1"]})
        self.assertIn("more than one fold", str(ctx.exception))

    def test_a_repeat_covering_a_different_corpus_is_refused(self) -> None:
        self.write_repeat(1, [("a", 1, 1)], [("b", 0, 0)])
        self.write_repeat(2, [("a", 1, 1)], [("zz", 0, 0)])
        with self.assertRaises(ValueError) as ctx:
            load_repeated_arm("pose", self.dir, "combined", {1: ["r1f0", "r1f1"], 2: ["r2f0", "r2f1"]})
        self.assertIn("different video set", str(ctx.exception))

    def test_a_missing_fold_is_an_error_not_a_shorter_vector(self) -> None:
        self.write_repeat(1, [("a", 1, 1)], [("b", 0, 0)])
        with self.assertRaises(FileNotFoundError):
            load_repeated_arm("pose", self.dir, "combined", {1: ["r1f0", "r1f1", "r1f2"]})


class MetricTests(unittest.TestCase):
    def test_decisions_are_scored_with_the_shared_metric_implementation(self) -> None:
        arm = arm_from("a", {1: [1, 1, 0, 0], 2: [1, 0, 0, 0]}, [1, 1, 0, 0])
        self.assertAlmostEqual(arm.metrics[1]["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(arm.metrics[2]["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(arm.summary()["balanced_accuracy"]["mean"], 0.875)

    def test_a_subset_keeps_the_decisions_the_full_corpus_produced(self) -> None:
        """Restricting after scoring is deliberate: each fold trained and chose its
        threshold on the whole corpus, which is what production would do."""
        arm = arm_from("a", {1: [1, 1, 0, 0]}, [1, 1, 0, 0])
        subset = arm.subset({"v0", "v2"})
        self.assertEqual(subset.video_ids, ["v0", "v2"])
        np.testing.assert_array_equal(subset.decisions[1], [1, 0])

    def test_an_empty_subset_is_refused(self) -> None:
        arm = arm_from("a", {1: [1, 0]}, [1, 0])
        with self.assertRaises(ValueError):
            arm.subset({"nope"})

    def test_table_lists_one_row_per_arm(self) -> None:
        table = format_arm_table([arm_from("pose", {1: [1, 0]}, [1, 0])])
        self.assertIn("pose", table)


class BootstrapTests(unittest.TestCase):
    def test_a_clear_improvement_gives_an_interval_above_zero(self) -> None:
        labels = [1] * 40 + [0] * 40
        weak = [1] * 20 + [0] * 20 + [0] * 40
        strong = [1] * 40 + [0] * 40
        baseline = arm_from("weak", {1: weak, 2: weak}, labels)
        candidate = arm_from("strong", {1: strong, 2: strong}, labels)

        result = paired_bootstrap_delta(baseline, candidate, resamples=400)

        self.assertGreater(result["observed_delta"], 0.2)
        self.assertGreater(result["ci_low"], 0.0)

    def test_two_identical_arms_give_a_zero_delta_and_a_zero_width_interval(self) -> None:
        labels = [1] * 10 + [0] * 10
        decisions = [1] * 10 + [0] * 10
        arm = arm_from("a", {1: decisions}, labels)
        other = arm_from("b", {1: decisions}, labels)

        result = paired_bootstrap_delta(arm, other, resamples=200)

        self.assertEqual(result["observed_delta"], 0.0)
        self.assertEqual(result["half_width"], 0.0)

    def test_one_draw_is_applied_to_every_repeat(self) -> None:
        """Letting each repeat draw independently would treat the same videos as
        five times as many and shrink the interval by about sqrt(5)."""
        labels = [1] * 30 + [0] * 30
        noisy = {r: [1] * 30 + [0] * 30 for r in (1, 2, 3, 4, 5)}
        weak = {r: [1] * 15 + [0] * 15 + [0] * 30 for r in (1, 2, 3, 4, 5)}
        five = paired_bootstrap_delta(arm_from("w", weak, labels), arm_from("n", noisy, labels), resamples=600)
        one = paired_bootstrap_delta(
            arm_from("w", {1: weak[1]}, labels), arm_from("n", {1: noisy[1]}, labels), resamples=600
        )
        # Identical repeats carry no extra information, so the interval must not narrow.
        self.assertAlmostEqual(five["half_width"], one["half_width"], places=2)

    def test_arms_scored_on_different_corpora_are_refused(self) -> None:
        left = arm_from("a", {1: [1, 0]}, [1, 0])
        right = arm_from("b", {1: [1, 0, 1]}, [1, 0, 1])
        with self.assertRaises(ValueError):
            align_arms(left, right)


class DenominatorGateTests(unittest.TestCase):
    def test_a_comparable_re_derivation_passes(self) -> None:
        arm = arm_from("pose", {1: [1, 1, 0, 0]}, [1, 1, 0, 0])  # 1.0
        gate = denominator_gate(arm, fixed_split_value=0.98, tolerance=0.03)
        self.assertTrue(gate["passed"])

    def test_a_materially_higher_score_is_flagged_as_leakage(self) -> None:
        """Each fold trains on fewer videos than the fixed split, so a higher score
        cannot be explained by more training data."""
        arm = arm_from("pose", {1: [1, 1, 0, 0]}, [1, 1, 0, 0])  # 1.0
        gate = denominator_gate(arm, fixed_split_value=0.650, tolerance=0.03)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["verdict"], "suspect leakage")

    def test_a_materially_lower_score_is_reported_as_a_harder_resampling(self) -> None:
        arm = arm_from("pose", {1: [1, 0, 0, 1]}, [1, 1, 0, 0])  # 0.0
        gate = denominator_gate(arm, fixed_split_value=0.650, tolerance=0.03)
        self.assertFalse(gate["passed"])
        self.assertIn("harder", str(gate["verdict"]))


if __name__ == "__main__":
    unittest.main()
