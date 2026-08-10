import unittest
from collections import Counter

import numpy as np

from src.rehab24.videomae_stage_a import (
    MIN_STRATUM_SAMPLES,
    arm_summary,
    paired_delta,
    permute_labels_within_subject,
    stratified_metrics,
    stratum_deltas,
)


def fold(subject: str, balanced_accuracy: float, n_test: int = 200, cameras: dict | None = None) -> dict:
    return {
        "test_subject": subject,
        "n_test": n_test,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": balanced_accuracy,
        "by_camera": cameras or {},
        "by_exercise": {},
    }


class PermuteLabelsWithinSubjectTest(unittest.TestCase):
    def setUp(self):
        # Each subject has 5 repetitions, each recorded from cam17 and cam18 --
        # the real REHAB24-6 structure (1072 reps, all with exactly two rows).
        self.subject_samples = {
            "1": [f"a_rep{i}_{cam}" for i in range(5) for cam in ("cam17", "cam18")],
            "2": [f"b_rep{i}_{cam}" for i in range(5) for cam in ("cam17", "cam18")],
        }
        self.metadata = {}
        self.labels = {}
        for prefix, positives in (("a", 2), ("b", 4)):
            for i in range(5):
                for cam in ("cam17", "cam18"):
                    sample_id = f"{prefix}_rep{i}_{cam}"
                    self.metadata[sample_id] = {"repetition": f"{prefix}_rep{i}"}
                    self.labels[sample_id] = int(i < positives)

    def permute(self, seed):
        return permute_labels_within_subject(self.labels, self.subject_samples, self.metadata, seed)

    def test_both_camera_rows_of_a_repetition_get_the_same_label(self):
        """Correctness is a property of the repetition. Permuting per sample would
        give the two views of one rep opposite labels -- a contradiction the real
        training set never contains, which would make the null unfairly hard."""
        for seed in range(5):
            permuted = self.permute(seed)
            for prefix in ("a", "b"):
                for i in range(5):
                    self.assertEqual(
                        permuted[f"{prefix}_rep{i}_cam17"],
                        permuted[f"{prefix}_rep{i}_cam18"],
                        f"rep {prefix}_rep{i} disagrees across cameras at seed {seed}",
                    )

    def test_preserves_each_subjects_positive_rate(self):
        """Each LOSO fold tests one subject, so holding per-subject balance fixed
        keeps the null's chance level identical to the real run's."""
        permuted = self.permute(0)
        for sample_ids in self.subject_samples.values():
            original = Counter(self.labels[sid] for sid in sample_ids)
            shuffled = Counter(permuted[sid] for sid in sample_ids)
            self.assertEqual(original, shuffled)

    def test_actually_reassigns_labels(self):
        self.assertTrue(any(self.permute(seed) != self.labels for seed in range(5)))

    def test_is_deterministic_for_a_seed(self):
        self.assertEqual(self.permute(7), self.permute(7))

    def test_different_seeds_give_different_permutations(self):
        self.assertTrue(any(self.permute(1) != self.permute(seed) for seed in range(2, 8)))

    def test_does_not_move_labels_between_subjects(self):
        permuted = self.permute(3)
        self.assertEqual(set(permuted), set(self.labels))
        for sample_ids in self.subject_samples.values():
            self.assertEqual(
                sum(permuted[sid] for sid in sample_ids),
                sum(self.labels[sid] for sid in sample_ids),
            )

    def test_ignores_samples_without_labels(self):
        subject_samples = {"1": ["a0", "a1", "missing"]}
        labels = {"a0": 1, "a1": 0}
        metadata = {"a0": {"repetition": "r0"}, "a1": {"repetition": "r1"}}
        permuted = permute_labels_within_subject(labels, subject_samples, metadata, seed=0)
        self.assertNotIn("missing", permuted)

    def test_falls_back_to_sample_id_when_repetition_is_unknown(self):
        subject_samples = {"1": ["a0", "a1"]}
        labels = {"a0": 1, "a1": 0}
        permuted = permute_labels_within_subject(labels, subject_samples, {}, seed=0)
        self.assertEqual(Counter(permuted.values()), Counter(labels.values()))


class StratifiedMetricsTest(unittest.TestCase):
    def setUp(self):
        self.n = 60
        self.ids = [f"s{i}" for i in range(self.n)]
        self.metadata = {sid: {"camera": "cam17" if i % 2 == 0 else "cam18"} for i, sid in enumerate(self.ids)}
        self.labels = np.array([i % 2 == 0 for i in range(self.n)], dtype=int)

    def test_splits_by_stratum(self):
        probs = np.where(self.labels == 1, 0.9, 0.1)
        # every sample in cam17 is positive -> single-class stratum, dropped
        result = stratified_metrics(self.ids, probs, self.labels, 0.5, self.metadata, "camera")
        self.assertEqual(result, {})

    def test_reports_per_stratum_balanced_accuracy(self):
        metadata = {sid: {"camera": "cam17" if i < 30 else "cam18"} for i, sid in enumerate(self.ids)}
        probs = np.where(self.labels == 1, 0.9, 0.1)
        result = stratified_metrics(self.ids, probs, self.labels, 0.5, metadata, "camera")
        self.assertEqual(set(result), {"cam17", "cam18"})
        self.assertAlmostEqual(result["cam17"]["balanced_accuracy"], 1.0)
        self.assertEqual(result["cam17"]["n"], 30)

    def test_drops_strata_below_minimum_size(self):
        metadata = {sid: {"camera": "cam17" if i < MIN_STRATUM_SAMPLES - 1 else "cam18"} for i, sid in enumerate(self.ids)}
        probs = np.where(self.labels == 1, 0.9, 0.1)
        result = stratified_metrics(self.ids, probs, self.labels, 0.5, metadata, "camera")
        self.assertNotIn("cam17", result)

    def test_ignores_samples_missing_from_metadata(self):
        probs = np.where(self.labels == 1, 0.9, 0.1)
        result = stratified_metrics(self.ids, probs, self.labels, 0.5, {}, "camera")
        self.assertEqual(result, {})


class PairedDeltaTest(unittest.TestCase):
    def test_computes_per_fold_deltas(self):
        candidate = [fold("1", 0.70), fold("2", 0.60)]
        baseline = [fold("1", 0.65), fold("2", 0.62)]
        result = paired_delta(candidate, baseline)
        self.assertAlmostEqual(result["folds"][0]["delta"], 0.05)
        self.assertAlmostEqual(result["folds"][1]["delta"], -0.02)
        self.assertEqual(result["n_positive"], 1)

    def test_majority_positive_flag(self):
        candidate = [fold(str(i), 0.7) for i in range(3)]
        baseline = [fold(str(i), 0.6) for i in range(3)]
        self.assertTrue(paired_delta(candidate, baseline)["majority_positive"])

        candidate = [fold("0", 0.7), fold("1", 0.5), fold("2", 0.5)]
        baseline = [fold("0", 0.6), fold("1", 0.6), fold("2", 0.6)]
        self.assertFalse(paired_delta(candidate, baseline)["majority_positive"])

    def test_drops_underpowered_p10_fold_by_default(self):
        candidate = [fold("1", 0.70), fold("10", 0.99, n_test=16)]
        baseline = [fold("1", 0.65), fold("10", 0.10, n_test=16)]
        result = paired_delta(candidate, baseline)
        self.assertEqual(result["n_folds"], 1)
        self.assertAlmostEqual(result["delta"]["mean"], 0.05)

    def test_rejects_misaligned_fold_order(self):
        with self.assertRaises(ValueError):
            paired_delta([fold("1", 0.7)], [fold("2", 0.6)])


class StratumDeltasTest(unittest.TestCase):
    def test_averages_delta_within_each_stratum(self):
        candidate = [
            fold("1", 0.7, cameras={"cam17": {"n": 50, "balanced_accuracy": 0.80}, "cam18": {"n": 50, "balanced_accuracy": 0.60}}),
            fold("2", 0.7, cameras={"cam17": {"n": 50, "balanced_accuracy": 0.70}, "cam18": {"n": 50, "balanced_accuracy": 0.50}}),
        ]
        baseline = [
            fold("1", 0.6, cameras={"cam17": {"n": 50, "balanced_accuracy": 0.60}, "cam18": {"n": 50, "balanced_accuracy": 0.60}}),
            fold("2", 0.6, cameras={"cam17": {"n": 50, "balanced_accuracy": 0.60}, "cam18": {"n": 50, "balanced_accuracy": 0.55}}),
        ]
        result = stratum_deltas(candidate, baseline, "by_camera")
        self.assertAlmostEqual(result["cam17"]["mean_delta"], 0.15)
        self.assertEqual(result["cam17"]["n_positive"], 2)
        self.assertAlmostEqual(result["cam18"]["mean_delta"], -0.025)
        self.assertEqual(result["cam18"]["n_positive"], 0)

    def test_skips_strata_absent_from_one_arm(self):
        candidate = [fold("1", 0.7, cameras={"cam17": {"n": 50, "balanced_accuracy": 0.8}})]
        baseline = [fold("1", 0.6, cameras={})]
        self.assertEqual(stratum_deltas(candidate, baseline, "by_camera"), {})


class ArmSummaryTest(unittest.TestCase):
    def test_excludes_p10_from_the_no_p10_summary(self):
        folds = [fold("1", 0.70), fold("2", 0.60), fold("10", 0.20, n_test=16)]
        summary = arm_summary(folds)
        self.assertAlmostEqual(summary["balanced_accuracy_no_p10"]["mean"], 0.65)
        self.assertAlmostEqual(summary["balanced_accuracy_all"]["mean"], 0.50)


if __name__ == "__main__":
    unittest.main()
