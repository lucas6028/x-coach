from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.video.late_fusion import (
    PlattCalibration,
    SplitPredictions,
    align,
    fit_platt,
    fuse_probabilities,
    fuse_run,
    logit,
    paired_bootstrap_delta,
    read_predictions,
)


def write_predictions_csv(path: Path, rows: list[tuple[str, str, int, float]]) -> None:
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
        for split, video_id, label, probability in rows:
            writer.writerow(
                {
                    "label_mode": "combined",
                    "split": split,
                    "video_id": video_id,
                    "label": label,
                    "probability": f"{probability:.8f}",
                    "fixed_0_5_prediction": int(probability >= 0.5),
                    "selected_threshold_prediction": int(probability >= 0.5),
                    "selected_threshold": "0.50000000",
                }
            )


def split_predictions(ids: list[str], probabilities: list[float], labels: list[int]) -> SplitPredictions:
    return SplitPredictions(
        video_ids=ids,
        probabilities=np.asarray(probabilities, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int32),
    )


class ReadPredictionsTests(unittest.TestCase):
    def test_rows_are_grouped_by_split(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.csv"
            write_predictions_csv(
                path,
                [("train", "a", 1, 0.9), ("val", "b", 0, 0.2), ("test", "c", 1, 0.7), ("test", "d", 0, 0.1)],
            )
            loaded = read_predictions(path)

            self.assertEqual(sorted(loaded), ["test", "train", "val"])
            self.assertEqual(loaded["test"].video_ids, ["c", "d"])
            np.testing.assert_allclose(loaded["test"].probabilities, [0.7, 0.1])
            np.testing.assert_array_equal(loaded["test"].labels, [1, 0])


class AlignTests(unittest.TestCase):
    def test_second_branch_is_reordered_onto_the_first(self) -> None:
        left = split_predictions(["a", "b", "c"], [0.1, 0.2, 0.3], [0, 1, 1])
        right = split_predictions(["c", "a", "b"], [0.9, 0.7, 0.8], [1, 0, 1])

        first, second, labels, ids = align(left, right)

        self.assertEqual(ids, ["a", "b", "c"])
        np.testing.assert_allclose(first, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(second, [0.7, 0.8, 0.9])
        np.testing.assert_array_equal(labels, [0, 1, 1])

    def test_mismatched_video_sets_are_refused(self) -> None:
        """Fusing by position would pair different videos and still print a number."""
        left = split_predictions(["a", "b"], [0.1, 0.2], [0, 1])
        right = split_predictions(["a", "z"], [0.3, 0.4], [0, 1])
        with self.assertRaises(ValueError):
            align(left, right)

    def test_disagreeing_labels_are_refused(self) -> None:
        left = split_predictions(["a", "b"], [0.1, 0.2], [0, 1])
        right = split_predictions(["a", "b"], [0.3, 0.4], [1, 1])
        with self.assertRaises(ValueError):
            align(left, right)


class PlattTests(unittest.TestCase):
    def test_calibration_recovers_a_known_distortion(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.normal(0.0, 2.0, size=2000)
        labels = (rng.random(2000) < 1.0 / (1.0 + np.exp(-scores))).astype(np.int32)
        # A miscalibrated branch: probabilities that are too soft by a factor of 3.
        soft = 1.0 / (1.0 + np.exp(-scores / 3.0))

        calibration = fit_platt(soft, labels)

        self.assertAlmostEqual(calibration.slope, 3.0, delta=0.4)
        self.assertAlmostEqual(calibration.intercept, 0.0, delta=0.2)

    def test_calibration_is_finite_on_a_perfectly_separable_split(self) -> None:
        """Platt target smoothing is why this terminates instead of diverging."""
        probabilities = np.asarray([0.01, 0.02, 0.97, 0.99])
        labels = np.asarray([0, 0, 1, 1])

        calibration = fit_platt(probabilities, labels)

        self.assertTrue(np.isfinite(calibration.slope))
        self.assertTrue(np.isfinite(calibration.intercept))
        calibrated = calibration.apply(probabilities)
        self.assertTrue(((calibrated > 0.0) & (calibrated < 1.0)).all())

    def test_calibration_preserves_ranking(self) -> None:
        """Calibration must not reorder a branch -- only rescale its confidence."""
        probabilities = np.asarray([0.05, 0.3, 0.6, 0.95])
        labels = np.asarray([0, 0, 1, 1])
        calibrated = fit_platt(probabilities, labels).apply(probabilities)
        self.assertTrue(np.all(np.diff(calibrated) > 0))

    def test_logit_clips_saturated_probabilities(self) -> None:
        self.assertTrue(np.isfinite(logit(np.asarray([0.0, 1.0]))).all())


class FuseTests(unittest.TestCase):
    def test_unweighted_mean_is_the_default(self) -> None:
        np.testing.assert_allclose(
            fuse_probabilities(np.asarray([0.2, 0.8]), np.asarray([0.4, 0.4])), [0.3, 0.6]
        )

    def test_weight_is_the_first_branches_share(self) -> None:
        np.testing.assert_allclose(
            fuse_probabilities(np.asarray([0.0]), np.asarray([1.0]), weight=0.25), [0.75]
        )

    def test_fuse_run_calibrates_on_val_and_scores_test_once(self) -> None:
        rng = np.random.default_rng(3)
        n = 120
        labels = (rng.random(n) < 0.6).astype(np.int32)
        signal = labels + rng.normal(0.0, 0.6, size=n)
        strong = 1.0 / (1.0 + np.exp(-(3.0 * (signal - 0.5))))
        weak = 1.0 / (1.0 + np.exp(-(0.4 * (signal - 0.5))))
        ids = [f"v{index}" for index in range(n)]
        half = n // 2

        first = {
            "val": split_predictions(ids[:half], strong[:half].tolist(), labels[:half].tolist()),
            "test": split_predictions(ids[half:], strong[half:].tolist(), labels[half:].tolist()),
        }
        second = {
            "val": split_predictions(ids[:half], weak[:half].tolist(), labels[:half].tolist()),
            "test": split_predictions(ids[half:], weak[half:].tolist(), labels[half:].tolist()),
        }

        result = fuse_run(first, second)

        self.assertIn("test_metrics", result)
        self.assertGreater(result["test_metrics"]["balanced_accuracy"], 0.6)
        self.assertEqual(result["weight"], 0.5)
        # Calibration is fitted on val only; the test block never touches the fit.
        self.assertEqual(len(result["test_video_ids"]), n - half)

    def test_calibration_makes_two_differently_scaled_branches_comparable(self) -> None:
        """The reason fusion calibrates first: an over-confident branch would
        otherwise dominate an unweighted mean regardless of how good it is."""
        labels = np.asarray([0, 0, 1, 1])
        confident_but_wrong = np.asarray([0.99, 0.99, 0.01, 0.01])
        timid_but_right = np.asarray([0.45, 0.46, 0.54, 0.55])

        naive = fuse_probabilities(confident_but_wrong, timid_but_right)
        calibrated = fuse_probabilities(
            fit_platt(confident_but_wrong, labels).apply(confident_but_wrong),
            fit_platt(timid_but_right, labels).apply(timid_but_right),
        )

        # Naive: a negative outranks a positive, because the confident branch wins.
        self.assertGreater(naive[0], naive[2])
        # Calibrated: the branch that is confidently wrong gets its slope flipped, so
        # the fused ordering agrees with the labels again.
        self.assertGreater(calibrated[3], calibrated[0])


class PairedBootstrapTests(unittest.TestCase):
    def test_interval_brackets_a_real_improvement_and_excludes_zero(self) -> None:
        rng = np.random.default_rng(7)
        n = 240
        labels = (rng.random(n) < 0.7).astype(np.int32)
        baseline = np.clip(labels * 0.35 + rng.normal(0.35, 0.2, size=n), 0.01, 0.99)
        candidate = np.clip(labels * 0.6 + rng.normal(0.25, 0.2, size=n), 0.01, 0.99)

        result = paired_bootstrap_delta([baseline], [0.5], [candidate], [0.5], labels, resamples=400)

        self.assertGreater(result["observed_delta"], 0.0)
        self.assertGreater(result["ci_low"], 0.0)
        self.assertLessEqual(result["ci_low"], result["observed_delta"])
        self.assertGreaterEqual(result["ci_high"], result["observed_delta"])

    def test_identical_arms_give_a_zero_delta_and_a_degenerate_interval(self) -> None:
        """Pairing must cancel everything the two arms share."""
        rng = np.random.default_rng(11)
        labels = (rng.random(150) < 0.6).astype(np.int32)
        probabilities = rng.random(150)

        result = paired_bootstrap_delta([probabilities], [0.5], [probabilities], [0.5], labels, resamples=200)

        self.assertEqual(result["observed_delta"], 0.0)
        self.assertEqual(result["ci_low"], 0.0)
        self.assertEqual(result["ci_high"], 0.0)

    def test_seeds_are_averaged_not_pooled(self) -> None:
        labels = np.asarray([0, 1] * 50, dtype=np.int32)
        good = np.where(labels == 1, 0.9, 0.1)
        bad = np.where(labels == 1, 0.1, 0.9)

        # One seed improves, one degrades by the same amount -> mean delta 0.
        result = paired_bootstrap_delta(
            [good, bad], [0.5, 0.5], [bad, good], [0.5, 0.5], labels, resamples=100
        )
        self.assertAlmostEqual(result["observed_delta"], 0.0, places=6)

    def test_mismatched_seed_counts_are_rejected(self) -> None:
        labels = np.asarray([0, 1], dtype=np.int32)
        with self.assertRaises(ValueError):
            paired_bootstrap_delta([np.asarray([0.1, 0.9])], [0.5], [], [], labels)

    def test_calibration_dataclass_applies_its_own_parameters(self) -> None:
        calibration = PlattCalibration(slope=2.0, intercept=0.0)
        np.testing.assert_allclose(calibration.apply(np.asarray([0.5])), [0.5], atol=1e-9)


if __name__ == "__main__":
    unittest.main()


class DuplicateIdTests(unittest.TestCase):
    def test_a_repeated_video_id_is_refused_rather_than_scattered(self) -> None:
        """The position map would collapse the repeat and leave one slot of the
        reordered array unwritten -- uninitialised memory scored as a probability."""
        left = split_predictions(["a", "a", "b"], [0.1, 0.2, 0.3], [0, 0, 1])
        right = split_predictions(["a", "a", "b"], [0.4, 0.5, 0.6], [0, 0, 1])
        with self.assertRaises(ValueError) as ctx:
            align(left, right)
        self.assertIn("Duplicate", str(ctx.exception))
