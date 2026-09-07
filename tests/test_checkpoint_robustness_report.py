from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.video.run_checkpoint_robustness_report import build_statistics
from src.video.checkpoint_robustness_report import (
    READING_REVERSED,
    READING_SUPPORTED,
    READING_UNDETERMINED,
    align_arms,
    arm_name,
    athlete_contribution,
    g4_gate,
    paired_multi_arm_bootstrap,
    primary_reading,
    retention,
)
from src.video.stage_b_report import ArmRuns, load_single_arm


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


def arm_from(
    name: str, probabilities: list[list[float]], labels: list[int], thresholds: list[float], video_ids: list[str] | None = None
) -> ArmRuns:
    return ArmRuns(
        name=name,
        seeds=list(range(1, len(probabilities) + 1)),
        probabilities=[np.asarray(values, dtype=np.float64) for values in probabilities],
        thresholds=thresholds,
        labels=np.asarray(labels, dtype=np.int32),
        video_ids=video_ids or [f"v{index}" for index in range(len(labels))],
    )


def write_arm_dir(
    tmp: Path, subdir: str, test_probabilities: list[float], threshold: float, ids: list[str], labels: list[int]
) -> Path:
    directory = tmp / subdir
    for seed in (1, 2, 3, 4, 5):
        write_predictions(
            directory / f"combined_seed{seed}_predictions.csv",
            {
                "val": (["val_a", "val_b"], [0.6, 0.3], [1, 0]),
                "test": (ids, test_probabilities, labels),
            },
            threshold=threshold,
        )
    return directory


class AthleteContributionAndRetentionTests(unittest.TestCase):
    def test_A_and_R_match_hand_computed_values(self) -> None:
        # full_frame BA fixed at 0.8, background_only BA fixed at 0.65 (all seeds identical
        # and separable at threshold 0.5), so A and R are exact, not just directional.
        labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        full_frame = arm_from("full", [[0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]] * 5, labels, [0.5] * 5)
        background_only = arm_from("bg", [[0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]] * 5, labels, [0.5] * 5)

        a = athlete_contribution(full_frame, background_only)
        r = retention(full_frame, background_only)

        full_ba = full_frame.summary()["balanced_accuracy"]["mean"]
        bg_ba = background_only.summary()["balanced_accuracy"]["mean"]
        self.assertAlmostEqual(a, full_ba - bg_ba, places=9)
        self.assertAlmostEqual(r, (bg_ba - 0.5) / (full_ba - 0.5), places=9)


class G4GateTests(unittest.TestCase):
    def test_gate_passes_when_kin_arms_reproduce_the_published_numbers(self) -> None:
        # Balanced accuracy is exactly determined by recall/specificity counts, so pick
        # probabilities/threshold combinations that land BA precisely on the targets.
        labels = [1] * 10 + [0] * 10
        # full_frame: 8/10 positives right, 9/10 negatives right -> recall .8 spec .9 -> BA .85
        full_probs = [0.9] * 8 + [0.1] * 2 + [0.1] * 9 + [0.9]
        full_frame = arm_from("kin_full_frame", [full_probs] * 5, labels, [0.5] * 5)
        background_only = arm_from("kin_background_only", [full_probs] * 5, labels, [0.5] * 5)
        person_crop = arm_from("kin_person_crop", [full_probs] * 5, labels, [0.5] * 5)
        arms = {
            "kin_full_frame": full_frame,
            "kin_background_only": background_only,
            "kin_person_crop": person_crop,
        }

        gate = g4_gate(arms, checkpoint="kin")

        self.assertAlmostEqual(gate["balanced_accuracy"]["full_frame"], 0.85, places=9)
        self.assertAlmostEqual(gate["R"], 1.0, places=9)
        # Identical to itself but not to the plan's published targets -> gate fails.
        self.assertFalse(gate["passed"])

    def test_gate_reads_real_kin_prediction_dirs_when_present(self) -> None:
        """Gate G4 against the actual repo data, if it is checked out locally."""
        root = Path(__file__).resolve().parents[1]
        full_dir = root / "data/Fitness-AQA/Squat/experiments/videomae_corrected/predictions"
        bg_dir = root / "data/Fitness-AQA/Squat/experiments/videomae_background_only/predictions"
        crop_dir = root / "data/Fitness-AQA/Squat/experiments/videomae_person_crop/predictions"
        if not (full_dir.exists() and bg_dir.exists() and crop_dir.exists()):
            self.skipTest("Real Fitness-AQA prediction dirs are not checked out on this machine.")

        seeds = [1, 2, 3, 4, 5]
        arms = {
            arm_name("kin", "full_frame"): load_single_arm(arm_name("kin", "full_frame"), full_dir, "combined", seeds),
            arm_name("kin", "background_only"): load_single_arm(
                arm_name("kin", "background_only"), bg_dir, "combined", seeds
            ),
            arm_name("kin", "person_crop"): load_single_arm(arm_name("kin", "person_crop"), crop_dir, "combined", seeds),
        }

        gate = g4_gate(arms, checkpoint="kin")

        self.assertAlmostEqual(gate["balanced_accuracy"]["full_frame"], 0.6401, places=4)
        self.assertAlmostEqual(gate["balanced_accuracy"]["background_only"], 0.6271, places=4)
        self.assertAlmostEqual(gate["balanced_accuracy"]["person_crop"], 0.6662, places=4)
        self.assertAlmostEqual(gate["R"], 0.907, places=2)
        self.assertTrue(gate["passed"])


class AlignArmsTests(unittest.TestCase):
    def test_arms_with_different_video_orders_are_paired_by_id(self) -> None:
        labels = [1, 0, 1, 0]
        reference = arm_from("ref", [[0.9, 0.1, 0.8, 0.2]], labels, [0.5], video_ids=["a", "b", "c", "d"])
        # Same videos, same labels, shuffled order -- probabilities carried along with the id.
        shuffled = arm_from(
            "shuffled",
            [[0.2, 0.8, 0.9, 0.1]],
            [0, 1, 1, 0],
            [0.5],
            video_ids=["d", "c", "a", "b"],
        )

        aligned = align_arms({"ref": reference, "shuffled": shuffled})

        self.assertEqual(aligned["shuffled"].video_ids, aligned["ref"].video_ids)
        np.testing.assert_array_equal(aligned["shuffled"].labels, aligned["ref"].labels)
        # Video "a" was probability 0.9 for shuffled at its own index 2 -- must land at
        # reference's index 0 after alignment.
        self.assertAlmostEqual(aligned["shuffled"].probabilities[0][0], 0.9)

    def test_mismatched_video_sets_are_refused(self) -> None:
        reference = arm_from("ref", [[0.9, 0.1]], [1, 0], [0.5], video_ids=["a", "b"])
        other = arm_from("other", [[0.9, 0.1]], [1, 0], [0.5], video_ids=["a", "zz"])
        with self.assertRaises(ValueError):
            align_arms({"ref": reference, "other": other})


class PairedMultiArmBootstrapTests(unittest.TestCase):
    def _two_checkpoint_arms(self) -> dict[str, ArmRuns]:
        rng = np.random.default_rng(42)
        n = 200
        labels = (rng.random(n) < 0.5).astype(np.int32)
        video_ids = [f"v{i}" for i in range(n)]

        # kin: full_frame clearly beats background_only (large A_kin).
        kin_full = np.clip(labels * 0.7 + rng.normal(0.15, 0.15, size=n), 0.01, 0.99)
        kin_background = np.clip(labels * 0.2 + rng.normal(0.4, 0.15, size=n), 0.01, 0.99)
        # ssv2: full_frame and background_only are nearly identical (tiny A_ssv2), so
        # delta A = A_ssv2 - A_kin is reliably negative -- the "ssv2 needs the athlete
        # less" branch.
        ssv2_full = np.clip(labels * 0.55 + rng.normal(0.25, 0.15, size=n), 0.01, 0.99)
        ssv2_background = np.clip(labels * 0.53 + rng.normal(0.26, 0.15, size=n), 0.01, 0.99)
        # person_crop arms are required by build_statistics(["kin", "ssv2"]) even though
        # this test only cares about delta A -- keep them simple but non-degenerate.
        kin_crop = np.clip(labels * 0.6 + rng.normal(0.2, 0.15, size=n), 0.01, 0.99)
        ssv2_crop = np.clip(labels * 0.5 + rng.normal(0.25, 0.15, size=n), 0.01, 0.99)

        def make(name: str, probabilities: np.ndarray) -> ArmRuns:
            return arm_from(name, [probabilities.tolist()] * 5, labels.tolist(), [0.5] * 5, video_ids=video_ids)

        return {
            "kin_full_frame": make("kin_full_frame", kin_full),
            "kin_background_only": make("kin_background_only", kin_background),
            "kin_person_crop": make("kin_person_crop", kin_crop),
            "ssv2_full_frame": make("ssv2_full_frame", ssv2_full),
            "ssv2_background_only": make("ssv2_background_only", ssv2_background),
            "ssv2_person_crop": make("ssv2_person_crop", ssv2_crop),
        }

    def test_delta_A_matches_hand_computed_difference_of_differences(self) -> None:
        arms = self._two_checkpoint_arms()
        aligned = align_arms(arms)
        statistics = build_statistics(["kin", "ssv2"], have_ssv2=True)

        result = paired_multi_arm_bootstrap(aligned, statistics, resamples=500, seed=1)

        a_kin = athlete_contribution(arms["kin_full_frame"], arms["kin_background_only"])
        a_ssv2 = athlete_contribution(arms["ssv2_full_frame"], arms["ssv2_background_only"])
        self.assertAlmostEqual(result["delta_A"]["observed"], a_ssv2 - a_kin, places=9)
        # Constructed so ssv2 needs the athlete much less than kin does.
        self.assertLess(result["delta_A"]["ci_high"], 0.0)

    def test_bootstrap_is_paired_same_draw_scores_every_arm(self) -> None:
        """Identical kin and ssv2 arms must give an exactly-zero delta A on every
        resample, because a paired draw cancels everything the two share -- if the
        four arms were resampled independently this would not be exactly zero."""
        rng = np.random.default_rng(5)
        n = 150
        labels = (rng.random(n) < 0.5).astype(np.int32)
        probabilities = rng.random(n)
        video_ids = [f"v{i}" for i in range(n)]

        def make(name: str) -> ArmRuns:
            return arm_from(name, [probabilities.tolist()] * 3, labels.tolist(), [0.5] * 3, video_ids=video_ids)

        arms = {
            "kin_full_frame": make("kin_full_frame"),
            "kin_background_only": make("kin_background_only"),
            "kin_person_crop": make("kin_person_crop"),
            "ssv2_full_frame": make("ssv2_full_frame"),
            "ssv2_background_only": make("ssv2_background_only"),
            "ssv2_person_crop": make("ssv2_person_crop"),
        }
        aligned = align_arms(arms)
        statistics = build_statistics(["kin", "ssv2"], have_ssv2=True)

        result = paired_multi_arm_bootstrap(aligned, statistics, resamples=300, seed=99)

        self.assertEqual(result["delta_A"]["observed"], 0.0)
        self.assertEqual(result["delta_A"]["ci_low"], 0.0)
        self.assertEqual(result["delta_A"]["ci_high"], 0.0)

    def test_bootstrap_is_deterministic_under_a_fixed_seed(self) -> None:
        arms = self._two_checkpoint_arms()
        aligned = align_arms(arms)
        statistics = build_statistics(["kin", "ssv2"], have_ssv2=True)

        first = paired_multi_arm_bootstrap(aligned, statistics, resamples=200, seed=7)
        second = paired_multi_arm_bootstrap(aligned, statistics, resamples=200, seed=7)

        self.assertEqual(first["delta_A"]["ci_low"], second["delta_A"]["ci_low"])
        self.assertEqual(first["delta_A"]["ci_high"], second["delta_A"]["ci_high"])

    def test_unaligned_arms_are_refused(self) -> None:
        arms = self._two_checkpoint_arms()
        # Deliberately skip align_arms -- the label vectors happen to already agree
        # (same labels array reused), so break it explicitly instead.
        broken = dict(arms)
        broken["kin_full_frame"] = arm_from(
            "kin_full_frame", [[0.9, 0.1, 0.5]], [1, 0, 1], [0.5], video_ids=["x", "y", "z"]
        )
        statistics = build_statistics(["kin", "ssv2"], have_ssv2=True)
        with self.assertRaises(ValueError):
            paired_multi_arm_bootstrap(broken, statistics, resamples=10, seed=1)


class PrimaryReadingTests(unittest.TestCase):
    def test_ci_entirely_above_zero_is_h2_supported(self) -> None:
        self.assertEqual(primary_reading(ci_low=0.01, ci_high=0.05), READING_SUPPORTED)

    def test_ci_straddling_zero_is_undetermined(self) -> None:
        self.assertEqual(primary_reading(ci_low=-0.01, ci_high=0.05), READING_UNDETERMINED)

    def test_ci_entirely_below_zero_is_ssv2_needs_the_athlete_less(self) -> None:
        self.assertEqual(primary_reading(ci_low=-0.05, ci_high=-0.01), READING_REVERSED)


class KinOnlyModeTests(unittest.TestCase):
    def test_kin_only_statistics_have_no_delta_A(self) -> None:
        statistics = build_statistics(["kin"], have_ssv2=False)
        self.assertNotIn("delta_A", statistics)
        self.assertIn("R_kin", statistics)
        self.assertIn("person_crop_minus_full_frame_kin", statistics)

    def test_kin_only_bootstrap_runs_without_ssv2_arms(self) -> None:
        labels = [1, 1, 0, 0, 1, 0, 1, 0]
        full = arm_from("kin_full_frame", [[0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.6, 0.4]] * 5, labels, [0.5] * 5)
        background = arm_from("kin_background_only", [[0.6, 0.55, 0.4, 0.45, 0.5, 0.5, 0.5, 0.5]] * 5, labels, [0.5] * 5)
        crop = arm_from("kin_person_crop", [[0.95, 0.85, 0.1, 0.05, 0.8, 0.2, 0.7, 0.3]] * 5, labels, [0.5] * 5)
        arms = {"kin_full_frame": full, "kin_background_only": background, "kin_person_crop": crop}
        aligned = align_arms(arms)
        statistics = build_statistics(["kin"], have_ssv2=False)

        result = paired_multi_arm_bootstrap(aligned, statistics, resamples=200, seed=3)

        self.assertIn("R_kin", result)
        self.assertIn("person_crop_minus_full_frame_kin", result)
        self.assertNotIn("delta_A", result)


class LoadRealArmsIntegrationTests(unittest.TestCase):
    def test_end_to_end_from_predictions_csvs(self) -> None:
        with TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            ids = [f"v{i}" for i in range(20)]
            labels = [1] * 10 + [0] * 10
            full_dir = write_arm_dir(
                tmp, "kin_full_frame", [0.9] * 8 + [0.1] * 2 + [0.1] * 9 + [0.9], 0.5, ids, labels
            )
            background_dir = write_arm_dir(
                tmp, "kin_background_only", [0.6] * 6 + [0.4] * 4 + [0.4] * 8 + [0.6] * 2, 0.5, ids, labels
            )
            crop_dir = write_arm_dir(
                tmp, "kin_person_crop", [0.9] * 9 + [0.1] + [0.1] * 9 + [0.9], 0.5, ids, labels
            )

            arms = {
                "kin_full_frame": load_single_arm("kin_full_frame", full_dir, "combined", [1, 2, 3, 4, 5]),
                "kin_background_only": load_single_arm(
                    "kin_background_only", background_dir, "combined", [1, 2, 3, 4, 5]
                ),
                "kin_person_crop": load_single_arm("kin_person_crop", crop_dir, "combined", [1, 2, 3, 4, 5]),
            }
            gate = g4_gate(arms, checkpoint="kin")
            aligned = align_arms(arms)
            statistics = build_statistics(["kin"], have_ssv2=False)
            bootstrap = paired_multi_arm_bootstrap(aligned, statistics, resamples=100, seed=1)

            self.assertIn("balanced_accuracy", gate)
            self.assertIn("R_kin", bootstrap)
            self.assertAlmostEqual(bootstrap["R_kin"]["observed"], gate["R"], places=9)


if __name__ == "__main__":
    unittest.main()
