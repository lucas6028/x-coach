"""Unit tests for the calibrated NLF + VideoMAE late fusion on REHAB24-6.

Synthetic data only: a small REHAB24-6-shaped cohort (nine primary subjects plus a tiny
P10, two sessions each, one interleaved and one label-segmented) and random feature
bundles inside a ``TemporaryDirectory``. ``MIN_VAL_SUBJECT_SAMPLES`` is patched down so
the cohort's subjects are eligible validation subjects; the real threshold, hashes and
session counts are only exercised by the CLI run against the real manifest.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# src.rehab24.loso_cross_validation exits at import time without torch, which the lean
# CI dependency set does not install; without this the whole module is a collection
# error rather than a skip.
torch = pytest.importorskip("torch")

from src.rehab24 import loso_cross_validation as loso
from src.rehab24 import pose_video_fusion as pvf
from src.rehab24 import video_backbone_comparison as vbc
from src.rehab24.dataset import CAMERAS, MANIFEST_FIELDS
from src.rehab24.videomae_position_control import position_rule_loso
from src.video.late_fusion import PROBABILITY_EPS, PlattCalibration
from src.video.videomae_video_classifier import clear_feature_cache

PRIMARY = [str(index) for index in range(1, 10)]
INTERLEAVED_LABELS = (1, 0, 1, 0, 1, 0)
SEGMENTED_LABELS = (1, 1, 1, 0, 0, 0)
SMALL_VAL_THRESHOLD = 8
FAST_CONFIG = loso.FoldConfig(epochs=2, early_stopping_patience=0)


def build_cohort(root: Path, rng: np.random.Generator, dims: dict[str, int]) -> dict:
    """Manifest, labels, Segmentation.csv and one feature directory per branch."""
    rows: list[dict[str, str]] = []
    labels: dict[str, int] = {}
    segmentation: list[dict[str, str]] = []
    sessions = [(subject, "1", f"V{subject}a", INTERLEAVED_LABELS) for subject in PRIMARY]
    sessions += [(subject, "2", f"V{subject}b", SEGMENTED_LABELS) for subject in PRIMARY]
    sessions.append(("10", "1", "V10a", (1, 0)))

    for subject, exercise_id, video_id, pattern in sessions:
        for number, correctness in enumerate(pattern, start=1):
            segmentation.append({"person_id": subject, "correctness": str(correctness), "repetition_number": str(number)})
            for camera in CAMERAS:
                sample_id = f"Ex{exercise_id}_{video_id}_rep{number}_{camera}"
                row = {name: "" for name in MANIFEST_FIELDS}
                row.update(
                    {
                        "sample_id": sample_id,
                        "split": "train",
                        "video_id": video_id,
                        "repetition_number": str(number),
                        "exercise_id": exercise_id,
                        "person_id": subject,
                        "camera": camera,
                        "correctness": str(correctness),
                    }
                )
                rows.append(row)
                labels[sample_id] = correctness

    manifest_path = root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    labels_path = root / "correctness.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    segmentation_path = root / "Segmentation.csv"
    with segmentation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["person_id", "correctness", "repetition_number"], delimiter=";")
        writer.writeheader()
        writer.writerows(segmentation)

    feature_dirs: dict[str, Path] = {}
    for branch, dim in dims.items():
        feature_dirs[branch] = root / f"features_{branch}"
        feature_dirs[branch].mkdir()
        for row in rows:
            vector = rng.normal(size=dim).astype(np.float32) + 0.8 * labels[row["sample_id"]]
            np.savez(feature_dirs[branch] / f"{row['sample_id']}.npz", video_feature=vector)

    return {
        "manifest_path": manifest_path,
        "labels_path": labels_path,
        "segmentation_path": segmentation_path,
        "feature_dirs": feature_dirs,
        "manifest_rows": {row["sample_id"]: row for row in rows},
        "labels": labels,
    }


def oof_rows(
    subject: str,
    test_subject: str,
    val_subject: str,
    probabilities: dict[tuple[int, str], float],
    labels: dict[int, int],
    seed: int = 42,
    video_id: str = "VX",
) -> list[dict]:
    """Parsed OOF-schema rows for one subject: ``probabilities`` is keyed by (repetition, camera)."""
    return [
        {
            "sample_id": f"Ex1_{video_id}_rep{number}_{camera}",
            "repetition_id": f"{video_id}_rep{number}",
            "session_id": f"1_{video_id}",
            "person_id": subject,
            "exercise_id": "1",
            "camera": camera,
            "seed": seed,
            "label": labels[number],
            "probability": probability,
            "threshold": 0.5,
            "test_subject": test_subject,
            "val_subject": val_subject,
        }
        for (number, camera), probability in probabilities.items()
    ]


def random_split(
    rng: np.random.Generator, subject: str, fold: dict, n_reps: int, video_id: str, signal: float = 0.25, positive_every: int = 2
) -> tuple[list[dict], list[dict]]:
    """Two branches' rows for one subject, mildly informative, never saturated.

    ``positive_every`` sets the positive rate: a positive ``k`` labels every k-th repetition
    correct (rate about 1/k), a negative ``k`` labels every k-th one incorrect. The splits
    of one fold are given different rates on purpose, so a quantity read off the wrong
    split cannot come out right by coincidence.
    """
    every = abs(positive_every)
    labels = {number: int((number % every == 0) == (positive_every > 0)) for number in range(1, n_reps + 1)}
    branches = []
    for _ in range(2):
        probabilities = {
            (number, camera): float(np.clip(0.5 + signal * (labels[number] - 0.5) + rng.normal(scale=0.12), 0.05, 0.95))
            for number in labels
            for camera in CAMERAS
        }
        branches.append(oof_rows(subject, fold["test_subject"], fold["val_subject"], probabilities, labels, video_id=video_id))
    return branches[0], branches[1]


FOLD = {"test_subject": "1", "val_subject": "2"}


class TrainOneFoldValidationExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(clear_feature_cache)
        rng = np.random.default_rng(0)
        self.feature_dir = Path(self.tmp.name)
        self.labels: dict[str, int] = {}
        self.ids = {"train": [], "val": [], "test": []}
        for split, count in (("train", 40), ("val", 16), ("test", 12)):
            for index in range(count):
                sample_id = f"{split}{index}"
                self.labels[sample_id] = index % 2
                np.savez(
                    self.feature_dir / f"{sample_id}.npz",
                    video_feature=rng.normal(size=5).astype(np.float32) + self.labels[sample_id],
                )
                self.ids[split].append(sample_id)
        self.device = torch.device("cpu")

    def fit(self, config: "loso.FoldConfig", test_split: str = "test", **kwargs):
        return loso.train_one_fold(
            self.feature_dir, self.ids["train"], self.ids["val"], self.ids[test_split], self.labels, config, self.device, 7, **kwargs
        )

    def test_default_return_is_unchanged_and_the_export_does_not_move_it(self) -> None:
        config = loso.FoldConfig(epochs=4)
        plain = self.fit(config)
        exported = self.fit(config, return_validation=True)
        self.assertEqual(len(plain), 3)
        self.assertEqual(len(exported), 4)
        self.assertEqual(plain[0], exported[0])
        np.testing.assert_array_equal(plain[1], exported[1])
        np.testing.assert_array_equal(plain[2], exported[2])
        self.assertEqual(exported[3].sample_ids, self.ids["val"])

    def test_export_is_the_best_checkpoint_not_the_last_epoch(self) -> None:
        validation_scores: list[np.ndarray] = []
        real_collect = loso.collect_sample_predictions

        def recording_collect(model, samples, *args, **kwargs):
            ids, probabilities, labels = real_collect(model, samples, *args, **kwargs)
            if ids == self.ids["val"]:
                validation_scores.append(probabilities.copy())
            return ids, probabilities, labels

        calls = {"n": 0}

        def first_epoch_wins(probabilities, labels, objective):
            calls["n"] += 1
            return 0.5, {objective: 1.0 / calls["n"]}

        with patch.object(loso, "collect_sample_predictions", recording_collect), patch.object(
            loso, "find_best_threshold", first_epoch_wins
        ):
            *_, validation = self.fit(loso.FoldConfig(epochs=4, early_stopping_patience=0), return_validation=True)

        self.assertEqual(len(validation_scores), 4)
        np.testing.assert_array_equal(validation.probabilities, validation_scores[0])
        self.assertFalse(np.array_equal(validation.probabilities, validation_scores[-1]))

    def test_export_equals_what_the_restored_checkpoint_scores(self) -> None:
        # Scoring the validation ids as the test set reads them off the restored checkpoint.
        _, test_prob, _, validation = self.fit(loso.FoldConfig(epochs=5, early_stopping_patience=2), test_split="val", return_validation=True)
        np.testing.assert_array_equal(validation.probabilities, test_prob)


class FusionRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(3)
        self.val_pose, self.val_video = random_split(self.rng, "2", FOLD, 12, "VV", positive_every=3)  # rate 1/3
        self.test_pose, self.test_video = random_split(self.rng, "1", FOLD, 10, "VT", positive_every=-5)  # rate 0.8

    def fuse(self, **overrides) -> dict:
        rows = {
            "validation": {pvf.POSE_BRANCH: self.val_pose, pvf.VIDEO_BRANCH: self.val_video},
            "held_out": {pvf.POSE_BRANCH: self.test_pose, pvf.VIDEO_BRANCH: self.test_video},
        }
        rows.update(overrides)
        return pvf.fuse_fold(FOLD, rows["validation"], rows["held_out"], shares=(0.0, 0.5, 1.0))

    def test_calibration_and_threshold_ignore_the_held_out_subject(self) -> None:
        reference = self.fuse()
        other_pose, other_video = random_split(np.random.default_rng(99), "1", FOLD, 6, "VO", signal=-0.4)
        changed = self.fuse(held_out={pvf.POSE_BRANCH: other_pose, pvf.VIDEO_BRANCH: other_video})
        for branch in pvf.BRANCHES:
            self.assertEqual(reference["calibrations"][branch], changed["calibrations"][branch])
        self.assertEqual(reference["by_share"][0.5]["threshold"], changed["by_share"][0.5]["threshold"])

    def test_the_provenance_record_is_read_off_the_rows_that_were_fitted(self) -> None:
        fused = self.fuse()
        self.assertEqual(fused["fitted_on"]["person_ids"], [FOLD["val_subject"]])
        self.assertEqual(fused["scored_on"]["person_ids"], [FOLD["test_subject"]])
        self.assertEqual(fused["fitted_on"]["ids_sha256"], vbc.hash_ids([row["sample_id"] for row in self.val_pose]))
        self.assertEqual(fused["fitted_on"]["n_rows"], len(self.val_pose))
        self.assertNotEqual(fused["fitted_on"]["ids_sha256"], fused["scored_on"]["ids_sha256"])

    def test_a_held_out_row_in_the_validation_set_raises(self) -> None:
        leaked = [{**self.test_pose[0]}]
        with self.assertRaises(pvf.FoldPurityError):
            self.fuse(validation={pvf.POSE_BRANCH: self.val_pose + leaked, pvf.VIDEO_BRANCH: self.val_video + [{**self.test_video[0]}]})

    def test_a_validation_row_in_the_held_out_set_raises(self) -> None:
        with self.assertRaises(pvf.FoldPurityError):
            self.fuse(held_out={pvf.POSE_BRANCH: self.test_pose + [self.val_pose[0]], pvf.VIDEO_BRANCH: self.test_video + [self.val_video[0]]})

    def test_a_fold_validating_on_its_own_held_out_subject_raises(self) -> None:
        with self.assertRaises(pvf.FoldPurityError):
            pvf.fuse_fold({"test_subject": "1", "val_subject": "1"}, {}, {})

    def test_branches_align_by_sample_id_not_row_order(self) -> None:
        reference = self.fuse()
        shuffled = self.fuse(
            validation={pvf.POSE_BRANCH: self.val_pose, pvf.VIDEO_BRANCH: self.val_video[::-1]},
            held_out={pvf.POSE_BRANCH: self.test_pose, pvf.VIDEO_BRANCH: self.test_video[::-1]},
        )
        self.assertEqual(reference["held_out"].sample_ids, shuffled["held_out"].sample_ids)
        np.testing.assert_array_equal(
            reference["by_share"][0.5]["held_out_probabilities"], shuffled["by_share"][0.5]["held_out_probabilities"]
        )

    def test_duplicate_mismatched_and_relabelled_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.fuse(held_out={pvf.POSE_BRANCH: self.test_pose + [self.test_pose[0]], pvf.VIDEO_BRANCH: self.test_video + [self.test_video[0]]})
        with self.assertRaises(ValueError):
            self.fuse(held_out={pvf.POSE_BRANCH: self.test_pose, pvf.VIDEO_BRANCH: self.test_video[1:]})
        relabelled = [{**self.test_video[0], "label": 1 - self.test_video[0]["label"]}] + self.test_video[1:]
        with self.assertRaises(ValueError):
            self.fuse(held_out={pvf.POSE_BRANCH: self.test_pose, pvf.VIDEO_BRANCH: relabelled})

    def test_degenerate_shares_return_a_calibrated_branch_bit_for_bit(self) -> None:
        fused = self.fuse()
        for split in ("validation", "held_out"):
            pose, video = fused[f"{split}_calibrated"]
            key = f"{split}_probabilities"
            self.assertTrue(np.array_equal(fused["by_share"][1.0][key], pose))
            self.assertTrue(np.array_equal(fused["by_share"][0.0][key], video))

    def test_platt_keeps_the_ranking_when_its_slope_is_positive(self) -> None:
        fused = self.fuse()
        self.assertGreater(fused["calibrations"][pvf.POSE_BRANCH].slope, 0)
        np.testing.assert_array_equal(pvf.midranks(fused["held_out"].pose), pvf.midranks(fused["held_out_calibrated"][0]))

    def test_the_constant_branch_is_one_number_read_off_validation_labels(self) -> None:
        rows = {
            "validation": {pvf.POSE_BRANCH: self.val_pose, pvf.VIDEO_BRANCH: self.val_video},
            "held_out": {pvf.POSE_BRANCH: self.test_pose, pvf.VIDEO_BRANCH: self.test_video},
        }
        constant = pvf.fuse_fold(FOLD, rows["validation"], rows["held_out"], constant_video=True)
        rate = float(np.mean([row["label"] for row in self.val_pose]))
        held_out_rate = float(np.mean([row["label"] for row in self.test_pose]))
        self.assertEqual((rate, held_out_rate), (1 / 3, 0.8))  # the fixture must be able to tell the two apart
        self.assertEqual(constant["constant_rate"], rate)
        self.assertNotIn(pvf.VIDEO_BRANCH, constant["calibrations"])  # nothing is fitted for the second branch
        for split in ("validation", "held_out"):
            self.assertEqual(set(constant[f"{split}_calibrated"][1].tolist()), {rate})
        np.testing.assert_array_equal(
            constant["by_share"][0.5]["held_out_probabilities"], 0.5 * constant["held_out_calibrated"][0] + 0.5 * rate
        )

        # No held-out label and no video probability of either split reaches it.
        flipped = [[{**row, "label": 1 - row["label"]} for row in branch] for branch in (self.test_pose, self.test_video)]
        scrambled_validation = [{**row, "probability": 1.0 - row["probability"]} for row in self.val_video]
        scrambled_held_out = [{**row, "probability": 1.0 - row["probability"]} for row in flipped[1]]
        other = pvf.fuse_fold(
            FOLD,
            {pvf.POSE_BRANCH: self.val_pose, pvf.VIDEO_BRANCH: scrambled_validation},
            {pvf.POSE_BRANCH: flipped[0], pvf.VIDEO_BRANCH: scrambled_held_out},
            constant_video=True,
        )
        self.assertEqual(other["constant_rate"], rate)
        self.assertEqual(other["by_share"][0.5]["threshold"], constant["by_share"][0.5]["threshold"])
        np.testing.assert_array_equal(other["by_share"][0.5]["held_out_probabilities"], constant["by_share"][0.5]["held_out_probabilities"])

    def test_fused_oof_rows_keep_identity_columns_and_round_trip_exactly(self) -> None:
        fused = self.fuse()
        result = fused["by_share"][0.5]
        rows = pvf.fused_oof_rows(self.test_pose, fused["held_out"].sample_ids, result["held_out_probabilities"], result["threshold"])
        self.assertEqual([row["sample_id"] for row in rows], fused["held_out"].sample_ids)
        self.assertEqual({row["person_id"] for row in rows}, {"1"})
        self.assertEqual([float(row["probability"]) for row in rows], result["held_out_probabilities"].tolist())


class TunedShareTests(unittest.TestCase):
    def test_a_unique_maximum_wins(self) -> None:
        scores = [0.5] * 11
        scores[8] = 0.7
        self.assertEqual(pvf.WEIGHT_GRID[pvf.select_tuned_share(scores)], 0.8)

    def test_ties_go_to_the_share_nearest_one_half(self) -> None:
        self.assertEqual(pvf.WEIGHT_GRID[pvf.select_tuned_share([0.6] * 11)], 0.5)
        scores = [0.5] * 11
        scores[1] = scores[4] = 0.7
        self.assertEqual(pvf.WEIGHT_GRID[pvf.select_tuned_share(scores)], 0.4)

    def test_equidistant_ties_go_to_the_larger_pose_share(self) -> None:
        scores = [0.5] * 11
        scores[4] = scores[6] = 0.7
        self.assertEqual(pvf.WEIGHT_GRID[pvf.select_tuned_share(scores)], 0.6)

    def test_a_wrong_length_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pvf.select_tuned_share([0.5] * 10)


class PermutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows, _ = random_split(np.random.default_rng(5), "1", FOLD, 14, "VP")

    @staticmethod
    def pairs(rows: list[dict]) -> dict[str, tuple[float, float]]:
        by_repetition: dict[str, dict[str, float]] = {}
        for row in rows:
            by_repetition.setdefault(row["repetition_id"], {})[row["camera"]] = row["probability"]
        return {key: (cameras[CAMERAS[0]], cameras[CAMERAS[1]]) for key, cameras in by_repetition.items()}

    def test_camera_pairs_move_together_and_nothing_else_changes(self) -> None:
        permuted = pvf.permute_branch_rows(self.rows, 101, "1", "held_out")
        self.assertEqual(sorted(self.pairs(permuted).values()), sorted(self.pairs(self.rows).values()))
        self.assertNotEqual(self.pairs(permuted), self.pairs(self.rows))
        original = {row["sample_id"]: row for row in self.rows}
        for row in permuted:
            self.assertEqual({k: v for k, v in row.items() if k != "probability"}, {k: v for k, v in original[row["sample_id"]].items() if k != "probability"})

    def test_the_shuffle_is_a_function_of_seed_subject_and_split_only(self) -> None:
        reference = self.pairs(pvf.permute_branch_rows(self.rows, 101, "1", "held_out"))
        self.assertEqual(reference, self.pairs(pvf.permute_branch_rows(self.rows[::-1], 101, "1", "held_out")))
        self.assertNotEqual(reference, self.pairs(pvf.permute_branch_rows(self.rows, 202, "1", "held_out")))
        self.assertNotEqual(reference, self.pairs(pvf.permute_branch_rows(self.rows, 101, "2", "held_out")))
        self.assertNotEqual(reference, self.pairs(pvf.permute_branch_rows(self.rows, 101, "1", "validation")))

    def test_an_incomplete_camera_pair_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pvf.permute_branch_rows(self.rows[1:], 101, "1", "held_out")

    def test_audit_seeds_are_1001_to_1020_with_model_seeds_assigned_cyclically(self) -> None:
        self.assertEqual(pvf.AUDIT_PERMUTATION_SEEDS, tuple(range(1001, 1021)))
        self.assertEqual([pvf.audit_model_seed(seed) for seed in (1001, 1002, 1003, 1004, 1005, 1006)], [42, 7, 1234, 42, 7, 1234])
        self.assertEqual([pvf.audit_model_seed(seed) for seed in (1019, 1020)], [42, 7])
        self.assertEqual([pvf.audit_model_seed(seed) for seed in pvf.AUDIT_PERMUTATION_SEEDS].count(1234), 6)
        self.assertEqual((pvf.UNINFORMATIVE_AUDIT_TRIGGER, pvf.LEAK_AUDIT_CLEAR_THRESHOLD), (0.01, 0.01))
        for unregistered in (101, 1000, 1021):
            with self.assertRaises(ValueError):
                pvf.audit_model_seed(unregistered)

    def test_the_registered_pairing_covers_each_model_seed_once(self) -> None:
        self.assertEqual(sorted(model_seed for _, model_seed in pvf.PERMUTATION_PAIRS), sorted(pvf.SEEDS))
        self.assertEqual(dict(pvf.PERMUTATION_PAIRS), {101: 42, 202: 7, 303: 1234})


class VerdictTests(unittest.TestCase):
    @staticmethod
    def primary(delta: float, p_value: float, n_positive: int = 9, ci: tuple[float, float] | None = None) -> dict:
        return {
            "mean_delta": delta,
            "n_positive": n_positive,
            "n_subjects": 9,
            "bootstrap_ci_95": list(ci if ci is not None else (delta - 0.01, delta + 0.01)),
            "sign_flip": {"p_value": p_value},
        }

    def verdict(self, primary: dict, gates_pass: bool = True, fused: float = 0.70, video: float = 0.66, pose_cal: float = 0.67) -> dict:
        return pvf.fusion_verdict(primary, gates_pass, fused, video, pose_cal)

    def test_row_0_a_failed_gate_outranks_every_significance_branch(self) -> None:
        verdict = self.verdict(self.primary(0.05, 0.004), gates_pass=False)
        self.assertEqual(verdict["row"], 0)
        self.assertFalse(verdict["stop_rule_applies"])

    def test_row_1_needs_every_condition(self) -> None:
        verdict = self.verdict(self.primary(0.03, 0.004))
        self.assertEqual(verdict["row"], 1)
        self.assertFalse(verdict["stop_rule_applies"])

    def test_row_2_names_each_failed_condition(self) -> None:
        cases = {
            "delta_at_least_margin": self.verdict(self.primary(0.015, 0.004)),
            "at_least_7_of_9_positive": self.verdict(self.primary(0.03, 0.004, n_positive=6)),
            "ci_excludes_zero": self.verdict(self.primary(0.03, 0.004, ci=(-0.001, 0.06))),
            "fused_mean_at_least_vm16_mean": self.verdict(self.primary(0.03, 0.004), fused=0.70, video=0.71),
            "fused_mean_at_least_nlf_cal_mean": self.verdict(self.primary(0.03, 0.004), fused=0.70, pose_cal=0.705),
        }
        for condition, verdict in cases.items():
            with self.subTest(condition=condition):
                self.assertEqual(verdict["row"], 2)
                self.assertEqual(verdict["failed_conditions"], [condition])
                self.assertTrue(verdict["stop_rule_applies"])

    def test_row_3_significant_negative(self) -> None:
        self.assertEqual(self.verdict(self.primary(-0.03, 0.004, n_positive=0))["row"], 3)

    def test_row_4_is_undetermined_whatever_the_delta(self) -> None:
        for delta in (0.05, -0.05, 0.0):
            verdict = self.verdict(self.primary(delta, 0.066))
            self.assertEqual(verdict["row"], 4)
            self.assertIn("undetermined", verdict["reading"])

    def test_boundaries_fall_on_the_side_the_plan_writes(self) -> None:
        passing = dict(delta=0.03, p_value=0.004)
        cases = [
            ("delta exactly at the margin", self.verdict(self.primary(0.02, 0.004)), 1, []),
            ("delta just under the margin", self.verdict(self.primary(0.0199999, 0.004)), 2, ["delta_at_least_margin"]),
            ("exactly 7 of 9 positive", self.verdict(self.primary(n_positive=7, **passing)), 1, []),
            ("6 of 9 positive", self.verdict(self.primary(n_positive=6, **passing)), 2, ["at_least_7_of_9_positive"]),
            ("p exactly alpha", self.verdict(self.primary(0.03, 0.05)), 4, []),
            ("p just under alpha", self.verdict(self.primary(0.03, 0.0499999)), 1, []),
            ("CI lower bound exactly zero", self.verdict(self.primary(ci=(0.0, 0.06), **passing)), 2, ["ci_excludes_zero"]),
            ("fused mean equal to the vm16 mean", self.verdict(self.primary(**passing), fused=0.70, video=0.70), 1, []),
            ("fused mean equal to the nlf_cal mean", self.verdict(self.primary(**passing), fused=0.70, pose_cal=0.70), 1, []),
            ("delta exactly zero at p < alpha", self.verdict(self.primary(0.0, 0.004, ci=(0.0, 0.0))), 4, []),
            ("any gate failing", self.verdict(self.primary(**passing), gates_pass=False), 0, []),
        ]
        for name, verdict, row, failed in cases:
            with self.subTest(case=name):
                self.assertEqual(verdict["row"], row)
                self.assertEqual(verdict["failed_conditions"], failed)

    def test_position_reading_only_qualifies_a_claimed_gain(self) -> None:
        positive = {"mean_delta": 0.02}
        self.assertIn("carries no information", pvf.position_reading(1, positive, {"significant": True}))
        self.assertIn("not separated", pvf.position_reading(1, positive, {"significant": False}))
        self.assertIn("not separated", pvf.position_reading(1, {"mean_delta": -0.01}, {"significant": True}))
        self.assertIn("changes nothing", pvf.position_reading(4, positive, {"significant": True}))


class RankingEvidenceTests(unittest.TestCase):
    calibration = PlattCalibration(slope=1.3, intercept=0.1)

    def test_ties_made_by_the_clip_are_counted_not_failed(self) -> None:
        raw = np.asarray([1e-9, 1e-8, 0.2, 0.5, 0.9, 1.0 - 1e-9])
        calibrated = self.calibration.apply(raw)
        self.assertEqual(calibrated[0], calibrated[1])  # both sit below eps, so the clip ties them
        self.assertFalse(np.array_equal(pvf.midranks(raw), pvf.midranks(calibrated)))
        self.assertEqual(
            pvf.ranking_evidence(raw, calibrated),
            {"clipped_rows": 3, "interior_saturation_ties": 0, "split_ties": 0, "inversions": 0, "interior_ranking_equal": True},
        )

    def test_the_interval_is_open_at_eps(self) -> None:
        raw = np.asarray([PROBABILITY_EPS, 0.3, 1.0 - PROBABILITY_EPS])
        self.assertEqual(pvf.ranking_evidence(raw, self.calibration.apply(raw))["clipped_rows"], 2)

    def test_an_inversion_is_never_excused_even_among_clipped_rows(self) -> None:
        raw = np.asarray([1e-9, 1e-8, 0.2, 0.5])
        evidence = pvf.ranking_evidence(raw, np.asarray([0.30, 0.10, 0.40, 0.60]))
        self.assertEqual((evidence["clipped_rows"], evidence["interior_ranking_equal"], evidence["inversions"]), (2, True, 1))

    def test_float64_saturation_ties_inside_the_interval_are_counted_not_inversions(self) -> None:
        raw = np.asarray([0.3, 0.999998, 0.9999985])  # all strictly inside (eps, 1 - eps)
        calibrated = PlattCalibration(slope=50.0, intercept=0.0).apply(raw)
        self.assertEqual(calibrated[1], calibrated[2])  # the sigmoid has run out of float64 resolution
        evidence = pvf.ranking_evidence(raw, calibrated)
        self.assertEqual((evidence["clipped_rows"], evidence["interior_saturation_ties"], evidence["inversions"], evidence["split_ties"]), (0, 1, 0, 0))
        self.assertFalse(evidence["interior_ranking_equal"])

    def test_a_raw_tie_that_comes_apart_is_counted_as_a_split(self) -> None:
        evidence = pvf.ranking_evidence(np.asarray([0.2, 0.2, 0.6]), np.asarray([0.25, 0.26, 0.7]))
        self.assertEqual((evidence["split_ties"], evidence["inversions"]), (1, 0))


class ProtocolSourceStateTests(unittest.TestCase):
    """`protocol_source_state` against a real, throwaway git repository."""

    def setUp(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is not available")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "test")
        self.git("config", "core.autocrlf", "false")
        (self.repo / "tracked.py").write_bytes(b"first = 1\nsecond = 2\n")
        (self.repo / "other.py").write_bytes(b"x = 0\n")
        self.git("add", "tracked.py", "other.py")
        self.git("commit", "-q", "-m", "initial")

    def git(self, *arguments: str) -> str:
        return subprocess.run(["git", *arguments], cwd=self.repo, capture_output=True, text=True, check=True).stdout

    def state(self, *files: str) -> list[str]:
        return pvf.protocol_source_state(self.repo, files)

    def test_a_committed_file_is_clean_and_a_modified_missing_or_untracked_one_is_not(self) -> None:
        self.assertEqual(self.state("tracked.py", "other.py"), [])
        (self.repo / "tracked.py").write_bytes(b"first = 1\nsecond = 3\n")
        self.assertEqual(self.state("tracked.py", "other.py"), ["M tracked.py"])
        self.assertEqual(self.state("other.py"), [])  # only the listed files count
        (self.repo / "new.py").write_bytes(b"y = 1\n")
        self.assertEqual(self.state("new.py"), ["?? new.py"])
        self.assertEqual(self.state("absent.py"), ["missing: absent.py"])
        self.git("add", "tracked.py")
        self.assertEqual(self.state("tracked.py"), ["M  tracked.py".strip()])  # staged but uncommitted is still not committed

    def test_a_line_ending_only_rewrite_under_autocrlf_is_not_reported_as_a_change(self) -> None:
        self.git("config", "core.autocrlf", "true")
        (self.repo / "tracked.py").write_bytes(b"first = 1\r\nsecond = 2\r\n")
        state = self.state("tracked.py")
        if state:
            # Some git builds report the file until its index entry is refreshed; the refusal message names this remedy.
            self.git("add", "tracked.py")
            state = self.state("tracked.py")
            if state:
                self.skipTest(f"this git build keeps reporting a CRLF-only rewrite: {state}")
        self.assertEqual(state, [])
        self.assertEqual(self.git("diff", "--stat", "HEAD"), "")

    def test_the_refusal_names_the_remedy(self) -> None:
        with patch.object(pvf, "protocol_source_state", return_value=["M src/rehab24/pose_video_fusion.py"]):
            with self.assertRaises(SystemExit) as refusal:
                pvf.require_committed_protocol(pvf.Expectations(), "fit")
        self.assertIn("git add <file>", str(refusal.exception))


class CohortTestCase(unittest.TestCase):
    dims = {pvf.POSE_BRANCH: 6, pvf.VIDEO_BRANCH: 4}

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(clear_feature_cache)
        self.root = Path(self.tmp.name)
        self.cohort = build_cohort(self.root, np.random.default_rng(11), self.dims)
        patcher = patch.object(loso, "MIN_VAL_SUBJECT_SAMPLES", SMALL_VAL_THRESHOLD)
        patcher.start()
        self.addCleanup(patcher.stop)

    def scored_rows(self, rng: np.random.Generator, seeds=pvf.SEEDS) -> list[dict]:
        rows = []
        for seed in seeds:
            for sample_id, row in self.cohort["manifest_rows"].items():
                rows.append(
                    {
                        "sample_id": sample_id,
                        "repetition_id": vbc.repetition_key(row),
                        "session_id": vbc.session_key(row),
                        "person_id": row["person_id"],
                        "exercise_id": row["exercise_id"],
                        "camera": row["camera"],
                        "seed": seed,
                        "label": int(row["correctness"]),
                        "probability": float(rng.uniform(0.1, 0.9)),
                        "threshold": 0.5,
                        "test_subject": row["person_id"],
                        "val_subject": "1",
                    }
                )
        return rows


class PositionJoinTests(CohortTestCase):
    def test_the_cohort_has_the_designed_session_split(self) -> None:
        counts = pvf.manifest_session_counts(self.cohort["manifest_rows"])
        self.assertEqual((counts["mixed_sessions"], counts["interleaved_sessions"]), (18, 9))
        self.assertEqual(counts["interleaved_by_subject"], {subject: 1 for subject in PRIMARY})

    def test_the_statistic_covers_exactly_the_interleaved_sessions(self) -> None:
        rows = self.scored_rows(np.random.default_rng(1))
        statistic = pvf.position_balanced_statistic(rows, self.cohort["manifest_rows"], pvf.SEEDS, 18, 9)
        self.assertEqual((statistic["n_sessions"], statistic["n_subjects"]), (9, 9))

    def test_the_within_session_statistic_asserts_its_session_count(self) -> None:
        rows = self.scored_rows(np.random.default_rng(1))
        statistic = pvf.within_session_statistic(rows, self.cohort["manifest_rows"], pvf.SEEDS, 18)
        self.assertEqual((statistic["n_sessions"], sorted(statistic["per_subject_auc"])), (18, PRIMARY))
        with self.assertRaises(ValueError):
            pvf.within_session_statistic(rows, self.cohort["manifest_rows"], pvf.SEEDS, 19)
        dropped = [row for row in rows if row["session_id"] != "2_V5b"]
        with self.assertRaises(ValueError):
            pvf.within_session_statistic(dropped, self.cohort["manifest_rows"], pvf.SEEDS, 18)
        self.assertEqual(pvf.EXPECTED_MIXED_SESSIONS, 61)

    def test_a_wrong_session_count_raises_instead_of_shrinking(self) -> None:
        rows = self.scored_rows(np.random.default_rng(1))
        with self.assertRaises(ValueError):
            pvf.position_balanced_statistic(rows, self.cohort["manifest_rows"], pvf.SEEDS, 18, 10)
        dropped = [row for row in rows if row["session_id"] != "1_V3a"]
        with self.assertRaises(ValueError):
            pvf.position_balanced_statistic(dropped, self.cohort["manifest_rows"], pvf.SEEDS, 18, 9)

    def test_an_unmatched_or_contradicting_row_raises(self) -> None:
        rows = self.scored_rows(np.random.default_rng(1))
        stranger = {**rows[0], "sample_id": "Ex9_nowhere_rep1_cam17"}
        with self.assertRaises(ValueError):
            pvf.join_to_manifest(rows + [stranger], self.cohort["manifest_rows"])
        with self.assertRaises(ValueError):
            pvf.join_to_manifest([{**rows[0], "label": 1 - rows[0]["label"]}], self.cohort["manifest_rows"])
        with self.assertRaises(ValueError):
            pvf.position_correlation_diagnostic(rows + [stranger], self.cohort["manifest_rows"], pvf.SEEDS)

    def test_the_join_overrides_a_foreign_session_key(self) -> None:
        rows = self.scored_rows(np.random.default_rng(1))
        respelled = [{**row, "session_id": f"Ex{row['session_id']}", "repetition_id": "x"} for row in rows]
        joined = pvf.join_to_manifest(respelled, self.cohort["manifest_rows"])
        self.assertEqual([row["session_id"] for row in joined], [row["session_id"] for row in rows])
        self.assertEqual([row["repetition_id"] for row in joined], [row["repetition_id"] for row in rows])

    def test_the_position_diagnostic_needs_every_primary_repetition_scored(self) -> None:
        rows = self.scored_rows(np.random.default_rng(1))
        diagnostic = pvf.position_correlation_diagnostic(rows, self.cohort["manifest_rows"], pvf.SEEDS)
        self.assertEqual(diagnostic["n_cells"], 36)  # 18 sessions x 2 classes, 3 repetitions each
        self.assertEqual(sorted(diagnostic["per_subject_median"]), PRIMARY)
        missing = [row for row in rows if row["repetition_id"] != "V2a_rep1"]
        with self.assertRaises(ValueError):
            pvf.position_correlation_diagnostic(missing, self.cohort["manifest_rows"], pvf.SEEDS)


class FeatureIntegrityTests(CohortTestCase):
    def audit(self) -> dict:
        clear_feature_cache()
        return pvf.feature_integrity(self.cohort["feature_dirs"][pvf.POSE_BRANCH], self.cohort["manifest_rows"], self.dims[pvf.POSE_BRANCH])

    def test_a_complete_directory_passes(self) -> None:
        self.assertTrue(self.audit()["passed"])

    def test_the_diff_is_by_id_not_by_count(self) -> None:
        directory = self.cohort["feature_dirs"][pvf.POSE_BRANCH]
        victim = next(iter(self.cohort["manifest_rows"]))
        (directory / f"{victim}.npz").rename(directory / "stranger.npz")
        audit = self.audit()
        self.assertEqual(audit["found_ids"], audit["expected_ids"])
        self.assertEqual((audit["missing"], audit["unexpected"]), ([victim], ["stranger"]))
        self.assertFalse(audit["passed"])
        self.assertTrue(audit["incomplete_camera_pairs"])

    def test_wrong_dimension_non_finite_and_duplicates_fail(self) -> None:
        directory = self.cohort["feature_dirs"][pvf.POSE_BRANCH]
        ids = list(self.cohort["manifest_rows"])
        np.savez(directory / f"{ids[0]}.npz", video_feature=np.zeros(3, dtype=np.float32))
        np.savez(directory / f"{ids[1]}.npz", video_feature=np.full(6, np.nan, dtype=np.float32))
        (directory / "copy").mkdir()
        np.savez(directory / "copy" / f"{ids[2]}.npz", video_feature=np.zeros(6, dtype=np.float32))
        audit = self.audit()
        self.assertEqual(audit["wrong_dim"], {ids[0]: 3})
        self.assertEqual(audit["non_finite"], [ids[1]])
        self.assertEqual(audit["duplicates"], [ids[2]])
        self.assertFalse(audit["passed"])


def write_rows(path: Path, rows: list[dict]) -> None:
    vbc.write_oof_csv(path, [{**row, "probability": repr(row["probability"]), "threshold": repr(row["threshold"])} for row in rows])


def trainer_source() -> bytes:
    return Path(loso.__file__).read_bytes()


def perturbed_trainer() -> "pvf.TrainerBaseline":
    """The trainer with a tripled learning rate: what an edit that is *not* inert looks like."""
    source = trainer_source()
    marker = b"lr=config.lr,"
    assert source.count(marker) == 1
    return pvf.trainer_from_source(source.replace(marker, b"lr=config.lr * 3,"), "perturbed")


#: One built fixture per distinct ``build`` recipe, shared by every class that uses it.
_FIXTURES: dict[object, dict] = {}


def tearDownModule() -> None:
    for fixture in _FIXTURES.values():
        fixture["_tmp"].cleanup()
    _FIXTURES.clear()
    clear_feature_cache()


class PipelineTestCase(unittest.TestCase):
    """One full synthetic run per build recipe; each test tampers with its own copy of it."""

    dims = {pvf.POSE_BRANCH: 6, pvf.VIDEO_BRANCH: 4}
    SHARED = ("_tmp", "root", "cohort", "source", "expectations", "device", "baseline", "template", "fuse_result")

    @classmethod
    def setUpClass(cls) -> None:
        recipe = cls.build.__func__
        if recipe not in _FIXTURES:
            cls.construct()
            _FIXTURES[recipe] = {name: getattr(cls, name) for name in cls.SHARED}
        for name, value in _FIXTURES[recipe].items():
            setattr(cls, name, value)

    @classmethod
    def construct(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.cohort = build_cohort(cls.root, np.random.default_rng(11), cls.dims)
        cls.source = cls.root / "source_run"
        with patch.object(loso, "MIN_VAL_SUBJECT_SAMPLES", SMALL_VAL_THRESHOLD):
            vbc.run_init(cls.source, cls.cohort["manifest_path"], cls.cohort["labels_path"], ["vm16"])
        rule = position_rule_loso(cls.cohort["segmentation_path"])
        cls.expectations = pvf.Expectations(
            folds_sha256=pvf.file_sha256(vbc.folds_path(cls.source)),
            folds_no_p10_sha256=pvf.file_sha256(vbc.folds_path(cls.source, no_p10=True)),
            manifest_md5=vbc.file_md5(cls.cohort["manifest_path"]),
            source_run_dir=cls.source,
            arm_feature_dirs=dict(cls.cohort["feature_dirs"]),
            arm_dims=dict(cls.dims),
            fold_config=asdict(FAST_CONFIG),
            baseline_trainer_commit="working-tree",
            baseline_trainer_blob_sha256=hashlib.sha256(trainer_source()).hexdigest(),
            audit_trigger=1.0,  # out of reach: the template run is one whose leak audit does not fire
            mixed_sessions=18,
            interleaved_sessions=9,
            position_rule_ba=round(rule["mean"], 4),
            require_clean_source=False,
        )
        cls.device = torch.device("cpu")
        cls.baseline = pvf.trainer_from_source(trainer_source(), "working-tree")
        cls.template = cls.root / "run_template"
        cls.build(cls.template)

    @classmethod
    def init(cls, run_dir: Path, expectations: "pvf.Expectations | None" = None) -> dict:
        return pvf.run_init(
            run_dir, cls.cohort["manifest_path"], cls.cohort["labels_path"], cls.cohort["segmentation_path"], expectations or cls.expectations
        )

    @classmethod
    def build(cls, run_dir: Path) -> None:
        cls.init(run_dir)
        pvf.run_feature_integrity(run_dir)
        pvf.run_ab_check(run_dir, pvf.BRANCHES, cls.device, baseline=cls.baseline, expectations=cls.expectations)
        for branch in pvf.BRANCHES:
            pvf.run_fit(run_dir, branch, cls.device, expectations=cls.expectations)
        cls.fuse_result = pvf.run_fuse(run_dir, expectations=cls.expectations)

    def fresh_run(self) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        run_dir = Path(directory.name) / "run"
        shutil.copytree(self.template, run_dir)
        return run_dir

    def gate(self, run_dir: Path, name: str, expectations: "pvf.Expectations | None" = None) -> dict:
        return pvf.compute_gates(run_dir, expectations or self.expectations, write=False)["gates"][name]

    def retuned(self, run_dir: Path, trigger: float, clear_threshold: float) -> "pvf.Expectations":
        """The same run under other audit thresholds: `run_config` and the expectations move together."""
        config = pvf.read_json(vbc.run_config_path(run_dir))
        config["leak_audit"] = {**config["leak_audit"], "trigger": trigger, "clear_threshold": clear_threshold}
        pvf.write_json(vbc.run_config_path(run_dir), config)
        return replace(self.expectations, audit_trigger=trigger, audit_clear_threshold=clear_threshold)


class InitTests(PipelineTestCase):
    @classmethod
    def build(cls, run_dir: Path) -> None:  # these tests need no fitted run
        cls.fuse_result = None

    def test_init_refuses_a_hash_that_moved(self) -> None:
        run_dir = self.root / "moved"
        with self.assertRaises(SystemExit):
            self.init(run_dir, replace(self.expectations, folds_sha256="0" * 64))
        self.assertFalse(vbc.run_config_path(run_dir).exists())

    def test_init_refuses_labels_that_disagree_with_the_frozen_manifest(self) -> None:
        labels = dict(self.cohort["labels"])
        victim = next(iter(labels))
        labels[victim] = 1 - labels[victim]
        path = self.root / "flipped.json"
        path.write_text(json.dumps(labels), encoding="utf-8")
        with self.assertRaises(SystemExit):
            pvf.run_init(self.root / "flipped", self.cohort["manifest_path"], path, self.cohort["segmentation_path"], self.expectations)

    def test_the_frozen_constants_are_the_plan_s(self) -> None:
        self.assertEqual(pvf.SEEDS, (42, 7, 1234))
        self.assertEqual(pvf.PRACTICAL_MARGIN, 0.02)
        self.assertEqual(pvf.ARM_DIMS, {"nlf": 2160, "vm16": 768})
        self.assertEqual(pvf.WEIGHT_GRID, (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
        self.assertEqual((vbc.BOOTSTRAP_RESAMPLES, vbc.BOOTSTRAP_SEED), (10_000, 20260918))
        self.assertEqual(pvf.ARM_SHARES, {"fused": 0.5, "nlf_cal": 1.0, "vm16_cal": 0.0})
        self.assertEqual(pvf.MAIN_TABLE_ARMS, ("nlf", "vm16", "fused", "nlf_cal"))
        self.assertEqual(len(pvf.GATE_NAMES), 8)

    def test_the_readout_recipe_is_pinned_as_literals_and_the_trainer_still_matches_it(self) -> None:
        recipe = {
            "epochs": 20,
            "batch_size": 32,
            "lr": 0.0003,
            "hidden_dim": 128,
            "dropout": 0.4,
            "weight_decay": 0.01,
            "early_stopping_patience": 5,
            "normalize_features": True,
            "threshold_objective": "balanced_accuracy",
        }
        self.assertEqual(pvf.FROZEN_FOLD_CONFIG, recipe)
        self.assertEqual(pvf.Expectations().fold_config, recipe)
        self.assertEqual(asdict(loso.FoldConfig()), recipe)

    def test_a_drifted_trainer_default_fails_the_frozen_protocol_gate(self) -> None:
        run_dir = self.root / "drifted"
        self.init(run_dir)
        self.assertEqual(pvf.gate_frozen_protocol(run_dir, self.expectations)["status"], "pass")
        with patch.object(loso.FoldConfig, "__init__", lambda config: config.__dict__.update({**pvf.FROZEN_FOLD_CONFIG, "epochs": 21})):
            gate = pvf.gate_frozen_protocol(run_dir, self.expectations)
        self.assertEqual(gate["status"], "fail")
        self.assertTrue(any("differ from the frozen recipe" in problem for problem in gate["evidence"]["problems"]))

    def test_protocol_files_cover_every_project_module_the_run_imports(self) -> None:
        code = (
            "import sys, pathlib, src.rehab24.pose_video_fusion as m\n"
            "root = pathlib.Path(m.REPO_ROOT)\n"
            "print('\\n'.join(sorted(pathlib.Path(v.__file__).resolve().relative_to(root).as_posix() "
            "for k, v in sys.modules.items() if (k == 'src' or k.startswith('src.')) and getattr(v, '__file__', None))))"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=pvf.REPO_ROOT, capture_output=True, text=True, check=True)
        imported = set(result.stdout.split())
        self.assertIn("src/rehab24/videomae_stage_a.py", imported)
        self.assertEqual(imported - set(pvf.PROTOCOL_FILES), set())
        self.assertIn("scripts/rehab24/run_pose_video_fusion.py", pvf.PROTOCOL_FILES)
        self.assertIn(pvf.PLAN_PATH, pvf.PROTOCOL_FILES)
        self.assertEqual([name for name in pvf.PROTOCOL_FILES if not (pvf.REPO_ROOT / name).exists()], [])

    def test_fuse_refuses_before_its_gates_and_the_report_withholds_outcomes(self) -> None:
        run_dir = self.root / "unfitted"
        self.init(run_dir)
        with self.assertRaises(SystemExit):
            pvf.run_fuse(run_dir, expectations=self.expectations)
        summary = pvf.run_report(run_dir, self.expectations)
        self.assertEqual(summary["verdict"]["row"], 0)
        self.assertNotIn("primary", summary)
        self.assertNotIn("primary", pvf.read_json(vbc.summary_path(run_dir)))


class BaselineTrainerTests(PipelineTestCase):
    @classmethod
    def build(cls, run_dir: Path) -> None:  # only `init`: every test here runs its own A/B cells
        cls.init(run_dir)
        cls.fuse_result = None

    def cell(self, baseline: "pvf.TrainerBaseline") -> dict:
        config = pvf.read_json(vbc.run_config_path(self.template))
        fold = vbc.load_folds(vbc.folds_path(self.template))[0]
        return pvf.ab_check_fold(
            self.cohort["feature_dirs"][pvf.POSE_BRANCH], fold, self.cohort["labels"], loso.FoldConfig(**config["fold_config"]), self.device, 42, baseline
        )

    def test_the_a_side_is_a_separate_module_with_its_provenance(self) -> None:
        self.assertIsNot(self.baseline.module, loso)
        self.assertIsNot(self.baseline.module.train_one_fold, loso.train_one_fold)
        cell = self.cell(self.baseline)
        self.assertTrue(cell["identical"])
        self.assertEqual(cell["baseline_trainer"], {"commit": "working-tree", "blob_sha256": self.baseline.blob_sha256})
        self.assertEqual(cell["device"], "cpu")
        self.assertIn("dirty", cell["source"])

    def test_a_perturbed_old_trainer_fails_the_cell_and_the_gate(self) -> None:
        self.assertFalse(self.cell(perturbed_trainer())["identical"])
        run_dir = self.fresh_run()
        perturbed = perturbed_trainer()
        pvf.run_ab_check(run_dir, [pvf.POSE_BRANCH], self.device, seeds=(42,), test_subjects=["1"], baseline=perturbed, expectations=self.expectations)
        expected = replace(self.expectations, baseline_trainer_commit="perturbed", baseline_trainer_blob_sha256=perturbed.blob_sha256)
        gate = self.gate(run_dir, "inert_trainer_change", expected)
        self.assertEqual(gate["status"], "fail")
        self.assertEqual(gate["evidence"]["cells_not_identical"], ["nlf|P1|seed42"])

    def test_the_pinned_trainer_is_read_from_git_and_predates_the_export(self) -> None:
        available = subprocess.run(
            ["git", "cat-file", "-e", f"{pvf.BASELINE_TRAINER_COMMIT}:{pvf.TRAINER_PATH}"], cwd=pvf.REPO_ROOT, capture_output=True
        )
        if available.returncode != 0:
            self.skipTest("the plan commit is not in this clone")
        baseline = pvf.load_baseline_trainer()
        self.assertEqual((baseline.commit, baseline.blob_sha256), (pvf.BASELINE_TRAINER_COMMIT, pvf.BASELINE_TRAINER_BLOB_SHA256))
        self.assertNotIn("return_validation", inspect.signature(baseline.module.train_one_fold).parameters)
        self.assertIn("return_validation", inspect.signature(loso.train_one_fold).parameters)
        self.assertTrue(self.cell(baseline)["identical"])


class EndToEndTests(PipelineTestCase):
    def test_every_gate_passes_and_no_score_leaves_fuse(self) -> None:
        gates = pvf.compute_gates(self.fresh_run(), self.expectations)
        self.assertEqual({name: gates["gates"][name]["status"] for name in pvf.GATE_NAMES}, {name: "pass" for name in pvf.GATE_NAMES})
        self.assertEqual(gates["gates"]["position_reproduction"]["evidence"]["arms"][pvf.FUSED_ARM]["n_sessions"], 9)
        self.assertEqual(gates["gates"]["degenerate_share_identity"]["evidence"]["clipped_rows_total"], 0)
        self.assertEqual(gates["gates"]["frozen_protocol"]["evidence"]["executions_checked"], 60 + 6 + 1)
        self.assertEqual(self.fuse_result["cells"], len(pvf.SEEDS) * 10)
        self.assertNotIn("balanced_accuracy", json.dumps(self.fuse_result))

    def test_a_fitted_run_may_not_be_re_stamped(self) -> None:
        with self.assertRaises(SystemExit):
            self.init(self.fresh_run())

    def test_validation_rows_sit_beside_the_held_out_oof_never_inside_it(self) -> None:
        held_out = vbc.load_arm_oof(self.template, pvf.POSE_BRANCH)
        validation = pvf.load_val_oof(self.template, pvf.POSE_BRANCH)
        self.assertTrue(all(row["person_id"] == row["test_subject"] for row in held_out))
        self.assertTrue(all(row["person_id"] == row["val_subject"] for row in validation))
        self.assertEqual(len(held_out), len(pvf.SEEDS) * len(self.cohort["manifest_rows"]))

    def test_every_fit_is_stamped_with_source_device_and_recipe(self) -> None:
        entry = pvf.read_json(pvf.fit_log_path(self.template, pvf.VIDEO_BRANCH))["seeds"]["42"]
        self.assertEqual((entry["device"], entry["fold_config"]), ("cpu", asdict(FAST_CONFIG)))
        self.assertEqual(sorted(entry["source"]), ["dirty", "git_commit"])

    def test_the_report_keeps_the_main_table_to_the_registered_arms(self) -> None:
        summary = pvf.run_report(self.fresh_run(), self.expectations)
        self.assertIn(summary["verdict"]["row"], (1, 2, 3, 4))
        self.assertEqual(summary["primary"]["n_subjects"], 9)
        self.assertEqual(sorted(summary["holm"]), sorted(summary["secondary"]))
        self.assertEqual(len(summary["secondary"]), 4)
        self.assertEqual(sorted(summary["main_table"]), sorted(["nlf", "vm16", "fused", "nlf_cal", "position_rule_reference", "fused_below_position_rule"]))
        self.assertEqual(
            sorted(summary["descriptive"]["other_arms_ba"]), sorted(["vm16_cal", "fused_tuned", "fused_perm101", "fused_perm202", "fused_perm303"])
        )
        self.assertEqual(sorted(summary["position_control"]["probability_vs_position"]), sorted(["nlf", "vm16", "fused"]))


def all_keys(payload: object) -> set[str]:
    if isinstance(payload, dict):
        return set(payload) | {key for value in payload.values() for key in all_keys(value)}
    if isinstance(payload, list):
        return {key for value in payload for key in all_keys(value)}
    return set()


#: Keys that would mean a score of an arm of the experiment has been written while outcomes are withheld.
OUTCOME_KEYS = {
    "primary", "secondary", "holm", "main_table", "position_control", "descriptive",
    "fused_perm_minus_nlf", "balanced_accuracy", "per_subject_auc", "candidate", "baseline", "candidate_mean", "baseline_mean",
}


class LeakAuditTests(PipelineTestCase):
    """The template run's audit does not fire (trigger 1.0). Firing and clearing are forced by re-fusing under other thresholds."""

    def audit_files(self, run_dir: Path) -> list[str]:
        return sorted(path.relative_to(run_dir / "oof_audit").as_posix() for path in (run_dir / "oof_audit").glob("*/seed*.csv"))

    def test_the_cli_has_no_override(self) -> None:
        subparsers = next(action for action in pvf.build_parser()._actions if isinstance(action.choices, dict))
        options = {option for action in subparsers.choices["report"]._actions for option in action.option_strings}
        self.assertEqual(options, {"-h", "--help", "--run-id", "--run-root"})
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            pvf.build_parser().parse_args(["report", "--run-id", "x", "--force"])
        self.assertEqual(list(inspect.signature(pvf.run_report).parameters), ["run_dir", "expectations"])

    def test_a_trigger_that_does_not_fire_needs_no_audit(self) -> None:
        run_dir = self.fresh_run()
        audit = pvf.read_json(pvf.leak_audit_path(run_dir))
        self.assertEqual((audit["trigger"]["fired"], audit["audit"], audit["released"]), (False, None, True))
        self.assertEqual(sorted(audit["trigger"]["runs"]), ["101", "202", "303"])
        self.assertFalse((run_dir / "oof_audit").exists())
        self.assertEqual(self.fuse_result["leak_audit"], {"fired": False, "released": True})
        summary = pvf.run_report(run_dir, self.expectations)
        self.assertEqual(summary["verdict"]["leak_audit"], "trigger did not fire")
        self.assertIn("primary", summary)

    def test_a_fired_audit_that_clears_releases_the_outcomes(self) -> None:
        run_dir = self.fresh_run()
        expectations = self.retuned(run_dir, trigger=-1.0, clear_threshold=1.0)
        self.assertEqual(pvf.run_fuse(run_dir, expectations=expectations)["leak_audit"], {"fired": True, "released": True})

        files = self.audit_files(run_dir)
        self.assertEqual(files[:3], [f"fused_const/seed{seed}.csv" for seed in sorted(pvf.SEEDS, key=str)])
        self.assertEqual(len(files), 3 + 20)
        self.assertIn("fused_perm1002/seed7.csv", files)
        self.assertIn("fused_perm1020/seed7.csv", files)
        self.assertEqual({row["seed"] for row in vbc.read_oof_csv(run_dir / "oof_audit" / "fused_perm1003" / "seed1234.csv")}, {1234})

        audit = pvf.read_json(pvf.leak_audit_path(run_dir))
        self.assertEqual(sorted(audit["audit"]["runs"], key=int), [str(seed) for seed in range(1001, 1021)])
        self.assertEqual([audit["audit"]["runs"][str(seed)]["model_seed"] for seed in (1001, 1002, 1003, 1004)], [42, 7, 1234, 42])
        deltas = [run["fused_perm_minus_fused_const"]["mean_delta"] for run in audit["audit"]["runs"].values()]
        self.assertAlmostEqual(audit["audit"]["statistic"], float(np.mean(deltas)), places=15)
        self.assertEqual((audit["audit"]["cleared"], audit["audit"]["clear_threshold"]), (True, 1.0))
        self.assertEqual(all_keys(audit) & OUTCOME_KEYS, set())

        summary = pvf.run_report(run_dir, expectations)
        self.assertEqual(summary["verdict"]["leak_audit"], "audit cleared")
        control = summary["descriptive"]["uninformative_branch_control"]
        self.assertIn("fused_perm_minus_nlf", control["trigger"]["runs"]["101"])  # released, so the experiment's own arm may appear
        self.assertEqual(control["audit"]["statistic"], audit["audit"]["statistic"])

    @staticmethod
    def balanced_accuracy_by_subject(rows: list[dict], seed: int) -> dict[str, float]:
        """Balanced accuracy of one seed's rows per held-out subject, written out here so the pairing is checked independently."""
        scores = {}
        for subject in PRIMARY:
            cell = [row for row in rows if row["test_subject"] == subject and row["seed"] == seed]
            labels = np.asarray([row["label"] for row in cell])
            predicted = np.asarray([row["probability"] >= row["threshold"] for row in cell])
            scores[subject] = (predicted[labels == 1].mean() + (~predicted[labels == 0]).mean()) / 2
        return scores

    def mean_delta(self, candidate: list[dict], candidate_seed: int, baseline: list[dict], baseline_seed: int) -> float:
        first = self.balanced_accuracy_by_subject(candidate, candidate_seed)
        second = self.balanced_accuracy_by_subject(baseline, baseline_seed)
        return float(np.mean([first[subject] - second[subject] for subject in PRIMARY]))

    def test_each_audit_run_is_measured_against_fused_const_of_its_own_model_seed(self) -> None:
        run_dir = self.fresh_run()
        pvf.run_fuse(run_dir, expectations=self.retuned(run_dir, trigger=-1.0, clear_threshold=1.0))
        stored = pvf.read_json(pvf.leak_audit_path(run_dir))["audit"]
        constant = [row for seed in pvf.SEEDS for row in vbc.read_oof_csv(pvf.audit_oof_seed_path(run_dir, pvf.CONSTANT_ARM, seed))]
        own_seed, fixed_seed = {}, {}
        for permutation_seed in pvf.AUDIT_PERMUTATION_SEEDS:
            model_seed = pvf.audit_model_seed(permutation_seed)
            permuted = vbc.read_oof_csv(pvf.audit_oof_seed_path(run_dir, f"fused_perm{permutation_seed}", model_seed))
            own_seed[permutation_seed] = self.mean_delta(permuted, model_seed, constant, model_seed)
            fixed_seed[permutation_seed] = self.mean_delta(permuted, model_seed, constant, 42)
            self.assertAlmostEqual(stored["runs"][str(permutation_seed)]["fused_perm_minus_fused_const"]["mean_delta"], own_seed[permutation_seed], places=12)
        self.assertAlmostEqual(stored["statistic"], float(np.mean(list(own_seed.values()))), places=12)
        # The check above has teeth only if another pairing gives another number.
        self.assertGreater(abs(float(np.mean(list(fixed_seed.values()))) - stored["statistic"]), 1e-6)

    def test_each_registered_run_is_measured_against_nlf_cal_of_its_paired_model_seed(self) -> None:
        run_dir = self.fresh_run()
        stored = pvf.read_json(pvf.leak_audit_path(run_dir))["trigger"]
        calibrated = vbc.load_arm_oof(run_dir, pvf.POSE_CALIBRATED_ARM)
        paired, pooled = [], []
        for permutation_seed, model_seed in pvf.PERMUTATION_PAIRS:
            permuted = vbc.load_arm_oof(run_dir, f"fused_perm{permutation_seed}")
            self.assertEqual({row["seed"] for row in permuted}, {model_seed})
            paired.append(self.mean_delta(permuted, model_seed, calibrated, model_seed))
            pooled.append(float(np.mean([self.mean_delta(permuted, model_seed, calibrated, seed) for seed in pvf.SEEDS])))
            self.assertAlmostEqual(stored["runs"][str(permutation_seed)]["fused_perm_minus_nlf_cal"]["mean_delta"], paired[-1], places=12)
        self.assertAlmostEqual(stored["mean_fused_perm_minus_nlf_cal"], float(np.mean(paired)), places=12)
        # The three runs cover each model seed once, so pooling nlf_cal over seeds leaves the MEAN unchanged
        # by arithmetic; it is the per-run deltas above that pin the pairing, and they must be able to differ.
        self.assertAlmostEqual(float(np.mean(pooled)), float(np.mean(paired)), places=12)
        self.assertGreater(max(abs(a - b) for a, b in zip(paired, pooled)), 1e-6)

    def test_an_edit_to_a_branch_file_makes_the_audit_stale(self) -> None:
        inputs = pvf.audit_input_hashes(self.template)
        for name in ("oof/nlf/seed42.csv", "oof/vm16/seed1234.csv", "oof_val/nlf/seed7.csv", "oof_val/vm16/seed42.csv", "calibration.json", "oof/nlf_cal/seed7.csv"):
            self.assertIn(name, inputs)
        for relative in ("oof/vm16/seed7.csv", "oof_val/nlf/seed42.csv"):
            with self.subTest(file=relative):
                run_dir = self.fresh_run()
                self.assertTrue(pvf.leak_audit_state(run_dir, self.expectations)["released"])
                with (run_dir / relative).open("ab") as handle:
                    handle.write(b"\n")  # same rows, another file
                state = pvf.leak_audit_state(run_dir, self.expectations)
                self.assertEqual((state["released"], "stale" in state["reason"]), (False, True))

    def test_fuse_writes_no_held_out_score_and_the_report_computes_brier_from_the_oof(self) -> None:
        run_dir = self.fresh_run()
        calibration = pvf.read_json(pvf.calibration_path(run_dir))
        self.assertEqual(set(calibration), {"execution", "shares", "negative_slopes", "folds"})
        allowed = {"test_subject", "seed", "fitted_on", "scored_on", "platt", "thresholds", "tuned_pose_share", "permuted_video_slope", "permuted_fitted_on"}
        self.assertEqual({key for record in calibration["folds"] for key in record} - allowed, set())
        self.assertEqual({key for key in all_keys(calibration) if "brier" in key or "accuracy" in key or "auc" in key}, set())

        brier = pvf.run_report(run_dir, self.expectations)["descriptive"]["platt"]["brier_held_out"]
        self.assertEqual(sorted(brier), sorted(pvf.BRANCHES))
        rows = vbc.load_arm_oof(run_dir, pvf.POSE_CALIBRATED_ARM)
        by_seed = [
            np.mean([(row["probability"] - row["label"]) ** 2 for row in rows if row["test_subject"] == "3" and row["seed"] == seed]) for seed in pvf.SEEDS
        ]
        self.assertAlmostEqual(brier["nlf"]["calibrated"]["per_subject"]["3"], float(np.mean(by_seed)), places=12)
        self.assertEqual(sorted(brier["vm16"]["raw"]["per_subject"]), PRIMARY)

    def test_a_fired_audit_that_does_not_clear_is_row_0_and_withheld_on_disk(self) -> None:
        run_dir = self.fresh_run()
        expectations = self.retuned(run_dir, trigger=-1.0, clear_threshold=-1.0)
        self.assertEqual(pvf.run_fuse(run_dir, expectations=expectations)["leak_audit"], {"fired": True, "released": False})
        self.assertTrue(pvf.compute_gates(run_dir, expectations, write=False)["passed"])  # a leak is not a gate failure; it is caught anyway
        for summary in (pvf.run_report(run_dir, expectations), pvf.read_json(vbc.summary_path(run_dir))):
            self.assertEqual(summary["verdict"]["row"], 0)
            self.assertIn("treated as leaking", summary["outcomes"])
            self.assertFalse(summary["leak_audit"]["audit"]["cleared"])
            self.assertEqual(all_keys(summary["leak_audit"]) & OUTCOME_KEYS, set())
            self.assertEqual(set(summary), {"gates", "verdict", "outcomes", "leak_audit"})

    def test_the_clearing_rule_is_at_most_the_threshold(self) -> None:
        run_dir = self.fresh_run()
        pvf.run_fuse(run_dir, expectations=self.retuned(run_dir, trigger=-1.0, clear_threshold=1.0))
        statistic = pvf.read_json(pvf.leak_audit_path(run_dir))["audit"]["statistic"]
        self.assertTrue(pvf.audit_statistic(run_dir, statistic)["cleared"])  # statistic == threshold clears
        self.assertFalse(pvf.audit_statistic(run_dir, float(np.nextafter(statistic, -1.0)))["cleared"])
        trigger = pvf.uninformative_trigger(run_dir, 0.0)["mean_fused_perm_minus_nlf_cal"]
        self.assertFalse(pvf.uninformative_trigger(run_dir, trigger)["fired"])  # the trigger needs strictly more
        self.assertTrue(pvf.uninformative_trigger(run_dir, float(np.nextafter(trigger, -1.0)))["fired"])

    def test_a_re_fuse_invalidates_the_audit_it_replaces(self) -> None:
        run_dir = self.fresh_run()
        stale = pvf.read_json(pvf.leak_audit_path(run_dir))
        pvf.run_fuse(run_dir, expectations=self.expectations)
        fresh = pvf.read_json(pvf.leak_audit_path(run_dir))
        self.assertNotEqual(stale["inputs"], fresh["inputs"])
        self.assertIn("primary", pvf.run_report(run_dir, self.expectations))

        pvf.write_json(pvf.leak_audit_path(run_dir), stale)
        summary = pvf.run_report(run_dir, self.expectations)
        self.assertEqual(summary["verdict"]["row"], 0)
        self.assertIn("stale", summary["outcomes"])
        self.assertNotIn("primary", pvf.read_json(vbc.summary_path(run_dir)))

        pvf.leak_audit_path(run_dir).unlink()
        self.assertIn("missing", pvf.run_report(run_dir, self.expectations)["outcomes"])

    def test_a_hand_edited_audit_record_releases_nothing(self) -> None:
        run_dir = self.fresh_run()
        expectations = self.retuned(run_dir, trigger=-1.0, clear_threshold=-1.0)
        pvf.run_fuse(run_dir, expectations=expectations)
        record = pvf.read_json(pvf.leak_audit_path(run_dir))
        record["released"] = True
        record["audit"]["cleared"] = True
        pvf.write_json(pvf.leak_audit_path(run_dir), record)
        self.assertEqual(pvf.run_report(run_dir, expectations)["verdict"]["row"], 0)

    def test_a_fuse_of_the_p10_sensitivity_leaves_the_audit_alone(self) -> None:
        run_dir = self.fresh_run()
        for branch in pvf.BRANCHES:
            pvf.run_fit(run_dir, branch, self.device, seeds=pvf.SEEDS, no_p10_training=True, expectations=self.expectations)
        before = pvf.read_json(pvf.leak_audit_path(run_dir))
        self.assertNotIn("leak_audit", pvf.run_fuse(run_dir, no_p10_training=True, expectations=self.expectations))
        self.assertEqual(pvf.read_json(pvf.leak_audit_path(run_dir)), before)
        self.assertIn("primary", pvf.run_report(run_dir, self.expectations))


class EndToEndDeviationTests(PipelineTestCase):
    def test_a_stored_run_mismatch_is_logged_and_never_blocks(self) -> None:
        run_dir = self.fresh_run()
        source = run_dir.parent / "source_copy"
        shutil.copytree(self.source, source)
        stored = vbc.read_oof_csv(vbc.oof_seed_path(self.template, pvf.VIDEO_BRANCH, 42))
        write_rows(vbc.oof_seed_path(source, pvf.VIDEO_BRANCH, 42), stored)
        config = pvf.read_json(vbc.run_config_path(run_dir))
        pvf.write_json(vbc.run_config_path(run_dir), {**config, "source_run_dir": str(source)})

        pvf.run_ab_check(run_dir, [pvf.VIDEO_BRANCH], self.device, seeds=(42,), test_subjects=["1"], baseline=self.baseline, expectations=self.expectations)
        cell = pvf.read_json(pvf.ab_check_path(run_dir))["cells"]["vm16|P1|seed42"]
        self.assertTrue(cell["stored_run"]["bit_identical"])
        self.assertEqual(pvf.read_json(pvf.deviations_path(run_dir)), [])

        first = next(index for index, row in enumerate(stored) if row["test_subject"] == "1")
        stored[first] = {**stored[first], "probability": stored[first]["probability"] + 1e-3}
        write_rows(vbc.oof_seed_path(source, pvf.VIDEO_BRANCH, 42), stored)
        for _ in range(2):  # a repeated mismatch replaces its entry, it does not pile up
            pvf.run_ab_check(run_dir, [pvf.VIDEO_BRANCH], self.device, seeds=(42,), test_subjects=["1"], baseline=self.baseline, expectations=self.expectations)
        entries = pvf.read_json(pvf.deviations_path(run_dir))
        self.assertEqual([entry["key"] for entry in entries], ["stored_vm16_oof_not_reproduced|vm16|P1|seed42"])
        gate = self.gate(run_dir, "inert_trainer_change")
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["evidence"]["stored_20260918_vm16_non_blocking"]["not_bit_identical"], ["vm16|P1|seed42"])

    def test_the_historical_baseline_check_logs_a_miss_and_never_raises(self) -> None:
        run_dir = self.fresh_run()
        folds = [{"test_subject": subject, "balanced_accuracy": 0.6 + int(subject) / 100, "threshold": 0.5} for subject in PRIMARY]
        stored, rerun = run_dir.parent / "stored.json", run_dir.parent / "rerun.json"
        pvf.write_json(stored, {"folds": folds})
        pvf.write_json(rerun, {"folds": folds})
        self.assertTrue(pvf.historical_baseline_check(run_dir, rerun, stored)["reproduced"])
        self.assertEqual(pvf.read_json(pvf.deviations_path(run_dir)), [])

        pvf.write_json(rerun, {"folds": [{**folds[0], "threshold": 0.4}, *folds[1:]]})
        report = pvf.historical_baseline_check(run_dir, rerun, stored)
        self.assertEqual((report["reproduced"], report["blocking"]), (False, False))
        self.assertEqual([entry["key"] for entry in pvf.read_json(pvf.deviations_path(run_dir))], ["historical_nlf_baseline_not_reproduced"])
        self.assertEqual(self.gate(run_dir, "frozen_protocol")["status"], "pass")


class GateFailureTests(PipelineTestCase):
    """One failing case per blocking gate, each on its own copy of a run that passes all eight."""

    def test_frozen_protocol_fails_on_a_hand_edited_run_config(self) -> None:
        for key, value in (
            ("leak_audit", {"trigger": 0.05, "clear_threshold": 0.01, "permutation_seeds": list(range(1001, 1021)), "model_seeds": []}),
            ("baseline_trainer", {"commit": "working-tree", "blob_sha256": "0" * 64, "path": pvf.TRAINER_PATH}),
            ("seeds", [42, 7, 99]),
            ("practical_margin", 0.01),
            ("source_run_dir", "elsewhere"),
            ("fold_config", {**asdict(FAST_CONFIG), "epochs": 3}),
            ("arm_feature_dirs", {"nlf": "a", "vm16": "b"}),
        ):
            with self.subTest(key=key):
                run_dir = self.fresh_run()
                config = pvf.read_json(vbc.run_config_path(run_dir))
                pvf.write_json(vbc.run_config_path(run_dir), {**config, key: value})
                gate = pvf.gate_frozen_protocol(run_dir, self.expectations)
                self.assertEqual(gate["status"], "fail")
                self.assertTrue(any(f"run_config {key}" in problem for problem in gate["evidence"]["problems"]))

    @staticmethod
    def clean_stamps(run_dir: Path) -> dict:
        """Rewrite every stamp as clean and on the commit `init` recorded, whatever state this checkout is in."""
        config = {**pvf.read_json(vbc.run_config_path(run_dir)), "source_dirty": []}
        pvf.write_json(vbc.run_config_path(run_dir), config)
        clean = {"git_commit": config["git_commit"], "dirty": []}
        for branch in pvf.BRANCHES:
            log = pvf.read_json(pvf.fit_log_path(run_dir, branch))
            for entry in log["seeds"].values():
                entry["source"] = dict(clean)
            pvf.write_json(pvf.fit_log_path(run_dir, branch), log)
        cells = pvf.read_json(pvf.ab_check_path(run_dir))
        for cell in cells["cells"].values():
            cell["source"] = dict(clean)
        pvf.write_json(pvf.ab_check_path(run_dir), cells)
        calibration = pvf.read_json(pvf.calibration_path(run_dir))
        calibration["execution"]["source"] = dict(clean)
        pvf.write_json(pvf.calibration_path(run_dir), calibration)
        return config

    def test_frozen_protocol_fails_on_a_dirty_tree_another_commit_or_another_device(self) -> None:
        strict = replace(self.expectations, require_clean_source=True)
        tampering = {
            "uncommitted protocol files": lambda entry, config: entry["source"].update(dirty=["M src/rehab24/pose_video_fusion.py"]),
            "init recorded": lambda entry, config: entry["source"].update(git_commit="0" * 40),
            "ran on device 'cuda'": lambda entry, config: entry.update(device="cuda"),
        }
        for message, tamper in tampering.items():
            for target in ("fit", "ab-check"):
                with self.subTest(message=message, target=target):
                    run_dir = self.fresh_run()
                    config = self.clean_stamps(run_dir)
                    self.assertEqual(pvf.gate_frozen_protocol(run_dir, strict)["status"], "pass")
                    path = pvf.fit_log_path(run_dir, pvf.POSE_BRANCH) if target == "fit" else pvf.ab_check_path(run_dir)
                    payload = pvf.read_json(path)
                    tamper(next(iter((payload["seeds"] if target == "fit" else payload["cells"]).values())), config)
                    pvf.write_json(path, payload)
                    gate = pvf.gate_frozen_protocol(run_dir, strict)
                    self.assertEqual((gate["status"], gate["evidence"]["n_problems"]), ("fail", 1))
                    self.assertIn(message, gate["evidence"]["problems"][0])
                    self.assertTrue(gate["evidence"]["problems"][0].startswith(target))

    def test_frozen_protocol_fails_on_a_fit_or_cell_without_a_stamp(self) -> None:
        cases = {}
        run_dir = self.fresh_run()
        pvf.fit_log_path(run_dir, pvf.POSE_BRANCH).unlink()
        cases["a missing fit log"] = (run_dir, "fit nlf seed42: OOF rows exist without a fit-log stamp")

        run_dir = self.fresh_run()
        log = pvf.read_json(pvf.fit_log_path(run_dir, pvf.VIDEO_BRANCH))
        del log["seeds"]["7"]
        pvf.write_json(pvf.fit_log_path(run_dir, pvf.VIDEO_BRANCH), log)
        cases["a seed the log does not know"] = (run_dir, "fit vm16 seed7: OOF rows exist without a fit-log stamp")

        run_dir = self.fresh_run()
        shutil.copytree(vbc.oof_dir(run_dir, pvf.POSE_BRANCH), vbc.oof_dir(run_dir, "nlf_nop10"))
        pvf.fit_log_path(run_dir, "nlf_nop10").unlink()
        cases["an unlogged P10-removed refit"] = (run_dir, "fit nlf_nop10 seed1234: OOF rows exist without a fit-log stamp")

        run_dir = self.fresh_run()
        payload = pvf.read_json(pvf.ab_check_path(run_dir))
        del payload["cells"]["vm16|P2|seed7"]["source"]
        pvf.write_json(pvf.ab_check_path(run_dir), payload)
        cases["an unstamped A/B cell"] = (run_dir, "ab-check vm16|P2|seed7: cell carries no stamp")

        for name, (run_dir, message) in cases.items():
            with self.subTest(case=name):
                gate = pvf.gate_frozen_protocol(run_dir, self.expectations)
                self.assertEqual(gate["status"], "fail")
                self.assertTrue(any(problem.startswith(message) for problem in gate["evidence"]["problems"]), gate["evidence"]["problems"])

    def test_every_writing_command_refuses_a_dirty_tree_or_another_head(self) -> None:
        strict = replace(self.expectations, require_clean_source=True)
        commands = {
            "init": lambda run_dir: self.init(run_dir.parent / "fresh", strict),
            "fit": lambda run_dir: pvf.run_fit(run_dir, pvf.POSE_BRANCH, self.device, seeds=(42,), expectations=strict),
            "ab-check": lambda run_dir: pvf.run_ab_check(
                run_dir, [pvf.POSE_BRANCH], self.device, seeds=(42,), test_subjects=["1"], baseline=self.baseline, expectations=strict
            ),
            "fuse": lambda run_dir: pvf.run_fuse(run_dir, expectations=strict),
        }
        for name, command in commands.items():
            run_dir = self.fresh_run()
            before = {path: path.stat().st_mtime_ns for path in run_dir.rglob("*") if path.is_file()}
            with self.subTest(command=name, reason="dirty"), patch.object(pvf, "protocol_source_state", return_value=["M src/rehab24/pose_video_fusion.py"]):
                with self.assertRaises(SystemExit) as refusal:
                    command(run_dir)
                self.assertIn("not committed", str(refusal.exception))
            if name != "init":  # `init` has no recorded commit to differ from
                with self.subTest(command=name, reason="head"), patch.object(pvf, "protocol_source_state", return_value=[]), patch.object(
                    vbc, "git_commit_hash", return_value="f" * 40
                ):
                    with self.assertRaises(SystemExit) as refusal:
                        command(run_dir)
                    self.assertIn("`init` recorded", str(refusal.exception))
            self.assertEqual({path: path.stat().st_mtime_ns for path in run_dir.rglob("*") if path.is_file()}, before)
            self.assertFalse((run_dir.parent / "fresh").exists())

    def test_a_clean_tree_on_the_recorded_head_is_not_refused(self) -> None:
        strict = replace(self.expectations, require_clean_source=True)
        with patch.object(pvf, "protocol_source_state", return_value=[]):
            pvf.require_committed_protocol(strict, "fit", self.fresh_run())
            pvf.require_committed_protocol(strict, "init")

    def test_fuse_refuses_a_fitted_run_whose_pre_fuse_gate_fails(self) -> None:
        def edit_config(run_dir: Path) -> None:
            pvf.write_json(vbc.run_config_path(run_dir), {**pvf.read_json(vbc.run_config_path(run_dir)), "practical_margin": 0.05})

        def fail_features(run_dir: Path) -> None:
            audit = pvf.read_json(pvf.feature_integrity_path(run_dir))
            audit[pvf.VIDEO_BRANCH]["passed"] = False
            pvf.write_json(pvf.feature_integrity_path(run_dir), audit)

        def break_cell(run_dir: Path) -> None:
            payload = pvf.read_json(pvf.ab_check_path(run_dir))
            payload["cells"]["nlf|P4|seed1234"]["identical"] = False
            pvf.write_json(pvf.ab_check_path(run_dir), payload)

        def drop_validation_row(run_dir: Path) -> None:
            path = pvf.val_oof_seed_path(run_dir, pvf.VIDEO_BRANCH, 42)
            write_rows(path, vbc.read_oof_csv(path)[1:])

        tampering = dict(zip(pvf.PRE_FUSE_GATES, (edit_config, fail_features, break_cell, drop_validation_row)))
        self.assertEqual(len(tampering), 4)
        for gate, tamper in tampering.items():
            with self.subTest(gate=gate):
                run_dir = self.fresh_run()
                pvf.run_fuse(run_dir, expectations=self.expectations)  # this run fuses; only the tampering below can stop it
                tamper(run_dir)
                calibration = pvf.calibration_path(run_dir).read_bytes()
                with self.assertRaises(SystemExit) as refusal:
                    pvf.run_fuse(run_dir, expectations=self.expectations)
                self.assertIn(f"gates not passed: ['{gate}']", str(refusal.exception))
                self.assertEqual(pvf.calibration_path(run_dir).read_bytes(), calibration)
                self.assertTrue(pvf.leak_audit_path(run_dir).exists())

    def test_inert_trainer_change_fails_on_a_cell_against_another_blob(self) -> None:
        run_dir = self.fresh_run()
        payload = pvf.read_json(pvf.ab_check_path(run_dir))
        payload["cells"]["vm16|P9|seed7"]["baseline_trainer"]["blob_sha256"] = "0" * 64
        pvf.write_json(pvf.ab_check_path(run_dir), payload)
        gate = self.gate(run_dir, "inert_trainer_change")
        self.assertEqual(gate["status"], "fail")
        self.assertEqual(gate["evidence"]["cells_against_another_baseline"], ["vm16|P9|seed7"])
        self.assertEqual(gate["evidence"]["pinned_baseline"]["blob_sha256"], self.expectations.baseline_trainer_blob_sha256)
        self.assertEqual(pvf.Expectations().baseline_trainer_blob_sha256, pvf.BASELINE_TRAINER_BLOB_SHA256)

    def test_degenerate_share_fails_on_an_inversion_or_a_split_tie_but_not_on_a_counted_tie(self) -> None:
        run_dir = self.fresh_run()
        clean = {"clipped_rows": 0, "interior_saturation_ties": 0, "split_ties": 0, "inversions": 0, "interior_ranking_equal": True}
        for name, evidence, status, message in (
            ("inversion", {**clean, "inversions": 2, "interior_ranking_equal": False}, "fail", "calibration inverts 2 pairs"),
            ("split tie", {**clean, "split_ties": 1, "interior_ranking_equal": False}, "fail", "separates 1 raw ties"),
            ("clip and saturation ties", {**clean, "clipped_rows": 3, "interior_saturation_ties": 4, "interior_ranking_equal": False}, "pass", None),
        ):
            with self.subTest(case=name), patch.object(pvf, "ranking_evidence", return_value=evidence):
                gate = pvf.gate_degenerate_share(run_dir)
                self.assertEqual(gate["status"], status)
                if message:
                    self.assertTrue(gate["evidence"]["problems"])
                    self.assertTrue(all(message in problem for problem in gate["evidence"]["problems"]))
                else:
                    checked = gate["evidence"]["rankings_checked"]
                    self.assertEqual((gate["evidence"]["clipped_rows_total"], gate["evidence"]["interior_saturation_ties_total"]), (3 * checked, 4 * checked))
                    self.assertEqual(len(gate["evidence"]["interior_saturation_ties_by_arm_fold_seed_split"]), checked)

    def test_degenerate_share_names_every_cell_it_skips_for_a_non_positive_slope(self) -> None:
        run_dir = self.fresh_run()
        calibration = pvf.read_json(pvf.calibration_path(run_dir))
        natural = sum(calibration["negative_slopes"].values())
        gate = pvf.gate_degenerate_share(run_dir)
        self.assertEqual(len(gate["evidence"]["cells_skipped_nonpositive_slope"]), natural)
        self.assertEqual(gate["evidence"]["rankings_checked"], 2 * (2 * len(calibration["folds"]) - natural))

        record = next(record for record in calibration["folds"] if float(record["platt"][pvf.POSE_BRANCH]["slope"]) > 0)
        record["platt"][pvf.POSE_BRANCH]["slope"] = "-0.5"
        pvf.write_json(pvf.calibration_path(run_dir), calibration)
        skipped = pvf.gate_degenerate_share(run_dir)["evidence"]["cells_skipped_nonpositive_slope"]
        self.assertIn(f"nlf|P{record['test_subject']} seed{record['seed']}|slope=-0.5", skipped)
        self.assertEqual(len(skipped), natural + 1)

    def test_feature_integrity_fails_when_a_bundle_is_missing(self) -> None:
        run_dir = self.fresh_run()
        features = run_dir.parent / "features_nlf"
        shutil.copytree(self.cohort["feature_dirs"][pvf.POSE_BRANCH], features)
        next(features.glob("*.npz")).unlink()
        config = pvf.read_json(vbc.run_config_path(run_dir))
        pvf.write_json(vbc.run_config_path(run_dir), {**config, "arm_feature_dirs": {**config["arm_feature_dirs"], pvf.POSE_BRANCH: str(features)}})
        clear_feature_cache()
        pvf.run_feature_integrity(run_dir)
        gate = pvf.gate_feature_integrity(run_dir)
        self.assertEqual(gate["status"], "fail")
        self.assertEqual(len(gate["evidence"][pvf.POSE_BRANCH]["missing"]), 1)

    def test_inert_trainer_change_fails_on_a_cell_against_another_baseline(self) -> None:
        run_dir = self.fresh_run()
        payload = pvf.read_json(pvf.ab_check_path(run_dir))
        payload["cells"]["nlf|P1|seed42"]["baseline_trainer"]["commit"] = "some-other-commit"
        pvf.write_json(pvf.ab_check_path(run_dir), payload)
        gate = self.gate(run_dir, "inert_trainer_change")
        self.assertEqual(gate["status"], "fail")
        self.assertEqual(gate["evidence"]["cells_against_another_baseline"], ["nlf|P1|seed42"])

    def test_inert_trainer_change_is_pending_while_cells_are_missing(self) -> None:
        run_dir = self.fresh_run()
        payload = pvf.read_json(pvf.ab_check_path(run_dir))
        del payload["cells"]["vm16|P3|seed7"]
        pvf.write_json(pvf.ab_check_path(run_dir), payload)
        self.assertEqual(self.gate(run_dir, "inert_trainer_change")["status"], "pending")

    def test_validation_export_fails_on_a_leaked_row_and_fuse_raises(self) -> None:
        run_dir = self.fresh_run()
        path = pvf.val_oof_seed_path(run_dir, pvf.POSE_BRANCH, 42)
        rows = vbc.read_oof_csv(path)
        rows[0] = {**rows[0], "person_id": rows[0]["test_subject"]}
        write_rows(path, rows)
        self.assertEqual(pvf.gate_validation_export(run_dir)["status"], "fail")
        with self.assertRaises(pvf.FoldPurityError):
            pvf.run_fuse(run_dir, enforce_gates=False, expectations=self.expectations)

    def test_validation_export_fails_when_the_rows_are_not_the_best_checkpoint_s(self) -> None:
        run_dir = self.fresh_run()
        path = pvf.val_oof_seed_path(run_dir, pvf.VIDEO_BRANCH, 7)
        rows = vbc.read_oof_csv(path)
        write_rows(path, [{**row, "probability": 1.0 - row["probability"]} for row in rows])
        gate = pvf.gate_validation_export(run_dir)
        self.assertEqual(gate["status"], "fail")
        self.assertTrue(any("threshold from exported validation probabilities" in problem for problem in gate["evidence"]))

    def test_fold_purity_fails_when_the_recorded_fit_rows_are_not_the_fold_s(self) -> None:
        tampering = [
            ("fit on the held-out subject", "the held-out subject reached a fit", lambda record: record["fitted_on"].update(person_ids=[record["test_subject"]])),
            ("fit on another row set", "not the validation ids", lambda record: record["fitted_on"].update(ids_sha256="0" * 64)),
            ("scored another row set", "not exactly the held-out ids", lambda record: record["scored_on"].update(ids_sha256="0" * 64)),
            ("permuted control fit on two subjects", "calibrated on subjects ['3', '4']", lambda record: record["permuted_fitted_on"].update(person_ids=["3", "4"])),
        ]
        for name, message, tamper in tampering:
            with self.subTest(case=name):
                run_dir = self.fresh_run()
                payload = pvf.read_json(pvf.calibration_path(run_dir))
                tamper(payload["folds"][0])
                pvf.write_json(pvf.calibration_path(run_dir), payload)
                gate = pvf.gate_fold_purity(run_dir)
                self.assertEqual(gate["status"], "fail")
                self.assertTrue(any(message in problem for problem in gate["evidence"]["problems"]), gate["evidence"])

    def test_fold_purity_reads_the_fold_file_not_the_record(self) -> None:
        run_dir = self.fresh_run()
        payload = pvf.read_json(vbc.folds_path(run_dir))
        payload["folds"][0]["val_ids"] = payload["folds"][0]["val_ids"][1:]  # still disjoint, so the structural check passes
        pvf.write_json(vbc.folds_path(run_dir), payload)
        self.assertEqual(vbc.gate_fold_purity(run_dir)["status"], "pass")
        gate = pvf.gate_fold_purity(run_dir)
        self.assertEqual(gate["status"], "fail")
        self.assertTrue(all("P1 " in problem and "not the validation ids" in problem for problem in gate["evidence"]["problems"]))

    def test_degenerate_share_fails_when_nlf_cal_is_not_the_calibrated_branch(self) -> None:
        run_dir = self.fresh_run()
        path = vbc.oof_seed_path(run_dir, pvf.POSE_CALIBRATED_ARM, 42)
        rows = vbc.read_oof_csv(path)
        rows[0] = {**rows[0], "probability": float(np.nextafter(rows[0]["probability"], 1.0))}
        write_rows(path, rows)
        gate = pvf.gate_degenerate_share(run_dir)
        self.assertEqual(gate["status"], "fail")
        self.assertIn("bit for bit", gate["evidence"]["problems"][0])

    def test_fused_oof_integrity_fails_on_a_missing_or_foreign_row(self) -> None:
        manifest_rows, labels = self.cohort["manifest_rows"], self.cohort["labels"]
        for name, tamper in (
            ("missing", lambda rows: rows[1:]),
            ("duplicate", lambda rows: rows + [rows[0]]),
            ("foreign fold", lambda rows: [{**rows[0], "test_subject": rows[0]["val_subject"]}] + rows[1:]),
        ):
            with self.subTest(case=name):
                run_dir = self.fresh_run()
                path = vbc.oof_seed_path(run_dir, pvf.FUSED_ARM, 1234)
                write_rows(path, tamper(vbc.read_oof_csv(path)))
                gate = pvf.gate_fused_oof_integrity(run_dir, manifest_rows, labels)
                self.assertEqual(gate["status"], "fail")
                self.assertTrue(gate["evidence"][pvf.FUSED_ARM])

    def test_position_reproduction_fails_on_a_wrong_count_or_a_moved_rule(self) -> None:
        run_dir = self.fresh_run()
        for expectations in (replace(self.expectations, interleaved_sessions=10), replace(self.expectations, position_rule_ba=0.7309)):
            gate = self.gate(run_dir, "position_reproduction", expectations)
            self.assertEqual(gate["status"], "fail")
            self.assertTrue(gate["evidence"]["problems"])

    def test_position_reproduction_fails_when_an_arm_loses_a_session(self) -> None:
        run_dir = self.fresh_run()
        for seed in pvf.SEEDS:
            path = vbc.oof_seed_path(run_dir, pvf.FUSED_ARM, seed)
            write_rows(path, [row for row in vbc.read_oof_csv(path) if row["session_id"] != "1_V4a"])
        gate = self.gate(run_dir, "position_reproduction")
        self.assertEqual(gate["status"], "fail")
        self.assertTrue(any(problem.startswith("fused:") for problem in gate["evidence"]["problems"]))

    def test_any_failed_gate_puts_the_report_on_row_0(self) -> None:
        run_dir = self.fresh_run()
        path = vbc.oof_seed_path(run_dir, pvf.FUSED_ARM, 42)
        write_rows(path, vbc.read_oof_csv(path)[1:])
        summary = pvf.run_report(run_dir, self.expectations)
        self.assertEqual(summary["verdict"]["row"], 0)
        self.assertNotIn("primary", pvf.read_json(vbc.summary_path(run_dir)))


if __name__ == "__main__":
    unittest.main()
