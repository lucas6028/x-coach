"""Unit tests for the VideoMAE-vs-V-JEPA-2 evaluation harness (REHAB24-6).

Everything here runs on synthetic manifests and tiny random feature bundles inside a
``TemporaryDirectory`` -- no real feature bundle exists yet (the extraction half is
built in parallel). ``MIN_VAL_SUBJECT_SAMPLES`` is patched down for the small
synthetic cohorts used by most tests; the real REHAB24-6 threshold (100) is only
exercised by the CLI acceptance run against the real manifest, done separately.
"""

from __future__ import annotations

import csv
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# src.rehab24.loso_cross_validation exits at import time without torch, which the lean
# CI dependency set does not install; without this the whole module is a collection
# error rather than a skip.
torch = pytest.importorskip("torch")

from src.rehab24 import loso_cross_validation as loso
from src.rehab24 import video_backbone_comparison as vbc
from src.rehab24.dataset import CAMERAS, MANIFEST_FIELDS, load_manifest


def write_manifest_and_labels(
    root: Path,
    subjects: list[str],
    reps_per_subject: int,
    rng: np.random.Generator,
) -> tuple[Path, Path, dict[str, list[str]]]:
    """A small synthetic REHAB24-6-shaped manifest: two camera rows per repetition."""
    rows: list[dict[str, str]] = []
    labels: dict[str, int] = {}
    subject_samples: dict[str, list[str]] = {subject: [] for subject in subjects}

    for subject in subjects:
        for rep in range(reps_per_subject):
            video_id = f"S{subject}R{rep}"
            exercise_id = "1" if rep % 2 == 0 else "2"
            correctness = int(rng.integers(0, 2))
            for camera in CAMERAS:
                sample_id = f"P{subject}_{video_id}_{camera}"
                row = {field: "" for field in MANIFEST_FIELDS}
                row.update(
                    {
                        "sample_id": sample_id,
                        "split": "train",
                        "video_id": video_id,
                        "repetition_number": "1",
                        "exercise_id": exercise_id,
                        "exercise_name": "test",
                        "person_id": subject,
                        "first_frame": "0",
                        "last_frame": "10",
                        "camera": camera,
                        "camera_orientation": "front",
                        "cam17_orientation": "front",
                        "correctness": str(correctness),
                        "mocap_erroneous": "False",
                        "lights_on": "True",
                        "extra_person_in_camera": "False",
                        "skeleton_3d_path": "x",
                        "skeleton_2d_path": "x",
                        "video_path": "x",
                    }
                )
                rows.append(row)
                labels[sample_id] = correctness
                subject_samples[subject].append(sample_id)

    manifest_path = root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    labels_path = root / "correctness.json"
    with labels_path.open("w", encoding="utf-8") as handle:
        json.dump(labels, handle)

    return manifest_path, labels_path, subject_samples


def write_raw_bundle(
    path: Path,
    sample_id: str,
    video_id: str,
    exercise_id: str,
    person_id: str,
    camera: str,
    correctness: int,
    dim: int,
    rng: np.random.Generator,
    arm: str = "vm16",
    clip_length: str = "16",
    resolution: str = "224",
    model_name: str = "test-model",
    num_clips: int = 2,
    frames_per_clip: int = 4,
    non_finite: bool = False,
    override_dim: int | None = None,
    provenance_overrides: dict[str, str] | None = None,
    frame_indices_override: np.ndarray | None = None,
    first_index_override: int | None = None,
    last_index_override: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clip_features = rng.normal(size=(num_clips, override_dim or dim)).astype(np.float32)
    if non_finite:
        clip_features[0, 0] = np.nan
    if frame_indices_override is not None:
        frame_indices = frame_indices_override
    else:
        frame_indices = np.arange(num_clips * frames_per_clip, dtype=np.int64).reshape(num_clips, frames_per_clip)
    first_index_value = 0 if first_index_override is None else first_index_override
    last_index_value = int(frame_indices.max()) if last_index_override is None else last_index_override

    payload = {
        "sample_id": np.asarray(sample_id),
        "video_id": np.asarray(video_id),
        "exercise_id": np.asarray(exercise_id),
        "person_id": np.asarray(person_id),
        "camera": np.asarray(camera),
        "correctness": np.asarray(int(correctness)),
        "clip_features": clip_features,
        "clip_starts": np.arange(num_clips, dtype=np.int64),
        "frame_indices": frame_indices,
        "unique_frame_count": np.full(num_clips, frames_per_clip, dtype=np.int64),
        "padding_fraction": np.zeros(num_clips, dtype=np.float32),
        "first_index": np.asarray(first_index_value),
        "last_index": np.asarray(last_index_value),
        "total_frames": np.asarray(int(frame_indices.max()) + 1),
        "provenance_model_name": np.asarray(model_name),
        "provenance_revision": np.asarray("rev1"),
        "provenance_arm": np.asarray(arm),
        "provenance_clip_length": np.asarray(clip_length),
        "provenance_frame_stride": np.asarray("2"),
        "provenance_num_clips": np.asarray(str(num_clips)),
        "provenance_resolution": np.asarray(resolution),
        "provenance_pooling": np.asarray("mean"),
        "provenance_output_field": np.asarray("last_hidden_state"),
        "provenance_sampler": np.asarray("uniform"),
        "provenance_variant": np.asarray("full_frame"),
        "provenance_dtype": np.asarray("float32"),
        "provenance_transformers_version": np.asarray("4.0"),
        "provenance_torch_version": np.asarray("2.0"),
        "provenance_code_fingerprint": np.asarray("abc123"),
    }
    for key, value in (provenance_overrides or {}).items():
        payload[key] = np.asarray(value)
    np.savez_compressed(path, **payload)


def write_materialized_bundle(path: Path, sample_id: str, video_feature: np.ndarray, correctness: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        video_feature=video_feature.astype(np.float32),
        sample_id=np.asarray(sample_id),
        correctness=np.asarray(int(correctness)),
    )


class FoldConstructionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        # Subject "4" mirrors P10: too few samples to ever be picked as a val subject.
        self.manifest_path, self.labels_path, self.subject_samples = write_manifest_and_labels(
            self.root, subjects=["1", "2", "3"], reps_per_subject=6, rng=rng
        )
        small_root = self.root / "small_subject"
        small_root.mkdir()
        # Hand-build a P10-style tiny subject on top of the 3 normal ones. The module
        # hardcodes "10" as the sensitivity subject (SENSITIVITY_SUBJECT), so a
        # synthetic stand-in for "too few samples to be a val subject or a training
        # member of the sensitivity fold set" must use that exact id.
        rows = load_manifest(self.manifest_path)
        labels = json.load(self.labels_path.open(encoding="utf-8"))
        extra_rows = []
        for camera in CAMERAS:
            sample_id = f"P10_extra_{camera}"
            row = {field: "" for field in MANIFEST_FIELDS}
            row.update(
                {
                    "sample_id": sample_id,
                    "video_id": "extra",
                    "repetition_number": "1",
                    "exercise_id": "1",
                    "person_id": "10",
                    "camera": camera,
                    "correctness": "1",
                    "video_path": "x",
                    "skeleton_2d_path": "x",
                    "skeleton_3d_path": "x",
                }
            )
            extra_rows.append(row)
            labels[sample_id] = 1
        with self.manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows + extra_rows)
        with self.labels_path.open("w", encoding="utf-8") as handle:
            json.dump(labels, handle)

    def test_val_rule_and_p10_style_exclusion_and_disjointness(self):
        with patch.object(loso, "MIN_VAL_SUBJECT_SAMPLES", 5):
            folds = vbc.build_all_folds(self.manifest_path, drop_p10_from_train=False)

        by_subject = {fold["test_subject"]: fold for fold in folds}
        # "10" has only 2 samples (< 5): never eligible as anyone's val subject.
        for fold in folds:
            self.assertNotEqual(fold["val_subject"], "10")
        # Cyclic rule: test=1 -> next eligible is 2; test=3 -> wraps to 1 (2 has enough too,
        # but 3's own successor going round is 1... check disjointness instead of the exact
        # link, since with 3 eligible subjects any of them could be picked depending on order).
        self.assertIn(by_subject["1"]["val_subject"], {"2", "3"})

        for fold in folds:
            train, val, test = set(fold["train_ids"]), set(fold["val_ids"]), set(fold["test_ids"])
            self.assertFalse(train & val)
            self.assertFalse(train & test)
            self.assertFalse(val & test)
            self.assertNotEqual(fold["val_subject"], fold["test_subject"])

    def test_no_p10_training_variant_strips_tiny_subject_from_every_train_set(self):
        with patch.object(loso, "MIN_VAL_SUBJECT_SAMPLES", 5):
            folds = vbc.build_all_folds(self.manifest_path, drop_p10_from_train=True)

        tiny_ids = {"P10_extra_cam17", "P10_extra_cam18"}
        for fold in folds:
            self.assertFalse(set(fold["train_ids"]) & tiny_ids, f"P10 leaked into training for test={fold['test_subject']}")
            if fold["test_subject"] == "10":
                # Its own test fold is untouched by the training-side exclusion.
                self.assertTrue(set(fold["test_ids"]) & tiny_ids)

    def test_fold_hashes_are_stable_sha256_of_sorted_ids(self):
        with patch.object(loso, "MIN_VAL_SUBJECT_SAMPLES", 5):
            folds = vbc.build_all_folds(self.manifest_path, drop_p10_from_train=False)
        payload = vbc.folds_payload(folds)
        for fold in payload["folds"]:
            self.assertEqual(fold["hashes"]["test"], vbc.hash_ids(fold["test_ids"]))


class ManifestAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_valid_manifest_reports_counts(self):
        rng = np.random.default_rng(1)
        manifest_path, labels_path, _ = write_manifest_and_labels(self.root, ["1", "2"], 3, rng)
        audit = vbc.manifest_audit(manifest_path, labels_path)
        self.assertEqual(audit["rows"], 2 * 3 * 2)
        self.assertEqual(audit["repetitions"], 2 * 3)
        self.assertFalse(audit["matches_expected_rehab24_6"]["rows"])  # small synthetic cohort

    def test_non_binary_label_raises(self):
        rng = np.random.default_rng(2)
        manifest_path, labels_path, _ = write_manifest_and_labels(self.root, ["1"], 2, rng)
        rows = load_manifest(manifest_path)
        rows[0]["correctness"] = "2"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(ValueError):
            vbc.manifest_audit(manifest_path, labels_path)

    def test_incomplete_camera_pair_raises(self):
        rng = np.random.default_rng(3)
        manifest_path, labels_path, _ = write_manifest_and_labels(self.root, ["1"], 2, rng)
        rows = load_manifest(manifest_path)
        rows = [row for row in rows if not (row["video_id"] == "S1R0" and row["camera"] == "cam18")]
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(ValueError):
            vbc.manifest_audit(manifest_path, labels_path)


class MaterializeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rng = np.random.default_rng(4)
        self.manifest_path, self.labels_path, self.subject_samples = write_manifest_and_labels(
            self.root, ["1", "2"], 2, self.rng
        )
        self.run_dir = self.root / "run"
        self.manifest_rows = {row["sample_id"]: row for row in load_manifest(self.manifest_path)}

    def _write_all_bundles(self, arm="vm16", dim=768, **kwargs):
        for sample_id, row in self.manifest_rows.items():
            path = vbc.raw_arm_dir(self.run_dir, arm) / "train" / f"{sample_id}.npz"
            write_raw_bundle(
                path,
                sample_id,
                row["video_id"],
                row["exercise_id"],
                row["person_id"],
                row["camera"],
                int(row["correctness"]),
                dim,
                self.rng,
                arm=arm,
                **kwargs,
            )

    def test_successful_materialize_writes_video_feature_bundles(self):
        self._write_all_bundles()
        audit = vbc.materialize(self.run_dir, "vm16", self.manifest_path)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["written"], len(self.manifest_rows))
        written_paths = list(vbc.features_arm_dir(self.run_dir, "vm16").rglob("*.npz"))
        self.assertEqual(len(written_paths), len(self.manifest_rows))
        with np.load(written_paths[0], allow_pickle=False) as data:
            self.assertEqual(data["video_feature"].shape, (768,))

    def test_missing_sample_refuses(self):
        self._write_all_bundles()
        any_path = next(vbc.raw_arm_dir(self.run_dir, "vm16").rglob("*.npz"))
        any_path.unlink()
        with self.assertRaises(SystemExit):
            vbc.materialize(self.run_dir, "vm16", self.manifest_path)
        audit = json.load(vbc.materialize_audit_path(self.run_dir, "vm16").open(encoding="utf-8"))
        self.assertFalse(audit["passed"])
        self.assertTrue(audit["missing_sample_ids"])

    def test_duplicate_sample_refuses(self):
        self._write_all_bundles()
        arm_dir = vbc.raw_arm_dir(self.run_dir, "vm16")
        any_path = next(arm_dir.rglob("*.npz"))
        duplicate_path = arm_dir / "val" / any_path.name
        duplicate_path.parent.mkdir(parents=True, exist_ok=True)
        duplicate_path.write_bytes(any_path.read_bytes())
        with self.assertRaises(SystemExit):
            vbc.materialize(self.run_dir, "vm16", self.manifest_path)
        audit = json.load(vbc.materialize_audit_path(self.run_dir, "vm16").open(encoding="utf-8"))
        self.assertTrue(audit["duplicate_sample_ids"])

    def test_non_finite_refuses(self):
        sample_id, row = next(iter(self.manifest_rows.items()))
        for sid, r in self.manifest_rows.items():
            path = vbc.raw_arm_dir(self.run_dir, "vm16") / "train" / f"{sid}.npz"
            write_raw_bundle(
                path, sid, r["video_id"], r["exercise_id"], r["person_id"], r["camera"],
                int(r["correctness"]), 768, self.rng, non_finite=(sid == sample_id),
            )
        with self.assertRaises(SystemExit):
            vbc.materialize(self.run_dir, "vm16", self.manifest_path)
        audit = json.load(vbc.materialize_audit_path(self.run_dir, "vm16").open(encoding="utf-8"))
        self.assertIn(sample_id, audit["non_finite_sample_ids"])

    def test_dim_mismatch_refuses(self):
        sample_id, row = next(iter(self.manifest_rows.items()))
        for sid, r in self.manifest_rows.items():
            path = vbc.raw_arm_dir(self.run_dir, "vm16") / "train" / f"{sid}.npz"
            write_raw_bundle(
                path, sid, r["video_id"], r["exercise_id"], r["person_id"], r["camera"],
                int(r["correctness"]), 768, self.rng, override_dim=(5 if sid == sample_id else None),
            )
        with self.assertRaises(SystemExit):
            vbc.materialize(self.run_dir, "vm16", self.manifest_path)
        audit = json.load(vbc.materialize_audit_path(self.run_dir, "vm16").open(encoding="utf-8"))
        self.assertIn(sample_id, audit["dim_mismatches"])

    def test_provenance_disagreement_refuses(self):
        sample_id, row = next(iter(self.manifest_rows.items()))
        for sid, r in self.manifest_rows.items():
            path = vbc.raw_arm_dir(self.run_dir, "vm16") / "train" / f"{sid}.npz"
            overrides = {"provenance_model_name": "different-model"} if sid == sample_id else None
            write_raw_bundle(
                path, sid, r["video_id"], r["exercise_id"], r["person_id"], r["camera"],
                int(r["correctness"]), 768, self.rng, provenance_overrides=overrides,
            )
        with self.assertRaises(SystemExit):
            vbc.materialize(self.run_dir, "vm16", self.manifest_path)
        audit = json.load(vbc.materialize_audit_path(self.run_dir, "vm16").open(encoding="utf-8"))
        self.assertIn("provenance_model_name", audit["provenance_disagreements"])


class OofIntegrityGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(5)
        self.manifest_path, self.labels_path, _ = write_manifest_and_labels(self.root, ["1", "2"], 2, rng)
        self.run_dir = self.root / "run"
        self.manifest_rows = {row["sample_id"]: row for row in load_manifest(self.manifest_path)}

    def _good_rows(self) -> list[dict]:
        rows = []
        for sample_id, row in self.manifest_rows.items():
            rows.append(
                {
                    "sample_id": sample_id,
                    "repetition_id": vbc.repetition_key(row),
                    "session_id": vbc.session_key(row),
                    "person_id": row["person_id"],
                    "exercise_id": row["exercise_id"],
                    "camera": row["camera"],
                    "seed": 42,
                    "label": int(row["correctness"]),
                    "probability": 0.6 if int(row["correctness"]) == 1 else 0.4,
                    "threshold": 0.5,
                    "test_subject": row["person_id"],
                    "val_subject": "other",
                }
            )
        return rows

    def test_well_formed_oof_passes(self):
        vbc.write_oof_csv(vbc.oof_seed_path(self.run_dir, "vm16", 42), self._good_rows())
        gate = vbc.gate_oof_integrity(self.run_dir, self.manifest_path, self.labels_path)
        self.assertEqual(gate["evidence"]["vm16"]["status"], "pass")

    def test_swapped_label_fails(self):
        rows = self._good_rows()
        rows[0] = {**rows[0], "label": 1 - rows[0]["label"]}
        vbc.write_oof_csv(vbc.oof_seed_path(self.run_dir, "vm16", 42), rows)
        gate = vbc.gate_oof_integrity(self.run_dir, self.manifest_path, self.labels_path)
        self.assertEqual(gate["evidence"]["vm16"]["status"], "fail")
        self.assertTrue(any("label disagrees" in problem for problem in gate["evidence"]["vm16"]["evidence"]))

    def test_missing_camera_row_fails(self):
        rows = self._good_rows()
        target_repetition = rows[0]["repetition_id"]
        rows = [row for row in rows if not (row["repetition_id"] == target_repetition and row["camera"] == "cam18")]
        vbc.write_oof_csv(vbc.oof_seed_path(self.run_dir, "vm16", 42), rows)
        gate = vbc.gate_oof_integrity(self.run_dir, self.manifest_path, self.labels_path)
        self.assertEqual(gate["evidence"]["vm16"]["status"], "fail")
        self.assertTrue(any("missing a camera" in problem for problem in gate["evidence"]["vm16"]["evidence"]))


class SignFlipTests(unittest.TestCase):
    def test_hand_computable_three_subject_case_matches_brute_force(self):
        deltas = [0.10, 0.20, -0.05]
        observed = sum(deltas) / 3
        extreme = 0
        total = 0
        for signs in itertools.product((1, -1), repeat=3):
            candidate = sum(s * d for s, d in zip(signs, deltas)) / 3
            total += 1
            if abs(candidate) >= abs(observed) - 1e-12:
                extreme += 1
        expected_p = extreme / total

        result = vbc.sign_flip_p_value(deltas)
        self.assertEqual(result["n_assignments"], 8)
        self.assertAlmostEqual(result["p_value"], expected_p)
        self.assertAlmostEqual(result["observed_mean_delta"], observed)

    def test_all_zero_deltas_give_p_one(self):
        result = vbc.sign_flip_p_value([0.0, 0.0, 0.0, 0.0])
        self.assertEqual(result["p_value"], 1.0)

    def test_all_same_sign_deltas_are_maximally_significant(self):
        # |mean| is maximized by the all-positive AND the all-negative assignment
        # (same magnitude, opposite sign); every other assignment is strictly smaller
        # in absolute value. So exactly 2 of the 8 assignments are "at least as
        # extreme" as the observed all-positive one: p = 2/2^n.
        deltas = [0.1, 0.1, 0.1]
        result = vbc.sign_flip_p_value(deltas)
        self.assertAlmostEqual(result["p_value"], 2 / 8)


class BootstrapDeterminismTests(unittest.TestCase):
    def test_same_seed_gives_identical_interval(self):
        values = [0.01, -0.02, 0.05, 0.0, 0.03, -0.01, 0.04, 0.02, -0.03]
        first = vbc.bootstrap_interval(values, 2000, 20260918)
        second = vbc.bootstrap_interval(values, 2000, 20260918)
        self.assertEqual(first, second)


class HolmTests(unittest.TestCase):
    def test_holm_correction_on_four_p_values(self):
        p_values = {"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.20}
        corrected = vbc.holm_correct(p_values)
        # Ascending order a,b,c,d -> multipliers 4,3,2,1 with a running max.
        self.assertAlmostEqual(corrected["a"]["holm"], 0.004)
        self.assertAlmostEqual(corrected["b"]["holm"], 0.06)
        self.assertAlmostEqual(corrected["c"]["holm"], max(0.06, 0.06))
        self.assertAlmostEqual(corrected["d"]["holm"], max(0.06, 0.20))
        self.assertTrue(corrected["a"]["significant"])
        self.assertFalse(corrected["d"]["significant"])


@unittest.skipIf(torch is None, "torch is required")
class LogisticRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.feature_dir = self.root / "features"

    def _write_separable(self, ids: list[str], labels: dict[str, int], rng) -> None:
        for sample_id in ids:
            label = labels[sample_id]
            feature = rng.normal(scale=0.3, size=6).astype(np.float32)
            feature[0] += 3.0 if label == 1 else -3.0
            write_materialized_bundle(self.feature_dir / f"{sample_id}.npz", sample_id, feature, label)

    def test_recovers_a_separable_problem_and_reports_converged(self):
        rng = np.random.default_rng(6)
        n = 60
        ids = [f"s{i}" for i in range(n)]
        labels = {sample_id: i % 2 for i, sample_id in enumerate(ids)}
        self._write_separable(ids, labels, rng)
        train_ids, val_ids, test_ids = ids[:40], ids[40:50], ids[50:]

        threshold, probabilities, test_labels, sample_ids, fit_log = vbc.train_logistic_fold(
            self.feature_dir, train_ids, val_ids, test_ids, labels, torch.device("cpu")
        )
        self.assertTrue(fit_log["converged"])
        predictions = (probabilities >= threshold).astype(int)
        accuracy = float(np.mean(predictions == test_labels))
        self.assertGreaterEqual(accuracy, 0.9)


class PaddingBucketTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(vbc.padding_bucket(0.0), "0")
        self.assertEqual(vbc.padding_bucket(0.3), "(0, 0.5]")
        self.assertEqual(vbc.padding_bucket(0.5), "(0, 0.5]")
        self.assertEqual(vbc.padding_bucket(0.51), ">0.5")
        self.assertEqual(vbc.padding_bucket(1.0), ">0.5")


class VerdictConditionsTests(unittest.TestCase):
    def test_promising_requires_every_condition(self):
        result = vbc.verdict_conditions(
            mean_delta=0.03, n_positive=8, n_subjects=9, ci=[0.01, 0.05], p_value=0.01, gates_pass=True
        )
        self.assertEqual(result["verdict"], "promising improvement")

    def test_significant_but_below_margin(self):
        result = vbc.verdict_conditions(
            mean_delta=0.01, n_positive=8, n_subjects=9, ci=[0.002, 0.02], p_value=0.01, gates_pass=True
        )
        self.assertEqual(result["verdict"], "significant positive below margin")

    def test_gate_failure_short_circuits_to_its_own_verdict(self):
        # The plan's fifth outcome row: a failed gate means "no valid primary
        # comparison", not a downgraded-but-still-significant reading -- so this must
        # NOT fall through to "significant positive below margin" even though every
        # significance condition here would otherwise qualify.
        result = vbc.verdict_conditions(
            mean_delta=0.03, n_positive=8, n_subjects=9, ci=[0.01, 0.05], p_value=0.01, gates_pass=False
        )
        self.assertEqual(result["verdict"], "gate failure: no valid primary comparison")
        self.assertFalse(result["conditions"]["gates_pass"])

    def test_non_significant_is_undetermined_never_equivalence_language(self):
        result = vbc.verdict_conditions(
            mean_delta=0.001, n_positive=5, n_subjects=9, ci=[-0.01, 0.02], p_value=0.4, gates_pass=True
        )
        self.assertEqual(result["verdict"], "undetermined")


class SamplingGateTests(unittest.TestCase):
    """gate_sampling must check indices against the MANIFEST'S bounds, not the
    bundle's own self-reported first_index/last_index (review defect #1): a buggy
    extractor that overruns the repetition but writes consistent-with-itself bounds
    must still fail."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(11)
        self.manifest_path, self.labels_path, _ = write_manifest_and_labels(self.root, ["1"], reps_per_subject=1, rng=rng)
        rows = load_manifest(self.manifest_path)
        # 1-based annotation frames 1..11 -> manifest-derived decoder bound is 0..10.
        for row in rows:
            row["first_frame"] = "1"
            row["last_frame"] = "11"
        with self.manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self.rows = rows
        self.run_dir = self.root / "run"
        self.rng = rng

    def test_manifest_bound_is_derived_correctly(self):
        bounds = vbc.manifest_frame_bounds(self.manifest_path)
        for row in self.rows:
            self.assertEqual(bounds[row["sample_id"]], (0, 10))

    def test_within_bounds_bundle_passes(self):
        for row in self.rows:
            path = vbc.raw_arm_dir(self.run_dir, "vm16") / "train" / f"{row['sample_id']}.npz"
            write_raw_bundle(
                path, row["sample_id"], row["video_id"], row["exercise_id"], row["person_id"], row["camera"],
                int(row["correctness"]), 768, self.rng,
                frame_indices_override=np.array([[0, 2, 4, 6], [4, 6, 8, 10]], dtype=np.int64),
                first_index_override=0, last_index_override=10,
            )
        gate = vbc.gate_sampling(self.run_dir, self.manifest_path)
        self.assertEqual(gate["evidence"]["vm16"]["status"], "pass")

    def test_overrun_with_self_consistent_stored_bounds_still_fails(self):
        # The extractor's own first_index/last_index (0, 15) are internally
        # consistent with its frame_indices (max 15) -- the OLD, self-referential
        # gate would have passed this. The manifest says the repetition only spans
        # decoder indices 0..10, so this must fail.
        for row in self.rows:
            path = vbc.raw_arm_dir(self.run_dir, "vm16") / "train" / f"{row['sample_id']}.npz"
            write_raw_bundle(
                path, row["sample_id"], row["video_id"], row["exercise_id"], row["person_id"], row["camera"],
                int(row["correctness"]), 768, self.rng,
                frame_indices_override=np.array([[0, 5, 10, 15], [0, 5, 10, 15]], dtype=np.int64),
                first_index_override=0, last_index_override=15,
            )
        gate = vbc.gate_sampling(self.run_dir, self.manifest_path)
        evidence = gate["evidence"]["vm16"]
        self.assertEqual(evidence["status"], "fail")
        self.assertTrue(evidence["evidence"]["out_of_manifest_bounds_sample_ids"])

    def test_stored_bounds_disagreeing_with_manifest_fails_even_if_indices_happen_to_fit(self):
        for row in self.rows:
            path = vbc.raw_arm_dir(self.run_dir, "vm16") / "train" / f"{row['sample_id']}.npz"
            write_raw_bundle(
                path, row["sample_id"], row["video_id"], row["exercise_id"], row["person_id"], row["camera"],
                int(row["correctness"]), 768, self.rng,
                frame_indices_override=np.array([[0, 2, 4, 6], [4, 6, 8, 10]], dtype=np.int64),
                first_index_override=0, last_index_override=20,  # disagrees with manifest's 10
            )
        gate = vbc.gate_sampling(self.run_dir, self.manifest_path)
        evidence = gate["evidence"]["vm16"]
        self.assertEqual(evidence["status"], "fail")
        self.assertTrue(evidence["evidence"]["stored_bounds_disagree_with_manifest_sample_ids"])


class ModelIdentityDimGateTests(unittest.TestCase):
    """gate_model_identity must compare a MEASURED dimension against ARM_DIMS, not
    compare the arm's own declared config against itself (review defect #2)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = Path(self.tmp.name) / "run"

    def _write_audit(self, arm: str, measured_dim: int, provenance: dict | None = None) -> None:
        path = vbc.materialize_audit_path(self.run_dir, arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = vbc.ARM_MODEL_CONFIG[arm]
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "arm": arm,
                    "expected_dim": vbc.ARM_DIMS[arm],
                    "measured_dim": measured_dim,
                    "provenance": dict(expected) if provenance is None else provenance,
                    "problems": [],
                    "passed": True,
                },
                handle,
            )

    def test_prefixed_provenance_keys_as_written_by_real_bundles_pass(self):
        # Real bundles stamp ``provenance_<key>``; the first full run failed this gate
        # on every arm with ``actual: None`` because the test audits above used bare
        # keys and the gate only tried the bare spelling.
        prefixed = {f"provenance_{k}": v for k, v in vbc.ARM_MODEL_CONFIG["vj16"].items()}
        self._write_audit("vj16", vbc.ARM_DIMS["vj16"], provenance=prefixed)
        gate = vbc.gate_model_identity(self.run_dir)
        self.assertEqual(gate["evidence"]["per_arm"]["vj16"]["status"], "pass")

    def test_prefixed_provenance_with_wrong_model_name_fails(self):
        prefixed = {f"provenance_{k}": v for k, v in vbc.ARM_MODEL_CONFIG["vj16"].items()}
        prefixed["provenance_model_name"] = "facebook/vjepa2-vitl-fpc16-256"
        self._write_audit("vj16", vbc.ARM_DIMS["vj16"], provenance=prefixed)
        gate = vbc.gate_model_identity(self.run_dir)
        evidence = gate["evidence"]["per_arm"]["vj16"]
        self.assertEqual(evidence["status"], "fail")
        self.assertIn("model_name", evidence["evidence"])

    def test_measured_dim_matching_expected_passes(self):
        self._write_audit("vm16", vbc.ARM_DIMS["vm16"])
        gate = vbc.gate_model_identity(self.run_dir)
        self.assertEqual(gate["evidence"]["per_arm"]["vm16"]["status"], "pass")

    def test_wrong_measured_dim_fails_even_though_expected_dim_field_agrees_with_itself(self):
        # expected_dim is still written as the correct 768 (an honest echo of config);
        # only the independently MEASURED dimension is wrong. The old check compared
        # expected_dim to ARM_DIMS[arm] -- literally the same value -- so it could
        # never have caught this.
        self._write_audit("vm16", 512)
        gate = vbc.gate_model_identity(self.run_dir)
        evidence = gate["evidence"]["per_arm"]["vm16"]
        self.assertEqual(evidence["status"], "fail")
        self.assertEqual(evidence["evidence"]["dim"]["actual"], 512)
        self.assertEqual(evidence["evidence"]["dim"]["expected"], vbc.ARM_DIMS["vm16"])

    def test_materialize_records_measured_dim_from_real_data(self):
        # End-to-end check that `materialize` itself populates `measured_dim` from the
        # actual array shape (not just an echo of the arm's declared config).
        root = Path(self.tmp.name)
        rng = np.random.default_rng(12)
        manifest_path, labels_path, _ = write_manifest_and_labels(root, ["1"], reps_per_subject=1, rng=rng)
        for row in load_manifest(manifest_path):
            path = vbc.raw_arm_dir(self.run_dir, "vm16") / "train" / f"{row['sample_id']}.npz"
            write_raw_bundle(
                path, row["sample_id"], row["video_id"], row["exercise_id"], row["person_id"], row["camera"],
                int(row["correctness"]), 768, rng, arm="vm16",
            )
        audit = vbc.materialize(self.run_dir, "vm16", manifest_path)
        self.assertEqual(audit["measured_dim"], 768)


class PooledPairwiseAucGuardTests(unittest.TestCase):
    def test_single_class_pool_returns_none_not_nan(self):
        rows = [{"test_subject": "1", "label": 1, "probability": 0.7}, {"test_subject": "1", "label": 1, "probability": 0.3}]
        self.assertIsNone(vbc.pooled_pairwise_auc(rows, ("1",)))

    def test_empty_pool_returns_none(self):
        self.assertIsNone(vbc.pooled_pairwise_auc([], ("1",)))

    def test_mixed_labels_return_a_float(self):
        rows = [
            {"test_subject": "1", "label": 1, "probability": 0.9},
            {"test_subject": "1", "label": 0, "probability": 0.1},
        ]
        result = vbc.pooled_pairwise_auc(rows, ("1",))
        self.assertIsInstance(result, float)


class EndToEndSmokeTest(unittest.TestCase):
    """init -> materialize -> evaluate (1 seed, tiny epochs) -> report, on synthetic data."""

    @unittest.skipIf(torch is None, "torch is required")
    def test_smoke_pipeline_produces_primary_contrast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(7)
            subjects = [str(i) for i in range(1, 10)]
            manifest_path, labels_path, _ = write_manifest_and_labels(root, subjects, reps_per_subject=6, rng=rng)
            run_dir = root / "run"
            manifest_rows = {row["sample_id"]: row for row in load_manifest(manifest_path)}

            with patch.object(loso, "MIN_VAL_SUBJECT_SAMPLES", 10):
                vbc.run_init(run_dir, manifest_path, labels_path, arms=("vm16", "vj64"))

                for arm, dim in (("vm16", 768), ("vj64", 1024)):
                    for sample_id, row in manifest_rows.items():
                        path = vbc.raw_arm_dir(run_dir, arm) / "train" / f"{sample_id}.npz"
                        write_raw_bundle(
                            path, sample_id, row["video_id"], row["exercise_id"], row["person_id"], row["camera"],
                            int(row["correctness"]), dim, rng, arm=arm,
                        )
                    vbc.materialize(run_dir, arm, manifest_path)

                config = loso.FoldConfig(epochs=2, batch_size=16, hidden_dim=16, early_stopping_patience=1)
                device = torch.device("cpu")
                eval_reports = {}
                for arm in ("vm16", "vj64"):
                    eval_reports[arm] = vbc.run_evaluate(
                        run_dir, arm, manifest_path, labels_path, config, device, seeds=[42]
                    )

                summary = vbc.run_report(run_dir)

            self.assertEqual(summary["primary"]["status"], "complete")
            self.assertIn("mean_delta", summary["primary"])
            self.assertIn("bootstrap_ci_95", summary["primary"])
            self.assertIn("sign_flip", summary["primary"])
            self.assertEqual(summary["primary"]["n_subjects"], 9)
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "gates.json").exists())

            # Advisor tip: the CSV round-trip (repr-encoded probability/threshold) must
            # not silently drift the reported BA away from the fold's own BA.
            vm16_folds = {fold["test_subject"]: fold for fold in eval_reports["vm16"]["mlp"][0]["folds"]}
            oof_rows = vbc.load_arm_oof(run_dir, "vm16")
            for subject in vbc.PRIMARY_SUBJECTS:
                rows = [row for row in oof_rows if row["test_subject"] == subject and row["seed"] == 42]
                reported_ba = vbc.fold_metrics(rows)["balanced_accuracy"]
                self.assertAlmostEqual(reported_ba, vm16_folds[subject]["balanced_accuracy"], places=9)


if __name__ == "__main__":
    unittest.main()
