import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.rehab24.videomae_audit import audit_feature_dir
from src.video.videomae_audit import audit_feature_dir as core_audit_feature_dir
from src.video.videomae_audit import load_labeled_ids, load_split_map

MANIFEST_HEADER = "sample_id,split,video_id,exercise_id,person_id,camera,correctness\n"


def write_manifest(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [MANIFEST_HEADER]
    for sample_id, split in rows:
        lines.append(f"{sample_id},{split},PM_000,1,1,cam17,1\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_feature(path: Path, feature: np.ndarray, provenance: dict[str, str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = provenance or {"model_name": "m", "token_pooling": "mean_pool_fc_norm"}
    np.savez_compressed(
        path,
        video_feature=feature,
        clip_features=np.stack([feature, feature], axis=0),
        **{f"provenance_{k}": np.asarray(v) for k, v in provenance.items()},
    )


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.manifest = self.tmp / "manifest.csv"
        self.labels = self.tmp / "correctness.json"
        self.feature_dir = self.tmp / "features"
        write_manifest(self.manifest, [("s1", "train"), ("s2", "train"), ("s3", "test")])
        self.labels.write_text(json.dumps({"s1": 1, "s2": 0, "s3": 1}), encoding="utf-8")
        rng = np.random.default_rng(0)
        self.good = {sid: rng.normal(size=8).astype(np.float32) for sid in ("s1", "s2", "s3")}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_all_good(self):
        for sid, split in (("s1", "train"), ("s2", "train"), ("s3", "test")):
            write_feature(self.feature_dir / split / f"{sid}.npz", self.good[sid])

    def test_clean_dir_passes(self):
        self.write_all_good()
        report = audit_feature_dir(self.feature_dir, self.manifest, self.labels)
        self.assertTrue(report["passed"], report["checks"])
        self.assertEqual(report["n_found"], 3)

    def test_detects_missing_coverage(self):
        write_feature(self.feature_dir / "train" / "s1.npz", self.good["s1"])
        report = audit_feature_dir(self.feature_dir, self.manifest, self.labels)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["coverage_complete"])
        self.assertEqual(report["missing"], ["s2", "s3"])

    def test_detects_duplicate_stems_across_splits(self):
        """build_samples takes the first rglob hit, so a duplicate silently
        decides which split's copy a fold trains on."""
        self.write_all_good()
        write_feature(self.feature_dir / "test" / "s1.npz", self.good["s1"])
        report = audit_feature_dir(self.feature_dir, self.manifest, self.labels)
        self.assertFalse(report["checks"]["no_duplicate_stems"])
        self.assertIn("s1", report["duplicates"])

    def test_detects_mixed_feature_dims(self):
        self.write_all_good()
        write_feature(self.feature_dir / "test" / "s3.npz", np.zeros(16, dtype=np.float32) + np.arange(16))
        report = audit_feature_dir(self.feature_dir, self.manifest, self.labels)
        self.assertFalse(report["checks"]["single_feature_dim"])
        self.assertEqual(set(report["feature_dims"]), {8, 16})

    def test_detects_nonfinite_values(self):
        self.write_all_good()
        broken = self.good["s2"].copy()
        broken[0] = np.nan
        write_feature(self.feature_dir / "train" / "s2.npz", broken)
        report = audit_feature_dir(self.feature_dir, self.manifest, self.labels)
        self.assertFalse(report["checks"]["all_finite"])
        self.assertIn("s2", report["nonfinite"])

    def test_detects_constant_feature_vector(self):
        self.write_all_good()
        write_feature(self.feature_dir / "train" / "s2.npz", np.zeros(8, dtype=np.float32))
        report = audit_feature_dir(self.feature_dir, self.manifest, self.labels)
        self.assertFalse(report["checks"]["no_constant_features"])

    def test_a_one_dimensional_control_is_not_called_constant(self):
        """Within-sample std is 0 by construction at dim 1, so applying that check to a
        duration-only control fails every legitimate dim-1 arm -- and a gate that cannot
        pass teaches people to ignore gates."""
        for sid, split, value in (("s1", "train", 15.0), ("s2", "train", 110.0), ("s3", "test", 585.0)):
            write_feature(self.feature_dir / split / f"{sid}.npz", np.asarray([value], dtype=np.float32))
        report = audit_feature_dir(self.feature_dir, self.manifest, self.labels)
        self.assertTrue(report["checks"]["no_constant_features"])
        self.assertTrue(report["checks"]["features_vary_across_samples"])
        self.assertTrue(report["passed"], report["checks"])

    def test_detects_a_dim_one_arm_that_carries_no_information(self):
        """The check that DOES matter at dim 1: identical values everywhere trains to
        chance and still reports a number."""
        for sid, split in (("s1", "train"), ("s2", "train"), ("s3", "test")):
            write_feature(self.feature_dir / split / f"{sid}.npz", np.asarray([42.0], dtype=np.float32))
        report = audit_feature_dir(self.feature_dir, self.manifest, self.labels)
        self.assertFalse(report["checks"]["features_vary_across_samples"])
        self.assertFalse(report["passed"])

    def test_detects_an_identical_vector_repeated_across_every_sample(self):
        """Same failure at full width: each sample varies internally, so the per-sample
        check passes, yet the dir carries no signal at all."""
        shared = np.arange(8, dtype=np.float32)
        for sid, split in (("s1", "train"), ("s2", "train"), ("s3", "test")):
            write_feature(self.feature_dir / split / f"{sid}.npz", shared.copy())
        report = audit_feature_dir(self.feature_dir, self.manifest, self.labels)
        self.assertTrue(report["checks"]["no_constant_features"])
        self.assertFalse(report["checks"]["features_vary_across_samples"])

    def test_detects_split_mismatch(self):
        write_feature(self.feature_dir / "train" / "s1.npz", self.good["s1"])
        write_feature(self.feature_dir / "train" / "s2.npz", self.good["s2"])
        write_feature(self.feature_dir / "train" / "s3.npz", self.good["s3"])  # manifest says test
        report = audit_feature_dir(self.feature_dir, self.manifest, self.labels)
        self.assertFalse(report["checks"]["splits_match_manifest"])
        self.assertTrue(any("s3" in entry for entry in report["split_mismatch"]))

    def test_detects_mixed_provenance(self):
        """Two extractions fused into one dir is the failure the separate output
        dir is meant to prevent; the audit is the backstop."""
        self.write_all_good()
        write_feature(
            self.feature_dir / "test" / "s3.npz",
            self.good["s3"],
            provenance={"model_name": "other", "token_pooling": "legacy_first_token"},
        )
        report = audit_feature_dir(self.feature_dir, self.manifest, self.labels)
        self.assertFalse(report["checks"]["single_provenance"])
        self.assertEqual(len(report["provenance_variants"]), 2)

    def test_detects_unexpected_and_unlabeled_samples(self):
        self.write_all_good()
        write_feature(self.feature_dir / "train" / "ghost.npz", self.good["s1"])
        report = audit_feature_dir(self.feature_dir, self.manifest, self.labels)
        self.assertFalse(report["checks"]["no_unexpected_samples"])
        self.assertEqual(report["unexpected"], ["ghost"])
        self.assertEqual(report["unlabeled"], ["ghost"])

    def test_reports_unreadable_bundle_without_raising(self):
        self.write_all_good()
        (self.feature_dir / "train" / "s2.npz").write_bytes(b"not an npz")
        report = audit_feature_dir(self.feature_dir, self.manifest, self.labels)
        self.assertFalse(report["checks"]["all_readable"])
        self.assertTrue(any("s2" in entry for entry in report["unreadable"]))


class SquatSplitSourceTests(unittest.TestCase):
    """The Fitness-AQA side feeds the same checks from split-key and label JSONs."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_split_map_covers_every_split_key_file(self):
        for split_name, ids in (("train", ["a", "b"]), ("val", ["c"]), ("test", ["d"])):
            (self.tmp / f"{split_name}_keys.json").write_text(json.dumps(ids), encoding="utf-8")

        self.assertEqual(
            load_split_map(self.tmp),
            {"a": "train", "b": "train", "c": "val", "d": "test"},
        )

    def test_labeled_ids_are_the_union_of_both_error_files(self):
        """``build_labels`` reads both files with ``.get``, so an id in neither is
        silently relabeled as a negative rather than reported as missing."""
        (self.tmp / "error_knees_forward.json").write_text(json.dumps({"a": [], "b": [[1, 2]]}), encoding="utf-8")
        (self.tmp / "error_knees_inward.json").write_text(json.dumps({"c": []}), encoding="utf-8")

        self.assertEqual(load_labeled_ids(self.tmp), {"a", "b", "c"})

    def test_missing_label_files_yield_an_empty_set_rather_than_raising(self):
        self.assertEqual(load_labeled_ids(self.tmp), set())

    def test_feature_dir_is_audited_against_the_split_map(self):
        feature_dir = self.tmp / "features"
        write_feature(feature_dir / "train" / "a.npz", np.asarray([1.0, 2.0], dtype=np.float32))
        write_feature(feature_dir / "train" / "b.npz", np.asarray([2.0, 1.0], dtype=np.float32))

        report = core_audit_feature_dir(feature_dir, {"a": "train", "b": "val"}, {"a", "b"})

        self.assertTrue(report["checks"]["coverage_complete"])
        self.assertFalse(report["checks"]["splits_match_manifest"])
        self.assertTrue(any("b" in entry for entry in report["split_mismatch"]))


if __name__ == "__main__":
    unittest.main()
