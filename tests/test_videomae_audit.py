import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.rehab24.videomae_audit import audit_feature_dir

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


if __name__ == "__main__":
    unittest.main()
