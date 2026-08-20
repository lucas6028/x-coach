from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.rehab24.videomae_framing_pairing import compare_raw_dirs
from src.video.videomae_pooling import MEAN_POOL_FC_NORM


def write_bundle(
    root: Path,
    sample_id: str,
    split: str = "train",
    clip_starts: tuple[int, ...] = (0, 10, 20, 30),
    features: np.ndarray | None = None,
    variant: str = "full_frame",
    camera: str = "cam17",
    correctness: int = 1,
) -> None:
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash((sample_id, variant))) % (2**32))
    np.savez_compressed(
        directory / f"{sample_id}.npz",
        sample_id=np.asarray(sample_id),
        video_id=np.asarray(sample_id.split("_rep")[0]),
        exercise_id=np.asarray("6"),
        person_id=np.asarray("1"),
        camera=np.asarray(camera),
        correctness=np.asarray(correctness, dtype=np.int64),
        clip_starts=np.asarray(clip_starts, dtype=np.int32),
        first_frame=np.asarray(0, dtype=np.int32),
        last_frame=np.asarray(100, dtype=np.int32),
        total_frames=np.asarray(1000, dtype=np.int32),
        **{f"clip_features_{MEAN_POOL_FC_NORM}": rng.normal(size=(4, 768)).astype(np.float32) if features is None else features},
        provenance_model_name=np.asarray("MCG-NJU/videomae-base-finetuned-kinetics"),
        provenance_clip_length=np.asarray("16"),
        provenance_variant=np.asarray(variant),
    )


class PairingGateTests(unittest.TestCase):
    def dirs(self, tmp: str) -> tuple[Path, Path]:
        return Path(tmp) / "baseline", Path(tmp) / "candidate"

    def build_pair(self, tmp: str, ids: tuple[str, ...] = ("Ex6_a_rep1", "Ex6_a_rep2")) -> tuple[Path, Path]:
        baseline, candidate = self.dirs(tmp)
        for sample_id in ids:
            write_bundle(baseline, sample_id, variant="full_frame")
            write_bundle(candidate, sample_id, variant="full_frame_letterbox")
        return baseline, candidate

    def test_a_correctly_paired_arm_passes(self) -> None:
        with TemporaryDirectory() as tmp:
            report = compare_raw_dirs(*self.build_pair(tmp))
            self.assertTrue(report["passed"], report["checks"])
            self.assertEqual(report["n_compared"], 2)

    def test_a_short_candidate_fails(self) -> None:
        """An arm missing samples trains on a smaller set and still prints a number."""
        with TemporaryDirectory() as tmp:
            baseline, candidate = self.dirs(tmp)
            write_bundle(baseline, "a", variant="full_frame")
            write_bundle(baseline, "b", variant="full_frame")
            write_bundle(candidate, "a", variant="full_frame_letterbox")
            report = compare_raw_dirs(baseline, candidate)
            self.assertFalse(report["checks"]["same_sample_ids"])
            self.assertEqual(report["missing_from_candidate"], ["b"])

    def test_drifted_clip_starts_fail(self) -> None:
        """Different frames under one label is not a paired comparison."""
        with TemporaryDirectory() as tmp:
            baseline, candidate = self.dirs(tmp)
            write_bundle(baseline, "a", clip_starts=(0, 10, 20, 30), variant="full_frame")
            write_bundle(candidate, "a", clip_starts=(0, 11, 20, 30), variant="full_frame_letterbox")
            report = compare_raw_dirs(baseline, candidate)
            self.assertFalse(report["checks"]["same_clip_starts"])

    def test_a_relabelled_sample_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            baseline, candidate = self.dirs(tmp)
            write_bundle(baseline, "a", correctness=1, variant="full_frame")
            write_bundle(candidate, "a", correctness=0, variant="full_frame_letterbox")
            report = compare_raw_dirs(baseline, candidate)
            self.assertFalse(report["checks"]["same_repetition_metadata"])

    def test_a_resplit_sample_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            baseline, candidate = self.dirs(tmp)
            write_bundle(baseline, "a", split="train", variant="full_frame")
            write_bundle(candidate, "a", split="test", variant="full_frame_letterbox")
            report = compare_raw_dirs(baseline, candidate)
            self.assertFalse(report["checks"]["same_splits"])

    def test_bit_identical_features_fail_with_zero_tolerance(self) -> None:
        """The check with teeth. Both REHAB24-6 cameras are non-square, so a sample
        whose features match the baseline exactly means the transform did not run --
        which reads as 'the manipulation did nothing' instead of as a bug."""
        with TemporaryDirectory() as tmp:
            baseline, candidate = self.dirs(tmp)
            shared = np.arange(4 * 768, dtype=np.float32).reshape(4, 768)
            write_bundle(baseline, "a", features=shared, variant="full_frame")
            write_bundle(candidate, "a", features=shared.copy(), variant="full_frame_letterbox")
            report = compare_raw_dirs(baseline, candidate)
            self.assertFalse(report["checks"]["features_actually_differ"])
            self.assertEqual(report["identical_features"], ["a"])

    def test_identical_features_can_be_allowed_for_an_identity_arm(self) -> None:
        with TemporaryDirectory() as tmp:
            baseline, candidate = self.dirs(tmp)
            shared = np.ones((4, 768), dtype=np.float32)
            write_bundle(baseline, "a", features=shared, variant="full_frame")
            write_bundle(candidate, "a", features=shared.copy(), variant="reencoded")
            report = compare_raw_dirs(baseline, candidate, require_different_features=False)
            self.assertNotIn("features_actually_differ", report["checks"])
            self.assertTrue(report["passed"], report["checks"])

    def test_two_arms_sharing_a_provenance_stamp_fail(self) -> None:
        """Same stamp means the same pixels were claimed for both arms."""
        with TemporaryDirectory() as tmp:
            baseline, candidate = self.dirs(tmp)
            write_bundle(baseline, "a", variant="full_frame")
            write_bundle(candidate, "a", variant="full_frame")
            report = compare_raw_dirs(baseline, candidate)
            self.assertFalse(report["checks"]["arms_declare_different_variants"])

    def test_non_finite_features_fail(self) -> None:
        with TemporaryDirectory() as tmp:
            baseline, candidate = self.dirs(tmp)
            broken = np.zeros((4, 768), dtype=np.float32)
            broken[0, 0] = np.nan
            write_bundle(baseline, "a", variant="full_frame")
            write_bundle(candidate, "a", features=broken, variant="full_frame_letterbox")
            report = compare_raw_dirs(baseline, candidate)
            self.assertFalse(report["checks"]["all_finite"])

    def test_completeness_is_measured_against_the_manifest_not_the_baseline(self) -> None:
        """Both arms being equally short is still incomplete."""
        with TemporaryDirectory() as tmp:
            baseline, candidate = self.build_pair(tmp, ids=("Ex6_a_rep1",))
            report = compare_raw_dirs(baseline, candidate, {"Ex6_a_rep1": "train", "Ex6_a_rep2": "train"})
            self.assertFalse(report["checks"]["baseline_is_complete"])
            self.assertFalse(report["checks"]["candidate_is_complete"])
            self.assertTrue(report["checks"]["same_sample_ids"])


if __name__ == "__main__":
    unittest.main()
