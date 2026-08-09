import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.rehab24.videomae_materialize import materialize_all, materialize_bundle, read_provenance
from src.video.videomae_pooling import LEGACY_FIRST_TOKEN, MEAN_POOL_FC_NORM, aggregate_clips


def write_raw_bundle(path: Path, sample_id: str, legacy: np.ndarray, corrected: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_id=np.asarray(sample_id),
        video_id=np.asarray("PM_000"),
        exercise_id=np.asarray("1"),
        person_id=np.asarray("1"),
        camera=np.asarray("cam17"),
        correctness=np.asarray(1, dtype=np.int64),
        clip_starts=np.asarray([0, 10], dtype=np.int32),
        **{
            f"clip_features_{LEGACY_FIRST_TOKEN}": legacy,
            f"clip_features_{MEAN_POOL_FC_NORM}": corrected,
        },
        provenance_model_name=np.asarray("MCG-NJU/videomae-base-finetuned-kinetics"),
        provenance_clip_length=np.asarray("16"),
        provenance_transformers_version=np.asarray("5.5.0"),
    )


class MaterializeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raw_dir = self.tmp / "videomae_raw"
        self.out_parent = self.tmp / "out"
        rng = np.random.default_rng(0)
        self.legacy = rng.normal(size=(2, 8)).astype(np.float32)
        self.corrected = rng.normal(size=(2, 8)).astype(np.float32)
        write_raw_bundle(self.raw_dir / "train" / "s1.npz", "s1", self.legacy, self.corrected)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_video_feature_matches_requested_aggregation(self):
        out = materialize_bundle(self.raw_dir / "train" / "s1.npz", self.out_parent, "train", MEAN_POOL_FC_NORM, "mean")
        with np.load(out, allow_pickle=False) as data:
            np.testing.assert_allclose(data["video_feature"], aggregate_clips(self.corrected, "mean"))

    def test_token_pooling_selects_the_right_stack(self):
        legacy_out = materialize_bundle(self.raw_dir / "train" / "s1.npz", self.out_parent / "a", "train", LEGACY_FIRST_TOKEN, "max")
        corrected_out = materialize_bundle(self.raw_dir / "train" / "s1.npz", self.out_parent / "b", "train", MEAN_POOL_FC_NORM, "max")
        with np.load(legacy_out, allow_pickle=False) as data:
            np.testing.assert_allclose(data["video_feature"], aggregate_clips(self.legacy, "max"))
        with np.load(corrected_out, allow_pickle=False) as data:
            np.testing.assert_allclose(data["video_feature"], aggregate_clips(self.corrected, "max"))

    def test_provenance_records_both_pooling_axes(self):
        out = materialize_bundle(self.raw_dir / "train" / "s1.npz", self.out_parent, "train", MEAN_POOL_FC_NORM, "mean")
        with np.load(out, allow_pickle=False) as data:
            provenance = read_provenance(data)
        self.assertEqual(provenance["token_pooling"], MEAN_POOL_FC_NORM)
        self.assertEqual(provenance["clip_aggregation"], "mean")
        self.assertEqual(provenance["transformers_version"], "5.5.0")

    def test_carries_sample_metadata_for_stratification(self):
        out = materialize_bundle(self.raw_dir / "train" / "s1.npz", self.out_parent, "train", MEAN_POOL_FC_NORM, "mean")
        with np.load(out, allow_pickle=False) as data:
            self.assertEqual(str(data["sample_id"]), "s1")
            self.assertEqual(str(data["camera"]), "cam17")
            self.assertEqual(str(data["person_id"]), "1")
            self.assertEqual(int(data["correctness"]), 1)

    def test_preserves_split_subdirectory(self):
        out = materialize_bundle(self.raw_dir / "train" / "s1.npz", self.out_parent, "train", MEAN_POOL_FC_NORM, "mean")
        self.assertEqual(out.parent.name, "train")

    def test_materialize_all_writes_every_combination(self):
        counts = materialize_all(self.raw_dir, self.out_parent)
        self.assertEqual(len(counts), 4)
        for name in counts:
            self.assertTrue((self.out_parent / name / "train" / "s1.npz").exists(), name)

    def test_combinations_produce_distinct_features(self):
        """If two dirs held identical vectors, a paired LOSO delta would be
        structurally zero and could be misread as 'pooling does not matter'."""
        materialize_all(self.raw_dir, self.out_parent)
        vectors = []
        for name in sorted(p.name for p in self.out_parent.iterdir()):
            with np.load(self.out_parent / name / "train" / "s1.npz", allow_pickle=False) as data:
                vectors.append(data["video_feature"])
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                self.assertFalse(np.allclose(vectors[i], vectors[j]), f"dirs {i} and {j} are identical")

    def test_raises_when_raw_dir_is_empty(self):
        with self.assertRaises(SystemExit):
            materialize_all(self.tmp / "does_not_exist", self.out_parent)


if __name__ == "__main__":
    unittest.main()
