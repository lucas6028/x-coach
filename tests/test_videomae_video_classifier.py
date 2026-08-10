from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.video.videomae_video_classifier import (
    FeatureDataset,
    Sample,
    compute_feature_normalization,
    feature_normalization_payload,
)


def write_feature(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, video_feature=np.asarray(values, dtype=np.float32))


class VideoMaeVideoClassifierTests(unittest.TestCase):
    def test_compute_feature_normalization_uses_train_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first_path = root / "first.npz"
            second_path = root / "second.npz"
            write_feature(first_path, [1.0, 4.0, 7.0])
            write_feature(second_path, [3.0, 4.0, 11.0])

            normalization = compute_feature_normalization(
                [
                    Sample(video_id="first", feature_path=first_path, label=0),
                    Sample(video_id="second", feature_path=second_path, label=1),
                ]
            )

            np.testing.assert_allclose(normalization.mean, np.asarray([2.0, 4.0, 9.0], dtype=np.float32))
            np.testing.assert_allclose(normalization.std, np.asarray([1.0, 1.0, 2.0], dtype=np.float32))

    def test_feature_dataset_applies_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.npz"
            eval_path = root / "eval.npz"
            write_feature(train_path, [1.0, 4.0])
            write_feature(eval_path, [3.0, 10.0])
            normalization = compute_feature_normalization(
                [
                    Sample(video_id="low", feature_path=train_path, label=0),
                    Sample(video_id="high", feature_path=eval_path, label=1),
                ]
            )

            dataset = FeatureDataset(
                [Sample(video_id="eval", feature_path=eval_path, label=1)],
                normalization=normalization,
            )

            feature, label = dataset[0]

            np.testing.assert_allclose(feature.numpy(), np.asarray([1.0, 1.0], dtype=np.float32))
            self.assertEqual(float(label), 1.0)

    def test_feature_normalization_payload_is_plain_python_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first_path = root / "first.npz"
            second_path = root / "second.npz"
            write_feature(first_path, [1.0])
            write_feature(second_path, [3.0])
            normalization = compute_feature_normalization(
                [
                    Sample(video_id="first", feature_path=first_path, label=0),
                    Sample(video_id="second", feature_path=second_path, label=1),
                ]
            )

            payload = feature_normalization_payload(normalization)

            self.assertEqual(payload["kind"], "train_set_zscore")
            self.assertEqual(payload["mean"], [2.0])
            self.assertEqual(payload["std"], [1.0])


if __name__ == "__main__":
    unittest.main()


class FeatureCacheTest(unittest.TestCase):
    """The cache must change only the number of disk reads, never the values."""

    def setUp(self):
        from src.video.videomae_video_classifier import clear_feature_cache

        clear_feature_cache()
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "s1.npz"
        self.feature = np.arange(8, dtype=np.float32)
        np.savez_compressed(self.path, video_feature=self.feature)

    def tearDown(self):
        from src.video.videomae_video_classifier import clear_feature_cache

        clear_feature_cache()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_the_stored_values(self):
        from src.video.videomae_video_classifier import load_video_feature

        np.testing.assert_allclose(load_video_feature(self.path), self.feature)

    def test_repeated_loads_are_equal(self):
        from src.video.videomae_video_classifier import load_video_feature

        np.testing.assert_allclose(load_video_feature(self.path), load_video_feature(self.path))

    def test_mutating_a_returned_array_does_not_poison_the_cache(self):
        """FeatureDataset passes the array to torch.from_numpy, which shares memory."""
        from src.video.videomae_video_classifier import load_video_feature

        first = load_video_feature(self.path)
        first[0] = 999.0
        np.testing.assert_allclose(load_video_feature(self.path), self.feature)

    def test_dataset_item_matches_an_uncached_read(self):
        from src.video.videomae_video_classifier import FeatureDataset, Sample, clear_feature_cache

        dataset = FeatureDataset([Sample(video_id="s1", feature_path=self.path, label=1)])
        cached_feature, cached_label = dataset[0]
        clear_feature_cache()
        fresh_feature, fresh_label = dataset[0]
        np.testing.assert_allclose(cached_feature.numpy(), fresh_feature.numpy())
        self.assertEqual(float(cached_label), float(fresh_label))
