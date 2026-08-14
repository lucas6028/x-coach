from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.video.early_fusion import concat_features, fuse_feature_dirs


def write_feature(path: Path, feature: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, video_feature=feature.astype(np.float32))


class ConcatFeaturesTests(unittest.TestCase):
    def test_first_dir_comes_first_in_the_concatenation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_feature(root / "a.npz", np.asarray([1.0, 2.0]))
            write_feature(root / "b.npz", np.asarray([3.0]))
            np.testing.assert_allclose(concat_features(root / "a.npz", root / "b.npz"), [1.0, 2.0, 3.0])


class FuseFeatureDirsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.first = self.root / "first"
        self.second = self.root / "second"
        self.output = self.root / "out"
        self.split_map = {"a": "train", "b": "val"}
        for video_id, split_name in self.split_map.items():
            write_feature(self.first / split_name / f"{video_id}.npz", np.asarray([1.0, 1.0]))
            write_feature(self.second / split_name / f"{video_id}.npz", np.asarray([2.0]))

    def test_every_id_is_written_under_its_split(self) -> None:
        result = fuse_feature_dirs(self.first, self.second, self.output, self.split_map)

        self.assertEqual(result["written"], 2)
        self.assertEqual(result["missing"], 0)
        with np.load(self.output / "train" / "a.npz", allow_pickle=False) as data:
            np.testing.assert_allclose(data["video_feature"], [1.0, 1.0, 2.0])
            self.assertEqual(str(data["split"]), "train")

    def test_ids_missing_from_one_side_are_reported_not_skipped(self) -> None:
        """A fusion arm silently covering fewer videos is not a paired comparison."""
        (self.second / "val" / "b.npz").unlink()

        result = fuse_feature_dirs(self.first, self.second, self.output, self.split_map)

        self.assertEqual(result["written"], 1)
        self.assertEqual(result["missing_ids"], ["b"])

    def test_existing_outputs_are_left_alone_unless_overwrite(self) -> None:
        fuse_feature_dirs(self.first, self.second, self.output, self.split_map)
        write_feature(self.first / "train" / "a.npz", np.asarray([9.0, 9.0]))

        fuse_feature_dirs(self.first, self.second, self.output, self.split_map)
        with np.load(self.output / "train" / "a.npz", allow_pickle=False) as data:
            np.testing.assert_allclose(data["video_feature"], [1.0, 1.0, 2.0])

        fuse_feature_dirs(self.first, self.second, self.output, self.split_map, overwrite=True)
        with np.load(self.output / "train" / "a.npz", allow_pickle=False) as data:
            np.testing.assert_allclose(data["video_feature"], [9.0, 9.0, 2.0])

    def test_provenance_records_both_source_dirs(self) -> None:
        fuse_feature_dirs(self.first, self.second, self.output, self.split_map)
        with np.load(self.output / "val" / "b.npz", allow_pickle=False) as data:
            self.assertEqual(str(data["provenance_first_dir"]), str(self.first))
            self.assertEqual(str(data["provenance_second_dir"]), str(self.second))


if __name__ == "__main__":
    unittest.main()
