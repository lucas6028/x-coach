from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.rehab24.duration_control import (
    CARRIED_KEYS,
    FEATURE_INDEX,
    FEATURE_NAME,
    derive_all,
    derive_bundle,
)
from src.video.box_geometry import FEATURE_NAMES


def write_box_bundle(root: Path, sample_id: str, split: str = "train", n_frames: float = 110.0) -> Path:
    feature = np.arange(len(FEATURE_NAMES), dtype=np.float32)
    feature[FEATURE_INDEX] = n_frames
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sample_id}.npz"
    np.savez_compressed(
        path,
        video_feature=feature,
        sample_id=np.asarray(sample_id),
        video_id=np.asarray("PM_000"),
        exercise_id=np.asarray("6"),
        person_id=np.asarray("1"),
        camera=np.asarray("cam17"),
        correctness=np.asarray(1, dtype=np.int64),
        first_frame=np.asarray(0, dtype=np.int32),
        last_frame=np.asarray(109, dtype=np.int32),
    )
    return path


class DurationControlTests(unittest.TestCase):
    def test_keeps_exactly_the_n_frames_term(self) -> None:
        """Sliced from the box control rather than recomputed, so the two floors are
        provably the same number and their difference is only the geometry terms."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_box_bundle(root / "box", "a", n_frames=137.0)
            out = derive_bundle(source, root / "dur")
            with np.load(out, allow_pickle=False) as data:
                self.assertEqual(data["video_feature"].shape, (1,))
                self.assertEqual(float(data["video_feature"][0]), 137.0)

    def test_the_index_points_at_the_documented_column(self) -> None:
        self.assertEqual(FEATURE_NAMES[FEATURE_INDEX], FEATURE_NAME)

    def test_preserves_the_split_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_box_bundle(root / "box", "a", split="test")
            out = derive_bundle(source, root / "dur")
            self.assertEqual(out.parent.name, "test")

    def test_carries_the_stratification_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = derive_bundle(write_box_bundle(root / "box", "a"), root / "dur")
            with np.load(out, allow_pickle=False) as data:
                for key in CARRIED_KEYS:
                    self.assertIn(key, data.files)
                self.assertEqual(str(data["provenance_variant"]), "n_frames")

    def test_leaves_no_partial_file_behind(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = derive_bundle(write_box_bundle(root / "box", "a"), root / "dur")
            self.assertEqual([p.name for p in out.parent.iterdir()], ["a.npz"])

    def test_rejects_a_feature_of_the_wrong_width(self) -> None:
        """A silently-truncated source would put some other geometry term in the
        duration column and the control would stop being a duration control."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "box" / "train"
            directory.mkdir(parents=True)
            path = directory / "a.npz"
            np.savez_compressed(path, video_feature=np.zeros(4, dtype=np.float32))
            with self.assertRaises(ValueError):
                derive_bundle(path, root / "dur")

    def test_derives_every_bundle_across_splits(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val", "test"):
                for name in ("a", "b"):
                    write_box_bundle(root / "box", f"{split}_{name}", split=split)
            self.assertEqual(derive_all(root / "box", root / "dur"), 6)
            self.assertEqual(len(list((root / "dur").rglob("*.npz"))), 6)

    def test_fails_loudly_on_an_empty_source_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                derive_all(Path(tmp) / "box", Path(tmp) / "dur")


if __name__ == "__main__":
    unittest.main()
