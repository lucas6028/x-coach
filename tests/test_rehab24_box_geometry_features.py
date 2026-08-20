from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from src.rehab24.box_geometry_features import (
    build_feature,
    read_manifest,
    save_feature,
    segment_points,
)


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_row(**overrides: object) -> dict:
    row = {
        "sample_id": "Ex1_PM_000_rep1_cam17",
        "split": "train",
        "video_id": "PM_000",
        "exercise_id": "1",
        "person_id": "1",
        "camera": "cam17",
        "correctness": "1",
        "first_frame": "2",
        "last_frame": "4",
        "skeleton_2d_path": "Ex1/skel.npy",
        "video_path": "Ex1/vid.mp4",
    }
    row.update({k: str(v) for k, v in overrides.items()})
    return row


class ManifestTests(unittest.TestCase):
    def test_a_manifest_missing_a_required_column_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.csv"
            row = sample_row()
            row.pop("skeleton_2d_path")
            write_manifest(path, [row])
            with self.assertRaises(ValueError) as ctx:
                read_manifest(path)
            self.assertIn("skeleton_2d_path", str(ctx.exception))

    def test_an_empty_manifest_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.csv"
            path.write_text("sample_id,split\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_manifest(path)


class SegmentPointsTests(unittest.TestCase):
    def skeleton(self, frames: int = 10, joints: int = 3) -> np.ndarray:
        data = np.zeros((frames, joints, 2), dtype=np.float64)
        for frame in range(frames):
            data[frame, :, 0] = frame * 10.0
            data[frame, :, 1] = frame * 5.0
        return data

    def test_the_range_includes_its_last_frame(self) -> None:
        """The dataset's segmentation is inclusive; dropping the last frame would
        silently shrink every repetition by one."""
        xs, _ = segment_points(self.skeleton(), first_frame=2, last_frame=4)
        self.assertEqual(sorted(set(xs.tolist())), [20.0, 30.0, 40.0])

    def test_non_finite_joints_are_dropped_rather_than_poisoning_the_box(self) -> None:
        skeleton = self.skeleton()
        skeleton[3, 1, 0] = np.nan
        skeleton[3, 2, 1] = np.inf
        xs, ys = segment_points(skeleton, 2, 4)
        self.assertTrue(np.isfinite(xs).all())
        self.assertTrue(np.isfinite(ys).all())
        self.assertEqual(xs.size, ys.size)

    def test_a_range_running_past_the_end_is_clipped(self) -> None:
        xs, _ = segment_points(self.skeleton(frames=5), first_frame=3, last_frame=99)
        self.assertEqual(sorted(set(xs.tolist())), [30.0, 40.0])

    def test_an_empty_range_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            segment_points(self.skeleton(), first_frame=8, last_frame=2)

    def test_a_malformed_skeleton_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            segment_points(np.zeros((10, 2)), 0, 5)


class BuildFeatureTests(unittest.TestCase):
    def test_the_feature_uses_the_video_frame_size_not_an_assumed_one(self) -> None:
        """cam18 is stored portrait (1080x1920) and its skeleton lives in that frame,
        so assuming landscape would normalise every term wrongly."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Ex1").mkdir(parents=True)
            skeleton = np.zeros((6, 2, 2))
            skeleton[:, 0] = [100.0, 200.0]
            skeleton[:, 1] = [300.0, 800.0]
            np.save(root / "Ex1" / "skel.npy", skeleton)

            with mock.patch("src.rehab24.box_geometry_features.frame_size", return_value=(1080, 1920)) as size:
                feature = build_feature(sample_row(first_frame=0, last_frame=5), root, {})

            size.assert_called_once()
            self.assertAlmostEqual(float(feature[0]), 100 / 1080, places=5)
            self.assertAlmostEqual(float(feature[8]), 1080.0, places=5)
            self.assertAlmostEqual(float(feature[9]), 1920.0, places=5)
            self.assertEqual(int(feature[11]), 6)

    def test_frame_size_is_read_once_per_video_not_once_per_repetition(self) -> None:
        """Each video carries roughly 20 repetitions; opening it per row is 20x the IO."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Ex1").mkdir(parents=True)
            skeleton = np.zeros((6, 2, 2))
            skeleton[:, 1] = [300.0, 800.0]
            np.save(root / "Ex1" / "skel.npy", skeleton)
            cache: dict[str, tuple[int, int]] = {}

            with mock.patch("src.rehab24.box_geometry_features.frame_size", return_value=(1920, 1080)) as size:
                for repetition in range(3):
                    build_feature(sample_row(first_frame=0, last_frame=5, sample_id=f"r{repetition}"), root, cache)

            self.assertEqual(size.call_count, 1)

    def test_a_sample_with_no_finite_joints_is_reported(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Ex1").mkdir(parents=True)
            np.save(root / "Ex1" / "skel.npy", np.full((6, 2, 2), np.nan))
            with mock.patch("src.rehab24.box_geometry_features.frame_size", return_value=(1920, 1080)):
                with self.assertRaises(ValueError):
                    build_feature(sample_row(first_frame=0, last_frame=5), root, {})


class SaveFeatureTests(unittest.TestCase):
    def test_the_bundle_carries_the_grouping_keys_loso_needs(self) -> None:
        with TemporaryDirectory() as tmp:
            path = save_feature(Path(tmp), sample_row(), np.arange(12, dtype=np.float32))
            self.assertEqual(path.parent.name, "train")
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(str(data["person_id"]), "1")
                self.assertEqual(str(data["camera"]), "cam17")
                self.assertEqual(int(data["correctness"]), 1)
                self.assertEqual(data["video_feature"].shape, (12,))


if __name__ == "__main__":
    unittest.main()
