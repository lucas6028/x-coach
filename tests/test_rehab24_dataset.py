from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.rehab24.dataset import (
    TEST_PERSON_IDS,
    TRAIN_PERSON_IDS,
    VAL_PERSON_IDS,
    build_manifest_rows,
    read_segmentation,
    validate_manifest_paths,
    write_manifest,
    write_splits_and_labels,
)
from src.rehab24.fuse_features import fuse_feature_files
from src.rehab24.skeleton_features import extract_features_for_manifest, extract_feature_vector, frame_bounds


SEGMENTATION_FIELDS = [
    "video_id",
    "repetition_number",
    "exercise_id",
    "person_id",
    "first_frame",
    "last_frame",
    "cam17_orientation",
    "mocap_erroneous",
    "exercise_subtype",
    "lights_on",
    "extra_person_in_cam17",
    "extra_person_in_cam18",
    "correctness",
]


def write_segmentation(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEGMENTATION_FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerow(
            {
                "video_id": "PM_000",
                "repetition_number": "1",
                "exercise_id": "6",
                "person_id": "6",
                "first_frame": "2",
                "last_frame": "6",
                "cam17_orientation": "front",
                "mocap_erroneous": "0",
                "exercise_subtype": "squat",
                "lights_on": "1",
                "extra_person_in_cam17": "0",
                "extra_person_in_cam18": "2",
                "correctness": "1",
            }
        )


def write_rehab24_files(root: Path) -> None:
    exercise_dir = root / "Ex6"
    exercise_dir.mkdir(parents=True)
    skeleton_3d = np.arange(10 * 26 * 4, dtype=np.float64).reshape(10, 26, 4)
    skeleton_2d = np.arange(10 * 26 * 2, dtype=np.float64).reshape(10, 26, 2)
    np.save(exercise_dir / "PM_000-30fps.npy", skeleton_3d)
    np.save(exercise_dir / "PM_000-c17-30fps.npy", skeleton_2d)
    np.save(exercise_dir / "PM_000-c18-30fps.npy", skeleton_2d + 1)
    (exercise_dir / "PM_000-Camera17-30fps.mp4").write_bytes(b"")
    (exercise_dir / "PM_000-Camera18-30fps-transposed.mp4").write_bytes(b"")


class Rehab24DatasetTests(unittest.TestCase):
    def test_default_person_splits_do_not_overlap(self) -> None:
        self.assertFalse(TRAIN_PERSON_IDS & VAL_PERSON_IDS)
        self.assertFalse(TRAIN_PERSON_IDS & TEST_PERSON_IDS)
        self.assertFalse(VAL_PERSON_IDS & TEST_PERSON_IDS)

    def test_manifest_has_two_camera_samples_and_resolvable_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_rehab24_files(root)
            segmentation_path = root / "Segmentation.csv"
            write_segmentation(segmentation_path)

            rows = build_manifest_rows(read_segmentation(segmentation_path))

            self.assertEqual([row["sample_id"] for row in rows], ["Ex6_PM_000_rep1_cam17", "Ex6_PM_000_rep1_cam18"])
            self.assertEqual([row["split"] for row in rows], ["val", "val"])
            self.assertEqual([row["camera_orientation"] for row in rows], ["front", "side"])
            self.assertEqual(validate_manifest_paths(root, rows), [])

    def test_frame_bounds_treat_last_frame_as_inclusive_one_based(self) -> None:
        self.assertEqual(frame_bounds(first_frame=2, last_frame=6, total_frames=10), (1, 6))

    def test_feature_extraction_writes_classifier_compatible_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_rehab24_files(root)
            segmentation_path = root / "Segmentation.csv"
            write_segmentation(segmentation_path)
            rows = build_manifest_rows(read_segmentation(segmentation_path), cameras=("cam17",))
            processed = root / "processed"
            manifest = processed / "manifest.csv"
            write_manifest(manifest, rows)
            write_splits_and_labels(processed, rows)

            written = extract_features_for_manifest(
                data_root=root,
                manifest_path=manifest,
                output_dir=processed / "skeleton_features",
            )

            self.assertEqual(written, 1)
            output = processed / "skeleton_features" / "val" / "Ex6_PM_000_rep1_cam17.npz"
            with np.load(output, allow_pickle=False) as data:
                self.assertIn("video_feature", data.files)
                self.assertEqual(data["video_feature"].ndim, 1)
                self.assertGreater(data["video_feature"].shape[0], 0)
                self.assertEqual(str(data["sample_id"]), "Ex6_PM_000_rep1_cam17")

    def test_extract_feature_vector_is_fixed_length(self) -> None:
        skeleton_3d = np.ones((8, 26, 4), dtype=np.float32)
        skeleton_2d = np.ones((8, 26, 2), dtype=np.float32)

        feature = extract_feature_vector(skeleton_3d, skeleton_2d, first_frame=1, last_frame=8)

        expected_dim = (26 * 6 + 26 * 4) * 9
        self.assertEqual(feature.shape, (expected_dim,))

    def test_fuse_feature_files_concatenates_video_feature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = root / "first.npz"
            second = root / "second.npz"
            np.savez_compressed(first, video_feature=np.asarray([1.0, 2.0], dtype=np.float32))
            np.savez_compressed(second, video_feature=np.asarray([3.0], dtype=np.float32))

            fused = fuse_feature_files(first, second)

            np.testing.assert_allclose(fused, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
