from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from src.video.box_geometry import FEATURE_DIM, FEATURE_NAMES, box_from_points, box_geometry_feature
from src.video.variant_geometry import Box

ARCHIVED = Path("data/Fitness-AQA/Squat/Labeled_Dataset/box_geometry_features")
POSE_JSON = Path("data/Fitness-AQA/Squat/Labeled_Dataset/pose_json")


class FeatureTests(unittest.TestCase):
    def test_the_twelve_terms_are_what_their_names_say(self) -> None:
        feature = box_geometry_feature(Box(100, 200, 300, 500), frame_width=400, frame_height=1000, n_frames=90)
        by_name = dict(zip(FEATURE_NAMES, feature.tolist()))

        self.assertAlmostEqual(by_name["x0_norm"], 0.25, places=6)
        self.assertAlmostEqual(by_name["y1_norm"], 0.5, places=6)
        self.assertAlmostEqual(by_name["width_norm"], 0.5, places=6)
        self.assertAlmostEqual(by_name["height_norm"], 0.3, places=6)
        self.assertAlmostEqual(by_name["area_norm"], 0.15, places=6)
        self.assertAlmostEqual(by_name["frame_aspect"], 0.4, places=6)
        self.assertEqual(by_name["n_frames"], 90)

    def test_box_aspect_is_measured_in_pixels_not_normalised_units(self) -> None:
        """A 480x600 frame would report a 2:1 athlete as 2.5:1 in normalised units."""
        feature = box_geometry_feature(Box(0, 0, 100, 200), frame_width=480, frame_height=600, n_frames=1)
        self.assertAlmostEqual(feature[FEATURE_NAMES.index("box_aspect")], 0.5, places=6)

    def test_a_degenerate_box_or_frame_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            box_geometry_feature(Box(10, 10, 10, 20), 480, 600, 30)
        with self.assertRaises(ValueError):
            box_geometry_feature(Box(0, 0, 10, 20), 0, 600, 30)

    def test_the_dimension_is_fixed(self) -> None:
        self.assertEqual(FEATURE_DIM, 12)
        self.assertEqual(box_geometry_feature(Box(0, 0, 4, 4), 8, 8, 2).shape, (12,))


class BoxFromPointsTests(unittest.TestCase):
    def test_rounding_matches_the_pose_derived_box(self) -> None:
        box = box_from_points(np.asarray([10.4, 30.6]), np.asarray([5.2, 40.9]), 100, 100)
        self.assertEqual(box.as_tuple(), (10, 5, 31, 41))

    def test_points_are_clamped_into_the_frame(self) -> None:
        box = box_from_points(np.asarray([-5.0, 200.0]), np.asarray([-3.0, 150.0]), 100, 120)
        self.assertEqual(box.as_tuple(), (0, 0, 100, 120))

    def test_an_empty_cloud_returns_none(self) -> None:
        self.assertIsNone(box_from_points(np.asarray([]), np.asarray([]), 10, 10))

    def test_a_collapsed_cloud_returns_none(self) -> None:
        self.assertIsNone(box_from_points(np.asarray([5.0, 5.0]), np.asarray([5.0, 5.0]), 10, 10))


@unittest.skipUnless(ARCHIVED.exists() and POSE_JSON.exists(), "Fitness-AQA data not present")
class ReproducesArchivedFeaturesTests(unittest.TestCase):
    """The definition was reverse-engineered from .npz files that had no source.

    Regenerating them from the pose JSON and requiring a bit-for-bit match is what
    proves the recovered definition is the one behind the reported 0.6120, and pins it
    so it cannot drift.
    """

    def test_regenerated_features_match_the_archived_ones_exactly(self) -> None:
        from src.video.squat_video_variants import person_box_from_pose

        archived = sorted(ARCHIVED.rglob("*.npz"))[:25]
        self.assertTrue(archived, "no archived box_geometry features found")

        checked = 0
        for path in archived:
            with np.load(path, allow_pickle=False) as data:
                expected = data["video_feature"]
                video_id = str(data["video_id"])
                split = str(data["split"])

            pose_path = POSE_JSON / split / f"{video_id}.json"
            if not pose_path.exists():
                continue
            with pose_path.open("r", encoding="utf-8") as f:
                pose = json.load(f)

            metadata = pose["metadata"]
            box = person_box_from_pose(pose)
            self.assertIsNotNone(box, f"{video_id} has no visible pose")
            actual = box_geometry_feature(
                box,
                frame_width=int(metadata["width"]),
                frame_height=int(metadata["height"]),
                n_frames=int(metadata["total_frames"]),
            )
            np.testing.assert_array_equal(actual, expected, err_msg=f"mismatch on {video_id}")
            checked += 1

        self.assertGreater(checked, 0, "no archived feature could be checked against its pose JSON")


if __name__ == "__main__":
    unittest.main()
