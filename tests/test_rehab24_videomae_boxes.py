from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.rehab24.videomae_boxes import (
    BOX_SOURCE,
    box_for_video,
    load_index,
    video_box,
    video_rows,
    write_index,
)
from src.video.variant_geometry import Box


def skeleton(points_per_frame: list[list[tuple[float, float]]]) -> np.ndarray:
    return np.asarray(points_per_frame, dtype=np.float64)


class VideoBoxTests(unittest.TestCase):
    def test_unions_over_every_frame_not_one_repetition(self) -> None:
        """Plan §4.1: a per-repetition box encodes how far THAT rep travelled, which is
        a function of its correctness -- the label would enter through the crop."""
        moving = skeleton([[(10.0, 10.0), (20.0, 20.0)], [(200.0, 300.0), (210.0, 310.0)]])
        box = video_box(moving, 640, 480)
        self.assertEqual(box, Box(10, 10, 210, 310))

    def test_is_identical_for_every_repetition_of_a_video(self) -> None:
        """The whole point: two rows of the same video get the same rectangle, so the
        pixel transform cannot vary with the repetition."""
        points = skeleton([[(30.0, 40.0)], [(300.0, 400.0)], [(100.0, 100.0)]])
        first = video_box(points, 640, 480)
        second = video_box(points, 640, 480)
        self.assertEqual(first, second)

    def test_drops_non_finite_joints_rather_than_poisoning_the_box(self) -> None:
        points = skeleton([[(np.nan, np.nan), (50.0, 60.0)], [(np.inf, 5.0), (80.0, 90.0)]])
        self.assertEqual(video_box(points, 640, 480), Box(50, 60, 80, 90))

    def test_clamps_to_the_frame(self) -> None:
        points = skeleton([[(-30.0, -40.0)], [(9000.0, 9000.0)]])
        self.assertEqual(video_box(points, 640, 480), Box(0, 0, 640, 480))

    def test_returns_none_when_nothing_is_finite(self) -> None:
        self.assertIsNone(video_box(skeleton([[(np.nan, np.nan)]]), 640, 480))

    def test_rejects_a_skeleton_of_the_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            video_box(np.zeros((10, 2)), 640, 480)


class VideoRowsTests(unittest.TestCase):
    def test_collapses_the_repetitions_of_a_video_to_one_row(self) -> None:
        rows = [
            {"video_path": "Ex1/a.mp4", "camera": "cam17"},
            {"video_path": "Ex1/a.mp4", "camera": "cam17"},
            {"video_path": "Ex1/b.mp4", "camera": "cam18"},
        ]
        self.assertEqual([row["video_path"] for row in video_rows(rows)], ["Ex1/a.mp4", "Ex1/b.mp4"])

    def test_keeps_the_two_cameras_of_one_recording_apart(self) -> None:
        """cam17 and cam18 are different files with different frame sizes and different
        skeletons; sharing a box between them would crop the wrong region."""
        rows = [
            {"video_path": "Ex1/a-Camera17-30fps.mp4", "camera": "cam17"},
            {"video_path": "Ex1/a-Camera18-30fps-transposed.mp4", "camera": "cam18"},
        ]
        self.assertEqual(len(video_rows(rows)), 2)


class BoxLookupTests(unittest.TestCase):
    def index(self) -> dict:
        return {
            "box_source": BOX_SOURCE,
            "margin": 0.15,
            "n_videos": 1,
            "videos": {"Ex1/a.mp4": {"camera": "cam17", "box": [10, 20, 110, 220], "frame_size": [640, 480]}},
        }

    def test_returns_the_expanded_box(self) -> None:
        self.assertEqual(box_for_video(self.index(), "Ex1/a.mp4"), Box(10, 20, 110, 220))

    def test_fails_closed_on_a_missing_video(self) -> None:
        """`apply_variant` reads a null box as 'leave untouched', so returning None here
        would write full-frame features into a control arm under that arm's name."""
        with self.assertRaises(KeyError):
            box_for_video(self.index(), "Ex1/missing.mp4")

    def test_fails_closed_on_a_null_box(self) -> None:
        index = self.index()
        index["videos"]["Ex1/a.mp4"]["box"] = None
        with self.assertRaises(KeyError):
            box_for_video(index, "Ex1/a.mp4")

    def test_survives_a_write_read_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "boxes.json"
            write_index(self.index(), path)
            self.assertEqual(box_for_video(load_index(path), "Ex1/a.mp4"), Box(10, 20, 110, 220))
            self.assertEqual(json.loads(path.read_text())["box_source"], BOX_SOURCE)


if __name__ == "__main__":
    unittest.main()
