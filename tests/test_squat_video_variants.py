from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from src.video.squat_video_variants import (
    Box,
    apply_variant,
    build_variant_video,
    estimate_background,
    person_box_from_pose,
    read_all_frames,
    square_crop_box,
    write_video,
)

WIDTH = 64
HEIGHT = 48


def landmark(x: float, y: float, visibility: float = 1.0) -> dict:
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility}


def pose_document(frames: list[list[dict]], width: int = WIDTH, height: int = HEIGHT, total_frames: int | None = None) -> dict:
    return {
        "metadata": {
            "fps": 30.0,
            "width": width,
            "height": height,
            "total_frames": total_frames if total_frames is not None else len(frames),
        },
        "frames": [{"frame_index": index, "landmarks": marks} for index, marks in enumerate(frames)],
    }


class PersonBoxTests(unittest.TestCase):
    def test_box_is_the_union_over_frames_in_pixels(self) -> None:
        pose = pose_document(
            [
                [landmark(0.25, 0.25), landmark(0.5, 0.5)],
                [landmark(0.5, 0.5), landmark(0.75, 0.75)],
            ]
        )
        self.assertEqual(person_box_from_pose(pose).as_tuple(), (16, 12, 48, 36))

    def test_low_visibility_landmarks_do_not_widen_the_box(self) -> None:
        pose = pose_document([[landmark(0.25, 0.25), landmark(0.5, 0.5), landmark(0.9, 0.9, visibility=0.1)]])
        box = person_box_from_pose(pose)
        self.assertEqual(box.as_tuple(), (16, 12, 32, 24))

    def test_no_visible_landmark_anywhere_returns_none(self) -> None:
        """A person-less video must be reported, not silently boxed at the origin."""
        pose = pose_document([[landmark(0.5, 0.5, visibility=0.0)], []])
        self.assertIsNone(person_box_from_pose(pose))

    def test_missing_frame_size_metadata_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            person_box_from_pose({"metadata": {}, "frames": []})


class SquareCropBoxTests(unittest.TestCase):
    def test_crop_is_square_and_padded_by_the_margin(self) -> None:
        box = square_crop_box(Box(20, 10, 30, 30), WIDTH, HEIGHT, margin=0.2)
        self.assertEqual(box.x1 - box.x0, box.y1 - box.y0)
        self.assertEqual(box.x1 - box.x0, 24)  # 20px tall * 1.2

    def test_crop_stays_inside_the_frame_when_the_person_hugs_an_edge(self) -> None:
        box = square_crop_box(Box(0, 0, 40, 40), WIDTH, HEIGHT, margin=0.5)
        self.assertGreaterEqual(box.x0, 0)
        self.assertGreaterEqual(box.y0, 0)
        self.assertLessEqual(box.x1, WIDTH)
        self.assertLessEqual(box.y1, HEIGHT)

    def test_side_is_capped_by_the_shorter_frame_dimension(self) -> None:
        box = square_crop_box(Box(0, 0, WIDTH, HEIGHT), WIDTH, HEIGHT, margin=1.0)
        self.assertEqual(box.y1 - box.y0, HEIGHT)


class ApplyVariantTests(unittest.TestCase):
    def make_frames(self, count: int = 5) -> list[np.ndarray]:
        frames = []
        for index in range(count):
            frame = np.full((HEIGHT, WIDTH, 3), 10, dtype=np.uint8)
            frame[10:30, 20:40] = 200 + index  # the "person"
            frames.append(frame)
        return frames

    def test_person_crop_keeps_only_the_box(self) -> None:
        cropped = apply_variant(self.make_frames(), "person_crop", Box(20, 10, 40, 30))
        self.assertEqual(cropped[0].shape[:2], (20, 20))
        self.assertTrue((cropped[0] >= 200).all())

    def test_background_only_paints_the_box_with_the_scene_median(self) -> None:
        frames = self.make_frames()
        masked = apply_variant(frames, "background_only", Box(20, 10, 40, 30))
        self.assertEqual(masked[0].shape, frames[0].shape)
        # The person's pixels are replaced by the temporal median (202 here), so no
        # frame-to-frame variation from the athlete survives inside the box.
        inside = np.stack([frame[10:30, 20:40] for frame in masked], axis=0)
        self.assertEqual(len(np.unique(inside)), 1)
        # Outside the box the scene is untouched.
        self.assertTrue((masked[0][0:10, 0:20] == 10).all())

    def test_missing_box_leaves_the_video_untouched(self) -> None:
        frames = self.make_frames()
        for variant in ("person_crop", "background_only"):
            result = apply_variant(frames, variant, None)
            self.assertTrue(np.array_equal(result[0], frames[0]))

    def test_unknown_variant_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_variant(self.make_frames(), "grayscale", Box(0, 0, 4, 4))

    def test_background_median_ignores_a_transient_object(self) -> None:
        frames = [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(9)]
        frames[4][:] = 255
        self.assertTrue((estimate_background(frames) == 0).all())


class VideoRoundTripTests(unittest.TestCase):
    def write_source(self, path: Path, frames: list[np.ndarray], fps: float = 30.0) -> None:
        write_video(frames, path, fps)

    def test_frame_count_is_padded_to_the_container_header(self) -> None:
        """Variants must sit on the source's frame grid or the clip starts diverge."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mp4"
            frames = [np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
            self.write_source(path, frames)

            self.assertEqual(len(read_all_frames(path, target_frames=5)), 5)
            self.assertEqual(len(read_all_frames(path, target_frames=2)), 2)

    def test_build_variant_video_writes_a_cropped_video_and_manifest_row(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "v.mp4"
            frames = []
            for index in range(6):
                frame = np.full((HEIGHT, WIDTH, 3), 5, dtype=np.uint8)
                frame[12:36, 16:48] = 180 + index
                frames.append(frame)
            self.write_source(source, frames)

            pose_path = root / "v.json"
            pose_path.write_text(
                json.dumps(
                    pose_document(
                        [[landmark(16 / WIDTH, 12 / HEIGHT), landmark(48 / WIDTH, 36 / HEIGHT)] for _ in frames],
                        total_frames=len(frames),
                    )
                ),
                encoding="utf-8",
            )

            output = root / "out" / "v.mp4"
            row = build_variant_video(source, pose_path, output, "person_crop")

            self.assertTrue(output.exists())
            self.assertTrue(row["pose_detected"])
            self.assertEqual(row["written_frames"], len(frames))
            self.assertEqual(row["frame_size"][0], row["frame_size"][1])

            cap = cv2.VideoCapture(str(output))
            try:
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), row["frame_size"][0])
            finally:
                cap.release()

    def test_build_variant_video_reports_a_pose_less_video(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "v.mp4"
            frames = [np.full((HEIGHT, WIDTH, 3), 7, dtype=np.uint8) for _ in range(4)]
            self.write_source(source, frames)

            pose_path = root / "v.json"
            pose_path.write_text(
                json.dumps(pose_document([[landmark(0.5, 0.5, visibility=0.0)] for _ in frames])),
                encoding="utf-8",
            )

            row = build_variant_video(source, pose_path, root / "out.mp4", "background_only")

            self.assertFalse(row["pose_detected"])
            self.assertIsNone(row["box"])


if __name__ == "__main__":
    unittest.main()
