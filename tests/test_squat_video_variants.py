from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import cv2
import numpy as np

from src.video.squat_video_variants import (
    Box,
    apply_variant,
    build_variant_video,
    describe_variant,
    fill_box_from_surroundings,
    person_box_from_pose,
    read_all_frames,
    expand_box,
    letterbox_to_square,
    verify_variant_video,
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


class CropBoxTests(unittest.TestCase):
    def test_margin_grows_the_box_without_changing_its_aspect(self) -> None:
        box = expand_box(Box(20, 10, 30, 30), WIDTH, HEIGHT, margin=0.2)
        self.assertEqual(box.as_tuple(), (19, 8, 31, 32))

    def test_expansion_never_leaves_the_frame(self) -> None:
        box = expand_box(Box(0, 0, WIDTH, HEIGHT), WIDTH, HEIGHT, margin=0.5)
        self.assertEqual(box.as_tuple(), (0, 0, WIDTH, HEIGHT))

    def test_a_tall_person_keeps_far_less_background_than_a_square_expansion(self) -> None:
        """The measured reason the crop letterboxes instead of expanding: on 300 train
        videos a square expansion kept a median 77% of the frame, the landmark box 25%."""
        tall = Box(24, 2, 40, 46)  # 16 x 44, a standing athlete
        cropped = expand_box(tall, WIDTH, HEIGHT, margin=0.0)
        kept = (cropped.x1 - cropped.x0) * (cropped.y1 - cropped.y0) / (WIDTH * HEIGHT)
        squared = max(tall.x1 - tall.x0, tall.y1 - tall.y0) ** 2 / (WIDTH * HEIGHT)
        self.assertLess(kept, 0.3)
        self.assertGreater(squared, 0.6)

    def test_letterbox_squares_the_frame_without_distorting_it(self) -> None:
        frame = np.zeros((40, 10, 3), dtype=np.uint8)
        frame[:] = 200
        padded = letterbox_to_square(frame)
        self.assertEqual(padded.shape[:2], (40, 40))
        self.assertTrue((padded[:, 15:25] == 200).all())
        self.assertTrue((padded[:, 0:14] == 114).all())

    def test_letterbox_leaves_an_already_square_frame_alone(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertIs(letterbox_to_square(frame), frame)


class ApplyVariantTests(unittest.TestCase):
    def make_frames(self, count: int = 5) -> list[np.ndarray]:
        frames = []
        for index in range(count):
            frame = np.full((HEIGHT, WIDTH, 3), 10, dtype=np.uint8)
            frame[10:30, 20:40] = 200 + index  # the "person"
            frames.append(frame)
        return frames

    def test_person_crop_keeps_only_the_box_and_pads_it_square(self) -> None:
        cropped = apply_variant(self.make_frames(), "person_crop", Box(24, 10, 36, 30))
        self.assertEqual(cropped[0].shape[:2], (20, 20))  # 12x20 letterboxed to 20x20
        self.assertTrue((cropped[0][:, 4:16] >= 200).all())
        self.assertTrue((cropped[0][:, 0:3] == 114).all())

    def test_background_only_replaces_the_person_with_the_surrounding_scene(self) -> None:
        """The fill must land on the *background* value, not the person's.

        A per-pixel temporal median passes a "nothing varies inside the box" check
        while painting a smeared athlete, because a squatter covers those pixels in
        every frame. Asserting the filled value matches the scene is what
        distinguishes the two.
        """
        frames = self.make_frames()
        masked = apply_variant(frames, "background_only", Box(20, 10, 40, 30))

        self.assertEqual(masked[0].shape, frames[0].shape)
        inside = masked[0][10:30, 20:40]
        self.assertTrue((inside == 10).all(), f"filled with {np.unique(inside)}, expected the scene value 10")
        # Outside the box the scene is untouched.
        self.assertTrue((masked[0][0:10, 0:20] == 10).all())

    def test_background_fill_ramps_between_the_two_flanking_columns(self) -> None:
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[:, 0] = 0
        frame[:, 1:5] = 255  # the "person"
        frame[:, 5] = 100
        filled = fill_box_from_surroundings(frame, Box(1, 0, 5, 4))
        self.assertEqual([int(v) for v in filled[0, 1:5, 0]], [0, 33, 67, 100])

    def test_background_fill_uses_one_side_when_the_box_touches_an_edge(self) -> None:
        frame = np.full((4, 6, 3), 20, dtype=np.uint8)
        frame[:, 0:4] = 200
        filled = fill_box_from_surroundings(frame, Box(0, 0, 4, 4))
        self.assertTrue((filled[:, 0:4] == 20).all())

    def test_background_fill_falls_back_to_a_vertical_blend_across_the_full_width(self) -> None:
        frame = np.zeros((5, 4, 3), dtype=np.uint8)
        frame[0] = 10
        frame[1:4] = 250
        frame[4] = 30
        filled = fill_box_from_surroundings(frame, Box(0, 1, 4, 4))
        self.assertTrue((filled[1:4] < 250).all())
        self.assertEqual(int(filled[1, 0, 0]), 10)

    def test_missing_box_leaves_the_video_untouched(self) -> None:
        frames = self.make_frames()
        for variant in ("person_crop", "background_only"):
            result = apply_variant(frames, variant, None)
            self.assertTrue(np.array_equal(result[0], frames[0]))

    def test_unknown_variant_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_variant(self.make_frames(), "grayscale", Box(0, 0, 4, 4))

    def test_fill_carries_no_body_structure_even_when_the_person_never_moves(self) -> None:
        """The failure the temporal median had: a static athlete IS the median.

        The fill is a blend of two columns *outside* the box, so whatever shape the
        person has inside it cannot appear in the output -- every filled row is
        constant-to-linear regardless of what the athlete was doing.
        """
        frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
        frame[12:28, 22:38] = 240  # a body-shaped blob, identical in every frame
        frame[16:20, 24:28] = 90  # ... with internal structure
        filled = fill_box_from_surroundings(frame, Box(20, 10, 40, 30))
        self.assertEqual(len(np.unique(filled[10:30, 20:40])), 1)
        self.assertEqual(int(np.unique(filled[10:30, 20:40])[0]), 30)


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



class ReencodedVariantTests(unittest.TestCase):
    """The identity arm exists to price the extra lossy generation the two box
    variants pay and the untouched full_frame arm does not."""

    def test_reencoded_returns_the_frames_untouched(self) -> None:
        frames = [np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
        result = apply_variant(frames, "reencoded", Box(10, 10, 20, 20))
        for original, produced in zip(frames, result):
            self.assertTrue(np.array_equal(original, produced))

    def test_reencoded_ignores_a_missing_box(self) -> None:
        frames = [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)]
        self.assertTrue(np.array_equal(apply_variant(frames, "reencoded", None)[0], frames[0]))


class AtomicWriteTests(unittest.TestCase):
    """A killed build must not leave a truncated video that the resume path skips."""

    def test_no_partial_file_survives_a_successful_write(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "v.mp4"
            write_video([np.zeros((8, 8, 3), dtype=np.uint8)] * 3, out, 30.0)
            self.assertTrue(out.exists())
            self.assertEqual(list(Path(tmp).glob("*.partial.mp4")), [])

    def test_a_failed_encode_leaves_the_destination_absent(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "v.mp4"
            with mock.patch.object(cv2.VideoWriter, "write", side_effect=RuntimeError("disk full")):
                with self.assertRaises(RuntimeError):
                    write_video([np.zeros((8, 8, 3), dtype=np.uint8)] * 3, out, 30.0)
            self.assertFalse(out.exists())


class VerifyVariantTests(unittest.TestCase):
    def test_a_matching_video_reports_no_problem(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "v.mp4"
            write_video([np.zeros((8, 8, 3), dtype=np.uint8)] * 4, out, 30.0)
            self.assertIsNone(verify_variant_video(out, 4))

    def test_a_truncated_video_reports_its_actual_count(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "v.mp4"
            write_video([np.zeros((8, 8, 3), dtype=np.uint8)] * 4, out, 30.0)
            self.assertEqual(verify_variant_video(out, 7), 4)

    def test_a_zero_byte_stub_is_reported_rather_than_crashing(self) -> None:
        """The exact shape of the six corrupt files an interrupted build left behind."""
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "v.mp4"
            out.write_bytes(b"")
            self.assertEqual(verify_variant_video(out, 10), 0)

    def test_a_missing_output_counts_as_zero_frames(self) -> None:
        with TemporaryDirectory() as tmp:
            self.assertEqual(verify_variant_video(Path(tmp) / "nope.mp4", 5), 0)


class DescribeVariantTests(unittest.TestCase):
    """Every video must carry a box, whether or not its video file was encoded.

    A row without a box used to be indistinguishable from a row whose box is null
    ("no person visible"), and the extractor treats null as "leave untouched" -- so
    the resume path silently fed untransformed videos into the control arms: 51% of
    person_crop and 26% of background_only.
    """

    def make_pose(self, tmp: Path, visible: bool = True) -> Path:
        marks = [landmark(0.25, 0.2), landmark(0.5, 0.8)] if visible else [landmark(0.5, 0.5, visibility=0.0)]
        path = tmp / "v.json"
        path.write_text(json.dumps(pose_document([marks, marks], total_frames=2)), encoding="utf-8")
        return path

    def test_box_is_produced_without_encoding_a_video(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = describe_variant(root / "v.mp4", self.make_pose(root), "person_crop")

            self.assertIn("box", row)
            self.assertTrue(row["pose_detected"])
            self.assertEqual(row["video_id"], "v")
            self.assertEqual(list(Path(tmp).glob("*.mp4")), [])

    def test_person_crop_box_is_the_expanded_one_and_background_the_raw_one(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pose = self.make_pose(root)
            crop = describe_variant(root / "v.mp4", pose, "person_crop")["box"]
            background = describe_variant(root / "v.mp4", pose, "background_only")["box"]
            self.assertNotEqual(crop, background)
            self.assertLessEqual(crop[0], background[0])

    def test_a_person_less_video_records_an_explicit_null_box(self) -> None:
        """Null is a real state and must stay distinguishable from 'not recorded'."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = describe_variant(root / "v.mp4", self.make_pose(root, visible=False), "background_only")
            self.assertIn("box", row)
            self.assertIsNone(row["box"])
            self.assertFalse(row["pose_detected"])


if __name__ == "__main__":
    unittest.main()
