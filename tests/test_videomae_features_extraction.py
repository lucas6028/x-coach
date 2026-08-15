import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from unittest import mock

from src.rehab24.videomae_features import (
    FRAMING_VARIANTS,
    assert_fc_norm_pretrained,
    assert_output_dir_matches_variant,
    group_rows_by_video,
    sample_clip_starts,
    save_feature,
    transform_frames,
)
from src.video.variant_geometry import LETTERBOX_FILL, Box


def row(sample_id: str, video_path: str, first_frame: int) -> dict[str, str]:
    return {"sample_id": sample_id, "video_path": video_path, "first_frame": str(first_frame)}


class GroupRowsByVideoTest(unittest.TestCase):
    def test_groups_every_row_of_a_video_together(self):
        rows = [row("a", "Ex1/v1.mp4", 100), row("b", "Ex2/v2.mp4", 5), row("c", "Ex1/v1.mp4", 10)]
        grouped = group_rows_by_video(rows)
        self.assertEqual([video for video, _ in grouped], ["Ex1/v1.mp4", "Ex2/v2.mp4"])
        self.assertEqual([r["sample_id"] for r in grouped[0][1]], ["c", "a"])

    def test_orders_rows_by_start_frame_within_a_video(self):
        """One capture is reused across a video's ~16 repetitions, so rows must be
        visited in increasing frame order to keep decoding forward."""
        rows = [row("a", "v.mp4", 900), row("b", "v.mp4", 30), row("c", "v.mp4", 400)]
        _, video_rows = group_rows_by_video(rows)[0]
        starts = [int(r["first_frame"]) for r in video_rows]
        self.assertEqual(starts, sorted(starts))

    def test_preserves_all_rows(self):
        rows = [row(str(i), f"v{i % 3}.mp4", i) for i in range(20)]
        grouped = group_rows_by_video(rows)
        self.assertEqual(sum(len(video_rows) for _, video_rows in grouped), 20)

    def test_handles_empty_input(self):
        self.assertEqual(group_rows_by_video([]), [])


class SampleClipStartsTest(unittest.TestCase):
    def test_returns_requested_number_of_clips(self):
        self.assertEqual(len(sample_clip_starts(180, 377, 16, 2, 4)), 4)

    def test_starts_are_non_decreasing_and_within_the_repetition(self):
        starts = sample_clip_starts(180, 377, 16, 2, 4)
        self.assertEqual(starts, sorted(starts))
        self.assertGreaterEqual(starts[0], 179)
        self.assertLessEqual(starts[-1], 377)

    def test_single_clip_is_centred(self):
        self.assertEqual(len(sample_clip_starts(180, 377, 16, 2, 1)), 1)

    def test_short_repetition_does_not_produce_negative_starts(self):
        starts = sample_clip_starts(1, 5, 16, 2, 4)
        self.assertTrue(all(start >= 0 for start in starts))


class AssertFcNormPretrainedTest(unittest.TestCase):
    def test_rejects_default_layer_norm_init(self):
        with self.assertRaises(SystemExit):
            assert_fc_norm_pretrained(np.ones(768, dtype=np.float32), np.zeros(768, dtype=np.float32), "m")

    def test_accepts_loaded_weights(self):
        rng = np.random.default_rng(0)
        assert_fc_norm_pretrained(rng.normal(size=768).astype(np.float32), rng.normal(size=768).astype(np.float32), "m")

    def test_accepts_weights_that_differ_only_in_bias(self):
        assert_fc_norm_pretrained(np.ones(768, dtype=np.float32), np.full(768, 0.1, dtype=np.float32), "m")


def frames(width: int = 1920, height: int = 1080, count: int = 3) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8) for _ in range(count)]


class TransformFramesTest(unittest.TestCase):
    """The framing arms' pixel transform, applied between decode and the processor."""

    def test_full_frame_is_the_untouched_source(self):
        source = frames(64, 32, count=2)
        result = transform_frames(source, "full_frame", None)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(result, source)))

    def test_letterbox_squares_a_landscape_frame_without_cropping_or_stretching(self):
        """cam17 is 1920x1080. Padding to 1920x1920 makes the processor's shortest-edge
        resize plus 224 centre crop a no-op, which is the whole primary manipulation."""
        source = frames(1920, 1080, count=2)
        result = transform_frames(source, "full_frame_letterbox", None)
        for original, padded in zip(source, result):
            self.assertEqual(padded.shape, (1920, 1920, 3))
            top = (1920 - 1080) // 2
            np.testing.assert_array_equal(padded[top : top + 1080, :, :], original)

    def test_letterbox_squares_a_portrait_frame_the_same_way(self):
        """cam18 is 1080x1920 -- the camera whose feet the centre crop can take."""
        result = transform_frames(frames(1080, 1920, count=1), "full_frame_letterbox", None)
        self.assertEqual(result[0].shape, (1920, 1920, 3))

    def test_letterbox_pads_with_the_neutral_grey_and_nothing_else(self):
        padded = transform_frames(frames(1920, 1080, count=1), "full_frame_letterbox", None)[0]
        self.assertTrue(np.all(padded[:100] == LETTERBOX_FILL))
        self.assertTrue(np.all(padded[-100:] == LETTERBOX_FILL))

    def test_letterbox_needs_no_box(self):
        """It must never consult one: a box-shaped dependency here would give the arm a
        null-box path, and a null box means 'leave untouched' one call deeper."""
        result = transform_frames(frames(1920, 1080, count=1), "full_frame_letterbox", None)
        self.assertNotEqual(result[0].shape, (1080, 1920, 3))

    def test_letterbox_changes_the_pixels_on_every_rehab24_frame_size(self):
        """Both REHAB24-6 cameras are non-square, so unlike Fitness-AQA there is no
        video for which this arm is a no-op. A bit-identical output means the transform
        did not run."""
        for width, height in ((1920, 1080), (1080, 1920)):
            source = frames(width, height, count=1)
            result = transform_frames(source, "full_frame_letterbox", None)
            self.assertNotEqual(result[0].shape, source[0].shape)

    def test_box_variants_refuse_to_run_without_a_box(self):
        for variant in ("person_crop", "background_only"):
            with self.subTest(variant=variant), self.assertRaises(RuntimeError):
                transform_frames(frames(640, 480, count=1), variant, None)

    def test_person_crop_and_background_only_take_the_same_box(self):
        box = Box(100, 50, 300, 400)
        source = frames(640, 480, count=1)
        cropped = transform_frames([f.copy() for f in source], "person_crop", box)[0]
        painted = transform_frames([f.copy() for f in source], "background_only", box)[0]
        self.assertEqual(cropped.shape, (350, 350, 3))  # 200x350 crop, letterboxed square
        self.assertFalse(np.array_equal(painted[50:400, 100:300], source[0][50:400, 100:300]))
        np.testing.assert_array_equal(painted[:50], source[0][:50])

    def test_every_advertised_variant_is_applicable(self):
        box = Box(10, 10, 100, 200)
        for variant in FRAMING_VARIANTS:
            with self.subTest(variant=variant):
                self.assertEqual(len(transform_frames(frames(640, 480, count=2), variant, box)), 2)


class SaveFeatureAtomicityTest(unittest.TestCase):
    """The resume path skips whatever already exists, so a half-written bundle would be
    accepted as a finished repetition rather than re-extracted."""

    def row(self) -> dict[str, str]:
        return {
            "sample_id": "Ex6_a_rep1_cam17",
            "video_id": "a",
            "exercise_id": "6",
            "person_id": "1",
            "camera": "cam17",
            "correctness": "1",
        }

    def bundle(self) -> dict[str, np.ndarray]:
        return {
            "clip_features_mean_pool_fc_norm": np.ones((4, 768), dtype=np.float32),
            "clip_starts": np.asarray([0, 10, 20, 30], dtype=np.int32),
        }

    def test_writes_a_readable_bundle_and_leaves_no_temp_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "train" / "Ex6_a_rep1_cam17.npz"
            save_feature(path, self.row(), self.bundle(), {"variant": "full_frame_letterbox"})
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["clip_features_mean_pool_fc_norm"].shape, (4, 768))
                self.assertEqual(str(data["provenance_variant"]), "full_frame_letterbox")
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["Ex6_a_rep1_cam17.npz"])

    def test_a_crash_mid_write_leaves_no_bundle_at_the_target_path(self):
        """Without the rename this is exactly how a truncated .npz survives a kill and
        is later counted as done."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "train" / "Ex6_a_rep1_cam17.npz"
            with mock.patch("numpy.savez_compressed", side_effect=OSError("killed mid-write")):
                with self.assertRaises(OSError):
                    save_feature(path, self.row(), self.bundle(), {})
            self.assertFalse(path.exists())

    def test_overwrites_an_existing_bundle_in_place(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "train" / "Ex6_a_rep1_cam17.npz"
            save_feature(path, self.row(), self.bundle(), {})
            replacement = self.bundle()
            replacement["clip_features_mean_pool_fc_norm"] = np.full((4, 768), 7.0, dtype=np.float32)
            save_feature(path, self.row(), replacement, {})
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(float(data["clip_features_mean_pool_fc_norm"][0, 0]), 7.0)


class OutputDirVariantGuardTest(unittest.TestCase):
    def write(self, directory: Path, name: str, **stamps: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / name, video_feature=np.zeros(4), **stamps)

    def test_accepts_an_empty_or_missing_dir(self):
        with TemporaryDirectory() as tmp:
            assert_output_dir_matches_variant(Path(tmp) / "nope", "full_frame_letterbox")
            assert_output_dir_matches_variant(Path(tmp), "full_frame_letterbox")

    def test_accepts_a_dir_already_holding_the_same_variant(self):
        with TemporaryDirectory() as tmp:
            self.write(Path(tmp) / "train", "a.npz", provenance_variant=np.asarray("full_frame_letterbox"))
            assert_output_dir_matches_variant(Path(tmp), "full_frame_letterbox")

    def test_rejects_writing_one_variant_into_another_variants_dir(self):
        """Nothing downstream would notice: the audit's single-provenance check runs per
        directory, and the resume path skips whatever already exists."""
        with TemporaryDirectory() as tmp:
            self.write(Path(tmp) / "train", "a.npz", provenance_variant=np.asarray("full_frame_letterbox"))
            with self.assertRaises(SystemExit):
                assert_output_dir_matches_variant(Path(tmp), "person_crop")

    def test_treats_an_unstamped_bundle_as_full_frame(self):
        """Bundles extracted before --variant existed carry no stamp; they are all
        full_frame, and appending a variant to that dir must still be refused."""
        with TemporaryDirectory() as tmp:
            self.write(Path(tmp) / "train", "a.npz")
            assert_output_dir_matches_variant(Path(tmp), "full_frame")
            with self.assertRaises(SystemExit):
                assert_output_dir_matches_variant(Path(tmp), "full_frame_letterbox")


if __name__ == "__main__":
    unittest.main()
