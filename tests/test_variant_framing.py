from __future__ import annotations

import unittest

from src.video.variant_geometry import Box
from src.video.variant_framing import (
    PROCESSOR_SIZE,
    frame_variant,
    processor_view,
    select_rows,
    split_counts,
    summarize_manifest,
    truncation_cause,
    variant_input,
)


class ProcessorViewTests(unittest.TestCase):
    def test_a_square_frame_loses_nothing_to_the_centre_crop(self) -> None:
        area, survival, top_loss, bottom_loss = processor_view(480, 480, Box(120, 60, 360, 420))
        self.assertAlmostEqual(survival, 1.0, places=6)
        self.assertAlmostEqual(area, (240 * 360) / (480 * 480), places=6)
        self.assertEqual((top_loss, bottom_loss), (0.0, 0.0))

    def test_a_tall_frame_truncates_a_box_that_reaches_its_edges(self) -> None:
        """480x852 is a real Fitness-AQA frame size; the resize makes the height 398
        and the centre crop keeps rows 87..311, so a full-height athlete loses both
        ends. This is the 38% truncation the plan's F2 factor is about."""
        area, survival, top_loss, bottom_loss = processor_view(480, 852, Box(0, 0, 480, 852))
        self.assertLess(survival, 0.6)
        self.assertAlmostEqual(area, 1.0, places=6)
        self.assertAlmostEqual(top_loss, bottom_loss, places=6)

    def test_the_end_that_goes_is_reported_separately_from_how_much(self) -> None:
        """Area is the wrong severity metric for F2: the same 12% off the bottom is
        the ankles, and squat depth is judged at the ankles. A box sitting low in a
        tall frame loses its feet and keeps its head, and the two must not average."""
        _, survival, top_loss, bottom_loss = processor_view(480, 900, Box(100, 400, 380, 900))
        self.assertLess(survival, 1.0)
        self.assertEqual(top_loss, 0.0)
        self.assertGreater(bottom_loss, 0.0)

    def test_losses_are_never_negative_zero(self) -> None:
        _, _, top_loss, bottom_loss = processor_view(480, 480, Box(0, 0, 480, 480))
        self.assertEqual(str(top_loss), "0.0")
        self.assertEqual(str(bottom_loss), "0.0")

    def test_a_box_entirely_outside_the_crop_window_survives_nothing(self) -> None:
        """A box pinned to the top of a very tall frame falls above the crop window
        entirely, so the whole athlete is lost off the TOP and the loss saturates."""
        _, survival, top_loss, bottom_loss = processor_view(480, 2000, Box(0, 0, 480, 100))
        self.assertEqual(survival, 0.0)
        self.assertEqual(top_loss, 1.0)
        self.assertEqual(bottom_loss, 0.0)

    def test_a_zero_sized_frame_is_an_error_not_a_division(self) -> None:
        with self.assertRaises(ValueError):
            processor_view(0, 480, Box(0, 0, 1, 1))


class VariantInputTests(unittest.TestCase):
    def test_letterbox_pads_to_the_long_edge_and_shifts_the_box(self) -> None:
        width, height, box, transformed = variant_input("full_frame_letterbox", 480, 852, Box(100, 0, 300, 852))
        self.assertEqual((width, height), (852, 852))
        self.assertEqual(box.as_tuple(), (286, 0, 486, 852))  # left pad = (852-480)//2 = 186
        self.assertTrue(transformed)

    def test_letterbox_reports_an_already_square_frame_as_untransformed(self) -> None:
        """47.3% of the corpus. Counting these as failures would be the wrong fix."""
        _, _, _, transformed = variant_input("full_frame_letterbox", 480, 480, Box(1, 1, 2, 2))
        self.assertFalse(transformed)

    def test_both_crop_arms_place_the_whole_box_inside_their_input(self) -> None:
        for variant, expected in (("person_crop", (240, 240)), ("person_crop_centercrop", (100, 240))):
            width, height, _, _ = variant_input(variant, 480, 600, Box(50, 100, 150, 340))
            self.assertEqual((width, height), expected)

    def test_full_frame_and_background_only_share_the_untouched_geometry(self) -> None:
        box = Box(10, 20, 30, 40)
        self.assertEqual(
            variant_input("full_frame", 480, 600, box),
            variant_input("background_only", 480, 600, box),
        )

    def test_an_unknown_variant_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            variant_input("greyscale", 480, 480, Box(0, 0, 1, 1))


class FrameVariantTests(unittest.TestCase):
    def test_letterboxing_restores_the_athlete_but_shrinks_them(self) -> None:
        """The design limitation that must be stated before any score is seen: this
        arm moves F2 up and F3 down, so two equal arms are also consistent with the
        two effects cancelling."""
        box = Box(60, 0, 420, 852)
        cropped = frame_variant("full_frame", 480, 852, box)
        padded = frame_variant("full_frame_letterbox", 480, 852, box)

        self.assertLess(cropped.box_survival, 1.0)
        self.assertAlmostEqual(padded.box_survival, 1.0, places=6)
        self.assertLess(padded.body_area_fraction, cropped.body_area_fraction)

    def test_the_letterboxed_crop_never_truncates_and_the_bare_crop_does(self) -> None:
        box = Box(180, 40, 300, 560)  # a tall narrow athlete
        padded = frame_variant("person_crop", 480, 600, box)
        bare = frame_variant("person_crop_centercrop", 480, 600, box)

        self.assertFalse(padded.truncated)
        self.assertTrue(bare.truncated)
        self.assertAlmostEqual(bare.box_survival, 120 / 520, places=2)

    def test_the_person_crop_body_fraction_is_the_box_aspect_ratio(self) -> None:
        """Measured: person_crop pins the athlete's height to the canvas in 99% of
        videos, so its F3 varies only through the box's width-to-height ratio."""
        framing = frame_variant("person_crop", 480, 600, Box(100, 50, 200, 450))
        self.assertAlmostEqual(framing.body_area_fraction, 100 / 400, places=3)
        self.assertEqual(framing.input_size, (400, 400))


class SummarizeManifestTests(unittest.TestCase):
    def rows(self) -> list[dict]:
        return [
            {"video_id": "square", "frame_size": [480, 480], "box": [120, 40, 300, 440]},
            {"video_id": "tall", "frame_size": [480, 852], "box": [100, 20, 340, 800]},
            {"video_id": "boxless", "frame_size": [480, 600], "box": None},
        ]

    def test_boxless_rows_are_counted_and_excluded(self) -> None:
        summary = summarize_manifest(self.rows())
        self.assertEqual(summary["n_rows"], 3)
        self.assertEqual(summary["n_boxless"], 1)

    def test_the_letterbox_arm_reports_its_no_op_videos_rather_than_hiding_them(self) -> None:
        summary = summarize_manifest(self.rows())["arms"]["full_frame_letterbox"]
        self.assertEqual(summary["n_identical_to_source"], 1)
        self.assertEqual(summary["n_transformed"], 1)

    def test_the_letterboxed_arms_truncate_nothing(self) -> None:
        """A box that fits exactly must not be reported as truncated: the resize
        arithmetic returns 1 - 1e-16, and taking that literally would mark every
        letterboxed video a truncation and erase the contrast this arm exists for."""
        summary = summarize_manifest(self.rows())["arms"]
        self.assertEqual(summary["person_crop"]["n_truncated"], 0)
        self.assertEqual(summary["full_frame_letterbox"]["n_truncated"], 0)
        self.assertLessEqual(summary["person_crop"]["box_survival"]["p10"], 1.0)

    def test_every_arm_reports_every_percentile_summary(self) -> None:
        summary = summarize_manifest(self.rows())["arms"]
        for arm in summary.values():
            for key in ("body_area_fraction", "box_survival", "top_loss", "bottom_loss"):
                self.assertEqual(set(arm[key]), {"p10", "median", "p90"})
            self.assertLessEqual(arm["body_area_fraction"]["median"], 1.0)

    def test_subsets_narrow_to_the_videos_the_manipulation_can_reach(self) -> None:
        """The marginal over all videos dilutes F2 with videos it cannot touch."""
        rows = self.rows()
        self.assertEqual(summarize_manifest(rows, subset="all")["n_selected"], 2)
        self.assertEqual(summarize_manifest(rows, subset="non_square")["n_selected"], 1)

    def test_an_unknown_subset_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_rows(self.rows(), "everything")

    def test_processor_size_matches_the_checkpoint_config(self) -> None:
        self.assertEqual(PROCESSOR_SIZE, 224)


class TruncationCauseTests(unittest.TestCase):
    """The two causes have different prices, so they must not be pooled."""

    def test_a_box_taller_than_the_crop_window_is_a_scale_problem(self) -> None:
        self.assertEqual(truncation_cause(480, 900, Box(100, 20, 340, 880)), "scale")

    def test_a_box_that_fits_but_sits_low_is_a_framing_problem(self) -> None:
        """Half of the 613 truncated squats. Re-centring the crop window on the
        athlete restores them at no zoom cost, so F2 need not cost 30% body area."""
        self.assertEqual(truncation_cause(480, 900, Box(100, 500, 340, 900)), "framing")

    def test_an_untruncated_video_has_no_cause(self) -> None:
        self.assertEqual(truncation_cause(480, 480, Box(100, 100, 300, 300)), "none")


class SplitCountTests(unittest.TestCase):
    def rows(self) -> list[dict]:
        return [
            {"video_id": "a", "frame_size": [480, 480], "box": [10, 10, 20, 20]},
            {"video_id": "b", "frame_size": [480, 852], "box": [10, 10, 20, 20]},
            {"video_id": "c", "frame_size": [480, 852], "box": [10, 10, 20, 20]},
        ]

    def test_counts_are_reported_per_split_for_the_chosen_subset(self) -> None:
        split_map = {"a": "train", "b": "test", "c": "test"}
        self.assertEqual(split_counts(self.rows(), split_map, "non_square"), {"test": 2})

    def test_a_video_missing_from_the_split_map_is_surfaced_not_dropped(self) -> None:
        """Silently dropping it would overstate how much test data a contrast has."""
        counts = split_counts(self.rows(), {"a": "train"}, "all")
        self.assertEqual(counts["unassigned"], 2)


if __name__ == "__main__":
    unittest.main()
