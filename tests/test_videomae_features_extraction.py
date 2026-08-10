import unittest

import numpy as np

from src.rehab24.videomae_features import (
    assert_fc_norm_pretrained,
    group_rows_by_video,
    sample_clip_starts,
)


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


if __name__ == "__main__":
    unittest.main()
