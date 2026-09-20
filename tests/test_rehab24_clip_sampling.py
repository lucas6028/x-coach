import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

# src.rehab24.clip_sampling decodes video with OpenCV, which the lean CI dependency
# set does not install; without this the whole module is a collection error.
pytest.importorskip("cv2")

from src.rehab24.clip_sampling import (
    assert_resume_provenance_matches,
    build_bounded_clips,
    clip_frame_indices,
    decode_frames_at_indices,
    provenance_mismatches,
    rep_index_bounds,
    sample_clip_starts,
    sha256_of_files,
)


class RepIndexBoundsTest(unittest.TestCase):
    """Frame-number (1-based, inclusive) -> decoder-index (0-based, inclusive) conversion."""

    def test_first_frame_one_is_decoder_index_zero(self):
        self.assertEqual(rep_index_bounds(1, 100), (0, 99))

    def test_last_frame_is_inclusive(self):
        first_index, last_index = rep_index_bounds(180, 377)
        self.assertEqual(first_index, 179)
        self.assertEqual(last_index, 376)

    def test_last_index_never_precedes_first_index(self):
        # A malformed one-frame-or-negative-length row must not yield an empty range.
        self.assertEqual(rep_index_bounds(50, 50), (49, 49))
        self.assertEqual(rep_index_bounds(50, 10), (49, 49))


class ClipFrameIndicesTest(unittest.TestCase):
    def test_no_clamping_when_the_repetition_is_long_enough(self):
        first_index, last_index = rep_index_bounds(1, 1000)
        indices = clip_frame_indices(0, 16, 2, first_index, last_index, total_frames=2000)
        self.assertEqual(np.unique(indices).size, 16)
        expected = np.arange(0, 32, 2, dtype=np.int64)
        np.testing.assert_array_equal(indices, expected)

    def test_short_repetition_pads_by_repeating_the_last_in_repetition_frame(self):
        # 5-frame repetition (decoder indices 0..4); a 16x2 clip cannot avoid padding.
        first_index, last_index = rep_index_bounds(1, 5)
        indices = clip_frame_indices(0, 16, 2, first_index, last_index, total_frames=100)
        self.assertTrue(np.all(indices >= first_index))
        self.assertTrue(np.all(indices <= last_index))
        # The raw (unclamped) sequence would be 0,2,4,...,30 -- everything past 4 clamps to 4.
        np.testing.assert_array_equal(indices[:3], [0, 2, 4])
        self.assertTrue(np.all(indices[3:] == last_index))

    def test_never_exceeds_total_frames_minus_one_even_if_last_index_would_allow_it(self):
        # The repetition's own bound says up to 500, but the decoded video is shorter.
        indices = clip_frame_indices(0, 16, 2, first_index=0, last_index=500, total_frames=50)
        self.assertTrue(np.all(indices <= 49))

    def test_raises_when_the_repetition_bound_is_entirely_after_the_decoded_video(self):
        with self.assertRaises(ValueError):
            clip_frame_indices(600, 16, 2, first_index=600, last_index=700, total_frames=50)


class BuildBoundedClipsTest(unittest.TestCase):
    def test_returns_requested_number_of_clips(self):
        clips, _, _ = build_bounded_clips(180, 377, total_frames=1000, clip_length=16, frame_stride=2, num_clips=4)
        self.assertEqual(len(clips), 4)

    def test_vm16_span_is_31_frames_before_clipping(self):
        clips, first_index, _ = build_bounded_clips(1, 1000, total_frames=2000, clip_length=16, frame_stride=2, num_clips=4)
        for clip in clips:
            self.assertEqual(clip["frame_indices"][-1] - clip["frame_indices"][0], 30)

    def test_vj64_span_is_127_frames_before_clipping(self):
        clips, first_index, _ = build_bounded_clips(1, 2000, total_frames=3000, clip_length=64, frame_stride=2, num_clips=4)
        for clip in clips:
            self.assertEqual(clip["frame_indices"][-1] - clip["frame_indices"][0], 126)

    def test_vm16_and_vj16_configurations_draw_byte_identical_indices(self):
        """vj16 uses vm16's exact clip length and stride, so with the same repetition
        bounds and clip count the two arms must draw exactly the same frame indices."""
        vm16_clips, vm16_first, vm16_last = build_bounded_clips(
            212, 340, total_frames=900, clip_length=16, frame_stride=2, num_clips=4
        )
        vj16_clips, vj16_first, vj16_last = build_bounded_clips(
            212, 340, total_frames=900, clip_length=16, frame_stride=2, num_clips=4
        )
        self.assertEqual(vm16_first, vj16_first)
        self.assertEqual(vm16_last, vj16_last)
        for a, b in zip(vm16_clips, vj16_clips):
            np.testing.assert_array_equal(a["frame_indices"], b["frame_indices"])

    def test_padding_fraction_arithmetic(self):
        clips, _, _ = build_bounded_clips(1, 5, total_frames=100, clip_length=16, frame_stride=2, num_clips=1)
        clip = clips[0]
        expected_unique = np.unique(clip["frame_indices"]).size
        self.assertEqual(clip["unique_frame_count"], expected_unique)
        self.assertAlmostEqual(clip["padding_fraction"], 1.0 - expected_unique / 16)

    def test_duplicate_clips_in_short_repetitions_are_retained_not_discarded(self):
        clips, _, _ = build_bounded_clips(1, 3, total_frames=50, clip_length=16, frame_stride=2, num_clips=4)
        self.assertEqual(len(clips), 4)


class SampleClipStartsTest(unittest.TestCase):
    """Unchanged from the historical implementation -- vm16 must keep drawing the
    exact historical starts."""

    def test_returns_requested_number_of_clips(self):
        self.assertEqual(len(sample_clip_starts(180, 377, 16, 2, 4)), 4)

    def test_starts_are_non_decreasing_and_within_the_repetition(self):
        starts = sample_clip_starts(180, 377, 16, 2, 4)
        self.assertEqual(starts, sorted(starts))
        self.assertGreaterEqual(starts[0], 179)
        self.assertLessEqual(starts[-1], 377)


class FakeCapture:
    """A minimal cv2.VideoCapture stand-in: `frames[i]` is returned once the
    cursor reaches `i`; `set`/`grab`/`read` mirror the real API's semantics
    (BGR frames, since `decode_frames_at_indices` converts to RGB itself)."""

    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames
        self.cursor = 0

    def set(self, prop, value):
        self.cursor = int(value)

    def grab(self):
        self.cursor += 1
        return self.cursor < len(self.frames)

    def read(self):
        if self.cursor >= len(self.frames):
            return False, None
        frame = self.frames[self.cursor]
        self.cursor += 1
        return True, frame


def bgr_frame(value: int) -> np.ndarray:
    return np.full((2, 2, 3), value, dtype=np.uint8)


class DecodeFramesAtIndicesTest(unittest.TestCase):
    def test_decodes_exactly_the_requested_indices(self):
        cap = FakeCapture([bgr_frame(i) for i in range(10)])
        decoded = decode_frames_at_indices(cap, [2, 5, 5, 9], total_frames=10)
        self.assertEqual(set(decoded.keys()), {2, 5, 9})
        self.assertEqual(int(decoded[2][0, 0, 0]), 2)
        self.assertEqual(int(decoded[9][0, 0, 0]), 9)

    def test_converts_bgr_to_rgb(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[..., 0] = 10  # B
        frame[..., 2] = 200  # R
        cap = FakeCapture([frame])
        decoded = decode_frames_at_indices(cap, [0], total_frames=1)
        self.assertEqual(int(decoded[0][0, 0, 0]), 200)  # R channel first now
        self.assertEqual(int(decoded[0][0, 0, 2]), 10)  # B channel last now

    def test_empty_indices_returns_empty_dict(self):
        cap = FakeCapture([bgr_frame(0)])
        self.assertEqual(decode_frames_at_indices(cap, [], total_frames=1), {})


class FakeCaptureFailingAt(FakeCapture):
    """Like FakeCapture, but `read()` fails (returns `False, None`) once the cursor
    reaches any position in `fail_at` -- used to simulate a mid-repetition decode
    gap on a request index that is NOT the first one asked for."""

    def __init__(self, frames: list[np.ndarray], fail_at: set):
        super().__init__(frames)
        self.fail_at = set(fail_at)

    def read(self):
        if self.cursor in self.fail_at:
            self.cursor += 1
            return False, None
        return super().read()


class DecodeFramesAtIndicesDecodeFailureTest(unittest.TestCase):
    """Regression coverage: an earlier version of this function only raised when
    the FIRST requested index failed to decode, and silently padded any LATER
    failure with the previously decoded frame -- exactly the "any decode error
    must block" requirement the plan states."""

    def test_raises_on_a_failure_at_the_third_requested_index(self):
        cap = FakeCaptureFailingAt([bgr_frame(i) for i in range(10)], fail_at={7})
        with self.assertRaises(RuntimeError) as ctx:
            decode_frames_at_indices(cap, [1, 4, 7], total_frames=10)
        self.assertIn("7", str(ctx.exception))

    def test_does_not_silently_pad_the_third_index_with_the_second(self):
        # Before the fix, this call would have returned successfully with
        # decoded[7] set to a copy of frame 4 instead of raising.
        cap = FakeCaptureFailingAt([bgr_frame(i) for i in range(10)], fail_at={7})
        with self.assertRaises(RuntimeError):
            decode_frames_at_indices(cap, [1, 4, 7], total_frames=10)

    def test_also_raises_when_the_first_requested_index_fails(self):
        cap = FakeCaptureFailingAt([bgr_frame(i) for i in range(10)], fail_at={1})
        with self.assertRaises(RuntimeError):
            decode_frames_at_indices(cap, [1, 4, 7], total_frames=10)

    def test_error_message_names_the_sample_and_video_from_context(self):
        cap = FakeCaptureFailingAt([bgr_frame(i) for i in range(10)], fail_at={7})
        with self.assertRaises(RuntimeError) as ctx:
            decode_frames_at_indices(cap, [1, 4, 7], total_frames=10, context="sample='s1' video='Ex1/v.mp4'")
        message = str(ctx.exception)
        self.assertIn("s1", message)
        self.assertIn("Ex1/v.mp4", message)


class Sha256OfFilesTest(unittest.TestCase):
    def test_deterministic_over_repeated_calls(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.py"
            path.write_text("x = 1\n", encoding="utf-8")
            self.assertEqual(sha256_of_files([path]), sha256_of_files([path]))

    def test_sensitive_to_file_content(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.py"
            path.write_text("x = 1\n", encoding="utf-8")
            before = sha256_of_files([path])
            path.write_text("x = 2\n", encoding="utf-8")
            after = sha256_of_files([path])
            self.assertNotEqual(before, after)

    def test_order_of_the_file_list_matters(self):
        with TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.py"
            b = Path(tmp) / "b.py"
            a.write_text("a", encoding="utf-8")
            b.write_text("b", encoding="utf-8")
            self.assertNotEqual(sha256_of_files([a, b]), sha256_of_files([b, a]))


class ProvenanceMismatchTest(unittest.TestCase):
    def test_no_mismatches_when_equal(self):
        self.assertEqual(provenance_mismatches({"arm": "vm16"}, {"arm": "vm16"}), {})

    def test_reports_differing_keys_present_in_both(self):
        mismatches = provenance_mismatches({"arm": "vm16", "device": "cpu"}, {"arm": "vj16", "device": "cpu"})
        self.assertEqual(mismatches, {"arm": ("vm16", "vj16")})

    def test_ignores_keys_only_present_on_one_side(self):
        mismatches = provenance_mismatches({"arm": "vm16"}, {"arm": "vm16", "revision": "abc"})
        self.assertEqual(mismatches, {})


class AssertResumeProvenanceMatchesTest(unittest.TestCase):
    def write_bundle(self, path: Path, **provenance: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, clip_features=np.zeros(4), **{f"provenance_{k}": np.asarray(v) for k, v in provenance.items()})

    def test_accepts_an_empty_or_missing_directory(self):
        with TemporaryDirectory() as tmp:
            assert_resume_provenance_matches(Path(tmp) / "missing", {"arm": "vm16"})
            assert_resume_provenance_matches(Path(tmp), {"arm": "vm16"})

    def test_accepts_matching_provenance(self):
        with TemporaryDirectory() as tmp:
            self.write_bundle(Path(tmp) / "train" / "a.npz", arm="vm16", sampler="bounded")
            assert_resume_provenance_matches(Path(tmp), {"arm": "vm16", "sampler": "bounded"})

    def test_refuses_mismatched_provenance(self):
        with TemporaryDirectory() as tmp:
            self.write_bundle(Path(tmp) / "train" / "a.npz", arm="vm16", sampler="bounded")
            with self.assertRaises(RuntimeError):
                assert_resume_provenance_matches(Path(tmp), {"arm": "vm16", "sampler": "legacy"})


if __name__ == "__main__":
    unittest.main()
