import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.rehab24.vjepa2_features import (
    ARM_CLIP_LENGTH,
    ARM_VJ16,
    ARM_VJ64,
    ARMS,
    FRAME_STRIDE,
    NUM_CLIPS,
    build_provenance,
    check_vm16_against_history,
    encode_clips,
    letterbox_and_resize,
    select_pilot_samples,
)
from src.rehab24.clip_sampling import build_bounded_clips
from src.rehab24.videomae_features import extract_bounded_repetition_features


def frame(width: int, height: int, value: int = 100) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


class LetterboxAndResizeTest(unittest.TestCase):
    def test_output_is_a_256x256_rgb_uint8_frame(self):
        result = letterbox_and_resize(frame(1920, 1080), resolution=256)
        self.assertEqual(result.shape, (256, 256, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_preserves_letterbox_padding_no_crop_of_the_subject(self):
        # A 1920x1080 frame letterboxed to 1920x1920 then resized to 256 keeps
        # the top/bottom grey bands; a naive crop-to-square would not.
        source = frame(1920, 1080, value=200)
        result = letterbox_and_resize(source, resolution=256)
        # Top row is the grey fill (114); a centre row is still the content (200).
        # Both must hold -- either alone would also pass a dropped letterbox.
        np.testing.assert_array_equal(result[0], 114)
        np.testing.assert_array_equal(result[128], 200)

    def test_a_square_frame_is_a_near_no_op_besides_resizing(self):
        result = letterbox_and_resize(frame(300, 300, value=77), resolution=256)
        self.assertEqual(result.shape, (256, 256, 3))
        # No letterbox padding was needed, so most pixels should still be ~77.
        self.assertGreater(int((result == 77).sum()), int(result.size * 0.9))


class FakeModel:
    """Returns a token count/hidden size matching V-JEPA 2's real (T/2*16*16, 1024)
    shape so the pooled dimension checked here (1024) matches the raw contract."""

    def __call__(self, pixel_values_videos, skip_predictor=False):
        batch, time_steps = pixel_values_videos.shape[0], pixel_values_videos.shape[1]
        tokens = (time_steps // 2) * 16 * 16
        hidden = torch.arange(tokens * 1024, dtype=torch.float32).reshape(1, tokens, 1024)
        return type("Output", (), {"last_hidden_state": hidden, "predictor_output": None})()


class EncodeClipsTest(unittest.TestCase):
    def test_stacks_one_1024_dim_vector_per_clip(self):
        clips, _, _ = build_bounded_clips(1, 200, total_frames=500, clip_length=16, frame_stride=2, num_clips=4)
        cache = {}
        for clip in clips:
            for index in clip["frame_indices"].tolist():
                cache.setdefault(int(index), np.zeros((256, 256, 3), dtype=np.uint8))
        bundle = encode_clips(FakeModel(), torch.device("cpu"), clips, cache)
        self.assertEqual(bundle["clip_features"].shape, (4, 1024))
        self.assertEqual(bundle["frame_indices"].shape, (4, 16))
        self.assertEqual(bundle["clip_starts"].shape, (4,))
        self.assertTrue(np.all(np.isfinite(bundle["clip_features"])))

    def test_padding_fraction_and_unique_count_pass_through_from_the_clips(self):
        clips, _, _ = build_bounded_clips(1, 5, total_frames=100, clip_length=16, frame_stride=2, num_clips=1)
        cache = {int(i): np.zeros((256, 256, 3), dtype=np.uint8) for i in clips[0]["frame_indices"].tolist()}
        bundle = encode_clips(FakeModel(), torch.device("cpu"), clips, cache)
        self.assertEqual(bundle["unique_frame_count"][0], clips[0]["unique_frame_count"])
        self.assertAlmostEqual(float(bundle["padding_fraction"][0]), clips[0]["padding_fraction"])


class BuildProvenanceTest(unittest.TestCase):
    def test_records_the_arm_specific_clip_length(self):
        provenance = build_provenance(ARM_VJ64, torch.device("cpu"))
        self.assertEqual(provenance["clip_length"], "64")
        self.assertEqual(provenance["arm"], "vj64")
        provenance16 = build_provenance(ARM_VJ16, torch.device("cpu"))
        self.assertEqual(provenance16["clip_length"], "16")

    def test_shared_fields_are_identical_across_arms_except_arm_and_clip_length(self):
        vj64 = build_provenance(ARM_VJ64, torch.device("cpu"))
        vj16 = build_provenance(ARM_VJ16, torch.device("cpu"))
        differing = {k for k in vj64 if vj64[k] != vj16.get(k)}
        self.assertEqual(differing, {"arm", "clip_length"})

    def test_rejects_an_unknown_arm(self):
        with self.assertRaises(ValueError):
            build_provenance("vj32", torch.device("cpu"))


def manifest_row(sample_id: str, exercise_id: str, camera: str, first_frame: int, last_frame: int) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "exercise_id": exercise_id,
        "camera": camera,
        "first_frame": str(first_frame),
        "last_frame": str(last_frame),
    }


class SelectPilotSamplesTest(unittest.TestCase):
    def build_rows(self) -> list[dict[str, str]]:
        rows = []
        counter = 0
        for exercise_id in "123456":
            for camera in ("cam17", "cam18"):
                for length in (10, 20, 30, 200, 300, 400):  # spans short and long quartiles
                    counter += 1
                    rows.append(manifest_row(f"s{counter}", exercise_id, camera, 1, length))
        return rows

    def test_selects_exactly_n(self):
        rows = self.build_rows()
        selected = select_pilot_samples(rows, n=24, seed=20260918)
        self.assertEqual(len(selected), 24)

    def test_is_deterministic(self):
        rows = self.build_rows()
        first = [row["sample_id"] for row in select_pilot_samples(rows, n=24, seed=20260918)]
        second = [row["sample_id"] for row in select_pilot_samples(rows, n=24, seed=20260918)]
        self.assertEqual(first, second)

    def test_covers_both_cameras_and_all_six_exercises(self):
        rows = self.build_rows()
        selected = select_pilot_samples(rows, n=24, seed=20260918)
        self.assertEqual({row["camera"] for row in selected}, {"cam17", "cam18"})
        self.assertEqual({row["exercise_id"] for row in selected}, set("123456"))

    def test_covers_both_short_and_long_quartile_reps(self):
        rows = self.build_rows()
        selected = select_pilot_samples(rows, n=24, seed=20260918)
        lengths = [int(row["last_frame"]) - int(row["first_frame"]) + 1 for row in selected]
        self.assertTrue(min(lengths) <= 20)
        self.assertTrue(max(lengths) >= 300)

    def test_raises_on_empty_row_set(self):
        with self.assertRaises(ValueError):
            select_pilot_samples([], n=24)

    def test_raises_when_n_does_not_divide_the_cells_evenly(self):
        rows = self.build_rows()
        with self.assertRaises(ValueError):
            select_pilot_samples(rows, n=25, seed=1)

    def test_raises_when_a_cell_has_no_eligible_candidate(self):
        # Exercise "1" is entirely short (global quartiles put it below q3), so its
        # (exercise=1, long) cell has no eligible candidate.
        rows = [manifest_row(f"a{i}", "1", "cam17", 1, 10) for i in range(4)]
        rows += [manifest_row(f"b{i}", "2", "cam17", 1, 400) for i in range(4)]
        with self.assertRaises(ValueError):
            select_pilot_samples(rows, n=4, seed=1)


class CheckVm16AgainstHistoryTest(unittest.TestCase):
    def write_new_bundle(self, run_dir: Path, sample_id: str, split: str, features: np.ndarray, padding) -> None:
        out = run_dir / "raw" / "vm16" / split / f"{sample_id}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, clip_features=features, padding_fraction=np.asarray(padding, dtype=np.float32))

    def write_historical_bundle(self, historical_dir: Path, split: str, sample_id: str, features: np.ndarray) -> None:
        out = historical_dir / split / f"{sample_id}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, clip_features_mean_pool_fc_norm=features)

    def test_compares_only_non_padded_samples_and_reports_similarity(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            historical_dir = Path(tmp) / "historical"
            features = np.ones((4, 768), dtype=np.float32)
            self.write_new_bundle(run_dir, "s1", "train", features, padding=[0.0, 0.0, 0.0, 0.0])
            self.write_historical_bundle(historical_dir, "train", "s1", features)

            report = check_vm16_against_history(run_dir, historical_dir, ["s1"])
            self.assertEqual(report["compared"], ["s1"])
            self.assertAlmostEqual(report["max_relative_l2"], 0.0, places=6)
            self.assertAlmostEqual(report["min_cosine"], 1.0, places=6)

    def test_skips_padded_samples(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            historical_dir = Path(tmp) / "historical"
            features = np.ones((4, 768), dtype=np.float32)
            self.write_new_bundle(run_dir, "s2", "train", features, padding=[0.0, 0.1, 0.0, 0.0])
            self.write_historical_bundle(historical_dir, "train", "s2", features)

            report = check_vm16_against_history(run_dir, historical_dir, ["s2"])
            self.assertEqual(report["skipped_padded"], ["s2"])
            self.assertEqual(report["compared"], [])

    def test_skips_missing_historical_bundles(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            historical_dir = Path(tmp) / "historical"
            features = np.ones((4, 768), dtype=np.float32)
            self.write_new_bundle(run_dir, "s3", "train", features, padding=[0.0] * 4)

            report = check_vm16_against_history(run_dir, historical_dir, ["s3"])
            self.assertEqual(report["skipped_missing_history"], ["s3"])

    def test_raises_when_the_new_bundle_itself_is_missing(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                check_vm16_against_history(Path(tmp) / "run", Path(tmp) / "historical", ["ghost"])

    def test_reports_a_nontrivial_relative_l2_for_differing_features(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            historical_dir = Path(tmp) / "historical"
            new_features = np.ones((4, 768), dtype=np.float32)
            old_features = np.full((4, 768), 2.0, dtype=np.float32)
            self.write_new_bundle(run_dir, "s4", "train", new_features, padding=[0.0] * 4)
            self.write_historical_bundle(historical_dir, "train", "s4", old_features)

            report = check_vm16_against_history(run_dir, historical_dir, ["s4"])
            self.assertGreater(report["max_relative_l2"], 0.0)
            self.assertLess(report["min_cosine"], 1.0 + 1e-6)


class Vm16Vj16IndexIdentityTest(unittest.TestCase):
    """Cross-module: the videomae bounded extractor (arm vm16) and the vjepa2
    extractor's vj16 sampling must draw byte-identical frame_indices, since both
    ultimately call build_bounded_clips with the same clip_length/frame_stride."""

    def test_vm16_and_vj16_clip_configurations_match(self):
        self.assertEqual(ARM_CLIP_LENGTH[ARM_VJ16], 16)
        vm16_clips, vm16_first, vm16_last = build_bounded_clips(
            212, 340, total_frames=900, clip_length=16, frame_stride=2, num_clips=4
        )
        vj16_clips, vj16_first, vj16_last = build_bounded_clips(
            212, 340, total_frames=900, clip_length=ARM_CLIP_LENGTH[ARM_VJ16], frame_stride=FRAME_STRIDE, num_clips=NUM_CLIPS
        )
        self.assertEqual(vm16_first, vj16_first)
        self.assertEqual(vm16_last, vj16_last)
        for a, b in zip(vm16_clips, vj16_clips):
            np.testing.assert_array_equal(a["frame_indices"], b["frame_indices"])


class FakeVideoMAEModule:
    """A stand-in torch.nn.Module for encode_clip's `backbone` argument, matching
    VideoMAEModel's contract: forward(pixel_values=...) -> object with
    last_hidden_state of shape (batch, tokens, hidden)."""

    def __call__(self, pixel_values):
        batch, time_steps = pixel_values.shape[0], pixel_values.shape[1]
        tokens = (time_steps // 2) * 14 * 14
        hidden = torch.zeros(1, tokens, 768)
        return type("Output", (), {"last_hidden_state": hidden})()


class BoundedVideomaeMatchesArmsTest(unittest.TestCase):
    def test_extract_bounded_repetition_features_frame_indices_match_vj16(self):
        """Runs the real videomae bounded extractor (mocked backbone/processor) and
        checks its emitted frame_indices equal what vj16 would draw for the same
        repetition -- the acceptance criterion's cross-arm index identity, exercised
        through the actual extractor entry point rather than only the shared helper."""

        class FakeProcessor:
            def __call__(self, frames, return_tensors="pt"):
                stacked = torch.zeros(1, len(frames), 3, 224, 224)
                return {"pixel_values": stacked}

        class FakeCaptureSequential:
            def __init__(self, total_frames):
                self.total_frames = total_frames
                self.cursor = 0

            def set(self, prop, value):
                self.cursor = int(value)

            def grab(self):
                self.cursor += 1
                return True

            def read(self):
                if self.cursor >= self.total_frames:
                    return False, None
                frame = np.full((4, 4, 3), self.cursor % 255, dtype=np.uint8)
                self.cursor += 1
                return True, frame

        cap = FakeCaptureSequential(total_frames=900)
        bundle = extract_bounded_repetition_features(
            backbone=FakeVideoMAEModule(),
            processor=FakeProcessor(),
            cap=cap,
            total_frames=900,
            first_frame=212,
            last_frame=340,
            clip_length=16,
            frame_stride=2,
            num_clips=4,
            device=torch.device("cpu"),
            fc_norm_weight=np.ones(768, dtype=np.float32),
            fc_norm_bias=np.zeros(768, dtype=np.float32),
            fc_norm_eps=1e-6,
        )
        vj16_clips, _, _ = build_bounded_clips(212, 340, total_frames=900, clip_length=16, frame_stride=2, num_clips=4)
        expected = np.stack([clip["frame_indices"] for clip in vj16_clips], axis=0)
        np.testing.assert_array_equal(bundle["frame_indices"], expected)


if __name__ == "__main__":
    unittest.main()
