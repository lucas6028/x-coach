"""Temporal-shuffle plan: reorder rules, seeding, gates and the paired inference."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.rehab24.videomae_features import (
    STATIC_FRAME_INDEX,
    TEMPORAL_ARMS,
    TEMPORAL_NONE,
    assert_output_dir_matches_variant,
    clip_permutation_seed,
    reorder_clip_frames,
    save_feature,
    temporal_permutation,
)
from src.rehab24.videomae_temporal_control import (
    EXPECTED_MEAN_DISPLACEMENT,
    ORDER_SHARE_BOUNDARY,
    frame_pairing_gate,
    mean_displacement,
    paired_comparison,
    permutation_gate,
    permutation_rule_ok,
    absolute_verdict,
    reading_table_row,
    reproduction_check,
    static_vs_shuffle,
)

N = 16


def frames(n: int = N) -> list[np.ndarray]:
    """Frame i is a constant image of value i, so order is readable from pixels."""
    return [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(n)]


def values(reordered: list[np.ndarray]) -> list[int]:
    return [int(frame[0, 0, 0]) for frame in reordered]


class TemporalPermutationTest(unittest.TestCase):
    def test_identity_and_none_are_the_natural_order(self):
        for arm in (TEMPORAL_NONE, "frame_identity"):
            np.testing.assert_array_equal(temporal_permutation(arm, N, "s", 0), np.arange(N))

    def test_reverse_is_15_minus_i(self):
        np.testing.assert_array_equal(temporal_permutation("frame_reverse", N, "s", 0), np.arange(N)[::-1])

    def test_static_repeats_index_8(self):
        permutation = temporal_permutation("rep_static_frame", N, "s", 0)
        self.assertEqual(STATIC_FRAME_INDEX, 8)
        self.assertTrue(np.all(permutation == 8))

    def test_shuffle_is_a_permutation_and_not_the_identity(self):
        permutation = temporal_permutation("frame_shuffle", N, "Ex1_PM_109_rep10_cam17", 2606)
        np.testing.assert_array_equal(np.sort(permutation), np.arange(N))
        self.assertFalse(np.array_equal(permutation, np.arange(N)))

    def test_tubelet_shuffle_keeps_pairs_adjacent_and_ordered(self):
        permutation = temporal_permutation("tubelet_shuffle", N, "Ex1_PM_109_rep10_cam17", 2606)
        np.testing.assert_array_equal(np.sort(permutation), np.arange(N))
        heads = permutation[::2]
        self.assertTrue(np.all(heads % 2 == 0))
        np.testing.assert_array_equal(permutation[1::2], heads + 1)
        self.assertFalse(np.array_equal(permutation, np.arange(N)))

    def test_shuffle_is_deterministic_in_sample_and_clip_start(self):
        a = temporal_permutation("frame_shuffle", N, "Ex1_PM_109_rep10_cam17", 2606)
        b = temporal_permutation("frame_shuffle", N, "Ex1_PM_109_rep10_cam17", 2606)
        np.testing.assert_array_equal(a, b)

    def test_shuffle_differs_across_clips_and_cameras(self):
        base = temporal_permutation("frame_shuffle", N, "Ex1_PM_109_rep10_cam17", 2606)
        other_clip = temporal_permutation("frame_shuffle", N, "Ex1_PM_109_rep10_cam17", 2647)
        other_camera = temporal_permutation("frame_shuffle", N, "Ex1_PM_109_rep10_cam18", 2606)
        self.assertFalse(np.array_equal(base, other_clip))
        self.assertFalse(np.array_equal(base, other_camera))

    def test_seed_is_64_bit_and_keyed_on_all_three_parts(self):
        seed = clip_permutation_seed("s", 1)
        self.assertLess(seed, 2**64)
        self.assertNotEqual(seed, clip_permutation_seed("s", 2))
        self.assertNotEqual(seed, clip_permutation_seed("t", 1))
        self.assertNotEqual(seed, clip_permutation_seed("s", 1, tag="other"))

    def test_unknown_arm_raises(self):
        with self.assertRaises(ValueError):
            temporal_permutation("bogus", N, "s", 0)

    def test_tubelet_shuffle_needs_an_even_length(self):
        with self.assertRaises(ValueError):
            temporal_permutation("tubelet_shuffle", 15, "s", 0)

    def test_displacement_of_uniform_draws_matches_expectation(self):
        # G2's statistic on 10,000 synthetic clips lands within ±0.05 of (n²−1)/(3n).
        stack = np.stack([temporal_permutation("frame_shuffle", N, f"clip{i}", i) for i in range(10_000)])
        self.assertAlmostEqual(EXPECTED_MEAN_DISPLACEMENT, 5.3125)
        self.assertLess(abs(mean_displacement(stack) - EXPECTED_MEAN_DISPLACEMENT), 0.05)


class ReorderClipFramesTest(unittest.TestCase):
    def test_identity_is_a_no_op_on_the_pixels(self):
        reordered, permutation = reorder_clip_frames(frames(), "frame_identity", "s", 0)
        self.assertEqual(values(reordered), list(range(N)))
        np.testing.assert_array_equal(permutation, np.arange(N))

    def test_reverse_reverses_the_pixels(self):
        reordered, _ = reorder_clip_frames(frames(), "frame_reverse", "s", 0)
        self.assertEqual(values(reordered), list(range(N))[::-1])

    def test_static_yields_16_identical_frames_from_index_8(self):
        reordered, _ = reorder_clip_frames(frames(), "rep_static_frame", "s", 0)
        self.assertEqual(values(reordered), [8] * N)

    def test_shuffle_moves_pixels_by_the_returned_permutation(self):
        reordered, permutation = reorder_clip_frames(frames(), "frame_shuffle", "s", 3)
        self.assertEqual(values(reordered), [int(i) for i in permutation])
        self.assertEqual(sorted(values(reordered)), list(range(N)))

    def test_does_not_mutate_the_input(self):
        original = frames()
        reorder_clip_frames(original, "frame_shuffle", "s", 3)
        self.assertEqual(values(original), list(range(N)))


class PermutationRuleTest(unittest.TestCase):
    def test_each_arm_accepts_its_own_output(self):
        for arm in TEMPORAL_ARMS:
            self.assertTrue(permutation_rule_ok(arm, temporal_permutation(arm, N, "s", 5)), arm)

    def test_rules_reject_the_wrong_arm(self):
        shuffle = temporal_permutation("frame_shuffle", N, "s", 5)
        self.assertFalse(permutation_rule_ok("frame_identity", shuffle))
        self.assertFalse(permutation_rule_ok("frame_reverse", shuffle))
        self.assertFalse(permutation_rule_ok("rep_static_frame", shuffle))
        self.assertFalse(permutation_rule_ok("tubelet_shuffle", shuffle))
        self.assertFalse(permutation_rule_ok("frame_shuffle", np.zeros(N, dtype=int)))


def bundle(temporal: str, sample_id: str, clip_starts: list[int], first: int = 10, last: int = 90, total: int = 500) -> dict:
    payload = {
        "clip_features_legacy_first_token": np.ones((len(clip_starts), 4), dtype=np.float32),
        "clip_features_mean_pool_fc_norm": np.ones((len(clip_starts), 4), dtype=np.float32),
        "clip_starts": np.asarray(clip_starts, dtype=np.int32),
        "first_frame": np.asarray(first, dtype=np.int32),
        "last_frame": np.asarray(last, dtype=np.int32),
        "total_frames": np.asarray(total, dtype=np.int32),
    }
    if temporal != TEMPORAL_NONE:
        payload["frame_permutations"] = np.stack(
            [temporal_permutation(temporal, N, sample_id, start) for start in clip_starts]
        ).astype(np.int8)
    return payload


def row(sample_id: str) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "video_id": "PM_1",
        "exercise_id": "1",
        "person_id": "1",
        "camera": "cam17",
        "correctness": "1",
    }


def write_dir(root: Path, temporal: str, samples: dict[str, list[int]], provenance: dict | None = None, **overrides) -> Path:
    provenance = provenance if provenance is not None else ({} if temporal == TEMPORAL_NONE else {"temporal_transform": temporal})
    for sample_id, starts in samples.items():
        payload = bundle(temporal, sample_id, starts, **overrides.get(sample_id, {}))
        save_feature(root / "train" / f"{sample_id}.npz", row(sample_id), payload, provenance)
    return root


class RawBundleGatesTest(unittest.TestCase):
    samples = {"a_rep1_cam17": [0, 10, 20, 30], "a_rep1_cam18": [0, 10, 20, 30], "b_rep2_cam17": [5, 15, 25, 35]}

    def test_frame_pairing_passes_on_identical_sampling(self):
        with TemporaryDirectory() as tmp:
            base = write_dir(Path(tmp) / "base", TEMPORAL_NONE, self.samples)
            arm = write_dir(Path(tmp) / "arm", "frame_shuffle", self.samples)
            gate = frame_pairing_gate(arm, base)
        self.assertEqual(gate["mismatched"], [])
        self.assertEqual(gate["missing"], [])
        self.assertEqual(gate["arm_bundles"], 3)
        self.assertAlmostEqual(gate["feature_relative_l2"]["max"], 0.0)
        # Only the bundle count keeps this synthetic dir from passing outright.
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["expected_bundles"], 2144)

    def test_frame_pairing_names_a_clip_start_mismatch_and_a_missing_bundle(self):
        with TemporaryDirectory() as tmp:
            base = write_dir(Path(tmp) / "base", TEMPORAL_NONE, self.samples)
            shifted = {**self.samples, "b_rep2_cam17": [6, 15, 25, 35]}
            del shifted["a_rep1_cam18"]
            arm = write_dir(Path(tmp) / "arm", "frame_shuffle", shifted)
            gate = frame_pairing_gate(arm, base)
        self.assertEqual(gate["mismatched"], ["train/b_rep2_cam17.npz:clip_starts"])
        self.assertEqual(gate["missing"], ["train/a_rep1_cam18.npz"])
        self.assertFalse(gate["passed"])

    def test_permutation_gate_validates_and_reseeds_every_arm(self):
        for arm in TEMPORAL_ARMS:
            with TemporaryDirectory() as tmp:
                raw = write_dir(Path(tmp) / arm, arm, self.samples)
                gate = permutation_gate(raw, arm)
            self.assertEqual(gate["invalid"], [], arm)
            self.assertEqual(gate["reseed_mismatch"], [], arm)
            self.assertEqual(gate["clips"], 12, arm)
            self.assertGreater(gate["reseed_checked"], 0, arm)

    def test_permutation_gate_catches_a_permutation_that_does_not_reseed(self):
        with TemporaryDirectory() as tmp:
            raw = write_dir(Path(tmp) / "arm", "frame_shuffle", self.samples)
            target = raw / "train" / "a_rep1_cam17.npz"
            with np.load(target, allow_pickle=False) as data:
                payload = {key: data[key] for key in data.files}
            permutations = payload["frame_permutations"].copy()
            permutations[0] = permutations[0][::-1]  # still a permutation, but not the seeded one
            payload["frame_permutations"] = permutations
            np.savez_compressed(target, **payload)
            gate = permutation_gate(raw, "frame_shuffle")
        self.assertEqual(gate["invalid"], [])
        self.assertEqual(gate["reseed_mismatch"], ["a_rep1_cam17:clip0"])
        self.assertFalse(gate["passed"])

    def test_permutation_gate_catches_a_wrong_rule(self):
        with TemporaryDirectory() as tmp:
            raw = write_dir(Path(tmp) / "arm", "frame_reverse", self.samples)
            gate = permutation_gate(raw, "tubelet_shuffle")
        self.assertEqual(len(gate["invalid"]), 12)
        self.assertFalse(gate["passed"])

    def test_output_dir_guard_separates_temporal_arms_from_the_baseline(self):
        with TemporaryDirectory() as tmp:
            base = write_dir(Path(tmp) / "base", TEMPORAL_NONE, {"a_rep1_cam17": [0]}, provenance={"variant": "full_frame_letterbox"})
            assert_output_dir_matches_variant(base, "full_frame_letterbox")
            with self.assertRaises(SystemExit):
                assert_output_dir_matches_variant(base, "full_frame_letterbox", "frame_identity")
            arm = write_dir(
                Path(tmp) / "arm",
                "frame_shuffle",
                {"a_rep1_cam17": [0]},
                provenance={"variant": "full_frame_letterbox", "temporal_transform": "frame_shuffle"},
            )
            assert_output_dir_matches_variant(arm, "full_frame_letterbox", "frame_shuffle")
            with self.assertRaises(SystemExit):
                assert_output_dir_matches_variant(arm, "full_frame_letterbox")
            with self.assertRaises(SystemExit):
                assert_output_dir_matches_variant(arm, "full_frame_letterbox", "frame_reverse")


class PairedInferenceTest(unittest.TestCase):
    baseline = {str(i): value for i, value in enumerate([0.84, 0.90, 0.88, 0.82, 0.84, 0.95, 0.89, 0.91, 0.87], start=1)}

    def test_hand_computed_nine_subject_case(self):
        candidate = {person: value - delta for (person, value), delta in zip(self.baseline.items(), [0.10, 0.12, 0.08, 0.11, 0.09, 0.13, 0.10, 0.12, 0.11])}
        result = paired_comparison(self.baseline, candidate, n_bootstrap=2000, seed=1)
        self.assertEqual(result["n_subjects"], 9)
        self.assertEqual(result["n_drop"], 9)
        self.assertAlmostEqual(result["mean_delta"], np.mean([0.10, 0.12, 0.08, 0.11, 0.09, 0.13, 0.10, 0.12, 0.11]))
        # all nine positive: exact two-sided Wilcoxon p = 2 / 2^9
        self.assertAlmostEqual(result["two_sided_p"], 2 / 512)
        expected_share = result["mean_delta"] / (np.mean(list(self.baseline.values())) - 0.5)
        self.assertAlmostEqual(result["share_of_above_chance_signal"], expected_share)
        lo, hi = result["bootstrap_95ci_delta"]
        self.assertLess(lo, result["mean_delta"])
        self.assertGreater(hi, result["mean_delta"])

    def test_zero_delta_yields_undetermined_row(self):
        result = paired_comparison(self.baseline, dict(self.baseline), n_bootstrap=200, seed=1)
        self.assertEqual(result["n_drop"], 0)
        self.assertIsNone(result["two_sided_p"])
        reading = reading_table_row(result, {"above_chance": True})
        self.assertTrue(reading["row"].startswith("row 3"))

    def test_reading_table_rows(self):
        big = {"mean_delta": 0.15, "n_drop": 9, "two_sided_p": 0.004}
        small = {"mean_delta": 0.04, "n_drop": 8, "two_sided_p": 0.02}
        unclear = {"mean_delta": 0.04, "n_drop": 5, "two_sided_p": 0.30}
        self.assertTrue(reading_table_row(big, {"above_chance": True})["row"].startswith("row 1"))
        self.assertTrue(reading_table_row(small, {"above_chance": True})["row"].startswith("row 2"))
        self.assertTrue(reading_table_row(unclear, {"above_chance": True})["row"].startswith("row 3"))
        self.assertTrue(reading_table_row(big, {"above_chance": False})["row"].startswith("row 4"))
        self.assertIn("unavailable", reading_table_row(big, None)["row"])
        self.assertEqual(ORDER_SHARE_BOUNDARY, 0.10)

    def test_static_vs_shuffle_row_5(self):
        static = {p: 0.80 for p in self.baseline}
        shuffle = {p: 0.75 for p in self.baseline}
        self.assertTrue(static_vs_shuffle(static, shuffle)["row_5_triggered"])
        self.assertFalse(static_vs_shuffle(shuffle, static)["row_5_triggered"])

    def test_absolute_verdict_carries_the_printers_reading_key(self):
        statistic = {"mean": 0.87, "n_subjects_above_chance": 9}
        verdict = absolute_verdict(statistic, 0.0001)
        self.assertTrue(verdict["above_chance"])
        self.assertIn("reading", verdict)
        self.assertFalse(absolute_verdict({"mean": 0.52, "n_subjects_above_chance": 5}, 0.3)["above_chance"])

    def test_reproduction_check_to_four_decimals(self):
        ok = reproduction_check(0.87412, {"mean": 0.66118})
        self.assertTrue(ok["passed"])
        off = reproduction_check(0.8752, {"mean": 0.6612})
        self.assertFalse(off["auc_reproduced"])
        self.assertTrue(off["ba_reproduced"])
        self.assertFalse(off["passed"])
        self.assertFalse(reproduction_check(0.8741, None)["passed"])


if __name__ == "__main__":
    unittest.main()
