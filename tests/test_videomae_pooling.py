import unittest

import numpy as np

try:  # CI installs no torch; this module is numpy-only so the rest still runs there.
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.video.videomae_pooling import (
    CLIP_AGGREGATIONS,
    LEGACY_FIRST_TOKEN,
    MEAN_POOL_FC_NORM,
    TOKEN_POOLING_MODES,
    aggregate_clips,
    build_provenance,
    feature_dir_name,
    layer_norm,
)


class LayerNormTest(unittest.TestCase):
    # Only the equivalence check needs torch, so it skips where torch is absent
    # rather than taking the whole numpy-only module down with it.
    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_matches_torch_layer_norm(self):
        """The numpy fc_norm must be numerically identical to torch's, or the
        corrected features would silently differ from the classification path."""
        rng = np.random.default_rng(0)
        features = rng.normal(size=(4, 768)).astype(np.float32)
        weight = rng.normal(size=768).astype(np.float32)
        bias = rng.normal(size=768).astype(np.float32)
        eps = 1e-6

        reference = torch.nn.functional.layer_norm(
            torch.from_numpy(features),
            normalized_shape=(768,),
            weight=torch.from_numpy(weight),
            bias=torch.from_numpy(bias),
            eps=eps,
        ).numpy()

        np.testing.assert_allclose(layer_norm(features, weight, bias, eps), reference, rtol=1e-5, atol=1e-6)

    def test_normalizes_with_biased_variance(self):
        features = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        weight = np.ones(4, dtype=np.float32)
        bias = np.zeros(4, dtype=np.float32)

        result = layer_norm(features, weight, bias, eps=0.0)

        self.assertAlmostEqual(float(result.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(result.std(ddof=0)), 1.0, places=5)

    def test_returns_float32(self):
        features = np.ones((2, 4), dtype=np.float64)
        result = layer_norm(features, np.ones(4), np.zeros(4), eps=1e-6)
        self.assertEqual(result.dtype, np.float32)


class AggregateClipsTest(unittest.TestCase):
    def setUp(self):
        self.stack = np.array([[1.0, -4.0], [3.0, 2.0]], dtype=np.float32)

    def test_max_takes_elementwise_maximum(self):
        np.testing.assert_allclose(aggregate_clips(self.stack, "max"), [3.0, 2.0])

    def test_mean_averages_across_clips(self):
        np.testing.assert_allclose(aggregate_clips(self.stack, "mean"), [2.0, -1.0])

    def test_max_and_mean_differ_on_normalized_clips(self):
        """The reason aggregation is an explicit axis: after fc_norm each clip is
        zero-mean, and max over clips reintroduces a positive bias that mean does not."""
        rng = np.random.default_rng(1)
        normalized = rng.normal(size=(4, 768)).astype(np.float32)
        normalized -= normalized.mean(axis=1, keepdims=True)

        self.assertGreater(float(aggregate_clips(normalized, "max").mean()), 0.5)
        self.assertAlmostEqual(float(aggregate_clips(normalized, "mean").mean()), 0.0, places=5)

    def test_rejects_unknown_aggregation(self):
        with self.assertRaises(ValueError):
            aggregate_clips(self.stack, "median")

    def test_rejects_non_2d_stack(self):
        with self.assertRaises(ValueError):
            aggregate_clips(np.zeros(768, dtype=np.float32), "mean")

    def test_rejects_empty_stack(self):
        with self.assertRaises(ValueError):
            aggregate_clips(np.zeros((0, 768), dtype=np.float32), "mean")


class FeatureDirNameTest(unittest.TestCase):
    def test_names_are_unique_per_combination(self):
        names = {
            feature_dir_name(pooling, aggregation)
            for pooling in TOKEN_POOLING_MODES
            for aggregation in CLIP_AGGREGATIONS
        }
        self.assertEqual(len(names), len(TOKEN_POOLING_MODES) * len(CLIP_AGGREGATIONS))

    def test_encodes_both_axes(self):
        self.assertEqual(
            feature_dir_name(MEAN_POOL_FC_NORM, "mean"),
            "videomae_mean_pool_fc_norm_mean",
        )
        self.assertEqual(
            feature_dir_name(LEGACY_FIRST_TOKEN, "max"),
            "videomae_legacy_first_token_max",
        )

    def test_rejects_unknown_axes(self):
        with self.assertRaises(ValueError):
            feature_dir_name("cls_token", "max")
        with self.assertRaises(ValueError):
            feature_dir_name(MEAN_POOL_FC_NORM, "median")


class BuildProvenanceTest(unittest.TestCase):
    def test_records_shared_extraction_settings(self):
        record = build_provenance(
            model_name="MCG-NJU/videomae-base-finetuned-kinetics",
            clip_length=16,
            frame_stride=2,
            num_clips=4,
            transformers_version="5.5.0",
        )
        self.assertEqual(record["model_name"], "MCG-NJU/videomae-base-finetuned-kinetics")
        self.assertEqual(record["clip_length"], "16")
        self.assertEqual(record["transformers_version"], "5.5.0")

    def test_pooling_axes_are_optional(self):
        shared = build_provenance("m", 16, 2, 4, "5.5.0")
        self.assertNotIn("token_pooling", shared)
        self.assertNotIn("clip_aggregation", shared)

        materialized = build_provenance("m", 16, 2, 4, "5.5.0", token_pooling=MEAN_POOL_FC_NORM, clip_aggregation="mean")
        self.assertEqual(materialized["token_pooling"], MEAN_POOL_FC_NORM)
        self.assertEqual(materialized["clip_aggregation"], "mean")

    def test_values_are_strings_for_npz_round_trip(self):
        record = build_provenance("m", 16, 2, 4, "5.5.0", token_pooling=LEGACY_FIRST_TOKEN, clip_aggregation="max")
        self.assertTrue(all(isinstance(value, str) for value in record.values()))


if __name__ == "__main__":
    unittest.main()
