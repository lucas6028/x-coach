import unittest
from unittest import mock

import numpy as np
import torch

from src.video.vjepa2_backbone import (
    EXPECTED_PARAMETER_COUNT,
    IMAGE_MEAN,
    IMAGE_STD,
    assert_expected_parameter_count,
    encode_clip,
    pool_tokens,
    preprocess_clip,
)


def solid_frame(value: int, size: int = 256) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


class PreprocessClipTest(unittest.TestCase):
    """Letterbox+resize happen in the caller; this is pure normalization + layout."""

    def test_output_shape_is_time_channels_height_width(self):
        frames = [solid_frame(0), solid_frame(128), solid_frame(255)]
        array = preprocess_clip(frames)
        self.assertEqual(array.shape, (3, 3, 256, 256))
        self.assertEqual(array.dtype, np.float32)

    def test_normalization_matches_imagenet_style_scaling_on_a_synthetic_frame(self):
        # A constant-value frame: every pixel should normalize to the same number,
        # matching (value/255 - mean) / std per channel.
        value = 128
        array = preprocess_clip([solid_frame(value, size=4)])
        expected = (np.float32(value) / 255.0 - IMAGE_MEAN) / IMAGE_STD
        for channel in range(3):
            np.testing.assert_allclose(array[0, channel], expected[channel], rtol=1e-6)

    def test_black_frame_normalizes_to_minus_mean_over_std(self):
        array = preprocess_clip([solid_frame(0, size=4)])
        expected = (0.0 - IMAGE_MEAN) / IMAGE_STD
        for channel in range(3):
            np.testing.assert_allclose(array[0, channel], expected[channel], rtol=1e-6)

    def test_channel_axis_is_moved_to_immediately_after_time(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[..., 0] = 50  # R channel distinguishable
        array = preprocess_clip([frame])
        # array[0, 0] must be the R channel's normalized plane, not G or B.
        expected_r = (50.0 / 255.0 - IMAGE_MEAN[0]) / IMAGE_STD[0]
        np.testing.assert_allclose(array[0, 0], expected_r, rtol=1e-6)


class PoolTokensTest(unittest.TestCase):
    """Pooling must average over the TOKEN axis (dim 1), never the batch axis."""

    def test_pools_over_tokens_not_batch(self):
        # (batch=1, tokens=4, hidden=6); rows are distinguishable per token.
        hidden = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
        pooled = pool_tokens(hidden)
        expected = hidden.mean(dim=1).squeeze(0).numpy()
        np.testing.assert_allclose(pooled, expected)
        self.assertEqual(pooled.shape, (6,))

    def test_returns_numpy_float32(self):
        hidden = torch.zeros(1, 8, 1024)
        pooled = pool_tokens(hidden)
        self.assertIsInstance(pooled, np.ndarray)
        self.assertEqual(pooled.dtype, np.float32)

    def test_a_single_distinctive_token_shifts_the_mean_proportionally(self):
        hidden = torch.zeros(1, 4, 3)
        hidden[0, 0] = torch.tensor([4.0, 4.0, 4.0])
        pooled = pool_tokens(hidden)
        np.testing.assert_allclose(pooled, [1.0, 1.0, 1.0])

    def test_rejects_a_non_3d_tensor(self):
        with self.assertRaises(ValueError):
            pool_tokens(torch.zeros(4, 6))


class FakeVJEPA2Output:
    def __init__(self, last_hidden_state, predictor_output=None):
        self.last_hidden_state = last_hidden_state
        self.predictor_output = predictor_output


class FakeVJEPA2Model:
    """Stands in for VJEPA2Model: returns a fixed last_hidden_state shaped like the
    real encoder's (batch, tokens, hidden) output, so CPU tests never load the
    actual 326M-parameter checkpoint."""

    def __init__(self, tokens: int = 8, hidden: int = 1024, predictor_output=None):
        self.tokens = tokens
        self.hidden = hidden
        self.predictor_output = predictor_output
        self.calls = []

    def __call__(self, pixel_values_videos, skip_predictor=False):
        self.calls.append({"shape": tuple(pixel_values_videos.shape), "skip_predictor": skip_predictor})
        hidden_state = torch.arange(self.tokens * self.hidden, dtype=torch.float32).reshape(1, self.tokens, self.hidden)
        return FakeVJEPA2Output(hidden_state, predictor_output=self.predictor_output)


class EncodeClipTest(unittest.TestCase):
    def test_returns_a_1024_dim_vector_shaped_like_pool_tokens_output(self):
        model = FakeVJEPA2Model(tokens=8, hidden=1024)
        frames = [solid_frame(10)] * 16
        pooled = encode_clip(model, frames, device=torch.device("cpu"))
        self.assertEqual(pooled.shape, (1024,))
        self.assertTrue(np.all(np.isfinite(pooled)))

    def test_calls_the_model_with_skip_predictor_true(self):
        model = FakeVJEPA2Model(tokens=8, hidden=16)
        encode_clip(model, [solid_frame(0)] * 4, device=torch.device("cpu"))
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(model.calls[0]["skip_predictor"])

    def test_pixel_values_have_the_expected_batch_time_channel_height_width_shape(self):
        model = FakeVJEPA2Model(tokens=8, hidden=16)
        frames = [solid_frame(0)] * 5
        encode_clip(model, frames, device=torch.device("cpu"))
        self.assertEqual(model.calls[0]["shape"], (1, 5, 3, 256, 256))

    def test_refuses_a_predictor_output_even_with_skip_predictor(self):
        """Future-proofing: if a transformers version ever returns a non-null
        predictor_output despite skip_predictor=True, refuse rather than silently
        pool a representation that might include predictor activity."""
        model = FakeVJEPA2Model(tokens=8, hidden=16, predictor_output=torch.zeros(1, 8, 16))
        with self.assertRaises(RuntimeError):
            encode_clip(model, [solid_frame(0)] * 4, device=torch.device("cpu"))


class AssertExpectedParameterCountTest(unittest.TestCase):
    def test_accepts_the_expected_count(self):
        model = torch.nn.Linear(10, 10, bias=False)  # 100 params
        assert_expected_parameter_count(model, expected=100)

    def test_rejects_a_drifted_count(self):
        model = torch.nn.Linear(10, 10, bias=False)  # 100 params
        with self.assertRaises(SystemExit):
            assert_expected_parameter_count(model, expected=101)

    def test_default_expected_is_the_pinned_vitl_fpc64_256_count(self):
        self.assertEqual(EXPECTED_PARAMETER_COUNT, 325_971_328)


if __name__ == "__main__":
    unittest.main()
