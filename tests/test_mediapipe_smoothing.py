from __future__ import annotations

import unittest

import numpy as np

from src.rehab24.mediapipe_skeleton_features import smooth_series


class SmoothSeriesTests(unittest.TestCase):
    def _jitter(self, series: np.ndarray) -> float:
        """Mean absolute frame-to-frame change — a proxy for estimation jitter."""
        return float(np.nanmean(np.abs(np.diff(series, axis=0))))

    def test_preserves_shape(self):
        series = np.random.RandomState(0).randn(40, 33, 3).astype(np.float32)
        out = smooth_series(series, window_length=7, polyorder=2)
        self.assertEqual(out.shape, series.shape)
        self.assertEqual(out.dtype, np.float32)

    def test_reduces_jitter(self):
        rng = np.random.RandomState(1)
        t = np.linspace(0, 4 * np.pi, 60)
        clean = np.sin(t)[:, None, None] * np.ones((1, 33, 3), dtype=np.float32)
        noisy = (clean + rng.normal(scale=0.3, size=clean.shape)).astype(np.float32)
        smoothed = smooth_series(noisy, window_length=9, polyorder=2)
        # Smoothing should bring the jittery signal closer to the clean one and
        # cut the frame-to-frame jitter that the velocity channels amplify.
        self.assertLess(self._jitter(smoothed), self._jitter(noisy))
        self.assertLess(
            float(np.mean((smoothed - clean) ** 2)),
            float(np.mean((noisy - clean) ** 2)),
        )

    def test_even_window_is_coerced_odd(self):
        series = np.random.RandomState(2).randn(40, 33, 3).astype(np.float32)
        # An even window must not raise (Savitzky-Golay requires odd length).
        out = smooth_series(series, window_length=8, polyorder=2)
        self.assertEqual(out.shape, series.shape)

    def test_window_leq_one_is_noop(self):
        series = np.random.RandomState(3).randn(40, 33, 3).astype(np.float32)
        np.testing.assert_array_equal(smooth_series(series, window_length=1), series)
        np.testing.assert_array_equal(smooth_series(series, window_length=0), series)

    def test_short_series_degrades_gracefully(self):
        series = np.random.RandomState(4).randn(2, 33, 3).astype(np.float32)
        # Fewer frames than a usable window: return unchanged rather than error.
        np.testing.assert_array_equal(smooth_series(series, window_length=7), series)

    def test_window_clamped_to_series_length(self):
        series = np.random.RandomState(5).randn(11, 33, 3).astype(np.float32)
        out = smooth_series(series, window_length=51, polyorder=2)
        self.assertEqual(out.shape, series.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_all_nan_channel_passthrough(self):
        series = np.random.RandomState(6).randn(40, 33, 3).astype(np.float32)
        series[:, 5, :] = np.nan  # a joint never detected, as interpolate_missing leaves it
        out = smooth_series(series, window_length=7, polyorder=2)
        self.assertTrue(np.all(np.isnan(out[:, 5, :])))
        self.assertTrue(np.all(np.isfinite(out[:, 0, :])))


if __name__ == "__main__":
    unittest.main()
