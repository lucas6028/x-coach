"""Self-tests for the MediaPipe BlazePose-33 -> H36M-17 mapping (no mediapipe/dataset needed)."""

from __future__ import annotations

import unittest

import numpy as np

from src.fit3d.mediapipe_baseline import (
    BP_L_HIP, BP_R_HIP, BP_R_KNEE, BP_R_SHOULDER, blazepose33_to_h36m17,
)


def _distinct_blazepose(frames=3):
    # landmark i -> (i, -i, 2i): distinct so midpoints/copies are checkable.
    base = np.array([[i, -i, 2 * i] for i in range(33)], dtype=np.float64)
    return base[None].repeat(frames, axis=0)  # (frames, 33, 3)


class BlazePoseMappingTests(unittest.TestCase):
    def test_shape_and_direct_joints(self):
        bp = _distinct_blazepose()
        h = blazepose33_to_h36m17(bp)
        self.assertEqual(h.shape, (3, 17, 3))
        np.testing.assert_allclose(h[0, 1], bp[0, BP_R_HIP])       # R hip
        np.testing.assert_allclose(h[0, 2], bp[0, BP_R_KNEE])      # R knee
        np.testing.assert_allclose(h[0, 11], bp[0, BP_R_SHOULDER])  # R shoulder

    def test_pelvis_is_mid_hip(self):
        bp = _distinct_blazepose()
        h = blazepose33_to_h36m17(bp)
        np.testing.assert_allclose(h[0, 0], 0.5 * (bp[0, BP_L_HIP] + bp[0, BP_R_HIP]))

    def test_spine_between_pelvis_and_thorax(self):
        bp = _distinct_blazepose()
        h = blazepose33_to_h36m17(bp)
        np.testing.assert_allclose(h[0, 7], 0.5 * (h[0, 0] + h[0, 8]))

    def test_nan_propagates_to_derived_joint(self):
        bp = _distinct_blazepose()
        bp[:, BP_R_HIP, :] = np.nan
        h = blazepose33_to_h36m17(bp)
        self.assertTrue(np.isnan(h[0, 0]).all())  # pelvis depends on R hip
        self.assertFalse(np.isnan(h[0, 4]).any())  # L hip untouched

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            blazepose33_to_h36m17(np.zeros((3, 17, 3)))


if __name__ == "__main__":
    unittest.main()
