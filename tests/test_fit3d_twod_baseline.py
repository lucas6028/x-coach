"""Self-tests for the RTMPose COCO-WholeBody -> Fit3D-25 mapping (synthetic, no rtmlib needed)."""

from __future__ import annotations

import unittest

import numpy as np

from src.fit3d import dataset as ds
from src.fit3d.twod_baseline import coco_wholebody_to_fit3d25
from src.fit3d.twod_vs_threed import cue_verdict


def _coco17():
    # distinct pixel per keypoint: index i -> (10*i, 20*i)
    return np.array([[10 * i, 20 * i] for i in range(17)], dtype=np.float64)


class CocoToFit3d25Tests(unittest.TestCase):
    def test_direct_and_midpoint_slots(self):
        kp = _coco17()
        out = coco_wholebody_to_fit3d25(kp, scores=np.ones(17))
        self.assertEqual(out.shape, (ds.NUM_JOINTS, 2))
        # direct joints
        np.testing.assert_allclose(out[ds.R_HIP], kp[12])
        np.testing.assert_allclose(out[ds.R_KNEE], kp[14])
        np.testing.assert_allclose(out[ds.R_ANKLE], kp[16])
        np.testing.assert_allclose(out[ds.L_HIP], kp[11])
        np.testing.assert_allclose(out[ds.L_KNEE], kp[13])
        np.testing.assert_allclose(out[ds.L_ANKLE], kp[15])
        np.testing.assert_allclose(out[ds.R_SHOULDER], kp[6])
        np.testing.assert_allclose(out[ds.L_SHOULDER], kp[5])
        # midpoints
        np.testing.assert_allclose(out[ds.ROOT], 0.5 * (kp[11] + kp[12]))
        np.testing.assert_allclose(out[ds.THORAX], 0.5 * (kp[5] + kp[6]))

    def test_low_confidence_joint_is_nan(self):
        kp = _coco17()
        sc = np.ones(17)
        sc[14] = 0.1  # right knee below threshold
        out = coco_wholebody_to_fit3d25(kp, scores=sc, score_thr=0.3)
        self.assertTrue(np.isnan(out[ds.R_KNEE]).all())
        self.assertFalse(np.isnan(out[ds.L_KNEE]).any())  # left knee still confident

    def test_too_few_keypoints_returns_all_nan(self):
        out = coco_wholebody_to_fit3d25(np.zeros((5, 2)))
        self.assertEqual(out.shape, (ds.NUM_JOINTS, 2))
        self.assertTrue(np.isnan(out).all())


class CueVerdictTests(unittest.TestCase):
    def test_need_3d_when_perfect_2d_fails_and_3d_fixes(self):
        # mocap-2D large, detector tiny, 3D << mocap -> projection geometry, need 3D
        self.assertEqual(cue_verdict(real=18.0, mocap=18.3, best3d=6.0), "need-3D")

    def test_better_2d_when_detector_dominates(self):
        # mocap-2D small, real-2D much larger -> detector error, a better 2D detector helps
        self.assertEqual(cue_verdict(real=0.10, mocap=0.03, best3d=None), "better-2D")

    def test_mixed_when_nothing_dominates(self):
        # small errors, 3D doesn't beat mocap-2D, detector negligible
        self.assertEqual(cue_verdict(real=0.06, mocap=0.07, best3d=0.06), "mixed")


if __name__ == "__main__":
    unittest.main()
