"""Video-level aggregation must summarise cues without leaking labels or the depth split."""

import unittest

import numpy as np

from src.fit3d.biomech import IMAGE2D
from src.fitness_aqa import video_features as vf
from src.fitness_aqa.cue_features import CAM3D
from tests.test_fitness_aqa_cue_features import squat_skeleton


def clip(depths, jitter=0.0):
    """A synthetic clip: one H36M-17 frame per knee-angle in ``depths`` (camera coords)."""
    frames = np.stack([squat_skeleton(d) for d in depths])
    if jitter:
        frames = frames + jitter
    return frames


class TestVideoFeatures(unittest.TestCase):
    def test_feature_dim_matches_names(self):
        frames = clip([90.0, 100.0, 120.0])
        det = np.ones(len(frames), bool)
        f = vf.video_feature(frames, det, CAM3D)
        self.assertEqual(f.shape, (len(vf.FEATURE_NAMES),))
        self.assertEqual(len(vf.FEATURE_NAMES), 2 * 14 * len(vf.STATS))

    def test_all_and_bottom_pools_differ(self):
        # Deep frames (small knee angle, low hips) should dominate the bottom pool, so the
        # bottom knee-angle mean is below the all-frames mean.
        frames = clip([60.0, 90.0, 150.0, 160.0, 170.0])
        det = np.ones(len(frames), bool)
        f = vf.video_feature(frames, det, CAM3D)
        all_mean = f[vf.FEATURE_NAMES.index("all__knee_angle_r__mean")]
        bot_mean = f[vf.FEATURE_NAMES.index("bottom__knee_angle_r__mean")]
        self.assertLess(bot_mean, all_mean)

    def test_bottom_pool_selected_by_pose_not_order(self):
        # Put the deepest frame last; the bottom pool must still pick it up (hip-height, not index).
        frames = clip([170.0, 165.0, 160.0, 55.0])
        det = np.ones(len(frames), bool)
        f = vf.video_feature(frames, det, CAM3D)
        bot_min = f[vf.FEATURE_NAMES.index("bottom__knee_angle_r__min")]
        self.assertLess(bot_min, 60.0)

    def test_undetected_frames_ignored(self):
        frames = clip([90.0, 100.0, 110.0])
        det = np.array([True, False, True])
        frames[1] = np.nan  # an undetected frame carries garbage
        f = vf.video_feature(frames, det, CAM3D)
        self.assertTrue(np.isfinite(f).all())

    def test_all_undetected_returns_nan_row(self):
        frames = clip([90.0, 100.0])
        det = np.zeros(len(frames), bool)
        f = vf.video_feature(frames, det, CAM3D)
        self.assertTrue(np.isnan(f).all())

    def test_2d_and_3d_produce_same_shape_different_values(self):
        frames = clip([70.0, 95.0, 130.0])
        det = np.ones(len(frames), bool)
        f3 = vf.video_feature(frames, det, CAM3D)
        f2 = vf.video_feature(frames[..., :2], det, IMAGE2D)
        self.assertEqual(f3.shape, f2.shape)
        # depth channel changes the knee-angle summary between the two arms
        k = vf.FEATURE_NAMES.index("all__knee_angle_r__mean")
        self.assertGreater(abs(f3[k] - f2[k]), 1.0)

    def test_build_matrix_orders_and_flags(self):
        frames = clip([90.0, 100.0])
        per_video = {
            "vA": (frames, np.ones(2, bool)),
            "vB": (frames, np.zeros(2, bool)),   # all undetected -> NaN row, not ok
        }
        feats, ok = vf.build_matrix(per_video, CAM3D, ["vA", "vB", "vMissing"])
        self.assertEqual(feats.shape, (3, len(vf.FEATURE_NAMES)))
        self.assertTrue(ok[0])
        self.assertFalse(ok[1])
        self.assertFalse(ok[2])
        self.assertTrue(np.isnan(feats[2]).all())


if __name__ == "__main__":
    unittest.main()
