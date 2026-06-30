"""Self-tests for the experiment-1 depth-error harness (synthetic, no dataset needed)."""

from __future__ import annotations

import unittest

import numpy as np

from src.fit3d import dataset as ds
from src.fit3d.depth_eval import (
    CORE_JOINTS, SMPL24_TO_H36M17, depth_decomposition, map_smpl24_to_h36m17,
    procrustes_align, resolve_lr,
)


def _synthetic_gt_cam(frames=8, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(scale=300.0, size=(ds.NUM_JOINTS, 3)) + np.array([0, 0, 4000.0])
    return base[None] + rng.normal(scale=20.0, size=(frames, ds.NUM_JOINTS, 3))


class DepthDecompositionTests(unittest.TestCase):
    def setUp(self):
        self.gt = _synthetic_gt_cam()
        self.joint = 5  # a non-root core joint (left knee)

    def test_perfect_prediction_is_zero_error(self):
        de = depth_decomposition(self.gt.copy(), self.gt.copy())
        self.assertAlmostEqual(de.mpjpe, 0.0, places=6)
        self.assertAlmostEqual(de.inplane, 0.0, places=6)
        self.assertAlmostEqual(de.depth, 0.0, places=6)

    def test_depth_only_noise_routes_to_depth(self):
        pred = self.gt.copy()
        pred[:, self.joint, 2] += 5.0  # perturb camera-axis (depth) of one joint
        de = depth_decomposition(pred, self.gt)
        expected = 5.0 / len(CORE_JOINTS)  # mean |dz| over core joints
        self.assertAlmostEqual(de.depth, expected, places=6)
        self.assertAlmostEqual(de.inplane, 0.0, places=6)

    def test_inplane_only_noise_routes_to_inplane(self):
        pred = self.gt.copy()
        pred[:, self.joint, 0] += 5.0  # perturb image-x of one joint
        de = depth_decomposition(pred, self.gt)
        expected = 5.0 / len(CORE_JOINTS)
        self.assertAlmostEqual(de.inplane, expected, places=6)
        self.assertAlmostEqual(de.depth, 0.0, places=6)

    def test_procrustes_recovers_similarity_transform(self):
        gt_core = self.gt[:, CORE_JOINTS, :]
        theta = 0.4
        Rm = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])
        transformed = 2.0 * (gt_core @ Rm.T) + np.array([100.0, -50.0, 30.0])
        aligned = procrustes_align(transformed, gt_core)
        self.assertLess(np.nanmax(np.abs(aligned - gt_core)), 1e-6)


class Smpl24MappingTests(unittest.TestCase):
    def _distinct_smpl24(self):
        # joint i -> coordinate (i, -i, 2i): distinct and L/R-asymmetric.
        base = np.array([[i, -i, 2 * i] for i in range(24)], dtype=np.float64)
        return base[None].repeat(4, axis=0)  # (4, 24, 3)

    def test_mapping_selects_expected_smpl_indices(self):
        smpl = self._distinct_smpl24()
        h = map_smpl24_to_h36m17(smpl, swap_lr=False)
        self.assertEqual(h.shape, (4, 17, 3))
        for h36m_idx, smpl_idx in enumerate(SMPL24_TO_H36M17):
            np.testing.assert_allclose(h[0, h36m_idx], smpl[0, smpl_idx])

    def test_resolve_lr_picks_matching_orientation(self):
        smpl = self._distinct_smpl24()
        gt_unswapped = map_smpl24_to_h36m17(smpl, swap_lr=False)
        _, swap, mpjpe = resolve_lr(smpl, gt_unswapped)
        self.assertFalse(swap)
        self.assertAlmostEqual(mpjpe, 0.0, places=6)

        gt_swapped = map_smpl24_to_h36m17(smpl, swap_lr=True)
        _, swap2, mpjpe2 = resolve_lr(smpl, gt_swapped)
        self.assertTrue(swap2)
        self.assertAlmostEqual(mpjpe2, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
