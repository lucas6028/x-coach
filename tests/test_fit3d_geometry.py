"""Projection + joint-layout sanity for the Fit3D loader."""

from __future__ import annotations

import unittest

import numpy as np

from src.fit3d import dataset as ds


def _synthetic_cam(f=(1000.0, 1000.0), c=(500.0, 400.0), cam_pos=(0.0, 0.0, -5.0)):
    """Pinhole camera at world position ``cam_pos``, axes aligned with the world (R = I)."""
    return {
        "extrinsics": {"R": np.eye(3), "T": np.array(cam_pos)},
        "intrinsics_wo_distortion": {"f": np.array(f), "c": np.array(c)},
        "intrinsics_w_distortion": {
            "f": np.array([f]), "c": np.array([c]),
            "k": np.zeros((1, 3)), "p": np.zeros((1, 2)),
        },
    }


class ProjectionTests(unittest.TestCase):
    def test_world_to_camera_shifts_by_camera_position(self):
        cam = _synthetic_cam(cam_pos=(0.0, 0.0, -5.0))
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
        cam_pts = ds.world_to_camera(pts, cam)
        # X_cam = R (X_world - T); R=I, T=(0,0,-5)  ->  +5 on z
        np.testing.assert_allclose(cam_pts, [[0, 0, 5], [1, 2, 5]])

    def test_point_on_axis_projects_to_principal_point(self):
        cam = _synthetic_cam(c=(500.0, 400.0))
        proj = ds.project_world_to_image(np.array([[0.0, 0.0, 0.0]]), cam, with_distortion=False)
        np.testing.assert_allclose(proj[0], [500.0, 400.0])

    def test_offaxis_projection_matches_hand_calc(self):
        cam = _synthetic_cam(f=(1000.0, 1000.0), c=(500.0, 400.0), cam_pos=(0.0, 0.0, -5.0))
        # world (1,0,0) -> cam (1,0,5) -> normalised (0.2,0) -> 1000*0.2 + 500 = 700
        proj = ds.project_world_to_image(np.array([[1.0, 0.0, 0.0]]), cam, with_distortion=False)
        np.testing.assert_allclose(proj[0], [700.0, 400.0])

    def test_projection_preserves_leading_shape(self):
        cam = _synthetic_cam()
        pts = np.zeros((7, ds.NUM_JOINTS, 3))
        proj = ds.project_world_to_image(pts, cam, with_distortion=False)
        self.assertEqual(proj.shape, (7, ds.NUM_JOINTS, 2))


class DatasetAvailableTests(unittest.TestCase):
    """Real-data checks; skipped when the (large) dataset is not extracted."""

    @classmethod
    def setUpClass(cls):
        cls.split, cls.action = "train", "squat"
        try:
            cls.subjs = ds.subjects(cls.split)
        except FileNotFoundError:
            cls.subjs = []
        if not cls.subjs:
            raise unittest.SkipTest("Fit3D not extracted under data/Fit3D")
        cls.subj = cls.subjs[0]
        cls.j3d = ds.load_joints3d(cls.split, cls.subj, cls.action)
        cls.cam = ds.cameras(cls.split, cls.subj)[0]
        cls.cp = ds.read_cam_params(cls.split, cls.subj, cls.cam, cls.action)

    def test_joints_shape(self):
        self.assertEqual(self.j3d.shape[1:], (ds.NUM_JOINTS, 3))

    def test_leg_bones_are_rigid(self):
        # Verifies the hip/knee/ankle indices form real bones: femur length is near-constant.
        femur = np.linalg.norm(self.j3d[:, ds.R_HIP] - self.j3d[:, ds.R_KNEE], axis=1)
        self.assertLess(np.std(femur) / np.mean(femur), 0.05)

    def test_kinematic_height_ordering(self):
        # Head above hips above knees above ankles (world Z up), on average.
        z = self.j3d[:, :, 2].mean(axis=0)
        self.assertGreater(z[ds.HEAD], z[ds.R_HIP])
        self.assertGreater(z[ds.R_HIP], z[ds.R_KNEE])
        self.assertGreater(z[ds.R_KNEE], z[ds.R_ANKLE])

    def test_projection_lands_in_frame_and_orders_vertically(self):
        proj = ds.project_world_to_image(self.j3d[:1], self.cp)[0]
        c = self.cp["intrinsics_w_distortion"]["c"].reshape(2)
        self.assertTrue(np.all(np.abs(proj - c) < 2000))         # near the image centre
        self.assertLess(proj[ds.HEAD, 1], proj[ds.R_ANKLE, 1])   # head above ankle in pixels


if __name__ == "__main__":
    unittest.main()
