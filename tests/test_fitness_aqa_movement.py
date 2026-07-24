"""Movement-general cues + OHP/BarbellRow loaders: depth-isolation and label handling."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.fit3d.biomech import IMAGE2D
from src.fitness_aqa import barbellrow_dataset as bd
from src.fitness_aqa import movement_cues as mc
from src.fitness_aqa import ohp_dataset as ohp
from src.fitness_aqa.cue_features import CAM3D
from tests.test_fitness_aqa_cue_features import rotate_about_vertical


def press_skeleton(elbow_flare_mm: float = 0.0) -> np.ndarray:
    """Synthetic H36M-17 overhead-press pose in camera coords (x right, y down, z away), mm.

    Upper arm goes shoulder -> elbow -> wrist mostly upward; ``elbow_flare_mm`` pushes the
    elbow out along the mediolateral (world-y / left-right) axis, the cue that foreshortens
    into depth in oblique views.
    """
    world = np.zeros((17, 3))
    # world frame: x forward, y left(+)/right(-), z up
    for sign, hip, knee, ankle, sh, el, wr in (
        (1, 1, 2, 3, 11, 12, 13), (-1, 4, 5, 6, 14, 15, 16)
    ):
        y = sign * 150.0
        world[ankle] = [0, y, 0]
        world[knee] = [0, y, 450]
        world[hip] = [0, y, 900]
        world[sh] = [0, y, 1400]
        world[el] = [0, y - sign * elbow_flare_mm, 1650]   # flare pushes elbow laterally
        world[wr] = [0, y, 1900]
    world[0] = 0.5 * (world[1] + world[4])
    world[8] = 0.5 * (world[11] + world[14])
    world[7] = 0.5 * (world[0] + world[8])
    world[9] = world[8] + [0, 0, 100]
    world[10] = world[8] + [0, 0, 200]
    # world (x fwd, y left, z up) -> camera (x right, y down, z away)
    return np.stack([world[:, 1], -world[:, 2], world[:, 0]], axis=-1)


class TestMovementCues(unittest.TestCase):
    def test_dim_and_shared_shape(self):
        pts = press_skeleton()[None]
        f3 = mc.compute_features(pts, CAM3D)
        f2 = mc.compute_features(pts[..., :2], IMAGE2D)
        self.assertEqual(f3.shape, (1, 14))
        self.assertEqual(f3.shape, f2.shape)

    def test_elbow_flare_is_mediolateral_view_dependent_in_2d_not_3d(self):
        base = press_skeleton(elbow_flare_mm=120.0)
        i = mc.FEATURE_NAMES.index("elbow_flare_r")
        r3, r2 = [], []
        for deg in (0.0, 30.0, 60.0, 85.0):
            rot = rotate_about_vertical(base, deg)[None]
            r3.append(mc.compute_features(rot, CAM3D)[0, i])
            r2.append(mc.compute_features(rot[..., :2], IMAGE2D)[0, i])
        # true 3D reads the same flare from every angle; 2D swings with the view
        self.assertLess(float(np.ptp(r3)), 1e-6)
        self.assertGreater(float(np.ptp(r2)), 0.1)

    def test_sagittal_angle_view_invariant_in_3d(self):
        base = press_skeleton()
        i = mc.FEATURE_NAMES.index("elbow_angle_r")
        vals = [mc.compute_features(rotate_about_vertical(base, d)[None], CAM3D)[0, i]
                for d in (0.0, 45.0, 90.0)]
        self.assertLess(float(np.ptp(vals)), 1e-6)

    def test_flare_increases_with_elbow_out(self):
        i = mc.FEATURE_NAMES.index("elbow_flare_r")
        straight = mc.compute_features(press_skeleton(0.0)[None], CAM3D)[0, i]
        flared = mc.compute_features(press_skeleton(200.0)[None], CAM3D)[0, i]
        self.assertGreater(flared, straight)

    def test_nan_joint_localised(self):
        pts = press_skeleton()[None].copy()
        pts[0, 13] = np.nan  # right wrist
        f = mc.compute_features(pts, CAM3D)
        self.assertTrue(np.isnan(f[0, mc.FEATURE_NAMES.index("elbow_angle_r")]))
        self.assertFalse(np.isnan(f[0, mc.FEATURE_NAMES.index("elbow_angle_l")]))

    def test_mode_validation(self):
        pts = press_skeleton()[None]
        with self.assertRaises(ValueError):
            mc.compute_features(pts, IMAGE2D)
        with self.assertRaises(ValueError):
            mc.compute_features(pts[..., :2], CAM3D)


class TestOHPDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "Labels").mkdir(parents=True)
        (self.root / "Splits").mkdir(parents=True)
        json.dump({"a": [[0.1, 0.2]], "b": [], "c": []},
                  open(self.root / "Labels" / "error_knees.json", "w"))
        json.dump({"a": [], "b": [[1, 2]], "c": []},
                  open(self.root / "Labels" / "error_elbows.json", "w"))
        for s, keys in (("train", ["a"]), ("val", ["b"]), ("test", ["c"])):
            json.dump(keys, open(self.root / "Splits" / f"{s}_keys.json", "w"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_binary_and_combined(self):
        self.assertEqual(ohp.load_binary_labels("knees", self.root), {"a": 1, "b": 0, "c": 0})
        comb = ohp.load_combined_labels(self.root)
        self.assertEqual((comb["a"], comb["b"], comb["c"]), (1, 1, 0))

    def test_split_of(self):
        self.assertEqual(ohp.split_of(self.root), {"a": "train", "b": "val", "c": "test"})

    def test_unknown_fault(self):
        with self.assertRaises(ValueError):
            ohp.load_spans("back", self.root)


class TestBarbellRowDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "Labels").mkdir(parents=True)
        (self.root / "Splits" / "Splits_Lumbar_Error").mkdir(parents=True)
        (self.root / "Splits" / "Splits_TorsoAngle_Error").mkdir(parents=True)
        json.dump({"10_1_5": 1, "10_1_6": 0, "11_2_3": 0},
                  open(self.root / "Labels" / "labels_lumbar_error.json", "w"))
        json.dump({"10_1_5": 0, "12_1_1": 1},
                  open(self.root / "Labels" / "labels_torso_angle_error.json", "w"))
        json.dump(["10_1_5", "10_1_6"], open(self.root / "Splits" / "Splits_Lumbar_Error" / "train_ids.json", "w"))
        json.dump(["11_2_3"], open(self.root / "Splits" / "Splits_Lumbar_Error" / "test_ids.json", "w"))
        json.dump([], open(self.root / "Splits" / "Splits_Lumbar_Error" / "val_ids.json", "w"))
        json.dump(["10_1_5"], open(self.root / "Splits" / "Splits_TorsoAngle_Error" / "train_ids.json", "w"))
        json.dump(["12_1_1"], open(self.root / "Splits" / "Splits_TorsoAngle_Error" / "test_ids.json", "w"))
        json.dump([], open(self.root / "Splits" / "Splits_TorsoAngle_Error" / "val_ids.json", "w"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_video_id(self):
        self.assertEqual(bd.video_id("10_1_5"), "10_1")

    def test_manifest_per_fault(self):
        man = bd.load_manifest("lumbar", self.root)
        ids = {r["id"]: r for r in man}
        self.assertEqual(ids["10_1_5"]["label"], 1)
        self.assertEqual(ids["10_1_5"]["split"], "train")
        self.assertEqual(ids["11_2_3"]["split"], "test")
        self.assertEqual(ids["10_1_5"]["video_id"], "10_1")

    def test_all_sample_ids_is_union(self):
        self.assertEqual(set(bd.all_sample_ids(self.root)), {"10_1_5", "10_1_6", "11_2_3", "12_1_1"})

    def test_faults_have_own_splits(self):
        self.assertEqual(bd.load_split("torso_angle", "test", self.root), ["12_1_1"])
        with self.assertRaises(ValueError):
            bd.load_split("lumbar", "nope", self.root)


if __name__ == "__main__":
    unittest.main()
