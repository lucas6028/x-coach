"""Self-tests for the experiment-3 verdict-fidelity harness (synthetic, no dataset needed)."""

from __future__ import annotations

import unittest

import numpy as np

from src.fit3d import dataset as ds
from src.fit3d.decision_eval import (
    _debias, _per_subject_flip, _swept_flip, _verdict_metrics, needs_3d_verdict,
    nlf_world_points,
)


class VerdictMetricsTests(unittest.TestCase):
    def test_perfect_reading_no_flips(self):
        gt = np.array([85.0, 95.0, 70.0, 100.0])
        m = _verdict_metrics(gt, gt.copy(), thr=90.0, fault_when="high")
        self.assertEqual(m["flip"], 0.0)
        self.assertEqual(m["false_alarm"], 0.0)
        self.assertEqual(m["miss"], 0.0)
        # GT fault when value > 90: indices 1 and 3 -> 2/4 prevalence.
        self.assertAlmostEqual(m["true_fault_rate"], 0.5)

    def test_false_alarm_vs_miss_directionality_high(self):
        gt = np.array([80.0, 80.0])       # both truly OK (<= 90)
        reading = np.array([100.0, 100.0])  # both read as fault
        m = _verdict_metrics(gt, reading, thr=90.0, fault_when="high")
        self.assertEqual(m["false_alarm"], 1.0)  # all OK reps falsely failed
        self.assertTrue(np.isnan(m["miss"]))     # no true faults to miss
        self.assertEqual(m["flip"], 1.0)

    def test_miss_when_reading_passes_a_true_fault(self):
        gt = np.array([100.0, 100.0])      # both truly faults (> 90)
        reading = np.array([80.0, 80.0])   # both read OK
        m = _verdict_metrics(gt, reading, thr=90.0, fault_when="high")
        self.assertEqual(m["miss"], 1.0)
        self.assertTrue(np.isnan(m["false_alarm"]))
        self.assertEqual(m["true_fault_rate"], 1.0)

    def test_fault_when_low_flips_direction(self):
        gt = np.array([0.9, 0.7])          # 0.7 is a fault when fault_when='low' at 0.8
        reading = np.array([0.9, 0.7])
        m = _verdict_metrics(gt, reading, thr=0.8, fault_when="low")
        self.assertEqual(m["flip"], 0.0)
        self.assertAlmostEqual(m["true_fault_rate"], 0.5)

    def test_nan_pairs_are_dropped(self):
        gt = np.array([80.0, np.nan, 100.0])
        reading = np.array([np.nan, 50.0, 100.0])  # only index 2 is a finite pair
        m = _verdict_metrics(gt, reading, thr=90.0, fault_when="high")
        self.assertEqual(m["n"], 1)


class DebiasTests(unittest.TestCase):
    def test_per_camera_offset_removed(self):
        gt = np.array([70.0, 80.0, 70.0, 80.0])
        cam = np.array(["a", "a", "b", "b"])
        reading = np.array([90.0, 100.0, 60.0, 70.0])  # cam a: +20 bias; cam b: -10 bias
        deb, offsets = _debias(reading, gt, cam)
        self.assertAlmostEqual(offsets["a"], 20.0)
        self.assertAlmostEqual(offsets["b"], -10.0)
        # after debiasing, each camera's mean matches GT's mean for that camera
        np.testing.assert_allclose(deb, np.array([70.0, 80.0, 70.0, 80.0]))

    def test_debias_preserves_within_camera_scatter(self):
        gt = np.array([70.0, 70.0])
        cam = np.array(["a", "a"])
        reading = np.array([70.0, 90.0])  # mean offset +10, but they differ by 20
        deb, _ = _debias(reading, gt, cam)
        self.assertAlmostEqual(deb[1] - deb[0], 20.0)  # scatter survives calibration


class SweptAndSubjectTests(unittest.TestCase):
    def test_swept_flip_zero_for_perfect_reading(self):
        rng = np.random.default_rng(0)
        gt = rng.normal(70, 10, size=40)
        self.assertAlmostEqual(_swept_flip(gt, gt.copy(), "high"), 0.0)

    def test_per_subject_flip_groups_by_subject(self):
        gt = np.array([80.0, 80.0, 100.0, 100.0])
        reading = np.array([100.0, 100.0, 100.0, 100.0])  # s1 both false-alarm, s2 both correct
        subj = np.array(["s1", "s1", "s2", "s2"])
        out = _per_subject_flip(gt, reading, subj, thr=90.0, fault_when="high")
        self.assertEqual(out["n_subjects"], 2)
        self.assertAlmostEqual(out["per_subject"]["s1"], 1.0)
        self.assertAlmostEqual(out["per_subject"]["s2"], 0.0)
        self.assertAlmostEqual(out["mean"], 0.5)


class NeedsThreeDVerdictTests(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(needs_3d_verdict(0.21, 0.11), "needs-3D")
        self.assertEqual(needs_3d_verdict(0.11, 0.21), "2D-better")
        self.assertEqual(needs_3d_verdict(0.13, 0.13), "tie")
        self.assertEqual(needs_3d_verdict(float("nan"), 0.1), "n/a")


class NlfWorldPointsTests(unittest.TestCase):
    def test_rotation_and_padding(self):
        # Identity rotation: world == camera; padded joints 17..24 are zero.
        cam_params = {"extrinsics": {"R": np.eye(3)}}
        pred = np.arange(2 * 17 * 3, dtype=np.float64).reshape(2, 17, 3)
        out = nlf_world_points(pred, cam_params)
        self.assertEqual(out.shape, (2, ds.NUM_JOINTS, 3))
        np.testing.assert_allclose(out[:, :17, :], pred)
        self.assertTrue(np.all(out[:, 17:, :] == 0.0))

    def test_rotation_is_applied(self):
        # 90 deg about z: (x,y,z) @ R should rotate the in-plane coords.
        theta = np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
        cam_params = {"extrinsics": {"R": R}}
        pred = np.zeros((1, 17, 3))
        pred[0, 1] = [1.0, 0.0, 0.0]
        out = nlf_world_points(pred, cam_params)
        np.testing.assert_allclose(out[0, 1], pred[0, 1] @ R, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
