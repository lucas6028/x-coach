import unittest

import numpy as np

from src.pose.movements.base import CoreFrame
from src.pose.movements.deadlift import (
    DEADLIFT_METRIC_KEYS,
    deadlift_assign_phases,
    deadlift_compute_raw,
)


def _landmarks(overrides: dict[int, tuple[float, float]]) -> list[dict]:
    """33 fully-visible landmarks in a plausible standing pose, with overrides applied."""
    base = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.99} for _ in range(33)]
    defaults = {
        11: (0.50, 0.30), 12: (0.52, 0.30),   # shoulders
        23: (0.50, 0.55), 24: (0.52, 0.55),   # hips
        25: (0.50, 0.75), 26: (0.52, 0.75),   # knees
        27: (0.50, 0.95), 28: (0.52, 0.95),   # ankles
        29: (0.49, 0.96), 30: (0.51, 0.96),   # heels
        31: (0.55, 0.97), 32: (0.57, 0.97),   # foot index
    }
    defaults.update(overrides)
    for index, (x, y) in defaults.items():
        base[index] = {"x": x, "y": y, "z": 0.0, "visibility": 0.99}
    return base


def _frame(index: int, overrides: dict[int, tuple[float, float]] | None = None) -> dict:
    return {"frame_index": index, "landmarks": _landmarks(overrides or {})}


class DeadliftMetricTests(unittest.TestCase):
    def test_metric_keys_match_the_emitted_metrics(self):
        """A key the tuple omits is dropped by run_detector and read back as NaN."""
        raw = deadlift_compute_raw([_frame(0)], fps=30.0)
        emitted = set(raw[0]) - {"frame_index", "time", "valid", "lower_body_visibility"}
        self.assertEqual(emitted, set(DEADLIFT_METRIC_KEYS))

    def test_an_upright_lockout_reads_near_180_degrees(self):
        raw = deadlift_compute_raw([_frame(0)], fps=30.0)
        self.assertTrue(raw[0]["valid"])
        self.assertGreater(raw[0]["hip_angle_deg"], 170.0)
        self.assertGreater(raw[0]["knee_angle_deg"], 170.0)
        self.assertLess(raw[0]["torso_pitch_deg"], 10.0)

    def test_a_pitched_trunk_reads_a_large_torso_pitch(self):
        # Shoulders driven forward of the hips: trunk near horizontal.
        raw = deadlift_compute_raw(
            [_frame(0, {11: (0.75, 0.50), 12: (0.77, 0.50)})], fps=30.0
        )
        self.assertGreater(raw[0]["torso_pitch_deg"], 60.0)

    def test_one_missing_landmark_invalidates_the_whole_frame(self):
        landmarks = _landmarks({})
        landmarks[24] = {"x": 0.52, "y": 0.55, "z": 0.0, "visibility": 0.01}
        raw = deadlift_compute_raw([{"frame_index": 0, "landmarks": landmarks}], fps=30.0)
        self.assertFalse(raw[0]["valid"])
        self.assertNotIn("hip_angle_deg", raw[0])

    def test_a_non_dict_frame_is_invalid_rather_than_raising(self):
        self.assertEqual(deadlift_compute_raw([None], fps=30.0), [{"valid": False}])


class DeadliftPhaseTests(unittest.TestCase):
    @staticmethod
    def _rep(hip_angles: list[float]) -> list[dict]:
        return [
            {"frame_index": i, "valid": True, "hip_angle_deg": a}
            for i, a in enumerate(hip_angles)
        ]

    @staticmethod
    def _pull(n: int, low: float, high: float) -> list[float]:
        """A flexed-start rep: floor -> lockout -> floor."""
        up = list(np.linspace(low, high, n // 2))
        return up + list(np.linspace(high, low, n - n // 2))

    def test_an_empty_clip_returns_no_phases(self):
        self.assertEqual(deadlift_assign_phases([]), [])

    def test_a_clip_with_no_finite_signal_is_entirely_unknown(self):
        raw = self._rep([np.nan] * 20)
        self.assertEqual(set(deadlift_assign_phases(raw)), {"unknown"})

    def test_an_invalid_frame_is_unknown_even_inside_the_setup_prefix(self):
        raw = self._rep(self._pull(40, 60.0, 178.0))
        raw[0]["valid"] = False
        self.assertEqual(deadlift_assign_phases(raw)[0], "unknown")

    def test_a_full_rep_produces_all_five_phases(self):
        raw = self._rep(self._pull(60, 60.0, 178.0))
        self.assertEqual(
            set(deadlift_assign_phases(raw)),
            {"setup", "lift_off", "mid_pull", "lockout", "lowering"},
        )

    def test_the_rep_opens_in_setup_because_a_flexed_start_begins_on_the_floor(self):
        raw = self._rep(self._pull(60, 60.0, 178.0))
        self.assertEqual(deadlift_assign_phases(raw)[0], "setup")

    def test_a_rep_that_never_locks_out_still_has_a_lockout_phase(self):
        """The fault IS failing to reach extension, so the phase must not vanish with it.

        The lockout threshold is a PERCENTILE of this rep's own hip-angle excursion, not an
        absolute angle, so a rep peaking at 150 degrees still yields a lockout phase for
        `rule_incomplete_lockout` to score. An absolute cutoff would silence the rule on
        exactly the reps it exists to catch.
        """
        raw = self._rep(self._pull(60, 60.0, 150.0))
        phases = deadlift_assign_phases(raw)
        self.assertGreaterEqual(phases.count("lockout"), 6)

    def test_phase_count_always_equals_frame_count(self):
        """run_detector raises if assign_phases returns a different length."""
        for n in (1, 7, 40, 61):
            raw = self._rep(self._pull(n, 60.0, 178.0))
            self.assertEqual(len(deadlift_assign_phases(raw)), n)
