import unittest
import numpy as np
from src.pose.movements.overhead_press import ohp_compute_raw, ohp_assign_phases


def ohp_frame(elbow_angle: float, wrist_y: float, shoulder_y: float = 0.4, frame_index: int = 0) -> dict:
    lm = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(33)]
    # shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24, ears 7/8
    lm[11] = {"x": 0.45, "y": shoulder_y, "z": 0, "visibility": 1.0}
    lm[12] = {"x": 0.55, "y": shoulder_y, "z": 0, "visibility": 1.0}
    lm[13] = {"x": 0.43, "y": shoulder_y + 0.15, "z": 0, "visibility": 1.0}
    lm[14] = {"x": 0.57, "y": shoulder_y + 0.15, "z": 0, "visibility": 1.0}
    lm[15] = {"x": 0.44, "y": wrist_y, "z": 0, "visibility": 1.0}
    lm[16] = {"x": 0.56, "y": wrist_y, "z": 0, "visibility": 1.0}
    lm[23] = {"x": 0.46, "y": 0.75, "z": 0, "visibility": 1.0}
    lm[24] = {"x": 0.54, "y": 0.75, "z": 0, "visibility": 1.0}
    lm[7] = {"x": 0.46, "y": shoulder_y - 0.08, "z": 0, "visibility": 1.0}
    lm[8] = {"x": 0.54, "y": shoulder_y - 0.08, "z": 0, "visibility": 1.0}
    return {"frame_index": frame_index, "landmarks": lm}


class OverheadPressMetricsTests(unittest.TestCase):
    def test_wrist_above_shoulder_sign(self) -> None:
        raw = ohp_compute_raw([ohp_frame(160, wrist_y=0.20)], 30.0)  # wrists above shoulders
        self.assertLess(raw[0]["wrist_above_shoulder"], 0.0)

    def test_phases_include_lockout_at_top(self) -> None:
        frames = ([ohp_frame(90, 0.45, frame_index=i) for i in range(4)]
                  + [ohp_frame(175, 0.15, frame_index=i + 4) for i in range(4)]
                  + [ohp_frame(90, 0.45, frame_index=i + 8) for i in range(4)])
        phases = ohp_assign_phases(ohp_compute_raw(frames, 30.0))
        self.assertIn("lockout", phases)
