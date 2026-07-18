import math
import unittest
import numpy as np
from src.pose.movements.overhead_press import ohp_compute_raw, ohp_assign_phases


def _elbow_xy(
    shoulder_xy: tuple[float, float],
    wrist_xy: tuple[float, float],
    elbow_angle: float,
    side: str,
) -> tuple[float, float]:
    """Compute an elbow landmark position such that the interior angle(shoulder, elbow,
    wrist) equals `elbow_angle` by construction. Places the elbow on the perpendicular
    bisector of the shoulder-wrist segment, offset laterally outward (left -> -x, right
    -> +x), using d = h / tan(angle/2) where h is half the shoulder-wrist distance."""
    sx, sy = shoulder_xy
    wx, wy = wrist_xy
    mx, my = (sx + wx) / 2.0, (sy + wy) / 2.0
    half_len = math.hypot(sx - wx, sy - wy) / 2.0
    if elbow_angle >= 179:
        d = 0.0
    else:
        d = half_len / math.tan(math.radians(elbow_angle) / 2.0)
    dx, dy = wx - sx, wy - sy
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        perp = (1.0, 0.0)
    else:
        ux, uy = dx / norm, dy / norm
        perp = (-uy, ux)
    desired_sign = -1.0 if side == "left" else 1.0
    if desired_sign * perp[0] < 0:
        perp = (-perp[0], -perp[1])
    return mx + d * perp[0], my + d * perp[1]


def ohp_frame(
    elbow_angle: float,
    wrist_y: float,
    shoulder_y: float = 0.4,
    frame_index: int = 0,
    shoulder_dx: float = 0.0,
) -> dict:
    lm = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(33)]
    # shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24, ears 7/8
    # shoulder_dx shifts both shoulders' x while hips stay put, simulating a
    # shoulders-behind-hips back-lean (positive torso_lean_signed_deg).
    left_shoulder = (0.45 + shoulder_dx, shoulder_y)
    right_shoulder = (0.55 + shoulder_dx, shoulder_y)
    left_wrist = (0.44, wrist_y)
    right_wrist = (0.56, wrist_y)
    left_elbow = _elbow_xy(left_shoulder, left_wrist, elbow_angle, "left")
    right_elbow = _elbow_xy(right_shoulder, right_wrist, elbow_angle, "right")
    lm[11] = {"x": left_shoulder[0], "y": left_shoulder[1], "z": 0, "visibility": 1.0}
    lm[12] = {"x": right_shoulder[0], "y": right_shoulder[1], "z": 0, "visibility": 1.0}
    lm[13] = {"x": left_elbow[0], "y": left_elbow[1], "z": 0, "visibility": 1.0}
    lm[14] = {"x": right_elbow[0], "y": right_elbow[1], "z": 0, "visibility": 1.0}
    lm[15] = {"x": left_wrist[0], "y": left_wrist[1], "z": 0, "visibility": 1.0}
    lm[16] = {"x": right_wrist[0], "y": right_wrist[1], "z": 0, "visibility": 1.0}
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

    def test_ohp_frame_fixture_controls_elbow_angle(self) -> None:
        # The fixture's elbow landmark must be computed from `elbow_angle`, not hardcoded,
        # so avg_elbow_angle actually tracks the caller's requested angle.
        raw_straight = ohp_compute_raw([ohp_frame(elbow_angle=178, wrist_y=0.15)], 30.0)
        self.assertAlmostEqual(raw_straight[0]["avg_elbow_angle"], 178, delta=3)

        raw_bent = ohp_compute_raw([ohp_frame(elbow_angle=140, wrist_y=0.15)], 30.0)
        self.assertAlmostEqual(raw_bent[0]["avg_elbow_angle"], 140, delta=3)

    def test_wrist_height_asymmetry_metric(self) -> None:
        frame = ohp_frame(160, wrist_y=0.15)
        # Push the right wrist down relative to the left to create asymmetry.
        frame["landmarks"][16]["y"] = 0.30
        raw = ohp_compute_raw([frame], 30.0)
        self.assertGreater(raw[0]["wrist_height_asymmetry"], 0.15)


class OverheadPressRulesTests(unittest.TestCase):
    def _run(self, frames, view="side", vc=0.8):
        from src.pose.movements import registry
        from src.pose.movements.base import run_detector
        return run_detector(registry.get_detector("Overhead Press"), frames, 30.0, view, vc)[1]

    def test_incomplete_lockout_flagged(self) -> None:
        frames = ([ohp_frame(90, 0.45, frame_index=i) for i in range(4)]
                  + [ohp_frame(140, 0.15, frame_index=i + 4) for i in range(6)]   # elbows never extend, wrists up
                  + [ohp_frame(90, 0.45, frame_index=i + 10) for i in range(4)])
        ids = {d.fault_id for d in self._run(frames)}
        self.assertIn("ohp_incomplete_lockout", ids)

    def test_full_lockout_not_flagged(self) -> None:
        frames = ([ohp_frame(90, 0.45, frame_index=i) for i in range(4)]
                  + [ohp_frame(178, 0.15, frame_index=i + 4) for i in range(6)]   # full extension, wrists up
                  + [ohp_frame(90, 0.45, frame_index=i + 10) for i in range(4)])
        ids = {d.fault_id for d in self._run(frames)}
        self.assertNotIn("ohp_incomplete_lockout", ids)

    def test_lockout_rule_carries_citation(self) -> None:
        frames = [ohp_frame(140, 0.15, frame_index=i) for i in range(12)]
        det = next((d for d in self._run(frames) if d.fault_id == "ohp_incomplete_lockout"), None)
        assert det is not None and det.citation and det.citation_support

    def test_asymmetric_press_flagged_when_wrists_uneven(self) -> None:
        frames = []
        for i in range(12):
            frame = ohp_frame(160, wrist_y=0.15, frame_index=i)
            # Shoulder width here is 0.10 (0.55 - 0.45); push the right wrist down by more
            # than 0.15 * shoulder_width (~0.015) to trip the asymmetry threshold.
            frame["landmarks"][16]["y"] = 0.35
            frames.append(frame)
        ids = {d.fault_id for d in self._run(frames, view="front")}
        self.assertIn("ohp_asymmetric_press", ids)

    def test_asymmetric_press_not_flagged_when_wrists_even(self) -> None:
        frames = [ohp_frame(160, wrist_y=0.15, frame_index=i) for i in range(12)]
        ids = {d.fault_id for d in self._run(frames, view="front")}
        self.assertNotIn("ohp_asymmetric_press", ids)

    def test_back_lean_flagged(self) -> None:
        # Same setup -> press -> lockout -> lower shape as the lockout tests, but with
        # shoulders shifted +0.15 in x on every frame (hips fixed) so torso_lean_signed_deg
        # is well past the 15 deg threshold for the press/lockout portion of the rep.
        frames = ([ohp_frame(90, 0.45, frame_index=i, shoulder_dx=0.15) for i in range(4)]
                  + [ohp_frame(178, 0.15, frame_index=i + 4, shoulder_dx=0.15) for i in range(6)]
                  + [ohp_frame(90, 0.45, frame_index=i + 10, shoulder_dx=0.15) for i in range(4)])
        ids = {d.fault_id for d in self._run(frames)}
        self.assertIn("ohp_lumbar_hyperextension", ids)

    def test_upright_not_flagged(self) -> None:
        # Identical rep shape with shoulder_dx=0.0 (shoulders directly above hips
        # throughout) must not trip the back-lean rule.
        frames = ([ohp_frame(90, 0.45, frame_index=i) for i in range(4)]
                  + [ohp_frame(178, 0.15, frame_index=i + 4) for i in range(6)]
                  + [ohp_frame(90, 0.45, frame_index=i + 10) for i in range(4)])
        ids = {d.fault_id for d in self._run(frames)}
        self.assertNotIn("ohp_lumbar_hyperextension", ids)
