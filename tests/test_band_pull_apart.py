import math
import unittest

import numpy as np

from src.pose.movements.band_pull_apart import (
    BAND_PULL_APART_METRIC_KEYS,
    band_pull_apart_assign_phases,
    band_pull_apart_compute_raw,
)


def _lm(x: float, y: float, z: float = 0.0, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": z, "visibility": visibility}


def _elbow_xy(
    shoulder: tuple[float, float],
    wrist: tuple[float, float],
    elbow_angle_deg: float,
    side_sign: float,
) -> tuple[float, float]:
    """Place an elbow so that angle(shoulder, elbow, wrist) EQUALS `elbow_angle_deg` exactly.

    Two equal-length segments of length r spanning a shoulder-wrist chord of length d subtend
    an elbow angle of 2*asin(d / (2r)), so the r producing a requested angle is
    r = d / (2*sin(angle/2)). The elbow then sits on the chord's perpendicular bisector at
    height h = sqrt(r^2 - (d/2)^2). Controlling the ANGLE directly is what the ROM rule's
    fixtures need: `min_elbow_angle` equals the requested number by construction, so a boundary
    fixture really does sit one step either side of the 150-degree threshold. Copied from
    tests/test_row.py, where the same construction backs the row's elbow fixtures.
    """
    sx, sy = shoulder
    wx, wy = wrist
    dx, dy = wx - sx, wy - sy
    d = math.hypot(dx, dy)
    half = math.radians(elbow_angle_deg) / 2.0
    r = d / (2.0 * math.sin(half))
    h = math.sqrt(max(r * r - (d / 2.0) ** 2, 0.0))
    ux, uy = dx / d, dy / d
    px, py = -uy, ux
    return (sx + dx / 2.0 + side_sign * h * px, sy + dy / 2.0 + side_sign * h * py)


def _elbow_xyz(
    shoulder: tuple[float, float, float],
    wrist: tuple[float, float, float],
    elbow_angle_deg: float,
    side_sign: float,
) -> tuple[float, float, float]:
    """3D-correcting wrapper around `_elbow_xy`, needed because `angle_degrees`
    (src/pose/geometry.py) computes `dims=3` and therefore consumes z.

    `_elbow_xy` places the elbow so a PURELY 2D (z=0) triangle subtends the requested angle. Once
    the wrist carries a nonzero z relative to the shoulder (via `wrist_depth_offset`), the TRUE
    shoulder-wrist chord in 3D is longer than its image-plane projection, and the 2D elbow
    position then understates the requested angle -- measured directly at
    elbow_angle_deg=140 with the fixture's default wrist_depth_offset=-0.30, the actual 3D angle
    came out 96.4 degrees, a 43.6-degree miss. Not "slightly off": a z-only correction (setting
    landmarks[13]/[14]'s z without moving x/y) cannot fix this -- it cannot lengthen the chord,
    only rotate the elbow's existing 2D position out of the z=0 plane, and no single z value
    reproduces the requested angle exactly.

    THE FIX: the elbow's distance from the shoulder-wrist midpoint is h = (d/2) * cot(angle/2),
    LINEAR in the chord length d for a FIXED angle. Scaling `_elbow_xy`'s in-plane offset from
    the midpoint by `d3 / d2` -- the ratio of the true 3D chord length to its 2D (z=0)
    projection -- reproduces the identical isosceles construction against the longer chord, so
    the resulting 3D angle equals `elbow_angle_deg` exactly (verified numerically to 5 decimal
    places for elbow_angle_deg in {140, 170}). The elbow's z is set to the shoulder-wrist
    midpoint's z: `_elbow_xy`'s offset direction (px, py) is perpendicular to (dx, dy) by
    construction and carries no z component, so it cannot move the elbow off the midpoint's z.
    """
    sx, sy, sz = shoulder
    wx, wy, wz = wrist
    d2 = math.hypot(wx - sx, wy - sy)
    d3 = math.sqrt((wx - sx) ** 2 + (wy - sy) ** 2 + (wz - sz) ** 2)
    scale = d3 / d2
    mx, my, mz = (sx + wx) / 2.0, (sy + wy) / 2.0, (sz + wz) / 2.0
    ex2, ey2 = _elbow_xy((sx, sy), (wx, wy), elbow_angle_deg, side_sign)
    return (mx + scale * (ex2 - mx), my + scale * (ey2 - my), mz)


def bpa_frame(
    spread_ratio: float = 1.8,
    shoulder_ear_gap: float = 0.12,
    left_gap_delta: float = 0.0,
    elbow_angle_deg: float = 170.0,
    trunk_lean_deg: float = 0.0,
    wrist_depth_offset: float = -0.30,
    frame_index: int = 0,
    visibility: float = 0.95,
) -> dict:
    """One standing band pull-apart frame, image y growing DOWNWARD, viewed from behind.

    Knobs, each controlling exactly one metric BY CONSTRUCTION:
      spread_ratio       -- wrist separation / shoulder width. Equals
                            `wrist_spread_shoulder_norm`.
      shoulder_ear_gap   -- image-y distance from each ear up to its shoulder. Equals both
                            `left_shoulder_ear_gap` and `right_shoulder_ear_gap`.
      left_gap_delta     -- added to the LEFT gap only, so a unilateral shrug can be built.
      elbow_angle_deg    -- angle(shoulder, elbow, wrist) per side; equals `min_elbow_angle`.
      trunk_lean_deg     -- signed pitch of hip_mid -> shoulder_mid from vertical. Positive
                            moves the shoulders toward +x. Equals
                            `trunk_lean_image_signed_deg`.
      wrist_depth_offset -- mean wrist z minus mean shoulder z. Equals `wrist_depth_offset`.
                            Negative = wrists nearer the camera = lifter faces the camera.
    """
    shoulder_width = 0.20
    hip_mid = (0.50, 0.70)
    trunk_len = 0.30
    theta = math.radians(trunk_lean_deg)
    # Pitch measured from VERTICAL, positive toward +x.
    shoulder_mid = (
        hip_mid[0] + trunk_len * math.sin(theta),
        hip_mid[1] - trunk_len * math.cos(theta),
    )
    sy = shoulder_mid[1]
    left_shoulder = (shoulder_mid[0] - shoulder_width / 2.0, sy)
    right_shoulder = (shoulder_mid[0] + shoulder_width / 2.0, sy)

    half_spread = spread_ratio * shoulder_width / 2.0
    wrist_y = sy + 0.02
    left_wrist = (shoulder_mid[0] - half_spread, wrist_y)
    right_wrist = (shoulder_mid[0] + half_spread, wrist_y)

    shoulder_z = 0.0
    wrist_z = shoulder_z + wrist_depth_offset

    # `_elbow_xyz`, NOT `_elbow_xy` directly, places the elbow -- see its docstring for why a
    # plain 2D placement understates `elbow_angle_deg` once the wrist carries nonzero z.
    left_elbow = _elbow_xyz(
        (left_shoulder[0], left_shoulder[1], shoulder_z),
        (left_wrist[0], left_wrist[1], wrist_z),
        elbow_angle_deg,
        side_sign=1.0,
    )
    right_elbow = _elbow_xyz(
        (right_shoulder[0], right_shoulder[1], shoulder_z),
        (right_wrist[0], right_wrist[1], wrist_z),
        elbow_angle_deg,
        side_sign=-1.0,
    )

    left_ear = (left_shoulder[0], sy - (shoulder_ear_gap + left_gap_delta))
    right_ear = (right_shoulder[0], sy - shoulder_ear_gap)

    landmarks = [_lm(0.5, 0.3, visibility=visibility) for _ in range(33)]
    landmarks[7] = _lm(*left_ear, visibility=visibility)
    landmarks[8] = _lm(*right_ear, visibility=visibility)
    landmarks[11] = _lm(*left_shoulder, z=shoulder_z, visibility=visibility)
    landmarks[12] = _lm(*right_shoulder, z=shoulder_z, visibility=visibility)
    landmarks[13] = _lm(*left_elbow, visibility=visibility)
    landmarks[14] = _lm(*right_elbow, visibility=visibility)
    landmarks[15] = _lm(*left_wrist, z=wrist_z, visibility=visibility)
    landmarks[16] = _lm(*right_wrist, z=wrist_z, visibility=visibility)
    landmarks[23] = _lm(hip_mid[0] - 0.08, hip_mid[1], visibility=visibility)
    landmarks[24] = _lm(hip_mid[0] + 0.08, hip_mid[1], visibility=visibility)
    return {"frame_index": frame_index, "landmarks": landmarks}


class BandPullApartMetricsTest(unittest.TestCase):
    def test_spread_ratio_knob_equals_the_emitted_metric(self) -> None:
        raw = band_pull_apart_compute_raw([bpa_frame(spread_ratio=1.35)], fps=30.0)
        self.assertTrue(raw[0]["valid"])
        self.assertAlmostEqual(raw[0]["wrist_spread_shoulder_norm"], 1.35, places=4)

    def test_shoulder_ear_gap_knob_equals_the_emitted_metric(self) -> None:
        raw = band_pull_apart_compute_raw([bpa_frame(shoulder_ear_gap=0.09)], fps=30.0)
        self.assertAlmostEqual(raw[0]["left_shoulder_ear_gap"], 0.09, places=4)
        self.assertAlmostEqual(raw[0]["right_shoulder_ear_gap"], 0.09, places=4)

    def test_left_gap_delta_moves_only_the_left_side(self) -> None:
        raw = band_pull_apart_compute_raw(
            [bpa_frame(shoulder_ear_gap=0.12, left_gap_delta=-0.05)], fps=30.0
        )
        self.assertAlmostEqual(raw[0]["left_shoulder_ear_gap"], 0.07, places=4)
        self.assertAlmostEqual(raw[0]["right_shoulder_ear_gap"], 0.12, places=4)

    def test_elbow_angle_knob_equals_min_elbow_angle(self) -> None:
        raw = band_pull_apart_compute_raw([bpa_frame(elbow_angle_deg=140.0)], fps=30.0)
        self.assertAlmostEqual(raw[0]["min_elbow_angle"], 140.0, places=2)

    def test_trunk_lean_knob_equals_the_signed_image_pitch(self) -> None:
        raw = band_pull_apart_compute_raw([bpa_frame(trunk_lean_deg=12.0)], fps=30.0)
        self.assertAlmostEqual(raw[0]["trunk_lean_image_signed_deg"], 12.0, places=2)
        raw = band_pull_apart_compute_raw([bpa_frame(trunk_lean_deg=-12.0)], fps=30.0)
        self.assertAlmostEqual(raw[0]["trunk_lean_image_signed_deg"], -12.0, places=2)

    def test_wrist_depth_offset_knob_equals_the_emitted_metric(self) -> None:
        raw = band_pull_apart_compute_raw([bpa_frame(wrist_depth_offset=-0.42)], fps=30.0)
        self.assertAlmostEqual(raw[0]["wrist_depth_offset"], -0.42, places=4)

    def test_one_dropped_landmark_invalidates_the_whole_frame(self) -> None:
        frame = bpa_frame()
        frame["landmarks"][15] = _lm(0.4, 0.5, visibility=0.10)
        raw = band_pull_apart_compute_raw([frame], fps=30.0)
        self.assertFalse(raw[0]["valid"])
        # An invalid frame carries NO metric keys at all -- every rule masking on
        # `frame.valid` therefore goes silent for it, not just the wrist-dependent ones.
        for key in BAND_PULL_APART_METRIC_KEYS:
            self.assertNotIn(key, raw[0])

    def test_metric_keys_match_the_emitted_metrics_exactly(self) -> None:
        raw = band_pull_apart_compute_raw([bpa_frame(frame_index=i) for i in range(5)], fps=30.0)
        emitted = set(raw[2]) - {"frame_index", "time", "valid", "lower_body_visibility"}
        self.assertEqual(emitted, set(BAND_PULL_APART_METRIC_KEYS))

    def test_trunk_speed_is_nan_at_both_boundaries(self) -> None:
        frames = [bpa_frame(trunk_lean_deg=float(i), frame_index=i) for i in range(5)]
        raw = band_pull_apart_compute_raw(frames, fps=30.0)
        self.assertTrue(math.isnan(raw[0]["trunk_angle_speed_deg_s"]))
        self.assertTrue(math.isnan(raw[-1]["trunk_angle_speed_deg_s"]))
        self.assertTrue(math.isfinite(raw[2]["trunk_angle_speed_deg_s"]))


class BandPullApartPhaseTest(unittest.TestCase):
    def _rep(self) -> list[dict]:
        # hands together -> spread -> together, the movement's excursion.
        ratios = [0.4, 0.6, 1.0, 1.5, 1.9, 2.0, 1.9, 1.4, 0.9, 0.5]
        return [bpa_frame(spread_ratio=r, frame_index=i) for i, r in enumerate(ratios)]

    def test_one_phase_per_frame(self) -> None:
        raw = band_pull_apart_compute_raw(self._rep(), fps=30.0)
        self.assertEqual(len(band_pull_apart_assign_phases(raw)), len(raw))

    def test_opening_frames_are_setup_and_the_widest_frames_are_peak(self) -> None:
        raw = band_pull_apart_compute_raw(self._rep(), fps=30.0)
        phases = band_pull_apart_assign_phases(raw)
        self.assertEqual(phases[0], "setup")
        widest = max(range(len(raw)), key=lambda i: raw[i]["wrist_spread_shoulder_norm"])
        self.assertEqual(phases[widest], "peak")

    def test_frames_before_the_peak_are_pull_and_after_are_return(self) -> None:
        raw = band_pull_apart_compute_raw(self._rep(), fps=30.0)
        phases = band_pull_apart_assign_phases(raw)
        self.assertIn("pull", phases)
        self.assertIn("return", phases)
        self.assertLess(phases.index("pull"), len(phases) - 1 - phases[::-1].index("return"))

    def test_empty_clip_and_all_nan_clip(self) -> None:
        self.assertEqual(band_pull_apart_assign_phases([]), [])
        self.assertEqual(band_pull_apart_assign_phases([{"valid": False}] * 3), ["unknown"] * 3)

    def test_an_occluded_opening_frame_is_unknown_not_setup(self) -> None:
        frames = self._rep()
        frames[0]["landmarks"][11] = _lm(0.4, 0.5, visibility=0.10)
        raw = band_pull_apart_compute_raw(frames, fps=30.0)
        self.assertEqual(band_pull_apart_assign_phases(raw)[0], "unknown")


if __name__ == "__main__":
    unittest.main()
