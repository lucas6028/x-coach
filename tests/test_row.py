import math
import unittest

import numpy as np

from src.pose.movements.base import RuleContext, run_detector


def _lm(x: float, y: float, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility}


def _elbow_xy(
    shoulder: tuple[float, float],
    wrist: tuple[float, float],
    elbow_angle_deg: float,
    side_sign: float,
) -> tuple[float, float]:
    """Place an elbow so that angle(shoulder, elbow, wrist) EQUALS `elbow_angle_deg` exactly.

    Two equal-length segments of length r spanning a shoulder-wrist chord of length d subtend
    an elbow angle of 2*asin(d / (2r)), so the r that produces a requested angle is
    r = d / (2*sin(angle/2)). The elbow then sits on the chord's perpendicular bisector at
    height h = sqrt(r^2 - (d/2)^2). Controlling the ANGLE directly is the property the ROM
    rule's fixtures need: `max_elbow_angle` equals the requested number by construction, so a
    boundary fixture really does sit one step either side of the 100-degree threshold.
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


def row_frame(
    trunk_angle_deg: float = 20.0,
    wrist_hip_dist: float = 0.08,
    elbow_angle_deg: float = 70.0,
    elbow_dy: float = 0.0,
    shoulder_tilt: float = 0.0,
    wrist_shift: float = 0.0,
    frame_index: int = 0,
    visibility: float = 0.95,
    strict_angle: bool = False,
) -> dict:
    """One bent-over row frame, image y growing DOWNWARD, viewed obliquely.

    Knobs, each controlling exactly one metric BY CONSTRUCTION:
      trunk_angle_deg -- angle of shoulder_mid -> hip_mid from horizontal. 0 = perfectly
                         hinged (torso horizontal), 90 = upright. Equals
                         `trunk_angle_from_horizontal_deg`.
      wrist_hip_dist  -- distance from each wrist to its same-side hip, in image units.
                         Equals `mean_wrist_hip_dist` when `wrist_shift` is 0.
      elbow_angle_deg -- angle(shoulder, elbow, wrist) per side; equals both
                         `min_elbow_angle` and `max_elbow_angle` when `elbow_dy` is 0.
      elbow_dy        -- extra downward displacement applied to the LEFT elbow only. Equals
                         `elbow_height_asymmetry`. The left elbow ANGLE then becomes derived
                         rather than requested, which is why no test asserts both at once.
      shoulder_tilt   -- signed image-y difference between the shoulders; equals
                         `shoulder_tilt` (the metric) in magnitude.
      wrist_shift     -- moves the RIGHT wrist further from its hip by this amount; equals
                         `wrist_travel_asymmetry`.
      strict_angle    -- opt-in guard, OFF by default (see CAVEAT). Does not itself change any
                         landmark.

    CAVEAT -- `elbow_angle_deg` is exact only when `shoulder_tilt` and `wrist_shift` are BOTH
    0. Both knobs move a landmark (the shoulder or the wrist) that is also the elbow's own
    chord endpoint; the elbow itself is anchored to the UNPERTURBED endpoint (see the comment
    below) so the two asymmetry metrics stay exact, but that means the frame's ACTUAL, measured
    elbow angle drifts off the requested value once either knob is nonzero -- by ~1.2 deg at
    shoulder_tilt=0.03 and by ~10.4 deg at wrist_shift=0.04 (both measured at
    elbow_angle_deg=99).

    This is NOT flagged unconditionally, because plenty of legitimate fixtures pass a nonzero
    `elbow_angle_deg` together with a nonzero `shoulder_tilt`/`wrist_shift` without caring about
    the angle's exactness at all -- e.g. a later task's per-frame clip helper that sets
    `elbow_angle_deg` on every frame just to keep the rep phased correctly, while the frame
    under test is really asserting on `shoulder_tilt`. Rejecting that combination outright would
    break those fixtures for a drift they never rely on. Instead this is OPT-IN: pass
    `strict_angle=True` when a test genuinely asserts on `elbow_angle_deg`/`min_elbow_angle`/
    `max_elbow_angle` for a frame that ALSO carries a nonzero asymmetry knob, and the loud
    `ValueError` fires instead of a silently-wrong boundary. Frames that merely set an angle in
    passing are unaffected.
    """
    if strict_angle and (shoulder_tilt != 0.0 or wrist_shift != 0.0):
        raise ValueError(
            "row_frame(strict_angle=True): elbow_angle_deg was combined with a nonzero "
            "shoulder_tilt or wrist_shift. Per the CAVEAT in this docstring, the elbow is "
            "anchored to the UNPERTURBED shoulder/wrist so the two asymmetry metrics stay "
            "exact, but that means the frame's ACTUAL measured elbow angle drifts off the "
            "requested elbow_angle_deg (~1.2 deg at shoulder_tilt=0.03, ~10.4 deg at "
            "wrist_shift=0.04, both measured at elbow_angle_deg=99) -- asserting on the angle "
            "here would silently check against the wrong number. Put the angle assertion and "
            "the asymmetry knobs on separate frames, or drop strict_angle=True if this frame "
            "does not actually assert on the angle."
        )

    hip_mid = (0.60, 0.55)
    trunk_len = 0.30
    theta = math.radians(trunk_angle_deg)
    shoulder_mid = (hip_mid[0] - trunk_len * math.cos(theta), hip_mid[1] - trunk_len * math.sin(theta))

    # FIXTURE NOTE (fixed during Task 1 implementation, not part of the original brief text):
    # both anchors use the SAME half-width, and elbow position is computed from the
    # UNPERTURBED (mirror-symmetric) shoulder/wrist anchors before `shoulder_tilt` and
    # `wrist_shift` are applied. `_elbow_xy` holds the ANGLE fixed but not the elbow's absolute
    # position -- for this geometry elbow_y = (shoulder_y + wrist_y) / 2 + C * dx (C a constant
    # depending only on the angle), an EXACT identity. Two consequences follow, both verified
    # numerically before this fix: (1) unequal shoulder/hip half-widths make dx differ
    # left-vs-right even with every asymmetry knob at 0, giving a nonzero baseline
    # `elbow_height_delta_signed`; (2) feeding a tilted shoulder or a shifted wrist straight
    # into `_elbow_xy` leaks exactly HALF of `shoulder_tilt` and HALF of `wrist_shift` into
    # `elbow_height_asymmetry`, because both terms sit inside that same `(shoulder_y +
    # wrist_y) / 2` average. Computing the elbow from the symmetric anchors and applying the
    # tilt/shift only to the OUTPUT landmark afterward removes both couplings, so each knob
    # controls exactly the one metric the docstring above promises -- including in
    # `test_asymmetry_metrics_equal_their_knobs`, which exercises all three knobs at once.
    half_width = 0.06
    left_shoulder_anchor = (shoulder_mid[0] - half_width, shoulder_mid[1])
    right_shoulder_anchor = (shoulder_mid[0] + half_width, shoulder_mid[1])
    left_hip = (hip_mid[0] - half_width, hip_mid[1])
    right_hip = (hip_mid[0] + half_width, hip_mid[1])

    # Wrists sit directly BELOW their own hip (the bar hangs under the shoulders in a hinge),
    # so wrist-to-hip distance is exactly the requested value.
    left_wrist = (left_hip[0], left_hip[1] + wrist_hip_dist)
    right_wrist_anchor = (right_hip[0], right_hip[1] + wrist_hip_dist)

    left_elbow = _elbow_xy(left_shoulder_anchor, left_wrist, elbow_angle_deg, +1.0)
    left_elbow = (left_elbow[0], left_elbow[1] + elbow_dy)
    right_elbow = _elbow_xy(right_shoulder_anchor, right_wrist_anchor, elbow_angle_deg, +1.0)

    # `shoulder_tilt` and `wrist_shift` are applied to the LANDMARKS only, after the elbow
    # position above is already fixed, so neither retroactively perturbs the elbow chord.
    left_shoulder = (left_shoulder_anchor[0], left_shoulder_anchor[1] - shoulder_tilt / 2.0)
    right_shoulder = (right_shoulder_anchor[0], right_shoulder_anchor[1] + shoulder_tilt / 2.0)
    right_wrist = (right_wrist_anchor[0], right_wrist_anchor[1] + wrist_shift)

    landmarks = [_lm(0.0, 0.0, 0.0) for _ in range(33)]
    landmarks[11] = _lm(*left_shoulder, visibility)
    landmarks[12] = _lm(*right_shoulder, visibility)
    landmarks[13] = _lm(*left_elbow, visibility)
    landmarks[14] = _lm(*right_elbow, visibility)
    landmarks[15] = _lm(*left_wrist, visibility)
    landmarks[16] = _lm(*right_wrist, visibility)
    landmarks[23] = _lm(*left_hip, visibility)
    landmarks[24] = _lm(*right_hip, visibility)
    return {"frame_index": frame_index, "landmarks": landmarks}


class RowFrameFixtureTest(unittest.TestCase):
    """Guards on the FIXTURE helper itself, not on `src.pose.movements.row`."""

    def test_strict_angle_combined_with_an_asymmetry_knob_raises(self) -> None:
        # Each combination independently trips the opt-in guard described in row_frame's
        # CAVEAT -- strict_angle=True asserts the caller cares about elbow_angle_deg's
        # exactness, and either asymmetry knob silently drifts the frame's actual elbow angle
        # off the requested value.
        with self.assertRaises(ValueError):
            row_frame(elbow_angle_deg=99.0, shoulder_tilt=0.03, strict_angle=True)
        with self.assertRaises(ValueError):
            row_frame(elbow_angle_deg=99.0, wrist_shift=0.04, strict_angle=True)
        with self.assertRaises(ValueError):
            row_frame(elbow_angle_deg=99.0, shoulder_tilt=0.03, wrist_shift=0.04, strict_angle=True)

    def test_the_same_call_without_strict_angle_does_not_raise(self) -> None:
        # Identical knobs, `strict_angle` simply omitted: a fixture that sets an angle in
        # passing (without asserting on it) must be unaffected by the guard.
        row_frame(elbow_angle_deg=99.0, shoulder_tilt=0.03)
        row_frame(elbow_angle_deg=99.0, wrist_shift=0.04)
        row_frame(elbow_angle_deg=99.0, shoulder_tilt=0.03, wrist_shift=0.04)

    def test_strict_angle_alone_does_not_raise(self) -> None:
        # strict_angle only fires together with a nonzero asymmetry knob; on its own (no tilt,
        # no shift) there is nothing to drift the angle, so it is a no-op.
        row_frame(elbow_angle_deg=99.0, strict_angle=True)


class RowMetricsTest(unittest.TestCase):
    def test_trunk_angle_equals_the_constructed_hinge_angle(self) -> None:
        from src.pose.movements.row import row_compute_raw

        for requested in (0.0, 20.0, 45.0, 80.0):
            raw = row_compute_raw([row_frame(trunk_angle_deg=requested)], fps=30.0)
            self.assertAlmostEqual(raw[0]["trunk_angle_from_horizontal_deg"], requested, places=4)

    def test_elbow_angles_equal_the_constructed_angle(self) -> None:
        from src.pose.movements.row import row_compute_raw

        raw = row_compute_raw([row_frame(elbow_angle_deg=95.0)], fps=30.0)
        self.assertAlmostEqual(raw[0]["left_elbow_angle"], 95.0, places=3)
        self.assertAlmostEqual(raw[0]["right_elbow_angle"], 95.0, places=3)
        self.assertAlmostEqual(raw[0]["min_elbow_angle"], 95.0, places=3)
        self.assertAlmostEqual(raw[0]["max_elbow_angle"], 95.0, places=3)

    def test_min_and_max_elbow_angle_pick_the_right_arm(self) -> None:
        from src.pose.movements.row import row_compute_raw

        # elbow_dy displaces the LEFT elbow only, so the two arms differ.
        raw = row_compute_raw([row_frame(elbow_angle_deg=90.0, elbow_dy=0.05)], fps=30.0)
        left, right = raw[0]["left_elbow_angle"], raw[0]["right_elbow_angle"]
        self.assertNotAlmostEqual(left, right, places=2)
        self.assertAlmostEqual(raw[0]["min_elbow_angle"], min(left, right), places=6)
        self.assertAlmostEqual(raw[0]["max_elbow_angle"], max(left, right), places=6)

    def test_wrist_hip_distance_equals_the_constructed_distance(self) -> None:
        from src.pose.movements.row import row_compute_raw

        raw = row_compute_raw([row_frame(wrist_hip_dist=0.15)], fps=30.0)
        self.assertAlmostEqual(raw[0]["left_wrist_hip_dist"], 0.15, places=6)
        self.assertAlmostEqual(raw[0]["right_wrist_hip_dist"], 0.15, places=6)
        self.assertAlmostEqual(raw[0]["mean_wrist_hip_dist"], 0.15, places=6)

    def test_asymmetry_metrics_equal_their_knobs(self) -> None:
        from src.pose.movements.row import row_compute_raw

        raw = row_compute_raw(
            [row_frame(elbow_dy=0.07, shoulder_tilt=0.03, wrist_shift=0.04)], fps=30.0
        )
        self.assertAlmostEqual(raw[0]["elbow_height_asymmetry"], 0.07, places=6)
        self.assertAlmostEqual(raw[0]["shoulder_tilt"], 0.03, places=6)
        self.assertAlmostEqual(raw[0]["wrist_travel_asymmetry"], 0.04, places=6)

    def test_the_signed_elbow_delta_records_which_side_is_lower(self) -> None:
        from src.pose.movements.row import row_compute_raw

        # elbow_dy displaces the LEFT elbow DOWNWARD (image y grows down), so the signed delta
        # (left_y - right_y) must be POSITIVE and equal to the knob.
        raw = row_compute_raw([row_frame(elbow_dy=0.07)], fps=30.0)
        self.assertAlmostEqual(raw[0]["elbow_height_delta_signed"], 0.07, places=6)
        raw = row_compute_raw([row_frame(elbow_dy=-0.07)], fps=30.0)
        self.assertAlmostEqual(raw[0]["elbow_height_delta_signed"], -0.07, places=6)

    def test_shoulder_normalized_diagnostic_is_the_ratio_of_the_two(self) -> None:
        from src.pose.movements.row import row_compute_raw

        raw = row_compute_raw([row_frame(wrist_hip_dist=0.12)], fps=30.0)
        expected = raw[0]["mean_wrist_hip_dist"] / raw[0]["shoulder_width"]
        self.assertAlmostEqual(raw[0]["wrist_hip_dist_shoulder_norm"], expected, places=6)

    def test_one_missing_landmark_invalidates_the_whole_frame(self) -> None:
        from src.pose.movements.row import row_compute_raw

        frame = row_frame()
        frame["landmarks"][13] = _lm(0.5, 0.5, 0.10)  # left elbow below VISIBILITY_THRESHOLD
        raw = row_compute_raw([frame], fps=30.0)
        self.assertFalse(raw[0]["valid"])
        self.assertNotIn("trunk_angle_from_horizontal_deg", raw[0])

    def test_non_dict_frame_is_refused_rather_than_crashing(self) -> None:
        from src.pose.movements.row import row_compute_raw

        raw = row_compute_raw([None, "nonsense"], fps=30.0)
        self.assertEqual([item["valid"] for item in raw], [False, False])


class RowDerivativeTest(unittest.TestCase):
    def test_constant_velocity_gives_zero_acceleration(self) -> None:
        from src.pose.movements.row import row_compute_raw

        frames = [
            row_frame(wrist_hip_dist=0.05 + 0.01 * i, frame_index=i) for i in range(7)
        ]
        raw = row_compute_raw(frames, fps=30.0)
        for item in raw[2:5]:
            self.assertAlmostEqual(item["wrist_accel_norm"], 0.0, places=4)

    def test_boundary_frames_carry_nan_rather_than_a_one_sided_estimate(self) -> None:
        from src.pose.movements.row import row_compute_raw

        frames = [row_frame(wrist_hip_dist=0.05 + 0.01 * i, frame_index=i) for i in range(7)]
        raw = row_compute_raw(frames, fps=30.0)
        for index in (0, 1, 5, 6):
            self.assertTrue(math.isnan(raw[index]["wrist_accel_norm"]))

    def test_trunk_angle_speed_is_degrees_per_second(self) -> None:
        from src.pose.movements.row import row_compute_raw

        # 2 degrees per frame at 30 fps == 60 deg/s.
        frames = [row_frame(trunk_angle_deg=10.0 + 2.0 * i, frame_index=i) for i in range(5)]
        raw = row_compute_raw(frames, fps=30.0)
        self.assertAlmostEqual(raw[2]["trunk_angle_speed_deg_s"], 60.0, places=3)

    def test_an_interior_invalid_frame_poisons_only_its_derivative_neighbours(self) -> None:
        """`wrist_accel_norm` is `_derivative` applied TWICE (velocity, then acceleration), so
        a single-frame hole propagates through two composed central differences with a
        DERIVED radius, not a guessed one:

        Frame 3 of 7 is made invalid, so `wrist_mid_x`/`wrist_mid_y` carry NaN at index 3.
        - PASS 1 (velocity): `_derivative`'s central difference at index i reads positions
          i-1 and i+1, never i itself, so a hole exactly AT index 3 poisons velocity at its
          two neighbours, indices 2 and 4 -- NOT at index 3, where the velocity estimate
          skips over the hole and is unaffected. Velocity is therefore NaN at {0 (boundary),
          2, 4, 6 (boundary)} and finite at {1, 3, 5}.
        - PASS 2 (acceleration): applying the same rule to the velocity series above, each
          NaN velocity index poisons acceleration at its own two neighbours. Velocity index 0
          poisons acceleration index 1; velocity index 2 poisons acceleration indices 1 and 3;
          velocity index 4 poisons acceleration indices 3 and 5; velocity index 6 poisons
          acceleration index 5. Acceleration indices 2 and 4 each depend on ONE poisoned and
          ONE clean velocity value (velocity[1]/velocity[3] and velocity[3]/velocity[5]
          respectively) plus the always-clean velocity[3] -- so they stay finite.

        Net radius for a hole at frame k=3 (verified numerically before writing this
        assertion): acceleration is NaN at k-2, k, k+2 -> frames {1, 3, 5}, in addition to the
        permanent boundary NaNs at {0, 6}; it stays finite at k-1, k+1 -> frames {2, 4}. Frame
        3 itself has no `wrist_accel_norm` key at all (the whole frame is invalid).
        """
        from src.pose.movements.row import row_compute_raw

        frames = [row_frame(wrist_hip_dist=0.05 + 0.01 * i, frame_index=i) for i in range(7)]
        frames[3]["landmarks"][15] = _lm(0.5, 0.5, 0.10)  # left wrist below VISIBILITY_THRESHOLD
        raw = row_compute_raw(frames, fps=30.0)

        self.assertFalse(raw[3]["valid"])
        self.assertNotIn("wrist_accel_norm", raw[3])

        for index in (0, 1, 5, 6):
            self.assertTrue(math.isnan(raw[index]["wrist_accel_norm"]), f"index {index}")
        for index in (2, 4):
            self.assertFalse(math.isnan(raw[index]["wrist_accel_norm"]), f"index {index}")


class RowPhaseTest(unittest.TestCase):
    def test_phases_run_setup_pull_peak_lower(self) -> None:
        from src.pose.movements.row import row_assign_phases, row_compute_raw

        angles = [170.0] * 4 + [150.0, 120.0, 95.0, 70.0, 60.0, 70.0, 95.0, 120.0, 150.0, 170.0]
        frames = [row_frame(elbow_angle_deg=a, frame_index=i) for i, a in enumerate(angles)]
        phases = row_assign_phases(row_compute_raw(frames, fps=30.0))
        self.assertEqual(phases[0], "setup")
        self.assertEqual(phases[8], "peak")
        self.assertIn("pull", phases)
        self.assertIn("lower", phases)
        self.assertEqual(len(phases), len(frames))

    def test_empty_clip_returns_empty_phases(self) -> None:
        from src.pose.movements.row import row_assign_phases

        self.assertEqual(row_assign_phases([]), [])

    def test_clip_with_no_finite_signal_is_entirely_unknown(self) -> None:
        from src.pose.movements.row import row_assign_phases

        self.assertEqual(row_assign_phases([{"valid": False}, {"valid": False}]), ["unknown", "unknown"])

    def test_an_invalid_frame_inside_the_setup_window_is_unknown_not_setup(self) -> None:
        from src.pose.movements.row import row_assign_phases, row_compute_raw

        frames = [row_frame(elbow_angle_deg=170.0 - 5.0 * i, frame_index=i) for i in range(14)]
        frames[0]["landmarks"][15] = _lm(0.5, 0.5, 0.10)
        phases = row_assign_phases(row_compute_raw(frames, fps=30.0))
        self.assertEqual(phases[0], "unknown")


if __name__ == "__main__":
    unittest.main()
