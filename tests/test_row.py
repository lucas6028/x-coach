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


def _row_clip(
    pull_frames: int = 10,
    setup_trunk: float = 20.0,
    peak_trunk: float = 20.0,
    setup_tilt: float = 0.0,
    peak_tilt: float = 0.0,
    peak_wrist_hip: float = 0.05,
    peak_elbow: float = 70.0,
    peak_elbow_dy: float = 0.0,
) -> list[dict]:
    """A synthetic single rep: 6 setup frames, then a descent into a held peak.

    CONSTANT-VALUE SEGMENTS ARE THE POINT. `run_detector` median-filters every metric with a
    5-frame window; a segment held at one value makes that filter a no-op, so an asserted
    severity is EXACT rather than approximately-whatever-the-filter-left. The OHP review found
    5 of 10 threshold mutants surviving because every fixture sat at an extreme instead of on
    a boundary, so boundary fixtures must be exact.
    """
    frames: list[dict] = []
    index = 0
    for _ in range(6):
        frames.append(
            row_frame(
                trunk_angle_deg=setup_trunk,
                shoulder_tilt=setup_tilt,
                elbow_angle_deg=170.0,
                wrist_hip_dist=0.30,
                frame_index=index,
            )
        )
        index += 1
    for _ in range(pull_frames):
        frames.append(
            row_frame(
                trunk_angle_deg=peak_trunk,
                shoulder_tilt=peak_tilt,
                elbow_angle_deg=peak_elbow,
                elbow_dy=peak_elbow_dy,
                wrist_hip_dist=peak_wrist_hip,
                frame_index=index,
            )
        )
        index += 1
    return frames


def _run_rule(rule, frames: list[dict], view_type: str = "rear_oblique", view_confidence: float = 0.8):
    """Run ONE rule over a clip, bypassing rep segmentation AND `run_detector`'s smoothing.

    Rules receive a per-rep slice from `run_detector`; here the whole clip IS the window, which
    is the `only_partial_reps` fallback shape and is what a single-rep fixture should exercise.

    This also bypasses `run_detector`'s 5-frame median filter over every metric key: `core` is
    built straight from `row_compute_raw`/`row_assign_phases`, never through `run_detector`
    itself. `_row_clip`'s docstring justifies its constant-value segments by citing that
    filter (a held segment makes a median filter a no-op) -- true in the full pipeline, but not
    the reason boundary fixtures need to be exact ON THIS PATH, since there is no filter here to
    be a no-op of. Constant segments are still the right fixture shape regardless: an exact
    input still makes an exact, non-approximate assertion possible, and the assertions stay
    correct if this helper is ever routed through `run_detector` later.
    """
    from src.pose.movements.base import CoreFrame, RuleContext
    from src.pose.movements.row import ROW_METRIC_KEYS, row_assign_phases, row_compute_raw

    raw = row_compute_raw(frames, fps=30.0)
    phases = row_assign_phases(raw)
    core = [
        CoreFrame(
            frame_index=int(item.get("frame_index", i) or i),
            time=float(item.get("time", 0.0) or 0.0),
            phase=phases[i],
            valid=bool(item.get("valid", False)),
            lower_body_visibility=float(item.get("lower_body_visibility", 0.0) or 0.0),
            metrics={key: float(item.get(key, np.nan)) for key in ROW_METRIC_KEYS},
        )
        for i, item in enumerate(raw)
    ]
    ctx = RuleContext(fps=30.0, view_type=view_type, view_confidence=view_confidence, min_frames=6)
    return rule(core, ctx)


class RowTorsoRisingTest(unittest.TestCase):
    def test_a_torso_held_at_the_setup_angle_does_not_fire(self) -> None:
        from src.pose.movements.row import rule_torso_rising

        self.assertEqual(_run_rule(rule_torso_rising, _row_clip(setup_trunk=20.0, peak_trunk=20.0)), [])

    def test_just_under_fifteen_degrees_of_rise_does_not_fire(self) -> None:
        from src.pose.movements.row import rule_torso_rising

        clip = _row_clip(setup_trunk=20.0, peak_trunk=34.9)
        self.assertEqual(_run_rule(rule_torso_rising, clip), [])

    def test_just_over_fifteen_degrees_of_rise_fires(self) -> None:
        from src.pose.movements.row import rule_torso_rising

        clip = _row_clip(setup_trunk=20.0, peak_trunk=35.1)
        detections = _run_rule(rule_torso_rising, clip)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "row_torso_rising")

    def test_severity_is_exact_at_the_ramp_midpoint(self) -> None:
        from src.pose.movements.row import rule_torso_rising

        # Ramp 15 -> 37.5; a 26.25-degree rise is exactly half way.
        clip = _row_clip(setup_trunk=20.0, peak_trunk=46.25)
        detections = _run_rule(rule_torso_rising, clip)
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].severity, 0.5, places=3)

    def test_severity_saturates_at_the_ramp_end(self) -> None:
        from src.pose.movements.row import rule_torso_rising

        clip = _row_clip(setup_trunk=10.0, peak_trunk=60.0)
        detections = _run_rule(rule_torso_rising, clip)
        self.assertAlmostEqual(detections[0].severity, 1.0, places=6)

    def test_an_off_axis_view_downgrades_rather_than_silencing(self) -> None:
        from src.pose.movements.row import rule_torso_rising

        clip = _row_clip(setup_trunk=20.0, peak_trunk=46.25)
        front = _run_rule(rule_torso_rising, clip, view_type="front")
        self.assertEqual(len(front), 1)
        self.assertEqual(front[0].observability, "medium")
        oblique = _run_rule(rule_torso_rising, clip, view_type="rear_oblique")
        self.assertEqual(oblique[0].observability, "high")
        self.assertLess(front[0].confidence, oblique[0].confidence)

    def test_a_window_with_no_setup_frames_emits_nothing(self) -> None:
        from src.pose.movements.row import rule_torso_rising

        clip = _row_clip(setup_trunk=20.0, peak_trunk=60.0)
        for frame in clip[:6]:
            frame["landmarks"][11] = _lm(0.5, 0.5, 0.10)
        self.assertEqual(_run_rule(rule_torso_rising, clip), [])


class RowSetupBaselineTest(unittest.TestCase):
    """Tests of `_setup_baseline` itself -- the shared helper Tasks 3-5 reuse -- not of any one
    rule that consumes it.
    """

    def test_the_baseline_is_the_median_not_the_mean_of_a_non_constant_setup(self) -> None:
        """Every `_row_clip` fixture holds `setup_trunk` CONSTANT across the whole setup slice,
        where median and mean coincide -- that is not enough to prove `_setup_baseline` actually
        calls `np.median` rather than `np.mean` (the same gap Task 1's `_derivative` had, for
        the same reason: a fixture that cannot distinguish the two implementations passes either
        way). Four setup frames at the true resting angle (20 deg) plus a fifth outlier at
        60 deg MEDIAN to 20 (the outlier is out-voted 4-to-1) but would MEAN to 28 -- a
        mean-based helper fails the `assertAlmostEqual(baseline, 20.0, ...)` below.

        Frame count is chosen, not assumed: `row_assign_phases`'s `setup_cutoff =
        max(1, int(frame_count * 0.15))` needs to land on EXACTLY 5 for this fixture's 5 crafted
        values to be the whole setup slice (34 frames -> int(34 * 0.15) == 5). The phase
        assertions below check that this landed as intended rather than trusting the arithmetic
        silently.
        """
        from src.pose.movements.base import CoreFrame
        from src.pose.movements.row import (
            ROW_METRIC_KEYS,
            _setup_baseline,
            row_assign_phases,
            row_compute_raw,
        )

        setup_trunks = [20.0, 20.0, 20.0, 20.0, 60.0]
        frames: list[dict] = []
        index = 0
        for trunk in setup_trunks:
            frames.append(
                row_frame(trunk_angle_deg=trunk, elbow_angle_deg=170.0, wrist_hip_dist=0.30, frame_index=index)
            )
            index += 1
        for _ in range(29):
            frames.append(
                row_frame(trunk_angle_deg=20.0, elbow_angle_deg=70.0, wrist_hip_dist=0.05, frame_index=index)
            )
            index += 1

        raw = row_compute_raw(frames, fps=30.0)
        phases = row_assign_phases(raw)
        # This fixture's phase cutoff must land on EXACTLY these 5 frames -- not 4, not 6 --
        # or the test below would silently exercise a different setup slice than intended.
        self.assertEqual(phases[:5], ["setup"] * 5)
        self.assertNotEqual(phases[5], "setup")

        core = [
            CoreFrame(
                frame_index=int(item.get("frame_index", i) or i),
                time=float(item.get("time", 0.0) or 0.0),
                phase=phases[i],
                valid=bool(item.get("valid", False)),
                lower_body_visibility=float(item.get("lower_body_visibility", 0.0) or 0.0),
                metrics={key: float(item.get(key, np.nan)) for key in ROW_METRIC_KEYS},
            )
            for i, item in enumerate(raw)
        ]
        baseline = _setup_baseline(core, "trunk_angle_from_horizontal_deg")
        self.assertAlmostEqual(baseline, 20.0, places=3)
        mean_of_the_same_values = (20.0 * 4 + 60.0) / 5
        # Not asserting the mean's exact value (that would just restate the arithmetic) -- only
        # that the median result is meaningfully far from what a mean-based helper would have
        # produced, which is the property a `np.median` -> `np.mean` mutant would violate.
        self.assertGreater(abs(baseline - mean_of_the_same_values), 5.0)

    def test_a_contaminated_setup_frame_biases_the_baseline_upward_and_can_mask_a_real_rise(self) -> None:
        """Regression test for `_setup_baseline`'s documented STATED LIMITATION: on a short rep,
        the setup slice (here 2 of 14 frames -- `row_assign_phases`'s
        `max(1, int(14 * 0.15)) == 2`) can already contain a risen frame, before the rule has
        seen a single `peak` frame. This asserts the DIRECTION of the resulting error, not a
        magic number: the contaminated baseline reads HIGHER than the true resting angle, which
        makes `peak - baseline` SMALLER than the true rise. Two consequences are checked
        together, both required by the documented limitation: the bias itself (baseline pulled
        toward the intruding frame) and its behavioral cost (a genuine ~20-degree rise, well
        past the 15-degree fire threshold, going completely undetected) -- i.e. the failure mode
        is a MISSED fault, never a false one.
        """
        from src.pose.movements.base import CoreFrame
        from src.pose.movements.row import (
            ROW_METRIC_KEYS,
            _setup_baseline,
            row_assign_phases,
            row_compute_raw,
            rule_torso_rising,
        )

        # Setup: frame 0 at the true resting angle (20 deg); frame 1 ALREADY risen to 39 deg,
        # simulating a lifter who starts rising before the setup window closes. Peak: 12 frames
        # held at 40 deg -- a genuine 20-degree rise over the TRUE baseline, but only a
        # 10.5-degree rise over the CONTAMINATED one (40 - 29.5), which does not clear 15.
        contaminated_clip = [
            row_frame(trunk_angle_deg=20.0, elbow_angle_deg=170.0, wrist_hip_dist=0.30, frame_index=0),
            row_frame(trunk_angle_deg=39.0, elbow_angle_deg=170.0, wrist_hip_dist=0.30, frame_index=1),
        ] + [
            row_frame(trunk_angle_deg=40.0, elbow_angle_deg=70.0, wrist_hip_dist=0.05, frame_index=i)
            for i in range(2, 14)
        ]
        self.assertEqual(len(contaminated_clip), 14)  # setup_cutoff = max(1, int(14*0.15)) == 2

        raw = row_compute_raw(contaminated_clip, fps=30.0)
        phases = row_assign_phases(raw)
        self.assertEqual(phases[:2], ["setup", "setup"])
        self.assertNotEqual(phases[2], "setup")
        core = [
            CoreFrame(
                frame_index=int(item.get("frame_index", i) or i),
                time=float(item.get("time", 0.0) or 0.0),
                phase=phases[i],
                valid=bool(item.get("valid", False)),
                lower_body_visibility=float(item.get("lower_body_visibility", 0.0) or 0.0),
                metrics={key: float(item.get(key, np.nan)) for key in ROW_METRIC_KEYS},
            )
            for i, item in enumerate(raw)
        ]

        # (1) THE BIAS, DIRECTLY: the contaminated baseline sits strictly between the true
        # resting angle and the intruding frame's value, and strictly above the true angle.
        baseline = _setup_baseline(core, "trunk_angle_from_horizontal_deg")
        self.assertGreater(baseline, 20.0)
        self.assertLess(baseline, 39.0)

        # (2) THE CONSEQUENCE: the same real ~20-degree rise, run through the actual rule, is
        # silently missed once contaminated -- never flagged as a false positive, just absent.
        contaminated_detections = _run_rule(rule_torso_rising, contaminated_clip)
        self.assertEqual(contaminated_detections, [])

        # CONTROL: an otherwise-identical clip whose setup is clean (both frames at the true
        # 20-degree angle). Same real 20-degree rise to the same 40-degree peak, no
        # contamination -- the rule fires as it should, proving the miss above was caused by the
        # contaminated setup frame and not by some other property of the fixture.
        clean_clip = [
            row_frame(trunk_angle_deg=20.0, elbow_angle_deg=170.0, wrist_hip_dist=0.30, frame_index=0),
            row_frame(trunk_angle_deg=20.0, elbow_angle_deg=170.0, wrist_hip_dist=0.30, frame_index=1),
        ] + [
            row_frame(trunk_angle_deg=40.0, elbow_angle_deg=70.0, wrist_hip_dist=0.05, frame_index=i)
            for i in range(2, 14)
        ]
        clean_detections = _run_rule(rule_torso_rising, clean_clip)
        self.assertEqual(len(clean_detections), 1)


class RowIncompleteRomTest(unittest.TestCase):
    def test_a_full_pull_does_not_fire(self) -> None:
        from src.pose.movements.row import rule_incomplete_rom

        clip = _row_clip(peak_wrist_hip=0.05, peak_elbow=70.0)
        self.assertEqual(_run_rule(rule_incomplete_rom, clip), [])

    def test_just_inside_both_thresholds_does_not_fire(self) -> None:
        from src.pose.movements.row import rule_incomplete_rom

        clip = _row_clip(peak_wrist_hip=0.119, peak_elbow=99.0)
        self.assertEqual(_run_rule(rule_incomplete_rom, clip), [])

    def test_a_short_pull_distance_alone_fires(self) -> None:
        from src.pose.movements.row import rule_incomplete_rom

        clip = _row_clip(peak_wrist_hip=0.121, peak_elbow=70.0)
        detections = _run_rule(rule_incomplete_rom, clip)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "row_incomplete_rom")
        self.assertEqual(detections[0].evidence["fired_on"], "pull_distance")

    def test_an_unbent_elbow_alone_fires(self) -> None:
        from src.pose.movements.row import rule_incomplete_rom

        clip = _row_clip(peak_wrist_hip=0.05, peak_elbow=101.0)
        detections = _run_rule(rule_incomplete_rom, clip)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].evidence["fired_on"], "elbow_angle")

    def test_severity_is_exact_at_the_distance_ramp_midpoint(self) -> None:
        from src.pose.movements.row import rule_incomplete_rom

        # Ramp 0.12 -> 0.30; 0.21 is exactly half way.
        clip = _row_clip(peak_wrist_hip=0.21, peak_elbow=70.0)
        detections = _run_rule(rule_incomplete_rom, clip)
        self.assertAlmostEqual(detections[0].severity, 0.5, places=3)

    def test_severity_is_exact_at_the_elbow_ramp_midpoint(self) -> None:
        from src.pose.movements.row import rule_incomplete_rom

        # Ramp 100 -> 140; 120 is exactly half way.
        clip = _row_clip(peak_wrist_hip=0.05, peak_elbow=120.0)
        detections = _run_rule(rule_incomplete_rom, clip)
        self.assertAlmostEqual(detections[0].severity, 0.5, places=3)

    def test_the_worse_of_the_two_conditions_sets_the_severity(self) -> None:
        """Also pins the `primary_*` display fields to the axis that actually drove the
        severity (review Finding 1): an earlier version branched on the categorical `fired_on`
        string, which is `"both"` here regardless of which axis was worse, and always reported
        the distance axis -- silently coaching pull distance when the elbow was the real fault.
        """
        from src.pose.movements.row import PEAK_ELBOW_MILD_DEG, rule_incomplete_rom

        # distance 0.21 -> 0.5; elbow 130 -> 0.75. The larger must win.
        clip = _row_clip(peak_wrist_hip=0.21, peak_elbow=130.0)
        detections = _run_rule(rule_incomplete_rom, clip)
        self.assertAlmostEqual(detections[0].severity, 0.75, places=3)
        self.assertEqual(detections[0].evidence["fired_on"], "both")
        self.assertEqual(detections[0].evidence["primary_label"], "elbow angle at peak")
        self.assertAlmostEqual(detections[0].evidence["primary_value"], 130.0, places=2)
        self.assertEqual(detections[0].evidence["primary_threshold"], PEAK_ELBOW_MILD_DEG)

    def test_the_worse_of_the_two_conditions_sets_the_severity_distance_axis(self) -> None:
        """Mirror of the test above with the two axes' roles swapped: distance is now the WORSE
        axis (severity 0.75) and elbow the milder one (severity 0.5), both still firing
        (`fired_on == "both"`). Without this case, Finding 1's fix is only half-pinned -- a
        version that always reports the distance axis would still pass the elbow-axis test above
        only by accident of which axis that fixture happened to make worse.
        """
        from src.pose.movements.row import PULL_DEPTH_MILD, rule_incomplete_rom

        # distance 0.255 -> 0.75; elbow 120 -> 0.5. The larger (distance) must win.
        clip = _row_clip(peak_wrist_hip=0.255, peak_elbow=120.0)
        detections = _run_rule(rule_incomplete_rom, clip)
        self.assertAlmostEqual(detections[0].severity, 0.75, places=3)
        self.assertEqual(detections[0].evidence["fired_on"], "both")
        self.assertEqual(detections[0].evidence["primary_label"], "wrist-to-hip distance at peak")
        self.assertAlmostEqual(detections[0].evidence["primary_value"], 0.255, places=3)
        self.assertEqual(detections[0].evidence["primary_threshold"], PULL_DEPTH_MILD)

    def test_peak_frame_is_the_worst_frame_even_when_the_ramp_saturates(self) -> None:
        """Regression pin for the unclipped `score_values` fix -- previously pinned by NOTHING.
        Every `_row_clip` peak slice holds a CONSTANT value, so `nanargmax` returns index 0
        whether `score_values` is clipped or unclipped; reverting `rule_incomplete_rom`'s
        `scores` to the old clipped `max(severity_from_range(...), severity_from_range(...))`
        form still passed all 38 tests in this file. `pushup.rule_head_drop`'s own
        `test_peak_frame_is_the_worst_frame_even_when_the_ramp_saturates` is the precedent this
        mirrors.

        The fixture needs a NON-constant peak slice, which `row_frame`/`_row_clip`'s geometry
        cannot produce (every knob controls one metric at one fixed value per call). Built
        directly as `CoreFrame`s instead, bypassing `row_compute_raw`/`row_assign_phases`
        entirely -- `_run_rule` does not smooth (see its own docstring), so these are exactly
        the raw per-frame values `rule_incomplete_rom` sees in production, just without the
        landmark geometry in between.

        Seven `peak`-phase frames; `mean_wrist_hip_dist` = [0.20, 0.35, 0.28, 0.50, 0.22, 0.20,
        0.20], `max_elbow_angle` held at a constant 70 (below the 100 fire threshold, so the
        elbow axis never fires and only the distance axis drives this segment). 0.20/0.28/0.22
        sit below the 0.30 severe end and are NOT saturated; 0.35 (frame_index 1) and 0.50
        (frame_index 3) both clip to severity 1.0. Under the CLIPPED formulation `nanargmax`
        returns the FIRST tied maximum, frame_index 1 -- the wrong answer, since 0.50 is the
        genuinely worse value. Under the UNCLIPPED formulation the two keep climbing past 1.0
        (1.278 vs 2.111), so frame_index 3 wins, which is what this test asserts.
        """
        from src.pose.movements.base import CoreFrame
        from src.pose.movements.row import ROW_METRIC_KEYS, rule_incomplete_rom

        distances = [0.20, 0.35, 0.28, 0.50, 0.22, 0.20, 0.20]
        core = [
            CoreFrame(
                frame_index=index,
                time=index / 30.0,
                phase="peak",
                valid=True,
                lower_body_visibility=1.0,
                metrics={
                    **{key: 0.0 for key in ROW_METRIC_KEYS},
                    "mean_wrist_hip_dist": distance,
                    "max_elbow_angle": 70.0,
                },
            )
            for index, distance in enumerate(distances)
        ]
        ctx = RuleContext(fps=30.0, view_type="rear_oblique", view_confidence=0.8, min_frames=6)
        detections = rule_incomplete_rom(core, ctx)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].peak_frame, 3)

    def test_it_reads_the_less_flexed_arm(self) -> None:
        """The conservative reading: a rep is incomplete if EITHER arm fell short."""
        from src.pose.movements.row import row_compute_raw, rule_incomplete_rom

        # peak_elbow_dy is NEGATIVE here (the brief's draft used +0.06): a positive dy shifts
        # the perturbed (left) elbow DOWN, which on this geometry DECREASES its angle (more
        # flexion), driving max_elbow_angle to the unperturbed ~90 and min_elbow_angle to ~73 --
        # both under 100, which cannot satisfy this test's own assertions. Measured instead:
        # -0.06 leaves the unperturbed right arm at ~90.0 (min_elbow_angle, < 100) and pushes the
        # perturbed left arm to ~111.1 (max_elbow_angle, > 100), which is the geometry this test
        # requires. Only the sign changed; the magnitude (0.06) is untouched.
        clip = _row_clip(peak_wrist_hip=0.05, peak_elbow=90.0, peak_elbow_dy=-0.06)
        raw = row_compute_raw(clip, fps=30.0)
        peak = raw[-1]
        self.assertGreater(peak["max_elbow_angle"], 100.0)
        self.assertLess(peak["min_elbow_angle"], 100.0)
        self.assertEqual(len(_run_rule(rule_incomplete_rom, clip)), 1)


_JERK_PULL_FRAMES = 26
_JERK_TOTAL_FRAMES = 38
_JERK_BURST_AT = 10
_JERK_BURST_WIDTH = 1.2


def _jerk_clip(burst_amplitude: float) -> list[dict]:
    """A smooth bell-shaped pull, optionally with one injected acceleration burst.

    THE BASELINE MUST NOT BE LINEAR, and this was established by measurement, not taste. A
    constant-velocity wrist path has zero acceleration everywhere, so the rule's median over the
    pull is ~1e-5 -- below `_DEGENERATE_ACCEL` -- and `rule_momentum_jerk` returns [] via the
    degenerate guard on EVERY such clip, spike or no spike. A fixture built that way tests the
    guard and nothing else. The cosine-ease travel below gives a half-sine velocity and a
    genuinely nonzero acceleration median, which is what puts the ratio test in play at all.

    THE TWO TEST PATHS SEE DIFFERENT NUMBERS, AND THE ASSERTIONS BELOW ARE PATH-SPECIFIC.
    `_run_rule` builds `CoreFrame`s straight from `row_compute_raw` and does NOT apply
    `run_detector`'s 5-frame median filter; only the one test that goes through `run_detector`
    sees the smoothed series. Measured on both paths, median taken over `pull` frames exactly
    as the rule takes it:

        burst_amp   _run_rule (UNSMOOTHED)                   run_detector (SMOOTHED)
                    ratio   severity  span                   ratio   severity  span
        0.000        1.71     --      silent                  1.56     --      silent
        0.010        2.71     --      silent                  1.64     --      silent
        0.015        4.25    0.279      1 (frame 10)           2.20     --      silent
        0.024        6.60    0.800      3x1 (frames 8/10/12)   3.40    0.090      1
        0.050       14.18    1.000      7                      6.94    0.876      7

    THE 0.024 ROW WAS RE-MEASURED; AN EARLIER DRAFT OF THIS TABLE CLAIMED A SINGLE 5-FRAME
    SPAN THERE, COMPUTED AS (last fired index - first fired index + 1) RATHER THAN AS THREE
    SEPARATE `contiguous_true_segments` RUNS. `_derivative` applied twice is a discrete
    second-difference stencil that samples `x[i-2], x[i], x[i+2]` -- a step of 2 frames -- and
    at this burst's width (sigma 1.2 frames, narrower than the stencil's own spacing) that
    stencil RINGS: the signed acceleration at frames 8/9/10/11/12 is -5.66/+1.93/+9.55/+2.29/
    -4.93, so frames 9 and 11 sit BELOW the 3x threshold (ratio 1.33 and 1.59) while 8, 10 and
    12 sit above it (ratio 3.91, 6.60, 3.40). Three isolated one-frame detections, not one
    five-frame one -- confirmed by manually re-deriving the stencil against the raw
    `wrist_hip_dist` series, independent of `rule_momentum_jerk` itself, so this is a property
    of the FIXTURE-AND-METRIC-LAYER combination, not a rule bug. `severity_from_range` is
    monotonic in the ratio, so the WORST of the three (frame 10, ratio 6.60) is still the
    0.800 severity this docstring and the tests below pin -- taking `max(...)` over the
    resulting detections reproduces the exact number an earlier, wrong, single-detection
    assumption also expected.

    Three facts those numbers carry, all pinned by the tests below:
      - A smooth pull does NOT fire on either path, at any pull speed tried (ratio 1.25-1.71
        for pulls of 10-26 frames). The design spec's "expected to over-fire" worry is not
        borne out on synthetic smooth profiles; it remains untested on real video.
      - The 0.024 burst's THREE fired frames are each span 1, every one SHORTER than
        `ctx.min_frames` (6 at 30 fps). That is the concrete case the event-rule deviation
        exists for -- and passing `ctx.min_frames` here would not shorten one detection, it
        would silence all three.
      - THE MEDIAN FILTER COSTS REAL SENSITIVITY: a burst of 0.012 fires unsmoothed but is
        silent through `run_detector`, and one of 0.024 drops from severity 0.800 to 0.090.
        Emitting the derivative as the metric keeps the transient measurable, which is the
        point of doing it, but it does not preserve its magnitude.
    """
    frames: list[dict] = []
    for index in range(_JERK_TOTAL_FRAMES):
        progress = min(index / _JERK_PULL_FRAMES, 1.0)
        # Cosine ease: velocity is a half-sine, so acceleration is nonzero across the pull.
        travel = 0.25 * (1.0 - math.cos(math.pi * progress)) / 2.0
        distance_value = 0.30 - travel
        if burst_amplitude:
            distance_value -= burst_amplitude * math.exp(
                -(((index - _JERK_BURST_AT) / _JERK_BURST_WIDTH) ** 2)
            )
        frames.append(
            row_frame(
                trunk_angle_deg=20.0,
                wrist_hip_dist=max(distance_value, 0.02),
                elbow_angle_deg=170.0 - 100.0 * progress,
                frame_index=index,
            )
        )
    return frames


class RowMomentumJerkTest(unittest.TestCase):
    def test_a_smooth_controlled_pull_does_not_fire(self) -> None:
        """Specificity on a realistic profile: measured peak/median ratio 1.56, well under 3."""
        from src.pose.movements.row import rule_momentum_jerk

        self.assertEqual(_run_rule(rule_momentum_jerk, _jerk_clip(burst_amplitude=0.0)), [])

    @unittest.skip("ROW_DETECTOR lands in Task 6")
    def test_a_three_frame_spike_survives_the_median_filter_and_fires(self) -> None:
        """The §4.6(b) claim, verified rather than asserted.

        `run_detector` median-filters every metric with a 5-frame window. This test runs the
        FULL detector path -- smoothing included -- so it fails if the derivative-as-metric
        decision is ever reverted to differencing a smoothed position series.
        """
        from src.pose.movements.base import run_detector
        from src.pose.movements.row import ROW_DETECTOR

        result = run_detector(
            ROW_DETECTOR, _jerk_clip(burst_amplitude=0.05), fps=30.0,
            view_type="rear_oblique", view_confidence=0.8, max_reps=None,
        )
        fired = [d for d in result.detections if d.fault_id == "row_momentum_jerk"]
        self.assertEqual(len(fired), 1)

    def test_a_burst_shorter_than_min_frames_still_fires(self) -> None:
        """min_frames is 6 at 30fps; this burst's fired frames are THREE isolated singletons
        (frames 8, 10, 12 -- see `_jerk_clip`'s table and its re-measurement note), each of
        span 1. None may be filtered out.

        This is the concrete case the event-rule deviation exists for: a `contiguous_true_segments`
        call passing `ctx.min_frames` instead of 1 would drop EVERY ONE of these three
        detections -- total silence, not merely a shortened one.
        """
        from src.pose.movements.row import rule_momentum_jerk

        detections = _run_rule(rule_momentum_jerk, _jerk_clip(burst_amplitude=0.024))
        self.assertEqual(len(detections), 3)
        for detection in detections:
            self.assertLess(detection.end_frame - detection.start_frame + 1, 6)

    def test_severity_rises_with_the_ratio(self) -> None:
        from src.pose.movements.row import rule_momentum_jerk

        small = _run_rule(rule_momentum_jerk, _jerk_clip(burst_amplitude=0.015))
        large = _run_rule(rule_momentum_jerk, _jerk_clip(burst_amplitude=0.024))
        self.assertTrue(small and large)
        # `large` is THREE detections (see `_jerk_clip`'s re-measurement note); take the worst
        # of each side rather than index [0], which would depend on segment order rather than
        # magnitude.
        small_severity = max(detection.severity for detection in small)
        large_severity = max(detection.severity for detection in large)
        self.assertLess(small_severity, large_severity)
        # Exact, on the UNSMOOTHED `_run_rule` path (see `_jerk_clip`'s table). Both points are
        # deliberately below saturation -- two clipped 1.0s would pin nothing about the ramp.
        self.assertAlmostEqual(small_severity, 0.279, places=2)
        self.assertAlmostEqual(large_severity, 0.800, places=2)

    def test_a_motionless_window_is_refused_rather_than_maximally_flagged(self) -> None:
        """A zero median makes every ratio infinite; the guard must silence, not fire."""
        from src.pose.movements.row import rule_momentum_jerk

        frames = [
            row_frame(wrist_hip_dist=0.10, elbow_angle_deg=170.0 - 4.0 * i, frame_index=i)
            for i in range(20)
        ]
        self.assertEqual(_run_rule(rule_momentum_jerk, frames), [])

    def test_observability_is_medium_in_every_view(self) -> None:
        from src.pose.movements.row import rule_momentum_jerk

        for view in ("side", "rear", "rear_oblique", "unknown"):
            detections = _run_rule(rule_momentum_jerk, _jerk_clip(burst_amplitude=0.024), view_type=view)
            # Three isolated singleton detections at this amplitude (see `_jerk_clip`'s
            # re-measurement note) -- the observability/confidence contract must hold on ALL
            # of them, not just whichever segment happens to sort first.
            self.assertEqual(len(detections), 3, view)
            for detection in detections:
                self.assertEqual(detection.observability, "medium", view)
                self.assertAlmostEqual(detection.confidence, detection.severity, places=6)

    def test_trunk_heave_is_evidence_and_never_a_fire_condition(self) -> None:
        from src.pose.movements.row import rule_momentum_jerk

        detections = _run_rule(rule_momentum_jerk, _jerk_clip(burst_amplitude=0.024))
        self.assertIn("trunk_heave", detections[0].evidence)
        self.assertIn(detections[0].evidence["trunk_heave"], ("yes", "no"))


if __name__ == "__main__":
    unittest.main()
