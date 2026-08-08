import math
import unittest

import numpy as np

from src.pose.movements.band_pull_apart import (
    BAND_PULL_APART_METRIC_KEYS,
    _clip_facing_sign,
    band_pull_apart_assign_phases,
    band_pull_apart_compute_raw,
    rule_incomplete_rom,
    rule_shrugging,
    rule_trunk_extension_compensation,
)
from src.pose.movements.base import CoreFrame, RuleContext, run_detector


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


def _ctx(view_type: str = "rear", view_confidence: float = 0.8, min_frames: int = 3) -> RuleContext:
    return RuleContext(
        fps=30.0, view_type=view_type, view_confidence=view_confidence, min_frames=min_frames
    )


def _core(frames: list[dict], fps: float = 30.0) -> list[CoreFrame]:
    """Metrics + phases WITHOUT run_detector, so a rule can be tested on an exact fixture.

    run_detector median-filters and re-slices per rep; both are correct in production and both
    would blur a boundary fixture built to sit one step either side of a threshold.
    """
    raw = band_pull_apart_compute_raw(frames, fps=fps)
    phases = band_pull_apart_assign_phases(raw)
    return [
        CoreFrame(
            frame_index=int(item.get("frame_index", i) or i),
            time=float(item.get("time", 0.0) or 0.0),
            phase=phases[i],
            valid=bool(item.get("valid", False)),
            lower_body_visibility=float(item.get("lower_body_visibility", 0.0) or 0.0),
            metrics={k: float(item.get(k, np.nan)) for k in BAND_PULL_APART_METRIC_KEYS},
        )
        for i, item in enumerate(raw)
    ]


def _shrug_rep(peak_gap: float, setup_gap: float = 0.12, n: int = 20) -> list[dict]:
    """A rep whose setup frames hold `setup_gap` and whose widest frames hold `peak_gap`."""
    frames = []
    for i in range(n):
        wide = i >= int(n * 0.4)
        frames.append(
            bpa_frame(
                spread_ratio=1.9 if wide else 0.6,
                shoulder_ear_gap=peak_gap if wide else setup_gap,
                frame_index=i,
            )
        )
    return frames


class ShruggingRuleTest(unittest.TestCase):
    def test_fires_when_the_gap_closes_past_the_spec_threshold(self) -> None:
        # setup 0.12 -> peak 0.08 is a 0.04 closure, past the spec's 0.03.
        core = _core(_shrug_rep(peak_gap=0.08))
        detections = rule_shrugging(core, _ctx())
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "bpa_shrugging")
        self.assertGreater(detections[0].severity, 0.0)

    def test_silent_just_inside_the_threshold(self) -> None:
        # 0.12 -> 0.095 is a 0.025 closure, inside the spec's 0.03.
        core = _core(_shrug_rep(peak_gap=0.095))
        self.assertEqual(rule_shrugging(core, _ctx()), [])

    def test_a_unilateral_shrug_still_fires(self) -> None:
        frames = []
        for i in range(20):
            wide = i >= 8
            frames.append(
                bpa_frame(
                    spread_ratio=1.9 if wide else 0.6,
                    shoulder_ear_gap=0.12,
                    left_gap_delta=-0.05 if wide else 0.0,
                    frame_index=i,
                )
            )
        detections = rule_shrugging(_core(frames), _ctx())
        self.assertEqual(len(detections), 1)

    def test_severity_reaches_one_at_the_ramp_endpoint(self) -> None:
        # 0.12 -> 0.045 is a 0.075 closure, the RULE-LEVEL ramp endpoint.
        core = _core(_shrug_rep(peak_gap=0.045))
        self.assertAlmostEqual(rule_shrugging(core, _ctx())[0].severity, 1.0, places=3)

    def test_the_metric_is_facing_free_so_rear_oblique_is_not_discounted(self) -> None:
        core = _core(_shrug_rep(peak_gap=0.08))
        rear = rule_shrugging(core, _ctx(view_type="rear"))[0]
        oblique = rule_shrugging(core, _ctx(view_type="rear_oblique"))[0]
        self.assertEqual(rear.observability, "high")
        self.assertEqual(oblique.observability, "high")
        self.assertAlmostEqual(rear.confidence, oblique.confidence, places=6)

    def test_a_nan_baseline_silences_the_rule(self) -> None:
        frames = _shrug_rep(peak_gap=0.08)
        for i in range(4):  # blank out every setup frame
            frames[i]["landmarks"][7] = _lm(0.4, 0.2, visibility=0.10)
        self.assertEqual(rule_shrugging(_core(frames), _ctx()), [])


def _rom_rep(spread_ratio: float, elbow_angle_deg: float = 170.0, n: int = 20) -> list[dict]:
    frames = []
    for i in range(n):
        wide = i >= int(n * 0.4)
        frames.append(
            bpa_frame(
                spread_ratio=spread_ratio if wide else 0.6,
                elbow_angle_deg=elbow_angle_deg if wide else 175.0,
                frame_index=i,
            )
        )
    return frames


class IncompleteRomRuleTest(unittest.TestCase):
    def test_fires_when_the_spread_falls_short(self) -> None:
        detections = rule_incomplete_rom(_core(_rom_rep(spread_ratio=1.3)), _ctx())
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "bpa_incomplete_rom")

    def test_silent_at_full_spread_with_straight_arms(self) -> None:
        self.assertEqual(rule_incomplete_rom(_core(_rom_rep(spread_ratio=1.9)), _ctx()), [])

    def test_fires_on_bent_elbows_even_at_full_spread(self) -> None:
        """THE SPEC'S INEQUALITY IS INVERTED AND THIS TEST PINS THE CORRECTION.

        Parent spec line 739 reads `elbow_angle > ~150deg` while its own parenthetical says a
        bent-elbow cheat is the fault -- and >150 degrees is nearly STRAIGHT arms. The
        parenthetical is right. Implemented as `< 150`, so this fixture (140 degrees, full
        spread) MUST fire; under the spec's literal `>` it would be silent.
        """
        detections = rule_incomplete_rom(
            _core(_rom_rep(spread_ratio=1.9, elbow_angle_deg=140.0)), _ctx()
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].evidence["primary_label"], "elbow flexion at peak")

    def test_straight_arms_at_full_spread_do_not_fire(self) -> None:
        self.assertEqual(
            rule_incomplete_rom(_core(_rom_rep(spread_ratio=1.9, elbow_angle_deg=170.0)), _ctx()),
            [],
        )

    def test_spread_severity_reaches_one_at_the_ramp_endpoint(self) -> None:
        detections = rule_incomplete_rom(_core(_rom_rep(spread_ratio=1.0)), _ctx())
        self.assertAlmostEqual(detections[0].severity, 1.0, places=3)

    def test_rear_oblique_downgrades_because_spread_foreshortens(self) -> None:
        core = _core(_rom_rep(spread_ratio=1.3))
        rear = rule_incomplete_rom(core, _ctx(view_type="rear"))[0]
        oblique = rule_incomplete_rom(core, _ctx(view_type="rear_oblique"))[0]
        self.assertEqual(rear.observability, "high")
        self.assertEqual(oblique.observability, "medium")
        self.assertLess(oblique.confidence, rear.confidence)


def _lean_rep(
    peak_lean_deg: float,
    setup_lean_deg: float = 0.0,
    wrist_depth_offset: float = -0.30,
    n: int = 20,
) -> list[dict]:
    frames = []
    for i in range(n):
        wide = i >= int(n * 0.4)
        frames.append(
            bpa_frame(
                spread_ratio=1.9 if wide else 0.6,
                trunk_lean_deg=peak_lean_deg if wide else setup_lean_deg,
                wrist_depth_offset=wrist_depth_offset,
                frame_index=i,
            )
        )
    return frames


def _facing_switch_rep(
    peak_offset: float,
    off_peak_offset: float,
    peak_frames: int = 6,
    off_peak_frames: int = 14,
) -> list[dict]:
    """A rep whose PEAK-phase frames carry `peak_offset` and whose setup/pull frames carry a
    DIFFERENT (here, opposite-signed) `off_peak_offset` -- built to PIN `_clip_facing_sign`'s
    `frame.phase == "peak"` filter as a load-bearing line, not incidental.

    Off-peak frames deliberately OUTNUMBER peak frames (14 vs 6, the default split) so that if
    the filter were ever dropped (falling back to every valid frame) or widened to
    `("pull", "peak")` (still admitting the 11 `pull` frames), the off-peak value's majority
    would dominate the median and flip the returned sign. A fixture where peak frames were the
    majority (as `_lean_rep`'s default 60/40 split is) would NOT catch that regression, because
    the median would still land on the peak value's side by sheer frame count -- that is
    precisely the gap the coordinator's review flagged: `_lean_rep` alone cannot pin this.
    """
    n = off_peak_frames + peak_frames
    frames = []
    for i in range(n):
        wide = i >= off_peak_frames
        frames.append(
            bpa_frame(
                spread_ratio=1.9 if wide else 0.6,
                wrist_depth_offset=peak_offset if wide else off_peak_offset,
                frame_index=i,
            )
        )
    return frames


class FacingDerivationTest(unittest.TestCase):
    def test_negative_offset_means_the_lifter_faces_the_camera(self) -> None:
        # Wrists nearer the camera than the shoulders (MediaPipe z is negative toward camera).
        self.assertEqual(_clip_facing_sign(_core(_lean_rep(0.0, wrist_depth_offset=-0.30))), 1.0)

    def test_positive_offset_means_the_lifter_faces_away(self) -> None:
        self.assertEqual(_clip_facing_sign(_core(_lean_rep(0.0, wrist_depth_offset=0.30))), -1.0)

    def test_all_zero_z_is_undetermined(self) -> None:
        """The RTMPose extraction path writes z=0.0 for EVERY landmark
        (src/pose/rtmpose_pose_extraction.py:121,131), so this is a real runtime, not a
        hypothetical. It must read as undetermined, never as a facing."""
        self.assertTrue(math.isnan(_clip_facing_sign(_core(_lean_rep(0.0, wrist_depth_offset=0.0)))))

    def test_offset_under_the_floor_is_undetermined(self) -> None:
        self.assertTrue(
            math.isnan(_clip_facing_sign(_core(_lean_rep(0.0, wrist_depth_offset=-0.01))))
        )

    def test_the_peak_only_filter_is_load_bearing_not_incidental(self) -> None:
        """Pins the `frame.phase == "peak"` filter directly, closing the gap the coordinator's
        review flagged: every OTHER test in this class holds `wrist_depth_offset` constant
        across all phases, so none of them would fail if the filter were dropped or widened.
        Here the majority off-peak frames (14) carry the OPPOSITE sign from the minority peak
        frames (6); only a genuine peak-only reduction returns the peak frames' sign."""
        core = _core(_facing_switch_rep(peak_offset=-0.30, off_peak_offset=0.30))
        self.assertEqual(_clip_facing_sign(core), 1.0)


class TrunkExtensionRuleTest(unittest.TestCase):
    def test_fires_on_a_backward_lean_past_the_threshold(self) -> None:
        # Facing the camera (offset -0.30) => facing sign +1 => a POSITIVE image lean is
        # backward. 15 degrees clears the spec's 10.
        core = _core(_lean_rep(peak_lean_deg=15.0, wrist_depth_offset=-0.30))
        detections = rule_trunk_extension_compensation(core, _ctx(view_type="rear_oblique"))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "bpa_trunk_extension_compensation")

    def test_a_forward_lean_of_the_same_size_does_not_fire(self) -> None:
        """The whole point of the facing derivation: magnitude alone would fire here."""
        core = _core(_lean_rep(peak_lean_deg=-15.0, wrist_depth_offset=-0.30))
        self.assertEqual(
            rule_trunk_extension_compensation(core, _ctx(view_type="rear_oblique")), []
        )

    def test_the_verdict_inverts_when_the_lifter_faces_away(self) -> None:
        core = _core(_lean_rep(peak_lean_deg=-15.0, wrist_depth_offset=0.30))
        self.assertEqual(
            len(rule_trunk_extension_compensation(core, _ctx(view_type="rear_oblique"))), 1
        )

    def test_silent_just_inside_the_threshold(self) -> None:
        core = _core(_lean_rep(peak_lean_deg=8.0, wrist_depth_offset=-0.30))
        self.assertEqual(
            rule_trunk_extension_compensation(core, _ctx(view_type="rear_oblique")), []
        )

    def test_hard_gated_silent_on_a_confident_rear_label(self) -> None:
        """A signed sagittal lean read from a pure rear view is FRONTAL-plane lateral sway --
        a confident reading of the wrong plane, which no confidence discount can express."""
        core = _core(_lean_rep(peak_lean_deg=15.0, wrist_depth_offset=-0.30))
        self.assertEqual(rule_trunk_extension_compensation(core, _ctx(view_type="rear")), [])

    def test_hard_gated_silent_on_unknown(self) -> None:
        core = _core(_lean_rep(peak_lean_deg=15.0, wrist_depth_offset=-0.30))
        self.assertEqual(rule_trunk_extension_compensation(core, _ctx(view_type="unknown")), [])

    def test_silent_when_the_facing_is_undetermined(self) -> None:
        core = _core(_lean_rep(peak_lean_deg=15.0, wrist_depth_offset=0.0))
        self.assertEqual(
            rule_trunk_extension_compensation(core, _ctx(view_type="rear_oblique")), []
        )

    def test_observability_is_medium_not_high(self) -> None:
        """Downgraded from the spec's `high` because the facing derivation is an unvalidated
        precondition; the observability field should say so."""
        core = _core(_lean_rep(peak_lean_deg=15.0, wrist_depth_offset=-0.30))
        detection = rule_trunk_extension_compensation(core, _ctx(view_type="rear_oblique"))[0]
        self.assertEqual(detection.observability, "medium")

    def test_whip_speed_is_recorded_as_evidence_not_as_a_fire_condition(self) -> None:
        core = _core(_lean_rep(peak_lean_deg=15.0, wrist_depth_offset=-0.30))
        detection = rule_trunk_extension_compensation(core, _ctx(view_type="rear_oblique"))[0]
        self.assertIn("trunk_whip_deg_s", detection.evidence)


if __name__ == "__main__":
    unittest.main()
