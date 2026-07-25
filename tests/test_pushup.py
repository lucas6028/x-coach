import math
import unittest

import numpy as np

from src.pose.movements.pushup import (
    PUSHUP_METRIC_KEYS,
    pushup_assign_phases,
    pushup_compute_raw,
)


def _elbow_xy(
    shoulder_xy: tuple[float, float],
    wrist_xy: tuple[float, float],
    elbow_angle: float,
    side: str,
) -> tuple[float, float]:
    """Compute an elbow landmark position such that the interior angle(shoulder, elbow,
    wrist) equals `elbow_angle` by construction. Places the elbow on the perpendicular
    bisector of the shoulder-wrist segment, offset laterally (left -> -x, right -> +x),
    using d = h / tan(angle/2) where h is half the shoulder-wrist distance.

    Copied verbatim from tests/test_overhead_press.py::_elbow_xy -- the proven construction
    for making the requested elbow angle actually materialise. The result is exact for ANY
    choice of perpendicular direction, so it stays correct even though this fixture's body
    axis runs along image x (see `pushup_frame`), which makes the offset run along the body
    rather than out to the side."""
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


# --- fixture geometry -------------------------------------------------------------------
# A "ladder": the body's long axis runs along image +x (head at small x, feet at large x),
# and the left/right pair separation runs along image y. Shoulders, hips, ankles and ears sit
# on two rails at y = 0.45 and y = 0.55, so with every knob at 0 the ear/shoulder/hip chain is
# exactly collinear on each side (neck angle 0) and the shoulder-mid -> ankle-mid plank line is
# exactly horizontal (body_axis_tilt_deg 0, hip_offset_ratio 0).
#
# DELIBERATELY NON-PHYSICAL: the spec says hip sag is observable from `side` and hand width
# from `front`/`rear` -- no single real camera resolves both, because one collapses whichever
# the other needs. This fixture puts left/right separation and the groundward direction on the
# same image axis so every metric is independently controllable in one frame. It is a test of
# METRIC ARITHMETIC, not a simulation of a real camera. Rule-level view gating is Task 6/7.
_MID_Y = 0.50
_HALF_WIDTH = 0.05          # half the shoulder / hip / ankle / ear pair separation
_SHOULDER_WIDTH = 2 * _HALF_WIDTH
_SHOULDER_X = 0.25
_WRIST_X = 0.32
_HIP_X = 0.55
_ANKLE_X = 0.85
_EAR_X = 0.17
_AXIS_LENGTH = _ANKLE_X - _SHOULDER_X   # 0.60


def _rotate(point: tuple[float, float], degrees: float) -> tuple[float, float]:
    """Rotate a point about the image centre (0.5, 0.5) by `degrees`. Used to prove the
    plank metrics are measured PERPENDICULAR to the body axis rather than vertically in
    image y: a rigid rotation must leave `hip_offset_ratio` untouched while
    `body_axis_tilt_deg` tracks the rotation."""
    if degrees == 0.0:
        return point
    theta = math.radians(degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx, dy = point[0] - 0.5, point[1] - 0.5
    return (0.5 + dx * cos_t - dy * sin_t, 0.5 + dx * sin_t + dy * cos_t)


def pushup_frame(
    elbow_angle: float = 160.0,
    hip_offset: float = 0.0,
    hand_width_ratio: float = 1.0,
    ear_offset: float = 0.0,
    frame_index: int = 0,
    right_elbow_angle: float | None = None,
    tilt_deg: float = 0.0,
    hand_drop: float = 0.10,
) -> dict:
    """Build one push-up frame in which EVERY asserted metric is controlled by construction.

    Knobs (all in normalized image units; y grows DOWNWARD, so +y is groundward here):
      elbow_angle        -- interior shoulder-elbow-wrist angle, both arms (see `_elbow_xy`).
      right_elbow_angle  -- override for the right arm only, so `min_elbow_angle` can be shown
                            to pick the more-flexed side rather than an average.
      hip_offset         -- displacement of BOTH hips along +y. Positive = hips toward the
                            ground = SAG; negative = PIKE. Yields
                            hip_offset_ratio == hip_offset / 0.60 exactly.
      hand_width_ratio   -- wrists are placed at ABSOLUTE lateral coordinates
                            0.50 -/+ ratio * shoulder_width / 2, so the wrist-to-wrist
                            distance is ratio * shoulder_width and the metric equals `ratio`.
                            (Placing them as an offset FROM the shoulders would silently yield
                            ratio + 1 -- the classic fixture-hardcoding bug.)
      ear_offset         -- displacement of BOTH ears along +y, i.e. head dropping toward the
                            floor out of the torso line. 0 => ears collinear with
                            shoulder+hip => neck_line_angle_deg exactly 0.
      tilt_deg           -- rigid rotation of the whole body about the image centre.
      hand_drop          -- displacement of BOTH wrists along +y, i.e. how far the planted
                            hands sit on the groundward side of the plank line. Defaults to
                            a non-zero value because that is anatomically true (the hands are
                            on the floor, the body is above it) AND because it is what makes
                            `hand_offset_ratio` -- the camera-inversion diagnostic -- live
                            rather than degenerate. It moves both wrists equally, so it does
                            not disturb `hand_width_ratio`, and the elbow angles are
                            constructed from whatever wrist position results, so it does not
                            disturb those either.
    """
    right_angle = elbow_angle if right_elbow_angle is None else right_elbow_angle
    half_hand = hand_width_ratio * _SHOULDER_WIDTH / 2.0

    left_shoulder = (_SHOULDER_X, _MID_Y - _HALF_WIDTH)
    right_shoulder = (_SHOULDER_X, _MID_Y + _HALF_WIDTH)
    left_hip = (_HIP_X, _MID_Y - _HALF_WIDTH + hip_offset)
    right_hip = (_HIP_X, _MID_Y + _HALF_WIDTH + hip_offset)
    left_ankle = (_ANKLE_X, _MID_Y - _HALF_WIDTH)
    right_ankle = (_ANKLE_X, _MID_Y + _HALF_WIDTH)
    left_ear = (_EAR_X, _MID_Y - _HALF_WIDTH + ear_offset)
    right_ear = (_EAR_X, _MID_Y + _HALF_WIDTH + ear_offset)
    left_wrist = (_WRIST_X, _MID_Y + hand_drop - half_hand)
    right_wrist = (_WRIST_X, _MID_Y + hand_drop + half_hand)
    left_elbow = _elbow_xy(left_shoulder, left_wrist, elbow_angle, "left")
    right_elbow = _elbow_xy(right_shoulder, right_wrist, right_angle, "right")
    # Knees ride midway down the leg so lower-body landmarks are plausible; no metric reads
    # them, they only feed the framework's lower_body_visibility field.
    left_knee = ((_HIP_X + _ANKLE_X) / 2.0, _MID_Y - _HALF_WIDTH + hip_offset / 2.0)
    right_knee = ((_HIP_X + _ANKLE_X) / 2.0, _MID_Y + _HALF_WIDTH + hip_offset / 2.0)
    nose = (_EAR_X - 0.03, _MID_Y + ear_offset)

    placed = {
        0: nose,
        7: left_ear, 8: right_ear,
        11: left_shoulder, 12: right_shoulder,
        13: left_elbow, 14: right_elbow,
        15: left_wrist, 16: right_wrist,
        23: left_hip, 24: right_hip,
        25: left_knee, 26: right_knee,
        27: left_ankle, 28: right_ankle,
    }

    lm = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(33)]
    for index, point in placed.items():
        x, y = _rotate(point, tilt_deg)
        # z stays 0 everywhere: geometry.angle_degrees is a 3D angle, so a non-zero z would
        # break the by-construction elbow angle.
        lm[index] = {"x": x, "y": y, "z": 0.0, "visibility": 1.0}
    return {"frame_index": frame_index, "landmarks": lm}


class PushupFixtureTests(unittest.TestCase):
    """The fixture's own knobs. If these fail, every assertion below is meaningless."""

    def test_neutral_frame_is_a_straight_horizontal_plank(self) -> None:
        raw = pushup_compute_raw([pushup_frame()], 30.0)[0]
        self.assertTrue(raw["valid"])
        self.assertAlmostEqual(raw["hip_offset_ratio"], 0.0, places=6)
        self.assertAlmostEqual(raw["plank_angle_deviation_deg"], 0.0, places=4)
        self.assertAlmostEqual(raw["body_axis_tilt_deg"], 0.0, places=6)
        # Pins the neck-angle convention for Task 7: NEUTRAL IS 0, not 180. A collinear
        # ear -> shoulder -> hip chain must read 0 deg of deviation.
        self.assertAlmostEqual(raw["neck_line_angle_deg"], 0.0, places=4)

    def test_every_declared_metric_key_is_emitted(self) -> None:
        raw = pushup_compute_raw([pushup_frame()], 30.0)[0]
        for key in PUSHUP_METRIC_KEYS:
            self.assertIn(key, raw, f"missing metric {key}")


class PushupElbowMetricTests(unittest.TestCase):
    def test_min_elbow_angle_tracks_the_requested_angle(self) -> None:
        # The fixture's elbow landmark must be COMPUTED from `elbow_angle`, not hardcoded.
        for requested in (170.0, 140.0, 95.0, 60.0):
            raw = pushup_compute_raw([pushup_frame(elbow_angle=requested)], 30.0)[0]
            self.assertAlmostEqual(raw["min_elbow_angle"], requested, delta=3)
            self.assertAlmostEqual(raw["left_elbow_angle"], requested, delta=3)
            self.assertAlmostEqual(raw["right_elbow_angle"], requested, delta=3)

    def test_min_elbow_angle_takes_the_more_flexed_arm_not_the_mean(self) -> None:
        raw = pushup_compute_raw(
            [pushup_frame(elbow_angle=150.0, right_elbow_angle=80.0)], 30.0
        )[0]
        self.assertAlmostEqual(raw["left_elbow_angle"], 150.0, delta=3)
        self.assertAlmostEqual(raw["right_elbow_angle"], 80.0, delta=3)
        # The mean would be 115; the more-flexed arm is 80.
        self.assertAlmostEqual(raw["min_elbow_angle"], 80.0, delta=3)


class PushupPlankMetricTests(unittest.TestCase):
    def test_hip_offset_ratio_is_positive_for_sag_and_negative_for_pike(self) -> None:
        sag = pushup_compute_raw([pushup_frame(hip_offset=0.06)], 30.0)[0]
        pike = pushup_compute_raw([pushup_frame(hip_offset=-0.06)], 30.0)[0]
        self.assertGreater(sag["hip_offset_ratio"], 0.0)
        self.assertLess(pike["hip_offset_ratio"], 0.0)
        # Exact by construction: offset / (ankle_x - shoulder_x) = 0.06 / 0.60.
        self.assertAlmostEqual(sag["hip_offset_ratio"], 0.1, places=5)
        self.assertAlmostEqual(pike["hip_offset_ratio"], -0.1, places=5)

    def test_hip_offset_sign_does_not_depend_on_which_way_the_subject_faces(self) -> None:
        # Mirroring the body left-to-right (a subject filmed from the other side) flips the
        # raw 2D cross product's sign. The groundward-oriented normal must NOT flip: a sag is
        # still a sag. This is the whole reason the sign is not read off the cross product.
        sag = pushup_frame(hip_offset=0.06)
        mirrored = pushup_frame(hip_offset=0.06)
        for landmark in mirrored["landmarks"]:
            landmark["x"] = 1.0 - landmark["x"]
        forward = pushup_compute_raw([sag], 30.0)[0]
        backward = pushup_compute_raw([mirrored], 30.0)[0]
        self.assertGreater(backward["hip_offset_ratio"], 0.0)
        self.assertAlmostEqual(
            forward["hip_offset_ratio"], backward["hip_offset_ratio"], places=5
        )

    def test_hip_offset_is_perpendicular_to_the_body_axis_not_vertical(self) -> None:
        # Rotate the whole body 20 deg. A vertical-distance implementation would inflate the
        # offset by 1/cos(20 deg) ~ 1.064; a true perpendicular projection is unchanged.
        flat = pushup_compute_raw([pushup_frame(hip_offset=0.06)], 30.0)[0]
        tilted = pushup_compute_raw([pushup_frame(hip_offset=0.06, tilt_deg=20.0)], 30.0)[0]
        self.assertAlmostEqual(tilted["body_axis_tilt_deg"], 20.0, places=4)
        self.assertAlmostEqual(
            tilted["hip_offset_ratio"], flat["hip_offset_ratio"], places=5
        )
        self.assertGreater(tilted["hip_offset_ratio"], 0.0)

    def test_hand_offset_ratio_is_positive_in_a_genuine_pushup(self) -> None:
        # The hands are planted on the floor, so they always sit on the groundward side of
        # the plank line. hand_drop=0.10 over a 0.60 body => +1/6.
        raw = pushup_compute_raw([pushup_frame(hand_drop=0.10)], 30.0)[0]
        self.assertAlmostEqual(raw["hand_offset_ratio"], 0.1 / 0.6, places=5)
        self.assertGreater(raw["hand_offset_ratio"], 0.0)

    def test_camera_inversion_flips_hip_sign_and_hand_offset_catches_it(self) -> None:
        # THE FAILURE `body_axis_tilt_deg` CANNOT SEE. A 180-degree-rotated clip (unapplied
        # rotation metadata, inverted mount) makes every sag read as a pike, while the folded
        # tilt diagnostic reports 0.0 -- the IDEAL plank value. `hand_offset_ratio` is what
        # detects it: the hands cannot be on the sky side of the body in a real push-up.
        upright = pushup_compute_raw([pushup_frame(hip_offset=0.06)], 30.0)[0]
        inverted = pushup_compute_raw(
            [pushup_frame(hip_offset=0.06, tilt_deg=180.0)], 30.0
        )[0]

        # The sign really does invert, and the tilt diagnostic really is blind to it.
        self.assertGreater(upright["hip_offset_ratio"], 0.0)
        self.assertLess(inverted["hip_offset_ratio"], 0.0)
        self.assertAlmostEqual(upright["body_axis_tilt_deg"], 0.0, places=4)
        self.assertAlmostEqual(inverted["body_axis_tilt_deg"], 0.0, places=4)

        # The emitted guard is not blind.
        self.assertGreater(upright["hand_offset_ratio"], 0.0)
        self.assertLess(inverted["hand_offset_ratio"], 0.0)

    def test_hand_offset_stays_positive_when_the_subject_merely_faces_the_other_way(
        self,
    ) -> None:
        # The case a SIGNED body-axis angle confuses with inversion: mirroring the subject
        # left-to-right gives the same +180 deg axis angle as a rotated camera, but the sign
        # of hip_offset_ratio is CORRECT here. The hand guard must not cry wolf.
        mirrored = pushup_frame(hip_offset=0.06)
        for landmark in mirrored["landmarks"]:
            landmark["x"] = 1.0 - landmark["x"]
        raw = pushup_compute_raw([mirrored], 30.0)[0]
        self.assertGreater(raw["hip_offset_ratio"], 0.0)
        self.assertGreater(raw["hand_offset_ratio"], 0.0)

    def test_body_axis_tilt_is_zero_for_a_horizontal_body(self) -> None:
        raw = pushup_compute_raw([pushup_frame()], 30.0)[0]
        self.assertAlmostEqual(raw["body_axis_tilt_deg"], 0.0, places=6)
        # ... and is folded to [0, 90], so it does not depend on facing direction.
        for tilt in (25.0, -25.0):
            tilted = pushup_compute_raw([pushup_frame(tilt_deg=tilt)], 30.0)[0]
            self.assertAlmostEqual(tilted["body_axis_tilt_deg"], 25.0, places=4)

    def test_plank_angle_deviation_grows_with_the_offset_and_is_unsigned(self) -> None:
        straight = pushup_compute_raw([pushup_frame()], 30.0)[0]
        sag = pushup_compute_raw([pushup_frame(hip_offset=0.06)], 30.0)[0]
        pike = pushup_compute_raw([pushup_frame(hip_offset=-0.06)], 30.0)[0]
        self.assertAlmostEqual(straight["plank_angle_deviation_deg"], 0.0, places=4)
        self.assertGreater(sag["plank_angle_deviation_deg"], 0.0)
        # abs(180 - angle) cannot distinguish sag from pike; hip_offset_ratio is what carries
        # the direction. Symmetric offsets must therefore give the same deviation.
        self.assertAlmostEqual(
            sag["plank_angle_deviation_deg"], pike["plank_angle_deviation_deg"], places=4
        )


class PushupHandWidthTests(unittest.TestCase):
    def test_hand_width_ratio_equals_the_requested_ratio(self) -> None:
        for requested in (0.8, 1.0, 1.6, 2.2):
            raw = pushup_compute_raw(
                [pushup_frame(hand_width_ratio=requested)], 30.0
            )[0]
            self.assertAlmostEqual(raw["hand_width_ratio"], requested, places=5)

    def test_hand_width_ratio_is_scale_free(self) -> None:
        # Rigid rotation preserves both distances, so the ratio must not move.
        raw = pushup_compute_raw(
            [pushup_frame(hand_width_ratio=1.6, tilt_deg=35.0)], 30.0
        )[0]
        self.assertAlmostEqual(raw["hand_width_ratio"], 1.6, places=5)


class PushupNeckLineTests(unittest.TestCase):
    def test_neck_line_angle_grows_as_the_head_leaves_the_torso_line(self) -> None:
        neutral = pushup_compute_raw([pushup_frame()], 30.0)[0]
        dropped = pushup_compute_raw([pushup_frame(ear_offset=0.03)], 30.0)[0]
        self.assertAlmostEqual(neutral["neck_line_angle_deg"], 0.0, places=4)
        # Exact by construction: ear sits 0.08 ahead of the shoulder along the body axis and
        # 0.03 groundward of it, against a torso vector that runs straight down the axis.
        expected = math.degrees(math.atan2(0.03, _SHOULDER_X - _EAR_X))
        self.assertAlmostEqual(dropped["neck_line_angle_deg"], expected, places=4)

    def test_hip_sag_does_not_manufacture_neck_deviation(self) -> None:
        # REGRESSION for the shoulder->hip chord reference. With the head perfectly on the
        # body line (ear_offset=0), referencing the neck angle to the shoulder->hip chord
        # made a sagging hip rotate the reference and invent neck deviation:
        #     hip_offset_ratio 0.067 -> 7.595 deg,  0.100 -> 11.310 deg,  0.150 -> 16.699 deg
        # against 20.556 deg for a genuine head drop -- so a 0.10 sag forged 55% of a full
        # head-drop signal, growing through the descent exactly as the sag does. Referencing
        # to the BODY AXIS (which a hip sag does not move) decouples them.
        for hip_offset in (0.0, 0.04, 0.06, 0.09, 0.15):
            raw = pushup_compute_raw([pushup_frame(hip_offset=hip_offset)], 30.0)[0]
            self.assertAlmostEqual(
                raw["neck_line_angle_deg"], 0.0, places=4,
                msg=f"sag of {hip_offset} manufactured neck deviation",
            )

    def test_neck_line_angle_is_unchanged_by_a_simultaneous_sag(self) -> None:
        # The head-drop reading must be the SAME whether or not the hips also sag, so Task 7
        # never sees hip_sag and head_drop as mutually corroborating.
        alone = pushup_compute_raw([pushup_frame(ear_offset=0.03)], 30.0)[0]
        with_sag = pushup_compute_raw(
            [pushup_frame(ear_offset=0.03, hip_offset=0.06)], 30.0
        )[0]
        self.assertAlmostEqual(
            alone["neck_line_angle_deg"], with_sag["neck_line_angle_deg"], places=4
        )
        # ... and the sag is still detected, by the metric that owns it.
        self.assertGreater(with_sag["hip_offset_ratio"], 0.0)

    def test_neck_line_angle_is_unsigned(self) -> None:
        # Documented limitation: a head LIFTED off the torso line reads the same as a head
        # DROPPED toward the floor. Task 7 owns recovering the direction.
        dropped = pushup_compute_raw([pushup_frame(ear_offset=0.03)], 30.0)[0]
        lifted = pushup_compute_raw([pushup_frame(ear_offset=-0.03)], 30.0)[0]
        self.assertAlmostEqual(
            dropped["neck_line_angle_deg"], lifted["neck_line_angle_deg"], places=4
        )

    def test_neck_line_angle_survives_a_sagittal_single_ear_view(self) -> None:
        # The primary push-up view hides the far ear. A per-side computation must still read
        # the near side rather than going NaN on a missing ear MIDPOINT.
        frame = pushup_frame(ear_offset=0.03)
        frame["landmarks"][8]["visibility"] = 0.0     # far (right) ear occluded
        raw = pushup_compute_raw([frame], 30.0)[0]
        self.assertTrue(raw["valid"])
        expected = math.degrees(math.atan2(0.03, _SHOULDER_X - _EAR_X))
        self.assertAlmostEqual(raw["neck_line_angle_deg"], expected, places=4)


class PushupUnmeasurableInputTests(unittest.TestCase):
    """The module's central promise: an unmeasurable metric comes back NaN (or the frame
    comes back with no metrics at all) rather than as a plausible-but-wrong number.

    This is the failure mode measured in `view_estimation._visible_midpoint`, where one
    occluded shoulder made the body-axis extent read 0.070 instead of ~0.60 -- 8.6x low, with
    no NaN and no other signal. Every plank metric here is built on left/right midpoints and
    is exposed to exactly that. NOTE the gate exercised here is
    `geometry.VISIBILITY_THRESHOLD` (0.50), which is a DIFFERENT gate from
    `view_estimation`'s 0.35 midpoint gate."""

    def test_occluded_ear_nans_only_the_neck_metric(self) -> None:
        # Ears are deliberately NOT in `required`, so this is the per-metric NaN path -- the
        # one the frame-level validity gate does not cover.
        frame = pushup_frame(elbow_angle=95.0, hip_offset=0.06, ear_offset=0.03)
        frame["landmarks"][7]["visibility"] = 0.0
        frame["landmarks"][8]["visibility"] = 0.0
        raw = pushup_compute_raw([frame], 30.0)[0]
        self.assertTrue(raw["valid"])
        self.assertFalse(np.isfinite(raw["neck_line_angle_deg"]))
        # ... and nothing else is disturbed.
        self.assertAlmostEqual(raw["min_elbow_angle"], 95.0, delta=3)
        self.assertAlmostEqual(raw["hip_offset_ratio"], 0.1, places=5)
        self.assertAlmostEqual(raw["hand_width_ratio"], 1.0, places=5)
        self.assertAlmostEqual(raw["body_axis_tilt_deg"], 0.0, places=6)

    def test_occluded_wrist_invalidates_the_frame(self) -> None:
        # Wrists are in `required`, so a wrist dropped below the visibility gate does not
        # leave one arm's angle NaN inside an otherwise-valid frame -- it invalidates the
        # frame outright, and nothing is silently computed from the surviving arm.
        frame = pushup_frame(elbow_angle=90.0)
        frame["landmarks"][16]["visibility"] = 0.0
        raw = pushup_compute_raw([frame], 30.0)[0]
        self.assertFalse(raw["valid"])
        self.assertNotIn("min_elbow_angle", raw)

    def test_degenerate_elbow_geometry_nans_only_the_elbow_metrics(self) -> None:
        # The one route to a NaN elbow angle inside a VALID frame: the elbow landmark
        # coinciding with the wrist, which makes geometry.angle_degrees' denominator
        # degenerate. All three landmarks stay visible, so the frame is still valid -- and
        # `min_elbow_angle` must come back NaN rather than fall back to some other number,
        # while the plank metrics (which do not depend on the elbows) stay finite.
        frame = pushup_frame(elbow_angle=90.0, hip_offset=0.06)
        for elbow, wrist in ((13, 15), (14, 16)):
            frame["landmarks"][elbow] = dict(frame["landmarks"][wrist])
        raw = pushup_compute_raw([frame], 30.0)[0]
        self.assertTrue(raw["valid"])
        self.assertFalse(np.isfinite(raw["left_elbow_angle"]))
        self.assertFalse(np.isfinite(raw["right_elbow_angle"]))
        self.assertFalse(np.isfinite(raw["min_elbow_angle"]))
        self.assertAlmostEqual(raw["hip_offset_ratio"], 0.1, places=5)
        self.assertAlmostEqual(raw["hand_width_ratio"], 1.0, places=5)

    def test_occluded_far_shoulder_yields_no_metrics_not_a_wrong_plank_line(self) -> None:
        loud = pushup_compute_raw([pushup_frame(hip_offset=0.06)], 30.0)[0]
        self.assertAlmostEqual(loud["hip_offset_ratio"], 0.1, places=5)

        frame = pushup_frame(hip_offset=0.06)
        frame["landmarks"][12]["visibility"] = 0.0    # far (right) shoulder occluded
        raw = pushup_compute_raw([frame], 30.0)[0]
        self.assertFalse(raw["valid"])
        # No stale/degraded metric survives on an invalid frame: a consumer reading these via
        # CoreFrame.m() gets NaN, never the 8.6x-low number a half-visible midpoint produces.
        for key in PUSHUP_METRIC_KEYS:
            self.assertNotIn(key, raw, f"invalid frame leaked metric {key}")

    def test_cropped_at_the_knees_invalidates_every_frame(self) -> None:
        # Ankles are required because the plank line needs them, so a clip framed from the
        # knees up silences ALL push-up rules, not just the plank ones. Documented at module
        # level; pinned here so it cannot regress into a silent partial answer.
        frames = [pushup_frame(hip_offset=0.06, frame_index=i) for i in range(10)]
        for frame in frames:
            frame["landmarks"][27]["visibility"] = 0.0
            frame["landmarks"][28]["visibility"] = 0.0
        raw = pushup_compute_raw(frames, 30.0)
        self.assertTrue(all(not item["valid"] for item in raw))
        for item in raw:
            self.assertNotIn("hip_offset_ratio", item)

    def test_invalid_frames_still_carry_index_time_and_visibility(self) -> None:
        frame = pushup_frame(frame_index=7)
        frame["landmarks"][27]["visibility"] = 0.0
        raw = pushup_compute_raw([frame], 30.0)[0]
        self.assertEqual(raw["frame_index"], 7)
        self.assertAlmostEqual(raw["time"], 7 / 30.0, places=6)
        self.assertIn("lower_body_visibility", raw)

    def test_visible_but_misplaced_landmark_is_trusted(self) -> None:
        """KNOWN GAP, pinned rather than claimed away. The "NaN, never a degraded number"
        guarantee covers landmark DROP-OUT only. `geometry.visible_point` trusts any landmark
        at visibility >= 0.50 completely, so one that MediaPipe reports confidently but places
        wrongly yields a finite, wrong metric with no signal at all. This matters because
        MediaPipe routinely gives mid-range visibility to HALLUCINATED far-side landmarks in
        exactly the sagittal view this module calls primary.

        Inherited from the shared gate, not introduced by this module, and not fixable here
        without inventing a plausibility threshold (out of scope). Asserted so the limit is
        documented behaviour: if someone later narrows it, this test tells them what changed."""
        # Just BELOW the gate: refused, as advertised.
        below = pushup_frame(hip_offset=0.06)
        below["landmarks"][12]["visibility"] = 0.49
        below["landmarks"][12]["x"] += 0.20
        self.assertFalse(pushup_compute_raw([below], 30.0)[0]["valid"])

        # Just ABOVE it: trusted, and wrong, with no NaN anywhere.
        above = pushup_frame(hip_offset=0.06)
        above["landmarks"][12]["visibility"] = 0.55
        above["landmarks"][12]["x"] += 0.20
        raw = pushup_compute_raw([above], 30.0)[0]
        self.assertTrue(raw["valid"])
        self.assertTrue(np.isfinite(raw["hand_width_ratio"]))
        self.assertAlmostEqual(raw["hand_width_ratio"], 0.44721, places=4)   # truth is 1.0
        self.assertAlmostEqual(raw["hip_offset_ratio"], 0.12, places=4)      # truth is 0.10

    def test_non_dict_frame_is_invalid(self) -> None:
        raw = pushup_compute_raw([None, "not a frame"], 30.0)
        self.assertEqual([item["valid"] for item in raw], [False, False])


class PushupPhaseTests(unittest.TestCase):
    @staticmethod
    def _rep() -> list[dict]:
        """A full rep: 4 frames at the top, 6 descending, 4 at the bottom, 6 ascending,
        4 back at the top."""
        angles = (
            [170.0] * 4
            + [155.0, 140.0, 125.0, 110.0, 100.0, 95.0]
            + [85.0] * 4
            + [95.0, 105.0, 120.0, 135.0, 150.0, 160.0]
            + [170.0] * 4
        )
        return [
            pushup_frame(elbow_angle=angle, frame_index=i) for i, angle in enumerate(angles)
        ]

    def test_phase_sequence_covers_setup_descent_bottom_ascent(self) -> None:
        phases = pushup_assign_phases(pushup_compute_raw(self._rep(), 30.0))
        self.assertEqual(len(phases), 24)
        self.assertEqual(set(phases), {"setup", "descent", "bottom", "ascent"})
        # 15% of 24 frames = 3 opening frames labelled setup.
        self.assertEqual(phases[:3], ["setup", "setup", "setup"])
        # The deepest frames are the bottom, and descent strictly precedes ascent.
        bottom_indices = [i for i, p in enumerate(phases) if p == "bottom"]
        descent_indices = [i for i, p in enumerate(phases) if p == "descent"]
        ascent_indices = [i for i, p in enumerate(phases) if p == "ascent"]
        self.assertTrue(all(i < min(bottom_indices) for i in descent_indices))
        self.assertTrue(all(i > min(bottom_indices) for i in ascent_indices))

    def test_setup_window_is_the_first_fifteen_percent(self) -> None:
        # Pins the FRACTION, not just "the clip opens with some setup frames". The earlier
        # `phases[:3] == [setup]*3` assertion left 0.25 free (it also makes frames 0-2 setup),
        # so the constant was unpinned in the direction it was most likely to drift.
        # `setup` is only ever assigned in the index < cutoff branch, so for an all-valid clip
        # the count IS the cutoff.
        for frame_count, expected in ((24, 3), (40, 6), (13, 1)):
            angles = [170.0 - 80.0 * math.sin(math.pi * i / (frame_count - 1))
                      for i in range(frame_count)]
            frames = [pushup_frame(elbow_angle=a, frame_index=i)
                      for i, a in enumerate(angles)]
            phases = pushup_assign_phases(pushup_compute_raw(frames, 30.0))
            self.assertEqual(
                phases.count("setup"), expected,
                msg=f"n={frame_count}: expected int(n*0.15)={expected} setup frames",
            )
            # ... and they are the LEADING frames, contiguously.
            self.assertEqual(phases[:expected], ["setup"] * expected)
            self.assertNotEqual(phases[expected], "setup")

    def test_bottom_is_the_deepest_thirty_percent(self) -> None:
        raw = pushup_compute_raw(self._rep(), 30.0)
        phases = pushup_assign_phases(raw)
        threshold = float(
            np.percentile([item["min_elbow_angle"] for item in raw], 30)
        )
        for phase, item in zip(phases, raw):
            if phase == "bottom":
                self.assertLessEqual(item["min_elbow_angle"], threshold + 1e-6)

    def test_empty_clip_returns_empty_phases(self) -> None:
        self.assertEqual(pushup_assign_phases([]), [])

    def test_clip_with_no_depth_signal_is_entirely_unknown(self) -> None:
        frames = [pushup_frame(frame_index=i) for i in range(8)]
        for frame in frames:
            frame["landmarks"][27]["visibility"] = 0.0
        phases = pushup_assign_phases(pushup_compute_raw(frames, 30.0))
        self.assertEqual(phases, ["unknown"] * 8)

    def test_invalid_frame_inside_the_setup_window_is_unknown_not_setup(self) -> None:
        # The validity check must precede the setup cutoff, mirroring ohp_assign_phases.
        frames = self._rep()
        frames[0]["landmarks"][27]["visibility"] = 0.0
        phases = pushup_assign_phases(pushup_compute_raw(frames, 30.0))
        self.assertEqual(phases[0], "unknown")
        self.assertEqual(phases[1], "setup")


if __name__ == "__main__":
    unittest.main()
