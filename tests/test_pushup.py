import math
import unittest

import numpy as np

from src.pose.movements.base import CoreFrame, MovementDetector, RuleContext, run_detector
from src.pose.movements.pushup import (
    PUSHUP_METRIC_KEYS,
    pushup_assign_phases,
    pushup_compute_raw,
    rule_elbow_flare,
    rule_head_drop,
    rule_hip_sag,
    rule_scapular_winging,
    rule_shallow_depth,
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


def _rotate_about(
    point: tuple[float, float], pivot: tuple[float, float], radians: float
) -> tuple[float, float]:
    """Rotate `point` about `pivot` by `radians`. Used by the `head_follows="chord"` knob to
    swing the head with the torso segment instead of translating it."""
    cos_t, sin_t = math.cos(radians), math.sin(radians)
    dx, dy = point[0] - pivot[0], point[1] - pivot[1]
    return (pivot[0] + dx * cos_t - dy * sin_t, pivot[1] + dx * sin_t + dy * cos_t)


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
    head_follows: str = "axis",
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
                            disturb those either. `hand_drop=0.0` reproduces the original
                            round-0 geometry exactly, for diffing behaviour against commits
                            ed40e087 / 20c1204a.
      head_follows       -- WHICH SEGMENT THE HEAD IS MODELLED AS RIDING ON, when the hips
                            sag. This is a modeling assumption, not a fault knob, and neither
                            setting is "the correct one" -- see `_neck_line_angle`'s docstring
                            and `test_neck_reference_assumption_is_visible`.
                              "axis"  (default) the head keeps its orientation relative to the
                                      shoulder-mid -> ankle-mid body axis; `ear_offset` simply
                                      translates it. This is what `neck_line_angle_deg`
                                      assumes, so a pure sag reads 0.
                              "chord" the head rotates WITH the shoulder->hip torso segment as
                                      the hips drop. Under this model the axis-referenced
                                      metric reports sag-proportional deviation with no head
                                      fault present.
    """
    if head_follows not in {"axis", "chord"}:
        raise ValueError(f"head_follows must be 'axis' or 'chord', got {head_follows!r}")
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
    if head_follows == "chord":
        # Swing each ear about its own shoulder by exactly the angle the shoulder->hip chord
        # rotated through when the hips dropped, so the head stays neutral to the TORSO
        # SEGMENT rather than to the body axis.
        chord_rotation = math.atan2(hip_offset, _HIP_X - _SHOULDER_X)
        left_ear = _rotate_about(left_ear, left_shoulder, chord_rotation)
        right_ear = _rotate_about(right_ear, right_shoulder, chord_rotation)
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

    def test_offsets_track_each_frame_independently_across_a_clip(self) -> None:
        # `hip_offset_ratio` / `hand_offset_ratio` are computed by a closure defined inside
        # the per-frame loop, capturing that frame's normal, shoulder_mid and axis_length.
        # Every other offset test uses a single frame or a constant offset, so nothing else
        # would notice if the closure leaked one frame's geometry into another's. Here the
        # sag CHANGES every frame and swings through zero into a pike.
        offsets = [0.06, 0.03, 0.0, -0.03, -0.06, 0.09]
        frames = [pushup_frame(hip_offset=o, frame_index=i) for i, o in enumerate(offsets)]
        raw = pushup_compute_raw(frames, 30.0)
        self.assertEqual(len(raw), len(offsets))
        for item, offset in zip(raw, offsets):
            self.assertAlmostEqual(item["hip_offset_ratio"], offset / 0.60, places=5)
            # The hands do not move, so their offset must be identical on every frame.
            self.assertAlmostEqual(item["hand_offset_ratio"], 0.1 / 0.6, places=5)

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
    def test_neck_line_angle_grows_as_the_head_leaves_the_body_axis(self) -> None:
        neutral = pushup_compute_raw([pushup_frame()], 30.0)[0]
        dropped = pushup_compute_raw([pushup_frame(ear_offset=0.03)], 30.0)[0]
        self.assertAlmostEqual(neutral["neck_line_angle_deg"], 0.0, places=4)
        # Exact by construction: the ear sits 0.08 along the BODY AXIS from the shoulder and
        # 0.03 off it, so the angle is atan2(0.03, 0.08). NOTE this expected value coincides
        # with the old shoulder->hip-chord reference only because hip_offset is 0 here, which
        # makes chord and axis the same line; `test_hip_sag_does_not_manufacture_neck_deviation`
        # is what pins the two apart.
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

    def test_neck_reference_assumption_is_visible(self) -> None:
        """DOCUMENTS a modeling assumption; does NOT assert it is correct.

        `neck_line_angle_deg` references the head to the BODY AXIS, which decouples it from
        hip sag EXACTLY ONLY IF the head stays neutral to that axis. If a lifter's head
        instead rides the shoulder->hip TORSO SEGMENT, rotating with it as the hips drop, the
        same metric reports sag-proportional deviation with no head fault present -- the same
        magnitudes the chord reference used to invent, just penalising the opposite posture.
        Which model is right is an EMPIRICAL question and no data in this repo can settle it;
        both branches below are therefore recorded as behaviour, not as pass/fail on realism.
        See `_neck_line_angle`'s docstring for the segment-kinematics argument for `axis`."""
        # Head neutral to the AXIS (what the metric assumes): a sag is invisible to the neck.
        for hip_offset in (0.06, 0.09, 0.15):
            axis_raw = pushup_compute_raw(
                [pushup_frame(hip_offset=hip_offset, head_follows="axis")], 30.0
            )[0]
            self.assertAlmostEqual(axis_raw["neck_line_angle_deg"], 0.0, places=4)

        # Head neutral to the CHORD: the metric reads deviation that is purely the sag.
        # Exact by construction -- atan2(hip_offset, hip_x - shoulder_x) = atan2(sag, 0.30).
        for hip_offset in (0.06, 0.09, 0.15):
            chord_raw = pushup_compute_raw(
                [pushup_frame(hip_offset=hip_offset, head_follows="chord")], 30.0
            )[0]
            expected = math.degrees(math.atan2(hip_offset, _HIP_X - _SHOULDER_X))
            self.assertAlmostEqual(chord_raw["neck_line_angle_deg"], expected, places=4)
        # Pinned as literals too, so the magnitudes stay legible to a future reader.
        self.assertAlmostEqual(
            pushup_compute_raw(
                [pushup_frame(hip_offset=0.06, head_follows="chord")], 30.0
            )[0]["neck_line_angle_deg"], 11.30993, places=4)
        self.assertAlmostEqual(
            pushup_compute_raw(
                [pushup_frame(hip_offset=0.15, head_follows="chord")], 30.0
            )[0]["neck_line_angle_deg"], 26.56505, places=4)

        # With NO sag the two models are indistinguishable -- the chord IS the axis.
        for follows in ("axis", "chord"):
            straight = pushup_compute_raw(
                [pushup_frame(hip_offset=0.0, head_follows=follows)], 30.0
            )[0]
            self.assertAlmostEqual(straight["neck_line_angle_deg"], 0.0, places=4)

    def test_neck_line_angle_is_nan_when_the_body_axis_degenerates(self) -> None:
        # Behaviour CHANGE from the chord reference, previously untested: with the ankles
        # collapsed onto the shoulders there is no body axis, so the axis-referenced neck
        # angle is NaN where the old chord version still returned a finite number. Refusing is
        # the right call -- with no axis there is no line for the head to deviate from -- but
        # it is a change, so it is pinned rather than left implicit.
        frame = pushup_frame(ear_offset=0.03)
        for ankle, shoulder in ((27, 11), (28, 12)):
            frame["landmarks"][ankle] = dict(frame["landmarks"][shoulder])
        raw = pushup_compute_raw([frame], 30.0)[0]
        self.assertTrue(raw["valid"])
        self.assertFalse(np.isfinite(raw["neck_line_angle_deg"]))
        # The other axis-dependent metrics go with it, for the same reason.
        self.assertFalse(np.isfinite(raw["hip_offset_ratio"]))
        self.assertFalse(np.isfinite(raw["hand_offset_ratio"]))
        self.assertFalse(np.isfinite(raw["body_axis_tilt_deg"]))
        # Metrics that do not need the axis are unaffected.
        self.assertAlmostEqual(raw["hand_width_ratio"], 1.0, places=5)

    def test_neck_line_angle_is_unsigned(self) -> None:
        # Documented limitation of THIS metric: a head LIFTED off the torso line reads the same
        # as a head DROPPED toward the floor. `neck_line_signed_deg` is the directional
        # companion that recovers it (see PushupSignedNeckLineTests); this one is kept unsigned
        # because it needs no groundward reference and so survives a rolled camera.
        dropped = pushup_compute_raw([pushup_frame(ear_offset=0.03)], 30.0)[0]
        lifted = pushup_compute_raw([pushup_frame(ear_offset=-0.03)], 30.0)[0]
        self.assertAlmostEqual(
            dropped["neck_line_angle_deg"], lifted["neck_line_angle_deg"], places=4
        )
        # ... and the signed companion does NOT confuse them.
        self.assertGreater(dropped["neck_line_signed_deg"], 0.0)
        self.assertLess(lifted["neck_line_signed_deg"], 0.0)

    def test_neck_line_angle_survives_a_sagittal_single_ear_view(self) -> None:
        # The primary push-up view hides the far ear. A per-side computation must still read
        # the near side rather than going NaN on a missing ear MIDPOINT.
        frame = pushup_frame(ear_offset=0.03)
        frame["landmarks"][8]["visibility"] = 0.0     # far (right) ear occluded
        raw = pushup_compute_raw([frame], 30.0)[0]
        self.assertTrue(raw["valid"])
        expected = math.degrees(math.atan2(0.03, _SHOULDER_X - _EAR_X))
        self.assertAlmostEqual(raw["neck_line_angle_deg"], expected, places=4)


class PushupSignedNeckLineTests(unittest.TestCase):
    """`neck_line_signed_deg`: same magnitude as `neck_line_angle_deg`, plus the direction.
    It exists because a per-clip BASELINE on an unsigned angle is actively inverted -- with a
    baseline of +5, a head lifted to -15 reads unsigned 15, i.e. deviation +10, and a head LIFT
    is reported as a DROP. See `_signed_neck_line_angle`."""

    def test_positive_is_a_drop_and_negative_is_a_lift(self) -> None:
        expected = math.degrees(math.atan2(0.03, _SHOULDER_X - _EAR_X))
        dropped = pushup_compute_raw([pushup_frame(ear_offset=0.03)], 30.0)[0]
        lifted = pushup_compute_raw([pushup_frame(ear_offset=-0.03)], 30.0)[0]
        neutral = pushup_compute_raw([pushup_frame(ear_offset=0.0)], 30.0)[0]
        self.assertAlmostEqual(dropped["neck_line_signed_deg"], expected, places=4)
        self.assertAlmostEqual(lifted["neck_line_signed_deg"], -expected, places=4)
        self.assertAlmostEqual(neutral["neck_line_signed_deg"], 0.0, places=4)

    def test_magnitude_matches_the_unsigned_angle_on_a_symmetric_fixture(self) -> None:
        # SYMMETRIC FIXTURE ON PURPOSE. Both metrics average their two sides with `mean_finite`,
        # so the identity |signed| == unsigned holds per SIDE but can break across sides: a clip
        # whose left and right readings disagree in sign averages toward 0 signed while the
        # unsigned mean stays large. This fixture moves both ears together, so the identity
        # holds; asserting it on an asymmetric one would be asserting something false.
        for ear_offset in (0.05, 0.02, 0.0, -0.02, -0.05):
            raw = pushup_compute_raw([pushup_frame(ear_offset=ear_offset)], 30.0)[0]
            self.assertAlmostEqual(
                abs(raw["neck_line_signed_deg"]), raw["neck_line_angle_deg"], places=4,
                msg=f"ear_offset {ear_offset}",
            )

    def test_camera_inversion_flips_the_sign_and_the_hand_guard_catches_it(self) -> None:
        # The sign rides on the SAME groundward normal as `hip_offset_ratio`, so it inherits
        # the same exposure: a 180-degree-rotated clip turns a genuine head DROP into a
        # confident head LIFT. `hand_offset_ratio` is what detects it, which is why
        # `rule_head_drop` carries `rule_hip_sag`'s guard verbatim.
        upright = pushup_compute_raw([pushup_frame(ear_offset=0.03)], 30.0)[0]
        inverted = pushup_compute_raw([pushup_frame(ear_offset=0.03, tilt_deg=180.0)], 30.0)[0]
        self.assertGreater(upright["neck_line_signed_deg"], 0.0)
        self.assertLess(inverted["neck_line_signed_deg"], 0.0)
        # The unsigned metric is blind to the inversion -- it is the same number either way.
        self.assertAlmostEqual(
            upright["neck_line_angle_deg"], inverted["neck_line_angle_deg"], places=4
        )
        self.assertGreater(upright["hand_offset_ratio"], 0.0)
        self.assertLess(inverted["hand_offset_ratio"], 0.0)

    def test_sign_does_not_depend_on_which_way_the_subject_faces(self) -> None:
        # Mirroring left-to-right (filmed from the other side) must not turn a drop into a lift.
        mirrored = pushup_frame(ear_offset=0.03)
        for landmark in mirrored["landmarks"]:
            landmark["x"] = 1.0 - landmark["x"]
        forward = pushup_compute_raw([pushup_frame(ear_offset=0.03)], 30.0)[0]
        backward = pushup_compute_raw([mirrored], 30.0)[0]
        self.assertGreater(backward["neck_line_signed_deg"], 0.0)
        self.assertAlmostEqual(
            forward["neck_line_signed_deg"], backward["neck_line_signed_deg"], places=4
        )

    def test_a_chord_following_head_contaminates_the_pike_direction(self) -> None:
        """DOCUMENTS the modeling assumption's cost WITH THE SIGN ATTACHED; does not assert the
        assumption is right. If a lifter's head rides the shoulder->hip torso segment rather
        than the body axis, a plank fault alone produces a neck reading -- and the sign says
        which plank fault contaminates which verdict. A SAG rotates the head SKYWARD relative to
        the axis, so it reads as a LIFT: `rule_head_drop` does not fire on it, but it MASKS a
        simultaneous genuine drop. A PIKE is the false-positive direction, and must exceed about
        0.09 of body length to forge the rule's 15 deg threshold unaided. Task 5's unsigned
        metric could not separate these two cases at all."""
        for hip_offset, expected in ((0.06, -11.3099), (0.09, -16.6992), (0.15, -26.5650)):
            raw = pushup_compute_raw(
                [pushup_frame(hip_offset=hip_offset, head_follows="chord")], 30.0
            )[0]
            self.assertAlmostEqual(raw["neck_line_signed_deg"], expected, places=3)
        for hip_offset, expected in ((-0.06, 11.3099), (-0.09, 16.6993), (-0.15, 26.5651)):
            raw = pushup_compute_raw(
                [pushup_frame(hip_offset=hip_offset, head_follows="chord")], 30.0
            )[0]
            self.assertAlmostEqual(raw["neck_line_signed_deg"], expected, places=3)
        # The head-neutral-to-the-axis model (what the metric assumes) reads 0 either way.
        for hip_offset in (0.15, -0.15):
            raw = pushup_compute_raw(
                [pushup_frame(hip_offset=hip_offset, head_follows="axis")], 30.0
            )[0]
            self.assertAlmostEqual(raw["neck_line_signed_deg"], 0.0, places=4)

    def test_signed_angle_survives_a_sagittal_single_ear_view(self) -> None:
        frame = pushup_frame(ear_offset=0.03)
        frame["landmarks"][8]["visibility"] = 0.0     # far (right) ear occluded
        raw = pushup_compute_raw([frame], 30.0)[0]
        expected = math.degrees(math.atan2(0.03, _SHOULDER_X - _EAR_X))
        self.assertAlmostEqual(raw["neck_line_signed_deg"], expected, places=4)

    def test_signed_angle_is_nan_wherever_the_unsigned_one_is(self) -> None:
        # No axis -> no normal -> no sign. The signed metric must not outlive its reference.
        frame = pushup_frame(ear_offset=0.03)
        for ankle, shoulder in ((27, 11), (28, 12)):
            frame["landmarks"][ankle] = dict(frame["landmarks"][shoulder])
        raw = pushup_compute_raw([frame], 30.0)[0]
        self.assertTrue(raw["valid"])
        self.assertFalse(np.isfinite(raw["neck_line_signed_deg"]))
        # ... and an occluded ear pair NaNs it the same way it NaNs the unsigned one.
        occluded = pushup_frame(ear_offset=0.03)
        occluded["landmarks"][7]["visibility"] = 0.0
        occluded["landmarks"][8]["visibility"] = 0.0
        raw = pushup_compute_raw([occluded], 30.0)[0]
        self.assertTrue(raw["valid"])
        self.assertFalse(np.isfinite(raw["neck_line_signed_deg"]))


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


# --- rule-layer fixtures ------------------------------------------------------------------
# The rule tests below set metric values DIRECTLY on CoreFrames rather than routing them
# through landmark geometry. That is deliberate and is the only way to place a boundary AT the
# threshold to the last bit: `_elbow_xy` reproduces a requested angle to ~1e-13, which is fine
# for "is this metric right" but useless for "does 100.0 exactly fire under a strict `>`".
# The landmark path is exercised end-to-end by PushupRuleIntegrationTests, which runs the real
# `pushup_compute_raw` -> `pushup_assign_phases` -> `run_detector` chain.

# What the fixture's default hand_drop=0.10 over a 0.60 body actually produces. Positive, as
# it must be in any genuine push-up -- see rule_hip_sag's inversion guard.
_GOOD_HAND_OFFSET = 0.1 / 0.6


def _rule_frames(count: int = 10, *, phase: str = "bottom", valid: bool = True, **metrics):
    """`count` identical CoreFrames carrying exactly the metrics named (plus a healthy
    `hand_offset_ratio` unless overridden). Identical frames also mean the centred-median
    smoothing inside `run_detector` would be a no-op on them, so the severities asserted
    against these are the same ones the real pipeline would compute."""
    values = {"hand_offset_ratio": _GOOD_HAND_OFFSET}
    values.update(metrics)
    return [
        CoreFrame(
            frame_index=index,
            time=index / 30.0,
            phase=phase,
            valid=valid,
            lower_body_visibility=1.0,
            metrics=dict(values),
        )
        for index in range(count)
    ]


def _ctx(view_type: str = "side", *, min_frames: int = 6, view_confidence: float = 0.9):
    """min_frames=6 is what `run_detector` computes at 30 fps -- max(3, ceil(30 * 0.20)) --
    so a segment-length mutant cannot hide behind an artificially permissive 1."""
    return RuleContext(
        fps=30.0,
        view_type=view_type,
        view_confidence=view_confidence,
        min_frames=min_frames,
    )


class PushupHipSagRuleTests(unittest.TestCase):
    def test_fires_when_the_hips_sag(self) -> None:
        detections = rule_hip_sag(_rule_frames(hip_offset_ratio=0.12), _ctx())
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.fault_id, "pushup_hip_sag")
        self.assertEqual(detection.fault_name, "Hip sag / broken plank line")
        self.assertEqual(detection.kg_query, "Trunk Sagging")
        self.assertEqual(detection.evidence["direction"], "sag")
        self.assertEqual(detection.evidence["threshold"], 0.06)
        self.assertEqual(detection.evidence["primary_threshold"], 0.06)
        self.assertAlmostEqual(detection.evidence["peak_hip_offset_ratio"], 0.12, places=4)
        self.assertEqual(detection.observability, "high")

    def test_does_not_fire_on_a_straight_plank(self) -> None:
        for offset in (0.0, 0.03, -0.03, -0.0001):
            self.assertEqual(
                rule_hip_sag(_rule_frames(hip_offset_ratio=offset), _ctx()), [],
                msg=f"offset {offset} must not be a fault",
            )

    def test_sag_boundary_just_inside_and_just_outside_and_exactly_on(self) -> None:
        # Strictly greater than 0.06. All three of these must behave differently from each
        # other, which is what kills a `>=` mutant and a shifted-threshold mutant alike.
        self.assertEqual(len(rule_hip_sag(_rule_frames(hip_offset_ratio=0.0601), _ctx())), 1)
        self.assertEqual(rule_hip_sag(_rule_frames(hip_offset_ratio=0.06), _ctx()), [])
        self.assertEqual(rule_hip_sag(_rule_frames(hip_offset_ratio=0.0599), _ctx()), [])

    def test_a_pike_is_reported_as_a_pike_not_a_sag(self) -> None:
        # THE INVERTED-FEEDBACK GUARD. Both directions share one fault_id per the spec, so if
        # the direction were not carried in the evidence the coach would tell a piking lifter
        # to lift their hips.
        detections = rule_hip_sag(_rule_frames(hip_offset_ratio=-0.12), _ctx())
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.fault_id, "pushup_hip_sag")
        self.assertEqual(detection.evidence["direction"], "pike")
        self.assertEqual(detection.evidence["primary_threshold"], -0.06)
        self.assertAlmostEqual(detection.evidence["peak_hip_offset_ratio"], -0.12, places=4)
        self.assertIn("pike", str(detection.evidence["primary_label"]))

    def test_pike_boundary_just_inside_and_just_outside_and_exactly_on(self) -> None:
        self.assertEqual(len(rule_hip_sag(_rule_frames(hip_offset_ratio=-0.0601), _ctx())), 1)
        self.assertEqual(rule_hip_sag(_rule_frames(hip_offset_ratio=-0.06), _ctx()), [])
        self.assertEqual(rule_hip_sag(_rule_frames(hip_offset_ratio=-0.0599), _ctx()), [])

    def test_exact_severity_midway_along_the_ramp(self) -> None:
        # MID-RAMP ON PURPOSE. 0.105 sits exactly halfway along the spec's 0.06 -> 0.15 ramp,
        # so this single number pins BOTH endpoints: moving 0.06 or 0.15 moves the answer off
        # 0.5. A fixture parked at or past 0.15 would saturate to 1.0 and see neither.
        sag = rule_hip_sag(_rule_frames(hip_offset_ratio=0.105), _ctx())[0]
        self.assertAlmostEqual(sag.severity, 0.5, places=6)
        self.assertAlmostEqual(sag.confidence, 0.5, places=6)
        # The ramp is on the MAGNITUDE, so a pike of the same size scores identically.
        pike = rule_hip_sag(_rule_frames(hip_offset_ratio=-0.105), _ctx())[0]
        self.assertAlmostEqual(pike.severity, 0.5, places=6)
        # A second mid-ramp point, to pin the ramp's slope as well as one of its values.
        # (places=4 because build_detection rounds the emitted severity to 4 dp.)
        third = rule_hip_sag(_rule_frames(hip_offset_ratio=0.09), _ctx())[0]
        self.assertAlmostEqual(third.severity, 1.0 / 3.0, places=4)

    def test_severity_saturates_at_the_severe_end_of_the_ramp(self) -> None:
        for offset in (0.15, 0.30):
            self.assertAlmostEqual(
                rule_hip_sag(_rule_frames(hip_offset_ratio=offset), _ctx())[0].severity,
                1.0, places=6,
            )

    def test_peak_frame_is_the_worst_frame_in_either_direction(self) -> None:
        # Passing the SIGNED offset series to build_detection would nanargmax a pike segment to
        # its LEAST-piked frame; the absolute series is what makes the peak the worst frame.
        offsets = [-0.07, -0.09, -0.13, -0.11, -0.08, -0.07, -0.07]
        core = [
            CoreFrame(
                frame_index=index, time=index / 30.0, phase="bottom", valid=True,
                lower_body_visibility=1.0,
                metrics={"hip_offset_ratio": offset, "hand_offset_ratio": _GOOD_HAND_OFFSET},
            )
            for index, offset in enumerate(offsets)
        ]
        detection = rule_hip_sag(core, _ctx())[0]
        self.assertEqual(detection.peak_frame, 2)
        self.assertEqual(detection.evidence["direction"], "pike")
        self.assertAlmostEqual(detection.evidence["peak_hip_offset_ratio"], -0.13, places=4)
        self.assertAlmostEqual(detection.evidence["max_abs_hip_offset_ratio"], 0.13, places=4)

    def test_a_clip_that_sags_then_pikes_yields_one_detection_of_each(self) -> None:
        # WHY `direction` IS TRUSTWORTHY AT ALL: a segment cannot contain both directions,
        # because getting from +0.061 to -0.061 must pass through |x| <= 0.06, which breaks the
        # mask. That property is argued in the rule's docstring; this pins it, so widening the
        # mask or adding hysteresis later cannot silently produce a mixed-direction segment
        # whose single reported direction is wrong for half its frames.
        mixed = (
            _rule_frames(7, hip_offset_ratio=0.12)
            + _rule_frames(7, hip_offset_ratio=0.0)
            + _rule_frames(7, hip_offset_ratio=-0.12)
        )
        core = [
            CoreFrame(
                frame_index=index, time=index / 30.0, phase=frame.phase, valid=frame.valid,
                lower_body_visibility=frame.lower_body_visibility, metrics=frame.metrics,
            )
            for index, frame in enumerate(mixed)
        ]
        detections = rule_hip_sag(core, _ctx())
        self.assertEqual([d.evidence["direction"] for d in detections], ["sag", "pike"])
        self.assertEqual((detections[0].start_frame, detections[0].end_frame), (0, 6))
        self.assertEqual((detections[1].start_frame, detections[1].end_frame), (14, 20))

    def test_camera_inversion_guard_refuses_a_directional_verdict(self) -> None:
        # A negative hand offset means the direction the metric layer believes is groundward
        # is not, so a genuine SAG arrives here as a large negative number. Emitting would
        # produce a confident, full-severity PIKE. The rule must stay silent instead.
        core = _rule_frames(hip_offset_ratio=-0.12, hand_offset_ratio=-_GOOD_HAND_OFFSET)
        self.assertEqual(rule_hip_sag(core, _ctx()), [])
        # ... in the other direction too, so this is a guard and not an accident of sign.
        core = _rule_frames(hip_offset_ratio=0.12, hand_offset_ratio=-_GOOD_HAND_OFFSET)
        self.assertEqual(rule_hip_sag(core, _ctx()), [])

    def test_zero_hand_offset_is_on_the_refusing_side_of_the_guard(self) -> None:
        # The cut is the sign boundary itself (`> 0.0`), not an invented margin around it.
        # Exactly zero -- hands on the body axis, nothing to arbitrate with -- refuses.
        core = _rule_frames(hip_offset_ratio=0.12, hand_offset_ratio=0.0)
        self.assertEqual(rule_hip_sag(core, _ctx()), [])

    def test_nan_hand_offset_guard_refuses_rather_than_assuming(self) -> None:
        # `nan > 0.0` is False, so an unmeasurable guard refuses for free. Pinned so it is
        # documented behaviour rather than a lucky accident of comparison semantics.
        core = _rule_frames(hip_offset_ratio=0.12, hand_offset_ratio=float("nan"))
        self.assertEqual(rule_hip_sag(core, _ctx()), [])

    def test_plank_angle_alone_does_not_fire_but_is_reported(self) -> None:
        # The spec's "Equivalent: ... departs from 180 deg by > ~12 deg" is a restatement, not
        # a second gate. It is UNSIGNED (Task 5 pins sag and pike to identical values), so
        # firing on it would emit a plank fault with no recoverable direction.
        quiet = _rule_frames(hip_offset_ratio=0.0, plank_angle_deviation_deg=30.0)
        self.assertEqual(rule_hip_sag(quiet, _ctx()), [])
        loud = _rule_frames(hip_offset_ratio=0.12, plank_angle_deviation_deg=30.0)
        detection = rule_hip_sag(loud, _ctx())[0]
        self.assertAlmostEqual(detection.evidence["max_plank_angle_deviation_deg"], 30.0, places=2)
        self.assertEqual(detection.evidence["plank_angle_deviation_threshold_deg"], 12.0)

    def test_a_segment_shorter_than_min_frames_does_not_fire(self) -> None:
        sagging = _rule_frames(5, hip_offset_ratio=0.12)
        clean = _rule_frames(5, hip_offset_ratio=0.0)
        self.assertEqual(rule_hip_sag(sagging + clean, _ctx()), [])
        self.assertEqual(len(rule_hip_sag(_rule_frames(6, hip_offset_ratio=0.12), _ctx())), 1)

    def test_setup_phase_is_out_of_scope(self) -> None:
        # Rule-level call, not a spec requirement: the lifter is still getting into position.
        for phase in ("setup", "unknown"):
            self.assertEqual(
                rule_hip_sag(_rule_frames(phase=phase, hip_offset_ratio=0.12), _ctx()), [],
                msg=f"phase {phase} must be out of scope",
            )
        for phase in ("descent", "bottom", "ascent"):
            self.assertEqual(
                len(rule_hip_sag(_rule_frames(phase=phase, hip_offset_ratio=0.12), _ctx())), 1,
                msg=f"phase {phase} must be in scope",
            )

    def test_invalid_frames_do_not_fire(self) -> None:
        core = _rule_frames(hip_offset_ratio=0.12, valid=False)
        self.assertEqual(rule_hip_sag(core, _ctx()), [])

    def test_head_on_views_are_hard_gated_to_silence(self) -> None:
        # The spec rates hip sag near-`none` from front/rear, and a foreshortened body axis
        # INFLATES the normalized offset -- a false-positive amplifier, so silence beats a
        # low-confidence guess.
        for view in ("front", "rear"):
            self.assertEqual(
                rule_hip_sag(_rule_frames(hip_offset_ratio=0.12), _ctx(view)), [],
                msg=f"view {view} must be hard-gated",
            )

    def test_oblique_view_downgrades_observability_and_confidence(self) -> None:
        detection = rule_hip_sag(_rule_frames(hip_offset_ratio=0.105), _ctx("front_oblique"))[0]
        self.assertEqual(detection.observability, "medium")
        self.assertAlmostEqual(detection.severity, 0.5, places=6)
        self.assertAlmostEqual(detection.confidence, 0.5 * 0.65, places=6)

    def test_a_weakly_classified_side_view_is_downgraded_not_trusted(self) -> None:
        # The view gate keys off the LABEL, so a weakly-classified `side` must not buy the
        # full-confidence treatment -- it is treated as an unclassified view instead. Inert in
        # production (score_view only emits `side` at side_score >= 0.62), so this test is the
        # only place the floor is exercised.
        strong = rule_hip_sag(_rule_frames(hip_offset_ratio=0.105), _ctx("side", view_confidence=0.62))[0]
        self.assertEqual(strong.observability, "high")
        self.assertAlmostEqual(strong.confidence, 0.5, places=6)
        # Just below the shared SIDE_VIEW_CONF_THRESHOLD (0.20).
        weak = rule_hip_sag(_rule_frames(hip_offset_ratio=0.105), _ctx("side", view_confidence=0.19))[0]
        self.assertEqual(weak.observability, "medium")
        self.assertAlmostEqual(weak.confidence, 0.5 * 0.65, places=6)
        # The floor only ever LOWERS confidence -- it never suppresses the detection, and it
        # never changes the fault itself.
        self.assertAlmostEqual(weak.severity, strong.severity, places=6)
        self.assertEqual(weak.evidence["direction"], strong.evidence["direction"])
        # Exactly at the floor is on the trusted side.
        at_floor = rule_hip_sag(_rule_frames(hip_offset_ratio=0.105), _ctx("side", view_confidence=0.20))[0]
        self.assertEqual(at_floor.observability, "high")

    def test_citation_is_copied_from_the_spec(self) -> None:
        detection = rule_hip_sag(_rule_frames(hip_offset_ratio=0.12), _ctx())[0]
        self.assertEqual(
            detection.citation,
            "Freeman S, Karpowicz A, Gray J, McGill S. Med Sci Sports Exerc (2006). "
            "DOI 10.1249/01.mss.0000189317.08635.1b.",
        )
        self.assertIn("spinal compression and torque generation in the L4-5 area",
                      detection.citation_support)
        # The spec writes "L4-5 area" inside the quoted material but "L4-L5 spine compression"
        # in its own prose. Round 1 silently dropped the second L here, and the assertions
        # above did not cover that phrase -- so it is asserted explicitly now.
        self.assertIn("large differences in L4-L5 spine compression", detection.citation_support)
        self.assertIn("the highest spine compression", detection.citation_support)
        self.assertIn("inferred from the loading mechanism", detection.citation_support)


class PushupShallowDepthRuleTests(unittest.TestCase):
    def test_fires_on_a_shallow_bottom(self) -> None:
        detections = rule_shallow_depth(_rule_frames(min_elbow_angle=125.0), _ctx())
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.fault_id, "pushup_shallow_depth")
        self.assertEqual(detection.fault_name, "Shallow depth (partial rep)")
        self.assertEqual(detection.kg_query, "Limited Range Of Motion")
        self.assertEqual(detection.evidence["threshold"], 100.0)
        self.assertAlmostEqual(detection.evidence["max_min_elbow_angle"], 125.0, places=2)
        self.assertEqual(detection.observability, "high")

    def test_does_not_fire_on_a_full_depth_rep(self) -> None:
        for angle in (60.0, 85.0, 90.0):
            self.assertEqual(
                rule_shallow_depth(_rule_frames(min_elbow_angle=angle), _ctx()), [],
                msg=f"{angle} deg is a full rep, not a fault",
            )

    def test_boundary_just_inside_and_just_outside_and_exactly_on(self) -> None:
        # Strictly greater than 100. Exactly 100.0 must NOT fire -- that is the `>=` mutant.
        self.assertEqual(len(rule_shallow_depth(_rule_frames(min_elbow_angle=100.1), _ctx())), 1)
        self.assertEqual(rule_shallow_depth(_rule_frames(min_elbow_angle=100.0), _ctx()), [])
        self.assertEqual(rule_shallow_depth(_rule_frames(min_elbow_angle=99.9), _ctx()), [])

    def test_exact_severity_midway_along_the_ramp(self) -> None:
        # 120 is exactly halfway along the spec's 100 -> 140 ramp, so this pins both endpoints.
        detection = rule_shallow_depth(_rule_frames(min_elbow_angle=120.0), _ctx())[0]
        self.assertAlmostEqual(detection.severity, 0.5, places=6)
        self.assertAlmostEqual(detection.confidence, 0.5, places=6)
        # A second mid-ramp point pins the slope too.
        quarter = rule_shallow_depth(_rule_frames(min_elbow_angle=110.0), _ctx())[0]
        self.assertAlmostEqual(quarter.severity, 0.25, places=6)

    def test_severity_saturates_at_the_severe_end_of_the_ramp(self) -> None:
        for angle in (140.0, 170.0):
            self.assertAlmostEqual(
                rule_shallow_depth(_rule_frames(min_elbow_angle=angle), _ctx())[0].severity,
                1.0, places=6,
            )

    def test_only_the_bottom_phase_counts(self) -> None:
        for phase in ("setup", "descent", "ascent", "unknown"):
            self.assertEqual(
                rule_shallow_depth(_rule_frames(phase=phase, min_elbow_angle=125.0), _ctx()), [],
                msg=f"depth must only be judged at the bottom, not during {phase}",
            )

    def test_nan_elbow_angle_does_not_fire(self) -> None:
        core = _rule_frames(min_elbow_angle=float("nan"))
        self.assertEqual(rule_shallow_depth(core, _ctx()), [])

    def test_invalid_frames_do_not_fire(self) -> None:
        core = _rule_frames(min_elbow_angle=125.0, valid=False)
        self.assertEqual(rule_shallow_depth(core, _ctx()), [])

    def test_a_segment_shorter_than_min_frames_does_not_fire(self) -> None:
        shallow = _rule_frames(5, min_elbow_angle=125.0)
        deep = _rule_frames(5, min_elbow_angle=85.0)
        self.assertEqual(rule_shallow_depth(shallow + deep, _ctx()), [])
        self.assertEqual(len(rule_shallow_depth(_rule_frames(6, min_elbow_angle=125.0), _ctx())), 1)

    def test_view_handling_follows_the_spec(self) -> None:
        for view in ("side", "front_oblique"):
            detection = rule_shallow_depth(_rule_frames(min_elbow_angle=120.0), _ctx(view))[0]
            self.assertEqual(detection.observability, "high", msg=view)
            self.assertAlmostEqual(detection.confidence, 0.5, places=6)
        # Head-on foreshortens the elbow angle: downgraded, but NOT silenced -- unlike hip sag
        # this rule makes no directional claim an unknown facing could invert.
        for view in ("front", "rear"):
            detection = rule_shallow_depth(_rule_frames(min_elbow_angle=120.0), _ctx(view))[0]
            self.assertEqual(detection.observability, "medium", msg=view)
            self.assertAlmostEqual(detection.confidence, 0.5 * 0.65, places=6)

    def test_citation_is_copied_from_the_spec(self) -> None:
        detection = rule_shallow_depth(_rule_frames(min_elbow_angle=125.0), _ctx())[0]
        self.assertEqual(
            detection.citation,
            "San Juan JG, Suprak DN, Roach SM, Lyda M. BMC Musculoskelet Disord (2015) "
            "PMC4327800.",
        )
        self.assertIn("displayed a significant linear decrease across the ROM",
                      detection.citation_support)
        self.assertIn("PMC4327800", detection.citation)


def _sequenced(frames: list[CoreFrame]) -> list[CoreFrame]:
    """Renumber a concatenation of `_rule_frames` blocks so `frame_index` runs 0..n-1 across the
    join. `_rule_frames` restarts its indices at 0 for every block, and `build_detection` reads
    `frame_index` for `start_frame`/`end_frame`, so a naive concatenation reports nonsense
    spans. Same fix `test_a_clip_that_sags_then_pikes_yields_one_detection_of_each` applies."""
    return [
        CoreFrame(
            frame_index=index,
            time=index / 30.0,
            phase=frame.phase,
            valid=frame.valid,
            lower_body_visibility=frame.lower_body_visibility,
            metrics=frame.metrics,
        )
        for index, frame in enumerate(frames)
    ]


def _head_clip(
    *,
    setup_neck: float = 0.0,
    active_neck: float = 25.0,
    setup_count: int = 8,
    active_count: int = 8,
    active_phase: str = "descent",
    setup_valid: bool = True,
    active_valid: bool = True,
    hand_offset_ratio: float = _GOOD_HAND_OFFSET,
) -> list[CoreFrame]:
    """A two-block clip for `rule_head_drop`: `setup` frames that DEFINE the baseline, then
    active frames judged against it. Both blocks are constant-valued, so `run_detector`'s
    centred median would be a no-op on them and the severities asserted here are the ones the
    real pipeline computes. 8 frames each clears `min_frames=6` with room to spare."""
    return _sequenced(
        _rule_frames(
            setup_count,
            phase="setup",
            valid=setup_valid,
            neck_line_signed_deg=setup_neck,
            hand_offset_ratio=hand_offset_ratio,
        )
        + _rule_frames(
            active_count,
            phase=active_phase,
            valid=active_valid,
            neck_line_signed_deg=active_neck,
            hand_offset_ratio=hand_offset_ratio,
        )
    )


class PushupHeadDropRuleTests(unittest.TestCase):
    def test_fires_when_the_head_drops_below_the_setup_baseline(self) -> None:
        detections = rule_head_drop(_head_clip(setup_neck=0.0, active_neck=25.0), _ctx())
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.fault_id, "pushup_head_drop")
        self.assertEqual(detection.fault_name, "Forward head / neck drop")
        self.assertEqual(detection.kg_query, "Forward Head Posture")
        self.assertEqual(detection.evidence["threshold"], 15.0)
        self.assertEqual(detection.evidence["primary_threshold"], 15.0)
        self.assertAlmostEqual(detection.evidence["max_neck_deviation_deg"], 25.0, places=2)
        self.assertAlmostEqual(detection.evidence["setup_baseline_deg"], 0.0, places=2)
        # The spec never rates this fault `high` from any view; `medium` is its ceiling.
        self.assertEqual(detection.observability, "medium")
        # The detected span is the ACTIVE block only, which is also what proves `_sequenced`
        # is doing its job -- without it these indices restart at 0 and collide.
        self.assertEqual((detection.start_frame, detection.end_frame), (8, 15))

    def test_does_not_fire_on_a_neutral_head(self) -> None:
        for neck in (0.0, 5.0, 10.0, 14.0):
            self.assertEqual(
                rule_head_drop(_head_clip(active_neck=neck), _ctx()), [],
                msg=f"{neck} deg of deviation is not yet a fault",
            )

    def test_boundary_just_inside_and_just_outside_and_exactly_on(self) -> None:
        # Strictly greater than 15. Exactly 15.0 must NOT fire -- that is the `>=` mutant.
        self.assertEqual(len(rule_head_drop(_head_clip(active_neck=15.1), _ctx())), 1)
        self.assertEqual(rule_head_drop(_head_clip(active_neck=15.0), _ctx()), [])
        self.assertEqual(rule_head_drop(_head_clip(active_neck=14.9), _ctx()), [])

    def test_the_threshold_is_measured_from_the_baseline_not_from_zero(self) -> None:
        # THE WHOLE POINT OF THE SPEC DEVIATION. With a baseline of 10, an absolute reading of
        # 25.0 is only 15.0 of deviation and must stay silent, while the SAME absolute reading
        # against a baseline of 0 fires. An implementation that ignored the baseline would pass
        # every other test in this class and fail this one.
        self.assertEqual(rule_head_drop(_head_clip(setup_neck=10.0, active_neck=25.0), _ctx()), [])
        self.assertEqual(
            len(rule_head_drop(_head_clip(setup_neck=10.0, active_neck=25.1), _ctx())), 1
        )
        self.assertEqual(len(rule_head_drop(_head_clip(setup_neck=0.0, active_neck=25.0), _ctx())), 1)
        # It works below zero too: a lifter who sets up with the head LIFTED gets faulted for a
        # smaller absolute reading, because they moved further.
        detection = rule_head_drop(_head_clip(setup_neck=-10.0, active_neck=6.0), _ctx())[0]
        self.assertAlmostEqual(detection.evidence["setup_baseline_deg"], -10.0, places=2)
        self.assertAlmostEqual(detection.evidence["max_neck_deviation_deg"], 16.0, places=2)

    def test_the_baseline_is_the_MEAN_over_setup_frames_not_the_median(self) -> None:
        # Mirrors `rule_heel_rise` (src/pose/movements/squat.py) and the spec's own squat
        # heel-rise wording, "Establish a `setup` baseline (mean over setup frames)".
        # Deliberately asymmetric: mean([0]*7 + [20]) = 2.5, median = 0.0.
        setup = _rule_frames(7, phase="setup", neck_line_signed_deg=0.0) + _rule_frames(
            1, phase="setup", neck_line_signed_deg=20.0
        )
        # 17.0 - 2.5 = 14.5 (silent under the mean); 17.0 - 0.0 = 17.0 would fire under a median.
        quiet = _sequenced(setup + _rule_frames(8, phase="descent", neck_line_signed_deg=17.0))
        self.assertEqual(rule_head_drop(quiet, _ctx()), [])
        # 17.6 - 2.5 = 15.1, just over.
        loud = _sequenced(setup + _rule_frames(8, phase="descent", neck_line_signed_deg=17.6))
        detection = rule_head_drop(loud, _ctx())[0]
        self.assertAlmostEqual(detection.evidence["setup_baseline_deg"], 2.5, places=4)
        self.assertAlmostEqual(detection.evidence["max_neck_deviation_deg"], 15.1, places=4)

    def test_exact_severity_midway_along_the_ramp(self) -> None:
        # 25 deg of deviation is exactly halfway along the rule-level 15 -> 35 ramp, so this one
        # number pins BOTH endpoints: moving either moves the answer off 0.5.
        detection = rule_head_drop(_head_clip(active_neck=25.0), _ctx())[0]
        self.assertAlmostEqual(detection.severity, 0.5, places=6)
        self.assertAlmostEqual(detection.confidence, 0.5, places=6)
        # A second mid-ramp point pins the slope as well as one of its values.
        quarter = rule_head_drop(_head_clip(active_neck=20.0), _ctx())[0]
        self.assertAlmostEqual(quarter.severity, 0.25, places=6)
        # ... and the ramp is on the DEVIATION, so a shifted baseline shifts it with them.
        shifted = rule_head_drop(_head_clip(setup_neck=5.0, active_neck=30.0), _ctx())[0]
        self.assertAlmostEqual(shifted.severity, 0.5, places=6)

    def test_severity_saturates_at_the_severe_end_of_the_ramp(self) -> None:
        for neck in (35.0, 60.0):
            self.assertAlmostEqual(
                rule_head_drop(_head_clip(active_neck=neck), _ctx())[0].severity, 1.0, places=6
            )

    def test_a_head_lift_is_not_a_head_drop(self) -> None:
        # THE REASON THE METRIC WAS MADE SIGNED. On Task 5's unsigned angle, a baseline of +5
        # and a head lifted to -15 reads |−15| = 15, deviation +10, and a big enough lift fires
        # as a drop. The signed metric refuses all of these.
        for setup_neck, active_neck in ((0.0, -25.0), (5.0, -15.0), (5.0, -60.0)):
            self.assertEqual(
                rule_head_drop(
                    _head_clip(setup_neck=setup_neck, active_neck=active_neck), _ctx()
                ),
                [],
                msg=f"baseline {setup_neck} -> {active_neck} is a head LIFT, not a drop",
            )

    def test_a_head_drop_held_from_setup_is_invisible(self) -> None:
        # DOCUMENTED COST of baselining: the rule measures CHANGE, not posture. A lifter who
        # sets up with the head already dropped and simply holds it there is never flagged. The
        # spec's absolute phrasing would catch them; the price of that would be faulting
        # long-necked lifters at rest, and this rule chooses the false-negative side.
        self.assertEqual(rule_head_drop(_head_clip(setup_neck=30.0, active_neck=30.0), _ctx()), [])
        self.assertEqual(rule_head_drop(_head_clip(setup_neck=60.0, active_neck=60.0), _ctx()), [])

    def test_camera_inversion_guard_refuses_a_directional_verdict(self) -> None:
        # The sign rides on the same groundward normal `hip_offset_ratio` uses, so an inverted
        # clip turns a head LIFT into a confident head DROP. Same guard, same reason.
        core = _head_clip(active_neck=25.0, hand_offset_ratio=-_GOOD_HAND_OFFSET)
        self.assertEqual(rule_head_drop(core, _ctx()), [])

    def test_zero_and_nan_hand_offset_are_on_the_refusing_side_of_the_guard(self) -> None:
        self.assertEqual(rule_head_drop(_head_clip(hand_offset_ratio=0.0), _ctx()), [])
        self.assertEqual(rule_head_drop(_head_clip(hand_offset_ratio=float("nan")), _ctx()), [])

    def test_no_setup_frames_means_no_baseline_and_therefore_no_verdict(self) -> None:
        # Refusing is the same call the rest of the module makes on unmeasurable input. A
        # fallback baseline of 0 would silently reinstate the absolute criterion the per-clip
        # baseline exists to avoid -- and would fire here, since 25 > 15.
        core = _sequenced(_rule_frames(16, phase="descent", neck_line_signed_deg=25.0))
        self.assertEqual(rule_head_drop(core, _ctx()), [])

    def test_a_setup_window_with_no_usable_neck_reading_refuses_too(self) -> None:
        self.assertEqual(
            rule_head_drop(_head_clip(setup_neck=float("nan"), active_neck=25.0), _ctx()), []
        )
        # Invalid setup frames are excluded from the baseline, which leaves none at all.
        self.assertEqual(
            rule_head_drop(_head_clip(setup_valid=False, active_neck=25.0), _ctx()), []
        )

    def test_nan_neck_angle_in_the_active_block_does_not_fire(self) -> None:
        self.assertEqual(rule_head_drop(_head_clip(active_neck=float("nan")), _ctx()), [])

    def test_phase_scope_is_descent_and_bottom_only(self) -> None:
        # RULE-LEVEL narrowing: the spec scopes this fault to no phase. `setup` is excluded
        # structurally (it defines the baseline), `ascent` because the spec describes the fault
        # as the chin reaching for the floor -- a descent/bottom event.
        for phase in ("descent", "bottom"):
            self.assertEqual(
                len(rule_head_drop(_head_clip(active_phase=phase), _ctx())), 1, msg=phase
            )
        for phase in ("setup", "ascent", "unknown"):
            self.assertEqual(
                rule_head_drop(_head_clip(active_phase=phase), _ctx()), [], msg=phase
            )

    def test_invalid_frames_do_not_fire(self) -> None:
        self.assertEqual(rule_head_drop(_head_clip(active_valid=False), _ctx()), [])

    def test_a_segment_shorter_than_min_frames_does_not_fire(self) -> None:
        self.assertEqual(rule_head_drop(_head_clip(active_count=5), _ctx()), [])
        self.assertEqual(len(rule_head_drop(_head_clip(active_count=6), _ctx())), 1)

    def test_view_handling_follows_the_spec(self) -> None:
        for view in ("side", "front_oblique"):
            detection = rule_head_drop(_head_clip(active_neck=25.0), _ctx(view))[0]
            self.assertEqual(detection.observability, "medium", msg=view)
            self.assertAlmostEqual(detection.confidence, 0.5, places=6)
        # The spec rates head-on `low`, not `none`, so these are downgraded rather than silenced
        # -- a foreshortened neck angle reads SMALLER, i.e. degrades toward the un-faulted
        # direction, so it is a weak signal and not a false-positive amplifier.
        for view in ("front", "rear", "rear_oblique", "unknown"):
            detection = rule_head_drop(_head_clip(active_neck=25.0), _ctx(view))[0]
            self.assertEqual(detection.observability, "low", msg=view)
            self.assertAlmostEqual(detection.confidence, 0.5 * 0.65, places=6)

    def test_a_weakly_classified_side_view_is_deliberately_NOT_downgraded(self) -> None:
        # DOCUMENTS AN OMISSION. `rule_hip_sag` carries a SIDE_VIEW_CONF_THRESHOLD floor because
        # a mislabelled `side` could be a head-on view where ITS metric inflates. Here the
        # directional hazard is camera ROLL, already covered by the hand-offset guard and
        # invisible to any view label; and `side`/`front_oblique` are treated identically, so a
        # floor would change nothing but a cosmetic number. If someone adds one later, this test
        # is where the decision gets re-argued rather than silently reversed.
        weak = rule_head_drop(_head_clip(active_neck=25.0), _ctx("side", view_confidence=0.01))[0]
        self.assertEqual(weak.observability, "medium")
        self.assertAlmostEqual(weak.confidence, 0.5, places=6)

    def test_citation_is_copied_from_the_spec(self) -> None:
        detection = rule_head_drop(_head_clip(active_neck=25.0), _ctx())[0]
        self.assertEqual(
            detection.citation,
            "Lee S et al. J Phys Ther Sci (2013) PMC3820220 (form/neutral-alignment standard); "
            "mechanism corroborated by Al Hammadi MI et al. Cureus (2025) PMC12514857.",
        )
        self.assertIn("the head, spine, and pelvis were positioned in a straight line",
                      detection.citation_support)
        self.assertIn("the cervical vertebrae in a neutral position",
                      detection.citation_support)
        self.assertIn("interfere with scapular movement", detection.citation_support)
        # The spec's own hedge must survive into the shipped support text.
        self.assertIn("cervical-injury evidence is thin", detection.citation_support)
        self.assertIn("PMC12514857", detection.citation)


class PushupElbowFlareRuleTests(unittest.TestCase):
    """NOTE the default `_ctx()` is a confident `side`, which this rule SILENCES on purpose, so
    every firing test here passes a non-side label. `rear` is used because it is what the spec
    asks for and, unlike `front`, is reachable in production (`allow_front=False`)."""

    def test_fires_when_the_hands_are_too_wide(self) -> None:
        detections = rule_elbow_flare(
            _rule_frames(hand_width_ratio=1.9, phase="descent"), _ctx("rear")
        )
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.fault_id, "pushup_elbow_flare")
        self.assertEqual(detection.fault_name, "Flared elbows / excessive hand width")
        self.assertEqual(detection.kg_query, "Elbow Valgus Torque")
        self.assertEqual(detection.evidence["threshold"], 1.6)
        self.assertEqual(detection.evidence["primary_threshold"], 1.6)
        self.assertAlmostEqual(detection.evidence["max_hand_width_ratio"], 1.9, places=4)
        self.assertEqual(detection.observability, "medium")

    def test_does_not_fire_at_or_near_shoulder_width(self) -> None:
        for ratio in (0.8, 1.0, 1.3, 1.5):
            self.assertEqual(
                rule_elbow_flare(
                    _rule_frames(hand_width_ratio=ratio, phase="descent"), _ctx("rear")
                ),
                [],
                msg=f"ratio {ratio} is not a fault",
            )

    def test_boundary_just_inside_and_just_outside_and_exactly_on(self) -> None:
        # Strictly greater than the spec's 1.6. Exactly 1.6 must NOT fire.
        self.assertEqual(
            len(rule_elbow_flare(_rule_frames(hand_width_ratio=1.61, phase="descent"), _ctx("rear"))), 1
        )
        self.assertEqual(
            rule_elbow_flare(_rule_frames(hand_width_ratio=1.6, phase="descent"), _ctx("rear")), []
        )
        self.assertEqual(
            rule_elbow_flare(_rule_frames(hand_width_ratio=1.59, phase="descent"), _ctx("rear")), []
        )

    def test_exact_severity_midway_along_the_ramp(self) -> None:
        # 1.9 is exactly halfway along the rule-level 1.6 -> 2.2 ramp, pinning both endpoints.
        detection = rule_elbow_flare(
            _rule_frames(hand_width_ratio=1.9, phase="descent"), _ctx("rear")
        )[0]
        self.assertAlmostEqual(detection.severity, 0.5, places=6)
        # No view discount is applied at all (see the rule's docstring), so confidence == severity.
        self.assertAlmostEqual(detection.confidence, 0.5, places=6)
        quarter = rule_elbow_flare(
            _rule_frames(hand_width_ratio=1.75, phase="descent"), _ctx("rear")
        )[0]
        self.assertAlmostEqual(quarter.severity, 0.25, places=6)

    def test_severity_saturates_at_the_severe_end_of_the_ramp(self) -> None:
        for ratio in (2.2, 3.0):
            self.assertAlmostEqual(
                rule_elbow_flare(
                    _rule_frames(hand_width_ratio=ratio, phase="descent"), _ctx("rear")
                )[0].severity,
                1.0, places=6,
            )

    def test_the_measurability_gate_cannot_change_any_outcome(self) -> None:
        """DOCUMENTS AN INERT TERM rather than claiming it is tested. The mask carries
        `hand_width_ratio > 0.25` -- the brief's "the wrists must actually be separated in the
        image", written as a ratio because `hand_width_ratio` IS `distance(15,16) /
        shoulder_width`. Every ratio it would reject is already rejected by the spec's 1.6, so
        no fixture can distinguish 0.25 from any other value below 1.6. It is reported as a
        knowingly-SURVIVING mutant, not engineered into a fake kill, and is kept so the
        measurability condition stays legible and becomes live if 1.6 is ever lowered."""
        for ratio in (0.0, 0.2, 0.25, 0.26, 1.0, 1.6):
            self.assertEqual(
                rule_elbow_flare(
                    _rule_frames(hand_width_ratio=ratio, phase="descent"), _ctx("rear")
                ),
                [],
                msg=f"ratio {ratio}",
            )

    def test_a_confident_side_view_is_silenced(self) -> None:
        # NEGATIVE view gate. "Is the camera confidently perpendicular to the body's long axis?"
        # IS answerable for a horizontal body (the `side` verdict rests on `body_axis_extent`,
        # which Task 3 validated for horizontal subjects) -- unlike "is this a front view?",
        # which Task 4 says carries no validated meaning here. If it is a side view the wrists
        # overlap and any ratio is noise/noise, so the rule refuses.
        for ratio in (1.9, 3.0, 10.0):
            self.assertEqual(
                rule_elbow_flare(_rule_frames(hand_width_ratio=ratio, phase="descent"), _ctx()),
                [],
                msg=f"a confident side view must silence ratio {ratio}",
            )
        # The cut is the existing shared SIDE_VIEW_CONF_THRESHOLD (0.20) -- no new number.
        at_threshold = rule_elbow_flare(
            _rule_frames(hand_width_ratio=1.9, phase="descent"), _ctx("side", view_confidence=0.20)
        )
        self.assertEqual(at_threshold, [])
        just_below = rule_elbow_flare(
            _rule_frames(hand_width_ratio=1.9, phase="descent"), _ctx("side", view_confidence=0.19)
        )
        self.assertEqual(len(just_below), 1)

    def test_observability_is_a_flat_medium_with_no_view_discount(self) -> None:
        # `medium` is the spec's CEILING for this fault, and the only label that would tier
        # below it is the front/rear one Task 4 discredited for a horizontal body -- so
        # discounting on it would be theatre. The one informative label, `side`, is handled by
        # silence above rather than by a discount.
        for view in ("front", "rear", "front_oblique", "rear_oblique", "unknown"):
            detection = rule_elbow_flare(
                _rule_frames(hand_width_ratio=1.9, phase="descent"), _ctx(view)
            )[0]
            self.assertEqual(detection.observability, "medium", msg=view)
            self.assertAlmostEqual(detection.confidence, 0.5, places=6, msg=view)

    def test_setup_phase_is_out_of_scope(self) -> None:
        # Same rule-level call `rule_hip_sag` makes: during setup the hands are still being
        # placed. The spec scopes this fault to no phase.
        for phase in ("setup", "unknown"):
            self.assertEqual(
                rule_elbow_flare(
                    _rule_frames(phase=phase, hand_width_ratio=1.9), _ctx("rear")
                ),
                [], msg=f"phase {phase} must be out of scope",
            )
        for phase in ("descent", "bottom", "ascent"):
            self.assertEqual(
                len(rule_elbow_flare(
                    _rule_frames(phase=phase, hand_width_ratio=1.9), _ctx("rear")
                )),
                1, msg=f"phase {phase} must be in scope",
            )

    def test_invalid_frames_and_nan_ratios_do_not_fire(self) -> None:
        self.assertEqual(
            rule_elbow_flare(
                _rule_frames(hand_width_ratio=1.9, phase="descent", valid=False), _ctx("rear")
            ),
            [],
        )
        self.assertEqual(
            rule_elbow_flare(
                _rule_frames(hand_width_ratio=float("nan"), phase="descent"), _ctx("rear")
            ),
            [],
        )

    def test_a_segment_shorter_than_min_frames_does_not_fire(self) -> None:
        wide = _rule_frames(5, phase="descent", hand_width_ratio=1.9)
        narrow = _rule_frames(5, phase="descent", hand_width_ratio=1.0)
        self.assertEqual(rule_elbow_flare(_sequenced(wide + narrow), _ctx("rear")), [])
        self.assertEqual(
            len(rule_elbow_flare(_rule_frames(6, phase="descent", hand_width_ratio=1.9), _ctx("rear"))),
            1,
        )

    def test_citation_is_copied_from_the_spec(self) -> None:
        detection = rule_elbow_flare(
            _rule_frames(hand_width_ratio=1.9, phase="descent"), _ctx("rear")
        )[0]
        self.assertEqual(
            detection.citation,
            "Donkers MJ, An KN, Chao EY, Morrey BF. J Biomech (1993). "
            "DOI 10.1016/0021-9290(93)90026-b.",
        )
        self.assertIn("averaged 45% of the body weight", detection.citation_support)
        self.assertIn("the maximum valgus torque at the elbow opposed by the medial "
                      "ligamentous structure", detection.citation_support)
        self.assertIn("rose 42% one-handed", detection.citation_support)


class PushupElbowFlareKnownGapTests(unittest.TestCase):
    def test_sagittal_collapse_can_forge_a_wide_ratio(self) -> None:
        """KNOWN GAP, pinned rather than claimed away -- the same idiom as
        `test_visible_but_misplaced_landmark_is_trusted`.

        The measurability gate assumes a sagittal view makes the WRIST separation vanish. It
        does -- but the SHOULDER separation vanishes with it, so `hand_width_ratio` becomes
        noise/noise and can land anywhere, including far above the spec's 1.6.
        `_DEGENERATE_LENGTH` guards division by zero, not this. Closing it needs the wrist span
        normalized by something that survives the sagittal view (shoulder-to-ankle length would),
        which is not emitted today and is deliberately NOT invented here: a fabricated
        anthropometric constant in a FIRE gate is worse than a documented gap. The `side` view
        gate narrows this -- but only partly, because Task 4's limit 3 records that a genuine
        sagittal clip often fails to EARN the `side` label."""
        frame = pushup_frame()
        # Both pairs collapsed onto near-coincident image points, as a camera looking down the
        # sagittal plane produces. Shoulder separation 0.0020, wrist separation 0.0050.
        for index, y in ((11, 0.4990), (12, 0.5010), (15, 0.5975), (16, 0.6025)):
            frame["landmarks"][index]["y"] = y
            frame["landmarks"][index]["x"] = _SHOULDER_X if index in (11, 12) else _WRIST_X
        raw = pushup_compute_raw([frame], 30.0)[0]
        self.assertTrue(raw["valid"])
        # A forged 2.5 from two sub-pixel separations -- truth here is "unmeasurable".
        self.assertAlmostEqual(raw["hand_width_ratio"], 2.5, places=3)

        # ... and the rule fires on it at FULL severity when the view label is not a confident
        # `side`, which is exactly the residual the view gate cannot reach.
        detections = rule_elbow_flare(
            _rule_frames(hand_width_ratio=raw["hand_width_ratio"], phase="descent"), _ctx("rear")
        )
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].severity, 1.0, places=6)


class PushupScapularWingingRuleTests(unittest.TestCase):
    def test_scapular_winging_never_emits(self) -> None:
        # PINNED ON PURPOSE. MediaPipe's 33 landmarks contain no scapular border points, so this
        # rule must stay silent from every view; the spec rates it observability `none` and says
        # "Recommend NOT emitting a confident verdict". If someone implements a proxy, THIS TEST
        # FAILING IS THE INTENDED SIGNAL to go and have a spec conversation first -- not a
        # nuisance failure to update. The rule is registered rather than omitted so the spec and
        # the code stay in 1:1 correspondence.
        frames = [pushup_frame(elbow_angle=90.0, frame_index=i) for i in range(12)]
        core, _ = run_detector(_TEST_DETECTOR, frames, 30.0, "side", 0.8)
        for view in ("side", "front", "rear", "front_oblique", "rear_oblique", "unknown"):
            self.assertEqual(rule_scapular_winging(core, _ctx(view)), [], msg=view)
        # Also silent on the postures a proxy would be tempted by (rounded upper back would
        # show up as a broken plank line / dropped head here).
        loud = _head_clip(active_neck=40.0)
        self.assertEqual(rule_scapular_winging(loud, _ctx()), [])
        self.assertEqual(rule_scapular_winging([], _ctx()), [])


# A test-only detector so the rules can be driven through the real
# compute_raw -> assign_phases -> centred-median -> rules chain. Task 8 owns the production
# MovementDetector and its registry.register call; this one is never registered.
_TEST_DETECTOR = MovementDetector(
    "Push-up (test harness)",
    PUSHUP_METRIC_KEYS,
    pushup_compute_raw,
    pushup_assign_phases,
    (rule_hip_sag, rule_shallow_depth, rule_head_drop, rule_elbow_flare, rule_scapular_winging),
)


class PushupRuleIntegrationTests(unittest.TestCase):
    """The rules driven over real landmark geometry, not hand-set metric values."""

    @staticmethod
    def _clip(**knobs) -> list[dict]:
        # Constant frames: the centred median (window 5) is a no-op on them, so the severity
        # the pipeline computes is exactly the severity the arithmetic predicts.
        return [pushup_frame(frame_index=i, **knobs) for i in range(20)]

    def test_a_sagging_shallow_rep_fires_both_rules_with_exact_severities(self) -> None:
        # hip_offset 0.063 over the fixture's 0.60 body axis => hip_offset_ratio 0.105, the
        # ramp midpoint; elbow 120 deg is the depth ramp's midpoint. Both must score 0.5.
        _, detections = run_detector(
            _TEST_DETECTOR, self._clip(hip_offset=0.063, elbow_angle=120.0), 30.0, "side", 0.9
        )
        by_id = {detection.fault_id: detection for detection in detections}
        self.assertEqual(set(by_id), {"pushup_hip_sag", "pushup_shallow_depth"})
        self.assertAlmostEqual(by_id["pushup_hip_sag"].severity, 0.5, places=4)
        self.assertEqual(by_id["pushup_hip_sag"].evidence["direction"], "sag")
        self.assertAlmostEqual(by_id["pushup_shallow_depth"].severity, 0.5, places=4)

    def test_a_clean_deep_rep_fires_nothing(self) -> None:
        _, detections = run_detector(
            _TEST_DETECTOR, self._clip(hip_offset=0.0, elbow_angle=85.0), 30.0, "side", 0.9
        )
        self.assertEqual(detections, [])

    def test_a_piking_rep_is_reported_as_a_pike_end_to_end(self) -> None:
        _, detections = run_detector(
            _TEST_DETECTOR, self._clip(hip_offset=-0.063, elbow_angle=85.0), 30.0, "side", 0.9
        )
        self.assertEqual([d.fault_id for d in detections], ["pushup_hip_sag"])
        self.assertEqual(detections[0].evidence["direction"], "pike")
        self.assertAlmostEqual(detections[0].severity, 0.5, places=4)

    def test_camera_inversion_silences_hip_sag_but_not_shallow_depth(self) -> None:
        # A 180-degree-rotated clip of a genuine sag. Without the hand-offset guard the sag
        # would be emitted as a full-severity PIKE; the depth rule, which reads no sign, is
        # unaffected and must keep working.
        _, detections = run_detector(
            _TEST_DETECTOR,
            self._clip(hip_offset=0.063, elbow_angle=120.0, tilt_deg=180.0),
            30.0, "side", 0.9,
        )
        self.assertEqual([d.fault_id for d in detections], ["pushup_shallow_depth"])

    @staticmethod
    def _rep_with_head(ear_offset: float, *, drop_from: int = 6, **knobs) -> list[dict]:
        """A real 24-frame rep whose head leaves the line partway through the descent. The
        opening frames stay neutral so the `setup` baseline is measured on an undropped head;
        the centred median (window 5) is a no-op on both constant stretches and only blends
        across the single step, which sits well before the first judged frame."""
        angles = (
            [170.0] * 4
            + [155.0, 140.0, 125.0, 110.0, 100.0, 95.0]
            + [85.0] * 4
            + [95.0, 105.0, 120.0, 135.0, 150.0, 160.0]
            + [170.0] * 4
        )
        return [
            pushup_frame(
                elbow_angle=angle,
                frame_index=i,
                ear_offset=(ear_offset if i >= drop_from else 0.0),
                **knobs,
            )
            for i, angle in enumerate(angles)
        ]

    def test_a_head_drop_fires_end_to_end_with_an_exact_severity(self) -> None:
        # The ear sits 0.08 along the body axis from the shoulder, so an offset of
        # 0.08 * tan(25 deg) puts the neck exactly 25 deg off the line -- the ramp midpoint --
        # against a setup baseline of 0.
        offset = 0.08 * math.tan(math.radians(25.0))
        _, detections = run_detector(
            _TEST_DETECTOR, self._rep_with_head(offset), 30.0, "side", 0.9
        )
        by_id = {detection.fault_id: detection for detection in detections}
        self.assertIn("pushup_head_drop", by_id)
        detection = by_id["pushup_head_drop"]
        self.assertAlmostEqual(detection.severity, 0.5, places=3)
        self.assertAlmostEqual(detection.evidence["setup_baseline_deg"], 0.0, places=3)
        self.assertAlmostEqual(detection.evidence["max_neck_deviation_deg"], 25.0, places=2)
        self.assertEqual(detection.observability, "medium")

    def test_a_head_lift_of_the_same_size_fires_nothing_end_to_end(self) -> None:
        # Directional over real landmark geometry, not just over hand-set metric values.
        offset = 0.08 * math.tan(math.radians(25.0))
        _, detections = run_detector(
            _TEST_DETECTOR, self._rep_with_head(-offset), 30.0, "side", 0.9
        )
        self.assertNotIn("pushup_head_drop", {d.fault_id for d in detections})

    def test_an_inverted_clip_silences_the_head_drop_end_to_end(self) -> None:
        # A genuine head drop filmed upside down: the sign flips, and without the hand-offset
        # guard this would be a confident head LIFT (silent) or, symmetrically, a lift would
        # become a confident drop. The guard refuses either way.
        offset = 0.08 * math.tan(math.radians(25.0))
        for signed_offset in (offset, -offset):
            _, detections = run_detector(
                _TEST_DETECTOR,
                self._rep_with_head(signed_offset, tilt_deg=180.0),
                30.0, "side", 0.9,
            )
            self.assertNotIn("pushup_head_drop", {d.fault_id for d in detections})

    def test_wide_hands_fire_only_when_the_view_is_not_a_confident_side(self) -> None:
        frames = self._clip(hand_width_ratio=1.9, elbow_angle=85.0)
        _, silenced = run_detector(_TEST_DETECTOR, frames, 30.0, "side", 0.9)
        self.assertEqual(silenced, [])
        _, detections = run_detector(_TEST_DETECTOR, frames, 30.0, "rear", 0.9)
        self.assertEqual([d.fault_id for d in detections], ["pushup_elbow_flare"])
        self.assertAlmostEqual(detections[0].severity, 0.5, places=4)

    def test_a_clip_cropped_at_the_knees_fires_nothing_at_all(self) -> None:
        # The module-wide silence risk, now visible at the rule layer: no ankles, no plank
        # line, no valid frames -- so even the depth rule (which does not need ankles) goes
        # quiet, because the validity gate precedes it.
        frames = self._clip(hip_offset=0.063, elbow_angle=120.0)
        for frame in frames:
            frame["landmarks"][27]["visibility"] = 0.0
            frame["landmarks"][28]["visibility"] = 0.0
        _, detections = run_detector(_TEST_DETECTOR, frames, 30.0, "side", 0.9)
        self.assertEqual(detections, [])


if __name__ == "__main__":
    unittest.main()
