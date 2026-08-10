import math
import unittest

import numpy as np

from src.pose.geometry import angle_degrees, landmarks_to_array
from src.pose.movements.base import CoreFrame, RuleContext, run_detector
from src.pose.movements.registry import get_detector, list_detectors
from src.pose.movements.torso_twist import (
    BRACE_MILD_DEG,
    BRACE_SEVERE_DEG,
    TORSO_TWIST_DETECTOR,
    TORSO_TWIST_METRIC_KEYS,
    rule_insufficient_rotation_rom,
    rule_trunk_not_braced,
    torso_twist_assign_phases,
    torso_twist_compute_raw,
)

_HIP = (0.50, 0.62)
_THIGH_LEN = 0.20
_TRUNK_LEN = 0.24
_SHOULDER_HALF_WIDTH = 0.07
_KNEE_HALF_WIDTH = 0.05
_HAND_HALF_WIDTH = 0.01
# The thighs of a seated twister: forward and slightly up out of the hips, in image
# coordinates where y grows DOWNWARD.
_THIGH_DIR = (0.94, -0.34)

# Landmark pairs that must swap identity when the image is mirrored: a subject filmed from
# behind has their anatomical left on the other side of the frame.
_MIRROR_PAIRS = ((11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32))


def _lm(x: float, y: float, z: float = 0.0, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": z, "visibility": visibility}


def _rot(vector: tuple[float, float], degrees: float) -> tuple[float, float]:
    rad = math.radians(degrees)
    cos, sin = math.cos(rad), math.sin(rad)
    return (vector[0] * cos - vector[1] * sin, vector[0] * sin + vector[1] * cos)


def _unit(vector: tuple[float, float]) -> tuple[float, float]:
    norm = math.hypot(*vector)
    return (vector[0] / norm, vector[1] / norm)


def twist_frame(
    trunk_thigh_deg: float = 95.0,
    twist_ratio: float = 0.0,
    shoulder_projection: float = 1.0,
    frame_index: int = 0,
    drop_landmark: int | None = None,
    roll_deg: float = 0.0,
    mirrored: bool = False,
) -> dict:
    """One SEATED RUSSIAN TWIST frame, image y growing DOWNWARD.

    Every landmark carries z=0, so the module's 2-D reads see exactly the image-plane geometry
    and each knob controls its metric BY CONSTRUCTION -- the same reason tests/test_situp.py and
    tests/test_leg_abduction.py need no depth correction.

    Knobs:

      trunk_thigh_deg      -- the angle between the trunk (hip-mid -> shoulder-mid) and the
                              thighs (hip-mid -> knee-mid). `trunk_thigh_angle_deg` equals this
                              exactly. This is the brace quantity: it OPENS as a seated twister
                              sags back toward the floor.
      twist_ratio          -- signed travel of the clasped hands across the body, measured along
                              the shoulder axis in shoulder widths. `twist_offset_ratio` equals
                              this exactly.
      shoulder_projection  -- foreshortening of the shoulder line, i.e. `cos` of an axial
                              rotation into the image plane. 1.0 = square to the camera. It moves
                              each individual shoulder but NOT the shoulder midpoint, which is
                              what `test_the_brace_angle_does_not_move_when_the_subject_only
                              _twists` exists to demonstrate.
      drop_landmark        -- zero one landmark's visibility, to exercise the all-or-nothing
                              8-landmark validity gate.
      roll_deg             -- rotate EVERY landmark about the image centre. Both metrics are
                              built from angles between body vectors and a dot product of body
                              vectors, so this must change nothing.
      mirrored             -- reflect about the image vertical AND swap every left/right landmark
                              pair, i.e. the subject filmed from behind. The brace angle must be
                              unchanged; the twist offset must change SIGN and keep its
                              magnitude, which is the honest limit of a monocular pipeline.
    """
    hip_mid = _HIP
    thigh = _unit(_THIGH_DIR)
    knee_mid = (hip_mid[0] + _THIGH_LEN * thigh[0], hip_mid[1] + _THIGH_LEN * thigh[1])
    knee_across = (-thigh[1], thigh[0])
    knee = {
        "left": [knee_mid[0] + _KNEE_HALF_WIDTH * knee_across[0], knee_mid[1] + _KNEE_HALF_WIDTH * knee_across[1]],
        "right": [knee_mid[0] - _KNEE_HALF_WIDTH * knee_across[0], knee_mid[1] - _KNEE_HALF_WIDTH * knee_across[1]],
    }

    # The trunk is the thigh direction rotated OPEN by `trunk_thigh_deg`; a larger angle lays
    # the torso further back.
    trunk = _unit(_rot(thigh, -trunk_thigh_deg))
    shoulder_mid = (hip_mid[0] + _TRUNK_LEN * trunk[0], hip_mid[1] + _TRUNK_LEN * trunk[1])
    across = (-trunk[1], trunk[0])  # the shoulder axis, perpendicular to the trunk
    half = _SHOULDER_HALF_WIDTH * shoulder_projection
    shoulder = {
        "left": [shoulder_mid[0] + half * across[0], shoulder_mid[1] + half * across[1]],
        "right": [shoulder_mid[0] - half * across[0], shoulder_mid[1] - half * across[1]],
    }

    # Hands clasped in front of the chest, carried `twist_ratio` shoulder widths across the
    # body. The component along the trunk is perpendicular to the shoulder axis, so it does not
    # enter the dot product and the knob stays exact.
    shoulder_width = 2.0 * half
    hand_mid = (
        hip_mid[0] + 0.45 * _TRUNK_LEN * trunk[0] + twist_ratio * shoulder_width * across[0],
        hip_mid[1] + 0.45 * _TRUNK_LEN * trunk[1] + twist_ratio * shoulder_width * across[1],
    )
    wrist = {
        "left": [hand_mid[0] + _HAND_HALF_WIDTH * across[0], hand_mid[1] + _HAND_HALF_WIDTH * across[1]],
        "right": [hand_mid[0] - _HAND_HALF_WIDTH * across[0], hand_mid[1] - _HAND_HALF_WIDTH * across[1]],
    }
    ankle = {
        "left": [knee["left"][0] + 0.02, knee["left"][1] + 0.14],
        "right": [knee["right"][0] + 0.02, knee["right"][1] + 0.14],
    }

    points = {
        11: shoulder["left"], 12: shoulder["right"],
        15: wrist["left"], 16: wrist["right"],
        23: [hip_mid[0] + 0.05, hip_mid[1]], 24: [hip_mid[0] - 0.05, hip_mid[1]],
        25: knee["left"], 26: knee["right"],
        27: ankle["left"], 28: ankle["right"],
        29: ankle["left"], 30: ankle["right"],
        31: ankle["left"], 32: ankle["right"],
    }
    if mirrored:
        points = {index: [1.0 - x, y] for index, (x, y) in points.items()}
        for left_index, right_index in _MIRROR_PAIRS:
            if left_index in points and right_index in points:
                points[left_index], points[right_index] = points[right_index], points[left_index]

    landmarks = []
    for index in range(33):
        x, y = points.get(index, (0.5, 0.30))
        if roll_deg:
            dx, dy = x - 0.5, y - 0.5
            rx, ry = _rot((dx, dy), roll_deg)
            x, y = rx + 0.5, ry + 0.5
        visibility = 0.0 if index == drop_landmark else 0.95
        landmarks.append(_lm(x, y, 0.0, visibility))
    return {"frame_index": frame_index, "landmarks": landmarks}


def swing_clip(
    swings: int = 3,
    frames_per_swing: int = 24,
    peak_ratio: float = 0.55,
    brace_deviation_deg: float = 0.0,
    brace_from_setup: bool = False,
    base_trunk_thigh_deg: float = 95.0,
) -> list[dict]:
    """A clip of `swings` alternating side-swings, centre -> peak -> centre each time.

    `brace_deviation_deg` opens the trunk-thigh angle in step with the swing, so a rep whose
    twist peaks is also the rep whose brace is worst -- the realistic coupling. With
    `brace_from_setup=True` the deviation is instead applied from the FIRST frame and held, which
    is the posture a baseline-relative rule is structurally blind to.
    """
    frames: list[dict] = []
    for swing in range(swings):
        direction = 1.0 if swing % 2 == 0 else -1.0
        for step in range(frames_per_swing):
            fraction = math.sin(math.pi * step / (frames_per_swing - 1))
            deviation = brace_deviation_deg if brace_from_setup else brace_deviation_deg * fraction
            frames.append(
                twist_frame(
                    trunk_thigh_deg=base_trunk_thigh_deg + deviation,
                    twist_ratio=direction * peak_ratio * fraction,
                    frame_index=len(frames),
                )
            )
    return frames


def _core(frames: list[dict], phase: str = "peak") -> list[CoreFrame]:
    """Raw frames -> CoreFrames carrying every metric key, all in one phase."""
    raw = torso_twist_compute_raw(frames, 30.0)
    core = []
    for index, item in enumerate(raw):
        core.append(
            CoreFrame(
                frame_index=int(item.get("frame_index", index)),
                time=float(item.get("time", 0.0)),
                phase=phase if item.get("valid") else "unknown",
                valid=bool(item.get("valid")),
                lower_body_visibility=float(item.get("lower_body_visibility", 0.0)),
                metrics={key: float(item.get(key, math.nan)) for key in TORSO_TWIST_METRIC_KEYS},
            )
        )
    return core


def _core_with_setup(frames: list[dict], setup_frames: int = 6) -> list[CoreFrame]:
    """As `_core`, but the opening `setup_frames` are labelled `setup` -- what the shipped rule
    needs in order to have a baseline at all."""
    core = _core(frames, phase="peak")
    out = []
    for index, frame in enumerate(core):
        phase = "unknown" if not frame.valid else ("setup" if index < setup_frames else "peak")
        out.append(
            CoreFrame(
                frame_index=frame.frame_index, time=frame.time, phase=phase, valid=frame.valid,
                lower_body_visibility=frame.lower_body_visibility, metrics=frame.metrics,
            )
        )
    return out


def _ctx(view_type: str = "front", min_frames: int = 3) -> RuleContext:
    return RuleContext(fps=30.0, view_type=view_type, view_confidence=0.8, min_frames=min_frames)


class MetricLayerTest(unittest.TestCase):
    def test_metric_keys_match_the_emitted_metrics_exactly(self) -> None:
        """Two-way: a key the tuple omits is silently NaN in every rule, and a key the tuple
        names but nothing emits is silently NaN too."""
        raw = torso_twist_compute_raw([twist_frame()], 30.0)[0]
        framework = {"frame_index", "time", "valid", "lower_body_visibility"}
        emitted = set(raw) - framework
        self.assertEqual(emitted, set(TORSO_TWIST_METRIC_KEYS))

    def test_the_knobs_control_their_metrics_exactly(self) -> None:
        raw = torso_twist_compute_raw([twist_frame(trunk_thigh_deg=112.0, twist_ratio=0.42)], 30.0)[0]
        self.assertAlmostEqual(raw["trunk_thigh_angle_deg"], 112.0, places=4)
        self.assertAlmostEqual(raw["twist_offset_ratio"], 0.42, places=4)

    def test_the_twist_offset_is_signed(self) -> None:
        left = torso_twist_compute_raw([twist_frame(twist_ratio=0.4)], 30.0)[0]
        right = torso_twist_compute_raw([twist_frame(twist_ratio=-0.4)], 30.0)[0]
        self.assertAlmostEqual(left["twist_offset_ratio"], -right["twist_offset_ratio"], places=6)

    def test_the_brace_angle_does_not_move_when_the_subject_only_twists(self) -> None:
        """THE ONE CONSTRUCTION CHOICE THIS MODULE MAKES THAT SIT-UP DOES NOT.

        `situp_compute_raw` reads a SAME-SIDE angle, `angle(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)`,
        so that a rolled subject never blends one side's shoulder with the other side's knee.
        Correct for a sit-up; wrong here, because an axial twist swings each individual shoulder
        across the frame while leaving the shoulder MIDPOINT -- which sits on the rotation axis --
        where it was. This asserts the contrast rather than asserting it away: with the trunk
        posture held fixed and only the shoulder line foreshortened, the module's midpoint
        construction is unmoved and Sit-up's same-side construction moves by degrees.
        """
        square = twist_frame(shoulder_projection=1.0)
        turned = twist_frame(shoulder_projection=0.3)  # ~72 deg of axial rotation, projected

        module_square = torso_twist_compute_raw([square], 30.0)[0]["trunk_thigh_angle_deg"]
        module_turned = torso_twist_compute_raw([turned], 30.0)[0]["trunk_thigh_angle_deg"]
        self.assertAlmostEqual(module_square, module_turned, places=6)

        same_side_square = angle_degrees(landmarks_to_array(square["landmarks"]), 11, 23, 25)
        same_side_turned = angle_degrees(landmarks_to_array(turned["landmarks"]), 11, 23, 25)
        self.assertGreater(abs(same_side_turned - same_side_square), 5.0)

    def test_one_dropped_landmark_invalidates_the_whole_frame(self) -> None:
        for index in (11, 12, 15, 16, 23, 24, 25, 26):
            with self.subTest(landmark=index):
                raw = torso_twist_compute_raw([twist_frame(drop_landmark=index)], 30.0)[0]
                self.assertFalse(raw["valid"])
                self.assertNotIn("trunk_thigh_angle_deg", raw)

    def test_the_wrists_are_required_and_the_hands_are_the_occlusion_risk(self) -> None:
        """Sit-up requires six landmarks and no wrist; this module requires eight because the
        rep signal IS the hands. Recorded as a test so the cost is not quietly relaxed later."""
        self.assertFalse(torso_twist_compute_raw([twist_frame(drop_landmark=15)], 30.0)[0]["valid"])
        self.assertTrue(torso_twist_compute_raw([twist_frame(drop_landmark=27)], 30.0)[0]["valid"])

    def test_a_non_dict_frame_is_invalid_rather_than_raising(self) -> None:
        raw = torso_twist_compute_raw([None, "nonsense", 7], 30.0)
        self.assertEqual([item["valid"] for item in raw], [False, False, False])


class InvarianceTest(unittest.TestCase):
    """The Group E re-anchoring mandate, carried into Group F and pinned executably."""

    def test_every_metric_is_invariant_under_camera_roll(self) -> None:
        reference = torso_twist_compute_raw([twist_frame(trunk_thigh_deg=104.0, twist_ratio=0.37)], 30.0)[0]
        for roll in (17.0, 90.0, 180.0, -90.0):
            with self.subTest(roll=roll):
                rolled = torso_twist_compute_raw(
                    [twist_frame(trunk_thigh_deg=104.0, twist_ratio=0.37, roll_deg=roll)], 30.0
                )[0]
                for key in TORSO_TWIST_METRIC_KEYS:
                    self.assertAlmostEqual(rolled[key], reference[key], places=4, msg=key)

    def test_the_twist_magnitude_survives_mirroring_and_its_sign_does_not(self) -> None:
        """The honest limit, asserted rather than hidden. A dot product onto a body axis is
        roll-invariant and mirror-invariant IN MAGNITUDE; the sign says which way the hands went
        in the IMAGE, and no monocular pipeline can map that onto the subject's own left and
        right. Nothing in this module claims a body side, which is why this is a limitation and
        not a defect: the rep signal is rectified and the ROM rule reads the magnitude.
        """
        forward = torso_twist_compute_raw([twist_frame(trunk_thigh_deg=104.0, twist_ratio=0.37)], 30.0)[0]
        behind = torso_twist_compute_raw(
            [twist_frame(trunk_thigh_deg=104.0, twist_ratio=0.37, mirrored=True)], 30.0
        )[0]
        self.assertAlmostEqual(behind["trunk_thigh_angle_deg"], forward["trunk_thigh_angle_deg"], places=4)
        self.assertAlmostEqual(abs(behind["twist_offset_ratio"]), abs(forward["twist_offset_ratio"]), places=4)
        self.assertAlmostEqual(behind["twist_offset_ratio"], -forward["twist_offset_ratio"], places=4)

    def test_detections_survive_roll_and_mirroring_all_the_way_to_the_verdict(self) -> None:
        """Invariance all the way to a detection, not just to a metric -- including through
        `segment_reps`, which sees the SIGNED metric and rectifies it.

        NOT ASSERTED AS BYTE-IDENTITY, AND THE REASON IS THE ROUNDING RATHER THAN THE GEOMETRY.
        Leg Abduction could assert byte-identity because its fixture rotates to coordinates that
        survive the round trip exactly. Here the roll is applied about the image centre in
        floating point, so the metrics agree to ~1e-13 while `build_detection` rounds its
        evidence to 2 decimals -- and a value sitting on a .005 boundary can round the two ways.
        Everything that DECIDES anything (the fault, the severity, the confidence, the frames,
        the phase, the observability) is compared exactly; the evidence numbers are compared to
        0.01, which is the rounding itself and not a slackened claim.
        """
        def run(roll: float, mirrored: bool):
            frames = [
                twist_frame(
                    trunk_thigh_deg=frame["_angle"], twist_ratio=frame["_twist"],
                    frame_index=index, roll_deg=roll, mirrored=mirrored,
                )
                for index, frame in enumerate(_SWING_SPEC)
            ]
            return run_detector(TORSO_TWIST_DETECTOR, frames, 30.0, "front", 0.8).detections

        decisive = ("fault_id", "severity", "confidence", "observability", "start_frame",
                    "end_frame", "peak_frame", "phase", "rep_count", "occurred_reps")
        reference = run(0.0, False)
        self.assertTrue(reference)
        for roll, mirrored in ((37.0, False), (0.0, True), (90.0, True)):
            with self.subTest(roll=roll, mirrored=mirrored):
                other = run(roll, mirrored)
                self.assertEqual(len(other), len(reference))
                for got, want in zip(other, reference):
                    for field in decisive:
                        self.assertEqual(getattr(got, field), getattr(want, field), msg=field)
                    self.assertEqual(set(got.evidence), set(want.evidence))
                    for key, value in want.evidence.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            # One unit in the last place of `build_detection`'s 2-decimal
                            # rounding, and no more.
                            self.assertLessEqual(abs(got.evidence[key] - value), 0.01 + 1e-9, msg=key)
                        else:
                            self.assertEqual(got.evidence[key], value, msg=key)


def _swing_spec(swings: int = 3, per: int = 24, peak: float = 0.55, deviation: float = 30.0):
    spec = []
    for swing in range(swings):
        direction = 1.0 if swing % 2 == 0 else -1.0
        for step in range(per):
            fraction = math.sin(math.pi * step / (per - 1))
            spec.append({"_angle": 95.0 + deviation * fraction, "_twist": direction * peak * fraction})
    return spec


_SWING_SPEC = _swing_spec()


class PhaseTest(unittest.TestCase):
    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(torso_twist_assign_phases([]), [])

    def test_a_clip_with_no_finite_signal_is_entirely_unknown(self) -> None:
        raw = torso_twist_compute_raw([twist_frame(drop_landmark=15) for _ in range(10)], 30.0)
        self.assertEqual(set(torso_twist_assign_phases(raw)), {"unknown"})

    def test_an_invalid_frame_inside_the_setup_window_is_unknown_not_setup(self) -> None:
        frames = [twist_frame(twist_ratio=0.1 * i, frame_index=i) for i in range(20)]
        frames[0] = twist_frame(drop_landmark=16, frame_index=0)
        phases = torso_twist_assign_phases(torso_twist_compute_raw(frames, 30.0))
        self.assertEqual(phases[0], "unknown")
        self.assertEqual(phases[1], "setup")

    def test_the_phases_run_setup_rotate_peak_return(self) -> None:
        frames = [
            twist_frame(twist_ratio=0.55 * math.sin(math.pi * i / 23), frame_index=i)
            for i in range(24)
        ]
        phases = torso_twist_assign_phases(torso_twist_compute_raw(frames, 30.0))
        self.assertEqual(phases[0], "setup")
        self.assertIn("rotate", phases)
        self.assertIn("peak", phases)
        self.assertIn("return", phases)
        self.assertEqual(phases[phases.index("peak")], "peak")
        self.assertLess(phases.index("rotate"), phases.index("peak"))
        self.assertLess(phases.index("peak"), phases.index("return"))


class BraceRuleTest(unittest.TestCase):
    def _window(self, deviation: float, setup_frames: int = 6, total: int = 24) -> list[CoreFrame]:
        frames = [twist_frame(trunk_thigh_deg=95.0, frame_index=i) for i in range(setup_frames)]
        frames += [
            twist_frame(trunk_thigh_deg=95.0 + deviation, frame_index=setup_frames + i)
            for i in range(total - setup_frames)
        ]
        return _core_with_setup(frames, setup_frames)

    def test_it_fires_above_the_threshold_and_is_silent_below(self) -> None:
        self.assertEqual(rule_trunk_not_braced(self._window(BRACE_MILD_DEG - 1.0), _ctx()), [])
        fired = rule_trunk_not_braced(self._window(BRACE_MILD_DEG + 5.0), _ctx())
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].fault_id, "tt_trunk_not_braced")

    def test_severity_ramps_between_the_two_thresholds(self) -> None:
        mild = rule_trunk_not_braced(self._window(BRACE_MILD_DEG + 0.5), _ctx())[0]
        severe = rule_trunk_not_braced(self._window(BRACE_SEVERE_DEG + 10.0), _ctx())[0]
        self.assertLess(mild.severity, 0.15)
        self.assertEqual(severe.severity, 1.0)
        self.assertEqual(severe.confidence, severe.severity)

    def test_it_reads_the_thighs_and_not_the_image_vertical(self) -> None:
        """The distinguishing property against the parent spec's own construction, which is a
        trunk angle "relative to VERTICAL". A rolled camera moves an image-vertical reading and
        must not move this one."""
        upright = rule_trunk_not_braced(self._window(BRACE_MILD_DEG + 8.0), _ctx())
        frames = [twist_frame(trunk_thigh_deg=95.0, frame_index=i, roll_deg=41.0) for i in range(6)]
        frames += [
            twist_frame(trunk_thigh_deg=95.0 + BRACE_MILD_DEG + 8.0, frame_index=6 + i, roll_deg=41.0)
            for i in range(18)
        ]
        rolled = rule_trunk_not_braced(_core_with_setup(frames, 6), _ctx())
        self.assertEqual(len(upright), 1)
        self.assertEqual(rolled, upright)

    def test_a_brace_lost_before_the_rep_opens_is_invisible(self) -> None:
        """PUSH-UP'S BLINDNESS, INHERITED AND PINNED. A baseline measures CHANGE, not POSTURE.
        A twister who sets up already collapsed and holds it is never flagged -- and here that
        blind spot is the whole verdict, because this is the detector's only live rule."""
        held = _core_with_setup(
            [twist_frame(trunk_thigh_deg=95.0 + 45.0, frame_index=i) for i in range(24)], 6
        )
        self.assertEqual(rule_trunk_not_braced(held, _ctx()), [])

    def test_a_window_with_no_setup_frame_is_silent_rather_than_baselined_on_the_whole_rep(self) -> None:
        self.assertEqual(rule_trunk_not_braced(_core(swing_clip(1)), _ctx()), [])

    def test_a_window_shorter_than_min_frames_does_not_fire(self) -> None:
        window = self._window(BRACE_SEVERE_DEG, setup_frames=1, total=2)
        self.assertEqual(rule_trunk_not_braced(window, _ctx(min_frames=3)), [])

    def test_the_evidence_reports_the_baseline_and_the_sag(self) -> None:
        fired = rule_trunk_not_braced(self._window(22.0), _ctx())[0]
        self.assertAlmostEqual(fired.evidence["setup_trunk_thigh_angle_deg"], 95.0, places=2)
        self.assertAlmostEqual(fired.evidence["max_trunk_sag_deg"], 22.0, places=2)
        self.assertEqual(fired.evidence["threshold_deg"], BRACE_MILD_DEG)

    def test_a_twister_who_TIGHTENS_is_not_told_they_lost_the_brace(self) -> None:
        """THE FALSE-POSITIVE DIRECTION OF THE BASELINE, WHICH THE FIRST IMPLEMENTATION HAD AND
        NO GREEN TEST CAUGHT.

        `trunk_thigh_angle_deg` is monotone in sag -- larger means the torso has laid further
        back toward the floor -- so an UNSIGNED deviation from the baseline fires on the opposite
        of the fault. Measured on the shipped path before the fix: a twister setting up loose at
        95 deg and then tightening to 50 deg for the swing was reported "Braced Torso Lost" at
        severity 1.0, quoting a 45 deg deviation.

        And the baseline makes that the ORDINARY case rather than an edge one: `setup` is the
        window's first 15%, i.e. the frames BEFORE the subject braces. Set up loose, brace, swing.

        `pushup_head_drop` recorded this exact inversion ("a baseline on the unsigned angle is not
        merely non-directional but actively inverted") and had to add a signed metric for it; here
        the sign lives in the comparison, so the fix costs nothing. This test is the mirror of
        `test_a_brace_lost_before_the_rep_opens_is_invisible` -- that one pins what the baseline
        cannot see, this one pins what it must not invent.
        """
        frames = [twist_frame(trunk_thigh_deg=95.0, frame_index=i) for i in range(6)]
        frames += [twist_frame(trunk_thigh_deg=50.0, frame_index=6 + i) for i in range(18)]
        self.assertEqual(rule_trunk_not_braced(_core_with_setup(frames, 6), _ctx()), [])

    def test_it_still_fires_on_the_sag_direction_so_the_test_above_is_not_vacuous(self) -> None:
        frames = [twist_frame(trunk_thigh_deg=95.0, frame_index=i) for i in range(6)]
        frames += [twist_frame(trunk_thigh_deg=140.0, frame_index=6 + i) for i in range(18)]
        fired = rule_trunk_not_braced(_core_with_setup(frames, 6), _ctx())
        self.assertEqual(len(fired), 1)
        self.assertAlmostEqual(fired[0].evidence["max_trunk_sag_deg"], 45.0, places=2)

    def test_the_citation_records_that_mcgill_never_mentions_this_exercise(self) -> None:
        """The parent spec marks this rule VERIFIED. Reading McGill in place shows he measured a
        laboratory axial-torque protocol and named no exercise, no range and no tolerance; the
        shipped `citation_support` has to say so where a reader will meet it."""
        support = rule_trunk_not_braced(self._window(22.0), _ctx())[0].citation_support
        self.assertIn("never mentions this exercise", support)
        self.assertIn("states no range, no tolerance and no fault threshold", support)
        self.assertIn("is not the number applied here", support)

    def test_no_view_gate_and_no_view_discount(self) -> None:
        """Fourth module in a row with neither. Leg Abduction measured the front/rear/oblique
        labels systematically inverted even on an upright subject; a seated twister sits between
        two regimes in both of which they have been measured wrong."""
        window = self._window(22.0)
        outputs = {view: rule_trunk_not_braced(window, _ctx(view)) for view in
                   ("front", "side", "rear", "front_oblique", "unknown")}
        first = outputs["front"]
        self.assertEqual(len(first), 1)
        for view, detections in outputs.items():
            self.assertEqual(detections, first, msg=view)


class SilentRomRuleTest(unittest.TestCase):
    def test_the_rom_rule_never_fires_even_on_a_repetition_that_trips_the_specs_cut(self) -> None:
        """The parent spec says to flag a swing whose hands fail to travel past the hip midline
        by more than ~0.08 of shoulder width. This builds exactly that repetition."""
        shallow = _core(
            [twist_frame(twist_ratio=0.03 * math.sin(math.pi * i / 23), frame_index=i) for i in range(24)]
        )
        peak = max(abs(frame.m("twist_offset_ratio")) for frame in shallow)
        self.assertLess(peak, 0.08)
        self.assertEqual(rule_insufficient_rotation_rom(shallow, _ctx()), [])

    def test_it_is_registered_rather_than_absent(self) -> None:
        """Registered so the spec and the code stay 1:1 and an auditor meets the rule with its
        reasoning instead of meeting a gap and closing it with an invented threshold."""
        self.assertIn(rule_insufficient_rotation_rom, TORSO_TWIST_DETECTOR.rules)


class RegistrationTest(unittest.TestCase):
    def test_the_detector_is_registered_under_its_canonical_name(self) -> None:
        self.assertIs(get_detector("Torso Twist"), TORSO_TWIST_DETECTOR)

    def test_it_is_the_fourteenth_of_sixteen(self) -> None:
        names = [detector.name for detector in list_detectors()]
        self.assertEqual(names[13], "Torso Twist")
        self.assertEqual(len(names), 14)

    def test_it_ships_beta(self) -> None:
        self.assertFalse(TORSO_TWIST_DETECTOR.validated)

    def test_the_rep_signal_is_a_declared_metric_key(self) -> None:
        self.assertIn(TORSO_TWIST_DETECTOR.rep_signal, TORSO_TWIST_METRIC_KEYS)

    def test_it_is_the_first_user_of_the_rectified_rep_signal_hook(self) -> None:
        """`base.py`'s `rep_rectify` was written with this movement named in its comment and has
        had no user until now."""
        self.assertTrue(TORSO_TWIST_DETECTOR.rep_rectify)
        self.assertEqual(TORSO_TWIST_DETECTOR.rep_polarity, "max")
        others = [d for d in list_detectors() if d.name != "Torso Twist" and d.rep_rectify]
        self.assertEqual(others, [])


class EndToEndSegmentationTest(unittest.TestCase):
    def test_alternating_swings_segment_and_score_through_run_detector(self) -> None:
        result = run_detector(TORSO_TWIST_DETECTOR, swing_clip(3, brace_deviation_deg=30.0), 30.0, "front", 0.8)
        self.assertIsNone(result.fallback)
        self.assertGreaterEqual(len(result.reps), 3)
        self.assertEqual([d.fault_id for d in result.detections], ["tt_trunk_not_braced"])

    def test_a_clean_clip_produces_no_detections_and_really_was_scored(self) -> None:
        """Non-vacuous: it asserts the clip segmented and was handed to the rules, so the empty
        result is a verdict rather than a pipeline that never ran."""
        result = run_detector(TORSO_TWIST_DETECTOR, swing_clip(3, brace_deviation_deg=0.0), 30.0, "front", 0.8)
        self.assertIsNone(result.fallback)
        self.assertTrue(result.analyzed)
        self.assertEqual(result.detections, [])

    def test_the_peak_phase_lands_on_the_most_twisted_frames(self) -> None:
        result = run_detector(TORSO_TWIST_DETECTOR, swing_clip(3), 30.0, "front", 0.8)
        peaks = [f for f in result.core if f.phase == "peak"]
        others = [f for f in result.core if f.phase in {"rotate", "return"}]
        self.assertTrue(peaks and others)
        self.assertGreater(
            min(abs(f.m("twist_offset_ratio")) for f in peaks),
            np.median([abs(f.m("twist_offset_ratio")) for f in others]),
        )

    def test_a_brace_held_from_setup_is_invisible_end_to_end_too(self) -> None:
        result = run_detector(
            TORSO_TWIST_DETECTOR, swing_clip(3, brace_deviation_deg=45.0, brace_from_setup=True),
            30.0, "front", 0.8,
        )
        self.assertEqual(result.detections, [])


class EffectiveThresholdTest(unittest.TestCase):
    """ROW'S SETUP-BASELINE DEFECT, CHECKED FOR HERE RATHER THAN INHERITED AS A FACTOR.

    `row_torso_rising`'s effective threshold is exactly 2x its nominal one on the segmented path,
    because `segment_reps` trims the window to the excursion and leaves a 2-frame `setup` slice
    of which one frame is already loaded. This rule has the same shape -- a baseline over the
    window's `setup` frames -- so the trap was measured rather than assumed, AND THE MEASUREMENT
    SEPARATES THE TWO MECHANISMS THAT CAN INFLATE IT.

    Result: the effective threshold is 18.0 deg against a nominal 15.0, an inflation of 1.20x --
    and NONE of it is Row's trimming, because on this fixture `segment_reps` trims nothing (the
    swings begin and end at exactly zero, so the excursion IS the window). All of it is the
    `setup` slice already carrying part of the ramp, which is a milder and different defect.
    That is a statement about this fixture, not a proof that the trimming cannot bite on real
    footage where a swing does not start from rest.
    """

    @staticmethod
    def _fires(deviation: float) -> bool:
        result = run_detector(
            TORSO_TWIST_DETECTOR, swing_clip(3, brace_deviation_deg=deviation), 30.0, "front", 0.8
        )
        return bool(result.detections)

    def test_the_effective_threshold_is_measured_and_inflated(self) -> None:
        smallest = next(
            (d for d in np.arange(BRACE_MILD_DEG, 4.0 * BRACE_MILD_DEG, 0.5) if self._fires(float(d))),
            None,
        )
        self.assertIsNotNone(smallest, "the rule never fired across the swept range")
        # PINNED. A change to `segment_reps`, to the smoothing window, or to this rule's baseline
        # will move this number, and the next reader should meet it rather than rediscover the
        # mechanism.
        self.assertAlmostEqual(float(smallest), 18.0, places=6)
        self.assertAlmostEqual(float(smallest) / BRACE_MILD_DEG, 1.2, places=6)

    def test_segmentation_trims_nothing_here_so_the_inflation_is_the_setup_slice_alone(self) -> None:
        """The half of the diagnosis that makes the 1.20x attributable. Row's 2x came from
        trimming; if the windows are untrimmed, whatever inflation remains cannot be."""
        frames = swing_clip(3, brace_deviation_deg=30.0)
        result = run_detector(TORSO_TWIST_DETECTOR, frames, 30.0, "front", 0.8)
        self.assertEqual(
            [(rep.start, rep.end) for rep in result.reps], [(0, 23), (24, 47), (48, 71)]
        )
        self.assertTrue(all(not rep.partial for rep in result.reps))

    def test_the_residual_inflation_is_derived_and_not_merely_observed(self) -> None:
        """`setup` is the window's first 15% -- 3 of 24 frames -- and the fixture's deviation
        ramps as `sin(pi * step / 23)`, so the baseline is the median of the ramp at steps 0, 1
        and 2 rather than zero. The measured peak is therefore `(1 - f) * D` for that median
        fraction `f`, and the effective cut is `15 / (1 - f)`. Asserting the algebra against the
        unsmoothed rule shows the 18.0 above is 15/(1-f) plus the framework's median-5 smoothing,
        not an unexplained number.
        """
        fraction = float(np.median([math.sin(math.pi * step / 23) for step in range(3)]))
        derived = BRACE_MILD_DEG / (1.0 - fraction)

        def fires_unsmoothed(deviation: float) -> bool:
            frames = swing_clip(1, brace_deviation_deg=deviation)
            raw = torso_twist_compute_raw(frames, 30.0)
            phases = torso_twist_assign_phases(raw)
            core = [
                CoreFrame(
                    frame_index=int(item.get("frame_index", index)),
                    time=float(item.get("time", 0.0)),
                    phase=phases[index],
                    valid=bool(item.get("valid")),
                    lower_body_visibility=float(item.get("lower_body_visibility", 0.0)),
                    metrics={k: float(item.get(k, math.nan)) for k in TORSO_TWIST_METRIC_KEYS},
                )
                for index, item in enumerate(raw)
            ]
            return bool(rule_trunk_not_braced(core, _ctx()))

        smallest = next(
            d for d in np.arange(BRACE_MILD_DEG, 4.0 * BRACE_MILD_DEG, 0.5)
            if fires_unsmoothed(float(d))
        )
        self.assertAlmostEqual(float(smallest), 17.5, places=6)
        # The derived cut is 17.36; 17.5 is the smallest step of the sweep past it.
        self.assertAlmostEqual(derived, 17.36, places=2)
        self.assertLess(derived, float(smallest))
        self.assertLess(float(smallest) - derived, 0.5)


if __name__ == "__main__":
    unittest.main()
