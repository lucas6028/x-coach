import math
import unittest

import numpy as np

from src.pose.movements.base import CoreFrame, RuleContext, run_detector
from src.pose.movements.registry import list_detectors
from src.pose.movements.jumping_jacks import (
    JUMPING_JACKS_DETECTOR,
    JUMPING_JACKS_METRIC_KEYS,
    LEG_ROM_MILD_RATIO,
    jumping_jacks_assign_phases,
    jumping_jacks_compute_raw,
    rule_incomplete_arm_rom,
    rule_incomplete_leg_rom,
)

_HIP_MID = (0.50, 0.60)
_SHOULDER_HALF_WIDTH = 0.07
_TRUNK_LEN = 0.22
_THIGH_LEN = 0.16
_SHANK_LEN = 0.16
_HEAD_ABOVE_SHOULDER = 0.09

# Landmark pairs that must swap identity when the image is mirrored: a subject filmed from
# behind has their anatomical left on the other side of the frame.
_MIRROR_PAIRS = ((11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32))


def _lm(x: float, y: float, z: float = 0.0, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": z, "visibility": visibility}


def _rot(vector: tuple[float, float], degrees: float) -> tuple[float, float]:
    rad = math.radians(degrees)
    cos, sin = math.cos(rad), math.sin(rad)
    return (vector[0] * cos - vector[1] * sin, vector[0] * sin + vector[1] * cos)


def jack_frame(
    stance_ratio: float = 1.6,
    valgus_ratio: float = 1.0,
    hands_ratio: float = 0.6,
    frame_index: int = 0,
    drop_landmark: int | None = None,
    roll_deg: float = 0.0,
    mirrored: bool = False,
) -> dict:
    """One JUMPING JACK frame, built so each knob controls its metric BY CONSTRUCTION.

    Every landmark carries z=0, so the module's 2-D reads see exactly the image-plane geometry --
    the same reason tests/test_situp.py and tests/test_torso_twist.py need no depth correction.
    The subject is upright: the trunk axis runs from the hip midpoint to the shoulder midpoint,
    straight up the image before any roll is applied.

    Knobs:

      stance_ratio  -- ankle separation in shoulder widths. `stance_width_ratio` equals it
                       exactly. This is the rep signal: ~0.2 with the feet together, ~1.6 at a
                       full side-straddle.
      valgus_ratio  -- knee separation as a fraction of ankle separation, i.e. the quantity the
                       WITHDRAWN valgus rule read. No metric key carries it any more; it is kept
                       as a knob so `StanceGeometryConfoundTest` can demonstrate WHY the rule was
                       withdrawn on synthetic geometry as well as on real footage.
      hands_ratio   -- how far the wrist midpoint sits above the nose ALONG THE TRUNK AXIS, in
                       shoulder widths. `hands_above_head_ratio` equals it exactly; negative means
                       the hands never got above the head.
      drop_landmark -- zero one landmark's visibility. Dropping one of the EIGHT required
                       landmarks invalidates the frame; dropping a wrist or the nose must not,
                       which is this module's departure from every earlier one.
      roll_deg      -- rotate EVERY landmark about the image centre. Both metrics are a ratio of
                       distances or a dot product onto a body axis, so this must change nothing.
      mirrored      -- reflect about the image vertical AND swap every left/right pair, i.e. the
                       subject filmed from behind. Nothing here may change: unlike Torso Twist's
                       signed twist offset, neither metric names a body side.
    """
    trunk_up = (0.0, -1.0)  # image y grows downward, so "up" is negative y

    shoulder_mid = (_HIP_MID[0] + _TRUNK_LEN * trunk_up[0], _HIP_MID[1] + _TRUNK_LEN * trunk_up[1])
    shoulder_width = 2.0 * _SHOULDER_HALF_WIDTH
    shoulder = {
        "left": [shoulder_mid[0] + _SHOULDER_HALF_WIDTH, shoulder_mid[1]],
        "right": [shoulder_mid[0] - _SHOULDER_HALF_WIDTH, shoulder_mid[1]],
    }
    hip = {
        "left": [_HIP_MID[0] + 0.05, _HIP_MID[1]],
        "right": [_HIP_MID[0] - 0.05, _HIP_MID[1]],
    }

    ankle_half = 0.5 * stance_ratio * shoulder_width
    knee_half = 0.5 * valgus_ratio * stance_ratio * shoulder_width
    knee = {
        "left": [_HIP_MID[0] + knee_half, _HIP_MID[1] + _THIGH_LEN],
        "right": [_HIP_MID[0] - knee_half, _HIP_MID[1] + _THIGH_LEN],
    }
    ankle = {
        "left": [_HIP_MID[0] + ankle_half, _HIP_MID[1] + _THIGH_LEN + _SHANK_LEN],
        "right": [_HIP_MID[0] - ankle_half, _HIP_MID[1] + _THIGH_LEN + _SHANK_LEN],
    }

    nose = [shoulder_mid[0], shoulder_mid[1] - _HEAD_ABOVE_SHOULDER]
    # The wrist midpoint sits `hands_ratio` shoulder widths above the nose along the trunk axis.
    # The two wrists are placed symmetrically ACROSS that axis, so their midpoint is exact and
    # the sideways placement cannot leak into the dot product.
    hand_mid = (nose[0], nose[1] - hands_ratio * shoulder_width)
    wrist = {
        "left": [hand_mid[0] + 0.06, hand_mid[1]],
        "right": [hand_mid[0] - 0.06, hand_mid[1]],
    }

    points = {
        0: nose,
        11: shoulder["left"], 12: shoulder["right"],
        15: wrist["left"], 16: wrist["right"],
        23: hip["left"], 24: hip["right"],
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


def jack_clip(
    reps: int = 3,
    frames_per_rep: int = 20,
    open_stance_ratio: float = 1.6,
    closed_stance_ratio: float = 0.2,
    hands_at_open: float = 0.6,
) -> list[dict]:
    """`reps` open-and-close cycles, feet together -> wide -> feet together."""
    frames: list[dict] = []
    for _ in range(reps):
        for step in range(frames_per_rep):
            fraction = math.sin(math.pi * step / (frames_per_rep - 1))
            frames.append(
                jack_frame(
                    stance_ratio=closed_stance_ratio
                    + (open_stance_ratio - closed_stance_ratio) * fraction,
                    hands_ratio=-0.4 + (hands_at_open + 0.4) * fraction,
                    frame_index=len(frames),
                )
            )
    return frames


def _core(frames: list[dict], phase: str = "open") -> list[CoreFrame]:
    raw = jumping_jacks_compute_raw(frames, 30.0)
    core = []
    for index, item in enumerate(raw):
        core.append(
            CoreFrame(
                frame_index=int(item.get("frame_index", index)),
                time=float(item.get("time", 0.0)),
                phase=phase if item.get("valid") else "unknown",
                valid=bool(item.get("valid")),
                lower_body_visibility=float(item.get("lower_body_visibility", 0.0)),
                metrics={key: float(item.get(key, math.nan)) for key in JUMPING_JACKS_METRIC_KEYS},
            )
        )
    return core


def _ctx(view_type: str = "front", min_frames: int = 3) -> RuleContext:
    return RuleContext(fps=30.0, view_type=view_type, view_confidence=0.8, min_frames=min_frames)


class MetricLayerTest(unittest.TestCase):
    def test_metric_keys_match_the_emitted_metrics_exactly(self) -> None:
        """Two-way: a key the tuple omits is silently NaN in every rule, and a key the tuple
        names but nothing emits is silently NaN too."""
        raw = jumping_jacks_compute_raw([jack_frame()], 30.0)[0]
        framework = {"frame_index", "time", "valid", "lower_body_visibility"}
        emitted = set(raw) - framework
        self.assertEqual(emitted, set(JUMPING_JACKS_METRIC_KEYS))

    def test_the_knobs_control_their_metrics_exactly(self) -> None:
        raw = jumping_jacks_compute_raw(
            [jack_frame(stance_ratio=1.45, hands_ratio=0.33)], 30.0
        )[0]
        self.assertAlmostEqual(raw["stance_width_ratio"], 1.45, places=5)
        self.assertAlmostEqual(raw["hands_above_head_ratio"], 0.33, places=5)

    def test_the_hand_height_is_signed_so_hands_below_the_head_read_negative(self) -> None:
        low = jumping_jacks_compute_raw([jack_frame(hands_ratio=-0.5)], 30.0)[0]
        self.assertLess(low["hands_above_head_ratio"], 0.0)

    def test_the_stance_ratio_is_a_distance_ratio_not_an_image_x_difference(self) -> None:
        """THE PARENT SPEC WRITES THIS METRIC AS `|x27-x28| / |x11-x12|`, AND THAT IS THE
        DISTINGUISHING PROPERTY THIS ASSERTS AWAY FROM.

        An image-x difference collapses under camera roll -- at 90 degrees the ankles share an x
        coordinate and the ratio reads 0 -- while the distance form is unchanged. The two agree
        exactly when the camera is upright, which is asserted first so the test is not just
        measuring a rewrite.
        """
        upright = jack_frame(stance_ratio=1.6)
        points = np.array([[lm["x"], lm["y"]] for lm in upright["landmarks"]], dtype=np.float64)
        image_x_form = abs(points[27][0] - points[28][0]) / abs(points[11][0] - points[12][0])
        self.assertAlmostEqual(image_x_form, 1.6, places=5)

        rolled = jack_frame(stance_ratio=1.6, roll_deg=90.0)
        rolled_points = np.array(
            [[lm["x"], lm["y"]] for lm in rolled["landmarks"]], dtype=np.float64
        )
        rolled_image_x_form = abs(rolled_points[27][0] - rolled_points[28][0]) / max(
            abs(rolled_points[11][0] - rolled_points[12][0]), 1e-9
        )
        self.assertLess(rolled_image_x_form, 0.01)
        self.assertAlmostEqual(
            jumping_jacks_compute_raw([rolled], 30.0)[0]["stance_width_ratio"], 1.6, places=5
        )


class ValidityGateTest(unittest.TestCase):
    """NINE LANDMARKS ARE READ AND ONLY EIGHT ARE REQUIRED -- the module's one structural
    departure from every earlier detector."""

    def test_dropping_a_required_landmark_invalidates_the_frame(self) -> None:
        for index in (11, 12, 23, 24, 25, 26, 27, 28):
            with self.subTest(landmark=index):
                raw = jumping_jacks_compute_raw([jack_frame(drop_landmark=index)], 30.0)[0]
                self.assertFalse(raw["valid"])

    def test_dropping_a_wrist_or_the_nose_leaves_the_leg_metric_intact(self) -> None:
        """The hands are the fastest-moving landmarks in this movement. Requiring them would let
        a motion-blurred hand silence a rule that never reads them."""
        for index in (0, 15, 16):
            with self.subTest(landmark=index):
                raw = jumping_jacks_compute_raw([jack_frame(drop_landmark=index)], 30.0)[0]
                self.assertTrue(raw["valid"])
                self.assertAlmostEqual(raw["stance_width_ratio"], 1.6, places=5)
                self.assertTrue(math.isnan(raw["hands_above_head_ratio"]))

    def test_one_missing_wrist_is_no_reading_rather_than_half_a_reading(self) -> None:
        raw = jumping_jacks_compute_raw([jack_frame(drop_landmark=15)], 30.0)[0]
        self.assertTrue(math.isnan(raw["hands_above_head_ratio"]))


class InvarianceTest(unittest.TestCase):
    def test_every_metric_is_invariant_under_camera_roll(self) -> None:
        """The Group E re-anchoring mandate, carried into Group F: no metric here may reference
        the image vertical or the image horizontal."""
        upright = jumping_jacks_compute_raw(
            [jack_frame(stance_ratio=1.45, hands_ratio=0.33)], 30.0
        )[0]
        for roll in (17.0, -33.0, 90.0, 180.0):
            with self.subTest(roll=roll):
                rolled = jumping_jacks_compute_raw(
                    [jack_frame(stance_ratio=1.45, hands_ratio=0.33, roll_deg=roll)], 30.0
                )[0]
                for key in JUMPING_JACKS_METRIC_KEYS:
                    self.assertAlmostEqual(rolled[key], upright[key], places=5, msg=key)

    def test_every_metric_survives_mirroring_including_its_sign(self) -> None:
        """CONTRAST WITH TORSO TWIST, WHICH COULD NOT MAKE THIS CLAIM. Its `twist_offset_ratio`
        names which way the hands went IN THE IMAGE, so mirroring flips its sign and only the
        magnitude is honest. Neither metric here names a body side: one is a ratio of distances
        and the other a projection onto the trunk axis, which mirroring leaves alone."""
        normal = jumping_jacks_compute_raw(
            [jack_frame(stance_ratio=1.45, hands_ratio=0.33)], 30.0
        )[0]
        mirrored = jumping_jacks_compute_raw(
            [jack_frame(stance_ratio=1.45, hands_ratio=0.33, mirrored=True)], 30.0
        )[0]
        for key in JUMPING_JACKS_METRIC_KEYS:
            self.assertAlmostEqual(mirrored[key], normal[key], places=5, msg=key)

    def test_the_metrics_survive_roll_and_mirroring_through_segmentation(self) -> None:
        """The invariance must survive the whole pipeline, not merely one frame. There are no
        detections to compare -- every rule is silent -- so what is compared is the segmentation
        and the per-frame metric series `run_detector` builds."""
        def run(roll_deg: float, mirrored: bool):
            frames = [
                jack_frame(
                    stance_ratio=item, frame_index=index, roll_deg=roll_deg, mirrored=mirrored
                )
                for index, item in enumerate(_stance_profile())
            ]
            result = run_detector(JUMPING_JACKS_DETECTOR, frames, 30.0, "front", 0.8)
            return (
                [(rep.start, rep.end) for rep in result.reps],
                # 5 decimals, not exact: rotating every landmark about the image centre is a
                # float operation, so the ratio agrees to ~1e-6 rather than bit-for-bit. The
                # tolerance IS that rounding, stated rather than papered over.
                [round(frame.m("stance_width_ratio"), 5) for frame in result.core],
            )

        base = run(0.0, False)
        self.assertTrue(base[0], "non-vacuity: the clip must really have segmented")
        for roll, mirrored in ((23.0, False), (0.0, True), (-41.0, True)):
            with self.subTest(roll=roll, mirrored=mirrored):
                self.assertEqual(base, run(roll, mirrored))


def _stance_profile() -> list[float]:
    profile = []
    for _ in range(3):
        for step in range(20):
            profile.append(0.2 + 1.4 * math.sin(math.pi * step / 19))
    return profile


class PhaseTest(unittest.TestCase):
    def test_phases_run_setup_opening_open_closing(self) -> None:
        raw = jumping_jacks_compute_raw(jack_clip(reps=1, frames_per_rep=20), 30.0)
        phases = jumping_jacks_assign_phases(raw)
        self.assertEqual(len(phases), len(raw))
        self.assertEqual(phases[0], "setup")
        self.assertIn("opening", phases)
        self.assertIn("open", phases)
        self.assertIn("closing", phases)

    def test_the_open_phase_is_where_the_stance_is_widest(self) -> None:
        raw = jumping_jacks_compute_raw(jack_clip(reps=1, frames_per_rep=20), 30.0)
        phases = jumping_jacks_assign_phases(raw)
        widths = [item["stance_width_ratio"] for item in raw]
        self.assertEqual(phases[int(np.argmax(widths))], "open")

    def test_an_empty_clip_and_a_signal_free_clip_do_not_raise(self) -> None:
        self.assertEqual(jumping_jacks_assign_phases([]), [])
        blind = jumping_jacks_compute_raw([jack_frame(drop_landmark=27)] * 5, 30.0)
        self.assertEqual(jumping_jacks_assign_phases(blind), ["unknown"] * 5)

    def test_an_invalid_frame_inside_the_setup_slice_is_unknown_not_setup(self) -> None:
        frames = jack_clip(reps=1, frames_per_rep=20)
        frames[1] = jack_frame(drop_landmark=27, frame_index=1)
        phases = jumping_jacks_assign_phases(jumping_jacks_compute_raw(frames, 30.0))
        self.assertEqual(phases[1], "unknown")


class SilentLegRomRuleTest(unittest.TestCase):
    """PERMANENTLY SILENT, and silenced by the labeled data rather than by an argument."""

    def test_it_never_fires_even_on_a_repetition_far_below_the_specs_cut(self) -> None:
        core = _core([jack_frame(stance_ratio=0.5, frame_index=i) for i in range(20)])
        self.assertEqual(rule_incomplete_leg_rom(core, _ctx()), [])

    def test_the_metric_it_would_have_read_is_computed_and_correct(self) -> None:
        """SILENT IS NOT BROKEN -- and here the metric is also the rep signal, so it has to work
        regardless."""
        narrow = jumping_jacks_compute_raw([jack_frame(stance_ratio=0.9)], 30.0)[0]
        wide = jumping_jacks_compute_raw([jack_frame(stance_ratio=1.7)], 30.0)[0]
        self.assertLess(narrow["stance_width_ratio"], LEG_ROM_MILD_RATIO)
        self.assertGreater(wide["stance_width_ratio"], LEG_ROM_MILD_RATIO)

    def test_the_specs_cut_is_kept_where_it_is_rather_than_moved(self) -> None:
        """The measured correct population sits BELOW 1.3, so a cut fitted to it could be
        manufactured at will. Silencing rather than moving is the whole point; this pins the
        constant so a later edit that quietly retunes it has to change a test that says why."""
        self.assertEqual(LEG_ROM_MILD_RATIO, 1.3)


class SilentArmRuleTest(unittest.TestCase):
    def test_the_arm_rule_never_fires_even_when_the_hands_stay_at_shoulder_height(self) -> None:
        """The exact case the parent spec says to flag: the hands never reach the head."""
        core = _core([jack_frame(hands_ratio=-0.9, frame_index=i) for i in range(20)])
        self.assertEqual(rule_incomplete_arm_rom(core, _ctx()), [])

    def test_the_metric_it_would_have_used_is_computed_and_correct(self) -> None:
        """§6's claim is that this rule's SENSOR is fine and only its corroboration is missing.
        That has to be checkable, which is why the metric is emitted even though no rule reads
        it -- the first time in this registry that a metric exists solely for a silent rule."""
        overhead = jumping_jacks_compute_raw([jack_frame(hands_ratio=0.8)], 30.0)[0]
        short = jumping_jacks_compute_raw([jack_frame(hands_ratio=-0.6)], 30.0)[0]
        self.assertGreater(overhead["hands_above_head_ratio"], 0.0)
        self.assertLess(short["hands_above_head_ratio"], 0.0)

    def test_both_silent_rules_are_present_rather_than_absent(self) -> None:
        self.assertEqual(
            JUMPING_JACKS_DETECTOR.rules, (rule_incomplete_leg_rom, rule_incomplete_arm_rom)
        )


class StanceGeometryConfoundTest(unittest.TestCase):
    """WHY `jj_knee_valgus_landing` IS WITHDRAWN, demonstrated on synthetic geometry.

    The rule read `knee_width / ankle_width < 0.82`. On real footage a zero-parameter control --
    replacing both knees with their projections onto the same-side hip->ankle line, i.e. a
    PERFECTLY straight limb -- still falls below that cut on 66.4% of open-phase frames
    (notes/jumping-jacks-rule-validation.md). These cases pin the mechanism itself, which is that
    the ratio is a function of the STANCE, so it cannot be read as a statement about the knees.
    """

    @staticmethod
    def _aligned_ratio(stance_ratio: float) -> float:
        """The knee/ankle ratio for a perfectly straight limb at this stance width."""
        frame = jack_frame(stance_ratio=stance_ratio)
        points = np.array([[lm["x"], lm["y"]] for lm in frame["landmarks"]], dtype=np.float64)
        ratio = []
        for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)):
            limb = points[ankle] - points[hip]
            fraction = float(np.dot(points[knee] - points[hip], limb)) / float(np.dot(limb, limb))
            ratio.append(points[hip] + fraction * limb)
        return float(np.linalg.norm(ratio[0] - ratio[1])) / float(
            np.linalg.norm(points[27] - points[28])
        )

    def test_a_perfectly_aligned_knee_trips_the_withdrawn_cut_at_a_real_jack_stance(self) -> None:
        """At a full side-straddle the legs splay from a pelvis that does not widen, so a
        correctly tracking knee sits well inside the ankle line -- below 0.82 with no valgus at
        all. THIS IS THE WITHDRAWAL."""
        self.assertLess(self._aligned_ratio(1.6), 0.82)

    def test_and_the_same_construction_is_fine_at_a_squat_stance(self) -> None:
        """Non-vacuity, and the reason the number is defensible for squats: at about shoulder
        width a straight limb keeps the knees essentially over the ankles."""
        self.assertGreater(self._aligned_ratio(1.0), 0.82)

    def test_the_confound_is_monotone_in_the_stance_the_movement_is_defined_by(self) -> None:
        ratios = [self._aligned_ratio(width) for width in (1.0, 1.3, 1.6, 1.9)]
        self.assertEqual(ratios, sorted(ratios, reverse=True))


class WithdrawnRulesTest(unittest.TestCase):
    def test_no_withdrawn_rule_leaves_a_metric_behind(self) -> None:
        """A withdrawn rule must leave nothing for something to quietly start reading -- neither
        the confounded valgus ratio nor the view-corrupted knee flexion angle."""
        self.assertNotIn("knee_width_to_ankle_width", JUMPING_JACKS_METRIC_KEYS)
        for key in JUMPING_JACKS_METRIC_KEYS:
            self.assertNotIn("knee_angle", key)

    def test_no_rule_produces_any_detection_at_all(self) -> None:
        """The whole-detector consequence of the roster: a clip built to trip every parent-spec
        rule produces nothing."""
        frames = jack_clip(reps=3, frames_per_rep=20, open_stance_ratio=0.8, hands_at_open=-0.9)
        result = run_detector(JUMPING_JACKS_DETECTOR, frames, 30.0, "front", 0.8)
        self.assertTrue(result.analyzed, "non-vacuity: the clip must really have segmented")
        self.assertEqual(result.detections, [])


class PhaseFractionTest(unittest.TestCase):
    """THE BICEP CURL TRAP, RECORDED FOR WHOEVER WAKES THE LEG RULE UP.

    A phase-SCOPED rule using `contiguous_true_segments(mask, min_frames)` needs
    `phase_fraction * rep_frames >= min_frames`. At 30 fps `min_frames` is 6, `open` is the top
    30% of a repetition, and a 2 Hz jack is 15 frames -- about 4 open frames.
    """

    def test_the_arithmetic_that_would_silence_a_mask_and_run_rule(self) -> None:
        fps = 30.0
        min_frames = max(3, int(math.ceil(max(fps, 1.0) * 0.20)))
        self.assertEqual(min_frames, 6)
        self.assertLess(0.30 * (fps / 2.0), min_frames)

    def test_a_two_hertz_clip_still_segments_and_still_reaches_the_rules(self) -> None:
        frames = jack_clip(reps=4, frames_per_rep=15)
        result = run_detector(JUMPING_JACKS_DETECTOR, frames, 30.0, "front", 0.8)
        self.assertTrue(result.analyzed)
        self.assertIsNone(result.fallback)
        open_frames = [frame for frame in result.core if frame.phase == "open"]
        self.assertTrue(open_frames, "the open phase must exist for a rule to scope to")


class SegmentationTest(unittest.TestCase):
    def test_the_segmenter_finds_one_repetition_per_open_and_close_cycle(self) -> None:
        frames = jack_clip(reps=3, frames_per_rep=20)
        result = run_detector(JUMPING_JACKS_DETECTOR, frames, 30.0, "front", 0.8)
        self.assertEqual(len(result.reps), 3)
        self.assertIsNone(result.fallback)

    def test_the_default_min_rep_seconds_admits_the_fastest_cited_cadence(self) -> None:
        """`base.py` NAMES THIS MOVEMENT AS ONE THAT MUST LOWER `min_rep_seconds`, AND IT DOES
        NOT NEED TO. The fastest jumping-jack cadence with a number anywhere in this project's
        sources is the RAG doc's Guinness record -- 136 repetitions in one minute, i.e. 2.27 Hz or
        0.44 s per repetition, which clears the 0.4 s floor. Asserted end to end at 13 frames per
        repetition; the real-footage measurement is in notes/jumping-jacks-rule-validation.md.
        """
        self.assertEqual(JUMPING_JACKS_DETECTOR.min_rep_seconds, 0.4)
        self.assertGreater(60.0 / 136.0, JUMPING_JACKS_DETECTOR.min_rep_seconds)
        frames = jack_clip(reps=4, frames_per_rep=13)
        result = run_detector(JUMPING_JACKS_DETECTOR, frames, 30.0, "front", 0.8)
        self.assertTrue(result.analyzed)
        self.assertIsNone(result.fallback)


class NotRegisteredTest(unittest.TestCase):
    """THE FIRST DETECTOR IN THE PROGRAMME THAT IS DELIBERATELY NOT REGISTERED."""

    def test_it_is_absent_from_the_registry(self) -> None:
        """Registration is what makes a movement analyzable in the app. With every rule silent or
        withdrawn, registering would offer an analysis that can never report a fault while wearing
        the Beta tag that says faults are possible."""
        self.assertNotIn("Jumping Jacks", [detector.name for detector in list_detectors()])

    def test_the_detector_object_still_exists_and_is_complete(self) -> None:
        """Not registered is not the same as not built: the metric layer, the phases and the
        segmentation all work and are what a future threshold would be dropped into."""
        self.assertEqual(JUMPING_JACKS_DETECTOR.name, "Jumping Jacks")
        self.assertEqual(JUMPING_JACKS_DETECTOR.rep_signal, "stance_width_ratio")
        self.assertIn(JUMPING_JACKS_DETECTOR.rep_signal, JUMPING_JACKS_DETECTOR.metric_keys)
        self.assertFalse(JUMPING_JACKS_DETECTOR.validated)

    def test_the_rep_signal_is_unipolar_unlike_torso_twists(self) -> None:
        """Torso Twist is the only user of `rep_rectify`; the feet in a jumping jack never cross,
        so rectifying here would fold the closed position onto itself."""
        self.assertEqual(JUMPING_JACKS_DETECTOR.rep_polarity, "max")
        self.assertFalse(JUMPING_JACKS_DETECTOR.rep_rectify)
        self.assertEqual(JUMPING_JACKS_DETECTOR.rep_start, "extended")


if __name__ == "__main__":
    unittest.main()
