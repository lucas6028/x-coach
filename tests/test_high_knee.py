import math
import unittest

import numpy as np

from src.pose.movements.base import run_detector
from src.pose.movements.registry import list_detectors
from src.pose.movements.high_knee import (
    HIGH_KNEE_DETECTOR,
    HIGH_KNEE_METRIC_KEYS,
    KNEE_LIFT_CITED_A_SKIP,
    KNEE_LIFT_IMPLEMENTED_B_SKIP,
    high_knee_assign_phases,
    high_knee_compute_raw,
    rule_insufficient_knee_lift,
)
from src.pose.rep_segmentation import segment_reps

_HIP_MID = (0.50, 0.62)
_HIP_HALF_WIDTH = 0.05
_SHOULDER_HALF_WIDTH = 0.07
_TRUNK_LEN = 0.22
_THIGH_LEN = 0.15
_SHANK_LEN = 0.15
_FOOT_LEN = 0.05

# Landmark pairs that must swap identity when the image is mirrored: a subject filmed from
# behind has their anatomical left on the other side of the frame.
_MIRROR_PAIRS = ((11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32))


def _lm(x: float, y: float, z: float = 0.0, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": z, "visibility": visibility}


def _rot(vector, degrees: float):
    rad = math.radians(degrees)
    cos, sin = math.cos(rad), math.sin(rad)
    return (vector[0] * cos - vector[1] * sin, vector[0] * sin + vector[1] * cos)


def high_knee_frame(
    left_elevation: float = -1.0,
    right_elevation: float = -1.0,
    trunk_lean_deg: float = 0.0,
    foot_length: float = _FOOT_LEN,
    hip_half_width: float = _HIP_HALF_WIDTH,
    frame_index: int = 0,
    drop_landmark: int | None = None,
    roll_deg: float = 0.0,
    mirrored: bool = False,
) -> dict:
    """One HIGH KNEE frame, built so each knob controls its metric BY CONSTRUCTION.

    Every landmark carries z=0, so the module's 2-D reads see exactly the image-plane geometry.
    The subject stands upright facing image-RIGHT, which is the anterior direction the feet encode.

    Knobs:

      left_elevation / right_elevation -- the target `thigh_elevation` for that side, i.e.
                        cos(angle between the hip->knee vector and trunk-up). -1.0 is a thigh
                        hanging straight down, 0.0 is the knee at hip height. The thigh is placed
                        at exactly the angle that produces the requested cosine.
      trunk_lean_deg -- tilts the TRUNK about the hip midpoint, positive FORWARD (anterior). The
                        legs are left alone, so this knob changes the trunk's relation to the
                        support limb and nothing else -- which is what the withdrawn trunk rules
                        would have read.
      foot_length    -- heel->toe length. Shrinking it models a camera looking down the subject's
                        line of travel, where the anterior axis projects to almost nothing.
      hip_half_width -- half the pelvis width IN THE IMAGE. Near zero models a SAGITTAL view, in
                        which the two hips project onto each other and the remaining image axis is
                        the anterior one. The default models a frontal view, where that axis is
                        instead mediolateral. A 2-D fixture cannot be both at once, and the
                        validation harness's quantities are sagittal, so its tests pass a small
                        value. See `SupportLimbGeometryTest`.
    """
    hip_mid = np.asarray(_HIP_MID, dtype=np.float64)

    def _place(vector):
        rolled = _rot(tuple(vector), roll_deg)
        return (hip_mid[0] + rolled[0], hip_mid[1] + rolled[1])

    # Trunk: straight up the image (image y increases downward), then leaned forward by the knob.
    trunk_vec = _rot((0.0, -_TRUNK_LEN), trunk_lean_deg)
    shoulder_mid = np.asarray(trunk_vec, dtype=np.float64)

    def _thigh(elevation: float):
        # elevation = cos(angle to trunk-up) and the trunk-up here is image-up, so a thigh at
        # angle theta from straight-down has elevation -cos(theta).
        theta = math.acos(max(-1.0, min(1.0, -elevation)))
        return (_THIGH_LEN * math.sin(theta), _THIGH_LEN * math.cos(theta))

    points: dict[int, tuple[float, float]] = {}
    points[11] = _place((shoulder_mid[0] - _SHOULDER_HALF_WIDTH, shoulder_mid[1]))
    points[12] = _place((shoulder_mid[0] + _SHOULDER_HALF_WIDTH, shoulder_mid[1]))
    points[23] = _place((-hip_half_width, 0.0))
    points[24] = _place((+hip_half_width, 0.0))

    for hip_index, knee_index, ankle_index, heel_index, toe_index, elevation, sign in (
        (23, 25, 27, 29, 31, left_elevation, -1.0),
        (24, 26, 28, 30, 32, right_elevation, +1.0),
    ):
        hip_offset = (sign * hip_half_width, 0.0)
        thigh = _thigh(elevation)
        knee_offset = (hip_offset[0] + thigh[0], hip_offset[1] + thigh[1])
        # The shank hangs straight down from the knee; the ankle's exact placement matters only
        # to the validation harness, which recomputes the withdrawn quantities from it.
        ankle_offset = (knee_offset[0], knee_offset[1] + _SHANK_LEN)
        points[knee_index] = _place(knee_offset)
        points[ankle_index] = _place(ankle_offset)
        points[heel_index] = _place(ankle_offset)
        points[toe_index] = _place((ankle_offset[0] + foot_length, ankle_offset[1]))

    landmarks = [_lm(0.5, 0.5) for _ in range(33)]
    for index, (x, y) in points.items():
        landmarks[index] = _lm(x, y)
    # The nose, so a frame is anatomically complete even though no rule reads it.
    landmarks[0] = _lm(*_place((shoulder_mid[0], shoulder_mid[1] - 0.06)))

    if mirrored:
        landmarks = [_lm(1.0 - lm["x"], lm["y"], lm["z"], lm["visibility"]) for lm in landmarks]
        for left_index, right_index in _MIRROR_PAIRS:
            landmarks[left_index], landmarks[right_index] = (
                landmarks[right_index],
                landmarks[left_index],
            )

    if drop_landmark is not None:
        landmarks[drop_landmark] = _lm(
            landmarks[drop_landmark]["x"], landmarks[drop_landmark]["y"], 0.0, 0.05
        )

    return {"frame_index": frame_index, "landmarks": landmarks}


def drive_clip(
    peak_elevation: float = -0.45,
    cycles: int = 4,
    frames_per_drive: int = 10,
    **frame_kwargs,
) -> list[dict]:
    """Alternating knee drives: one drive per `frames_per_drive` frames, sides alternating.

    At the default 10 frames per drive and 30 fps this is a 3 Hz drill -- the cadence
    `base.py:55` names for this movement.
    """
    frames: list[dict] = []
    index = 0
    for cycle in range(cycles * 2):
        left_side = cycle % 2 == 0
        for step in range(frames_per_drive):
            phase = math.sin(math.pi * step / max(1, frames_per_drive - 1))
            lifted = -1.0 + (peak_elevation + 1.0) * phase
            frames.append(
                high_knee_frame(
                    left_elevation=lifted if left_side else -1.0,
                    right_elevation=-1.0 if left_side else lifted,
                    frame_index=index,
                    **frame_kwargs,
                )
            )
            index += 1
    return frames


class MetricLayerTest(unittest.TestCase):
    def test_metric_keys_match_the_emitted_metrics_exactly(self) -> None:
        """Two-way: a key the tuple omits is silently NaN in every rule, and a key the tuple
        names but nothing emits is silently NaN too."""
        raw = high_knee_compute_raw([high_knee_frame()], 30.0)[0]
        framework = {"frame_index", "time", "valid", "lower_body_visibility"}
        self.assertEqual(set(raw) - framework, set(HIGH_KNEE_METRIC_KEYS))

    def test_the_knobs_control_their_metrics_exactly(self) -> None:
        raw = high_knee_compute_raw(
            [high_knee_frame(left_elevation=-0.30, right_elevation=-0.85)], 30.0
        )[0]
        self.assertAlmostEqual(raw["thigh_elevation_left"], -0.30, places=5)
        self.assertAlmostEqual(raw["thigh_elevation_right"], -0.85, places=5)
        self.assertAlmostEqual(raw["thigh_elevation_difference"], 0.55, places=5)

    def test_a_hanging_thigh_reads_minus_one_and_a_level_thigh_reads_zero(self) -> None:
        """The two ends of the scale the citation's two targets live on."""
        hanging = high_knee_compute_raw([high_knee_frame(left_elevation=-1.0)], 30.0)[0]
        self.assertAlmostEqual(hanging["thigh_elevation_left"], -1.0, places=5)
        level = high_knee_compute_raw([high_knee_frame(left_elevation=0.0)], 30.0)[0]
        self.assertAlmostEqual(level["thigh_elevation_left"], 0.0, places=5)

    def test_the_elevation_is_a_cosine_not_an_image_y_difference(self) -> None:
        """THE PARENT SPEC WRITES THIS RULE AS `y_knee - y_hip`, AND THAT IS WHAT THIS ASSERTS
        AWAY FROM.

        The corpus this detector was measured on ships its side cameras ROLLED 90 DEGREES, where
        an image-y difference reads the subject's fore-aft position instead of their knee height.
        The two agree when the camera is upright, which is asserted first so the test is not just
        measuring a rewrite.
        """
        upright = high_knee_frame(left_elevation=0.0)
        points = np.array([[lm["x"], lm["y"]] for lm in upright["landmarks"]])
        self.assertAlmostEqual(points[25][1] - points[23][1], 0.0, places=5)

        rolled = high_knee_frame(left_elevation=0.0, roll_deg=90.0)
        rolled_points = np.array([[lm["x"], lm["y"]] for lm in rolled["landmarks"]])
        self.assertGreater(abs(rolled_points[25][1] - rolled_points[23][1]), 0.10)
        self.assertAlmostEqual(
            high_knee_compute_raw([rolled], 30.0)[0]["thigh_elevation_left"], 0.0, places=5
        )

    def test_the_view_gate_is_a_length_and_collapses_when_the_feet_point_at_the_camera(self) -> None:
        wide = high_knee_compute_raw([high_knee_frame(foot_length=0.05)], 30.0)[0]
        narrow = high_knee_compute_raw([high_knee_frame(foot_length=0.002)], 30.0)[0]
        self.assertGreater(wide["anterior_axis_length"], narrow["anterior_axis_length"])
        self.assertGreaterEqual(narrow["anterior_axis_length"], 0.0)


class ValidityGateTest(unittest.TestCase):
    """SIX LANDMARKS ARE REQUIRED AND TWELVE ARE READ -- the jumping_jacks principle applied to a
    movement whose feet are the least reliable landmarks it has."""

    def test_dropping_a_required_landmark_invalidates_the_frame(self) -> None:
        for index in (11, 12, 23, 24, 25, 26):
            with self.subTest(landmark=index):
                raw = high_knee_compute_raw([high_knee_frame(drop_landmark=index)], 30.0)[0]
                self.assertFalse(raw["valid"])

    def test_dropping_a_foot_landmark_leaves_the_knee_metrics_intact(self) -> None:
        for index in (29, 30, 31, 32):
            with self.subTest(landmark=index):
                raw = high_knee_compute_raw(
                    [high_knee_frame(left_elevation=-0.4, drop_landmark=index)], 30.0
                )[0]
                self.assertTrue(raw["valid"])
                self.assertAlmostEqual(raw["thigh_elevation_left"], -0.4, places=5)

    def test_losing_both_feet_is_no_view_reading_rather_than_a_small_one(self) -> None:
        frame = high_knee_frame(drop_landmark=31)
        frame["landmarks"][32] = _lm(frame["landmarks"][32]["x"], frame["landmarks"][32]["y"],
                                     0.0, 0.05)
        raw = high_knee_compute_raw([frame], 30.0)[0]
        self.assertTrue(math.isnan(raw["anterior_axis_length"]))


class InvarianceTest(unittest.TestCase):
    def test_every_metric_is_invariant_under_camera_roll(self) -> None:
        """The property the whole module is built on: this corpus's side cameras are rolled 90
        degrees and every parent-spec heuristic for this movement is written in image y."""
        base = high_knee_compute_raw(
            [high_knee_frame(left_elevation=-0.35, right_elevation=-0.90)], 30.0
        )[0]
        for roll in (17.0, 90.0, 180.0, -63.0):
            with self.subTest(roll=roll):
                rolled = high_knee_compute_raw(
                    [high_knee_frame(left_elevation=-0.35, right_elevation=-0.90, roll_deg=roll)],
                    30.0,
                )[0]
                for key in HIGH_KNEE_METRIC_KEYS:
                    self.assertAlmostEqual(rolled[key], base[key], places=5, msg=key)

    def test_mirroring_swaps_the_sides_and_negates_the_difference(self) -> None:
        base = high_knee_compute_raw(
            [high_knee_frame(left_elevation=-0.35, right_elevation=-0.90)], 30.0
        )[0]
        mirrored = high_knee_compute_raw(
            [high_knee_frame(left_elevation=-0.35, right_elevation=-0.90, mirrored=True)], 30.0
        )[0]
        # Mirroring models filming the SAME subject from the other side, so MediaPipe's
        # anatomical left now sits where its right was: the per-side readings swap and the signed
        # difference negates. What the rep signal reads -- the MAGNITUDE -- is unchanged, which is
        # the invariance that matters and the reason the signal is rectified.
        self.assertAlmostEqual(mirrored["thigh_elevation_left"], base["thigh_elevation_right"], 5)
        self.assertAlmostEqual(mirrored["thigh_elevation_right"], base["thigh_elevation_left"], 5)
        self.assertAlmostEqual(
            mirrored["thigh_elevation_difference"], -base["thigh_elevation_difference"], 5
        )
        self.assertAlmostEqual(
            abs(mirrored["thigh_elevation_difference"]),
            abs(base["thigh_elevation_difference"]),
            places=5,
        )
        self.assertAlmostEqual(
            mirrored["anterior_axis_length"], base["anterior_axis_length"], places=5
        )

    def test_the_metrics_survive_roll_and_mirroring_through_segmentation(self) -> None:
        """Invariance of one frame is not invariance of a verdict: the rep signal is derived from
        these metrics, so the SEGMENTATION has to be invariant too."""
        plain = drive_clip()
        transformed = drive_clip(roll_deg=90.0, mirrored=True)
        self.assertEqual(
            len(run_detector(HIGH_KNEE_DETECTOR, plain, 30.0, "side", 0.8, max_reps=None).reps),
            len(
                run_detector(
                    HIGH_KNEE_DETECTOR, transformed, 30.0, "side", 0.8, max_reps=None
                ).reps
            ),
        )


class PhaseTest(unittest.TestCase):
    def test_phases_run_setup_drive_peak_recovery(self) -> None:
        phases = set(high_knee_assign_phases(high_knee_compute_raw(drive_clip(), 30.0)))
        self.assertTrue({"setup", "peak"} <= phases)
        self.assertTrue(phases <= {"setup", "drive", "peak", "recovery", "unknown"})

    def test_the_peak_phase_is_where_the_thighs_are_farthest_apart(self) -> None:
        raw = high_knee_compute_raw(drive_clip(), 30.0)
        phases = high_knee_assign_phases(raw)
        peak_values = [
            abs(raw[i]["thigh_elevation_difference"])
            for i, phase in enumerate(phases)
            if phase == "peak"
        ]
        other = [
            abs(raw[i]["thigh_elevation_difference"])
            for i, phase in enumerate(phases)
            if phase in ("drive", "recovery")
        ]
        self.assertGreater(min(peak_values), max(other) - 1e-9)

    def test_an_empty_clip_and_a_signal_free_clip_do_not_raise(self) -> None:
        self.assertEqual(high_knee_assign_phases([]), [])
        self.assertEqual(
            high_knee_assign_phases([{"valid": False}, {"valid": False}]),
            ["unknown", "unknown"],
        )

    def test_an_invalid_frame_inside_the_setup_slice_is_unknown_not_setup(self) -> None:
        frames = drive_clip()
        frames[1] = high_knee_frame(frame_index=1, drop_landmark=25)
        phases = high_knee_assign_phases(high_knee_compute_raw(frames, 30.0))
        self.assertEqual(phases[1], "unknown")


class SilentKneeLiftRuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = RuleContextFactory()

    def test_it_never_fires_even_on_a_drive_far_below_both_cuts(self) -> None:
        core = run_detector(
            HIGH_KNEE_DETECTOR, drive_clip(peak_elevation=-0.95), 30.0, "side", 0.9,
            max_reps=None,
        ).core
        self.assertEqual(rule_insufficient_knee_lift(core, self.ctx.build()), [])

    def test_the_metric_it_would_have_read_is_computed_and_correct(self) -> None:
        raw = high_knee_compute_raw([high_knee_frame(left_elevation=-0.62)], 30.0)[0]
        self.assertAlmostEqual(raw["thigh_elevation_left"], -0.62, places=5)

    def test_both_of_the_specs_disagreeing_cuts_are_kept_where_they_are(self) -> None:
        """THE POINT OF THIS RULE IS THAT THE SPEC SUPPLIES TWO NUMBERS AND THEY DISAGREE. Pinning
        both, and pinning that they differ, is what stops a later edit from quietly picking one
        and calling it settled."""
        self.assertAlmostEqual(KNEE_LIFT_CITED_A_SKIP, -math.cos(math.radians(45.0)), places=6)
        self.assertAlmostEqual(KNEE_LIFT_IMPLEMENTED_B_SKIP, 0.0, places=6)
        self.assertLess(KNEE_LIFT_CITED_A_SKIP, KNEE_LIFT_IMPLEMENTED_B_SKIP)

    def test_the_silent_rule_is_present_rather_than_absent(self) -> None:
        self.assertIn(rule_insufficient_knee_lift, HIGH_KNEE_DETECTOR.rules)


class RuleContextFactory:
    def build(self, view_type: str = "side", confidence: float = 0.9, min_frames: int = 3):
        from src.pose.movements.base import RuleContext

        return RuleContext(
            fps=30.0, view_type=view_type, view_confidence=confidence, min_frames=min_frames
        )


class WithdrawnRulesTest(unittest.TestCase):
    def test_no_withdrawn_rule_leaves_a_metric_behind(self) -> None:
        """A withdrawn rule's quantity must not sit in the metric tuple where something could
        quietly start reading it. Both trunk rules' scalar and the pelvic rule's obliquity are
        recomputed in `src/egoexo/high_knee_validation.py` instead."""
        for forbidden in ("trunk_lean_forward", "pelvic_obliquity", "hip_line_tilt"):
            self.assertNotIn(forbidden, HIGH_KNEE_METRIC_KEYS)
        raw = high_knee_compute_raw([high_knee_frame()], 30.0)[0]
        self.assertNotIn("trunk_lean_forward", raw)

    def test_no_rule_produces_any_detection_at_all(self) -> None:
        """The claim the non-registration rests on: with one rule silent and four withdrawn, the
        detector cannot report a fault on any input."""
        for peak in (-0.99, -0.70, -0.20, 0.10):
            with self.subTest(peak=peak):
                result = run_detector(
                    HIGH_KNEE_DETECTOR, drive_clip(peak_elevation=peak), 30.0, "side", 0.9,
                    max_reps=None,
                )
                self.assertEqual(result.detections, [])


class SegmentationTest(unittest.TestCase):
    def test_the_segmenter_finds_one_repetition_per_knee_drive(self) -> None:
        result = run_detector(
            HIGH_KNEE_DETECTOR, drive_clip(cycles=3, frames_per_drive=12), 30.0, "side", 0.8,
            max_reps=None,
        )
        # 3 cycles x 2 drives, minus at most one boundary window the segmenter cannot close.
        self.assertGreaterEqual(len(result.reps), 5)
        self.assertLessEqual(len(result.reps), 6)

    def test_the_lowered_floor_is_what_admits_this_movements_cadence(self) -> None:
        """THE KNOB `base.py:55` RESERVED FOR THIS MOVEMENT BY NAME, exercised on the cadence that
        comment names: ~3 Hz, about 10 frames per repetition at 30 fps.

        Measured the non-circular way -- differencing rep counts at two floors, because every
        window `segment_reps` RETURNS is at least `min_rep_seconds` long by construction and so
        can never show the floor biting.
        """
        from src.pose.geometry import centered_median

        frames = drive_clip(cycles=4, frames_per_drive=10)
        raw = high_knee_compute_raw(frames, 30.0)
        signal = centered_median(
            [float(item["thigh_elevation_difference"]) for item in raw], window=5
        )
        at_shipped = segment_reps(
            signal, fps=30.0, polarity="max", rectify=True, min_rep_seconds=0.15
        )
        at_default = segment_reps(
            signal, fps=30.0, polarity="max", rectify=True, min_rep_seconds=0.4
        )
        self.assertGreater(len(at_shipped), len(at_default))
        self.assertEqual(len(at_default), 0)

    def test_the_rep_signal_is_bipolar_and_rectified_unlike_jumping_jacks(self) -> None:
        self.assertEqual(HIGH_KNEE_DETECTOR.rep_signal, "thigh_elevation_difference")
        self.assertTrue(HIGH_KNEE_DETECTOR.rep_rectify)
        self.assertEqual(HIGH_KNEE_DETECTOR.rep_polarity, "max")
        raw = high_knee_compute_raw(drive_clip(), 30.0)
        values = [item["thigh_elevation_difference"] for item in raw]
        self.assertGreater(max(values), 0.0)
        self.assertLess(min(values), 0.0)

    def test_the_shipped_floor_is_the_frameworks_own_arithmetic(self) -> None:
        """0.15 s is half the 0.33 s `base.py:55` states for this movement -- not a value fitted
        to the 1.31 Hz the measured corpus happens to show."""
        self.assertAlmostEqual(HIGH_KNEE_DETECTOR.min_rep_seconds, 0.15, places=6)
        self.assertLess(HIGH_KNEE_DETECTOR.min_rep_seconds, 1.0 / 3.0)


class NotRegisteredTest(unittest.TestCase):
    def test_it_is_absent_from_the_registry(self) -> None:
        self.assertNotIn("High Knee", {detector.name for detector in list_detectors()})

    def test_the_detector_object_still_exists_and_is_complete(self) -> None:
        """Not registered is not "not built": everything that works is kept and testable, so
        waking the movement is a threshold plus one line."""
        self.assertEqual(HIGH_KNEE_DETECTOR.name, "High Knee")
        self.assertEqual(HIGH_KNEE_DETECTOR.metric_keys, HIGH_KNEE_METRIC_KEYS)
        self.assertFalse(HIGH_KNEE_DETECTOR.validated)
        self.assertEqual(len(HIGH_KNEE_DETECTOR.rules), 1)

    def test_the_module_is_not_imported_by_the_registry(self) -> None:
        source = (
            __import__("pathlib").Path("src/pose/movements/registry.py").read_text(encoding="utf-8")
        )
        self.assertNotIn("import high_knee", source)
        self.assertIn("high_knee", source)  # the reason is recorded there


if __name__ == "__main__":
    unittest.main()
