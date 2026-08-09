import math
import unittest

import numpy as np

from src.pose.movements.base import CoreFrame, RuleContext, run_detector
from src.pose.movements.registry import get_detector, list_detectors
from src.pose.movements.situp import (
    ROM_MILD_DEG,
    ROM_SEVERE_DEG,
    SITUP_DETECTOR,
    SITUP_METRIC_KEYS,
    rule_hip_flexor_dominance,
    rule_incomplete_rom,
    situp_assign_phases,
    situp_compute_raw,
)


def _lm(x: float, y: float, z: float = 0.0, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": z, "visibility": visibility}


_TRUNK_LEN = 0.24
_THIGH_LEN = 0.16
_HALF_WIDTH = 0.03


def situp_frame(
    hip_angle_deg: float = 140.0,
    right_hip_angle_deg: float | None = None,
    frame_index: int = 0,
    visibility: float = 0.95,
    drop_landmark: int | None = None,
    roll_deg: float = 0.0,
) -> dict:
    """One SUPINE sit-up frame, image y growing DOWNWARD.

    Every landmark carries z=0, so `angle_degrees` (which computes with dims=3) sees exactly the
    image-plane triangle and the knob controls its metric BY CONSTRUCTION -- the same reason
    tests/test_arm_vw.py and tests/test_arm_abduction.py need no depth correction.

    Geometry: the subject lies along the image x axis with the hips at the origin point, the thigh
    running toward the feet (hook lying, so it also rises off the mat), and the trunk placed by
    rotating the hip->knee direction through the requested `angle(shoulder, hip, knee)`. Left and
    right are congruent triangles offset perpendicular to the body axis, so both sides read the
    same angle exactly unless `right_hip_angle_deg` overrides one.

    Knobs:
      hip_angle_deg        -- `angle(shoulder, hip, knee)`, both sides unless overridden. ~180 =
                              trunk and thigh in a straight line, smaller = trunk curled toward the
                              thigh. Equals `left_hip_angle_deg` and `hip_angle_deg`.
      right_hip_angle_deg  -- right-side override, so a fixture can drive the two sides apart.
      drop_landmark        -- zero the visibility of one landmark index, to exercise the
                              all-or-nothing validity gate.
      roll_deg             -- rotate EVERY landmark about the image centre by this angle. The
                              module reads only joint-relative angles, so this must change nothing:
                              it is the executable form of the design spec's claim that the image
                              horizontal is not a usable floor reference. EgoExo-Fitness ships its
                              sagittal sit-up frames rolled by exactly 90 degrees, with no EXIF tag.
    """
    hip_mid = (0.40, 0.62)
    # Thigh direction: toward the feet (+x) and up off the mat (-y in image coords).
    thigh_dir = (math.cos(math.radians(-25.0)), math.sin(math.radians(-25.0)))

    def side(angle_deg: float, sign: float):
        hip = (hip_mid[0], hip_mid[1] + sign * _HALF_WIDTH)
        knee = (hip[0] + _THIGH_LEN * thigh_dir[0], hip[1] + _THIGH_LEN * thigh_dir[1])
        # Rotate the hip->knee unit vector by the requested angle to place the shoulder, so
        # `angle_degrees(shoulder, hip, knee)` equals `angle_deg` exactly rather than approximately
        # -- a boundary fixture needs to sit one hundredth of a degree either side of a threshold.
        phi = math.radians(angle_deg)
        rx = thigh_dir[0] * math.cos(phi) - thigh_dir[1] * math.sin(phi)
        ry = thigh_dir[0] * math.sin(phi) + thigh_dir[1] * math.cos(phi)
        shoulder = (hip[0] + _TRUNK_LEN * rx, hip[1] + _TRUNK_LEN * ry)
        return shoulder, hip, knee

    left_shoulder, left_hip, left_knee = side(hip_angle_deg, -1.0)
    right_shoulder, right_hip, right_knee = side(
        hip_angle_deg if right_hip_angle_deg is None else right_hip_angle_deg, 1.0
    )

    landmarks = [_lm(0.5, 0.5, visibility=visibility) for _ in range(33)]
    landmarks[11] = _lm(*left_shoulder, visibility=visibility)
    landmarks[12] = _lm(*right_shoulder, visibility=visibility)
    landmarks[23] = _lm(*left_hip, visibility=visibility)
    landmarks[24] = _lm(*right_hip, visibility=visibility)
    landmarks[25] = _lm(*left_knee, visibility=visibility)
    landmarks[26] = _lm(*right_knee, visibility=visibility)

    if roll_deg:
        theta = math.radians(roll_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rolled = []
        for point in landmarks:
            dx, dy = point["x"] - 0.5, point["y"] - 0.5
            rolled.append(
                _lm(
                    0.5 + dx * cos_t - dy * sin_t,
                    0.5 + dx * sin_t + dy * cos_t,
                    visibility=point["visibility"],
                )
            )
        landmarks = rolled

    if drop_landmark is not None:
        point = landmarks[drop_landmark]
        landmarks[drop_landmark] = _lm(point["x"], point["y"], visibility=0.0)
    return {"frame_index": frame_index, "landmarks": landmarks}


def _ctx(view_type: str = "rear_oblique", view_confidence: float = 0.8, min_frames: int = 3):
    """Defaults to `rear_oblique` -- 37 of the 49 real pose JSONs. The value is deliberately
    irrelevant to every assertion below: this module's shipped rule reads no view (design spec
    section 7.3), and `ViewIndifferenceTest` pins that."""
    return RuleContext(
        fps=30.0, view_type=view_type, view_confidence=view_confidence, min_frames=min_frames
    )


def _core(frames: list[dict], fps: float = 30.0) -> list[CoreFrame]:
    """Metrics + phases WITHOUT run_detector, so a rule can be tested on an exact fixture.

    run_detector median-filters and re-slices per rep; both are correct in production and both
    would blur a boundary fixture built to sit one step either side of a threshold.
    """
    raw = situp_compute_raw(frames, fps=fps)
    phases = situp_assign_phases(raw)
    return [
        CoreFrame(
            frame_index=int(item.get("frame_index", i) or i),
            time=float(item.get("time", 0.0) or 0.0),
            phase=phases[i],
            valid=bool(item.get("valid", False)),
            lower_body_visibility=float(item.get("lower_body_visibility", 0.0) or 0.0),
            metrics={k: float(item.get(k, np.nan)) for k in SITUP_METRIC_KEYS},
        )
        for i, item in enumerate(raw)
    ]


def _rep(
    n: int = 40,
    supine_deg: float = 140.0,
    curled_deg: float = 90.0,
    start_index: int = 0,
    invalidate: range | None = None,
    roll_deg: float = 0.0,
) -> list[dict]:
    """One curl-up rep: supine hold -> curl -> top hold -> lower -> supine hold.

    The rep OPENS lying flat (`supine_deg`, the trunk and thigh at their most open) and bottoms at
    the top of the curl (`curled_deg`) in the middle, which is what makes the signal's MINIMUM the
    effort peak and gives the registry entry `("hip_angle_deg", "min", "extended")`.

    Both ends are flat on purpose: the parent spec's phase list opens and closes at `setup
    (supine)` / `rest`, and flat ends make `supine_deg` and `curled_deg` land EXACTLY on sampled
    frames, which a boundary fixture needs.

    `invalidate` blanks a contiguous block of frames, to test that `rule_incomplete_rom` measures
    the whole window rather than each contiguous valid run.
    """
    frames: list[dict] = []
    for i in range(n):
        t = i / (n - 1)
        if t < 0.2 or t > 0.8:
            angle = supine_deg
        elif t < 0.4:
            angle = supine_deg - (supine_deg - curled_deg) * math.sin(math.pi * (t - 0.2) / 0.4) ** 2
        elif t <= 0.6:
            angle = curled_deg
        else:
            angle = curled_deg + (supine_deg - curled_deg) * math.sin(math.pi * (t - 0.6) / 0.4) ** 2
        frames.append(
            situp_frame(
                hip_angle_deg=angle,
                frame_index=start_index + i,
                drop_landmark=11 if invalidate is not None and i in invalidate else None,
                roll_deg=roll_deg,
            )
        )
    return frames


class FixtureAndMetricTest(unittest.TestCase):
    def test_the_fixture_produces_the_requested_hip_angles_exactly(self) -> None:
        """The fixture's own contract. Every threshold test below is only as trustworthy as this --
        a rotation-sign error would put the trunk on the wrong side of the thigh and quietly
        produce the supplementary angle instead of the requested one."""
        for requested in (60.0, 90.0, 140.0, 175.0):
            raw = situp_compute_raw([situp_frame(hip_angle_deg=requested)], fps=30.0)[0]
            self.assertAlmostEqual(raw["left_hip_angle_deg"], requested, places=4)
            self.assertAlmostEqual(raw["right_hip_angle_deg"], requested, places=4)
            self.assertAlmostEqual(raw["hip_angle_deg"], requested, places=4)

    def test_the_two_sides_average_into_the_rep_signal(self) -> None:
        raw = situp_compute_raw(
            [situp_frame(hip_angle_deg=120.0, right_hip_angle_deg=100.0)], fps=30.0
        )[0]
        self.assertAlmostEqual(raw["left_hip_angle_deg"], 120.0, places=4)
        self.assertAlmostEqual(raw["right_hip_angle_deg"], 100.0, places=4)
        self.assertAlmostEqual(raw["hip_angle_deg"], 110.0, places=4)

    def test_the_signal_degrades_to_one_side_rather_than_going_nan(self) -> None:
        """Deliberate asymmetry with `arm_vw.arm_elevation_asymmetry_deg`, which is NaN unless BOTH
        sides are finite. This is the REP SIGNAL: refusing it when one side is occluded would
        disable segmentation on exactly the sagittal geometry this movement is filmed in, where the
        far-side landmarks are most often lost. There is no asymmetry rule in this module, so no
        rule can silently read a one-sided value as a two-sided one."""
        frame = situp_frame(hip_angle_deg=120.0)
        # Degenerate the right side's trunk to a zero-length vector: the point stays visible, so
        # the frame is still valid, but `angle_degrees` cannot form the triangle.
        frame["landmarks"][12] = dict(frame["landmarks"][24])
        raw = situp_compute_raw([frame], fps=30.0)[0]
        self.assertTrue(raw["valid"])
        self.assertFalse(np.isfinite(raw["right_hip_angle_deg"]))
        self.assertAlmostEqual(raw["hip_angle_deg"], raw["left_hip_angle_deg"], places=4)

    def test_one_dropped_required_landmark_invalidates_the_whole_frame(self) -> None:
        """All-or-nothing, matching every movement module since Push-up. It bites hardest here: a
        sagittal view of a supine subject is the geometry in which the far-side shoulder, hip and
        knee are most often occluded."""
        for index in (11, 12, 23, 24, 25, 26):
            raw = situp_compute_raw([situp_frame(drop_landmark=index)], fps=30.0)[0]
            self.assertFalse(raw["valid"], f"landmark {index} should invalidate the frame")
            for key in SITUP_METRIC_KEYS:
                self.assertNotIn(key, raw)

    def test_the_heel_and_ankle_landmarks_are_not_required(self) -> None:
        """The parent spec's heel proxy for `hip_flexor_dominance` is NOT implemented (see that
        rule's docstring), so the heel landmarks stay out of `required`. If they were in it, one
        lost heel would silence the rule that DOES fire."""
        for index in (27, 28, 29, 30, 31, 32):
            raw = situp_compute_raw([situp_frame(drop_landmark=index)], fps=30.0)[0]
            self.assertTrue(raw["valid"], f"landmark {index} must not invalidate the frame")

    def test_a_non_dict_frame_is_invalid_rather_than_an_exception(self) -> None:
        raw = situp_compute_raw(["not a frame", None], fps=30.0)
        self.assertEqual([item["valid"] for item in raw], [False, False])

    def test_metric_keys_match_the_emitted_metrics_exactly(self) -> None:
        """Two-way. A key the tuple omits is dropped by `run_detector` (which builds each
        CoreFrame's metrics dict FROM this tuple) and read back as NaN by every rule; a key the
        tuple names but nothing emits is a silent NaN column."""
        raw = situp_compute_raw([situp_frame()], fps=30.0)[0]
        emitted = set(raw) - {"frame_index", "time", "valid", "lower_body_visibility"}
        self.assertEqual(emitted, set(SITUP_METRIC_KEYS))


class RollInvarianceTest(unittest.TestCase):
    """THE DESIGN SPEC'S CENTRAL GEOMETRIC CLAIM, AS AN EXECUTABLE ASSERTION.

    The parent spec defines every Group E quantity against "the floor/horizontal". The image
    horizontal is not the floor: EgoExo-Fitness -- the only dataset with labeled sit-ups -- stores
    its sagittal `exo_l`/`exo_r` frames rotated a quarter turn with NO EXIF orientation tag, so a
    trunk-flexion angle read against the image axis is 90 degrees wrong on the only real footage
    that exists. This module re-anchors to the body instead. These tests fail the moment anyone
    re-introduces an image-horizontal reference.
    """

    def test_the_metric_is_unchanged_by_any_camera_roll(self) -> None:
        for roll in (0.0, 17.0, 90.0, 180.0, -90.0):
            raw = situp_compute_raw([situp_frame(hip_angle_deg=118.0, roll_deg=roll)], fps=30.0)[0]
            self.assertAlmostEqual(raw["hip_angle_deg"], 118.0, places=3, msg=f"roll={roll}")

    def test_detections_are_identical_under_a_ninety_degree_roll(self) -> None:
        """90 degrees is not an arbitrary choice -- it is the roll EgoExo-Fitness actually ships."""
        upright = rule_incomplete_rom(_core(_rep(curled_deg=132.0)), _ctx())
        rolled = rule_incomplete_rom(_core(_rep(curled_deg=132.0, roll_deg=90.0)), _ctx())
        self.assertEqual(len(upright), 1)
        self.assertEqual(len(rolled), 1)
        self.assertEqual(upright[0].severity, rolled[0].severity)
        self.assertEqual(upright[0].evidence, rolled[0].evidence)
        self.assertEqual(upright[0].peak_frame, rolled[0].peak_frame)


class PhaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = situp_compute_raw(_rep(), fps=30.0)
        self.phases = situp_assign_phases(self.raw)

    def test_the_curl_is_the_top_and_the_opening_supine_hold_is_the_setup(self) -> None:
        lowest = int(np.nanargmin([item["hip_angle_deg"] for item in self.raw]))
        self.assertEqual(self.phases[lowest], "top")
        self.assertEqual(self.phases[0], "setup")
        self.assertEqual(set(self.phases), {"setup", "concentric", "top", "eccentric"})

    def test_an_invalid_frame_inside_the_setup_window_is_unknown_not_setup(self) -> None:
        frames = _rep()
        frames[1] = situp_frame(hip_angle_deg=140.0, frame_index=1, drop_landmark=23)
        phases = situp_assign_phases(situp_compute_raw(frames, fps=30.0))
        self.assertEqual(phases[1], "unknown")

    def test_a_clip_with_no_finite_signal_is_entirely_unknown(self) -> None:
        frames = [situp_frame(frame_index=i, drop_landmark=25) for i in range(10)]
        phases = situp_assign_phases(situp_compute_raw(frames, fps=30.0))
        self.assertEqual(set(phases), {"unknown"})

    def test_an_empty_clip_returns_an_empty_phase_list(self) -> None:
        self.assertEqual(situp_assign_phases([]), [])


class IncompleteRomRuleTest(unittest.TestCase):
    def _excursion(self, excursion_deg: float):
        return rule_incomplete_rom(
            _core(_rep(supine_deg=140.0, curled_deg=140.0 - excursion_deg)), _ctx()
        )

    def test_fires_just_under_the_spec_threshold(self) -> None:
        detections = self._excursion(ROM_MILD_DEG - 0.01)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "situp_incomplete_rom")

    def test_silent_just_over_the_spec_threshold(self) -> None:
        self.assertEqual(self._excursion(ROM_MILD_DEG + 0.01), [])

    def test_silent_on_a_real_sized_curl(self) -> None:
        """A curl-up that reaches the 35-40 degree target the sources describe must say nothing."""
        self.assertEqual(self._excursion(38.0), [])

    def test_severity_ramps_between_the_rule_level_bounds(self) -> None:
        mild = self._excursion(ROM_MILD_DEG - 0.01)[0].severity
        severe = self._excursion(ROM_SEVERE_DEG)[0].severity
        self.assertLess(mild, severe)
        self.assertAlmostEqual(severe, 1.0, places=3)

    def test_the_excursion_is_measured_over_the_whole_window_not_per_valid_run(self) -> None:
        """An occlusion gap must not split one rep into two half-excursions. Splitting would hand
        each half a smaller travel and FIRE ON A GOOD REP -- the opposite of this rule's intended
        failure direction."""
        frames = _rep(n=40, supine_deg=140.0, curled_deg=95.0, invalidate=range(16, 24))
        self.assertEqual(rule_incomplete_rom(_core(frames), _ctx()), [])

    def test_evidence_reports_the_excursion_and_both_endpoints(self) -> None:
        detection = self._excursion(8.0)[0]
        evidence = detection.evidence
        self.assertAlmostEqual(evidence["hip_angle_excursion_deg"], 8.0, places=2)
        self.assertAlmostEqual(evidence["supine_hip_angle_deg"], 140.0, places=2)
        self.assertAlmostEqual(evidence["curled_hip_angle_deg"], 132.0, places=2)
        self.assertEqual(evidence["threshold_deg"], ROM_MILD_DEG)
        self.assertEqual(evidence["primary_threshold"], ROM_MILD_DEG)

    def test_the_reported_peak_frame_is_the_most_curled_frame(self) -> None:
        core = _core(_rep(supine_deg=140.0, curled_deg=132.0))
        detection = rule_incomplete_rom(core, _ctx())[0]
        most_curled = min(
            (frame for frame in core if frame.valid), key=lambda f: f.m("hip_angle_deg")
        )
        self.assertEqual(detection.peak_frame, most_curled.frame_index)

    def test_a_window_shorter_than_min_frames_is_not_judged(self) -> None:
        core = _core(_rep(supine_deg=140.0, curled_deg=136.0))[:2]
        self.assertEqual(rule_incomplete_rom(core, _ctx(min_frames=3)), [])

    def test_the_whole_rep_scope_reduces_the_min_frames_requirement_to_point_two_seconds(
        self,
    ) -> None:
        """No shipped rule here is phase-scoped, so the Bicep Curl trap
        (`phase_fraction * T >= min_frames / fps`) collapses to `T >= min_frames / fps`. Both sides
        of that cliff, pinned: at 30 fps `min_frames` is 6, so a 6-frame window scores and a
        5-frame one does not."""
        ctx = _ctx(min_frames=6)
        core = _core(_rep(n=40, supine_deg=140.0, curled_deg=132.0))
        self.assertEqual(len(rule_incomplete_rom(core[:6], ctx)), 1)
        self.assertEqual(rule_incomplete_rom(core[:5], ctx), [])


class ViewIndifferenceTest(unittest.TestCase):
    """THE FIRST RULE IN THIS REGISTRY WITH NEITHER A VIEW GATE NOR A VIEW DISCOUNT.

    `src/pose/view_estimation.py`'s docstring, limit 1: "for a horizontal body the frontal axis no
    longer maps onto image x, so the front/rear/*_oblique labels carry no validated meaning there.
    Do not gate a horizontal-movement rule on them." A discount keyed on a meaningless label is not
    conservative, it is arbitrary. Design spec section 7.3.
    """

    def test_every_view_produces_byte_identical_detections(self) -> None:
        core = _core(_rep(supine_deg=140.0, curled_deg=132.0))
        baseline = rule_incomplete_rom(core, _ctx(view_type="rear"))[0]
        for view in ("front", "front_oblique", "side", "rear_oblique", "rear", "unknown"):
            for confidence in (0.0, 0.5, 1.0):
                detection = rule_incomplete_rom(
                    core, _ctx(view_type=view, view_confidence=confidence)
                )[0]
                self.assertEqual(detection.confidence, baseline.confidence, msg=view)
                self.assertEqual(detection.observability, baseline.observability, msg=view)
                self.assertEqual(detection.severity, baseline.severity, msg=view)

    def test_confidence_is_severity_undiscounted(self) -> None:
        detection = rule_incomplete_rom(
            _core(_rep(supine_deg=140.0, curled_deg=132.0)), _ctx(view_type="unknown")
        )[0]
        self.assertEqual(detection.confidence, detection.severity)


class HipFlexorRuleIsPermanentlySilentTest(unittest.TestCase):
    """NON-VACUOUS SILENCE. The Bicep Curl pass shipped an "asserts silence" test that passed
    because its fixture could not have fired anything, and had to correct it. So the first test
    here proves the fixture is live before the second proves this rule says nothing on it."""

    def setUp(self) -> None:
        self.core = _core(_rep(supine_deg=140.0, curled_deg=132.0))

    def test_the_live_rule_fires_on_this_clip(self) -> None:
        self.assertEqual(len(rule_incomplete_rom(self.core, _ctx())), 1)

    def test_the_hip_flexor_rule_says_nothing_on_the_same_clip(self) -> None:
        self.assertEqual(rule_hip_flexor_dominance(self.core, _ctx()), [])

    def test_the_hip_flexor_rule_says_nothing_on_any_view_or_shape(self) -> None:
        for view in ("front", "side", "rear", "rear_oblique", "unknown"):
            for curled in (40.0, 90.0, 139.0, 179.0):
                core = _core(_rep(supine_deg=180.0, curled_deg=curled))
                self.assertEqual(rule_hip_flexor_dominance(core, _ctx(view_type=view)), [])

    def test_a_rigid_trunk_is_indistinguishable_from_a_segmental_curl_in_the_metrics(self) -> None:
        """The structural reason the rule is silent, pinned. MediaPipe has no landmark between the
        shoulders (11/12) and the hips (23/24), so the trunk is ONE segment: two reps with the same
        hip-angle trajectory are byte-identical in every emitted metric no matter what the spine
        did. There is nothing for a segmental-curl rule to read."""
        raw = situp_compute_raw(_rep(), fps=30.0)
        emitted = set(raw[0]) - {"frame_index", "time", "valid", "lower_body_visibility"}
        self.assertEqual(emitted, set(SITUP_METRIC_KEYS))
        self.assertNotIn("trunk_curl_deg", emitted)
        self.assertNotIn("heel_displacement", emitted)


class WithdrawnRulesAreAbsentTest(unittest.TestCase):
    def test_only_two_rules_are_registered(self) -> None:
        """`excessive_speed` and `excessive_rom` are ABSENT, not silent. A silent stub asserts
        "real fault, the sensor cannot see it"; both of these fail on citations that do not support
        them as written. Design spec sections 5 and 6."""
        self.assertEqual(
            [rule.__name__ for rule in SITUP_DETECTOR.rules],
            ["rule_incomplete_rom", "rule_hip_flexor_dominance"],
        )

    def test_no_rule_fires_on_a_fast_or_a_deep_repetition(self) -> None:
        """A full sit-up performed quickly -- the exact shape `excessive_speed` and `excessive_rom`
        would have flagged -- must produce nothing. EgoExo-Fitness's canonical guidance PRESCRIBES
        this range ("touch your feet with your hands") and faults the failure to reach it on 28/82
        judged actions."""
        fast_and_deep = _rep(n=8, supine_deg=150.0, curled_deg=40.0)
        core = _core(fast_and_deep)
        detections = [d for rule in SITUP_DETECTOR.rules for d in rule(core, _ctx())]
        self.assertEqual(detections, [])


class RegistrationTest(unittest.TestCase):
    def test_the_detector_is_registered_unvalidated_with_the_stated_rep_interface(self) -> None:
        detector = get_detector("Sit-up")
        self.assertIs(detector, SITUP_DETECTOR)
        self.assertFalse(detector.validated)
        self.assertEqual(detector.rep_signal, "hip_angle_deg")
        self.assertIn(detector.rep_signal, detector.metric_keys)
        self.assertEqual(detector.rep_polarity, "min")
        self.assertEqual(detector.rep_start, "extended")
        self.assertFalse(detector.rep_rectify)

    def test_it_is_the_eleventh_detector_in_registration_order(self) -> None:
        names = [detector.name for detector in list_detectors()]
        self.assertEqual(names[-1], "Sit-up")
        self.assertEqual(len(names), 11)
        self.assertEqual(names[0], "Squat")

    def test_lookup_is_case_insensitive(self) -> None:
        self.assertIs(get_detector("sit-up"), SITUP_DETECTOR)


class EndToEndSegmentationTest(unittest.TestCase):
    def test_three_reps_segment_and_the_rule_scores_each_one(self) -> None:
        frames: list[dict] = []
        for index in range(3):
            frames.extend(
                _rep(n=40, supine_deg=140.0, curled_deg=132.0, start_index=index * 40)
            )
        result = run_detector(
            SITUP_DETECTOR, frames, fps=30.0, view_type="rear_oblique", view_confidence=0.8
        )
        self.assertIsNone(result.fallback)
        self.assertGreaterEqual(len(result.reps), 3)
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(result.detections[0].fault_id, "situp_incomplete_rom")
        self.assertGreaterEqual(result.detections[0].rep_count, 3)

    def test_a_shallow_curl_still_segments_because_the_segmenter_is_scale_free(self) -> None:
        """The trap worth checking whenever a rule reads the same signal that defines a rep: could
        segmentation structurally hide the fault? It cannot. `segment_reps` thresholds on
        PERCENTILES of the signal (rep_segmentation.py:186-194), so a 3-degree curl segments exactly
        as a 50-degree one does and reaches the rule with its small excursion intact. Contrast the
        Bicep Curl extension term, silenced by an ABSOLUTE interaction."""
        frames: list[dict] = []
        for index in range(3):
            frames.extend(
                _rep(n=40, supine_deg=140.0, curled_deg=137.0, start_index=index * 40)
            )
        result = run_detector(
            SITUP_DETECTOR, frames, fps=30.0, view_type="rear_oblique", view_confidence=0.8
        )
        self.assertIsNone(result.fallback)
        self.assertGreaterEqual(len(result.reps), 3)
        self.assertEqual(len(result.detections), 1)

    def test_a_motionless_clip_fires_this_rule_at_full_severity(self) -> None:
        """PINS A KNOWN, MEASURED, DELIBERATELY UNREPAIRED FAILURE -- read this before "fixing" it.

        `segment_reps` has no noise floor by design: it thresholds on PERCENTILES of the signal's
        own range, which is exactly what lets a genuinely shallow curl be found as a rep at all.
        The cost is that 0.4 deg of jitter on a motionless subject also segments, and this rule
        then reports the resulting ~0.7 deg excursion as a maximally severe incomplete range.

        NOT INTRODUCED BY THIS MODULE: the identical probe against
        `arm_vw.rule_incomplete_excursion` (shipped and merged) also yields 3 reps at severity 1.0.
        Every whole-rep "not enough travel" rule inherits it. Any in-rule guard would be a
        minimum-excursion floor -- an invented threshold. The honest repairs are framework-level
        (a noise floor in `segment_reps`, or `RunResult.fallback` threaded into `RuleContext`).

        WHEN THAT REPAIR LANDS THIS TEST SHOULD FAIL, and its failure is the intended signal.
        """
        frames = [
            situp_frame(hip_angle_deg=140.0 + 0.4 * math.sin(i / 3.0), frame_index=i)
            for i in range(60)
        ]
        result = run_detector(
            SITUP_DETECTOR, frames, fps=30.0, view_type="rear_oblique", view_confidence=0.8
        )
        self.assertIsNone(result.fallback, "the jitter is segmented, not sent to the fallback path")
        self.assertEqual(len(result.detections), 1)
        detection = result.detections[0]
        self.assertEqual(detection.fault_id, "situp_incomplete_rom")
        self.assertEqual(detection.severity, 1.0)
        self.assertEqual(detection.confidence, 1.0)
        self.assertEqual(detection.observability, "high")
        self.assertLess(detection.evidence["hip_angle_excursion_deg"], 1.0)

    def test_a_good_clip_produces_no_detections_end_to_end(self) -> None:
        frames: list[dict] = []
        for index in range(3):
            frames.extend(_rep(n=40, supine_deg=145.0, curled_deg=100.0, start_index=index * 40))
        result = run_detector(
            SITUP_DETECTOR, frames, fps=30.0, view_type="rear_oblique", view_confidence=0.8
        )
        self.assertIsNone(result.fallback)
        self.assertEqual(result.detections, [])


if __name__ == "__main__":
    unittest.main()
