import math
import unittest

import numpy as np

from src.pose.movements.base import CoreFrame, RuleContext, run_detector
from src.pose.movements.registry import get_detector, list_detectors
from src.pose.movements.leg_abduction import (
    ACTIVE_PHASES,
    LEG_ABDUCTION_DETECTOR,
    LEG_ABDUCTION_METRIC_KEYS,
    MOVING_SIDE_MIN_SEPARATION_DEG,
    PEAK_PHASE,
    ROM_MILD_DEG,
    TRUNK_LEAN_MILD_DEG,
    TRUNK_LEAN_MILD_RATIO,
    TRUNK_LEAN_SEVERE_DEG,
    leg_abduction_assign_phases,
    leg_abduction_compute_raw,
    resolve_moving_side,
    rule_insufficient_abduction_rom,
    rule_trunk_lean_compensation,
)

_HIP_HALF_WIDTH = 0.05
_THIGH_LEN = 0.18
_SHANK_LEN = 0.18
_TRUNK_LEN = 0.25
_SHOULDER_HALF_WIDTH = 0.07
_HIP_Y = 0.60
_CENTRE_X = 0.50


def _lm(x: float, y: float, z: float = 0.0, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": z, "visibility": visibility}


def _rot(vector: tuple[float, float], degrees: float) -> tuple[float, float]:
    rad = math.radians(degrees)
    cos, sin = math.cos(rad), math.sin(rad)
    return (vector[0] * cos - vector[1] * sin, vector[0] * sin + vector[1] * cos)


def abduction_frame(
    abduction_deg: float = 40.0,
    trunk_tilt_deg: float = 0.0,
    pelvic_hike_ratio: float = 0.0,
    moving: str = "left",
    frame_index: int = 0,
    drop_landmark: int | None = None,
    roll_deg: float = 0.0,
    mirrored: bool = False,
) -> dict:
    """One STANDING unilateral hip-abduction frame, image y growing DOWNWARD.

    Every landmark carries z=0, so the module's 2-D reads see exactly the image-plane geometry
    and each knob controls its metric BY CONSTRUCTION -- the same reason tests/test_situp.py and
    tests/test_shoulder_bridge.py need no depth correction.

    Geometry: the stance leg is planted straight down from its hip, so the support-limb
    reference every metric here is measured against points exactly along image `-y` before any
    roll is applied. The knobs are then exact:

      abduction_deg      -- angle between the MOVING thigh and the support limb. `{moving}
                            _abduction_deg` equals this exactly.
      trunk_tilt_deg     -- angle between the trunk axis and the support limb. `{moving}
                            _trunk_tilt_deg` equals this exactly. Its `sin` is the parent spec's
                            "lateral-lean as a fraction of trunk length".
      pelvic_hike_ratio  -- how far the MOVING-side hip rides up the support limb, in hip
                            widths. Positive = HIKE (the direction the labeled data separates
                            on), negative = DROP (the direction the parent spec cites).
      drop_landmark      -- zero one landmark's visibility, to exercise the all-or-nothing
                            8-landmark validity gate.
      roll_deg           -- rotate EVERY landmark about the image centre. Every quantity here is
                            an angle between two body vectors or a dot product of two body
                            vectors, so this must change nothing.
      mirrored           -- reflect every landmark about the image vertical, i.e. the subject
                            facing AWAY from the camera rather than toward it. This must also
                            change nothing, and it is the property that a cross-product sign
                            construction would NOT have -- see MirrorInvarianceTest.
    """
    stance = "right" if moving == "left" else "left"
    # Sign that puts the subject's left on image +x, flipped by `mirrored`.
    side_sign = {"left": 1.0, "right": -1.0}
    flip = -1.0 if mirrored else 1.0

    hip = {}
    for name in ("left", "right"):
        hip[name] = [
            _CENTRE_X + flip * side_sign[name] * _HIP_HALF_WIDTH,
            _HIP_Y,
        ]
    # The support limb runs straight down from the stance hip, so "up along it" is (0, -1) and
    # a hike of `h` hip widths lifts the moving hip by h * hip_width in image -y.
    hip[moving][1] -= pelvic_hike_ratio * (2.0 * _HIP_HALF_WIDTH)

    ankle = {stance: [hip[stance][0], hip[stance][1] + _THIGH_LEN + _SHANK_LEN]}
    knee = {stance: [hip[stance][0], hip[stance][1] + _THIGH_LEN]}

    # Moving thigh: the DOWNWARD direction rotated out to the side by `abduction_deg`.
    outward = -flip * side_sign[moving]
    thigh_dir = _rot((0.0, 1.0), outward * abduction_deg)
    knee[moving] = [
        hip[moving][0] + _THIGH_LEN * thigh_dir[0],
        hip[moving][1] + _THIGH_LEN * thigh_dir[1],
    ]
    ankle[moving] = [
        knee[moving][0] + _SHANK_LEN * thigh_dir[0],
        knee[moving][1] + _SHANK_LEN * thigh_dir[1],
    ]

    hip_mid = (
        (hip["left"][0] + hip["right"][0]) / 2.0,
        (hip["left"][1] + hip["right"][1]) / 2.0,
    )
    # Trunk: the UPWARD direction rotated by `trunk_tilt_deg`, leaning toward the stance side.
    lean_dir = _rot((0.0, -1.0), -outward * trunk_tilt_deg)
    shoulder_mid = (
        hip_mid[0] + _TRUNK_LEN * lean_dir[0],
        hip_mid[1] + _TRUNK_LEN * lean_dir[1],
    )
    across = (-lean_dir[1], lean_dir[0])  # perpendicular to the trunk, so the pair stays level
    shoulder = {}
    for name in ("left", "right"):
        offset = flip * side_sign[name] * _SHOULDER_HALF_WIDTH
        shoulder[name] = [shoulder_mid[0] + offset * across[0], shoulder_mid[1] + offset * across[1]]

    points = {
        11: shoulder["left"], 12: shoulder["right"],
        23: hip["left"], 24: hip["right"],
        25: knee["left"], 26: knee["right"],
        27: ankle["left"], 28: ankle["right"],
        29: ankle["left"], 30: ankle["right"],
        31: ankle["left"], 32: ankle["right"],
    }

    landmarks = []
    for index in range(33):
        x, y = points.get(index, (_CENTRE_X, 0.30))
        if roll_deg:
            dx, dy = x - 0.5, y - 0.5
            rx, ry = _rot((dx, dy), roll_deg)
            x, y = rx + 0.5, ry + 0.5
        visibility = 0.0 if index == drop_landmark else 0.95
        landmarks.append(_lm(x, y, 0.0, visibility))
    return {"frame_index": frame_index, "landmarks": landmarks}


def _core(frames: list[dict], phase: str = PEAK_PHASE) -> list[CoreFrame]:
    """Raw frames -> CoreFrames carrying every metric key, all in one phase."""
    raw = leg_abduction_compute_raw(frames, 30.0)
    core = []
    for index, item in enumerate(raw):
        core.append(
            CoreFrame(
                frame_index=int(item.get("frame_index", index)),
                time=float(item.get("time", 0.0)),
                phase=phase if item.get("valid") else "unknown",
                valid=bool(item.get("valid")),
                lower_body_visibility=float(item.get("lower_body_visibility", 0.0)),
                metrics={key: float(item.get(key, math.nan)) for key in LEG_ABDUCTION_METRIC_KEYS},
            )
        )
    return core


def _ctx(view_type: str = "rear", min_frames: int = 3) -> RuleContext:
    return RuleContext(fps=30.0, view_type=view_type, view_confidence=0.8, min_frames=min_frames)


class MetricLayerTest(unittest.TestCase):
    def test_metric_keys_match_the_emitted_metrics_exactly(self) -> None:
        """Two-way: a key the tuple omits is silently NaN in every rule, and a key the tuple
        names but nothing emits is silently NaN too."""
        raw = leg_abduction_compute_raw([abduction_frame()], 30.0)[0]
        emitted = set(raw) - {"frame_index", "time", "valid", "lower_body_visibility"}
        self.assertEqual(emitted, set(LEG_ABDUCTION_METRIC_KEYS))

    def test_the_knobs_control_their_metrics_exactly(self) -> None:
        raw = leg_abduction_compute_raw(
            [abduction_frame(abduction_deg=37.0, trunk_tilt_deg=11.0)], 30.0
        )[0]
        self.assertAlmostEqual(raw["left_abduction_deg"], 37.0, places=4)
        self.assertAlmostEqual(raw["left_trunk_tilt_deg"], 11.0, places=4)

    def test_pelvic_hike_is_signed_and_hike_is_positive(self) -> None:
        hiked = leg_abduction_compute_raw([abduction_frame(pelvic_hike_ratio=0.4)], 30.0)[0]
        dropped = leg_abduction_compute_raw([abduction_frame(pelvic_hike_ratio=-0.4)], 30.0)[0]
        self.assertGreater(hiked["left_pelvic_hike_ratio"], 0.0)
        self.assertLess(dropped["left_pelvic_hike_ratio"], 0.0)
        # The two directions are DISTINGUISHABLE, which is what Shoulder Bridge's
        # `angle_degrees` could not do for its sag/arch pair.
        self.assertNotAlmostEqual(
            hiked["left_pelvic_hike_ratio"], dropped["left_pelvic_hike_ratio"], places=3
        )

    def test_one_dropped_landmark_invalidates_the_whole_frame(self) -> None:
        for index in (11, 12, 23, 24, 25, 26, 27, 28):
            with self.subTest(landmark=index):
                raw = leg_abduction_compute_raw([abduction_frame(drop_landmark=index)], 30.0)[0]
                self.assertFalse(raw["valid"])
                for key in LEG_ABDUCTION_METRIC_KEYS:
                    self.assertNotIn(key, raw)

    def test_the_ankles_are_required_which_no_other_group_e_module_needs(self) -> None:
        """The support limb IS the ankle-to-hip line, so an occluded ankle removes this module's
        only vertical reference. Sit-up and Shoulder Bridge both require six landmarks and
        neither requires an ankle."""
        raw = leg_abduction_compute_raw([abduction_frame(drop_landmark=28)], 30.0)[0]
        self.assertFalse(raw["valid"])

    def test_a_non_dict_frame_is_invalid_rather_than_raising(self) -> None:
        self.assertEqual(leg_abduction_compute_raw(["not a frame", None], 30.0),
                         [{"valid": False}, {"valid": False}])

    def test_the_rep_signal_is_the_larger_of_the_two_trunk_referenced_angles(self) -> None:
        raw = leg_abduction_compute_raw([abduction_frame(abduction_deg=45.0)], 30.0)[0]
        self.assertAlmostEqual(
            raw["max_thigh_trunk_deg"],
            max(raw["left_thigh_trunk_deg"], raw["right_thigh_trunk_deg"]),
            places=6,
        )


class InvarianceTest(unittest.TestCase):
    """Both invariances the design argument rests on, pinned as executable claims."""

    def _metrics(self, **kwargs) -> dict:
        raw = leg_abduction_compute_raw([abduction_frame(**kwargs)], 30.0)[0]
        return {key: raw[key] for key in LEG_ABDUCTION_METRIC_KEYS}

    def test_every_metric_is_invariant_under_camera_roll(self) -> None:
        base = self._metrics()
        for roll in (17.0, 90.0, 180.0, -90.0):
            with self.subTest(roll=roll):
                rolled = self._metrics(roll_deg=roll)
                for key, value in base.items():
                    self.assertAlmostEqual(rolled[key], value, places=3)

    def test_every_metric_is_invariant_under_mirroring(self) -> None:
        """THE PROPERTY A CROSS-PRODUCT SIGN WOULD NOT HAVE.

        Shoulder Bridge's two refuted sign constructions were built from cross products, whose
        sign flips when the subject faces away from the camera instead of toward it -- something
        no monocular pipeline can tell. Every signed quantity here is a DOT product against a
        body axis, so mirroring changes nothing, including the sign of the pelvic-tilt metric.
        """
        base = self._metrics(pelvic_hike_ratio=0.35, trunk_tilt_deg=9.0)
        mirrored = self._metrics(pelvic_hike_ratio=0.35, trunk_tilt_deg=9.0, mirrored=True)
        for key, value in base.items():
            with self.subTest(metric=key):
                self.assertAlmostEqual(mirrored[key], value, places=3)

    def test_detections_are_byte_identical_under_roll_and_mirroring(self) -> None:
        def detections(**kwargs):
            frames = [
                abduction_frame(trunk_tilt_deg=14.0, frame_index=i, **kwargs) for i in range(12)
            ]
            return [
                (d.fault_id, round(d.severity, 6), d.start_frame, d.end_frame)
                for d in rule_trunk_lean_compensation(_core(frames), _ctx())
            ]

        base = detections()
        self.assertTrue(base, "fixture must fire for this test to mean anything")
        self.assertEqual(detections(roll_deg=90.0), base)
        self.assertEqual(detections(mirrored=True), base)


class PhaseTest(unittest.TestCase):
    def _rising_falling(self, count: int = 30) -> list[dict]:
        peak = count // 2
        frames = []
        for i in range(count):
            fraction = i / peak if i <= peak else (count - 1 - i) / max(1, count - 1 - peak)
            frames.append(abduction_frame(abduction_deg=45.0 * fraction, frame_index=i))
        return frames

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(leg_abduction_assign_phases([]), [])

    def test_a_clip_with_no_finite_signal_is_entirely_unknown(self) -> None:
        raw = leg_abduction_compute_raw([abduction_frame(drop_landmark=23) for _ in range(5)], 30.0)
        self.assertEqual(leg_abduction_assign_phases(raw), ["unknown"] * 5)

    def test_an_invalid_frame_inside_the_setup_window_is_unknown_not_setup(self) -> None:
        frames = self._rising_falling()
        frames[1] = abduction_frame(abduction_deg=0.0, frame_index=1, drop_landmark=27)
        phases = leg_abduction_assign_phases(leg_abduction_compute_raw(frames, 30.0))
        self.assertEqual(phases[1], "unknown")

    def test_the_phases_run_setup_concentric_peak_eccentric(self) -> None:
        phases = leg_abduction_assign_phases(
            leg_abduction_compute_raw(self._rising_falling(), 30.0)
        )
        self.assertEqual(phases[0], "setup")
        self.assertIn("concentric", phases)
        self.assertIn(PEAK_PHASE, phases)
        self.assertIn("eccentric", phases)
        self.assertEqual(phases[len(phases) // 2], PEAK_PHASE)


class ResolveMovingSideTest(unittest.TestCase):
    def test_it_names_the_leg_that_is_carried_out(self) -> None:
        for moving in ("left", "right"):
            with self.subTest(moving=moving):
                core = _core([abduction_frame(abduction_deg=40.0, moving=moving)])
                self.assertEqual(resolve_moving_side(core), moving)

    def test_it_refuses_rather_than_guesses_when_the_legs_are_close(self) -> None:
        core = _core([abduction_frame(abduction_deg=MOVING_SIDE_MIN_SEPARATION_DEG - 1.0)])
        self.assertIsNone(resolve_moving_side(core))

    def test_it_refuses_on_a_window_with_no_valid_frame(self) -> None:
        self.assertIsNone(resolve_moving_side(_core([abduction_frame(drop_landmark=24)])))

    def test_a_rule_is_silent_when_the_side_is_unresolvable(self) -> None:
        """A refused side must silence the rule, not fall back to a default leg."""
        frames = [
            abduction_frame(abduction_deg=1.0, trunk_tilt_deg=25.0, frame_index=i)
            for i in range(12)
        ]
        self.assertEqual(rule_trunk_lean_compensation(_core(frames), _ctx()), [])


class TrunkLeanRuleTest(unittest.TestCase):
    def _frames(self, tilt: float, count: int = 12, **kwargs) -> list[dict]:
        return [abduction_frame(trunk_tilt_deg=tilt, frame_index=i, **kwargs) for i in range(count)]

    def test_the_threshold_is_the_spec_ratio_rendered_as_an_angle(self) -> None:
        self.assertAlmostEqual(math.sin(math.radians(TRUNK_LEAN_MILD_DEG)),
                               TRUNK_LEAN_MILD_RATIO, places=9)

    def test_it_fires_above_the_threshold_and_is_silent_below(self) -> None:
        self.assertEqual(rule_trunk_lean_compensation(
            _core(self._frames(TRUNK_LEAN_MILD_DEG - 0.01)), _ctx()), [])
        fired = rule_trunk_lean_compensation(_core(self._frames(TRUNK_LEAN_MILD_DEG + 0.01)), _ctx())
        self.assertEqual([d.fault_id for d in fired], ["abd_pelvic_drop_trunk_lean"])

    def test_severity_ramps_between_the_two_ratios(self) -> None:
        mild = rule_trunk_lean_compensation(
            _core(self._frames(TRUNK_LEAN_MILD_DEG + 0.5)), _ctx())[0]
        severe = rule_trunk_lean_compensation(
            _core(self._frames(TRUNK_LEAN_SEVERE_DEG + 5.0)), _ctx())[0]
        self.assertLess(mild.severity, 0.2)
        self.assertEqual(severe.severity, 1.0)

    def test_a_brief_exceedance_shorter_than_min_frames_does_not_fire(self) -> None:
        frames = self._frames(0.0, count=12)
        for index in (4, 5):
            frames[index] = abduction_frame(trunk_tilt_deg=25.0, frame_index=index)
        self.assertEqual(rule_trunk_lean_compensation(_core(frames), _ctx(min_frames=3)), [])

    def test_it_ignores_the_setup_phase(self) -> None:
        core = _core(self._frames(25.0), phase="setup")
        self.assertEqual(rule_trunk_lean_compensation(core, _ctx()), [])
        self.assertTrue(ACTIVE_PHASES)

    def test_confidence_is_discounted_off_a_frontal_view_but_severity_is_not(self) -> None:
        core = _core(self._frames(TRUNK_LEAN_SEVERE_DEG + 5.0))
        frontal = rule_trunk_lean_compensation(core, _ctx(view_type="rear"))[0]
        oblique = rule_trunk_lean_compensation(core, _ctx(view_type="rear_oblique"))[0]
        self.assertEqual(frontal.severity, oblique.severity)
        self.assertLess(oblique.confidence, frontal.confidence)
        self.assertEqual(frontal.observability, "high")
        self.assertEqual(oblique.observability, "medium")

    def test_the_evidence_reports_both_the_angle_and_the_specs_own_ratio(self) -> None:
        detection = rule_trunk_lean_compensation(_core(self._frames(20.0)), _ctx())[0]
        self.assertAlmostEqual(detection.evidence["max_trunk_lean_deg"], 20.0, places=2)
        self.assertAlmostEqual(
            detection.evidence["max_trunk_lean_ratio"], math.sin(math.radians(20.0)), places=3
        )
        self.assertEqual(detection.evidence["threshold_ratio"], TRUNK_LEAN_MILD_RATIO)
        self.assertEqual(detection.evidence["moving_side"], "left")

    def test_it_reads_the_support_limb_and_not_the_image_vertical(self) -> None:
        """The distinguishing property against `arm_abduction.rule_contralateral_trunk_lean`,
        which measures the same compensation from the IMAGE vertical.

        BOTH HALVES ARE ASSERTED, because the silence alone proves nothing: a fixture with zero
        trunk tilt is silent under either reference frame. So this also computes the
        image-vertical form on the SAME rolled frames and asserts it would have fired well past
        the cut. A subject rolled 90 degrees whose trunk is still aligned with their planted leg
        has not leaned; an image-vertical rule calls that a 90-degree lean.
        """
        frames = self._frames(0.0, roll_deg=90.0)
        self.assertEqual(rule_trunk_lean_compensation(_core(frames), _ctx()), [])

        from src.pose.geometry import landmarks_to_array, midpoint

        image_vertical_leans = []
        for frame in frames:
            points = landmarks_to_array(frame["landmarks"])
            shoulder_mid = midpoint(points, 11, 12, dims=2)
            hip_mid = midpoint(points, 23, 24, dims=2)
            trunk = shoulder_mid - hip_mid
            image_vertical_leans.append(
                abs(math.degrees(math.atan2(float(trunk[0]), -float(trunk[1]))))
            )
        self.assertGreater(min(image_vertical_leans), TRUNK_LEAN_MILD_DEG)
        self.assertAlmostEqual(min(image_vertical_leans), 90.0, places=3)

    def test_the_citation_records_that_its_support_is_secondary(self) -> None:
        detection = rule_trunk_lean_compensation(_core(self._frames(20.0)), _ctx())[0]
        self.assertIn("SECONDARY", detection.citation_support)
        self.assertIn("PMC12372021", detection.citation)


class SilentRomRuleTest(unittest.TestCase):
    def test_it_never_fires_even_on_a_repetition_that_trips_the_specs_cut(self) -> None:
        """`rule_insufficient_abduction_rom` is registered PERMANENTLY SILENT. The fixture below
        peaks well under the spec's own ~30 degree cut, which is exactly the case the spec says
        to flag -- and the rule must still return nothing."""
        frames = [
            abduction_frame(abduction_deg=ROM_MILD_DEG - 20.0, frame_index=i) for i in range(20)
        ]
        core = _core(frames)
        peak = max(frame.m("left_abduction_deg") for frame in core)
        self.assertLess(peak, ROM_MILD_DEG)
        self.assertEqual(rule_insufficient_abduction_rom(core, _ctx()), [])

    def test_it_is_registered_rather_than_absent(self) -> None:
        self.assertIn(rule_insufficient_abduction_rom, LEG_ABDUCTION_DETECTOR.rules)


class RegistrationTest(unittest.TestCase):
    def test_the_detector_is_registered_under_its_canonical_name(self) -> None:
        self.assertIs(get_detector("Leg Abduction"), LEG_ABDUCTION_DETECTOR)
        self.assertIn(LEG_ABDUCTION_DETECTOR, list_detectors())

    def test_it_ships_beta(self) -> None:
        self.assertFalse(LEG_ABDUCTION_DETECTOR.validated)

    def test_the_rep_signal_is_a_declared_metric_key(self) -> None:
        self.assertIn(LEG_ABDUCTION_DETECTOR.rep_signal, LEG_ABDUCTION_METRIC_KEYS)
        self.assertEqual(LEG_ABDUCTION_DETECTOR.rep_polarity, "max")


class EndToEndSegmentationTest(unittest.TestCase):
    def _clip(self, reps: int, tilt: float, frames_per_rep: int = 30) -> list[dict]:
        frames: list[dict] = []
        for _ in range(reps):
            half = frames_per_rep // 2
            for i in range(frames_per_rep):
                fraction = i / half if i <= half else (frames_per_rep - 1 - i) / max(1, half - 1)
                frames.append(
                    abduction_frame(
                        abduction_deg=45.0 * max(0.0, min(1.0, fraction)),
                        trunk_tilt_deg=tilt * max(0.0, min(1.0, fraction)),
                        frame_index=len(frames),
                    )
                )
        return frames

    def test_a_three_rep_clip_segments_and_scores_through_run_detector(self) -> None:
        result = run_detector(LEG_ABDUCTION_DETECTOR, self._clip(3, tilt=25.0), 30.0, "rear", 0.8)
        self.assertGreaterEqual(len(result.reps), 2)
        self.assertEqual([d.fault_id for d in result.detections], ["abd_pelvic_drop_trunk_lean"])

    def test_a_clean_clip_produces_no_detections(self) -> None:
        result = run_detector(LEG_ABDUCTION_DETECTOR, self._clip(3, tilt=0.0), 30.0, "rear", 0.8)
        # NOT VACUOUS: assert the clip really was segmented and scored, so "no detections"
        # means the rule declined rather than that nothing reached it.
        self.assertGreaterEqual(len(result.analyzed), 2)
        self.assertIsNone(result.fallback)
        self.assertEqual(result.detections, [])

    def test_the_peak_phase_lands_on_the_most_abducted_frames(self) -> None:
        """Pins WHICH end of the signal `peak` names, so the polarity cannot silently invert."""
        result = run_detector(LEG_ABDUCTION_DETECTOR, self._clip(2, tilt=0.0), 30.0, "rear", 0.8)
        peaks = [f.m("max_thigh_trunk_deg") for f in result.core if f.phase == PEAK_PHASE]
        others = [
            f.m("max_thigh_trunk_deg")
            for f in result.core
            if f.valid and f.phase in ("concentric", "eccentric")
        ]
        self.assertTrue(peaks and others)
        self.assertGreater(float(np.nanmin(peaks)), float(np.nanmin(others)))


if __name__ == "__main__":
    unittest.main()
