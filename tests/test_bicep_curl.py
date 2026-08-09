import math
import unittest

import numpy as np

from src.pose.movements.bicep_curl import (
    BICEP_CURL_DETECTOR,
    BICEP_CURL_METRIC_KEYS,
    DRIFT_MILD_DEG,
    bicep_curl_assign_phases,
    bicep_curl_compute_raw,
    rule_elbow_drift_forward,
    rule_incomplete_rom,
    rule_trunk_swing_momentum,
)
from src.pose.movements.base import CoreFrame, RuleContext, run_detector
from src.pose.movements import registry


def _lm(x: float, y: float, z: float = 0.0, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": z, "visibility": visibility}


_UPPER_ARM = 0.15
_FOREARM = 0.14


def curl_frame(
    elbow_angle_deg: float = 170.0,
    upper_arm_lean_deg: float = 0.0,
    trunk_lean_deg: float = 0.0,
    left_elbow_angle_deg: float | None = None,
    left_lean_deg: float | None = None,
    frame_index: int = 0,
    visibility: float = 0.95,
) -> dict:
    """One standing bicep-curl frame, image y growing DOWNWARD.

    Every landmark carries z=0, so `angle_degrees` (which computes with dims=3) sees exactly the
    image-plane triangle and each knob controls its metric BY CONSTRUCTION. That is the whole
    reason this fixture needs no `_elbow_xyz`-style 3D correction of the kind
    tests/test_band_pull_apart.py requires: that movement's fixture gives the wrists a nonzero z
    to encode facing, and a curl's rules read no depth at all.

    Knobs:
      elbow_angle_deg     -- angle(shoulder, elbow, wrist) on the RIGHT arm, and on the left too
                             unless `left_elbow_angle_deg` overrides it. 180 = straight.
      upper_arm_lean_deg  -- unsigned angle of shoulder->elbow from image-vertical-down, on the
                             RIGHT arm and on the left unless `left_lean_deg` overrides it.
                             Equals `right_upper_arm_lean_deg`.
      trunk_lean_deg      -- signed pitch of hip_mid -> shoulder_mid from vertical. Positive
                             moves the shoulders toward +x. Equals
                             `trunk_lean_image_signed_deg`.
      left_elbow_angle_deg / left_lean_deg -- left-arm overrides, so a fixture can separate
                             `min_elbow_angle` from `max_elbow_angle` and pin which arm each
                             rule reads.
    """
    shoulder_width = 0.20
    hip_mid = (0.50, 0.70)
    trunk_len = 0.30
    theta = math.radians(trunk_lean_deg)
    shoulder_mid = (
        hip_mid[0] + trunk_len * math.sin(theta),
        hip_mid[1] - trunk_len * math.cos(theta),
    )
    sy = shoulder_mid[1]
    left_shoulder = (shoulder_mid[0] - shoulder_width / 2.0, sy)
    right_shoulder = (shoulder_mid[0] + shoulder_width / 2.0, sy)

    def arm(shoulder: tuple[float, float], lean_deg: float, angle_deg: float, side_sign: float):
        """Place elbow and wrist so BOTH the upper-arm lean and the elbow angle are exact.

        The elbow sits one upper-arm length from the shoulder at `lean_deg` off
        image-vertical-DOWN (+y), which is what `line_angle_from_vertical(shoulder, elbow)`
        measures. The wrist is then the elbow->shoulder unit vector rotated by the requested
        elbow angle, so `angle_degrees(shoulder, elbow, wrist)` equals it exactly by
        construction rather than approximately.
        """
        lean = math.radians(lean_deg)
        elbow = (
            shoulder[0] + side_sign * _UPPER_ARM * math.sin(lean),
            shoulder[1] + _UPPER_ARM * math.cos(lean),
        )
        ux = (shoulder[0] - elbow[0]) / _UPPER_ARM
        uy = (shoulder[1] - elbow[1]) / _UPPER_ARM
        phi = side_sign * math.radians(angle_deg)
        rx = ux * math.cos(phi) - uy * math.sin(phi)
        ry = ux * math.sin(phi) + uy * math.cos(phi)
        wrist = (elbow[0] + _FOREARM * rx, elbow[1] + _FOREARM * ry)
        return elbow, wrist

    right_elbow, right_wrist = arm(right_shoulder, upper_arm_lean_deg, elbow_angle_deg, 1.0)
    left_elbow, left_wrist = arm(
        left_shoulder,
        upper_arm_lean_deg if left_lean_deg is None else left_lean_deg,
        elbow_angle_deg if left_elbow_angle_deg is None else left_elbow_angle_deg,
        -1.0,
    )

    landmarks = [_lm(0.5, 0.3, visibility=visibility) for _ in range(33)]
    landmarks[11] = _lm(*left_shoulder, visibility=visibility)
    landmarks[12] = _lm(*right_shoulder, visibility=visibility)
    landmarks[13] = _lm(*left_elbow, visibility=visibility)
    landmarks[14] = _lm(*right_elbow, visibility=visibility)
    landmarks[15] = _lm(*left_wrist, visibility=visibility)
    landmarks[16] = _lm(*right_wrist, visibility=visibility)
    landmarks[23] = _lm(hip_mid[0] - 0.08, hip_mid[1], visibility=visibility)
    landmarks[24] = _lm(hip_mid[0] + 0.08, hip_mid[1], visibility=visibility)
    return {"frame_index": frame_index, "landmarks": landmarks}


def _ctx(view_type: str = "rear_oblique", view_confidence: float = 0.8, min_frames: int = 3) -> RuleContext:
    """Defaults to `rear_oblique`, the MODAL production view (37 of 49 real pose JSONs), rather
    than to `side`, which the design spec measured as never occurring in production."""
    return RuleContext(
        fps=30.0, view_type=view_type, view_confidence=view_confidence, min_frames=min_frames
    )


def _core(frames: list[dict], fps: float = 30.0) -> list[CoreFrame]:
    """Metrics + phases WITHOUT run_detector, so a rule can be tested on an exact fixture.

    run_detector median-filters and re-slices per rep; both are correct in production and both
    would blur a boundary fixture built to sit one step either side of a threshold.
    """
    raw = bicep_curl_compute_raw(frames, fps=fps)
    phases = bicep_curl_assign_phases(raw)
    return [
        CoreFrame(
            frame_index=int(item.get("frame_index", i) or i),
            time=float(item.get("time", 0.0) or 0.0),
            phase=phases[i],
            valid=bool(item.get("valid", False)),
            lower_body_visibility=float(item.get("lower_body_visibility", 0.0) or 0.0),
            metrics={k: float(item.get(k, np.nan)) for k in BICEP_CURL_METRIC_KEYS},
        )
        for i, item in enumerate(raw)
    ]


class BicepCurlMetricsTest(unittest.TestCase):
    def test_elbow_angle_knob_equals_the_emitted_metrics(self) -> None:
        raw = bicep_curl_compute_raw([curl_frame(elbow_angle_deg=140.0)], fps=30.0)
        self.assertTrue(raw[0]["valid"])
        for key in ("left_elbow_angle", "right_elbow_angle", "avg_elbow_angle",
                    "min_elbow_angle", "max_elbow_angle"):
            self.assertAlmostEqual(raw[0][key], 140.0, places=3, msg=key)

    def test_asymmetric_arms_separate_min_avg_and_max(self) -> None:
        raw = bicep_curl_compute_raw(
            [curl_frame(elbow_angle_deg=170.0, left_elbow_angle_deg=110.0)], fps=30.0
        )
        self.assertAlmostEqual(raw[0]["min_elbow_angle"], 110.0, places=3)
        self.assertAlmostEqual(raw[0]["max_elbow_angle"], 170.0, places=3)
        self.assertAlmostEqual(raw[0]["avg_elbow_angle"], 140.0, places=3)

    def test_upper_arm_lean_knob_equals_the_emitted_metric(self) -> None:
        raw = bicep_curl_compute_raw([curl_frame(upper_arm_lean_deg=32.0)], fps=30.0)
        self.assertAlmostEqual(raw[0]["right_upper_arm_lean_deg"], 32.0, places=3)
        self.assertAlmostEqual(raw[0]["left_upper_arm_lean_deg"], 32.0, places=3)
        self.assertAlmostEqual(raw[0]["max_upper_arm_lean_deg"], 32.0, places=3)

    def test_max_upper_arm_lean_takes_the_worse_arm(self) -> None:
        raw = bicep_curl_compute_raw(
            [curl_frame(upper_arm_lean_deg=5.0, left_lean_deg=40.0)], fps=30.0
        )
        self.assertAlmostEqual(raw[0]["max_upper_arm_lean_deg"], 40.0, places=3)

    def test_upper_arm_lean_is_unsigned_so_both_directions_read_alike(self) -> None:
        """The parent spec's "toward the anterior side" qualifier is deliberately dropped: a
        backward drift is the same departure from the vertical hang, and reading a direction
        would need a facing proxy no citation supplies (design spec 4.8)."""
        forward = bicep_curl_compute_raw([curl_frame(upper_arm_lean_deg=30.0)], fps=30.0)
        backward = bicep_curl_compute_raw([curl_frame(upper_arm_lean_deg=-30.0)], fps=30.0)
        # places=4, not 6: `landmarks_to_array` stores float32, so mirroring the fixture moves
        # the elbow through a different rounding and the two leans differ by ~1e-5 degrees. That
        # is the storage precision, not a signed residual.
        self.assertAlmostEqual(
            forward[0]["max_upper_arm_lean_deg"], backward[0]["max_upper_arm_lean_deg"], places=4
        )

    def test_trunk_lean_knob_equals_the_signed_image_pitch(self) -> None:
        raw = bicep_curl_compute_raw([curl_frame(trunk_lean_deg=14.0)], fps=30.0)
        self.assertAlmostEqual(raw[0]["trunk_lean_image_signed_deg"], 14.0, places=3)
        raw = bicep_curl_compute_raw([curl_frame(trunk_lean_deg=-14.0)], fps=30.0)
        self.assertAlmostEqual(raw[0]["trunk_lean_image_signed_deg"], -14.0, places=3)

    def test_one_missing_landmark_invalidates_the_whole_frame(self) -> None:
        """All-or-nothing validity: a dropped wrist silences EVERY rule for that frame, not just
        the one that reads wrists."""
        frame = curl_frame()
        frame["landmarks"][16] = _lm(0.5, 0.5, visibility=0.01)
        raw = bicep_curl_compute_raw([frame], fps=30.0)
        self.assertFalse(raw[0]["valid"])
        self.assertNotIn("avg_elbow_angle", raw[0])

    def test_non_dict_frame_is_refused_rather_than_guessed(self) -> None:
        raw = bicep_curl_compute_raw(["not a frame"], fps=30.0)
        self.assertFalse(raw[0]["valid"])

    def test_metric_keys_match_the_emitted_metrics_exactly(self) -> None:
        """Two-way match. A key the tuple omits is dropped by run_detector (which builds each
        CoreFrame's metrics dict FROM this tuple) and read back as NaN by every rule; a key the
        tuple names but compute_raw never emits is a silent NaN column."""
        raw = bicep_curl_compute_raw([curl_frame()], fps=30.0)
        emitted = set(raw[0]) - {"frame_index", "time", "valid", "lower_body_visibility"}
        self.assertEqual(emitted, set(BICEP_CURL_METRIC_KEYS))


class BicepCurlPhaseTest(unittest.TestCase):
    def _rep(self, n: int = 20) -> list[dict]:
        """One curl: extended at both ends, most flexed in the middle."""
        frames = []
        for i in range(n):
            t = i / (n - 1)
            angle = 170.0 - 130.0 * math.sin(math.pi * t)
            frames.append(curl_frame(elbow_angle_deg=angle, frame_index=i))
        return frames

    def test_phases_run_setup_concentric_peak_eccentric(self) -> None:
        phases = bicep_curl_assign_phases(bicep_curl_compute_raw(self._rep(), fps=30.0))
        self.assertEqual(phases[0], "setup")
        self.assertIn("concentric", phases)
        self.assertIn("peak", phases)
        self.assertIn("eccentric", phases)

    def test_setup_is_the_extended_end_not_the_flexed_one(self) -> None:
        """rep_start="extended" means the window opens at the bottom of the curl, which is what
        makes `setup` the right scope for the incomplete-EXTENSION term."""
        raw = bicep_curl_compute_raw(self._rep(), fps=30.0)
        phases = bicep_curl_assign_phases(raw)
        setup_angles = [raw[i]["avg_elbow_angle"] for i, p in enumerate(phases) if p == "setup"]
        peak_angles = [raw[i]["avg_elbow_angle"] for i, p in enumerate(phases) if p == "peak"]
        self.assertGreater(min(setup_angles), max(peak_angles))

    def test_empty_clip_returns_empty(self) -> None:
        self.assertEqual(bicep_curl_assign_phases([]), [])

    def test_clip_with_no_finite_signal_is_entirely_unknown(self) -> None:
        raw = [{"valid": False}, {"valid": False}]
        self.assertEqual(bicep_curl_assign_phases(raw), ["unknown", "unknown"])

    def test_an_invalid_frame_inside_the_setup_window_is_unknown_not_setup(self) -> None:
        """The validity check precedes the setup cutoff on purpose: `_setup_baseline` reduces
        over exactly these frames, so an occluded one must not be counted as setup."""
        frames = self._rep()
        frames[0]["landmarks"][15] = _lm(0.5, 0.5, visibility=0.01)
        phases = bicep_curl_assign_phases(bicep_curl_compute_raw(frames, fps=30.0))
        self.assertEqual(phases[0], "unknown")


def _drift_rep(lean_deg: float, n: int = 20, peak_only: bool = False) -> list[dict]:
    """A curl whose CONCENTRIC frames carry `lean_deg` of upper-arm lean.

    `peak_only=True` puts the lean on the peak frames instead, to pin that the rule's phase
    scope really is the spec's `concentric` and was not quietly widened.
    """
    frames = []
    for i in range(n):
        t = i / (n - 1)
        angle = 170.0 - 130.0 * math.sin(math.pi * t)
        flexed = angle <= 70.0
        target = (flexed if peak_only else (not flexed and i >= int(n * 0.15) and t < 0.5))
        frames.append(
            curl_frame(
                elbow_angle_deg=angle,
                upper_arm_lean_deg=lean_deg if target else 0.0,
                frame_index=i,
            )
        )
    return frames


class ElbowDriftRuleTest(unittest.TestCase):
    def test_fires_above_the_spec_threshold(self) -> None:
        detections = rule_elbow_drift_forward(_core(_drift_rep(40.0)), _ctx())
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "curl_elbow_drift_forward")
        self.assertGreater(detections[0].severity, 0.0)

    def test_silent_just_below_the_spec_threshold(self) -> None:
        self.assertEqual(rule_elbow_drift_forward(_core(_drift_rep(24.0)), _ctx()), [])

    def test_silent_when_the_drift_sits_outside_the_concentric(self) -> None:
        """Phase scope is the spec's own ("at any frame during concentric"). `peak` is NOT
        included even though drift is largest there -- widening a phase scope is a rule-level
        change the spec's wording does not authorise."""
        self.assertEqual(
            rule_elbow_drift_forward(_core(_drift_rep(40.0, peak_only=True)), _ctx()), []
        )

    def test_gated_off_the_two_frontal_views(self) -> None:
        """A sagittal quantity read from a pure front/rear view is a CONFIDENT reading of the
        WRONG PLANE -- lateral elbow flare, not forward drift -- so it gates rather than
        downgrades."""
        core = _core(_drift_rep(40.0))
        self.assertEqual(rule_elbow_drift_forward(core, _ctx(view_type="rear")), [])
        self.assertEqual(rule_elbow_drift_forward(core, _ctx(view_type="front")), [])

    def test_survives_on_the_oblique_and_on_unknown(self) -> None:
        """The gate is written in the NEGATIVE so `unknown` (3 of 49 real pose JSONs) passes
        rather than being excluded by an omission from a whitelist."""
        core = _core(_drift_rep(40.0))
        self.assertEqual(len(rule_elbow_drift_forward(core, _ctx(view_type="rear_oblique"))), 1)
        self.assertEqual(len(rule_elbow_drift_forward(core, _ctx(view_type="unknown"))), 1)

    def test_confidence_is_discounted_off_the_sagittal_view(self) -> None:
        core = _core(_drift_rep(40.0))
        side = rule_elbow_drift_forward(core, _ctx(view_type="side"))[0]
        oblique = rule_elbow_drift_forward(core, _ctx(view_type="rear_oblique"))[0]
        self.assertEqual(side.observability, "high")
        self.assertEqual(oblique.observability, "medium")
        self.assertLess(oblique.confidence, side.confidence)
        self.assertEqual(side.severity, oblique.severity)

    def test_the_displacement_disjunct_is_unreachable(self) -> None:
        """Pins design spec 4.9 numerically, so a future edit that "restores" the parent spec's
        second drift cue has to confront the arithmetic first.

        The cue reads "elbow x-displacement anterior of the shoulder-hip vertical line exceeds
        0.5 x upper_arm_length". Displacement = upper_arm_length * sin(lean), so its threshold IS
        lean > arcsin(0.5) = 30 degrees -- strictly inside the 25 degrees the angular term
        already applies. Every frame it could catch, the angular term has caught.
        """
        self.assertAlmostEqual(math.degrees(math.asin(0.5)), 30.0, places=6)
        self.assertLess(DRIFT_MILD_DEG, 30.0)
        # The band between the two thresholds is live: the angular term fires there and the
        # displacement term does not, so the latter can never be the deciding condition.
        raw = bicep_curl_compute_raw([curl_frame(upper_arm_lean_deg=27.0)], fps=30.0)
        lean = raw[0]["max_upper_arm_lean_deg"]
        displacement_ratio = math.sin(math.radians(lean))
        self.assertGreater(lean, DRIFT_MILD_DEG)
        self.assertLess(displacement_ratio, 0.5)


def _swing_rep(amplitude: float, n: int = 20) -> list[dict]:
    """A curl that holds a STEADY trunk through setup and then oscillates by +/-`amplitude`.

    The steady setup is what makes this fixture isolate term (a). An earlier version oscillated
    from frame 0, which put the setup baseline at one EXTREME of the oscillation -- so a
    +/-7-degree swing produced a 14-degree deviation from baseline and fired term (b) as well,
    making the fixture useless for showing the terms are independent. The baseline must sit at
    the CENTRE of the swing for the range term to be the only one that can fire.
    """
    frames = []
    cutoff = max(1, int(n * 0.15))
    for i in range(n):
        t = i / (n - 1)
        angle = 170.0 - 130.0 * math.sin(math.pi * t)
        lean = 0.0 if i < cutoff else (amplitude if i % 2 else -amplitude)
        frames.append(
            curl_frame(elbow_angle_deg=angle, trunk_lean_deg=lean, frame_index=i)
        )
    return frames


def _sustained_lean_rep(setup_lean: float, later_lean: float, n: int = 20) -> list[dict]:
    frames = []
    cutoff = max(1, int(n * 0.15))
    for i in range(n):
        t = i / (n - 1)
        angle = 170.0 - 130.0 * math.sin(math.pi * t)
        frames.append(
            curl_frame(
                elbow_angle_deg=angle,
                trunk_lean_deg=setup_lean if i < cutoff else later_lean,
                frame_index=i,
            )
        )
    return frames


class TrunkSwingRuleTest(unittest.TestCase):
    def test_within_rep_range_term_fires_on_its_own(self) -> None:
        detections = rule_trunk_swing_momentum(
            _core(_sustained_lean_rep(setup_lean=0.0, later_lean=0.0)), _ctx()
        )
        self.assertEqual(detections, [])
        detections = rule_trunk_swing_momentum(_core(_swing_rep(8.0)), _ctx())
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "curl_trunk_swing_momentum")
        self.assertEqual(
            detections[0].evidence["primary_label"], "within-rep trunk swing range"
        )

    def test_the_two_terms_are_a_genuine_disjunction_not_a_subset(self) -> None:
        """Neither term nests inside the other, and this is the check `row.rule_momentum_jerk`
        did not get -- its second condition turned out to be a strict subset of its first.

        (a) alone: oscillating +/-7 degrees about a steady setup gives a 14-degree range
        (fires, > 12) but only a 7-degree deviation from baseline (does not fire, < 10).
        (b) alone: a sustained 11-degree lean away from setup gives an 11-degree range (does not
        fire, < 12) but an 11-degree deviation (fires, > 10).
        """
        range_only = rule_trunk_swing_momentum(_core(_swing_rep(7.0)), _ctx())
        self.assertEqual(len(range_only), 1)
        self.assertEqual(
            range_only[0].evidence["primary_label"], "within-rep trunk swing range"
        )

        baseline_only = rule_trunk_swing_momentum(
            _core(_sustained_lean_rep(setup_lean=0.0, later_lean=11.0)), _ctx()
        )
        self.assertEqual(len(baseline_only), 1)
        self.assertEqual(
            baseline_only[0].evidence["primary_label"], "trunk lean vs setup baseline"
        )
        self.assertLess(baseline_only[0].evidence["within_rep_swing_range_deg"], 12.0)

    def test_silent_on_a_steady_trunk(self) -> None:
        self.assertEqual(
            rule_trunk_swing_momentum(_core(_swing_rep(4.0)), _ctx()), []
        )

    def test_is_facing_free_so_a_mirrored_rep_scores_identically(self) -> None:
        """Neither term reads the sign of `trunk_lean_image_signed_deg` -- one is a range, the
        other an absolute deviation -- which is why this rule needs no facing proxy."""
        forward = rule_trunk_swing_momentum(
            _core(_sustained_lean_rep(setup_lean=0.0, later_lean=14.0)), _ctx()
        )[0]
        mirrored = rule_trunk_swing_momentum(
            _core(_sustained_lean_rep(setup_lean=0.0, later_lean=-14.0)), _ctx()
        )[0]
        self.assertAlmostEqual(forward.severity, mirrored.severity, places=6)

    def test_a_nan_setup_baseline_silences_only_the_second_term(self) -> None:
        """Pins the design spec's claim that a missing baseline is not a whole-rule kill switch.

        With every setup frame occluded, `_setup_baseline` returns NaN, so term (b) cannot fire
        at all -- but term (a) is a range over the rep and needs no baseline, so the rule still
        reports. The detection must still nominate a peak frame, which is what the median
        fallback for the DISPLAY reference exists for; that fallback decides nothing about
        firing.
        """
        frames = _swing_rep(8.0)
        for i in range(max(1, int(len(frames) * 0.15))):
            frames[i]["landmarks"][15] = _lm(0.5, 0.5, visibility=0.01)
        detections = rule_trunk_swing_momentum(_core(frames), _ctx())
        self.assertEqual(len(detections), 1)
        self.assertIsNone(detections[0].evidence["setup_baseline_lean_deg"])
        self.assertEqual(
            detections[0].evidence["primary_label"], "within-rep trunk swing range"
        )
        self.assertIn("peak_time", detections[0].evidence)

    def test_gated_off_the_two_frontal_views(self) -> None:
        core = _core(_swing_rep(8.0))
        self.assertEqual(rule_trunk_swing_momentum(core, _ctx(view_type="rear")), [])
        self.assertEqual(rule_trunk_swing_momentum(core, _ctx(view_type="front")), [])
        self.assertEqual(len(rule_trunk_swing_momentum(core, _ctx(view_type="unknown"))), 1)


def _rom_rep(
    setup_angle: float = 170.0, peak_angle: float = 40.0, n: int = 20,
    left_setup_angle: float | None = None, left_peak_angle: float | None = None,
) -> list[dict]:
    """A curl holding `setup_angle` at the bottom and `peak_angle` at the top.

    The left-arm overrides exist to pin WHICH arm each ROM term reads: extension takes the
    less-flexed (straighter) arm and flexion the more-flexed one, so a rep is called incomplete
    only when BOTH arms fell short at that end.
    """
    frames = []
    cutoff = max(1, int(n * 0.15))
    for i in range(n):
        top = i >= int(n * 0.45) and i < int(n * 0.75)
        if i < cutoff:
            right, left = setup_angle, left_setup_angle
        elif top:
            right, left = peak_angle, left_peak_angle
        else:
            right, left = 110.0, None
        frames.append(
            curl_frame(
                elbow_angle_deg=right,
                left_elbow_angle_deg=left,
                frame_index=i,
            )
        )
    return frames


class IncompleteRomRuleTest(unittest.TestCase):
    def test_fires_on_incomplete_extension_at_the_bottom(self) -> None:
        detections = rule_incomplete_rom(_core(_rom_rep(setup_angle=130.0)), _ctx())
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "curl_incomplete_rom")
        self.assertEqual(detections[0].evidence["fired_on"], "extension")
        self.assertEqual(detections[0].evidence["primary_threshold"], 150.0)

    def test_fires_on_incomplete_flexion_at_the_top(self) -> None:
        detections = rule_incomplete_rom(_core(_rom_rep(peak_angle=95.0)), _ctx())
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].evidence["fired_on"], "flexion")
        self.assertEqual(detections[0].evidence["primary_threshold"], 60.0)

    def test_both_ends_produce_two_separate_detections(self) -> None:
        """The two terms live in DIFFERENT, non-adjacent phases (`concentric` separates them),
        so no contiguous segment can span both and no detection mixes their evidence."""
        detections = rule_incomplete_rom(
            _core(_rom_rep(setup_angle=130.0, peak_angle=95.0)), _ctx()
        )
        self.assertEqual(len(detections), 2)
        self.assertEqual(
            {d.evidence["fired_on"] for d in detections}, {"extension", "flexion"}
        )

    def test_silent_on_a_full_range_rep(self) -> None:
        self.assertEqual(rule_incomplete_rom(_core(_rom_rep()), _ctx()), [])

    def test_extension_reads_the_straighter_arm_so_one_good_arm_silences_it(self) -> None:
        """The generous reading, chosen because design spec 2 measured this threshold sitting
        within ~1 degree of the edge of the real-rep distribution."""
        self.assertEqual(
            rule_incomplete_rom(
                _core(_rom_rep(setup_angle=130.0, left_setup_angle=170.0)), _ctx()
            ),
            [],
        )

    def test_flexion_reads_the_more_flexed_arm_so_one_good_arm_silences_it(self) -> None:
        self.assertEqual(
            rule_incomplete_rom(
                _core(_rom_rep(peak_angle=95.0, left_peak_angle=40.0)), _ctx()
            ),
            [],
        )

    def test_downgrades_rather_than_gates_on_the_frontal_views(self) -> None:
        """An elbow angle is the RIGHT quantity from every view; obliquity makes it noisier, not
        different in kind. Contrast the two sagittal rules, which gate."""
        core = _core(_rom_rep(setup_angle=130.0))
        side = rule_incomplete_rom(core, _ctx(view_type="side"))[0]
        rear = rule_incomplete_rom(core, _ctx(view_type="rear"))[0]
        self.assertEqual(side.observability, "high")
        self.assertEqual(rear.observability, "medium")
        self.assertLess(rear.confidence, side.confidence)
        self.assertEqual(side.severity, rear.severity)


class BicepCurlDetectorRegistrationTest(unittest.TestCase):
    def test_registered_under_its_canonical_movement_name(self) -> None:
        self.assertIs(registry.get_detector("Bicep Curl"), BICEP_CURL_DETECTOR)

    def test_ships_unvalidated_because_no_labeled_curl_exists(self) -> None:
        """Fit3D has curls with 3D ground truth and rep boundaries but NO correct/incorrect
        label, and REHAB24-6 has no curl at all. Beta is the factual label."""
        self.assertFalse(BICEP_CURL_DETECTOR.validated)

    def test_ships_three_rules_with_the_wrist_rule_absent_not_silent(self) -> None:
        """The parent spec's fourth rule is WITHDRAWN (a citation failure: Parpa discusses
        forearm ROTATION and grip, never wrist flexion), so unlike pushup's scapular-winging
        stub it leaves no registered-but-silent function behind -- a silent stub is this
        codebase's way of saying "the citation holds, the sensor does not"."""
        self.assertEqual(len(BICEP_CURL_DETECTOR.rules), 3)
        self.assertEqual(
            [rule.__name__ for rule in BICEP_CURL_DETECTOR.rules],
            ["rule_elbow_drift_forward", "rule_trunk_swing_momentum", "rule_incomplete_rom"],
        )

    def test_fault_ids_are_prefixed_so_they_cannot_collide(self) -> None:
        """`row_incomplete_rom` and `bpa_incomplete_rom` already exist, and merge_by_fault, the
        analyses table and the frontend's byFault map all key on fault_id with no movement
        qualifier -- so the parent spec's bare `incomplete_rom` would be indistinguishable."""
        core = _core(_rom_rep(setup_angle=130.0))
        detections = rule_incomplete_rom(core, _ctx())
        self.assertTrue(all(d.fault_id.startswith("curl_") for d in detections))


_FRAMES_PER_SAMPLE = 3
# One curl, extended -> flexed -> extended. `avg_elbow_angle` peaks at its MINIMUM, and the rep
# starts extended, which is what `rep_polarity="min"` / `rep_start="extended"` encode.
_ANGLES = [170.0, 140.0, 100.0, 55.0, 45.0, 55.0, 100.0, 140.0, 170.0]


def _three_rep_clip() -> list[dict]:
    frames: list[dict] = []
    for _ in range(3):
        for angle in _ANGLES:
            for _ in range(_FRAMES_PER_SAMPLE):
                frames.append(curl_frame(elbow_angle_deg=angle, frame_index=len(frames)))
    return frames


class EndToEndSegmentationTest(unittest.TestCase):
    """Verifies the rep-signal interface-design inference end to end.

    `rep_signal="avg_elbow_angle"` / `rep_polarity="min"` / `rep_start="extended"` are an
    interface-design choice backed by measurement (L/R elbow correlation r=0.992-0.996 across 8
    Fit3D subjects); this is what actually checks that `segment_reps` agrees.

    At 30 fps each rep here is 27 frames = 0.9 s, comfortably above
    DEFAULT_MIN_REP_SECONDS (0.4 s) -- and well below the 1.92-3.68 s/rep measured on real
    Fit3D curls, so the fixture is conservative in the direction that matters.
    """

    def test_three_reps_are_segmented_and_phased(self) -> None:
        result = run_detector(
            BICEP_CURL_DETECTOR,
            _three_rep_clip(),
            fps=30.0,
            view_type="rear_oblique",
            view_confidence=0.8,
        )
        self.assertIsNone(result.fallback)
        self.assertEqual(len(result.reps), 3)
        phases = {frame.phase for frame in result.core}
        self.assertIn("setup", phases)
        self.assertIn("peak", phases)

    def test_a_clean_clip_raises_no_faults(self) -> None:
        """The fixture curls from 170 to 45 degrees with a still trunk and vertical upper arms,
        so all three rules should stay silent -- the check that none of them fires on ordinary
        form."""
        result = run_detector(
            BICEP_CURL_DETECTOR,
            _three_rep_clip(),
            fps=30.0,
            view_type="rear_oblique",
            view_confidence=0.8,
        )
        self.assertEqual(result.detections, [])


if __name__ == "__main__":
    unittest.main()
