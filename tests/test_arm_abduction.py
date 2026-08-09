import math
import unittest

import numpy as np

from src.pose.movements.arm_abduction import (
    ARM_ABDUCTION_DETECTOR,
    ARM_ABDUCTION_METRIC_KEYS,
    ASYMMETRY_MILD_DEG,
    TRUNK_LEAN_MILD_DEG,
    arm_abduction_assign_phases,
    arm_abduction_compute_raw,
    rule_contralateral_trunk_lean,
    rule_lr_asymmetry,
    rule_shoulder_shrug,
)
from src.pose.movements.base import CoreFrame, RuleContext, run_detector
from src.pose.pose_rule_detector import VIEW_UNAVAILABLE_CONFIDENCE_SCALE


def _lm(x: float, y: float, z: float = 0.0, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": z, "visibility": visibility}


_UPPER_ARM = 0.16


def abduction_frame(
    elevation_deg: float = 5.0,
    left_elevation_deg: float | None = None,
    trunk_lean_deg: float = 0.0,
    frame_index: int = 0,
    visibility: float = 0.95,
    drop_landmark: int | None = None,
) -> dict:
    """One standing arm-abduction frame, image y growing DOWNWARD.

    Every landmark carries z=0, so `angle_degrees` (which computes with dims=3) sees exactly the
    image-plane triangle and each knob controls its metric BY CONSTRUCTION -- the same reason
    tests/test_bicep_curl.py needs no depth correction, and unlike test_band_pull_apart.py whose
    fixture must encode facing in z.

    Knobs:
      elevation_deg       -- `angle(hip, shoulder, elbow)` on the RIGHT arm, and on the left too
                             unless `left_elevation_deg` overrides it. ~0 = arm at the side,
                             90 = horizontal. Equals `right_arm_elevation_deg`.
      left_elevation_deg  -- left-arm override, so a fixture can drive
                             `arm_elevation_asymmetry_deg` directly.
      trunk_lean_deg      -- UNSIGNED lateral lean of hip_mid -> shoulder_mid from vertical;
                             equals `lateral_trunk_lean_deg`. Positive leans toward +x, but the
                             metric is unsigned so the sign is invisible to every rule (see
                             `test_trunk_lean_is_unsigned`).
      drop_landmark       -- zero the visibility of one landmark index, to exercise the
                             all-or-nothing validity gate.
    """
    shoulder_width = 0.20
    hip_half = 0.08
    hip_mid = (0.50, 0.70)
    trunk_len = 0.30
    theta = math.radians(trunk_lean_deg)
    shoulder_mid = (
        hip_mid[0] + trunk_len * math.sin(theta),
        hip_mid[1] - trunk_len * math.cos(theta),
    )
    left_shoulder = (shoulder_mid[0] - shoulder_width / 2.0, shoulder_mid[1])
    right_shoulder = (shoulder_mid[0] + shoulder_width / 2.0, shoulder_mid[1])
    left_hip = (hip_mid[0] - hip_half, hip_mid[1])
    right_hip = (hip_mid[0] + hip_half, hip_mid[1])

    def elbow_for(shoulder, hip, elev_deg: float, side_sign: float):
        """Place the elbow so `angle_degrees(hip, shoulder, elbow)` equals `elev_deg` exactly.

        The shoulder->hip unit vector is rotated by the requested elevation, outward from the
        midline (`side_sign` picks the direction), and the elbow placed one upper-arm length
        along it. Exact by construction rather than approximately, so a boundary fixture can sit
        one hundredth of a degree either side of a threshold.
        """
        ux, uy = hip[0] - shoulder[0], hip[1] - shoulder[1]
        norm = math.hypot(ux, uy)
        ux, uy = ux / norm, uy / norm
        phi = side_sign * math.radians(elev_deg)
        rx = ux * math.cos(phi) - uy * math.sin(phi)
        ry = ux * math.sin(phi) + uy * math.cos(phi)
        return (shoulder[0] + _UPPER_ARM * rx, shoulder[1] + _UPPER_ARM * ry)

    right_elbow = elbow_for(right_shoulder, right_hip, elevation_deg, -1.0)
    left_elbow = elbow_for(
        left_shoulder,
        left_hip,
        elevation_deg if left_elevation_deg is None else left_elevation_deg,
        1.0,
    )

    landmarks = [_lm(0.5, 0.3, visibility=visibility) for _ in range(33)]
    landmarks[11] = _lm(*left_shoulder, visibility=visibility)
    landmarks[12] = _lm(*right_shoulder, visibility=visibility)
    landmarks[13] = _lm(*left_elbow, visibility=visibility)
    landmarks[14] = _lm(*right_elbow, visibility=visibility)
    landmarks[23] = _lm(*left_hip, visibility=visibility)
    landmarks[24] = _lm(*right_hip, visibility=visibility)
    if drop_landmark is not None:
        point = landmarks[drop_landmark]
        landmarks[drop_landmark] = _lm(point["x"], point["y"], visibility=0.0)
    return {"frame_index": frame_index, "landmarks": landmarks}


def _ctx(
    view_type: str = "rear", view_confidence: float = 0.8, min_frames: int = 3
) -> RuleContext:
    """Defaults to `rear` -- the view this movement's rules are rated `high` on AND which the
    production estimator really emits (9 of 49 real pose JSONs; `rear_oblique` 37 more). Every
    previous detector's tests default to a view its rules only partly earn."""
    return RuleContext(
        fps=30.0, view_type=view_type, view_confidence=view_confidence, min_frames=min_frames
    )


def _core(frames: list[dict], fps: float = 30.0) -> list[CoreFrame]:
    """Metrics + phases WITHOUT run_detector, so a rule can be tested on an exact fixture.

    run_detector median-filters and re-slices per rep; both are correct in production and both
    would blur a boundary fixture built to sit one step either side of a threshold.
    """
    raw = arm_abduction_compute_raw(frames, fps=fps)
    phases = arm_abduction_assign_phases(raw)
    return [
        CoreFrame(
            frame_index=int(item.get("frame_index", i) or i),
            time=float(item.get("time", 0.0) or 0.0),
            phase=phases[i],
            valid=bool(item.get("valid", False)),
            lower_body_visibility=float(item.get("lower_body_visibility", 0.0) or 0.0),
            metrics={k: float(item.get(k, np.nan)) for k in ARM_ABDUCTION_METRIC_KEYS},
        )
        for i, item in enumerate(raw)
    ]


def _rep(
    n: int = 40,
    base_deg: float = 10.0,
    peak_deg: float = 95.0,
    left_offset_deg: float = 0.0,
    lean_deg: float = 0.0,
    lean_where: str = "all",
    start_index: int = 0,
) -> list[dict]:
    """One smooth abduction rep: base -> peak -> base, following sin^2.

    `left_offset_deg` subtracts from the left arm at every frame (so the asymmetry is constant
    and equals it). `lean_where` restricts the trunk lean to `"rising"` (strictly before the
    apex, i.e. `concentric` once the top 30% is carved off) or `"apex"` (the middle third, which
    lands inside `peak`), so phase scoping can be tested rather than assumed.
    """
    frames: list[dict] = []
    for i in range(n):
        t = i / (n - 1)
        elevation = base_deg + (peak_deg - base_deg) * math.sin(math.pi * t) ** 2
        if lean_where == "all":
            lean = lean_deg
        elif lean_where == "rising":
            lean = lean_deg if 0.20 <= t < 0.38 else 0.0
        else:
            lean = lean_deg if 0.42 <= t <= 0.58 else 0.0
        frames.append(
            abduction_frame(
                elevation_deg=elevation,
                left_elevation_deg=elevation - left_offset_deg,
                trunk_lean_deg=lean,
                frame_index=start_index + i,
            )
        )
    return frames


class FixtureAndMetricTest(unittest.TestCase):
    def test_the_fixture_produces_the_requested_elevations_exactly(self) -> None:
        """The fixture's own contract. Every threshold test below is only as trustworthy as this
        -- a rotation-sign error would put the arm on the wrong side of the body and quietly
        produce a different angle from the one the test names."""
        raw = arm_abduction_compute_raw(
            [abduction_frame(elevation_deg=73.0, left_elevation_deg=41.0)], fps=30.0
        )[0]
        self.assertTrue(raw["valid"])
        self.assertAlmostEqual(raw["right_arm_elevation_deg"], 73.0, places=4)
        self.assertAlmostEqual(raw["left_arm_elevation_deg"], 41.0, places=4)
        self.assertAlmostEqual(raw["avg_arm_elevation_deg"], 57.0, places=4)
        self.assertAlmostEqual(raw["arm_elevation_asymmetry_deg"], 32.0, places=4)

    def test_the_arms_abduct_outward_not_across_the_body(self) -> None:
        """Guards the `side_sign` in `elbow_for`: at 90 degrees each elbow must sit OUTSIDE its
        own shoulder in image x, not crossed over the midline."""
        frame = abduction_frame(elevation_deg=90.0)
        lms = frame["landmarks"]
        self.assertLess(lms[13]["x"], lms[11]["x"], "left elbow should sit left of left shoulder")
        self.assertGreater(
            lms[14]["x"], lms[12]["x"], "right elbow should sit right of right shoulder"
        )

    def test_trunk_lean_is_unsigned(self) -> None:
        """`lateral_trunk_lean_deg` uses `line_angle_from_vertical`, which takes abs() of both
        components. Leaning either way reads the same magnitude -- which is the point: the parent
        spec's "away from the raising arm" qualifier is undefined on a bilateral raise, so the
        metric deliberately cannot express a direction (design spec section 5.1)."""
        for sign in (1.0, -1.0):
            raw = arm_abduction_compute_raw(
                [abduction_frame(trunk_lean_deg=sign * 14.0)], fps=30.0
            )[0]
            self.assertAlmostEqual(raw["lateral_trunk_lean_deg"], 14.0, places=4)

    def test_one_dropped_required_landmark_invalidates_the_whole_frame(self) -> None:
        """The all-or-nothing gate: no metric key survives, so EVERY rule goes silent for that
        frame, not just the one whose input vanished."""
        for index in (11, 12, 13, 14, 23, 24):
            with self.subTest(landmark=index):
                raw = arm_abduction_compute_raw(
                    [abduction_frame(elevation_deg=90.0, drop_landmark=index)], fps=30.0
                )[0]
                self.assertFalse(raw["valid"])
                for key in ARM_ABDUCTION_METRIC_KEYS:
                    self.assertNotIn(key, raw)

    def test_the_ears_are_not_required_so_a_lost_ear_silences_nothing(self) -> None:
        """The deliberate inverse of the test above, and the reason `shoulder_ear_gap` is not
        emitted at all: the shrug rule is permanently silent, so putting landmarks 7/8 in
        `required` would let one lost ear silence the two rules that DO fire, for a metric
        nothing reads. Design spec section 3.3."""
        for index in (7, 8):
            with self.subTest(landmark=index):
                raw = arm_abduction_compute_raw(
                    [abduction_frame(elevation_deg=90.0, drop_landmark=index)], fps=30.0
                )[0]
                self.assertTrue(raw["valid"])

    def test_asymmetry_is_nan_when_either_arm_is_unmeasurable(self) -> None:
        """An asymmetry between one measured arm and one unmeasured arm is not a small
        asymmetry, it is no measurement. Driven by a degenerate elbow (coincident with its
        shoulder), which `angle_degrees` cannot resolve, rather than by visibility -- a dropped
        landmark would invalidate the whole frame before this branch is reached."""
        frame = abduction_frame(elevation_deg=90.0)
        frame["landmarks"][14] = _lm(
            frame["landmarks"][12]["x"], frame["landmarks"][12]["y"]
        )
        raw = arm_abduction_compute_raw([frame], fps=30.0)[0]
        self.assertTrue(raw["valid"])
        self.assertFalse(np.isfinite(raw["right_arm_elevation_deg"]))
        self.assertFalse(np.isfinite(raw["arm_elevation_asymmetry_deg"]))
        self.assertTrue(
            np.isfinite(raw["avg_arm_elevation_deg"]),
            "the rep SIGNAL degrades to whichever arm was seen; only the asymmetry refuses",
        )

    def test_a_non_dict_frame_is_refused_rather_than_raising(self) -> None:
        """Pose JSON reaches `compute_raw` from several producers (server extraction, the
        browser MediaPipe path, the REHAB24-6 replay harness). A malformed entry must degrade to
        an invalid frame, not abort the clip."""
        raw = arm_abduction_compute_raw([None, "not a frame", abduction_frame()], fps=30.0)
        self.assertEqual([item["valid"] for item in raw], [False, False, True])

    def test_metric_keys_match_the_emitted_metrics_exactly(self) -> None:
        """Two-way. A key the tuple omits is silently dropped by `run_detector` (which builds
        each CoreFrame's metrics dict FROM this tuple) and read back as NaN by every rule; a key
        the tuple names but nothing emits is read back as NaN too."""
        raw = arm_abduction_compute_raw([abduction_frame(elevation_deg=60.0)], fps=30.0)[0]
        emitted = set(raw) - {"frame_index", "time", "valid", "lower_body_visibility"}
        self.assertEqual(emitted, set(ARM_ABDUCTION_METRIC_KEYS))


class PhaseAssignmentTest(unittest.TestCase):
    def test_peak_is_the_most_elevated_thirty_percent(self) -> None:
        """The polarity inverse of `row`/`bicep_curl`, which take the 30th percentile and BELOW.
        Getting this backwards would scope `rule_lr_asymmetry` to the bottom of the rep."""
        core = _core(_rep(n=40))
        peak = [f for f in core if f.phase == "peak"]
        rest = [f for f in core if f.phase in {"concentric", "eccentric"}]
        self.assertTrue(peak)
        self.assertGreater(
            min(f.m("avg_arm_elevation_deg") for f in peak),
            max(f.m("avg_arm_elevation_deg") for f in rest),
        )

    def test_concentric_precedes_the_apex_and_eccentric_follows_it(self) -> None:
        core = _core(_rep(n=40))
        apex = max(range(len(core)), key=lambda i: core[i].m("avg_arm_elevation_deg"))
        concentric = [i for i, f in enumerate(core) if f.phase == "concentric"]
        eccentric = [i for i, f in enumerate(core) if f.phase == "eccentric"]
        self.assertTrue(concentric and eccentric)
        self.assertLess(max(concentric), apex)
        self.assertGreater(min(eccentric), apex)

    def test_an_invalid_frame_in_the_opening_fifteen_percent_is_unknown_not_setup(self) -> None:
        """The validity check precedes the setup cutoff. An occluded opening frame labelled
        `setup` would be counted as a measurement of the arms-down position it never saw."""
        frames = _rep(n=40)
        frames[1] = abduction_frame(elevation_deg=10.0, frame_index=1, drop_landmark=23)
        core = _core(frames)
        self.assertEqual(core[1].phase, "unknown")
        self.assertEqual(core[0].phase, "setup")

    def test_degenerate_clips(self) -> None:
        self.assertEqual(arm_abduction_assign_phases([]), [])
        self.assertEqual(
            arm_abduction_assign_phases([{"valid": False}, {"valid": False}]),
            ["unknown", "unknown"],
        )


class ShoulderShrugSilenceTest(unittest.TestCase):
    """`rule_shoulder_shrug` is registered and PERMANENTLY SILENT.

    Bicep Curl's equivalent "asserts silence" test was green for the wrong reason -- the phase
    window it depended on was structurally too narrow to fire, so the assertion would have
    passed even if the rule had been live. Every test here therefore proves the frames actually
    reached the rules, by showing another rule firing on the SAME core and context.
    """

    def _shrugging_clip(self) -> list[dict]:
        """Arms raised hard AND asymmetrically, so a live shrug rule would have every reason to
        fire and `rule_lr_asymmetry` demonstrably does."""
        return _rep(n=40, peak_deg=140.0, left_offset_deg=25.0)

    def test_it_is_silent_where_another_rule_fires(self) -> None:
        core = _core(self._shrugging_clip())
        ctx = _ctx()
        self.assertNotEqual(
            rule_lr_asymmetry(core, ctx), [], "control: the frames did reach the rules"
        )
        self.assertEqual(rule_shoulder_shrug(core, ctx), [])

    def test_it_is_silent_on_every_reachable_view(self) -> None:
        core = _core(self._shrugging_clip())
        for view in ("front", "rear", "rear_oblique", "front_oblique", "side", "unknown"):
            with self.subTest(view=view):
                self.assertEqual(rule_shoulder_shrug(core, _ctx(view_type=view)), [])

    def test_it_is_registered_rather_than_absent(self) -> None:
        """Silent, not withdrawn -- the parent spec and the code stay in 1:1 correspondence so an
        auditor gets "yes, accounted for, and here is why it says nothing". Contrast the
        impingement-arc rule, which is ABSENT from the module entirely."""
        self.assertIn(rule_shoulder_shrug, ARM_ABDUCTION_DETECTOR.rules)
        self.assertEqual(len(ARM_ABDUCTION_DETECTOR.rules), 3)


class TrunkLeanRuleTest(unittest.TestCase):
    def _fire(self, lean_deg: float, **kwargs) -> list:
        core = _core(_rep(n=40, lean_deg=lean_deg, lean_where="rising"))
        return rule_contralateral_trunk_lean(core, _ctx(**kwargs))

    def test_fires_just_past_the_spec_threshold_and_is_silent_just_short(self) -> None:
        self.assertEqual(self._fire(TRUNK_LEAN_MILD_DEG - 0.5), [])
        detections = self._fire(TRUNK_LEAN_MILD_DEG + 0.5)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "arm_abd_contralateral_trunk_lean")
        self.assertAlmostEqual(
            detections[0].evidence["primary_value"], TRUNK_LEAN_MILD_DEG + 0.5, places=2
        )
        self.assertEqual(detections[0].evidence["primary_threshold"], TRUNK_LEAN_MILD_DEG)

    def test_the_phase_scope_is_concentric_only(self) -> None:
        """FROM THE SPEC ("during concentric"). A lean confined to the apex sits in `peak` and
        must not fire -- widening the scope is a rule-level change the spec's wording exempts."""
        core = _core(_rep(n=40, lean_deg=20.0, lean_where="apex"))
        leaning = [f for f in core if f.m("lateral_trunk_lean_deg") > TRUNK_LEAN_MILD_DEG]
        self.assertTrue(leaning, "fixture must actually lean")
        self.assertEqual({f.phase for f in leaning}, {"peak"})
        self.assertEqual(rule_contralateral_trunk_lean(core, _ctx()), [])

    def test_severity_ramps_between_the_rule_level_bounds(self) -> None:
        mild = self._fire(13.0)[0].severity
        severe = self._fire(35.0)[0].severity
        self.assertLess(mild, 0.2)
        self.assertEqual(severe, 1.0)

    def test_a_short_lean_below_min_frames_does_not_fire(self) -> None:
        core = _core(_rep(n=40, lean_deg=20.0, lean_where="rising"))
        self.assertEqual(rule_contralateral_trunk_lean(core, _ctx(min_frames=99)), [])


class AsymmetryRuleTest(unittest.TestCase):
    def _fire(self, offset_deg: float, **kwargs) -> list:
        core = _core(_rep(n=40, left_offset_deg=offset_deg))
        return rule_lr_asymmetry(core, _ctx(**kwargs))

    def test_fires_just_past_the_spec_threshold_and_is_silent_just_short(self) -> None:
        self.assertEqual(self._fire(ASYMMETRY_MILD_DEG - 0.5), [])
        detections = self._fire(ASYMMETRY_MILD_DEG + 0.5)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "arm_abd_lr_asymmetry")
        self.assertAlmostEqual(
            detections[0].evidence["primary_value"], ASYMMETRY_MILD_DEG + 0.5, places=2
        )

    def test_it_is_side_agnostic(self) -> None:
        """`|L - R|` is sign-invariant, which is what lets the rule read the same from `front`
        and `rear` with no facing determination."""
        left_lags = rule_lr_asymmetry(_core(_rep(n=40, left_offset_deg=20.0)), _ctx())
        right_lags = rule_lr_asymmetry(_core(_rep(n=40, left_offset_deg=-20.0)), _ctx())
        self.assertEqual(len(left_lags), 1)
        self.assertEqual(len(right_lags), 1)
        self.assertAlmostEqual(
            left_lags[0].evidence["primary_value"], right_lags[0].evidence["primary_value"], places=4
        )

    def test_the_phase_scope_is_the_peak_hold_only(self) -> None:
        """FROM THE SPEC ("at the top-hold"). An asymmetry that resolves before the top must not
        fire, so the mask is checked against a fixture where only the rising frames are
        asymmetric."""
        frames = []
        for i, frame in enumerate(_rep(n=40)):
            t = i / 39.0
            offset = 25.0 if t < 0.30 else 0.0
            elevation = 10.0 + 85.0 * math.sin(math.pi * t) ** 2
            frames.append(
                abduction_frame(
                    elevation_deg=elevation,
                    left_elevation_deg=elevation - offset,
                    frame_index=i,
                )
            )
        core = _core(frames)
        asymmetric = [f for f in core if f.m("arm_elevation_asymmetry_deg") > ASYMMETRY_MILD_DEG]
        self.assertTrue(asymmetric)
        self.assertNotIn("peak", {f.phase for f in asymmetric})
        self.assertEqual(rule_lr_asymmetry(core, _ctx()), [])

    def test_the_wrist_height_disjunct_is_frame_scale_dependent(self) -> None:
        """The parent spec's second asymmetry cue -- "peak wrist heights differ by > 0.05
        normalized units" -- is NOT implemented, and unlike Bicep Curl's dropped disjunct the
        reason is not redundancy. `0.05` in RAW normalized image units means different physical
        asymmetries at different framings, because normalized coordinates scale with how much of
        the frame the subject occupies.

        Pinned here so a future edit that "restores" the cue has to confront the arithmetic.
        Measured across the 43 production pose JSONs under data/runtime/pose_json carrying a
        usable shoulder width, the per-clip median `shoulder_width` runs 0.0591 to 0.4923
        normalized units. `shoulder_width` is emitted by the metric layer precisely so this stays
        checkable without re-deriving it. Design spec section 6.7.
        """
        narrowest, widest = 0.0591, 0.4923
        self.assertAlmostEqual(0.05 / narrowest, 0.846, places=3)
        self.assertAlmostEqual(0.05 / widest, 0.102, places=3)
        self.assertAlmostEqual(widest / narrowest, 8.33, places=2)
        self.assertIn("shoulder_width", ARM_ABDUCTION_METRIC_KEYS)

    def test_severity_ramps_between_the_rule_level_bounds(self) -> None:
        self.assertLess(self._fire(13.0)[0].severity, 0.2)
        self.assertEqual(self._fire(35.0)[0].severity, 1.0)


class ViewHandlingTest(unittest.TestCase):
    """NO VIEW SILENCES EITHER LIVE RULE -- the first detector since Lunge for which that is
    true, and the inverse of the gating tests every detector since carries. Both metrics are
    frontal-plane and unsigned, so a rear view reads the same plane as a front one and obliquity
    makes them noisier rather than different. Design spec section 6.6.
    """

    def _both(self, view: str) -> list:
        core = _core(_rep(n=40, left_offset_deg=20.0, lean_deg=20.0, lean_where="rising"))
        ctx = _ctx(view_type=view)
        return rule_contralateral_trunk_lean(core, ctx) + rule_lr_asymmetry(core, ctx)

    def test_no_view_silences_either_rule(self) -> None:
        for view in ("front", "rear", "rear_oblique", "front_oblique", "side", "unknown"):
            with self.subTest(view=view):
                self.assertEqual(len(self._both(view)), 2)

    def test_front_and_rear_earn_full_confidence(self) -> None:
        """`rear` is the load-bearing half: it occurs 9 times in the 49 real production pose
        JSONs, whereas `front` is unreachable under allow_front=False. Every previous detector's
        `high` rating went to `side`, which occurs zero times."""
        for view in ("front", "rear"):
            with self.subTest(view=view):
                for detection in self._both(view):
                    self.assertEqual(detection.observability, "high")
                    self.assertAlmostEqual(detection.confidence, detection.severity, places=6)

    def test_every_other_view_is_downgraded_not_gated(self) -> None:
        for view in ("rear_oblique", "front_oblique", "side", "unknown"):
            with self.subTest(view=view):
                for detection in self._both(view):
                    self.assertEqual(detection.observability, "medium")
                    # `build_detection` rounds both fields to 4 dp, so the comparison must too.
                    self.assertAlmostEqual(
                        detection.confidence,
                        detection.severity * VIEW_UNAVAILABLE_CONFIDENCE_SCALE,
                        places=3,
                    )


class EndToEndSegmentationTest(unittest.TestCase):
    """The check that actually verifies the `avg_arm_elevation_deg` / `max` / `extended`
    interface-design inference -- everything above bypasses `run_detector` deliberately.

    At 30 fps each rep here is 60 frames = 2.0 s, inside the measured real range (1.40-4.96 s on
    Fit3D `side_lateral_raise`, 2.77-10.53 s on REHAB24-6 Ex1) and 5x DEFAULT_MIN_REP_SECONDS.
    """

    def _clip(self, n_per_rep: int = 60, **kwargs) -> list[dict]:
        frames: list[dict] = []
        for _ in range(3):
            frames.extend(_rep(n=n_per_rep, start_index=len(frames), **kwargs))
        return frames

    def _run(self, frames: list[dict], view_type: str = "rear"):
        return run_detector(
            ARM_ABDUCTION_DETECTOR, frames, fps=30.0, view_type=view_type, view_confidence=0.8
        )

    def test_three_reps_are_segmented_and_phased(self) -> None:
        result = self._run(self._clip())
        self.assertIsNone(result.fallback)
        self.assertEqual(len(result.reps), 3)
        phases = {frame.phase for frame in result.core}
        self.assertIn("setup", phases)
        self.assertIn("peak", phases)
        self.assertIn("concentric", phases)

    def test_a_clean_clip_raises_no_faults(self) -> None:
        """Symmetric arms, upright trunk. Not a "no false positives" claim on its own -- see
        `test_the_peak_window_survives_rep_trimming`, which is what stops this assertion being
        vacuous the way Bicep Curl's equivalent was."""
        self.assertEqual(self._run(self._clip()).detections, [])

    def test_faults_survive_the_full_pipeline(self) -> None:
        result = self._run(self._clip(left_offset_deg=25.0, lean_deg=20.0, lean_where="rising"))
        self.assertEqual(len(result.reps), 3)
        self.assertEqual(
            {d.fault_id for d in result.detections},
            {"arm_abd_lr_asymmetry", "arm_abd_contralateral_trunk_lean"},
        )

    def test_the_peak_window_survives_rep_trimming(self) -> None:
        """THE NON-VACUITY CHECK, and the direct answer to the defect Bicep Curl shipped.

        `segment_reps` trims each window to the signal's EXCURSION, so a clip that holds the arms
        DOWN between reps has that hold cut away and every phase window shrinks. Bicep Curl's
        `setup`-scoped term died there silently. `peak` is 30% of the window rather than 15%, so
        the requirement is `T >= min_frames / (0.30 * fps)` = 0.667 s rather than 1.333 s -- and
        this asserts it holds AFTER trimming, on a fixture that includes exactly the
        between-reps hold that does the trimming.
        """
        frames: list[dict] = []
        for _ in range(3):
            frames.extend(_rep(n=45, left_offset_deg=25.0, start_index=len(frames)))
            for _ in range(15):  # the arms-down hold segment_reps will trim away
                frames.append(abduction_frame(elevation_deg=10.0, frame_index=len(frames)))
        result = self._run(frames)
        self.assertEqual(len(result.reps), 3)
        window = result.core[result.reps[0].start : result.reps[0].end + 1]
        peak = [f for f in window if f.phase == "peak"]
        self.assertGreaterEqual(
            len(peak), 6, "peak must still clear min_frames = max(3, ceil(0.20 * 30)) after trimming"
        )
        self.assertIn("arm_abd_lr_asymmetry", {d.fault_id for d in result.detections})


if __name__ == "__main__":
    unittest.main()
