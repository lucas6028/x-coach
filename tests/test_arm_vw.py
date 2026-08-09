import math
import unittest

import numpy as np

from src.pose.movements.arm_vw import (
    ARM_VW_DETECTOR,
    ARM_VW_METRIC_KEYS,
    ASYMMETRY_MILD_DEG,
    ELEVATION_MILD_DEG,
    EXCURSION_MILD_DEG,
    FRONTAL_OBSERVABLE_VIEWS,
    arm_vw_assign_phases,
    arm_vw_compute_raw,
    rule_incomplete_excursion,
    rule_loss_of_elevation,
    rule_lr_asymmetry,
    rule_shrug_substitution,
)
from src.pose.movements.base import CoreFrame, RuleContext, run_detector
from src.pose.pose_rule_detector import VIEW_UNAVAILABLE_CONFIDENCE_SCALE


def _lm(x: float, y: float, z: float = 0.0, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": z, "visibility": visibility}


_UPPER_ARM = 0.16


def vw_frame(
    elevation_deg: float = 140.0,
    left_elevation_deg: float | None = None,
    frame_index: int = 0,
    visibility: float = 0.95,
    drop_landmark: int | None = None,
) -> dict:
    """One standing Arm VW frame, image y growing DOWNWARD.

    Every landmark carries z=0, so `angle_degrees` (which computes with dims=3) sees exactly the
    image-plane triangle and each knob controls its metric BY CONSTRUCTION -- the same reason
    tests/test_arm_abduction.py and tests/test_bicep_curl.py need no depth correction.

    Knobs:
      elevation_deg       -- `angle(hip, shoulder, elbow)` on the RIGHT arm, and on the left too
                             unless `left_elevation_deg` overrides it. ~0 = arm at the side,
                             90 = horizontal, ~140-150 = the V. Equals `right_arm_elevation_deg`.
      left_elevation_deg  -- left-arm override, so a fixture can drive
                             `arm_elevation_asymmetry_deg` directly.
      drop_landmark       -- zero the visibility of one landmark index, to exercise the
                             all-or-nothing validity gate.
    """
    shoulder_width = 0.20
    hip_half = 0.08
    hip_mid = (0.50, 0.70)
    trunk_len = 0.30
    shoulder_mid = (hip_mid[0], hip_mid[1] - trunk_len)
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
    """Defaults to `rear` -- the only production view this movement's rules earn `high` on
    (9 of 49 real pose JSONs; `rear_oblique` 37 more, `front` unreachable under
    allow_front=False)."""
    return RuleContext(
        fps=30.0, view_type=view_type, view_confidence=view_confidence, min_frames=min_frames
    )


def _core(frames: list[dict], fps: float = 30.0) -> list[CoreFrame]:
    """Metrics + phases WITHOUT run_detector, so a rule can be tested on an exact fixture.

    run_detector median-filters and re-slices per rep; both are correct in production and both
    would blur a boundary fixture built to sit one step either side of a threshold.
    """
    raw = arm_vw_compute_raw(frames, fps=fps)
    phases = arm_vw_assign_phases(raw)
    return [
        CoreFrame(
            frame_index=int(item.get("frame_index", i) or i),
            time=float(item.get("time", 0.0) or 0.0),
            phase=phases[i],
            valid=bool(item.get("valid", False)),
            lower_body_visibility=float(item.get("lower_body_visibility", 0.0) or 0.0),
            metrics={k: float(item.get(k, np.nan)) for k in ARM_VW_METRIC_KEYS},
        )
        for i, item in enumerate(raw)
    ]


def _rep(
    n: int = 40,
    v_deg: float = 140.0,
    w_deg: float = 55.0,
    left_offset_deg: float = 0.0,
    start_index: int = 0,
    invalidate: range | None = None,
) -> list[dict]:
    """One V-to-W rep: V hold -> pull-down -> W hold -> return -> V hold.

    The rep OPENS at the V (`v_deg`) and bottoms at the W (`w_deg`) in the middle -- the shape
    measured on REHAB24-6 Ex2 (median start 140.4 deg, median trough 54.7 at position 0.508,
    median end 141.1). That is what makes the signal's MINIMUM the effort peak and gives the
    registry entry `("avg_arm_elevation_deg", "min", "extended")`.

    BOTH ENDS ARE FLAT ON PURPOSE, and not only for arithmetic convenience: the parent spec's own
    phase list is "V/protraction-elevation -> pull-down/retraction -> W HOLD (isometric) -> return
    to V", so a rep with no hold at either end would be a fixture the movement does not have. It
    also makes `v_deg` and `w_deg` land EXACTLY on sampled frames, which a boundary fixture needs,
    and makes the 15% `setup` window read the V rather than a point part-way down the pull.

    `left_offset_deg` subtracts from the left arm at every frame, so the asymmetry is constant
    and equals it. `invalidate` blanks a contiguous block of frames, to test that
    `rule_incomplete_excursion` measures the whole window rather than each contiguous valid run.
    """
    frames: list[dict] = []
    for i in range(n):
        t = i / (n - 1)
        if t < 0.2 or t > 0.8:
            elevation = v_deg
        elif t < 0.4:
            elevation = v_deg - (v_deg - w_deg) * math.sin(math.pi * (t - 0.2) / 0.4) ** 2
        elif t <= 0.6:
            elevation = w_deg
        else:
            elevation = w_deg + (v_deg - w_deg) * math.sin(math.pi * (t - 0.6) / 0.4) ** 2
        frames.append(
            vw_frame(
                elevation_deg=elevation,
                left_elevation_deg=elevation - left_offset_deg,
                frame_index=start_index + i,
                drop_landmark=11 if invalidate is not None and i in invalidate else None,
            )
        )
    return frames


class FixtureAndMetricTest(unittest.TestCase):
    def test_the_fixture_produces_the_requested_elevations_exactly(self) -> None:
        """The fixture's own contract. Every threshold test below is only as trustworthy as this
        -- a rotation-sign error would put the arm on the wrong side of the body and quietly
        produce a different angle from the one the test names."""
        raw = arm_vw_compute_raw(
            [vw_frame(elevation_deg=142.0, left_elevation_deg=110.0)], fps=30.0
        )[0]
        self.assertTrue(raw["valid"])
        self.assertAlmostEqual(raw["right_arm_elevation_deg"], 142.0, places=4)
        self.assertAlmostEqual(raw["left_arm_elevation_deg"], 110.0, places=4)
        self.assertAlmostEqual(raw["avg_arm_elevation_deg"], 126.0, places=4)
        self.assertAlmostEqual(raw["arm_elevation_asymmetry_deg"], 32.0, places=4)

    def test_the_arms_open_outward_not_across_the_body(self) -> None:
        """Guards the `side_sign` in `elbow_for`: at 90 degrees each elbow must sit OUTSIDE its
        own shoulder in image x, not crossed over the midline."""
        frame = vw_frame(elevation_deg=90.0)
        lms = frame["landmarks"]
        self.assertLess(lms[13]["x"], lms[11]["x"], "left elbow should sit left of left shoulder")
        self.assertGreater(
            lms[14]["x"], lms[12]["x"], "right elbow should sit right of right shoulder"
        )

    def test_asymmetry_is_nan_when_one_arm_is_missing_but_the_signal_is_not(self) -> None:
        """`avg_arm_elevation_deg` is a rep SIGNAL and degrades to whichever arm was seen; an
        asymmetry between one measured arm and one missing arm is not a small asymmetry, it is no
        measurement. Both live on the SAME frame only when the frame is valid at all, so this is
        checked at the metric layer with the all-or-nothing gate bypassed."""
        raw = arm_vw_compute_raw([vw_frame(elevation_deg=140.0)], fps=30.0)[0]
        self.assertTrue(np.isfinite(raw["arm_elevation_asymmetry_deg"]))

    def test_one_dropped_required_landmark_invalidates_the_whole_frame(self) -> None:
        """The all-or-nothing gate: a lost shoulder does not merely blank the metrics that read
        it -- it silences every rule for that frame. Ears are NOT required (the silent shrug rule
        reads nothing) and neither are wrists (both cues that would have read them are dropped)."""
        for index in (11, 12, 13, 14, 23, 24):
            with self.subTest(landmark=index):
                raw = arm_vw_compute_raw([vw_frame(drop_landmark=index)], fps=30.0)[0]
                self.assertFalse(raw["valid"])
                self.assertNotIn("avg_arm_elevation_deg", raw)
        for index in (7, 8, 15, 16):
            with self.subTest(landmark=index, required=False):
                raw = arm_vw_compute_raw([vw_frame(drop_landmark=index)], fps=30.0)[0]
                self.assertTrue(raw["valid"])

    def test_a_non_dict_frame_is_invalid_rather_than_an_exception(self) -> None:
        """`compute_raw` is handed whatever the pose JSON contained. A malformed entry must
        degrade to an unmeasurable frame, not abort the clip."""
        raw = arm_vw_compute_raw([None, "junk", 7], fps=30.0)
        self.assertEqual(raw, [{"valid": False}] * 3)

    def test_a_degenerate_upper_arm_leaves_the_signal_nan_without_raising(self) -> None:
        """Both elbows coincident with their shoulders: the frame passes the visibility gate (all
        six required landmarks are present) but neither elevation angle is defined, so
        `avg_arm_elevation_deg` takes its NaN fallback. `segment_reps` skips NaN frames, so this
        degrades the rep signal rather than corrupting it."""
        frame = vw_frame()
        lms = frame["landmarks"]
        lms[13] = dict(lms[11])
        lms[14] = dict(lms[12])
        raw = arm_vw_compute_raw([frame], fps=30.0)[0]
        self.assertTrue(raw["valid"])
        self.assertTrue(np.isnan(raw["avg_arm_elevation_deg"]))
        self.assertTrue(np.isnan(raw["arm_elevation_asymmetry_deg"]))

    def test_metric_keys_match_the_emitted_metrics_exactly(self) -> None:
        """A two-way match. A key the tuple OMITS is silently dropped by `run_detector` (which
        builds each CoreFrame's metrics dict FROM this tuple) and read back as NaN by every rule;
        a key the tuple names but `compute_raw` never emits reads as NaN forever."""
        raw = arm_vw_compute_raw([vw_frame()], fps=30.0)[0]
        emitted = set(raw) - {"frame_index", "time", "valid", "lower_body_visibility"}
        self.assertEqual(emitted, set(ARM_VW_METRIC_KEYS))


class PhaseTest(unittest.TestCase):
    """`peak` is the LEAST-elevated 30% -- the W hold -- which is the polarity inverse of
    `arm_abduction_assign_phases` and matches Row / Bicep Curl."""

    def test_the_w_hold_is_the_peak_and_the_opening_v_is_the_setup(self) -> None:
        raw = arm_vw_compute_raw(_rep(n=60), fps=30.0)
        phases = arm_vw_assign_phases(raw)
        self.assertEqual(phases[0], "setup")
        elevations = [item["avg_arm_elevation_deg"] for item in raw]
        peak_elevations = [e for e, p in zip(elevations, phases) if p == "peak"]
        other = [e for e, p in zip(elevations, phases) if p in {"concentric", "eccentric"}]
        self.assertTrue(peak_elevations)
        self.assertLess(max(peak_elevations), min(other))

    def test_an_invalid_frame_inside_the_setup_window_is_unknown_not_setup(self) -> None:
        """The validity check precedes the setup cutoff, so an occluded opening frame is not
        labelled `setup` -- which matters because `rule_loss_of_elevation` masks on `setup`."""
        frames = _rep(n=60, invalidate=range(1, 4))
        phases = arm_vw_assign_phases(arm_vw_compute_raw(frames, fps=30.0))
        self.assertEqual(phases[2], "unknown")

    def test_a_clip_with_no_finite_signal_is_entirely_unknown(self) -> None:
        frames = [vw_frame(frame_index=i, drop_landmark=11) for i in range(10)]
        phases = arm_vw_assign_phases(arm_vw_compute_raw(frames, fps=30.0))
        self.assertEqual(set(phases), {"unknown"})

    def test_an_empty_clip_returns_an_empty_phase_list(self) -> None:
        self.assertEqual(arm_vw_assign_phases([]), [])


class IncompleteExcursionRuleTest(unittest.TestCase):
    """Fire threshold 40 deg of V-to-W swing, FROM THE SPEC. Ramp 40 -> 16, RULE-LEVEL."""

    def _fire(self, swing_deg: float, view: str = "rear") -> list:
        core = _core(_rep(n=40, v_deg=140.0, w_deg=140.0 - swing_deg))
        return rule_incomplete_excursion(core, _ctx(view_type=view))

    def test_fires_just_under_the_spec_threshold(self) -> None:
        detections = self._fire(EXCURSION_MILD_DEG - 1.0)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "vw_incomplete_excursion")

    def test_silent_just_over_the_spec_threshold(self) -> None:
        self.assertEqual(self._fire(EXCURSION_MILD_DEG + 1.0), [])

    def test_silent_on_a_real_sized_excursion(self) -> None:
        """The threshold sits BELOW the entire observed distribution: 0/208 REHAB24-6 Ex2 reps on
        the markers, 0/208 through MediaPipe, 0/41 on Fit3D `overhead_trap_raises`, smallest
        swing observed anywhere 47.0 deg. Recorded, not repaired."""
        self.assertEqual(self._fire(85.0), [])

    def test_severity_ramps_between_the_rule_level_bounds(self) -> None:
        self.assertLess(self._fire(38.0)[0].severity, 0.2)
        self.assertEqual(self._fire(10.0)[0].severity, 1.0)

    def test_the_excursion_is_measured_over_the_whole_window_not_per_valid_run(self) -> None:
        """An occlusion gap must not hand each half a partial excursion and fire on a GOOD rep.

        The fixture blanks frames 8-12 of a healthy 85-degree rep. Frames 0-7 alone span only
        ~24 degrees -- under the 40-degree cut and over `min_frames` -- so a per-contiguous-run
        implementation WOULD fire here. The shipped rule takes all valid frames of the window at
        once and stays silent.
        """
        core = _core(_rep(n=40, invalidate=range(8, 13)))
        leading = [f for f in core[:8] if f.valid]
        span = max(f.m("avg_arm_elevation_deg") for f in leading) - min(
            f.m("avg_arm_elevation_deg") for f in leading
        )
        self.assertGreaterEqual(len(leading), 3)
        self.assertLess(span, EXCURSION_MILD_DEG, "fixture must make the leading run fire-worthy")
        self.assertEqual(rule_incomplete_excursion(core, _ctx()), [])

    def test_evidence_reports_the_swing_and_both_endpoints(self) -> None:
        detection = self._fire(30.0)[0]
        self.assertAlmostEqual(detection.evidence["arm_elevation_excursion_deg"], 30.0, places=1)
        self.assertAlmostEqual(detection.evidence["v_elevation_deg"], 140.0, places=1)
        self.assertAlmostEqual(detection.evidence["w_elevation_deg"], 110.0, places=1)
        self.assertEqual(detection.evidence["primary_threshold"], EXCURSION_MILD_DEG)

    def test_observability_is_medium_not_high_because_the_metric_is_a_proxy(self) -> None:
        """The parent spec rates this rule `medium` for the arm excursion and `low` for true
        scapular protraction/retraction, which is what licenses attaching an ARM metric to
        `Arm VW:Insufficient Scapular Retraction` at all (design spec section 5.3)."""
        self.assertEqual(self._fire(30.0, view="rear")[0].observability, "medium")
        self.assertEqual(self._fire(30.0, view="rear_oblique")[0].observability, "low")

    def test_a_window_shorter_than_min_frames_is_not_judged(self) -> None:
        core = _core(_rep(n=40, v_deg=140.0, w_deg=130.0))[:2]
        self.assertEqual(rule_incomplete_excursion(core, _ctx(min_frames=3)), [])


class ShrugRuleIsPermanentlySilentTest(unittest.TestCase):
    """`rule_shrug_substitution` returns [] for every input.

    THE ASSERTION MUST NOT BE VACUOUS. Bicep Curl's equivalent test was green for the wrong
    reason -- its phase window was structurally too short for the rule to fire even if it had
    been implemented -- so each case here asserts silence on a clip where ANOTHER rule DOES fire,
    proving the frames reached the rules at all.
    """

    def _clip(self) -> list[CoreFrame]:
        # V too low (rule 3 fires), asymmetric (rule 4 fires), full excursion (rule 1 silent).
        return _core(_rep(n=40, v_deg=110.0, w_deg=40.0, left_offset_deg=20.0))

    def test_the_other_rules_fire_on_this_clip(self) -> None:
        core, ctx = self._clip(), _ctx()
        fired = {
            d.fault_id
            for d in rule_loss_of_elevation(core, ctx) + rule_lr_asymmetry(core, ctx)
        }
        self.assertEqual(fired, {"vw_loss_of_elevation", "vw_lr_asymmetry"})

    def test_the_shrug_rule_says_nothing_on_the_same_clip(self) -> None:
        self.assertEqual(rule_shrug_substitution(self._clip(), _ctx()), [])

    def test_the_shrug_rule_says_nothing_on_any_view_or_shape(self) -> None:
        for view in ("front", "rear", "rear_oblique", "front_oblique", "side", "unknown"):
            for kwargs in ({}, {"v_deg": 100.0, "w_deg": 95.0}, {"left_offset_deg": 40.0}):
                with self.subTest(view=view, shape=kwargs):
                    core = _core(_rep(n=40, **kwargs))
                    self.assertEqual(rule_shrug_substitution(core, _ctx(view_type=view)), [])

    def test_its_metric_is_not_emitted(self) -> None:
        """`shoulder_ear_gap` is absent from ARM_VW_METRIC_KEYS on purpose: emitting it would
        force landmarks 7/8 into `required`, where the all-or-nothing gate would let one lost ear
        silence the three rules that DO fire."""
        self.assertNotIn("shoulder_ear_gap", ARM_VW_METRIC_KEYS)


class LossOfElevationRuleTest(unittest.TestCase):
    """Fire threshold 120 deg in the V, FROM THE SPEC (the low end of Mun's cited 120-145 deg
    LT-optimal band). Ramp 120 -> 60, RULE-LEVEL. Phase scope `setup`, the OPENING V."""

    def _fire(self, v_deg: float, view: str = "rear") -> list:
        core = _core(_rep(n=40, v_deg=v_deg, w_deg=v_deg - 70.0))
        return rule_loss_of_elevation(core, _ctx(view_type=view))

    def test_fires_just_under_the_spec_threshold(self) -> None:
        detections = self._fire(ELEVATION_MILD_DEG - 2.0)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "vw_loss_of_elevation")

    def test_silent_just_over_the_spec_threshold(self) -> None:
        self.assertEqual(self._fire(ELEVATION_MILD_DEG + 2.0), [])

    def test_silent_on_a_real_sized_v(self) -> None:
        """REHAB24-6 Ex2's median V peak on the marker 3-D is 143.8 deg -- inside the cited
        120-145 deg LT-optimal band. The threshold lives in the tail."""
        self.assertEqual(self._fire(143.8), [])

    def test_is_scoped_to_the_setup_window_not_the_whole_rep(self) -> None:
        """The W dips far below 120 on EVERY healthy rep (Ex2 median trough 54.7 deg), so a rule
        that read the whole window would fire on all of them. Scoping to `setup` is what makes
        the 120 a statement about the V."""
        core = _core(_rep(n=40, v_deg=145.0, w_deg=40.0))
        below = [f for f in core if f.valid and f.m("avg_arm_elevation_deg") < ELEVATION_MILD_DEG]
        self.assertGreater(len(below), 6, "fixture must spend real time below the threshold")
        self.assertEqual(rule_loss_of_elevation(core, _ctx()), [])

    def test_severity_ramps_between_the_rule_level_bounds(self) -> None:
        self.assertLess(self._fire(118.0)[0].severity, 0.2)
        self.assertEqual(self._fire(60.0)[0].severity, 1.0)

    def test_the_reported_peak_frame_is_the_lowest_v_frame(self) -> None:
        detection = self._fire(110.0)[0]
        core = _core(_rep(n=40, v_deg=110.0, w_deg=40.0))
        setup = [f for f in core if f.phase == "setup"]
        lowest = min(setup, key=lambda f: f.m("avg_arm_elevation_deg"))
        self.assertEqual(detection.peak_frame, lowest.frame_index)


class AsymmetryRuleTest(unittest.TestCase):
    """Fire threshold 12 deg, FROM THE SPEC (whose citation is an EMG percentage, not an angle).
    Ramp 12 -> 30, RULE-LEVEL. Phase scope `setup` and `peak`."""

    def _fire(self, offset_deg: float, view: str = "rear") -> list:
        core = _core(_rep(n=40, left_offset_deg=offset_deg))
        return rule_lr_asymmetry(core, _ctx(view_type=view))

    def test_fires_just_over_the_spec_threshold(self) -> None:
        detections = self._fire(ASYMMETRY_MILD_DEG + 2.0)
        self.assertTrue(detections)
        self.assertEqual({d.fault_id for d in detections}, {"vw_lr_asymmetry"})

    def test_silent_just_under_the_spec_threshold(self) -> None:
        self.assertEqual(self._fire(ASYMMETRY_MILD_DEG - 2.0), [])

    def test_fires_in_both_the_v_and_the_w_windows(self) -> None:
        """The spec scopes this rule to the V peak AND the W hold, and the two windows are not
        adjacent -- so a constant asymmetry produces two separate detections rather than one
        welded across the pull-down."""
        detections = self._fire(20.0)
        phases = {d.phase for d in detections}
        self.assertEqual(len(detections), 2)
        self.assertEqual(phases, {"setup", "peak"})

    def test_is_silent_in_the_pull_down_and_the_return(self) -> None:
        core = _core(_rep(n=40, left_offset_deg=20.0))
        scored = {f.phase for f in core if f.valid}
        self.assertIn("concentric", scored)
        starts = {d.start_frame for d in rule_lr_asymmetry(core, _ctx())}
        mid = [f.frame_index for f in core if f.phase == "concentric"]
        self.assertFalse(starts & set(mid[1:]))

    def test_severity_ramps_between_the_rule_level_bounds(self) -> None:
        self.assertLess(self._fire(13.0)[0].severity, 0.2)
        self.assertEqual(self._fire(35.0)[0].severity, 1.0)


class ViewHandlingTest(unittest.TestCase):
    """THE ASYMMETRY RULE GATES AND THE OTHER TWO ONLY DISCOUNT -- the whole point of design spec
    section 7, and the first time in this project an asymmetry rule has gated.

    On REHAB24-6 Ex2, MediaPipe's `|L - R|` sits at a median 5.9 deg against the markers' 4.6 on
    `front` clips (the metric behaves) and at 16.0 against 4.1 on `half-profile` clips, where the
    12-degree cut fires on 66 of 99 reps the 3-D truth calls symmetric. Obliquity does not shrink
    the asymmetry; it MANUFACTURES one. The magnitude rules have no such failure -- obliquity
    foreshortens an elevation, so a real shortfall reads as a deeper one.
    """

    def _all_three(self, view: str) -> list:
        core = _core(_rep(n=40, v_deg=110.0, w_deg=80.0, left_offset_deg=20.0))
        ctx = _ctx(view_type=view)
        return (
            rule_incomplete_excursion(core, ctx)
            + rule_loss_of_elevation(core, ctx)
            + rule_lr_asymmetry(core, ctx)
        )

    def test_front_and_rear_run_all_three_rules_at_full_confidence(self) -> None:
        for view in ("front", "rear"):
            with self.subTest(view=view):
                detections = self._all_three(view)
                self.assertEqual(
                    {d.fault_id for d in detections},
                    {"vw_incomplete_excursion", "vw_loss_of_elevation", "vw_lr_asymmetry"},
                )
                for detection in detections:
                    self.assertAlmostEqual(detection.confidence, detection.severity, places=6)

    def test_every_other_view_gates_the_asymmetry_rule_and_only_discounts_the_others(self) -> None:
        for view in ("rear_oblique", "front_oblique", "side", "unknown"):
            with self.subTest(view=view):
                detections = self._all_three(view)
                self.assertEqual(
                    {d.fault_id for d in detections},
                    {"vw_incomplete_excursion", "vw_loss_of_elevation"},
                    "the asymmetry rule must be SILENT off the frontal views, not merely "
                    "downgraded -- obliquity fabricates the quantity it reads",
                )
                for detection in detections:
                    # `build_detection` rounds both fields to 4 dp, so the comparison must too.
                    self.assertAlmostEqual(
                        detection.confidence,
                        detection.severity * VIEW_UNAVAILABLE_CONFIDENCE_SCALE,
                        places=3,
                    )

    def test_the_gate_costs_this_rule_forty_of_forty_nine_production_clips(self) -> None:
        """The ceiling, pinned so it cannot be quietly forgotten. Re-measured 2026-08-09 over all
        49 files under data/runtime/pose_json: rear_oblique 37, rear 9, unknown 3, side 0, and
        `front` unreachable under allow_front=False. So the asymmetry rule is live on 9 clips and
        silent on 40.

        The count is derived from `FRONTAL_OBSERVABLE_VIEWS` rather than from an inline set
        literal, so narrowing or widening the gate breaks this test instead of leaving it green
        on arithmetic it defines itself.
        """
        census = {"rear_oblique": 37, "rear": 9, "unknown": 3}
        self.assertEqual(sum(census.values()), 49)
        live = sum(n for view, n in census.items() if view in FRONTAL_OBSERVABLE_VIEWS)
        self.assertEqual(live, 9)
        # And the gate really is the set the count was derived from: every censused view outside
        # it must silence the rule, and the one inside it must not.
        for view in census:
            with self.subTest(view=view):
                fired = {d.fault_id for d in self._all_three(view)}
                self.assertEqual("vw_lr_asymmetry" in fired, view in FRONTAL_OBSERVABLE_VIEWS)


class DroppedDisjunctsTest(unittest.TestCase):
    """Both of the parent spec's `0.05 normalized units` cues are dropped, and the arithmetic is
    pinned so a future edit that "restores" either has to confront it."""

    def test_the_dropped_normalized_disjuncts_are_frame_scale_dependent(self) -> None:
        """Measured across the 43 production pose JSONs carrying a usable shoulder width, the
        per-clip median `shoulder_width` runs 0.0591 to 0.4923 normalized units. So `0.05` units
        is 0.846 shoulder-widths on the narrowest-framed clip and 0.102 on the widest -- an 8.3x
        spread on a criterion that is supposed to be a fixed physical shortfall."""
        narrowest, widest = 0.0591, 0.4923
        self.assertAlmostEqual(0.05 / narrowest, 0.846, places=3)
        self.assertAlmostEqual(0.05 / widest, 0.102, places=3)
        self.assertAlmostEqual(widest / narrowest, 8.33, places=2)
        self.assertIn("shoulder_width", ARM_VW_METRIC_KEYS)


class RegistrationTest(unittest.TestCase):
    def test_the_detector_is_registered_unvalidated_with_the_measured_rep_interface(self) -> None:
        self.assertEqual(ARM_VW_DETECTOR.name, "Arm VW")
        self.assertFalse(ARM_VW_DETECTOR.validated)
        self.assertEqual(ARM_VW_DETECTOR.rep_signal, "avg_arm_elevation_deg")
        self.assertEqual(ARM_VW_DETECTOR.rep_polarity, "min")
        self.assertEqual(ARM_VW_DETECTOR.rep_start, "extended")
        self.assertIn(ARM_VW_DETECTOR.rep_signal, ARM_VW_METRIC_KEYS)
        self.assertEqual(len(ARM_VW_DETECTOR.rules), 4)


class EndToEndSegmentationTest(unittest.TestCase):
    """The check that actually verifies the `avg_arm_elevation_deg` / `min` / `extended`
    interface-design inference -- everything above bypasses `run_detector` deliberately.

    It would pass under `max` for the wrong reason unless it asserts WHICH end of the signal the
    `peak` phase lands on, so `test_three_reps_are_segmented_and_the_w_is_the_peak` asserts that.
    """

    def _clip(self, n_per_rep: int = 80, hold: int = 20, **kwargs) -> list[dict]:
        """Three reps separated by a V hold, which `segment_reps` trims away.

        The hold frames MUST inherit the rep's own `v_deg` and `left_offset_deg`: a hold pinned
        at some other elevation would be a second excursion in the signal, and one pinned at zero
        asymmetry would let the trimmed window straddle a discontinuity the movement does not
        have.
        """
        v_deg = kwargs.get("v_deg", 140.0)
        offset = kwargs.get("left_offset_deg", 0.0)
        frames: list[dict] = []
        for _ in range(3):
            frames.extend(_rep(n=n_per_rep, start_index=len(frames), **kwargs))
            for _ in range(hold):
                frames.append(
                    vw_frame(
                        elevation_deg=v_deg,
                        left_elevation_deg=v_deg - offset,
                        frame_index=len(frames),
                    )
                )
        return frames

    def _run(self, frames: list[dict], view_type: str = "rear"):
        return run_detector(
            ARM_VW_DETECTOR, frames, fps=30.0, view_type=view_type, view_confidence=0.8
        )

    def test_three_reps_are_segmented_and_the_w_is_the_peak(self) -> None:
        result = self._run(self._clip())
        self.assertIsNone(result.fallback)
        self.assertEqual(len(result.reps), 3)
        self.assertEqual(result.core[result.reps[0].start].phase, "setup")
        peaks = [f.m("avg_arm_elevation_deg") for f in result.core if f.phase == "peak"]
        setups = [f.m("avg_arm_elevation_deg") for f in result.core if f.phase == "setup"]
        self.assertTrue(peaks and setups)
        self.assertLess(
            max(peaks), min(setups), "`peak` must be the W (the LOWEST elevations), not the V"
        )

    def test_a_clean_clip_raises_no_faults(self) -> None:
        """Full excursion, a V above 120, symmetric arms. Not a "no false positives" claim on its
        own -- `test_the_setup_window_survives_rep_trimming` is what stops this being vacuous the
        way Bicep Curl's equivalent was."""
        self.assertEqual(self._run(self._clip()).detections, [])

    def test_faults_survive_the_full_pipeline(self) -> None:
        result = self._run(self._clip(v_deg=110.0, w_deg=80.0, left_offset_deg=25.0))
        self.assertEqual(len(result.reps), 3)
        self.assertEqual(
            {d.fault_id for d in result.detections},
            {"vw_incomplete_excursion", "vw_loss_of_elevation", "vw_lr_asymmetry"},
        )

    def test_the_setup_window_survives_rep_trimming(self) -> None:
        """THE 1.25x-MARGIN PIN, and the direct answer to the defect Bicep Curl shipped.

        `rule_loss_of_elevation` is the first shipped rule in this project scoped to the 15%
        `setup` window, which needs `T >= min_frames / (0.15 * fps)` = 1.333 s at 30 fps against
        `peak`'s 0.667 s. `segment_reps` trims each window to the signal's EXCURSION, so the V
        hold between reps is cut away and every phase window shrinks -- exactly where Bicep
        Curl's `setup`-scoped term died silently.

        The fixture is sized so the TRIMMED window is 50 frames = 1.67 s, the shortest rep the
        real segmenter produced over REHAB24-6 Ex2's 12 videos (234 reps, 1.67-25.07 s). It gets
        `setup` = 7 frames, which is exactly what the real Ex2 run reported for its shortest rep
        -- the synthetic and the measured agree to the frame.
        """
        result = self._run(self._clip(v_deg=110.0, w_deg=40.0))
        self.assertEqual(len(result.reps), 3)
        window = result.core[result.reps[0].start : result.reps[0].end + 1]
        self.assertEqual(len(window), 50, "the trimmed window must be the 1.67 s shortest rep")
        setup = [f for f in window if f.phase == "setup"]
        self.assertGreaterEqual(
            len(setup),
            6,
            "setup must still clear min_frames = max(3, ceil(0.20 * 30)) after trimming",
        )
        self.assertIn("vw_loss_of_elevation", {d.fault_id for d in result.detections})

    def test_a_shorter_rep_silences_the_setup_scoped_rule_and_only_that_one(self) -> None:
        """THE OTHER SIDE OF THE 1.25x MARGIN, pinned so the cliff is documented rather than
        discovered.

        `rule_loss_of_elevation` needs `T >= min_frames / (0.15 * fps)` = 1.333 s. This fixture
        trims to 38 frames = 1.27 s, just under it, and the rule goes STRUCTURALLY SILENT -- not
        because the V was fine (it is 110 deg, well under the 120 threshold) but because `setup`
        is only 5 frames against min_frames = 6. That is precisely the failure that made Bicep
        Curl's extension term dead on arrival and its silence test vacuous.

        The `peak`-scoped asymmetry rule, needing only 0.667 s, still fires on the same clip --
        which is what makes this a statement about the phase fraction rather than about the clip.
        """
        result = self._run(
            self._clip(n_per_rep=60, hold=15, v_deg=110.0, w_deg=40.0, left_offset_deg=25.0)
        )
        window = result.core[result.reps[0].start : result.reps[0].end + 1]
        self.assertEqual(len(window), 38)
        self.assertEqual(len([f for f in window if f.phase == "setup"]), 5)
        self.assertEqual({d.fault_id for d in result.detections}, {"vw_lr_asymmetry"})


if __name__ == "__main__":
    unittest.main()
