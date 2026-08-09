import math
import unittest

import numpy as np

from src.pose.movements.base import CoreFrame, RuleContext, run_detector
from src.pose.movements.registry import get_detector, list_detectors
from src.pose.movements.shoulder_bridge import (
    EXTENSION_MILD_DEG,
    EXTENSION_SEVERE_DEG,
    SHOULDER_BRIDGE_DETECTOR,
    SHOULDER_BRIDGE_METRIC_KEYS,
    TOP_PHASE,
    rule_incomplete_hip_extension,
    rule_lumbar_hyperextension,
    shoulder_bridge_assign_phases,
    shoulder_bridge_compute_raw,
)


def _lm(x: float, y: float, z: float = 0.0, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": z, "visibility": visibility}


_TRUNK_LEN = 0.24
_THIGH_LEN = 0.16
_HALF_WIDTH = 0.03


def bridge_frame(
    hip_angle_deg: float = 150.0,
    right_hip_angle_deg: float | None = None,
    frame_index: int = 0,
    visibility: float = 0.95,
    drop_landmark: int | None = None,
    roll_deg: float = 0.0,
    arched: bool = False,
) -> dict:
    """One SUPINE shoulder-bridge frame, image y growing DOWNWARD.

    Every landmark carries z=0, so `angle_degrees` (which computes with dims=3) sees exactly the
    image-plane triangle and the knob controls its metric BY CONSTRUCTION -- the same reason
    tests/test_situp.py and tests/test_arm_vw.py need no depth correction.

    Geometry: the subject lies along the image x axis with the hips at the origin point and the
    thigh running toward the feet and up off the mat (hook lying). The shoulder is placed by
    rotating the hip->knee direction through the requested `angle(shoulder, hip, knee)`. Left and
    right are congruent triangles offset perpendicular to the body axis, so both sides read the
    same angle exactly unless `right_hip_angle_deg` overrides one.

    Knobs:
      hip_angle_deg        -- `angle(shoulder, hip, knee)`, both sides unless overridden. ~180 =
                              shoulders, hips and knees in the straight line BOTH cited sources
                              define as the endpoint; smaller = the pelvis is off that line.
      right_hip_angle_deg  -- right-side override, so a fixture can drive the two sides apart.
      drop_landmark        -- zero the visibility of one landmark index, to exercise the
                              all-or-nothing validity gate.
      roll_deg             -- rotate EVERY landmark about the image centre. The module reads only
                              joint-relative angles, so this must change nothing: it is the
                              executable form of the design spec's claim that the image horizontal
                              is not a usable floor reference. EgoExo-Fitness ships these frames
                              rolled, with no EXIF tag.
      arched               -- place the pelvis on the OPPOSITE side of the shoulder->knee line,
                              i.e. a bridge driven PAST the straight line into lumbar
                              hyperextension rather than one that never reached it. Physically the
                              opposite fault; `angle_degrees` cannot tell them apart, which is what
                              `MetricConflationTest` pins.
    """
    hip_mid = (0.40, 0.62)
    # Thigh direction: toward the feet (+x) and up off the mat (-y in image coords).
    thigh_dir = (math.cos(math.radians(-25.0)), math.sin(math.radians(-25.0)))

    def side(angle_deg: float, sign: float):
        hip = (hip_mid[0], hip_mid[1] + sign * _HALF_WIDTH)
        knee = (hip[0] + _THIGH_LEN * thigh_dir[0], hip[1] + _THIGH_LEN * thigh_dir[1])
        # Rotate the hip->knee unit vector by the requested angle to place the shoulder, so
        # `angle_degrees(shoulder, hip, knee)` equals `angle_deg` EXACTLY rather than
        # approximately -- a boundary fixture needs to sit one hundredth of a degree either side
        # of a threshold. Negating the rotation swings the trunk to the other side of the thigh,
        # which is the `arched` case.
        phi = math.radians(-angle_deg if arched else angle_deg)
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
        landmarks = [
            _lm(
                0.5 + (p["x"] - 0.5) * cos_t - (p["y"] - 0.5) * sin_t,
                0.5 + (p["x"] - 0.5) * sin_t + (p["y"] - 0.5) * cos_t,
                visibility=p["visibility"],
            )
            for p in landmarks
        ]

    if drop_landmark is not None:
        point = landmarks[drop_landmark]
        landmarks[drop_landmark] = _lm(point["x"], point["y"], visibility=0.0)
    return {"frame_index": frame_index, "landmarks": landmarks}


def _ctx(view_type: str = "rear_oblique", view_confidence: float = 0.8, min_frames: int = 3):
    """Defaults to `rear_oblique`, which the estimator returned on 3 of the 6 real clip-views.

    The value is deliberately irrelevant to every assertion below: this module's shipped rule
    reads no view (design spec section 1.2), and `ViewIndifferenceTest` pins that.
    """
    return RuleContext(
        fps=30.0, view_type=view_type, view_confidence=view_confidence, min_frames=min_frames
    )


def _core(frames: list[dict], fps: float = 30.0) -> list[CoreFrame]:
    """Metrics + phases WITHOUT run_detector, so a rule can be tested on an exact fixture.

    run_detector median-filters and re-slices per rep; both are correct in production and both
    would blur a boundary fixture built to sit one step either side of a threshold.
    """
    raw = shoulder_bridge_compute_raw(frames, fps=fps)
    phases = shoulder_bridge_assign_phases(raw)
    return [
        CoreFrame(
            frame_index=int(item.get("frame_index", i) or i),
            time=float(item.get("time", 0.0) or 0.0),
            phase=phases[i],
            valid=bool(item.get("valid", False)),
            lower_body_visibility=float(item.get("lower_body_visibility", 0.0) or 0.0),
            metrics={k: float(item.get(k, np.nan)) for k in SHOULDER_BRIDGE_METRIC_KEYS},
        )
        for i, item in enumerate(raw)
    ]


def _rep(
    n: int = 40,
    down_deg: float = 130.0,
    top_deg: float = 175.0,
    start_index: int = 0,
    invalidate: range | None = None,
    roll_deg: float = 0.0,
    arched: bool = False,
) -> list[dict]:
    """One bridge rep: supine hold -> lift -> top hold -> lower -> supine hold.

    The rep OPENS lying on the mat with the hips flexed (`down_deg`) and PEAKS at the top of the
    lift (`top_deg`), which is what makes the signal's MAXIMUM the effort peak and gives the
    registry entry `("hip_angle_deg", "max", "extended")` -- the inverse of Sit-up's polarity on
    the identical signal in the identical body position.

    Defaults are Escamilla's stated positions: ~50 degrees of hip flexion at the start (so a hip
    angle near 130) rising toward the straight line.

    Both ends are flat on purpose: the parent spec's phase list opens and closes at `setup
    (supine)` / `rest`, and flat ends make `down_deg` and `top_deg` land EXACTLY on sampled
    frames, which a boundary fixture needs.
    """
    frames: list[dict] = []
    for i in range(n):
        t = i / (n - 1)
        if t < 0.2 or t > 0.8:
            angle = down_deg
        elif t < 0.4:
            angle = down_deg + (top_deg - down_deg) * math.sin(math.pi * (t - 0.2) / 0.4) ** 2
        elif t <= 0.6:
            angle = top_deg
        else:
            angle = top_deg - (top_deg - down_deg) * math.sin(math.pi * (t - 0.6) / 0.4) ** 2
        frames.append(
            bridge_frame(
                hip_angle_deg=angle,
                frame_index=start_index + i,
                roll_deg=roll_deg,
                arched=arched,
                visibility=0.0 if invalidate is not None and i in invalidate else 0.95,
            )
        )
    return frames


class ComputeRawTest(unittest.TestCase):
    def test_metric_keys_match_the_emitted_metrics_exactly(self) -> None:
        """A two-way match, and both directions matter.

        `run_detector` builds each CoreFrame's metrics dict FROM `SHOULDER_BRIDGE_METRIC_KEYS`, so
        a key `compute_raw` emits but the tuple omits is silently DROPPED, and every rule reading
        it gets NaN. A key the tuple declares but `compute_raw` never emits is smoothed as an
        all-NaN column.
        """
        raw = shoulder_bridge_compute_raw([bridge_frame()], fps=30.0)
        emitted = set(raw[0]) - {"frame_index", "time", "valid", "lower_body_visibility"}
        self.assertEqual(emitted, set(SHOULDER_BRIDGE_METRIC_KEYS))

    def test_the_hip_angle_is_the_spec_quantity(self) -> None:
        raw = shoulder_bridge_compute_raw([bridge_frame(hip_angle_deg=163.0)], fps=30.0)
        self.assertAlmostEqual(raw[0]["hip_angle_deg"], 163.0, places=4)
        self.assertAlmostEqual(raw[0]["left_hip_angle_deg"], 163.0, places=4)
        self.assertAlmostEqual(raw[0]["right_hip_angle_deg"], 163.0, places=4)

    def test_the_two_sides_are_averaged(self) -> None:
        """Measured on the six real clip-views, the sides disagree by a median 5.9-11.1 degrees."""
        raw = shoulder_bridge_compute_raw(
            [bridge_frame(hip_angle_deg=150.0, right_hip_angle_deg=170.0)], fps=30.0
        )
        self.assertAlmostEqual(raw[0]["hip_angle_deg"], 160.0, places=4)

    def test_one_dropped_landmark_of_six_invalidates_the_frame(self) -> None:
        """Every rule masks on `frame.valid`, so this silences the whole module for that frame."""
        for index in (11, 12, 23, 24, 25, 26):
            with self.subTest(landmark=index):
                raw = shoulder_bridge_compute_raw(
                    [bridge_frame(drop_landmark=index)], fps=30.0
                )
                self.assertFalse(raw[0]["valid"])
                self.assertNotIn("hip_angle_deg", raw[0])

    def test_an_ankle_is_not_required(self) -> None:
        """The ankles feed only the two withdrawn rules and the sign construction that does not
        ship (design spec section 4), so losing one must not silence the rule that does."""
        raw = shoulder_bridge_compute_raw([bridge_frame(drop_landmark=27)], fps=30.0)
        self.assertTrue(raw[0]["valid"])

    def test_a_non_dict_frame_is_refused_rather_than_crashing(self) -> None:
        raw = shoulder_bridge_compute_raw(["not a frame", None], fps=30.0)
        self.assertEqual([item["valid"] for item in raw], [False, False])


class MetricConflationTest(unittest.TestCase):
    """`angle_degrees` is unsigned, and that is a defect the shipped rule carries.

    Design spec section 3. This is not a test of a bug to be fixed later -- it is the executable
    statement of a limitation the rule ships with, so a future reader meets it here rather than
    rediscovering it on a user's screen.
    """

    def test_the_metric_cannot_distinguish_a_sag_from_an_arch(self) -> None:
        for offset in (10.0, 20.0, 30.0):
            with self.subTest(offset=offset):
                sagging = shoulder_bridge_compute_raw(
                    [bridge_frame(hip_angle_deg=180.0 - offset)], fps=30.0
                )
                arched = shoulder_bridge_compute_raw(
                    [bridge_frame(hip_angle_deg=180.0 - offset, arched=True)], fps=30.0
                )
                self.assertAlmostEqual(
                    sagging[0]["hip_angle_deg"], arched[0]["hip_angle_deg"], places=4
                )

    def test_angle_degrees_never_exceeds_180(self) -> None:
        """Which is why `rule_lumbar_hyperextension`'s "> ~190 deg" test can never fire -- the
        fifth vacuous-branch defect in this registry, and the second caught before implementation.
        """
        for offset in (5.0, 20.0, 45.0):
            raw = shoulder_bridge_compute_raw(
                [bridge_frame(hip_angle_deg=180.0 - offset, arched=True)], fps=30.0
            )
            self.assertLessEqual(raw[0]["hip_angle_deg"], 180.0)

    def test_an_arched_bridge_is_reported_as_incomplete_extension(self) -> None:
        """The mislabel, stated end to end: the user is told to lift HIGHER when the fault is the
        opposite. Shipped because in the only labeled data the assumed direction is the one
        annotators fault (16/77 actions) and the other direction is not among the twelve criteria
        at all. Design spec section 5.4."""
        core = _core(_rep(top_deg=140.0, down_deg=130.0, arched=True))
        detections = rule_incomplete_hip_extension(core, _ctx())
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "bridge_incomplete_hip_extension")


class PhaseTest(unittest.TestCase):
    def test_top_is_the_most_extended_frames(self) -> None:
        """POLARITY IS THE INVERSE OF SIT-UP'S on the identical signal: a bridge's effort peak is
        the hip OPENING, so `top` is the 70th percentile and ABOVE."""
        frames = _rep()
        raw = shoulder_bridge_compute_raw(frames, fps=30.0)
        phases = shoulder_bridge_assign_phases(raw)
        top_values = [
            raw[i]["hip_angle_deg"] for i, p in enumerate(phases) if p == TOP_PHASE
        ]
        other = [
            raw[i]["hip_angle_deg"]
            for i, p in enumerate(phases)
            if p in {"concentric", "eccentric"}
        ]
        self.assertTrue(top_values)
        self.assertTrue(other)
        self.assertGreater(min(top_values), max(other))

    def test_an_empty_clip_returns_no_phases(self) -> None:
        self.assertEqual(shoulder_bridge_assign_phases([]), [])

    def test_a_clip_with_no_finite_signal_is_entirely_unknown(self) -> None:
        raw = shoulder_bridge_compute_raw([bridge_frame(drop_landmark=23) for _ in range(8)], fps=30.0)
        self.assertEqual(set(shoulder_bridge_assign_phases(raw)), {"unknown"})

    def test_an_invalid_frame_in_the_setup_window_is_unknown_not_setup(self) -> None:
        """The validity check precedes the setup cutoff, matching every other module."""
        frames = _rep(n=40)
        frames[1] = bridge_frame(frame_index=1, drop_landmark=25)
        raw = shoulder_bridge_compute_raw(frames, fps=30.0)
        self.assertEqual(shoulder_bridge_assign_phases(raw)[1], "unknown")


class IncompleteHipExtensionTest(unittest.TestCase):
    def test_a_bridge_reaching_the_straight_line_is_silent(self) -> None:
        core = _core(_rep(top_deg=178.0))
        self.assertEqual(rule_incomplete_hip_extension(core, _ctx()), [])

    def test_a_bridge_stopping_short_fires(self) -> None:
        core = _core(_rep(top_deg=150.0))
        detections = rule_incomplete_hip_extension(core, _ctx())
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.fault_id, "bridge_incomplete_hip_extension")
        self.assertAlmostEqual(detection.evidence["peak_hip_angle_deg"], 150.0, places=1)
        self.assertAlmostEqual(detection.evidence["residual_hip_flexion_deg"], 30.0, places=1)
        self.assertEqual(detection.evidence["primary_threshold"], EXTENSION_MILD_DEG)

    def test_the_threshold_is_exclusive(self) -> None:
        """A peak above the cited-endpoint rendering does NOT fire, matching every other
        `not-enough` rule in the registry.

        Tested half a degree either side rather than exactly ON the cut: the fixture places the
        shoulder by rotating a unit vector, so `angle_degrees` reconstructs the requested angle to
        within float rounding and a value sitting exactly on the threshold lands arbitrarily on
        either side of a strict `<`. What is being pinned is the rule's comparison, not numpy's
        last bit.
        """
        self.assertEqual(
            rule_incomplete_hip_extension(_core(_rep(top_deg=EXTENSION_MILD_DEG + 0.5)), _ctx()), []
        )
        fired = rule_incomplete_hip_extension(
            _core(_rep(top_deg=EXTENSION_MILD_DEG - 0.5)), _ctx()
        )
        self.assertEqual(len(fired), 1)
        self.assertGreater(fired[0].severity, 0.0)

    def test_severity_ramps_between_the_spec_cut_and_escamillas_start_position(self) -> None:
        mild = rule_incomplete_hip_extension(_core(_rep(top_deg=158.0)), _ctx())[0]
        severe = rule_incomplete_hip_extension(
            _core(_rep(top_deg=EXTENSION_SEVERE_DEG, down_deg=EXTENSION_SEVERE_DEG - 20.0)), _ctx()
        )[0]
        self.assertLess(mild.severity, 0.2)
        self.assertAlmostEqual(severe.severity, 1.0, places=4)
        self.assertEqual(mild.confidence, mild.severity)

    def test_a_window_shorter_than_min_frames_is_not_scored(self) -> None:
        core = _core(_rep(n=6, top_deg=140.0))
        self.assertEqual(rule_incomplete_hip_extension(core, _ctx(min_frames=99)), [])

    def test_the_payload_carries_no_nan(self) -> None:
        detection = rule_incomplete_hip_extension(_core(_rep(top_deg=145.0)), _ctx())[0]
        for key, value in detection.evidence.items():
            if isinstance(value, float):
                self.assertTrue(np.isfinite(value), msg=key)
        self.assertTrue(np.isfinite(detection.severity))
        self.assertTrue(np.isfinite(detection.confidence))
        self.assertIn("PMC11048684", detection.citation)
        self.assertIn("PMC11981018", detection.citation)


class AxialViewFalsePositiveTest(unittest.TestCase):
    """The measured defect, pinned so the next reader meets it. Design spec section 8.

    On the two real clips filmed by three SIMULTANEOUS cameras, the head-on axial `exo_m` view
    reads peak hip angles of 110.6, 112.3, 124.9 and 131.4, 139.3, 143.9 degrees for repetitions
    that human annotators marked CORRECT on this rule's exact criterion -- because viewed down the
    body's long axis the sagittal hip angle is foreshortened into meaninglessness (median 90
    degrees on exo_m against 128-134 on exo_r, same repetitions). The rule cannot tell those
    cameras apart, because the view estimator labels the axial views `rear_oblique` and one
    near-sagittal view `rear_oblique` too.

    The fixture reproduces the MEASURED READINGS rather than simulating the camera; what is pinned
    is the consequence.
    """

    def test_the_axial_view_fires_this_rule_at_near_full_severity_on_a_correct_rep(self) -> None:
        for measured_peak in (110.6, 112.3, 124.9):
            with self.subTest(peak=measured_peak):
                core = _core(_rep(top_deg=measured_peak, down_deg=measured_peak - 15.0))
                detections = rule_incomplete_hip_extension(core, _ctx())
                self.assertEqual(len(detections), 1)
                self.assertGreater(detections[0].severity, 0.9)

    def test_the_near_sagittal_view_fires_only_marginally_on_the_same_reps(self) -> None:
        """The other half of the census: 159.5 and 155.5 degrees are what the near-sagittal
        cameras read on the same footage, and they produce a low-severity card rather than a
        confident one."""
        for measured_peak in (159.5, 155.5):
            with self.subTest(peak=measured_peak):
                detections = rule_incomplete_hip_extension(
                    _core(_rep(top_deg=measured_peak)), _ctx()
                )
                self.assertEqual(len(detections), 1)
                self.assertLess(detections[0].severity, 0.2)


class MotionlessClipTest(unittest.TestCase):
    """This rule reads an ABSOLUTE POSITION, not a range, so it does not inherit the fail-open
    that `situp.rule_incomplete_rom` and `arm_vw.rule_incomplete_excursion` both carry.

    `segment_reps` thresholds on PERCENTILES and is therefore scale-free: a motionless subject's
    jitter segments into repetitions with a tiny excursion, which any excursion rule reads as a
    maximal fault. Judging position instead makes a still subject judged on where they actually
    are. Design spec section 5.6.
    """

    @staticmethod
    def _still(angle: float, n: int = 60) -> list[dict]:
        rng = np.random.default_rng(20260809)
        return [
            bridge_frame(hip_angle_deg=angle + float(rng.normal(0.0, 0.2)), frame_index=i)
            for i in range(n)
        ]

    def test_a_motionless_clip_is_judged_on_position_not_range(self) -> None:
        flat = run_detector(
            SHOULDER_BRIDGE_DETECTOR, self._still(130.0), 30.0, "rear_oblique", 0.8
        )
        held = run_detector(
            SHOULDER_BRIDGE_DETECTOR, self._still(175.0), 30.0, "rear_oblique", 0.8
        )
        self.assertTrue(
            [d for d in flat.detections if d.fault_id == "bridge_incomplete_hip_extension"],
            msg="a subject lying flat throughout never bridged and must be told so",
        )
        self.assertEqual(
            [d for d in held.detections if d.fault_id == "bridge_incomplete_hip_extension"],
            [],
            msg="a subject holding a good bridge must not be faulted for holding it still",
        )


class RollInvarianceTest(unittest.TestCase):
    """The module reads only joint-relative angles, so a rolled camera must change NOTHING.

    EgoExo-Fitness ships these frames rolled with no EXIF orientation tag (verified again on
    `z8RAua`: `PIL.getexif()` empty on all three exo views), which is why the parent spec's
    Group E convention of measuring "vs the floor/horizontal" is unusable. Unlike Sit-up, this
    module needed no re-anchoring to get here -- `angle(shoulder, hip, knee)` was already
    body-relative. This test stops that from being silently reverted.
    """

    def test_detections_are_identical_under_camera_roll(self) -> None:
        baseline = run_detector(
            SHOULDER_BRIDGE_DETECTOR, _rep(top_deg=148.0), 30.0, "rear_oblique", 0.8
        )
        for roll in (17.0, 90.0, 180.0, -90.0):
            with self.subTest(roll=roll):
                rolled = run_detector(
                    SHOULDER_BRIDGE_DETECTOR,
                    _rep(top_deg=148.0, roll_deg=roll),
                    30.0,
                    "rear_oblique",
                    0.8,
                )
                self.assertEqual(len(rolled.detections), len(baseline.detections))
                for got, want in zip(rolled.detections, baseline.detections):
                    self.assertEqual(got.fault_id, want.fault_id)
                    self.assertAlmostEqual(got.severity, want.severity, places=6)
                    self.assertAlmostEqual(
                        got.evidence["peak_hip_angle_deg"],
                        want.evidence["peak_hip_angle_deg"],
                        places=4,
                    )


class ViewIndifferenceTest(unittest.TestCase):
    """No view gate and no view discount -- the second rule in the registry with neither.

    `src/pose/view_estimation.py`'s docstring limit 1 forbids gating a horizontal-movement rule on
    these labels. Measured here on six real clip-views, the estimator returns `rear` three times
    and `rear_oblique` three times, never `side`, and the SAME camera disagrees with itself between
    two clips of the same person in the same room. Design spec section 1.2.
    """

    def test_the_verdict_is_identical_under_every_view_label(self) -> None:
        core = _core(_rep(top_deg=150.0))
        reference = rule_incomplete_hip_extension(core, _ctx(view_type="rear"))[0]
        for view_type in ("side", "rear", "rear_oblique", "front", "front_oblique", "unknown"):
            for confidence in (0.0, 0.5, 1.0):
                with self.subTest(view=view_type, confidence=confidence):
                    detection = rule_incomplete_hip_extension(
                        core, _ctx(view_type=view_type, view_confidence=confidence)
                    )[0]
                    self.assertEqual(detection.severity, reference.severity)
                    self.assertEqual(detection.confidence, reference.confidence)
                    self.assertEqual(detection.observability, reference.observability)


class PhaseScopeFloorTest(unittest.TestCase):
    """The Bicep Curl phase-fraction interaction binds here, because this module's shipped rule IS
    phase-scoped (Sit-up's is not).

    `min_frames = max(3, ceil(0.20 * fps))` and `top` covers 30% of a rep, so a rep needs
    `0.30 * T * fps >= min_frames`, i.e. `T >= 0.67 s` -- 1.67x the 0.4 s segmentation floor. A rep
    that segments but cannot be scored is a real gap, documented rather than closed by tuning
    `min_rep_seconds`. Design spec section 5.5.
    """

    def test_a_rep_below_the_phase_scope_floor_segments_but_is_not_scored(self) -> None:
        short = _rep(n=15, top_deg=140.0)  # 0.5 s at 30 fps, above 0.4 s and below 0.67 s
        core = _core(short)
        top_frames = [f for f in core if f.phase == TOP_PHASE and f.valid]
        self.assertLess(len(top_frames), 6)
        self.assertEqual(rule_incomplete_hip_extension(core, _ctx(min_frames=6)), [])


class SilentRuleTest(unittest.TestCase):
    """`rule_lumbar_hyperextension` is REGISTERED and PERMANENTLY SILENT.

    Real fault, cited for THIS exercise (Colonna, secondarily). It cannot ship because its test
    can never fire against an unsigned angle and because the sign it needs is not recoverable --
    two body-relative constructions were built and both were measured to fail on real footage.
    Design spec sections 4 and 6.
    """

    def test_it_is_silent_on_every_input_including_a_grossly_arched_bridge(self) -> None:
        for core in (
            _core(_rep(top_deg=175.0)),
            _core(_rep(top_deg=140.0)),
            _core(_rep(top_deg=150.0, arched=True)),
            _core(_rep(top_deg=120.0, arched=True)),
            [],
        ):
            self.assertEqual(rule_lumbar_hyperextension(core, _ctx()), [])

    def test_it_is_registered_so_the_silence_is_visible(self) -> None:
        self.assertIn(rule_lumbar_hyperextension, SHOULDER_BRIDGE_DETECTOR.rules)


class WithdrawnRulesTest(unittest.TestCase):
    """The two withdrawn rules are ABSENT, not silent, and nothing may quietly re-add them.

    A silent stub asserts "real fault, the sensor cannot see it"; an absent rule asserts "no
    citation supports this as written". `asymmetric_pelvic_drop`'s citation describes gait and
    `knee_valgus`'s describes landing and patellofemoral pain. Design spec section 7.
    """

    def test_no_pelvic_drop_or_valgus_rule_is_registered(self) -> None:
        fault_ids = set()
        for frames in (_rep(top_deg=140.0), _rep(top_deg=178.0)):
            result = run_detector(
                SHOULDER_BRIDGE_DETECTOR, frames, 30.0, "rear_oblique", 0.8
            )
            fault_ids.update(d.fault_id for d in result.detections)
        self.assertNotIn("bridge_asymmetric_pelvic_drop", fault_ids)
        self.assertNotIn("bridge_knee_valgus", fault_ids)
        self.assertEqual(len(SHOULDER_BRIDGE_DETECTOR.rules), 2)

    def test_no_ankle_or_frontal_width_metric_is_emitted(self) -> None:
        """Neither withdrawn rule's input is computed, so nothing can be wired up by accident."""
        for key in SHOULDER_BRIDGE_METRIC_KEYS:
            self.assertNotIn("width", key)
            self.assertNotIn("tilt", key)
            self.assertNotIn("ankle", key)


class RegistrationTest(unittest.TestCase):
    def test_the_detector_is_registered_under_its_movement_name(self) -> None:
        self.assertIs(get_detector("Shoulder Bridge"), SHOULDER_BRIDGE_DETECTOR)
        self.assertIs(get_detector("shoulder bridge"), SHOULDER_BRIDGE_DETECTOR)
        self.assertIn(SHOULDER_BRIDGE_DETECTOR, list_detectors())

    def test_the_segmentation_knobs_match_the_movement(self) -> None:
        """`max` polarity is the INVERSE of Sit-up's on the identical signal in the identical body
        position: a bridge's effort peak is the hip OPENING, a sit-up's is the hip CLOSING."""
        self.assertEqual(SHOULDER_BRIDGE_DETECTOR.rep_signal, "hip_angle_deg")
        self.assertEqual(SHOULDER_BRIDGE_DETECTOR.rep_polarity, "max")
        self.assertEqual(SHOULDER_BRIDGE_DETECTOR.rep_start, "extended")
        self.assertIn(SHOULDER_BRIDGE_DETECTOR.rep_signal, SHOULDER_BRIDGE_METRIC_KEYS)

    def test_it_is_not_marked_validated(self) -> None:
        """A FOURTH distinct reason in this registry: the labels exist and match the variant, and
        the pixels are missing. 77 human-judged EgoExo-Fitness Shoulder Bridge actions, of which
        exactly 2 fall in the part of the truncated `frames_open` archive that decodes. Design
        spec section 2.3."""
        self.assertFalse(SHOULDER_BRIDGE_DETECTOR.validated)


class EndToEndTest(unittest.TestCase):
    def test_a_three_rep_clip_merges_to_one_card(self) -> None:
        frames: list[dict] = []
        for rep_index in range(3):
            frames.extend(_rep(n=40, top_deg=148.0, start_index=rep_index * 40))
        result = run_detector(SHOULDER_BRIDGE_DETECTOR, frames, 30.0, "rear_oblique", 0.8)
        cards = [d for d in result.detections if d.fault_id == "bridge_incomplete_hip_extension"]
        self.assertEqual(len(cards), 1)
        self.assertGreaterEqual(cards[0].rep_count, 1)

    def test_a_good_clip_produces_no_cards(self) -> None:
        frames: list[dict] = []
        for rep_index in range(3):
            frames.extend(_rep(n=40, top_deg=178.0, start_index=rep_index * 40))
        result = run_detector(SHOULDER_BRIDGE_DETECTOR, frames, 30.0, "rear_oblique", 0.8)
        self.assertEqual(result.detections, [])


if __name__ == "__main__":
    unittest.main()
