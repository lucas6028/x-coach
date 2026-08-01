import math
import unittest
import warnings

import numpy as np

from src.pose.geometry import contiguous_true_segments
from src.pose.movements.base import CoreFrame, RuleContext, run_detector
from src.pose.movements.deadlift import (
    DEADLIFT_LOCKOUT_MILD_DEG,
    DEADLIFT_METRIC_KEYS,
    deadlift_assign_phases,
    deadlift_compute_raw,
    rule_hips_shoot_up,
    rule_incomplete_lockout,
    rule_lumbar_flexion,
    setup_baseline,
)
from src.pose.movements import registry
from src.pose.movements.deadlift import DEADLIFT_DETECTOR


def _ctx(view: str = "side", conf: float = 0.9, min_frames: int = 6) -> RuleContext:
    return RuleContext(fps=30.0, view_type=view, view_confidence=conf, min_frames=min_frames)


def _frames(metrics: dict, count: int = 12, phase: str = "lockout") -> list[CoreFrame]:
    """A window of `count` identical CoreFrames carrying `metrics`."""
    return [
        CoreFrame(
            frame_index=i,
            time=i / 30.0,
            phase=phase,
            valid=True,
            lower_body_visibility=0.9,
            metrics=dict(metrics),
        )
        for i in range(count)
    ]


def _rep_window(setup: dict, active: dict, setup_n: int = 8, active_n: int = 12):
    """A window whose opening frames are `setup` phase and whose remainder is `mid_pull`."""
    return (
        _frames(setup, count=setup_n, phase="setup")
        + _frames(active, count=active_n, phase="mid_pull")
    )


def _varying_window(
    setup: dict, active_metrics: list[dict], setup_n: int = 8
) -> list[CoreFrame]:
    """Like `_rep_window`, but the active block carries a DIFFERENT metrics dict per frame.

    `_frames`/`_rep_window` deliberately build blocks of IDENTICAL frames, which is fine for
    fire/silence assertions but cannot distinguish which frame within a segment a rule picks
    as its "worst" one -- every candidate ties, so `np.nanargmax` (`build_detection`) returns
    index 0 regardless of sign conventions or min/max mistakes. Tests that pin
    `start_frame`/`end_frame`/`peak_frame` need frame-to-frame variation, which is what this
    helper provides.
    """
    return _frames(setup, count=setup_n, phase="setup") + [
        CoreFrame(
            frame_index=setup_n + i,
            time=(setup_n + i) / 30.0,
            phase="mid_pull",
            valid=True,
            lower_body_visibility=0.9,
            metrics=dict(metrics),
        )
        for i, metrics in enumerate(active_metrics)
    ]


def _varying_frames(metrics_list: list[dict], phase: str = "lockout") -> list[CoreFrame]:
    """A window of `len(metrics_list)` frames, each carrying a DIFFERENT metrics dict.

    Same motivation as `_varying_window`, but without a `setup` prefix: `rule_incomplete_lockout`
    has no setup-baseline dependency, so this is the minimal per-frame-varying builder for it.
    """
    return [
        CoreFrame(
            frame_index=i,
            time=i / 30.0,
            phase=phase,
            valid=True,
            lower_body_visibility=0.9,
            metrics=dict(metrics),
        )
        for i, metrics in enumerate(metrics_list)
    ]


def _landmarks(overrides: dict[int, tuple[float, float]]) -> list[dict]:
    """33 fully-visible landmarks in a plausible standing pose, with overrides applied."""
    base = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.99} for _ in range(33)]
    defaults = {
        11: (0.50, 0.30), 12: (0.52, 0.30),   # shoulders
        23: (0.50, 0.55), 24: (0.52, 0.55),   # hips
        25: (0.50, 0.75), 26: (0.52, 0.75),   # knees
        27: (0.50, 0.95), 28: (0.52, 0.95),   # ankles
        29: (0.49, 0.96), 30: (0.51, 0.96),   # heels
        31: (0.55, 0.97), 32: (0.57, 0.97),   # foot index
    }
    defaults.update(overrides)
    for index, (x, y) in defaults.items():
        base[index] = {"x": x, "y": y, "z": 0.0, "visibility": 0.99}
    return base


def _frame(index: int, overrides: dict[int, tuple[float, float]] | None = None) -> dict:
    return {"frame_index": index, "landmarks": _landmarks(overrides or {})}


def _deadlift_reps(n_reps: int, frames_per_rep: int = 30, low_offset: float = 0.30) -> list[dict]:
    """`n_reps` deadlifts built from landmark frames: the shoulders swing forward of the hips
    at each rep boundary (a hip hinge -- LOW hip_angle_deg, the floor) and return directly
    above the hips at the midpoint (upright lockout -- HIGH hip_angle_deg), tracing one
    hip-EXTENSION excursion per rep that starts and ends flexed.

    Mirrors `squat_reps` in tests/test_run_detector_per_rep.py, but mirrored in polarity: squat
    starts extended and dips to a bottom, this starts flexed and rises to a peak. That
    difference is exactly why `DEADLIFT_DETECTOR` needs `rep_start="flexed"` while
    `rep_polarity` stays "min" -- see `DeadliftRunDetectorTests` below for why this is not just
    restating the config: `hip_angle_deg` is naturally LOW at the floor and HIGH at lockout, so
    the floor (the flexed boundary) is already the raw signal's minimum and needs no inversion.
    """
    frames: list[dict] = []
    for index in range(n_reps * frames_per_rep):
        theta = 2.0 * math.pi * (index % frames_per_rep) / frames_per_rep
        # Forward lean is maximal at theta=0 (rep boundary, floor) and zero at theta=pi (rep
        # midpoint, lockout) -- the cosine shape squat_reps' hip_y uses, phase-shifted.
        offset = low_offset * (0.5 + 0.5 * math.cos(theta))
        frames.append(_frame(index, {11: (0.50 + offset, 0.30), 12: (0.52 + offset, 0.30)}))
    return frames


# ---------------------------------------------------------------------------------------
# SEAM FIXTURES: real landmarks -> deadlift_compute_raw -> deadlift_assign_phases -> a rule.
# ---------------------------------------------------------------------------------------
# Every OTHER rule test in this file hand-builds `CoreFrame`s with hand-set `phase` strings,
# which cannot catch a disagreement between what `deadlift_assign_phases` actually labels and
# what a rule expects to be handed. That gap hid a real defect: `lockout` is the 75th PERCENTILE
# of the rep's own hip-angle excursion, so on a rep that locks out fully the band still reaches
# below 165 degrees, and the original per-frame `< 165` mask fired "incomplete lockout" on it.
# See `DeadliftSeamTests` below.
#
# `_pose_landmarks` places the trunk and thigh so that `deadlift_compute_raw` recovers EXACTLY
# the requested `hip_angle_deg` and `torso_pitch_deg` (verified to 0.01 deg), with the knee held
# collinear at 180 deg so the hip axis alone drives `rule_incomplete_lockout`. That independent
# control is the point: in a naive stick model trunk pitch is a function of hip angle, so a
# fixture could not vary one without the other and `rule_hips_shoot_up` could not be posed at
# all.
_THIGH_LEN, _SHANK_LEN, _TORSO_LEN = 0.20, 0.20, 0.25


def _pose_landmarks(
    hip_angle_deg: float,
    torso_pitch_deg: float,
    hip_y: float = 0.55,
    torso_len: float = _TORSO_LEN,
) -> list[dict]:
    hip = np.array([0.51, hip_y])
    # Trunk: `torso_pitch_deg` from vertical, leaning forward (+x), shoulders above the hips.
    phi = math.radians(torso_pitch_deg)
    trunk = np.array([math.sin(phi), -math.cos(phi)])
    shoulder = hip + torso_len * trunk
    # Thigh: the trunk direction rotated by `hip_angle_deg`, so the interior angle at the hip
    # between shoulder and knee is exactly that.
    th = math.radians(hip_angle_deg)
    cos_t, sin_t = math.cos(th), math.sin(th)
    thigh = np.array(
        [cos_t * trunk[0] - sin_t * trunk[1], sin_t * trunk[0] + cos_t * trunk[1]]
    )
    knee = hip + _THIGH_LEN * thigh
    ankle = knee + _SHANK_LEN * thigh  # collinear with the thigh -> knee angle 180
    points = {
        11: shoulder + np.array([-0.01, 0.0]), 12: shoulder + np.array([0.01, 0.0]),
        23: hip + np.array([-0.01, 0.0]), 24: hip + np.array([0.01, 0.0]),
        25: knee + np.array([-0.01, 0.0]), 26: knee + np.array([0.01, 0.0]),
        27: ankle + np.array([-0.01, 0.0]), 28: ankle + np.array([0.01, 0.0]),
        29: ankle + np.array([-0.02, 0.01]), 30: ankle + np.array([0.0, 0.01]),
        31: ankle + np.array([0.04, 0.02]), 32: ankle + np.array([0.06, 0.02]),
    }
    base = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.99} for _ in range(33)]
    for index, point in points.items():
        base[index] = {"x": float(point[0]), "y": float(point[1]), "z": 0.0, "visibility": 0.99}
    return base


def _hip_track(n: int, low: float, peak: float) -> list[float]:
    """One flexed-start rep's hip angle: floor -> peak -> floor, constant angular velocity.

    Constant velocity is deliberate, not incidental. It is the profile that spreads the rep's
    frames UNIFORMLY over the hip-angle range, which is what pushes the 75th-percentile lockout
    cutoff far below the peak -- the condition the 2026-08-01 false positive needed. A fixture
    that dwelt near the peak would hide the very thing these tests exist to pin.
    """
    half = n // 2
    return list(np.linspace(low, peak, half)) + list(np.linspace(peak, low, n - half))


def _hinge_pitch(hip_angle: float) -> float:
    """A well-executed hinge: the trunk uprights monotonically as the hip extends."""
    return float(np.interp(hip_angle, [60.0, 180.0], [60.0, 2.0]))


def _shoot_pitch(hip_angle: float) -> float:
    """Hips shoot up: the chest DROPS early -- pitch rises well past its setup value."""
    return float(np.interp(hip_angle, [60.0, 95.0, 180.0], [40.0, 78.0, 2.0]))


def _rep_frames(n: int, peak: float, pitch_fn, torso_fn=None) -> list[dict]:
    return [
        {
            "frame_index": i,
            "landmarks": _pose_landmarks(
                hip, pitch_fn(hip), torso_len=torso_fn(hip) if torso_fn else _TORSO_LEN
            ),
        }
        for i, hip in enumerate(_hip_track(n, 60.0, peak))
    ]


def _core_from(raw: list[dict], phases: list[str]) -> list[CoreFrame]:
    """Assemble `CoreFrame`s the way `run_detector` does, but without smoothing or segmentation.

    Deliberately NOT re-implementing rule dispatch -- this is only the glue that lets a test
    drive `compute_raw` -> `assign_phases` -> one rule directly, so a phase-label mismatch shows
    up as a rule-level failure.
    """
    return [
        CoreFrame(
            frame_index=int(item.get("frame_index", i) or i),
            time=float(item.get("time", 0.0) or 0.0),
            phase=phase,
            valid=bool(item.get("valid", False)),
            lower_body_visibility=float(item.get("lower_body_visibility", 0.0) or 0.0),
            metrics={key: float(item.get(key, np.nan)) for key in DEADLIFT_METRIC_KEYS},
        )
        for i, (item, phase) in enumerate(zip(raw, phases))
    ]


def _seam(frames: list[dict], rule, ctx: RuleContext | None = None):
    """compute_raw -> assign_phases -> rule. Returns (core, phases, detections)."""
    raw = deadlift_compute_raw(frames, 30.0)
    phases = deadlift_assign_phases(raw)
    core = _core_from(raw, phases)
    return core, phases, rule(core, ctx or _ctx())


class DeadliftRunDetectorTests(unittest.TestCase):
    """The one design point the parent task calls out: `rep_start="flexed"` with
    `rep_polarity="min"` is the combination that makes `segment_reps` treat the FLOOR (a low
    raw `hip_angle_deg`) as the rep boundary, rather than the mid-rep event. Every other test in
    this module only echoes the config (`DEADLIFT_DETECTOR.rep_start == "flexed"`); none of them
    exercises `run_detector`, so a config that LOOKED right but found zero reps (silently
    falling back to whole-clip analysis, which breaks the per-rep setup baseline
    `rule_hips_shoot_up`/`rule_lumbar_flexion` depend on) would have passed every other test in
    this file. This one drives the real path end to end and would catch that.
    """

    def test_three_reps_are_detected_through_the_real_segmentation_path(self):
        result = run_detector(DEADLIFT_DETECTOR, _deadlift_reps(3), 30.0, "side", 0.9)
        self.assertIsNone(result.fallback)
        self.assertEqual(len(result.reps), 3)
        self.assertEqual(len(result.analyzed), 3)

    def test_each_rep_opens_at_the_floor_not_at_lockout(self):
        """A flexed-start rep's opening frames ARE the bar-on-the-floor setup -- the property
        `setup_baseline` depends on. Checked on the METRIC VALUE, not the phase label:
        `deadlift_assign_phases` labels the first `max(1, frame_count * 0.10)` frames of ANY
        per-rep window "setup" POSITIONALLY (see its `setup_cutoff`), without ever inspecting
        `hip_angle_deg` -- so that label is IDENTICAL whether or not the rep boundary actually
        landed on the floor, and asserting on it proves nothing about `rep_start`/`rep_polarity`.
        Verified directly: swapping in `rep_start="extended"` still returns reps whose opening
        frame is labeled "setup" even though its `hip_angle_deg` is ~179 (lockout), not ~130
        (floor). Only the value discriminates, which is why this asserts the value: under the
        fixture's ~130/~180 range, 150.0 sits strictly between the two.

        Also guards against a vacuous pass: an empty `result.reps` would let the loop body below
        never run and the test pass having checked nothing, which is exactly the failure mode a
        broken `rep_start`/`rep_polarity` combination produces (silent whole-clip fallback).
        """
        result = run_detector(DEADLIFT_DETECTOR, _deadlift_reps(3), 30.0, "side", 0.9)
        self.assertTrue(result.reps, "segmentation found no reps -- nothing below would run")
        for rep in result.reps:
            with self.subTest(rep=rep.index):
                self.assertLess(result.core[rep.start].m("hip_angle_deg"), 150.0)


class DeadliftMetricTests(unittest.TestCase):
    def test_metric_keys_match_the_emitted_metrics(self):
        """A key the tuple omits is dropped by run_detector and read back as NaN."""
        raw = deadlift_compute_raw([_frame(0)], fps=30.0)
        emitted = set(raw[0]) - {"frame_index", "time", "valid", "lower_body_visibility"}
        self.assertEqual(emitted, set(DEADLIFT_METRIC_KEYS))

    def test_an_upright_lockout_reads_near_180_degrees(self):
        raw = deadlift_compute_raw([_frame(0)], fps=30.0)
        self.assertTrue(raw[0]["valid"])
        self.assertGreater(raw[0]["hip_angle_deg"], 170.0)
        self.assertGreater(raw[0]["knee_angle_deg"], 170.0)
        self.assertLess(raw[0]["torso_pitch_deg"], 10.0)

    def test_a_pitched_trunk_reads_a_large_torso_pitch(self):
        # Shoulders driven forward of the hips: trunk near horizontal.
        raw = deadlift_compute_raw(
            [_frame(0, {11: (0.75, 0.50), 12: (0.77, 0.50)})], fps=30.0
        )
        self.assertGreater(raw[0]["torso_pitch_deg"], 60.0)

    def test_one_missing_landmark_invalidates_the_whole_frame(self):
        landmarks = _landmarks({})
        landmarks[24] = {"x": 0.52, "y": 0.55, "z": 0.0, "visibility": 0.01}
        raw = deadlift_compute_raw([{"frame_index": 0, "landmarks": landmarks}], fps=30.0)
        self.assertFalse(raw[0]["valid"])
        self.assertNotIn("hip_angle_deg", raw[0])

    def test_a_non_dict_frame_is_invalid_rather_than_raising(self):
        self.assertEqual(deadlift_compute_raw([None], fps=30.0), [{"valid": False}])


class DeadliftPhaseTests(unittest.TestCase):
    @staticmethod
    def _rep(hip_angles: list[float]) -> list[dict]:
        return [
            {"frame_index": i, "valid": True, "hip_angle_deg": a}
            for i, a in enumerate(hip_angles)
        ]

    @staticmethod
    def _pull(n: int, low: float, high: float) -> list[float]:
        """A flexed-start rep: floor -> lockout -> floor."""
        up = list(np.linspace(low, high, n // 2))
        return up + list(np.linspace(high, low, n - n // 2))

    def test_an_empty_clip_returns_no_phases(self):
        self.assertEqual(deadlift_assign_phases([]), [])

    def test_a_clip_with_no_finite_signal_is_entirely_unknown(self):
        raw = self._rep([np.nan] * 20)
        self.assertEqual(set(deadlift_assign_phases(raw)), {"unknown"})

    def test_an_invalid_frame_is_unknown_even_inside_the_setup_prefix(self):
        raw = self._rep(self._pull(40, 60.0, 178.0))
        raw[0]["valid"] = False
        self.assertEqual(deadlift_assign_phases(raw)[0], "unknown")

    def test_a_full_rep_produces_all_five_phases(self):
        raw = self._rep(self._pull(60, 60.0, 178.0))
        self.assertEqual(
            set(deadlift_assign_phases(raw)),
            {"setup", "lift_off", "mid_pull", "lockout", "lowering"},
        )

    def test_the_rep_opens_in_setup_because_a_flexed_start_begins_on_the_floor(self):
        raw = self._rep(self._pull(60, 60.0, 178.0))
        self.assertEqual(deadlift_assign_phases(raw)[0], "setup")

    def test_a_rep_that_never_locks_out_still_has_a_lockout_phase_AND_the_rule_scores_it(self):
        """The fault IS failing to reach extension, so the phase must not vanish with it.

        The lockout threshold is a PERCENTILE of this rep's own hip-angle excursion, not an
        absolute angle, so a rep peaking at 150 degrees still yields a lockout phase for
        `rule_incomplete_lockout` to score. An absolute cutoff would silence the rule on
        exactly the reps it exists to catch.

        A raw `phases.count("lockout")` was the original assertion and it was not enough on two
        counts, both of which this now covers. (1) `contiguous_true_segments(mask, min_frames)`
        needs the lockout frames CONTIGUOUS, not merely numerous -- 8 lockout frames split into
        two runs of 4 yield no segment at all -- so the count is asserted on the longest RUN.
        (2) The scoring half was never exercised: the docstring claimed the phase had to survive
        "for `rule_incomplete_lockout` to score" and then never checked that it does. Design
        spec section 7 promised this fixture would pin the rule firing; now it does.
        """
        raw = self._rep(self._pull(60, 60.0, 150.0))
        phases = deadlift_assign_phases(raw)
        mask = [p == "lockout" for p in phases]
        runs = contiguous_true_segments(mask, 6)
        self.assertEqual(len(runs), 1, "the lockout phase must be ONE contiguous run")
        start, end = runs[0]
        self.assertGreaterEqual(end - start + 1, 6)

        # ...and the rule actually scores it. `knee_angle_deg` is absent from this raw fixture,
        # so the hip ramp alone drives the verdict.
        core = _core_from(raw, phases)
        out = rule_incomplete_lockout(core, _ctx())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].fault_id, "deadlift_incomplete_lockout")
        self.assertAlmostEqual(out[0].evidence["peak_hip_angle_deg"], 150.0, places=1)

    def test_phase_count_always_equals_frame_count(self):
        """run_detector raises if assign_phases returns a different length."""
        for n in (1, 7, 40, 61):
            raw = self._rep(self._pull(n, 60.0, 178.0))
            self.assertEqual(len(deadlift_assign_phases(raw)), n)


class SetupBaselineTests(unittest.TestCase):
    def test_the_baseline_is_the_median_of_the_setup_frames_only(self):
        window = _rep_window({"torso_pitch_deg": 50.0}, {"torso_pitch_deg": 80.0})
        self.assertAlmostEqual(setup_baseline(window, "torso_pitch_deg"), 50.0, places=4)

    def test_a_window_with_no_setup_frames_has_no_baseline(self):
        window = _frames({"torso_pitch_deg": 60.0}, phase="mid_pull")
        self.assertTrue(np.isnan(setup_baseline(window, "torso_pitch_deg")))


class LumbarFlexionTests(unittest.TestCase):
    SETUP = {"torso_len": 0.25, "hip_y": 0.60}

    def test_a_shortening_torso_over_stationary_hips_fires_at_low_observability(self):
        window = _rep_window(self.SETUP, {"torso_len": 0.22, "hip_y": 0.60})
        out = rule_lumbar_flexion(window, _ctx())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].fault_id, "deadlift_lumbar_flexion")
        self.assertEqual(out[0].observability, "low")

    def test_a_rigid_torso_is_silent(self):
        window = _rep_window(self.SETUP, {"torso_len": 0.25, "hip_y": 0.50})
        self.assertEqual(rule_lumbar_flexion(window, _ctx()), [])

    def test_shortening_while_the_hips_travel_is_silent(self):
        """Hips moving means the shortening is the lift, not the spine."""
        window = _rep_window(self.SETUP, {"torso_len": 0.22, "hip_y": 0.40})
        self.assertEqual(rule_lumbar_flexion(window, _ctx()), [])

    def test_confidence_carries_the_low_observability_discount(self):
        out = rule_lumbar_flexion(
            _rep_window(self.SETUP, {"torso_len": 0.22, "hip_y": 0.60}), _ctx()
        )[0]
        # Discounted even in its own view: a proxy is never a measurement.
        self.assertAlmostEqual(out.confidence, out.severity * 0.65, places=4)

    def test_an_off_view_window_emits_nothing_at_all(self):
        """HARD GATE, unlike the other two rules: off-view, trunk pitch alone shortens the
        projected segment, so the proxy produces FALSE POSITIVES rather than silence."""
        window = _rep_window(self.SETUP, {"torso_len": 0.22, "hip_y": 0.60})
        for view in ("front", "rear", "rear_oblique", "unknown"):
            self.assertEqual(rule_lumbar_flexion(window, _ctx(view=view)), [], msg=view)

    def test_a_weakly_classified_side_view_emits_nothing(self):
        window = _rep_window(self.SETUP, {"torso_len": 0.22, "hip_y": 0.60})
        self.assertEqual(rule_lumbar_flexion(window, _ctx(view="side", conf=0.05)), [])

    def test_a_window_without_a_setup_baseline_is_silent(self):
        window = _frames({"torso_len": 0.22, "hip_y": 0.60}, phase="mid_pull")
        self.assertEqual(rule_lumbar_flexion(window, _ctx()), [])

    def test_a_degenerate_setup_torso_length_is_silent_rather_than_dividing_by_zero(self):
        window = _rep_window({"torso_len": 0.0, "hip_y": 0.60}, {"torso_len": 0.0, "hip_y": 0.60})
        self.assertEqual(rule_lumbar_flexion(window, _ctx()), [])

    def test_nan_metrics_are_silent(self):
        window = _rep_window(self.SETUP, {"torso_len": np.nan, "hip_y": np.nan})
        self.assertEqual(rule_lumbar_flexion(window, _ctx()), [])

    def test_severity_saturates_at_the_severe_endpoint(self):
        out = rule_lumbar_flexion(
            _rep_window(self.SETUP, {"torso_len": 0.20, "hip_y": 0.60}), _ctx()
        )[0]
        self.assertAlmostEqual(out.severity, 1.0, places=4)

    def test_peak_and_segment_bounds_pin_the_worst_frame_not_index_zero(self):
        """Regression tripwire: with IDENTICAL active frames every candidate ties for "worst",
        so `np.nanargmax` (`build_detection`) returns index 0 regardless of whether
        `score_values` is correctly negated or whether `worst` uses `nanmin` vs `nanmax`. This
        uses a window with frame-to-frame variation so a broken sign or a broken min/max both
        produce a wrong, checkable answer.

        Ratios (torso_len / setup 0.25): 0.92, 0.88, 0.80, 0.84, 0.88, 0.92 -- all below the
        0.95 mild threshold (so the whole 6-frame active block stays masked in, pinning
        start_frame at the setup-block offset), with a genuine minimum at position 2.
        """
        active = [
            {"torso_len": 0.230, "hip_y": 0.60},  # ratio 0.92
            {"torso_len": 0.220, "hip_y": 0.60},  # ratio 0.88
            {"torso_len": 0.200, "hip_y": 0.60},  # ratio 0.80 -- the worst frame
            {"torso_len": 0.210, "hip_y": 0.60},  # ratio 0.84
            {"torso_len": 0.220, "hip_y": 0.60},  # ratio 0.88
            {"torso_len": 0.230, "hip_y": 0.60},  # ratio 0.92
        ]
        window = _varying_window(self.SETUP, active)
        out = rule_lumbar_flexion(window, _ctx())
        self.assertEqual(len(out), 1)
        detection = out[0]
        # setup_n=8, so the active block starts at frame_index 8 and the segment must span
        # the full 6 frames (every ratio is below the mild threshold).
        self.assertEqual(detection.start_frame, 8)
        self.assertEqual(detection.end_frame, 13)
        # The worst (smallest-ratio) frame is active[2], at window frame_index 8 + 2 = 10.
        # A `score_values=ratios` mutation (dropping the negation) would instead pick the
        # first-occurring MAXIMUM ratio, frame_index 8.
        self.assertEqual(detection.peak_frame, 10)
        # A `nanmax` mutation for `worst` would report 0.92 instead of 0.80 here.
        self.assertAlmostEqual(detection.evidence["min_torso_length_ratio"], 0.80, places=4)


class IncompleteLockoutTests(unittest.TestCase):
    LOCKED = {"hip_angle_deg": 178.0, "knee_angle_deg": 176.0}

    def test_a_locked_out_rep_is_silent(self):
        self.assertEqual(rule_incomplete_lockout(_frames(self.LOCKED), _ctx()), [])

    def test_a_soft_hip_fires(self):
        out = rule_incomplete_lockout(
            _frames({"hip_angle_deg": 150.0, "knee_angle_deg": 176.0}), _ctx()
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].fault_id, "deadlift_incomplete_lockout")
        self.assertEqual(out[0].evidence["driver"], "hip")
        self.assertAlmostEqual(
            out[0].evidence["threshold"], DEADLIFT_LOCKOUT_MILD_DEG, places=4
        )

    def test_a_soft_knee_fires(self):
        out = rule_incomplete_lockout(
            _frames({"hip_angle_deg": 178.0, "knee_angle_deg": 150.0}), _ctx()
        )
        self.assertEqual(len(out), 1)

    def test_a_hip_only_failure_still_scores_the_hip_ramp(self):
        """The OHP mis-attribution regression: selecting the ramp by which reading is finite
        scored the wrong axis when a segment fired on one criterion alone."""
        soft = rule_incomplete_lockout(
            _frames({"hip_angle_deg": 145.0, "knee_angle_deg": 178.0}), _ctx()
        )[0]
        softer = rule_incomplete_lockout(
            _frames({"hip_angle_deg": 141.0, "knee_angle_deg": 178.0}), _ctx()
        )[0]
        self.assertGreater(softer.severity, soft.severity)

    def test_the_worse_of_the_two_axes_drives_severity(self):
        both = rule_incomplete_lockout(
            _frames({"hip_angle_deg": 160.0, "knee_angle_deg": 141.0}), _ctx()
        )[0]
        knee_only = rule_incomplete_lockout(
            _frames({"hip_angle_deg": 178.0, "knee_angle_deg": 141.0}), _ctx()
        )[0]
        self.assertAlmostEqual(both.severity, knee_only.severity, places=4)
        # Knee (141) is worse than hip (160) here, so the reported driver/primary_value must
        # name the knee, not the hip. Swapping the "hip"/"knee" branches in the rule's driver
        # ternary would flip this without changing `severity`, so the assertion above alone
        # cannot catch a mis-attribution -- this one can.
        self.assertEqual(both.evidence["driver"], "knee")
        self.assertAlmostEqual(both.evidence["primary_value"], 141.0, places=4)

    def test_severity_saturates_at_the_severe_endpoint(self):
        out = rule_incomplete_lockout(
            _frames({"hip_angle_deg": 100.0, "knee_angle_deg": 178.0}), _ctx()
        )[0]
        self.assertAlmostEqual(out.severity, 1.0, places=4)

    def test_only_the_lockout_phase_is_scored(self):
        soft = {"hip_angle_deg": 120.0, "knee_angle_deg": 120.0}
        for phase in ("setup", "lift_off", "mid_pull", "lowering", "rest"):
            self.assertEqual(
                rule_incomplete_lockout(_frames(soft, phase=phase), _ctx()), [],
                msg=f"{phase} must not be scored",
            )

    def test_nan_metrics_are_silent_rather_than_firing(self):
        out = rule_incomplete_lockout(
            _frames({"hip_angle_deg": np.nan, "knee_angle_deg": np.nan}), _ctx()
        )
        self.assertEqual(out, [])

    def test_a_hip_nan_segment_flagged_purely_by_the_knee_criterion_is_evidence_safe(self):
        """Reachable occluded-landmark case: `_flagged` ORs two independently-finite-checked
        clauses, so a segment can be flagged ENTIRELY by the knee reading while hip is NaN
        throughout every frame -- exactly the module header's "one occluded shoulder silently
        reverts... exactly in the view most likely to trigger it" scenario. An unguarded
        `np.nanmin` over an all-NaN hip slice both raises `RuntimeWarning: All-NaN slice
        encountered` and lets a bare `nan` reach `evidence`, which unsanitized `asdict()`
        serialization (`pose_rule_detector.py:664`) would carry to postgrest's
        `allow_nan=False` encoder -- a silently swallowed `ValueError` that drops the
        analysis from the user's history (see `json_safe_view_payload`'s docstring).

        The RuntimeWarning-to-error promotion is scoped to just this call (not the module or
        session) so the guard's absence fails THIS test under a plain `pytest tests/` run --
        without it, `evidence["peak_hip_angle_deg"] == 0.0` alone can never catch a removed
        guard, because the evidence-dict `if np.isfinite(...) else 0.0` fallback converts the
        NaN to 0.0 regardless of whether the upstream `nanmax` call was guarded.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = rule_incomplete_lockout(
                _frames({"hip_angle_deg": np.nan, "knee_angle_deg": 150.0}), _ctx()
            )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].evidence["driver"], "knee")
        self.assertEqual(out[0].evidence["peak_hip_angle_deg"], 0.0)

    def test_a_knee_nan_segment_flagged_purely_by_the_hip_criterion_is_evidence_safe(self):
        """Symmetric case: knee NaN throughout every frame, hip alone flags the segment. See
        the hip-NaN test above for why the RuntimeWarning-to-error promotion, scoped to this
        call only, is what makes the guard's absence actually fail this test."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = rule_incomplete_lockout(
                _frames({"hip_angle_deg": 150.0, "knee_angle_deg": np.nan}), _ctx()
            )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].evidence["driver"], "hip")
        self.assertEqual(out[0].evidence["peak_knee_angle_deg"], 0.0)

    def test_an_off_view_reading_is_discounted_but_not_suppressed(self):
        soft = {"hip_angle_deg": 150.0, "knee_angle_deg": 178.0}
        on = rule_incomplete_lockout(_frames(soft), _ctx(view="side"))[0]
        off = rule_incomplete_lockout(_frames(soft), _ctx(view="rear"))[0]
        self.assertEqual(on.observability, "high")
        self.assertEqual(off.observability, "medium")
        self.assertLess(off.confidence, on.confidence)
        # Pin the discount to the named constant, not just "some" reduction: full confidence
        # in-view, exactly the `_OFF_VIEW_CONFIDENCE` (0.65) scale off-view.
        self.assertAlmostEqual(on.confidence, on.severity, places=4)
        self.assertAlmostEqual(off.confidence, off.severity * 0.65, places=4)

    def test_a_run_shorter_than_min_frames_is_not_a_detection(self):
        soft = {"hip_angle_deg": 150.0, "knee_angle_deg": 178.0}
        self.assertEqual(rule_incomplete_lockout(_frames(soft, count=3), _ctx()), [])

    def test_the_reported_angle_is_the_reps_BEST_extension_not_its_worst_frame(self):
        """Regression tripwire, and the one that pins the 2026-08-01 peak-scoring rewrite.

        This rule aggregates with `nanmax` -- "how far did this rep actually extend" -- while
        every other rule in this module takes a per-frame worst. `_frames` builds IDENTICAL
        frames, so every candidate ties and `np.nanargmax` (`build_detection`, driven by
        `score_values`) returns index 0 whatever the aggregate is; this window varies
        `hip_angle_deg` frame-to-frame so a `nanmax` -> `nanmin` slip produces a wrong,
        checkable answer instead of an invisible one.

        Hip angles 143, 150, 158, 155, 152, 147 -- all below the 165 deg mild threshold, so the
        rep is genuinely a failed lockout and the whole 6-frame run (exactly `min_frames`) is
        one lockout segment. The MAXIMUM sits at position 2, deliberately NOT at index 0 or at
        the end, so `peak_frame` discriminates. Knee is held at 178 (unflagged) throughout so
        the hip ramp alone drives the result.
        """
        hip_angles = [143.0, 150.0, 158.0, 155.0, 152.0, 147.0]
        window = _varying_frames(
            [{"hip_angle_deg": h, "knee_angle_deg": 178.0} for h in hip_angles]
        )
        out = rule_incomplete_lockout(window, _ctx())
        self.assertEqual(len(out), 1)
        detection = out[0]
        # The lockout phase starts at frame_index 0 -- pins the slice arithmetic.
        self.assertEqual(detection.start_frame, 0)
        # `score_values` is the driver axis's raw angles, so `peak_frame` names the frame that
        # achieved the reported peak extension (index 2). A `score_values=[-v ...]` mutation
        # would pick the smallest angle (index 0) instead.
        self.assertEqual(detection.peak_frame, 2)
        # THE CORE ASSERTION: the rep reached 158, so that -- not the 143 of its worst single
        # frame -- is what "this rep failed to lock out" is scored on. A `nanmax` -> `nanmin`
        # mutation in `_peak_extension` reports 143.0 here.
        self.assertAlmostEqual(detection.evidence["peak_hip_angle_deg"], 158.0, places=4)
        self.assertAlmostEqual(detection.evidence["primary_value"], 158.0, places=4)
        # Severity follows the peak: (165-158)/(165-140) = 0.28. The same `nanmin` mutation
        # would report 0.88, inflating a marginal miss into a Moderate fault.
        self.assertAlmostEqual(detection.severity, 0.28, places=4)


    def test_the_driver_is_chosen_from_the_two_PEAKS_when_both_axes_vary(self):
        """Pins the `driver` tie-break under the 2026-08-01 peak aggregate.

        Every other driver assertion in this class uses constant frames, where per-frame and
        per-peak selection agree and so cannot tell them apart. Here BOTH axes vary and both
        fail, and the axis with the higher single-frame minimum is the one that must win: hip
        dips to 150 but PEAKS at 158 (severity 0.28), knee dips to 140 but peaks at 148
        (severity 0.68). Peak-scoring must name the KNEE. A per-frame aggregate would compare
        the minima (150 vs 140) and reach the same answer here by luck, so the value assertions
        below -- 158/148, not 150/140 -- are what actually discriminate.
        """
        hips = [150.0, 155.0, 158.0, 156.0, 152.0, 151.0]
        knees = [140.0, 144.0, 148.0, 146.0, 142.0, 141.0]
        window = _varying_frames(
            [{"hip_angle_deg": h, "knee_angle_deg": k} for h, k in zip(hips, knees)]
        )
        out = rule_incomplete_lockout(window, _ctx())
        self.assertEqual(len(out), 1)
        detection = out[0]
        self.assertEqual(detection.evidence["driver"], "knee")
        self.assertAlmostEqual(detection.evidence["peak_hip_angle_deg"], 158.0, places=4)
        self.assertAlmostEqual(detection.evidence["peak_knee_angle_deg"], 148.0, places=4)
        self.assertAlmostEqual(detection.evidence["primary_value"], 148.0, places=4)
        # severity == max(hip_sev, knee_sev) == (165-148)/25, so `driver` cannot disagree with
        # `severity` -- the invariant the parent spec's section 8 bullet states.
        self.assertAlmostEqual(detection.severity, 0.68, places=4)
        # `score_values` follows the DRIVER axis, so the peak frame is the knee's best frame.
        self.assertEqual(detection.peak_frame, 2)


class HipsShootUpTests(unittest.TestCase):
    def test_a_trunk_that_flattens_past_the_gate_fires(self):
        window = _rep_window({"torso_pitch_deg": 50.0}, {"torso_pitch_deg": 65.0})
        out = rule_hips_shoot_up(window, _ctx())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].fault_id, "deadlift_hips_shoot_up")

    def test_a_rep_that_stays_flat_without_flattening_further_is_silent(self):
        """Setting up flat is not the sequencing fault the citation describes."""
        window = _rep_window({"torso_pitch_deg": 70.0}, {"torso_pitch_deg": 68.0})
        self.assertEqual(rule_hips_shoot_up(window, _ctx()), [])

    def test_a_trunk_that_flattens_but_stays_upright_is_silent(self):
        window = _rep_window({"torso_pitch_deg": 20.0}, {"torso_pitch_deg": 40.0})
        self.assertEqual(rule_hips_shoot_up(window, _ctx()), [])

    def test_a_good_hinge_that_becomes_more_upright_is_silent(self):
        window = _rep_window({"torso_pitch_deg": 60.0}, {"torso_pitch_deg": 25.0})
        self.assertEqual(rule_hips_shoot_up(window, _ctx()), [])

    def test_severity_ramps_with_peak_pitch(self):
        mild = rule_hips_shoot_up(
            _rep_window({"torso_pitch_deg": 50.0}, {"torso_pitch_deg": 60.0}), _ctx()
        )[0]
        worse = rule_hips_shoot_up(
            _rep_window({"torso_pitch_deg": 50.0}, {"torso_pitch_deg": 72.0}), _ctx()
        )[0]
        self.assertGreater(worse.severity, mild.severity)

    def test_severity_saturates_at_the_severe_endpoint(self):
        out = rule_hips_shoot_up(
            _rep_window({"torso_pitch_deg": 50.0}, {"torso_pitch_deg": 85.0}), _ctx()
        )[0]
        self.assertAlmostEqual(out.severity, 1.0, places=4)

    def test_a_window_without_a_setup_baseline_is_silent(self):
        window = _frames({"torso_pitch_deg": 80.0}, phase="mid_pull")
        self.assertEqual(rule_hips_shoot_up(window, _ctx()), [])

    def test_nan_pitch_is_silent(self):
        window = _rep_window({"torso_pitch_deg": 50.0}, {"torso_pitch_deg": np.nan})
        self.assertEqual(rule_hips_shoot_up(window, _ctx()), [])

    def test_lowering_frames_are_never_scored(self):
        window = (
            _frames({"torso_pitch_deg": 50.0}, count=8, phase="setup")
            + _frames({"torso_pitch_deg": 80.0}, count=12, phase="lowering")
        )
        self.assertEqual(rule_hips_shoot_up(window, _ctx()), [])

    def test_an_off_view_reading_is_discounted_but_not_suppressed(self):
        window = _rep_window({"torso_pitch_deg": 50.0}, {"torso_pitch_deg": 65.0})
        on = rule_hips_shoot_up(window, _ctx(view="side"))[0]
        off = rule_hips_shoot_up(window, _ctx(view="rear"))[0]
        self.assertEqual(on.observability, "high")
        self.assertEqual(off.observability, "medium")
        self.assertLess(off.confidence, on.confidence)
        # Pin the discount to the named constant, not just "some" reduction: full confidence
        # in-view, exactly the `_OFF_VIEW_CONFIDENCE` (0.65) scale off-view.
        self.assertAlmostEqual(on.confidence, on.severity, places=4)
        self.assertAlmostEqual(off.confidence, off.severity * 0.65, places=4)

    def test_peak_and_start_frame_pin_the_worst_frame_not_index_zero(self):
        """Regression tripwire, same shape as the other two rules': `_rep_window` builds
        IDENTICAL active frames, so every candidate ties for "worst" and `np.nanargmax`
        (`build_detection`) returns index 0 regardless of a broken `np.nanmax` -> `np.nanmin`
        swap. This window varies `torso_pitch_deg` frame-to-frame so that mutation produces a
        wrong, checkable answer.

        Pitches 58, 62, 71, 66, 60, 59 -- all above both the setup baseline (50.0) and the
        55-degree gate (so the whole 6-frame active block, exactly `min_frames`, stays
        flagged), with a genuine maximum (worst flattening) at position 2.
        """
        pitches = [58.0, 62.0, 71.0, 66.0, 60.0, 59.0]
        window = _varying_window(
            {"torso_pitch_deg": 50.0},
            [{"torso_pitch_deg": p} for p in pitches],
        )
        out = rule_hips_shoot_up(window, _ctx())
        self.assertEqual(len(out), 1)
        detection = out[0]
        # setup_n=8 (the `_varying_window` default), so the active block starts at frame_index
        # 8; every pitch clears both clauses, so the segment spans the full 6-frame run.
        self.assertEqual(detection.start_frame, 8)
        # The worst (largest-pitch) frame is active[2], at window frame_index 8 + 2 = 10.
        # A `np.nanmax` -> `np.nanmin` mutation would instead pick the SMALLEST pitch (58.0,
        # index 0), reporting peak_frame 8 and evidence 58.0 here.
        self.assertEqual(detection.peak_frame, 10)
        self.assertAlmostEqual(detection.evidence["peak_torso_pitch_deg"], 71.0, places=4)


class DeadliftSeamTests(unittest.TestCase):
    """Real `deadlift_assign_phases` output driving each real rule's phase mask.

    Nothing else in this file exercises that seam: every other rule test hand-sets `phase`
    strings, so a rule and the phase function could disagree indefinitely without failing. They
    did. The 2026-08-01 defect these tests pin is recorded in `rule_incomplete_lockout`'s
    docstring; the two lockout tests below are its regression, in both directions.
    """

    def test_a_shallow_finish_fires_through_the_real_phase_labels(self):
        """A rep peaking at ~150 deg never locks out, and the rule must say so -- driven by the
        phases `deadlift_assign_phases` actually produced, not by hand-set labels."""
        for n in (60, 90):
            with self.subTest(frames=n):
                _, phases, out = _seam(_rep_frames(n, 150.0, _hinge_pitch), rule_incomplete_lockout)
                self.assertIn("lockout", phases)
                self.assertEqual(len(out), 1)
                self.assertEqual(out[0].fault_id, "deadlift_incomplete_lockout")
                self.assertEqual(out[0].evidence["driver"], "hip")
                self.assertAlmostEqual(out[0].evidence["peak_hip_angle_deg"], 150.0, places=1)
                # Ramp check: (165 - 150) / (165 - 140) = 0.60.
                self.assertAlmostEqual(out[0].severity, 0.60, places=2)

    def test_a_FULL_LOCKOUT_rep_stays_silent_even_though_the_phase_band_dips_below_165(self):
        """THE 2026-08-01 REGRESSION. This is the direction nobody checked, and it failed.

        `deadlift_assign_phases` sets `lockout` at the 75th PERCENTILE of the rep's own
        hip-angle excursion -- a RANK cutoff, not an angle. A rep that extends fully to 178 deg
        but spends under 25% of its frames above 165 therefore gets a `lockout` band reaching
        down to ~148 deg. The original rule masked frames individually on `< 165`, so those
        band-edge frames formed a contiguous run and it reported "incomplete lockout, minimum
        hip angle 148.5" at severity 0.66 on a rep that locked out perfectly.

        The durations are the ones that actually fired: 84, 90 and 120 frames (2.8 / 3.0 /
        4.0 s per rep at 30 fps). Below ~2.8 s/rep the sub-165 run fell short of `min_frames`
        and the bug hid, which is exactly why a single-duration fixture would not have caught it.

        This test asserts BOTH halves, and the first half is what stops it from silently
        decaying into a tautology: the lockout band genuinely still contains a sub-165 RUN long
        enough for the old bug to have fired (so the percentile behaviour that makes
        shallow-finish detection work is intact and NOT quietly replaced by an absolute cutoff),
        AND the rule is nevertheless silent, because it now scores the rep's PEAK extension
        instead of a run of frames.

        The guard is on the longest CONTIGUOUS run, not on a count of sub-165 frames. Those two
        differ by about a factor of two here, because the band dips below 165 on BOTH sides of
        the peak: measured counts are 12 / 14 / 16 against actual runs of [6, 6] / [7, 7] /
        [8, 8]. A count would still pass on a fixture that had drifted to runs of 3+3, which no
        longer poses the hazard at all -- and `contiguous_true_segments` is what the old bug
        actually depended on. Same correction as the sibling shallow-finish test.

        NOTE FOR ANYONE TIDYING THIS FIXTURE: at n=84 the longest run is exactly 6, equal to
        `min_frames` at 30 fps, with ZERO margin. That is deliberate -- it is the shortest rep
        that reproduced the original defect. Shortening the rep, damping the profile, or adding
        a pause at lockout drops the run below 6 and silently disarms this test.
        """
        min_frames = 6  # == max(3, ceil(30 fps * 0.20)), the floor `_ctx()` passes
        for n in (84, 90, 120):
            with self.subTest(frames=n):
                core, phases, out = _seam(
                    _rep_frames(n, 178.0, _hinge_pitch), rule_incomplete_lockout
                )
                band = [
                    frame.m("hip_angle_deg")
                    for frame, phase in zip(core, phases)
                    if phase == "lockout"
                ]
                self.assertGreater(max(band), 165.0, "the rep must genuinely reach lockout")
                # The percentile band still dips below the threshold, for long enough in one
                # unbroken stretch that the old per-frame rule WOULD have fired here...
                runs = contiguous_true_segments(
                    [value < DEADLIFT_LOCKOUT_MILD_DEG for value in band], 1
                )
                longest = max((end - start + 1 for start, end in runs), default=0)
                self.assertGreaterEqual(
                    longest, min_frames,
                    f"longest contiguous sub-165 lockout run is {longest} < min_frames "
                    f"{min_frames}: the old bug could not have fired on this fixture either, "
                    "so it no longer proves anything",
                )
                # ...and the rule is silent anyway, because it scores the PEAK.
                self.assertEqual(out, [], f"false 'incomplete lockout' on a 178 deg rep (n={n})")

    def test_a_full_lockout_clip_is_clean_end_to_end_through_run_detector(self):
        """The same regression on the FULL production path, which is where it was found.

        `_seam` skips `run_detector`'s smoothing and rep segmentation; this drives all of it.
        `fallback` must stay None -- on a fallback the clip is scored whole and the per-rep
        percentile band never forms, so the test would pass without proving anything.
        """
        for n in (84, 90, 120):
            with self.subTest(frames=n):
                frames: list[dict] = []
                for _ in range(3):
                    for hip in _hip_track(n, 60.0, 178.0):
                        frames.append(
                            {
                                "frame_index": len(frames),
                                "landmarks": _pose_landmarks(hip, _hinge_pitch(hip)),
                            }
                        )
                result = run_detector(DEADLIFT_DETECTOR, frames, 30.0, "side", 0.9)
                self.assertIsNone(result.fallback)
                self.assertEqual(len(result.analyzed), 3)
                self.assertEqual(
                    [d.fault_id for d in result.detections], [],
                    "a clean, fully locked-out deadlift must produce no faults at all",
                )

    def test_hips_shooting_up_fires_through_the_real_phase_labels(self):
        """Trunk pitch rising past its own setup value during the pull, via real phases.

        Paired with the good-hinge control below, this pins that the `setup` label
        `deadlift_assign_phases` emits actually lands on the floor portion of the rep -- if it
        did not, `setup_baseline` would return a mid-pull pitch and the relative clause would
        stop discriminating.
        """
        for n in (60, 90):
            with self.subTest(frames=n):
                _, _, out = _seam(_rep_frames(n, 175.0, _shoot_pitch), rule_hips_shoot_up)
                self.assertEqual(len(out), 1)
                self.assertEqual(out[0].fault_id, "deadlift_hips_shoot_up")
                evidence = out[0].evidence
                self.assertGreater(evidence["peak_torso_pitch_deg"], 70.0)
                # The baseline must be a FLOOR pitch (~40-52 deg here), not a mid-pull one.
                self.assertLess(evidence["setup_torso_pitch_deg"], 60.0)
                self.assertGreater(
                    evidence["peak_torso_pitch_deg"], evidence["setup_torso_pitch_deg"]
                )

    def test_a_good_hinge_is_silent_through_the_real_phase_labels(self):
        """Control for the test above: same seam, same durations, trunk uprighting normally."""
        for n in (60, 90):
            with self.subTest(frames=n):
                _, _, out = _seam(_rep_frames(n, 175.0, _hinge_pitch), rule_hips_shoot_up)
                self.assertEqual(out, [])

    def test_lumbar_flexion_fires_through_the_real_phase_labels(self):
        """Projected torso shortening over stationary hips, via real phases.

        `torso_len` is held at its full 0.25 while the hip angle is still near the floor (where
        `deadlift_assign_phases` puts `setup`) and shortened to 0.22 once the pull is under way,
        so the 0.88 ratio is measured against a genuine setup baseline.
        """
        def torso_fn(hip_angle: float) -> float:
            return float(np.interp(hip_angle, [60.0, 75.0, 90.0, 180.0], [0.25, 0.25, 0.22, 0.22]))

        for n in (60, 90):
            with self.subTest(frames=n):
                _, _, out = _seam(
                    _rep_frames(n, 175.0, _hinge_pitch, torso_fn=torso_fn), rule_lumbar_flexion
                )
                self.assertEqual(len(out), 1)
                self.assertEqual(out[0].fault_id, "deadlift_lumbar_flexion")
                self.assertEqual(out[0].observability, "low")
                self.assertAlmostEqual(out[0].evidence["min_torso_length_ratio"], 0.88, places=2)

    def test_a_rigid_torso_is_silent_through_the_real_phase_labels(self):
        """Control for the test above: identical seam with the torso length held constant."""
        for n in (60, 90):
            with self.subTest(frames=n):
                _, _, out = _seam(_rep_frames(n, 175.0, _hinge_pitch), rule_lumbar_flexion)
                self.assertEqual(out, [])


class DeadliftDetectorTests(unittest.TestCase):
    def test_it_is_registered_under_its_canonical_name(self):
        self.assertIs(registry.get_detector("Deadlift"), DEADLIFT_DETECTOR)

    def test_lookup_is_case_insensitive(self):
        self.assertIs(registry.get_detector("deadlift"), DEADLIFT_DETECTOR)

    def test_it_ships_unvalidated_because_no_labeled_deadlift_data_exists(self):
        self.assertFalse(DEADLIFT_DETECTOR.validated)

    def test_the_rep_signal_is_a_declared_metric_key(self):
        self.assertIn(DEADLIFT_DETECTOR.rep_signal, DEADLIFT_DETECTOR.metric_keys)

    def test_the_rep_starts_flexed_because_the_bar_starts_on_the_floor(self):
        self.assertEqual(DEADLIFT_DETECTOR.rep_start, "flexed")

    def test_all_three_surviving_rules_are_wired(self):
        names = {rule.__name__ for rule in DEADLIFT_DETECTOR.rules}
        self.assertEqual(
            names,
            {"rule_hips_shoot_up", "rule_incomplete_lockout", "rule_lumbar_flexion"},
        )

    def test_bar_drift_is_absent_because_it_was_withdrawn(self):
        self.assertNotIn(
            "rule_bar_drift", {rule.__name__ for rule in DEADLIFT_DETECTOR.rules}
        )
