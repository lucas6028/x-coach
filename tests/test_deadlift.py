import unittest

import numpy as np

from src.pose.movements.base import CoreFrame, RuleContext
from src.pose.movements.deadlift import (
    DEADLIFT_LOCKOUT_MILD_DEG,
    DEADLIFT_METRIC_KEYS,
    deadlift_assign_phases,
    deadlift_compute_raw,
    rule_incomplete_lockout,
    rule_lumbar_flexion,
    setup_baseline,
)


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

    def test_a_rep_that_never_locks_out_still_has_a_lockout_phase(self):
        """The fault IS failing to reach extension, so the phase must not vanish with it.

        The lockout threshold is a PERCENTILE of this rep's own hip-angle excursion, not an
        absolute angle, so a rep peaking at 150 degrees still yields a lockout phase for
        `rule_incomplete_lockout` to score. An absolute cutoff would silence the rule on
        exactly the reps it exists to catch.
        """
        raw = self._rep(self._pull(60, 60.0, 150.0))
        phases = deadlift_assign_phases(raw)
        self.assertGreaterEqual(phases.count("lockout"), 6)

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

    def test_peak_and_start_frame_pin_the_worst_frame_not_index_zero(self):
        """Regression tripwire, same shape as lumbar flexion's: `_frames` builds IDENTICAL
        frames, so every candidate ties for "worst" and `np.nanargmax` (`build_detection`,
        driven by `score_values`) returns index 0 regardless of a broken `score_values` sign,
        and `worst_hip`/`severity` are indistinguishable between `nanmin`/`nanmax` or
        `max`/`min`. This window varies `hip_angle_deg` frame-to-frame so those mutations each
        produce a wrong, checkable answer.

        Hip angles 158, 152, 143, 150, 155, 157 -- all below the 165 deg mild threshold (so the
        whole 6-frame run, exactly `min_frames`, stays flagged), with a genuine minimum (worst
        extension) at position 2. Knee angle is held at 178 (unflagged) throughout so the hip
        ramp alone drives the result.
        """
        hip_angles = [158.0, 152.0, 143.0, 150.0, 155.0, 157.0]
        window = _varying_frames(
            [{"hip_angle_deg": h, "knee_angle_deg": 178.0} for h in hip_angles]
        )
        out = rule_incomplete_lockout(window, _ctx())
        self.assertEqual(len(out), 1)
        detection = out[0]
        # The first flagged frame is frame_index 0 -- pins the slice arithmetic.
        self.assertEqual(detection.start_frame, 0)
        # The worst (smallest hip angle) frame is index 2.
        self.assertEqual(detection.peak_frame, 2)
        # A `nanmax` mutation for `worst_hip` would report 158.0 instead of 143.0 here.
        self.assertAlmostEqual(detection.evidence["min_hip_angle_deg"], 143.0, places=4)
        # Severity is the WORST frame's score (index 2: (165-143)/(165-140) = 0.88), not the
        # best one. A `severity = min(scores)` mutation would report 0.28 (index 0) instead.
        self.assertAlmostEqual(detection.severity, 0.88, places=4)
