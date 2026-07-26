import math
import unittest
import numpy as np
from src.pose.movements.overhead_press import ohp_compute_raw, ohp_assign_phases


def _elbow_xy(
    shoulder_xy: tuple[float, float],
    wrist_xy: tuple[float, float],
    elbow_angle: float,
    side: str,
) -> tuple[float, float]:
    """Compute an elbow landmark position such that the interior angle(shoulder, elbow,
    wrist) equals `elbow_angle` by construction. Places the elbow on the perpendicular
    bisector of the shoulder-wrist segment, offset laterally outward (left -> -x, right
    -> +x), using d = h / tan(angle/2) where h is half the shoulder-wrist distance."""
    sx, sy = shoulder_xy
    wx, wy = wrist_xy
    mx, my = (sx + wx) / 2.0, (sy + wy) / 2.0
    half_len = math.hypot(sx - wx, sy - wy) / 2.0
    if elbow_angle >= 179:
        d = 0.0
    else:
        d = half_len / math.tan(math.radians(elbow_angle) / 2.0)
    dx, dy = wx - sx, wy - sy
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        perp = (1.0, 0.0)
    else:
        ux, uy = dx / norm, dy / norm
        perp = (-uy, ux)
    desired_sign = -1.0 if side == "left" else 1.0
    if desired_sign * perp[0] < 0:
        perp = (-perp[0], -perp[1])
    return mx + d * perp[0], my + d * perp[1]


def ohp_frame(
    elbow_angle: float,
    wrist_y: float,
    shoulder_y: float = 0.4,
    frame_index: int = 0,
    shoulder_dx: float = 0.0,
    ear_dx: float = 0.0,
    nose_y: float | None = None,
) -> dict:
    lm = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(33)]
    # shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24, ears 7/8
    # shoulder_dx shifts both shoulders' x while hips stay put, simulating a
    # shoulders-behind-hips back-lean (positive torso_lean_signed_deg).
    left_shoulder = (0.45 + shoulder_dx, shoulder_y)
    right_shoulder = (0.55 + shoulder_dx, shoulder_y)
    left_wrist = (0.44, wrist_y)
    right_wrist = (0.56, wrist_y)
    left_elbow = _elbow_xy(left_shoulder, left_wrist, elbow_angle, "left")
    right_elbow = _elbow_xy(right_shoulder, right_wrist, elbow_angle, "right")
    lm[11] = {"x": left_shoulder[0], "y": left_shoulder[1], "z": 0, "visibility": 1.0}
    lm[12] = {"x": right_shoulder[0], "y": right_shoulder[1], "z": 0, "visibility": 1.0}
    lm[13] = {"x": left_elbow[0], "y": left_elbow[1], "z": 0, "visibility": 1.0}
    lm[14] = {"x": right_elbow[0], "y": right_elbow[1], "z": 0, "visibility": 1.0}
    lm[15] = {"x": left_wrist[0], "y": left_wrist[1], "z": 0, "visibility": 1.0}
    lm[16] = {"x": right_wrist[0], "y": right_wrist[1], "z": 0, "visibility": 1.0}
    lm[23] = {"x": 0.46, "y": 0.75, "z": 0, "visibility": 1.0}
    lm[24] = {"x": 0.54, "y": 0.75, "z": 0, "visibility": 1.0}
    # Head landmarks ride WITH the shoulders (a torso lean carries the head with it), so
    # `ear_dx` is a pure head-jut knob measured relative to the shoulder line. Sign follows
    # the module's single facing convention (anterior = -x, see ohp_compute_raw): a NEGATIVE
    # ear_dx means the head juts ANTERIOR of the shoulders, i.e. forward head.
    lm[7] = {"x": 0.46 + shoulder_dx + ear_dx, "y": shoulder_y - 0.08, "z": 0, "visibility": 1.0}
    lm[8] = {"x": 0.54 + shoulder_dx + ear_dx, "y": shoulder_y - 0.08, "z": 0, "visibility": 1.0}
    # Nose (landmark 0) must be placed explicitly: the all-33 default sits at y=0.5, i.e.
    # BELOW the shoulders (y=0.4), which is anatomically impossible and would make any
    # nose-relative metric meaningless. Placed just above the ear line by default.
    lm[0] = {"x": 0.50 + shoulder_dx + ear_dx,
             "y": shoulder_y - 0.10 if nose_y is None else nose_y,
             "z": 0, "visibility": 1.0}
    return {"frame_index": frame_index, "landmarks": lm}


class OverheadPressMetricsTests(unittest.TestCase):
    def test_wrist_above_shoulder_sign(self) -> None:
        raw = ohp_compute_raw([ohp_frame(160, wrist_y=0.20)], 30.0)  # wrists above shoulders
        self.assertLess(raw[0]["wrist_above_shoulder"], 0.0)

    def test_phases_include_lockout_at_top(self) -> None:
        frames = ([ohp_frame(90, 0.45, frame_index=i) for i in range(4)]
                  + [ohp_frame(175, 0.15, frame_index=i + 4) for i in range(4)]
                  + [ohp_frame(90, 0.45, frame_index=i + 8) for i in range(4)])
        phases = ohp_assign_phases(ohp_compute_raw(frames, 30.0))
        self.assertIn("lockout", phases)

    def test_ohp_frame_fixture_controls_elbow_angle(self) -> None:
        # The fixture's elbow landmark must be computed from `elbow_angle`, not hardcoded,
        # so avg_elbow_angle actually tracks the caller's requested angle.
        raw_straight = ohp_compute_raw([ohp_frame(elbow_angle=178, wrist_y=0.15)], 30.0)
        self.assertAlmostEqual(raw_straight[0]["avg_elbow_angle"], 178, delta=3)

        raw_bent = ohp_compute_raw([ohp_frame(elbow_angle=140, wrist_y=0.15)], 30.0)
        self.assertAlmostEqual(raw_bent[0]["avg_elbow_angle"], 140, delta=3)

    def test_wrist_above_nose_sign(self) -> None:
        # Nose sits at shoulder_y - 0.10 = 0.30 by default; wrists at 0.20 are ABOVE it, and
        # MediaPipe y grows downward, so the normalized clearance must be negative.
        raw = ohp_compute_raw([ohp_frame(160, wrist_y=0.20)], 30.0)
        self.assertLess(raw[0]["wrist_above_nose"], 0.0)
        # Wrists stalled below the nose flips the sign.
        raw_low = ohp_compute_raw([ohp_frame(160, wrist_y=0.40)], 30.0)
        self.assertGreater(raw_low[0]["wrist_above_nose"], 0.0)

    def test_anterior_offset_sign_matches_back_lean_convention(self) -> None:
        # Anchors the single module-wide facing convention: in ONE frame, the shift the
        # back-lean rule reads as "shoulders behind the hips" (positive torso_lean_signed_deg)
        # and the shift the forward-head rule reads as "ear anterior of the shoulder line"
        # (positive ear_forward_offset) must agree in sign. If they ever disagree, the module
        # has two conflicting facing conventions.
        # (Re-anchored from `wrist_forward_offset` to `ear_forward_offset` when the bar-path
        # sub-criterion was withdrawn; `ear_dx` is needed because the fixture's head rides
        # WITH the shoulders, so shoulder_dx alone leaves ear_forward_offset at exactly 0.)
        raw = ohp_compute_raw([ohp_frame(160, wrist_y=0.15, shoulder_dx=0.15, ear_dx=-0.05)], 30.0)
        self.assertGreater(raw[0]["torso_lean_signed_deg"], 15.0)
        self.assertGreater(raw[0]["ear_forward_offset"], 0.0)

    def test_wrist_forward_offset_metric_is_gone(self) -> None:
        # The bar-path sub-criterion was withdrawn (see `rule_forward_head`); its metric must
        # not survive as dead pre-wiring that a future reader mistakes for a live signal.
        from src.pose.movements.overhead_press import OHP_METRIC_KEYS
        raw = ohp_compute_raw([ohp_frame(160, wrist_y=0.15, shoulder_dx=0.15)], 30.0)
        self.assertNotIn("wrist_forward_offset", raw[0])
        self.assertNotIn("wrist_forward_offset", OHP_METRIC_KEYS)

    def test_ear_forward_offset_metric(self) -> None:
        # shoulder width is 0.10; ear_dx=-0.05 puts the ears 0.5 shoulder-widths anterior.
        raw = ohp_compute_raw([ohp_frame(160, wrist_y=0.15, ear_dx=-0.05)], 30.0)
        self.assertAlmostEqual(raw[0]["ear_forward_offset"], 0.5, places=6)
        # Head stacked over the shoulders => no anterior offset, even during a back lean.
        raw_stacked = ohp_compute_raw([ohp_frame(160, wrist_y=0.15, shoulder_dx=0.15)], 30.0)
        self.assertAlmostEqual(raw_stacked[0]["ear_forward_offset"], 0.0, places=6)

    def test_wrist_height_asymmetry_metric(self) -> None:
        frame = ohp_frame(160, wrist_y=0.15)
        # Push the right wrist down relative to the left to create asymmetry.
        frame["landmarks"][16]["y"] = 0.30
        raw = ohp_compute_raw([frame], 30.0)
        self.assertGreater(raw[0]["wrist_height_asymmetry"], 0.15)


def run_ohp(frames, view="side", vc=0.8):
    from src.pose.movements import registry
    from src.pose.movements.base import run_detector
    return run_detector(registry.get_detector("Overhead Press"), frames, 30.0, view, vc).detections


class OverheadPressRulesTests(unittest.TestCase):
    _run = staticmethod(run_ohp)

    def test_incomplete_lockout_flagged(self) -> None:
        frames = ([ohp_frame(90, 0.45, frame_index=i) for i in range(4)]
                  + [ohp_frame(140, 0.15, frame_index=i + 4) for i in range(6)]   # elbows never extend, wrists up
                  + [ohp_frame(90, 0.45, frame_index=i + 10) for i in range(4)])
        ids = {d.fault_id for d in self._run(frames)}
        self.assertIn("ohp_incomplete_lockout", ids)

    def test_full_lockout_not_flagged(self) -> None:
        frames = ([ohp_frame(90, 0.45, frame_index=i) for i in range(4)]
                  + [ohp_frame(178, 0.15, frame_index=i + 4) for i in range(6)]   # full extension, wrists up
                  + [ohp_frame(90, 0.45, frame_index=i + 10) for i in range(4)])
        ids = {d.fault_id for d in self._run(frames)}
        self.assertNotIn("ohp_incomplete_lockout", ids)

    def test_lockout_severity_follows_the_criterion_that_fired(self) -> None:
        # A maximally failed press: elbows locked straight (178 deg, so the ELBOW criterion
        # does NOT fire) but the bar never rises above the shoulders at all (wrist_y 0.45 vs
        # shoulder_y 0.40 => wrist_above_shoulder = +0.05, so the WRIST criterion fires).
        # Selecting the severity ramp by "is the elbow reading finite?" used to take the
        # elbow ramp here and emit severity 0.0 / confidence 0.0 -- a phantom detection whose
        # own evidence said the elbow was fine.
        frames = [ohp_frame(178, wrist_y=0.45, frame_index=i) for i in range(12)]
        det = next(
            d for d in self._run(frames) if d.fault_id == "ohp_incomplete_lockout"
        )
        self.assertGreater(det.severity, 0.0)
        self.assertGreater(det.confidence, 0.0)
        # severity_from_range(0.05, mild=0.0, severe=0.15) = 0.05 / 0.15 = 0.3333
        self.assertAlmostEqual(det.severity, 0.3333, places=4)
        self.assertEqual(det.evidence["primary_label"], "wrist height above shoulder")
        self.assertAlmostEqual(det.evidence["primary_value"], 0.05, places=4)

    def test_lockout_rule_carries_citation(self) -> None:
        frames = [ohp_frame(140, 0.15, frame_index=i) for i in range(12)]
        det = next((d for d in self._run(frames) if d.fault_id == "ohp_incomplete_lockout"), None)
        assert det is not None and det.citation and det.citation_support

    def test_asymmetric_press_flagged_when_wrists_uneven(self) -> None:
        frames = []
        for i in range(12):
            frame = ohp_frame(160, wrist_y=0.15, frame_index=i)
            # Shoulder width here is 0.10 (0.55 - 0.45); push the right wrist down by more
            # than 0.15 * shoulder_width (~0.015) to trip the asymmetry threshold.
            frame["landmarks"][16]["y"] = 0.35
            frames.append(frame)
        ids = {d.fault_id for d in self._run(frames, view="front")}
        self.assertIn("ohp_asymmetric_press", ids)

    def test_asymmetric_press_not_flagged_when_wrists_even(self) -> None:
        frames = [ohp_frame(160, wrist_y=0.15, frame_index=i) for i in range(12)]
        ids = {d.fault_id for d in self._run(frames, view="front")}
        self.assertNotIn("ohp_asymmetric_press", ids)

    def test_back_lean_flagged(self) -> None:
        # Same setup -> press -> lockout -> lower shape as the lockout tests, but with
        # shoulders shifted +0.15 in x on every frame (hips fixed) so torso_lean_signed_deg
        # is well past the 15 deg threshold for the press/lockout portion of the rep.
        frames = ([ohp_frame(90, 0.45, frame_index=i, shoulder_dx=0.15) for i in range(4)]
                  + [ohp_frame(178, 0.15, frame_index=i + 4, shoulder_dx=0.15) for i in range(6)]
                  + [ohp_frame(90, 0.45, frame_index=i + 10, shoulder_dx=0.15) for i in range(4)])
        ids = {d.fault_id for d in self._run(frames)}
        self.assertIn("ohp_lumbar_hyperextension", ids)

    def test_upright_not_flagged(self) -> None:
        # Identical rep shape with shoulder_dx=0.0 (shoulders directly above hips
        # throughout) must not trip the back-lean rule.
        frames = ([ohp_frame(90, 0.45, frame_index=i) for i in range(4)]
                  + [ohp_frame(178, 0.15, frame_index=i + 4) for i in range(6)]
                  + [ohp_frame(90, 0.45, frame_index=i + 10) for i in range(4)])
        ids = {d.fault_id for d in self._run(frames)}
        self.assertNotIn("ohp_lumbar_hyperextension", ids)


class OverheadPressElevationRuleTests(unittest.TestCase):
    """`ohp_insufficient_elevation` -- and, crucially, its separation from
    `ohp_incomplete_lockout`, which the spec explicitly demands be distinguishable."""

    _run = staticmethod(run_ohp)

    @staticmethod
    def _rep(elbow_angle: float, top_wrist_y: float) -> list[dict]:
        """setup -> press -> lockout -> lower, with the top of the rep held for 8 frames
        (min_frames is 6 at 30 fps)."""
        return ([ohp_frame(90, 0.45, frame_index=i) for i in range(4)]
                + [ohp_frame(elbow_angle, top_wrist_y, frame_index=i + 4) for i in range(8)]
                + [ohp_frame(90, 0.45, frame_index=i + 12) for i in range(4)])

    def test_insufficient_elevation_flagged_when_hands_stall_at_nose(self) -> None:
        # Nose sits at 0.30 by default, so wrists held at 0.30 never clear the head.
        ids = {d.fault_id for d in self._run(self._rep(178, 0.30))}
        self.assertIn("ohp_insufficient_elevation", ids)

    def test_elevation_not_flagged_when_hands_finish_overhead(self) -> None:
        # Same rep with the hands 1.5 shoulder-widths above the nose.
        ids = {d.fault_id for d in self._run(self._rep(178, 0.15))}
        self.assertNotIn("ohp_insufficient_elevation", ids)

    def test_elevation_fires_without_incomplete_lockout(self) -> None:
        # Elbows essentially locked (178 deg) but the hands stall at nose level: this is a
        # short press, NOT an incomplete lockout. Only the elevation rule may fire.
        ids = {d.fault_id for d in self._run(self._rep(178, 0.30))}
        self.assertIn("ohp_insufficient_elevation", ids)
        self.assertNotIn("ohp_incomplete_lockout", ids)

    def test_incomplete_lockout_fires_without_elevation(self) -> None:
        # The inverse: hands well clear of the nose but elbows stuck at 140 deg.
        ids = {d.fault_id for d in self._run(self._rep(140, 0.15))}
        self.assertIn("ohp_incomplete_lockout", ids)
        self.assertNotIn("ohp_insufficient_elevation", ids)

    def test_elevation_rule_carries_citation(self) -> None:
        det = next(
            (d for d in self._run(self._rep(178, 0.30))
             if d.fault_id == "ohp_insufficient_elevation"),
            None,
        )
        assert det is not None and det.citation and det.citation_support
        self.assertIn("PMC9354811", det.citation)

    @staticmethod
    def _flat_rep(wrist_y: float) -> list[dict]:
        """12 IDENTICAL frames. The window-5 median smoothing in `run_detector` is then a
        no-op, so `wrist_above_nose` equals its per-frame value exactly and the severity ramp
        can be asserted as a number. (`_rep` above cannot: its 0.45 setup frames blend into
        the top frames under smoothing, and the resulting worst value saturates the ramp.)
        The default nose sits at y=0.30 and shoulder width is 0.10, so
        wrist_above_nose == (wrist_y - 0.30) / 0.10."""
        return [ohp_frame(178, wrist_y, frame_index=i) for i in range(12)]

    def test_elevation_fires_just_inside_threshold(self) -> None:
        # wrist_y=0.29 => wrist_above_nose = -0.10, just past the -0.15 fire threshold.
        det = next(
            (d for d in self._run(self._flat_rep(0.29))
             if d.fault_id == "ohp_insufficient_elevation"),
            None,
        )
        self.assertIsNotNone(det)
        self.assertAlmostEqual(det.evidence["max_wrist_above_nose"], -0.10, places=4)
        # Deterministic ramp: severity_from_range(v, mild=-0.15, severe=0.35)
        # = (-0.10 + 0.15) / (0.35 + 0.15) = 0.1. Pins BOTH range endpoints, so moving
        # `severe` (e.g. 0.35 -> 0.9) fails here.
        self.assertAlmostEqual(det.severity, 0.1, places=4)

    def test_elevation_silent_just_outside_threshold(self) -> None:
        # wrist_y=0.28 => wrist_above_nose = -0.20, just short of -0.15: hands barely clear
        # the head but they DO clear it. Any fire threshold below -0.20 (e.g. -1.0) would
        # wrongly fire here.
        ids = {d.fault_id for d in self._run(self._flat_rep(0.28))}
        self.assertNotIn("ohp_insufficient_elevation", ids)

    def test_elevation_confidence_is_downgraded_off_observable_views(self) -> None:
        # `rear` is outside {side, front, front_oblique}, so confidence must be severity
        # scaled by the 0.65 unobservable-view multiplier, not severity itself.
        observable = next(
            d for d in self._run(self._flat_rep(0.29), view="side")
            if d.fault_id == "ohp_insufficient_elevation"
        )
        unobservable = next(
            d for d in self._run(self._flat_rep(0.29), view="rear")
            if d.fault_id == "ohp_insufficient_elevation"
        )
        self.assertEqual(observable.severity, unobservable.severity)
        self.assertAlmostEqual(observable.confidence, 0.1, places=4)
        self.assertAlmostEqual(unobservable.confidence, 0.065, places=4)
        self.assertEqual(unobservable.observability, "medium")

    def test_elevation_not_flagged_when_nose_invisible(self) -> None:
        # No nose landmark => wrist_above_nose is NaN => the rule must stay silent rather
        # than guess, and must not invalidate the frame for the other OHP rules.
        frames = self._rep(140, 0.30)
        for frame in frames:
            frame["landmarks"][0]["visibility"] = 0.0
        ids = {d.fault_id for d in self._run(frames)}
        self.assertNotIn("ohp_insufficient_elevation", ids)
        self.assertIn("ohp_incomplete_lockout", ids)


class OverheadPressForwardHeadRuleTests(unittest.TestCase):
    """`ohp_forward_head` -- the ear-anterior cue, its hard sagittal-view gate, and the
    view-confidence floor. The bar-path sub-criterion is WITHDRAWN (see `rule_forward_head`);
    `test_back_lean_rep_does_not_emit_forward_head` and
    `test_wrists_unstacked_alone_emits_nothing` are its regressions."""

    _run = staticmethod(run_ohp)

    @staticmethod
    def _frames(**kwargs) -> list[dict]:
        # All 12 frames identical, so the window-5 median smoothing in `run_detector` is a
        # no-op and every metric equals its per-frame value exactly. That is what lets the
        # severity assertions below name an exact number.
        # elbow_angle=175 (not 160): the fixture's trig construction lands 160 at
        # 159.99998 deg, which is `< 160.0` and so tripped `ohp_incomplete_lockout` at
        # severity ~5e-7 in every forward-head test. That knife-edge artifact is a FIXTURE
        # bug, not a production one, and it must not pollute a rule-isolation fixture.
        return [ohp_frame(175, wrist_y=0.15, frame_index=i, **kwargs) for i in range(12)]

    def test_forward_head_flagged_when_ear_anterior(self) -> None:
        # ear_dx=-0.05 => ears 0.5 shoulder-widths anterior of the shoulder line (> 0.30).
        dets = self._run(self._frames(ear_dx=-0.05), view="side")
        det = next((d for d in dets if d.fault_id == "ohp_forward_head"), None)
        self.assertIsNotNone(det)
        self.assertGreater(det.evidence["max_ear_forward_offset"], 0.30)

    def test_forward_head_severity_is_the_exact_ramp_value(self) -> None:
        # Deterministic ramp: severity_from_range(offset, mild=0.30, severe=0.60).
        # ear_dx=-0.05 => offset 0.50 => (0.50 - 0.30) / (0.60 - 0.30) = 0.6667.
        # Pins BOTH range endpoints, so widening `severe` (e.g. 0.60 -> 5.0) fails here.
        det = next(
            d for d in self._run(self._frames(ear_dx=-0.05), view="side")
            if d.fault_id == "ohp_forward_head"
        )
        self.assertAlmostEqual(det.evidence["max_ear_forward_offset"], 0.5, places=4)
        self.assertAlmostEqual(det.severity, 0.6667, places=4)
        self.assertAlmostEqual(det.confidence, 0.6667, places=4)

    def test_forward_head_fires_just_inside_threshold(self) -> None:
        # ear_dx=-0.032 => offset 0.32, just past the 0.30 fire threshold.
        det = next(
            (d for d in self._run(self._frames(ear_dx=-0.032), view="side")
             if d.fault_id == "ohp_forward_head"),
            None,
        )
        self.assertIsNotNone(det)
        self.assertAlmostEqual(det.severity, 0.0667, places=4)

    def test_forward_head_silent_just_outside_threshold(self) -> None:
        # ear_dx=-0.025 => offset 0.25, just short of 0.30: a real forward-head lean that is
        # not yet a fault. Any threshold LOWER than 0.30 (e.g. 0.05) would fire here.
        ids = {d.fault_id for d in self._run(self._frames(ear_dx=-0.025), view="side")}
        self.assertNotIn("ohp_forward_head", ids)

    def test_forward_head_not_flagged_when_stacked(self) -> None:
        ids = {d.fault_id for d in self._run(self._frames(), view="side")}
        self.assertNotIn("ohp_forward_head", ids)

    def test_back_lean_rep_does_not_emit_forward_head(self) -> None:
        # REGRESSION for the withdrawn bar-path sub-criterion. A PURE back lean (the exact
        # fixture `test_back_lean_flagged` uses) moves the shoulders posterior while the hands
        # stay over the base of support. The old wrist-vs-shoulder cue read that as a
        # saturated bar-path fault (severity 1.0) and outranked the fault that actually
        # occurred. Only the back-lean rule may speak for this rep.
        frames = ([ohp_frame(90, 0.45, frame_index=i, shoulder_dx=0.15) for i in range(4)]
                  + [ohp_frame(178, 0.15, frame_index=i + 4, shoulder_dx=0.15) for i in range(6)]
                  + [ohp_frame(90, 0.45, frame_index=i + 10, shoulder_dx=0.15) for i in range(4)])
        dets = self._run(frames, view="side")
        ids = [d.fault_id for d in dets]
        self.assertNotIn("ohp_forward_head", ids)
        self.assertIn("ohp_lumbar_hyperextension", ids)
        # ... and it must be the top-ranked detection, not buried under a phantom.
        self.assertEqual(dets[0].fault_id, "ohp_lumbar_hyperextension")

    def test_wrists_unstacked_alone_emits_nothing(self) -> None:
        # The old `test_bar_forward_flagged_when_wrists_not_stacked` fixture, inverted.
        # shoulder_dx=+0.05 leaves the wrists 0.5 shoulder-widths anterior of the shoulders
        # but implies only an 8.1 deg lean -- under the 15 deg back-lean threshold. With the
        # bar-path cue withdrawn this rep is now (correctly) silent end to end.
        dets = self._run(self._frames(shoulder_dx=0.05), view="side")
        self.assertEqual([d.fault_id for d in dets], [])

    def test_forward_head_is_hard_gated_to_sagittal_views(self) -> None:
        # IDENTICAL faulty frames: they fire from a sagittal view and must produce NO
        # detection at all (not a low-confidence one) from a frontal view, where the sign of
        # a pure horizontal offset is arbitrary.
        frames = self._frames(ear_dx=-0.05)
        self.assertIn("ohp_forward_head", {d.fault_id for d in self._run(frames, view="side")})
        self.assertIn(
            "ohp_forward_head",
            {d.fault_id for d in self._run(frames, view="front_oblique")},
        )
        for blind_view in ("front", "rear", "rear_oblique"):
            ids = {d.fault_id for d in self._run(frames, view=blind_view)}
            self.assertNotIn("ohp_forward_head", ids, f"fired from view={blind_view}")

    def test_forward_head_requires_confident_view_classification(self) -> None:
        # A LOW-confidence `side`/`front_oblique` classification must not authorize a
        # directional claim -- the same floor `rule_knees_forward` applies in squat.py
        # (SIDE_VIEW_CONF_THRESHOLD = 0.20 in src.pose.pose_rule_detector).
        from src.pose.pose_rule_detector import SIDE_VIEW_CONF_THRESHOLD
        frames = self._frames(ear_dx=-0.05)
        for view in ("side", "front_oblique"):
            below = {d.fault_id for d in self._run(frames, view=view, vc=SIDE_VIEW_CONF_THRESHOLD - 0.01)}
            self.assertNotIn("ohp_forward_head", below, f"fired at low confidence, view={view}")
            at = {d.fault_id for d in self._run(frames, view=view, vc=SIDE_VIEW_CONF_THRESHOLD)}
            self.assertIn("ohp_forward_head", at, f"did not fire at the floor, view={view}")

    def test_head_behind_shoulders_does_not_fire(self) -> None:
        # Positive ear_dx pushes the head POSTERIOR. Forward head is directional: the
        # opposite deviation must not trip it.
        ids = {d.fault_id for d in self._run(self._frames(ear_dx=0.05), view="side")}
        self.assertNotIn("ohp_forward_head", ids)

    def test_forward_head_rule_carries_citation(self) -> None:
        det = next(
            (d for d in self._run(self._frames(ear_dx=-0.05), view="side")
             if d.fault_id == "ohp_forward_head"),
            None,
        )
        assert det is not None and det.citation and det.citation_support
        self.assertIn("PMC13116542", det.citation)


class OverheadPressFrameValidityTests(unittest.TestCase):
    """The module-level silence risk: BOTH shoulders are in `ohp_compute_raw`'s `required`
    tuple, so dropping either one invalidates the frame outright and silences ALL FIVE OHP
    rules -- not just the forward-head rule whose normalizer it also degrades.

    SCOPE OF THE PROOF: this exercises the VALIDITY-GATE mechanism on a synthetic frame. It
    does NOT reproduce the real-world scenario (a hard sagittal view where the far shoulder
    is hidden behind the near one), because the fixture's landmark layout is not sagittal
    geometry. The mechanism is view-independent, which is why that is enough to pin it; the
    real-world hit rate remains unmeasured."""

    _run = staticmethod(run_ohp)

    def test_occluded_far_shoulder_silences_every_ohp_rule(self) -> None:
        # A rep that is faulty on several axes at once: elbows stuck at 140 deg (incomplete
        # lockout), hands stalled at nose level (insufficient elevation), head jutting
        # anterior (forward head), shoulders behind the hips (back lean).
        frames = [
            ohp_frame(140, wrist_y=0.30, frame_index=i, shoulder_dx=0.15, ear_dx=-0.05)
            for i in range(12)
        ]
        loud = {d.fault_id for d in self._run(frames, view="side")}
        self.assertIn("ohp_incomplete_lockout", loud)
        self.assertIn("ohp_insufficient_elevation", loud)
        self.assertIn("ohp_forward_head", loud)
        self.assertIn("ohp_lumbar_hyperextension", loud)

        for frame in frames:
            frame["landmarks"][12]["visibility"] = 0.0   # far (right) shoulder occluded
        raw = ohp_compute_raw(frames, 30.0)
        self.assertTrue(all(not item["valid"] for item in raw))
        self.assertEqual([d.fault_id for d in self._run(frames, view="side")], [])
