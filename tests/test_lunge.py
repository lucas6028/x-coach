import math
import unittest

import numpy as np

from src.pose.movements.base import RuleContext, run_detector


def _lm(x: float, y: float, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility}


def _knee_at(
    hip_xy: tuple[float, float],
    ankle_xy: tuple[float, float],
    along: float,
    perpendicular: float,
) -> tuple[float, float]:
    """Place a knee at (`along`, `perpendicular`) in the leg's own frame.

    `along` is the fraction of the way from hip to ankle; `perpendicular` is the signed
    displacement off the hip-ankle line in ABSOLUTE image units, positive along the leg's
    left-hand normal.

    WHY NOT the perpendicular-bisector angle construction that tests/test_pushup.py::_elbow_xy
    uses: for THIS metric that construction is the wrong control. `_medial_offset_ratio`
    measures exactly the perpendicular displacement, so a fixture that requests a knee ANGLE
    is implicitly requesting a perpendicular offset -- and a 90-degree in-image knee bend over
    a 0.40-long leg puts the knee 0.20 off the line, which is 1.7 HIP WIDTHS, an order of
    magnitude past the spec's 0.10 fire threshold. Controlling the offset directly makes
    `left_knee_medial_offset_ratio` equal the requested value BY CONSTRUCTION, which is the
    property a fixture is supposed to have; the knee angle is then derived and asserted as
    measured rather than requested.
    """
    hx, hy = hip_xy
    ax, ay = ankle_xy
    dx, dy = ax - hx, ay - hy
    norm = math.hypot(dx, dy)
    ux, uy = (0.0, 1.0) if norm < 1e-9 else (dx / norm, dy / norm)
    px, py = -uy, ux
    return (hx + along * dx + perpendicular * px, hy + along * dy + perpendicular * py)


def lunge_frame(
    lead: str = "left",
    lead_medial: float = 0.0,
    trail_medial: float = 0.0,
    lead_anterior: float = 0.60,
    pelvis_tilt_deg: float = 0.0,
    lead_offset: float = 0.10,
    frame_index: int = 0,
) -> dict:
    """One OBLIQUE-view lunge frame, y growing DOWNWARD. Split stance along image x.

    Knobs:
      lead          -- "left" or "right"; which leg is forward. The lead ankle is displaced
                       by `lead_offset` along image x, which is what gives the lead leg a
                       genuinely different in-image knee angle from the trailing leg.
      lead_medial   -- lead-knee displacement toward the midline, IN HIP WIDTHS.
      trail_medial  -- the same for the trailing leg.
      lead_anterior -- lead-knee displacement in the ANTERIOR (step) direction, in hip widths.
                       This is what bends the lead knee in-image; the default 0.60 gives a
                       clearly-flexed lead leg against a straight trailing one.
      pelvis_tilt_deg -- rotates the hip pair about the mid-hip; POSITIVE = RIGHT hip lower,
                       matching `pelvis_tilt_signed_deg`'s convention exactly.

    ------------------------------------------------------------------------------------
    TWO PROJECTION FACTS THIS FIXTURE ENCODES. Both are properties of monocular geometry,
    not fixture conveniences, and both must be carried into `rule_knee_valgus`'s docstring.
    ------------------------------------------------------------------------------------

    (1) IN A STRICTLY FRONTAL VIEW, in-image knee flexion and medial offset are the SAME
        degree of freedom. A knee on the hip-ankle line has an interior angle of exactly
        180 degrees, so the ONLY way to bend a knee in-image is to move it off that line --
        which is precisely what `_medial_offset_ratio` measures. A lunge's real flexion is
        sagittal and projects onto the leg line frontally, contributing nothing to either.
        Consequence: `resolve_lead_side` and `rule_knee_valgus` read the same quantity in a
        pure frontal view. This fixture uses an OBLIQUE stance to break the degeneracy, and
        `front_oblique`/`rear_oblique` are the labels production actually reaches anyway.

    (2) OBLIQUELY, ANTERIOR KNEE TRAVEL CONTAMINATES THE VALGUS PROXY. `lead_anterior` and
        `lead_medial` add to the SAME perpendicular axis, because an oblique camera gives the
        anterior direction an in-image component. So `_medial_offset_ratio` cannot separate
        "knee travelled forward" from "knee caved inward" off-axis; it is clean only in a true
        frontal view, which is the view production never emits. That is a genuine limitation
        of the spec's frontal-plane proxy under this pipeline's reachable view labels, and it
        is documented rather than corrected -- correcting it needs depth this pipeline lacks.

    THE OPEN QUESTIONS THESE RAISE, which Phase 2 must answer on real data: does
    `resolve_lead_side` work on the 88 cam17 reps the dataset calls `front` (if in-image knee
    angles there are near-symmetric, the ambiguity guard fires and the frontal rules go silent
    on exactly the camera routed to them), and does `lunge_knee_valgus` fire in proportion to
    step depth rather than to correctness (the signature of contamination (2))? Task 8 Step 3
    reports the unresolved rate and Step 4 reads the valgus/depth relationship for exactly
    these reasons -- do not treat either as a harness bug.
    """
    half_hip = 0.06
    tilt = math.radians(pelvis_tilt_deg)
    left_hip = (0.50 - half_hip * math.cos(tilt), 0.50 - half_hip * math.sin(tilt))
    right_hip = (0.50 + half_hip * math.cos(tilt), 0.50 + half_hip * math.sin(tilt))
    # The lead ankle steps forward along image x; the trailing ankle stays under its hip.
    lead_shift = lead_offset if lead == "left" else -lead_offset
    left_ankle = (0.44 + (lead_shift if lead == "left" else 0.0), 0.90)
    right_ankle = (0.56 + (lead_shift if lead == "right" else 0.0), 0.90)

    hip_width = math.dist(left_hip, right_hip)
    # Medial is toward the mid-hip. `_knee_at`'s +perpendicular is the leg's left-hand normal
    # P = (-uy, ux); for a mostly-vertical leg (uy near 1, the case every fixture frame here
    # produces) P_x is close to -1 for BOTH legs, so P points AWAY from the midline for the
    # left leg (whose hip sits left of mid-hip, so "toward" is +x) and TOWARD it for the right
    # leg (whose hip sits right of mid-hip, so "toward" is -x) -- the opposite of what a naive
    # "left leg -> +1" reading suggests. FIXTURE ADJUSTMENT (see task-2-report.md): the sign
    # was originally assigned the other way around and every medial-offset test read the exact
    # negative of the requested value, confirmed independently by
    # `test_anterior_knee_travel_contaminates_the_valgus_proxy`'s own docstring, which the
    # inverted sign contradicted (a "clean, deep" lunge reading NEGATIVE/varus instead of the
    # positive/valgus-reading limitation the test exists to pin).
    left_sign = -1.0 if left_hip[0] < right_hip[0] else 1.0
    left_medial = lead_medial if lead == "left" else trail_medial
    right_medial = lead_medial if lead == "right" else trail_medial
    # Anterior travel lands on the same perpendicular axis -- see fact (2) in the docstring.
    # It is applied to the LEAD leg only; the trailing leg stays straight.
    left_anterior = lead_anterior if lead == "left" else 0.0
    right_anterior = lead_anterior if lead == "right" else 0.0

    lm = [_lm(0.50, 0.50) for _ in range(33)]
    lm[11], lm[12] = _lm(0.44, 0.25), _lm(0.56, 0.25)
    lm[23], lm[24] = _lm(*left_hip), _lm(*right_hip)
    lm[25] = _lm(*_knee_at(left_hip, left_ankle, 0.5,
                           left_sign * (left_medial + left_anterior) * hip_width))
    lm[26] = _lm(*_knee_at(right_hip, right_ankle, 0.5,
                           -left_sign * (right_medial + right_anterior) * hip_width))
    lm[27], lm[28] = _lm(*left_ankle), _lm(*right_ankle)
    lm[29], lm[30] = _lm(left_ankle[0] - 0.02, 0.92), _lm(right_ankle[0] - 0.02, 0.92)
    lm[31], lm[32] = _lm(left_ankle[0] + 0.04, 0.94), _lm(right_ankle[0] + 0.04, 0.94)
    return {"frame_index": frame_index, "landmarks": lm}


def mirrored(frame: dict) -> dict:
    """The same body FACING THE OTHER WAY: positions reflect about x=0.5, indices unchanged.

    MediaPipe labels landmarks ANATOMICALLY -- index 23 is the subject's left hip whether they
    face the camera or away from it -- so turning around reflects where each landmark projects
    without renumbering anything. (Swapping the CONTENTS of 23/24 would model a left/right
    identity swap, a different operation, and would negate `pelvis_tilt_signed_deg` by
    construction for any implementation -- see task-2-report.md's fix-report addendum for the
    arithmetic.)

    This transform is what makes the facing-independence test discriminating: reflection flips
    `sign(right_hip.x - left_hip.x)`, so an implementation using SIGNED dx reads 168 degrees
    here while the `abs(dx)` form the metric actually uses holds at 12. Which hip is physically
    lower is unchanged by turning around, and the metric must agree.
    """
    lm = [dict(item) for item in frame["landmarks"]]
    for item in lm:
        item["x"] = 1.0 - item["x"]
    return {"frame_index": frame.get("frame_index", 0), "landmarks": lm}


class LungeMetricTests(unittest.TestCase):
    def test_medial_offset_equals_the_requested_displacement(self) -> None:
        # Controlled by construction: `lead_medial` is in hip widths and the metric normalizes
        # by hip width, so the two must agree to within the fixture's rounding.
        from src.pose.movements.lunge import lunge_compute_raw

        raw = lunge_compute_raw(
            [lunge_frame(lead="left", lead_medial=0.18, lead_anterior=0.0)], 30.0
        )
        self.assertTrue(raw[0]["valid"])
        self.assertAlmostEqual(raw[0]["left_knee_medial_offset_ratio"], 0.18, delta=0.01)

    def test_medial_offset_means_toward_the_midline_on_the_right_leg_too(self) -> None:
        # The sign convention is the whole point: "medial" is toward the mid-hip for BOTH
        # legs, which is opposite image-x directions for left and right.
        from src.pose.movements.lunge import lunge_compute_raw

        raw = lunge_compute_raw(
            [lunge_frame(lead="right", lead_medial=0.18, lead_anterior=0.0)], 30.0
        )
        self.assertAlmostEqual(raw[0]["right_knee_medial_offset_ratio"], 0.18, delta=0.01)

    def test_a_knee_tracking_outside_reads_negative(self) -> None:
        from src.pose.movements.lunge import lunge_compute_raw

        raw = lunge_compute_raw(
            [lunge_frame(lead="left", lead_medial=-0.18, lead_anterior=0.0)], 30.0
        )
        self.assertLess(raw[0]["left_knee_medial_offset_ratio"], 0.0)

    def test_a_well_tracked_lunge_sits_below_the_fire_threshold(self) -> None:
        # THE SCALE CHECK. Without it, the rule-level boundary tests (which inject the metric
        # directly) would prove `severity_from_range` works while never establishing that
        # `_medial_offset_ratio` produces spec-scale values from an actual body. A correct
        # lunge must land well under the spec's 0.10; a caved one inside its 0.10-0.25 ramp.
        from src.pose.movements.lunge import lunge_compute_raw

        good = lunge_compute_raw(
            [lunge_frame(lead="left", lead_medial=0.02, lead_anterior=0.0)], 30.0
        )[0]
        self.assertLess(abs(good["left_knee_medial_offset_ratio"]), 0.10)

        caved = lunge_compute_raw(
            [lunge_frame(lead="left", lead_medial=0.18, lead_anterior=0.0)], 30.0
        )[0]
        self.assertGreater(caved["left_knee_medial_offset_ratio"], 0.10)
        self.assertLess(caved["left_knee_medial_offset_ratio"], 0.25)

    def test_anterior_knee_travel_contaminates_the_valgus_proxy(self) -> None:
        # PINNED ON PURPOSE -- this documents a limitation, it does not endorse it. Off-axis,
        # anterior travel and medial collapse land on the same perpendicular measurement, so a
        # deep, perfectly-tracked lunge reads as valgus. The rule ships with this stated in its
        # docstring; if someone later separates the two (it needs depth this pipeline lacks),
        # this test should fail and force a spec conversation rather than silently changing
        # what a stored `lunge_knee_valgus` severity meant.
        from src.pose.movements.lunge import lunge_compute_raw

        deep_but_clean = lunge_compute_raw(
            [lunge_frame(lead="left", lead_medial=0.0, lead_anterior=0.60)], 30.0
        )[0]
        self.assertGreater(deep_but_clean["left_knee_medial_offset_ratio"], 0.10)

    def test_the_lead_leg_reads_a_smaller_knee_angle_than_the_trailing_leg(self) -> None:
        # Derived, not requested: the split stance is what makes the lead knee measurably more
        # flexed in-image. Asserting the ORDERING (not a specific angle) is what
        # `resolve_lead_side` actually depends on.
        from src.pose.movements.lunge import lunge_compute_raw

        raw = lunge_compute_raw([lunge_frame(lead="left")], 30.0)
        self.assertLess(raw[0]["left_knee_angle"], raw[0]["right_knee_angle"])
        self.assertAlmostEqual(raw[0]["min_knee_angle"], raw[0]["left_knee_angle"], places=6)

    def test_pelvis_tilt_is_positive_when_the_right_hip_is_lower(self) -> None:
        from src.pose.movements.lunge import lunge_compute_raw

        raw = lunge_compute_raw([lunge_frame(pelvis_tilt_deg=12.0)], 30.0)
        self.assertAlmostEqual(raw[0]["pelvis_tilt_signed_deg"], 12.0, delta=1.0)

    def test_pelvis_tilt_sign_does_not_depend_on_facing(self) -> None:
        # A real turn-around (see `mirrored`): landmark indices stay anatomically bound, only
        # x reflects. Which hip is physically lower is unchanged, so the metric must not flip.
        # A tilt built with the RIGHT hip lower stays positive after the subject turns around.
        # THIS TEST HAS TEETH: reflection flips sign(right_hip.x - left_hip.x), so a signed-dx
        # implementation of this metric would read 168.0 here, not 12.0 -- `mirrored()` is
        # exactly the transform that discriminates `abs(dx)` from a signed dx.
        from src.pose.movements.lunge import lunge_compute_raw

        raw = lunge_compute_raw([mirrored(lunge_frame(pelvis_tilt_deg=12.0))], 30.0)
        self.assertAlmostEqual(raw[0]["pelvis_tilt_signed_deg"], 12.0, delta=1.0)

    def test_a_frame_missing_a_required_landmark_is_invalid_and_carries_no_metrics(self) -> None:
        from src.pose.movements.lunge import lunge_compute_raw

        frame = lunge_frame()
        frame["landmarks"][25] = _lm(0.44, 0.70, visibility=0.10)
        raw = lunge_compute_raw([frame], 30.0)
        self.assertFalse(raw[0]["valid"])
        self.assertNotIn("left_knee_angle", raw[0])

    def test_lunge_metric_keys_match_the_emitted_metrics(self) -> None:
        """TWO-WAY equality, mirroring
        test_movement_registry.py::test_pushup_metric_keys_match_the_emitted_metrics.
        `run_detector` builds each CoreFrame's metrics dict FROM `LUNGE_METRIC_KEYS`, so a key
        `lunge_compute_raw` emits but the tuple omits is silently dropped and reads back as
        NaN -- for `left_knee_angle`/`right_knee_angle` that would permanently silence
        `resolve_lead_side` with no error anywhere."""
        from src.pose.movements.lunge import LUNGE_METRIC_KEYS, lunge_compute_raw

        raw = lunge_compute_raw([lunge_frame()], 30.0)
        self.assertTrue(raw[0]["valid"], "fixture frame must be valid for this comparison")
        framework_keys = {"frame_index", "time", "valid", "lower_body_visibility"}
        emitted = set(raw[0]) - framework_keys
        self.assertEqual(set(LUNGE_METRIC_KEYS), emitted)
        # No duplicates hiding a missing key behind a matching set size.
        self.assertEqual(len(LUNGE_METRIC_KEYS), len(set(LUNGE_METRIC_KEYS)))


class LungePhaseTests(unittest.TestCase):
    def _descend_and_rise(self) -> list[dict]:
        # 30 frames: the lead knee bends in and back out, i.e. one clean rep on the left leg.
        # Depth is driven by `lead_anterior`, which is what bends the knee in-image.
        depths = list(np.linspace(0.0, 0.80, 15)) + list(np.linspace(0.80, 0.0, 15))
        return [
            lunge_frame(lead="left", lead_anterior=float(d), frame_index=i)
            for i, d in enumerate(depths)
        ]

    def test_phases_run_setup_descent_bottom_ascent(self) -> None:
        from src.pose.movements.lunge import lunge_assign_phases, lunge_compute_raw

        phases = lunge_assign_phases(lunge_compute_raw(self._descend_and_rise(), 30.0))
        self.assertEqual(len(phases), 30)
        self.assertEqual(phases[0], "setup")
        self.assertIn("descent", phases)
        self.assertIn("bottom", phases)
        self.assertIn("ascent", phases)
        self.assertLess(phases.index("descent"), phases.index("ascent"))

    def test_an_empty_clip_returns_no_phases(self) -> None:
        from src.pose.movements.lunge import lunge_assign_phases

        self.assertEqual(lunge_assign_phases([]), [])

    def test_a_clip_with_no_finite_signal_is_entirely_unknown(self) -> None:
        from src.pose.movements.lunge import lunge_assign_phases

        raw = [{"valid": False} for _ in range(10)]
        self.assertEqual(lunge_assign_phases(raw), ["unknown"] * 10)


def _rule_frames(metrics: dict, count: int = 12, phase: str = "bottom") -> list:
    """A window of `count` identical CoreFrames carrying `metrics`.

    Constant values on purpose: `run_detector`'s median smoothing is a no-op over constants,
    so an asserted severity is EXACT rather than approximately right.
    """
    from src.pose.movements.base import CoreFrame

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


def _ctx(view_type: str = "front_oblique", *, min_frames: int = 6, view_confidence: float = 0.9):
    """min_frames=6 is what `run_detector` computes at 30 fps -- max(3, ceil(30 * 0.20)) --
    so a segment-length mutant cannot hide behind an artificially permissive 1."""
    return RuleContext(fps=30.0, view_type=view_type, view_confidence=view_confidence,
                       min_frames=min_frames)


class LeadSideResolutionTests(unittest.TestCase):
    def test_resolves_the_more_flexed_leg_at_the_windows_bottom(self) -> None:
        from src.pose.movements.lunge import resolve_lead_side

        window = _rule_frames({"min_knee_angle": 85.0, "left_knee_angle": 85.0,
                               "right_knee_angle": 165.0})
        self.assertEqual(resolve_lead_side(window), "left")

    def test_resolves_the_right_leg_when_it_is_the_flexed_one(self) -> None:
        from src.pose.movements.lunge import resolve_lead_side

        window = _rule_frames({"min_knee_angle": 85.0, "left_knee_angle": 165.0,
                               "right_knee_angle": 85.0})
        self.assertEqual(resolve_lead_side(window), "right")

    def test_returns_none_when_both_knees_are_within_the_ambiguity_guard(self) -> None:
        # A near-symmetric bottom is not a lunge. Guessing a side here would mis-attribute
        # every fault in the rep, so the rules go silent instead.
        from src.pose.movements.lunge import resolve_lead_side

        window = _rule_frames({"min_knee_angle": 100.0, "left_knee_angle": 100.0,
                               "right_knee_angle": 102.0})
        self.assertIsNone(resolve_lead_side(window))

    def test_returns_none_when_no_frame_is_valid(self) -> None:
        from src.pose.movements.base import CoreFrame
        from src.pose.movements.lunge import resolve_lead_side

        window = [
            CoreFrame(frame_index=i, time=i / 30.0, phase="unknown", valid=False,
                      lower_body_visibility=0.0, metrics={})
            for i in range(12)
        ]
        self.assertIsNone(resolve_lead_side(window))

    def test_resolution_uses_the_bottom_frame_not_the_first(self) -> None:
        # The rep opens with the RIGHT knee incidentally more flexed, but the bottom is
        # unambiguously a LEFT-lead lunge. Resolving on frame 0 would answer "right".
        from src.pose.movements.lunge import resolve_lead_side

        window = _rule_frames({"min_knee_angle": 160.0, "left_knee_angle": 168.0,
                               "right_knee_angle": 160.0}, count=6)
        window += _rule_frames({"min_knee_angle": 85.0, "left_knee_angle": 85.0,
                                "right_knee_angle": 150.0}, count=6)
        self.assertEqual(resolve_lead_side(window), "left")

    def test_silent_just_inside_the_separation_guard(self) -> None:
        # LEAD_SIDE_MIN_SEPARATION_DEG = 5.0. A 4.9-degree gap must still read as ambiguous --
        # a mutant that widens or shrinks the guard to 3.0 or 10.0 would pass the wide-margin
        # cases above but flip this one.
        from src.pose.movements.lunge import resolve_lead_side

        window = _rule_frames({"min_knee_angle": 100.0, "left_knee_angle": 100.0,
                               "right_knee_angle": 104.9})
        self.assertIsNone(resolve_lead_side(window))

    def test_resolves_just_outside_the_separation_guard(self) -> None:
        from src.pose.movements.lunge import resolve_lead_side

        window = _rule_frames({"min_knee_angle": 100.0, "left_knee_angle": 100.0,
                               "right_knee_angle": 105.1})
        self.assertEqual(resolve_lead_side(window), "left")

    def test_a_gap_of_exactly_the_guard_value_already_resolves(self) -> None:
        # Pins the guard as a strict `<` comparison: a gap of EXACTLY 5.0 degrees is already
        # "separated enough" (only gaps strictly below 5.0 are ambiguous), which distinguishes
        # this from a mutant that flipped the comparison to `<=`.
        from src.pose.movements.lunge import resolve_lead_side

        window = _rule_frames({"min_knee_angle": 100.0, "left_knee_angle": 100.0,
                               "right_knee_angle": 105.0})
        self.assertEqual(resolve_lead_side(window), "left")


class LungeDepthRuleTests(unittest.TestCase):
    def _window(self, lead_angle: float, view: str = "side", phase: str = "bottom", **kwargs):
        from src.pose.movements.lunge import rule_insufficient_depth

        window = _rule_frames({"min_knee_angle": lead_angle, "left_knee_angle": lead_angle,
                               "right_knee_angle": 170.0}, phase=phase)
        return rule_insufficient_depth(window, _ctx(view, **kwargs))

    def test_fires_on_a_shallow_lunge(self) -> None:
        detections = self._window(120.0)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "lunge_insufficient_depth")

    def test_silent_on_a_deep_lunge(self) -> None:
        self.assertEqual(self._window(88.0), [])

    def test_silent_just_inside_the_threshold(self) -> None:
        self.assertEqual(self._window(99.0), [])

    def test_fires_just_outside_the_threshold(self) -> None:
        self.assertEqual(len(self._window(101.0)), 1)

    def test_severity_is_exact_at_the_ramp_midpoint(self) -> None:
        # ramp 100 -> 130, so 115 is exactly half way.
        self.assertAlmostEqual(self._window(115.0)[0].severity, 0.5, places=3)

    def test_severity_saturates_at_the_ramp_end(self) -> None:
        self.assertAlmostEqual(self._window(130.0)[0].severity, 1.0, places=3)

    def test_off_view_is_downgraded_not_silenced(self) -> None:
        detections = self._window(120.0, view="rear")
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].observability, "medium")
        self.assertAlmostEqual(
            detections[0].confidence, round(detections[0].severity * 0.65, 4), places=3
        )

    def test_silent_when_the_lead_side_is_unresolvable(self) -> None:
        from src.pose.movements.lunge import rule_insufficient_depth

        window = _rule_frames({"min_knee_angle": 120.0, "left_knee_angle": 120.0,
                               "right_knee_angle": 121.0})
        self.assertEqual(rule_insufficient_depth(window, _ctx()), [])

    def test_silent_outside_the_active_phases(self) -> None:
        from src.pose.movements.lunge import rule_insufficient_depth

        window = _rule_frames({"min_knee_angle": 120.0, "left_knee_angle": 120.0,
                               "right_knee_angle": 170.0}, phase="setup")
        self.assertEqual(rule_insufficient_depth(window, _ctx()), [])

    def test_still_silent_during_descent_and_ascent(self) -> None:
        # The mask is `phase == "bottom"` ONLY, unlike every other lunge rule -- a mutant that
        # widened it back to `phase in LUNGE_ACTIVE_PHASES` would pass every constant-valued
        # fixture in this class (they all default to phase="bottom") while re-introducing the
        # real defect this task found: firing on the ordinary >100-degree transit every rep
        # makes on the way down/up, even a deep one. This constant-valued check pins the mask
        # shape directly; the trajectory tests below pin the CONSEQUENCE on a real rep.
        detections = self._window(120.0, phase="descent")
        self.assertEqual(detections, [])
        detections = self._window(120.0, phase="ascent")
        self.assertEqual(detections, [])

    def _trajectory(self, bottom_angle: float, per_leg: int = 30, margin: float = 8.0):
        """A real rep: lead knee travels 170 -> `bottom_angle` -> 170 over `2 * per_leg`
        frames. Every frame above the 100-degree fire threshold during descent/ascent is on
        purpose -- this is what a genuine rep's transit looks like, unlike the constant-valued
        fixtures elsewhere in this class. `phase` is assigned the same way `lunge_assign_phases`
        would: the `margin`-degree-wide slice around `bottom_angle` is "bottom", everything
        before it "descent", everything after "ascent". `per_leg=30, margin=8.0` are tuned (not
        arbitrary) so the "bottom" phase run is >= `min_frames=6` for BOTH trajectories used
        below -- verified directly against `rule_insufficient_depth` before this test was
        written, not assumed.
        """
        from src.pose.movements.base import CoreFrame

        angles = list(np.linspace(170.0, bottom_angle, per_leg)) + list(
            np.linspace(bottom_angle, 170.0, per_leg)
        )
        bottom_cutoff = bottom_angle + margin
        frames = []
        past_bottom = False
        for i, angle in enumerate(angles):
            if angle <= bottom_cutoff:
                phase = "bottom"
                past_bottom = True
            elif not past_bottom:
                phase = "descent"
            else:
                phase = "ascent"
            frames.append(
                CoreFrame(
                    frame_index=i,
                    time=i / 30.0,
                    phase=phase,
                    valid=True,
                    lower_body_visibility=0.9,
                    metrics={
                        "min_knee_angle": float(angle),
                        "left_knee_angle": float(angle),
                        "right_knee_angle": 170.0,
                    },
                )
            )
        return frames

    def test_silent_on_a_deep_rep_that_transits_through_the_threshold(self) -> None:
        # THE DEFECT THIS TEST PINS: a rep that bottoms at a clean 85 degrees still spends many
        # frames above the 100-degree threshold on the way down and back up. Before the mask
        # was narrowed to `phase == "bottom"`, this fired twice at severity 1.0 (verified
        # empirically against the pre-fix code: two detections, max_lead_knee_angle_deg=170.0).
        from src.pose.movements.lunge import rule_insufficient_depth

        detections = rule_insufficient_depth(self._trajectory(85.0), _ctx("side"))
        self.assertEqual(detections, [])

    def test_fires_once_on_a_shallow_rep_with_severity_from_the_bottom_not_the_transit(self) -> None:
        from src.pose.movements.lunge import rule_insufficient_depth

        detections = rule_insufficient_depth(self._trajectory(118.0), _ctx("side"))
        self.assertEqual(len(detections), 1)
        # Severity must come from the ~118-degree bottom, not the ~170-degree transit through
        # descent/ascent that a wider mask would have included.
        self.assertLess(detections[0].evidence["max_lead_knee_angle_deg"], 130.0)
        self.assertGreater(detections[0].severity, 0.0)
        self.assertLess(detections[0].severity, 1.0)


class LungeKneePastToesRuleTests(unittest.TestCase):
    def _fire(self, ratio: float, view: str = "side", conf: float = 0.9):
        from src.pose.movements.lunge import rule_knee_past_toes

        window = _rule_frames({"min_knee_angle": 90.0, "left_knee_angle": 90.0,
                               "right_knee_angle": 170.0, "left_knee_forward_ratio": ratio,
                               "right_knee_forward_ratio": 0.0})
        return rule_knee_past_toes(window, _ctx(view, view_confidence=conf))

    def test_fires_when_the_lead_knee_travels_past_the_toes(self) -> None:
        detections = self._fire(0.20)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "lunge_knee_past_toes")
        self.assertEqual(detections[0].evidence["lead_side"], "left")

    def test_silent_when_the_knee_stays_behind_the_toes(self) -> None:
        self.assertEqual(self._fire(0.02), [])

    def test_silent_just_inside_the_threshold(self) -> None:
        self.assertEqual(self._fire(0.09), [])

    def test_fires_just_outside_the_threshold(self) -> None:
        self.assertEqual(len(self._fire(0.11)), 1)

    def test_severity_is_exact_at_the_ramp_midpoint(self) -> None:
        self.assertAlmostEqual(self._fire(0.20)[0].severity, 0.5, places=3)

    def test_hard_gated_to_silence_off_the_sagittal_view(self) -> None:
        # Not a downgrade: sagittal knee travel is not resolvable head-on, and the squat's
        # rule_knees_forward sets the precedent of silence rather than a low-confidence claim.
        self.assertEqual(self._fire(0.20, view="rear_oblique"), [])

    def test_hard_gated_to_silence_on_a_weakly_classified_side_view(self) -> None:
        self.assertEqual(self._fire(0.20, view="side", conf=0.10), [])

    def test_reads_the_lead_legs_ratio_not_the_trailing_legs(self) -> None:
        # Trailing leg way past its toes, lead leg fine -> nothing fires. The whole point of
        # per-window lead resolution.
        from src.pose.movements.lunge import rule_knee_past_toes

        window = _rule_frames({"min_knee_angle": 90.0, "left_knee_angle": 90.0,
                               "right_knee_angle": 170.0, "left_knee_forward_ratio": 0.01,
                               "right_knee_forward_ratio": 0.90})
        self.assertEqual(rule_knee_past_toes(window, _ctx("side")), [])


class LungeKneeValgusRuleTests(unittest.TestCase):
    def _fire(self, offset: float, view: str = "front_oblique"):
        from src.pose.movements.lunge import rule_knee_valgus

        window = _rule_frames({"min_knee_angle": 90.0, "left_knee_angle": 90.0,
                               "right_knee_angle": 170.0,
                               "left_knee_medial_offset_ratio": offset,
                               "right_knee_medial_offset_ratio": 0.0})
        return rule_knee_valgus(window, _ctx(view))

    def test_fires_when_the_lead_knee_caves_medially(self) -> None:
        detections = self._fire(0.18)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "lunge_knee_valgus")

    def test_silent_when_the_knee_tracks_over_the_foot(self) -> None:
        self.assertEqual(self._fire(0.01), [])

    def test_silent_when_the_knee_tracks_laterally(self) -> None:
        # A NEGATIVE offset is the knee bowing outward -- the opposite fault, not this one.
        self.assertEqual(self._fire(-0.30), [])

    def test_silent_just_inside_the_threshold(self) -> None:
        self.assertEqual(self._fire(0.09), [])

    def test_fires_just_outside_the_threshold(self) -> None:
        self.assertEqual(len(self._fire(0.11)), 1)

    def test_severity_is_exact_at_the_ramp_midpoint(self) -> None:
        # ramp 0.10 -> 0.25, midpoint 0.175
        self.assertAlmostEqual(self._fire(0.175)[0].severity, 0.5, places=2)

    def test_off_view_is_downgraded_not_silenced(self) -> None:
        detections = self._fire(0.18, view="side")
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].observability, "medium")

    def test_fires_from_a_rear_view_at_full_observability(self) -> None:
        # `front` is unreachable in production (allow_front=False), so a rule that only rated
        # `front` highly would be permanently downgraded. Medial-vs-midline reads the same
        # from behind, so `rear` earns the same rating -- matching squat.rule_knees_inward.
        self.assertEqual(self._fire(0.18, view="rear")[0].observability, "high")


class LungePelvicDropRuleTests(unittest.TestCase):
    def _fire(self, tilt: float, lead: str = "left", view: str = "front_oblique",
              phase: str = "bottom"):
        from src.pose.movements.lunge import rule_pelvic_drop

        angles = ({"left_knee_angle": 90.0, "right_knee_angle": 170.0} if lead == "left"
                  else {"left_knee_angle": 170.0, "right_knee_angle": 90.0})
        window = _rule_frames(
            {"min_knee_angle": 90.0, "pelvis_tilt_signed_deg": tilt, **angles}, phase=phase
        )
        return rule_pelvic_drop(window, _ctx(view))

    def test_fires_when_the_contralateral_hip_drops_on_a_left_lead(self) -> None:
        # Left lead -> contralateral is the RIGHT hip -> positive tilt is the fault.
        detections = self._fire(14.0, lead="left")
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].fault_id, "lunge_pelvic_drop")

    def test_fires_when_the_contralateral_hip_drops_on_a_right_lead(self) -> None:
        # Right lead -> contralateral is the LEFT hip -> NEGATIVE tilt is the fault. Checks the
        # REPORTED magnitude and severity, not just the count: a mutant that drops the `sign *`
        # from the `drops` list (while leaving it in the mask) would still fire once here but
        # report -14.0 and clamp severity to 0.0 -- `len == 1` alone would miss that.
        detections = self._fire(-14.0, lead="right")
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].evidence["lead_side"], "right")
        self.assertAlmostEqual(detections[0].evidence["max_contralateral_drop_deg"], 14.0, places=2)
        self.assertAlmostEqual(detections[0].severity, 0.5, places=3)

    def test_silent_when_the_LEAD_side_hip_drops(self) -> None:
        # This is the sign error the rule exists to avoid: an ipsilateral drop is not
        # Trendelenburg, and reporting it would invert the coaching cue.
        self.assertEqual(self._fire(-14.0, lead="left"), [])

    def test_silent_on_a_level_pelvis(self) -> None:
        self.assertEqual(self._fire(1.0), [])

    def test_silent_just_inside_the_threshold(self) -> None:
        self.assertEqual(self._fire(7.0), [])

    def test_fires_just_outside_the_threshold(self) -> None:
        self.assertEqual(len(self._fire(9.0)), 1)

    def test_severity_is_exact_at_the_ramp_midpoint(self) -> None:
        # ramp 8 -> 20, midpoint 14
        self.assertAlmostEqual(self._fire(14.0)[0].severity, 0.5, places=3)

    def test_silent_from_a_pure_side_view(self) -> None:
        # The parent spec: "not observable from a pure side view". A frontal-plane tilt has no
        # meaning in the sagittal projection, so this is silence, not a downgrade.
        self.assertEqual(self._fire(14.0, view="side"), [])

    def test_silent_when_the_drop_is_confined_to_descent(self) -> None:
        # OVERRIDE 1 (spec vs. brief): the spec scopes this fault to "sustained through
        # bottom/ascent" only, unlike LUNGE_ACTIVE_PHASES (which includes descent and is what
        # every other lunge rule uses). A drop that only ever occurs during descent must not
        # fire -- pins PELVIC_DROP_PHASES as its own set, not an alias of LUNGE_ACTIVE_PHASES.
        self.assertEqual(self._fire(14.0, phase="descent"), [])

    def test_fires_when_the_drop_is_sustained_through_ascent(self) -> None:
        # `ascent` is the other half of the spec's "bottom/ascent" scope. Every other fixture in
        # this class defaults to phase="bottom", so without this test a mutant that narrowed
        # PELVIC_DROP_PHASES to {"bottom"} alone would pass the whole suite.
        self.assertEqual(len(self._fire(14.0, phase="ascent")), 1)

    def test_silent_when_the_lead_side_is_unresolvable(self) -> None:
        # Deleting the `lead is None` guard would not crash -- `None == "left"` is False, so
        # `sign` would silently become -1.0 and the rule would emit a garbage-attributed
        # detection instead of staying silent. Mirrors LungeDepthRuleTests' equivalent test.
        from src.pose.movements.lunge import rule_pelvic_drop

        window = _rule_frames({"min_knee_angle": 90.0, "pelvis_tilt_signed_deg": 14.0,
                               "left_knee_angle": 90.0, "right_knee_angle": 91.0})
        self.assertEqual(rule_pelvic_drop(window, _ctx()), [])

    def test_medium_observability_on_front(self) -> None:
        # OVERRIDE 2 (spec vs. brief): the spec rates this fault `medium` on front/rear -- its
        # TOP tier, never `high`.
        detections = self._fire(14.0, view="front")
        self.assertEqual(detections[0].observability, "medium")
        self.assertAlmostEqual(detections[0].confidence, detections[0].severity, places=3)

    def test_medium_observability_on_rear(self) -> None:
        detections = self._fire(14.0, view="rear")
        self.assertEqual(detections[0].observability, "medium")

    def test_low_observability_off_view(self) -> None:
        # OVERRIDE 2: off the alignment-observable views (and not `side`, which is silenced
        # separately) the rule downgrades to `low` -- a RULE-LEVEL rating, not a number the
        # spec names for this case -- which also demotes the detection behind every other one
        # via `run_detector`'s sort key `(observability == "low", -severity, start_frame)`.
        detections = self._fire(14.0, view="unknown")
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].observability, "low")
        self.assertAlmostEqual(
            detections[0].confidence, round(detections[0].severity * 0.65, 4), places=3
        )


class LungeAlternatingLeadTests(unittest.TestCase):
    """The regression guard on lead-side resolution living in the RULES, not in compute_raw.

    The Phase 2 validation harness feeds ONE REP PER CLIP, so a clip-level lead side is
    per-rep by construction there and this defect would be invisible to it. Lunges are
    normally performed alternating legs, so production sees exactly this shape.
    """

    def _alternating_clip(self) -> list[dict]:
        # Three reps: lead left, lead right, lead left. Depth is driven by `lead_anterior`,
        # which is what bends the lead knee in-image.
        frames: list[dict] = []
        index = 0
        for lead in ("left", "right", "left"):
            depths = list(np.linspace(0.0, 0.80, 12)) + list(np.linspace(0.80, 0.0, 12))
            for depth in depths:
                frames.append(
                    lunge_frame(lead=lead, lead_anterior=float(depth), frame_index=index)
                )
                index += 1
        return frames

    def test_each_rep_attributes_its_fault_to_the_leg_that_actually_led_it(self) -> None:
        from src.pose.movements.lunge import LUNGE_DETECTOR

        result = run_detector(LUNGE_DETECTOR, self._alternating_clip(), 30.0, "front_oblique", 0.9)
        self.assertGreaterEqual(len(result.reps), 2, "segmentation did not find the reps")
        # Whatever fires, no detection may name a lead side whose knee was the EXTENDED one:
        # that is the signature of a lead side resolved over the wrong window.
        for detection in result.detections:
            lead = detection.evidence.get("lead_side")
            if lead is None:
                continue
            peak = next(f for f in result.core if f.frame_index == detection.peak_frame)
            self.assertLess(
                peak.m(f"{lead}_knee_angle"),
                peak.m("left_knee_angle" if lead == "right" else "right_knee_angle"),
                f"detection {detection.fault_id} named {lead} as the lead leg, but that leg "
                f"was the more EXTENDED one at its peak frame",
            )


if __name__ == "__main__":
    unittest.main()
