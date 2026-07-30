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
    """The same body FACING THE OTHER WAY: left/right landmark CONTENTS swapped, then the
    whole thing reflected about x=0.5.

    Reflecting x alone is not a facing flip -- it moves landmark 23 to the right-hand side of
    the image while leaving it the "left hip", and since `pelvis_tilt_signed_deg` reads
    `right_hip[1] - left_hip[1]` and reflection does not touch y, such a test passes trivially
    without exercising anything. Swapping the contents of every left/right pair is what a real
    turn-around does.
    """
    lm = [dict(item) for item in frame["landmarks"]]
    for left, right in ((11, 12), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32), (7, 8)):
        lm[left], lm[right] = lm[right], lm[left]
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
        # A real turn-around (see `mirrored`): left/right landmark CONTENTS swap and the image
        # reflects. Which hip is physically lower is unchanged, so the metric must not flip.
        # A tilt built with the RIGHT hip lower stays positive after the subject turns around.
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


if __name__ == "__main__":
    unittest.main()
