"""Pure helpers of the High Knee validation harness.

The harness's INPUT (EgoExo pose JSON) is not in the repository, so what is testable here is the
arithmetic every reported number passes through -- and, more importantly, the two constructions
the withdrawals rest on: that the trunk quantity is genuinely SIGNED, and that the reference axis
it is measured against has its own error measured reference-free.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from src.egoexo.high_knee_validation import (
    WITHDRAWN_BACK_LEAN_CUT_DEG,
    classify_knee_lift_comment,
    criterion_failure_rates,
    WITHDRAWN_FORWARD_LEAN_CUT_DEG,
    cadence_hz,
    cross_camera_spread,
    fire_rate,
    floor_discarded,
    load_judgements,
    pearson,
    pelvic_obliquity_deg,
    signed_trunk_lean_deg,
    trunk_to_support_limb_deg,
)
from src.pose.geometry import landmarks_to_array
from tests.test_high_knee import high_knee_frame


# A SAGITTAL fixture. Every quantity in this module is a fore-aft one, and a 2-D frame has only
# one image axis left over once the trunk has taken the other -- so it is either the mediolateral
# axis (frontal view) or the anterior one (sagittal view), never both. These rules need the
# sagittal view, so the pelvis is collapsed to near-zero projected width, which is what a side
# camera actually shows. `SupportLimbGeometryTest` is about what happens when it is not.
SAGITTAL = {"hip_half_width": 0.002}


def _points(**kwargs):
    return landmarks_to_array(high_knee_frame(**{**SAGITTAL, **kwargs})["landmarks"])


class SignedTrunkLeanTest(unittest.TestCase):
    """THE FIXTURE IS RAMPED BOTH WAYS ON PURPOSE.

    An UNSIGNED deviation from a baseline is actively inverted -- it reports the opposite fault at
    full severity -- and this project has now shipped that bug twice (`pushup_head_drop`, then
    Torso Twist's brace rule). Both times the reason no green test caught it was that every
    fixture ramped the SAME way. So every assertion here comes in a matched pair.
    """

    def test_leaning_forward_and_backward_have_opposite_signs(self) -> None:
        forward = signed_trunk_lean_deg(_points(trunk_lean_deg=20.0))
        backward = signed_trunk_lean_deg(_points(trunk_lean_deg=-20.0))
        self.assertGreater(forward, 0.0)
        self.assertLess(backward, 0.0)
        self.assertAlmostEqual(forward, -backward, places=3)

    def test_an_upright_trunk_reads_about_zero(self) -> None:
        # NOT exactly zero, and the residual is the point of `SupportLimbGeometryTest`: even in
        # this fixture the stance ankle sits under the hip JOINT rather than under the pelvis
        # midpoint, so the reference axis is already tilted by atan(hip_half_width / leg_length).
        self.assertAlmostEqual(signed_trunk_lean_deg(_points(trunk_lean_deg=0.0)), 0.0, delta=0.5)

    def test_the_magnitude_tracks_the_knob(self) -> None:
        for angle in (5.0, 12.0, 25.0):
            with self.subTest(angle=angle):
                self.assertAlmostEqual(
                    signed_trunk_lean_deg(_points(trunk_lean_deg=angle)), angle, delta=0.5
                )

    def test_the_two_withdrawn_cuts_would_have_fired_on_opposite_inputs(self) -> None:
        """The property that makes them TWO rules rather than one, and the property an unsigned
        construction destroys."""
        leaning_back = signed_trunk_lean_deg(_points(trunk_lean_deg=-18.0))
        leaning_forward = signed_trunk_lean_deg(_points(trunk_lean_deg=18.0))
        self.assertLess(leaning_back, -WITHDRAWN_BACK_LEAN_CUT_DEG)
        self.assertGreater(leaning_back, -180.0)
        self.assertGreater(leaning_forward, WITHDRAWN_FORWARD_LEAN_CUT_DEG)
        # And neither input fires the other rule.
        self.assertFalse(leaning_back > WITHDRAWN_FORWARD_LEAN_CUT_DEG)
        self.assertFalse(leaning_forward < -WITHDRAWN_BACK_LEAN_CUT_DEG)

    def test_it_is_roll_invariant_so_the_rolled_corpus_can_be_measured_at_all(self) -> None:
        for roll in (37.0, 90.0, -120.0):
            with self.subTest(roll=roll):
                self.assertAlmostEqual(
                    signed_trunk_lean_deg(_points(trunk_lean_deg=14.0, roll_deg=roll)),
                    14.0,
                    delta=0.5,
                )

    def test_it_survives_mirroring_with_its_sign_intact(self) -> None:
        """The sign comes from the SUBJECT's feet, not from image x, so filming the same lean from
        the other side must not turn a backward lean into a forward one. A cross-product
        construction would fail exactly here -- Shoulder Bridge's two attempts did."""
        plain = signed_trunk_lean_deg(_points(trunk_lean_deg=-16.0))
        mirrored = signed_trunk_lean_deg(_points(trunk_lean_deg=-16.0, mirrored=True))
        self.assertLess(plain, 0.0)
        self.assertLess(mirrored, 0.0)
        self.assertAlmostEqual(plain, mirrored, places=2)


class SupportLimbGeometryTest(unittest.TestCase):
    """THE SUPPORT LIMB IS NOT A VERTICAL, AND PART OF THE REASON IS PURE ANATOMY.

    The stance foot sits under the hip JOINT; the reference axis is drawn from the PELVIS
    MIDPOINT. Those differ by half a pelvis width, so the axis is tilted by
    atan(half_pelvis / leg_length) before the subject has moved at all -- about 6 degrees on adult
    proportions (0.09 m over 0.85 m). Nothing a performer does removes it.

    That is the floor under the 6.4-14.2 degrees measured between trunk and support limb on the
    real corpus, against trunk-lean thresholds of 10-15 degrees. Design spec section 7.1.
    """

    def test_a_stance_foot_under_the_hip_joint_tilts_the_axis_before_anyone_moves(self) -> None:
        upright_frontal = trunk_to_support_limb_deg(
            landmarks_to_array(high_knee_frame()["landmarks"])
        )
        expected = math.degrees(math.atan2(0.05, 0.30))  # fixture pelvis half-width / leg length
        self.assertAlmostEqual(upright_frontal, expected, delta=0.2)
        self.assertGreater(upright_frontal, 9.0)

    def test_and_it_shrinks_as_the_projected_pelvis_narrows(self) -> None:
        wide = trunk_to_support_limb_deg(
            landmarks_to_array(high_knee_frame(hip_half_width=0.05)["landmarks"])
        )
        narrow = trunk_to_support_limb_deg(
            landmarks_to_array(high_knee_frame(hip_half_width=0.002)["landmarks"])
        )
        self.assertGreater(wide, narrow)
        self.assertLess(narrow, 1.0)


class ReferenceAxisTest(unittest.TestCase):
    def test_a_vertical_support_limb_gives_a_zero_axis_error(self) -> None:
        self.assertAlmostEqual(trunk_to_support_limb_deg(_points()), 0.0, delta=0.5)

    def test_a_leaning_trunk_shows_up_as_axis_error_because_the_two_are_indistinguishable(self) -> None:
        """THE WHOLE WITHDRAWAL IN ONE ASSERTION. This quantity cannot tell whether the trunk moved
        or the support limb did -- it only reports that they differ. That is precisely why a trunk
        lean measured against the support limb attributes an unknown share of the limb's own
        inclination to the trunk."""
        self.assertAlmostEqual(
            trunk_to_support_limb_deg(_points(trunk_lean_deg=11.0)), 11.0, delta=0.5
        )
        self.assertAlmostEqual(
            trunk_to_support_limb_deg(_points(trunk_lean_deg=-11.0)), 11.0, delta=0.5
        )

    def test_it_needs_no_vertical_and_so_is_roll_invariant(self) -> None:
        for roll in (23.0, 90.0, 180.0):
            with self.subTest(roll=roll):
                self.assertAlmostEqual(
                    trunk_to_support_limb_deg(_points(trunk_lean_deg=9.0, roll_deg=roll)),
                    9.0,
                    delta=0.5,
                )


class PelvicObliquityTest(unittest.TestCase):
    def test_a_level_pelvis_reads_about_zero(self) -> None:
        self.assertAlmostEqual(pelvic_obliquity_deg(_points()), 0.0, places=3)

    def test_it_is_roll_invariant(self) -> None:
        base = pelvic_obliquity_deg(_points())
        for roll in (41.0, 90.0):
            with self.subTest(roll=roll):
                self.assertAlmostEqual(pelvic_obliquity_deg(_points(roll_deg=roll)), base, 3)

    def test_tilting_the_hip_line_is_signed(self) -> None:
        frame = high_knee_frame()
        landmarks = frame["landmarks"]
        landmarks[23]["y"] -= 0.03  # left hip up the image
        up = pelvic_obliquity_deg(landmarks_to_array(landmarks))
        landmarks[23]["y"] += 0.06  # and now down
        down = pelvic_obliquity_deg(landmarks_to_array(landmarks))
        self.assertGreater(abs(up - down), 1.0)
        self.assertNotAlmostEqual(math.copysign(1.0, up), math.copysign(1.0, down))


class ArithmeticTest(unittest.TestCase):
    def test_cadence_uses_the_rep_span_and_refuses_to_divide_by_nothing(self) -> None:
        self.assertAlmostEqual(cadence_hz(10, 150, 30.0), 2.0, places=6)
        for bad in (cadence_hz(0, 150, 30.0), cadence_hz(10, 0, 30.0), cadence_hz(10, 150, 0.0)):
            self.assertTrue(math.isnan(bad))

    def test_floor_discarded_is_the_difference_and_never_negative(self) -> None:
        self.assertEqual(floor_discarded(150, 52), 98)
        self.assertEqual(floor_discarded(52, 150), 0)

    def test_fire_rate_ignores_nan_and_respects_direction(self) -> None:
        values = [-0.9, -0.5, float("nan"), 0.2]
        self.assertAlmostEqual(fire_rate(values, 0.0), 2 / 3, places=6)
        self.assertAlmostEqual(fire_rate(values, 0.0, below=False), 1 / 3, places=6)
        self.assertTrue(math.isnan(fire_rate([float("nan")], 0.0)))

    def test_pearson_needs_ten_usable_pairs_and_handles_flat_series(self) -> None:
        self.assertTrue(math.isnan(pearson([1.0] * 5, [2.0] * 5)))
        self.assertTrue(math.isnan(pearson([1.0] * 20, list(range(20)))))
        rising = [float(i) for i in range(20)]
        self.assertAlmostEqual(pearson(rising, rising), 1.0, places=6)
        self.assertAlmostEqual(pearson(rising, [-v for v in rising]), -1.0, places=6)

    def test_cross_camera_spread_needs_two_usable_views(self) -> None:
        per_view = {
            "exo_l": {"obliquity_median_deg": 2.0},
            "exo_m": {"obliquity_median_deg": 0.1},
            "exo_r": {"obliquity_median_deg": -7.0},
        }
        self.assertAlmostEqual(
            cross_camera_spread(per_view, "obliquity_median_deg", ("exo_l", "exo_r")), 9.0, 6
        )
        self.assertTrue(
            math.isnan(cross_camera_spread(per_view, "obliquity_median_deg", ("exo_l",)))
        )

    def test_cross_camera_spread_excludes_the_view_the_gate_rejects(self) -> None:
        """Pooling a camera its own gate says cannot see the quantity is not a second opinion --
        and here it would UNDERSTATE the spread, i.e. bias the withdrawal's evidence toward
        keeping the rule."""
        per_view = {
            "exo_l": {"obliquity_median_deg": 2.0},
            "exo_m": {"obliquity_median_deg": 0.1},
            "exo_r": {"obliquity_median_deg": -7.0},
        }
        gated = cross_camera_spread(per_view, "obliquity_median_deg", ("exo_l", "exo_r"))
        pooled = cross_camera_spread(per_view, "obliquity_median_deg", ("exo_l", "exo_m", "exo_r"))
        self.assertAlmostEqual(gated, pooled, places=6)  # here they agree...
        per_view["exo_m"] = {"obliquity_median_deg": 40.0}  # ...and here the degenerate view lies
        self.assertLess(
            cross_camera_spread(per_view, "obliquity_median_deg", ("exo_l", "exo_r")),
            cross_camera_spread(per_view, "obliquity_median_deg", ("exo_l", "exo_m", "exo_r")),
        )


class JudgementLoadingTest(unittest.TestCase):
    def test_it_keeps_only_high_knee_and_applies_a_strict_majority(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        payload = {
            "a_action_1": {
                "annotations": [
                    {"action_name": "High Knee",
                     "key_point_verification": [["c", "True"], ["d", "False"]]},
                    {"action_name": "High Knee",
                     "key_point_verification": [["c", "False"], ["d", "False"]]},
                ]
            },
            "b_action_2": {
                "annotations": [
                    {"action_name": "Sit-ups", "key_point_verification": [["c", "False"]]}
                ]
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "j.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_judgements(path)
        self.assertEqual(set(loaded), {"a_action_1"})
        # 1 of 2 False is NOT a strict majority; 2 of 2 is.
        self.assertFalse(loaded["a_action_1"]["c"])
        self.assertTrue(loaded["a_action_1"]["d"])


if __name__ == "__main__":
    unittest.main()


class SupportLimbSelectionTest(unittest.TestCase):
    """THE SUPPORT LIMB MUST BE THE GROUNDED LEG, AND THE FIXTURES MUST STAND ON ONE.

    Every other fixture in this file stands on BOTH feet, where the two ankles sit at the same
    height and `max` resolves the tie regardless of the comparator's sign -- so an inverted
    selection survives them all. (Verified: inverting `_support_ankle` left the rest of this file
    green.) A drill whose whole point is that one knee is driven has to be tested with one knee
    driven.
    """

    def _lean_with(self, driven: str, lean_deg: float) -> float:
        kwargs = {"left_elevation": -0.45, "right_elevation": -1.0} if driven == "left" else {
            "left_elevation": -1.0, "right_elevation": -0.45
        }
        return signed_trunk_lean_deg(_points(trunk_lean_deg=lean_deg, **kwargs))

    def test_the_grounded_leg_is_chosen_whichever_side_it_is_on(self) -> None:
        """If the AIRBORNE leg were picked, the reference axis would run to a raised ankle and the
        reported lean would differ between the two mirror-image postures. It must not."""
        driving_left = self._lean_with("left", 12.0)
        driving_right = self._lean_with("right", 12.0)
        self.assertAlmostEqual(driving_left, 12.0, delta=1.0)
        self.assertAlmostEqual(driving_right, 12.0, delta=1.0)
        self.assertAlmostEqual(driving_left, driving_right, delta=1.0)

    def test_and_the_sign_survives_the_side_switch(self) -> None:
        self.assertLess(self._lean_with("left", -14.0), 0.0)
        self.assertLess(self._lean_with("right", -14.0), 0.0)

    def test_the_axis_error_is_not_inflated_by_the_driven_leg(self) -> None:
        """`trunk_to_support_limb_deg` must read the STANCE limb too -- otherwise section 7.1's
        6.4-23.6 degree measurement would be reporting hip flexion, not limb inclination."""
        upright_one_leg = trunk_to_support_limb_deg(
            _points(left_elevation=-0.45, right_elevation=-1.0)
        )
        upright_two_legs = trunk_to_support_limb_deg(_points())
        self.assertAlmostEqual(upright_one_leg, upright_two_legs, delta=0.5)


class CriterionAggregationTest(unittest.TestCase):
    def test_failure_rates_are_per_action_and_sorted_worst_first(self) -> None:
        judgements = {
            "a": {"speed": True, "back": False},
            "b": {"speed": True, "back": False},
            "c": {"speed": False, "back": False},
        }
        self.assertEqual(
            criterion_failure_rates(judgements),
            [("speed", 2, 3), ("back", 0, 3)],
        )

    def test_it_survives_a_criterion_missing_from_one_action(self) -> None:
        rows = {name: (failed, total)
                for name, failed, total in
                criterion_failure_rates({"a": {"x": True}, "b": {"y": False}})}
        self.assertEqual(rows["x"], (1, 1))
        self.assertEqual(rows["y"], (0, 1))


class CommentClassifierTest(unittest.TestCase):
    """The PRE-REGISTERED rule. These cases are the ones the rule was fixed to discriminate."""

    def test_a_leg_height_complaint_is_positive(self) -> None:
        for comment in (
            "The leg raising range is too small, should be lifted higher.",
            "it is suggested to raise the legs a bit higher",
            "insufficient height in lifting the legs, and relatively slow",
        ):
            with self.subTest(comment=comment):
                self.assertEqual(classify_knee_lift_comment(comment), "positive")

    def test_a_bare_range_complaint_is_UNATTRIBUTABLE_not_positive(self) -> None:
        """The load-bearing case: the same phrase is used about the ARM SWING elsewhere in this
        corpus, so counting it as a leg complaint would manufacture the positive class."""
        self.assertEqual(
            classify_knee_lift_comment("The speed is too slow, range of motion is too small."),
            "unattributable",
        )

    def test_an_arm_complaint_is_not_a_leg_complaint(self) -> None:
        self.assertEqual(
            classify_knee_lift_comment("the amplitude of arm swings was not sufficient"),
            "negative",
        )

    def test_praise_and_absence_are_handled(self) -> None:
        self.assertEqual(classify_knee_lift_comment("The movement was executed very smoothly."),
                         "negative")
        self.assertEqual(classify_knee_lift_comment(None), "missing")
        self.assertEqual(classify_knee_lift_comment(""), "missing")

    def test_the_leg_and_insufficiency_tokens_must_share_a_SENTENCE(self) -> None:
        """Split on sentence punctuation, NOT on commas -- these are translated comments whose
        clauses run on commas, and a comma split would fragment the evidence."""
        self.assertEqual(
            classify_knee_lift_comment("The legs are fine. The arms are not high enough."),
            "negative",
        )
        self.assertEqual(
            classify_knee_lift_comment("the legs are fine, but they should be higher"),
            "positive",
        )
