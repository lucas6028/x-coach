"""Unit tests for the PURE half of the Leg Abduction validation harness.

The orchestration half reads a gitignored pose corpus off disk and is exercised only when that
corpus is present; everything tested here takes frames, records and numbers.
"""

import math
import unittest

from src.pose.movements.leg_abduction import ACTIVE_PHASES, LEG_ABDUCTION_DETECTOR
from src.rehab24 import leg_abduction_rule_validation as harness


class _Frame:
    """The minimum of `CoreFrame` that `window_scores` touches."""

    def __init__(self, phase: str, valid: bool = True, **metrics: float) -> None:
        self.phase = phase
        self.valid = valid
        self._metrics = metrics

    def m(self, key: str) -> float:
        return float(self._metrics.get(key, math.nan))


class _Segment:
    def __init__(self, correctness: int = 1, subtype: str = "left leg",
                 orientation: str = "front", person: str = "1") -> None:
        self.video_id = "PM_005"
        self.repetition_number = "1"
        self.exercise_id = "4"
        self.person_id = person
        self.first_frame = 0
        self.last_frame = 10
        self.cam17_orientation = orientation
        self.mocap_erroneous = "0"
        self.exercise_subtype = subtype
        self.lights_on = "0"
        self.extra_person_in_cam17 = "0"
        self.extra_person_in_cam18 = "0"
        self.correctness = correctness


class CameraRoutingTests(unittest.TestCase):
    def test_the_frontal_rule_reads_cam17(self) -> None:
        """A trunk lean is a frontal-plane cue and `front` in cam17 means `side` in cam18."""
        self.assertEqual(harness.RULE_CAMERAS["abd_pelvic_drop_trunk_lean"], "cam17")

    def test_every_rule_that_can_fire_has_a_camera(self) -> None:
        """A rule added to the detector without a camera entry would be silently unmeasured.

        The expected set is DERIVED by running every registered rule on a fixture built to trip
        all of them, rather than restated as a literal -- a literal would make this test
        `X == X` and it would pass unchanged after someone registered a fifth rule.
        """
        from tests.test_leg_abduction import _core, _ctx, abduction_frame

        # Legs far enough apart for the side to resolve, trunk far past the lean cut, and peak
        # abduction far below the silenced ROM rule's cut: anything that CAN fire, does.
        frames = [
            abduction_frame(abduction_deg=12.0, trunk_tilt_deg=30.0, frame_index=i)
            for i in range(20)
        ]
        core = _core(frames)
        emitted = {
            detection.fault_id
            for rule in LEG_ABDUCTION_DETECTOR.rules
            for detection in rule(core, _ctx())
        }
        self.assertTrue(emitted, "fixture must fire something for this test to mean anything")
        self.assertEqual(set(harness.RULE_CAMERAS), emitted)


class OracleViewsTests(unittest.TestCase):
    def test_front_maps_to_a_label_production_can_never_emit(self) -> None:
        """That is the point of the oracle pass, not a bug: `estimate_view_for_pose` is called
        with `allow_front=False` in production, so asking "would this fire if the view label
        were right?" requires bypassing the gate."""
        self.assertEqual(harness.ORACLE_VIEWS["front"], "front")

    def test_half_profile_maps_to_front_oblique(self) -> None:
        self.assertEqual(harness.ORACLE_VIEWS["half-profile"], "front_oblique")

    def test_oracle_view_confidence_is_pinned_at_one(self) -> None:
        self.assertEqual(harness.ORACLE_VIEW_CONFIDENCE, 1.0)

    def test_every_orientation_the_dataset_can_record_is_mapped(self) -> None:
        self.assertEqual(set(harness.ORACLE_VIEWS), {"front", "half-profile", "side", "profile"})


class SubtypeMappingTests(unittest.TestCase):
    def test_the_leg_word_maps_to_the_anatomical_side(self) -> None:
        """MediaPipe's landmark names are anatomical, so no camera-facing correction applies."""
        self.assertEqual(harness.SUBTYPE_MOVING_SIDE,
                         {"left leg": "left", "right leg": "right"})

    def test_an_unmapped_subtype_stops_the_run(self) -> None:
        with self.assertRaises(SystemExit):
            harness.assert_dataset_shape([_Segment(subtype="both legs")])


class DatasetShapeTests(unittest.TestCase):
    def test_a_short_load_stops_the_run_rather_than_reporting_on_it(self) -> None:
        with self.assertRaises(SystemExit):
            harness.assert_dataset_shape([_Segment() for _ in range(5)])

    def test_the_pinned_counts_are_the_ones_the_writeup_quotes(self) -> None:
        self.assertEqual(harness.EX4_EXPECTED["reps"], 210)
        self.assertEqual(harness.EX4_EXPECTED["correct"], 120)
        self.assertEqual(harness.EX4_EXPECTED["incorrect"], 90)
        self.assertEqual(harness.EX4_EXPECTED["subjects"], 9)

    def test_correctness_one_means_performed_correctly(self) -> None:
        """Inverting this would swap sensitivity and specificity everywhere, silently."""
        self.assertTrue(harness.is_correct(_Segment(correctness=1)))
        self.assertFalse(harness.is_correct(_Segment(correctness=0)))


class MovingSideAccuracyTests(unittest.TestCase):
    def test_refusals_are_counted_apart_from_errors(self) -> None:
        result = harness.moving_side_accuracy([
            {"expected_side": "left", "resolved_side": "left", "fallback": None},
            {"expected_side": "left", "resolved_side": "left", "fallback": None},
            {"expected_side": "right", "resolved_side": "left", "fallback": None},
            {"expected_side": "right", "resolved_side": None, "fallback": None},
        ])
        self.assertEqual(
            (result["correct"], result["wrong"], result["refused_ambiguous"]), (2, 1, 1)
        )
        # A refusal costs COVERAGE, not accuracy: it silences the rule rather than misplacing it.
        self.assertAlmostEqual(result["accuracy_when_resolved"], 2 / 3)
        self.assertAlmostEqual(result["coverage"], 3 / 4)

    def test_a_fallback_rep_is_not_blamed_on_the_resolver(self) -> None:
        """`segment_reps` finding nothing hands the resolver an empty window. Counting that as a
        refusal would attribute a segmentation outcome to a different component."""
        result = harness.moving_side_accuracy([
            {"expected_side": "left", "resolved_side": "left", "fallback": None},
            {"expected_side": "right", "resolved_side": None, "fallback": "no_reps_detected"},
            {"expected_side": "right", "resolved_side": None, "fallback": "only_partial_reps"},
        ])
        self.assertEqual(result["refused_no_window"], 2)
        self.assertEqual(result["refused_ambiguous"], 0)
        self.assertEqual(result["reached_the_resolver"], 1)
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["total"], 3)

    def test_no_records_yields_nan_rather_than_a_flattering_one(self) -> None:
        result = harness.moving_side_accuracy([])
        self.assertTrue(math.isnan(result["accuracy_when_resolved"]))


class FireRateTests(unittest.TestCase):
    def test_a_high_cut_fires_on_the_reps_above_it(self) -> None:
        result = harness.fire_rates([1.0, 9.0, 2.0, 8.0], [True, True, False, False], 5.0,
                                    fires_below=False)
        self.assertEqual((result["correct_fired"], result["correct_n"]), (1, 2))
        self.assertEqual((result["incorrect_fired"], result["incorrect_n"]), (1, 2))

    def test_fires_below_inverts_the_comparison(self) -> None:
        result = harness.fire_rates([1.0, 9.0], [True, False], 5.0, fires_below=True)
        self.assertEqual(result["correct_fired"], 1)
        self.assertEqual(result["incorrect_fired"], 0)

    def test_non_finite_scores_are_dropped_rather_than_counted_as_silent(self) -> None:
        result = harness.fire_rates([math.nan, 9.0], [True, True], 5.0, fires_below=False)
        self.assertEqual(result["correct_n"], 1)


class WindowScoresTests(unittest.TestCase):
    def _window(self) -> list[_Frame]:
        return [
            _Frame("setup", left_trunk_tilt_deg=40.0, left_abduction_deg=0.0,
                   left_pelvic_hike_ratio=0.9),
            _Frame(ACTIVE_PHASES[0], left_trunk_tilt_deg=5.0, left_abduction_deg=20.0,
                   left_pelvic_hike_ratio=0.1),
            _Frame(ACTIVE_PHASES[1], left_trunk_tilt_deg=11.0, left_abduction_deg=42.0,
                   left_pelvic_hike_ratio=0.4),
        ]

    def test_it_scopes_to_the_active_phases_like_the_rules_do(self) -> None:
        """The `setup` frame carries the largest value of every metric; none may leak through."""
        scores = harness.window_scores(self._window(), "left")
        self.assertAlmostEqual(scores["trunk_tilt_deg"], 11.0)
        self.assertAlmostEqual(scores["abduction_deg"], 42.0)
        self.assertAlmostEqual(scores["pelvic_hike_ratio"], 0.4)

    def test_an_unresolved_side_yields_nan_rather_than_a_default_leg(self) -> None:
        scores = harness.window_scores(self._window(), None)
        self.assertTrue(all(math.isnan(v) for v in scores.values()))

    def test_invalid_frames_do_not_contribute(self) -> None:
        window = [_Frame(ACTIVE_PHASES[1], valid=False, left_trunk_tilt_deg=99.0)]
        self.assertTrue(math.isnan(harness.window_scores(window, "left")["trunk_tilt_deg"]))


class ViewConfusionTests(unittest.TestCase):
    def test_it_counts_emitted_labels_per_recorded_orientation(self) -> None:
        table = harness.view_label_confusion([
            {"cam17_orientation": "front", "view_type": "rear"},
            {"cam17_orientation": "front", "view_type": "rear"},
            {"cam17_orientation": "half-profile", "view_type": "rear_oblique"},
        ])
        self.assertEqual(table, {"front": {"rear": 2}, "half-profile": {"rear_oblique": 1}})


class SpearmanTests(unittest.TestCase):
    def test_a_monotone_pair_correlates_at_one(self) -> None:
        self.assertAlmostEqual(harness.spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]),
                               1.0)

    def test_an_inverted_pair_correlates_at_minus_one(self) -> None:
        self.assertAlmostEqual(harness.spearman([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]),
                               -1.0)

    def test_ties_share_a_rank_rather_than_ordering_arbitrarily(self) -> None:
        self.assertAlmostEqual(harness.spearman([1.0, 1.0, 2.0, 2.0], [5.0, 5.0, 9.0, 9.0]), 1.0)

    def test_too_few_finite_pairs_yields_nan_not_a_number(self) -> None:
        self.assertTrue(math.isnan(harness.spearman([1.0, math.nan], [1.0, 2.0])))


class SplitByOrientationTests(unittest.TestCase):
    def test_it_tables_the_rule_separately_per_recorded_orientation(self) -> None:
        records = [
            {"cam17_orientation": "front", "production_fired": {"abd_pelvic_drop_trunk_lean": 1.0},
             "correct": False},
            {"cam17_orientation": "front", "production_fired": {}, "correct": True},
            {"cam17_orientation": "half-profile",
             "production_fired": {"abd_pelvic_drop_trunk_lean": 1.0}, "correct": True},
        ]
        table = harness.split_by_orientation(records, "abd_pelvic_drop_trunk_lean")
        self.assertEqual(table["front"], {"tp": 1, "fp": 0, "fn": 0, "tn": 1})
        self.assertEqual(table["half-profile"], {"tp": 0, "fp": 1, "fn": 0, "tn": 0})


class ReportTests(unittest.TestCase):
    def test_the_report_renders_from_an_empty_run_without_raising(self) -> None:
        """A run against a missing corpus must produce a readable report, not a traceback."""
        summary = harness.summarize({"records": [], "skipped": [{"video_id": "PM_005",
                                                                 "reason": "missing"}]})
        text = harness.render_report(summary)
        self.assertIn("Leg Abduction rule validation", text)
        self.assertIn("SKIPPED", text)

    def test_the_silenced_rom_signal_is_scored_even_though_no_rule_fires(self) -> None:
        """The reason `rule_insufficient_abduction_rom` is silent is a number this harness
        produces, so the signal must be in the report whether or not a rule reads it."""
        summary = harness.summarize({"records": [], "skipped": []})
        self.assertIn("abduction_deg", summary["signals"])
        self.assertIn("pelvic_hike_ratio", summary["signals"])


if __name__ == "__main__":
    unittest.main()
