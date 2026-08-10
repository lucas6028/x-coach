"""Pure helpers of the Jumping Jacks validation harness.

The harness's INPUT is not in the repository (the `frames_open` archive is truncated), so these
cases pin the arithmetic and the reporting conventions rather than replaying the corpus. That is
the same split `tests/test_rotation_proxy_fidelity.py` uses for the Fit3D harness.
"""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.egoexo.jumping_jacks_validation import (
    EXO_VIEWS,
    FOOT_SPLIT_CRITERION,
    cadence_hz,
    cross_view_agreement,
    floor_discarded,
    load_labels,
    seconds_per_rep,
    spread,
    summarize,
    validity_rate,
)


class _Frame:
    def __init__(self, valid: bool) -> None:
        self.valid = valid


class CadenceTest(unittest.TestCase):
    def test_cadence_is_reps_per_second_over_the_analysed_span(self) -> None:
        self.assertAlmostEqual(cadence_hz(4, 60, 30.0), 2.0, places=6)

    def test_an_unsegmented_action_is_NaN_rather_than_zero(self) -> None:
        """A 0.0 would average in as "infinitely slow" and drag the cadence summary toward the
        answer that justifies the `min_rep_seconds` knob. `view_estimation` records the same
        NaN-not-zero reasoning for a different quantity."""
        self.assertTrue(math.isnan(cadence_hz(0, 60, 30.0)))
        self.assertTrue(math.isnan(cadence_hz(4, 0, 30.0)))
        self.assertTrue(math.isnan(cadence_hz(4, 60, 0.0)))

    def test_the_floor_probe_measures_what_the_direct_reading_cannot(self) -> None:
        """EVERY WINDOW `segment_reps` RETURNS IS ALREADY AT LEAST `min_rep_seconds` LONG, so the
        shortest returned repetition can never show the floor biting. Differencing the counts at
        two floors can, and a positive difference means `base.py:55`'s "must lower it" was right
        about this movement."""
        self.assertEqual(floor_discarded(6, 9), 3)
        self.assertEqual(floor_discarded(6, 6), 0)

    def test_a_lower_floor_finding_FEWER_reps_is_not_reported_as_a_negative_loss(self) -> None:
        self.assertEqual(floor_discarded(6, 5), 0)

    def test_seconds_per_rep_inverts_cadence_and_propagates_NaN(self) -> None:
        self.assertAlmostEqual(seconds_per_rep(2.27), 1.0 / 2.27, places=6)
        self.assertTrue(math.isnan(seconds_per_rep(math.nan)))
        self.assertTrue(math.isnan(seconds_per_rep(0.0)))


class ValidityRateTest(unittest.TestCase):
    def test_it_counts_valid_frames(self) -> None:
        core = [_Frame(True), _Frame(False), _Frame(True), _Frame(True)]
        self.assertAlmostEqual(validity_rate(core), 0.75, places=6)

    def test_an_empty_clip_is_zero_not_a_division_error(self) -> None:
        self.assertEqual(validity_rate([]), 0.0)


class AgreementTest(unittest.TestCase):
    def test_the_three_verdicts(self) -> None:
        both = {"exo_l": {"jj_incomplete_leg_rom"}, "exo_r": {"jj_incomplete_leg_rom"}}
        neither = {"exo_l": set(), "exo_r": set()}
        mixed = {"exo_l": {"jj_incomplete_leg_rom"}, "exo_r": set()}
        self.assertEqual(cross_view_agreement(both, "jj_incomplete_leg_rom"), "unanimous_fire")
        self.assertEqual(cross_view_agreement(neither, "jj_incomplete_leg_rom"), "unanimous_silent")
        self.assertEqual(cross_view_agreement(mixed, "jj_incomplete_leg_rom"), "split")

    def test_a_single_camera_is_not_agreement(self) -> None:
        """The truncated archive leaves one action with only `exo_r`. Reporting it as unanimous
        would inflate the agreement count with an action nothing could disagree about."""
        self.assertIsNone(cross_view_agreement({"exo_r": set()}, "jj_incomplete_leg_rom"))
        self.assertIsNone(cross_view_agreement({}, "jj_incomplete_leg_rom"))


class SpreadTest(unittest.TestCase):
    def test_it_is_the_range_over_the_finite_values(self) -> None:
        self.assertAlmostEqual(spread([1.4, 1.9, 1.6]), 0.5, places=6)

    def test_fewer_than_two_finite_values_is_NaN(self) -> None:
        self.assertTrue(math.isnan(spread([1.4])))
        self.assertTrue(math.isnan(spread([1.4, math.nan])))


class LabelsTest(unittest.TestCase):
    def test_it_reads_the_fault_flag_per_criterion(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "tkv.json"
            path.write_text(
                json.dumps(
                    {
                        "a_action_1": {
                            FOOT_SPLIT_CRITERION: {"fault": 1, "n_true": 0, "n_false": 2},
                            "Other.": {"fault": 0, "n_true": 2, "n_false": 0},
                        }
                    }
                ),
                encoding="utf-8",
            )
            labels = load_labels(path)
        self.assertEqual(labels["a_action_1"][FOOT_SPLIT_CRITERION], 1)
        self.assertEqual(labels["a_action_1"]["Other."], 0)


class SummarizeTest(unittest.TestCase):
    def _payload(self) -> dict:
        def view(reps: int, fired: list[str], validity: float, stance: float, knee: float) -> dict:
            return {
                "frames": 100,
                "validity_rate": validity,
                "reps_found": reps,
                "reps_analyzed": min(reps, 3),
                "fallback": None if reps else "no_reps_detected",
                "cadence_hz": cadence_hz(reps, 100, 30.0),
                "fired": [],
                # Both rules are silent, so `fired` is empty on every real pair; the agreement and
                # rate statistics are computed from what the parent spec's cuts WOULD have said.
                "would_fire": fired,
                "max_stance_width_ratio": stance,
                "min_knee_ankle_ratio": knee,
                "scored_reps": min(reps, 3),
                "rom_rep_hits": 1 if fired else 0,
                "valgus_rep_hits": 0,
                "per_rep_widest": [stance],
                "per_rep_tightest": [knee],
                "open_frames": 10,
                "open_observed_below_cut": 8,
                "open_aligned_below_cut": 7,
                "open_observed_median": 0.77,
                "open_aligned_median": 0.81,
            }

        return {
            "actions": [
                {
                    "sample_id": "a_action_1",
                    "foot_split_fault": 0,
                    "views": {
                        "exo_l": view(6, [], 0.9, 1.55, 0.95),
                        "exo_m": view(6, ["jj_incomplete_leg_rom"], 0.9, 1.20, 0.95),
                    },
                    "agreement": {
                        "jj_incomplete_leg_rom": "split",
                        "jj_knee_valgus_landing": "unanimous_silent",
                    },
                    "stance_spread": 0.35,
                    "valgus_spread": 0.0,
                }
            ]
        }

    def test_no_detection_is_ever_actually_emitted(self) -> None:
        """The roster's whole-detector consequence: every rule is silent or withdrawn, so the
        rates below describe what the parent spec's cuts WOULD have said, never what a user saw."""
        summary = summarize(self._payload())
        self.assertEqual(summary["detections_emitted"], 0)

    def test_the_confound_control_is_carried_through_the_summary(self) -> None:
        """The aligned-knee control is the evidence that withdrew the valgus rule, so it has to
        survive aggregation rather than living in a one-off probe."""
        summary = summarize(self._payload())
        self.assertEqual(summary["open_frames"], 20)
        self.assertAlmostEqual(summary["valgus_observed_frame_rate"], 0.8, places=6)
        self.assertAlmostEqual(summary["valgus_aligned_frame_rate"], 0.7, places=6)

    def test_the_fire_rate_is_over_action_camera_pairs_not_actions(self) -> None:
        """Three simultaneous cameras of one action are three chances to fire, and reporting one
        rate per ACTION would hide exactly the cross-camera disagreement the exo rig exists to
        expose."""
        summary = summarize(self._payload())
        self.assertEqual(summary["actions"], 1)
        self.assertEqual(summary["action_camera_pairs"], 2)
        self.assertAlmostEqual(summary["leg_rom_fire_rate"], 0.5, places=6)
        self.assertAlmostEqual(summary["valgus_fire_rate"], 0.0, places=6)

    def test_it_counts_the_actions_humans_judged_correct(self) -> None:
        summary = summarize(self._payload())
        self.assertEqual(summary["actions_judged_correct_on_foot_split"], 1)

    def test_the_agreement_counters_carry_all_three_verdicts(self) -> None:
        summary = summarize(self._payload())
        self.assertEqual(
            summary["agreement_leg_rom"],
            {"unanimous_fire": 0, "unanimous_silent": 0, "split": 1},
        )

    def test_the_exo_view_set_is_the_three_third_person_cameras(self) -> None:
        self.assertEqual(EXO_VIEWS, ("exo_l", "exo_m", "exo_r"))


if __name__ == "__main__":
    unittest.main()
