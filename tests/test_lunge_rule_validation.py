import math
import unittest


class SliceRepTests(unittest.TestCase):
    def test_slices_inclusive_of_both_bounds(self) -> None:
        from src.rehab24.lunge_rule_validation import slice_rep

        frames = [{"frame_index": i} for i in range(100)]
        window = slice_rep(frames, 10, 19)
        self.assertEqual(len(window), 10)
        self.assertEqual(window[0]["frame_index"], 10)
        self.assertEqual(window[-1]["frame_index"], 19)

    def test_clamps_a_last_frame_beyond_the_clip(self) -> None:
        # Labels come from the mocap timeline; a video can be a few frames shorter.
        from src.rehab24.lunge_rule_validation import slice_rep

        frames = [{"frame_index": i} for i in range(20)]
        self.assertEqual(len(slice_rep(frames, 15, 40)), 5)

    def test_returns_empty_when_the_window_starts_past_the_clip(self) -> None:
        from src.rehab24.lunge_rule_validation import slice_rep

        self.assertEqual(slice_rep([{"frame_index": i} for i in range(5)], 90, 99), [])

    def test_raises_on_negative_first_frame(self) -> None:
        from src.rehab24.lunge_rule_validation import slice_rep

        with self.assertRaises(ValueError):
            slice_rep([{"frame_index": i} for i in range(5)], -1, 3)

    def test_raises_when_last_frame_precedes_first_frame(self) -> None:
        from src.rehab24.lunge_rule_validation import slice_rep

        with self.assertRaises(ValueError):
            slice_rep([{"frame_index": i} for i in range(5)], 3, 1)


class ContingencyTests(unittest.TestCase):
    def test_counts_positives_as_incorrect_reps(self) -> None:
        from src.rehab24.lunge_rule_validation import contingency

        # rep 1: incorrect + fired  -> tp
        # rep 2: correct   + fired  -> fp
        # rep 3: correct   + silent -> tn
        # rep 4: incorrect + silent -> fn
        table = contingency(fired=[True, True, False, False],
                            correct=[False, True, True, False])
        self.assertEqual(table, {"tp": 1, "fp": 1, "tn": 1, "fn": 1})

    def test_the_positive_convention_is_not_symmetric_under_a_swap(self) -> None:
        # Asymmetric counts on purpose. The four-cells-at-one case above cannot catch a
        # `positive = is_correct` inversion: that swap permutes the cells without changing any
        # count. Here a swapped implementation would report {"tp":1,"fp":2,...} instead.
        from src.rehab24.lunge_rule_validation import contingency

        table = contingency(fired=[True, True, True], correct=[False, False, True])
        self.assertEqual(table, {"tp": 2, "fp": 1, "tn": 0, "fn": 0})

    def test_rejects_mismatched_lengths(self) -> None:
        from src.rehab24.lunge_rule_validation import contingency

        with self.assertRaises(ValueError):
            contingency(fired=[True], correct=[True, False])

    def test_empty_inputs_yield_an_all_zero_table(self) -> None:
        from src.rehab24.lunge_rule_validation import contingency

        self.assertEqual(contingency(fired=[], correct=[]), {"tp": 0, "fp": 0, "tn": 0, "fn": 0})


class RankAucTests(unittest.TestCase):
    def test_perfect_separation_scores_one(self) -> None:
        from src.rehab24.lunge_rule_validation import rank_auc

        self.assertAlmostEqual(
            rank_auc([1.0, 2.0, 8.0, 9.0], [False, False, True, True]), 1.0, places=6
        )

    def test_inverted_separation_scores_zero(self) -> None:
        from src.rehab24.lunge_rule_validation import rank_auc

        self.assertAlmostEqual(
            rank_auc([8.0, 9.0, 1.0, 2.0], [False, False, True, True]), 0.0, places=6
        )

    def test_ties_score_one_half(self) -> None:
        from src.rehab24.lunge_rule_validation import rank_auc

        self.assertAlmostEqual(rank_auc([5.0, 5.0], [True, False]), 0.5, places=6)

    def test_is_nan_when_one_class_is_empty(self) -> None:
        # A rule the dataset never exercises must report NaN, not a misleading 0.5.
        from src.rehab24.lunge_rule_validation import rank_auc

        self.assertTrue(math.isnan(rank_auc([1.0, 2.0], [True, True])))

    def test_ignores_non_finite_scores(self) -> None:
        from src.rehab24.lunge_rule_validation import rank_auc

        self.assertAlmostEqual(
            rank_auc([float("nan"), 1.0, 9.0], [True, False, True]), 1.0, places=6
        )


class CameraRoutingTests(unittest.TestCase):
    def test_sagittal_rules_read_cam18_and_frontal_rules_read_cam17(self) -> None:
        from src.rehab24.lunge_rule_validation import RULE_CAMERAS

        self.assertEqual(RULE_CAMERAS["lunge_knee_past_toes"], "cam18")
        self.assertEqual(RULE_CAMERAS["lunge_insufficient_depth"], "cam18")
        self.assertEqual(RULE_CAMERAS["lunge_knee_valgus"], "cam17")
        self.assertEqual(RULE_CAMERAS["lunge_pelvic_drop"], "cam17")

    def test_every_registered_lunge_rule_has_a_camera(self) -> None:
        # A rule added later without a routing entry would be silently dropped from the report.
        from src.pose.movements.lunge import LUNGE_DETECTOR
        from src.rehab24.lunge_rule_validation import RULE_CAMERAS

        emitted = {
            "lunge_knee_past_toes", "lunge_knee_valgus",
            "lunge_insufficient_depth", "lunge_pelvic_drop",
        }
        self.assertEqual(len(LUNGE_DETECTOR.rules), len(emitted))
        self.assertEqual(set(RULE_CAMERAS), emitted)


class OracleViewsTests(unittest.TestCase):
    def test_front_maps_to_a_label_production_can_never_emit(self) -> None:
        # Production always calls estimate_view_for_pose(allow_front=False), so "front" never
        # reaches the rules on the production path. The oracle pass deliberately bypasses that
        # gate to ask "would this rule fire if the view label were correct?"
        from src.rehab24.lunge_rule_validation import ORACLE_VIEWS

        self.assertEqual(ORACLE_VIEWS["front"], "front")

    def test_half_profile_maps_to_front_oblique(self) -> None:
        from src.rehab24.lunge_rule_validation import ORACLE_VIEWS

        self.assertEqual(ORACLE_VIEWS["half-profile"], "front_oblique")

    def test_side_maps_to_side(self) -> None:
        from src.rehab24.lunge_rule_validation import ORACLE_VIEWS

        self.assertEqual(ORACLE_VIEWS["side"], "side")

    def test_oracle_view_confidence_is_pinned_at_one(self) -> None:
        from src.rehab24.lunge_rule_validation import ORACLE_VIEW_CONFIDENCE

        self.assertEqual(ORACLE_VIEW_CONFIDENCE, 1.0)


class PerSubjectTests(unittest.TestCase):
    def test_groups_records_by_key_and_applies_value_fn_per_group(self) -> None:
        from src.rehab24.lunge_rule_validation import per_subject

        records = [
            {"person_id": "1", "score": 1.0},
            {"person_id": "1", "score": 3.0},
            {"person_id": "2", "score": 10.0},
        ]
        result = per_subject(
            records,
            key_fn=lambda r: r["person_id"],
            value_fn=lambda group: sum(r["score"] for r in group) / len(group),
        )
        self.assertEqual(result, {"1": 2.0, "2": 10.0})

    def test_does_not_pool_across_subjects(self) -> None:
        # 174 reps from 8 people are not independent: a pooled mean would let one subject's
        # count dominate. Grouping first and only then reducing is what per_subject exists for.
        from src.rehab24.lunge_rule_validation import per_subject

        records = [{"person_id": "1"}] * 100 + [{"person_id": "2"}] * 1
        result = per_subject(records, key_fn=lambda r: r["person_id"], value_fn=len)
        self.assertEqual(result, {"1": 100, "2": 1})

    def test_empty_records_yields_an_empty_dict(self) -> None:
        from src.rehab24.lunge_rule_validation import per_subject

        result = per_subject([], key_fn=lambda r: r["person_id"], value_fn=len)
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
