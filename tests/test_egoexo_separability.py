"""Statistics behind the silent-rule separability run (src/egoexo/separability.py).

Each property is pinned by a case that would FAIL under the plausible wrong implementation named
in its docstring, not only by a case the right one passes.
"""
from __future__ import annotations

import math
import unittest

from src.egoexo.separability import (
    BootstrapCI,
    auc,
    cluster_bootstrap_auc,
    fires,
    high_knee_action_scores,
    jumping_jacks_action_scores,
    leave_one_group_out,
    verdict,
    youden_cut,
)


class AucTest(unittest.TestCase):
    def test_orientation_follows_the_fault_side(self) -> None:
        """Wrong implementation: ignoring `lower_is_fault` (one of these two would read 0.0)."""
        scores, labels = [1.0, 2.0, 3.0, 4.0], [1, 1, 0, 0]
        self.assertEqual(auc(scores, labels, lower_is_fault=True), 1.0)
        self.assertEqual(auc(scores, labels, lower_is_fault=False), 0.0)

    def test_ties_count_half(self) -> None:
        self.assertEqual(auc([2.0, 2.0], [1, 0], lower_is_fault=True), 0.5)
        self.assertEqual(auc([1.0, 2.0, 2.0], [1, 1, 0], lower_is_fault=True), 0.75)

    def test_an_empty_class_is_nan_not_a_number(self) -> None:
        self.assertTrue(math.isnan(auc([1.0, 2.0], [0, 0], lower_is_fault=True)))
        self.assertTrue(math.isnan(auc([], [], lower_is_fault=True)))


class ClusterBootstrapTest(unittest.TestCase):
    def test_it_resamples_participants_not_actions(self) -> None:
        """Wrong implementation: resampling ACTIONS. With two participants, one all-positive and
        one all-negative, half of all participant resamples draw one person twice and have a single
        class -- so about half are skipped. An action-level resample would skip almost none."""
        scores = [1.0] * 10 + [2.0] * 10
        labels = [1] * 10 + [0] * 10
        groups = ["p"] * 10 + ["q"] * 10
        ci = cluster_bootstrap_auc(scores, labels, groups, lower_is_fault=True, n_boot=400)
        self.assertGreater(ci.skipped, 120)
        self.assertLess(ci.skipped, 280)
        self.assertEqual(ci.used + ci.skipped, 400)
        self.assertEqual((ci.low, ci.high), (1.0, 1.0))

    def test_it_is_seeded(self) -> None:
        scores = [0.1, 0.4, 0.35, 0.8, 0.5, 0.9, 0.2, 0.7]
        labels = [1, 1, 0, 0, 1, 0, 1, 0]
        groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
        first = cluster_bootstrap_auc(scores, labels, groups, lower_is_fault=True, n_boot=200)
        second = cluster_bootstrap_auc(scores, labels, groups, lower_is_fault=True, n_boot=200)
        self.assertEqual(first, second)
        self.assertLessEqual(first.low, auc(scores, labels, lower_is_fault=True))
        self.assertGreaterEqual(first.high, auc(scores, labels, lower_is_fault=True))


class VerdictTest(unittest.TestCase):
    def test_the_three_words(self) -> None:
        self.assertEqual(verdict(BootstrapCI(0.51, 0.9, 10, 0)), "separates")
        self.assertEqual(verdict(BootstrapCI(0.1, 0.49, 10, 0)), "sorted backwards")
        self.assertEqual(verdict(BootstrapCI(0.45, 0.8, 10, 0)), "undetermined")
        # A CI touching 0.5 is not separation.
        self.assertEqual(verdict(BootstrapCI(0.5, 0.9, 10, 0)), "undetermined")
        self.assertEqual(verdict(BootstrapCI(math.nan, math.nan, 0, 10)), "undetermined")


class CutTest(unittest.TestCase):
    def test_youden_picks_the_separating_midpoint(self) -> None:
        cut = youden_cut([1.0, 1.1, 1.5, 1.6], [1, 1, 0, 0], lower_is_fault=True)
        self.assertAlmostEqual(cut, 1.3)
        self.assertTrue(fires(1.1, cut, lower_is_fault=True))
        self.assertFalse(fires(1.5, cut, lower_is_fault=True))

    def test_ties_in_j_break_toward_firing_less(self) -> None:
        """Wrong implementation: first/last candidate wins, or the widest cut wins. Here J peaks at
        0.5 on TWO cuts: 1.5 (fires on one action) and 3.5 (fires on three). It must pick 1.5."""
        scores, labels = [1.0, 2.0, 3.0, 4.0], [1, 0, 1, 0]
        cut = youden_cut(scores, labels, lower_is_fault=True)
        self.assertAlmostEqual(cut, 1.5)
        self.assertEqual(sum(fires(s, cut, lower_is_fault=True) for s in scores), 1)
        # Mirror orientation: the cut must again be the one that fires less.
        cut = youden_cut([4.0, 3.0, 2.0, 1.0], labels, lower_is_fault=False)
        self.assertAlmostEqual(cut, 3.5)

    def test_a_class_missing_yields_no_cut(self) -> None:
        self.assertTrue(math.isnan(youden_cut([1.0, 2.0], [0, 0], lower_is_fault=True)))

    def test_leave_one_out_never_scores_a_person_with_their_own_cut(self) -> None:
        """Wrong implementation: fitting on all data. Participant `c` is the ONLY one with
        positives, so without `c` no cut exists; that fold must be counted, not scored."""
        scores = [2.0, 2.1, 1.9, 2.2, 1.0, 1.1]
        labels = [0, 0, 0, 0, 1, 1]
        groups = ["a", "a", "b", "b", "c", "c"]
        held = leave_one_group_out(scores, labels, groups, lower_is_fault=True)
        self.assertEqual(held.folds_without_a_cut, 1)
        self.assertEqual(held.tp + held.fn, 0)  # c's positives were never scored
        self.assertTrue(math.isnan(held.sensitivity))
        self.assertEqual(held.tn + held.fp, 4)


class ActionScoreTest(unittest.TestCase):
    def test_jumping_jacks_takes_median_per_view_then_across_views(self) -> None:
        payload = {"detail": {"actions": [
            {"sample_id": "a", "views": {
                "exo_l": {"per_rep_widest": [1.0, 1.2, 1.4]},   # median 1.2
                "exo_m": {"per_rep_widest": [1.5]},             # 1.5
                "exo_r": {"per_rep_widest": []},                # no reps: ignored
            }},
            {"sample_id": "empty", "views": {"exo_m": {"per_rep_widest": []}}},
        ]}}
        out = jumping_jacks_action_scores(payload, spec_cut=1.3)
        self.assertEqual(set(out), {"a"})
        self.assertAlmostEqual(out["a"]["score"], 1.35)
        self.assertAlmostEqual(out["a"]["spec_cut_rep_rate"], 0.5)  # 1.0, 1.2 of 4 pooled reps

    def test_high_knee_reads_only_the_gated_cameras(self) -> None:
        """Wrong implementation: pooling the frontal camera, which cannot see thigh elevation."""
        payload = {"per_action": {
            "a": {
                "exo_l": {"peak_elevation_median": -0.5, "fire_rate_cited_cut": 0.0},
                "exo_r": {"peak_elevation_median": -0.3, "fire_rate_cited_cut": 0.2},
                "exo_m": {"peak_elevation_median": -0.99, "fire_rate_cited_cut": 1.0},
            },
            "front_only": {"exo_m": {"peak_elevation_median": -0.9, "fire_rate_cited_cut": 1.0}},
        }}
        out = high_knee_action_scores(payload, gated_views=("exo_l", "exo_r"))
        self.assertEqual(set(out), {"a"})
        self.assertAlmostEqual(out["a"]["score"], -0.4)
        self.assertAlmostEqual(out["a"]["cited_cut_rate"], 0.1)


if __name__ == "__main__":
    unittest.main()
