"""Table-driven tests for ``backend/app/services/redflags.py`` -- a positive and a negative case
for every rule ((a) pain_high, (b) pain_rise, (c) form_drop), plus the exact vectors from the WP2
spec (§1 "Red-flag rules") and the task's item 4, and the pure-function edge cases (missing data,
naive timestamps, cross-plan isolation) that give the module's branches real coverage rather than
leaving them merely reachable.
"""

from __future__ import annotations

import unittest

from backend.app.services import redflags

PLAN = "p1"
OTHER_PLAN = "p0"


def _c(pain_nrs: int | None = None, form_score: int | None = None, created_at: str | None = None,
       plan_id: str = PLAN) -> dict:
    """One history/new-checkin row. Keys are omitted (not set to None) when not given, so a test
    can exercise "the field is simply absent" rather than "the field is null" -- both are real
    shapes ``evaluate`` must survive."""
    row: dict = {"plan_id": plan_id}
    if pain_nrs is not None:
        row["pain_nrs"] = pain_nrs
    if form_score is not None:
        row["form_score"] = form_score
    if created_at is not None:
        row["created_at"] = created_at
    return row


# ---------------------------------------------------------------------------------------------
# Rule (a): pain_high.
# ---------------------------------------------------------------------------------------------


class PainHighTests(unittest.TestCase):
    def test_positive_pain_at_threshold_flags(self) -> None:
        self.assertEqual(redflags.evaluate(_c(pain_nrs=6), []), ["pain_high"])

    def test_negative_pain_just_below_threshold_does_not_flag(self) -> None:
        self.assertEqual(redflags.evaluate(_c(pain_nrs=5), []), [])

    def test_exact_vector_pain_7_no_history(self) -> None:
        self.assertEqual(redflags.evaluate(_c(pain_nrs=7), []), ["pain_high"])


# ---------------------------------------------------------------------------------------------
# Rule (b): pain_rise.
# ---------------------------------------------------------------------------------------------


class PainRiseTests(unittest.TestCase):
    def test_positive_rise_of_three_flags(self) -> None:
        history = [_c(pain_nrs=1, created_at="2026-09-01T00:00:00+00:00")]
        self.assertEqual(redflags.evaluate(_c(pain_nrs=4), history), ["pain_rise"])

    def test_negative_rise_of_two_does_not_flag(self) -> None:
        history = [_c(pain_nrs=2, created_at="2026-09-01T00:00:00+00:00")]
        self.assertEqual(redflags.evaluate(_c(pain_nrs=4), history), [])

    def test_exact_vector_3_then_6_flags_both_pain_rules_in_order(self) -> None:
        history = [_c(pain_nrs=3, created_at="2026-09-01T00:00:00+00:00")]
        self.assertEqual(redflags.evaluate(_c(pain_nrs=6), history), ["pain_high", "pain_rise"])

    def test_exact_vector_1_then_4_flags_only_pain_rise(self) -> None:
        history = [_c(pain_nrs=1, created_at="2026-09-01T00:00:00+00:00")]
        self.assertEqual(redflags.evaluate(_c(pain_nrs=4), history), ["pain_rise"])

    def test_exact_vector_5_then_5_flags_nothing(self) -> None:
        history = [_c(pain_nrs=5, created_at="2026-09-01T00:00:00+00:00")]
        self.assertEqual(redflags.evaluate(_c(pain_nrs=5), history), [])

    def test_no_earlier_checkin_on_the_plan_cannot_trigger_pain_rise(self) -> None:
        self.assertEqual(redflags.evaluate(_c(pain_nrs=10), []), ["pain_high"])

    def test_compares_against_the_most_recent_earlier_checkin_not_the_oldest(self) -> None:
        # Oldest check-in has pain 0 (a rise of 6 from it), but the MOST RECENT one has pain 8
        # (a rise of only 2) -- the recent one must win the comparison.
        history = [
            _c(pain_nrs=0, created_at="2026-09-01T00:00:00+00:00"),
            _c(pain_nrs=8, created_at="2026-09-05T00:00:00+00:00"),
        ]
        self.assertEqual(redflags.evaluate(_c(pain_nrs=10), history), ["pain_high"])

    def test_a_naive_created_at_does_not_crash_the_comparison(self) -> None:
        # PostgREST always sends tz-aware timestamps, but a hand-built fixture is easy to leave
        # naive -- _parse_ts must default it to UTC rather than raising on the tz-aware comparison.
        history = [_c(pain_nrs=1, created_at="2026-09-01T00:00:00")]
        self.assertEqual(redflags.evaluate(_c(pain_nrs=4), history), ["pain_rise"])

    def test_a_history_row_missing_pain_nrs_does_not_crash_and_does_not_flag(self) -> None:
        history = [_c(created_at="2026-09-01T00:00:00+00:00")]  # no pain_nrs at all
        self.assertEqual(redflags.evaluate(_c(pain_nrs=5), history), [])

    def test_a_new_checkin_missing_pain_nrs_cannot_trigger_pain_high_or_pain_rise(self) -> None:
        history = [_c(pain_nrs=1, created_at="2026-09-01T00:00:00+00:00")]
        self.assertEqual(redflags.evaluate(_c(), history), [])

    def test_history_rows_without_created_at_are_ignored(self) -> None:
        # Cannot be ordered, so cannot be "the most recent earlier check-in".
        history = [_c(pain_nrs=0)]  # no created_at
        self.assertEqual(redflags.evaluate(_c(pain_nrs=10), history), ["pain_high"])


# ---------------------------------------------------------------------------------------------
# Rule (c): form_drop.
# ---------------------------------------------------------------------------------------------


def _scored_history(scores: list[int]) -> list[dict]:
    """``len(scores)`` history rows on PLAN, oldest first, pain held constant at 2 (below
    pain_high, no rise) so a form_drop test's result isn't polluted by the pain rules. The first
    row's timestamp is deliberately naive to exercise that branch here too."""
    rows = []
    for i, score in enumerate(scores):
        created = (
            "2026-09-01T00:00:00" if i == 0 else f"2026-09-0{i + 1}T00:00:00+00:00"
        )
        rows.append(_c(pain_nrs=2, form_score=score, created_at=created))
    return rows


class FormDropTests(unittest.TestCase):
    def test_exact_vector_90_90_90_60_60_then_new_60_flags_form_drop(self) -> None:
        history = _scored_history([90, 90, 90, 60, 60])
        new = _c(pain_nrs=2, form_score=60)
        self.assertEqual(redflags.evaluate(new, history), ["form_drop"])

    def test_negative_no_drop_across_six_scored_checkins_does_not_flag(self) -> None:
        history = _scored_history([90, 90, 90, 85, 85])
        new = _c(pain_nrs=2, form_score=85)
        self.assertEqual(redflags.evaluate(new, history), [])

    def test_exact_vector_only_five_scored_in_total_does_not_flag(self) -> None:
        history = _scored_history([90, 90, 90, 60])  # 4 scored + the new one below = 5 total
        new = _c(pain_nrs=2, form_score=60)
        self.assertEqual(redflags.evaluate(new, history), [])

    def test_exact_vector_unscored_new_checkin_after_a_drop_does_not_flag(self) -> None:
        # The same history that flagged above, but the NEW check-in has no form_score.
        history = _scored_history([90, 90, 90, 60, 60])
        new = _c(pain_nrs=2)  # no form_score
        self.assertEqual(redflags.evaluate(new, history), [])

    def test_exact_vector_interleaved_null_score_rows_are_skipped_when_counting(self) -> None:
        history = [
            _c(pain_nrs=2, form_score=90, created_at="2026-09-01T00:00:00+00:00"),
            _c(pain_nrs=2, created_at="2026-09-01T12:00:00+00:00"),  # no form_score -- skipped
            _c(pain_nrs=2, form_score=90, created_at="2026-09-02T00:00:00+00:00"),
            _c(pain_nrs=2, created_at="2026-09-02T12:00:00+00:00"),  # no form_score -- skipped
            _c(pain_nrs=2, form_score=90, created_at="2026-09-03T00:00:00+00:00"),
            _c(pain_nrs=2, form_score=60, created_at="2026-09-04T00:00:00+00:00"),
            _c(pain_nrs=2, form_score=60, created_at="2026-09-05T00:00:00+00:00"),
        ]
        new = _c(pain_nrs=2, form_score=60)
        # 5 scored history rows (the null ones don't count) + the new one = 6 -- still flags,
        # proving the null rows neither pad the count nor shift the window.
        self.assertEqual(redflags.evaluate(new, history), ["form_drop"])

    def test_exact_vector_other_plans_rows_are_ignored(self) -> None:
        other_plan_drop = [
            {**row, "plan_id": OTHER_PLAN} for row in _scored_history([90, 90, 90, 60, 60])
        ]
        new = _c(pain_nrs=2, form_score=60, plan_id=PLAN)
        self.assertEqual(redflags.evaluate(new, other_plan_drop), [])

    def test_a_rise_in_form_score_does_not_flag(self) -> None:
        history = _scored_history([60, 60, 60, 90, 90])
        new = _c(pain_nrs=2, form_score=90)
        self.assertEqual(redflags.evaluate(new, history), [])


# ---------------------------------------------------------------------------------------------
# All three rules can fire together; order is always pain_high, pain_rise, form_drop.
# ---------------------------------------------------------------------------------------------


class CombinedOrderingTests(unittest.TestCase):
    def test_all_three_rules_fire_in_the_fixed_order(self) -> None:
        history = _scored_history([90, 90, 90, 60, 60])
        # Rewrite the last history row's pain so a big rise is also present.
        history[-1]["pain_nrs"] = 0
        new = _c(pain_nrs=9, form_score=60)
        self.assertEqual(redflags.evaluate(new, history), ["pain_high", "pain_rise", "form_drop"])


if __name__ == "__main__":
    unittest.main()
