"""Drift guard for ``backend/app/services/scoring.py`` against its frontend source of truth
(``frontend/src/lib/formScore.ts`` + ``quality.ts``'s ``wasMeasured``), and the shared score
vectors both sides must agree on. Same drift-guard pattern as ``tests/test_movement_muscles.py``:
parse the TS source straight off disk (regex, no Node) and assert the two sides agree, then prove
the guard has teeth by re-running the SAME comparison against a corrupted copy of the text.

NOTE: another agent may be editing ``frontend/`` concurrently on this machine. If this file's
drift assertions ever fail, check ``git diff`` on ``formScore.ts``/``quality.ts`` before assuming
the Python port is the one that is wrong.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from backend.app.services import scoring

REPO_ROOT = Path(__file__).resolve().parents[1]
FORM_SCORE_TS = REPO_ROOT / "frontend" / "src" / "lib" / "formScore.ts"
QUALITY_TS = REPO_ROOT / "frontend" / "src" / "lib" / "quality.ts"

_CONSTANT_NAMES = ("PENALTY_PER_SEVERITY", "FLOOR")


def _parse_constants(text: str) -> dict[str, int]:
    """``const NAME = N;`` for each of ``_CONSTANT_NAMES``, parsed straight out of the TS source."""
    out: dict[str, int] = {}
    for name in _CONSTANT_NAMES:
        match = re.search(rf"const {name}\s*=\s*(-?\d+)\s*;", text)
        if match is None:
            raise AssertionError(f"could not find `const {name} = N;` in formScore.ts")
        out[name] = int(match.group(1))
    return out


class DriftGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ts_text = FORM_SCORE_TS.read_text(encoding="utf-8")
        cls.quality_text = QUALITY_TS.read_text(encoding="utf-8")
        cls.ts_constants = _parse_constants(cls.ts_text)

    def test_parser_sanity_both_constants_are_found(self) -> None:
        # A regex that silently returned {} (or dropped one name) would make every comparison
        # below vacuously pass -- this is what stops that.
        self.assertEqual(set(self.ts_constants), set(_CONSTANT_NAMES))

    def _assert_constants_agree(self, constants: dict[str, int]) -> None:
        """The whole comparison, in one place, so the corruption test below re-runs THIS rather
        than a hand-written echo of it -- the same reasoning as
        ``test_movement_muscles.py::_assert_agrees_with_frontend``."""
        self.assertEqual(constants["PENALTY_PER_SEVERITY"], scoring.PENALTY_PER_SEVERITY)
        self.assertEqual(constants["FLOOR"], scoring.FLOOR)

    def test_constants_match_the_frontend(self) -> None:
        self._assert_constants_agree(self.ts_constants)

    def test_a_corrupted_constant_makes_this_guard_fail(self) -> None:
        corrupted_text = self.ts_text.replace("const FLOOR = 20;", "const FLOOR = 99;")
        corrupted = _parse_constants(corrupted_text)
        with self.assertRaises(AssertionError):
            self._assert_constants_agree(corrupted)

    def test_frontend_still_rounds_with_math_round(self) -> None:
        # The whole point of scoring._js_round: if formScore.ts ever switched off Math.round, the
        # Python port's rounding semantics would silently disagree with it.
        self.assertRegex(self.ts_text, r"Math\.round\(")

    def test_quality_ts_still_reads_valid_frame_ratio_with_gt_zero(self) -> None:
        # scoring.was_measured mirrors `(quality?.valid_frame_ratio ?? 0) > 0` exactly -- this
        # regex is deliberately tight to that shape, not just "mentions valid_frame_ratio somewhere".
        self.assertRegex(self.quality_text, r"valid_frame_ratio\s*\?\?\s*0\s*\)\s*>\s*0")


class FormScoreVectorTests(unittest.TestCase):
    """The shared vectors from the WP2 spec (§1 / the task's item 2), each also hand-verifiable:
    penalty = 25 * sum(clamped severities); value = clamp(100 - penalty, 20, 100); JS-rounded."""

    @staticmethod
    def _det(severity: float) -> dict:
        return {"severity": severity}

    def test_no_detections_measured_scores_100(self) -> None:
        self.assertEqual(scoring.form_score([], {"valid_frame_ratio": 1.0}), 100)

    def test_one_severity_0_5_scores_88(self) -> None:
        # penalty = 12.5, value = 87.5 -> Math.round -> 88.
        dets = [self._det(0.5)]
        self.assertEqual(scoring.form_score(dets, {"valid_frame_ratio": 1.0}), 88)

    def test_one_severity_0_3_scores_93(self) -> None:
        # penalty = 7.5, value = 92.5 -> Math.round -> 93 (NOT 92 -- proves floor(x+0.5), not
        # Python's banker's round()).
        dets = [self._det(0.3)]
        self.assertEqual(scoring.form_score(dets, {"valid_frame_ratio": 1.0}), 93)

    def test_two_severity_0_8_scores_60(self) -> None:
        dets = [self._det(0.8), self._det(0.8)]
        self.assertEqual(scoring.form_score(dets, {"valid_frame_ratio": 1.0}), 60)

    def test_five_severity_1_0_scores_the_floor_20(self) -> None:
        dets = [self._det(1.0)] * 5
        self.assertEqual(scoring.form_score(dets, {"valid_frame_ratio": 1.0}), 20)

    def test_severity_above_1_clamps_to_75(self) -> None:
        dets = [self._det(1.5)]
        self.assertEqual(scoring.form_score(dets, {"valid_frame_ratio": 1.0}), 75)

    def test_negative_severity_clamps_to_zero_scores_100(self) -> None:
        dets = [self._det(-1)]
        self.assertEqual(scoring.form_score(dets, {"valid_frame_ratio": 1.0}), 100)

    def test_missing_quality_is_unmeasured_returns_none(self) -> None:
        self.assertIsNone(scoring.form_score([], None))

    def test_zero_valid_frame_ratio_is_unmeasured_returns_none(self) -> None:
        dets = [self._det(0.5)]
        self.assertIsNone(scoring.form_score(dets, {"valid_frame_ratio": 0}))

    def test_empty_quality_dict_is_unmeasured_returns_none(self) -> None:
        self.assertIsNone(scoring.form_score([], {}))


class FaultSeverityEdgeCaseTests(unittest.TestCase):
    """Missing/non-numeric severities, and non-dict detections, all count as severity 0."""

    def test_missing_or_non_numeric_severity_counts_as_zero(self) -> None:
        dets = [
            {"note": "no severity key at all"},
            {"severity": "not a number"},
            {"severity": None},
        ]
        self.assertEqual(scoring.form_score(dets, {"valid_frame_ratio": 1.0}), 100)

    def test_a_non_dict_detection_counts_as_zero(self) -> None:
        dets = [None, "not a dict", 42]
        self.assertEqual(scoring.form_score(dets, {"valid_frame_ratio": 1.0}), 100)

    def test_a_boolean_severity_counts_as_zero(self) -> None:
        # bool is an int subclass in Python -- must not be treated as severity 1/0.
        dets = [{"severity": True}]
        self.assertEqual(scoring.form_score(dets, {"valid_frame_ratio": 1.0}), 100)


class WasMeasuredTests(unittest.TestCase):
    def test_none_quality_is_unmeasured(self) -> None:
        self.assertFalse(scoring.was_measured(None))

    def test_positive_ratio_is_measured(self) -> None:
        self.assertTrue(scoring.was_measured({"valid_frame_ratio": 0.4}))

    def test_zero_ratio_is_unmeasured(self) -> None:
        self.assertFalse(scoring.was_measured({"valid_frame_ratio": 0}))

    def test_missing_key_is_unmeasured(self) -> None:
        self.assertFalse(scoring.was_measured({}))


class FormScoreForAnalysisTests(unittest.TestCase):
    def test_reads_detections_and_quality_off_the_result(self) -> None:
        result = {
            "detections": [{"severity": 0.5}],
            "quality": {"valid_frame_ratio": 1.0},
        }
        self.assertEqual(scoring.form_score_for_analysis(result), 88)

    def test_none_result_is_none(self) -> None:
        self.assertIsNone(scoring.form_score_for_analysis(None))

    def test_empty_result_is_none(self) -> None:
        self.assertIsNone(scoring.form_score_for_analysis({}))


if __name__ == "__main__":
    unittest.main()
