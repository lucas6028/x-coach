"""Server-side port of ``frontend/src/lib/formScore.ts`` (plus ``quality.ts``'s ``wasMeasured``
criterion) for WP2 (session check-ins, §1 of the 20-day plan).

WHY THIS IS A BACKEND MODULE, NOT JUST A FRONTEND ONE. Before WP2, ``formScore()`` was a
presentation-layer computation only -- rendered in the studio, never stored. WP2 stores the number
ON THE CHECK-IN (``session_checkins.form_score``) because the clinician trend chart and the
``form_drop`` red-flag rule (``services/redflags.py``) both need it QUERYABLE at check-in time, from
whatever ``detections``/``quality`` the linked analysis carried -- recomputing client-side and
trusting the client to send it back would let a tampered request fabricate a flag-suppressing
score. So "25 points per severity point, floored at 20, only when the clip was measured" is ONE
rule, expressed in TWO languages, kept in lockstep by ``tests/test_scoring_drift.py`` (same
drift-guard pattern as ``tests/test_movement_muscles.py`` for the muscle table) rather than by hand.

ROUNDING MUST MATCH JAVASCRIPT'S ``Math.round``, NOT PYTHON'S BUILTIN ``round()``. Python's
``round()`` uses banker's rounding (round-half-to-even): ``round(92.5)`` is ``92``. JavaScript's
``Math.round`` is ``floor(x + 0.5)`` for every real number, always rounding a ``.5`` UP:
``Math.round(92.5)`` is ``93``. A clip with one severity-0.3 fault scores exactly 92.5 before
rounding, so this distinction is not a corner case -- it is the difference between agreeing with
the frontend and disagreeing with it by one point on an ordinary fault.
"""

from __future__ import annotations

import math
from typing import Any

# How much one fault can cost -- mirrors ``frontend/src/lib/formScore.ts``'s
# ``PENALTY_PER_SEVERITY``. A severity-1.0 fault removes a quarter of the score.
PENALTY_PER_SEVERITY = 25
# The floor -- mirrors ``formScore.ts``'s ``FLOOR``. A measured clip full of faults is still a clip
# with signal in it; 0 would read as "the analysis failed", a different state (see ``form_score``
# returning ``None`` for that case instead).
FLOOR = 20


def _js_round(value: float) -> int:
    """``Math.round`` semantics: ``floor(x + 0.5)``, always rounding ``.5`` up. See the module
    docstring for why Python's builtin ``round()`` (banker's rounding) is the wrong function here."""
    return math.floor(value + 0.5)


def was_measured(quality: dict[str, Any] | None) -> bool:
    """Whether a clip was ever measurable -- the SAME criterion as ``frontend/src/lib/quality.ts``'s
    ``wasMeasured``: ``(quality?.valid_frame_ratio ?? 0) > 0``. A missing/absent ``quality`` counts
    as unmeasured, same as a missing field in the TS version -- "we cannot tell" must not resolve to
    "everything is fine". See ``quality.ts`` for the full reasoning (an empty ``detections`` list is
    otherwise indistinguishable between "no faults found" and "nothing was ever measured").
    """
    return ((quality or {}).get("valid_frame_ratio") or 0) > 0


def _fault_severity(detection: Any) -> float:
    """One detection's severity, clamped to ``[0, 1]``. A missing key, a non-numeric value, or a
    detection that isn't even a dict counts as severity 0 -- the task's explicit rule for "we don't
    know how bad this was" (matches ``formScore.ts``'s ``Math.min(1, Math.max(0, d.severity))``,
    which would coerce a non-number to ``NaN`` and silently zero out the whole penalty in JS; the
    Python port makes that same "unknown counts as harmless" outcome explicit rather than accidental).
    """
    if not isinstance(detection, dict):
        return 0.0
    severity = detection.get("severity")
    if not isinstance(severity, (int, float)) or isinstance(severity, bool):
        return 0.0
    return min(1.0, max(0.0, float(severity)))


def form_score(
    detections: list[Any] | None, quality: dict[str, Any] | None
) -> int | None:
    """The headline 0-100 form score for one clip, or ``None`` when it was never measured.

    ``score = clamp(100 - 25 * sum(clamp(severity, 0, 1) for each fault), FLOOR, 100)``, rounded
    with JS ``Math.round`` semantics (see the module docstring). Split from
    ``form_score_for_analysis`` so it is testable directly against hand-built fault lists, matching
    ``formScore.ts``'s own split into ``formScore``/``scoreFromDetections``.
    """
    if not was_measured(quality):
        return None
    penalty = sum(PENALTY_PER_SEVERITY * _fault_severity(d) for d in (detections or []))
    value = min(100.0, max(float(FLOOR), 100.0 - penalty))
    return _js_round(value)


def form_score_for_analysis(result: dict[str, Any] | None) -> int | None:
    """``form_score`` read off a stored analysis's ``result`` JSONB (``detections`` + ``quality``).

    ``None`` both when ``result`` itself is falsy (no linked analysis -- the check-in router's own
    "no analysis_id" case never even calls this) and when the clip inside it was never measured --
    the two callers of this function never need to tell those apart.
    """
    if not result:
        return None
    return form_score(result.get("detections"), result.get("quality"))
