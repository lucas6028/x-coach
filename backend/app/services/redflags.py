"""Deterministic red-flag rules (WP2, §1 "Red-flag rules" of the 20-day plan): a PURE function over
one new check-in and the patient's own prior check-ins on the same plan. No model, no I/O -- the
router (``routers/checkins.py``) fetches the history and passes it in, the same split
``services/clinic.py`` uses between its I/O reads and its pure aggregation functions.

Rules implemented here, in the fixed order the reasons list is returned:
  (a) ``pain_high``  -- the new check-in's ``pain_nrs`` is >= ``PAIN_HIGH_THRESHOLD`` on its own.
  (b) ``pain_rise``  -- pain rose by >= ``PAIN_RISE_DELTA`` versus the most recent EARLIER
      check-in on the SAME plan (no earlier check-in on this plan -> the rule cannot fire).
  (c) ``form_drop``  -- evaluated ONLY when the new check-in has a form score. Takes every
      SCORED (non-null ``form_score``) check-in on the plan, oldest to newest, appends the new
      one, and flags when there are at least ``FORM_DROP_MIN_SCORED`` of them AND the mean of the
      last ``FORM_DROP_WINDOW`` is at least ``FORM_DROP_THRESHOLD`` points below the mean of the
      ``FORM_DROP_WINDOW`` before them. Fewer scored check-ins than the minimum never flags --
      "the last 3 sessions" is not a meaningful comparison with only one or two on record.

Rule (d) -- 7 days without a check-in on the current assigned plan -- is NOT here: it depends on
"now" and the current prescription rather than on one new check-in event, and is computed on read
for the clinician dashboard instead (``services/clinic.inactive_7d``, WP1).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Rule (a): pain at or above this threshold is flagged on its own -- no history needed.
PAIN_HIGH_THRESHOLD = 6
# Rule (b): a rise of at least this many points versus the previous check-in on the same plan.
PAIN_RISE_DELTA = 3
# Rule (c): the scored series (history + the new check-in) must reach this length before
# form_drop can ever fire.
FORM_DROP_MIN_SCORED = 6
# Rule (c): compare the mean of the last WINDOW scored check-ins against the WINDOW immediately
# before them.
FORM_DROP_WINDOW = 3
# Rule (c): flag when the mean falls by at least this many points.
FORM_DROP_THRESHOLD = 20


def _parse_ts(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, defaulting a naive one to UTC -- same reasoning as
    ``services/clinic.py``'s ``_parse_ts``: PostgREST always sends tz-aware timestamps, but a
    hand-built test fixture is easy to leave naive."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def evaluate(new_checkin: dict[str, Any], history: list[dict[str, Any]]) -> list[str]:
    """The reason codes ``new_checkin`` trips, given the caller's OWN prior check-ins.

    ``history`` may span multiple plans -- rows whose ``plan_id`` does not match
    ``new_checkin["plan_id"]`` are ignored HERE rather than trusted to be pre-filtered by the
    caller, so a caller that forgets to scope its query cannot silently over- or under-flag.
    ``new_checkin`` need not carry ``created_at`` (the router calls this BEFORE the insert that
    assigns one): it is always treated as the newest entry, both for the pain-rise comparison and
    at the end of the form-score series.
    """
    plan_id = new_checkin.get("plan_id")
    same_plan = [c for c in history if c.get("plan_id") == plan_id]

    pain_nrs = new_checkin.get("pain_nrs")
    pain_high = pain_nrs is not None and pain_nrs >= PAIN_HIGH_THRESHOLD

    dated_history = [c for c in same_plan if c.get("created_at")]
    pain_rise = False
    if dated_history and pain_nrs is not None:
        previous = max(dated_history, key=lambda c: _parse_ts(c["created_at"]))
        prev_pain = previous.get("pain_nrs")
        pain_rise = prev_pain is not None and (pain_nrs - prev_pain) >= PAIN_RISE_DELTA

    form_drop = False
    new_form_score = new_checkin.get("form_score")
    if new_form_score is not None:
        scored = sorted(
            (
                c
                for c in same_plan
                if c.get("form_score") is not None and c.get("created_at")
            ),
            key=lambda c: _parse_ts(c["created_at"]),
        )
        series = scored + [new_checkin]
        if len(series) >= FORM_DROP_MIN_SCORED:
            last = series[-FORM_DROP_WINDOW:]
            before = series[-2 * FORM_DROP_WINDOW : -FORM_DROP_WINDOW]
            mean_last = sum(c["form_score"] for c in last) / FORM_DROP_WINDOW
            mean_before = sum(c["form_score"] for c in before) / FORM_DROP_WINDOW
            form_drop = (mean_before - mean_last) >= FORM_DROP_THRESHOLD

    reasons: list[str] = []
    if pain_high:
        reasons.append("pain_high")
    if pain_rise:
        reasons.append("pain_rise")
    if form_drop:
        reasons.append("form_drop")
    return reasons
