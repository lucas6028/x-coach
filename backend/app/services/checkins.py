"""Persistence for ``session_checkins`` (WP2): a patient's post-session pain/RPE report, the
server-computed form score of its linked analysis, and any red flags it trips.

Same contract as ``services/plans.py`` and ``services/clinic.py``: every I/O call runs through a
Supabase client authenticated with the CALLER'S OWN JWT (``_user_client``, imported from ``store``
rather than redefined -- one place builds the client, one place the unit tests patch), so the
``session_checkins`` RLS policies (insert: owner only, with ``acknowledged_*`` null; select: the
owner, or a linked clinician) are the ownership backstop, not this module. There is no update or
delete policy on this table at all -- see the migration header for why the one mutation (a
clinician acknowledging a flag) goes through the ``ack_checkin_flag`` RPC instead
(``services/clinic.ack_checkin_flag``), not a function here.
"""

from __future__ import annotations

from typing import Any

from backend.app.services.store import _user_client

# Every column the API returns for one check-in, spelled out rather than "*" for the same reason as
# plans.py's _PLAN_COLUMNS: a column added later does not silently start appearing in responses.
_CHECKIN_COLUMNS = (
    "id, user_id, plan_id, plan_item_id, analysis_id, pain_nrs, rpe, note, form_score, "
    "flagged, flag_reasons, acknowledged_at, acknowledged_by, created_at"
)

# How many of the caller's own prior check-ins on one plan ``redflags.evaluate`` is handed as
# history. Rule (c) needs 5 SCORED rows before the new one, and check-ins reported from the plan
# page without a studio clip have no form score; 30 keeps a month of daily reports in the window
# so interleaved unscored rows cannot hide a drop, without pulling the patient's entire history.
HISTORY_LIMIT = 30


def recent_checkins(
    *, token: str, user_id: str, plan_id: str, limit: int = HISTORY_LIMIT
) -> list[dict[str, Any]]:
    """The caller's own last ``limit`` check-ins on ONE plan, newest first -- the ``history`` input
    to ``redflags.evaluate``.

    Both ``user_id`` and ``plan_id`` are filtered EXPLICITLY even though the table's SELECT policy
    already scopes a read to rows the caller may see: since the clinic migration that also includes
    a linked clinician reading a PATIENT's check-ins, so RLS alone is no longer "mine" -- the same
    reasoning as ``store.list_analyses``'s explicit ``user_id`` predicate.
    """
    client = _user_client(token)
    resp = (
        client.table("session_checkins")
        .select(_CHECKIN_COLUMNS)
        .eq("user_id", user_id)
        .eq("plan_id", plan_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data or []


def create_checkin(*, token: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Insert one check-in row; return it as stored (``id``/``created_at`` filled in by Postgres).

    ``payload`` is assembled entirely by the router -- this function trusts its caller the same way
    ``plans.add_item`` does, since the column set (and which fields the RLS insert policy permits at
    all -- ``acknowledged_at``/``acknowledged_by`` must stay null) is the router's business, not a
    second copy of it here.
    """
    client = _user_client(token)
    resp = client.table("session_checkins").insert(payload).execute()
    rows = resp.data or []
    if not rows:
        raise RuntimeError("Check-in insert returned no row.")
    return rows[0]


def list_checkins(
    *, token: str, user_id: str, plan_id: str | None = None, limit: int = 30
) -> list[dict[str, Any]]:
    """The caller's own check-ins, newest first, optionally scoped to one plan.

    ``user_id`` is filtered explicitly for the same reason as ``recent_checkins`` -- the clinician
    SELECT policy means "rows I can read" is no longer "rows that are mine". ``plan_id`` is applied
    only when given, so an unscoped ``GET /api/checkins`` returns the caller's whole history.
    """
    client = _user_client(token)
    query = client.table("session_checkins").select(_CHECKIN_COLUMNS).eq("user_id", user_id)
    if plan_id is not None:
        query = query.eq("plan_id", plan_id)
    resp = query.order("created_at", desc=True).limit(limit).execute()
    return resp.data or []
