"""Clinic persistence + pure aggregation: the therapist -> patient care loop (WP1, "clinic core").

Same contract as ``store.py`` and ``plans.py``: every I/O call runs through a Supabase client
authenticated with the CALLER'S OWN JWT (``_user_client``, imported from ``store`` -- one place
builds the client, one place the unit tests patch), so PostgREST executes as that user and the
``db/migrations/20260916000000_clinic.sql`` RLS policies are the ownership/consent backstop. The
backend holds no service_role key on this path: a clinician sees a patient's rows only because an
ACTIVE ``care_links`` row makes ``is_linked(auth.uid(), patient_id)`` true in Postgres, not because
this module decided to trust a role check alone.

This module is split in two halves on purpose:
  * I/O functions (``create_invite``, ``list_invites``, ``accept_invite``, ``revoke_link``,
    ``list_patients``, ``my_clinicians``, ``list_checkins``, ``batch_patient_data``) talk to
    Supabase and are exercised against the ``_FakeDb`` pattern from ``tests/test_plans_store.py``.
  * Aggregation functions (``last_checkin_at``, ``adherence_7d``, ``open_flags``, ``inactive_7d``,
    plus the small views ``recent_checkins``/``trend``/``open_flag_rows``) are PURE: no I/O, ``now``
    passed in by the caller so tests can pin the clock and the 7-day/Asia-Taipei boundaries are
    exercised directly, without a fake database in the way.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.app.services.store import _user_client

# Codes are drawn from an alphabet with the visually-ambiguous characters removed (no I/1/O/0),
# so a clinician reading a code aloud, or a patient copying it off a phone screen, doesn't trip
# over a glyph that looks like another one. 6 characters over a 33-symbol alphabet is ~30 bits of
# entropy per code -- plenty against a 7-day expiry and no online guessing surface (accept_care_invite
# is one exact-match lookup, not enumerable).
_INVITE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_INVITE_CODE_LEN = 6
_MAX_INVITE_ATTEMPTS = 5

# A LINE-login account's email is a synthetic ``line_<sub>@line.invalid`` placeholder (the existing
# identity key -- see ``services/line_bot.py``), never a real address. Surfacing it as "the
# patient's email" would be actively misleading, so the display helper blanks it.
_SYNTHETIC_EMAIL_SUFFIX = "@line.invalid"

# Taipei has no DST, so a fixed UTC+8 offset is exactly correct year-round -- and it avoids
# ``zoneinfo``, which needs the `tzdata` package on Windows and is not in the lean CI dependency
# set (see ``services/line_bot.py``'s ``_DISPLAY_TZ`` for the same reasoning).
_TAIPEI = timezone(timedelta(hours=8))

# Columns for a linked patient's check-ins, read as the clinician. Full row (not a lean subset):
# the dashboard's ``open_flags`` count needs only ``flagged``/``acknowledged_at`` but the patient
# detail page needs the rest (pain/rpe/note/form_score/trend/flag_reasons), and fetching the full
# row once is simpler than keeping two column lists in lockstep.
_CHECKIN_COLUMNS = (
    "id, user_id, plan_id, plan_item_id, analysis_id, pain_nrs, rpe, note, form_score, "
    "flagged, flag_reasons, acknowledged_at, acknowledged_by, created_at"
)
# Minimal plan/item columns for the AGGREGATION math only (adherence/last-checkin/inactive). The
# full plan display (name, notes, template_key, ...) goes through ``services/plans.list_plans``
# instead -- this is a separate, narrower read so the dashboard's batched query stays cheap.
_CLINIC_PLAN_COLUMNS = "id, user_id, created_at, assigned_by"
_CLINIC_ITEM_COLUMNS = "plan_id, user_id, day_index"


# ---------------------------------------------------------------------------------------------
# Invites: a clinician generates a short code; the patient consumes it via ``care.accept_invite``.
# ---------------------------------------------------------------------------------------------


def _generate_code() -> str:
    return "".join(secrets.choice(_INVITE_ALPHABET) for _ in range(_INVITE_CODE_LEN))


def create_invite(*, token: str, clinician_id: str) -> dict[str, Any]:
    """Insert a fresh pending ``care_links`` row with a random 6-char code; return the row.

    ``invite_code`` is globally unique (across all time, used and revoked codes included -- see
    the migration), so a fresh random draw can collide with a still-pending code from anyone. On a
    unique-violation (Postgres SQLSTATE ``23505``, read with ``getattr`` so this needs no
    ``postgrest``/``postgrest`` exception import and the tests can raise a plain stand-in), retry
    with a new draw, up to ``_MAX_INVITE_ATTEMPTS`` times, then give up loudly rather than loop
    forever. Any OTHER exception is not retried -- a collision is the only failure this function
    knows how to recover from.
    """
    client = _user_client(token)
    last_exc: Exception | None = None
    for _ in range(_MAX_INVITE_ATTEMPTS):
        code = _generate_code()
        try:
            resp = (
                client.table("care_links")
                .insert(
                    {
                        "clinician_id": clinician_id,
                        "invite_code": code,
                        "status": "pending",
                    }
                )
                .execute()
            )
        except Exception as exc:  # noqa: BLE001 — only a 23505 collision is swallowed and retried.
            if getattr(exc, "code", None) == "23505":
                last_exc = exc
                continue
            raise
        rows = resp.data or []
        if not rows:
            raise RuntimeError("Invite insert returned no row.")
        return rows[0]
    raise RuntimeError(
        f"Could not generate a unique invite code after {_MAX_INVITE_ATTEMPTS} attempts."
    ) from last_exc


def list_invites(*, token: str, clinician_id: str) -> list[dict[str, Any]]:
    """The caller's own pending, unexpired invites, newest first.

    Filtered explicitly by ``clinician_id`` and ``status`` even though the ``care_links`` SELECT
    policy already scopes rows to a party of the link -- the predicate is what turns "every link I
    am a party to" into "the invites I could still hand out", the same belt-and-braces pattern as
    ``plans.list_plans``'s explicit ``user_id`` filter under an RLS-scoped read.
    """
    client = _user_client(token)
    now = datetime.now(timezone.utc).isoformat()
    resp = (
        client.table("care_links")
        .select("id, clinician_id, invite_code, status, created_at, expires_at")
        .eq("clinician_id", clinician_id)
        .eq("status", "pending")
        .gte("expires_at", now)
        .order("created_at", desc=True)
        .execute()
    )
    return resp.data or []


# ---------------------------------------------------------------------------------------------
# Link lifecycle + display. accept/revoke are thin wrappers over the SECURITY DEFINER RPCs --
# see the migration header for why care_links itself has no UPDATE policy.
# ---------------------------------------------------------------------------------------------


def accept_invite(*, token: str, code: str) -> dict[str, Any]:
    """Call ``accept_care_invite(code)``; return ``{"link": {...}}`` or ``{"error": "invalid"|"self"}``."""
    client = _user_client(token)
    resp = client.rpc("accept_care_invite", {"p_code": code}).execute()
    return resp.data or {}


def revoke_link(*, token: str, link_id: str) -> dict[str, Any]:
    """Call ``revoke_care_link(id)``; return ``{"link": {...}}`` or ``{"error": "not_found"}``.

    Works for either party (clinician or patient) and for a still-pending invite -- the RPC itself
    decides who may revoke which row; this is a thin, role-agnostic wrapper reused by both
    ``routers/clinic.py`` and ``routers/care.py``.
    """
    client = _user_client(token)
    resp = client.rpc("revoke_care_link", {"p_link": link_id}).execute()
    return resp.data or {}


def _apply_display(row: dict[str, Any]) -> dict[str, Any]:
    """Blank a synthetic LINE-login email and fall back ``display_name`` to a real one.

    A LINE-login account's ``email`` is a synthetic ``line_<sub>@line.invalid`` placeholder (never
    shown to a human), so a row carrying one gets ``email: None`` outright. Otherwise, when there
    is no profile name, ``display_name`` falls back to the (real) email so the dashboard never
    shows a blank row.
    """
    row = dict(row)
    email = row.get("email")
    synthetic = bool(email) and email.endswith(_SYNTHETIC_EMAIL_SUFFIX)
    if synthetic:
        row["email"] = None
    elif not row.get("display_name"):
        row["display_name"] = email
    return row


def list_patients(*, token: str) -> list[dict[str, Any]]:
    """The caller's linked (active) patients via the ``clinic_patients()`` definer RPC, display-mapped.

    The RPC itself raises SQLSTATE 42501 for a non-clinician -- this function does not re-check the
    role; every caller of it is already behind ``get_clinician_user``.
    """
    client = _user_client(token)
    resp = client.rpc("clinic_patients").execute()
    return [_apply_display(row) for row in (resp.data or [])]


def my_clinicians(*, token: str) -> list[dict[str, Any]]:
    """The caller's linked (active) clinicians via the ``my_clinicians()`` definer RPC, display-mapped."""
    client = _user_client(token)
    resp = client.rpc("my_clinicians").execute()
    return [_apply_display(row) for row in (resp.data or [])]


def find_patient(*, token: str, patient_id: str) -> dict[str, Any] | None:
    """One row of ``list_patients`` matching ``patient_id``, or ``None`` if not (actively) linked.

    This is the 404 boundary for every ``/api/clinic/patients/{id}...`` route: a patient id that is
    not on the caller's ``clinic_patients()`` list answers 404, whether it never existed, was never
    linked to this clinician, or the link was revoked -- RLS makes those indistinguishable by
    construction, the same reasoning as ``store.get_analysis``.
    """
    for row in list_patients(token=token):
        if row.get("patient_id") == patient_id:
            return row
    return None


# ---------------------------------------------------------------------------------------------
# Batched cross-patient reads for the dashboard, plus the per-patient check-in read for the
# detail page. Both are I/O; the VIEWS derived from their output (below) are pure.
# ---------------------------------------------------------------------------------------------


def batch_patient_data(
    *, token: str, patient_ids: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """One plans query, one items query, one check-ins query covering EVERY id in ``patient_ids``.

    Used by ``GET /api/clinic/patients`` (one round trip per table for N patients, not one round
    trip per patient per table) and reused as-is by ``GET /api/clinic/patients/{id}`` with a
    single-element list, so the aggregation math never has two code paths. Every read still goes
    through the caller's own JWT: the ``is_linked()`` RLS policies are what actually scope these
    rows to patients who accepted THIS clinician's invite, not the ``in_`` filter.
    """
    if not patient_ids:
        return [], [], []
    client = _user_client(token)
    plans = (
        client.table("training_plans")
        .select(_CLINIC_PLAN_COLUMNS)
        .in_("user_id", patient_ids)
        .execute()
        .data
        or []
    )
    items = (
        client.table("plan_items")
        .select(_CLINIC_ITEM_COLUMNS)
        .in_("user_id", patient_ids)
        .execute()
        .data
        or []
    )
    checkins = (
        client.table("session_checkins")
        .select(_CHECKIN_COLUMNS)
        .in_("user_id", patient_ids)
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )
    return plans, items, checkins


def list_checkins(*, token: str, patient_id: str) -> list[dict[str, Any]]:
    """All of one linked patient's check-ins, newest first.

    Equivalent to ``batch_patient_data(patient_ids=[patient_id])[2]`` restricted to just the
    check-ins query; kept as its own function because the detail-page router reads plans through
    ``plans.list_plans`` instead (for the fuller column set) and only needs this one table here.
    """
    client = _user_client(token)
    resp = (
        client.table("session_checkins")
        .select(_CHECKIN_COLUMNS)
        .eq("user_id", patient_id)
        .order("created_at", desc=True)
        .execute()
    )
    return resp.data or []


# ---------------------------------------------------------------------------------------------
# Pure aggregation. No I/O; ``now`` is always passed in so callers (and tests) control the clock.
# Every function here takes data ALREADY SCOPED to one patient -- callers filter by patient_id
# themselves (the batched reads above return cross-patient rows).
# ---------------------------------------------------------------------------------------------


def _parse_ts(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, defaulting a naive one to UTC (PostgREST always sends tz-aware,
    but a hand-built test fixture is easy to leave naive)."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _current_prescription(plans: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The patient's most recently CREATED plan with ``assigned_by`` not null, or ``None``.

    This is "the current prescription" every other aggregation rule below is defined against. A
    self-authored plan (``assigned_by`` null) never counts, even if it is newer -- adherence and
    the inactivity flag are about what the CLINICIAN assigned, not what the patient built alone.
    """
    assigned = [p for p in plans if p.get("assigned_by")]
    if not assigned:
        return None
    return max(assigned, key=lambda p: p.get("created_at") or "")


def last_checkin_at(checkins: list[dict[str, Any]]) -> str | None:
    """The patient's newest check-in ``created_at`` (as the original ISO string), or ``None``."""
    timestamps = [c.get("created_at") for c in checkins if c.get("created_at")]
    if not timestamps:
        return None
    return max(timestamps, key=_parse_ts)


def adherence_7d(
    plans: list[dict[str, Any]],
    items: list[dict[str, Any]],
    checkins: list[dict[str, Any]],
    *,
    now: datetime,
) -> float | None:
    """7-day adherence against the current prescription: ``round(min(1, done / planned), 2)``.

    ``planned`` = the number of DISTINCT ``day_index`` values among that plan's items (how many
    day-slots the prescription actually uses). ``done`` = the number of distinct Asia/Taipei
    CALENDAR DATES in the half-open window ``(now - 7d, now]`` that have at least one check-in
    tied to that same plan (``plan_id`` match) -- a patient who does two sessions in one Taipei day
    still only earns one day of credit, and a session logged just after Taipei midnight counts for
    the new day, not the old one, which is the whole reason this buckets in Taipei time rather than
    UTC or server time.

    ``None`` when there is no assigned plan, or the assigned plan has no items (``planned == 0``,
    which would otherwise divide by zero and is also just not a measurable prescription).
    """
    plan = _current_prescription(plans)
    if plan is None:
        return None
    plan_id = plan.get("id")

    planned = len(
        {
            it.get("day_index")
            for it in items
            if it.get("plan_id") == plan_id and it.get("day_index") is not None
        }
    )
    if planned == 0:
        return None

    window_start = now - timedelta(days=7)
    done_dates = set()
    for c in checkins:
        if c.get("plan_id") != plan_id:
            continue
        created = c.get("created_at")
        if not created:
            continue
        dt = _parse_ts(created)
        if window_start < dt <= now:  # half-open: the start instant itself does not count.
            done_dates.add(dt.astimezone(_TAIPEI).date())

    return round(min(1.0, len(done_dates) / planned), 2)


def open_flags(checkins: list[dict[str, Any]]) -> int:
    """Count of check-ins that are flagged AND not yet acknowledged by a clinician."""
    return sum(1 for c in checkins if c.get("flagged") and not c.get("acknowledged_at"))


def inactive_7d(
    plans: list[dict[str, Any]],
    checkins: list[dict[str, Any]],
    *,
    now: datetime,
) -> bool:
    """Red-flag rule (d): the current prescription is stale -- assigned a week+ ago, untouched since.

    True exactly when ALL of: a current prescription exists; it was created at least 7 days before
    ``now``; and it has no check-in (tied to that plan) in the half-open window ``(now - 7d, now]``.
    A brand-new prescription (assigned within the last 7 days) is never flagged, even with zero
    check-ins yet -- the patient hasn't had a full week to fall behind on it.
    """
    plan = _current_prescription(plans)
    if plan is None:
        return False
    created = plan.get("created_at")
    if not created:
        return False

    window_start = now - timedelta(days=7)
    if _parse_ts(created) > window_start:  # assigned less than 7 days ago -> too new to flag.
        return False

    plan_id = plan.get("id")
    for c in checkins:
        if c.get("plan_id") != plan_id:
            continue
        created_c = c.get("created_at")
        if not created_c:
            continue
        dt = _parse_ts(created_c)
        if window_start < dt <= now:
            return False  # a recent check-in on this plan exists -> not inactive.
    return True


def recent_checkins(checkins: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    """The newest ``limit`` check-ins. ``checkins`` is expected already newest-first (query order)."""
    return checkins[:limit]


def trend(
    checkins: list[dict[str, Any]], *, now: datetime, days: int = 30
) -> list[dict[str, Any]]:
    """``[{created_at, form_score, pain_nrs}]`` for the last ``days`` days, OLDEST first (chart order)."""
    cutoff = now - timedelta(days=days)
    rows = [
        {
            "created_at": c["created_at"],
            "form_score": c.get("form_score"),
            "pain_nrs": c.get("pain_nrs"),
        }
        for c in checkins
        if c.get("created_at") and _parse_ts(c["created_at"]) >= cutoff
    ]
    rows.sort(key=lambda r: r["created_at"])
    return rows


def open_flag_rows(checkins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The full rows behind ``open_flags`` — flagged and not yet acknowledged."""
    return [c for c in checkins if c.get("flagged") and not c.get("acknowledged_at")]


# ---------------------------------------------------------------------------------------------
# Check-in flag acknowledgement (WP2). A thin RPC wrapper, same shape as accept_invite/
# revoke_link: the RPC itself (ack_checkin_flag, see the migration) is the security boundary --
# only a clinician actively linked to the check-in's owner may act on it -- this function does not
# re-check that.
# ---------------------------------------------------------------------------------------------


def ack_checkin_flag(*, token: str, checkin_id: str) -> dict[str, Any]:
    """Call ``ack_checkin_flag(id)``; return ``{"checkin": {...}}`` or
    ``{"error": "not_found" | "not_flagged"}``. Idempotent on the Postgres side -- see the
    migration's docstring for ``ack_checkin_flag``."""
    client = _user_client(token)
    resp = client.rpc("ack_checkin_flag", {"p_checkin": checkin_id}).execute()
    return resp.data or {}
