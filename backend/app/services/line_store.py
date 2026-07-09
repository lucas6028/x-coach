"""Service-role persistence for the LINE bot: link codes + LINE↔analysis bindings.

Unlike ``services/store`` (user-JWT + RLS), the LINE webhook is called by LINE with **no user
token**, so this store authenticates with the Supabase **service_role** key and talks to two
LINE-only tables:

  * ``line_link_codes`` — short-lived one-time codes a signed-in web user generates to connect a
    LINE account to a specific analysis. Each row snapshots the analysis grounding ``context`` so
    the webhook never has to read the user's RLS-protected ``analyses``/``conversations`` tables.
  * ``line_bindings``   — one row per LINE user: which web ``user_id``/``video_id`` they're bound
    to, the grounding ``context`` copied from the redeemed code, and the running ``messages`` thread.

Both tables deny all ``anon``/``authenticated`` access (see the migration); only this store, via the
service_role key, touches them. Keeping this module's queries strictly on the two ``line_*`` tables
is the discipline that bounds the service_role blast radius — it must never read core user data.

The ``supabase`` import is deferred into ``_service_client`` (mirroring ``services/store``) so the
routers import cheaply and the unit tests patch ``_service_client`` without the package installed.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

# Human-friendly one-time code: an unambiguous alphabet (no 0/O/1/I/L) so a user can retype it into
# LINE without confusion. 6 chars over a 30-symbol alphabet ≈ 7e8 combinations — ample for codes
# that live only minutes and are deleted on first use.
_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_CODE_LENGTH = 6

# A link code is single-use and short-lived: long enough for the user to switch to the LINE app and
# type it, short enough that a leaked code is useless. Redeeming it deletes it regardless.
_CODE_TTL = timedelta(minutes=15)


def _service_client() -> Any:
    """Build a Supabase client authenticated with the service_role key (bypasses RLS).

    Deferred heavy import, same pattern as ``services/store._user_client``. The service_role key is
    server-side only; this client is used ONLY against the ``line_*`` tables in this module.
    """
    from supabase import create_client  # deferred heavy import (gotrue/postgrest/...)

    from backend.app.settings import get_settings

    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def _now() -> datetime:
    """Current UTC time (isolated so tests can monkeypatch a fixed clock)."""
    return datetime.now(timezone.utc)


def _new_code() -> str:
    """A fresh random link code over the unambiguous alphabet."""
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LENGTH))


def create_link_code(*, user_id: str, video_id: str | None, context: dict[str, Any]) -> str:
    """Create a one-time link code for ``user_id`` snapshotting ``context``; return the code.

    Called from the auth-gated ``POST /api/line/link-code``. The analysis grounding blob is stored on
    the code row so that, once redeemed, the LINE binding is fully self-contained and the webhook
    never needs the user's JWT. Codes are not deduplicated per user by design — generating a new one
    (e.g. after re-analysing) just supersedes the old, and both expire.
    """
    client = _service_client()
    code = _new_code()
    client.table("line_link_codes").insert(
        {
            "code": code,
            "user_id": user_id,
            "video_id": video_id,
            "context": context,
            "expires_at": (_now() + _CODE_TTL).isoformat(),
        }
    ).execute()
    return code


def redeem_link_code(*, line_user_id: str, code: str) -> dict[str, Any] | None:
    """Bind ``line_user_id`` to the analysis behind ``code``; return the new binding, or ``None``.

    Looks the code up case-insensitively (users may retype in lower case), rejects an expired one,
    upserts the ``line_bindings`` row (resetting the message thread to the new analysis), then deletes
    the code so it can't be reused. ``None`` means the code was unknown or expired — the caller replies
    with a "code invalid" message.
    """
    normalized = (code or "").strip().upper()
    if not normalized:
        return None

    client = _service_client()
    resp = (
        client.table("line_link_codes")
        .select("code, user_id, video_id, context, expires_at")
        .eq("code", normalized)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return None
    row = rows[0]

    expires_at = _parse_ts(row.get("expires_at"))
    if expires_at is not None and expires_at < _now():
        client.table("line_link_codes").delete().eq("code", normalized).execute()  # tidy the stale code
        return None

    binding = {
        "line_user_id": line_user_id,
        "user_id": row["user_id"],
        "video_id": row.get("video_id"),
        "context": row.get("context") or {},
        "messages": [],  # a fresh thread each time a (re)binding happens
    }
    client.table("line_bindings").upsert(binding, on_conflict="line_user_id").execute()
    client.table("line_link_codes").delete().eq("code", normalized).execute()  # one-time use
    return binding


def get_binding(*, line_user_id: str) -> dict[str, Any] | None:
    """Return the binding for ``line_user_id`` (``{user_id, video_id, context, messages}``), or
    ``None`` when the LINE user has not linked yet."""
    client = _service_client()
    resp = (
        client.table("line_bindings")
        .select("user_id, video_id, context, messages")
        .eq("line_user_id", line_user_id)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return None
    row = rows[0]
    return {
        "user_id": row.get("user_id"),
        "video_id": row.get("video_id"),
        "context": row.get("context") or {},
        "messages": row.get("messages") or [],
    }


def save_binding_messages(*, line_user_id: str, messages: list[dict[str, Any]]) -> None:
    """Persist the running chat thread for a bound LINE user (whole-array update, like the web
    conversations store). Only ``messages`` changes; ``user_id``/``context`` stay as bound."""
    client = _service_client()
    client.table("line_bindings").update({"messages": messages}).eq(
        "line_user_id", line_user_id
    ).execute()


def _parse_ts(value: Any) -> datetime | None:
    """Parse a Postgres/ISO timestamp into an aware UTC datetime, or ``None`` if unparseable.

    PostgREST returns ``timestamptz`` as ISO-8601; tolerate a trailing ``Z`` and naive values (assumed
    UTC) so an odd serialization never crashes the redeem path — an unparseable expiry is treated as
    "no expiry information", so the code is accepted rather than silently dropped.
    """
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
