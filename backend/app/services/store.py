"""Persistence service: read/write per-user video + analysis rows in Supabase Postgres.

Uses **supabase-py in user-JWT mode**: each call builds a client authenticated with the
caller's access token (``postgrest.auth(token)``), so PostgREST runs the query as that user
and the RLS policies from the schema migration scope every row to ``auth.uid() = user_id``.
The backend never holds a service_role key, so a bug here cannot leak across users — the DB
is the backstop.

The ``supabase`` import is deferred into ``_user_client`` so importing this module (and the
routers that use it) stays light, and the unit tests can patch ``_user_client`` without the
``supabase`` package installed.
"""

from __future__ import annotations

from typing import Any


def _user_client(token: str) -> Any:
    """Build a Supabase client acting as the user identified by ``token`` (RLS-scoped)."""
    from supabase import create_client  # deferred heavy import (gotrue/postgrest/...)

    from backend.app.settings import get_settings

    settings = get_settings()
    client = create_client(settings.supabase_url, settings.supabase_anon_key)
    client.postgrest.auth(token)
    return client


def _summarize(result: dict[str, Any]) -> tuple[str | None, int, str | None, str | None]:
    """Promote the list-view columns out of the nested analysis document."""
    view_type = (result.get("view") or {}).get("view_type")
    fault_count = len(result.get("detections") or [])
    pipeline_version = result.get("pipeline_version")
    # Which detector produced this. Null for rows predating per-movement analysis; the frontend
    # falls back to result["movement"], then to Squat, rather than guessing here.
    movement = result.get("movement")
    return view_type, fault_count, pipeline_version, movement


def is_admin(*, token: str, user_id: str) -> bool:
    """Return whether ``user_id`` holds the 'admin' role, queried as the user (RLS-scoped).

    Uses the caller's own JWT: the ``user_roles`` SELECT policy lets any authenticated user read the
    table, so a user can learn whether they themselves are admin without a service_role key. A
    patchable seam — the unit tests replace ``_user_client``.
    """
    client = _user_client(token)
    resp = (
        client.table("user_roles")
        .select("user_id")
        .eq("user_id", user_id)
        .eq("role", "admin")
        .limit(1)
        .execute()
    )
    return bool(resp.data)


def count_admins(*, token: str) -> int:
    """Return how many users currently hold the 'admin' role.

    Used by the role PUT's last-admin guard to refuse a revoke that would leave zero admins. This
    goes through the ``count_admins()`` SECURITY DEFINER RPC, NOT a direct table read: the tightened
    ``user_roles`` SELECT policy scopes an authenticated caller to their OWN row, so a plain
    ``.table("user_roles")`` count would always return 1 for the acting admin and wrongly block every
    demotion. The definer function bypasses RLS and returns the true total. A patchable seam — the
    unit tests replace ``_user_client``.
    """
    client = _user_client(token)
    resp = client.rpc("count_admins").execute()
    return int(resp.data or 0)


def get_app_settings(*, token: str) -> dict[str, Any]:
    """Return every ``app_settings`` override as ``{key: value}``, read as the caller (RLS-scoped).

    Used by the admin ``GET /api/admin/settings`` handler to show the currently-persisted overrides.
    The table's SELECT policy lets any authenticated user read it; the admin gate is enforced by the
    endpoint's ``get_admin_user`` dependency. A patchable seam — the unit tests replace ``_user_client``.
    """
    client = _user_client(token)
    resp = client.table("app_settings").select("key, value").execute()
    return {row["key"]: row["value"] for row in (resp.data or []) if "key" in row}


def upsert_app_settings(*, token: str, items: dict[str, Any]) -> None:
    """Upsert the given ``{key: value}`` overrides into ``app_settings`` as the caller (RLS-scoped).

    Writes are gated in Postgres by the ``is_admin(auth.uid())`` INSERT/UPDATE policies, so a
    non-admin's write is rejected by RLS even though the endpoint already checks ``get_admin_user``.
    Only the provided keys are touched; omitted knobs keep their previous value (or their env default
    if never set). A patchable seam — the unit tests replace ``_user_client``.
    """
    if not items:
        return
    rows = [{"key": key, "value": value} for key, value in items.items()]
    client = _user_client(token)
    client.table("app_settings").upsert(rows, on_conflict="key").execute()


def admin_list_users(*, token: str) -> list[dict[str, Any]]:
    """Return one row per user (id, email, activity counts, is_admin) for the admin overview.

    Calls the ``admin_list_users()`` SECURITY DEFINER function with the caller's own JWT: the function
    gates itself internally on ``is_admin(auth.uid())`` (raising 42501 for a non-admin), so no
    service_role key is needed to read ``auth.users``. A patchable seam — the unit tests replace
    ``_user_client``. The endpoint's ``get_admin_user`` dependency is the frontline gate; the function's
    internal guard is the Postgres-side backstop.
    """
    client = _user_client(token)
    resp = client.rpc("admin_list_users").execute()
    return resp.data or []


def set_user_role(*, token: str, user_id: str, make_admin: bool) -> None:
    """Grant or revoke the 'admin' role for ``user_id``, written as the caller (RLS-scoped).

    Writes to ``user_roles`` are gated in Postgres by the ``is_admin(auth.uid())`` INSERT/DELETE
    policies, so only an admin's write lands even though the endpoint already checks ``get_admin_user``.
    ``make_admin`` → upsert the role row (idempotent on ``user_id``); otherwise → delete it. A patchable
    seam — the unit tests replace ``_user_client``.
    """
    client = _user_client(token)
    if make_admin:
        client.table("user_roles").upsert(
            {"user_id": user_id, "role": "admin"}, on_conflict="user_id"
        ).execute()
    else:
        client.table("user_roles").delete().eq("user_id", user_id).execute()


def persist_analysis(
    *,
    token: str,
    user_id: str,
    video_id: str,
    source: str,
    result: dict[str, Any],
    filename: str | None = None,
) -> str:
    """Upsert the video row and insert the analysis; return the new analysis id.

    The ``videos`` row carries the (currently trivial) status machine and the storage key,
    which P2 will repoint at object storage. ``result`` is stored verbatim as JSONB so history
    replay is self-contained.
    """
    client = _user_client(token)

    client.table("videos").upsert(
        {
            "user_id": user_id,
            "video_id": video_id,
            "filename": filename,
            "storage_key": f"runtime/uploads/{video_id}",
            "status": "done",
        },
        on_conflict="user_id,video_id",
    ).execute()

    view_type, fault_count, pipeline_version, movement = _summarize(result)
    resp = (
        client.table("analyses")
        .insert(
            {
                "user_id": user_id,
                "video_id": video_id,
                "source": source,
                "view_type": view_type,
                "fault_count": fault_count,
                "pipeline_version": pipeline_version,
                "movement": movement,
                "result": result,
            }
        )
        .execute()
    )
    rows = resp.data or []
    return str(rows[0]["id"]) if rows else ""


def list_analyses(*, token: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    """Return the caller's analyses (newest first) as ``{"total", "items"}`` for the history page."""
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    client = _user_client(token)
    resp = (
        client.table("analyses")
        .select(
            "id, video_id, source, view_type, fault_count, movement, created_at",
            count="exact",
        )
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    return {"total": resp.count or 0, "items": resp.data or []}


def delete_all_analyses(*, token: str, user_id: str) -> int:
    """Delete every analysis (and source video row) owned by the caller; return how many analyses
    were removed.

    RLS already scopes writes to ``auth.uid() = user_id``, but we also filter by ``user_id``
    explicitly: PostgREST refuses an unfiltered bulk delete, and the predicate is a second guard.
    """
    client = _user_client(token)
    resp = client.table("analyses").delete().eq("user_id", user_id).execute()
    # Drop the (now orphaned) source video rows and chat threads too, so a "clear" leaves no residue.
    client.table("videos").delete().eq("user_id", user_id).execute()
    client.table("conversations").delete().eq("user_id", user_id).execute()
    return len(resp.data or [])


def delete_analysis(*, token: str, analysis_id: str, user_id: str) -> bool:
    """Delete ONE analysis owned by the caller; return whether a row was actually removed.

    The video row and chat thread are keyed on ``video_id``, not on the analysis: re-analysing one
    clip inserts a *second* ``analyses`` row against the same ``video_id`` (``persist_analysis``
    upserts ``videos`` but inserts ``analyses``). So they are dropped only once this was the LAST
    analysis referencing that video -- never unconditionally the way ``delete_all_analyses`` can,
    or deleting one record would silently wipe a sibling record's chat thread.

    The uploaded file under ``runtime/uploads/`` is deliberately left on disk, matching the
    clear-all path; reaping those is a separate change that should cover both.
    """
    client = _user_client(token)
    found = (
        client.table("analyses")
        .select("video_id")
        .eq("id", analysis_id)
        .limit(1)
        .execute()
    )
    rows = found.data or []
    if not rows:
        # Missing, or someone else's -- RLS scopes the read, so the two are indistinguishable.
        return False
    video_id = rows[0]["video_id"]

    # RLS already scopes writes to auth.uid() = user_id; the explicit predicate is a second guard.
    resp = (
        client.table("analyses")
        .delete()
        .eq("id", analysis_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not (resp.data or []):
        return False

    siblings = (
        client.table("analyses")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("video_id", video_id)
        .limit(1)
        .execute()
    )
    if (siblings.count or 0) == 0:
        client.table("videos").delete().eq("user_id", user_id).eq("video_id", video_id).execute()
        client.table("conversations").delete().eq("user_id", user_id).eq(
            "video_id", video_id
        ).execute()
    return True


def get_analysis(*, token: str, analysis_id: str) -> dict[str, Any] | None:
    """Return one analysis row (full ``result`` JSONB) owned by the caller, or ``None``."""
    client = _user_client(token)
    resp = (
        client.table("analyses")
        .select("*")
        .eq("id", analysis_id)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    return rows[0] if rows else None


def upsert_conversation(
    *,
    token: str,
    user_id: str,
    video_id: str,
    messages: list[dict[str, Any]],
    followups: list[str] | None = None,
) -> None:
    """Save the caller's chat thread for ``video_id`` (one row per ``(user_id, video_id)``).

    The whole message array is written each turn — coaching threads are short, so a full-array
    upsert is simpler than an append and keeps the row self-contained for replay. ``followups`` is
    the *latest* answer's two grounded next-question chips (ephemeral React state otherwise lost on
    reload); it is persisted so a history-replay restores the chips, not just the answer.
    """
    client = _user_client(token)
    client.table("conversations").upsert(
        {
            "user_id": user_id,
            "video_id": video_id,
            "messages": messages,
            "followups": followups or [],
        },
        on_conflict="user_id,video_id",
    ).execute()


def get_conversation(*, token: str, video_id: str) -> dict[str, Any] | None:
    """Return the caller's saved thread as ``{"messages": [...], "followups": [...]}``, or ``None``.

    ``None`` means no thread has been saved yet (a fresh upload). A saved-but-empty thread returns
    empty lists so the caller can tell "no thread" from "an empty thread". ``followups`` are the
    latest answer's chips (empty for pre-followups rows or a cleared thread).
    """
    client = _user_client(token)
    resp = (
        client.table("conversations")
        .select("messages, followups")
        .eq("video_id", video_id)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return None
    return {
        "messages": rows[0].get("messages") or [],
        "followups": rows[0].get("followups") or [],
    }
