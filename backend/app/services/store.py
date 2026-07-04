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


def _summarize(result: dict[str, Any]) -> tuple[str | None, int, str | None]:
    """Promote the list-view columns out of the nested analysis document."""
    view_type = (result.get("view") or {}).get("view_type")
    fault_count = len(result.get("detections") or [])
    pipeline_version = result.get("pipeline_version")
    return view_type, fault_count, pipeline_version


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

    view_type, fault_count, pipeline_version = _summarize(result)
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
            "id, video_id, source, view_type, fault_count, created_at",
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
    *, token: str, user_id: str, video_id: str, messages: list[dict[str, Any]]
) -> None:
    """Save the caller's chat thread for ``video_id`` (one row per ``(user_id, video_id)``).

    The whole message array is written each turn — coaching threads are short, so a full-array
    upsert is simpler than an append and keeps the row self-contained for replay.
    """
    client = _user_client(token)
    client.table("conversations").upsert(
        {"user_id": user_id, "video_id": video_id, "messages": messages},
        on_conflict="user_id,video_id",
    ).execute()


def get_conversation(*, token: str, video_id: str) -> dict[str, Any] | None:
    """Return the caller's saved thread for ``video_id`` as ``{"messages": [...]}``, or ``None``.

    ``None`` means no thread has been saved yet (a fresh upload). A saved-but-empty thread returns
    ``{"messages": []}`` so the caller can tell "no thread" from "an empty thread".
    """
    client = _user_client(token)
    resp = (
        client.table("conversations")
        .select("messages")
        .eq("video_id", video_id)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return None
    return {"messages": rows[0].get("messages") or []}
