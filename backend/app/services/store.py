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
    size_bytes: int = 0,
    storage_key: str | None = None,
) -> str:
    """Upsert the video row and insert the analysis; return the new analysis id.

    The ``videos`` row carries the (currently trivial) status machine, the byte size (for the
    per-user storage quota), and the storage key — the R2 object key when the upload was pushed to
    object storage, else the local runtime path. ``result`` is stored verbatim as JSONB so history
    replay is self-contained.
    """
    client = _user_client(token)

    client.table("videos").upsert(
        {
            "user_id": user_id,
            "video_id": video_id,
            "filename": filename,
            "storage_key": storage_key or f"runtime/uploads/{video_id}",
            "size_bytes": size_bytes,
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


def get_usage(*, token: str) -> dict[str, int]:
    """Return the caller's current storage usage as ``{"count", "bytes"}`` for the quota check.

    Sums ``size_bytes`` over the caller's own (RLS-scoped) video rows. A user keeps a handful of
    short clips, so summing client-side is cheaper than a PostgREST aggregate and avoids its quirks.
    """
    client = _user_client(token)
    resp = client.table("videos").select("size_bytes", count="exact").execute()
    rows = resp.data or []
    total = sum(int(row.get("size_bytes") or 0) for row in rows)
    count = resp.count if resp.count is not None else len(rows)
    return {"count": count, "bytes": total}


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
    # Purge the source videos from object storage before dropping their rows, so a "clear" leaves no
    # residue in R2 either. Best-effort per object: a storage hiccup must not abort the DB cleanup.
    from backend.app.services import object_store

    if object_store.is_configured():
        vids = client.table("videos").select("video_id").eq("user_id", user_id).execute()
        for row in vids.data or []:
            try:
                object_store.delete_video(row["video_id"])
            except Exception:  # noqa: BLE001 — never let a storage error strand the DB delete
                pass
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
