"""Runtime settings resolution: read admin-editable overrides from the ``app_settings`` table.

This is the one place the backend reads the ``app_settings`` override table (see the
``20260713000100_app_settings.sql`` migration). The settings *getters* in
``backend.app.settings`` layer these values over their env/constant defaults, so an admin can
retune the LLM / RAG / analysis knobs without a redeploy.

Design constraints, all deliberate:
  * **Anon client, no user token.** Overrides are non-secret operational knobs and are read on
    paths that may be unauthenticated (e.g. the public ``/api/health`` model picker), so we read
    with a plain anon client (``create_client(url, anon_key)`` with no ``postgrest.auth``). The
    table's SELECT policy grants ``anon``.
  * **Offline-safe.** ANY failure — auth not configured, ``supabase`` not installed, a network
    error, a missing table — returns ``{}`` so every caller falls back cleanly to its env default.
    Many callers are offline paths (analysis suffix checks, KG/RAG defaults) that must never
    depend on a reachable database.
  * **Short TTL cache.** A single process-wide snapshot with a ~30s TTL keeps this off the hot
    path without going stale for long after an admin write; ``clear_cache()`` (called by the admin
    PUT handler) forces the next read to refetch immediately.

The actual DB read is factored into ``_fetch_rows`` so the unit tests can patch that single seam
(mirroring how ``services.store._user_client`` is patchable) without ``supabase`` installed.
"""

from __future__ import annotations

import time
from typing import Any

# How long a fetched override snapshot is trusted before the next read refetches (seconds).
_TTL_SECONDS = 30.0

# Process-wide cache: the last fetched overrides and the monotonic time they were fetched at.
# ``None`` means "never fetched / invalidated" so the next ``get_overrides`` refetches.
_cache: dict[str, Any] | None = None
_cache_at: float = 0.0


def _fetch_rows() -> list[dict[str, Any]]:
    """Read every ``app_settings`` row with an anon client. The single patchable DB seam.

    Uses ``create_client(url, anon_key)`` WITHOUT ``postgrest.auth(token)`` — an unauthenticated
    read, which the table's anon SELECT policy allows. The ``supabase`` import is deferred so this
    module (and everything that transitively imports a settings getter) stays import-light and the
    tests can patch this function without the package installed.
    """
    from supabase import create_client  # deferred heavy import (gotrue/postgrest/...)

    from backend.app.settings import get_settings

    settings = get_settings()
    client = create_client(settings.supabase_url, settings.supabase_anon_key)
    resp = client.table("app_settings").select("key, value").execute()
    return resp.data or []


def get_overrides() -> dict[str, Any]:
    """Return the admin overrides as ``{key: value}``, or ``{}`` on any failure / when offline.

    Served from a short-TTL in-process cache. Returns ``{}`` immediately when auth isn't configured
    (no Supabase project to read from) and on ANY exception during the read, so callers always have
    a clean env-default fallback and no offline path can be broken by a DB hiccup.
    """
    global _cache, _cache_at

    # auth_configured is checked FIRST, before any import, so the offline path is fully hermetic.
    # ``getattr`` with a default keeps this robust when a caller has patched ``get_settings`` to a
    # lightweight stand-in without the property (several existing unit tests do exactly that).
    from backend.app.settings import get_settings

    if not getattr(get_settings(), "auth_configured", False):
        return {}

    now = time.monotonic()
    if _cache is not None and (now - _cache_at) < _TTL_SECONDS:
        return _cache

    try:
        rows = _fetch_rows()
        overrides = {row["key"]: row["value"] for row in rows if "key" in row}
    except Exception:  # noqa: BLE001 — any failure means "no overrides", preserving env fallback.
        # Cache the empty result too: a best-effort override read that failed shouldn't hammer the DB
        # on every getter call for the next TTL window (env defaults apply meanwhile).
        overrides = {}

    _cache = overrides
    _cache_at = now
    return overrides


def clear_cache() -> None:
    """Invalidate the TTL cache so the next ``get_overrides`` refetches (called after an admin write)."""
    global _cache, _cache_at
    _cache = None
    _cache_at = 0.0
