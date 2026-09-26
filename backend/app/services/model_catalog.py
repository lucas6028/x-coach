"""Live LLM model availability: is the configured model actually still served?

The provider retired the configured default model once already and every chat request failed
until someone noticed (2026-09-26). This module closes that gap two ways:

  * **Before a request** — a process-wide, TTL-cached snapshot of the provider's ``/models``
    catalog (fetched in the background, never on the request path) filters
    ``settings.chat_models()`` down to ``settings.available_chat_models()``, so a default/pinned
    model the provider has delisted is never handed to a client in the first place.
  * **During a request** — ``services.chat`` marks a model "dead" the moment the provider itself
    says so (HTTP 410, or 404 on a request that carries no ``tools``) and falls back to the next
    available candidate for the REST OF THE PROCESS LIFETIME (or 30 minutes, whichever is sooner),
    so a single bad response doesn't repeat on every subsequent request.

Both signals are **fail-open**: an unreachable/never-fetched catalog answers "available" for every
model (``is_available`` returns ``True`` when the catalog is unknown), because withholding a model
on our OWN inability to check it would be strictly worse than the status quo. Only a positive,
provider-confirmed signal (in the catalog with no dead-mark, or a fresh dead-mark) changes anything.

NOT ALL CATALOGS ARE TRUSTWORTHY. OpenRouter's is authoritative; NVIDIA NIM's ``/models`` endpoint
still lists models that 404/410 on ``/chat/completions`` (verified 2026-09-26:
``meta/llama2-70b`` lists but 404s; ``openai/gpt-oss-120b`` isn't even listed but 410s). So the
catalog is used only to ADD confidence (``available`` / ``not_listed``), never to override a dead
mark — a catalog refresh must never clear one (see ``mark_unavailable``).

STATE AND LOCKING. Two module-level locks, deliberately not one: ``_refresh_lock`` serializes the
actual network fetch (``httpx.get``, up to the 5s timeout) so two refreshes never race each other;
``_state_lock`` guards the small state variables and is held only for a plain dict/attribute
read-or-write, never across I/O. Every READ path in this module (``catalog_ids``,
``is_available``, ``is_marked_unavailable``, ``model_status``, ``catalog_state``) touches only
``_state_lock`` and returns immediately — none of them ever blocks on the network, which is what
lets ``GET /api/health`` stay synchronous and instant even while a refresh is in flight.

Mirrors ``services/runtime_config.py``'s pattern: a single patchable network seam
(``_fetch_catalog``), a process-wide cache, and every failure swallowed so an offline/misconfigured
process degrades to its safe default (here, fail-open) rather than raising.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

# --------------------------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------------------------

# How long a successfully fetched catalog is trusted before the next read triggers a refresh.
_CATALOG_SUCCESS_TTL_S = 600.0
# After a FAILED refresh, don't retry sooner than this — a dead/misconfigured provider must not be
# hammered on every request.
_CATALOG_FAILURE_RETRY_S = 120.0
# How long a dead-mark (a model the provider itself rejected mid-request) is trusted. Long enough
# that one bad response doesn't repeat all over again a minute later; short enough that a model the
# provider brings back doesn't stay wrongly blacklisted for the life of the process.
_DEAD_MARK_TTL_S = 1800.0

# --------------------------------------------------------------------------------------------
# Module state (process-wide; see the module docstring for the locking discipline)
# --------------------------------------------------------------------------------------------

_state_lock = threading.Lock()
_refresh_lock = threading.Lock()

# Everything below is keyed to ``_cache_base_url`` as a bundle: if the CURRENT provider base URL
# (``settings.chat_base_url()``) doesn't match it, every read treats the bundle as absent — an
# admin switching providers must not serve the old provider's catalog under the new one's name.
_cache_base_url: str | None = None
_cache_ids: dict[str, str | None] | None = None
_cache_fetched_at: float | None = None  # monotonic time of the last SUCCESSFUL fetch
_last_attempt_at: float | None = None  # monotonic time of the last fetch ATTEMPT (success or not)
_last_attempt_wall: str | None = None  # same moment, as an ISO-8601 UTC string (for catalog_state)
_last_error: str | None = None  # None when the last attempt succeeded

_refreshing = False  # single-flight guard for _start_refresh

# Dead marks: (base_url, model_id) -> (marked_at monotonic, detail string). Kept OUTSIDE the
# base_url-bundle above and NEVER cleared by a catalog refresh — see ``mark_unavailable``.
_dead: dict[tuple[str, str], tuple[float, str]] = {}


def _now() -> float:
    """The monotonic clock, as its own seam so tests can control elapsed time without patching the
    global ``time`` module (which every other timing-sensitive getter in this codebase reads)."""
    return time.monotonic()


def _iso_now() -> str:
    """Wall-clock "now" as ISO-8601 UTC with second precision (``…+00:00``, no microseconds) — the
    exact shape ``GET /api/admin/llm/models`` documents for ``catalog.checked_at``."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _current_base_url() -> str | None:
    """The provider base URL to key every cache read/write on, or ``None`` when settings can't be
    resolved (e.g. a test's lightweight stand-in ``get_settings()`` return value). NEVER RAISES —
    every caller in this module treats ``None`` as "nothing to key a cache lookup on" and degrades
    to its own fail-open default rather than letting an AttributeError escape."""
    try:
        from backend.app import settings as _settings

        return _settings.chat_base_url()
    except Exception:  # noqa: BLE001 — this module must never break a caller over a settings shape.
        return None


def _fetch_catalog(base_url: str) -> dict[str, str | None]:
    """The ONE network seam: fetch ``{base_url}/models`` and return ``{model_id: expiration_date}``.

    ``base_url`` must come from ``settings.chat_base_url()`` (the allowlist guard) — this function
    trusts its caller for that and sends the ``Authorization: Bearer <key>`` header wherever it
    points, so nothing upstream of this call may source ``base_url`` any other way. Raises on any
    HTTP or transport failure (``raise_for_status``); the caller (``refresh``) is what swallows it.

    Tolerates both shapes seen in the wild: OpenRouter's ``{"data": [{"id", "expiration_date", ...}]}``
    and NVIDIA NIM's ``{"data": [{"id", "object", "created", "owned_by"}]}`` (no ``expiration_date``
    at all, which becomes ``None`` — NOT a signal of anything; NIM's catalog is known to list dead
    models regardless, see the module docstring).
    """
    import httpx  # deferred: only needed on a live fetch, keeps this module import-light.

    from backend.app.settings import get_settings

    key = getattr(get_settings(), "llm_api_key", "") or ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    resp = httpx.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=5.0)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data") if isinstance(payload, dict) else None

    out: dict[str, str | None] = {}
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not model_id:
            continue
        out[str(model_id)] = item.get("expiration_date")
    return out


def refresh() -> None:
    """Synchronous fetch + store. Swallows every exception into ``_last_error`` — NEVER RAISES —
    because this runs unattended on a background thread (``_start_refresh``) with nothing to catch
    it, and also directly from the admin ``refresh=true`` endpoint, where a raised exception would
    turn a "please recheck" click into a 500.

    A FAILED refresh keeps the previous good ids (for the SAME base URL only — a brand-new base URL
    with no prior success has nothing of its own to keep, so it starts from ``None``). This is what
    makes a transient provider hiccup harmless: ``catalog_ids()`` keeps answering from the last known
    truth instead of going blank the moment one fetch fails.

    Protected by ``_refresh_lock`` so two concurrent refreshes (the background single-flighted one
    and an admin-triggered synchronous one) never race the same network call.
    """
    global _cache_base_url, _cache_ids, _cache_fetched_at
    global _last_attempt_at, _last_attempt_wall, _last_error, _refreshing

    with _refresh_lock:
        base_url = _current_base_url()
        now = _now()
        now_wall = _iso_now()

        if base_url is None:
            with _state_lock:
                _last_attempt_at = now
                _last_attempt_wall = now_wall
                _last_error = "chat base URL is not resolvable"
                _refreshing = False
            return

        try:
            ids = _fetch_catalog(base_url)
        except Exception as exc:  # noqa: BLE001 — a best-effort refresh must never raise.
            with _state_lock:
                if _cache_base_url != base_url:
                    # A different (or no) provider was cached before this attempt -- there is no
                    # "previous good ids" of THIS url's to keep.
                    _cache_base_url = base_url
                    _cache_ids = None
                    _cache_fetched_at = None
                _last_attempt_at = now
                _last_attempt_wall = now_wall
                _last_error = f"{type(exc).__name__}: {exc}"
                _refreshing = False
            return

        with _state_lock:
            _cache_base_url = base_url
            _cache_ids = ids
            _cache_fetched_at = now
            _last_attempt_at = now
            _last_attempt_wall = now_wall
            _last_error = None
            _refreshing = False


def _start_refresh() -> None:
    """Single-flight kickoff: start a daemon thread running ``refresh()``, unless one is already
    running. This is the seam ``tests/conftest.py`` stubs to a no-op for every other test in the
    suite (so no test ever spawns a real background thread), and the one this module's own tests
    patch back to the real implementation to exercise single-flight directly."""
    global _refreshing
    with _state_lock:
        if _refreshing:
            return
        _refreshing = True
    threading.Thread(target=refresh, daemon=True).start()


def catalog_ids() -> dict[str, str | None] | None:
    """The cached model-id -> expiration_date map for the CURRENT base URL, or ``None`` when
    unknown (never fetched, the fetch failed with nothing to fall back on, or the base URL just
    changed). NEVER BLOCKS — this is what ``GET /api/health`` and every settings getter rely on.

    Returns whatever is cached, stale or not, and only kicks off a background refresh (never an
    inline fetch) when the cache is stale/missing AND ``settings.chat_configured`` is true. A
    failure backs off ``_CATALOG_FAILURE_RETRY_S``; a success is trusted for
    ``_CATALOG_SUCCESS_TTL_S`` before the next read triggers another background refresh.
    """
    base_url = _current_base_url()
    if base_url is None:
        return None

    now = _now()
    with _state_lock:
        if _cache_base_url == base_url:
            ids = _cache_ids
            fetched_at = _cache_fetched_at
            last_attempt = _last_attempt_at
            last_error = _last_error
        else:
            ids = fetched_at = last_attempt = last_error = None

    stale = ids is None or fetched_at is None or (now - fetched_at) >= _CATALOG_SUCCESS_TTL_S
    if stale:
        try:
            from backend.app.settings import get_settings

            configured = bool(getattr(get_settings(), "chat_configured", False))
        except Exception:  # noqa: BLE001 — an unresolvable settings object means "don't fetch".
            configured = False
        if configured:
            min_gap = _CATALOG_FAILURE_RETRY_S if last_error else _CATALOG_SUCCESS_TTL_S
            if last_attempt is None or (now - last_attempt) >= min_gap:
                _start_refresh()

    return ids


def mark_unavailable(model: str, detail: str) -> None:
    """Record that the provider itself just rejected ``model`` as gone (HTTP 410, or a plain 404).

    Keyed to the CURRENT base URL — a model dead on one provider says nothing about a same-named
    model on another. TTL'd (``_DEAD_MARK_TTL_S``) rather than permanent, and — critically — never
    cleared by a catalog ``refresh()``: NIM's ``/models`` keeps listing models that 404, so trusting
    a refresh to "un-mark" a model would silently resurrect exactly the failure this module exists
    to prevent.
    """
    base_url = _current_base_url()
    if base_url is None:
        return  # nothing to key the mark on; the real request path always has a resolvable URL.
    with _state_lock:
        _dead[(base_url, model)] = (_now(), detail)


def is_marked_unavailable(model: str) -> bool:
    """True when ``model`` was dead-marked for the CURRENT base URL and the mark hasn't expired.

    The empty-dict short-circuit below means the common case (nothing has ever been marked dead)
    never even resolves the current base URL — so a caller with an incomplete/stand-in settings
    object never sees this reach for ``chat_base_url()`` unless something has actually gone wrong.
    """
    with _state_lock:
        if not _dead:
            return False

    base_url = _current_base_url()
    if base_url is None:
        return False

    key = (base_url, model)
    now = _now()
    with _state_lock:
        entry = _dead.get(key)
        if entry is None:
            return False
        marked_at, _detail = entry
        if now - marked_at >= _DEAD_MARK_TTL_S:
            del _dead[key]
            return False
        return True


def _catalog_key(model: str, ids: dict[str, str | None]) -> str | None:
    """The catalog id that vouches for ``model``, or ``None`` when the catalog doesn't list it.

    Exact match first. Failing that, a ``vendor/model:suffix`` id falls back to its base id — but
    only when NO catalog id carries that suffix. OpenRouter lists some variants as ids of their own
    (``:free``, ``:batch``) and accepts routing suffixes it never lists (``:nitro``, ``:floor``,
    ``:online``). A listed-kind variant that is missing is genuinely gone (a free tier that ended),
    while an unlisted-kind suffix is only ever a routing hint on a base model that must be judged
    by the base id, or a working ``…:nitro`` would be dropped from the picker.
    """
    if model in ids:
        return model
    base, sep, suffix = model.partition(":")
    if not sep or base not in ids:
        return None
    if any(i.partition(":")[2] == suffix for i in ids):
        return None
    return base


def is_available(model: str) -> bool:
    """True unless ``model`` is dead-marked. FAILS OPEN when the catalog is unknown (never fetched,
    or not resolvable) — an unlisted-but-uncertain model is treated as available, since the whole
    point of this module is to never make the picker MORE restrictive than "no check at all" absent
    a positive signal that the model is actually gone."""
    if is_marked_unavailable(model):
        return False
    ids = catalog_ids()
    if ids is None:
        return True
    return _catalog_key(model, ids) is not None


def model_status(model: str) -> dict[str, Any]:
    """One model's full status for the admin panel: ``{"status", "expires", "detail"}``.

    ``status`` is one of ``"unavailable"`` (dead-marked; ``detail`` carries why), ``"not_listed"``
    (the catalog is known and ``model`` isn't in it), ``"available"`` (listed, with its
    ``expires``/``expiration_date`` when the provider sends one), or ``"unknown"`` (the catalog
    hasn't been fetched/isn't resolvable and the model isn't dead-marked either).
    """
    if is_marked_unavailable(model):
        detail = None
        base_url = _current_base_url()
        if base_url is not None:
            with _state_lock:
                entry = _dead.get((base_url, model))
            if entry is not None:
                detail = entry[1]
        ids = catalog_ids()
        key = _catalog_key(model, ids) if isinstance(ids, dict) else None
        expires = ids.get(key) if key is not None else None
        return {"status": "unavailable", "expires": expires, "detail": detail}

    ids = catalog_ids()
    if ids is None:
        return {"status": "unknown", "expires": None, "detail": None}
    key = _catalog_key(model, ids)
    if key is not None:
        return {"status": "available", "expires": ids.get(key), "detail": None}
    return {"status": "not_listed", "expires": None, "detail": None}


def catalog_state() -> dict[str, Any]:
    """The catalog's own health, for ``GET /api/admin/llm/models``: ``{"status", "checked_at",
    "model_count", "error"}``.

    ``status`` is ``"ok"`` (a successful fetch is cached for the current base URL — possibly stale,
    that's not this field's concern), ``"error"`` (the last attempt against the current base URL
    failed — even if a stale-but-good ids map is still being served underneath it), or ``"unknown"``
    (never attempted for this base URL, or the base URL just changed).
    """
    base_url = _current_base_url()
    with _state_lock:
        if base_url is not None and _cache_base_url == base_url:
            ids, checked_at, last_error = _cache_ids, _last_attempt_wall, _last_error
        else:
            ids, checked_at, last_error = None, None, None

    if last_error is not None:
        status = "error"
    elif ids is not None:
        status = "ok"
    else:
        status = "unknown"

    return {
        "status": status,
        "checked_at": checked_at,
        "model_count": len(ids) if ids is not None else None,
        "error": last_error,
    }


def _reset() -> None:
    """Clear ALL module state — the cache, dead marks, and the single-flight guard. Test-only (see
    ``tests/conftest.py``'s autouse fixture, which calls this before AND after every test so no
    test's catalog/dead-marks leak into the next one)."""
    global _cache_base_url, _cache_ids, _cache_fetched_at
    global _last_attempt_at, _last_attempt_wall, _last_error, _refreshing, _dead
    with _state_lock:
        _cache_base_url = None
        _cache_ids = None
        _cache_fetched_at = None
        _last_attempt_at = None
        _last_attempt_wall = None
        _last_error = None
        _refreshing = False
        _dead = {}
