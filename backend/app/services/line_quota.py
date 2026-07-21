"""Read the LINE Messaging API push-message quota + this month's consumption (read-only).

Companion to services/line_bot: the bot SENDS (reply, which is free and uncounted); this module
READS the account's push quota so the admin panel can show "used / limit / remaining". The numbers
therefore reflect ONLY push/multicast/broadcast usage — exactly what LINE bills.

Two LINE endpoints, both authorised with the Messaging channel access token (server-side only;
never exposed to the browser):
    GET /v2/bot/message/quota             -> {"type": "none"} | {"type": "limited", "value": N}
    GET /v2/bot/message/quota/consumption -> {"totalUsage": N}

Defensive throughout (mirrors line_bot.reply): ANY failure — no token, network error, non-200,
malformed shape — returns None so the panel degrades to "unavailable" rather than raising. A short
process-wide TTL cache keeps admin refreshes from hammering LINE (its quota endpoints rate-limit).

``httpx`` is a top-level import (as in line_bot); ``get_settings`` is read through the module
namespace so tests can patch it.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from backend.app.settings import get_settings

logger = logging.getLogger(__name__)

LINE_QUOTA_URL = "https://api.line.me/v2/bot/message/quota"
LINE_CONSUMPTION_URL = "https://api.line.me/v2/bot/message/quota/consumption"
_QUOTA_TIMEOUT_S = 10.0
_TTL_SECONDS = 60.0

# Process-wide snapshot: last fetched result (may be None) and when it was fetched.
_cache: dict[str, Any] | None = None
_cache_at: float = 0.0
_cache_valid: bool = False  # distinguishes "cached None" from "never fetched".


def _safe_int(value: Any) -> int | None:
    """int() or None — the LINE payload is trusted but not guaranteed well-formed."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get(url: str, token: str) -> dict[str, Any]:
    """GET a LINE quota endpoint; return its JSON dict. Raises on non-200 or non-dict payload."""
    resp = httpx.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=_QUOTA_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("unexpected LINE quota payload")
    return data


def _fetch() -> dict[str, Any] | None:
    """One live read of both endpoints -> the assembled quota dict, or None on any failure."""
    token = get_settings().line_messaging_access_token
    if not token:
        return None
    try:
        quota = _get(LINE_QUOTA_URL, token)
        consumption = _get(LINE_CONSUMPTION_URL, token)
    except Exception:  # noqa: BLE001 — any failure means "unavailable"; never propagate.
        logger.warning("LINE quota: read failed")
        return None

    used = _safe_int(consumption.get("totalUsage"))
    if used is None:
        return None
    result: dict[str, Any] = {
        "type": "limited" if quota.get("type") == "limited" else "none",
        "used": used,
    }
    if result["type"] == "limited":
        value = _safe_int(quota.get("value"))
        if value is None:
            return None
        result["value"] = value
        result["remaining"] = max(0, value - used)
    return result


def fetch_quota() -> dict[str, Any] | None:
    """Push-quota snapshot ({"type","used",[value,remaining]}) or None, served from a 60s TTL cache."""
    global _cache, _cache_at, _cache_valid
    now = time.monotonic()
    if _cache_valid and (now - _cache_at) < _TTL_SECONDS:
        return _cache
    _cache = _fetch()
    _cache_at = now
    _cache_valid = True
    return _cache


def clear_cache() -> None:
    """Invalidate the TTL cache so the next fetch_quota re-reads (used by tests)."""
    global _cache, _cache_at, _cache_valid
    _cache = None
    _cache_at = 0.0
    _cache_valid = False
