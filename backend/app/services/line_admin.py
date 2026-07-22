"""Read the LINE Messaging API for the admin panel (read-only).

Companion to services/line_bot (which SENDS): this module READS account state for the admin
diagnostics panel — push-message quota, bot info, webhook health, and daily delivery counts.
Every read uses the Messaging channel access token server-side only (never exposed to the browser).

Defensive throughout (mirrors line_bot.reply): ANY failure — no token, network error, non-200,
malformed shape — returns None so the panel degrades to "unavailable" rather than raising. Read-only
lookups share a keyed 60s TTL cache so admin refreshes don't hammer LINE (its endpoints rate-limit).

``httpx`` is a top-level import (as in line_bot); ``get_settings`` is read through the module
namespace so tests can patch it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from backend.app.settings import get_settings

logger = logging.getLogger(__name__)

LINE_QUOTA_URL = "https://api.line.me/v2/bot/message/quota"
LINE_CONSUMPTION_URL = "https://api.line.me/v2/bot/message/quota/consumption"
LINE_BOT_INFO_URL = "https://api.line.me/v2/bot/info"
LINE_WEBHOOK_ENDPOINT_URL = "https://api.line.me/v2/bot/channel/webhook/endpoint"
LINE_DELIVERY_REPLY_URL = "https://api.line.me/v2/bot/message/delivery/reply"
LINE_DELIVERY_PUSH_URL = "https://api.line.me/v2/bot/message/delivery/push"
LINE_WEBHOOK_TEST_URL = "https://api.line.me/v2/bot/channel/webhook/test"
_TIMEOUT_S = 10.0
_TTL_SECONDS = 60.0

# Delivery counts are per-day and only complete the day after, so we always read YESTERDAY.
_DISPLAY_TZ = timezone(timedelta(hours=8))  # LINE OA account timezone (Taiwan), matches services/line_bot.

# Keyed process-wide TTL cache: values may be None (a real "unavailable" result worth caching for the
# window); ``key in _cache_at`` distinguishes cached-None from never-fetched.
_cache: dict[str, dict[str, Any] | None] = {}
_cache_at: dict[str, float] = {}


def _cached(key: str, producer: Callable[[], dict[str, Any] | None]) -> dict[str, Any] | None:
    """Serve ``key`` from the 60s TTL cache, calling ``producer()`` on miss/expiry."""
    now = time.monotonic()
    if key in _cache_at and (now - _cache_at[key]) < _TTL_SECONDS:
        return _cache[key]
    result = producer()
    _cache[key] = result
    _cache_at[key] = now
    return result


def _safe_int(value: Any) -> int | None:
    """int() or None — the LINE payload is trusted but not guaranteed well-formed."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get(url: str, token: str) -> dict[str, Any]:
    """GET a LINE endpoint; return its JSON dict. Raises on non-200 or non-dict payload."""
    resp = httpx.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("unexpected LINE quota payload")
    return data


def _post(url: str, token: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
    """POST to a LINE endpoint; return its JSON dict. Raises on non-200 or non-dict payload."""
    resp = httpx.post(url, headers={"Authorization": f"Bearer {token}"}, json=json or {}, timeout=_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("unexpected LINE payload")
    return data


def _fetch_quota() -> dict[str, Any] | None:
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
    """Push-quota snapshot ({"type","used",[value,remaining]}) or None, served from the 60s TTL cache."""
    return _cached("quota", _fetch_quota)


def _token() -> str:
    """The Messaging channel access token (empty string when unconfigured)."""
    return get_settings().line_messaging_access_token


def _yesterday_yyyymmdd() -> str:
    """Yesterday in the OA timezone as ``yyyymmdd`` — the newest day LINE has complete delivery data for."""
    return (datetime.now(_DISPLAY_TZ) - timedelta(days=1)).strftime("%Y%m%d")


def _fetch_bot_info() -> dict[str, Any] | None:
    token = _token()
    if not token:
        return None
    try:
        data = _get(LINE_BOT_INFO_URL, token)
    except Exception:  # noqa: BLE001 — any failure means "unavailable"; never propagate.
        logger.warning("LINE bot info: read failed")
        return None
    premium_id = data.get("premiumId")
    return {
        "display_name": str(data.get("displayName") or ""),
        "basic_id": str(data.get("basicId") or ""),
        "premium_id": premium_id if isinstance(premium_id, str) else None,
        "chat_mode": str(data.get("chatMode") or ""),
        "mark_as_read_mode": str(data.get("markAsReadMode") or ""),
    }


def _fetch_webhook() -> dict[str, Any] | None:
    token = _token()
    if not token:
        return None
    try:
        data = _get(LINE_WEBHOOK_ENDPOINT_URL, token)
    except Exception:  # noqa: BLE001
        logger.warning("LINE webhook endpoint: read failed")
        return None
    endpoint = data.get("endpoint")
    if not isinstance(endpoint, str):
        return None
    return {"endpoint": endpoint, "active": bool(data.get("active"))}


def _fetch_delivery() -> dict[str, Any] | None:
    token = _token()
    if not token:
        return None
    date = _yesterday_yyyymmdd()
    try:
        reply = _get(f"{LINE_DELIVERY_REPLY_URL}?date={date}", token)
        push = _get(f"{LINE_DELIVERY_PUSH_URL}?date={date}", token)
    except Exception:  # noqa: BLE001
        logger.warning("LINE delivery: read failed")
        return None
    # ``success`` is present only when status == "ready"; _safe_int -> None otherwise.
    return {"date": date, "reply": _safe_int(reply.get("success")), "push": _safe_int(push.get("success"))}


def fetch_bot_info() -> dict[str, Any] | None:
    """OA display name / basic id / chat mode, or None. 60s TTL cache."""
    return _cached("bot_info", _fetch_bot_info)


def fetch_webhook() -> dict[str, Any] | None:
    """Configured webhook endpoint + active flag, or None. 60s TTL cache."""
    return _cached("webhook", _fetch_webhook)


def fetch_delivery() -> dict[str, Any] | None:
    """Yesterday's reply/push delivery counts, or None. 60s TTL cache."""
    return _cached("delivery", _fetch_delivery)


def test_webhook() -> dict[str, Any] | None:
    """Actively ask LINE to POST a test event to the configured webhook; report the outcome.

    Has a SIDE EFFECT (LINE delivers a test event to the webhook), so this is never called on a plain
    status read — only on an explicit admin action. NOT cached. Returns None on any transport failure;
    ``success`` reflects whether the webhook itself answered LINE with 200.
    """
    token = _token()
    if not token:
        return None
    try:
        data = _post(LINE_WEBHOOK_TEST_URL, token)
    except Exception:  # noqa: BLE001
        logger.warning("LINE webhook test: request failed")
        return None
    return {
        "success": bool(data.get("success")),
        "status_code": _safe_int(data.get("statusCode")),
        "reason": data.get("reason") if isinstance(data.get("reason"), str) else None,
        "detail": data.get("detail") if isinstance(data.get("detail"), str) else None,
    }


def clear_cache() -> None:
    """Invalidate the whole TTL cache so the next read re-fetches (used by tests)."""
    _cache.clear()
    _cache_at.clear()
