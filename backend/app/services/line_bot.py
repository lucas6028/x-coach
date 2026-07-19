"""LINE Messaging API bot: answer "my training summary" inside the LINE chat room.

The companion to ``services/line_auth`` (which signs LINE users in). Both channels live
under the SAME LINE provider, so the webhook's ``source.userId`` is byte-identical to the
Login ID token's ``sub`` that ``line_auth`` stored in ``user_metadata.line_sub`` — that is
what lets the bot resolve the account with no binding flow (LINE docs: a user ID is issued
per *provider*, not per channel).

The webhook has no user JWT, so it cannot use ``services/store`` (every call there runs as
the user with RLS). Instead it calls ONE SECURITY DEFINER function,
``public.line_training_summary(p_line_sub)``, granted to service_role only — the smallest
possible widening of the "backend never touches data with service_role" posture.

``httpx`` is a top-level import (as in ``line_auth``); the ``supabase`` import is deferred so
the module stays light and unit tests can fake the package without it installed.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from backend.app.settings import get_settings

logger = logging.getLogger(__name__)

# LINE's reply endpoint. Replies are free and need no push quota, but the reply token is
# single-use and expires ~1 minute after the event — never retry a failed reply.
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"


def verify_signature(raw_body: bytes, signature: str | None) -> bool:
    """Whether ``signature`` is LINE's HMAC-SHA256 of ``raw_body`` under our channel secret.

    Must be computed over the RAW request bytes — re-serialising the parsed JSON would change
    whitespace/key order and never match. Compared with ``compare_digest`` so a wrong
    signature leaks no timing information.
    """
    secret = get_settings().line_messaging_channel_secret
    if not secret or not signature or not signature.isascii():
        # ``compare_digest`` raises TypeError on non-ASCII str input rather than just
        # returning False; a base64 signature is ASCII by construction, so anything else
        # is definitionally a mismatch and short-circuits before we get there.
        return False
    digest = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("ascii")
    return hmac.compare_digest(expected, signature)


# The SECURITY DEFINER function granted to service_role (see the matching migration). It is
# the ONLY data the bot can reach: one user's aggregate summary, keyed by their LINE user id.
SUMMARY_RPC = "line_training_summary"


def _service_client() -> Any:
    """Build a service_role Supabase client (needed to call the summary RPC).

    Deferred import, as in ``line_auth._admin_client``: keeps the module light and lets unit
    tests fake the ``supabase`` package without it installed.
    """
    from supabase import create_client  # deferred heavy import (gotrue/postgrest/...)

    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def summary_for_line_user(line_user_id: str) -> dict[str, Any] | None:
    """Return this LINE user's training summary, or ``None`` if they have no x-coach account.

    The RPC returns SQL NULL for an unknown ``sub``; PostgREST surfaces that as ``None``. Any
    other shape is treated as "unknown" rather than trusted.
    """
    response = _service_client().rpc(SUMMARY_RPC, {"p_line_sub": line_user_id}).execute()
    data = getattr(response, "data", None)
    return data if isinstance(data, dict) else None


# Replies are read in Taiwan; fix the display offset rather than depending on a tz database
# (zoneinfo needs the `tzdata` package on Windows, which is not in the lean CI dependency set).
_DISPLAY_TZ = timezone(timedelta(hours=8))

# Localised labels, kept in step with the frontend's i18n keys (`fault.*`, `view.*` in
# frontend/src/lib/i18n.tsx) so the chat room and the web app name the same thing the same way.
FAULT_LABELS: dict[str, str] = {
    "knees_inward": "膝蓋內夾",
    "knees_forward": "膝蓋前移",
    "shallow_depth": "深度不足",
    "excessive_forward_lean": "軀幹過度前傾",
    "heel_rise": "腳跟離地",
    "butt_wink": "骨盆後傾",
    "asymmetric_shift": "左右不對稱",
}
VIEW_LABELS: dict[str, str] = {
    "front": "正面",
    "front_oblique": "正面斜角",
    "side": "側面",
    "rear": "背面",
    "rear_oblique": "背面斜角",
    "left": "左側",
    "right": "右側",
    "unknown": "未知",
}


def _liff_link() -> str:
    """The deep link into the LIFF app, or "" when no LIFF id is configured."""
    liff_id = (getattr(get_settings(), "line_liff_id", "") or "").strip()
    return f"https://liff.line.me/{liff_id}" if liff_id else ""


def _safe_int(value: Any, default: int = 0) -> int:
    """Best-effort ``int()``: the RPC payload is trusted but not guaranteed, and a webhook
    that has already been acknowledged must never raise on a malformed field (e.g. a count
    that arrives as ``None``, a non-numeric string, or some other unexpected type)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_time(raw: Any) -> str:
    """Render a PostgREST ISO timestamp in UTC+8, or "未知時間" if it can't be parsed."""
    if not isinstance(raw, str) or not raw:
        return "未知時間"
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return "未知時間"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(_DISPLAY_TZ).strftime("%Y-%m-%d %H:%M")


def _fault_label(fault: dict[str, Any]) -> str:
    """Localised fault name, falling back to the English name for ids we don't know yet."""
    fault_id = str(fault.get("id") or "")
    return FAULT_LABELS.get(fault_id) or str(fault.get("name") or fault_id or "未知問題")


def _with_link(lines: list[str], call_to_action: str) -> str:
    link = _liff_link()
    if link:
        lines += ["", f"{call_to_action} 👉 {link}"]
    return "\n".join(lines)


def format_summary(summary: dict[str, Any]) -> str:
    """Render one training summary as a single LINE text message.

    Every field is defensive: the RPC shape is trusted, but a malformed row must degrade into
    a readable message rather than raise inside a webhook that has already been acknowledged.
    """
    lines = ["📊 你的訓練摘要", "", f"累積分析：{_safe_int(summary.get('total'))} 次"]

    latest = summary.get("latest")
    if isinstance(latest, dict):
        view = VIEW_LABELS.get(str(latest.get("view_type") or "unknown"), "未知")
        faults = _safe_int(latest.get("fault_count"))
        lines.append(
            f"最近一次：{_format_time(latest.get('created_at'))}"
            f"（{view}視角，偵測到 {faults} 個問題）"
        )

    top = [f for f in (summary.get("top_faults") or []) if isinstance(f, dict)]
    if top:
        lines += ["", "最常出現的問題"]
        for rank, fault in enumerate(top, start=1):
            lines.append(f"{rank}. {_fault_label(fault)} ×{_safe_int(fault.get('count'))}")

    return _with_link(lines, "打開 x-coach 看完整報告")


def unbound_message() -> str:
    """Reply for a LINE user with no matching x-coach account."""
    return _with_link(
        ["還沒有找到你的 x-coach 帳號。", "請先用 LINE 登入 x-coach，之後就能在這裡查詢訓練摘要。"],
        "前往登入",
    )


def empty_message() -> str:
    """Reply for a known user who has no analyses yet."""
    return _with_link(
        ["你還沒有分析紀錄。", "上傳一支深蹲影片做第一次分析，之後就能在這裡看到摘要。"],
        "開始分析",
    )


def help_message() -> str:
    """Reply for text we don't recognise."""
    return _with_link(
        ["傳「摘要」或點下方選單，就能看到你的訓練摘要。"],
        "打開 x-coach",
    )


_REPLY_TIMEOUT_S = 10.0

# Text that means "show me my summary". The rich-menu button is configured as a *message*
# action sending "我的訓練摘要", so the menu and typed keywords share one code path. Compared
# after stripping and lower-casing (lower-casing is a no-op for the Chinese entries).
SUMMARY_KEYWORDS = frozenset({"我的訓練摘要", "摘要", "訓練", "紀錄", "summary"})


def _reply_text_for(line_user_id: str) -> str:
    """Decide what to say to this user, degrading to an apology if the lookup fails."""
    try:
        summary = summary_for_line_user(line_user_id)
    except Exception:  # noqa: BLE001 — a webhook must answer, never propagate.
        logger.exception("LINE bot: training-summary lookup failed")
        return "暫時查不到你的訓練摘要，請稍後再試一次。"
    if summary is None:
        return unbound_message()
    if _safe_int(summary.get("total")) == 0:
        return empty_message()
    return format_summary(summary)


def handle_events(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Turn a webhook payload into the replies to send.

    Returns planned replies instead of sending them so the decision logic stays a pure-ish
    function (one mocked seam) and the router owns all the I/O. Non-text and malformed events
    are skipped silently: LINE delivers many event types we don't answer.
    """
    replies: list[dict[str, str]] = []
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("type") != "message":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("type") != "text":
            continue
        reply_token = event.get("replyToken")
        source = event.get("source")
        line_user_id = source.get("userId") if isinstance(source, dict) else None
        if not reply_token or not line_user_id:
            continue

        text = str(message.get("text") or "").strip().lower()
        answer = _reply_text_for(str(line_user_id)) if text in SUMMARY_KEYWORDS else help_message()
        replies.append({"reply_token": str(reply_token), "text": answer})
    return replies


def reply(reply_token: str, text: str) -> None:
    """Send one text message back through LINE's reply API; failures are logged, never raised.

    The reply token is single-use and expires ~1 minute after the event, so a failed reply is
    not retried — and the webhook has already been acknowledged either way.
    """
    settings = get_settings()
    try:
        response = httpx.post(
            LINE_REPLY_URL,
            headers={"Authorization": f"Bearer {settings.line_messaging_access_token}"},
            json={"replyToken": reply_token, "messages": [{"type": "text", "text": text}]},
            timeout=_REPLY_TIMEOUT_S,
        )
    except httpx.HTTPError:
        logger.warning("LINE bot: reply request failed")
        return
    if response.status_code != 200:
        # Never log the body or the token — it can carry user-identifying content.
        logger.warning("LINE bot: reply rejected with status %s", response.status_code)
