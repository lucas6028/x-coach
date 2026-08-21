"""LINE Messaging API bot: training summary + LLM coaching conversation in the LINE chat room.

Two kinds of text are answered. The summary keywords (and the rich-menu button) return a
formatted training summary. Anything else is a question for the coach: it goes to the same
OpenAI-compatible LLM the web chat uses (``services/chat._stream_completion``), grounded in the
user's summary and a few turns of short-lived, in-process history — LINE has no conversation
state of its own, and the chat room is a demo surface, so the history is deliberately NOT
persisted (no migration, nothing to leak across deploys). When no LLM key is configured the
bot falls back to the keyword-only help reply, so the summary feature never depends on the LLM.

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
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from backend.app.services import chat as chat_service
from backend.app.settings import default_chat_model, get_settings

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


# The summary RPC is a prerequisite of every reply and the reply token lives ~1 minute, so the
# PostgREST call gets its own short budget instead of the client's 120s default — a hung
# Supabase must turn into the "暫時查不到" apology while the token can still carry it.
_RPC_TIMEOUT_S = 8


def _service_client() -> Any:
    """Build a service_role Supabase client (needed to call the summary RPC).

    Deferred import, as in ``line_auth._admin_client``: keeps the module light and lets unit
    tests fake the ``supabase`` package without it installed. ``ClientOptions`` is looked up
    leniently for the same reason (the test fake need not provide it).
    """
    import supabase  # deferred heavy import (gotrue/postgrest/...)

    settings = get_settings()
    options_cls = getattr(supabase, "ClientOptions", None)
    options = options_cls(postgrest_client_timeout=_RPC_TIMEOUT_S) if options_cls else None
    if options is None:
        return supabase.create_client(settings.supabase_url, settings.supabase_service_role_key)
    return supabase.create_client(
        settings.supabase_url, settings.supabase_service_role_key, options=options
    )


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


def group_summary_message() -> str:
    """Reply for a summary keyword sent in a group/room: the summary is personal, so go 1:1."""
    return _with_link(
        ["訓練摘要是個人資料，請私訊我（1 對 1 聊天）再傳「摘要」查看。"],
        "打開 x-coach",
    )


def help_message() -> str:
    """Reply for text we don't recognise."""
    lines = ["傳「摘要」或點下方選單，就能看到你的訓練摘要。"]
    if getattr(get_settings(), "chat_configured", False):
        lines.append("也可以直接問我訓練問題，例如「膝蓋內夾怎麼改善？」")
    return _with_link(lines, "打開 x-coach")


_REPLY_TIMEOUT_S = 10.0

# Text that means "show me my summary". The rich-menu button is configured as a *message*
# action sending "我的訓練摘要", so the menu and typed keywords share one code path. Compared
# after stripping and lower-casing (lower-casing is a no-op for the Chinese entries).
SUMMARY_KEYWORDS = frozenset({"我的訓練摘要", "摘要", "訓練", "紀錄", "summary"})

# Text that means "what can you do?". Everything that is neither a summary nor a help keyword is
# a question for the coach (``chat_reply_for``) when an LLM key is configured.
HELP_KEYWORDS = frozenset({"help", "幫助", "說明", "指令", "?", "？"})


# ---------------------------------------------------------------------------
# LLM conversation
# ---------------------------------------------------------------------------

# The reply token dies ~1 minute after the event and the reply call itself needs time, so the LLM
# budget is far tighter than the web chat's 60s. It is enforced twice: a monotonic wall-clock
# deadline on the whole stream (a slow but never-stalling provider), and a SHORTER httpx
# per-operation timeout for a stall between chunks — the deadline can only be checked when a
# chunk arrives, so a chunk just before it followed by a stall can still cost this much more.
# Worst case ≈ RPC 8s + deadline 25s + stall 10s + reply call, comfortably inside the token.
_CHAT_TIMEOUT_S = 25.0
_CHAT_STALL_TIMEOUT_S = 10.0
# Hard cap on generation so a verbose model cannot stream for the whole budget; ~700 tokens is
# several times the "under 200 characters" the prompt asks for.
_MAX_COMPLETION_TOKENS = 700
# LINE caps a text message at 5000 chars; a chat-room answer should be far shorter than that.
_MAX_REPLY_CHARS = 1500
# Short-lived per-user memory: the last N messages (user+assistant), dropped after the TTL.
_HISTORY_MAX_MESSAGES = 8
_HISTORY_TTL_S = 30 * 60
_MAX_USER_TEXT_CHARS = 1000

# LINE's typing indicator (1:1 chats only). Best-effort: it makes a 5–20s LLM wait feel alive.
LINE_LOADING_URL = "https://api.line.me/v2/bot/chat/loading/start"
_LOADING_SECONDS = 20
# Cosmetic call: give up fast so a slow LINE API can't eat the reply token's lifetime.
_LOADING_TIMEOUT_S = 3.0

_history_lock = threading.Lock()
# Keyed by CHAT (group id / room id / 1:1 user id), not by user: a user's private conversation
# must never be continued — and thereby quoted — inside a group they share with the bot.
_history: dict[str, tuple[float, list[dict[str, str]]]] = {}

# One FIFO "ticket lock" per chat, held for a whole turn (lookup → LLM → remember). Each webhook
# request is its own background task, so without it a follow-up sent while the previous answer is
# still streaming would run concurrently, read history that lacks that answer, and could even be
# answered first. A plain ``threading.Lock`` would stop the overlap but not the reordering (its
# waiters wake in no defined order), so turns take a ticket on ARRIVAL and run in ticket order.
# Entries are reference-counted: created by the first turn for a chat, removed when the last turn
# holding or awaiting it finishes — the registry only ever holds chats with a turn in flight, and
# nothing can drop a lane from under a caller that already holds a ticket.
_chat_lanes_guard = threading.Lock()
_chat_lanes: dict[str, "_ChatLane"] = {}


class _ChatLane:
    """FIFO turn ordering for one chat: ``next_ticket`` is issued on arrival, ``serving`` advances
    as turns finish, and ``refs`` counts turns holding or awaiting the lane."""

    __slots__ = ("cond", "next_ticket", "serving", "refs")

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.next_ticket = 0
        self.serving = 0
        self.refs = 0


@contextmanager
def _chat_turn_lock(chat_id: str) -> Iterator[None]:
    with _chat_lanes_guard:
        lane = _chat_lanes.get(chat_id)
        if lane is None:
            lane = _chat_lanes[chat_id] = _ChatLane()
        lane.refs += 1
        ticket = lane.next_ticket
        lane.next_ticket += 1
    try:
        with lane.cond:
            while lane.serving != ticket:
                lane.cond.wait()
        yield
    finally:
        with lane.cond:
            lane.serving = ticket + 1
            lane.cond.notify_all()
        with _chat_lanes_guard:
            lane.refs -= 1
            if lane.refs <= 0:
                del _chat_lanes[chat_id]

# Webhook event ids already handled, for LINE's redelivery. A redelivered event's reply token
# MAY still be valid (LINE only says it may have expired), so redeliveries are answered — unless
# this process already handled that exact event, in which case answering again would re-run the
# LLM and re-use a spent token. Bounded by TTL and size; in-process like the history.
_SEEN_EVENTS_TTL_S = 10 * 60
_SEEN_EVENTS_MAX = 2000
_seen_events_lock = threading.Lock()
_seen_events: dict[str, float] = {}


def _first_sighting(event_id: Any, now: float) -> bool:
    """Record ``event_id`` and return True the first time it is seen within the TTL.

    Events without an id (older payload shapes, malformed input) are always treated as new.
    """
    if not isinstance(event_id, str) or not event_id:
        return True
    with _seen_events_lock:
        if len(_seen_events) >= _SEEN_EVENTS_MAX:
            for eid, stamped in list(_seen_events.items()):
                if now - stamped > _SEEN_EVENTS_TTL_S:
                    del _seen_events[eid]
            if len(_seen_events) >= _SEEN_EVENTS_MAX:
                _seen_events.clear()  # pathological flood: forgetting is safer than growing.
        stamped = _seen_events.get(event_id)
        if stamped is not None and now - stamped <= _SEEN_EVENTS_TTL_S:
            return False
        _seen_events[event_id] = now
        return True


def _history_for(line_user_id: str, now: float) -> list[dict[str, str]]:
    """This user's recent turns, or ``[]`` when there are none or they have expired."""
    with _history_lock:
        entry = _history.get(line_user_id)
        if entry is None:
            return []
        stamped, turns = entry
        if now - stamped > _HISTORY_TTL_S:
            del _history[line_user_id]
            return []
        return list(turns)


def _remember(line_user_id: str, new_turns: list[dict[str, str]], now: float) -> None:
    """APPEND ``new_turns`` to this user's stored history (atomically) and evict expired users.

    Append-under-lock rather than "write back the caller's snapshot": two messages from the same
    user can be in flight at once (each webhook event is its own threadpool task), and a
    snapshot write-back would let whichever LLM call finishes last erase the other's turn.
    """
    with _history_lock:
        for uid, (stamped, _turns) in list(_history.items()):
            if now - stamped > _HISTORY_TTL_S:
                del _history[uid]
        stamped_turns = _history.get(line_user_id)
        current = stamped_turns[1] if stamped_turns is not None else []
        _history[line_user_id] = (now, (current + new_turns)[-_HISTORY_MAX_MESSAGES:])


def clear_history() -> None:
    """Forget every conversation and seen event (tests, and an admin reset if ever wired up)."""
    with _history_lock:
        _history.clear()
    with _seen_events_lock:
        _seen_events.clear()


def _summary_facts(summary: dict[str, Any] | None) -> str:
    """Render what the bot actually knows about this user, for the system prompt.

    Mirrors ``format_summary`` field-for-field so the coach and the summary card never disagree.
    """
    if summary is None:
        return (
            "USER DATA: none available here. Either this LINE user has not signed in to x-coach, "
            "or this is a group chat, where personal records are never shown. If they ask about "
            "their own records, say to sign in to x-coach with LINE and ask the bot in a 1:1 chat."
        )
    total = _safe_int(summary.get("total"))
    if total == 0:
        return (
            "USER DATA: the user has an x-coach account but no analyses yet. If they ask about "
            "their own records, say so and suggest uploading a first squat video."
        )
    lines = [f"USER DATA (from their x-coach analyses): total analyses = {total}."]
    latest = summary.get("latest")
    if isinstance(latest, dict):
        view = VIEW_LABELS.get(str(latest.get("view_type") or "unknown"), "未知")
        lines.append(
            f"Latest analysis: {_format_time(latest.get('created_at'))} (UTC+8), {view} view, "
            f"{_safe_int(latest.get('fault_count'))} fault(s) detected."
        )
    top = [f for f in (summary.get("top_faults") or []) if isinstance(f, dict)]
    if top:
        ranked = ", ".join(f"{_fault_label(f)} x{_safe_int(f.get('count'))}" for f in top)
        lines.append(f"Most frequent faults across all analyses: {ranked}.")
    return "\n".join(lines)


def _chat_system_prompt(summary: dict[str, Any] | None) -> str:
    """The LINE coach's system prompt: persona, honesty rules, the user's facts, LINE formatting."""
    return (
        "You are the x-coach AI coach, chatting with a user inside the LINE messaging app. "
        "You help with strength-training technique (squat and other common movements), training "
        "habits, and reading their x-coach results.\n\n"
        "HONESTY RULES — these are absolute:\n"
        "- The only facts you have about THIS user are in USER DATA below. Never invent analyses, "
        "measurements, faults, dates, or progress that are not listed there.\n"
        "- General coaching knowledge (what a fault is, why it happens, standard corrective cues) "
        "is fine, but say plainly when something is general advice rather than something measured "
        "in their videos.\n"
        "- You are not a doctor: for pain or injury, advise seeing a professional.\n"
        "- If you cannot answer from what you know, say so briefly.\n\n"
        f"{_summary_facts(summary)}\n\n"
        "FORMAT — this is a chat bubble, not a document:\n"
        "- Reply in Traditional Chinese (Taiwan) unless the user writes in another language.\n"
        "- Plain text only: NO Markdown (no **bold**, no # headings, no tables). Short lines and "
        "simple numbered steps are fine.\n"
        "- Keep it short: usually under 200 characters, at most a few short paragraphs.\n"
        "- To see their full report or analyse a new video, point them to the x-coach app."
    )


def show_loading(line_user_id: str) -> None:
    """Show LINE's typing indicator while the LLM works; failures are logged, never raised."""
    settings = get_settings()
    try:
        response = httpx.post(
            LINE_LOADING_URL,
            headers={"Authorization": f"Bearer {settings.line_messaging_access_token}"},
            json={"chatId": line_user_id, "loadingSeconds": _LOADING_SECONDS},
            timeout=_LOADING_TIMEOUT_S,
        )
    except httpx.HTTPError:
        logger.info("LINE bot: loading indicator request failed")
        return
    if response.status_code not in (200, 202):
        logger.info("LINE bot: loading indicator rejected with status %s", response.status_code)


_CHAT_UNAVAILABLE = "教練暫時離線，請稍後再試一次。"


def chat_reply_for(line_user_id: str, text: str, *, chat_id: str | None = None) -> str:
    """Answer a free-text message with the LLM, grounded in this user's summary and recent turns.

    ``chat_id`` is where the message was sent — the group/room id, or the user id for a 1:1 chat
    (the default). It keys the history, and it gates the personal data: the sender's training
    summary is loaded ONLY in their private chat, so nothing about one member's training is ever
    spoken into a group. The typing indicator is likewise 1:1-only (LINE rejects it elsewhere).

    Degrades rather than raises at every step: no LLM key → the help reply; summary lookup failure
    → chat without user facts; LLM failure / empty completion → a short apology (and the failed
    turn is not remembered, so the next message starts clean).
    """
    if not getattr(get_settings(), "chat_configured", False):
        return help_message()

    chat_id = chat_id or line_user_id
    with _chat_turn_lock(chat_id):
        return _chat_turn(line_user_id, text, chat_id)


def _chat_turn(line_user_id: str, text: str, chat_id: str) -> str:
    """One LLM turn for ``chat_id`` — the body of ``chat_reply_for``, run under the chat's lock."""
    private = chat_id == line_user_id
    text = text.strip()[:_MAX_USER_TEXT_CHARS]
    summary: dict[str, Any] | None = None
    if private:
        try:
            summary = summary_for_line_user(line_user_id)
        except Exception:  # noqa: BLE001 — chat is still useful without the user's facts.
            logger.exception("LINE bot: training-summary lookup failed; chatting without it")
        show_loading(line_user_id)

    now = time.time()
    history = _history_for(chat_id, now)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _chat_system_prompt(summary)},
        *history,
        {"role": "user", "content": text},
    ]
    parts: list[str] = []
    deadline = time.monotonic() + _CHAT_TIMEOUT_S
    cut_short = False
    try:
        for chunk in chat_service._stream_completion(
            messages,
            default_chat_model(),
            timeout=_CHAT_STALL_TIMEOUT_S,
            extra_body={"max_tokens": _MAX_COMPLETION_TOKENS},
        ):
            parts.append(chunk)
            if time.monotonic() > deadline:
                # Breaking out closes the generator and with it the httpx stream. Whatever has
                # arrived so far is still a better reply than an apology.
                cut_short = True
                logger.warning("LINE bot: LLM stream hit the wall-clock deadline; replying early")
                break
    except RuntimeError:
        logger.warning("LINE bot: LLM request failed")
        return _CHAT_UNAVAILABLE
    answer = "".join(parts).strip()
    if not answer:
        return _CHAT_UNAVAILABLE
    if len(answer) > _MAX_REPLY_CHARS:
        answer = answer[: _MAX_REPLY_CHARS - 1].rstrip() + "…"
    elif cut_short:
        answer += "…"

    _remember(
        chat_id,
        [{"role": "user", "content": text}, {"role": "assistant", "content": answer}],
        now,
    )
    return answer


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


def chat_key_for(event: Any) -> str | None:
    """Where an event was sent — group id, room id, or (1:1) the user id; ``None`` if unknowable.

    This is the unit of conversation: history is stored per key, the sender's personal summary
    is only ever loaded when the key IS the sender (a private chat), and the router processes
    events that share a key in order so a follow-up always sees the turn before it.
    """
    source = event.get("source") if isinstance(event, dict) else None
    if not isinstance(source, dict):
        return None
    key = source.get("groupId") or source.get("roomId") or source.get("userId")
    return str(key) if key else None


def iter_replies(payload: dict[str, Any]) -> Iterator[dict[str, str]]:
    """Yield the reply for each answerable event in a webhook payload, one at a time.

    A generator rather than a list so the router can SEND each reply before the next one is
    computed: LINE batches events, and each free-text event can spend up to ``_CHAT_TIMEOUT_S``
    in the LLM — computing a whole batch first could push the first event's reply token past its
    ~1-minute life. The decision logic stays a pure-ish function (mocked seams: the summary RPC
    and the LLM stream) and the router owns the reply I/O. Non-text and malformed events are
    skipped silently: LINE delivers many event types we don't answer. Routing: summary keywords
    → summary; help keywords / empty → help; anything else → the LLM coach (or help when no LLM
    key is configured).
    """
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("type") != "message":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("type") != "text":
            continue
        reply_token = event.get("replyToken")
        source = event.get("source") if isinstance(event.get("source"), dict) else {}
        line_user_id = source.get("userId")
        if not reply_token or not line_user_id:
            continue
        if not _first_sighting(event.get("webhookEventId"), time.time()):
            continue  # this process already answered this exact event (LINE redelivery).
        line_user_id = str(line_user_id)
        chat_id = chat_key_for(event) or line_user_id

        raw_text = str(message.get("text") or "").strip()
        text = raw_text.lower()
        if text in SUMMARY_KEYWORDS:
            # Personal data stays in the sender's private chat — never posted into a group/room.
            answer = _reply_text_for(line_user_id) if chat_id == line_user_id else group_summary_message()
        elif not text or text in HELP_KEYWORDS:
            answer = help_message()
        else:
            answer = chat_reply_for(line_user_id, raw_text, chat_id=chat_id)
        yield {"reply_token": str(reply_token), "text": answer}


def handle_events(payload: dict[str, Any]) -> list[dict[str, str]]:
    """All planned replies for a payload as a list — ``iter_replies`` fully drained."""
    return list(iter_replies(payload))


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
