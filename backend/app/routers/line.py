"""LINE Messaging API endpoints: the webhook that powers the LINE AI coach + link-code minting.

Product shape ("continue a web analysis on LINE"): a signed-in web user mints a one-time **link
code** (``POST /api/line/link-code``, auth-gated), types it to the LINE bot, and from then on chats
with the SAME grounded coach they used on the web — the LINE binding carries the analysis grounding
snapshot, so answers stay grounded in that specific squat rep.

Two endpoints:
  * ``POST /api/line/webhook``    — called by LINE. Verifies the ``X-Line-Signature`` over the raw
    body, then processes each event in a background task and returns ``200 {}`` immediately (LINE
    expects a fast ack; the LLM round-trip happens after the ack, and the reply is sent with the
    event's reply token). Never trusts the body until the signature checks out.
  * ``POST /api/line/link-code``  — called by the authenticated web app to mint a code that binds a
    LINE account to the current analysis. 503 unless the integration is fully configured.

The webhook is unauthenticated by HTTP standards (no bearer token) — its authenticity comes solely
from the HMAC signature, so signature verification is the security boundary and runs first.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from backend.app.auth import CurrentUser, get_current_user
from backend.app.routers.chat import ChatContext
from backend.app.services import chat as chat_service
from backend.app.services import line_client, line_store
from backend.app.settings import default_chat_model, get_settings

logger = logging.getLogger("x_coach.line")

router = APIRouter(prefix="/api/line", tags=["line"])

# Keywords that mark a message as "redeem this link code" rather than a coaching question. A bare
# 6-char code (what the web hands the user) is also accepted, so either "連結 ABC123" or "ABC123" works.
_BIND_KEYWORDS = ("連結", "綁定", "link", "bind")

# User-facing bot copy (the coach answers in the user's language via the system prompt; these are the
# fixed control messages). Traditional Chinese to match the product's primary audience.
_MSG_LINK_HELP = (
    "你還沒連結網頁上的分析。請到 x-coach 網頁產生連結碼，再傳給我，"
    "例如「連結 ABC123」或直接貼上代碼。"
)
_MSG_LINK_INVALID = "連結碼無效或已過期，請回網頁重新產生一組再傳給我。"
_MSG_LINK_OK = "✅ 已連結你的分析！現在可以直接問我關於這次深蹲的任何問題。"
_MSG_LLM_ERROR = "抱歉，我現在無法回覆，請稍後再試一次。"
_MSG_NON_TEXT = "目前我只能回覆文字訊息喔 🙏 用文字問我關於你深蹲的問題吧！"
_MSG_WELCOME = (
    "嗨！我是你的 x-coach 深蹲教練 🏋️\n"
    "先到 x-coach 網頁分析你的深蹲並產生連結碼，傳給我之後（例如「連結 ABC123」），"
    "就能在這裡繼續追問這次的分析結果。"
)


class LinkCodeRequest(BaseModel):
    # The same grounding blob ``buildChatContext(analysis)`` builds for the web chat. Snapshotted onto
    # the code so the redeemed LINE binding is self-contained and the webhook never reads user tables.
    context: ChatContext


class LinkCodeResponse(BaseModel):
    code: str
    # Minutes the code stays valid, surfaced so the web UI can tell the user how long they have.
    expires_in_minutes: int = Field(default=15)


@router.post("/link-code", response_model=LinkCodeResponse)
def create_link_code(
    body: LinkCodeRequest,
    user: CurrentUser = Depends(get_current_user),
) -> LinkCodeResponse:
    """Mint a one-time code binding the caller's current analysis to whichever LINE account redeems it.

    Auth-gated (only a signed-in user can link their own analysis) and 503 unless the LINE integration
    is fully configured — minting a code the webhook can't honour would be a dead end.
    """
    if not get_settings().line_configured:
        raise HTTPException(status_code=503, detail="The LINE integration is not configured.")

    context = body.context.model_dump()
    code = line_store.create_link_code(
        user_id=user.id, video_id=context.get("video_id"), context=context
    )
    return LinkCodeResponse(code=code)


@router.post("/webhook")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_line_signature: str | None = Header(default=None),
) -> dict:
    """Receive LINE webhook events. Verify the signature, then ack fast and process in the background.

    Returns ``200 {}`` in all normal cases so LINE doesn't retry: a bad signature is a ``400`` (the
    request isn't from LINE), but an unconfigured server or an unparseable/eventless body simply acks
    and does nothing. The actual work (LLM + reply) runs in a background task after the ack, using the
    event's reply token.
    """
    body = await request.body()
    settings = get_settings()

    # Unconfigured: ack without processing (nothing to verify against, nothing that could reply).
    if not settings.line_configured:
        return {}

    if not line_client.verify_signature(settings.line_channel_secret, body, x_line_signature):
        raise HTTPException(status_code=400, detail="Invalid LINE signature.")

    try:
        events = json.loads(body).get("events", [])
    except (json.JSONDecodeError, AttributeError):
        return {}  # a malformed body that still signed correctly — nothing to do.

    if events:
        background_tasks.add_task(_process_events, events)
    return {}


def _process_events(events: list[dict[str, Any]]) -> None:
    """Handle each webhook event, isolating failures so one bad event can't drop the rest.

    Runs in a background task (post-ack), so exceptions here never reach LINE as a 500 (which would
    make it retry and double-answer). Each event is wrapped: a failure is logged and skipped.
    """
    for event in events:
        try:
            _handle_event(event)
        except Exception:  # noqa: BLE001 — a single event's failure must not abort the batch.
            logger.exception("Failed to handle LINE event")


def _handle_event(event: dict[str, Any]) -> None:
    """Dispatch one event: text message, non-text message, or a new follower (``follow``)."""
    etype = event.get("type")
    reply_token = event.get("replyToken")
    line_user_id = (event.get("source") or {}).get("userId")

    if etype == "follow" and reply_token:
        _reply(reply_token, _MSG_WELCOME)
        return

    if etype != "message" or not reply_token:
        return  # unfollow/join/postback/etc. — nothing to say.

    message = event.get("message") or {}
    if message.get("type") != "text":
        _reply(reply_token, _MSG_NON_TEXT)
        return

    text = (message.get("text") or "").strip()
    if not line_user_id or not text:
        return

    code = _extract_link_code(text)
    if code is not None:
        _redeem(line_user_id=line_user_id, code=code, reply_token=reply_token)
    else:
        _answer(line_user_id=line_user_id, text=text, reply_token=reply_token)


def _extract_link_code(text: str) -> str | None:
    """Return the link code if ``text`` is a redeem request, else ``None`` (it's a coaching question).

    Recognises an explicit "連結 ABC123"/"link ABC123" prefix, or a bare token that has the exact shape
    of a code (the code length, all from the code alphabet) so a user can just paste it. A normal
    question — which has spaces or characters outside the code alphabet — falls through to the coach.
    """
    lowered = text.lower()
    for kw in _BIND_KEYWORDS:
        if lowered.startswith(kw):
            candidate = text[len(kw):].strip()
            return candidate or None
    stripped = text.strip()
    if len(stripped) == line_store._CODE_LENGTH and all(
        ch in line_store._CODE_ALPHABET for ch in stripped.upper()
    ):
        return stripped
    return None


def _redeem(*, line_user_id: str, code: str, reply_token: str) -> None:
    """Redeem a link code into a binding and confirm, or report an invalid/expired code."""
    binding = line_store.redeem_link_code(line_user_id=line_user_id, code=code)
    _reply(reply_token, _MSG_LINK_OK if binding else _MSG_LINK_INVALID)


def _answer(*, line_user_id: str, text: str, reply_token: str) -> None:
    """Generate a grounded coaching reply for a bound LINE user and send it.

    Unbound users are steered to the linking flow. For a bound user the new turn is appended, answered
    with the grounded ``answer_once`` over the binding's snapshot context, persisted, and replied. An
    LLM failure sends a graceful fallback and does NOT persist the failed turn (so a retry re-asks
    cleanly instead of stacking an unanswered user turn).
    """
    binding = line_store.get_binding(line_user_id=line_user_id)
    if binding is None:
        _reply(reply_token, _MSG_LINK_HELP)
        return

    history = list(binding.get("messages") or [])
    history.append({"role": "user", "content": text})
    try:
        answer = chat_service.answer_once(
            messages=history, context=binding.get("context") or {}, model=default_chat_model()
        )
    except RuntimeError:
        logger.exception("LLM answer failed for LINE user")
        _reply(reply_token, _MSG_LLM_ERROR)
        return

    history.append({"role": "assistant", "content": answer})
    line_store.save_binding_messages(line_user_id=line_user_id, messages=history)
    _reply(reply_token, answer)


def _reply(reply_token: str, text: str) -> None:
    """Send a text reply, swallowing (but logging) transport errors so a failed send never propagates.

    A raised error out of a background task is invisible to LINE anyway; catching it here keeps the
    log clean and lets a multi-event batch continue.
    """
    try:
        line_client.reply_message(
            access_token=get_settings().line_channel_access_token,
            reply_token=reply_token,
            text=text,
        )
    except RuntimeError:
        logger.exception("Failed to send LINE reply")
