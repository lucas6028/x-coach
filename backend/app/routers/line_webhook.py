"""POST /api/line/webhook — the LINE Messaging API bot's event sink.

The thin HTTP layer over ``services/line_bot`` (which documents the whole flow). Two things
are specific to this router and worth stating here:

* The handler is ``async`` and reads ``await request.body()`` because the signature must be
  computed over the RAW bytes — FastAPI's parsed model would not round-trip byte-identically.
* Once the signature checks out, it ALWAYS answers 200. LINE treats a non-2xx as a webhook
  failure, and the event has already been consumed, so a retry would only duplicate replies.
* The 200 is sent BEFORE the events are handled. LINE expects a prompt acknowledgement (a slow
  webhook is flagged, and with redelivery enabled re-sent — which would re-run the LLM and
  re-use a spent reply token), and a free-text event can spend ~25s in the LLM. So once the
  signature checks out the handler only schedules ``_process`` as a ``BackgroundTasks`` job
  (a sync function, which Starlette runs in its threadpool — never on the event loop) and
  returns. The reply token (~1 minute) is what keeps the answer deliverable after the ack.
* Events from DIFFERENT chats are independent, so ``_process`` answers them in parallel
  (bounded pool) and sends each reply the moment it is computed: serial handling of two or
  three free-text events would push the later reply tokens past their lifetime. Events from the
  SAME chat stay in order on one worker, so a follow-up ("那第二點呢？") is answered after — and
  with the history of — the message before it.

Returns 503 when the bot isn't configured, mirroring the rest of the API.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

from backend.app.services import line_bot
from backend.app.settings import get_settings

router = APIRouter(prefix="/api/line", tags=["line"])

logger = logging.getLogger(__name__)


@router.post("/webhook")
async def line_webhook(request: Request, background: BackgroundTasks) -> dict[str, bool]:
    settings = get_settings()
    if not getattr(settings, "line_messaging_configured", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The LINE bot is not configured on the server.",
        )

    raw_body = await request.body()
    if not line_bot.verify_signature(raw_body, request.headers.get("x-line-signature")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid LINE signature.",
        )

    background.add_task(_process, raw_body)
    return {"ok": True}


# Upper bound on events answered concurrently for ONE webhook payload (LINE batches at most a
# handful); each may hold an LLM stream open, so this also bounds provider fan-out per request.
_MAX_PARALLEL_EVENTS = 4


def _answer_events(payload: dict[str, Any]) -> None:
    """Compute and send the reply for each event in ``payload``, one event at a time; never raises."""
    try:
        for planned in line_bot.iter_replies(payload):
            try:
                line_bot.reply(planned["reply_token"], planned["text"])
            except Exception:  # noqa: BLE001 — one bad reply must not skip the rest of the events.
                # ``line_bot.reply`` only catches httpx.HTTPError itself; anything else (a bad
                # reply dict, an unexpected exception type) must not abort the loop, or every
                # event after the failing one in this webhook request goes unanswered.
                # Never log the reply token/text/userId here — same reasoning as below.
                logger.exception("LINE bot: reply failed for one event")
    except Exception:  # noqa: BLE001 — a signed event is acknowledged no matter what.
        # Never log the raw body/payload here: it can carry a LINE userId or reply text.
        logger.exception("LINE bot: webhook handling failed")


def _process(raw_body: bytes) -> None:
    """Background job: parse the payload and answer its events, in parallel when there are several.

    Synchronous by design — ``BackgroundTasks`` runs a sync callable in the threadpool, so the
    blocking work (RPC, LINE HTTP, LLM stream) never touches the event loop. Never raises.
    """
    try:
        payload = json.loads(raw_body.decode("utf-8"))
        events = payload.get("events") if isinstance(payload, dict) else None
        events = list(events) if isinstance(events, list) else []
    except Exception:  # noqa: BLE001 — a signed event is acknowledged no matter what.
        logger.exception("LINE bot: webhook payload could not be parsed")
        return
    lanes = _lanes_by_chat(events)
    if len(lanes) <= 1:
        _answer_events({"events": events})
        return
    with ThreadPoolExecutor(max_workers=min(_MAX_PARALLEL_EVENTS, len(lanes))) as pool:
        # ``list(...)`` drains the iterator so every event is answered before the pool closes;
        # ``_answer_events`` swallows its own exceptions, so nothing propagates out of ``map``.
        list(pool.map(lambda lane: _answer_events({"events": lane}), lanes))


def _lanes_by_chat(events: list[Any]) -> list[list[Any]]:
    """Group events by chat (payload order preserved within a lane and across lanes' first events).

    One lane per chat so the pool can run chats side by side while each chat's messages are
    answered strictly in the order LINE delivered them. Events whose chat can't be determined
    (malformed — ``iter_replies`` will skip them anyway) share one lane.
    """
    lanes: dict[str | None, list[Any]] = {}
    for event in events:
        lanes.setdefault(line_bot.chat_key_for(event), []).append(event)
    return list(lanes.values())
