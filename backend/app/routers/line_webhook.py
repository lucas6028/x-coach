"""POST /api/line/webhook — the LINE Messaging API bot's event sink.

The thin HTTP layer over ``services/line_bot`` (which documents the whole flow). Two things
are specific to this router and worth stating here:

* The handler is ``async`` and reads ``await request.body()`` because the signature must be
  computed over the RAW bytes — FastAPI's parsed model would not round-trip byte-identically.
* Once the signature checks out, it ALWAYS answers 200. LINE treats a non-2xx as a webhook
  failure, and the event has already been consumed, so a retry would only duplicate replies.

Returns 503 when the bot isn't configured, mirroring the rest of the API.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request, status

from backend.app.services import line_bot
from backend.app.settings import get_settings

router = APIRouter(prefix="/api/line", tags=["line"])

logger = logging.getLogger(__name__)


@router.post("/webhook")
async def line_webhook(request: Request) -> dict[str, bool]:
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

    try:
        payload = json.loads(raw_body.decode("utf-8"))
        for planned in line_bot.handle_events(payload):
            line_bot.reply(planned["reply_token"], planned["text"])
    except Exception:  # noqa: BLE001 — a signed event is acknowledged no matter what.
        # Never log the raw body/payload here: it can carry a LINE userId or reply text.
        logger.exception("LINE bot: webhook handling failed")
    return {"ok": True}
