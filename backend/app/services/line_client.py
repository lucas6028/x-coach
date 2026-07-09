"""LINE Messaging API transport: webhook signature verification + sending replies.

Two concerns, both stdlib/httpx only (no ``line-bot-sdk`` dependency — the surface we use is tiny):

  * ``verify_signature`` — every LINE webhook request carries an ``X-Line-Signature`` header, the
    Base64 of HMAC-SHA256(channel_secret, raw_request_body). We recompute it over the **raw** body
    (not the re-serialized JSON, which would differ byte-for-byte) and compare in constant time. A
    failing check means the request did not come from LINE and is rejected before any processing.
  * ``reply_message`` / ``push_message`` — post a text reply. ``reply_message`` uses the one-time
    ``replyToken`` from the webhook event (free, but valid only briefly); ``push_message`` targets a
    user id directly (used only as a fallback). Both authenticate with the channel access token.

The ``httpx`` import is deferred into the send functions (mirroring ``services/chat``) so importing
this module — and the router — stays light and the unit tests patch the send seam without the network.
"""

from __future__ import annotations

import base64
import hashlib
import hmac

# LINE endpoints. Kept as module constants so tests can assert the target without hard-coding it twice.
_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
_PUSH_URL = "https://api.line.me/v2/bot/message/push"
_REQUEST_TIMEOUT_S = 10.0


def verify_signature(channel_secret: str, body: bytes, signature: str | None) -> bool:
    """True iff ``signature`` is the valid LINE HMAC over the raw ``body`` for ``channel_secret``.

    Computed over the exact bytes LINE sent — the caller must pass ``await request.body()``, never a
    re-serialized dict. A missing secret or signature returns ``False`` (fail closed). The comparison
    is constant-time to avoid leaking the expected digest via timing.
    """
    if not channel_secret or not signature:
        return False
    digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)


def _post(url: str, access_token: str, payload: dict) -> None:
    """POST ``payload`` to a LINE Messaging API endpoint, raising on any transport/HTTP failure.

    Deferred ``httpx`` import keeps the router import light. Raises ``RuntimeError`` so the caller can
    log-and-swallow (a failed reply must never 500 the webhook back at LINE, which would trigger
    retries and duplicate answers).
    """
    import httpx  # deferred: only needed when actually sending.

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=_REQUEST_TIMEOUT_S)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 — any failure here is an upstream/transport problem.
        raise RuntimeError(f"LINE API request failed: {exc}") from exc


def reply_message(*, access_token: str, reply_token: str, text: str) -> None:
    """Send a single text reply using the webhook event's one-time ``reply_token``."""
    _post(
        _REPLY_URL,
        access_token,
        {"replyToken": reply_token, "messages": [{"type": "text", "text": text}]},
    )


def push_message(*, access_token: str, to: str, text: str) -> None:
    """Send a single text message to a LINE user id (fallback for when no reply token is available)."""
    _post(_PUSH_URL, access_token, {"to": to, "messages": [{"type": "text", "text": text}]})
