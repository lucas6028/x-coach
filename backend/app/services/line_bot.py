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
