"""LINE Login bridge: verify a LIFF ID token, then mint a real Supabase session.

The LIFF app runs inside the LINE client where the user is already LINE-authenticated, so
instead of a redirect round-trip it POSTs its LINE ID token to ``/api/auth/line`` and gets
back a normal Supabase session (see ``routers/auth_line``). Flow:

  1. Verify the ID token against LINE's official verify endpoint — LINE checks the
     signature, expiry and audience (our channel id); we additionally require ``sub``.
  2. Find-or-create the matching ``auth.users`` row. LINE users get a *deterministic
     synthetic auth email* derived from the (pairwise) LINE ``sub`` —
     ``line_<sub>@line.invalid`` — so find-or-create needs no mapping table, is idempotent
     under concurrent first logins, and can never collide with a real email/password
     account. A real LINE email (only present if the channel passed LINE's ``email``
     scope review) is kept in ``user_metadata`` for display, never as the auth email.
  3. Mint a session via the Admin API (``generate_link`` magiclink → ``verify_otp``) and
     return access+refresh tokens; the frontend hands them to ``supabase.auth.setSession``
     and from there everything (RLS, history, chat) behaves exactly like any other login.

This is the ONE place the backend uses the service_role key (``SUPABASE_SERVICE_ROLE_KEY``,
see ``settings``): it is needed only to create the user and generate the one-shot link.
The minted session is an ordinary user JWT — every downstream DB call still runs as the
user with RLS as the backstop, unchanged from the other login methods.

The ``supabase`` import is deferred (as in ``services/store``) so the module stays light
and unit tests can fake the package without it installed. ``httpx`` is a top-level import:
it is part of the lean CI set and the verify call is this module's whole reason to exist.
"""

from __future__ import annotations

from typing import Any

import httpx

from backend.app.settings import get_settings

# LINE's ID-token verification endpoint (LINE Login v2.1). Signature/expiry/audience are
# validated by LINE itself; a bad or expired token comes back as HTTP 400.
LINE_VERIFY_URL = "https://api.line.me/oauth2/v2.1/verify"
_VERIFY_TIMEOUT_S = 10.0


class LineAuthError(Exception):
    """A LINE-login failure the router should surface as an HTTP error."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def synthetic_email(sub: str) -> str:
    """The deterministic auth email for a LINE subject.

    ``.invalid`` is the reserved never-resolvable TLD, so nothing is ever delivered there
    (``generate_link`` doesn't send mail anyway) and it reads as clearly synthetic in the
    admin users table. Lower-cased because Supabase normalises emails to lower case — the
    derived address must be byte-identical on every login for find-or-create to work.
    """
    return f"line_{sub.strip().lower()}@line.invalid"


def verify_line_id_token(id_token: str) -> dict[str, Any]:
    """Validate ``id_token`` with LINE and return its claims, or raise ``LineAuthError``.

    Uses LINE's verify endpoint rather than local JWKS verification for the same reason
    ``auth._verify`` delegates to Supabase: the platform checks signature (ES256), expiry
    and audience (our channel id) authoritatively, at the cost of one round-trip.
    """
    settings = get_settings()
    try:
        response = httpx.post(
            LINE_VERIFY_URL,
            data={"id_token": id_token, "client_id": settings.line_channel_id},
            timeout=_VERIFY_TIMEOUT_S,
        )
    except httpx.HTTPError as exc:
        raise LineAuthError(502, "Could not reach LINE to verify the token.") from exc

    if response.status_code != 200:
        # LINE answers 400 with an error_description for invalid/expired/wrong-audience
        # tokens; anything not-200 means the token is unusable.
        raise LineAuthError(401, "LINE ID token is invalid or expired.")

    claims = response.json()
    if not isinstance(claims, dict) or not claims.get("sub"):
        raise LineAuthError(401, "LINE ID token is missing its subject.")
    return claims


def _admin_client() -> Any:
    """Build a service_role Supabase client (Admin API access; bypasses RLS)."""
    from supabase import create_client  # deferred heavy import (gotrue/postgrest/...)

    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def _anon_client() -> Any:
    """Build an anon Supabase client (used to redeem the one-shot link into a session)."""
    from supabase import create_client  # deferred heavy import

    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_anon_key)


def _ensure_user(admin: Any, claims: dict[str, Any]) -> None:
    """Create the ``auth.users`` row for this LINE subject if it doesn't exist yet.

    ``create_user`` on an already-registered (synthetic) email raises — that is the normal
    "returning user" case (or a concurrent first login losing the race), so the error is
    swallowed. If creation failed for a *real* reason (bad service key, project down), the
    session mint right after fails loudly with a 502, so nothing is silently lost.
    """
    metadata = {
        key: value
        for key, value in {
            "full_name": claims.get("name"),
            "avatar_url": claims.get("picture"),
            "line_sub": claims.get("sub"),
            "line_email": claims.get("email"),
        }.items()
        if value
    }
    try:
        admin.auth.admin.create_user(
            {
                "email": synthetic_email(claims["sub"]),
                # The synthetic address can't receive a confirmation mail; LINE already
                # authenticated the person, so mark the account confirmed at creation.
                "email_confirm": True,
                "user_metadata": metadata,
            }
        )
    except Exception:  # noqa: BLE001 — duplicate email == returning user, by design.
        pass


def _mint_session(admin: Any, email: str) -> dict[str, str]:
    """Mint a Supabase session for ``email`` and return its access+refresh tokens.

    ``generate_link`` (Admin API) produces a one-shot magiclink token *without sending any
    mail*; redeeming its ``hashed_token`` through the anon client's ``verify_otp`` yields a
    normal user session — the documented Admin-API way to sign a user in server-side.
    """
    link = admin.auth.admin.generate_link({"type": "magiclink", "email": email})
    hashed_token = getattr(getattr(link, "properties", None), "hashed_token", None)
    if not hashed_token:
        raise LineAuthError(502, "Could not create a sign-in link for the LINE user.")

    result = _anon_client().auth.verify_otp({"type": "magiclink", "token_hash": hashed_token})
    session = getattr(result, "session", None)
    access_token = getattr(session, "access_token", None)
    refresh_token = getattr(session, "refresh_token", None)
    if not (access_token and refresh_token):
        raise LineAuthError(502, "Could not mint a session for the LINE user.")
    return {"access_token": str(access_token), "refresh_token": str(refresh_token)}


def login_with_line(id_token: str) -> dict[str, str]:
    """The full bridge: LINE ID token in, Supabase session tokens out (or ``LineAuthError``)."""
    claims = verify_line_id_token(id_token)
    admin = _admin_client()
    _ensure_user(admin, claims)
    try:
        return _mint_session(admin, synthetic_email(claims["sub"]))
    except LineAuthError:
        raise
    except Exception as exc:  # noqa: BLE001 — any Supabase failure here is a gateway error.
        raise LineAuthError(502, "Supabase rejected the LINE sign-in.") from exc
