"""Supabase token validation and FastAPI auth dependencies.

The backend is stateless: it does not issue or store sessions. The frontend authenticates
with Supabase Auth, receives an access token, and sends it as ``Authorization: Bearer
<token>``. We validate that token through Supabase's Auth API (``_verify``) — which works for
any signing scheme the project uses — and carry the raw token forward so DB calls can act *as
the user* (``services/store``), letting Postgres RLS be the ownership backstop.

Two dependencies:
  - ``get_current_user``  — required auth; 401 when no valid token is present (history endpoints).
  - ``get_optional_user`` — returns the user if a token is present, else ``None`` (demo uploads).
    A present-but-invalid token still 401s, so a logged-in client with a stale session is told
    to refresh rather than silently dropping into anonymous mode.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status

from backend.app.services import store
from backend.app.settings import get_settings


@dataclass(frozen=True)
class CurrentUser:
    """The authenticated caller. ``token`` is the raw JWT, forwarded to Supabase for RLS."""

    id: str
    token: str
    email: str | None = None


def _extract_bearer(authorization: str | None) -> str | None:
    """Pull the token out of an ``Authorization: Bearer <token>`` header, or ``None``."""
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        return None
    return parts[1].strip()


def _verify(token: str) -> CurrentUser:
    """Validate a Supabase access token via the Auth API and return the caller, or raise 401/503.

    We ask Supabase to validate the token (``GET /auth/v1/user`` under the hood) rather than
    verifying its signature locally. That works regardless of the project's JWT signing scheme:
    legacy HS256 shared secret OR the newer asymmetric signing keys (ES256/RS256 via JWKS) that
    recent projects default to. Local HS256 verification silently breaks the moment a project
    uses asymmetric keys; this does not. Cost is one Auth round-trip per request, which is fine
    at prototype scale (cache or move to JWKS verification if it ever matters).
    """
    settings = get_settings()
    if not settings.auth_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured on the server.",
        )

    from supabase import create_client  # deferred: only needed when a request carries a token.

    client = create_client(settings.supabase_url, settings.supabase_anon_key)
    try:
        response = client.auth.get_user(token)
    except Exception as exc:  # noqa: BLE001 — any failure here means the token isn't usable.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user = getattr(response, "user", None)
    user_id = getattr(user, "id", None)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return CurrentUser(id=str(user_id), token=token, email=getattr(user, "email", None))


def get_current_user(authorization: str | None = Header(default=None)) -> CurrentUser:
    """Required auth: raise 401 unless a valid Supabase JWT is present."""
    token = _extract_bearer(authorization)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _verify(token)


def get_admin_user(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Admin-gated auth: require a valid session AND the 'admin' role, else 403.

    Delegates the role check to ``store.is_admin`` (queried with the user's own JWT, RLS-scoped), so
    every admin endpoint re-verifies server-side on each request — the frontend gating is only UX.
    ``store`` is imported at module top: it defers its own ``supabase`` import and does not import
    ``auth``, so there is no import cycle.
    """
    if not store.is_admin(token=user.token, user_id=user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges are required.",
        )
    return user


def get_optional_user(authorization: str | None = Header(default=None)) -> CurrentUser | None:
    """Optional auth: return the user if a (well-formed) token is present, else ``None``.

    A malformed/absent header → anonymous (demo). A present token that fails verification →
    401, so a logged-in client refreshes its session instead of silently not persisting.
    """
    token = _extract_bearer(authorization)
    if token is None:
        return None
    return _verify(token)
