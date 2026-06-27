"""Supabase JWT verification and FastAPI auth dependencies.

The backend is stateless: it does not issue or store sessions. The frontend authenticates
with Supabase Auth, receives an access token (JWT), and sends it as ``Authorization: Bearer
<token>``. We verify that token locally with the project's JWT secret (HS256) to reject forged
or expired tokens early, and we carry the raw token forward so DB calls can act *as the user*
(``services/store``) — letting Postgres RLS be the ownership backstop.

Two dependencies:
  - ``get_current_user``  — required auth; 401 when no valid token is present (history endpoints).
  - ``get_optional_user`` — returns the user if a token is present, else ``None`` (demo uploads).
    A present-but-invalid token still 401s, so a logged-in client with a stale session is told
    to refresh rather than silently dropping into anonymous mode.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, status

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
    """Verify a Supabase access token and return the caller, or raise 401/503."""
    import jwt  # deferred: PyJWT is only needed when a request actually carries a token.

    settings = get_settings()
    if not settings.supabase_jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured on the server.",
        )
    try:
        claims = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
        )
    except jwt.PyJWTError as exc:  # invalid signature, expired, wrong audience, malformed...
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has no subject.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return CurrentUser(id=str(user_id), token=token, email=claims.get("email"))


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


def get_optional_user(authorization: str | None = Header(default=None)) -> CurrentUser | None:
    """Optional auth: return the user if a (well-formed) token is present, else ``None``.

    A malformed/absent header → anonymous (demo). A present token that fails verification →
    401, so a logged-in client refreshes its session instead of silently not persisting.
    """
    token = _extract_bearer(authorization)
    if token is None:
        return None
    return _verify(token)
