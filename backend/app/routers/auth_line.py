"""POST /api/auth/line — exchange a LINE (LIFF) ID token for a Supabase session.

The thin HTTP layer over ``services/line_auth`` (which documents the whole flow). Kept as
its own router because it is the only *unauthenticated* POST that ends in a session: the
caller proves who they are with the LINE ID token itself, not a bearer header.

Returns 503 when the bridge isn't configured (missing LINE channel id / service_role key),
mirroring how the rest of the API reports unconfigured auth.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.services import line_auth
from backend.app.settings import get_settings

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LineLoginRequest(BaseModel):
    # A LINE ID token is a three-part JWT, typically ~700-1500 chars; the bounds just
    # reject junk early (422) before we spend a round-trip on LINE's verify endpoint.
    id_token: str = Field(min_length=16, max_length=8192)


class LineLoginResponse(BaseModel):
    """A minted Supabase session — the frontend passes both to ``supabase.auth.setSession``."""

    access_token: str
    refresh_token: str


@router.post("/line", response_model=LineLoginResponse)
def line_login(payload: LineLoginRequest) -> LineLoginResponse:
    settings = get_settings()
    if not settings.line_login_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LINE login is not configured on the server.",
        )
    try:
        session = line_auth.login_with_line(payload.id_token)
    except line_auth.LineAuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return LineLoginResponse(**session)
