"""History endpoints: list and fetch a user's persisted analyses ("我的紀錄").

Both require a valid Supabase JWT (``get_current_user``). Ownership is enforced twice: the
backend only ever queries as the authenticated user, and Postgres RLS scopes rows to
``auth.uid()`` — so a fetch for someone else's id simply returns nothing (404), never their data.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.app.auth import CurrentUser, get_current_user
from backend.app.services import store

router = APIRouter(prefix="/api", tags=["analyses"])


@router.get("/analyses")
def list_my_analyses(
    limit: int = 50,
    offset: int = 0,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """List the caller's analyses (newest first): ``{"total", "items": [summary rows]}``."""
    return store.list_analyses(token=user.token, limit=limit, offset=offset)


@router.delete("/analyses")
def delete_my_analyses(
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Delete all of the caller's saved analyses (and source video rows): ``{"deleted": n}``."""
    deleted = store.delete_all_analyses(token=user.token, user_id=user.id)
    return {"deleted": deleted}


@router.get("/analyses/{analysis_id}")
def get_my_analysis(
    analysis_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Return one of the caller's analyses (full ``result`` JSONB), or 404."""
    row = store.get_analysis(token=user.token, analysis_id=analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No analysis '{analysis_id}'.")
    return row
