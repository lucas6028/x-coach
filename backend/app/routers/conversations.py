"""Chat-thread persistence endpoints: save and restore a user's grounded conversation for a video.

Both require a valid Supabase JWT (``get_current_user``); ownership is enforced by the store's
user-JWT path plus Postgres RLS, so a request for another user's ``video_id`` simply finds nothing.
A thread is keyed per ``(user, video_id)`` — one grounded conversation per analysed clip — and is
reachable again only via the persisted-analysis history-replay path.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.app.auth import CurrentUser, get_current_user
from backend.app.services import store

router = APIRouter(prefix="/api", tags=["conversations"])


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ConversationBody(BaseModel):
    # The full thread, oldest first. Empty is allowed (a cleared thread); the client normally PUTs
    # after a completed turn, so it is non-empty in practice.
    messages: list[ConversationMessage] = Field(default_factory=list)
    # The latest answer's grounded follow-up chips. Optional — a PUT that omits it (or sends [])
    # clears the stored chips, matching the client's "clear on new send" behaviour.
    followups: list[str] = Field(default_factory=list)


@router.put("/conversations/{video_id}")
def save_conversation(
    video_id: str,
    body: ConversationBody,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Upsert the caller's chat thread for ``video_id`` (idempotent; the whole thread is stored)."""
    messages = [m.model_dump() for m in body.messages]
    store.upsert_conversation(
        token=user.token,
        user_id=user.id,
        video_id=video_id,
        messages=messages,
        followups=body.followups,
    )
    return {"video_id": video_id, "messages": messages, "followups": body.followups}


@router.get("/conversations/{video_id}")
def load_conversation(
    video_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Return the caller's saved thread for ``video_id``; empty lists when none exists."""
    row = store.get_conversation(token=user.token, video_id=video_id) or {}
    return {
        "video_id": video_id,
        "messages": row.get("messages", []),
        "followups": row.get("followups", []),
    }
