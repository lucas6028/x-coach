"""Conversational-coaching endpoint: a grounded LLM follow-up chat over an analysis.

Gated behind ``get_current_user`` (unlike ``/api/analyze``): the LLM call is metered, so a
signed-in session is required to keep an anonymous caller from running up cost. The endpoint is
otherwise stateless — the client sends the conversation so far plus a compact grounding blob
derived from the analysis it already holds, and the service builds the system prompt server-side.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.app.auth import CurrentUser, get_current_user
from backend.app.services import chat as chat_service
from backend.app.settings import get_settings

router = APIRouter(prefix="/api", tags=["chat"])


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class FaultContext(BaseModel):
    """One detected fault plus its retrieved knowledge, as the frontend already derived it."""

    fault_name: str
    phase: str | None = None
    severity: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    evidence: str | None = None
    causes: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    rag_snippet: str | None = None


class ChatContext(BaseModel):
    """Compact grounding blob built by ``buildChatContext(analysis)`` on the client."""

    video_id: str | None = None
    view_type: str | None = None
    view_confidence: float | None = None
    fault_count: int = 0
    quality: dict[str, Any] = Field(default_factory=dict)
    faults: list[FaultContext] = Field(default_factory=list)


class ChatRequest(BaseModel):
    # The conversation so far, oldest first; the last entry is the new user turn.
    messages: list[ChatMessage] = Field(..., min_length=1)
    context: ChatContext


@router.post("/chat")
async def chat(
    body: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
) -> StreamingResponse:
    """Stream a grounded coaching reply as Server-Sent Events (``delta``/``done``/``error``).

    Pre-flight failures return a real HTTP status *before* the stream opens: 401 (no session, via
    the dependency), 503 (LLM unconfigured), 422 (last turn not the user's). Once the 200 stream
    starts, any OpenRouter failure or empty completion is an in-band ``error`` event instead — the
    status is already committed and cannot change.
    """
    if not get_settings().chat_configured:
        raise HTTPException(
            status_code=503,
            detail="Conversational coaching is not configured on the server.",
        )

    if body.messages[-1].role != "user":
        raise HTTPException(status_code=422, detail="The last message must be from the user.")

    messages = [m.model_dump() for m in body.messages]
    context = body.context.model_dump()

    # The sync generator's blocking httpx calls are iterated off the event loop by StreamingResponse
    # (Starlette runs a non-async iterator in a threadpool). Disable proxy/browser buffering so
    # tokens flush as they arrive.
    return StreamingResponse(
        chat_service.answer_stream(messages=messages, context=context),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
