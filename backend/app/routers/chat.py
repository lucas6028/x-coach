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
from starlette.concurrency import run_in_threadpool

from backend.app.auth import CurrentUser, get_current_user
from backend.app.services import chat as chat_service
from backend.app.settings import followup_chat_model, get_settings, resolve_chat_model

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
    # Which detector produced this analysis. Optional so a client predating per-movement
    # analysis still validates; _resolve_movement (chat_service) falls back to the pipeline
    # default when absent.
    #
    # UNVALIDATED, BY DESIGN: this string is interpolated verbatim into the LLM system prompt,
    # right beside the GROUNDING RULES block (see chat_service._system_preamble /
    # _resolve_movement). That is acceptable, not an oversight -- every sibling field on this
    # model (fault_name, evidence, causes, risks, corrections, rag_snippet) already interpolates
    # unescaped into the same prompt, and ChatMessage.content (the user's own turns) reaches the
    # same model with only a min_length=1 check, a strictly easier injection channel. Adding
    # validation to this one field would not close that surface -- it would only misrepresent it
    # as closed. The endpoint is auth-gated (get_current_user), so anything injected here is
    # confined to the calling user's own conversation.
    movement: str | None = None
    view_type: str | None = None
    view_confidence: float | None = None
    fault_count: int = 0
    quality: dict[str, Any] = Field(default_factory=dict)
    faults: list[FaultContext] = Field(default_factory=list)
    # The FULL analysis document (detections + retrievals, minus the heavy `pose` block), shipped so
    # the `get_analysis` tool can read the detail `buildChatContext` compresses away — exact measured
    # values and complete reference passages. Optional: absent from a client predating v3, and
    # deliberately omitted by `/api/chat/followups`, which shares this model but can never use it.
    #
    # This is NOT persisted (`upsert_conversation` stores messages + followups only) and never enters
    # the prompt unless a tool returns part of it, so its cost is request body size, not tokens.
    detail: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    # The conversation so far, oldest first; the last entry is the new user turn.
    messages: list[ChatMessage] = Field(..., min_length=1)
    context: ChatContext
    # The user's chosen model (a provider model slug). Validated against the server allowlist; an
    # unknown/absent value falls back to the configured default.
    model: str | None = None


@router.post("/chat")
async def chat(
    body: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
) -> StreamingResponse:
    """Stream a grounded coaching reply as Server-Sent Events (``delta``/``done``/``error``).

    Pre-flight failures return a real HTTP status *before* the stream opens: 401 (no session, via
    the dependency), 503 (LLM unconfigured), 422 (last turn not the user's). Once the 200 stream
    starts, any upstream LLM failure or empty completion is an in-band ``error`` event instead — the
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
    # ``resolve_chat_model`` reads the admin overrides, which can do a synchronous Supabase round-trip
    # on a cold cache — run it in a threadpool so it never blocks the event loop.
    model = await run_in_threadpool(resolve_chat_model, body.model)  # allow-list guard.

    # The sync generator's blocking httpx calls are iterated off the event loop by StreamingResponse
    # (Starlette runs a non-async iterator in a threadpool). Disable proxy/browser buffering so
    # tokens flush as they arrive.
    return StreamingResponse(
        chat_service.answer_stream(messages=messages, context=context, model=model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class FollowupsResponse(BaseModel):
    questions: list[str] = Field(default_factory=list)


@router.post("/chat/followups", response_model=FollowupsResponse)
async def chat_followups(
    body: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
) -> FollowupsResponse:
    """Two grounded next-question suggestions for a completed turn (a separate, best-effort call).

    The client fires this *after* the answer has rendered — fire-and-forget — so it never blocks or
    delays the answer. ``body.messages`` is the conversation ending on the assistant answer (so no
    last-must-be-user check, unlike ``/chat``). Any upstream failure surfaces as an empty list rather
    than an error: a missing chip is not worth failing the request the UI didn't wait on. 503 without
    a key, 401 without a session (via the dependency) still apply.

    The model is the server-pinned ``followup_chat_model`` (a fast one), NOT ``body.model`` — chips
    should stay snappy even when the user picked a slow/reasoning answer model.
    """
    if not get_settings().chat_configured:
        raise HTTPException(
            status_code=503,
            detail="Conversational coaching is not configured on the server.",
        )

    # ``followup_chat_model`` reads the admin overrides (a possible cold-cache Supabase round-trip),
    # so resolve it off the event loop before the best-effort suggestion call.
    model = await run_in_threadpool(followup_chat_model)  # fast, server-pinned; independent of answer
    questions = chat_service.suggest_followups(
        messages=[m.model_dump() for m in body.messages],
        context=body.context.model_dump(),
        model=model,
    )
    return FollowupsResponse(questions=questions)
