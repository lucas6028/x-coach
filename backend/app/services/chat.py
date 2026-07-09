"""Conversational-coaching service: ground an LLM answer in an existing analysis.

The product's credibility is its *groundedness* — so this service is deliberately narrow: it
takes the faults + retrieved knowledge (causes / risks / corrections) the pipeline already
produced and lets an LLM (served via a configurable OpenAI-compatible provider — OpenRouter by
default, or any peer such as NVIDIA NIM) explain and answer follow-ups **only from those facts**.
The system prompt is built here on the server, never by the client, so the grounding and honesty
constraints can't be tampered with from the browser.

The reply is **streamed** (v2): the network call is isolated in the ``_stream_completion``
generator and the ``httpx`` import is deferred into it — mirroring how ``services/store`` defers
``supabase`` — so the routers import cheaply and the unit tests patch ``_stream_completion``
without touching the network or needing an API key. ``answer_stream`` wraps the token stream in
Server-Sent Events; because the HTTP 200 is committed the moment streaming starts, *every*
upstream failure (connect, mid-stream, or an empty completion) surfaces as an in-band ``error``
event rather than an HTTP status code — the only pre-flight HTTP errors are 503/401/422, raised in
the router before the stream opens.

Follow-up suggestions (v2.1) are a **separate, fire-and-forget** step (``suggest_followups`` +
``POST /api/chat/followups``), not part of the answer stream: the client renders the answer the moment
it completes, then asks for two grounded next-question chips in the background and drops them in when
they arrive. Keeping them off the answer path means a slow or failed suggestion can never delay or
corrupt the answer — and the answer stream stays the clean ``delta``/``done``/``error`` contract.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from backend.app.settings import get_settings

# LLM round-trip budget (seconds). Generous enough for a reasoning model, bounded so a
# hung upstream can't pin a worker thread indefinitely.
_REQUEST_TIMEOUT_S = 60.0

# The follow-up call is a separate, best-effort background request; bound it tightly so a slow/hung
# suggestion never leaves the chips "loading" — a miss just means no chips, and the answer is already
# on screen regardless.
_FOLLOWUP_TIMEOUT_S = 15.0

_SYSTEM_PREAMBLE = (
    "You are the x-coach squat coach. You explain an ALREADY-COMPUTED analysis of one squat "
    "repetition and answer the user's follow-up questions about it.\n\n"
    "GROUNDING RULES — these are absolute:\n"
    "- Speak ONLY from the analysis facts given below. They are the single source of truth.\n"
    "- Do NOT invent faults, causes, injury risks, corrective cues, measurements, or camera "
    "views that are not listed. If the user asks about something not in the analysis, say it "
    "was not detected or not measured in this rep — never fabricate it.\n"
    "- Base any corrective advice on the retrieved corrections/cues for the detected faults.\n"
    "- Be concise, specific, and encouraging. Reference the timecodes and phases when useful.\n"
    "- Reply in the same language the user writes in.\n"
    "- You may use light Markdown for readability — bold for key cues, short bulleted lists, and "
    "inline code for measurements/timecodes. Formatting never loosens the grounding rules above.\n"
)

# Appended to the grounded system prompt for the follow-up call. The full analysis grounding precedes
# it, so a suggested question can never reference a fault, cue, or measurement outside the analysis —
# the same honesty bar the answer holds.
_FOLLOWUP_INSTRUCTION = (
    "FOLLOW-UP TASK: The user has just read your answer. Propose EXACTLY TWO short follow-up "
    "questions the user might naturally ask you next about THIS squat. Each is from the user's point "
    "of view (addressed to you, the coach), grounded ONLY in the analysis facts above (never a "
    "fault/cue/measurement not listed), at most ~12 words, in the user's language. Output ONLY a "
    'compact JSON array of exactly two strings and nothing else — e.g. ["...", "..."].'
)

# A short trailing *user* turn for the follow-up call. Without it the request array would end on the
# assistant answer, which several OpenRouter-routed models continue (more prose) rather than treat as
# a cue to run the task — so the ask is restated as the final user turn (the conventional "do X over
# this conversation" shape); the grounding + detailed instruction still live in the system prompt.
_FOLLOWUP_NUDGE = "Now output ONLY the JSON array of exactly two follow-up questions."

# Extra OpenRouter body for the follow-up call: route to the lowest-latency provider for the pinned
# model. Measured to be the real fix for the 3–10s chip-latency variance — without it, OpenRouter can
# route the same model to a cold/slow provider (2s one call, 9s the next); with it, ~1.5s consistently.
# ``provider`` is an OpenRouter-only body field; it is sent only when the base URL is OpenRouter's (see
# ``_is_openrouter``) so an OpenAI-compatible peer like NVIDIA NIM isn't handed a field it may 400 on.
_FOLLOWUP_ROUTING = {"provider": {"sort": "latency"}}


def _is_openrouter(base_url: str) -> bool:
    """True when the configured LLM base URL is OpenRouter's.

    The transport speaks the plain OpenAI-compatible chat-completions dialect, so any peer that also
    speaks it (NVIDIA NIM at ``integrate.api.nvidia.com``, a self-hosted vLLM, …) works by only
    swapping ``LLM_BASE_URL`` + key + model ids. A couple of extras are OpenRouter-specific,
    though — the attribution headers and the ``provider`` routing body — and a stricter peer can reject
    an unknown body field. This gate keeps those extras on the OpenRouter path only.
    """
    return "openrouter.ai" in base_url


def _fmt_list(items: Any) -> str:
    """Join a list of short strings for prompt embedding, or return ``"—"`` when empty."""
    if not items:
        return "—"
    return ", ".join(str(x) for x in items if x)


def _build_system_prompt(context: dict[str, Any]) -> str:
    """Render the analysis ``context`` into the grounded system prompt.

    Handles the clean-rep case (no ``faults``) distinctly so the model reinforces good form
    instead of being handed an empty fault list it might feel obliged to fill.
    """
    lines: list[str] = [_SYSTEM_PREAMBLE, "ANALYSIS FACTS:"]

    view = context.get("view_type") or "unknown"
    conf = context.get("view_confidence")
    conf_txt = f" (confidence {conf:.2f})" if isinstance(conf, (int, float)) else ""
    lines.append(f"- Camera view: {view}{conf_txt}")

    quality = context.get("quality") or {}
    vis = quality.get("lower_body_visibility_mean")
    if isinstance(vis, (int, float)):
        lines.append(f"- Lower-body visibility: {vis:.2f}")

    faults = context.get("faults") or []
    fault_count = context.get("fault_count", len(faults))
    lines.append(f"- Faults detected: {fault_count}")

    if not faults:
        lines.append(
            "- This is a CLEAN REP: no faults were detected. Congratulate the user and "
            "reinforce what good form looks like; do not manufacture problems."
        )
    else:
        lines.append("")
        lines.append("DETECTED FAULTS (each with its retrieved knowledge):")
        for i, f in enumerate(faults, start=1):
            name = f.get("fault_name", "unknown")
            phase = f.get("phase", "—")
            sev = f.get("severity")
            sev_txt = f"{sev:.2f}" if isinstance(sev, (int, float)) else "—"
            start = f.get("start_time")
            end = f.get("end_time")
            when = (
                f"{start:.2f}s–{end:.2f}s"
                if isinstance(start, (int, float)) and isinstance(end, (int, float))
                else "—"
            )
            lines.append(
                f"  {i}. {name} — phase: {phase}, severity: {sev_txt}, window: {when}"
            )
            if f.get("evidence"):
                lines.append(f"     evidence: {f['evidence']}")
            lines.append(f"     likely causes: {_fmt_list(f.get('causes'))}")
            lines.append(f"     injury risks: {_fmt_list(f.get('risks'))}")
            lines.append(f"     corrective cues: {_fmt_list(f.get('corrections'))}")
            snippet = f.get("rag_snippet")
            if snippet:
                lines.append(f"     reference: {snippet}")

    return "\n".join(lines)


def _sse(event: str, data: dict[str, Any]) -> str:
    """Render one Server-Sent Event frame (``event:`` + JSON ``data:``, terminated by a blank line)."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_completion(
    messages: list[dict[str, str]],
    model: str,
    timeout: float = _REQUEST_TIMEOUT_S,
    extra_body: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Stream reply-text chunks from the configured provider's OpenAI-compatible chat-completions API.

    Isolated (and network-deferred) so tests patch this seam. Parses the OpenAI SSE shape
    (``data: {choices:[{delta:{content}}]}`` lines, ``data: [DONE]`` terminator) and yields each
    non-empty ``delta.content``. ``model`` is the already-resolved provider slug; ``timeout`` is the
    per-request budget (the follow-up call passes a tighter one); ``extra_body`` merges extra request
    fields (the follow-up call passes provider-routing preferences). Raises ``RuntimeError`` on any
    transport/HTTP failure so the caller can surface it (an in-band ``error`` for the answer stream, or
    an empty suggestion list for the best-effort follow-up call).
    """
    import httpx  # deferred: only needed on a live request, keeps router import light.

    settings = get_settings()
    body: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
    if extra_body:
        body.update(extra_body)
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    if _is_openrouter(settings.llm_base_url):
        # OpenRouter attribution headers (optional but recommended); other OpenAI-compatible peers
        # (e.g. NVIDIA NIM) don't use them, so keep them off those requests.
        headers["HTTP-Referer"] = "https://x-coach.local"
        headers["X-Title"] = "x-coach"
    try:
        with httpx.stream(
            "POST",
            f"{settings.llm_base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=body,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue  # blank keep-alives and ``:`` comment lines carry no payload.
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue  # a partial/keep-alive/unexpected frame — skip, don't abort.
                if delta:
                    yield delta
    except Exception as exc:  # noqa: BLE001 — any failure here is an upstream/transport problem.
        raise RuntimeError(f"LLM request failed: {exc}") from exc


def _parse_followups(text: str) -> list[str]:
    """Extract up to two follow-up questions from the model's JSON-array reply.

    Tolerant of a model that wraps the array in code fences or stray prose: the outer ``[...]`` is
    sliced out before parsing. Any non-list / unparseable payload (or fewer than the questions asked
    for) yields whatever *did* parse — ``[]`` in the worst case — so a malformed suggestion never
    breaks a good answer; the caller simply omits the followups frame.
    """
    text = text.strip()
    if "[" in text and "]" in text:  # slice to the outer array — tolerates ```json fences / prose.
        text = text[text.index("[") : text.rindex("]") + 1]
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out = [str(q).strip() for q in data if str(q).strip()]
    return out[:2]


def suggest_followups(
    *, messages: list[dict[str, str]], context: dict[str, Any], model: str
) -> list[str]:
    """Two grounded next-question suggestions for a completed turn — best-effort, never raises.

    Called by ``POST /api/chat/followups`` *after* the answer has already rendered, so it is fully
    off the answer's critical path: any transport failure, timeout, or unparseable reply just returns
    ``[]`` (no chips). ``messages`` is the conversation ending on the assistant answer the questions
    follow from; the grounded system prompt (same honesty rules as the answer) precedes the follow-up
    instruction, and a trailing user nudge keeps the request from ending on the assistant turn.
    """
    system = _build_system_prompt(context) + "\n\n" + _FOLLOWUP_INSTRUCTION
    convo = [
        {"role": "system", "content": system},
        *messages,  # the conversation, ending on the assistant answer these questions follow from
        {"role": "user", "content": _FOLLOWUP_NUDGE},  # restate the ask so the array isn't left open
    ]
    # The latency routing is OpenRouter-only; on any other OpenAI-compatible peer send no extra body.
    routing = _FOLLOWUP_ROUTING if _is_openrouter(get_settings().llm_base_url) else None
    parts: list[str] = []
    try:
        for chunk in _stream_completion(
            convo, model, timeout=_FOLLOWUP_TIMEOUT_S, extra_body=routing
        ):
            parts.append(chunk)
    except RuntimeError:
        return []
    return _parse_followups("".join(parts))


# LINE text messages are capped at 5000 characters by the Messaging API; stay comfortably under it
# so a reply is never rejected. Coaching answers are short, so this only guards a pathological case.
_LINE_MAX_CHARS = 4900


def answer_once(
    *, messages: list[dict[str, str]], context: dict[str, Any], model: str, max_chars: int = _LINE_MAX_CHARS
) -> str:
    """Return one grounded coaching reply as a plain string (no SSE) — the non-streaming sibling of
    ``answer_stream``, used by the LINE webhook where a reply is a single pushed message.

    Reuses the same grounded system prompt and the tested ``_stream_completion`` transport, simply
    joining the token chunks. Raises ``RuntimeError`` on any transport failure (propagated from
    ``_stream_completion``) or an empty completion, so the caller can reply with a graceful fallback
    instead of pushing a blank message. The result is truncated to ``max_chars`` for LINE's limit.
    """
    system = _build_system_prompt(context)
    text = "".join(_stream_completion([{"role": "system", "content": system}, *messages], model)).strip()
    if not text:
        raise RuntimeError("The LLM returned an empty message.")
    return text[:max_chars]


def answer_stream(
    *, messages: list[dict[str, str]], context: dict[str, Any], model: str
) -> Iterator[str]:
    """Stream a grounded coaching reply for ``messages`` as SSE frames.

    ``messages`` is the client-held conversation (roles ``user``/``assistant``), newest last; the
    backend prepends the grounded system prompt. ``model`` is the already-resolved (allow-listed)
    provider slug. Yields zero or more ``delta`` frames, then exactly one terminator: ``done``
    (carrying the model actually used) on success, or ``error`` on any transport failure or an empty
    completion. The empty-completion guard preserves the v1 invariant — the client must never keep an
    empty assistant turn, which the next send's ``content min_length=1`` would reject. Follow-up
    suggestions are intentionally NOT part of this stream (see ``suggest_followups``): the answer path
    stays clean and closes the moment the answer is done, so nothing delays it.
    """
    system = _build_system_prompt(context)
    parts: list[str] = []
    try:
        for chunk in _stream_completion([{"role": "system", "content": system}, *messages], model):
            parts.append(chunk)
            yield _sse("delta", {"text": chunk})
    except RuntimeError as exc:
        yield _sse("error", {"detail": str(exc)})
        return

    if not "".join(parts).strip():
        yield _sse("error", {"detail": "The LLM returned an empty message."})
        return

    yield _sse("done", {"model": model})
