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
from dataclasses import dataclass
from typing import Any

from backend.app import config
from backend.app.settings import (
    chat_base_url,
    chat_temperature,
    chat_timeout,
    followup_timeout,
    get_settings,
)

# Fallback LLM round-trip budgets (seconds). The effective values now come from the override-aware
# getters ``chat_timeout()`` / ``followup_timeout()`` (which default to these), so an admin can retune
# them at runtime. Kept here as the documented defaults and for backwards-compatible test references.
_REQUEST_TIMEOUT_S = 60.0

# The follow-up call is a separate, best-effort background request; bound it tightly so a slow/hung
# suggestion never leaves the chips "loading" — a miss just means no chips, and the answer is already
# on screen regardless.
_FOLLOWUP_TIMEOUT_S = 15.0

def _system_preamble(movement: str) -> str:
    """The grounded preamble, scoped to the movement whose rules actually ran.

    The movement is named rather than assumed because it is now USER-ASSERTED input: the studio
    lets the user pick, so a clip can be measured by rules that do not describe it (spec section
    9). Naming it makes every claim true relative to the assertion the user made, and puts that
    assertion in front of the model instead of leaving it implicit.
    """
    return (
        f"You are the x-coach {movement} coach. You explain an ALREADY-COMPUTED analysis of one "
        f"{movement} repetition and answer the user's follow-up questions about it.\n\n"
        "GROUNDING RULES — these are absolute:\n"
        "- Speak ONLY from the analysis facts given below. They are the single source of truth.\n"
        "- Do NOT invent faults, causes, injury risks, corrective cues, measurements, or camera "
        "views that are not listed. If the user asks about something not in the analysis, say it "
        "was not detected or not measured in this rep — never fabricate it.\n"
        "- Base any corrective advice on the retrieved corrections/cues for the detected faults.\n"
        "- Be concise, specific, and encouraging. Reference the timecodes and phases when useful.\n"
        "- Reply in the same language the user writes in.\n"
        "- You may use light Markdown for readability — bold for key cues, short bulleted lists, "
        "and inline code for measurements/timecodes. Formatting never loosens the grounding rules "
        "above.\n"
    )


def _followup_instruction(movement: str) -> str:
    """Appended to the grounded system prompt for the follow-up call. The full analysis grounding
    precedes it, so a suggested question can never reference a fault, cue, or measurement outside
    the analysis — the same honesty bar the answer holds."""
    return (
        "FOLLOW-UP TASK: The user has just read your answer. Propose EXACTLY TWO short follow-up "
        f"questions the user might naturally ask you next about THIS {movement}. Each is from the "
        "user's point of view (addressed to you, the coach), grounded ONLY in the analysis facts "
        "above (never a fault/cue/measurement not listed), at most ~12 words, in the user's "
        'language. Output ONLY a compact JSON array of exactly two strings and nothing else — '
        'e.g. ["...", "..."].'
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


def _resolve_movement(context: dict[str, Any]) -> str:
    """The movement name to interpolate into the system prompt, straight from client input.

    Falls back to the pipeline default so a client predating ``ChatContext.movement`` still gets
    a coherent prompt rather than an empty movement name.

    THIS VALUE IS UNVALIDATED AND GOES STRAIGHT INTO THE SYSTEM PROMPT, twice, as the opening two
    sentences immediately ahead of the ``GROUNDING RULES`` block (see ``_system_preamble``). That
    is deliberate, not an oversight: every other field on ``ChatContext`` (``fault_name``,
    ``evidence``, ``causes``, ``risks``, ``corrections``, ``rag_snippet``) already interpolates
    unescaped into the same system prompt, and ``ChatMessage.content`` -- the user's own chat
    turns -- reaches the same model with only a ``min_length=1`` check, a strictly easier
    injection channel than this field. Validating ``movement`` alone would not close that
    surface, only misrepresent it as closed. The endpoint is auth-gated (``get_current_user``),
    so the blast radius of anything injected here is the calling user's own conversation, not
    other users' data or other tenants.
    """
    return str(context.get("movement") or config.DEFAULT_ANALYSIS_MOVEMENT)


def _build_system_prompt(context: dict[str, Any]) -> str:
    """Render the analysis ``context`` into the grounded system prompt.

    Handles the clean-rep case (no ``faults``) distinctly so the model reinforces good form
    instead of being handed an empty fault list it might feel obliged to fill.

    THE EMPTY FAULT LIST IS AMBIGUOUS, AND THE PROMPT MUST NOT RESOLVE IT OPTIMISTICALLY.
    ``run_detector`` returns an empty ``detections`` list identically for "no faults found" and
    "no frame was ever measurable" -- a clip cropped above the ankles, say, fails every rule's
    validity gate and is byte-for-byte indistinguishable here from a flawless rep. Emitting the
    CLEAN REP instruction in that case tells the model to congratulate the user on form nothing
    measured, which is a fabricated claim of exactly the kind this prompt's honesty constraint
    forbids everywhere else.

    ``context["quality"]`` already carries the distinction -- ``buildChatContext`` ships the whole
    quality dict (frontend/src/lib/grounding.ts) and ``ChatContext.quality`` accepts it -- it was
    simply never surfaced, so the model could not recover it even in principle. Both
    ``valid_frame_ratio`` and the CLEAN REP / NOT MEASURED branch now come from it.

    THE CRITERION IS CATEGORICAL (exactly zero measurable frames), matching ``wasMeasured`` in
    frontend/src/lib/quality.ts verbatim so the tray banner, the metrics HUD and the coach cannot
    contradict each other. It is deliberately NOT a low-but-nonzero band: no "enough frames to
    trust a verdict" threshold has been measured in this repo, and inventing one for a user-facing
    verdict is not acceptable here. A MISSING ``valid_frame_ratio`` counts as UNMEASURED for the
    same reason -- the analyze pipeline always emits it, so its absence means the payload did not
    say, and "we cannot tell" must not resolve to "everything is fine".
    """
    movement = _resolve_movement(context)
    lines: list[str] = [_system_preamble(movement), "ANALYSIS FACTS:"]

    view = context.get("view_type") or "unknown"
    conf = context.get("view_confidence")
    conf_txt = f" (confidence {conf:.2f})" if isinstance(conf, (int, float)) else ""
    lines.append(f"- Camera view: {view}{conf_txt}")

    quality = context.get("quality") or {}
    vis = quality.get("lower_body_visibility_mean")
    if isinstance(vis, (int, float)):
        lines.append(f"- Lower-body visibility: {vis:.2f}")

    valid_ratio = quality.get("valid_frame_ratio")
    measured = isinstance(valid_ratio, (int, float)) and valid_ratio > 0
    if isinstance(valid_ratio, (int, float)):
        lines.append(f"- Measurable frames: {valid_ratio:.0%} of the clip")

    faults = context.get("faults") or []
    fault_count = context.get("fault_count", len(faults))
    lines.append(f"- Faults detected: {fault_count}")

    if not faults and not measured:
        lines.append(
            "- NOT MEASURED: no frame in this clip could be analysed, so the empty fault list "
            "means the analysis never ran on any frame -- it does NOT mean the form was good. "
            "Do NOT congratulate the user or comment on their form. Say plainly that the clip "
            "could not be measured and suggest re-recording with the whole body in frame."
        )
    elif not faults:
        lines.append(
            f"- This is a CLEAN {movement} REP: no {movement} faults were detected. "
            "Congratulate the user and reinforce what good form looks like; do not manufacture "
            "problems."
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


class _LLMError(RuntimeError):
    """An upstream LLM failure, carrying the HTTP status when the request got that far.

    ``status`` is the upstream response code for a non-2xx reply, or ``None`` for a transport-level
    failure (connect / read / timeout). The tool loop needs that distinction: a 4xx means *this
    model rejects the ``tools`` field* and the round can be retried plainly, while a dead provider
    cannot be retried into working. Subclasses ``RuntimeError`` so every pre-existing
    ``except RuntimeError`` in this module and its tests keeps catching it unchanged.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _stream_raw_chunks(
    messages: list[dict[str, Any]],
    model: str,
    timeout: float | None = None,
    extra_body: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream *parsed chunk dicts* from the provider's OpenAI-compatible chat-completions API.

    The transport half of what ``_stream_completion`` used to do alone: HTTP, SSE line framing, and
    JSON parsing — and deliberately nothing else. It does not reach into ``choices``/``delta``,
    because the answer path now needs ``tool_calls`` and ``finish_reason`` as much as ``content``,
    and the shape of a chunk is the caller's business.

    TOLERANCE OWNERSHIP (spec v3 section 7). This layer skips only what it cannot *parse* — a
    truncated or garbage ``data:`` payload. It never drops a well-formed chunk for lacking a field:
    a chunk with no ``content`` may still carry a ``tool_calls`` fragment, and swallowing it here
    would corrupt the caller's reassembly with no error raised anywhere. Shape tolerance is the
    caller's job.
    """
    import httpx  # deferred: only needed on a live request, keeps router import light.

    # The API key is a secret and stays pure-env; the base URL / temperature / timeout are the
    # admin-tunable knobs, read through the override-aware getters. A ``None`` timeout means "use the
    # configured answer budget" (the follow-up call passes its own tighter value explicitly).
    settings = get_settings()
    base_url = chat_base_url()
    if timeout is None:
        timeout = chat_timeout()
    body: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
    temperature = chat_temperature()
    if temperature is not None:  # omit entirely by default, preserving today's behaviour.
        body["temperature"] = temperature
    if extra_body:
        body.update(extra_body)
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    if _is_openrouter(base_url):
        # OpenRouter attribution headers (optional but recommended); other OpenAI-compatible peers
        # (e.g. NVIDIA NIM) don't use them, so keep them off those requests.
        headers["HTTP-Referer"] = "https://x-coach.local"
        headers["X-Title"] = "x-coach"
    try:
        with httpx.stream(
            "POST",
            f"{base_url.rstrip('/')}/chat/completions",
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
                    chunk = json.loads(payload)
                except (json.JSONDecodeError, ValueError):
                    continue  # a partial/garbage frame — skip, don't abort the stream.
                if isinstance(chunk, dict):
                    yield chunk
    except httpx.HTTPStatusError as exc:
        raise _LLMError(
            f"LLM request failed: {exc}", status=exc.response.status_code
        ) from exc
    except Exception as exc:  # noqa: BLE001 — anything else here is a transport problem.
        raise _LLMError(f"LLM request failed: {exc}") from exc


def _stream_completion(
    messages: list[dict[str, str]],
    model: str,
    timeout: float | None = None,
    extra_body: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Stream reply-*text* chunks — the v1/v2 seam, unchanged in signature and behaviour.

    Now a thin shell over ``_stream_raw_chunks``, which owns the transport. This layer keeps the
    text-only contract that ``suggest_followups`` (and its measured ~1.5s chip latency) depends on,
    so the follow-up path is byte-for-byte what it was. Raises ``RuntimeError`` — specifically
    ``_LLMError`` — on any transport/HTTP failure.
    """
    for chunk in _stream_raw_chunks(messages, model, timeout=timeout, extra_body=extra_body):
        try:
            delta = chunk["choices"][0]["delta"].get("content")
        except (KeyError, IndexError, TypeError, AttributeError):
            # a keep-alive/unexpected shape — skip, don't abort. AttributeError covers a
            # ``"delta": null`` chunk: ``None.get`` is an attribute miss, not a TypeError.
            continue
        if delta:
            yield delta


@dataclass
class _Turn:
    """Everything one model round produced: its text, its tool calls, and why it stopped.

    ``tool_calls`` entries are the *reassembled* form — ``{"id", "name", "arguments"}`` with
    ``arguments`` as the concatenated JSON string, not yet parsed (parsing is the dispatcher's job,
    and a model can emit malformed JSON there).
    """

    text: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str | None


def _stream_turn(
    messages: list[dict[str, Any]],
    model: str,
    *,
    timeout: float | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> Iterator[str | _Turn]:
    """Run ONE model round: yield each text delta as it arrives, then exactly one ``_Turn`` last.

    The odd ``str | _Turn`` shape exists because the caller is itself a generator streaming SSE to a
    live client: it has to forward text the instant it arrives *and* receive the round's summary at
    the end, and a plain return value cannot do both. The ``_Turn`` is always the final item.

    STREAMED TOOL CALLS ARRIVE FRAGMENTED, and this is the sharp edge of the whole feature.
    ``delta.tool_calls[i]`` carries ``id`` and ``function.name`` typically only on the *first* chunk
    that mentions index ``i``, then ``function.arguments`` as successive string fragments. Two calls
    can interleave freely. Everything is therefore accumulated in a dict keyed by the fragment's own
    ``index`` field — never by arrival order, never by position in the incoming list.

    Shape tolerance lives here (``_stream_raw_chunks`` owns only JSON parsing, spec v3 section 7): a
    usage-only frame, an empty ``choices``, a non-dict ``delta``, or a non-dict ``tool_calls`` entry
    are each skipped without aborting the round.
    """
    parts: list[str] = []
    acc: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    # ``tools``/``tool_choice`` are standard OpenAI-compatible fields, so unlike the ``provider``
    # routing body they are NOT gated on ``_is_openrouter`` — any peer speaking the dialect takes them.
    extra_body = {"tools": tools, "tool_choice": "auto"} if tools else None

    for chunk in _stream_raw_chunks(messages, model, timeout=timeout, extra_body=extra_body):
        try:
            choice = chunk["choices"][0]
        except (KeyError, IndexError, TypeError):
            continue  # usage-only frame or an unexpected envelope — nothing to fold in.
        if not isinstance(choice, dict):
            continue
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        text = delta.get("content")
        if text:
            parts.append(text)
            yield text
        for frag in delta.get("tool_calls") or []:
            if not isinstance(frag, dict):
                continue
            slot = acc.setdefault(frag.get("index", 0), {"id": "", "name": "", "arguments": ""})
            if frag.get("id"):
                slot["id"] = frag["id"]
            fn = frag.get("function") or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]

    # A slot with no name never became a real call (a stray fragment) — drop it rather than dispatch
    # an unnamed tool. Ordering is by ``index`` so the tool messages match the assistant turn's list.
    yield _Turn(
        text="".join(parts),
        tool_calls=[acc[i] for i in sorted(acc) if acc[i]["name"]],
        finish_reason=finish_reason,
    )


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
    movement = _resolve_movement(context)
    system = _build_system_prompt(context) + "\n\n" + _followup_instruction(movement)
    convo = [
        {"role": "system", "content": system},
        *messages,  # the conversation, ending on the assistant answer these questions follow from
        {"role": "user", "content": _FOLLOWUP_NUDGE},  # restate the ask so the array isn't left open
    ]
    # The latency routing is OpenRouter-only; on any other OpenAI-compatible peer send no extra body.
    routing = _FOLLOWUP_ROUTING if _is_openrouter(chat_base_url()) else None
    parts: list[str] = []
    try:
        for chunk in _stream_completion(
            convo, model, timeout=followup_timeout(), extra_body=routing
        ):
            parts.append(chunk)
    except RuntimeError:
        return []
    return _parse_followups("".join(parts))


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
