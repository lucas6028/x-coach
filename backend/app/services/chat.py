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
    kg_seeds_default,
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
    usage-only frame, an empty ``choices``, a non-dict ``choice``, a non-dict ``delta``, or a
    non-dict ``tool_calls`` entry are each skipped without aborting the round.
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
            # ``index`` is provider-supplied, not a value this module controls -- a peer could send
            # it as a numeric JSON string instead of an int. ``acc`` must have homogeneous key types
            # or the final ``sorted(acc)`` raises ``TypeError`` (str vs int), which is not a
            # ``RuntimeError``/``_LLMError`` and would escape this generator mid-stream, *after* the
            # HTTP 200 is already committed to the client. Coerce defensively; an unparseable index
            # folds into slot 0 rather than crashing the round.
            try:
                index = int(frag.get("index", 0))
            except (TypeError, ValueError):
                index = 0
            slot = acc.setdefault(index, {"id": "", "name": "", "arguments": ""})
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


# Tool results are pasted back into the next round's context verbatim. RAG hits carry full passage
# text and a 5-hit result can run to tens of kilobytes, which would dominate (or overflow) the
# window on round 2 — so every result is truncated to this budget before it goes back to the model.
_MAX_TOOL_RESULT_CHARS = 4000

# The tool catalogue, in the OpenAI-compatible ``tools`` schema. The descriptions do real work here:
# they are the only place the model learns that kg_query/rag_search return GENERAL knowledge rather
# than observations about this clip, which is the honesty risk live retrieval introduces (spec v3
# section 4). Keep that framing in any future edit.
_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_analysis",
            "description": (
                "Read the FULL detail of THIS video's analysis — the exact measured values and the "
                "complete retrieved reference text that the summary in your instructions "
                "compressed away. Use it when the user asks for a specific number, frame, or the "
                "full source passage behind a detected fault."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fault_name": {
                        "type": ["string", "null"],
                        "description": (
                            "The detected fault to expand, named exactly as in ANALYSIS FACTS. "
                            "Pass null for clip-level information instead (video metadata, quality, "
                            "camera view, and a one-line summary of every detection)."
                        ),
                    },
                    "include": {
                        "type": "string",
                        "enum": ["evidence", "knowledge", "all"],
                        "description": (
                            "'evidence' = the measured values and the frame/time window; "
                            "'knowledge' = the retrieved subgraph and full reference passages; "
                            "'all' = both."
                        ),
                    },
                },
                "required": ["include"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kg_query",
            "description": (
                "Search the movement knowledge graph for causes, injury risks, and corrective cues "
                "related to a term. Returns GENERAL REFERENCE knowledge about the movement — it "
                "says nothing about what happened in this video, and a fault it mentions was NOT "
                "necessarily committed by this user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A fault, joint, or biomechanical concept, in English.",
                    },
                    "hops": {
                        "type": "integer",
                        "description": "Graph traversal depth, 1 or 2. Defaults to 1.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Search the indexed sports-science literature for passages about a topic. Returns "
                "GENERAL REFERENCE knowledge — it says nothing about what happened in this video, "
                "and a fault it mentions was NOT necessarily committed by this user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic to look up, in English.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many passages to return, 1 to 8. Defaults to 5.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def _clamp_int(value: Any, *, low: int, high: int, default: int) -> int:
    """Coerce a MODEL-SUPPLIED number into ``[low, high]``, falling back to ``default``.

    Tool arguments are untrusted input in exactly the way ``routers/knowledge.py``'s query params
    are: an unbounded ``hops`` drives needlessly expensive traversal and a negative ``top_k`` is an
    invalid slice. Unlike the router this cannot answer 422 — the model is mid-conversation and
    there is no one to return a status to — so a bad value is silently clamped instead. Models emit
    ``"2"`` about as often as ``2``, and occasionally prose, so a non-numeric value falls back to
    the default rather than raising.
    """
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        # ``OverflowError`` is the extra case beyond the brief's own tuple: ``json.loads`` accepts
        # bare ``Infinity``/``-Infinity`` as an extension, so a model argument can arrive as an
        # actual ``float('inf')`` — and ``int(float('inf'))`` raises ``OverflowError``, not
        # ``ValueError``. Caught here rather than relying on ``_dispatch_tool``'s blanket
        # ``except Exception`` so this function's own contract ("never raises, always returns an
        # int in range") holds regardless of what wraps the call.
        return default
    return max(low, min(high, n))


def _parse_tool_args(raw: str) -> dict[str, Any]:
    """Parse a tool call's accumulated ``arguments`` JSON, tolerating nothing and tolerating junk.

    A model can stream truncated or malformed JSON here, and a round is too expensive to throw away
    over it: an empty dict lets each tool fall back to its own defaults and produce *something* the
    model can react to. A non-object payload (a bare array, a string) is treated the same way.
    """
    try:
        data = json.loads(raw or "{}")
    except (json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _tool_query_label(name: str, args: dict[str, Any]) -> str:
    """The human-readable subject of a tool call, for the ``tool`` SSE frame the tray displays."""
    if name == "get_analysis":
        return str(args.get("fault_name") or "")
    return str(args.get("query") or "")


def _tool_get_analysis(detail: dict[str, Any], args: dict[str, Any]) -> Any:
    """The ``get_analysis`` tool: read ``ChatContext.detail`` — the client's own analysis document.

    Reads the request payload, never the database (spec v3 decision 2). The client already holds the
    analysis it is asking about, so shipping it removes the entire auth/ownership surface a DB
    lookup would add — no token threading into this service, no UUID validation against a
    model-fabricated id, no IDOR — and guarantees the tool and the on-screen analysis are literally
    the same document.
    """
    include = args.get("include") or "all"
    name = args.get("fault_name")
    detections = detail.get("detections") or []
    retrievals = detail.get("retrievals") or []

    if not name:
        return {
            "metadata": detail.get("metadata"),
            "quality": detail.get("quality"),
            "view": detail.get("view"),
            "detections": [
                {
                    "fault_name": d.get("fault_name"),
                    "phase": d.get("phase"),
                    "severity": d.get("severity"),
                    "start_time": d.get("start_time"),
                    "end_time": d.get("end_time"),
                }
                for d in detections
            ],
        }

    # Match case-insensitively: the model is quoting a name back out of the prompt and will not
    # always preserve casing. An unmatched name lists what *was* detected rather than returning
    # nothing, so the model can correct itself instead of guessing again.
    matches = [d for d in detections if str(d.get("fault_name", "")).lower() == str(name).lower()]
    if not matches:
        return {
            "error": f"No detected fault named {name!r} in this analysis.",
            "detected_faults": [d.get("fault_name") for d in detections],
        }

    d = matches[0]
    out: dict[str, Any] = {"fault_name": d.get("fault_name")}
    if include in ("evidence", "all"):
        out["evidence"] = d.get("evidence")
        # The rep fields are the whole reason "第 2 rep 膝蓋幾度" is answerable at all: per-rep
        # attribution lives on the detection (`pose_rule_detector.py:105-107`) and survives
        # `asdict` into the payload, but it is absent from the compact blob the prompt is built
        # from. Dropping it here would silently downgrade the feature to whole-clip answers.
        # They sit at their zero defaults on the whole-clip fallback path, which is honest: the
        # model then sees rep_count 0 and knows there was no per-rep segmentation to speak of.
        out["measured"] = {
            k: d.get(k)
            for k in (
                "phase",
                "severity",
                "confidence",
                "observability",
                "start_time",
                "end_time",
                "start_frame",
                "end_frame",
                "peak_frame",
                "rep_index",
                "occurred_reps",
                "rep_count",
            )
        }
    if include in ("knowledge", "all"):
        out["knowledge"] = [
            r.get("context") for r in retrievals if r.get("fault_id") == d.get("fault_id")
        ]
    return out


def _dispatch_tool(name: str, args: dict[str, Any], context: dict[str, Any]) -> str:
    """Run one tool call; return its result as the string content of a ``role:"tool"`` message.

    NOTHING RAISES OUT OF HERE, deliberately. A failing tool (a missing KG graphml, an unbuilt RAG
    vector db) or a hallucinated tool name becomes an ``{"error": ...}`` payload the model can read
    and account for out loud. The alternative — letting it propagate — kills an answer stream that
    was otherwise fine, and the HTTP 200 is already committed by then so the user would just see it
    die. The result is finally truncated to ``_MAX_TOOL_RESULT_CHARS``.
    """
    from backend.app.services import knowledge  # deferred: the KG/RAG import chain is heavy.

    try:
        if name == "get_analysis":
            result: Any = _tool_get_analysis(context.get("detail") or {}, args)
        elif name == "kg_query":
            result = knowledge.graph_context(
                str(args.get("query") or ""),
                hops=_clamp_int(args.get("hops"), low=1, high=2, default=1),
                max_seeds=kg_seeds_default(),
                # Forced to the thread's movement: without it the KG happily returns knowledge for a
                # different exercise, which the coach would then present as relevant to this clip.
                movement=_resolve_movement(context),
            )
        elif name == "rag_search":
            result = knowledge.rag_snippets(
                str(args.get("query") or ""),
                top_k=_clamp_int(args.get("top_k"), low=1, high=8, default=5),
            )
        else:
            result = {"error": f"Unknown tool {name!r}."}
    except Exception as exc:  # noqa: BLE001 — a tool failure must never kill the answer stream.
        result = {"error": f"{name} failed: {exc}"}

    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) > _MAX_TOOL_RESULT_CHARS:
        text = text[:_MAX_TOOL_RESULT_CHARS] + "…[truncated]"
    return text


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
