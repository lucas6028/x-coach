"""Conversational-coaching service: ground an LLM answer in an existing analysis.

The product's credibility is its *groundedness* — so this service is deliberately narrow: it
takes the faults + retrieved knowledge (causes / risks / corrections) the pipeline already
produced and lets an LLM (served via OpenRouter) explain and answer follow-ups **only from those
facts**. The system prompt is built here on the server, never by the client, so the grounding
and honesty constraints can't be tampered with from the browser.

The reply is **streamed** (v2): the network call is isolated in the ``_stream_completion``
generator and the ``httpx`` import is deferred into it — mirroring how ``services/store`` defers
``supabase`` — so the routers import cheaply and the unit tests patch ``_stream_completion``
without touching the network or needing an API key. ``answer_stream`` wraps the token stream in
Server-Sent Events; because the HTTP 200 is committed the moment streaming starts, *every*
OpenRouter failure (connect, mid-stream, or an empty completion) surfaces as an in-band ``error``
event rather than an HTTP status code — the only pre-flight HTTP errors are 503/401/422, raised in
the router before the stream opens.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from backend.app.settings import get_settings

# OpenRouter round-trip budget (seconds). Generous enough for a reasoning model, bounded so a
# hung upstream can't pin a worker thread indefinitely.
_REQUEST_TIMEOUT_S = 60.0

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


def _stream_completion(messages: list[dict[str, str]], model: str) -> Iterator[str]:
    """Stream reply-text chunks from OpenRouter's OpenAI-compatible chat-completions API.

    Isolated (and network-deferred) so tests patch this seam. Parses the OpenAI SSE shape
    (``data: {choices:[{delta:{content}}]}`` lines, ``data: [DONE]`` terminator) and yields each
    non-empty ``delta.content``. ``model`` is the already-resolved (allow-listed) OpenRouter slug.
    Raises ``RuntimeError`` on any transport/HTTP failure so ``answer_stream`` can turn it into an
    in-band ``error`` event.
    """
    import httpx  # deferred: only needed on a live request, keeps router import light.

    settings = get_settings()
    try:
        with httpx.stream(
            "POST",
            f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                # OpenRouter attribution headers (optional but recommended).
                "HTTP-Referer": "https://x-coach.local",
                "X-Title": "x-coach",
            },
            json={"model": model, "messages": messages, "stream": True},
            timeout=_REQUEST_TIMEOUT_S,
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
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc


def answer_stream(
    *, messages: list[dict[str, str]], context: dict[str, Any], model: str
) -> Iterator[str]:
    """Stream a grounded coaching reply for ``messages`` as SSE frames.

    ``messages`` is the client-held conversation (roles ``user``/``assistant``), newest last; the
    backend prepends the grounded system prompt. ``model`` is the already-resolved (allow-listed)
    OpenRouter slug the caller picked. Yields zero or more ``delta`` frames, then exactly one
    terminator: ``done`` (carrying the model actually used) on success, or ``error`` on any
    transport failure or an empty completion. The empty-completion guard preserves the v1 invariant
    — the client must never keep an empty assistant turn, which the next send's
    ``content min_length=1`` would reject.
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
        yield _sse("error", {"detail": "OpenRouter returned an empty message."})
        return

    yield _sse("done", {"model": model})
