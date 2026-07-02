"""Conversational-coaching service: ground an LLM answer in an existing analysis.

The product's credibility is its *groundedness* — so this service is deliberately narrow: it
takes the faults + retrieved knowledge (causes / risks / corrections) the pipeline already
produced and lets an LLM (served via OpenRouter) explain and answer follow-ups **only from those
facts**. The system prompt is built here on the server, never by the client, so the grounding
and honesty constraints can't be tampered with from the browser.

The network call is isolated in ``_chat_completion`` and the ``httpx`` import is deferred into
it — mirroring how ``services/store`` defers ``supabase`` — so the routers import cheaply and the
unit tests patch ``_chat_completion`` without touching the network or needing an API key.
"""

from __future__ import annotations

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


def _chat_completion(messages: list[dict[str, str]]) -> str:
    """Call OpenRouter's OpenAI-compatible chat-completions API and return the reply text.

    Isolated (and network-deferred) so tests patch this seam. Raises ``RuntimeError`` on any
    transport/HTTP failure so the router can map it to a clean 502.
    """
    import httpx  # deferred: only needed on a live request, keeps router import light.

    settings = get_settings()
    try:
        resp = httpx.post(
            f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                # OpenRouter attribution headers (optional but recommended).
                "HTTP-Referer": "https://x-coach.local",
                "X-Title": "x-coach",
            },
            json={"model": settings.openrouter_model, "messages": messages},
            timeout=_REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001 — any failure here is an upstream/transport problem.
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc


def answer(*, messages: list[dict[str, str]], context: dict[str, Any]) -> dict[str, Any]:
    """Produce a grounded coaching reply for ``messages`` given the analysis ``context``.

    ``messages`` is the client-held conversation (roles ``user``/``assistant``), newest last;
    the backend prepends the grounded system prompt. Chat is stateless/ephemeral in v1 — nothing
    is persisted.
    """
    system = _build_system_prompt(context)
    reply = _chat_completion([{"role": "system", "content": system}, *messages])
    return {"reply": reply, "model": get_settings().openrouter_model}
