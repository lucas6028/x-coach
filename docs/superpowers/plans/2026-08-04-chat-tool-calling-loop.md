# Chat Tool-Calling Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the grounded coach chat into an agent that can call three server-side tools — `get_analysis`, `kg_query`, `rag_search` — inside a single `/api/chat` request.

**Architecture:** The existing `_stream_completion` seam is split *underneath* rather than changed: a new `_stream_raw_chunks` owns HTTP + SSE framing + JSON parsing, `_stream_completion` becomes a thin text-only shell over it (signature and every existing test unchanged, so the follow-up chip path is untouched), and a new `_stream_turn` runs one model round while reassembling fragmented streamed tool calls. `answer_stream` loops over `_stream_turn` up to 3 tool rounds inside one shared wall-clock budget, dispatching tools inline. Tool round-trips never leave the request: the thread contract (`ChatMessage.role`) and the `conversations` table are untouched.

**Tech Stack:** FastAPI + Starlette `StreamingResponse` (SSE), `httpx` (deferred import), Pydantic v2, `unittest.TestCase` under `tests/`; React 18 + TypeScript + vitest under `frontend/src/test/`.

**Spec:** `specs/llm-chat-spec.md`, section `# v3: Tool-calling loop`. Read it before Task 1.

## Global Constraints

- **Python interpreter is always `.venv\Scripts\python.exe`** from the repo root. There is no `python` on PATH on this machine. Never `source .venv/bin/activate`.
- **Backend tests are always scoped to `tests/`**: `.venv\Scripts\python.exe -m pytest tests/`. Never bare `pytest`.
- **Backend coverage gate is 95%**, enforced by CI: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`.
- **All frontend commands run with cwd = `frontend/`.** The Bash and PowerShell tools share one working directory; a stray `cd` to the repo root mass-fails vitest.
- Frontend commands: `yarn test` (vitest run), `yarn test:coverage`, `yarn build`.
- Backend tests are `unittest.TestCase` classes under `tests/`; frontend tests are vitest files under `frontend/src/test/`.
- ML/backend modules favour a local-first, dependency-light style (stdlib + numpy/networkx). **Do not add any new dependency in this plan** — everything needed is already imported somewhere in the repo.
- `_stream_completion`'s existing tests in `tests/test_chat_endpoint.py` must pass **unmodified** throughout. If a step requires editing one of them, the refactor is wrong — stop and reconsider.
- The follow-up chip path (`suggest_followups`, `POST /api/chat/followups`) must never send `tools` and never send `detail`. Its ~1.5s latency is a defended, measured property.
- Comment density in `backend/app/services/chat.py` is high and explanatory — match it. Every non-obvious constant and guard in that file carries a *why*.

---

### Task 1: Split the transport layer (no behaviour change)

**Files:**
- Modify: `backend/app/services/chat.py:239-301` (`_stream_completion`)
- Test: `tests/test_chat_endpoint.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `_LLMError(RuntimeError)` with attribute `status: int | None`
  - `_stream_raw_chunks(messages: list[dict[str, Any]], model: str, timeout: float | None = None, extra_body: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]`
  - `_stream_completion(...) -> Iterator[str]` — **signature unchanged**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chat_endpoint.py`, inside the same class that holds the existing `_stream_completion` transport tests (the one defining `_settings`, around line 380):

```python
    def test_raw_chunks_yield_parsed_dicts_and_skip_unparseable(self) -> None:
        # The raw layer owns JSON-level tolerance ONLY: a truncated/garbage payload is skipped, but a
        # well-formed chunk is yielded whole even when it carries no ``content`` -- it may hold a
        # ``tool_calls`` fragment, and silently dropping it would corrupt the caller's reassembly.
        fake_resp = mock.Mock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.iter_lines.return_value = iter(
            [
                ": keep-alive ping",  # SSE comment -> skip
                'data: {"choices":[{"delta":{"content":"Hi"}}]}',
                "data: {broken",  # truncated JSON -> skip
                'data: {"choices":[{"delta":{}}]}',  # no content, but still a real chunk -> KEPT
                "data: [DONE]",
                'data: {"choices":[{"delta":{"content":"unreached"}}]}',
            ]
        )
        cm = mock.MagicMock()
        cm.__enter__.return_value = fake_resp
        cm.__exit__.return_value = False
        with mock.patch.object(
            chat_service, "get_settings", return_value=self._settings()
        ), mock.patch("httpx.stream", return_value=cm):
            chunks = list(
                chat_service._stream_raw_chunks([{"role": "user", "content": "hi"}], "m")
            )
        self.assertEqual(
            chunks,
            [
                {"choices": [{"delta": {"content": "Hi"}}]},
                {"choices": [{"delta": {}}]},
            ],
        )

    def test_raw_chunks_http_status_error_carries_the_status(self) -> None:
        # A 4xx must be distinguishable from a transport failure: the tool loop retries without
        # ``tools`` on a 4xx (the model rejects the field) but not on a dead provider.
        import httpx

        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        response = httpx.Response(400, request=request)
        fake_resp = mock.Mock()
        fake_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400", request=request, response=response
        )
        cm = mock.MagicMock()
        cm.__enter__.return_value = fake_resp
        cm.__exit__.return_value = False
        with mock.patch.object(
            chat_service, "get_settings", return_value=self._settings()
        ), mock.patch("httpx.stream", return_value=cm):
            with self.assertRaises(chat_service._LLMError) as ctx:
                list(chat_service._stream_raw_chunks([{"role": "user", "content": "hi"}], "m"))
        self.assertEqual(ctx.exception.status, 400)
        self.assertIsInstance(ctx.exception, RuntimeError)  # existing callers still catch it

    def test_completion_shell_skips_chunks_with_no_usable_choice(self) -> None:
        # After the split the shell's own ``except (KeyError, IndexError, TypeError)`` has nothing
        # driving it from the SSE-level tests: ``data: not-json`` is now swallowed by the raw layer,
        # and a well-formed empty delta returns None from ``.get("content")`` without ever raising.
        # Drive it directly, or the branch goes uncovered under the 95% gate.
        with mock.patch.object(
            chat_service,
            "_stream_raw_chunks",
            return_value=iter(
                [
                    {"choices": []},  # IndexError
                    {"usage": {"total_tokens": 3}},  # KeyError
                    {"choices": [{"delta": None}]},  # TypeError (None has no .get)
                    {"choices": [{"delta": {"content": "ok"}}]},
                ]
            ),
        ):
            self.assertEqual(
                list(chat_service._stream_completion([{"role": "user", "content": "hi"}], "m")),
                ["ok"],
            )

    def test_raw_chunks_transport_failure_has_no_status(self) -> None:
        with mock.patch.object(
            chat_service, "get_settings", return_value=self._settings()
        ), mock.patch("httpx.stream", side_effect=Exception("connection reset")):
            with self.assertRaises(chat_service._LLMError) as ctx:
                list(chat_service._stream_raw_chunks([{"role": "user", "content": "hi"}], "m"))
        self.assertIsNone(ctx.exception.status)
```

- [ ] **Step 2: Run the tests to verify they fail**

```
.venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py -k "raw_chunks or completion_shell" -v
```

Expected: FAIL — `AttributeError: module 'backend.app.services.chat' has no attribute '_stream_raw_chunks'`.

- [ ] **Step 3: Add `_LLMError` and `_stream_raw_chunks`, and reduce `_stream_completion` to a shell**

In `backend/app/services/chat.py`, replace the whole body of `_stream_completion` (currently lines 239–301) with the following three definitions:

```python
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
        except (KeyError, IndexError, TypeError):
            continue  # a keep-alive/unexpected shape — skip, don't abort.
        if delta:
            yield delta
```

- [ ] **Step 4: Run the new tests plus every pre-existing transport test**

```
.venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py -v
```

Expected: PASS, including `test_parses_content_deltas_from_openai_sse`, `test_non_openrouter_base_url_omits_attribution_headers`, `test_extra_body_merges_into_the_request`, `test_stream_ending_without_done_terminator_exhausts_cleanly` and `test_transport_failure_becomes_runtime_error` — **with no edits to any of them**. If any needed editing, revert and redo Step 3.

- [ ] **Step 5: Run the full backend suite**

```
.venv\Scripts\python.exe -m pytest tests/
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/chat.py tests/test_chat_endpoint.py
git commit -m "refactor: split LLM transport out of _stream_completion

_stream_raw_chunks owns HTTP + SSE framing + JSON parsing and yields parsed
chunk dicts; _stream_completion becomes a thin text-only shell over it, with
its signature, behaviour, and every existing test unchanged. _LLMError
carries the upstream status so a 4xx (model rejects the tools field) is
distinguishable from a transport failure. No behaviour change."
```

---

### Task 2: `_stream_turn` — one round, with streamed tool-call reassembly

**Files:**
- Modify: `backend/app/services/chat.py` (add after `_stream_completion`)
- Test: `tests/test_chat_endpoint.py`

**Interfaces:**
- Consumes: `_stream_raw_chunks` from Task 1.
- Produces:
  - `_Turn` dataclass with fields `text: str`, `tool_calls: list[dict[str, Any]]`, `finish_reason: str | None`. Each entry in `tool_calls` is `{"id": str, "name": str, "arguments": str}`.
  - `_stream_turn(messages: list[dict[str, Any]], model: str, *, timeout: float | None = None, tools: list[dict[str, Any]] | None = None) -> Iterator[str | _Turn]` — yields text deltas as `str` as they arrive, then yields **exactly one** `_Turn` as its final item.

- [ ] **Step 1: Write the failing tests**

Add a new class to `tests/test_chat_endpoint.py`:

```python
class StreamTurnTests(unittest.TestCase):
    """One model round: text passthrough plus reassembly of fragmented streamed tool calls."""

    def _settings(self):
        return types.SimpleNamespace(
            llm_api_key="sk-or-test",
            llm_base_url="https://openrouter.ai/api/v1",
        )

    def _run(self, chunks, **kwargs):
        """Drive _stream_turn over a canned chunk list; return (text_deltas, turn)."""
        with mock.patch.object(chat_service, "_stream_raw_chunks", return_value=iter(chunks)):
            items = list(chat_service._stream_turn([{"role": "user", "content": "hi"}], "m", **kwargs))
        turn = items[-1]
        self.assertIsInstance(turn, chat_service._Turn)
        return [i for i in items[:-1]], turn

    def test_text_only_round_yields_deltas_then_a_turn(self) -> None:
        deltas, turn = self._run(
            [
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
        )
        self.assertEqual(deltas, ["Hel", "lo"])
        self.assertEqual(turn.text, "Hello")
        self.assertEqual(turn.tool_calls, [])
        self.assertEqual(turn.finish_reason, "stop")

    def test_tool_call_arguments_split_across_chunks_are_reassembled(self) -> None:
        # ``id``/``name`` arrive only on the first fragment; ``arguments`` streams in pieces.
        _, turn = self._run(
            [
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "call_a", "function": {"name": "kg_query", "arguments": ""}}
                ]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": '{"que'}}
                ]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": 'ry": "knee valgus"}'}}
                ]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
        )
        self.assertEqual(
            turn.tool_calls,
            [{"id": "call_a", "name": "kg_query", "arguments": '{"query": "knee valgus"}'}],
        )
        self.assertEqual(turn.finish_reason, "tool_calls")

    def test_two_interleaved_tool_calls_are_keyed_by_index_not_arrival(self) -> None:
        # Fragments for index 1 arrive before index 0 finishes; accumulation must key on ``index``.
        _, turn = self._run(
            [
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "a", "function": {"name": "kg_query", "arguments": '{"q'}}
                ]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 1, "id": "b", "function": {"name": "rag_search", "arguments": '{"x'}}
                ]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": 'uery": "a"}'}}
                ]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 1, "function": {"arguments": '": "b"}'}}
                ]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
        )
        self.assertEqual(
            turn.tool_calls,
            [
                {"id": "a", "name": "kg_query", "arguments": '{"query": "a"}'},
                {"id": "b", "name": "rag_search", "arguments": '{"x": "b"}'},
            ],
        )

    def test_narration_and_a_tool_call_in_one_round_are_both_captured(self) -> None:
        deltas, turn = self._run(
            [
                {"choices": [{"delta": {"content": "Let me look that up."}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "c", "function": {"name": "rag_search", "arguments": "{}"}}
                ]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
        )
        self.assertEqual(deltas, ["Let me look that up."])
        self.assertEqual(turn.text, "Let me look that up.")
        self.assertEqual(len(turn.tool_calls), 1)

    def test_malformed_chunk_shapes_are_tolerated(self) -> None:
        # Shape tolerance lives HERE, not in the raw layer: usage-only frames, a non-dict delta, and
        # a non-dict tool_calls entry must all be skipped without aborting the round.
        deltas, turn = self._run(
            [
                {"usage": {"total_tokens": 7}},  # no ``choices``
                {"choices": []},  # empty ``choices``
                {"choices": [{"delta": None}]},  # non-dict delta
                {"choices": [{"delta": {"tool_calls": ["not-a-dict"]}}]},
                {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
            ]
        )
        self.assertEqual(deltas, ["ok"])
        self.assertEqual(turn.tool_calls, [])

    def test_tools_are_sent_as_extra_body_only_when_offered(self) -> None:
        with mock.patch.object(
            chat_service, "_stream_raw_chunks", return_value=iter([])
        ) as raw:
            list(chat_service._stream_turn([], "m", tools=[{"type": "function"}]))
        self.assertEqual(
            raw.call_args.kwargs["extra_body"],
            {"tools": [{"type": "function"}], "tool_choice": "auto"},
        )
        with mock.patch.object(
            chat_service, "_stream_raw_chunks", return_value=iter([])
        ) as raw:
            list(chat_service._stream_turn([], "m", tools=None))
        self.assertIsNone(raw.call_args.kwargs["extra_body"])
```

- [ ] **Step 2: Run the tests to verify they fail**

```
.venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py::StreamTurnTests -v
```

Expected: FAIL — `AttributeError: module 'backend.app.services.chat' has no attribute '_stream_turn'`.

- [ ] **Step 3: Implement `_Turn` and `_stream_turn`**

Add `from dataclasses import dataclass` and `from collections.abc import Iterator` (already imported) to the imports at the top of `backend/app/services/chat.py`, then add after `_stream_completion`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```
.venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py::StreamTurnTests -v
```

Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/chat.py tests/test_chat_endpoint.py
git commit -m "feat: add _stream_turn with streamed tool-call reassembly

One model round: yields text deltas as they arrive, then a _Turn carrying the
text, the reassembled tool calls, and finish_reason. Streamed tool calls are
fragmented -- id/name land only on the first fragment for an index and
arguments stream in pieces -- so accumulation is keyed by the fragment's own
index, never by arrival order. Shape tolerance lives here; the raw layer owns
only JSON parsing."
```

---

### Task 3: The tool catalogue, argument clamping, and dispatch

**Files:**
- Modify: `backend/app/services/chat.py` (add after `_stream_turn`)
- Test: `tests/test_chat_endpoint.py`

**Interfaces:**
- Consumes: `_resolve_movement` (existing, `chat.py:122`).
- Produces:
  - `_TOOLS: list[dict[str, Any]]` — the OpenAI function-tool schemas.
  - `_MAX_TOOL_RESULT_CHARS: int`
  - `_clamp_int(value: Any, *, low: int, high: int, default: int) -> int`
  - `_parse_tool_args(raw: str) -> dict[str, Any]`
  - `_tool_query_label(name: str, args: dict[str, Any]) -> str`
  - `_dispatch_tool(name: str, args: dict[str, Any], context: dict[str, Any]) -> str`

- [ ] **Step 1: Write the failing tests**

Add a new class to `tests/test_chat_endpoint.py`:

```python
class ToolDispatchTests(unittest.TestCase):
    """The three tools: argument clamping, routing, and total failure containment."""

    DETAIL = {
        "metadata": {"fps": 30.0, "total_frames": 120},
        "quality": {"valid_frame_ratio": 0.9},
        "view": {"view_type": "side"},
        "detections": [
            {
                "fault_id": "f1",
                "fault_name": "Insufficient Depth",
                "phase": "bottom",
                "severity": 0.7,
                "confidence": 0.9,
                "observability": "clear",
                "start_time": 1.0,
                "end_time": 2.0,
                "start_frame": 30,
                "end_frame": 60,
                "peak_frame": 45,
                "evidence": {"hip_knee_delta_deg": 12.5},
                "rep_index": 2,
                "occurred_reps": [2],
                "rep_count": 1,
            }
        ],
        "retrievals": [
            {"fault_id": "f1", "context": {"results": [{"text": "squat depth passage"}]}},
            {"fault_id": "other", "context": {"results": [{"text": "unrelated"}]}},
        ],
    }

    def _ctx(self):
        return {"movement": "Squat", "detail": self.DETAIL}

    def test_clamp_int_bounds_coerces_and_falls_back(self) -> None:
        self.assertEqual(chat_service._clamp_int(9, low=1, high=2, default=1), 2)
        self.assertEqual(chat_service._clamp_int(-4, low=1, high=8, default=5), 1)
        self.assertEqual(chat_service._clamp_int("3", low=1, high=8, default=5), 3)
        self.assertEqual(chat_service._clamp_int("lots", low=1, high=8, default=5), 5)
        self.assertEqual(chat_service._clamp_int(None, low=1, high=8, default=5), 5)

    def test_parse_tool_args_tolerates_missing_and_malformed_json(self) -> None:
        self.assertEqual(chat_service._parse_tool_args('{"a": 1}'), {"a": 1})
        self.assertEqual(chat_service._parse_tool_args(""), {})
        self.assertEqual(chat_service._parse_tool_args("{not json"), {})
        self.assertEqual(chat_service._parse_tool_args("[1,2]"), {})  # not an object

    def test_get_analysis_without_a_fault_name_returns_clip_level_material(self) -> None:
        out = json.loads(chat_service._dispatch_tool("get_analysis", {"include": "all"}, self._ctx()))
        self.assertEqual(out["quality"], {"valid_frame_ratio": 0.9})
        self.assertEqual(out["view"], {"view_type": "side"})
        self.assertEqual([d["fault_name"] for d in out["detections"]], ["Insufficient Depth"])
        self.assertNotIn("evidence", out["detections"][0])  # summary only, not the full dict

    def test_get_analysis_evidence_carries_the_rep_attribution(self) -> None:
        # Per-rep identity is what makes "第 2 rep 膝蓋幾度" answerable; it exists on the detection
        # but not in the compact blob the prompt is built from, so this tool is the only path to it.
        out = json.loads(
            chat_service._dispatch_tool(
                "get_analysis",
                {"fault_name": "Insufficient Depth", "include": "evidence"},
                self._ctx(),
            )
        )
        self.assertEqual(out["measured"]["rep_index"], 2)
        self.assertEqual(out["measured"]["rep_count"], 1)

    def test_get_analysis_evidence_returns_measurements_and_no_knowledge(self) -> None:
        out = json.loads(
            chat_service._dispatch_tool(
                "get_analysis",
                {"fault_name": "insufficient depth", "include": "evidence"},  # case-insensitive
                self._ctx(),
            )
        )
        self.assertEqual(out["evidence"], {"hip_knee_delta_deg": 12.5})
        self.assertEqual(out["measured"]["peak_frame"], 45)
        self.assertNotIn("knowledge", out)

    def test_get_analysis_knowledge_returns_only_that_faults_retrievals(self) -> None:
        out = json.loads(
            chat_service._dispatch_tool(
                "get_analysis",
                {"fault_name": "Insufficient Depth", "include": "knowledge"},
                self._ctx(),
            )
        )
        self.assertEqual(out["knowledge"], [{"results": [{"text": "squat depth passage"}]}])
        self.assertNotIn("evidence", out)

    def test_get_analysis_unknown_fault_names_what_was_detected(self) -> None:
        out = json.loads(
            chat_service._dispatch_tool(
                "get_analysis", {"fault_name": "Butt Wink", "include": "all"}, self._ctx()
            )
        )
        self.assertIn("error", out)
        self.assertEqual(out["detected_faults"], ["Insufficient Depth"])

    def test_kg_query_clamps_hops_and_forces_the_thread_movement(self) -> None:
        from backend.app.services import knowledge as knowledge_service

        with mock.patch.object(
            knowledge_service, "graph_context", return_value={"ok": True}
        ) as gc:
            chat_service._dispatch_tool("kg_query", {"query": "valgus", "hops": 99}, self._ctx())
        self.assertEqual(gc.call_args.args[0], "valgus")
        self.assertEqual(gc.call_args.kwargs["hops"], 2)  # clamped from 99
        self.assertEqual(gc.call_args.kwargs["movement"], "Squat")

    def test_rag_search_clamps_top_k(self) -> None:
        from backend.app.services import knowledge as knowledge_service

        with mock.patch.object(
            knowledge_service, "rag_snippets", return_value={"results": []}
        ) as rs:
            chat_service._dispatch_tool("rag_search", {"query": "ankle", "top_k": 500}, self._ctx())
        self.assertEqual(rs.call_args.kwargs["top_k"], 8)  # clamped from 500

    def test_unknown_tool_name_is_an_error_payload_not_an_exception(self) -> None:
        out = json.loads(chat_service._dispatch_tool("launch_missiles", {}, self._ctx()))
        self.assertIn("error", out)

    def test_a_raising_tool_becomes_an_error_payload(self) -> None:
        # A missing KG file must not kill an otherwise-fine answer stream.
        from backend.app.services import knowledge as knowledge_service

        with mock.patch.object(
            knowledge_service, "graph_context", side_effect=FileNotFoundError("no graphml")
        ):
            out = json.loads(chat_service._dispatch_tool("kg_query", {"query": "x"}, self._ctx()))
        self.assertIn("error", out)
        self.assertIn("no graphml", out["error"])

    def test_a_huge_tool_result_is_truncated(self) -> None:
        from backend.app.services import knowledge as knowledge_service

        with mock.patch.object(
            knowledge_service, "rag_snippets", return_value={"results": ["x" * 50_000]}
        ):
            out = chat_service._dispatch_tool("rag_search", {"query": "x"}, self._ctx())
        self.assertLessEqual(len(out), chat_service._MAX_TOOL_RESULT_CHARS + 32)
        self.assertTrue(out.endswith("…[truncated]"))

    def test_query_label_picks_the_right_argument_per_tool(self) -> None:
        self.assertEqual(
            chat_service._tool_query_label("kg_query", {"query": "knee valgus"}), "knee valgus"
        )
        self.assertEqual(
            chat_service._tool_query_label("get_analysis", {"fault_name": "Depth"}), "Depth"
        )
        self.assertEqual(chat_service._tool_query_label("get_analysis", {}), "")

    def test_tool_schemas_are_the_three_expected_functions(self) -> None:
        names = [t["function"]["name"] for t in chat_service._TOOLS]
        self.assertEqual(names, ["get_analysis", "kg_query", "rag_search"])
        for tool in chat_service._TOOLS:
            self.assertEqual(tool["type"], "function")
            self.assertIn("description", tool["function"])
            self.assertIn("properties", tool["function"]["parameters"])
```

`types`, `unittest` and `mock` are already imported at the top of `tests/test_chat_endpoint.py`, but **`json` is not** — add `import json` to the stdlib import block (it belongs before `import types`).

- [ ] **Step 2: Run the tests to verify they fail**

```
.venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py::ToolDispatchTests -v
```

Expected: FAIL — `AttributeError: ... has no attribute '_clamp_int'`.

- [ ] **Step 3: Implement the catalogue and dispatcher**

Add `from backend.app.settings import kg_seeds_default` to the existing `backend.app.settings` import block at the top of `backend/app/services/chat.py`, then add after `_stream_turn`:

```python
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
    except (TypeError, ValueError):
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```
.venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py::ToolDispatchTests -v
```

Expected: PASS (13 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/chat.py tests/test_chat_endpoint.py
git commit -m "feat: add the chat tool catalogue and dispatcher

Three tools: get_analysis (reads ChatContext.detail, never the DB, so no auth
or IDOR surface), kg_query (forced to the thread's movement, hops clamped
1-2), rag_search (top_k clamped 1-8). Model-supplied arguments are untrusted
and clamped the way routers/knowledge.py bounds its query params, but silently
rather than 422 since there is no one to answer. Nothing raises out of the
dispatcher -- an unknown tool name or a raising tool becomes an error payload
the model can read, because the HTTP 200 is already committed by then."
```

---

### Task 4: The loop in `answer_stream`, the red line, and `ChatContext.detail`

**Files:**
- Modify: `backend/app/services/chat.py` — `_build_system_prompt` (append the rule) and `answer_stream` (lines 356-385)
- Modify: `backend/app/routers/chat.py:45-68` (`ChatContext`)
- Test: `tests/test_chat_endpoint.py`

**Interfaces:**
- Consumes: `_stream_turn`, `_Turn` (Task 2); `_TOOLS`, `_dispatch_tool`, `_parse_tool_args`, `_tool_query_label` (Task 3).
- Produces:
  - `_MAX_TOOL_ROUNDS = 3`
  - `_TOOL_GROUNDING_RULE: str`
  - `answer_stream(*, messages, context, model) -> Iterator[str]` — same signature, now emitting `tool` and `reset` frames in addition to `delta`/`done`/`error`.
  - `ChatContext.detail: dict[str, Any] | None`

- [ ] **Step 1: Write the failing tests**

Add a new class to `tests/test_chat_endpoint.py`:

```python
class ToolLoopTests(unittest.TestCase):
    """The answer loop: tool rounds, retraction, round/time caps, and tools-unsupported fallback."""

    CONTEXT = {"movement": "Squat", "faults": [], "quality": {"valid_frame_ratio": 0.9}}

    def _turns(self, *rounds):
        """Build a _stream_turn stub that replays one canned round per call.

        Each ``round`` is ``(text_deltas, tool_calls)``; the stub yields the deltas then a _Turn.
        """
        calls = []

        def fake(messages, model, *, timeout=None, tools=None):
            calls.append({"messages": [dict(m) for m in messages], "tools": tools})
            deltas, tool_calls = rounds[len(calls) - 1]
            for d in deltas:
                yield d
            yield chat_service._Turn(
                text="".join(deltas),
                tool_calls=list(tool_calls),
                finish_reason="tool_calls" if tool_calls else "stop",
            )

        return fake, calls

    @staticmethod
    def _events(frames):
        """Parse rendered SSE frames into [(event, data), ...]."""
        out = []
        for frame in "".join(frames).split("\n\n"):
            if not frame.strip():
                continue
            lines = frame.split("\n")
            event = lines[0][len("event:") :].strip()
            data = json.loads(lines[1][len("data:") :].strip())
            out.append((event, data))
        return out

    def _run(self, fake_turn, dispatch=None):
        with mock.patch.object(chat_service, "_stream_turn", fake_turn), mock.patch.object(
            chat_service, "_dispatch_tool", dispatch or (lambda n, a, c: '{"ok": true}')
        ), mock.patch.object(chat_service, "chat_timeout", return_value=60.0):
            return self._events(
                list(
                    chat_service.answer_stream(
                        messages=[{"role": "user", "content": "hi"}],
                        context=self.CONTEXT,
                        model="m",
                    )
                )
            )

    def test_a_plain_turn_streams_deltas_with_no_tool_or_reset_frames(self) -> None:
        # Regression lock: the common path (no tool call) must stream token-by-token exactly as it
        # does today -- not be buffered and flushed at the end.
        fake, calls = self._turns((["Hel", "lo"], []))
        events = self._run(fake)
        self.assertEqual(
            events, [("delta", {"text": "Hel"}), ("delta", {"text": "lo"}), ("done", {"model": "m"})]
        )
        self.assertIsNotNone(calls[0]["tools"])  # tools were offered on round 1

    def test_a_tool_round_emits_a_tool_frame_then_the_answer(self) -> None:
        fake, calls = self._turns(
            ([], [{"id": "c1", "name": "kg_query", "arguments": '{"query": "valgus"}'}]),
            (["Answer"], []),
        )
        events = self._run(fake)
        self.assertEqual(
            events,
            [
                ("tool", {"name": "kg_query", "query": "valgus"}),
                ("delta", {"text": "Answer"}),
                ("done", {"model": "m"}),
            ],
        )
        # Round 2 saw the assistant tool_calls turn plus the tool result appended.
        roles = [m["role"] for m in calls[1]["messages"]]
        self.assertEqual(roles[-2:], ["assistant", "tool"])
        self.assertEqual(calls[1]["messages"][-1]["tool_call_id"], "c1")

    def test_narration_plus_a_tool_call_is_retracted_with_a_reset_frame(self) -> None:
        fake, _ = self._turns(
            (["Let me check."], [{"id": "c1", "name": "rag_search", "arguments": '{"query": "x"}'}]),
            (["Real answer"], []),
        )
        events = self._run(fake)
        self.assertEqual(
            events,
            [
                ("delta", {"text": "Let me check."}),
                ("reset", {}),
                ("tool", {"name": "rag_search", "query": "x"}),
                ("delta", {"text": "Real answer"}),
                ("done", {"model": "m"}),
            ],
        )

    def test_a_tool_round_with_no_narration_emits_no_reset(self) -> None:
        fake, _ = self._turns(
            ([], [{"id": "c1", "name": "kg_query", "arguments": "{}"}]),
            (["A"], []),
        )
        self.assertNotIn("reset", [e for e, _ in self._run(fake)])

    def test_the_loop_stops_after_three_tool_rounds_and_forces_a_toolless_answer(self) -> None:
        call = [{"id": "c", "name": "kg_query", "arguments": "{}"}]
        fake, calls = self._turns(
            ([], call), ([], call), ([], call), (["Forced"], [])
        )
        events = self._run(fake)
        self.assertEqual(events[-2:], [("delta", {"text": "Forced"}), ("done", {"model": "m"})])
        self.assertEqual(len(calls), 4)
        self.assertIsNotNone(calls[2]["tools"])  # round 3 still offers tools
        self.assertIsNone(calls[3]["tools"])  # the forced final round does not

    def test_an_exhausted_time_budget_stops_the_loop(self) -> None:
        # chat_timeout() of 0 means there is no budget left before the first round even starts.
        fake, _ = self._turns((["never"], []))
        with mock.patch.object(chat_service, "_stream_turn", fake), mock.patch.object(
            chat_service, "chat_timeout", return_value=0.0
        ):
            events = self._events(
                list(
                    chat_service.answer_stream(
                        messages=[{"role": "user", "content": "hi"}],
                        context=self.CONTEXT,
                        model="m",
                    )
                )
            )
        self.assertEqual([e for e, _ in events], ["error"])

    def test_a_4xx_before_any_delta_retries_once_without_tools(self) -> None:
        attempts = []

        def fake(messages, model, *, timeout=None, tools=None):
            attempts.append(tools)
            if len(attempts) == 1:
                raise chat_service._LLMError("no tool support", status=400)
            yield "Plain"
            yield chat_service._Turn(text="Plain", tool_calls=[], finish_reason="stop")

        events = self._run(fake)
        self.assertEqual(events, [("delta", {"text": "Plain"}), ("done", {"model": "m"})])
        self.assertIsNotNone(attempts[0])  # first attempt offered tools
        self.assertIsNone(attempts[1])  # the retry did not

    def test_a_4xx_after_a_delta_is_a_terminal_error_with_no_retry(self) -> None:
        attempts = []

        def fake(messages, model, *, timeout=None, tools=None):
            attempts.append(tools)
            yield "partial"
            raise chat_service._LLMError("died mid-stream", status=400)

        events = self._run(fake)
        self.assertEqual([e for e, _ in events], ["delta", "error"])
        self.assertEqual(len(attempts), 1)  # committed output -> no retry

    def test_a_transport_failure_is_never_retried(self) -> None:
        attempts = []

        def fake(messages, model, *, timeout=None, tools=None):
            attempts.append(tools)
            raise chat_service._LLMError("connection reset")  # status=None
            yield  # pragma: no cover — makes this a generator

        events = self._run(fake)
        self.assertEqual([e for e, _ in events], ["error"])
        self.assertEqual(len(attempts), 1)

    def test_an_empty_completion_is_still_an_error(self) -> None:
        fake, _ = self._turns(([], []))
        events = self._run(fake)
        self.assertEqual([e for e, _ in events], ["error"])

    def test_the_tool_grounding_rule_is_in_the_system_prompt(self) -> None:
        fake, calls = self._turns((["ok"], []))
        self._run(fake)
        system = calls[0]["messages"][0]["content"]
        self.assertEqual(calls[0]["messages"][0]["role"], "system")
        self.assertIn("REFERENCE MATERIAL", system)
        self.assertIn("ANALYSIS FACTS", system)  # the v2 prompt is still there, unchanged
```

Also add, to the existing `ChatRouterTests` class:

```python
    def test_chat_context_accepts_a_detail_blob(self) -> None:
        from backend.app.routers.chat import ChatContext

        ctx = ChatContext(fault_count=0, detail={"detections": [], "retrievals": []})
        self.assertEqual(ctx.model_dump()["detail"], {"detections": [], "retrievals": []})
        self.assertIsNone(ChatContext(fault_count=0).model_dump()["detail"])  # optional
```

- [ ] **Step 2: Run the tests to verify they fail**

```
.venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py::ToolLoopTests -v
```

Expected: FAIL — the current `answer_stream` emits no `tool`/`reset` frames and never calls `_stream_turn`.

- [ ] **Step 3: Add the red line to the system prompt**

In `backend/app/services/chat.py`, add this constant just after `_followup_instruction` (around line 88):

```python
# The honesty rule that live retrieval makes necessary, and that v1/v2 never needed. kg_query and
# rag_search return knowledge about faults THIS REP DID NOT EXHIBIT; nothing in the v2 GROUNDING
# RULES stops a model from relaying a retrieved fault as an observation, because before v3 there was
# no channel through which one could arrive. Appended to (never woven into) _build_system_prompt, so
# the CLEAN REP / NOT MEASURED branches and their tests stay byte-identical.
_TOOL_GROUNDING_RULE = (
    "\nTOOL RULES:\n"
    "- You may call tools to look things up before answering.\n"
    "- `get_analysis` returns MORE DETAIL ABOUT THIS VIDEO. It is as authoritative as the facts "
    "above — it is the same analysis, just uncompressed.\n"
    "- Knowledge returned by `kg_query` and `rag_search` is REFERENCE MATERIAL, NOT AN OBSERVATION "
    "ABOUT THIS VIDEO. If they mention a fault, that does NOT mean the user committed it. Only the "
    "faults listed in ANALYSIS FACTS were actually detected. When you use retrieved knowledge, say "
    "plainly that it is general reference rather than something measured in this rep.\n"
)
```

- [ ] **Step 4: Replace `answer_stream` with the loop**

Add `import time` to the imports at the top of `backend/app/services/chat.py`, add the round cap constant beside `_MAX_TOOL_RESULT_CHARS`:

```python
# Three tools, one call each, is the realistic ceiling for a coaching follow-up; beyond that the
# model is looping rather than researching. The (N+1)th round is the forced tools-free one.
_MAX_TOOL_ROUNDS = 3
```

then replace the entire `answer_stream` function (currently lines 356-385) with:

```python
def answer_stream(
    *, messages: list[dict[str, str]], context: dict[str, Any], model: str
) -> Iterator[str]:
    """Stream a grounded coaching reply for ``messages`` as SSE frames, running tools as needed.

    ``messages`` is the client-held conversation (roles ``user``/``assistant``), newest last; the
    backend prepends the grounded system prompt. ``model`` is the already-resolved (allow-listed)
    provider slug. Yields ``delta`` frames, optional ``tool``/``reset`` frames, then exactly one
    terminator: ``done`` (carrying the model used) or ``error``.

    TOOL ROUND-TRIPS NEVER LEAVE THIS FUNCTION. ``ChatMessage.role`` is ``Literal["user",
    "assistant"]``, the client holds the conversation, and ``store.upsert_conversation`` persists it
    — so a ``role:"tool"`` turn has nowhere to live. The loop therefore runs entirely server-side
    inside one request and only the final assistant text is streamed and persisted (spec v3
    decision 3).

    RETRACTION, NOT BUFFERING (spec v3 section 1). A round streams its text the moment it arrives,
    because ``finish_reason`` only lands at the *end* of a round — "stream only the final round"
    is undecidable in flight and degenerates into buffering every round, which would cost
    token-by-token streaming on the commonest path of all, the turn that calls no tools. Instead, if
    a round turns out to have produced BOTH text and tool calls, a ``reset`` frame tells the client
    to discard what it has. That is safe because the client commits the assistant turn only after
    the stream ends.

    THE TIME BUDGET IS SHARED, NOT PER-ROUND. This endpoint is metered; N rounds must not cost N×
    ``chat_timeout()``. Each round is given only what is left.
    """
    system = _build_system_prompt(context) + _TOOL_GROUNDING_RULE
    convo: list[dict[str, Any]] = [{"role": "system", "content": system}, *messages]
    deadline = time.monotonic() + chat_timeout()
    # True once the HTTP 200 has carried output, which makes a retry illegal (it would double-emit).
    # NOTE it is always False at the TOP of an iteration, by induction: it starts False, and reaching
    # the next iteration requires tool calls, after which a round that narrated has emitted ``reset``
    # and cleared it while a round that did not never set it. That is why the timeout branch above
    # cannot have an "already streamed" case — do not add one back; it would be dead code and would
    # show up as a partial branch under the 95% coverage gate.
    streamed_any = False
    answer = ""

    for round_index in range(_MAX_TOOL_ROUNDS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # Nothing has reached the client at this point, ever — see the ``streamed_any`` note
            # below — so an honest error is the only outcome available here.
            yield _sse("error", {"detail": "The coach ran out of time before answering."})
            return

        # The extra final iteration never offers tools: it is the "answer from what you gathered"
        # round the caps fall through to, so the user gets prose instead of a failure.
        offer_tools = round_index < _MAX_TOOL_ROUNDS
        turn: _Turn | None = None
        narrated = False

        try:
            for item in _stream_turn(
                convo, model, timeout=remaining, tools=_TOOLS if offer_tools else None
            ):
                if isinstance(item, _Turn):
                    turn = item
                else:
                    narrated = True
                    streamed_any = True
                    yield _sse("delta", {"text": item})
        except _LLMError as exc:
            # A 4xx *before anything reached the client* is the signature of a model that rejects the
            # ``tools`` field. Retry this same round once, plainly, so an unsupported model degrades
            # to today's behaviour instead of failing visibly. A transport failure (status None)
            # cannot be retried into working, and once output is committed a retry would double-emit.
            retryable = (
                offer_tools
                and not streamed_any
                and exc.status is not None
                and 400 <= exc.status < 500
            )
            if not retryable:
                yield _sse("error", {"detail": str(exc)})
                return
            try:
                for item in _stream_turn(
                    convo, model, timeout=max(deadline - time.monotonic(), 1.0), tools=None
                ):
                    if isinstance(item, _Turn):
                        turn = item
                    else:
                        streamed_any = True
                        yield _sse("delta", {"text": item})
            except _LLMError as retry_exc:
                yield _sse("error", {"detail": str(retry_exc)})
                return
            offer_tools = False  # the retry ran without tools, so this round is final by construction

        if turn is None:  # the round produced no terminal _Turn at all — treat as an empty reply.
            yield _sse("error", {"detail": "The LLM returned an empty message."})
            return

        # ``not offer_tools`` closes the door on a model that emits tool_calls it was never offered:
        # without it such a round would loop until the range is exhausted and fall out with no answer.
        if not turn.tool_calls or not offer_tools:
            answer += turn.text
            break

        if narrated:
            # This round narrated AND called a tool. The narration is not the answer — retract it,
            # and re-arm the retry precondition, since the client is back to a clean slate.
            yield _sse("reset", {})
            streamed_any = False

        convo.append(
            {
                "role": "assistant",
                "content": turn.text,
                "tool_calls": [
                    {
                        "id": c["id"] or f"call_{i}",
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                    for i, c in enumerate(turn.tool_calls)
                ],
            }
        )
        for i, call in enumerate(turn.tool_calls):
            args = _parse_tool_args(call["arguments"])
            yield _sse("tool", {"name": call["name"], "query": _tool_query_label(call["name"], args)})
            convo.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"] or f"call_{i}",
                    "content": _dispatch_tool(call["name"], args, context),
                }
            )

    if not answer.strip():
        # Preserves the v1 invariant: the client must never keep an empty assistant turn, which the
        # next send's ``content min_length=1`` would reject.
        yield _sse("error", {"detail": "The LLM returned an empty message."})
        return

    yield _sse("done", {"model": model})
```

- [ ] **Step 5: Add `detail` to `ChatContext`**

In `backend/app/routers/chat.py`, add to the `ChatContext` model (after the `faults` field, around line 67):

```python
    # The FULL analysis document (detections + retrievals, minus the heavy `pose` block), shipped so
    # the `get_analysis` tool can read the detail `buildChatContext` compresses away — exact measured
    # values and complete reference passages. Optional: absent from a client predating v3, and
    # deliberately omitted by `/api/chat/followups`, which shares this model but can never use it.
    #
    # This is NOT persisted (`upsert_conversation` stores messages + followups only) and never enters
    # the prompt unless a tool returns part of it, so its cost is request body size, not tokens.
    detail: dict[str, Any] | None = None
```

- [ ] **Step 6: Run the tests to verify they pass**

```
.venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py -v
```

Expected: PASS, including every pre-existing test in the file.

- [ ] **Step 7: Run the full backend suite and the coverage gate**

```
.venv\Scripts\python.exe -m pytest tests/
.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95
```

Expected: both PASS. If coverage on `chat.py` dropped, add tests for the uncovered branches — do not lower the gate.

- [ ] **Step 8: Commit**

```bash
git add backend/app/services/chat.py backend/app/routers/chat.py tests/test_chat_endpoint.py
git commit -m "feat: run a bounded tool-calling loop in answer_stream

Up to 3 tool rounds plus a forced tools-free final round, all inside one
shared chat_timeout() budget rather than one budget per round. Tool
round-trips never leave the request: ChatMessage.role has no 'tool' and
threads are persisted, so only the final assistant text is streamed and
saved. A round that narrates and then calls a tool emits a reset frame to
retract the narration -- streaming stays optimistic so the common no-tool
turn is unaffected. A 4xx before any committed output retries once without
tools, so a model lacking function calling degrades to today's behaviour.
ChatContext gains an optional detail blob for get_analysis to read."
```

---

### Task 5: Frontend wire-up — `detail`, `tool`/`reset` frames, followups stay lean

**Files:**
- Modify: `frontend/src/api.ts:166-183` (`ChatContext`, `ChatStreamHandlers`), `:361-379` (`dispatchSSE`), `:655-669` (`chatFollowups`)
- Modify: `frontend/src/lib/grounding.ts`
- Test: `frontend/src/test/api.test.ts`

**Interfaces:**
- Consumes: the SSE contract from Task 4 (`tool` / `reset` frames).
- Produces:
  - `ChatContext.detail?: Record<string, unknown>`
  - `ChatStreamHandlers.onTool?: (name: string, query: string) => void`
  - `ChatStreamHandlers.onReset?: () => void`

- [ ] **Step 1: Write the failing tests**

Add to `frontend/src/test/api.test.ts`:

```ts
describe("chat SSE tool frames", () => {
  it("routes tool and reset frames to their handlers", async () => {
    const seen: string[] = [];
    const body = [
      'event: tool\ndata: {"name":"kg_query","query":"knee valgus"}\n\n',
      'event: delta\ndata: {"text":"hm"}\n\n',
      "event: reset\ndata: {}\n\n",
      'event: delta\ndata: {"text":"answer"}\n\n',
      'event: done\ndata: {"model":"m"}\n\n',
    ].join("");
    stubFetchStream(body);
    await api.chatStream([{ role: "user", content: "hi" }], { fault_count: 0, quality: {}, faults: [] }, {
      onDelta: (t) => seen.push(`delta:${t}`),
      onDone: () => seen.push("done"),
      onError: () => seen.push("error"),
      onTool: (n, q) => seen.push(`tool:${n}:${q}`),
      onReset: () => seen.push("reset"),
    });
    expect(seen).toEqual([
      "tool:kg_query:knee valgus",
      "delta:hm",
      "reset",
      "delta:answer",
      "done",
    ]);
  });

  it("ignores tool and reset frames when the handlers are absent", async () => {
    const seen: string[] = [];
    stubFetchStream('event: tool\ndata: {"name":"kg_query","query":"x"}\n\nevent: reset\ndata: {}\n\nevent: done\ndata: {"model":"m"}\n\n');
    await api.chatStream([{ role: "user", content: "hi" }], { fault_count: 0, quality: {}, faults: [] }, {
      onDelta: () => seen.push("delta"),
      onDone: () => seen.push("done"),
      onError: () => seen.push("error"),
    });
    expect(seen).toEqual(["done"]);
  });

  it("strips detail from the followups request", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ questions: [] }),
    });
    vi.stubGlobal("fetch", fetchMock);
    await api.chatFollowups(
      [{ role: "user", content: "hi" }],
      { fault_count: 0, quality: {}, faults: [], detail: { detections: [] } },
    );
    const sent = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(sent.context.detail).toBeUndefined();
    expect(sent.context.fault_count).toBe(0);
  });
});
```

Use whatever fetch-stream stub helper the file already defines for `chatStream`; if there is none, write `stubFetchStream` locally:

```ts
function stubFetchStream(body: string) {
  const bytes = new TextEncoder().encode(body);
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      body: {
        getReader: () => {
          let sent = false;
          return {
            read: async () =>
              sent ? { done: true, value: undefined } : ((sent = true), { done: false, value: bytes }),
          };
        },
      },
    }),
  );
}
```

Add to `frontend/src/test/` a check that `buildChatContext` ships `detail` without `pose` — put it in the existing grounding test file if one exists, otherwise create `frontend/src/test/grounding.detail.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { buildChatContext } from "../lib/grounding";
import type { Analysis } from "../api";

const ANALYSIS = {
  video_id: "v1",
  metadata: { fps: 30, width: 1, height: 1, total_frames: 10 },
  view: { view_type: "side", view_confidence: 0.9 },
  quality: { valid_frame_ratio: 0.9 },
  detections: [
    {
      fault_id: "f1",
      fault_name: "Insufficient Depth",
      kg_query: "depth",
      retrieval_mode: "kg",
      severity: 0.7,
      confidence: 0.9,
      observability: "clear",
      start_time: 1,
      end_time: 2,
      start_frame: 30,
      end_frame: 60,
      peak_frame: 45,
      phase: "bottom",
      evidence: { hip_knee_delta_deg: 12.5 },
    },
  ],
  retrievals: [
    { fault_id: "f1", fault_name: "Insufficient Depth", query_text: "d", retrieval_mode: "kg", context: {} },
  ],
  pose: { fps: 30, width: 1, height: 1, frames: [{ i: 0, lm: null }] },
  source: "upload",
  movement: "Squat",
} as unknown as Analysis;

describe("buildChatContext detail", () => {
  it("ships the full detections and retrievals for the get_analysis tool", () => {
    const detail = buildChatContext(ANALYSIS).detail as Record<string, unknown>;
    expect((detail.detections as unknown[])[0]).toMatchObject({
      evidence: { hip_knee_delta_deg: 12.5 },
      peak_frame: 45,
    });
    expect(detail.retrievals).toHaveLength(1);
    expect(detail.quality).toEqual({ valid_frame_ratio: 0.9 });
  });

  it("omits the heavy pose block", () => {
    const detail = buildChatContext(ANALYSIS).detail as Record<string, unknown>;
    expect(detail.pose).toBeUndefined();
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

With cwd = `frontend/`:

```
yarn test
```

Expected: FAIL — `onTool` is not a valid handler property (TypeScript) and `detail` is undefined.

- [ ] **Step 3: Extend the types and `dispatchSSE`**

In `frontend/src/api.ts`, add to `ChatContext` (after `movement`):

```ts
  // The full analysis document (detections + retrievals, no `pose`), read server-side by the
  // `get_analysis` tool. Never persisted, and never sent on the followups call.
  detail?: Record<string, unknown>;
```

Extend `ChatStreamHandlers`:

```ts
export interface ChatStreamHandlers {
  onDelta: (text: string) => void;
  onDone: (model: string) => void;
  onError: (detail: string) => void;
  // The coach called a tool. Optional so a caller that doesn't surface tool progress is unaffected.
  onTool?: (name: string, query: string) => void;
  // Discard everything streamed so far this turn: the round that produced it also called a tool, so
  // its text was narration ("let me look that up"), not the answer. Safe because the caller commits
  // the assistant turn only once the stream ends.
  onReset?: () => void;
}
```

In `dispatchSSE`, add two branches after the `delta` branch:

```ts
  if (event === "delta") handlers.onDelta(data.text ?? "");
  else if (event === "tool") handlers.onTool?.(data.name ?? "", data.query ?? "");
  else if (event === "reset") handlers.onReset?.();
  else if (event === "done") handlers.onDone(data.model ?? "");
  else if (event === "error") handlers.onError(data.detail ?? "Chat failed");
```

In `chatFollowups`, drop `detail` before sending:

```ts
  async chatFollowups(
    messages: ChatMessage[],
    context: ChatContext,
    model?: string
  ): Promise<string[]> {
    // Strip `detail`: it is the bulk of the payload (full RAG passage text for every fault) and this
    // fire-and-forget call can never use it — the followups endpoint runs no tools. Chip latency is
    // a defended ~1.5s and nothing gets added to this path.
    const { detail: _unused, ...lean } = context;
    const res = await fetch("/api/chat/followups", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(await authHeader()) },
      body: JSON.stringify(
        model ? { messages, context: lean, model } : { messages, context: lean }
      ),
    });
    if (!res.ok) return [];
    const data = (await res.json().catch(() => ({}))) as { questions?: string[] };
    return data.questions ?? [];
  },
```

- [ ] **Step 4: Ship `detail` from `buildChatContext`**

In `frontend/src/lib/grounding.ts`, add to the returned object (after `movement`):

```ts
    // The uncompressed analysis, for the backend's `get_analysis` tool: the exact measured values
    // and the complete retrieved passages this function summarises away above. `pose` is excluded —
    // it is by far the heaviest block and the coach has no use for raw landmarks.
    detail: {
      metadata: analysis.metadata,
      quality: analysis.quality,
      view: analysis.view,
      detections: analysis.detections,
      retrievals: analysis.retrievals,
    },
```

- [ ] **Step 5: Run the tests to verify they pass**

With cwd = `frontend/`:

```
yarn test
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/api.ts frontend/src/lib/grounding.ts frontend/src/test/
git commit -m "feat: wire the chat client to tool and reset SSE frames

ChatContext gains a detail blob (full detections + retrievals, no pose) for
the backend's get_analysis tool; dispatchSSE routes the new tool and reset
events to optional handlers so a caller that ignores them is unaffected; and
chatFollowups strips detail before sending, since that fire-and-forget call
runs no tools and its ~1.5s latency is a defended property."
```

---

### Task 6: `CoachTray` — the tool status line

**Files:**
- Modify: `frontend/src/components/CoachTray.tsx:57-61` (state), `:161-179` (handlers), `:207-210` (`finally`), `:418-432` (render)
- Modify: `frontend/src/lib/i18n.tsx` (both dictionaries)
- Test: `frontend/src/test/components.CoachTray.chat.test.tsx`

**Interfaces:**
- Consumes: `onTool` / `onReset` from Task 5.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing tests**

Add to `frontend/src/test/components.CoachTray.chat.test.tsx`:

**The stream must be held open across these assertions.** `send`'s `finally` clears both `tool` and
`streaming` the moment `chatStream` resolves. A mock that resolves immediately would make the
"shows the line" test fail (the line is already gone) and make the "clears the line" test pass
*vacuously* — it would pass even if `onDelta` cleared nothing. So the mock parks on a promise the
test releases:

```tsx
  it("shows a named tool status line, then clears it when the answer streams", async () => {
    let releaseTool!: () => void;
    let releaseDelta!: () => void;
    vi.spyOn(api, "chatStream").mockImplementation(async (_m, _c, h) => {
      h.onTool?.("kg_query", "knee valgus");
      await new Promise<void>((r) => (releaseTool = r)); // assert while the tool is running
      h.onDelta("Answer");
      await new Promise<void>((r) => (releaseDelta = r)); // assert while the answer streams
      h.onDone("m");
    });
    renderTray();
    void sendMessage("why?"); // NOT awaited — the stream is deliberately still open

    // The tool line is up, naming both the tool and its subject.
    expect(await screen.findByText(/knee valgus/)).toBeTruthy();

    // The first answer token retires it.
    releaseTool();
    expect(await screen.findByText("Answer")).toBeTruthy();
    expect(screen.queryByText(/knee valgus/)).toBeNull();

    releaseDelta();
  });

  it("falls back to a generic label for a tool it has no i18n string for", async () => {
    // `t()` returns the key itself on a miss (i18n.tsx:1421), so an unguarded lookup would render
    // "chat.tool.something_else" straight into the tray.
    let release!: () => void;
    vi.spyOn(api, "chatStream").mockImplementation(async (_m, _c, h) => {
      h.onTool?.("something_else", "x");
      await new Promise<void>((r) => (release = r));
      h.onDelta("A");
      h.onDone("m");
    });
    renderTray();
    void sendMessage("why?");
    expect(await screen.findByText(/x/)).toBeTruthy();
    expect(screen.queryByText(/chat\.tool\./)).toBeNull();
    release();
  });

  it("discards streamed narration when the server sends reset", async () => {
    vi.spyOn(api, "chatStream").mockImplementation(async (_m, _c, h) => {
      h.onDelta("Let me check.");
      h.onReset?.();
      h.onTool?.("kg_query", "valgus");
      h.onDelta("Real answer");
      h.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    expect(await screen.findByText("Real answer")).toBeTruthy();
    expect(screen.queryByText(/Let me check/)).toBeNull();
  });
```

The `reset` test *can* await `sendMessage`, unlike the two above: it asserts on the **committed**
assistant turn, which survives the `finally`.

Reuse the file's existing `renderTray` / `sendMessage` helpers and its `api` import.

- [ ] **Step 2: Run the tests to verify they fail**

With cwd = `frontend/`:

```
yarn test components.CoachTray.chat
```

Expected: FAIL — no tool status line is rendered, and the narration is not discarded.

- [ ] **Step 3: Add the i18n strings**

In `frontend/src/lib/i18n.tsx`, add to the English dictionary beside `"chat.thinking"` (around line 98):

```ts
  "chat.tool.get_analysis": "Re-reading the analysis",
  "chat.tool.kg_query": "Searching the knowledge graph",
  "chat.tool.rag_search": "Searching the literature",
  "chat.tool.generic": "Looking something up",
```

and to the Traditional Chinese dictionary beside its `"chat.thinking"` (around line 777):

```ts
  "chat.tool.get_analysis": "重讀分析細節",
  "chat.tool.kg_query": "搜尋知識圖譜",
  "chat.tool.rag_search": "查詢文獻",
  "chat.tool.generic": "查詢中",
```

- [ ] **Step 4: Wire the state, handlers, and render**

In `frontend/src/components/CoachTray.tsx`, add beside the other chat state (around line 57):

```tsx
  // The tool the coach is currently running, shown as a transient status line. Cleared the moment
  // the real answer starts streaming — the tool work is finished by then, by construction.
  const [tool, setTool] = useState<{ name: string; query: string } | null>(null);
```

In `send`, clear it beside `setStreaming("")` (around line 155):

```tsx
    setStreaming("");
    setTool(null);
```

and extend the handlers object passed to `api.chatStream` (around line 165):

```tsx
        {
          onDelta: (tkn) => {
            acc += tkn;
            setStreaming(acc);
            setTool(null); // the answer is arriving; any tool work is done.
          },
          onDone: () => undefined,
          onTool: (name, query) => setTool({ name, query }),
          // The round that produced this text also called a tool, so it was narration, not the
          // answer. Drop it — `acc` is what gets committed when the stream ends.
          onReset: () => {
            acc = "";
            setStreaming("");
          },
          // An in-band error (LLM provider connect/mid-stream/empty) isn't thrown — capture it and
          // rethrow below so success and failure share one rollback path.
          onError: (detail) => {
            inbandError = detail;
          },
        },
```

In the `finally` block (around line 207):

```tsx
    } finally {
      setLoading(false);
      setStreaming("");
      setTool(null);
    }
```

In the render, replace the existing thinking-dots block (around line 426) so the tool line takes its place while a tool runs:

```tsx
                {/* A tool is running: name what the coach is looking up, and what for. The project's
                    whole thesis is explainability, so the query is shown, not just a spinner. */}
                {tool && !streaming && (
                  <div className="flex items-center gap-2 text-xs text-muted">
                    <LumenLoader variant="dots" />
                    <span>
                      {TOOL_LABEL_KEYS.includes(tool.name as (typeof TOOL_LABEL_KEYS)[number])
                        ? t(`chat.tool.${tool.name}` as never)
                        : t("chat.tool.generic")}
                      {tool.query ? `：${tool.query}` : ""}
                    </span>
                  </div>
                )}
                {/* Lumen's dots only until the first token lands; then the streaming text carries it. */}
                {loading && !streaming && !tool && (
                  <div className="flex items-center gap-2 text-xs text-muted">
                    <LumenLoader variant="dots" />
                    {t("chat.thinking")}
                  </div>
                )}
```

The whitelist is **required**, not optional: `i18n.tsx:1421` is
`DICTS[lang][key] ?? en[key] ?? key`, so a miss returns the *key string*, which is truthy — a
`t(...) || t("chat.tool.generic")` fallback would silently render `chat.tool.launch_missiles` to the
user if the model ever names an unknown tool. Declare the whitelist at module scope in
`CoachTray.tsx`, beside the other module-level constants:

```tsx
// The tools we have i18n labels for. A name outside this list falls back to the generic label —
// `t()` returns the key itself on a miss (i18n.tsx:1421), so an unguarded lookup would render a raw
// key like "chat.tool.something_else" into the tray.
const TOOL_LABEL_KEYS = ["get_analysis", "kg_query", "rag_search"] as const;
```

- [ ] **Step 5: Run the tests to verify they pass**

With cwd = `frontend/`:

```
yarn test
```

Expected: PASS.

- [ ] **Step 6: Run the full frontend gate and the build**

With cwd = `frontend/`:

```
yarn test:coverage
yarn build
```

Expected: both PASS.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/CoachTray.tsx frontend/src/lib/i18n.tsx frontend/src/test/
git commit -m "feat: show a named tool status line in the coach tray

While the coach runs a tool, the tray names both the tool and its query
('搜尋知識圖譜:knee valgus') in place of the thinking dots, cleared the moment
the answer starts streaming. Showing the query rather than a generic spinner
matches the project's explainability thesis. onReset drops narration the
server retracted."
```

---

### Task 7: Live verification against real models

**Files:** none — this task changes no code unless it finds a defect.

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces: a `## Build notes (v3)` block appended to `specs/llm-chat-spec.md`.

This task exists because three things in the spec's `Known risks` cannot be settled by the test suite, and two of them decide whether the feature works at all.

- [ ] **Step 1: Start the stack**

Backend from the repo root, frontend from `frontend/`:

```
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

```
yarn dev
```

- [ ] **Step 2: Check every allow-listed model for function-calling support**

For each slug in `LLM_MODELS` (default: `deepseek/deepseek-v4-flash`, `xiaomi/mimo-v2.5`, `minimax/minimax-m3`, `tencent/hy3-preview`), pick it in Settings, then ask a question that forces a tool: **"文獻上怎麼說腳踝活動度不足對深蹲的影響?"** (nothing about ankle mobility is in the analysis, so it must call `rag_search`).

Record per model: did a `tool` frame appear; did the answer arrive; did the 4xx fallback fire silently.

- [ ] **Step 3: Test the honesty red line — the one that can damage the product**

Ask the same question on a **clean-rep** analysis (no detected faults). The coach must return the retrieved ankle-mobility knowledge while stating plainly it is general reference and was **not** observed in this rep. It must not say or imply the user has an ankle mobility problem.

Repeat on an analysis with one detected fault, asking about a *different* fault ("我有 butt wink 嗎?" when butt wink was not detected).

If either leaks — if retrieved knowledge is relayed as an observation — apply the spec's stated fallback: prefix every tool-returned knowledge block inline with `REFERENCE ONLY — not measured in this video:` in `_dispatch_tool`, rather than relying on the system-prompt rule alone. Re-test.

- [ ] **Step 4: Verify the detail path and the retraction path**

- Ask the spec's success criterion 1 verbatim — **"我第 2 rep 的膝蓋角度是多少?"** — on a multi-rep clip. Expect a `get_analysis` tool frame and a real number, not "分析中沒有測量". This is answerable because `rep_index`/`occurred_reps`/`rep_count` reach the payload via `asdict` (`pose_rule_detector.py:105-107, 664`) and Task 3 puts them in `measured`; if the answer comes back rep-agnostic, check that whitelist first.
- On a clip that fell back to whole-clip detection (`rep_count` 0), ask the same question. The coach must say per-rep attribution is unavailable rather than inventing a rep number.
- Watch for a turn where the coach narrates before calling a tool. If the narration ever remains on screen next to the final answer, `reset` is not reaching the client — check the frame ordering in `answer_stream` (the `reset` must precede the `tool` frames).

- [ ] **Step 5: Re-measure follow-up chip latency**

Time the chips over several turns. The v2.1 baseline is ~1.5s. If it regressed, confirm `chatFollowups` is stripping `detail` and that `suggest_followups` is not sending `tools`.

- [ ] **Step 6: Record the findings in the spec**

Append a `## Build notes (v3)` section to `specs/llm-chat-spec.md` covering: which models support tools, whether the red line held, measured chip latency, and any fallback applied. Keep the spec alive the way v2 and v2.1 were.

- [ ] **Step 7: Commit**

```bash
git add specs/llm-chat-spec.md
git commit -m "docs: v3 build notes — live model and grounding verification"
```

---

## Self-Review

**Spec coverage.** v3 §1 transport split → Tasks 1–2. §2 loop, caps, degradation → Task 4. §3 tool catalogue and clamping → Task 3. §4 red line → Task 4 Step 3, verified live in Task 7 Step 3. §5 SSE contract → Task 4 (emit) + Task 5 (consume). §6 client → Tasks 5–6. §7 testing → tests in every task, gates in Task 4 Step 7 and Task 6 Step 6. Success criteria 1–4 → Task 7 Steps 2–4; criterion 5 (chip latency, flagged in the spec as a live measurement) → Task 7 Step 5; criterion 6 (coverage gates) → Task 4 Step 7 and Task 6 Step 6. Known risks → Task 7.

**Type consistency.** `_Turn` fields (`text`, `tool_calls`, `finish_reason`) are used identically in Tasks 2 and 4. `tool_calls` entries are `{"id", "name", "arguments"}` in both `_stream_turn`'s output and `answer_stream`'s consumption. `_dispatch_tool(name, args, context)` has the same signature in Task 3's definition, Task 3's tests, and Task 4's call site. `_clamp_int` is keyword-only for `low`/`high`/`default` everywhere. `ChatContext.detail` is `dict[str, Any] | None` in Python and `Record<string, unknown> | undefined` in TypeScript, and the keys `buildChatContext` emits (`metadata`, `quality`, `view`, `detections`, `retrievals`) are exactly the keys `_tool_get_analysis` reads.

**Verified while writing, not assumed.** `i18n.tsx:1421` resolves a missing key to the key string
(`DICTS[lang][key] ?? en[key] ?? key`), so Task 6's tool label needs the explicit whitelist rather
than a falsy-fallback — the plan states this as fact, not as a choice. `tests/test_chat_endpoint.py`
imports `types`/`unittest`/`mock` but not `json`, so Task 3 adds it. Per-rep attribution
(`rep_index`, `occurred_reps`, `rep_count`) exists on the detection dataclass
(`pose_rule_detector.py:105-107`) and reaches the payload through `asdict` (`:664`), so spec success
criterion 1 is supported by the data — Task 3 puts those fields in `measured` and Task 7 asks the
criterion verbatim rather than a weaker substitute.

**Two coverage traps deliberately pre-empted**, because the 95% gate in Task 4 Step 7 would
otherwise stall on them. (1) `answer_stream`'s timeout branch has no "already streamed" case:
`streamed_any` is provably False at the top of every iteration, so such a case would be dead code
and a partial branch. (2) `_stream_completion`'s shape-tolerance `except` loses its only driver in
Task 1 — `data: not-json` moves to the raw layer — so Task 1 adds a test that patches
`_stream_raw_chunks` directly.

**Three async-timing traps in Task 6.** `send`'s `finally` clears `tool` and `streaming` as soon as
`chatStream` resolves, so a mock that resolves immediately makes the "shows the tool line" assertion
fail and makes a naive "clears the tool line" assertion pass vacuously. Task 6's mocks park on a
test-released promise and do not await `sendMessage`; only the `reset` test awaits it, because that
one asserts on committed content.

**Known soft spot.** Task 5's `stubFetchStream` helper is written defensively because the existing
`api.test.ts` may already define an equivalent under another name; the step says to reuse the file's
helper if one exists. This is a naming collision risk, not a correctness one.
