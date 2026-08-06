# Persistent Tool Records with Citations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the coach's tool records persist beside the answer they produced, and show which sources the retrieval actually returned.

**Architecture:** `_dispatch_tool` currently fuses run → prefix → serialise → truncate; it splits into `_run_tool` (raw result, never raises) + `_tool_sources` (pure extraction from the RAW result, before truncation) + `_dispatch_tool` returning a `_ToolResult(text, sources)`. The `tool` SSE frame gains a `sources` array. On the client the single transient `tool` state becomes an appended `toolRuns` list that survives the first `delta` and a `reset`, is attached to the assistant message on commit, and is persisted in `conversations.messages` (jsonb — no migration).

**Tech Stack:** FastAPI + Starlette SSE, Pydantic v2, `unittest.TestCase` under `tests/`; React 18 + TypeScript + vitest under `frontend/src/test/`.

**Spec:** `specs/llm-chat-spec.md`, section `# v3.1: Persistent tool records with citations`. Read it before Task 1.

## Global Constraints

- **Python is always `.venv\Scripts\python.exe`** from the repo root. There is no `python`/`pip` on PATH; a guard hook blocks bare invocations. Never `source .venv/bin/activate`.
- **Backend tests are always scoped to `tests/`**: `.venv\Scripts\python.exe -m pytest tests/`. Never bare `pytest`.
- **Backend coverage gate is 95%**: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`.
- **All yarn/vitest commands run with cwd = `frontend/`.** The Bash and PowerShell tools share one working directory; a stray `cd` to the repo root mass-fails vitest.
- **Do NOT run `yarn test:coverage`.** It is red on this machine for pre-existing reasons — vitest's default 5s `testTimeout` under coverage instrumentation, and `main` fails the same way. Use `yarn test`, which passes 90/90 cleanly. If it returns one to three `Test timed out in 5000ms` failures in components unrelated to chat/CoachTray, re-run those files in isolation to confirm and treat as environmental. Any chat/CoachTray failure is yours.
- No new dependencies.
- `backend/app/services/chat.py` carries a high explanatory comment density: every non-obvious guard has a *why*, not a *what*. Match it.
- Every user-facing string goes through `t()` in **both** the `en` and `zhHant` dictionaries in `frontend/src/lib/i18n.tsx`. A key in one only is a defect; `lib.i18n.test.ts` has a locale-parity test that will catch it.
- Tests are `unittest.TestCase` under `tests/` (backend) and vitest under `frontend/src/test/` (frontend).

### The one contract that must not regress

`_dispatch_tool` **never raises, for any input**. Its caller is a generator streaming SSE to a client whose HTTP 200 is already committed by Starlette *before the generator body runs*, so an escaping exception gives a dead stream instead of an `error` frame. This contract cost a full fix round to get right in v3, and the subtlety is that `json.dumps` is part of it: `default=str` does **not** rescue a non-string dict key (never consulted for keys), a circular structure (cycle detection fires first), or a value whose own `__str__` raises. Task 1 moves code around this contract — preserving it is the task's hard gate.

---

### Task 1: Split `_dispatch_tool` and extract sources from the raw result

**Files:**
- Modify: `backend/app/services/chat.py:698-756` (`_dispatch_tool`)
- Test: `tests/test_chat_endpoint.py` (`ToolDispatchTests`)

**Interfaces:**
- Consumes: `_tool_get_analysis`, `_clamp_int`, `_resolve_movement`, `kg_seeds_default`, `_REFERENCE_ONLY_PREFIX`, `_MAX_TOOL_RESULT_CHARS` (all existing).
- Produces:
  - `_MAX_TOOL_SOURCES: int = 5`
  - `_ToolResult` dataclass with fields `text: str`, `sources: list[dict[str, str]]`
  - `_run_tool(name: str, args: dict[str, Any], context: dict[str, Any]) -> Any`
  - `_tool_sources(name: str, result: Any) -> list[dict[str, str]]`
  - `_dispatch_tool(name: str, args: dict[str, Any], context: dict[str, Any]) -> _ToolResult`

- [ ] **Step 1: Write the failing tests**

Add to the existing `ToolDispatchTests` class in `tests/test_chat_endpoint.py`:

```python
    def test_rag_sources_use_reference_and_source_type(self) -> None:
        from backend.app.services import knowledge as knowledge_service

        payload = {
            "query": "ankle",
            "results": [
                {
                    "text": "…",
                    "metadata": {
                        "reference": "Wikipedia: Squat (exercise)",
                        "source_type": "encyclopedia",
                        "source": r"data\rag\docs\squat_wiki.txt",
                    },
                }
            ],
        }
        with mock.patch.object(knowledge_service, "rag_snippets", return_value=payload):
            out = chat_service._dispatch_tool("rag_search", {"query": "ankle"}, self._ctx())
        self.assertEqual(
            out.sources, [{"label": "Wikipedia: Squat (exercise)", "kind": "encyclopedia"}]
        )

    def test_rag_sources_never_leak_the_server_path(self) -> None:
        # metadata.source is a server filesystem path. It must not reach the client in ANY field,
        # including as the fallback label — the fallback is the BASENAME only.
        from backend.app.services import knowledge as knowledge_service

        payload = {
            "results": [
                {"text": "…", "metadata": {"source": r"data\rag\docs\squat_wiki.txt"}},
                {"text": "…", "metadata": {"source": "/srv/x-coach/data/rag/docs/ohp.pdf"}},
            ]
        }
        with mock.patch.object(knowledge_service, "rag_snippets", return_value=payload):
            out = chat_service._dispatch_tool("rag_search", {"query": "x"}, self._ctx())
        labels = [s["label"] for s in out.sources]
        self.assertEqual(labels, ["squat_wiki.txt", "ohp.pdf"])
        blob = json.dumps(out.sources)
        self.assertNotIn("data", blob.replace("squat_wiki", "").replace("ohp", ""))
        self.assertNotIn("srv", blob)

    def test_rag_sources_are_deduped_and_capped(self) -> None:
        from backend.app.services import knowledge as knowledge_service

        results = [
            {"metadata": {"reference": f"Doc {i // 3}", "source_type": "paper"}} for i in range(24)
        ]
        with mock.patch.object(
            knowledge_service, "rag_snippets", return_value={"results": results}
        ):
            out = chat_service._dispatch_tool("rag_search", {"query": "x"}, self._ctx())
        labels = [s["label"] for s in out.sources]
        self.assertEqual(len(labels), chat_service._MAX_TOOL_SOURCES)
        self.assertEqual(len(set(labels)), chat_service._MAX_TOOL_SOURCES)  # deduped
        self.assertEqual(labels[0], "Doc 0")  # first-seen order preserved

    def test_kg_sources_are_concepts_not_citations(self) -> None:
        # A KG node has no source field anywhere, so its `kind` is the literal "concept" — that is
        # what the renderer keys off to keep graph nodes out of the citation slot.
        from backend.app.services import knowledge as knowledge_service

        payload = {
            "matched_nodes": ["Squat:Insufficient Depth"],
            "subgraph": {
                "nodes": [
                    {"node_id": "Depth", "name": "Depth", "label": "QualityDimension"},
                    {"node_id": "Ankle", "name": "Ankle Mobility", "label": "Concept"},
                ],
                "edges": [],
            },
        }
        with mock.patch.object(knowledge_service, "graph_context", return_value=payload):
            out = chat_service._dispatch_tool("kg_query", {"query": "depth"}, self._ctx())
        self.assertEqual({s["kind"] for s in out.sources}, {"concept"})
        labels = [s["label"] for s in out.sources]
        self.assertIn("Insufficient Depth", labels)  # the "Squat:" prefix is stripped
        self.assertNotIn("Squat:Insufficient Depth", labels)
        self.assertIn("Ankle Mobility", labels)
        self.assertNotIn("QualityDimension", labels)  # internal taxonomy is not a source

    def test_get_analysis_reports_no_sources(self) -> None:
        out = chat_service._dispatch_tool("get_analysis", {"include": "all"}, self._ctx())
        self.assertEqual(out.sources, [])

    def test_a_failing_tool_reports_no_sources(self) -> None:
        from backend.app.services import knowledge as knowledge_service

        with mock.patch.object(
            knowledge_service, "graph_context", side_effect=FileNotFoundError("no graphml")
        ):
            out = chat_service._dispatch_tool("kg_query", {"query": "x"}, self._ctx())
        self.assertEqual(out.sources, [])
        self.assertIn("no graphml", out.text)

    def test_sources_survive_truncation_of_a_huge_result(self) -> None:
        # THE LOAD-BEARING TEST. Sources are derived from the RAW result, before truncation — a hit
        # big enough to be cut is exactly the one whose provenance matters most.
        from backend.app.services import knowledge as knowledge_service

        payload = {
            "results": [
                {
                    "text": "x" * 60_000,
                    "metadata": {"reference": "Big Review 2026", "source_type": "paper"},
                }
            ]
        }
        with mock.patch.object(knowledge_service, "rag_snippets", return_value=payload):
            out = chat_service._dispatch_tool("rag_search", {"query": "x"}, self._ctx())
        self.assertTrue(out.text.endswith("…[truncated]"))
        self.assertEqual(out.sources, [{"label": "Big Review 2026", "kind": "paper"}])

    def test_tool_sources_is_total_over_garbage(self) -> None:
        # _tool_sources runs inside the never-raises path, so it must survive any shape a tool or a
        # future provider could hand it.
        for junk in (None, 42, "text", [], {"results": "not-a-list"}, {"results": [None, 7]},
                     {"subgraph": None}, {"subgraph": {"nodes": "nope"}}, {"matched_nodes": 5}):
            for name in ("rag_search", "kg_query", "get_analysis"):
                self.assertIsInstance(chat_service._tool_sources(name, junk), list)
```

Then update **every existing** `_dispatch_tool` call site in the file (21 of them) to use `.text`. They currently read like `json.loads(chat_service._dispatch_tool(...))` or `out = chat_service._dispatch_tool(...)`; they become `json.loads(chat_service._dispatch_tool(...).text)` and `out = chat_service._dispatch_tool(...).text`. Change nothing else about what they assert.

- [ ] **Step 2: Run the tests to verify they fail**

```
.venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py::ToolDispatchTests -v
```

Expected: FAIL — `AttributeError: 'str' object has no attribute 'sources'` / no `_tool_sources`.

- [ ] **Step 3: Implement the split**

In `backend/app/services/chat.py`, replace the whole of `_dispatch_tool` (currently lines 698-756) with:

```python
# One document yields many RAG chunks and a KG node can be both matched and 1-hop, so the list is
# deduped; the cap then keeps a wide retrieval from turning the tray into a bibliography.
_MAX_TOOL_SOURCES = 5


@dataclass
class _ToolResult:
    """One tool call's outcome: what the MODEL reads, and what the USER is shown.

    ``text`` is the ``role:"tool"`` message content — prefixed and truncated. ``sources`` is the
    provenance the client renders, derived from the RAW result *before* truncation, because a hit
    big enough to be cut is exactly the one whose citations matter most.
    """

    text: str
    sources: list[dict[str, str]]


def _run_tool(name: str, args: dict[str, Any], context: dict[str, Any]) -> Any:
    """Run one tool call and return its RAW, unserialised result. NEVER RAISES.

    A failing tool (a missing KG graphml, an unbuilt RAG vector db) or a hallucinated tool name
    becomes an ``{"error": ...}`` payload the model can read and account for out loud. Letting it
    propagate would kill an answer stream that was otherwise fine, and the HTTP 200 is already
    committed by then, so the user would just see it die.
    """
    from backend.app.services import knowledge  # deferred: the KG/RAG import chain is heavy.

    try:
        if name == "get_analysis":
            return _tool_get_analysis(context.get("detail") or {}, args)
        if name == "kg_query":
            return knowledge.graph_context(
                str(args.get("query") or ""),
                hops=_clamp_int(args.get("hops"), low=1, high=2, default=1),
                max_seeds=kg_seeds_default(),
                # Forced to the thread's movement: without it the KG happily returns knowledge for a
                # different exercise, which the coach would then present as relevant to this clip.
                movement=_resolve_movement(context),
            )
        if name == "rag_search":
            return knowledge.rag_snippets(
                str(args.get("query") or ""),
                top_k=_clamp_int(args.get("top_k"), low=1, high=8, default=5),
            )
        return {"error": f"Unknown tool {name!r}."}
    except Exception as exc:  # noqa: BLE001 — a tool failure must never kill the answer stream.
        return {"error": f"{name} failed: {exc}"}


def _dedupe_cap(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop repeat labels (first-seen wins) and cap the list. Order is preserved deliberately —
    it is retrieval rank, which is the most useful order to show."""
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for s in sources:
        if s["label"] in seen:
            continue
        seen.add(s["label"])
        out.append(s)
        if len(out) >= _MAX_TOOL_SOURCES:
            break
    return out


def _tool_sources(name: str, result: Any) -> list[dict[str, str]]:
    """The provenance to show the user for one tool call, as ``[{"label", "kind"}]``.

    TOTAL BY CONSTRUCTION — it runs inside the never-raises path, so every access is guarded and any
    shape it does not recognise yields ``[]`` rather than an exception.

    THE THREE TOOLS DO NOT HAVE COMPARABLE PROVENANCE, and this function is where that is enforced
    (spec v3.1 section 1):

    * ``rag_search`` returns real literature, so ``kind`` carries the corpus's own ``source_type``.
    * ``kg_query`` returns GRAPH NODES, which carry no source field anywhere in the subgraph — only
      ``node_id``/``name``/``label``. Its ``kind`` is therefore the literal ``"concept"``, which is
      what the renderer keys off to keep a graph node out of the citation slot. Rendering one beside
      a cited paper would tell the user a concept is a source: exactly the false authority this
      feature exists to prevent.
    * ``get_analysis`` reads the user's own analysis. There is no outside source to credit, so it
      reports none and the client omits the block.

    ``metadata["source"]`` is NEVER emitted: it is a server filesystem path
    (``data\\rag\\docs\\squat_wiki.txt``), useless to a user and a gratuitous internals leak. It is
    used only as a last-resort label, and then only its basename.
    """
    if not isinstance(result, dict) or result.get("error"):
        return []  # a failed tool retrieved nothing, so it cites nothing.

    if name == "rag_search":
        out: list[dict[str, str]] = []
        for hit in result.get("results") or []:
            if not isinstance(hit, dict):
                continue
            meta = hit.get("metadata")
            if not isinstance(meta, dict):
                continue
            label = str(meta.get("reference") or "").strip()
            if not label:
                # Basename only — never the directories. Split on both separators: the corpus is
                # indexed on Windows, so stored paths use backslashes even on a POSIX host.
                raw = str(meta.get("source") or "").strip()
                label = raw.replace("\\", "/").rstrip("/").split("/")[-1]
            if label:
                out.append({"label": label, "kind": str(meta.get("source_type") or "document")})
        return _dedupe_cap(out)

    if name == "kg_query":
        names: list[str] = []
        for node in result.get("matched_nodes") or []:
            names.append(str(node))
        subgraph = result.get("subgraph")
        if isinstance(subgraph, dict):
            for node in subgraph.get("nodes") or []:
                if isinstance(node, dict) and node.get("name"):
                    names.append(str(node["name"]))
        # Matched nodes are stored movement-qualified ("Squat:Insufficient Depth"); the movement is
        # already established by the thread, so showing it again is noise.
        return _dedupe_cap(
            [{"label": n.split(":", 1)[-1].strip(), "kind": "concept"} for n in names if n.strip()]
        )

    return []  # get_analysis, and any tool added later without its own extractor.


def _dispatch_tool(name: str, args: dict[str, Any], context: dict[str, Any]) -> _ToolResult:
    """Run one tool call; return what the model reads plus what the user is shown.

    NOTHING RAISES OUT OF HERE, deliberately — see ``_run_tool``. THE SERIALISATION IS PART OF THAT
    GUARANTEE, which is easy to miss: ``default=str`` does not make ``json.dumps`` safe. A
    non-string dict key skips ``default`` entirely and raises ``TypeError``; a circular structure is
    caught by json's own cycle detector before ``default`` is ever consulted, raising ``ValueError``;
    and ``default=str`` calling a value's own broken ``__str__`` propagates whatever THAT raises. So
    it gets its own guard, whose fallback is a FRESH dict with a literal string key and a value
    built from the exception's CLASS NAME — not ``str(exc)``, because ``BaseException.__str__``
    returns ``str(args[0])`` for a single argument, so an exception carrying the very object whose
    ``__str__`` just failed would re-raise here, one level deeper.

    ``kg_query``/``rag_search`` results get ``_REFERENCE_ONLY_PREFIX`` stamped on before truncation
    (never ``get_analysis`` — see that constant's docstring), so the honesty rule lives next to the
    data itself, not only in one system-prompt sentence a model could pay less attention to.
    """
    result = _run_tool(name, args, context)  # cannot raise
    sources = _tool_sources(name, result)  # total by construction; derived BEFORE truncation

    try:
        text = json.dumps(result, ensure_ascii=False, default=str)
    except Exception as exc:  # noqa: BLE001 — see the docstring: default=str is not a safety net.
        text = json.dumps(
            {"error": f"{name} produced a result that could not be serialised ({type(exc).__name__})."},
            ensure_ascii=False,
        )

    if name in ("kg_query", "rag_search"):
        # The marker lives on EVERY result from these two tools, success or error — an error payload
        # ("no graphml found") is still general-knowledge-shaped, not an observation about this
        # video, so it gets the same honesty prefix. Applied here, before the truncation below, so
        # the prefix is never the part that gets cut off a long result.
        text = _REFERENCE_ONLY_PREFIX + text

    if len(text) > _MAX_TOOL_RESULT_CHARS:  # a plain str op — cannot raise.
        text = text[:_MAX_TOOL_RESULT_CHARS] + "…[truncated]"
    return _ToolResult(text=text, sources=sources)
```

`dataclass` is already imported at the top of the file (Task 2 of the v3 plan added it for `_Turn`). Verify rather than assume.

- [ ] **Step 4: Update `answer_stream`'s call site**

`answer_stream` (inside `_answer_stream_inner`) currently does
`"content": _dispatch_tool(call["name"], args, context)`. It must become
`_dispatch_tool(...).text` for now — Task 2 uses the `sources` half.

- [ ] **Step 5: Run the tests to verify they pass**

```
.venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py -v
```

Expected: PASS, including every pre-existing test.

- [ ] **Step 6: Prove the never-raises contract survived**

The three v3 tests that pin it — the non-string dict key, the circular reference, and the raising
`__str__` — are in `ToolDispatchTests` and must still pass **unedited apart from the `.text`
suffix**. Additionally, confirm by inspection that no statement in `_dispatch_tool` sits outside a
guard: `_run_tool` has its own try, `_tool_sources` is total, `json.dumps` has its own try, and the
prefix/truncation are plain string operations.

- [ ] **Step 7: Run the full suite and the coverage gate**

```
.venv\Scripts\python.exe -m pytest tests/
.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95
```

- [ ] **Step 8: Commit**

```bash
git add backend/app/services/chat.py tests/test_chat_endpoint.py
git commit -m "refactor: split _dispatch_tool and extract tool sources

_run_tool returns the raw result and never raises; _tool_sources derives the
provenance from that raw result BEFORE truncation, so a hit large enough to
be cut still reports its citations; _dispatch_tool returns a _ToolResult
carrying both. The never-raises contract is unchanged and still covers
serialisation, with the fallback now built from the exception CLASS NAME
rather than str(exc), which could itself raise for an exception carrying a
broken __str__.

rag_search maps reference/source_type; kg_query emits kind 'concept'
literally, because KG nodes carry no source field at all and must not be
rendered as citations; get_analysis reports none."
```

---

### Task 2: Put `sources` on the `tool` SSE frame

**Files:**
- Modify: `backend/app/services/chat.py` (`_answer_stream_inner`, the tool-frame yield)
- Test: `tests/test_chat_endpoint.py` (`ToolLoopTests`)

**Interfaces:**
- Consumes: `_ToolResult`, `_dispatch_tool` from Task 1.
- Produces: the `tool` frame shape `{"name", "query", "sources"?}` — `sources` **omitted** when empty.

- [ ] **Step 1: Write the failing tests**

Add to `ToolLoopTests` in `tests/test_chat_endpoint.py`:

```python
    def test_the_tool_frame_carries_sources(self) -> None:
        fake, _ = self._turns(
            ([], [{"id": "c1", "name": "rag_search", "arguments": '{"query": "ankle"}'}]),
            (["Answer"], []),
        )
        result = chat_service._ToolResult(
            text='{"ok": true}',
            sources=[{"label": "Wikipedia: Squat (exercise)", "kind": "encyclopedia"}],
        )
        events = self._run(fake, dispatch=lambda n, a, c: result)
        tool_frames = [d for e, d in events if e == "tool"]
        self.assertEqual(
            tool_frames,
            [
                {
                    "name": "rag_search",
                    "query": "ankle",
                    "sources": [{"label": "Wikipedia: Squat (exercise)", "kind": "encyclopedia"}],
                }
            ],
        )

    def test_the_tool_frame_omits_sources_when_there_are_none(self) -> None:
        # get_analysis has no outside source to credit; the key is absent, not an empty array, so a
        # client can tell "nothing to cite" from "cited nothing".
        fake, _ = self._turns(
            ([], [{"id": "c1", "name": "get_analysis", "arguments": '{"include": "all"}'}]),
            (["Answer"], []),
        )
        result = chat_service._ToolResult(text='{"ok": true}', sources=[])
        events = self._run(fake, dispatch=lambda n, a, c: result)
        tool_frames = [d for e, d in events if e == "tool"]
        self.assertEqual(tool_frames, [{"name": "get_analysis", "query": ""}])

    def test_the_tool_message_content_is_the_result_text(self) -> None:
        # Regression lock: the model must receive `.text`, never a repr of the _ToolResult.
        fake, calls = self._turns(
            ([], [{"id": "c1", "name": "kg_query", "arguments": "{}"}]),
            (["A"], []),
        )
        result = chat_service._ToolResult(text='{"marker": 1}', sources=[{"label": "L", "kind": "concept"}])
        self._run(fake, dispatch=lambda n, a, c: result)
        self.assertEqual(calls[1]["messages"][-1]["content"], '{"marker": 1}')
```

`ToolLoopTests._run` currently defaults `dispatch` to `lambda n, a, c: '{"ok": true}'` (a `str`).
Change that default to `lambda n, a, c: chat_service._ToolResult(text='{"ok": true}', sources=[])`,
and leave every existing assertion alone.

- [ ] **Step 2: Run the tests to verify they fail**

```
.venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py::ToolLoopTests -v
```

Expected: FAIL — the frame has no `sources` key.

- [ ] **Step 3: Emit the sources**

In `_answer_stream_inner`, the tool loop currently reads:

```python
        for i, call in enumerate(turn.tool_calls):
            args = _parse_tool_args(call["arguments"])
            yield _sse("tool", {"name": call["name"], "query": _tool_query_label(call["name"], args)})
            convo.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"] or f"call_{i}",
                    "content": _dispatch_tool(call["name"], args, context).text,
                }
            )
```

Replace it with:

```python
        for i, call in enumerate(turn.tool_calls):
            args = _parse_tool_args(call["arguments"])
            # Dispatch BEFORE the frame: the frame carries the sources, which only exist once the
            # tool has run. The cost is that the tray's status line appears after the retrieval
            # rather than before it — acceptable, because the sources are the point of the frame,
            # and the loop already blocks on this call either way.
            outcome = _dispatch_tool(call["name"], args, context)
            frame: dict[str, Any] = {
                "name": call["name"],
                "query": _tool_query_label(call["name"], args),
            }
            if outcome.sources:
                # Omitted rather than [] so a client can tell "this tool has nothing to cite"
                # (get_analysis) from "this tool cited nothing".
                frame["sources"] = outcome.sources
            yield _sse("tool", frame)
            convo.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"] or f"call_{i}",
                    "content": outcome.text,
                }
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

```
.venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py -v
```

- [ ] **Step 5: Run the full suite and the coverage gate**

```
.venv\Scripts\python.exe -m pytest tests/
.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/chat.py tests/test_chat_endpoint.py
git commit -m "feat: carry tool sources on the tool SSE frame

The frame gains an optional sources array, omitted rather than empty so a
client can tell 'nothing to cite' (get_analysis) from 'cited nothing'. The
dispatch now happens before the frame is yielded, since the sources only
exist once the tool has run; the loop already blocked on that call."
```

---

### Task 3: Accept and persist `tools` on a conversation message

**Files:**
- Modify: `backend/app/routers/conversations.py:22-24` (`ConversationMessage`)
- Test: `tests/test_chat_endpoint.py` or the existing conversations test module — put the test beside the other conversation-router tests; find them with `grep -rn "ConversationMessage\|putConversation\|/api/conversations" tests/`.

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ConversationMessage.tools: list[ToolRun]`, where `ToolRun` is `{name: str, query: str = "", sources: list[ToolSource] = []}` and `ToolSource` is `{label: str, kind: str}`.

- [ ] **Step 1: Write the failing test**

```python
    def test_conversation_message_accepts_and_round_trips_tools(self) -> None:
        from backend.app.routers.conversations import ConversationMessage

        m = ConversationMessage(
            role="assistant",
            content="…",
            tools=[
                {
                    "name": "rag_search",
                    "query": "ankle",
                    "sources": [{"label": "Wikipedia: Squat (exercise)", "kind": "encyclopedia"}],
                }
            ],
        )
        dumped = m.model_dump()
        self.assertEqual(dumped["tools"][0]["name"], "rag_search")
        self.assertEqual(dumped["tools"][0]["sources"][0]["kind"], "encyclopedia")
        # Absent is the common case (every user turn, and every pre-v3.1 stored row).
        self.assertEqual(ConversationMessage(role="user", content="hi").model_dump()["tools"], [])
```

- [ ] **Step 2: Run the test to verify it fails**

```
.venv\Scripts\python.exe -m pytest tests/ -k conversation_message_accepts -v
```

Expected: FAIL — `tools` is not a field, so Pydantic drops it and the lookup raises `KeyError`.

- [ ] **Step 3: Add the models**

In `backend/app/routers/conversations.py`, above `ConversationMessage`:

```python
class ToolSource(BaseModel):
    """One provenance entry the tray shows under a tool call.

    ``kind`` is a corpus ``source_type`` for ``rag_search`` but the literal ``"concept"`` for
    ``kg_query``, whose knowledge-graph nodes carry no source field at all — the client keys off it
    to keep a graph concept out of the citation slot (spec v3.1 section 1).
    """

    label: str
    kind: str


class ToolRun(BaseModel):
    name: str
    query: str = ""
    sources: list[ToolSource] = Field(default_factory=list)
```

and extend `ConversationMessage`:

```python
class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)
    # The tool calls that produced this answer, persisted so a reload or a history replay restores
    # the answer's provenance and not just its text. Empty for every user turn and for any row
    # written before v3.1. `conversations.messages` is jsonb, so this needed no migration.
    tools: list[ToolRun] = Field(default_factory=list)
```

- [ ] **Step 4: Run the test to verify it passes**

```
.venv\Scripts\python.exe -m pytest tests/ -k conversation_message_accepts -v
```

- [ ] **Step 5: Run the full suite**

```
.venv\Scripts\python.exe -m pytest tests/
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/routers/conversations.py tests/
git commit -m "feat: persist tool records on a conversation message

ConversationMessage gains an optional tools array so a reload or history
replay restores an answer's provenance, not just its text. conversations.
messages is jsonb, so no migration was needed. /api/chat's own ChatMessage is
deliberately left as {role, content}, which means Pydantic drops the records
before they could ever reach the LLM prompt."
```

---

### Task 4: Client types, `onTool` signature, and stripping `tools` on send

**Files:**
- Modify: `frontend/src/api.ts` — `ChatMessage`, `ChatStreamHandlers`, `dispatchSSE`, `chatStream`, `chatFollowups`
- Test: `frontend/src/test/api.test.ts`

**Interfaces:**
- Consumes: the `tool` frame shape from Task 2.
- Produces:
  - `ToolSource` = `{ label: string; kind: string }`
  - `ToolRun` = `{ name: string; query: string; sources?: ToolSource[] }`
  - `ChatMessage.tools?: ToolRun[]`
  - `ChatStreamHandlers.onTool?: (name: string, query: string, sources: ToolSource[]) => void`

- [ ] **Step 1: Write the failing tests**

Add to `frontend/src/test/api.test.ts`, reusing the file's existing `mockStream` helper:

```ts
  it("passes tool sources to onTool, and an empty array when the frame omits them", async () => {
    const seen: Array<[string, string, unknown]> = [];
    mockStream([
      'event: tool\ndata: {"name":"rag_search","query":"ankle","sources":[{"label":"Wikipedia: Squat (exercise)","kind":"encyclopedia"}]}\n\n',
      'event: tool\ndata: {"name":"get_analysis","query":"Depth"}\n\n',
      'event: delta\ndata: {"text":"A"}\n\n',
      'event: done\ndata: {"model":"m"}\n\n',
    ]);
    await api.chatStream([{ role: "user", content: "hi" }], { fault_count: 0, quality: {}, faults: [] }, {
      onDelta: () => undefined,
      onDone: () => undefined,
      onError: () => undefined,
      onTool: (n, q, s) => seen.push([n, q, s]),
    });
    expect(seen).toEqual([
      ["rag_search", "ankle", [{ label: "Wikipedia: Squat (exercise)", kind: "encyclopedia" }]],
      ["get_analysis", "Depth", []],
    ]);
  });

  it("strips tools from messages on both chat endpoints", async () => {
    const thread = [
      { role: "user" as const, content: "hi" },
      {
        role: "assistant" as const,
        content: "answer",
        tools: [{ name: "rag_search", query: "ankle", sources: [{ label: "L", kind: "paper" }] }],
      },
    ];
    const ctx = { fault_count: 0, quality: {}, faults: [] };

    const followupFetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ questions: [] }) });
    vi.stubGlobal("fetch", followupFetch);
    await api.chatFollowups(thread, ctx);
    const followupBody = JSON.parse(followupFetch.mock.calls[0][1].body);
    expect(followupBody.messages[1].tools).toBeUndefined();
    expect(followupBody.messages[1].content).toBe("answer");

    mockStream(['event: done\ndata: {"model":"m"}\n\n']);
    await api.chatStream(thread, ctx, {
      onDelta: () => undefined,
      onDone: () => undefined,
      onError: () => undefined,
    });
    const chatBody = JSON.parse((globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1].body);
    expect(chatBody.messages[1].tools).toBeUndefined();
  });
```

`mockStream` is at the top of that file and its signature is `mockStream(chunks: string[])` — it
takes an array of frame strings, which is what the tests above pass. It returns the `vi.spyOn` on
`globalThis.fetch`, so the second test can read `fetch.mock.calls` directly.

- [ ] **Step 2: Run the tests to verify they fail**

With cwd = `frontend/`:

```
yarn test src/test/api.test.ts
```

Expected: FAIL — `onTool` receives two arguments, and `tools` is sent verbatim.

- [ ] **Step 3: Add the types**

In `frontend/src/api.ts`, beside the other chat types:

```ts
// One provenance entry under a tool call. `kind` is a corpus source_type for rag_search but the
// literal "concept" for kg_query, whose knowledge-graph nodes carry no source field at all — the
// renderer keys off it so a graph concept is never shown as a literature citation.
export interface ToolSource {
  label: string;
  kind: string;
}

// One tool call the coach made while answering. `sources` is absent when the tool has nothing to
// cite (get_analysis reads the user's own analysis), which is distinct from citing nothing.
export interface ToolRun {
  name: string;
  query: string;
  sources?: ToolSource[];
}
```

Extend `ChatMessage`:

```ts
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  // The tool calls that produced this answer. Rendered above the message and persisted with it, so
  // a reload restores the answer's provenance. Stripped before either chat endpoint is called — the
  // LLM has no use for it and it would be re-uploaded every turn.
  tools?: ToolRun[];
}
```

Extend `ChatStreamHandlers.onTool` and widen `dispatchSSE`'s local `data` type:

```ts
  onTool?: (name: string, query: string, sources: ToolSource[]) => void;
```

```ts
  let data: {
    text?: string;
    model?: string;
    detail?: string;
    name?: string;
    query?: string;
    sources?: ToolSource[];
  };
```

```ts
  else if (event === "tool") handlers.onTool?.(data.name ?? "", data.query ?? "", data.sources ?? []);
```

- [ ] **Step 4: Strip `tools` before sending**

Add near the other chat helpers in `frontend/src/api.ts`:

```ts
// Both chat endpoints get the conversation with `tools` removed. The records are a rendering and
// persistence concern only: the backend's ChatMessage is {role, content}, so Pydantic would drop
// them anyway — but relying on implicit stripping still re-uploads the whole array every turn, and
// on a multi-tool thread that is not small.
function leanMessages(messages: ChatMessage[]): Array<{ role: string; content: string }> {
  return messages.map(({ role, content }) => ({ role, content }));
}
```

Use it in **both** `chatStream` and `chatFollowups`, replacing `messages` in the JSON body with
`leanMessages(messages)`. In `chatFollowups` this sits alongside the existing `detail` strip.

- [ ] **Step 5: Run the tests to verify they pass**

With cwd = `frontend/`:

```
yarn test src/test/api.test.ts
```

- [ ] **Step 6: Run the frontend suite and the build**

With cwd = `frontend/`:

```
yarn test
yarn build
```

- [ ] **Step 7: Commit**

```bash
git add frontend/src/api.ts frontend/src/test/api.test.ts
git commit -m "feat: client types for tool records and their sources

onTool now receives the sources array (empty when the frame omits it), and
both chat endpoints send the conversation with `tools` stripped -- the
backend's ChatMessage is {role, content} so Pydantic would drop them anyway,
but relying on that still re-uploaded the whole array every turn."
```

---

### Task 5: Render persistent tool records in the tray

**Files:**
- Create: `frontend/src/components/ToolRunList.tsx`
- Modify: `frontend/src/components/CoachTray.tsx` (state, `send`, render), `frontend/src/lib/i18n.tsx`
- Test: `frontend/src/test/components.CoachTray.chat.test.tsx`

**Interfaces:**
- Consumes: `ToolRun`, `ToolSource`, `ChatMessage.tools`, `onTool(name, query, sources)` from Task 4.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing tests**

Add to `frontend/src/test/components.CoachTray.chat.test.tsx`. Note the timing rule the file already
follows: `send`'s `finally` clears transient state, so a test asserting on **live** state must park
the mock on a promise and NOT await `sendMessage`; tests asserting on **committed** state may await.

```tsx
  it("keeps tool records visible after the answer starts streaming", async () => {
    vi.spyOn(api, "chatStream").mockImplementation(async (_m, _c, h) => {
      h.onTool?.("rag_search", "zzq-ankle-subject", [{ label: "zzq-Wiki-Source", kind: "encyclopedia" }]);
      h.onDelta("Real answer");
      h.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    // Committed, so this survives the finally — the whole point of v3.1.
    expect(await screen.findByText("Real answer")).toBeTruthy();
    expect(screen.getByText(/zzq-ankle-subject/)).toBeTruthy();
    expect(screen.getByText("zzq-Wiki-Source")).toBeTruthy();
  });

  it("appends successive tool calls instead of replacing them", async () => {
    vi.spyOn(api, "chatStream").mockImplementation(async (_m, _c, h) => {
      h.onTool?.("kg_query", "zzq-first-subject", []);
      h.onTool?.("rag_search", "zzq-second-subject", []);
      h.onDelta("A");
      h.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    expect(await screen.findByText("A")).toBeTruthy();
    expect(screen.getByText(/zzq-first-subject/)).toBeTruthy();
    expect(screen.getByText(/zzq-second-subject/)).toBeTruthy();
  });

  it("keeps tool records when the server retracts narration with reset", async () => {
    // reset retracts the model's narration, but the tool calls really happened and really fed the
    // answer — erasing them would misreport the reasoning chain.
    vi.spyOn(api, "chatStream").mockImplementation(async (_m, _c, h) => {
      h.onDelta("zzq-narration");
      h.onReset?.();
      h.onTool?.("rag_search", "zzq-kept-subject", []);
      h.onDelta("Real answer");
      h.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    expect(await screen.findByText("Real answer")).toBeTruthy();
    expect(screen.queryByText(/zzq-narration/)).toBeNull();
    expect(screen.getByText(/zzq-kept-subject/)).toBeTruthy();
  });

  it("persists tool records with the committed turn", async () => {
    const put = vi.spyOn(api, "putConversation").mockResolvedValue(undefined as never);
    vi.spyOn(api, "chatStream").mockImplementation(async (_m, _c, h) => {
      h.onTool?.("rag_search", "ankle", [{ label: "zzq-Persisted-Source", kind: "paper" }]);
      h.onDelta("A");
      h.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    await screen.findByText("A");
    const thread = put.mock.calls[0][1] as Array<{ role: string; tools?: unknown[] }>;
    expect(thread[thread.length - 1].tools).toEqual([
      { name: "rag_search", query: "ankle", sources: [{ label: "zzq-Persisted-Source", kind: "paper" }] },
    ]);
  });

  it("restores tool records from a stored conversation", async () => {
    vi.spyOn(api, "getConversation").mockResolvedValue({
      video_id: "v1",
      messages: [
        { role: "user", content: "why?" },
        {
          role: "assistant",
          content: "stored answer",
          tools: [{ name: "kg_query", query: "zzq-restored-subject", sources: [] }],
        },
      ],
      followups: [],
    } as never);
    renderTray();
    expect(await screen.findByText("stored answer")).toBeTruthy();
    expect(screen.getByText(/zzq-restored-subject/)).toBeTruthy();
  });
```

The `zzq-` prefixes are deliberate: this file's fixture renders real analysis text (`keyEvidence()`
puts strings like "knee valgus ratio 0.82" on the page), and a previous round of this work shipped
tests that passed spuriously by matching that fixture text instead of the tool line. Distinctive
subjects make the assertions mean what they say.

`api.getConversation(videoId)` resolves to `Conversation`, which is
`{ video_id: string; messages: ChatMessage[]; followups?: string[] }` — so once Task 4 adds
`tools?` to `ChatMessage`, stored records deserialise with no further change and the `as never` cast
in that last test can be dropped.

- [ ] **Step 2: Run the tests to verify they fail**

With cwd = `frontend/`:

```
yarn test src/test/components.CoachTray.chat.test.tsx
```

Expected: FAIL — records vanish on the first delta, are replaced rather than appended, and are not persisted.

- [ ] **Step 3: Add the i18n strings**

In `frontend/src/lib/i18n.tsx`, beside the existing `chat.tool.*` keys in the **`en`** dictionary:

```ts
  "chat.tool.sources": "Sources",
  "chat.tool.concepts": "Knowledge-graph concepts",
```

and beside their counterparts in the **`zhHant`** dictionary:

```ts
  "chat.tool.sources": "引用來源",
  "chat.tool.concepts": "知識圖譜概念",
```

- [ ] **Step 4: Create the render component**

Create `frontend/src/components/ToolRunList.tsx`:

```tsx
import { useI18n } from "../lib/i18n";
import type { ToolRun } from "../api";

// The tools we have i18n labels for. A name outside this list falls back to the generic label —
// `t()` returns the key itself on a miss (i18n.tsx), so an unguarded lookup would render a raw key
// like "chat.tool.something_else" into the tray.
const TOOL_LABEL_KEYS = ["get_analysis", "kg_query", "rag_search"] as const;

/** The tool calls behind one answer, in call order, above the answer they produced.
 *
 * Used twice: for a committed assistant message (from `message.tools`) and for the turn currently
 * streaming (from live state). Same markup both times, so nothing shifts when the turn commits.
 *
 * `kg_query`'s entries are headed differently from the other tools' on purpose. Its "sources" are
 * knowledge-graph concepts, which carry no citation anywhere in the graph; showing them under the
 * same heading as a retrieved document would tell the user a concept is a source.
 */
export function ToolRunList({ runs }: { runs: ToolRun[] }) {
  const { t } = useI18n();
  if (!runs.length) return null;
  return (
    <div className="flex flex-col gap-2">
      {runs.map((run, i) => {
        const known = TOOL_LABEL_KEYS.includes(run.name as (typeof TOOL_LABEL_KEYS)[number]);
        const heading = run.name === "kg_query" ? t("chat.tool.concepts") : t("chat.tool.sources");
        return (
          <div key={i} className="text-xs text-muted">
            <div>
              {known ? t(`chat.tool.${run.name}` as never) : t("chat.tool.generic")}
              {run.query ? `${t("chat.tool.sep")}${run.query}` : ""}
            </div>
            {run.sources && run.sources.length > 0 && (
              <div className="mt-1 pl-3 flex flex-col gap-0.5">
                <div className="text-faint">{heading}</div>
                {run.sources.map((s, j) => (
                  <div key={j} className="text-faint">
                    {s.label}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
```

Check the surrounding components for the real token names (`text-muted`, `text-faint`) and match
them — do not invent classes.

- [ ] **Step 5: Rewire `CoachTray` state**

Replace the `tool` state (around line 65) with:

```tsx
  // Every tool call this turn, in order. Unlike v3's single transient line these are NOT cleared
  // when the answer starts: the record is the answer's provenance and belongs beside it.
  const [toolRuns, setToolRuns] = useState<ToolRun[]>([]);
```

In `send`, replace `setTool(null)` at the top with `setToolRuns([])`, and rewire the handlers.
**The local `runs` accumulator matters:** `toolRuns` state is captured stale by this closure, exactly
as `acc` is for the streamed text, so the commit must read the local variable, not the state.

```tsx
    let acc = "";
    let runs: ToolRun[] = [];
```

```tsx
          onDelta: (tkn) => {
            acc += tkn;
            setStreaming(acc);
            // NOTE: unlike v3, the tool records are deliberately NOT cleared here.
          },
          onDone: () => {
            finished = true;
          },
          onTool: (name, query, sources) => {
            runs = [...runs, { name, query, ...(sources.length ? { sources } : {}) }];
            setToolRuns(runs);
          },
          // The round that produced this text also called a tool, so it was narration, not the
          // answer. Drop the text — but NOT `runs`: those calls really happened and really fed the
          // answer, so erasing them would misreport the reasoning chain.
          onReset: () => {
            acc = "";
            setStreaming("");
          },
```

Attach the records to the committed turn:

```tsx
      const thread: ChatMessage[] = [
        ...next,
        { role: "assistant", content: acc, ...(runs.length ? { tools: runs } : {}) },
      ];
```

In `finally`, replace `setTool(null)` with `setToolRuns([])` — the committed message now owns them,
so leaving the live copy would render them twice.

- [ ] **Step 6: Render**

Import `ToolRunList` and `ToolRun`. In the committed-message map, render each assistant message's
records above its content:

```tsx
                    {m.role === "assistant" && m.tools && m.tools.length > 0 && (
                      <ToolRunList runs={m.tools} />
                    )}
```

For the in-flight turn, replace the old single-line tool block with:

```tsx
                {toolRuns.length > 0 && <ToolRunList runs={toolRuns} />}
                {/* Lumen's dots only until the first token lands; then the streaming text carries it. */}
                {loading && !streaming && toolRuns.length === 0 && (
                  <div className="flex items-center gap-2 text-xs text-muted">
                    <LumenLoader variant="dots" />
                    {t("chat.thinking")}
                  </div>
                )}
```

The `TOOL_LABEL_KEYS` constant and its comment move to `ToolRunList.tsx` — delete them from
`CoachTray.tsx` so there is one copy.

- [ ] **Step 7: Run the tests to verify they pass**

With cwd = `frontend/`:

```
yarn test src/test/components.CoachTray.chat.test.tsx
```

- [ ] **Step 8: Confirm the tests have teeth**

Three of these tests exist because the behaviour they check is easy to get wrong in a way that still
looks fine. Break each line, watch the test fail, restore it:

1. Re-add `setToolRuns([])` to `onDelta` → "keeps tool records visible after the answer starts streaming" must FAIL.
2. Change `onTool` to `setToolRuns([{ name, query }])` (replace rather than append) → "appends successive tool calls" must FAIL.
3. Clear `runs` inside `onReset` → "keeps tool records when the server retracts narration" must FAIL.

- [ ] **Step 9: Run the frontend suite and the build**

With cwd = `frontend/`:

```
yarn test
yarn build
```

- [ ] **Step 10: Commit**

```bash
git add frontend/src/components/ToolRunList.tsx frontend/src/components/CoachTray.tsx frontend/src/lib/i18n.tsx frontend/src/test/
git commit -m "feat: keep tool records beside the answer, with their sources

The transient single-line status becomes an appended list that survives the
first delta and a reset, is attached to the assistant message on commit, and
is restored from the stored conversation. A reset retracts the narration but
keeps the records: those calls really happened and really fed the answer.

kg_query's entries are headed 'knowledge-graph concepts' rather than
'sources', because KG nodes carry no citation anywhere in the graph and
rendering them beside a retrieved document would present a concept as a
source."
```

---

## Self-Review

**Spec coverage.** v3.1 §1 (the three tools' provenance differ) → Task 1's `_tool_sources` plus
Task 5's per-tool heading. §2 (extraction before truncation, narrow wire shape, no server path) →
Task 1, with `test_sources_survive_truncation_of_a_huge_result` as the load-bearing regression. §3
(SSE contract) → Task 2. §4 (client state; `reset` keeps records) → Task 5, teeth-checked. §5
(persistence, no migration) → Task 3 (backend model) + Task 5 (attach on commit) + Task 4 (strip on
send). §6 (rendering, one component two callers) → Task 5. §7 (testing) → tests in every task.
Success criteria 1-5 → Task 5's five tests; criterion 6 (coverage gates) → Task 1 Step 7 and Task 2
Step 5 for the backend, Task 5 Step 9 for the frontend.

**Type consistency.** `_ToolResult(text, sources)` is used identically in Tasks 1 and 2. The wire
shape `{label, kind}` is the same in `_tool_sources` (Task 1), the frame (Task 2), `ToolSource`
(Tasks 3 and 4) and the renderer (Task 5). `onTool(name, query, sources)` has three parameters in
Task 4's definition, Task 4's tests and Task 5's handler. `ChatMessage.tools` is optional in TS and
defaults to `[]` in Pydantic — deliberate: the client omits the key when empty, and Pydantic's
default absorbs both that and every pre-v3.1 stored row.

**Verified while writing, not assumed.** KG subgraph nodes really do carry only `node_id`/`name`/
`label` (queried live), which is why `kind` is a literal for `kg_query`. RAG metadata really does
carry `reference`/`source_type`/`source`, with `source` a Windows-style server path — hence the
basename fallback splitting on both separators. `conversations.messages` really is jsonb with no
per-element constraint, so Task 3 needs no migration.

Both former soft spots were checked rather than left to the implementer: `mockStream(chunks:
string[])` takes an array (the tests pass one) and returns the `fetch` spy; `api.getConversation`
resolves to `Conversation` with `messages: ChatMessage[]`, so stored `tools` deserialise for free
once Task 4 lands.

**Known soft spot.** Task 5 Step 4's Tailwind token names (`text-muted`, `text-faint`) are copied
from the existing tray markup but must be confirmed against the neighbouring components rather than
assumed, since this plan does not otherwise touch styling.

**One risk worth stating plainly.** Task 2 moves `_dispatch_tool` to *before* the `tool` frame is
yielded, because the frame now carries the sources and those only exist once the tool has run. The
user-visible effect is that the tray's status line for a tool appears *after* that tool's retrieval
instead of before it — on a cold RAG process, where the embedding model load alone ran past two
minutes in measurement, that is a visibly longer silence than v3 had. The alternative is two frames
per call (a `tool` on start, a `sources` on completion), which keeps the early feedback at the cost
of a second frame type and a client-side correlation key. Not worth it for the first cut, but it is
the fix if the silence turns out to matter.
