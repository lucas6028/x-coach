# Live Tool Status & Condensed Sources (v3.2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the tool and its query *while* the lookup runs, and collapse the finished source list to a clickable count.

**Architecture:** The single `tool` SSE frame splits in two — `tool` yielded *before* dispatch (names the lookup) and `tool_done` yielded after (carries the sources) — correlated by an integer `id` that is monotonic across the whole stream. The client tracks a `pending` flag per run in memory only, strips it (and `id`) when the turn commits, and renders finished sources as a count row that expands on click.

**Tech Stack:** FastAPI + SSE (backend), React 18 + TypeScript + vitest (frontend), Python `unittest` under `tests/`.

## Global Constraints

Copied verbatim from `CLAUDE.md` and `specs/llm-chat-spec.md`; every task's requirements implicitly include these.

- **Python is always `.venv\Scripts\python.exe`.** Never bare `python`/`pip`, never `source .venv/bin/activate` (POSIX-only, fails on this machine).
- **Backend tests are always scoped to `tests/`**: `.venv\Scripts\python.exe -m pytest tests/`. Never bare `pytest`.
- **Backend coverage gate is 95%**, enforced by CI: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`.
- **All yarn/vitest commands run with cwd = `frontend/`.** The Bash and PowerShell tools share one cwd; a stray `cd` elsewhere mass-fails vitest.
- **Do not run `yarn test:coverage`.** It is red on this machine for reasons predating this branch, and `main` fails identically. The frontend gate is `yarn test` plus `yarn build`.
- **`yarn build` (`tsc -b && vite build`) is the real TypeScript gate.** vitest does not typecheck — a task is not done until `yarn build` is clean.
- **`metadata["source"]` must never reach a client.** It is a server filesystem path. Nothing in this plan touches source extraction, but no step may reintroduce it.
- **Never stage `.env.bak-20260804`** (an untracked local file in the working tree).
- New i18n keys go in **both** `en` and `zhHant` — `frontend/src/test/lib.i18n.test.ts` enforces parity.
- Interpolation uses braces: `t("key", { n: 3 })` against `"{n} sources"`.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `backend/app/services/chat.py` | The tool loop: yield `tool` before dispatch, `tool_done` after, both carrying a stream-monotonic `id` | 1 |
| `tests/test_chat_endpoint.py` | Backend frame-ordering, unconditional `tool_done`, cross-round id uniqueness | 1 |
| `frontend/src/api.ts` | `LiveToolRun` type, `onTool`/`onToolDone` handler signatures, `tool_done` routing in `dispatchSSE` | 2 |
| `frontend/src/test/api.test.ts` | SSE routing for both frames | 2 |
| `frontend/src/components/CoachTray.tsx` | Pending lifecycle, id-matched resolution, allow-list strip at commit | 2 |
| `frontend/src/test/components.CoachTray.chat.test.tsx` | Correlation, unknown-id drop, strip-at-commit | 2 |
| `frontend/src/components/ToolRunList.tsx` | `ToolRunRow` child: pending dots, collapsed count button, expanded labels | 3 |
| `frontend/src/lib/i18n.tsx` | `chat.tool.sourcesN` / `chat.tool.conceptsN`; delete the two now-unused heading keys | 3 |
| `frontend/src/test/components.ToolRunList.test.tsx` | New — component-level rendering tests driven by props | 3 |

Three tasks. Each leaves the tree green (backend pytest / `yarn test` / `yarn build`) on its own.

---

### Task 1: Backend — split the tool frame, add a stream-monotonic id

**Files:**
- Modify: `backend/app/services/chat.py:1015-1016` (declare the counter), `:1134-1156` (the dispatch loop)
- Test: `tests/test_chat_endpoint.py` — `ToolLoopTests` (add 3, rewrite 4)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: the SSE wire contract Task 2 reads —
  - `event: tool` → `{"id": <int>, "name": <str>, "query": <str>}` (no `sources` key, ever)
  - `event: tool_done` → `{"id": <int>}` plus `"sources": [{"label": str, "kind": str}, ...]` **only when non-empty**
  - `id` starts at 0 and increments once per tool call across **all** rounds of one `answer_stream` call.

**Context you need:** `answer_stream` runs up to `_MAX_TOOL_ROUNDS = 3` rounds plus one forced tool-less round (`chat.py:1018`). Inside each round, `for i, call in enumerate(turn.tool_calls)` restarts at `i = 0`. That is why the id cannot be the loop index: two rounds that each call `rag_search` would both emit `id: 0`. The `i` in `f"call_{i}"` on line 1153 is a *different* thing (the OpenAI `tool_call_id` fallback) and must be left exactly as it is.

- [ ] **Step 1: Write the three failing tests**

Add to `tests/test_chat_endpoint.py` inside `class ToolLoopTests`, after `test_the_tool_message_content_is_the_result_text`:

```python
    def test_the_tool_frame_is_yielded_before_the_tool_runs(self) -> None:
        # The whole point of v3.2: v3.1 dispatched first so the frame could carry sources, which
        # left the tray silent for the entire retrieval (past two minutes on a cold RAG process).
        # Asserting frame content is not enough -- only the ORDER proves the user sees the lookup
        # named while it is still running, so the dispatch stub records the frames emitted so far.
        seen_at_dispatch: list[list[str]] = []
        emitted: list[str] = []

        fake, _ = self._turns(
            ([], [{"id": "c1", "name": "rag_search", "arguments": '{"query": "ankle"}'}]),
            (["Answer"], []),
        )

        def dispatch(name, args, ctx):
            seen_at_dispatch.append(list(emitted))
            return chat_service._ToolResult(text='{"ok": true}', sources=[])

        with mock.patch.object(chat_service, "_stream_turn", fake), mock.patch.object(
            chat_service, "_dispatch_tool", dispatch
        ), mock.patch.object(chat_service, "chat_timeout", return_value=60.0):
            for frame in chat_service.answer_stream(
                messages=[{"role": "user", "content": "hi"}], context=self.CONTEXT, model="m"
            ):
                emitted.append(frame.split("\n")[0][len("event:") :].strip())

        self.assertEqual(seen_at_dispatch, [["tool"]])

    def test_tool_done_is_emitted_even_when_the_tool_yields_no_sources(self) -> None:
        # get_analysis never has an outside source to credit. If tool_done were conditional on
        # sources, its row would sit on screen as "still running" for the rest of the session.
        fake, _ = self._turns(
            ([], [{"id": "c1", "name": "get_analysis", "arguments": '{"include": "all"}'}]),
            (["Answer"], []),
        )
        result = chat_service._ToolResult(text='{"ok": true}', sources=[])
        events = self._run(fake, dispatch=lambda n, a, c: result)
        self.assertEqual(
            events,
            [
                ("tool", {"id": 0, "name": "get_analysis", "query": ""}),
                ("tool_done", {"id": 0}),
                ("delta", {"text": "Answer"}),
                ("done", {"model": "m"}),
            ],
        )

    def test_tool_ids_are_unique_across_rounds(self) -> None:
        # enumerate(turn.tool_calls) restarts at 0 every round, so a per-round index would collide
        # the moment two rounds each call a tool -- which is the NORMAL shape of a multi-round
        # conversation, not an edge case. Two rounds, two calls in the first, one in the second.
        fake, _ = self._turns(
            (
                [],
                [
                    {"id": "c1", "name": "kg_query", "arguments": '{"query": "a"}'},
                    {"id": "c2", "name": "rag_search", "arguments": '{"query": "b"}'},
                ],
            ),
            ([], [{"id": "c3", "name": "kg_query", "arguments": '{"query": "c"}'}]),
            (["Answer"], []),
        )
        events = self._run(fake)
        starts = [d["id"] for e, d in events if e == "tool"]
        dones = [d["id"] for e, d in events if e == "tool_done"]
        self.assertEqual(starts, [0, 1, 2])
        self.assertEqual(dones, [0, 1, 2])
```

- [ ] **Step 2: Run them to verify they fail**

```
.venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py::ToolLoopTests -v
```

Expected: the three new tests FAIL. `test_the_tool_frame_is_yielded_before_the_tool_runs` fails with `seen_at_dispatch == [[]]` (no frame emitted yet); the other two fail on the missing `id` key and the absent `tool_done` event.

- [ ] **Step 3: Declare the counter**

In `backend/app/services/chat.py`, immediately after line 1016 (`answer = ""`), add:

```python
    # Correlates each `tool` frame with its later `tool_done`. Monotonic across ALL rounds, not the
    # per-round enumerate index: `enumerate(turn.tool_calls)` restarts at 0 each round, so a
    # per-round index would collide as soon as two rounds each call a tool. Unrelated to the
    # `f"call_{i}"` tool_call_id fallback below, which stays as it is.
    tool_seq = 0
```

- [ ] **Step 4: Rewrite the dispatch loop**

Replace `backend/app/services/chat.py:1134-1156` (from `for i, call in enumerate(turn.tool_calls):` through the closing paren of the `convo.append(...)` inside it) with:

```python
        for i, call in enumerate(turn.tool_calls):
            args = _parse_tool_args(call["arguments"])
            # Frame BEFORE dispatch: the lookup can take minutes on a cold RAG process, and naming
            # it while it runs is the point of v3.2. The sources cannot ride along, so they follow
            # in `tool_done` -- which is emitted unconditionally, even with no sources, because it
            # is the COMPLETION signal. A tool with nothing to cite (get_analysis) would otherwise
            # render as still-running forever.
            uid = tool_seq
            tool_seq += 1
            yield _sse(
                "tool",
                {"id": uid, "name": call["name"], "query": _tool_query_label(call["name"], args)},
            )
            outcome = _dispatch_tool(call["name"], args, context)
            finished: dict[str, Any] = {"id": uid}
            if outcome.sources:
                # Omitted rather than [] so a client can tell "this tool has nothing to cite"
                # (get_analysis) from "this tool cited nothing".
                finished["sources"] = outcome.sources
            yield _sse("tool_done", finished)
            convo.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"] or f"call_{i}",
                    "content": outcome.text,
                }
            )
```

- [ ] **Step 5: Update the four existing tests that assert the old frame shape**

In `tests/test_chat_endpoint.py`:

At `test_a_tool_round_emits_a_tool_frame_then_the_answer` (line ~341), replace the `assertEqual(events, [...])` list with:

```python
            [
                ("tool", {"id": 0, "name": "kg_query", "query": "valgus"}),
                ("tool_done", {"id": 0}),
                ("delta", {"text": "Answer"}),
                ("done", {"model": "m"}),
            ],
```

At `test_narration_plus_a_tool_call_is_retracted_with_a_reset_frame` (line ~366), replace the list with:

```python
            [
                ("delta", {"text": "Let me check."}),
                ("reset", {}),
                ("tool", {"id": 0, "name": "rag_search", "query": "x"}),
                ("tool_done", {"id": 0}),
                ("delta", {"text": "Real answer"}),
                ("done", {"model": "m"}),
            ],
```

Rename `test_the_tool_frame_carries_sources` to `test_the_tool_done_frame_carries_sources` and replace its last two statements (`tool_frames = ...` and the `assertEqual`) with:

```python
        self.assertEqual(
            [d for e, d in events if e == "tool"],
            [{"id": 0, "name": "rag_search", "query": "ankle"}],
        )
        self.assertEqual(
            [d for e, d in events if e == "tool_done"],
            [
                {
                    "id": 0,
                    "sources": [{"label": "Wikipedia: Squat (exercise)", "kind": "encyclopedia"}],
                }
            ],
        )
```

Rename `test_the_tool_frame_omits_sources_when_there_are_none` to `test_the_tool_done_frame_omits_sources_when_there_are_none` and replace its last two statements with:

```python
        self.assertEqual([d for e, d in events if e == "tool_done"], [{"id": 0}])
```

- [ ] **Step 6: Run the backend suite**

```
.venv\Scripts\python.exe -m pytest tests/ -q
```

Expected: PASS, no fewer tests than before plus 3.

- [ ] **Step 7: Run the coverage gate**

```
.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95
```

Expected: PASS at ≥95%. Report `chat.py`'s own percentage in the task report.

- [ ] **Step 8: Commit**

```bash
git add backend/app/services/chat.py tests/test_chat_endpoint.py
git commit -m "feat: yield the tool frame before dispatch, sources in tool_done"
```

---

### Task 2: Client transport and CoachTray state

**Files:**
- Modify: `frontend/src/api.ts:162-168` (add `LiveToolRun`), `:211-221` (handler signatures), `:408-425` (`dispatchSSE`)
- Modify: `frontend/src/components/CoachTray.tsx:61` (state type), `:166` (accumulator type), `:191-194` (`onTool`), `:212-215` (commit)
- Test: `frontend/src/test/api.test.ts` (rewrite 2, add 1), `frontend/src/test/components.CoachTray.chat.test.tsx` (migrate ~9 call sites, add 3)

**Interfaces:**
- Consumes: Task 1's wire contract (`tool` with `id`/`name`/`query`; `tool_done` with `id` and optional `sources`).
- Produces, for Task 3:
  - `export interface LiveToolRun extends ToolRun { id: number; pending: boolean }` in `api.ts`
  - `ToolRunList` receives an array where each element may carry `pending?: boolean`. `ToolRun` itself (the persisted shape `{name, query, sources?}`) is **unchanged**.

**Context you need:**

`ChatStreamHandlers.onTool` currently has the signature `(name, query, sources) => void`. This task changes it to `(id, name, query) => void` and adds `onToolDone: (id, sources) => void`. Every existing call site in the tests passes the old 3-arg form — they all migrate in this task, and `yarn build` will not pass until they do.

Why an unknown `id` on `tool_done` is dropped rather than defaulted: attaching a citation to the wrong tool is a worse failure than losing one, and this feature exists to make provenance trustworthy. A `tool` frame with a non-numeric `id`, by contrast, still renders (with `id: -1`) — showing the lookup is worth something even when it can never be resolved, and it settles at commit anyway (see Step 6).

- [ ] **Step 1: Write the failing api.ts tests**

In `frontend/src/test/api.test.ts`, inside `describe("chat SSE tool frames", ...)`:

Replace the body of `it("routes tool and reset frames to their handlers", ...)` — change the first body line and the `onTool` handler:

```ts
      'event: tool\ndata: {"id":0,"name":"kg_query","query":"knee valgus"}\n\n',
```
```ts
      onTool: (_id, n, q) => seen.push(`tool:${n}:${q}`),
```

Replace `it("passes tool sources to onTool, and an empty array when the frame omits them", ...)` entirely with:

```ts
  it("routes tool_done sources to onToolDone, and an empty array when the frame omits them", async () => {
    const seen: Array<[number, unknown]> = [];
    mockStream([
      'event: tool\ndata: {"id":0,"name":"rag_search","query":"ankle"}\n\n',
      'event: tool_done\ndata: {"id":0,"sources":[{"label":"Wikipedia: Squat (exercise)","kind":"encyclopedia"}]}\n\n',
      'event: tool\ndata: {"id":1,"name":"get_analysis","query":"Depth"}\n\n',
      'event: tool_done\ndata: {"id":1}\n\n',
      'event: delta\ndata: {"text":"A"}\n\n',
      'event: done\ndata: {"model":"m"}\n\n',
    ]);
    await api.chatStream([{ role: "user", content: "hi" }], { fault_count: 0, quality: {}, faults: [] }, {
      onDelta: () => undefined,
      onDone: () => undefined,
      onError: () => undefined,
      onToolDone: (id, s) => seen.push([id, s]),
    });
    expect(seen).toEqual([
      [0, [{ label: "Wikipedia: Squat (exercise)", kind: "encyclopedia" }]],
      [1, []],
    ]);
  });

  it("drops a tool_done frame whose id is not a number", async () => {
    // A citation attached to the wrong tool is a worse failure than a citation lost: this feature
    // exists to make provenance trustworthy, so an uncorrelatable frame is discarded outright.
    const seen: unknown[] = [];
    mockStream([
      'event: tool_done\ndata: {"sources":[{"label":"L","kind":"paper"}]}\n\n',
      'event: done\ndata: {"model":"m"}\n\n',
    ]);
    await api.chatStream([{ role: "user", content: "hi" }], { fault_count: 0, quality: {}, faults: [] }, {
      onDelta: () => undefined,
      onDone: () => undefined,
      onError: () => undefined,
      onToolDone: (id, s) => seen.push([id, s]),
    });
    expect(seen).toEqual([]);
  });
```

- [ ] **Step 2: Run them to verify they fail**

From `frontend/`:
```
yarn test src/test/api.test.ts
```
Expected: FAIL — `onToolDone` is not a known handler, so nothing is pushed and both new assertions fail.

- [ ] **Step 3: Add the type and the handler signatures**

In `frontend/src/api.ts`, after the `ToolRun` interface (line 168), add:

```ts
// One tool call as the client tracks it WHILE the turn streams. `id` correlates the `tool` frame
// with the `tool_done` that follows it; `pending` is true until that frame arrives. Both are
// transport/UI state — they are stripped when the run is committed to a ChatMessage, so the
// persisted shape stays exactly `ToolRun`.
export interface LiveToolRun extends ToolRun {
  id: number;
  pending: boolean;
}
```

Replace `frontend/src/api.ts:215-216` with:

```ts
  // The coach started a tool call. Fires BEFORE the tool runs, so the UI can name the lookup while
  // it is still in flight. Optional so a caller that doesn't surface tool progress is unaffected.
  onTool?: (id: number, name: string, query: string) => void;
  // That tool call finished. Always fires once per `onTool`, even with no sources — it is the
  // completion signal, not the sources signal, so a tool with nothing to cite still settles.
  onToolDone?: (id: number, sources: ToolSource[]) => void;
```

- [ ] **Step 4: Route the frames**

In `frontend/src/api.ts`, add `id?: number;` to the `data` type at line 408-415 (after `detail?: string;`), then replace line 422 with:

```ts
  else if (event === "tool")
    handlers.onTool?.(typeof data.id === "number" ? data.id : -1, data.name ?? "", data.query ?? "");
  // An uncorrelatable tool_done is dropped, not defaulted: mis-attributing a citation is worse than
  // losing one. A `tool` with no id still renders (id -1) and simply never resolves — it settles
  // when the turn commits, since `pending` is stripped there.
  else if (event === "tool_done" && typeof data.id === "number")
    handlers.onToolDone?.(data.id, data.sources ?? []);
```

- [ ] **Step 5: Run the api tests**

From `frontend/`:
```
yarn test src/test/api.test.ts
```
Expected: PASS.

- [ ] **Step 6: Wire CoachTray**

In `frontend/src/components/CoachTray.tsx`:

Change the `ToolRun` import to also bring in `LiveToolRun` (the import sits with the other `api` type imports at the top of the file).

Line 61 → `const [toolRuns, setToolRuns] = useState<LiveToolRun[]>([]);`

Line 166 → `let runs: LiveToolRun[] = [];`

Replace lines 191-194 (`onTool: ...`) with:

```ts
          onTool: (id, name, query) => {
            runs = [...runs, { id, name, query, pending: true }];
            setToolRuns(runs);
          },
          // Match on `id`, and only while still pending: a duplicate or replayed frame must not
          // overwrite a run that has already settled.
          onToolDone: (id, sources) => {
            runs = runs.map((r) =>
              r.id === id && r.pending
                ? { ...r, pending: false, ...(sources.length ? { sources } : {}) }
                : r,
            );
            setToolRuns(runs);
          },
```

Replace lines 212-215 (the `thread` construction) with:

```ts
      // Strip the in-memory transport/UI fields by REBUILDING each run from an allow-list rather
      // than destructuring them away: a field added to LiveToolRun later must not silently ride
      // along into stored jsonb. This also settles any run still pending because its `tool_done`
      // was lost or uncorrelatable — the committed record simply shows no sources, which is the
      // truth, instead of a row that claims to still be running.
      const committed: ToolRun[] = runs.map((r) => ({
        name: r.name,
        query: r.query,
        ...(r.sources?.length ? { sources: r.sources } : {}),
      }));
      const thread: ChatMessage[] = [
        ...next,
        { role: "assistant", content: acc, ...(committed.length ? { tools: committed } : {}) },
      ];
```

- [ ] **Step 7: Migrate every `onTool` call site in the CoachTray tests**

In `frontend/src/test/components.CoachTray.chat.test.tsx`, every `handlers.onTool?.(name, query, sources)` becomes `handlers.onTool?.(<id>, name, query)` followed by `handlers.onToolDone?.(<id>, sources)`. Ids number from 0 within each mock implementation. Precisely:

**Work this table bottom-up, or match on the unique `zzq-` subject string instead of the line number.** Every replacement turns one line into two, so the moment you edit the first row top-down every line number below it is off by one.

| Line (pre-edit) | Replace with |
|---|---|
| 495 | `handlers.onTool?.(0, "kg_query", "zzqury-kg-subject");` then, on the next line, `handlers.onToolDone?.(0, []);` |
| 529 | `handlers.onTool?.(0, "something_else", "zzqux-subject");` + `handlers.onToolDone?.(0, []);` |
| 543 | `handlers.onTool?.(0, "kg_query", "zzq-early-subject");` + `handlers.onToolDone?.(0, []);` |
| 546 | `handlers.onTool?.(1, "kg_query", "valgus");` + `handlers.onToolDone?.(1, []);` |
| 567-571 | `handlers.onTool?.(0, "rag_search", "zzq-live-subject");` + `handlers.onToolDone?.(0, [{ label: "zzq-Live-Source", kind: "encyclopedia" }]);` |
| 592-596 | `handlers.onTool?.(0, "rag_search", "zzq-ankle-subject");` + `handlers.onToolDone?.(0, [{ label: "zzq-Wiki-Source", kind: "encyclopedia" }]);` |
| 610 | `handlers.onTool?.(0, "kg_query", "zzq-first-subject");` + `handlers.onToolDone?.(0, []);` |
| 611 | `handlers.onTool?.(1, "rag_search", "zzq-second-subject");` + `handlers.onToolDone?.(1, []);` |
| 629-631 | `handlers.onTool?.(0, "kg_query", "zzq-concept-subject");` + `handlers.onToolDone?.(0, [{ label: "zzq-Concept-Label", kind: "concept" }]);` |
| 632-634 | `handlers.onTool?.(1, "rag_search", "zzq-paper-subject");` + `handlers.onToolDone?.(1, [{ label: "zzq-Paper-Label", kind: "paper" }]);` |
| 660 | `handlers.onTool?.(0, "kg_query", "zzq-early-subject");` + `handlers.onToolDone?.(0, []);` |
| 663 | `handlers.onTool?.(1, "rag_search", "zzq-kept-subject");` + `handlers.onToolDone?.(1, []);` |
| 677 | `handlers.onTool?.(0, "rag_search", "ankle");` + `handlers.onToolDone?.(0, [{ label: "zzq-Persisted-Source", kind: "paper" }]);` |

One test needs more than a mechanical swap. In `it("keeps tool records visible after the answer starts streaming", ...)` (line ~561) the assertion runs while the stream is parked, so the `onToolDone` call must come **before** the `await new Promise(...)` park, not after — put it directly under the `onTool` line.

In `it("keeps tool records on the committed assistant message once the turn completes", ...)` nothing else changes.

- [ ] **Step 8: Add the three new CoachTray tests**

Append inside the same `describe` block that holds the tool-record tests:

```tsx
  it("lands each tool call's sources on its own row when the same tool is called twice", async () => {
    // The case that actually exercises correlation, and it is reachable today: one round can call
    // rag_search twice, and two rounds routinely do. tool_done arrives OUT of start order here, so
    // a "last pending run" rule would attach both source sets to the wrong rows.
    h.chatStream.mockImplementation(async (_m, _c, handlers) => {
      handlers.onTool?.(0, "rag_search", "zzq-first-query");
      handlers.onTool?.(1, "rag_search", "zzq-second-query");
      handlers.onToolDone?.(1, [{ label: "zzq-Second-Source", kind: "paper" }]);
      handlers.onToolDone?.(0, [{ label: "zzq-First-Source", kind: "paper" }]);
      handlers.onDelta("A");
      handlers.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    await screen.findByText("A");
    const thread = h.putConversation.mock.calls[0][1] as Array<{ tools?: unknown[] }>;
    expect(thread[thread.length - 1].tools).toEqual([
      { name: "rag_search", query: "zzq-first-query", sources: [{ label: "zzq-First-Source", kind: "paper" }] },
      { name: "rag_search", query: "zzq-second-query", sources: [{ label: "zzq-Second-Source", kind: "paper" }] },
    ]);
  });

  it("drops a tool_done whose id matches no run", async () => {
    h.chatStream.mockImplementation(async (_m, _c, handlers) => {
      handlers.onTool?.(0, "rag_search", "zzq-orphan-query");
      handlers.onToolDone?.(99, [{ label: "zzq-Orphan-Source", kind: "paper" }]);
      handlers.onDelta("A");
      handlers.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    await screen.findByText("A");
    const thread = h.putConversation.mock.calls[0][1] as Array<{ tools?: unknown[] }>;
    expect(thread[thread.length - 1].tools).toEqual([
      { name: "rag_search", query: "zzq-orphan-query" },
    ]);
  });

  it("commits tool records without the in-memory id and pending fields", async () => {
    // These are transport/UI state. The backend's ToolRun model would ignore them, but that
    // backstop is coincidental — the strip is the mechanism, exactly as with `tools` on /api/chat.
    h.chatStream.mockImplementation(async (_m, _c, handlers) => {
      handlers.onTool?.(0, "kg_query", "zzq-strip-query");
      handlers.onToolDone?.(0, []);
      handlers.onDelta("A");
      handlers.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    await screen.findByText("A");
    const thread = h.putConversation.mock.calls[0][1] as Array<{ tools?: unknown[] }>;
    const blob = JSON.stringify(thread[thread.length - 1].tools);
    expect(blob).not.toContain("pending");
    expect(blob).not.toContain('"id"');
  });
```

- [ ] **Step 9: Run the frontend tests and the typecheck**

From `frontend/`:
```
yarn test src/test/api.test.ts src/test/components.CoachTray.chat.test.tsx
yarn build
```
Expected: both PASS. `yarn build` is the real TypeScript gate — vitest does not typecheck, so a signature mismatch surfaces only here.

- [ ] **Step 10: Commit**

```bash
git add frontend/src/api.ts frontend/src/components/CoachTray.tsx frontend/src/test/api.test.ts frontend/src/test/components.CoachTray.chat.test.tsx
git commit -m "feat: track tool runs as pending until their tool_done frame"
```

---

### Task 3: Rendering — pending dots and a collapsible source count

**Files:**
- Modify: `frontend/src/components/ToolRunList.tsx` (whole file)
- Modify: `frontend/src/lib/i18n.tsx:107-108` and `:793-794` (replace two keys with two count keys, in both dictionaries)
- Create: `frontend/src/test/components.ToolRunList.test.tsx`
- Modify: `frontend/src/test/components.CoachTray.chat.test.tsx` (3 assertions that expect source labels to be visible without a click)

**Interfaces:**
- Consumes: `LiveToolRun` from Task 2 (`api.ts`); `LumenLoader` from `frontend/src/components/LumenLoader.tsx` (`<LumenLoader variant="dots" />`, already used by `CoachTray` at line 474).
- Produces: nothing later tasks depend on. This is the last task.

**Context you need:**

`ToolRunList` is called from two places in `CoachTray.tsx` — with `m.tools` (a committed `ToolRun[]`, no `pending`) and with `toolRuns` (a live `LiveToolRun[]`). The props type must accept both, so it takes `pending` as optional. Do **not** widen it to accept `id`: the renderer has no use for it.

Keep the label-and-query line as a **single element with the dots inside it**. Several existing CoachTray tests do `screen.getByText(/zzq-…-subject/).parentElement` to reach the row; testing-library's default matcher joins an element's direct text-node children and ignores element children, so putting `<LumenLoader/>` inside that line keeps both the text match and the parent relationship intact.

- [ ] **Step 1: Write the failing component tests**

Create `frontend/src/test/components.ToolRunList.test.tsx`:

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { I18nProvider } from "../lib/i18n";
import { ToolRunList } from "../components/ToolRunList";
import type { ToolSource } from "../api";

function renderRuns(runs: Array<{ name: string; query: string; sources?: ToolSource[]; pending?: boolean }>) {
  return render(
    <I18nProvider>
      <ToolRunList runs={runs} />
    </I18nProvider>
  );
}

describe("ToolRunList", () => {
  it("shows an animated marker while a run is pending, and none once it settles", async () => {
    // The dots are now the ONLY signal that anything is happening across a retrieval that can run
    // minutes — CoachTray's own dots are suppressed as soon as a tool record exists.
    const { rerender, container } = renderRuns([
      { name: "rag_search", query: "zzq-q", pending: true },
    ]);
    expect(container.querySelector(".lm-dots")).toBeTruthy();
    rerender(
      <I18nProvider>
        <ToolRunList runs={[{ name: "rag_search", query: "zzq-q", pending: false }]} />
      </I18nProvider>
    );
    expect(container.querySelector(".lm-dots")).toBeNull();
  });

  it("renders no source row at all for a settled run with nothing to cite", async () => {
    // get_analysis reads the user's own analysis — there is no outside source to credit, so a
    // "0 sources" row would be noise, and a pending marker would be a lie.
    const { container } = renderRuns([
      { name: "get_analysis", query: "Depth", pending: false },
    ]);
    expect(screen.queryByRole("button")).toBeNull();
    expect(container.querySelector(".lm-dots")).toBeNull();
  });

  it("collapses sources to a count and reveals the labels on click", async () => {
    renderRuns([
      {
        name: "rag_search",
        query: "ankle",
        sources: [
          { label: "zzq-Source-A", kind: "paper" },
          { label: "zzq-Source-B", kind: "encyclopedia" },
        ],
      },
    ]);
    const toggle = screen.getByRole("button", { name: /Sources · 2/ });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByText("zzq-Source-A")).toBeNull();
    await userEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByText("zzq-Source-A")).toBeTruthy();
    expect(screen.getByText("zzq-Source-B")).toBeTruthy();
  });

  it("counts kg_query results as knowledge-graph concepts, not sources", async () => {
    // v3.1's red line survives the collapse: a graph node carries no citation anywhere in the
    // graph, so counting it must not rename it into a citation. Keyed off `kind`, not tool name.
    renderRuns([
      { name: "kg_query", query: "valgus", sources: [{ label: "zzq-Concept", kind: "concept" }] },
      { name: "rag_search", query: "ankle", sources: [{ label: "zzq-Paper", kind: "paper" }] },
    ]);
    expect(screen.getByRole("button", { name: /Knowledge-graph concepts · 1/ })).toBeTruthy();
    expect(screen.getByRole("button", { name: /Sources · 1/ })).toBeTruthy();
  });

  it("expands each row independently", async () => {
    renderRuns([
      { name: "rag_search", query: "a", sources: [{ label: "zzq-Row-One", kind: "paper" }] },
      { name: "rag_search", query: "b", sources: [{ label: "zzq-Row-Two", kind: "paper" }] },
    ]);
    const [first] = screen.getAllByRole("button");
    await userEvent.click(first);
    expect(screen.getByText("zzq-Row-One")).toBeTruthy();
    expect(screen.queryByText("zzq-Row-Two")).toBeNull();
  });
});
```

- [ ] **Step 2: Run them to verify they fail**

From `frontend/`:
```
yarn test src/test/components.ToolRunList.test.tsx
```
Expected: FAIL — no button role exists (sources render as plain divs), and `.lm-dots` is never rendered.

**On the `getByRole(…, { name })` regexes:** accessible-name computation drops the `aria-hidden` chevron but can leave leading/interior whitespace, and testing-library's name matching may normalize case. The regexes here are unanchored and the two labels share no substring, so both cases are safe — but if one fails, read the actual "Here are the accessible roles" dump vitest prints and fix from that, do not adjust the regex by inspection.

- [ ] **Step 3: Swap the i18n keys**

In `frontend/src/lib/i18n.tsx`, replace lines 107-108 (the `en` dict) with:

```ts
  "chat.tool.sourcesN": "Sources · {n}",
  "chat.tool.conceptsN": "Knowledge-graph concepts · {n}",
```

The count trails the noun rather than leading it (`"{n} sources"`) so English never renders "1 sources" — this UI has no plural machinery, and a single-source retrieval is a common case, not an edge one.

and lines 793-794 (the `zhHant` dict) with:

```ts
  "chat.tool.sourcesN": "引用來源 {n} 筆",
  "chat.tool.conceptsN": "知識圖譜概念 {n} 筆",
```

The old `chat.tool.sources` / `chat.tool.concepts` keys are removed: the count row *is* the heading now, so a second heading under it would be redundant. `ToolRunList` was their only consumer.

- [ ] **Step 4: Rewrite ToolRunList**

Replace `frontend/src/components/ToolRunList.tsx` entirely:

```tsx
import { useState } from "react";
import { useI18n } from "../lib/i18n";
import type { ToolRun } from "../api";
import { LumenLoader } from "./LumenLoader";

// The tools we have i18n labels for. A name outside this list falls back to the generic label —
// `t()` returns the key itself on a miss (i18n.tsx), so an unguarded lookup would render a raw key
// like "chat.tool.something_else" into the tray.
const TOOL_LABEL_KEYS = ["get_analysis", "kg_query", "rag_search"] as const;

// A committed `ToolRun` plus the one live field the renderer cares about. `pending` is absent on a
// message restored from history — a stored record is finished by definition.
type DisplayToolRun = ToolRun & { pending?: boolean };

/** The tool calls behind one answer, in call order, above the answer they produced.
 *
 * Used twice: for a committed assistant message (from `message.tools`) and for the turn currently
 * streaming (from live state). Same markup both times, so nothing shifts when the turn commits.
 */
export function ToolRunList({ runs }: { runs: DisplayToolRun[] }) {
  if (!runs.length) return null;
  return (
    <div className="flex flex-col gap-2">
      {runs.map((run, i) => (
        <ToolRunRow key={i} run={run} />
      ))}
    </div>
  );
}

/** One tool call: the lookup line, plus a collapsed count of what it cited.
 *
 * A child component rather than inline markup because the expanded state is per-row and a hook
 * cannot live inside a `.map`. That state is deliberately ephemeral — a row collapses when the turn
 * commits, since the live list is replaced by the committed message's own. Preserving it would mean
 * hoisting row state into CoachTray and keying it across both render paths, which is not worth it
 * for a state the user usually enters after reading.
 *
 * A source's `kind` decides the heading, not the tool that produced it. `kg_query`'s entries come
 * back with `kind: "concept"` because knowledge-graph nodes carry no citation anywhere in the
 * graph; counting them under the same word as retrieved documents would tell the user a concept is
 * a source. Keying off `kind` means a future tool that also returns concept-kind sources gets the
 * safe wording automatically.
 */
function ToolRunRow({ run }: { run: DisplayToolRun }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const known = TOOL_LABEL_KEYS.includes(run.name as (typeof TOOL_LABEL_KEYS)[number]);
  const sources = Array.isArray(run.sources) ? run.sources : [];
  const isConcept = sources.some((s) => s.kind === "concept");
  return (
    <div className="text-xs text-muted" aria-busy={!!run.pending}>
      {/* The label, the query, and the pending marker share ONE element: the marker is an element
          child, which testing-library's text matcher ignores, so the line still matches by text and
          still parents the source block below it. */}
      <div className="flex items-center gap-2">
        {known ? t(`chat.tool.${run.name}`) : t("chat.tool.generic")}
        {run.query ? `${t("chat.tool.sep")}${run.query}` : ""}
        {run.pending && (
          // aria-hidden because LumenLoader's dots carry role="status": today exactly one exists at
          // a time (CoachTray's, gated on toolRuns.length === 0), but a three-tool turn would mount
          // three simultaneous live regions all announcing the same string. `aria-busy` on the row
          // states the same thing once, in the right place.
          <span aria-hidden="true" className="inline-flex">
            <LumenLoader variant="dots" />
          </span>
        )}
      </div>
      {sources.length > 0 && (
        <div className="mt-1 flex flex-col gap-0.5 pl-3">
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            className="self-start text-left text-faint transition-colors hover:text-muted"
          >
            <span aria-hidden="true" className="mr-1 inline-block">
              {open ? "⌄" : "›"}
            </span>
            {t(isConcept ? "chat.tool.conceptsN" : "chat.tool.sourcesN", { n: sources.length })}
          </button>
          {open &&
            sources.map((s, j) => (
              <div key={j} className="text-faint">
                {s.label}
              </div>
            ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Run the component tests**

From `frontend/`:
```
yarn test src/test/components.ToolRunList.test.tsx
```
Expected: PASS, 5/5.

- [ ] **Step 6: Update the three CoachTray assertions that expect open sources**

Sources are collapsed by default now, so three existing assertions in `frontend/src/test/components.CoachTray.chat.test.tsx` must click first.

In `it("keeps tool records visible after the answer starts streaming", ...)`, replace `expect(screen.getByText("zzq-Live-Source")).toBeTruthy();` with:

```tsx
    // Collapsed by default (v3.2) — the count is what shows; the label is one click away.
    await userEvent.click(screen.getByRole("button", { name: /Sources · 1/ }));
    expect(screen.getByText("zzq-Live-Source")).toBeTruthy();
```

In `it("keeps tool records on the committed assistant message once the turn completes", ...)`, replace `expect(screen.getByText("zzq-Wiki-Source")).toBeTruthy();` with:

```tsx
    await userEvent.click(screen.getByRole("button", { name: /Sources · 1/ }));
    expect(screen.getByText("zzq-Wiki-Source")).toBeTruthy();
```

Replace the whole body of `it("heads a kg_query source block \"Knowledge-graph concepts\" and a rag_search block \"Sources\", keyed by kind not tool name", ...)` — rename it and rewrite the assertions, since the headings are now count buttons:

```tsx
  it("counts a kg_query run as concepts and a rag_search run as sources, keyed by kind not tool name", async () => {
    // v3.1's red line, carried through the v3.2 collapse: a knowledge-graph node carries no citation
    // anywhere in the graph, so it must never be counted under the same word as a retrieved paper.
    h.chatStream.mockImplementation(async (_m, _c, handlers) => {
      handlers.onTool?.(0, "kg_query", "zzq-concept-subject");
      handlers.onToolDone?.(0, [{ label: "zzq-Concept-Label", kind: "concept" }]);
      handlers.onTool?.(1, "rag_search", "zzq-paper-subject");
      handlers.onToolDone?.(1, [{ label: "zzq-Paper-Label", kind: "paper" }]);
      handlers.onDelta("A");
      handlers.onDone("m");
    });
    renderTray();
    await sendMessage("why?");
    expect(await screen.findByText("A")).toBeTruthy();

    const conceptToggle = screen.getByRole("button", { name: /Knowledge-graph concepts · 1/ });
    const sourceToggle = screen.getByRole("button", { name: /Sources · 1/ });
    await userEvent.click(conceptToggle);
    await userEvent.click(sourceToggle);

    // Each row lists only its own tool's label, under its own count — not the other's.
    const conceptBlock = conceptToggle.parentElement as HTMLElement;
    const sourceBlock = sourceToggle.parentElement as HTMLElement;
    expect(conceptBlock.textContent).toContain("zzq-Concept-Label");
    expect(conceptBlock.textContent).not.toContain("zzq-Paper-Label");
    expect(sourceBlock.textContent).toContain("zzq-Paper-Label");
    expect(sourceBlock.textContent).not.toContain("zzq-Concept-Label");
  });
```

(`userEvent` is already imported at the top of this file.)

- [ ] **Step 7: Run the full frontend suite and the typecheck**

From `frontend/`:
```
yarn test
yarn build
```
Expected: PASS. **Known noise:** the full suite intermittently reports 1-2 `Test timed out in 5000ms` failures in files unrelated to chat (`pages.Movements.test.tsx`, `App.movement.test.tsx`) — vitest's default 5s `testTimeout` against a ~478s whole-suite run. If they appear, re-run those files in isolation to confirm they pass, and say so in the report. Do **not** run `yarn test:coverage`.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/ToolRunList.tsx frontend/src/lib/i18n.tsx frontend/src/test/components.ToolRunList.test.tsx frontend/src/test/components.CoachTray.chat.test.tsx
git commit -m "feat: collapse tool sources to a clickable count, show pending dots"
```

---

## Verification against the spec's success criteria

| Criterion (spec §"Success criteria (v3.2)") | Pinned by |
|---|---|
| 1. The tray names the tool and query while it runs | Task 1 Step 1 `test_the_tool_frame_is_yielded_before_the_tool_runs`; Task 3 Step 1 pending-marker test |
| 2. Sources appear as a count; clicking reveals them | Task 3 Step 1 `collapses sources to a count and reveals the labels on click` |
| 3. `get_analysis` completes visibly, shows no source row | Task 1 Step 1 `test_tool_done_is_emitted_even_when_the_tool_yields_no_sources`; Task 3 Step 1 `renders no source row at all for a settled run with nothing to cite` |
| 4. Two `rag_search` calls each show their own sources | Task 1 Step 1 `test_tool_ids_are_unique_across_rounds`; Task 2 Step 8 `lands each tool call's sources on its own row…` |
| 5. A stream dying mid-tool leaves no row claiming to run | Task 2 Step 6 allow-list strip (commit path) + the pre-existing `finally { setToolRuns([]) }` (rollback path); Task 2 Step 8 `drops a tool_done whose id matches no run` |
| 6. No v3.1 regression | Task 3 Step 6 concept-vs-source counting; the migrated persistence/restore tests in Task 2 Step 7 |

**Known gap, accepted:** a stream that neither ends nor errors leaves a pending row on screen indefinitely. That is a hung connection — the v3.1 thinking dots hung in exactly the same way, and the whole turn is stuck regardless — so v3.2 adds no new failure mode there.
