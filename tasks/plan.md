# Implementation Plan — LLM Chat v2 (Streaming · Persistence · Markdown)

Source of truth: [`specs/llm-chat-spec.md`](../specs/llm-chat-spec.md) (v2 section). This plan turns
that spec into an ordered, dependency-aware set of vertically-sliced tasks. Task checklist:
[`tasks/todo.md`](./todo.md).

**Status:** awaiting review (Phase 3 gate). No code changes until approved.

---

## 1. Dependency graph

```
                 ┌─────────────────────────────────────────────┐
                 │  A. STREAMING (reshapes the /api/chat seam)  │
                 │  A1 backend SSE ──► A2 frontend stream read  │
                 └───────────────┬──────────────┬──────────────┘
                                 │              │
              (markdown renders  │              │ (persistence saves a turn only
               partial stream)   ▼              ▼  AFTER a clean `done`)
                 ┌───────────────────────┐   ┌──────────────────────────────────┐
                 │  B. MARKDOWN          │   │  C. PERSISTENCE                  │
                 │  B1 render + deps     │   │  C1 migration ─► C2 store ─►     │
                 │  B2 prompt line       │   │  C3 router ─► C4 client restore  │
                 └───────────────────────┘   └──────────────────────────────────┘
                                 └──────────────┬──────────────┘
                                                ▼
                                 ┌──────────────────────────────┐
                                 │  D. REGRESSION + SPEC CLOSE   │
                                 └──────────────────────────────┘
```

**Hard edges (must precede):**
- **A1 → A2** — the client can't consume a stream until the endpoint emits one.
- **A → B** — markdown renders the assistant turn that streaming now fills incrementally; B must
  tolerate partial markdown, so streaming exists first. (B is otherwise the most independent block.)
- **A → C4** — the client persists a turn only after it observes a clean `done` event.
- **C1 → C2 → C3 → C4** — schema, then the store layer over it, then the HTTP surface, then the
  client that calls it. Standard DB→store→API→UI order (mirrors the shipped `analyses` stack).

**No edge (parallelizable if desired):** B and C are independent of each other. Recommended serial
order for a solo pass is **A → B → C** (spec build order); B and C could interleave.

## 2. Vertical slices (each task = one complete, verifiable path)

Every task lands with its own tests green and is independently reviewable. No task touches >5 files.
Slices are cut so each delivers a working path end-of-layer, not a horizontal "all backends then all
frontends" split.

### Phase A — Streaming

- **A1 — Backend: `/api/chat` emits SSE.** Replace the buffered service+router path with a streaming
  one. `services/chat.py`: swap `_chat_completion` → a sync generator `_stream_completion(messages)
  -> Iterator[str]` (deferred `httpx`, `httpx.stream(...)` parsing OpenAI SSE lines) and add
  `answer_stream(*, messages, context)` that yields `delta`/`done`/`error` SSE frames, preserving the
  empty-completion invariant (blank accumulation → `error`, never `done`). `routers/chat.py`: keep
  pre-flight 503/422 checks, then return `StreamingResponse(answer_stream(...),
  media_type="text/event-stream")` run so the sync generator iterates off the event loop. Rewrite
  `tests/test_chat_endpoint.py` streaming cases (normal / mid-stream error / empty) by patching
  `_stream_completion`; keep the `_build_system_prompt` groundedness tests unchanged.

- **A2 — Frontend: consume the stream.** `api.ts`: replace `chat()` with a streaming client
  (`chatStream(messages, context, { onDelta, onDone, onError, signal })`) that throws `ChatError(status)`
  on a pre-flight non-200 and routes in-band `error` frames to `onError`. `CoachTray.tsx`: append
  `delta`s into the in-progress assistant turn, finalize on `done`, roll back the optimistic user
  turn on error (v1 behavior preserved). Update `components.CoachTray.chat.test.tsx` to mock a fake
  stream.

- **✅ Checkpoint A** — see §4.

### Phase B — Markdown

- **B1 — Frontend: render assistant turns as sanitized markdown.** Add `react-markdown` +
  `rehype-sanitize` (⚠ **Ask-first dep — confirm at review**). New `components/Markdown.tsx` (or
  inline) rendering the assistant `content` with a coaching-safe sanitize schema (no raw HTML/script;
  links per the open question, default off), replacing the raw `<p>` at `CoachTray.tsx:283`. User
  turns stay plain text. Add a render test (`**bold**`→`<strong>`, `<script>` stripped, partial
  markdown renders).

- **B2 — Backend: permit light markdown in the system prompt.** Add one line to `_SYSTEM_PREAMBLE`
  allowing bold/short-lists/inline-code/timecodes **without** relaxing any grounding rule. Extend a
  `_build_system_prompt` test to assert the markdown-permission line is present and the honesty
  constraints are still intact.

- **✅ Checkpoint B** — see §4.

### Phase C — Persistence

- **C1 — Schema: `conversations` migration.** New `db/migrations/<ts>_conversations.sql`:
  `conversations(id, user_id → auth.users, video_id text /* no FK, by design */, messages jsonb
  default '[]', created_at, updated_at, unique(user_id, video_id))` + RLS `conversations_owner_all`
  (authenticated, `auth.uid() = user_id`) + reuse `public.touch_updated_at()` trigger. Verify:
  migration reads cleanly and mirrors the `analyses` precedent (no FK to `videos`/`analyses`).

- **C2 — Store: conversation read/write.** `services/store.py`: `upsert_conversation(*, token,
  user_id, video_id, messages)` (upsert on `user_id,video_id`), `get_conversation(*, token,
  video_id) -> {messages} | None`; extend `delete_all_analyses` to also clear the caller's
  conversations. New `tests/test_conversations_store.py` (or extend `test_backend.py`) patching
  `_user_client`, mirroring the analyses-store tests.

- **C3 — Router: `/api/conversations/{video_id}`.** New `routers/conversations.py`: `PUT`
  (upsert full thread) + `GET` (restore, `{messages: []}` when none), both `Depends(get_current_user)`
  (401 without session). Wire into `main.py`. New `tests/test_conversations_endpoint.py` (401 without
  session; PUT→GET round-trip; delegates to store) mirroring `test_chat_endpoint.py`'s direct-coroutine
  style.

- **C4 — Client: restore + save the thread.** `api.ts`: `getConversation(videoId)` /
  `putConversation(videoId, messages)` + `Conversation` type. `CoachTray.tsx`: on working session +
  `analysis.video_id`, GET the conversation to seed `messages` (replacing the v1 "reset to []"
  effect at `CoachTray.tsx:51-54`); after a turn reaches `done`, PUT the updated thread. Restore keys
  on `video_id` only — no replay-flag plumbing needed (a fresh upload's id simply has no saved row).
  Update the chat test for restore-on-load + save-after-turn.

- **✅ Checkpoint C** — see §4.

### Phase D — Close-out

- **D1 — Full regression + spec close.** Run the whole backend suite + `yarn build` + `yarn test`;
  confirm `backend/app` coverage ≥ 95%. Tick the v2 Success Criteria in `specs/llm-chat-spec.md`,
  record the resolved open questions, and note anything discovered during build (keep the doc alive).

## 3. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **SSE status is frozen after first flush** — a naive `raise HTTPException` mid-stream would 500 the socket, not message the user. | A1 emits mid-stream failures as in-band `error` frames; only pre-flight (503/401/422/immediate-502) uses HTTP status. Tests drive the mid-stream-error branch explicitly. |
| **Empty completion wedges the next send** (v1 `content min_length=1`). | Preserve the invariant: blank accumulation → `error`, never `done`; client never stores/persists an empty assistant turn. Dedicated test. |
| **Coverage gate (`backend/app` ≥ 95%)** breaks if the generator's branches aren't all hit. | Seam is a patchable generator; A1 tests exercise normal / mid-stream-error / empty branches. |
| **Unsanitized LLM HTML = XSS.** | B1 uses `rehype-sanitize` (default schema strips script/handlers); test asserts a `<script>` payload is stripped. Never `dangerouslySetInnerHTML` without sanitize. |
| **FK to `videos`/`analyses` would reject chat-before-save.** | C1 keys by `video_id text` with **no FK** (matches `analyses`), RLS by `user_id`. |
| **New frontend dependency** (react-markdown) is an Ask-first boundary. | Flagged in B1 and §5; do not install until confirmed at review. |
| **Streaming behind a proxy can buffer.** | Set `Cache-Control: no-cache` / `X-Accel-Buffering: no` on the StreamingResponse; note for deploy. |

## 4. Checkpoints (human/verify gates between phases)

- **Checkpoint A (after A2):** `python -m pytest tests/test_chat_endpoint.py` green; `cd frontend &&
  yarn test components.CoachTray` green; manual: a real signed-in send streams tokens into one
  assistant turn; a forced upstream error rolls back the turn and shows the error. **Gate:** streaming
  works end-to-end before adding markdown.
- **Checkpoint B (after B2):** frontend markdown render test + backend prompt test green; `yarn build`
  passes; manual: an answer with `**bold**`/a list renders formatted, partial markdown mid-stream is
  acceptable. **Gate:** rendering is safe (script stripped) before persistence.
- **Checkpoint C (after C4):** conversations store + endpoint tests green; chat restore/save test
  green; manual: send a turn, revisit the saved analysis from history → thread restored; delete-all
  leaves no conversation residue. **Gate:** persistence correct before close-out.
- **Checkpoint D:** full suite + `yarn build` + coverage ≥ 95% green; spec criteria ticked.

## 5. Decisions to confirm at review (carried from the spec)

1. **Markdown dep** — `react-markdown` + `rehype-sanitize` (default). Blocks **B1** only.
2. **Persistence granularity** — one thread per analysis, restored on replay (default). Shapes **C**.
3. **Links in markdown** — default **off** (text/emphasis/list/code only). Shapes **B1** sanitize schema.

Approve these (or redirect) and I'll start at **A1**.
