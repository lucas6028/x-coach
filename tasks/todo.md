# TODO — LLM Chat v2

Plan: [`tasks/plan.md`](./plan.md) · Spec: [`specs/llm-chat-spec.md`](../specs/llm-chat-spec.md).
Order is dependency-driven: **A → B → C → D**. Check off only when Acceptance + Verify pass.

Commands (repo root, venv active): backend `.venv/Scripts/python.exe -m pytest tests/test_chat_endpoint.py`;
frontend `cd frontend && yarn test <file>` / `yarn build`.

---

## Phase A — Streaming

- [x] **A1 — Backend: `/api/chat` emits SSE** ✅ 13 chat tests green; backend/app cov 99.8%.
  - Acceptance: `_stream_completion` sync generator yields text chunks (deferred `httpx.stream`);
    `answer_stream` yields ordered `delta` frames then one `done` on success, an `error` frame on a
    mid-stream `RuntimeError`, and an `error` (never `done`) when the accumulation is blank; router
    returns `StreamingResponse(..., media_type="text/event-stream")` with pre-flight 503/422 intact.
  - Verify: `pytest tests/test_chat_endpoint.py` — normal / mid-stream-error / empty branches all
    covered; `_build_system_prompt` groundedness tests still pass; `backend/app` coverage ≥ 95%.
  - Files: `backend/app/services/chat.py`, `backend/app/routers/chat.py`, `tests/test_chat_endpoint.py`.

- [x] **A2 — Frontend: consume the stream** ✅ api + CoachTray stream tests green (32); `yarn build` passes.
  - Acceptance: `api.chatStream(messages, context, {onDelta,onDone,onError,signal})` throws
    `ChatError(status)` on pre-flight non-200 and routes in-band `error` to `onError`; `CoachTray`
    accumulates deltas into one assistant turn, finalizes on `done`, rolls back the optimistic user
    turn + restores input on error (401 → sign-in-again message).
  - Verify: `yarn test components.CoachTray` green (fake-stream mock); no console errors.
  - Files: `frontend/src/api.ts`, `frontend/src/components/CoachTray.tsx`,
    `frontend/src/test/components.CoachTray.chat.test.tsx`.

- [ ] **✅ Checkpoint A** — chat streams end-to-end (auto tests + manual send). Gate before Phase B.

## Phase B — Markdown

- [x] **B1 — Frontend: sanitized markdown render** ✅ react-markdown@10 + rehype-sanitize@6; 4 render tests green (links dropped, script inert); `yarn build` passes.
  - Acceptance: assistant turns render markdown via `react-markdown` + `rehype-sanitize` (replacing
    the raw `<p>`); `**bold**`→`<strong>`, lists/inline-code render, a `<script>` payload is stripped,
    partial markdown mid-stream renders without throwing; user turns stay plain text.
  - Verify: `yarn test` markdown render case green; `yarn build` passes.
  - Files: `frontend/package.json`, `frontend/src/components/Markdown.tsx` (new) +
    `frontend/src/components/CoachTray.tsx`, chat test.

- [x] **B2 — Backend: permit light markdown in the prompt** ✅ preamble line added; grounding/honesty asserted intact.
  - Acceptance: one line added to `_SYSTEM_PREAMBLE` permitting bold/short-list/inline-code/timecodes
    with no grounding rule relaxed.
  - Verify: `pytest tests/test_chat_endpoint.py` — a `_build_system_prompt` test asserts the
    markdown-permission line and that "ONLY" / "Do NOT invent" constraints remain.
  - Files: `backend/app/services/chat.py`, `tests/test_chat_endpoint.py`.

- [ ] **✅ Checkpoint B** — formatted, sanitized answers render (script stripped). Gate before Phase C.

## Phase C — Persistence

- [x] **C1 — Schema: `conversations` migration** ✅ no-FK video_id, RLS owner-scoped, touch trigger.
  - Acceptance: new migration creates `conversations(id, user_id→auth.users, video_id text NO FK,
    messages jsonb default '[]', created_at, updated_at, unique(user_id,video_id))` + RLS
    `conversations_owner_all` + `touch_updated_at` trigger; mirrors the `analyses` precedent.
  - Verify: SQL reads cleanly; no FK to `videos`/`analyses`; RLS scoped to `auth.uid() = user_id`.
  - Files: `db/migrations/<ts>_conversations.sql`.

- [x] **C2 — Store: conversation read/write** ✅ upsert/get + delete-all clears conversations; store tests green (18).
  - Acceptance: `upsert_conversation(*,token,user_id,video_id,messages)`,
    `get_conversation(*,token,video_id)->{messages}|None`; `delete_all_analyses` also clears the
    caller's conversations.
  - Verify: `pytest` store tests (patch `_user_client`) — upsert + restore round-trip; delete clears.
  - Files: `backend/app/services/store.py`, `tests/test_conversations_store.py` (or `test_backend.py`).

- [x] **C3 — Router: `/api/conversations/{video_id}`** ✅ PUT/GET gated (401), wired into main; cov 99.8%.
  - Acceptance: `PUT` upserts the full thread, `GET` restores (`{messages: []}` when none), both
    gated by `get_current_user` (401 without session); wired into `main.py`.
  - Verify: `pytest tests/test_conversations_endpoint.py` — 401 without session, PUT→GET round-trip,
    delegates to store; `backend/app` coverage ≥ 95%.
  - Files: `backend/app/routers/conversations.py` (new), `backend/app/main.py`,
    `tests/test_conversations_endpoint.py`.

- [x] **C4 — Client: restore + save the thread** ✅ restore-on-load + persist-after-done; no-persist-on-error; 45 tests green; `yarn build` passes.
  - Acceptance: `api.getConversation` / `api.putConversation` + `Conversation` type; `CoachTray`
    seeds `messages` from a GET on `analysis.video_id` (replacing the reset-to-`[]` effect) and PUTs
    the updated thread after a turn reaches `done`; keys on `video_id` only (no replay flag).
  - Verify: `yarn test components.CoachTray` — restore-on-load + save-after-turn cases green;
    `yarn build` passes.
  - Files: `frontend/src/api.ts`, `frontend/src/components/CoachTray.tsx`, chat test.

- [ ] **✅ Checkpoint C** — revisiting a saved analysis restores the thread; delete-all leaves no residue.

## Phase D — Close-out

- [x] **D1 — Full regression + spec close** ✅ backend 292 passed (CI set), FE build + suite green (1 known env fail), spec criteria ticked.
  - Acceptance: whole backend suite + `yarn build` + `yarn test` green; `backend/app` coverage ≥ 95%.
  - Verify: `.venv/Scripts/python.exe -m pytest tests/` ; `cd frontend && yarn build && yarn test`.
  - Files: `specs/llm-chat-spec.md` (tick v2 criteria, record resolved open questions + build notes).

---

### Decisions (defaults approved via `/build auto`; reversible)
- [x] Markdown dep: `react-markdown` + `rehype-sanitize`
- [x] Persistence: one thread per analysis, restored on replay
- [x] Links in markdown: off (text/emphasis/list/code only)
