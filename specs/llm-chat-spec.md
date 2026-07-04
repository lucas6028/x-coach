# Spec: Conversational Coaching (LLM chat layer, OpenRouter)

Status: **v1 shipped · v2 shipped** · Owner: — · Supersedes the "chat disabled / coming soon" placeholder.

> v2 scope (this revision): **streaming · chat persistence/history · markdown output**. The v2
> section is at the bottom of this doc; the v1 spec above it is retained as the shipped baseline.
> Tool-calling live RAG, anonymous chat, and multi-analysis memory are deferred to v3.

## Objective

Add the deferred **LLM conversation layer**: after an analysis, a signed-in user can ask
follow-up questions ("why is my depth shallow?", "how do I fix the knee valgus?") and get a
coaching answer from an LLM served via **OpenRouter**.

The non-negotiable product constraint (from `frontend/APP_REDESIGN_PROMPT.md` §1, §10): **the
credibility of x-coach is its groundedness.** The chat must speak *only* from the analysis we
already computed — the same `detections` + `retrievals` (causes / risks / corrections) that
`ReasoningLog` renders. It must not invent faults, cues, or metrics that aren't in the analysis.

Success = a reviewer can ask a follow-up about a real analysis and get an answer that references
only the detected faults and retrieved knowledge, with the LLM key never exposed to the browser.

## Tech Stack

- **Backend**: FastAPI (existing `backend/app`), `httpx` (already transitive via supabase; pin
  it) for the OpenRouter call. OpenRouter is OpenAI-compatible (`POST {base}/chat/completions`,
  `Authorization: Bearer <key>`). No OpenAI/Anthropic SDK.
- **Frontend**: React 18 + Vite + TS, Tailwind 3 tokens, `motion/react`, i18n via `lib/i18n`.
- **Auth**: existing Supabase bearer-token flow; chat is **gated** (`get_current_user`).
- **Model**: env-configurable. Default `anthropic/claude-sonnet-5` (verified on OpenRouter's
  live model list; namespace is `vendor/model`). Key stays server-side.

## Commands

- Backend tests: `.venv/Scripts/python.exe -m pytest tests/test_chat_endpoint.py tests/test_backend.py`
- Backend run: `uvicorn backend.app.main:app --reload --port 8000` (from repo root, venv active)
- Frontend build: `cd frontend && yarn build`  · Frontend tests: `cd frontend && yarn test`

## Project Structure (touch points)

```
backend/app/settings.py              + openrouter_api_key / _model / _base_url + chat_configured
backend/app/services/chat.py         NEW: build grounding → call OpenRouter (deferred httpx import)
backend/app/routers/chat.py          NEW: POST /api/chat (get_current_user, 503 if unconfigured)
backend/app/main.py                  wire chat router
tests/test_chat_endpoint.py          NEW: contract + grounding + mocked network
.env.example                         + OPENROUTER_* keys (documented)
requirements.txt                     pin httpx

frontend/src/api.ts                  + chat() client method + ChatMessage/ChatContext types
frontend/src/lib/grounding.ts        NEW: buildChatContext(analysis) → compact grounding blob
frontend/src/components/ChatInput.tsx  rewrite: working chat panel (3 states below)
frontend/src/App.tsx                 pass analysis to <ChatInput/>
frontend/src/lib/i18n.tsx            + chat.* keys in BOTH en and zh-Hant
frontend/src/test/components.ChatInput.test.tsx  updated for the new states
```

## Data Contract

Client owns the `Analysis` already; it sends a **compact grounding blob** + the message history
each turn (backend is stateless, chat is ephemeral / client-held — no DB persistence in v1).

`POST /api/chat` (auth required)
```jsonc
{
  "messages": [ { "role": "user"|"assistant", "content": "..." } ],   // conversation so far, last = new user turn
  "context": {                                                        // built by buildChatContext(analysis)
    "video_id": "…", "view_type": "front", "view_confidence": 0.9,
    "fault_count": 2,
    "quality": { "lower_body_visibility_mean": 0.87, "valid_frame_ratio": 0.95 },
    "faults": [ {
      "fault_name": "knees_inward", "phase": "descent",
      "severity": 0.8, "start_time": 1.2, "end_time": 1.8,
      "evidence": "knee_valgus_ratio 0.82",
      "causes": ["…"], "risks": ["…"], "corrections": ["…"], "rag_snippet": "…"|null
    } ]
  }
}
```
Response: `{ "reply": "…", "model": "anthropic/claude-sonnet-5" }`

Errors: 401 (no session) · 503 (`OPENROUTER_API_KEY` unset) · 502 (OpenRouter unreachable/errored)
· 422 (bad body / empty messages).

## Grounding & honesty (the core requirement)

The backend builds the **system prompt** (never the client) from `context`:
1. Injects the analysis facts (view, quality, each fault with severity/phase/evidence and its
   retrieved causes/risks/corrections) as the sole source of truth.
2. Instructs: answer only from these facts; if asked about something not in the analysis, say it
   wasn't detected/measured rather than inventing it; keep advice to the retrieved corrections;
   be concise and encouraging; respond in the user's language.
3. On a **clean rep** (0 faults) → reinforce good form, don't manufacture problems.

## Code Style

Backend mirrors `services/store.py`: deferred heavy import, small patchable seam for tests.
```python
def answer(*, messages: list[dict], context: dict) -> dict:
    system = _build_system_prompt(context)
    reply = _chat_completion([{"role": "system", "content": system}, *messages])  # patched in tests
    return {"reply": reply, "model": get_settings().openrouter_model}
```

## Testing Strategy

- **Backend** (`tests/test_chat_endpoint.py`, unittest, network mocked like
  `test_analyze_endpoint.py`): 503 when unconfigured; empty-messages → 422; system prompt embeds
  the fault names + corrections and forbids invention; `_chat_completion` called with system +
  history; OpenRouter error → 502; clean-rep prompt has no fault list. Keep `backend/app`
  coverage ≥ 95% (CI gate).
- **Frontend** (`components.ChatInput.test.tsx`): unconfigured/logged-out → honest disabled
  state (preserves the existing disabled-input assertions); a light test for the send flow with
  `api.chat` mocked.

## Boundaries

- **Always**: run backend + frontend tests before declaring done; every user-facing string
  through `t()` in both dictionaries; only semantic Tailwind tokens; keep the LLM key server-side.
- **Ask first**: adding chat persistence (a Supabase migration), streaming/SSE, tool-calling
  retrieval, allowing anonymous chat, changing the default model tier.
- **Never**: expose `OPENROUTER_API_KEY` to the browser; let the LLM answer from anything but the
  provided analysis context; fake a working chat when the key/session is absent.

## Success Criteria

- [x] `POST /api/chat` returns a grounded reply for a real analysis; 503 without a key; 401 without a session.
- [x] System prompt is backend-built and constrains answers to the analysis (verified by test).
- [x] ChatInput: working when signed-in + server-configured; honest disabled state otherwise; both langs.
- [x] `yarn build` passes; backend suite green (backend/app coverage 100%); chat/grounding frontend tests green.
- [x] LLM key never reaches the client; no invented faults/metrics.

### Note added during implementation (spec kept alive)

Chat availability depends on **two independent** flags: Supabase auth (`auth_configured`) *and*
the server's OpenRouter key (`chat_configured`). `/api/health` now exposes both; `ChatInput`
reads `chat_configured` so a signed-in user whose backend lacks `OPENROUTER_API_KEY` gets the
honest disabled state, not a live input that 503s on every send. Pre-existing local test
`lib.supabase.test.ts` fails only when a populated `frontend/.env` is present (passes on CI where
`.env` is absent) — unrelated to this feature.

---

# v2: Streaming · Persistence · Markdown

Status: **in spec** — awaiting review before Phase 2 (Plan). The three v2 features build on the
shipped v1 endpoint/service/tray without changing the groundedness contract.

## Objective (v2)

Make the grounded coaching chat feel like a real conversation and survive a page reload:

1. **Streaming** — the coach's reply renders token-by-token instead of appearing all at once after
   a multi-second wait. Same grounded content; better perceived latency and a live "thinking" feel.
2. **Persistence / history** — a conversation is saved per analysis, so revisiting a saved analysis
   from history restores its chat thread instead of a blank composer.
3. **Markdown output** — the coach can answer with structure (bold cues, short lists, inline
   `code`/timecodes) and the client renders it safely, instead of a flat paragraph.

Success = a signed-in reviewer asks a follow-up and watches the answer stream in as formatted
markdown; reloads / revisits that saved analysis and the thread is still there; the groundedness,
honesty, and "LLM key never in the browser" guarantees from v1 are all preserved.

### Scope

- **In v2**: SSE streaming of `/api/chat`; per-analysis conversation persistence in Supabase
  (client-orchestrated, mirroring analyses persistence); markdown rendering of assistant turns.
- **Deferred to v3** (unchanged intent, explicitly out): tool-calling live RAG, anonymous chat,
  multi-analysis / cross-thread memory.

### ASSUMPTIONS (chosen defaults — correct at spec review or they stand)

1. **Markdown renderer**: add `react-markdown` + `rehype-sanitize` (a new frontend dependency —
   an "Ask first" boundary item; flagged here as the decision to confirm). LLM text is rendered as
   HTML, so sanitization is non-negotiable; `rehype-sanitize` strips scripts/handlers by default.
2. **Persistence granularity**: **one thread per `(user, video_id)`**, restored only via the
   persisted-analysis **history-replay** path. A fresh upload chats into an in-memory thread that
   becomes persisted once its analysis is saved; there is no standalone "my conversations" browser
   in v2 (that would be v3-shaped scope).
3. **Transport**: **replace** `/api/chat` with an SSE stream (one code path), rather than adding a
   parallel `/api/chat/stream`. The v1 buffered contract tests are updated, not kept in parallel.
4. **Persistence orchestration**: **client-driven**, mirroring `analyses` — the streaming endpoint
   stays stateless (generation only); separate `/api/conversations` endpoints save/restore the
   thread. This keeps the LLM seam testable in isolation and the DB layer patchable like `store.py`.
5. **Sync generator seam**: streaming uses a **sync** generator over `httpx.stream(...)`, iterated by
   `StreamingResponse` in a threadpool — least churn from v1's sync/deferred-import pattern.

## Build order (each depends on the prior)

**Streaming → Markdown → Persistence.** Streaming reshapes the endpoint contract and the service
seam first. Markdown is next and mostly independent, but must render *partial* content mid-stream.
Persistence is last because a turn can only be saved *after* the stream ends cleanly (non-empty).

## 1. Streaming (SSE) — the endpoint contract change

`POST /api/chat` becomes a `text/event-stream` response. Pre-flight validation still returns real
HTTP status **before the stream opens**; once bytes flush (200), status is frozen, so any later
failure is an **in-band** SSE error event.

### Error model (the part most likely to bite — nail it here)

| When | Condition | Surfaced as |
|---|---|---|
| Pre-flight | `chat_configured` false | HTTP **503** (before stream) |
| Pre-flight | no/expired session | HTTP **401** (before stream) |
| Pre-flight | last message not `user` / empty body | HTTP **422** (before stream) |
| **In-band** | OpenRouter connect/transport failure (even on the first chunk) | **in-band `error` event**, stream closes |
| **In-band** | OpenRouter drops/errors after chunks started | **in-band `error` event**, stream closes |
| **In-band** | completion accumulates to empty/blank | **in-band `error` event**; no assistant turn kept |

> **Refinement (during A1):** there is no pre-flight **502**. `StreamingResponse` commits the HTTP
> 200 (status + headers) *before* the first generator item is pulled, so an immediate OpenRouter
> connect failure is already inside a 200 stream — it surfaces as an in-band `error`, same as a
> mid-stream drop. Pre-flight HTTP is therefore only **503 / 401 / 422**. The client handles in-band
> errors uniformly (roll back the optimistic turn), so this is simpler *and* more honest.

**SSE event shape** (one event type per line-block; keep it minimal and explicit):
```
event: delta                       // zero or more; token text
data: {"text": "…"}

event: done                        // exactly one on success
data: {"model": "anthropic/claude-sonnet-5"}

event: error                       // instead of `done` on mid-stream failure
data: {"detail": "OpenRouter request failed: …"}
```

### Service seam (redesign for the coverage gate)

`backend/app` is held near 100% (CI gate) and v1 tests patch `_chat_completion`. Replace that seam
with a **generator** `_stream_completion(messages) -> Iterator[str]` (deferred `httpx` import,
`httpx.stream(...)` parsing OpenAI-compatible SSE lines). Tests patch it to yield a fixed sequence,
so **normal stream / mid-stream error / empty stream** are all drivable with no network:
```python
def answer_stream(*, messages, context):          # generator the router wraps in StreamingResponse
    system = _build_system_prompt(context)         # unchanged from v1
    acc = []
    try:
        for chunk in _stream_completion([{"role": "system", "content": system}, *messages]):
            acc.append(chunk)
            yield _sse("delta", {"text": chunk})
    except RuntimeError as exc:
        yield _sse("error", {"detail": str(exc)}); return
    if not "".join(acc).strip():                    # v1 empty-completion invariant, preserved
        yield _sse("error", {"detail": "OpenRouter returned an empty message."}); return
    yield _sse("done", {"model": get_settings().openrouter_model})
```
The empty-completion invariant (v1 `chat.py:129-139`) **must survive**: a blank completion emits an
`error`, never a `done`, so the client never stores an empty assistant turn that the next send's
`content min_length=1` validator would reject — the wedge v1 guarded against.

### Client (CoachTray) changes

- `api.chat(...)` returns a stream reader instead of a resolved `ChatResponse`. Consume `delta`
  events, appending to the in-progress assistant turn; finalize on `done`; on `error`, roll back the
  optimistic user turn (as v1 does) and show `t("chat.error")` / `t("chat.sessionExpired")`.
- `ChatError(status)` still covers the **pre-flight** 401/5xx (thrown before the stream opens); a new
  in-band-error path handles mid-stream failures. Both funnel into the existing error UI.

## 2. Markdown output

- System prompt: add one line permitting light markdown (bold, short lists, inline code/timecodes) —
  it must **not** relax any grounding rule.
- Render assistant turns with `react-markdown` + `rehype-sanitize` (replacing the raw `<p>` at
  `CoachTray.tsx:283`). User turns stay plain text. Renderer must tolerate **partial/incomplete
  markdown** during streaming (react-markdown re-parses each frame; acceptable).
- Constrain rendered elements to a coaching-safe subset via the sanitize schema (no raw HTML, no
  images/links unless we decide to allow links); style through the existing Tailwind tokens.

## 3. Persistence / history

Client-orchestrated, mirroring `analyses`. The generation endpoint stays stateless.

### Schema — new migration `db/migrations/<ts>_conversations.sql`

Follow the `analyses` precedent **exactly**: `analyses.video_id` is `text not null` with **no FK**
(only `user_id → auth.users`). A user can upload-and-chat *before* any video/analysis row is
persisted, so the conversations table must **not** FK to `videos`/`analyses` — key it by `video_id`
text, RLS-scoped by `user_id`, messages stored as JSONB (mirroring `analyses.result`):
```sql
create table if not exists public.conversations (
    id         uuid primary key default gen_random_uuid(),
    user_id    uuid not null references auth.users (id) on delete cascade,
    video_id   text not null,                 -- matches analyses.video_id (per user); no FK, by design
    messages   jsonb not null default '[]',   -- [{role, content}], mirrors client ChatMessage[]
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (user_id, video_id)
);
-- RLS: conversations_owner_all, authenticated, using/with check (auth.uid() = user_id).
-- touch updated_at via the existing public.touch_updated_at() trigger.
```

### Store functions — `backend/app/services/store.py` (patchable, like the rest)

- `upsert_conversation(*, token, user_id, video_id, messages) -> None` — upsert on `(user_id, video_id)`.
- `get_conversation(*, token, video_id) -> {messages} | None` — restore a thread.
- Deletion rides along: `delete_all_analyses` also clears the caller's conversations (no residue).

### Endpoints — `backend/app/routers/` (gated by `get_current_user`, like analyses)

- `PUT /api/conversations/{video_id}` — save the full thread (idempotent upsert). 401 without session.
- `GET /api/conversations/{video_id}` — restore, or `{messages: []}` / 404 when none.

### Client orchestration

- After a turn **streams to completion (`done`)**, the client PUTs the updated thread. A mid-stream
  `error` (nothing persisted) leaves the DB untouched — matches the roll-back of the optimistic turn.
- On **history-replay** load of a saved analysis, GET the conversation and seed `messages`. A fresh
  upload keeps v1's behavior (empty thread) until its analysis — and thus its conversation — is saved.
- v1's "new analysis → reset thread" effect (`CoachTray.tsx:51-54`) is replaced by "load the saved
  thread if present, else empty".

## Data contract (v2 deltas)

- `POST /api/chat` request body is **unchanged** (`{messages, context}`); the **response** changes
  from `{reply, model}` JSON to a `text/event-stream` of `delta` / `done` / `error` events (above).
- New `Conversation` shape: `{ video_id, messages: ChatMessage[] }` (frontend `api.ts` types +
  `PUT/GET /api/conversations/{video_id}`).

## Testing Strategy (v2)

- **Backend streaming** (`tests/test_chat_endpoint.py`, network mocked): patch `_stream_completion`
  to drive (a) a normal multi-chunk stream → ordered `delta`s then one `done` with the model; (b) a
  mid-stream `RuntimeError` → `delta`s then an `error`, no `done`; (c) an empty/blank accumulation →
  `error`, no `done`. Pre-flight still returns **503** unconfigured, **401** no session, **422** on a
  non-user last message — asserted on the response *before* any stream body. Keep `backend/app`
  coverage ≥ 95% (CI gate); the generator's three branches must all be exercised.
- **Backend persistence** (`tests/test_conversations_endpoint.py`, `store` patched like the analyses
  tests): PUT upserts and GET restores; both **401 without a session**; delete-all clears
  conversations; RLS reliance documented (user-JWT path, same as `store.py`).
- **Frontend** (`components.CoachTray.test.tsx`, `api.chat` mocked to yield a fake stream): deltas
  accumulate into one assistant turn; markdown renders (`**bold**` → `<strong>`, a `<script>` is
  sanitized away); an in-band `error` rolls back the optimistic user turn and shows the error; a
  restored thread renders on replay. Preserve v1's honest disabled/sign-in composer assertions.

## Boundaries (v2 deltas — v1 boundaries otherwise stand)

- **Always**: run backend + frontend tests before done; every new string through `t()` in **both**
  dictionaries; only semantic Tailwind tokens; keep the LLM key server-side; sanitize all rendered
  markdown.
- **Ask first** (explicit confirmations this revision needs): **adding `react-markdown` +
  `rehype-sanitize`** (new deps); **the `conversations` Supabase migration** (schema change);
  switching the default model tier.
- **Never**: expose `OPENROUTER_API_KEY` to the browser; let the LLM answer from anything but the
  provided analysis context; render unsanitized LLM HTML; persist an empty assistant turn; return a
  post-stream HTTP error code once the stream has started (must be in-band).

## Success Criteria (v2)

- [x] `/api/chat` streams `delta`/`done` SSE for a real analysis; pre-flight 503/401/422 still fire
      before the stream; a mid-stream failure and an empty completion both emit an in-band `error`
      and persist nothing. *(A1; `test_chat_endpoint.py` AnswerStreamTests / ChatRouterTests.)*
- [x] Assistant turns render sanitized markdown (bold/list/inline code); a `<script>` payload is
      inert; partial markdown renders mid-stream without throwing. *(B1; `components.Markdown.test.tsx`.)*
- [x] Revisiting a saved analysis restores its chat thread; a fresh upload starts empty; delete-all
      leaves no conversation residue. *(C2/C4; store + CoachTray restore/persist tests.)*
- [x] `yarn build` passes; backend suite green (`backend/app` coverage 99.8% ≥ 95%); new streaming +
      persistence + markdown tests green. *(i18n keys unchanged — v2 reused v1's `chat.*` strings.)*
- [x] LLM key never reaches the client; no invented faults/metrics; groundedness unchanged from v1.
      *(B2 test asserts the markdown-permission line does not relax the ONLY / Do-NOT-invent rules.)*

## Open Questions — resolved

- **Markdown dependency** → shipped with `react-markdown@10` + `rehype-sanitize@6` (assumption #1),
  per the plan default the user approved via `/build auto`. Reversible via `git revert` if redirected.
- **Persistence granularity** → one thread per `(user, video_id)`, restored via history-replay
  (assumption #2). No standalone conversations browser (that stays v3-shaped).
- **Link rendering** → **no links in v2**. The sanitize schema derives from rehype-sanitize's default
  and additionally drops `<a>`/`<img>`; `components.Markdown.test.tsx` asserts a markdown link renders
  as plain text with no `<a>`.

## Build notes (kept alive)

- **No pre-flight 502.** `StreamingResponse` commits the 200 before the first generator item, so
  *all* OpenRouter failures (connect/mid-stream/empty) are in-band `error` events; only 503/401/422
  are pre-flight HTTP. See the refinement note under the v2 error table.
- **Persistence is FK-free by design.** `conversations.video_id` is `text` with no FK (mirrors
  `analyses`), so a fresh upload can be chatted and saved before any analysis/video row exists.
- **Landed A1→D1** on branch `feat/llm-chat-v2`, one commit per task. Backend `backend/app` coverage
  99.8%. Pre-existing, unrelated local failures (not caused by v2): `test_backend_analysis.py`
  (missing Squat dataset — CI `--ignore`s it) and `lib.supabase.test.ts` (fails only with a populated
  local `frontend/.env`; passes on CI).

## Out of scope (v3)

Tool-calling live RAG, anonymous chat, multi-analysis / cross-thread memory.
