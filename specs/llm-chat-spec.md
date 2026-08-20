# Spec: Conversational Coaching (LLM chat layer)

Status: **v1 shipped · v2 shipped** · Owner: — · Supersedes the "chat disabled / coming soon" placeholder.

> v2 scope (this revision): **streaming · chat persistence/history · markdown output**. The v2
> section is at the bottom of this doc; the v1 spec above it is retained as the shipped baseline.
> Tool-calling live RAG, anonymous chat, and multi-analysis memory are deferred to v3.

## Objective

Add the deferred **LLM conversation layer**: after an analysis, a signed-in user can ask
follow-up questions ("why is my depth shallow?", "how do I fix the knee valgus?") and get a
coaching answer from an LLM served via a **configurable OpenAI-compatible provider** (OpenRouter by
default via `LLM_BASE_URL`; any peer such as NVIDIA NIM works by swapping base URL + key + model ids).

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
backend/app/settings.py              + llm_api_key / _models / _base_url / _followup_model + chat_configured
backend/app/services/chat.py         NEW: build grounding → call LLM provider (deferred httpx import)
backend/app/routers/chat.py          NEW: POST /api/chat (get_current_user, 503 if unconfigured)
backend/app/main.py                  wire chat router
tests/test_chat_endpoint.py          NEW: contract + grounding + mocked network
.env.example                         + LLM_* keys (documented)
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

Errors: 401 (no session) · 503 (`LLM_API_KEY` unset) · 502 (LLM provider unreachable/errored)
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
    return {"reply": reply, "model": default_chat_model()}
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
- **Never**: expose `LLM_API_KEY` to the browser; let the LLM answer from anything but the
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
reads `chat_configured` so a signed-in user whose backend lacks `LLM_API_KEY` gets the
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
data: {"detail": "LLM request failed: …"}
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
        yield _sse("error", {"detail": "The LLM returned an empty message."}); return
    yield _sse("done", {"model": default_chat_model()})
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
- **Never**: expose `LLM_API_KEY` to the browser; let the LLM answer from anything but the
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

---

# v2.1: Per-answer follow-up suggestions

Status: **shipped**. After each coach answer, the tray offers **two grounded next-question chips**
— dynamically generated from that answer (like the opening starter chips, but contextual), matching
the request "在每次 LLM 回答後，加入 2 個 follow up questions, 和一開始的選項一樣".

> **Design history.** This shipped in three iterations: (1) a blocking second call after the answer;
> (2) a single-request *sentinel split* (answer→`===FOLLOWUPS===`→JSON array in one stream) to kill the
> second round-trip; then (3) the **fire-and-forget** design below, after the sentinel's worst-case
> (a model emitting the array *without* the sentinel would stream raw `["…"]` into the answer, un-retractable)
> was judged too fragile. Fire-and-forget keeps the low perceived latency without that failure mode.

- **Generation — separate, fire-and-forget request.** `POST /api/chat` streams **only** the answer
  (`delta`/`done`/`error`, the clean v2 contract, closes the instant the answer is done). Follow-ups
  are a distinct endpoint, `POST /api/chat/followups` → `{questions: [...]}`, backed by
  `chat.py:suggest_followups` (grounded system prompt + `_FOLLOWUP_INSTRUCTION` + a trailing user
  `_FOLLOWUP_NUDGE`, tight `_FOLLOWUP_TIMEOUT_S`). Grounding unchanged — same analysis facts +
  "Do NOT invent" rule, so a suggestion can't reference a fault outside the analysis.
- **Client (`CoachTray`).** The answer commits and the composer re-enables the moment the stream
  ends (unchanged commit/rollback path). Immediately after committing, the client fires
  `api.chatFollowups(thread, context, model)` **without awaiting it** and drops the chips in when it
  resolves. A `followupSeq` ref guards the race: a slow result is applied only if its turn is still
  the latest (a newer send or a switched analysis bumps the seq and the stale result is ignored).
  `followups` state is ephemeral (not persisted); chips render via the shared `chipClass`.
- **Why this over the sentinel:** robustness. The answer path is a clean stream again (no delimiter
  to leak), a slow/failed suggestion can't delay or corrupt the answer (it's off the critical path),
  and `suggest_followups` swallows everything to `[]`. Cost: a second HTTP request + round-trip — but
  the user never waits on it, so perceived latency matches the sentinel version. The one real downside
  vs sentinel: `/api/chat/followups` re-sends the grounding blob and makes a second model call (mitigable
  later with prompt caching). Verified live (fault + clean-rep) — deepseek/minimax/xiaomi all return two
  grounded questions.
- **Latency tuning (chip snappiness).** The chips were measured at 3–10s wall-clock — traced to two
  causes: (1) OpenRouter's *provider routing* swinging the **same** model 2s→9s call-to-call, and (2)
  *reasoning* answer models (minimax) spending 6s "thinking" before emitting the tiny array; output
  length is not the bottleneck (TTFT is). Fix: the follow-up call is pinned to a fast model
  (`LLM_FOLLOWUP_MODEL`, default `openai/gpt-oss-120b`) via `settings.followup_chat_model` —
  **independent of the answer model the user picked** (the `/chat/followups` router ignores `body.model`)
  — and sends `provider: {sort: "latency"}` (`chat._FOLLOWUP_ROUTING`, merged via `_stream_completion`'s
  `extra_body`). Measured effect: a consistent **~1.5s** end-to-end (was 3–10s), verified over the live
  endpoint. `max_tokens` was tried and rejected — gpt-oss is a reasoning model and the cap gets consumed
  by hidden reasoning tokens, yielding empty content.
- **Tests**: `backend/app/services/chat.py` + `routers/chat.py` coverage 100% (`SuggestFollowupsTests`,
  `ParseFollowupsTests`, `ChatRouterTests` cover the endpoint's 503 + resolved-model + thread-ending-on-
  assistant; `AnswerStreamTests` locks that the answer stream carries no followups); frontend
  `api.test.ts` (`chatFollowups`) + `components.CoachTray.chat.test.tsx` (fire-and-forget chips render +
  click-to-send + clear-on-send). No new i18n — question text is model-generated.

## Out of scope (v3)

Anonymous chat, multi-analysis / cross-thread memory. (Tool-calling live RAG **moved into v3** —
see below.)

---

# v3: Tool-calling loop

Status: **built** (branch `feat/chat-tool-calling`). The coach becomes an agent over three server-side tools —
`get_analysis`, `kg_query`, `rag_search` — instead of a single grounded completion. This is the
"tool-calling live RAG" line the v2.1 out-of-scope list deferred.

## Objective (v3)

Today the coach can only speak from the compact blob `buildChatContext` shipped: a summarised
causes/risks/corrections list plus the single top RAG snippet per fault. It cannot answer
"我第 2 rep 的膝蓋實際幾度" (the full `evidence` dict was compressed away), nor "腳踝活動度不足會怎樣"
(nothing in the analysis retrieved it). Function calling lets the model **pull** the detail and the
knowledge it needs, on demand, without widening the prompt for every turn.

## Decisions (settled at design review — these stand)

1. **The fat prompt stays.** `_build_system_prompt` is unchanged; tools are added *alongside* it.
   The alternative (thin prompt + pull-on-demand) is the bigger token win but rewrites the grounding
   contract, and the CLEAN REP / NOT MEASURED honesty invariants (`chat.py:148-167`) are built on
   that prompt. Not worth risking for this iteration.
2. **`get_analysis` reads from the request, not the DB.** The client already holds the full
   `Analysis`; `ChatContext` gains a `detail` field carrying it. No `user.token` threading into the
   service, no new `store` function, no UUID validation, no IDOR surface. It also guarantees the
   tool and the on-screen analysis are the same document.
3. **Tool round-trips are ephemeral.** They live and die inside one `/api/chat` request.
4. **Named tool progress is shown to the user** (`event: tool`), not a generic spinner.
5. **Tool support is discovered by failure**, not by a capability list or a probe of
   OpenRouter `/models`.

## Non-goals (v3)

Persisting tool turns; `role:"tool"` in the thread contract; tools on the follow-up path; a
tool-capability column in the model picker; letting the model reach *other* analyses.

## 1. Transport: refactor under `_stream_completion`, not through it

`_stream_completion` yields `str` and drops `tool_calls`/`finish_reason` (`chat.py:295`). It has 13
test reference points and is shared with `suggest_followups`, whose ~1.5s chip latency is a measured,
defended property. So the layer is split **underneath** it and its signature is left alone:

```
_stream_raw_chunks(...) -> Iterator[dict]   # NEW. httpx + SSE line parsing only; yields parsed chunk dicts.
_stream_completion(...) -> Iterator[str]    # EXISTING, now a thin shell over the above. Signature, behaviour,
                                            # and every existing test unchanged.
_stream_turn(...)       -> TurnResult       # NEW. Consumes raw chunks; accumulates assistant text,
                                            # tool_calls, and finish_reason for ONE round.
```

`TurnResult` carries the assistant text, the reassembled `tool_calls`, and `finish_reason`.

**A round that narrates *and* calls a tool must retract its narration.** Many models emit prose
before a tool call ("讓我查一下知識圖譜…"). If that text is streamed as `delta` frames it gets
concatenated with the real answer the model writes after the tool result — on screen and in the
persisted assistant turn.

The obvious fix — "only the final round streams" — **does not work**, because `finish_reason` only
arrives at the *end* of a round. Whether a round is final is undecidable while it is in flight, so
that rule degenerates into "buffer every round and flush at the end", which costs token-by-token
streaming on the **most common path of all**: the turn that calls no tools. That is a worse
regression than the bug it fixes.

So: **stream optimistically, and retract on the rare collision.** A round emits `delta` frames as
text arrives. If that same round turns out to have produced `tool_calls`, the server emits a
`reset` frame (§5) before the `tool` frames; the client clears its accumulator and the streaming
bubble, and the narration goes into the message array as the assistant turn preceding the tool
results — which is what the OpenAI-compatible contract expects anyway. Normal turns stream exactly
as they do today; only the narrate-then-call collision pays anything.

`reset` is safe because the client's streaming buffer is transient: `CoachTray` commits the
assistant turn only after the stream ends, so nothing user-visible has been persisted at retraction
time.

Consequence for §2's retry precondition: "no `delta` has been yielded yet" becomes a genuine runtime
check rather than a structural guarantee, and a `reset` re-arms it (after retraction the client is
back to a clean slate, so a retry is legal again).

**Fragment reassembly is the sharp edge.** Streamed tool calls arrive split: `delta.tool_calls[i]`
carries `id` and `function.name` typically only on the first chunk, and `function.arguments` as
successive JSON string fragments. Accumulate keyed by `index`, never by position of arrival. This
gets its own tests (see §7); it is the single most likely source of a silent bug here.

`_stream_raw_chunks` raises `_LLMError(RuntimeError)` carrying `.status`, so the caller can tell a
4xx (model rejects `tools`) from a transport failure. `tools`/`tool_choice` are standard
OpenAI-compatible fields and are **not** gated by `_is_openrouter`; the `provider` routing body still is.

## 2. The loop

Server-side, inside `answer_stream`, entirely within one request:

- **Max 3 tool rounds.** Three tools, one call each, is the realistic ceiling. **Hitting this cap does
  not error.** Issue one final request with `tools` omitted so the model answers in prose from
  whatever it already gathered — the user gets an answer, not a failure. That extra request is cheap
  (no `tools` field, no further round-trips possible after it) and there is always budget for it: it
  runs before the time cap below, never after.
- **Cumulative wall-clock budget = `chat_timeout()`**, shared across all rounds — *not* a fresh
  budget per round. This endpoint is metered; N rounds must not cost N× the timeout. Each individual
  upstream request is given a per-request timeout of the *remaining* budget, so the sum of the round
  timeouts can never exceed the whole. **Hitting this cap DOES error.** If the remaining budget is
  exhausted before the next round can even start, the loop stops with an honest in-band `error`
  rather than issuing a request that is certain to time out.

  **The two caps behave differently, and that is deliberate, not an oversight.** The round cap always
  has a cheap, fast answer waiting on the other side of one more request — `tools=None` costs nothing
  extra and cannot itself trigger another round, so there is no way for the forced round to blow the
  budget. The time cap has, by construction, no budget left for that same request: issuing it anyway
  would not buy an answer, only spend more of a metered budget on a call already destined to time out
  — trading an honest, immediate failure for a slower one that still fails, and costs more. Do not
  "fix" this into a single rule that always issues a final request; that would restore the guaranteed
  timeout this section exists to avoid.
- **A tool that raises is not fatal.** Feed the error text back as that tool's result and let the
  model say it couldn't look it up. The stream never dies over a missing KG file.
- **An unknown tool name** (model hallucinates one) is handled the same way: an error result, not a
  crash.

**Degradation for models without function calling.** Always send `tools`. On a 4xx **and only if no
`delta` has been yielded yet**, retry the request once without `tools`, reverting to today's exact
behaviour. The "nothing yielded yet" precondition is hard: once streaming starts the HTTP 200 is
committed and retrying would double-emit. A 400 is raised at request time, before any delta, so the
retry path is reachable in practice. Zero configuration, self-healing, and no dependency on
OpenRouter-only metadata (an OpenAI-compatible peer such as NIM has no `/models` capability API).

## 3. Tool catalogue

| Tool | Arguments (all clamped server-side) | Backed by |
|---|---|---|
| `get_analysis` | `fault_name: str \| null`, `include: "evidence" \| "knowledge" \| "all"` | pure read of `context.detail` |
| `kg_query` | `query: str`, `hops: int` (clamped 1–2) | `knowledge.graph_context(..., movement=<thread movement>)` |
| `rag_search` | `query: str`, `top_k: int` (clamped 1–8) | `knowledge.rag_snippets(...)` |

- **Model-supplied numbers are untrusted input.** Clamp `hops` and `top_k` exactly the way
  `routers/knowledge.py:20,38` bounds its query params — an unbounded or negative value must not
  reach retrieval.
- **`kg_query` is forced to the thread's `movement`.** Without it the KG returns knowledge for a
  different exercise and the coach will present it as relevant.
- **Every tool result is truncated** to a per-result character cap before going back into the
  message array. RAG chunks are full-text and will otherwise blow the context on round 2.
- `get_analysis`'s `include` selects which half of a fault's detail to return: `"evidence"` is the
  detection's full `evidence` dict plus its frame/time/severity/confidence fields; `"knowledge"` is
  that fault's full `retrievals` context (subgraph nodes+edges, all RAG chunks with text and score);
  `"all"` is both. With `fault_name=null` it returns the analysis-level material instead (metadata,
  quality, view, and a one-line summary per detection).
- Both knowledge calls are **synchronous and local**. `StreamingResponse` already runs this
  non-async generator in a threadpool, so they are called inline; no async plumbing.

## 4. Grounding: the new red line

Live retrieval introduces a failure mode the v1/v2 prompt never had — `kg_query` and `rag_search`
return knowledge about faults **this rep did not exhibit**. Nothing in the current rules stops the
model from relaying a retrieved fault as an observation. A rule of the same rank as the existing
GROUNDING RULES is added:

> Knowledge returned by tools is **reference material, not an observation about this video**. If
> `kg_query` or `rag_search` mentions a fault, that does **not** mean the user committed it. Only the
> faults listed in ANALYSIS FACTS were actually detected.

Everything else in `_build_system_prompt` is byte-identical, including the CLEAN REP and NOT MEASURED
branches and their tests.

## 5. SSE contract (v3 delta)

Two new frames; `delta` / `done` / `error` are untouched.

```
event: tool
data: {"name": "kg_query", "query": "knee valgus"}

event: reset
data: {}
```

`tool` is emitted once per tool call, at call time. No terminating counterpart — the next `delta`
implies the work finished.

`reset` tells the client to discard everything streamed so far this turn (see §1). It is emitted
only when a round produced both text and tool calls, immediately before that round's `tool` frames.

`dispatchSSE` (`api.ts:363-379`) is an if/else-if chain that silently ignores unknown events, so an
old client tolerates both frames without crashing. Note the caveat: an old client that ignores
`reset` would render the narration concatenated with the answer on a collision turn, so **ship the
client change in the same release**, even though the wire format itself is backward-compatible.

## 6. Client

- `ChatStreamHandlers` gains an optional `onTool` and an optional `onReset`.
- `CoachTray` renders a transient status line above the streaming bubble ("搜尋知識圖譜:knee
  valgus"), cleared on the first `delta` of the final answer. `onReset` clears both the local
  accumulator and the `streaming` state.
- i18n for the three tool display names; the `query` string is rendered verbatim (model-generated,
  like the follow-up chips).
- `buildChatContext` (`lib/grounding.ts`) adds `detail`: the full `detections` + `retrievals`,
  **excluding `pose`** (the heavy block, and useless to the coach).
- `ChatContext.detail` is **not persisted** — `upsert_conversation` stores messages + followups only,
  never the context. The cost of `detail` is HTTP body size, not tokens or storage: it enters the
  prompt only when a tool actually returns it.
- **`/api/chat/followups` sends `detail: undefined`.** It shares `ChatRequest`, so it would otherwise
  re-upload the whole blob — dominated by `retrievals[].context.results[].text` (full RAG chunks,
  the bulk of the payload on a multi-fault clip) — on a fire-and-forget call that can never use it.
  `detail` is optional on the model precisely so this call can omit it. Chip latency is a defended
  property; nothing gets added to that path.
- `suggest_followups` sends **no** `tools`. Chips stay on the fast pinned model at ~1.5s.

## 7. Testing (v3)

Backend (`tests/test_chat_endpoint.py`, `unittest.TestCase`):

- Fragment reassembly: `arguments` split across chunks; `id`/`name` present only on the first chunk;
  two concurrent calls interleaved and keyed by `index`.
- Loop bounds: stops at 3 rounds; the forced final tools-free request is issued; the cumulative
  timeout trips before a 4th round.
- Degradation: 4xx before any delta → one retry without `tools`; 4xx *after* a delta → in-band
  `error`, no retry.
- Per tool: dispatch, argument clamping (`hops`, `top_k` out of range and non-numeric), unknown tool
  name, a tool that raises.
- Narration on a collision round is retracted: a round returning both prose and `tool_calls` emits
  its `delta` frames, then a `reset` **before** the `tool` frames, and its text lands in the message
  array handed to the next round. A round with tool calls but no prose emits **no** `reset`.
- A normal no-tool turn emits its `delta` frames and no `reset` (regression lock: streaming is not
  buffered on the common path).
- The `tool` SSE frame shape.
- `suggest_followups` sends no `tools` (regression lock on the chip path).
- `_stream_completion`'s existing tests must pass **unmodified**. Necessary but *not sufficient*:
  four of them (`test_chat_endpoint.py:409,439,462,484,492`) mock `httpx.stream`, so they sit above
  the new split and would still pass over a raw layer that silently dropped frames the old parser
  handled. The raw layer therefore gets its own direct tests for the three behaviours currently
  fused into `chat.py:288-299`: the `[DONE]` terminator, non-`data:` keep-alive/comment lines
  skipped, and malformed-frame tolerance.

**Decide where malformed-frame tolerance lives, and write it down.** Today the
`except (JSONDecodeError, KeyError, IndexError, TypeError): continue` guard sits in the *same* `try`
as content extraction, so "unparseable JSON" and "well-formed chunk with no content" are handled
identically. After the split they must not be: `_stream_raw_chunks` owns **only** JSON-level
tolerance (skip a chunk that will not parse), and `_stream_turn` owns shape tolerance (a parsed chunk
with no `choices`, no `delta`, or neither `content` nor `tool_calls`). If the raw layer silently
swallows a partially-delivered chunk instead, a dropped `tool_calls` fragment corrupts the
reassembly with no error anywhere — and the §7 reassembly tests will not catch it, because they feed
well-formed fragments. Test the raw layer with a truncated/garbage `data:` line explicitly.

Frontend (`frontend/src/test/`): `dispatchSSE` routes `tool`; an old handler set without `onTool`
ignores it; `CoachTray` renders and clears the status line; `buildChatContext` emits `detail` and
omits `pose`.

Gates: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`, and
`yarn test:coverage` with cwd = `frontend/`.

## Success criteria (v3)

1. Asking about a measurement the compact blob dropped ("第 2 rep 膝蓋角度") gets a real number via
   `get_analysis`, not "分析中沒有測量".
2. Asking about a topic outside the detected faults gets a `rag_search`/`kg_query`-backed answer that
   is explicitly framed as general reference, never as an observation about the clip.
3. A clean rep still gets the CLEAN REP response, and an unmeasurable clip still gets the NOT
   MEASURED response — unchanged from v2.
4. A model that rejects `tools` still answers, with no user-visible error.
5. Follow-up chip latency is unchanged (~1.5s). **Live measurement, not a suite gate** — the unit
   tests can only lock that the chip path sends no `tools` and no `detail`; the timing itself has to
   be checked against the real endpoint, the way v2.1's ~1.5s figure was.
6. Both coverage gates pass.

## Known risks

- **Time-to-first-token regresses** on turns that call tools: an extra LLM round trip plus retrieval
  before the first word. The `tool` frame is the mitigation (the user sees progress), not a fix.
- **The tool schemas are the real recurring token cost**, not `detail`. `detail` never enters the
  prompt unless a tool returns part of it, but `_TOOLS` — three function schemas with the
  descriptions that carry the reference-vs-observation framing — is ~400–600 tokens sent on *every*
  round of *every* turn, including turns that call nothing. That is the standing price of v3, and
  it is charged against a metered endpoint. If it needs cutting later, the lever is trimming the
  descriptions, not dropping `detail`.
- ~~**Default model tool support is unverified.**~~ **RESOLVED by live probe, 2026-08-04.** Note the
  risk as originally written named the `.env.example` defaults, which are *not* what this deployment
  runs. The real configuration is **NVIDIA NIM** (`LLM_BASE_URL=https://integrate.api.nvidia.com/v1`),
  not OpenRouter, with `LLM_MODELS=openai/gpt-oss-120b, openai/gpt-oss-20b, minimaxai/minimax-m2.7`.
  Probed each with a live tool-enabled streaming request:

  | Model | Result |
  |---|---|
  | `openai/gpt-oss-120b` | **tool call**, `finish_reason=tool_calls` |
  | `openai/gpt-oss-20b` | **tool call**, `finish_reason=tool_calls` |
  | `minimaxai/minimax-m2.7` | **HTTP 410 Gone** — end-of-life 2026-07-27 |

  So v3 is not inert: both live models genuinely emit tool calls. Because the base URL is NIM rather
  than OpenRouter, `_is_openrouter` is False throughout, so the attribution headers and the
  `provider` routing body are correctly omitted — and `tools`/`tool_choice`, being standard
  OpenAI-compatible fields, are sent regardless. That is the intended behaviour.

- **`minimaxai/minimax-m2.7` is dead and still in the picker — a PRE-EXISTING bug, not introduced by
  v3.** It returns HTTP 410 for *every* request, tools or not, so any user who selects it gets a
  failed chat today. v3's degradation path handles it correctly but cannot rescue it: 410 is a 4xx,
  so the loop retries once without `tools`, gets 410 again, and emits a single in-band `error`
  frame — honest, at the cost of one wasted round trip. **Fix by removing it from `LLM_MODELS`**;
  that is a config change, independent of this branch.
- **The red line in §4 is prompt-enforced, not mechanically enforced.** A model can still blur
  retrieved knowledge into observation. Verify against live models before shipping; if it leaks,
  the fallback is to tag every tool-returned knowledge block inline with a "REFERENCE ONLY" prefix
  rather than relying on one system-prompt rule.

## Build notes (v3)

Kept alive the way v2 and v2.1 were.

- **Implemented across 6 commits** on `feat/chat-tool-calling`: transport split → `_stream_turn` →
  tool catalogue → the loop → client wiring → tray UI. Backend coverage 96.8% (gate 95%),
  `chat.py` 98%.
- **Model support — measured, not assumed** (see Known risks above): both live NIM models emit real
  tool calls. The third entry in `LLM_MODELS` is end-of-life and 410s; that is a pre-existing config
  bug this branch merely surfaces.
- **Six defects were found during implementation, and every one was a guard that did not cover the
  input space it claimed to.** Recording them because the pattern is the lesson, not the individual
  bugs:
  1. `(KeyError, IndexError, TypeError)` does not catch `None.get(...)` — that is `AttributeError`.
  2. A provider-supplied `tool_calls[].index` was trusted to be an int; a mixed str/int set made
     `sorted()` raise `TypeError`.
  3. `_clamp_int` missed `OverflowError`: `json.loads` accepts a bare `Infinity` literal and
     `int(float('inf'))` overflows.
  4. `json.dumps` sat *outside* the `try` whose docstring promised "nothing raises out of here" —
     and `default=str` rescues neither a non-string dict key (never consulted for keys), a circular
     reference (cycle detection fires first), nor a value whose `__str__` raises.
  5. `answer_stream` caught only `_LLMError`, but `_stream_raw_chunks` wraps only the *transport* in
     `except Exception → _LLMError`; `_stream_turn`'s reassembly is unwrapped, so five distinct
     valid-JSON-but-misshapen chunk shapes raised straight out of the generator.
  6. `_stream_turn`'s nameless-slot filter had no test driving its false branch.

  All six share one failure mode, and it is the exact one this spec's error model exists to prevent:
  they escape as exception types nobody catches, out of a generator whose **HTTP 200 is already
  committed**, so the client sees a dead stream instead of an `error` frame. The structural fix was
  an `except Exception` in `answer_stream` — the only frame in the stack that knows the 200 is
  committed and can convert a crash into an in-band frame. Note a sibling `except` clause never
  catches what is raised inside *another* clause's body, so the retry call needed its own.
- **Correction to §1's framing.** The 200 is committed *earlier* than "the first frame is yielded":
  Starlette sends `http.response.start` before it ever enters the body iterator. Every statement in
  `answer_stream` is therefore in scope, including those before the first `yield`.
- **Three findings were invisible to coverage**, which is worth internalising before trusting a
  coverage number again: coverage.py does not split a one-line comprehension `if` or a one-line
  `or` into separate branch arcs, and it never models an argument's *value*. Concretely — deleting
  the `not offer_tools` disjunct left the whole suite green, and swapping `_dispatch_tool`'s third
  argument from `context` to `convo` also left it green while `get_analysis` would have been
  silently dead forever. Both are now covered by tests verified to fail against exactly those
  mutations.
- **The frontend coverage gate is red on the dev machine, and was already red on `main`** (3
  failures there, 4 on this branch, all `Test timed out in 5000ms` in components this branch never
  touches). `yarn test` without coverage passes 90/90 cleanly. Cause is vitest's default 5s
  `testTimeout` losing races under coverage instrumentation on a slow machine (~490s wall,
  ~3340s cumulative import). Raising `testTimeout` in `vite.config.ts` is the obvious fix but edits
  shared CI config, so it was deliberately left out of this branch.

### Live verification results (2026-08-04, `openai/gpt-oss-120b` on NIM)

Run via `scripts/chat/try_tools.py`, which drives `answer_stream` directly — real loop, real tools,
real model, no browser or Supabase session needed.

| Check | Result |
|---|---|
| **Criterion 1 — `get_analysis` detail path** | **PASS.** "所有量測數值 + 第幾個 rep" returned `min_knee_angle_deg 104.7`, `hip_depth_ratio 0.83`, `torso_lean_deg 41.2`, `peak_frame 108`, `rep_index 2`, `confidence 0.88` — every one present ONLY in `detail`, never in the prompt. |
| **Criterion 2 — `rag_search`** | **PASS.** Returned real corpus citations (Lee et al. 2019 JSC 33(3); Human Kinetics), framed as 「在一般文獻中」. |
| **`kg_query`** | **PASS**, but only fires on a graph-shaped question. For ordinary knowledge questions the model consistently prefers `rag_search`. Not a defect; worth knowing the tool is not dead, just narrowly selected. |
| **Criterion 3 — §4 red line, undetected fault** | **PASS.** Asked "我有 butt wink 嗎?" (a real KG fault, not detected). Opened with 「這次的分析並沒有偵測到 butt wink」 and labelled the retrieved knowledge 「參考一般文獻,未在此影片中測得」. |
| **Criterion 3 — §4 red line, NOT MEASURED clip** | **PASS, and this is the one that mattered.** On a 0%-measurable clip it refused to judge depth (「系統無法對任何畫格進行分析,也就無法判斷您的深蹲深度是否足夠」), told the user to re-record, and labelled the literature 「**並非從您的影片中觀測到的問題**」. FIX 7's clause plus the REFERENCE ONLY prefix held — the escape hatch is closed in practice, not just on paper. |

**The real cost, and it is not small: time-to-first-token on a tool turn is 31–36s.** A non-tool turn
on the same model is ~4.7s. So a tool round costs roughly **+27s** — partly `gpt-oss-120b` being a
reasoning model that thinks before both the tool call and the answer, partly the round trip plus
retrieval. The `tool` status frame makes the wait legible rather than blank, but it does not make it
short. This is materially worse than the spec's Known-risks section anticipated and is the strongest
argument for pinning a faster model for tool turns, the way v2.1 pinned one for the follow-up chips.

### Still outstanding

Two items still need the browser, because they are about the tray's rendering rather than the loop:

1. **The retraction path.** `gpt-oss-120b` did not narrate before calling a tool in any of the five
   runs above, so no `reset` frame was observed in the wild. The behaviour is unit-tested on both
   sides; what remains unverified is only that a real narrate-then-call turn *looks* right in the
   tray. A model more prone to narrating (or a deliberately provocative prompt) would surface it.
2. **Follow-up chip latency** — the v2.1 baseline is ~1.5s. Unchanged by this work in principle, since
   the chip path sends neither `tools` nor `detail`, but not re-measured live.

---

# v3.1: Persistent tool records with citations

Status: **built** (branch `feat/chat-tool-calling`; see `## Build notes (v3.1)` at the end of this
section). Requested as "tool calling 的紀錄像 Claude Code 一樣,不會開始回答
就消失,也可以讓使用者知道引用了哪些".

## Objective (v3.1)

v3 shows a **transient** tool status line that is cleared on the first `delta`. So the moment the
answer starts, the evidence for it disappears — the user sees a claim with no visible provenance,
which is the opposite of what a product whose thesis is explainability should do. v3.1 makes the
tool record **persist beside the answer it produced**, and adds the one thing v3 never put on the
wire at all: **which sources the retrieval actually returned**.

## Decisions (settled at design review)

1. Tool records live **above the answer**, in call order, **expanded by default** *(the source list's
   default is reversed to collapsed by v3.2 Decision 2; the tool line itself stays visible)* — the Claude Code
   shape, so the reasoning chain reads top-to-bottom: what was looked up, what came back, what the
   coach concluded.
2. They **persist across reload and history replay**, stored alongside the answer text.
3. **`reset` does NOT clear them.** See §4 — this is an honesty decision, not a technical one.

## 1. The three tools do not have comparable "sources", and the UI must not pretend they do

Measured against the live data, not assumed:

| Tool | What it can cite | Nature |
|---|---|---|
| `rag_search` | `metadata.reference` + `metadata.source_type`, e.g. `Wikipedia: Squat (exercise)` / `encyclopedia` | a **real literature citation** |
| `kg_query` | `matched_nodes` and the 1-hop node names — nothing else | **graph nodes, NOT citations.** Verified: KG nodes carry only `node_id`, `name`, `label`; there is no source/reference/citation field anywhere in the subgraph |
| `get_analysis` | nothing external | it reads the user's own analysis; there is no outside source to credit |

So the UI **labels `kg_query`'s sources as knowledge-graph concepts, never as references**. Rendering
a graph node in the same visual slot as a cited paper would tell the user a concept is a source —
manufacturing exactly the false authority this feature exists to prevent. `get_analysis` shows its
subject (the fault name) and no source list.

## 2. Where the sources come from — and why the split is required, not cosmetic

`_dispatch_tool` currently fuses four steps: run, prefix, serialise, truncate. Sources must be
derived from the **raw** result, *before* truncation, or a large `rag_search` hit gets its citations
sliced off by `_MAX_TOOL_RESULT_CHARS` — the exact results where provenance matters most.

```
_run_tool(name, args, context) -> Any            # the raw result; the never-raises try moves here
_tool_sources(name, result) -> list[dict]        # pure; derives citations from the RAW result
_dispatch_tool(...) -> _ToolResult(text, sources)
```

`_dispatch_tool`'s never-raises contract is unchanged and still covers serialisation (the v3 fix
wave's finding). The 21 test call sites gain `.text` — mechanical.

**The wire shape is deliberately narrow:**

```json
{"label": "Wikipedia: Squat (exercise)", "kind": "encyclopedia"}
```

Per tool, precisely:

- **`rag_search`** — one entry per retrieved chunk. `label` from `metadata.reference`; when that is
  absent, the basename of `metadata.source` with directories stripped. `kind` from
  `metadata.source_type` (`encyclopedia`, `paper`, ...).
- **`kg_query`** — one entry per `matched_nodes` entry plus the 1-hop `subgraph.nodes` names.
  `label` is the node `name` with any `Movement:` prefix stripped (`Squat:Insufficient Depth` renders
  as `Insufficient Depth`). **`kind` is the literal string `"concept"`** — never a `source_type` —
  because that is what §1 requires the renderer to key off to keep graph nodes out of the citation
  slot. The node `label` field (`QualityDimension`, …) is an internal taxonomy and is not sent.
- **`get_analysis`** — the key is omitted entirely.

Common rules:

- **`metadata.source` is NEVER sent.** It is a server filesystem path (`data\rag\docs\squat_wiki.txt`)
  — useless to a user and a gratuitous internals leak.
- Deduplicated by `label` (one document yields many chunks; a KG node can be both matched and 1-hop),
  capped at **5** per tool call, preserving first-seen order.

## 3. SSE contract (v3.1 delta)

The `tool` frame gains one field; no new frame types. *(Superseded by v3.2 §1, which splits this
into `tool` + `tool_done` so the status line can appear before the tool runs.)*

```
event: tool
data: {"name":"rag_search","query":"ankle dorsiflexion","sources":[{"label":"...","kind":"encyclopedia"}]}
```

Backward compatible: a client reading only `name`/`query` is unaffected, and `sources` is absent
rather than `[]` for tools that have none to give.

## 4. Client state — the actual behaviour change

`tool: {name, query} | null` becomes `toolRuns: ToolRun[]`: **appended, never replaced**, and **not
cleared on the first `delta`**.

**`reset` does not clear `toolRuns`.** The retraction exists because a round's *narration* was not
the answer — but the tool calls in that round genuinely happened, and their results genuinely fed
the next round. Erasing them alongside the narration would misreport the reasoning chain, showing an
answer whose real inputs are invisible. `reset` therefore clears only `acc`/`streaming`, exactly as
in v3.

`toolRuns` clears on: a new send, an analysis switch, and the error rollback path.

## 5. Persistence — no migration needed

`conversations.messages` is `jsonb` with no per-element constraint, so the storage side is free:

- Frontend type: `ChatMessage.tools?: ToolRun[]`.
- Backend: `ConversationMessage` (the **conversations** router) gains `tools`.
- **`/api/chat`'s `ChatMessage` stays `{role, content}`.** Pydantic drops unknown fields, so the tool
  records can never reach the LLM prompt — but the client should still strip them before sending, the
  way it strips `detail`, rather than re-uploading them every turn and relying on implicit stripping.

## 6. Rendering

One component, two callers: a committed assistant message renders its own `tools`; the in-flight turn
renders the live `toolRuns`. Both sit above the content, so nothing moves when the turn commits.

`get_analysis` renders as subject-only. `rag_search` renders its citation list. `kg_query` renders its
matched concepts under a label that says they are graph concepts.

## 7. Testing (v3.1)

Backend: `_tool_sources` per tool (RAG with and without `reference`; KG; `get_analysis` yields `[]`);
`metadata.source` never appears in any frame; dedupe; the 5-cap; and the load-bearing one — **a
result large enough to be truncated still yields its sources**, which is the regression that proves
extraction happens before truncation.

Frontend: `onTool` appends rather than replaces; a `reset` leaves `toolRuns` intact while clearing the
streamed text; commit attaches `tools` to the message; a reload restores them; and both `/api/chat`
and `/api/chat/followups` are sent messages with `tools` stripped.

## Success criteria (v3.1)

1. After an answer completes, the tools that produced it are still on screen, in call order.
2. A `rag_search` turn names the documents it retrieved; no server path is visible anywhere.
3. A `kg_query` turn's concepts are not presented as literature citations.
4. Reloading the page, or replaying the analysis from history, restores the tool records.
5. A narrate-then-call turn retracts the narration but keeps the tool record.
6. Both coverage gates still pass.

## Build notes (v3.1)

Built across 9 commits (`461f93ea`..`4d889920`). Backend 1558 passed / 1 xfailed, coverage 97.0%
against the 95% gate (`chat.py` 99%, `conversations.py` 100%); frontend 90/90 files, 818/818 tests,
`yarn build` clean. Criteria 1-5 are pinned by tests; criterion 6 holds for the backend gate, and the
frontend gate is `yarn test` — `yarn test:coverage` is red on this machine for reasons that predate
this work (vitest's default 5s `testTimeout` under coverage instrumentation) and `main` fails
identically.

Four things the build changed relative to the text above:

1. **§2's `_dispatch_tool` split shipped with a stronger never-raises guarantee than v3 had.** The
   handler that formats a failed tool's message is itself wrapped, because `f"{exc}"` calls
   `BaseException.__str__`, which returns `str(args[0])` — so an exception carrying an object whose
   own `__str__` raises would raise *inside the handler*. That is a live path: the same reasoning had
   already been applied to `json.dumps`'s fallback and not to this one.

2. **§3's frame is now yielded *after* the tool runs, not before.** The frame carries the sources, and
   those only exist once the tool has returned. The cost is real and was accepted: the tray shows
   generic "thinking" dots through the whole retrieval instead of naming the lookup, and on a cold RAG
   process that silence ran past two minutes in measurement. The fix, if it bites: yield `name`/`query`
   before dispatch as v3 did, then a second frame whose sources attach to the last-appended run — no
   correlation key needed, since the loop yields strictly sequentially.

3. **§1's mechanism is now real rather than declared.** The spec said `kind` is what the renderer keys
   off to keep graph nodes out of the citation slot; the first implementation keyed off the tool
   *name* and rendered `kind` nowhere, leaving the stated safety mechanism inert — a fourth tool
   returning concept-kind sources would have been headed "Sources". The renderer now derives the
   heading from `kind`, so the guarantee travels with the data.

4. **§4's list is rendered under a shared byline block.** The in-flight and committed turns render
   `coachTag → ToolRunList → content` identically, so nothing shifts at commit and a tool record shows
   its byline from the moment it lands — before the first token, which is exactly when a record is
   most likely to be the only thing on screen.

Not carried out, deliberately, and worth knowing: `_tool_sources` failing degrades to "no citations"
indistinguishably from "nothing to cite", with nothing logged — `chat.py` has no logger at all, so
fixing it means introducing logging to a module that has none.

---

# v3.2: Live tool status, condensed sources

Status: **designed** (approved at design review 2026-08-05; not built). Requested as "我想要像
Claude Code 一樣搜尋的時候顯示,搜尋完成後只顯示部分或濃縮的結果,而不是整個完整的標題".

## Objective (v3.2)

v3.1 bought provenance at the cost of live feedback and screen space, and both bills are now due:

- **The tray goes silent during retrieval.** `## Build notes (v3.1)` item 2 records the tradeoff —
  the `tool` frame moved *after* dispatch because it carries the sources, so the user sees generic
  thinking dots for the whole lookup, measured past two minutes on a cold RAG process. v3 named the
  lookup; v3.1 stopped doing so. v3.2 restores it without giving back the sources.
- **Every source is rendered in full, always.** Three `rag_search` hits push the answer down the
  tray behind a wall of document titles the user did not ask to read. Provenance should be
  *available*, not *unavoidable*.

v3.2 splits the tool frame in two and collapses the source list to a count.

## Decisions (settled at design review)

1. **Two frames per tool call**: one when it starts, one when it finishes.
2. **Sources render collapsed by default**, as a count, expandable on click. This **reverses v3.1
   Decision 1's "expanded by default"** for the source list specifically — the tool line itself
   (name + query) stays always-visible, which is the part v3.1 got right.
3. `kg_query` keeps its own heading when collapsed. §1 of v3.1 is not relaxed by counting.

## 1. SSE contract (v3.2 delta)

`tool` reverts to being yielded **before** dispatch and loses `sources`; a new `tool_done` carries
them. Both frames carry an `id`.

```
event: tool       data: {"id":0,"name":"rag_search","query":"ankle dorsiflexion"}
                  ... the tool actually runs here ...
event: tool_done  data: {"id":0,"sources":[{"label":"...","kind":"encyclopedia"}]}
```

Three properties, each load-bearing:

**`tool_done` is sent unconditionally, including with no sources.** It is the *completion* signal,
not the *sources* signal — that is why it is not named `tool_sources`. `get_analysis` never has
sources (v3.1 §1) and would otherwise sit on screen as "still running" forever. `sources` is omitted
rather than `[]` when empty, preserving v3.1's "nothing to cite" / "cited nothing" distinction.

**`id` is a counter monotonic across the entire `_answer_stream_inner` call, not per round.** The
loop runs up to `_MAX_TOOL_ROUNDS = 3` rounds (chat.py:1018) and `enumerate(turn.tool_calls)`
restarts at 0 in each, so a per-round index collides across rounds — as would matching on `name`,
the moment two rounds both call `rag_search`, which is the *expected* shape of a multi-round
conversation rather than a hypothetical. The counter makes correlation exact and independent of
dispatch ordering, so it also survives a future parallel dispatch that "last pending run" would not.
(The pre-existing `tool_call_id` fallback `f"call_{i}"` has the same latent per-round collision. Out
of scope — do not chase it.)

**A new event name, not a `done: true` flag on `tool`.** An older client's `dispatchSSE` chain
(api.ts:421-425 — one site, verify it is still the only one) silently ignores an unknown event and
leaves a row that never resolves; a flag on `tool` would render a duplicate row instead. Both are
wrong, but a stuck row is the smaller lie.

## 2. Client state

`ToolRun` — the persisted shape — is **unchanged**: `{name, query, sources?}`. The two new fields
live only in memory:

```ts
type LiveToolRun = ToolRun & { id: number; pending: boolean };
```

`onTool` appends with `pending: true`; `onToolDone` finds the run with the matching `id`, writes
`sources`, and clears `pending`. A `tool_done` whose `id` matches nothing is **dropped** — silently
mis-attributing a citation is worse than losing one.

**No pending run may outlive the turn.** This matters because a lost `tool_done` is a real path, not
a hypothetical: per v2 §1's error model Starlette has already committed HTTP 200 before the body
iterator runs, so an exception escaping the generator — including one raised while serialising
`tool_done` itself — produces a dead stream with no `error` frame at all. A dropped or
uncorrelatable frame does the same thing more quietly.

It needs no separate settle step, and deliberately does not get one. Both exits already cover it:
the commit path rebuilds each run from an allow-list, which drops `pending` along with `id`, so a
never-resolved run commits as "finished, cited nothing"; the rollback path clears the live list
wholesale. The invariant is therefore structural rather than a rule someone must remember to apply,
and no audit of backend serialisability is required.

The one case left open: a stream that neither ends nor errors holds its row pending indefinitely.
That is a hung connection — v3.1's thinking dots hung identically and the whole turn is stuck
regardless — so v3.2 adds no new failure mode.

**`id` and `pending` are stripped at commit**, alongside nothing else — the message keeps
`{name, query, sources}`. This is the same reasoning as v3.1 §5's "strip before sending rather than
relying on Pydantic dropping unknown fields": the backend's `ToolRun` model would ignore them, but
that backstop is coincidental and must not become the mechanism. **No backend model changes in
v3.2** — `conversations.py` is untouched.

## 3. Rendering

```
running    檢索文獻:ankle dorsiflexion  ⋯
done       檢索文獻:ankle dorsiflexion
              › 引用來源 3 筆
expanded   檢索文獻:ankle dorsiflexion
              ⌄ 引用來源 3 筆
                Wikipedia: Squat (exercise)
                Lee et al. 2019
                Human Kinetics: Strength Training
```

- **The pending marker reuses `<LumenLoader variant="dots" />`.** It is now the *only* signal that
  anything is happening across a retrieval that can run minutes, so it must animate; a static `⋯` is
  a much weaker signal than the dots it replaces. Reusing the existing primitive rather than
  inventing a second spinner idiom is deliberate. **But it must be wrapped `aria-hidden`**: the dots
  carry `role="status"`, and today exactly one exists at a time (CoachTray's, gated on
  `toolRuns.length === 0`). A three-tool turn would otherwise mount three simultaneous live regions
  all announcing the same string. The row instead carries `aria-busy`, which states the same thing
  once, in the right place.
- **A run with no sources renders no summary row at all**, collapsed or otherwise. `get_analysis`
  therefore looks exactly as it does today once complete.
- **The summary row is a `<button>` with `aria-expanded`**, not a clickable `div`.
- `ToolRunList` gains a `ToolRunRow` child holding the expanded state — a hook cannot live inside
  the `.map`. State is ephemeral and per-row.
- **The heading still comes from `kind`, not from the tool name** (v3.1 Build note 3). Collapsed,
  that is "知識圖譜概念 N 筆" vs "引用來源 N 筆". Counting must not flatten the distinction that v3.1
  §1 exists to enforce.
- **Known and accepted:** expanded rows collapse when the turn commits, because the live list is
  replaced by the committed message's own list. Preserving it means hoisting row state into
  `CoachTray` and keying it across both render paths — not worth it for a state the user usually
  enters after reading.

New i18n keys in **both** `en` and `zhHant` (`lib.i18n.test.ts` enforces parity), using the existing
brace convention (`"video.faultMany": "{count} faults detected"`):

```
"chat.tool.sourcesN":  "Sources · {n}"                   / "引用來源 {n} 筆"
"chat.tool.conceptsN": "Knowledge-graph concepts · {n}"  / "知識圖譜概念 {n} 筆"
```

`chat.tool.sources` / `chat.tool.concepts` become **unused** — the count row *is* the heading, and a
second heading under it would be redundant — and are deleted from both dictionaries.

## 4. Testing (v3.2)

Backend — the `tool` frame is yielded **before** the tool runs (assert ordering against a dispatch
that records when it was called); `tool_done` is emitted even when the tool yields no sources; ids
are unique across a **two-round** turn.

Frontend, the load-bearing four:

1. `get_analysis` does not stay pending — a `tool_done` with no `sources` clears it.
2. **The same tool called twice in one turn lands its sources on the right rows.** This is what
   actually exercises correlation, and it is reachable today, not hypothetically.
3. A `tool_done` with an unknown `id` is dropped rather than attached to anything.
4. A stream that ends without `tool_done` (error, or `done` alone) leaves no pending row.

Plus: collapsed `kg_query` says concepts, not sources; the expand toggle reveals the labels; and the
committed message's `tools` contains neither `id` nor `pending`.

## Success criteria (v3.2)

1. During a slow retrieval the tray names the tool and the query, and shows it as running.
2. When it completes, the sources appear as a count on one line; clicking reveals them.
3. `get_analysis` completes visibly and shows no source row.
4. Two `rag_search` calls in one turn each show their own sources.
5. A stream that dies mid-tool leaves no row claiming to still be running.
6. Nothing v3.1 pinned regresses: records still persist, no server path is visible, graph concepts
   are still not called citations.
