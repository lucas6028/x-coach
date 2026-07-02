# Spec: Conversational Coaching (LLM chat layer, OpenRouter)

Status: **v1 in progress** · Owner: — · Supersedes the "chat disabled / coming soon" placeholder.

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

## Out of scope (v2)

Streaming, chat persistence/history, tool-calling live RAG, anonymous chat, multi-analysis memory.
