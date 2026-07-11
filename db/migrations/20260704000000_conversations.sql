-- x-coach LLM chat v2: per-user, per-analysis conversation persistence.
--
-- Design notes (see specs/llm-chat-spec.md, v2 §3):
--   * One thread per (user, video_id). The whole conversation is stored as a JSONB array of
--     {role, content} messages — mirroring how `analyses.result` stores the full document verbatim
--     rather than normalising into a messages table. Coaching threads are short; a whole-array
--     upsert on each turn is simpler and cheap, and keeps replay self-contained.
--   * `video_id` is `text` with **NO foreign key** to `videos`/`analyses` — intentionally, exactly
--     like `analyses.video_id`. A signed-in user can chat about a *fresh upload* before any video
--     or analysis row is persisted, so a FK would reject the very first save. Ownership is still
--     enforced by `user_id -> auth.users` + RLS; the conversation is only ever reachable again via
--     the persisted-analysis history-replay path, keyed on the same per-user video_id.
--
-- RLS strategy is identical to videos/analyses: every row is scoped to auth.uid() = user_id, so the
-- user-JWT (postgrest.auth(token)) path the backend uses has the DB as the ownership backstop.

create extension if not exists pgcrypto;

-- ---------------------------------------------------------------------------
-- conversations: one grounded chat thread per (user, video_id)
-- ---------------------------------------------------------------------------
create table if not exists public.conversations (
    id         uuid primary key default gen_random_uuid(),
    user_id    uuid not null references auth.users (id) on delete cascade,
    video_id   text not null,                 -- matches analyses.video_id (per user); no FK, by design
    messages   jsonb not null default '[]',   -- [{role: 'user'|'assistant', content: text}], oldest first
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (user_id, video_id)
);

create index if not exists conversations_user_video_idx
    on public.conversations (user_id, video_id);

-- Keep updated_at fresh as the thread grows (reuses the trigger fn from the init migration).
drop trigger if exists conversations_touch_updated_at on public.conversations;
create trigger conversations_touch_updated_at
    before update on public.conversations
    for each row execute function public.touch_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: a user can only see/write their own threads.
-- ---------------------------------------------------------------------------
alter table public.conversations enable row level security;

drop policy if exists "conversations_owner_all" on public.conversations;
create policy "conversations_owner_all" on public.conversations
    for all
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);
