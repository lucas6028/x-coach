-- x-coach LINE bot: link codes + LINE↔analysis bindings (see backend/app/services/line_store.py).
--
-- These two tables power "continue a web analysis on LINE": a signed-in web user mints a one-time
-- code (line_link_codes) snapshotting their analysis grounding, then redeems it from LINE to create
-- a binding (line_bindings) that the webhook uses to answer grounded coaching questions.
--
-- Access model — DIFFERENT from videos/analyses/conversations:
--   * The LINE webhook is called by LINE with NO user JWT, so the backend touches these tables with
--     the Supabase **service_role** key (services/line_store), which bypasses RLS.
--   * There is therefore no per-user JWT to scope rows to. Instead, RLS is enabled with **NO policy
--     for anon/authenticated** — i.e. the anon/authenticated roles (a leaked anon key, the browser)
--     can read/write NOTHING here. Only service_role (which bypasses RLS) can. This is the opposite
--     of the user-owned tables and is deliberate: these rows are backend-owned bot state.
--   * user_id references auth.users only for cascade cleanup (delete the user -> drop their codes and
--     bindings); ownership enforcement lives in the service_role code path, not RLS.

create extension if not exists pgcrypto;

-- ---------------------------------------------------------------------------
-- line_link_codes: short-lived one-time codes to connect a LINE account to an analysis
-- ---------------------------------------------------------------------------
create table if not exists public.line_link_codes (
    code       text primary key,                -- human-typed code, e.g. "ABC123" (unambiguous alphabet)
    user_id    uuid not null references auth.users (id) on delete cascade,
    video_id   text,                            -- the analysed clip this code grounds to (nullable)
    context    jsonb not null default '{}',     -- buildChatContext() grounding snapshot, copied to the binding
    expires_at timestamptz not null,            -- redeem rejects after this; codes live ~15 min
    created_at timestamptz not null default now()
);

-- Sweep helper: expired, never-redeemed codes can be reaped by expiry.
create index if not exists line_link_codes_expires_idx
    on public.line_link_codes (expires_at);

-- ---------------------------------------------------------------------------
-- line_bindings: one row per LINE user, bound to a web analysis + running thread
-- ---------------------------------------------------------------------------
create table if not exists public.line_bindings (
    line_user_id text primary key,              -- LINE's opaque per-channel user id (source.userId)
    user_id      uuid not null references auth.users (id) on delete cascade,
    video_id     text,                          -- the bound analysed clip (per user)
    context      jsonb not null default '{}',   -- grounding snapshot the webhook answers from
    messages     jsonb not null default '[]',   -- [{role, content}] running thread, oldest first
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now()
);

create index if not exists line_bindings_user_idx
    on public.line_bindings (user_id);

-- Keep updated_at fresh as the thread grows (reuses the trigger fn from the init migration).
drop trigger if exists line_bindings_touch_updated_at on public.line_bindings;
create trigger line_bindings_touch_updated_at
    before update on public.line_bindings
    for each row execute function public.touch_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: enable, but grant NO policy to anon/authenticated.
-- Result: only the service_role key (RLS-exempt) can touch these tables. A leaked anon key or the
-- browser sees nothing here — these are backend-owned bot-state tables, not user-facing rows.
-- ---------------------------------------------------------------------------
alter table public.line_link_codes enable row level security;
alter table public.line_bindings   enable row level security;
