-- x-coach P1 schema: per-user video metadata + persisted analysis results.
--
-- Design notes (see notes/engineering.md, sections 2 & 5):
--   * Supabase Auth owns the users table (auth.users); we do NOT create our own.
--   * Videos and analyses are split: `videos` is one row per uploaded source clip
--     (carries the async status machine for P2), `analyses` is one row per completed
--     run. The full nested Analysis document (matching frontend/src/api.ts `Analysis`)
--     lives in `analyses.result` JSONB; `view_type` / `fault_count` are promoted to
--     columns so the "我的紀錄" list can sort/filter without parsing JSON.
--   * The source video binary lives in object storage (Cloudflare R2, local filesystem
--     store in dev/CI — see backend/app/services/storage.py). `storage_key` holds the key
--     PREFIX for one upload, `uploads/{owner}/{video_id}`, under which the raw video
--     (`/source`), the pose JSON (`/pose.json`) and the captured frame (`/thumb.jpg`) sit;
--     deleting a video reaps the whole prefix. As predicted when this migration was written,
--     the move off local disk changed only what storage_key points at — the schema did not
--     move, so there is no follow-up migration.
--
-- RLS strategy: every row is scoped to its owner via auth.uid() = user_id.
--   - If the FastAPI backend talks to Postgres with the *user's* JWT (anon key +
--     per-request access token, the supabase-py `postgrest.auth(token)` pattern),
--     these policies are enforced automatically.
--   - If the backend uses the *service_role* key it BYPASSES RLS, so it must then
--     filter/insert by user_id itself. Prefer the user-JWT path so the DB is the
--     backstop. Either way the policies below protect against a leaked anon key.

-- gen_random_uuid() ships with pgcrypto; Supabase enables it by default, but be explicit.
create extension if not exists pgcrypto;

-- ---------------------------------------------------------------------------
-- videos: one row per uploaded source clip
-- ---------------------------------------------------------------------------
create table if not exists public.videos (
    id           uuid primary key default gen_random_uuid(),
    user_id      uuid not null references auth.users (id) on delete cascade,
    video_id     text not null,                  -- app slug, e.g. "upload_ab12cd34ef56"
    filename     text,                            -- original client filename (display only)
    storage_key  text,                            -- object-store key PREFIX, e.g. "uploads/{user_id}/{video_id}"
    status       text not null default 'pending'
                 check (status in ('pending', 'processing', 'done', 'failed')),
    error_detail text,                            -- failure reason surfaced to the user
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now(),
    unique (user_id, video_id)
);

create index if not exists videos_user_created_idx
    on public.videos (user_id, created_at desc);

-- ---------------------------------------------------------------------------
-- analyses: one row per completed analysis of a video
-- ---------------------------------------------------------------------------
create table if not exists public.analyses (
    id               uuid primary key default gen_random_uuid(),
    user_id          uuid not null references auth.users (id) on delete cascade,
    video_id         text not null,               -- matches videos.video_id (per user)
    source           text not null default 'upload'
                     check (source in ('upload', 'library')),
    view_type        text,                         -- promoted from result.view.view_type
    fault_count      int  not null default 0,      -- promoted: length(result.detections)
    pipeline_version text,                          -- reproducibility: detector/threshold version
    result           jsonb not null,               -- full Analysis doc (api.ts `Analysis`)
    created_at       timestamptz not null default now()
);

create index if not exists analyses_user_created_idx
    on public.analyses (user_id, created_at desc);

-- GIN index so future queries can reach into the JSONB (e.g. filter by a fault_id).
create index if not exists analyses_result_gin_idx
    on public.analyses using gin (result);

-- Keep videos.updated_at fresh as the status machine advances.
create or replace function public.touch_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists videos_touch_updated_at on public.videos;
create trigger videos_touch_updated_at
    before update on public.videos
    for each row execute function public.touch_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: a user can only see/write their own rows.
-- ---------------------------------------------------------------------------
alter table public.videos   enable row level security;
alter table public.analyses enable row level security;

drop policy if exists "videos_owner_all" on public.videos;
create policy "videos_owner_all" on public.videos
    for all
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

drop policy if exists "analyses_owner_all" on public.analyses;
create policy "analyses_owner_all" on public.analyses
    for all
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);
