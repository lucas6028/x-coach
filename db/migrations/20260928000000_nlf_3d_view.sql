-- NLF 3D view: an opt-in per-video 3D reconstruction, produced by a worker PC (not the backend
-- process) that polls for queued jobs and does the (GPU-bound, NLF) 3D lift offline, then uploads
-- the result and marks the job done.
--
-- Design notes:
--   * Two new roles on the existing user_roles table: 'nlf_user' (may request a 3D view of their
--     own video) and 'nlf_worker' (the worker PC's own Supabase login; a service account, not a
--     patient). Both are ordinary rows in the existing (user_id, role) user_roles table, gated by
--     is_admin()-shaped SECURITY DEFINER functions (has_nlf_access, is_nlf_worker below).
--   * The worker never gets broader table access: it has NO RLS policy of its own on nlf_jobs,
--     videos or analyses (selecting those directly returns zero rows for it, same as any other
--     stranger). Every claim/complete/fail/heartbeat action goes through a SECURITY DEFINER
--     function gated by is_nlf_worker(auth.uid()); this is the only door the worker has.
--   * nlf_jobs has no insert/update/delete policy for ANYONE (including the owner): a patient
--     enqueues only via request_nlf_job(), and the worker only via the nlf_* functions below.
--     Owner gets SELECT only, to watch their own job's status.
--   * nlf_jobs is 1:1 with (user_id, video_id) and FK's onto videos(user_id, video_id) ON DELETE
--     CASCADE, so deleting a video silently drops any in-flight or finished 3D job for it too --
--     nlf_complete_job then just returns false for a job id the worker was already holding (the
--     row it would update is gone); nlf_fail_job's UPDATE likewise matches zero rows and is a
--     silent no-op.
--   * The worker reads the "latest analysis" for a video, not a specific analysis id, because the
--     3D-view request only ever names a video: at claim time nlf_claim_job looks up that video's
--     most recent analyses row (by created_at) for the movement/pose/rep/detection context the
--     worker needs, and records which analysis it used on the job row (analysis_id).
--
-- This migration is idempotent (db/checks/run_pglite.mjs applies the newest migration twice):
-- constraints are dropped-if-exists then re-added, tables are CREATE TABLE IF NOT EXISTS, policies
-- are dropped-if-exists then re-created, and functions are CREATE OR REPLACE (or, where the return
-- shape changes, DROP FUNCTION IF EXISTS first).

-- ---------------------------------------------------------------------------
-- 1. user_roles: add 'nlf_user' and 'nlf_worker' to the allowed roles.
-- ---------------------------------------------------------------------------
alter table public.user_roles drop constraint if exists user_roles_role_check;
alter table public.user_roles
    add constraint user_roles_role_check
    check (role in ('admin', 'clinician', 'nlf_user', 'nlf_worker'));

-- has_nlf_access(uid): true when the user may request a 3D view of their own videos.
create or replace function public.has_nlf_access(uid uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
    select exists(
        select 1 from public.user_roles where user_id = uid and role = 'nlf_user'
    );
$$;

revoke all on function public.has_nlf_access(uuid) from public;
revoke all on function public.has_nlf_access(uuid) from anon;
grant execute on function public.has_nlf_access(uuid) to authenticated;

-- is_nlf_worker(uid): true for the worker PC's own login only.
create or replace function public.is_nlf_worker(uid uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
    select exists(
        select 1 from public.user_roles where user_id = uid and role = 'nlf_worker'
    );
$$;

revoke all on function public.is_nlf_worker(uuid) from public;
revoke all on function public.is_nlf_worker(uuid) from anon;
grant execute on function public.is_nlf_worker(uuid) to authenticated;

-- ---------------------------------------------------------------------------
-- 2. nlf_jobs: one row per (user, video) 3D-view request/result.
-- ---------------------------------------------------------------------------
create table if not exists public.nlf_jobs (
    id           uuid primary key default gen_random_uuid(),
    user_id      uuid not null,
    video_id     text not null,
    status       text not null default 'queued'
                 check (status in ('queued', 'claimed', 'done', 'failed')),
    attempts     int not null default 0,
    error        text,
    result_key   text,
    meta         jsonb,
    analysis_id  uuid,
    claimed_at   timestamptz,
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now(),
    unique (user_id, video_id),
    foreign key (user_id, video_id) references public.videos (user_id, video_id) on delete cascade
);

create index if not exists nlf_jobs_status_created_idx
    on public.nlf_jobs (status, created_at);

alter table public.nlf_jobs enable row level security;

-- Owner reads their own job's status; no other policy exists, so every write (insert included)
-- must go through a SECURITY DEFINER function below.
drop policy if exists "nlf_jobs_owner_select" on public.nlf_jobs;
create policy "nlf_jobs_owner_select" on public.nlf_jobs
    for select
    to authenticated
    using (auth.uid() = user_id);

-- ---------------------------------------------------------------------------
-- 3. nlf_worker_heartbeat: single-row liveness marker the worker pings, and anyone signed in can
--    read (e.g. to show "worker last seen ..." in an admin/status view). No policies: only the
--    definer functions below touch it.
-- ---------------------------------------------------------------------------
create table if not exists public.nlf_worker_heartbeat (
    id        int primary key check (id = 1),
    last_seen timestamptz
);

alter table public.nlf_worker_heartbeat enable row level security;

-- ---------------------------------------------------------------------------
-- 4. request_nlf_job(video_id): the patient-facing entry point.
-- ---------------------------------------------------------------------------
create or replace function public.request_nlf_job(p_video_id text)
returns public.nlf_jobs
language plpgsql
security definer
set search_path = public
as $$
declare
    v_uid uuid := auth.uid();
    v_job public.nlf_jobs;
begin
    if not public.has_nlf_access(v_uid) then
        raise exception 'not authorized' using errcode = '42501';
    end if;

    if not exists (
        select 1 from public.videos where user_id = v_uid and video_id = p_video_id
    ) then
        raise exception 'video not found' using errcode = 'P0002';
    end if;

    -- INSERT .. ON CONFLICT DO NOTHING rather than "select, then insert if not found": two
    -- concurrent first requests for the same video would otherwise both miss the SELECT and one
    -- would hit a raw unique_violation on its INSERT instead of just seeing the row the other
    -- created.
    insert into public.nlf_jobs (user_id, video_id)
    values (v_uid, p_video_id)
    on conflict (user_id, video_id) do nothing
    returning * into v_job;

    if found then
        return v_job;
    end if;

    select * into v_job
    from public.nlf_jobs
    where user_id = v_uid and video_id = p_video_id
    for update;

    if v_job.status = 'failed' then
        update public.nlf_jobs
        set status = 'queued', attempts = 0, error = null, claimed_at = null, updated_at = now()
        where id = v_job.id
        returning * into v_job;
    end if;

    return v_job;
end;
$$;

revoke all on function public.request_nlf_job(text) from public;
revoke all on function public.request_nlf_job(text) from anon;
grant execute on function public.request_nlf_job(text) to authenticated;

-- nlf_worker_last_seen(): the heartbeat, or null. Any signed-in user may read it (not gated to
-- nlf_user/nlf_worker) -- it carries no data beyond "is the worker alive".
create or replace function public.nlf_worker_last_seen()
returns timestamptz
language sql
stable
security definer
set search_path = public
as $$
    select last_seen from public.nlf_worker_heartbeat where id = 1;
$$;

revoke all on function public.nlf_worker_last_seen() from public;
revoke all on function public.nlf_worker_last_seen() from anon;
grant execute on function public.nlf_worker_last_seen() to authenticated;

-- ---------------------------------------------------------------------------
-- 5. Worker functions. Each is gated on is_nlf_worker(auth.uid()) INSIDE the body -- the shim
--    (and Supabase's own default grants) hand execute to every role, so the body check is the
--    only real gate; see db/checks/00_local_supabase_shim.sql.
-- ---------------------------------------------------------------------------

-- nlf_claim_job(): (1) recover stale claims, (2) claim the oldest queued job, (3) attach the
-- video's storage_key and the latest analysis for that (user, video).
create or replace function public.nlf_claim_job()
returns table (
    job_id       uuid,
    owner_id     uuid,
    video_id     text,
    attempts     int,
    storage_key  text,
    analysis_id  uuid,
    movement     text,
    pose_fps     double precision,
    rep_segments jsonb,
    detections   jsonb
)
language plpgsql
security definer
set search_path = public
as $$
declare
    v_job      public.nlf_jobs;
    v_analysis public.analyses;
begin
    if not public.is_nlf_worker(auth.uid()) then
        raise exception 'not authorized' using errcode = '42501';
    end if;

    -- (1) stale claims: back to queued, or failed once attempts are exhausted.
    update public.nlf_jobs j
    set status = 'queued',
        claimed_at = null,
        updated_at = now()
    where j.status = 'claimed'
      and j.claimed_at < now() - interval '30 minutes'
      and j.attempts < 3;

    update public.nlf_jobs j
    set status = 'failed',
        error = 'claim timed out',
        updated_at = now()
    where j.status = 'claimed'
      and j.claimed_at < now() - interval '30 minutes'
      and j.attempts >= 3;

    -- (2) claim the oldest queued job.
    select *
    into v_job
    from public.nlf_jobs
    where status = 'queued'
    order by created_at
    limit 1
    for update skip locked;

    if not found then
        return;
    end if;

    -- (3) latest analysis for this (user, video), if any.
    select *
    into v_analysis
    from public.analyses a
    where a.user_id = v_job.user_id and a.video_id = v_job.video_id
    order by a.created_at desc
    limit 1;

    update public.nlf_jobs j
    set status = 'claimed',
        attempts = j.attempts + 1,
        claimed_at = now(),
        analysis_id = v_analysis.id,
        updated_at = now()
    where j.id = v_job.id
    returning j.* into v_job;

    return query
    select
        v_job.id,
        v_job.user_id,
        v_job.video_id,
        v_job.attempts,
        vv.storage_key,
        v_job.analysis_id,
        v_analysis.movement,
        case
            when jsonb_typeof(v_analysis.result -> 'pose' -> 'fps') = 'number'
            then (v_analysis.result -> 'pose' ->> 'fps')::double precision
        end,
        v_analysis.result -> 'reps' -> 'segments',
        coalesce(
            (
                select jsonb_agg(
                           jsonb_build_object(
                               'fault_id', d.value -> 'fault_id',
                               'peak_frame', d.value -> 'peak_frame'
                           )
                           order by d.ordinality
                       )
                from jsonb_array_elements(
                         case
                             when jsonb_typeof(v_analysis.result -> 'detections') = 'array'
                             then v_analysis.result -> 'detections'
                             else '[]'::jsonb
                         end
                     ) with ordinality as d(value, ordinality)
            ),
            '[]'::jsonb
        )
    from public.videos vv
    where vv.user_id = v_job.user_id and vv.video_id = v_job.video_id;
end;
$$;

revoke all on function public.nlf_claim_job() from public;
revoke all on function public.nlf_claim_job() from anon;
grant execute on function public.nlf_claim_job() to authenticated;

-- nlf_complete_job(job, result_key, meta): true if a claimed job was marked done, false if the
-- job is gone (its video was deleted) or was not in 'claimed' state.
create or replace function public.nlf_complete_job(p_job_id uuid, p_result_key text, p_meta jsonb)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
    v_count int;
begin
    if not public.is_nlf_worker(auth.uid()) then
        raise exception 'not authorized' using errcode = '42501';
    end if;

    update public.nlf_jobs
    set status = 'done',
        result_key = p_result_key,
        meta = p_meta,
        error = null,
        updated_at = now()
    where id = p_job_id and status = 'claimed';

    get diagnostics v_count = row_count;
    return v_count > 0;
end;
$$;

revoke all on function public.nlf_complete_job(uuid, text, jsonb) from public;
revoke all on function public.nlf_complete_job(uuid, text, jsonb) from anon;
grant execute on function public.nlf_complete_job(uuid, text, jsonb) to authenticated;

-- nlf_fail_job(job, error): retry (back to 'queued') under 3 attempts, otherwise 'failed'. Only
-- acts on a job currently 'claimed' by this worker call -- a queued/done/failed or missing job id
-- is a silent no-op, same as nlf_complete_job's guard.
create or replace function public.nlf_fail_job(p_job_id uuid, p_error text)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
    if not public.is_nlf_worker(auth.uid()) then
        raise exception 'not authorized' using errcode = '42501';
    end if;

    update public.nlf_jobs
    set status = case when attempts >= 3 then 'failed' else 'queued' end,
        claimed_at = case when attempts >= 3 then claimed_at else null end,
        error = p_error,
        updated_at = now()
    where id = p_job_id and status = 'claimed';
end;
$$;

revoke all on function public.nlf_fail_job(uuid, text) from public;
revoke all on function public.nlf_fail_job(uuid, text) from anon;
grant execute on function public.nlf_fail_job(uuid, text) to authenticated;

-- nlf_worker_heartbeat(): upsert the single liveness row.
create or replace function public.nlf_worker_heartbeat()
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
    if not public.is_nlf_worker(auth.uid()) then
        raise exception 'not authorized' using errcode = '42501';
    end if;

    insert into public.nlf_worker_heartbeat (id, last_seen) values (1, now())
    on conflict (id) do update set last_seen = excluded.last_seen;
end;
$$;

revoke all on function public.nlf_worker_heartbeat() from public;
revoke all on function public.nlf_worker_heartbeat() from anon;
grant execute on function public.nlf_worker_heartbeat() to authenticated;

-- ---------------------------------------------------------------------------
-- 6. admin_list_users(): add has_nlf. The return type changes, so drop-then-create (same as the
--    is_clinician addition in 20260916000000_clinic.sql), then re-apply the same revoke/grant.
-- ---------------------------------------------------------------------------
drop function if exists public.admin_list_users();
create function public.admin_list_users()
returns table (
    id                  uuid,
    email               text,
    created_at          timestamptz,
    last_sign_in_at     timestamptz,
    analyses_count      bigint,
    conversations_count bigint,
    is_admin            boolean,
    is_clinician        boolean,
    has_nlf             boolean
)
language plpgsql
security definer
set search_path = public
as $$
begin
    if not public.is_admin(auth.uid()) then
        raise exception 'not authorized' using errcode = '42501';
    end if;

    return query
    select
        u.id,
        u.email::text,
        u.created_at,
        u.last_sign_in_at,
        coalesce(a.cnt, 0) as analyses_count,
        coalesce(c.cnt, 0) as conversations_count,
        exists (
            select 1 from public.user_roles r
            where r.user_id = u.id and r.role = 'admin'
        ) as is_admin,
        exists (
            select 1 from public.user_roles r
            where r.user_id = u.id and r.role = 'clinician'
        ) as is_clinician,
        exists (
            select 1 from public.user_roles r
            where r.user_id = u.id and r.role = 'nlf_user'
        ) as has_nlf
    from auth.users u
    left join (
        select user_id, count(*) as cnt from public.analyses group by user_id
    ) a on a.user_id = u.id
    left join (
        select user_id, count(*) as cnt from public.conversations group by user_id
    ) c on c.user_id = u.id
    order by u.created_at desc;
end;
$$;

revoke all on function public.admin_list_users() from public;
revoke all on function public.admin_list_users() from anon;
grant execute on function public.admin_list_users() to authenticated;
