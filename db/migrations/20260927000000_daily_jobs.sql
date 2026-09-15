-- Daily care-loop job (WP5): who gets a LINE reminder tonight, and each clinician's one-line summary.
-- Plan: docs/superpowers/plans/2026-09-15-innoserve-rehab-20day-plan.md (WP5).
--
-- The job is triggered by a scheduler (GitHub Actions cron -> POST /api/jobs/daily), so it has no
-- user JWT and cannot go through the RLS-scoped path. Same answer as line_training_summary: the
-- backend's service_role client calls a few SECURITY DEFINER functions that return EXACTLY what the
-- job sends -- a LINE user id and a handful of counts -- never raw rows. Execute is granted to
-- service_role only; anon and authenticated cannot call any of them.
--
-- Only LINE-login accounts can be pushed to. Their LINE user id is recovered from the synthetic auth
-- email (line_<sub lowercased>@line.invalid, see services/line_auth.synthetic_email), NOT from
-- user_metadata.line_sub: user_metadata is writable by the user, so trusting it would let anyone
-- point the job's pushes at someone else's LINE account (the same reasoning as
-- 20260720000000_line_training_summary.sql). LINE user ids are 'U' + 32 lowercase hex, so the
-- lowercased email loses nothing but the leading 'U', which is restored here.
--
-- Rule (d) of the red flags ("7 days without a check-in on the current assigned plan") is computed
-- here with the same definition as services/clinic.inactive_7d: the current prescription is the
-- patient's most recently created plan with assigned_by set; it is inactive when that plan was
-- created at least 7 days before p_now and has no check-in in (p_now - 7 days, p_now].

-- ---------------------------------------------------------------------------
-- job_runs: one row per (job, local date) that has run. The primary key makes the daily claim
-- atomic, so a cron retry and a manual workflow_dispatch on the same day cannot both push.
-- RLS on, no policies: only the definer function below touches it.
-- ---------------------------------------------------------------------------
create table if not exists public.job_runs (
    job      text not null,
    run_date date not null,
    ran_at   timestamptz not null default now(),
    primary key (job, run_date)
);

alter table public.job_runs enable row level security;

create or replace function public.claim_job_run(p_job text, p_run_date date)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
begin
    insert into public.job_runs (job, run_date) values (p_job, p_run_date);
    return true;
exception when unique_violation then
    return false;
end;
$$;

revoke all on function public.claim_job_run(text, date) from public;
revoke all on function public.claim_job_run(text, date) from anon;
revoke all on function public.claim_job_run(text, date) from authenticated;
grant execute on function public.claim_job_run(text, date) to service_role;

-- line_user_id_of(email): the LINE user id behind a synthetic LINE-login email, or NULL for any
-- other account (email/Google users cannot receive a LINE push).
create or replace function public.line_user_id_of(p_email text)
returns text
language sql
immutable
set search_path = public
as $$
    select case
        when p_email ~ '^line_u[0-9a-f]{32}@line\.invalid$'
        then 'U' || substr(p_email, 7, 32)
    end;
$$;

revoke all on function public.line_user_id_of(text) from public;
revoke all on function public.line_user_id_of(text) from anon;
revoke all on function public.line_user_id_of(text) from authenticated;
grant execute on function public.line_user_id_of(text) to service_role;

-- daily_patient_reminders(since): one row per linked patient whose current prescription has no
-- check-in since `since` (the job passes the start of today, Asia/Taipei). The prescription must
-- come from a clinician still actively linked to the patient, so revoking the link stops reminders.
create or replace function public.daily_patient_reminders(p_since timestamptz)
returns table (
    patient_id uuid,
    line_user_id text,
    plan_id uuid,
    plan_name text
)
language sql
stable
security definer
set search_path = public
as $$
    with current_plan as (
        select distinct on (p.user_id) p.user_id, p.id, p.name, p.assigned_by
        from public.training_plans p
        where p.assigned_by is not null
        order by p.user_id, p.created_at desc
    )
    select cp.user_id, public.line_user_id_of(u.email::text), cp.id, cp.name
    from current_plan cp
    join auth.users u on u.id = cp.user_id
    where public.is_linked(cp.assigned_by, cp.user_id)
      and public.line_user_id_of(u.email::text) is not null
      and not exists (
          select 1 from public.session_checkins c
          where c.user_id = cp.user_id
            and c.plan_id = cp.id
            and c.created_at >= p_since
      )
    order by cp.user_id;
$$;

revoke all on function public.daily_patient_reminders(timestamptz) from public;
revoke all on function public.daily_patient_reminders(timestamptz) from anon;
revoke all on function public.daily_patient_reminders(timestamptz) from authenticated;
grant execute on function public.daily_patient_reminders(timestamptz) to service_role;

-- daily_clinician_summaries(since, now): one row per clinician (still holding the role, with a LINE
-- account) who has at least one active patient: how many patients, how many checked in since
-- `since`, open (unacknowledged) flags across them, and how many are inactive by rule (d).
create or replace function public.daily_clinician_summaries(p_since timestamptz, p_now timestamptz)
returns table (
    clinician_id     uuid,
    line_user_id     text,
    patients         int,
    checked_in_today int,
    open_flags       int,
    inactive_7d      int
)
language sql
stable
security definer
set search_path = public
as $$
    with links as (
        select l.clinician_id, l.patient_id
        from public.care_links l
        where l.status = 'active'
          and public.is_clinician(l.clinician_id)
    ),
    current_plan as (
        select distinct on (p.user_id) p.user_id, p.id, p.created_at
        from public.training_plans p
        where p.assigned_by is not null
        order by p.user_id, p.created_at desc
    ),
    per_patient as (
        select
            lk.clinician_id,
            lk.patient_id,
            exists (
                select 1 from public.session_checkins c
                where c.user_id = lk.patient_id and c.created_at >= p_since
            ) as checked_in,
            (
                select count(*) from public.session_checkins c
                where c.user_id = lk.patient_id and c.flagged and c.acknowledged_at is null
            ) as flags,
            (
                cp.id is not null
                and cp.created_at <= p_now - interval '7 days'
                and not exists (
                    select 1 from public.session_checkins c
                    where c.user_id = lk.patient_id
                      and c.plan_id = cp.id
                      and c.created_at > p_now - interval '7 days'
                      and c.created_at <= p_now
                )
            ) as inactive
        from links lk
        left join current_plan cp on cp.user_id = lk.patient_id
    )
    select
        pp.clinician_id,
        public.line_user_id_of(u.email::text),
        count(*)::int,
        count(*) filter (where pp.checked_in)::int,
        coalesce(sum(pp.flags), 0)::int,
        count(*) filter (where pp.inactive)::int
    from per_patient pp
    join auth.users u on u.id = pp.clinician_id
    where public.line_user_id_of(u.email::text) is not null
    group by pp.clinician_id, u.email
    order by pp.clinician_id;
$$;

revoke all on function public.daily_clinician_summaries(timestamptz, timestamptz) from public;
revoke all on function public.daily_clinician_summaries(timestamptz, timestamptz) from anon;
revoke all on function public.daily_clinician_summaries(timestamptz, timestamptz) from authenticated;
grant execute on function public.daily_clinician_summaries(timestamptz, timestamptz) to service_role;

-- ⚠️ Like the other auth.users readers (admin_user_overview, line_training_summary): if the definer
-- cannot read auth.users on your project, grant it once: `grant select on auth.users to postgres;`
