-- Grants and results of 20260927000000_daily_jobs.sql. LOCAL DEV ONLY; run via run_local.sh or
-- run_pglite.mjs. Every check raises on failure.
--
-- Cast (LINE-login accounts use the synthetic email line_<u+32 hex>@line.invalid):
--   C  clinician, LINE            P1 patient, LINE, plan assigned 10 days ago, never checked in
--   P2 patient, email login       P3 patient, LINE, checked in today, one open flag
--   S  stranger, LINE, own plan, no link

\set as_service 'reset role; set request.jwt.claims = ''''; set role service_role;'
\set as_p1 'reset role; set request.jwt.claims = ''{"sub":"00000000-0000-0000-0000-0000000001b1"}''; set role authenticated;'
\set as_anon 'reset role; set request.jwt.claims = ''''; set role anon;'
\set as_owner 'reset role; set request.jwt.claims = '''';'

\echo '  fixtures'
:as_owner
insert into auth.users (id, email) values
    ('00000000-0000-0000-0000-0000000001c1', 'line_u0000000000000000000000000000c1c1@line.invalid'),
    ('00000000-0000-0000-0000-0000000001b1', 'line_u0000000000000000000000000000b1b1@line.invalid'),
    ('00000000-0000-0000-0000-0000000001b2', 'p2@example.test'),
    ('00000000-0000-0000-0000-0000000001b3', 'line_u0000000000000000000000000000b3b3@line.invalid'),
    ('00000000-0000-0000-0000-0000000001d1', 'line_u0000000000000000000000000000d1d1@line.invalid');
insert into public.user_roles (user_id, role) values ('00000000-0000-0000-0000-0000000001c1', 'clinician');
insert into public.care_links (clinician_id, patient_id, invite_code, status, accepted_at) values
    ('00000000-0000-0000-0000-0000000001c1', '00000000-0000-0000-0000-0000000001b1', 'JOBP01', 'active', now()),
    ('00000000-0000-0000-0000-0000000001c1', '00000000-0000-0000-0000-0000000001b2', 'JOBP02', 'active', now()),
    ('00000000-0000-0000-0000-0000000001c1', '00000000-0000-0000-0000-0000000001b3', 'JOBP03', 'active', now());
insert into public.training_plans (id, user_id, name, assigned_by, created_at) values
    ('10000000-0000-0000-0000-0000000001b1', '00000000-0000-0000-0000-0000000001b1', 'Knee rehab', '00000000-0000-0000-0000-0000000001c1', now() - interval '10 days'),
    ('10000000-0000-0000-0000-0000000001b2', '00000000-0000-0000-0000-0000000001b2', 'Knee rehab', '00000000-0000-0000-0000-0000000001c1', now() - interval '1 day'),
    ('10000000-0000-0000-0000-0000000001b3', '00000000-0000-0000-0000-0000000001b3', 'Shoulder rehab', '00000000-0000-0000-0000-0000000001c1', now() - interval '10 days'),
    ('10000000-0000-0000-0000-0000000001d1', '00000000-0000-0000-0000-0000000001d1', 'My own plan', null, now() - interval '10 days');
insert into public.session_checkins (user_id, plan_id, pain_nrs, flagged, flag_reasons, created_at) values
    ('00000000-0000-0000-0000-0000000001b3', '10000000-0000-0000-0000-0000000001b3', 7, true, '["pain_high"]', now() - interval '1 hour');

\echo '  only service_role can execute the job functions'
:as_p1
do $$ begin
    begin
        perform * from public.daily_patient_reminders(now() - interval '12 hours');
        raise exception 'FAIL: authenticated executed daily_patient_reminders';
    exception when insufficient_privilege then null;
    end;
    begin
        perform * from public.daily_clinician_summaries(now() - interval '12 hours', now());
        raise exception 'FAIL: authenticated executed daily_clinician_summaries';
    exception when insufficient_privilege then null;
    end;
    begin
        perform public.claim_job_run('daily', current_date);
        raise exception 'FAIL: authenticated executed claim_job_run';
    exception when insufficient_privilege then null;
    end;
    begin
        perform public.line_user_id_of('line_u0000000000000000000000000000b1b1@line.invalid');
        raise exception 'FAIL: authenticated executed line_user_id_of';
    exception when insufficient_privilege then null;
    end;
    assert (select count(*) from public.job_runs) = 0, 'FAIL: authenticated can read job_runs';
end $$;

:as_anon
do $$ begin
    begin
        perform * from public.daily_patient_reminders(now() - interval '12 hours');
        raise exception 'FAIL: anon executed daily_patient_reminders';
    exception when insufficient_privilege then null;
    end;
end $$;

\echo '  reminders: linked LINE patients without a check-in today, LINE id restored with its U'
:as_service
do $$
declare n int; uid text;
begin
    select count(*), min(line_user_id) into n, uid
    from public.daily_patient_reminders(now() - interval '12 hours');
    assert n = 1, 'FAIL: expected exactly P1 to be reminded, got ' || n;
    assert uid = 'U0000000000000000000000000000b1b1', 'FAIL: wrong LINE user id ' || coalesce(uid, 'null');
    assert (select plan_name from public.daily_patient_reminders(now() - interval '12 hours')) = 'Knee rehab',
        'FAIL: wrong plan name';
    assert public.line_user_id_of('p2@example.test') is null, 'FAIL: email account mapped to a LINE id';
    assert public.line_user_id_of('line_Uabc@line.invalid') is null, 'FAIL: malformed synthetic email accepted';
end $$;

\echo '  clinician summary: patients, checked in today, open flags, inactive by rule (d)'
do $$
declare r record;
begin
    select * into r from public.daily_clinician_summaries(now() - interval '12 hours', now());
    assert r.line_user_id = 'U0000000000000000000000000000c1c1', 'FAIL: clinician LINE id';
    assert r.patients = 3, 'FAIL: patients ' || r.patients;
    assert r.checked_in_today = 1, 'FAIL: checked_in_today ' || r.checked_in_today;
    assert r.open_flags = 1, 'FAIL: open_flags ' || r.open_flags;
    -- P1: plan 10 days old, no check-in -> inactive. P2: plan 1 day old -> not yet. P3: checked in.
    assert r.inactive_7d = 1, 'FAIL: inactive_7d ' || r.inactive_7d;
end $$;

\echo '  the daily claim is atomic'
do $$ begin
    assert public.claim_job_run('daily', date '2026-09-16'), 'FAIL: first claim refused';
    assert not public.claim_job_run('daily', date '2026-09-16'), 'FAIL: second claim on the same day accepted';
    assert public.claim_job_run('daily', date '2026-09-17'), 'FAIL: next day refused';
end $$;

\echo '  revoking the link or the role stops the pushes'
:as_owner
update public.care_links set status = 'revoked', revoked_at = now()
where patient_id = '00000000-0000-0000-0000-0000000001b1';
:as_service
do $$ begin
    assert (select count(*) from public.daily_patient_reminders(now() - interval '12 hours')) = 0,
        'FAIL: a revoked patient is still reminded';
    assert (select patients from public.daily_clinician_summaries(now() - interval '12 hours', now())) = 2,
        'FAIL: revoked patient still counted in the summary';
end $$;
:as_owner
delete from public.user_roles where user_id = '00000000-0000-0000-0000-0000000001c1';
:as_service
do $$ begin
    assert (select count(*) from public.daily_clinician_summaries(now() - interval '12 hours', now())) = 0,
        'FAIL: a de-roled clinician still gets a summary';
end $$;

:as_owner
\echo '  all daily-job checks passed'
