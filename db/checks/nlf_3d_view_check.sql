-- RLS/function behaviour of 20260928000000_nlf_3d_view.sql, exercised as real roles. LOCAL DEV
-- ONLY; run via db/checks/run_local.sh or run_pglite.mjs. Every check raises on failure.
--
-- Runs after clinic_rls_check.sql and daily_jobs_check.sql against the SAME database (checks run
-- in sorted filename order), so fixture uuids here use a distinct '...2xxx' suffix family to avoid
-- colliding with those files' '...0xxx' / '...1xxx' users. It also reuses the admin fixture (a1)
-- those files already inserted, since it is still an admin at this point.
--
-- Cast: N1/N2 nlf_user patients, W1 the worker (nlf_worker only), U1 an ordinary authenticated
-- user with neither role.

\set as_owner 'reset role; set request.jwt.claims = '''';'
\set as_anon 'reset role; set request.jwt.claims = ''''; set role anon;'
\set as_n1 'reset role; set request.jwt.claims = ''{"sub":"00000000-0000-0000-0000-0000000002a1"}''; set role authenticated;'
\set as_n2 'reset role; set request.jwt.claims = ''{"sub":"00000000-0000-0000-0000-0000000002a2"}''; set role authenticated;'
\set as_w1 'reset role; set request.jwt.claims = ''{"sub":"00000000-0000-0000-0000-0000000002b1"}''; set role authenticated;'
\set as_u1 'reset role; set request.jwt.claims = ''{"sub":"00000000-0000-0000-0000-0000000002c1"}''; set role authenticated;'
\set as_a1 'reset role; set request.jwt.claims = ''{"sub":"00000000-0000-0000-0000-0000000000a1"}''; set role authenticated;'

\echo '  fixtures'
:as_owner
insert into auth.users (id, email) values
    ('00000000-0000-0000-0000-0000000002a1', 'nlf-n1@example.test'),
    ('00000000-0000-0000-0000-0000000002a2', 'nlf-n2@example.test'),
    ('00000000-0000-0000-0000-0000000002b1', 'nlf-worker@example.test'),
    ('00000000-0000-0000-0000-0000000002c1', 'nlf-u1@example.test');
insert into public.user_roles (user_id, role) values
    ('00000000-0000-0000-0000-0000000002a1', 'nlf_user'),
    ('00000000-0000-0000-0000-0000000002a2', 'nlf_user'),
    ('00000000-0000-0000-0000-0000000002b1', 'nlf_worker');

:as_n1
insert into public.videos (user_id, video_id, storage_key) values
    ('00000000-0000-0000-0000-0000000002a1', 'vid-n1', 'uploads/2a1/vid-n1'),
    ('00000000-0000-0000-0000-0000000002a1', 'vid-stale', 'uploads/2a1/vid-stale'),
    ('00000000-0000-0000-0000-0000000002a1', 'vid-attempts3', 'uploads/2a1/vid-attempts3'),
    ('00000000-0000-0000-0000-0000000002a1', 'vid-requeue', 'uploads/2a1/vid-requeue');
:as_n2
insert into public.videos (user_id, video_id, storage_key) values
    ('00000000-0000-0000-0000-0000000002a2', 'vid-n2', 'uploads/2a2/vid-n2');

-- ---------------------------------------------------------------------------------------------
\echo '  (a) a user without nlf_user gets 42501 from request_nlf_job'
:as_u1
do $$ begin
    begin
        perform public.request_nlf_job('whatever');
        raise exception 'FAIL: non-nlf user requested a job';
    exception when insufficient_privilege then null;
    end;
end $$;

\echo '  anon cannot call the gate functions or request a job'
:as_anon
do $$ begin
    begin
        perform public.has_nlf_access('00000000-0000-0000-0000-0000000002a1');
        raise exception 'FAIL: anon executed has_nlf_access';
    exception when insufficient_privilege then null;
    end;
    begin
        perform public.is_nlf_worker('00000000-0000-0000-0000-0000000002a1');
        raise exception 'FAIL: anon executed is_nlf_worker';
    exception when insufficient_privilege then null;
    end;
    begin
        perform public.request_nlf_job('whatever');
        raise exception 'FAIL: anon executed request_nlf_job';
    exception when insufficient_privilege then null;
    end;
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  (b) an enabled user can request only their own video'
:as_n1
do $$
declare j public.nlf_jobs;
begin
    j := public.request_nlf_job('vid-n1');
    assert j.status = 'queued', 'FAIL: fresh request not queued: ' || j.status;
    assert j.user_id = '00000000-0000-0000-0000-0000000002a1', 'FAIL: wrong owner on job';

    begin
        perform public.request_nlf_job('vid-n2');
        raise exception 'FAIL: N1 requested N2''s video';
    exception when no_data_found then null;
    end;

    begin
        perform public.request_nlf_job('no-such-video');
        raise exception 'FAIL: N1 requested a video that does not exist';
    exception when no_data_found then null;
    end;
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  (c) a user cannot read another user''s nlf_jobs row'
:as_n2
do $$ begin
    assert (select count(*) from public.nlf_jobs where video_id = 'vid-n1') = 0,
        'FAIL: N2 can see N1''s job';
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  (d) a non-worker gets 42501 from every worker function'
:as_u1
do $$ begin
    begin
        perform * from public.nlf_claim_job();
        raise exception 'FAIL: non-worker claimed a job';
    exception when insufficient_privilege then null;
    end;
    begin
        perform public.nlf_complete_job('00000000-0000-0000-0000-000000000000', 'k', '{}'::jsonb);
        raise exception 'FAIL: non-worker completed a job';
    exception when insufficient_privilege then null;
    end;
    begin
        perform public.nlf_fail_job('00000000-0000-0000-0000-000000000000', 'err');
        raise exception 'FAIL: non-worker failed a job';
    exception when insufficient_privilege then null;
    end;
    begin
        perform public.nlf_worker_heartbeat();
        raise exception 'FAIL: non-worker sent a heartbeat';
    exception when insufficient_privilege then null;
    end;
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  (e) the worker has no RLS access of its own to jobs, videos or analyses'
:as_w1
do $$ begin
    assert (select count(*) from public.nlf_jobs) = 0, 'FAIL: worker reads nlf_jobs via RLS';
    assert (select count(*) from public.videos) = 0, 'FAIL: worker reads videos via RLS';
    assert (select count(*) from public.analyses) = 0, 'FAIL: worker reads analyses via RLS';
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  (f) nlf_claim_job returns storage_key, attempts and the latest analysis (extra fields dropped)'
:as_n1
insert into public.analyses (id, user_id, video_id, movement, result) values
    ('00000000-0000-0000-0000-0000000002e1', '00000000-0000-0000-0000-0000000002a1', 'vid-n1', 'squat',
     '{"pose": {"fps": 30.5}, "reps": {"segments": [[0, 10], [11, 20]]}, "detections": [
        {"fault_id": "knee_valgus", "peak_frame": 15, "confidence": 0.9, "note": "drop me"},
        {"fault_id": "torso_lean", "peak_frame": 22, "extra": "also drop"}
     ]}'::jsonb);

:as_w1
do $$
declare r record;
begin
    select * into r from public.nlf_claim_job();
    assert found, 'FAIL: nlf_claim_job returned no row';
    assert r.owner_id = '00000000-0000-0000-0000-0000000002a1', 'FAIL: wrong owner_id ' || r.owner_id;
    assert r.video_id = 'vid-n1', 'FAIL: wrong video_id ' || r.video_id;
    assert r.attempts = 1, 'FAIL: attempts ' || r.attempts;
    assert r.storage_key = 'uploads/2a1/vid-n1', 'FAIL: storage_key ' || coalesce(r.storage_key, 'null');
    assert r.analysis_id = '00000000-0000-0000-0000-0000000002e1',
        'FAIL: analysis_id ' || coalesce(r.analysis_id::text, 'null');
    assert r.movement = 'squat', 'FAIL: movement ' || coalesce(r.movement, 'null');
    assert r.pose_fps = 30.5, 'FAIL: pose_fps ' || coalesce(r.pose_fps::text, 'null');
    assert r.rep_segments = '[[0, 10], [11, 20]]'::jsonb,
        'FAIL: rep_segments ' || coalesce(r.rep_segments::text, 'null');
    assert r.detections = '[{"fault_id": "knee_valgus", "peak_frame": 15}, {"fault_id": "torso_lean", "peak_frame": 22}]'::jsonb,
        'FAIL: detections ' || coalesce(r.detections::text, 'null');

    perform set_config('nlf_check.claimed_job', r.job_id::text, false);
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  (g) deleting the video cascades the job; completing the gone job then returns false'
:as_n1
do $$ begin
    delete from public.videos
    where user_id = '00000000-0000-0000-0000-0000000002a1' and video_id = 'vid-n1';
    assert (select count(*) from public.nlf_jobs where video_id = 'vid-n1') = 0,
        'FAIL: job survived its video''s delete';
end $$;

:as_w1
do $$
declare v_job_id uuid := current_setting('nlf_check.claimed_job')::uuid;
    ok boolean;
begin
    ok := public.nlf_complete_job(v_job_id, 'result-key', '{}'::jsonb);
    assert not ok, 'FAIL: completed a job whose video was deleted';
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  (h) a stale claim is re-queued by the next claim; one past its attempts becomes failed'
:as_n1
do $$ begin perform public.request_nlf_job('vid-stale'); end $$;

:as_w1
do $$
declare r record;
begin
    select * into r from public.nlf_claim_job();
    assert r.video_id = 'vid-stale', 'FAIL: expected to claim vid-stale, got ' || coalesce(r.video_id, 'null');
    assert r.attempts = 1, 'FAIL: first claim attempts ' || r.attempts;
end $$;

:as_owner
update public.nlf_jobs set claimed_at = now() - interval '31 minutes'
where video_id = 'vid-stale';

:as_w1
do $$
declare r record;
begin
    select * into r from public.nlf_claim_job();
    assert r.video_id = 'vid-stale', 'FAIL: stale claim not reclaimed, got ' || coalesce(r.video_id, 'null');
    assert r.attempts = 2, 'FAIL: reclaimed attempts ' || r.attempts;
end $$;

:as_n1
do $$ begin perform public.request_nlf_job('vid-attempts3'); end $$;

:as_owner
update public.nlf_jobs
set status = 'claimed', attempts = 3, claimed_at = now() - interval '31 minutes'
where video_id = 'vid-attempts3';

:as_w1
do $$ begin
    -- drains whatever else is queued; only vid-attempts3's post-state is asserted below.
    perform public.nlf_claim_job();
end $$;

:as_owner
do $$ begin
    assert (select status from public.nlf_jobs where video_id = 'vid-attempts3') = 'failed',
        'FAIL: attempts-exhausted stale claim was not failed';
    assert (select error from public.nlf_jobs where video_id = 'vid-attempts3') = 'claim timed out',
        'FAIL: wrong error message on a timed-out claim';
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  (i) request_nlf_job on a failed job re-queues it with attempts reset to 0'
:as_n1
do $$ begin perform public.request_nlf_job('vid-requeue'); end $$;

:as_owner
update public.nlf_jobs
set status = 'failed', attempts = 5, error = 'boom', claimed_at = now()
where video_id = 'vid-requeue';

:as_n1
do $$
declare j public.nlf_jobs;
begin
    j := public.request_nlf_job('vid-requeue');
    assert j.status = 'queued', 'FAIL: failed job not re-queued: ' || j.status;
    assert j.attempts = 0, 'FAIL: attempts not reset: ' || j.attempts;
    assert j.error is null, 'FAIL: error not cleared';
    assert j.claimed_at is null, 'FAIL: claimed_at not cleared';
end $$;

:as_w1
do $$ begin
    -- drain the re-queued job so the next test's FIFO claim isn't conflated with it.
    perform public.nlf_claim_job();
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  nlf_fail_job requeues under the attempt limit, and fails once exhausted'
:as_n1
insert into public.videos (user_id, video_id, storage_key) values
    ('00000000-0000-0000-0000-0000000002a1', 'vid-fail-cycle', 'uploads/2a1/vid-fail-cycle'),
    ('00000000-0000-0000-0000-0000000002a1', 'vid-complete', 'uploads/2a1/vid-complete');
do $$ begin perform public.request_nlf_job('vid-fail-cycle'); end $$;

:as_w1
do $$
declare r record;
begin
    select * into r from public.nlf_claim_job();
    assert r.video_id = 'vid-fail-cycle',
        'FAIL: expected to claim vid-fail-cycle, got ' || coalesce(r.video_id, 'null');
    perform set_config('nlf_check.fail_job', r.job_id::text, false);
    perform public.nlf_fail_job(r.job_id, 'transient');
end $$;

:as_n1
do $$
declare v_job_id uuid := current_setting('nlf_check.fail_job')::uuid;
begin
    assert (select status from public.nlf_jobs where id = v_job_id) = 'queued',
        'FAIL: fail under the attempt limit did not re-queue';
    assert (select claimed_at from public.nlf_jobs where id = v_job_id) is null,
        'FAIL: claimed_at not cleared on requeue';
    assert (select error from public.nlf_jobs where id = v_job_id) = 'transient',
        'FAIL: error not stored on requeue';
end $$;

:as_owner
update public.nlf_jobs
set status = 'claimed', attempts = 3, claimed_at = now()
where video_id = 'vid-fail-cycle';

:as_w1
do $$
declare v_job_id uuid := current_setting('nlf_check.fail_job')::uuid;
begin
    perform public.nlf_fail_job(v_job_id, 'permanent');
end $$;

:as_n1
do $$
declare v_job_id uuid := current_setting('nlf_check.fail_job')::uuid;
begin
    assert (select status from public.nlf_jobs where id = v_job_id) = 'failed',
        'FAIL: fail at the attempt limit did not fail the job';
    assert (select error from public.nlf_jobs where id = v_job_id) = 'permanent',
        'FAIL: error not updated on the final fail';
    assert (select claimed_at from public.nlf_jobs where id = v_job_id) is not null,
        'FAIL: claimed_at was cleared on a failed (not requeued) job';
end $$;

\echo '  nlf_complete_job marks a claimed job done and stores result_key/meta'
:as_n1
do $$ begin perform public.request_nlf_job('vid-complete'); end $$;

:as_w1
do $$
declare r record;
begin
    select * into r from public.nlf_claim_job();
    assert r.video_id = 'vid-complete',
        'FAIL: expected to claim vid-complete, got ' || coalesce(r.video_id, 'null');
    assert public.nlf_complete_job(r.job_id, 'result-key-2', '{"foo": 1}'::jsonb),
        'FAIL: nlf_complete_job returned false for a freshly claimed job';
    perform set_config('nlf_check.done_job', r.job_id::text, false);
end $$;

:as_n1
do $$
declare v_job_id uuid := current_setting('nlf_check.done_job')::uuid;
begin
    assert (select status from public.nlf_jobs where id = v_job_id) = 'done', 'FAIL: job not marked done';
    assert (select result_key from public.nlf_jobs where id = v_job_id) = 'result-key-2',
        'FAIL: result_key not stored';
    assert (select meta from public.nlf_jobs where id = v_job_id) = '{"foo": 1}'::jsonb,
        'FAIL: meta not stored';
    assert (select error from public.nlf_jobs where id = v_job_id) is null,
        'FAIL: error not cleared on completion';
end $$;

\echo '  nlf_fail_job is a no-op on a job that is not currently claimed'
:as_w1
do $$
declare v_job_id uuid := current_setting('nlf_check.done_job')::uuid;
begin
    perform public.nlf_fail_job(v_job_id, 'should not apply');
end $$;

:as_n1
do $$
declare v_job_id uuid := current_setting('nlf_check.done_job')::uuid;
begin
    assert (select status from public.nlf_jobs where id = v_job_id) = 'done',
        'FAIL: nlf_fail_job changed a done job''s status';
    assert (select error from public.nlf_jobs where id = v_job_id) is null,
        'FAIL: nlf_fail_job set an error on a done job';
    assert (select result_key from public.nlf_jobs where id = v_job_id) = 'result-key-2',
        'FAIL: nlf_fail_job touched a done job''s result_key';
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  nlf_claim_job returns pose_fps NULL (not a raised error) when fps is not a JSON number'
:as_n1
insert into public.videos (user_id, video_id, storage_key) values
    ('00000000-0000-0000-0000-0000000002a1', 'vid-bad-fps', 'uploads/2a1/vid-bad-fps');
insert into public.analyses (id, user_id, video_id, movement, result) values
    ('00000000-0000-0000-0000-0000000002e2', '00000000-0000-0000-0000-0000000002a1', 'vid-bad-fps', 'squat',
     '{"pose": {"fps": "thirty"}, "reps": {"segments": []}, "detections": []}'::jsonb);
do $$ begin perform public.request_nlf_job('vid-bad-fps'); end $$;

:as_w1
do $$
declare r record;
begin
    select * into r from public.nlf_claim_job();
    assert r.video_id = 'vid-bad-fps',
        'FAIL: expected to claim vid-bad-fps, got ' || coalesce(r.video_id, 'null');
    assert r.pose_fps is null, 'FAIL: non-numeric fps did not come back as NULL';
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  worker heartbeat: the worker updates it, any signed-in user can read it'
:as_w1
do $$ begin perform public.nlf_worker_heartbeat(); end $$;

:as_u1
do $$ begin
    assert public.nlf_worker_last_seen() is not null, 'FAIL: last_seen not visible after heartbeat';
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  (j) admin_list_users exposes has_nlf, still admin-only'
:as_a1
do $$ begin
    assert (select has_nlf from public.admin_list_users() where id = '00000000-0000-0000-0000-0000000002a1'),
        'FAIL: has_nlf false for the enabled user';
    assert not (select has_nlf from public.admin_list_users() where id = '00000000-0000-0000-0000-0000000002c1'),
        'FAIL: has_nlf true for a user without the role';
end $$;

:as_n1
do $$ begin
    begin
        perform * from public.admin_list_users();
        raise exception 'FAIL: non-admin listed users';
    exception when insufficient_privilege then null;
    end;
end $$;

:as_owner
\echo '  all nlf 3d view checks passed'
