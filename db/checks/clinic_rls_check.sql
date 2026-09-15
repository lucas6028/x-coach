-- RLS behaviour of 20260916000000_clinic.sql, exercised as real roles. LOCAL DEV ONLY; run via
-- db/checks/run_local.sh. Every check is a DO block that raises on failure, so psql's
-- ON_ERROR_STOP turns the first broken policy into a non-zero exit.
--
-- The identity switch matters: connected as postgres (owner/superuser) RLS is bypassed and every
-- policy would "pass". Each block below runs after `set role authenticated` with the caller's id
-- in request.jwt.claims, which is how PostgREST presents a user JWT.
--
-- Cast: A1 admin (+clinician), C1 clinician, C2 clinician (never linked), B1 patient, D1 stranger.

\set as_a1 'reset role; set request.jwt.claims = ''{"sub":"00000000-0000-0000-0000-0000000000a1"}''; set role authenticated;'
\set as_c1 'reset role; set request.jwt.claims = ''{"sub":"00000000-0000-0000-0000-0000000000c1"}''; set role authenticated;'
\set as_c2 'reset role; set request.jwt.claims = ''{"sub":"00000000-0000-0000-0000-0000000000c2"}''; set role authenticated;'
\set as_b1 'reset role; set request.jwt.claims = ''{"sub":"00000000-0000-0000-0000-0000000000b1"}''; set role authenticated;'
\set as_d1 'reset role; set request.jwt.claims = ''{"sub":"00000000-0000-0000-0000-0000000000d1"}''; set role authenticated;'
\set as_anon 'reset role; set request.jwt.claims = ''''; set role anon;'
\set as_owner 'reset role; set request.jwt.claims = '''';'

-- ---------------------------------------------------------------------------------------------
\echo '  fixtures: users and roles (admin also holds clinician -> composite user_roles key)'
insert into auth.users (id, email, raw_user_meta_data) values
    ('00000000-0000-0000-0000-0000000000a1', 'admin@example.test',    '{"full_name": "Admin"}'),
    ('00000000-0000-0000-0000-0000000000c1', 'clin@example.test',     '{"full_name": "Dr Clin"}'),
    ('00000000-0000-0000-0000-0000000000c2', 'clin2@example.test',    '{}'),
    ('00000000-0000-0000-0000-0000000000b1', 'pat@example.test',      '{"name": "Pat"}'),
    ('00000000-0000-0000-0000-0000000000d1', 'stranger@example.test', '{}');
insert into public.user_roles (user_id, role) values
    ('00000000-0000-0000-0000-0000000000a1', 'admin'),
    ('00000000-0000-0000-0000-0000000000a1', 'clinician'),
    ('00000000-0000-0000-0000-0000000000c1', 'clinician'),
    ('00000000-0000-0000-0000-0000000000c2', 'clinician');

do $$ begin
    begin
        insert into public.user_roles (user_id, role)
        values ('00000000-0000-0000-0000-0000000000d1', 'superuser');
        raise exception 'FAIL: user_roles accepted an unknown role';
    exception when check_violation then null;
    end;
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  patient writes own data; check-ins are append-only'
:as_b1
do $$
declare n int;
begin
    insert into public.training_plans (id, user_id, name)
    values ('10000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-0000000000b1', 'Own plan');
    insert into public.plan_items (id, plan_id, user_id, day_index, movement)
    values ('20000000-0000-0000-0000-000000000001', '10000000-0000-0000-0000-000000000001',
            '00000000-0000-0000-0000-0000000000b1', 1, 'Squat');
    insert into public.analyses (id, user_id, video_id, result)
    values ('30000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-0000000000b1', 'v1', '{}');
    insert into public.videos (user_id, video_id, storage_key)
    values ('00000000-0000-0000-0000-0000000000b1', 'v1', 'uploads/b1/v1');
    insert into public.conversations (user_id, video_id)
    values ('00000000-0000-0000-0000-0000000000b1', 'v1');
    insert into public.session_checkins (id, user_id, plan_id, pain_nrs)
    values ('40000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-0000000000b1',
            '10000000-0000-0000-0000-000000000001', 3);

    begin
        insert into public.session_checkins (user_id, pain_nrs, acknowledged_at)
        values ('00000000-0000-0000-0000-0000000000b1', 2, now());
        raise exception 'FAIL: patient inserted a pre-acknowledged check-in';
    exception when insufficient_privilege then null;
    end;

    begin
        insert into public.session_checkins (user_id, pain_nrs)
        values ('00000000-0000-0000-0000-0000000000d1', 2);
        raise exception 'FAIL: patient inserted a check-in for another user';
    exception when insufficient_privilege then null;
    end;

    update public.session_checkins set pain_nrs = 0;
    get diagnostics n = row_count;
    assert n = 0, 'FAIL: patient updated a check-in (must be append-only)';

    delete from public.session_checkins;
    get diagnostics n = row_count;
    assert n = 0, 'FAIL: patient deleted a check-in (must be append-only)';

    begin
        insert into public.session_checkins (user_id, pain_nrs)
        values ('00000000-0000-0000-0000-0000000000b1', 11);
        raise exception 'FAIL: pain_nrs 11 accepted';
    exception when check_violation then null;
    end;
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  unlinked clinician sees nothing and cannot forge links or assignments'
:as_c1
do $$
declare n int;
begin
    assert (select count(*) from public.training_plans   where user_id = '00000000-0000-0000-0000-0000000000b1') = 0, 'FAIL: C1 sees plans before link';
    assert (select count(*) from public.plan_items       where user_id = '00000000-0000-0000-0000-0000000000b1') = 0, 'FAIL: C1 sees items before link';
    assert (select count(*) from public.analyses         where user_id = '00000000-0000-0000-0000-0000000000b1') = 0, 'FAIL: C1 sees analyses before link';
    assert (select count(*) from public.session_checkins where user_id = '00000000-0000-0000-0000-0000000000b1') = 0, 'FAIL: C1 sees check-ins before link';
    assert (select count(*) from public.clinic_patients()) = 0, 'FAIL: C1 has patients before link';

    insert into public.care_links (clinician_id, invite_code)
    values ('00000000-0000-0000-0000-0000000000c1', 'ABC234');

    begin
        insert into public.care_links (clinician_id, patient_id, invite_code, status, accepted_at)
        values ('00000000-0000-0000-0000-0000000000c1', '00000000-0000-0000-0000-0000000000b1', 'FORGE2', 'active', now());
        raise exception 'FAIL: clinician created an active link without consent';
    exception when insufficient_privilege then null;
    end;

    begin
        insert into public.care_links (clinician_id, invite_code)
        values ('00000000-0000-0000-0000-0000000000c2', 'OTHER2');
        raise exception 'FAIL: clinician created an invite for another clinician';
    exception when insufficient_privilege then null;
    end;

    begin
        insert into public.care_links (clinician_id, invite_code, expires_at)
        values ('00000000-0000-0000-0000-0000000000c1', 'LONG22', now() + interval '1 year');
        raise exception 'FAIL: invite with a one-year expiry accepted';
    exception when insufficient_privilege then null;
    end;

    begin
        insert into public.training_plans (user_id, name, assigned_by)
        values ('00000000-0000-0000-0000-0000000000b1', 'Early', '00000000-0000-0000-0000-0000000000c1');
        raise exception 'FAIL: C1 assigned a plan before the patient accepted';
    exception when insufficient_privilege then null;
    end;
end $$;

\echo '  non-clinician cannot create invites or list patients; anon cannot call the helpers'
:as_d1
do $$ begin
    begin
        insert into public.care_links (clinician_id, invite_code)
        values ('00000000-0000-0000-0000-0000000000d1', 'STR234');
        raise exception 'FAIL: non-clinician created an invite';
    exception when insufficient_privilege then null;
    end;

    begin
        perform * from public.clinic_patients();
        raise exception 'FAIL: non-clinician listed patients';
    exception when insufficient_privilege then null;
    end;
end $$;

:as_anon
do $$ begin
    begin
        perform public.is_linked('00000000-0000-0000-0000-0000000000c1', '00000000-0000-0000-0000-0000000000b1');
        raise exception 'FAIL: anon executed is_linked';
    exception when insufficient_privilege then null;
    end;
    begin
        perform public.accept_care_invite('ABC234');
        raise exception 'FAIL: anon executed accept_care_invite';
    exception when insufficient_privilege then null;
    end;
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  patient accepts: pending invite invisible, bad code rejected, code normalized'
:as_b1
do $$
declare r jsonb;
begin
    assert (select count(*) from public.care_links) = 0, 'FAIL: patient sees a pending invite before accepting';

    r := public.accept_care_invite('NOPE00');
    assert r ->> 'error' = 'invalid', 'FAIL: unknown code not rejected: ' || r::text;

    r := public.accept_care_invite(' abc234 ');
    assert r -> 'link' ->> 'status' = 'active', 'FAIL: accept did not activate: ' || r::text;
    assert r -> 'link' ->> 'patient_id' = '00000000-0000-0000-0000-0000000000b1', 'FAIL: wrong patient on link';

    r := public.accept_care_invite('ABC234');
    assert r ->> 'error' = 'invalid', 'FAIL: a used code was accepted twice';

    assert (select count(*) from public.care_links) = 1, 'FAIL: patient does not see the accepted link';
    assert (select display_name from public.my_clinicians()) = 'Dr Clin', 'FAIL: my_clinicians display name';
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  linked clinician reads plans/results/check-ins (not clips or chat), writes only what they assigned'
:as_c1
do $$
declare n int; r jsonb;
begin
    assert (select count(*) from public.training_plans   where user_id = '00000000-0000-0000-0000-0000000000b1') = 1, 'FAIL: C1 cannot read patient plans';
    assert (select count(*) from public.plan_items       where user_id = '00000000-0000-0000-0000-0000000000b1') = 1, 'FAIL: C1 cannot read patient items';
    assert (select count(*) from public.analyses         where user_id = '00000000-0000-0000-0000-0000000000b1') = 1, 'FAIL: C1 cannot read patient analyses';
    assert (select count(*) from public.session_checkins where user_id = '00000000-0000-0000-0000-0000000000b1') = 1, 'FAIL: C1 cannot read patient check-ins';
    assert (select count(*) from public.clinic_patients()) = 1, 'FAIL: clinic_patients count';
    assert (select display_name from public.clinic_patients()) = 'Pat', 'FAIL: clinic_patients display name';

    -- No clinician policy on videos or conversations: the analysis result is readable, the clip
    -- (routers/videos.py resolves it through videos.storage_key, RLS-only) and the chat are not.
    assert (select count(*) from public.videos        where user_id = '00000000-0000-0000-0000-0000000000b1') = 0, 'FAIL: C1 can read the patient''s video rows (clip reachable)';
    assert (select count(*) from public.conversations where user_id = '00000000-0000-0000-0000-0000000000b1') = 0, 'FAIL: C1 can read the patient''s coaching chat';

    update public.training_plans set name = 'hijack' where id = '10000000-0000-0000-0000-000000000001';
    get diagnostics n = row_count;
    assert n = 0, 'FAIL: C1 edited a plan the patient made';

    delete from public.training_plans where user_id = '00000000-0000-0000-0000-0000000000b1';
    get diagnostics n = row_count;
    assert n = 0, 'FAIL: C1 deleted a patient plan';

    insert into public.training_plans (id, user_id, name, assigned_by)
    values ('10000000-0000-0000-0000-000000000002', '00000000-0000-0000-0000-0000000000b1',
            'Knee rehab', '00000000-0000-0000-0000-0000000000c1');

    begin
        insert into public.training_plans (user_id, name)
        values ('00000000-0000-0000-0000-0000000000b1', 'Unattributed');
        raise exception 'FAIL: C1 created a patient plan without assigned_by';
    exception when insufficient_privilege then null;
    end;

    begin
        insert into public.training_plans (user_id, name, assigned_by)
        values ('00000000-0000-0000-0000-0000000000b1', 'Framed', '00000000-0000-0000-0000-0000000000c2');
        raise exception 'FAIL: C1 attributed an assignment to another clinician';
    exception when insufficient_privilege then null;
    end;

    insert into public.plan_items (id, plan_id, user_id, day_index, movement)
    values ('20000000-0000-0000-0000-000000000002', '10000000-0000-0000-0000-000000000002',
            '00000000-0000-0000-0000-0000000000b1', 1, 'Lunge');

    begin
        insert into public.plan_items (plan_id, user_id, day_index, movement)
        values ('10000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-0000000000b1', 2, 'Row');
        raise exception 'FAIL: C1 added an item to the patient''s own plan';
    exception when insufficient_privilege then null;
    end;

    update public.plan_items set reps = 12 where plan_id = '10000000-0000-0000-0000-000000000002';
    get diagnostics n = row_count;
    assert n = 1, 'FAIL: C1 could not edit an item of the plan they assigned';

    update public.plan_items set reps = 12 where plan_id = '10000000-0000-0000-0000-000000000001';
    get diagnostics n = row_count;
    assert n = 0, 'FAIL: C1 edited an item of the patient''s own plan';

    update public.care_links set patient_id = '00000000-0000-0000-0000-0000000000d1';
    get diagnostics n = row_count;
    assert n = 0, 'FAIL: C1 rewrote a care link';

    delete from public.care_links;
    get diagnostics n = row_count;
    assert n = 0, 'FAIL: C1 deleted a care link';

    update public.session_checkins set pain_nrs = 0;
    get diagnostics n = row_count;
    assert n = 0, 'FAIL: C1 rewrote a patient check-in';

    r := public.ack_checkin_flag('40000000-0000-0000-0000-000000000001');
    assert r ->> 'error' = 'not_flagged', 'FAIL: acked an unflagged check-in: ' || r::text;

    insert into public.care_links (clinician_id, invite_code)
    values ('00000000-0000-0000-0000-0000000000c1', 'SELF23');
    r := public.accept_care_invite('SELF23');
    assert r ->> 'error' = 'self', 'FAIL: clinician accepted own invite: ' || r::text;
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  patient sees the assigned plan as their own; flagged check-in'
:as_b1
do $$ begin
    assert (select count(*) from public.training_plans) = 2, 'FAIL: patient does not own the assigned plan';
    assert (select assigned_by from public.training_plans where id = '10000000-0000-0000-0000-000000000002')
        = '00000000-0000-0000-0000-0000000000c1', 'FAIL: assigned_by not visible to patient';

    insert into public.session_checkins (id, user_id, plan_id, pain_nrs, flagged, flag_reasons)
    values ('40000000-0000-0000-0000-000000000002', '00000000-0000-0000-0000-0000000000b1',
            '10000000-0000-0000-0000-000000000002', 7, true, '["pain_high"]');
end $$;

\echo '  unlinked clinician and stranger still see nothing'
:as_c2
do $$
declare r jsonb;
begin
    assert (select count(*) from public.training_plans   where user_id = '00000000-0000-0000-0000-0000000000b1') = 0, 'FAIL: C2 sees patient plans';
    assert (select count(*) from public.analyses         where user_id = '00000000-0000-0000-0000-0000000000b1') = 0, 'FAIL: C2 sees patient analyses';
    assert (select count(*) from public.session_checkins where user_id = '00000000-0000-0000-0000-0000000000b1') = 0, 'FAIL: C2 sees patient check-ins';
    r := public.ack_checkin_flag('40000000-0000-0000-0000-000000000002');
    assert r ->> 'error' = 'not_found', 'FAIL: C2 acked a stranger''s check-in: ' || r::text;
end $$;

:as_d1
do $$
declare r jsonb;
begin
    assert (select count(*) from public.training_plans)   = 0, 'FAIL: stranger sees plans';
    assert (select count(*) from public.plan_items)       = 0, 'FAIL: stranger sees items';
    assert (select count(*) from public.analyses)         = 0, 'FAIL: stranger sees analyses';
    assert (select count(*) from public.session_checkins) = 0, 'FAIL: stranger sees check-ins';
    assert (select count(*) from public.care_links)       = 0, 'FAIL: stranger sees care links';
    r := public.revoke_care_link((select id from public.care_links limit 1));
    assert r ->> 'error' = 'not_found', 'FAIL: stranger revoked a link';
end $$;

\echo '  linked clinician acknowledges the flag, idempotently'
:as_c1
do $$
declare r jsonb; first_ack text;
begin
    r := public.ack_checkin_flag('40000000-0000-0000-0000-000000000002');
    assert r -> 'checkin' ->> 'acknowledged_by' = '00000000-0000-0000-0000-0000000000c1', 'FAIL: ack not recorded: ' || r::text;
    first_ack := r -> 'checkin' ->> 'acknowledged_at';
    r := public.ack_checkin_flag('40000000-0000-0000-0000-000000000002');
    assert r -> 'checkin' ->> 'acknowledged_at' = first_ack, 'FAIL: second ack moved acknowledged_at';
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  expired and duplicate invites'
:as_owner
insert into public.care_links (clinician_id, invite_code, created_at, expires_at)
values ('00000000-0000-0000-0000-0000000000c1', 'EXP234', now() - interval '8 days', now() - interval '1 day');
insert into public.care_links (clinician_id, invite_code)
values ('00000000-0000-0000-0000-0000000000c1', 'DUP234');

:as_b1
do $$
declare r jsonb; first_link uuid;
begin
    r := public.accept_care_invite('EXP234');
    assert r ->> 'error' = 'invalid', 'FAIL: expired code accepted: ' || r::text;

    select id into first_link from public.care_links where status = 'active';
    r := public.accept_care_invite('DUP234');
    assert (r -> 'link' ->> 'id')::uuid = first_link, 'FAIL: second invite from the same clinician did not return the existing link: ' || r::text;
    assert (select count(*) from public.care_links where status = 'active') = 1, 'FAIL: duplicate active link created';
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  revoking the clinician role ends access without touching the link'
:as_owner
delete from public.user_roles where user_id = '00000000-0000-0000-0000-0000000000c1' and role = 'clinician';
:as_c1
do $$ begin
    assert (select count(*) from public.training_plans where user_id = '00000000-0000-0000-0000-0000000000b1') = 0, 'FAIL: de-roled clinician still reads patient plans';
end $$;
:as_owner
insert into public.user_roles (user_id, role) values ('00000000-0000-0000-0000-0000000000c1', 'clinician');

-- ---------------------------------------------------------------------------------------------
\echo '  patient revokes: access ends at once'
:as_b1
do $$
declare r jsonb;
begin
    r := public.revoke_care_link((select id from public.care_links where status = 'active'));
    assert r -> 'link' ->> 'status' = 'revoked', 'FAIL: patient could not revoke: ' || r::text;
    r := public.revoke_care_link((r -> 'link' ->> 'id')::uuid);
    assert r ->> 'error' = 'not_found', 'FAIL: revoked link revoked twice';
    assert (select count(*) from public.my_clinicians()) = 0, 'FAIL: my_clinicians lists a revoked link';
end $$;

:as_c1
do $$ begin
    assert (select count(*) from public.training_plans   where user_id = '00000000-0000-0000-0000-0000000000b1') = 0, 'FAIL: C1 reads plans after revoke';
    assert (select count(*) from public.analyses         where user_id = '00000000-0000-0000-0000-0000000000b1') = 0, 'FAIL: C1 reads analyses after revoke';
    assert (select count(*) from public.session_checkins where user_id = '00000000-0000-0000-0000-0000000000b1') = 0, 'FAIL: C1 reads check-ins after revoke';
    assert (select count(*) from public.clinic_patients()) = 0, 'FAIL: clinic_patients lists a revoked link';
end $$;

-- ---------------------------------------------------------------------------------------------
\echo '  admin_list_users exposes is_clinician, still admin-only'
:as_a1
do $$ begin
    assert (select is_clinician from public.admin_list_users() where id = '00000000-0000-0000-0000-0000000000c1'), 'FAIL: is_clinician false for C1';
    assert (select is_admin and is_clinician from public.admin_list_users() where id = '00000000-0000-0000-0000-0000000000a1'), 'FAIL: A1 should be admin and clinician';
    assert not (select is_clinician from public.admin_list_users() where id = '00000000-0000-0000-0000-0000000000d1'), 'FAIL: is_clinician true for D1';
    assert public.is_admin('00000000-0000-0000-0000-0000000000a1'), 'FAIL: is_admin broken by the composite key';
end $$;

:as_b1
do $$ begin
    begin
        perform * from public.admin_list_users();
        raise exception 'FAIL: non-admin listed users';
    exception when insufficient_privilege then null;
    end;
end $$;

:as_owner
\echo '  all clinic RLS checks passed'
