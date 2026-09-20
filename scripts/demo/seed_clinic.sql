-- DEMO DATA ONLY — the care-loop scenario for the InnoServe screenshots and 3-minute video.
--
-- Run in the Supabase SQL editor (it runs as postgres), AFTER both migrations
-- (20260916000000_clinic.sql, 20260927000000_daily_jobs.sql) and AFTER the three demo accounts
-- below have each signed in to the app once (their auth.users rows must exist). Edit the three
-- emails first; keep them free of any school or personal name (the submission is anonymous).
--
-- Why SQL and not the API: check-ins are stamped by the server (created_at default now()), so two
-- or three weeks of history cannot be created through POST /api/checkins. The rows below carry
-- backdated timestamps and the flags the red-flag rules would have produced at each point.
--
-- Re-runnable: it first deletes ONLY rows it created itself — the two demo patients' plans that
-- this clinician assigned with exactly the c_marker note below, the check-ins on those plans, and
-- the links with the DEMO invite codes. Nothing else is touched, so pointing it at a real account
-- cannot erase that account's own data. The marker is a plan note that reads naturally on screen,
-- because everything seeded here appears in the video.
--
-- The scenario:
--   Patient A — knee rehab for 20 days, ten check-ins, pain easing 3 -> 1, form score 72 -> 88;
--               6 of 9 plan items ticked (this cycle's days 1 and 3).
--   Patient B — shoulder rehab for 16 days, eight check-ins. Day 7: form score falls >= 20 against
--               the previous three (rule c), acknowledged by the therapist. Today: pain 7, up 3 from
--               the last report (rules a + b) -> an OPEN red flag on /clinic; 4 of 9 items ticked.

do $$
declare
    c_clinician text := 'demo-therapist@example.com';
    c_patient_a text := 'demo-patient-a@example.com';
    c_patient_b text := 'demo-patient-b@example.com';
    c_marker    text := '每週三次，每次約 15 分鐘；疼痛超過 5 分請先暫停並回報。';

    v_clin   uuid;
    v_a      uuid;
    v_b      uuid;
    v_plan_a uuid;
    v_plan_b uuid;
    i        int;
    -- Patient A: ten check-ins, every other day from 19 days ago to yesterday.
    a_pain   int[] := array[3, 3, 2, 2, 2, 1, 2, 1, 1, 1];
    a_form   int[] := array[72, 75, 78, 80, 79, 83, 85, 84, 87, 88];
    -- Patient B: eight check-ins, every other day from 14 days ago; the last one is today.
    b_pain   int[] := array[2, 2, 3, 2, 3, 3, 4, 7];
    b_form   int[] := array[90, 92, 88, 90, 68, 66, 64, 62];
begin
    select id into v_clin from auth.users where email = c_clinician;
    select id into v_a    from auth.users where email = c_patient_a;
    select id into v_b    from auth.users where email = c_patient_b;
    if v_clin is null or v_a is null or v_b is null then
        raise exception 'Sign in once as %, % and % before seeding.', c_clinician, c_patient_a, c_patient_b;
    end if;

    -- Clean up a previous run (only rows this script created). Check-ins first: deleting a plan
    -- sets their plan_id to null, after which they could no longer be told apart.
    delete from public.session_checkins
    where plan_id in (
        select id from public.training_plans
        where user_id in (v_a, v_b) and assigned_by = v_clin and notes = c_marker
    );
    delete from public.training_plans
    where user_id in (v_a, v_b) and assigned_by = v_clin and notes = c_marker;
    delete from public.care_links
    where invite_code in ('DEMOA1', 'DEMOB1');

    insert into public.user_roles (user_id, role) values (v_clin, 'clinician')
    on conflict do nothing;

    -- Consent links, unless the demo accounts were already linked through the real invite flow.
    insert into public.care_links (clinician_id, patient_id, invite_code, status, created_at, expires_at, accepted_at)
    select v_clin, v_a, 'DEMOA1', 'active', now() - interval '21 days', now() - interval '14 days', now() - interval '21 days'
    where not exists (
        select 1 from public.care_links
        where clinician_id = v_clin and patient_id = v_a and status = 'active'
    );
    insert into public.care_links (clinician_id, patient_id, invite_code, status, created_at, expires_at, accepted_at)
    select v_clin, v_b, 'DEMOB1', 'active', now() - interval '17 days', now() - interval '10 days', now() - interval '17 days'
    where not exists (
        select 1 from public.care_links
        where clinician_id = v_clin and patient_id = v_b and status = 'active'
    );

    -- Assigned plans: copies of the knee_rehab and shoulder_rehab templates (routers/plans.py).
    insert into public.training_plans (user_id, name, notes, template_key, assigned_by, started_at, created_at, updated_at)
    values (v_a, '膝關節復健', c_marker, 'knee_rehab', v_clin, now() - interval '20 days', now() - interval '20 days', now() - interval '20 days')
    returning id into v_plan_a;
    insert into public.plan_items (plan_id, user_id, day_index, position, movement, sets, reps) values
        (v_plan_a, v_a, 1, 0, 'Shoulder Bridge', 2, 10), (v_plan_a, v_a, 1, 1, 'Leg Abduction', 2, 10), (v_plan_a, v_a, 1, 2, 'Squat', 2, 10),
        (v_plan_a, v_a, 3, 0, 'Shoulder Bridge', 2, 10), (v_plan_a, v_a, 3, 1, 'Leg Abduction', 2, 10), (v_plan_a, v_a, 3, 2, 'Lunge', 2, 10),
        (v_plan_a, v_a, 5, 0, 'Squat', 2, 10), (v_plan_a, v_a, 5, 1, 'Lunge', 2, 10), (v_plan_a, v_a, 5, 2, 'Shoulder Bridge', 2, 10);
    -- Ticks show the CURRENT cycle only (a restart wipes them), so they must NOT add up to the
    -- whole 20 days of check-ins: patient A is through days 1 and 3 of this week -> 6/9 on screen.
    update public.plan_items set completed_at = now() - interval '3 days'
    where plan_id = v_plan_a and day_index = 1;
    update public.plan_items set completed_at = now() - interval '1 day'
    where plan_id = v_plan_a and day_index = 3;

    insert into public.training_plans (user_id, name, notes, template_key, assigned_by, started_at, created_at, updated_at)
    values (v_b, '肩關節復健', c_marker, 'shoulder_rehab', v_clin, now() - interval '16 days', now() - interval '16 days', now() - interval '16 days')
    returning id into v_plan_b;
    insert into public.plan_items (plan_id, user_id, day_index, position, movement, sets, reps) values
        (v_plan_b, v_b, 1, 0, 'Arm Abduction', 2, 10), (v_plan_b, v_b, 1, 1, 'Band Pull Apart', 2, 10), (v_plan_b, v_b, 1, 2, 'Row', 2, 10),
        (v_plan_b, v_b, 3, 0, 'Arm VW', 2, 10), (v_plan_b, v_b, 3, 1, 'Band Pull Apart', 2, 10), (v_plan_b, v_b, 3, 2, 'Arm Abduction', 2, 10),
        (v_plan_b, v_b, 5, 0, 'Row', 2, 10), (v_plan_b, v_b, 5, 1, 'Arm VW', 2, 10), (v_plan_b, v_b, 5, 2, 'Band Pull Apart', 2, 10);
    -- Patient B is mid-session today (the check-in below is 3 hours old) -> 4/9 on screen.
    update public.plan_items set completed_at = now() - interval '2 days'
    where plan_id = v_plan_b and day_index = 1;
    update public.plan_items set completed_at = now() - interval '3 hours'
    where plan_id = v_plan_b and day_index = 3 and position = 0;

    -- Patient A: steady improvement, never flagged.
    for i in 1..10 loop
        insert into public.session_checkins (user_id, plan_id, pain_nrs, rpe, note, form_score, flagged, flag_reasons, created_at)
        values (v_a, v_plan_a, a_pain[i], 4, null, a_form[i], false, '[]',
                now() - make_interval(days => 21 - 2 * i, hours => 3));
    end loop;

    -- Patient B: flags exactly where services/redflags.evaluate would raise them.
    for i in 1..8 loop
        insert into public.session_checkins (user_id, plan_id, pain_nrs, rpe, note, form_score, flagged, flag_reasons,
                                             acknowledged_at, acknowledged_by, created_at)
        values (
            v_b, v_plan_b, b_pain[i], case when i = 8 then 7 else 5 end,
            case when i = 8 then '今天肩膀前側比較痛' end,
            b_form[i],
            i in (7, 8),
            case i
                when 7 then '["form_drop"]'::jsonb          -- mean(68,66,64)=66 vs mean(92,88,90)=90
                when 8 then '["pain_high", "pain_rise"]'::jsonb -- 7 >= 6, and 7 - 4 >= 3
                else '[]'::jsonb
            end,
            case when i = 7 then now() - interval '1 day' end,
            case when i = 7 then v_clin end,
            case when i = 8 then now() - interval '3 hours'
                 else now() - make_interval(days => 16 - 2 * i, hours => 5) end
        );
    end loop;

    raise notice 'Seeded: clinician %, patients % and %.', c_clinician, c_patient_a, c_patient_b;
end $$;
