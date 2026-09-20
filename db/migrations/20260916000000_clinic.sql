-- Care loop (clinician -> patient home exercise): clinician role, consent links, assigned plans,
-- and the append-only check-in record that feeds adherence, the form-score trend and red flags.
-- Plan: docs/superpowers/plans/2026-09-15-innoserve-rehab-20day-plan.md (WP0).
--
-- Design notes:
--   * The backend still talks to Postgres with the USER'S OWN JWT (no service_role), exactly as
--     the admin panel does. Every cross-user read a clinician gets is granted HERE, by RLS, and only
--     while an ACTIVE care_links row joins the clinician to the patient.
--   * A policy on a table must not select from that same table (it recurses). The link check is
--     therefore the SECURITY DEFINER function is_linked(), which reads care_links bypassing RLS.
--   * care_links has NO update/delete policy. Every state change goes through a SECURITY DEFINER
--     function that checks the caller itself:
--       - accept_care_invite(code): the patient is not on a pending row yet (patient_id is null),
--         so no RLS policy could let them find or update it without also letting anyone enumerate
--         invite codes.
--       - revoke_care_link(id): an UPDATE policy would let a clinician rewrite patient_id to any
--         user and read that user's data. The function only ever sets status = 'revoked'.
--   * session_checkins is append-only for the patient (insert + select, no update/delete). The one
--     mutation, a clinician acknowledging a red flag, is ack_checkin_flag(id), again so a clinician
--     cannot rewrite a patient's pain score through a broad UPDATE policy.
--   * Clinicians get NO policy on videos or conversations: the dashboard can read an analysis
--     result, not play the patient's clip or read their coaching chat.
--
-- DEPLOY ORDER: apply this together with the backend that ships it. The user_roles primary key
-- changes from (user_id) to (user_id, role); the previous backend's admin-role upsert uses
-- on_conflict=user_id and fails against the new key, so the admin role toggle breaks until the
-- new backend is live. services/plans.py also starts selecting training_plans.assigned_by, so the
-- NEW backend against an UNMIGRATED database fails every /api/plans call.

-- ---------------------------------------------------------------------------
-- 1. user_roles: one row per (user, role), so an admin can also be a clinician.
--    The admin migration keyed the table on user_id alone, which allowed exactly one role.
-- ---------------------------------------------------------------------------
alter table public.user_roles drop constraint if exists user_roles_pkey;
alter table public.user_roles add constraint user_roles_pkey primary key (user_id, role);

alter table public.user_roles drop constraint if exists user_roles_role_check;
alter table public.user_roles
    add constraint user_roles_role_check check (role in ('admin', 'clinician'));

-- is_clinician(uid): same shape and reasoning as is_admin() (admin_roles migration).
create or replace function public.is_clinician(uid uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
    select exists(
        select 1 from public.user_roles where user_id = uid and role = 'clinician'
    );
$$;

revoke all on function public.is_clinician(uuid) from public;
revoke all on function public.is_clinician(uuid) from anon;
grant execute on function public.is_clinician(uuid) to authenticated;

-- ---------------------------------------------------------------------------
-- 2. care_links: a clinician's invite, and once a patient accepts it, the consent link itself.
-- ---------------------------------------------------------------------------
create table if not exists public.care_links (
    id           uuid primary key default gen_random_uuid(),
    clinician_id uuid not null references auth.users (id) on delete cascade,
    -- NULL while the invite is pending; set by accept_care_invite() to the accepting user.
    patient_id   uuid references auth.users (id) on delete cascade,
    -- Short code the clinician hands over in person. Unique across all time (used and revoked codes
    -- included), so a code can never silently start pointing at a different clinician.
    invite_code  text not null unique check (invite_code ~ '^[A-Z0-9]{6,12}$'),
    status       text not null default 'pending'
                 check (status in ('pending', 'active', 'revoked')),
    created_at   timestamptz not null default now(),
    -- A pending code stops working after this. Accepted links do not expire.
    expires_at   timestamptz not null default (now() + interval '7 days'),
    accepted_at  timestamptz,
    revoked_at   timestamptz,
    constraint care_links_not_self check (patient_id is null or patient_id <> clinician_id),
    constraint care_links_active_has_patient
        check (status <> 'active' or (patient_id is not null and accepted_at is not null))
);

create index if not exists care_links_clinician_idx on public.care_links (clinician_id, status);
create index if not exists care_links_patient_idx   on public.care_links (patient_id, status);
-- At most one active link per clinician/patient pair.
create unique index if not exists care_links_active_pair_idx
    on public.care_links (clinician_id, patient_id) where status = 'active';

alter table public.care_links enable row level security;

drop policy if exists "care_links_select_party" on public.care_links;
create policy "care_links_select_party" on public.care_links
    for select
    to authenticated
    using (clinician_id = auth.uid() or patient_id = auth.uid());

-- Only a clinician creates invites, and only as a fresh pending row for themselves.
drop policy if exists "care_links_insert_clinician" on public.care_links;
create policy "care_links_insert_clinician" on public.care_links
    for insert
    to authenticated
    with check (
        clinician_id = auth.uid()
        and public.is_clinician(auth.uid())
        and status = 'pending'
        and patient_id is null
        and accepted_at is null
        and revoked_at is null
        and expires_at <= now() + interval '30 days'
    );

-- is_linked(clinician, patient): true while an ACTIVE link joins the two AND the clinician still
-- holds the role (revoking the role ends access without touching the links).
create or replace function public.is_linked(p_clinician uuid, p_patient uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
    select p_clinician is not null
       and p_patient is not null
       and exists (
           select 1 from public.care_links l
           where l.clinician_id = p_clinician
             and l.patient_id = p_patient
             and l.status = 'active'
       )
       and public.is_clinician(p_clinician);
$$;

revoke all on function public.is_linked(uuid, uuid) from public;
revoke all on function public.is_linked(uuid, uuid) from anon;
grant execute on function public.is_linked(uuid, uuid) to authenticated;

-- accept_care_invite(code): the patient consents. Returns {"link": {...}} or {"error": reason},
-- reason one of invalid (unknown, used, revoked or expired code), self (own invite). A result
-- object rather than RAISE so the API maps reasons to statuses without depending on PostgREST's
-- SQLSTATE-to-HTTP table.
create or replace function public.accept_care_invite(p_code text)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
    v_uid  uuid := auth.uid();
    v_link public.care_links;
begin
    if v_uid is null then
        raise exception 'not authenticated' using errcode = '42501';
    end if;

    select * into v_link
    from public.care_links
    where invite_code = upper(btrim(coalesce(p_code, '')))
    for update;

    if not found or v_link.status <> 'pending' or v_link.expires_at <= now() then
        return jsonb_build_object('error', 'invalid');
    end if;
    if v_link.clinician_id = v_uid then
        return jsonb_build_object('error', 'self');
    end if;

    -- Already actively linked to this clinician: consume the code and return the existing link,
    -- instead of tripping the one-active-link-per-pair index with a raw unique violation.
    if exists (
        select 1 from public.care_links
        where clinician_id = v_link.clinician_id and patient_id = v_uid and status = 'active'
    ) then
        update public.care_links
        set status = 'revoked', revoked_at = now()
        where id = v_link.id;

        select * into v_link
        from public.care_links
        where clinician_id = v_link.clinician_id and patient_id = v_uid and status = 'active';
        return jsonb_build_object('link', to_jsonb(v_link));
    end if;

    update public.care_links
    set patient_id = v_uid, status = 'active', accepted_at = now()
    where id = v_link.id
    returning * into v_link;

    return jsonb_build_object('link', to_jsonb(v_link));
end;
$$;

revoke all on function public.accept_care_invite(text) from public;
revoke all on function public.accept_care_invite(text) from anon;
grant execute on function public.accept_care_invite(text) to authenticated;

-- revoke_care_link(id): either party ends a link (or a clinician withdraws a pending invite).
-- Returns {"link": {...}} or {"error": "not_found"} (no such link, not a party, already revoked).
create or replace function public.revoke_care_link(p_link uuid)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
    v_uid  uuid := auth.uid();
    v_link public.care_links;
begin
    if v_uid is null then
        raise exception 'not authenticated' using errcode = '42501';
    end if;

    update public.care_links
    set status = 'revoked', revoked_at = now()
    where id = p_link
      and status in ('pending', 'active')
      and (clinician_id = v_uid or patient_id = v_uid)
    returning * into v_link;

    if not found then
        return jsonb_build_object('error', 'not_found');
    end if;
    return jsonb_build_object('link', to_jsonb(v_link));
end;
$$;

revoke all on function public.revoke_care_link(uuid) from public;
revoke all on function public.revoke_care_link(uuid) from anon;
grant execute on function public.revoke_care_link(uuid) to authenticated;

-- Display info for the other party of a link. auth.users is not readable with a user JWT, so both
-- directions go through a definer function that returns only the caller's own active links.
-- display_name is the Google/LINE profile name when there is one; LINE-login accounts have a
-- synthetic line_<sub>@line.invalid email, which the API hides.
create or replace function public.clinic_patients()
returns table (
    link_id      uuid,
    patient_id   uuid,
    email        text,
    display_name text,
    accepted_at  timestamptz
)
language plpgsql
stable
security definer
set search_path = public
as $$
begin
    if not public.is_clinician(auth.uid()) then
        raise exception 'not authorized' using errcode = '42501';
    end if;

    return query
    select
        l.id,
        l.patient_id,
        u.email::text,
        coalesce(u.raw_user_meta_data ->> 'full_name', u.raw_user_meta_data ->> 'name')::text,
        l.accepted_at
    from public.care_links l
    join auth.users u on u.id = l.patient_id
    where l.clinician_id = auth.uid()
      and l.status = 'active'
    order by l.accepted_at desc;
end;
$$;

revoke all on function public.clinic_patients() from public;
revoke all on function public.clinic_patients() from anon;
grant execute on function public.clinic_patients() to authenticated;

create or replace function public.my_clinicians()
returns table (
    link_id      uuid,
    clinician_id uuid,
    email        text,
    display_name text,
    accepted_at  timestamptz
)
language sql
stable
security definer
set search_path = public
as $$
    select
        l.id,
        l.clinician_id,
        u.email::text,
        coalesce(u.raw_user_meta_data ->> 'full_name', u.raw_user_meta_data ->> 'name')::text,
        l.accepted_at
    from public.care_links l
    join auth.users u on u.id = l.clinician_id
    where l.patient_id = auth.uid()
      and l.status = 'active'
    order by l.accepted_at desc;
$$;

revoke all on function public.my_clinicians() from public;
revoke all on function public.my_clinicians() from anon;
grant execute on function public.my_clinicians() to authenticated;

-- ---------------------------------------------------------------------------
-- 3. training_plans.assigned_by: the clinician who assigned this plan, NULL for a self-made plan.
--    The OWNER (user_id) stays the patient, so every owner-only path keeps working unchanged.
-- ---------------------------------------------------------------------------
alter table public.training_plans
    add column if not exists assigned_by uuid references auth.users (id) on delete set null;

-- Clinician access to a linked patient's plans: read all of them; create and edit only the ones
-- this clinician assigned. Permissive policies OR with training_plans_owner_all.
drop policy if exists "training_plans_clinician_select" on public.training_plans;
create policy "training_plans_clinician_select" on public.training_plans
    for select
    to authenticated
    using (public.is_linked(auth.uid(), user_id));

drop policy if exists "training_plans_clinician_insert" on public.training_plans;
create policy "training_plans_clinician_insert" on public.training_plans
    for insert
    to authenticated
    with check (assigned_by = auth.uid() and public.is_linked(auth.uid(), user_id));

drop policy if exists "training_plans_clinician_update" on public.training_plans;
create policy "training_plans_clinician_update" on public.training_plans
    for update
    to authenticated
    using (assigned_by = auth.uid() and public.is_linked(auth.uid(), user_id))
    with check (assigned_by = auth.uid() and public.is_linked(auth.uid(), user_id));

-- Items of a linked patient's plans: read all; create/edit only inside a plan this clinician
-- assigned. The subquery reads training_plans (not plan_items), so it does not recurse; it runs
-- under the clinician's own training_plans policies, which the select policy above satisfies.
drop policy if exists "plan_items_clinician_select" on public.plan_items;
create policy "plan_items_clinician_select" on public.plan_items
    for select
    to authenticated
    using (public.is_linked(auth.uid(), user_id));

drop policy if exists "plan_items_clinician_insert" on public.plan_items;
create policy "plan_items_clinician_insert" on public.plan_items
    for insert
    to authenticated
    with check (
        public.is_linked(auth.uid(), plan_items.user_id)
        and exists (
            select 1 from public.training_plans p
            where p.id = plan_items.plan_id
              and p.user_id = plan_items.user_id
              and p.assigned_by = auth.uid()
        )
    );

drop policy if exists "plan_items_clinician_update" on public.plan_items;
create policy "plan_items_clinician_update" on public.plan_items
    for update
    to authenticated
    using (
        public.is_linked(auth.uid(), plan_items.user_id)
        and exists (
            select 1 from public.training_plans p
            where p.id = plan_items.plan_id
              and p.user_id = plan_items.user_id
              and p.assigned_by = auth.uid()
        )
    )
    with check (
        public.is_linked(auth.uid(), plan_items.user_id)
        and exists (
            select 1 from public.training_plans p
            where p.id = plan_items.plan_id
              and p.user_id = plan_items.user_id
              and p.assigned_by = auth.uid()
        )
    );

-- A linked patient's analysis results, read-only. NOTE for the backend: any analyses read that
-- relied on RLS alone to mean "mine" (store.list_analyses) must now filter user_id explicitly, or
-- a clinician's own history would list their patients' analyses too.
drop policy if exists "analyses_clinician_select" on public.analyses;
create policy "analyses_clinician_select" on public.analyses
    for select
    to authenticated
    using (public.is_linked(auth.uid(), user_id));

-- ---------------------------------------------------------------------------
-- 4. session_checkins: one row per "how did today's session go" report. Append-only.
--    Kept apart from plan_items on purpose: POST /api/plans/{id}/start wipes every item's tick and
--    analysis link, so adherence history could never be rebuilt from plan_items.
-- ---------------------------------------------------------------------------
create table if not exists public.session_checkins (
    id              uuid primary key default gen_random_uuid(),
    user_id         uuid not null references auth.users (id) on delete cascade,
    -- set null, not cascade: a check-in outlives the plan/item/analysis it was about.
    plan_id         uuid references public.training_plans (id) on delete set null,
    plan_item_id    uuid references public.plan_items (id) on delete set null,
    analysis_id     uuid references public.analyses (id) on delete set null,
    pain_nrs        smallint not null check (pain_nrs between 0 and 10),
    rpe             smallint check (rpe is null or rpe between 0 and 10),
    note            text check (note is null or char_length(note) <= 500),
    -- Server-side port of frontend/src/lib/formScore.ts, computed from the linked analysis at
    -- check-in time. NULL when there is no analysis or the clip was never measurable.
    form_score      smallint check (form_score is null or form_score between 0 and 100),
    flagged         boolean not null default false,
    flag_reasons    jsonb not null default '[]'::jsonb
                    check (jsonb_typeof(flag_reasons) = 'array'),
    acknowledged_at timestamptz,
    acknowledged_by uuid references auth.users (id) on delete set null,
    created_at      timestamptz not null default now()
);

create index if not exists session_checkins_user_created_idx
    on public.session_checkins (user_id, created_at desc);
create index if not exists session_checkins_plan_created_idx
    on public.session_checkins (plan_id, created_at desc);

alter table public.session_checkins enable row level security;

drop policy if exists "session_checkins_owner_select" on public.session_checkins;
create policy "session_checkins_owner_select" on public.session_checkins
    for select
    to authenticated
    using (user_id = auth.uid());

drop policy if exists "session_checkins_owner_insert" on public.session_checkins;
create policy "session_checkins_owner_insert" on public.session_checkins
    for insert
    to authenticated
    with check (user_id = auth.uid() and acknowledged_at is null and acknowledged_by is null);

drop policy if exists "session_checkins_clinician_select" on public.session_checkins;
create policy "session_checkins_clinician_select" on public.session_checkins
    for select
    to authenticated
    using (public.is_linked(auth.uid(), user_id));

-- ack_checkin_flag(id): a linked clinician marks a flagged check-in as handled. Idempotent: an
-- already-acknowledged row comes back unchanged. Returns {"checkin": {...}} or
-- {"error": "not_found" | "not_flagged"}.
create or replace function public.ack_checkin_flag(p_checkin uuid)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
    v_uid uuid := auth.uid();
    v_row public.session_checkins;
begin
    if v_uid is null then
        raise exception 'not authenticated' using errcode = '42501';
    end if;

    select * into v_row from public.session_checkins where id = p_checkin for update;
    if not found or not public.is_linked(v_uid, v_row.user_id) then
        return jsonb_build_object('error', 'not_found');
    end if;
    if not v_row.flagged then
        return jsonb_build_object('error', 'not_flagged');
    end if;

    if v_row.acknowledged_at is null then
        update public.session_checkins
        set acknowledged_at = now(), acknowledged_by = v_uid
        where id = p_checkin
        returning * into v_row;
    end if;
    return jsonb_build_object('checkin', to_jsonb(v_row));
end;
$$;

revoke all on function public.ack_checkin_flag(uuid) from public;
revoke all on function public.ack_checkin_flag(uuid) from anon;
grant execute on function public.ack_checkin_flag(uuid) to authenticated;

-- ---------------------------------------------------------------------------
-- 5. admin_list_users(): add is_clinician for the /admin/users role toggle. The return type
--    changes, which CREATE OR REPLACE cannot do, so drop and recreate (same body otherwise as
--    20260713000200_admin_user_overview.sql, including its auth.users grant note).
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
    is_clinician        boolean
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
        ) as is_clinician
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
