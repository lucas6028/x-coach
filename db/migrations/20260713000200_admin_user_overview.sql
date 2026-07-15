-- x-coach admin panel P3: a read-only cross-user overview for the admin dashboard.
--
-- The backend never holds a service_role key, so it cannot read auth.users (or another user's rows)
-- with the user's own JWT — RLS scopes every table read to auth.uid(). To surface a users list to an
-- admin WITHOUT a service_role key we expose exactly one SECURITY DEFINER function that:
--   * runs as its owner (the migration runner / postgres), so it CAN read auth.users and aggregate
--     per-user counts across public.analyses / public.conversations;
--   * gates itself INTERNALLY on public.is_admin(auth.uid()) as its first statement, raising 42501
--     for a non-admin caller — the definer privilege is useless to a non-admin.
-- This is the single, tightly-scoped exception to "backend reads only as the user".

-- ---------------------------------------------------------------------------
-- admin_list_users(): one row per user with lightweight activity counts.
--
-- SECURITY DEFINER + `set search_path = public` (pins name resolution so the definer body can't be
-- hijacked by a caller-controlled search_path). The internal is_admin() guard is the access control;
-- `grant execute ... to authenticated` only lets a signed-in user *attempt* the call.
-- ---------------------------------------------------------------------------
create or replace function public.admin_list_users()
returns table (
    id                  uuid,
    email               text,
    created_at          timestamptz,
    last_sign_in_at     timestamptz,
    analyses_count      bigint,
    conversations_count bigint,
    is_admin            boolean
)
language plpgsql
security definer
set search_path = public
as $$
begin
    -- Admin-gate INSIDE the function: the definer privilege must never be usable by a non-admin.
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
        ) as is_admin
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

-- authenticated may CALL it; the internal is_admin() guard is the real gate (a non-admin gets 42501).
grant execute on function public.admin_list_users() to authenticated;

-- ---------------------------------------------------------------------------
-- ⚠️ VERIFY IN THE SUPABASE PROJECT: this is the one place that reads auth.users, and it does so via
-- SECURITY DEFINER (owner = the migration role). Whether the definer can select from auth.users
-- depends on the project's grants. In a standard Supabase project the migration runs as `postgres`,
-- which already owns / can read auth.users, so no extra grant is needed. If on your project the
-- function raises "permission denied for table users", grant the definer's role read access once:
--
--   grant usage on schema auth to postgres;
--   grant select on auth.users to postgres;
--
-- (Substitute the actual function-owner role if it is not `postgres`.) Keep this grant as narrow as
-- possible — only the definer role, only SELECT, only auth.users.
-- ---------------------------------------------------------------------------
