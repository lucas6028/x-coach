-- x-coach admin panel P1: a role table + an is_admin() helper so admin identity is visible to
-- Postgres itself. This lets RLS enforce "only an admin may write admin data" and lets future
-- SECURITY DEFINER functions do cross-user reads — all WITHOUT the backend ever holding a
-- service_role key (the backend still talks to Postgres with the user's own JWT).
--
-- RLS strategy (mirrors 20260620000000_init_videos_analyses.sql):
--   * The backend uses the *user's* JWT (anon key + postgrest.auth(token)), so these policies
--     are enforced automatically. It never uses service_role, so a bug here cannot leak roles.
--   * A policy on user_roles must NOT itself `select ... from user_roles`, or it recurses. We
--     route the admin check through the SECURITY DEFINER function is_admin(), which bypasses RLS
--     inside its own body, to break that recursion.

-- ---------------------------------------------------------------------------
-- user_roles: one row per privileged user (role defaults to 'admin')
-- ---------------------------------------------------------------------------
create table if not exists public.user_roles (
    user_id    uuid primary key references auth.users (id) on delete cascade,
    role       text not null default 'admin',
    created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- is_admin(uid): true when the given user has the 'admin' role.
--
-- SECURITY DEFINER (runs as the function owner) so it reads user_roles regardless of the caller's
-- RLS — this is what lets a policy ON user_roles ask "is the caller an admin?" without recursing
-- back through user_roles' own policies. `set search_path = public` pins name resolution so the
-- definer body can't be hijacked by a caller-controlled search_path.
-- ---------------------------------------------------------------------------
create or replace function public.is_admin(uid uuid)
returns boolean
language sql
security definer
set search_path = public
as $$
    select exists(
        select 1 from public.user_roles where user_id = uid and role = 'admin'
    );
$$;

-- Only authenticated users may call it: the frontend/backend ask "am I admin?" for a signed-in user,
-- and the policies below call it for auth.uid(). anon is deliberately NOT granted — an unauthenticated
-- caller has no auth.uid() and must not be able to probe arbitrary uuids for admin status. (The
-- app_settings anon SELECT is a table policy, unaffected; the RLS write policies that call
-- is_admin(auth.uid()) only run for authenticated users.)
grant execute on function public.is_admin(uuid) to authenticated;

-- Count the current admins. SECURITY DEFINER so it bypasses the self-only SELECT policy below:
-- an admin needs the TOTAL admin count for the "don't remove the last admin" guard, but RLS would
-- otherwise let the caller see only their own row. Returns just an integer (not sensitive), and is
-- only consulted on the admin-gated role PUT.
create or replace function public.count_admins()
returns integer
language sql
security definer
set search_path = public
as $$
    select count(*)::int from public.user_roles where role = 'admin';
$$;
grant execute on function public.count_admins() to authenticated;

-- ---------------------------------------------------------------------------
-- Row-Level Security on user_roles.
--   * An authenticated user may SELECT only their OWN role row (so they can learn whether they
--     themselves are admin) — NOT every row, so a user cannot enumerate the full admin list.
--     Admin-facing cross-user reads go through the admin_list_users() SECURITY DEFINER function,
--     which bypasses RLS; count_admins runs on the admin-gated PUT path.
--   * Only an admin may INSERT/UPDATE/DELETE (in-app role assignment), gated by is_admin().
-- ---------------------------------------------------------------------------
alter table public.user_roles enable row level security;

drop policy if exists "user_roles_select_authenticated" on public.user_roles;
create policy "user_roles_select_authenticated" on public.user_roles
    for select
    to authenticated
    using (user_id = auth.uid());

drop policy if exists "user_roles_insert_admin" on public.user_roles;
create policy "user_roles_insert_admin" on public.user_roles
    for insert
    to authenticated
    with check (public.is_admin(auth.uid()));

drop policy if exists "user_roles_update_admin" on public.user_roles;
create policy "user_roles_update_admin" on public.user_roles
    for update
    to authenticated
    using (public.is_admin(auth.uid()))
    with check (public.is_admin(auth.uid()));

drop policy if exists "user_roles_delete_admin" on public.user_roles;
create policy "user_roles_delete_admin" on public.user_roles
    for delete
    to authenticated
    using (public.is_admin(auth.uid()));

-- ---------------------------------------------------------------------------
-- Seed the first admin by email. This only lands a row if that user has ALREADY signed up
-- (auth.users has the row). On a fresh project where the email hasn't logged in yet, this inserts
-- nothing and is a silent no-op.
-- ---------------------------------------------------------------------------
insert into public.user_roles (user_id, role)
select id, 'admin' from auth.users where email = 'lucas60303@gmail.com'
on conflict (user_id) do nothing;

-- NOTE: if 'lucas60303@gmail.com' had not signed up when this migration ran, no admin was seeded.
-- After that user logs in once (so their auth.users row exists), re-run exactly this to grant it:
--
--   insert into public.user_roles (user_id, role)
--   select id, 'admin' from auth.users where email = 'lucas60303@gmail.com'
--   on conflict (user_id) do nothing;
--
-- Thereafter that admin can assign other admins in-app (the insert/update policies above allow it).
