-- LOCAL DEV ONLY. NEVER apply this to the Supabase project.
--
-- A plain postgres:16 container has none of what every file in db/migrations/ assumes Supabase
-- provides: the anon/authenticated/service_role roles, the auth schema, auth.users and auth.uid().
-- This file fakes exactly those, the way Supabase sets them up, so db/checks/run_local.sh can
-- apply the real migrations and then exercise their RLS policies as each role.
--
-- It lives in db/checks/, not db/migrations/, because the migrations README says to apply every
-- file in db/migrations/ in filename order; a shim there would plant fake auth objects in prod.

create role anon nologin noinherit;
create role authenticated nologin noinherit;
create role service_role nologin noinherit bypassrls;

create schema if not exists auth;
grant usage on schema auth to anon, authenticated, service_role;

create table auth.users (
    id                 uuid primary key default gen_random_uuid(),
    email              varchar(255),
    raw_user_meta_data jsonb default '{}'::jsonb,
    created_at         timestamptz default now(),
    last_sign_in_at    timestamptz
);

-- Supabase reads the caller from the JWT claims PostgREST puts in request.jwt.claims.
-- The check script sets that GUC directly: `set request.jwt.claims = '{"sub":"<uuid>"}'`.
create or replace function auth.uid()
returns uuid
language sql
stable
as $$
    select nullif(
        coalesce(
            current_setting('request.jwt.claim.sub', true),
            nullif(current_setting('request.jwt.claims', true), '')::jsonb ->> 'sub'
        ),
        ''
    )::uuid;
$$;

create or replace function auth.role()
returns text
language sql
stable
as $$
    select coalesce(
        current_setting('request.jwt.claim.role', true),
        nullif(current_setting('request.jwt.claims', true), '')::jsonb ->> 'role'
    )::text;
$$;

-- Supabase grants every API role full table/function privileges in public and lets RLS do the
-- gating. Reproduce that, so a missing policy (not a missing grant) is what the checks catch.
grant usage on schema public to anon, authenticated, service_role;
alter default privileges in schema public grant all on tables    to anon, authenticated, service_role;
alter default privileges in schema public grant all on sequences to anon, authenticated, service_role;
alter default privileges in schema public grant all on functions to anon, authenticated, service_role;
