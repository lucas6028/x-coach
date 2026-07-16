-- x-coach admin panel P2: a runtime settings override table so an admin can tune the LLM / RAG /
-- analysis-pipeline knobs without a redeploy. The backend resolution layer
-- (backend/app/services/runtime_config.py) reads this table and layers its values over the
-- env-driven defaults, with a short in-process TTL cache.
--
-- SECURITY — this table holds NON-SECRET operational knobs ONLY. NEVER store the LLM API key,
-- Supabase credentials, or any other secret here: rows are readable by anon (see the SELECT
-- policy below), so anything here is effectively public. Secrets stay pure-env (see settings.py).
--
-- RLS strategy (mirrors 20260620000000_init_videos_analyses.sql + 20260713000000_admin_roles.sql):
--   * SELECT is granted to BOTH anon and authenticated. The resolution layer reads overrides with
--     an ANON client on paths that may be unauthenticated (e.g. the public /api/health model
--     picker), so anon read is required — and safe, because only non-secret knobs live here.
--   * INSERT/UPDATE/DELETE are gated on public.is_admin(auth.uid()) — the SECURITY DEFINER helper
--     from the admin_roles migration — so only an admin may write, enforced by Postgres itself.
--     The backend still talks to Postgres with the user's own JWT (no service_role key).

-- ---------------------------------------------------------------------------
-- app_settings: a key/value override table (value is JSONB so a knob can be a
-- scalar, a list, or an object without a schema change).
-- ---------------------------------------------------------------------------
create table if not exists public.app_settings (
    key        text primary key,
    value      jsonb not null,
    updated_at timestamptz not null default now()
);

-- Reuse the touch_updated_at() trigger function defined in the init migration so updated_at stays
-- fresh on every write (the resolution layer can surface "last changed" in the admin UI later).
drop trigger if exists app_settings_touch_updated_at on public.app_settings;
create trigger app_settings_touch_updated_at
    before update on public.app_settings
    for each row execute function public.touch_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security.
--   * anon + authenticated may SELECT (the resolution layer reads on public paths).
--   * only an admin may INSERT/UPDATE/DELETE, gated by is_admin(auth.uid()).
-- ---------------------------------------------------------------------------
alter table public.app_settings enable row level security;

drop policy if exists "app_settings_select_all" on public.app_settings;
create policy "app_settings_select_all" on public.app_settings
    for select
    to anon, authenticated
    using (true);

drop policy if exists "app_settings_insert_admin" on public.app_settings;
create policy "app_settings_insert_admin" on public.app_settings
    for insert
    to authenticated
    with check (public.is_admin(auth.uid()));

drop policy if exists "app_settings_update_admin" on public.app_settings;
create policy "app_settings_update_admin" on public.app_settings
    for update
    to authenticated
    using (public.is_admin(auth.uid()))
    with check (public.is_admin(auth.uid()));

drop policy if exists "app_settings_delete_admin" on public.app_settings;
create policy "app_settings_delete_admin" on public.app_settings
    for delete
    to authenticated
    using (public.is_admin(auth.uid()));
