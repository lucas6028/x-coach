-- LINE Messaging API bot: one read-only, tightly-scoped entry point for the webhook.
--
-- The webhook is authenticated by LINE's request signature, not by a Supabase JWT, so it has
-- no user token and cannot go through the usual RLS-scoped path (services/store). Rather than
-- letting the backend read tables with service_role, we expose exactly ONE SECURITY DEFINER
-- function that takes a LINE user id and returns only that user's aggregate summary.
--
-- Resolution works because the Messaging API channel and the LINE Login channel live under the
-- same LINE provider: LINE issues a user id per provider, so the webhook's source.userId equals
-- the Login ID token's `sub`, which services/line_auth stored in user_metadata.line_sub.
--
-- Access control is the GRANT: service_role only. anon/authenticated cannot call it at all, so
-- no signed-in user can pass someone else's `sub` and read their data.

create or replace function public.line_training_summary(p_line_sub text)
returns jsonb
language plpgsql
security definer
-- Pin name resolution so the definer body can't be hijacked by a caller-controlled search_path.
set search_path = public, auth
as $$
declare
    v_user_id uuid;
    v_total   bigint;
    v_latest  jsonb;
    v_top     jsonb;
begin
    if p_line_sub is null or length(p_line_sub) = 0 then
        return null;
    end if;

    -- Key the lookup on the synthetic auth email, NOT on raw_user_meta_data->>'line_sub'.
    -- user_metadata is writable by the signed-in user themselves (supabase.auth.updateUser),
    -- so any x-coach account could set its own line_sub to someone else's LINE user id; with
    -- two rows matching, `limit 1` with no ORDER BY would pick arbitrarily and could hand the
    -- bot's reply to the wrong person. The synthetic email is derived deterministically by
    -- services/line_auth.synthetic_email (line_<sub, stripped+lowercased>@line.invalid) and
    -- Supabase's find-or-create already depends on its uniqueness, so it is the actual
    -- identity key for a LINE-login account — reproduce that derivation exactly here.
    select u.id into v_user_id
    from auth.users u
    where u.email = 'line_' || lower(trim(p_line_sub)) || '@line.invalid'
    limit 1;

    -- No x-coach account for this LINE user: the bot turns this into a "sign in first" reply.
    if v_user_id is null then
        return null;
    end if;

    select count(*) into v_total
    from public.analyses a
    where a.user_id = v_user_id;

    select to_jsonb(x) into v_latest
    from (
        select a.created_at, a.view_type, a.fault_count
        from public.analyses a
        where a.user_id = v_user_id
        order by a.created_at desc
        limit 1
    ) x;

    -- Top 3 faults by how often they were detected across every analysis. Grouped by the
    -- stable fault_id (the backend maps it to a localised label); the English fault_name rides
    -- along as a fallback for ids the backend doesn't know yet.
    --
    -- The inner query does the ranking (order by count desc, limit 3); jsonb_agg's own ORDER BY
    -- is required to preserve that ranking in the output array — an aggregate over a FROM
    -- subquery is not otherwise guaranteed to visit rows in the subquery's order, so without it
    -- the returned top_faults could come back in an arbitrary order.
    select coalesce(
        jsonb_agg(
            jsonb_build_object('id', t.id, 'name', t.name, 'count', t.cnt)
            order by t.cnt desc, t.id
        ),
        '[]'::jsonb
    ) into v_top
    from (
        select
            d ->> 'fault_id'      as id,
            min(d ->> 'fault_name') as name,
            count(*)              as cnt
        from public.analyses a
        cross join lateral jsonb_array_elements(
            coalesce(a.result -> 'detections', '[]'::jsonb)
        ) as d
        where a.user_id = v_user_id
        group by d ->> 'fault_id'
        order by count(*) desc, d ->> 'fault_id'
        limit 3
    ) t;

    return jsonb_build_object('total', v_total, 'latest', v_latest, 'top_faults', v_top);
end;
$$;

-- Least privilege: revoke the implicit PUBLIC execute grant (and the roles that inherit it),
-- then grant only to service_role, which only the backend holds.
revoke all on function public.line_training_summary(text) from public;
revoke all on function public.line_training_summary(text) from anon;
revoke all on function public.line_training_summary(text) from authenticated;
grant execute on function public.line_training_summary(text) to service_role;

-- ---------------------------------------------------------------------------
-- ⚠️ VERIFY IN THE SUPABASE PROJECT: like 20260713000200_admin_user_overview.sql, this function
-- reads auth.users via SECURITY DEFINER (owner = the migration role). Whether the definer can
-- select from auth.users depends on the project's grants. In a standard Supabase project the
-- migration runs as `postgres`, which already owns / can read auth.users, so no extra grant is
-- needed. If it raises "permission denied for table users", grant the definer's role read
-- access once (substitute the actual function-owner role if not `postgres`):
--
--   grant usage on schema auth to postgres;
--   grant select on auth.users to postgres;
-- ---------------------------------------------------------------------------
