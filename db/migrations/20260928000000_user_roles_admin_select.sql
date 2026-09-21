-- Let an admin SELECT every user_roles row, so the in-app role toggle can write another user's row.
--
-- The admin_roles migration made SELECT self-only (user_id = auth.uid()) and gated the writes on
-- is_admin(). That is not enough for a write to ANOTHER user's row:
--   * INSERT ... ON CONFLICT DO UPDATE (what the backend's upsert sends) checks the SELECT policy
--     against the new row, so a grant failed with 42501 "new row violates row-level security
--     policy" and the role PUT answered 500.
--   * DELETE ... WHERE user_id = ... only sees rows the SELECT policy exposes, so a revoke matched
--     zero rows and silently did nothing.
-- Admins already read every user's roles through admin_list_users(), so this exposes nothing new.
-- is_admin() is SECURITY DEFINER, so calling it from a user_roles policy does not recurse.

drop policy if exists "user_roles_select_authenticated" on public.user_roles;
create policy "user_roles_select_authenticated" on public.user_roles
    for select
    to authenticated
    using (user_id = auth.uid() or public.is_admin(auth.uid()));
