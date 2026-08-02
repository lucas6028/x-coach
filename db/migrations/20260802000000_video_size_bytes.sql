-- How many bytes each upload's stored artifacts occupy, for the per-user storage quota.
--
-- Additive with NOT NULL DEFAULT 0 on purpose: rows predating this column consume no quota
-- (there is deliberately no backfill), RLS policies are unaffected, and the
-- admin_user_overview view needs no change.
--
-- bigint, not integer: the value this column is summed against is an ADMIN-TUNABLE override,
-- so a silent overflow once someone raises the quota past 2 GB is not an acceptable failure
-- mode for a limit that exists to be adjusted.

alter table public.videos
    add column if not exists size_bytes bigint not null default 0;
