-- Record which movement's rules produced each analysis.
--
-- Additive and nullable on purpose: existing rows keep working (they predate per-movement
-- analysis, when everything was Squat), RLS policies are unaffected, and the
-- admin_user_overview view needs no change.
--
-- The analysis document in `result` also carries `movement` for anything analysed after this
-- lands, so the frontend can fall back to it for rows where this column is null.

alter table public.analyses
    add column if not exists movement text;
