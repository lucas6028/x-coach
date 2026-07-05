-- x-coach LLM chat v2.1: persist the latest answer's follow-up chips alongside the thread.
--
-- The per-answer follow-up suggestions (two grounded next-questions) were ephemeral React state, so
-- a page reload restored the messages but dropped the chips. We store only the *latest* answer's
-- chips — a single small array, mirroring how the frontend holds one `followups` at a time — in a
-- new JSONB column on the existing per-(user, video_id) conversation row.
--
-- `default '[]'` + `add column if not exists` keeps this backfill-safe: pre-existing rows read as an
-- empty chip list, and a PUT that omits `followups` is a valid clear rather than an error.

alter table public.conversations
    add column if not exists followups jsonb not null default '[]';  -- ["question?", ...] for the latest answer
