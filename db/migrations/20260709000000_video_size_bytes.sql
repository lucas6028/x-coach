-- x-coach: track each upload's byte size so the backend can enforce a per-user storage quota.
--
-- The quota check (see backend/app/services/store.get_usage) sums `size_bytes` over the caller's
-- own rows (RLS-scoped), so 30–40 demo users can't each fill object storage without bound. Existing
-- rows default to 0 (they predate object storage; their bytes are unknown and don't count).
--
-- `storage_key` (added in the init migration) now holds the R2 object key for uploads that were
-- pushed to object storage; it still falls back to the local runtime path when R2 is unconfigured.

alter table public.videos
    add column if not exists size_bytes bigint not null default 0;
