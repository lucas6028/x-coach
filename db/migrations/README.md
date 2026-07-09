# Database migrations (Supabase Postgres)

SQL migrations for the x-coach web backend's per-user data (P1: auth + history).
Apply them in filename order.

## Why `db/migrations/` and **not** `supabase/`

The Supabase CLI's convention is `supabase/migrations/`, but the backend is run **from the
repository root** (`uvicorn backend.app.main:app`, per `backend/README.md` and `CLAUDE.md`),
which puts the repo root on `sys.path`. A top-level `supabase/` directory is then picked up as
a PEP 420 namespace package and **shadows the installed `supabase` SDK** — `from supabase import
create_client` would resolve to the empty migrations folder and fail. So migrations live here
instead. If you later adopt the Supabase CLI, keep its config dir out of the import path (e.g.
run the API as an installed package, or symlink), don't reintroduce a top-level `supabase/`.

## Applying

**Option A — Dashboard (simplest).** Supabase project → SQL Editor → paste the file → Run.

**Option B — Supabase CLI.** With `--db-url` (no `supabase/` dir needed):

```bash
psql "$SUPABASE_DB_URL" -f db/migrations/20260620000000_init_videos_analyses.sql
```

## Migrations

| File | What it does |
|------|--------------|
| `20260620000000_init_videos_analyses.sql` | `videos` + `analyses` tables, owner-scoped RLS, indexes. `users` is Supabase's `auth.users`. |
| `20260704000000_conversations.sql` | `conversations` table (one saved chat thread per `(user_id, video_id)`), owner-scoped RLS. |
| `20260705000000_conversation_followups.sql` | Adds `conversations.followups` for the persisted next-question chips. |
| `20260709000000_video_size_bytes.sql` | Adds `videos.size_bytes` for the per-user storage quota (object storage on R2). |

After applying, confirm in Table Editor that `videos` and `analyses` exist with RLS enabled.
