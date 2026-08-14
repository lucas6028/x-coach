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
| `20260725000000_analysis_movement.sql` | Adds nullable `analyses.movement` (which detector produced the row). **Apply before deploying the movement-selection backend/frontend** — `store.py` writes this column unconditionally, so against an unmigrated database `POST /api/analyze` still returns 200 but silently drops the row (`analysis_id: null`, error only in the server log) and `GET /api/analyses` hard-500s for every signed-in user. See `backend/README.md` § Auth & persistence. |
| `20260802000000_video_size_bytes.sql` | Adds `videos.size_bytes bigint not null default 0` (bytes each upload's stored artifacts occupy), for the per-user storage quota. Additive, no backfill, RLS unaffected. **Apply before deploying the upload-quota backend** — the quota gate calls `store.get_storage_used`, whose `select("size_bytes")` fails at PostgREST against an unmigrated database, and the router turns that into a **503 on every signed-in upload** (anonymous uploads keep working). That is fail-closed and therefore the correct direction, but it is a total upload outage for signed-in users until the migration lands. |
| `20260813000000_training_plans.sql` | `training_plans` + `plan_items` tables for 訓練菜單 (a reusable Day 1–7 routine), owner-scoped RLS, and `plan_items.analysis_id → analyses(id) on delete set null` so deleting an analysis cannot leave a plan item linking to a 404. **Apply before deploying the plans backend/frontend** — without it every `/api/plans` call fails at PostgREST and the whole feature errors out. Nothing else in the app touches these tables, so an unmigrated database leaves the rest of the app working normally; the blast radius is the plans page alone. Read the file's header comment before applying: it states exactly what restarting a plan throws away, which is deliberate and reads like a bug otherwise. |

After applying, confirm in Table Editor that `videos` and `analyses` exist with RLS enabled.

Note: the table above is missing rows for `20260704000000_conversations.sql` through
`20260720000000_line_training_summary.sql` (pre-existing gap, not fixed here — see the
2026-07-25 review). Apply every file in `db/migrations/` in filename order regardless of
whether it has a row here.
