# Per-user upload storage quota and per-file size cap

Companion to `2026-08-02-r2-object-storage-design.md`. That change moved uploads into
Cloudflare R2 and made deletion reap objects; it deliberately left the growth of
`uploads/{user_id}/` unbounded. This closes that.

## Problem

Two gaps, one of which is the unbounded-growth problem the R2 change moved rather than solved.

**No per-user limit.** `uploads/anon/` is expired by a 7-day bucket lifecycle rule, and deleting
an analysis reaps its objects — but a signed-in user who keeps uploading and never deletes has
no ceiling. The R2 spec set out to stop unbounded growth on disk and succeeded; the growth just
moved to a bucket that bills by the gigabyte.

**No per-file limit at all.** `backend/app/config.py` carries only `MAX_CONCURRENT_ANALYSES`.
Nothing anywhere rejects a large upload, and `_stage_analyze_persist` calls an unbounded
`await file.read()`. Starlette spools the multipart body to a temp FILE past 1 MB, so an
oversized upload does not arrive in RAM — but that unbounded read then materialises the whole
clip as one `bytes` object, which does. A quota check that happens *after* that read is not
much of a defence, so the two land together.

## Non-goals

- **No eviction.** Reaching the quota refuses the new upload; it never deletes the user's
  existing data. Silently discarding a user's records to make room for a new one is a worse
  failure than a clear refusal.
- **No usage-warning UI.** No "80% full" banner, no usage endpoint, no progress meter. If that
  is wanted later it is additive.
- **No backfill.** Rows predating the new column count as zero. Same decision, for the same
  reason, as the thumbnail work on the companion branch.
- **No quota for anonymous uploads.** They have no `videos` row to count against, and
  `uploads/anon/` is already handled by the lifecycle rule. They still get the per-file cap.
- **No reverse-proxy configuration.** Called out as a deployment prerequisite, not built here.

## Limits

| Limit | Default | Override key | Applies to |
|---|---|---|---|
| Per-file upload size | 100 MB | `max_upload_bytes` | every upload, incl. anonymous |
| Per-user total stored | 500 MB | `user_storage_quota_bytes` | signed-in uploads only |

Reference point for those numbers: a 4.6 s 720x1280 clip from this app's own capture path is
~5 MB, so 100 MB is roughly a 90-second clip and 500 MB is roughly 100 clips.

Both are read through the existing runtime-override layer (`backend/app/settings.py`,
`_overrides()` + `_coerce_int` with clamping) — the same shape as `rag_top_k_default()` —
so an operator can retune them from the admin panel without a redeploy.

## Architecture

### Enforcement point — `backend/app/routers/analyze.py`

Both checks go inside `_stage_analyze_persist`, the helper `/api/analyze` and
`/api/analyze/pose` already share. One implementation covers both endpoints, and they cannot
drift apart later.

Replacing today's `data = await file.read()`:

1. **Per-file cap.** `data = await file.read(max_upload_bytes() + 1)`; if
   `len(data) > max_upload_bytes()` raise `413`. Reading one byte past the limit is enough to
   detect "too large" without ever materialising more than that — the same technique
   `_read_thumbnail` already uses for the thumbnail part.
2. **Empty-upload check.** Unchanged, still `400`.
3. **Quota.** Only when `user is not None`:
   `used = store.get_storage_used(token=user.token, user_id=user.id)`; if
   `used + len(data) > user_storage_quota_bytes()` raise `413`.
4. **Everything downstream unchanged.** `stage_upload` still puts the source first and is still
   allowed to raise into a `503`.

Both checks run BEFORE `stage_upload`, so a refused upload writes no object and spends no CPU.
That is the same fail-fast ordering the storage-unavailable `503` already uses.

### What the cap does and does not buy

The capped read bounds what the process **materialises and stores**. It does not stop the bytes
being transmitted, and Starlette has already spooled the full body to temp disk before the
handler runs. Rejecting an oversized upload at the door is a reverse-proxy concern
(`client_max_body_size` or equivalent) and is listed under deployment prerequisites.

Stating this explicitly because the opposite is easy to assume: this is a storage-and-memory
bound, not an ingress bound.

### Accounting — `db/migrations/20260802000000_video_size_bytes.sql` (new)

```sql
alter table public.videos add column if not exists size_bytes bigint not null default 0;
```

`bigint`, not `integer`: a 100 MB cap fits in `int4` today, but the column is the thing an
operator would raise via an override, and a silent overflow at 2 GB is not an acceptable
failure mode for a value that is deliberately tunable.

Existing rows default to `0`, so they consume no quota. No backfill.

### Recording the size — `backend/app/services/analysis.py`

The quota must reflect everything an upload actually stores, not just the video:

- `stage_upload` already holds the source bytes, so the source size is `len(data)` at the
  call site.
- `_put_artifact` returns the number of bytes it wrote, `0` when its swallow-everything
  handler fires. It must keep never raising.
- `store_artifacts` returns the sum of what it actually stored (`pose.json` + `thumb.jpg`).
  It must keep never raising, and a partial failure must be reflected honestly — recording
  bytes that were not written would charge a user for storage they do not occupy.

`_stage_analyze_persist` adds the two and passes the total to `persist_analysis`.

### `persist_analysis` — `backend/app/services/store.py`

Gains a REQUIRED `size_bytes: int` parameter, the same treatment `storage_key` received on the
companion branch. Required rather than defaulted: a caller that forgets it should fail loudly
in review, not silently write a row that consumes no quota.

### Usage query — `store.get_storage_used` (new)

```python
def get_storage_used(*, token: str, user_id: str) -> int:
    """Total bytes this user's uploads occupy. Read with the CALLER'S OWN JWT, so RLS scopes it."""
```

Selects the caller's `size_bytes` values under their own JWT and sums them in Python. This
matches `get_storage_keys`, which likewise selects rows and processes them in Python, and it
keeps the `_user_client` patch seam the unit tests already rely on.

Considered and rejected: a PostgREST `sum()` aggregate (availability depends on
`db-aggregates-enabled`, which is not ours to guarantee) and a Postgres RPC (one more object to
migrate and stub, for a row count the quota itself bounds). At 500 MB with realistic clip
sizes this is ~100 rows. If that ever stops being true, the upgrade path is an RPC returning
`coalesce(sum(size_bytes), 0)` for `auth.uid()`.

### Settings — `backend/app/settings.py`

```python
def max_upload_bytes() -> int: ...            # default 100 MB, clamped 1 MB .. 2 GB
def user_storage_quota_bytes() -> int: ...    # default 500 MB, clamped 10 MB .. 100 GB
```

Clamped rather than rejected, matching `_coerce_int`'s existing contract, so an out-of-band or
direct-DB write cannot drive an absurd value downstream.

## Deletion frees space

No new code. `delete_analysis` already removes the row and reaps the objects, so the row's
`size_bytes` leaves the sum with it. This is what makes the hard gate usable: the user's own
remedy is the delete button that already exists.

## Error handling summary

| Condition | Status | Detail |
|---|---|---|
| File exceeds `max_upload_bytes` | 413 | names the limit in MB |
| Signed-in user would exceed quota | 413 | names used and limit in MB |
| Anonymous, file within cap | — | proceeds; no quota applies |
| Empty upload | 400 | unchanged |
| Storage unavailable | 503 | unchanged |
| Usage query fails | 503 | see below |

A failing usage query is a `503`, not a silent pass. Treating "cannot determine usage" as
"under quota" turns a database hiccup into an unbounded write path — the exact thing this spec
exists to prevent. Refusing is the conservative direction, and the caller can retry.

## Accepted imprecision

Three, all stated rather than engineered away.

**Overshoot by one upload's derived artifacts.** The quota is checked against the source size,
because that is all that is known before anything is stored, but the recorded `size_bytes` is
the true total including `pose.json` and the thumbnail. A user can therefore finish marginally
over the limit — bounded by roughly 30% of one upload — and the next upload is refused. The
alternative is estimating derived sizes before producing them, which trades a known small
error for an unknown one.

**TOCTOU between concurrent uploads.** Each in-flight upload passes the check before any of
them writes its row, so N concurrent uploads can collectively overshoot. Bounded in practice by
`MAX_CONCURRENT_ANALYSES` (2) plus request queueing, and it self-corrects on the next upload.
A correct fix is a database-side reservation, which is disproportionate machinery here.

**Orphans undercount.** An upload whose `persist_analysis` fails leaves objects with no row,
so they occupy space that no quota counts. This is the gap already documented and accepted in
the companion spec; the quota inherits it rather than widening it.

## Frontend

`api.analyzePose` already throws the server's `detail`, and `runPoseAnalysis`'s catch already
renders it, so the refusal reaches the user with no wiring change. It would reach them in
English, in a product whose UI is Traditional Chinese.

So: map `413` to a localised message in `frontend/src/api.ts` and add the two i18n keys
(zh-TW + en). Nothing else on the frontend changes — no usage display, no pre-flight size check
in the browser.

## Testing

Backend unit tests, `tests/`:

- **Per-file cap** — an upload of exactly `max_upload_bytes` succeeds; `+1` byte gets `413`.
  The boundary is tested from both sides so an off-by-one cannot pass.
- **Quota** — under the limit succeeds; over gets `413`; an anonymous upload over the quota
  still succeeds (cap only). A user at exactly the limit is refused the next non-empty upload.
- **Size recording** — `size_bytes` equals source + pose + thumbnail; an upload whose thumbnail
  is absent records source + pose only; a `_put_artifact` failure records only what was stored.
- **Never-raises contracts hold** — `store_artifacts` still returns rather than raising when a
  put fails, now while also returning a byte count. Needs a mutation check: the companion
  branch shipped five defects of exactly this shape, where a docstring promised a guarantee the
  code did not deliver.
- **Usage query** — `get_storage_used` sums the caller's rows; a row with `NULL`/absent
  `size_bytes` counts as 0; a failing query surfaces as `503`, not as a pass.
- **Settings** — defaults, override applied, out-of-range override clamped.

Frontend: `413` renders the localised message rather than the raw English detail.

The backend coverage gate (95%) applies. New test files must be added to
`scripts/run_backend_coverage.py`'s `_DEFAULT_TESTS` — a suite absent from that list is
measured but never run, which the companion branch hit.

## Deployment prerequisites

1. Apply `db/migrations/20260802000000_video_size_bytes.sql` to Supabase.
2. Set a reverse-proxy request-body limit at or slightly above `max_upload_bytes`. Without it
   the application cap still protects memory and storage, but oversized bodies are transmitted
   and spooled to temp disk before being refused.
3. Optionally set `max_upload_bytes` / `user_storage_quota_bytes` overrides in the admin panel;
   the built-in defaults apply if not.
