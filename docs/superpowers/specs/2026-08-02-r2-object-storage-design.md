# Cloudflare R2 object storage for user uploads

Date: 2026-08-02
Status: approved, ready for implementation planning

## Problem

User-uploaded artifacts live on the backend's local disk:

- raw video — `data/runtime/uploads/{video_id}{suffix}` (`analysis.save_upload`)
- pose JSON — `data/runtime/pose_json/{video_id}.json` (`analysis.analyze_video_file`,
  `analysis.analyze_pose_payload`)
- thumbnail — does not exist anywhere in the codebase today

Local disk does not survive a redeploy, cannot be shared across processes, and grows without
bound (`store.delete_analysis` explicitly leaves the file behind). `store.persist_analysis`
already writes a placeholder `storage_key` of `runtime/uploads/{video_id}` with a comment
saying "P2 will repoint at object storage" — this is that change.

Two decisions were made alongside it:

- **Thumbnails become real.** The browser generates one frame per upload; the History page
  renders it. Storing an artifact nothing reads would not be worth the plumbing.
- **The `GET /api/video-file/{video_id}` IDOR is closed.** That endpoint has no auth
  dependency at all, so any caller who knows a `video_id` can fetch any user's upload.
  Presigned URLs make the URL itself the capability, so shipping them without an ownership
  check would move the same hole to a CDN. The read path is being rewritten anyway; fixing
  it here costs almost nothing.

## Non-goals

- **No backfill.** The app is not yet deployed; `data/runtime/` holds only test data. The
  cutover is a straight switch with no dual-read fallback and no migration script.
- No CDN custom domain, no image resizing service, no video transcoding.
- No change to the analysis pipeline itself (`src/pose/...` is untouched).

## Architecture

### Storage abstraction — `backend/app/services/storage.py` (new)

A narrow interface so `analysis.py`, `videos.py`, and `store.py` never learn that R2 exists:

```python
put(key: str, data: bytes, *, content_type: str) -> None
presigned_url(key: str, *, expires_in: int) -> str
delete_prefix(prefix: str) -> None
```

Two implementations behind it:

- `LocalObjectStore` — writes under `data/runtime/objects/{key}`. `presigned_url` returns a
  backend-served URL (`/api/local-object/{key}`), so the frontend contract is identical in
  both modes: fetch a URL, set it as `src`. That endpoint is **development-only**: it carries
  no signature and is registered only when `storage_configured` is false, so a deployment with
  R2 credentials never exposes it. Reaching an object still requires having called the
  ownership-checked `/api/uploads/{video_id}/url` first.
- `R2ObjectStore` — boto3 S3 client with
  `endpoint_url=https://{account_id}.r2.cloudflarestorage.com`, `region_name="auto"`,
  signature v4. `presigned_url` delegates to `generate_presigned_url("get_object", ...)`.

`get_object_store()` is `lru_cache`d and selects R2 when every `R2_*` setting is present,
Local otherwise. Consequence: CI and offline development need no network and no credentials,
which keeps the 95% backend coverage gate reachable.

New dependency: `boto3` in `requirements.txt` and `requirements-ci.txt`.

### Settings — `backend/app/settings.py`

```
R2_ACCOUNT_ID
R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY
R2_BUCKET
```

Plus a `storage_configured` property (all four present), mirroring the existing
`auth_configured` / `chat_configured` / `line_messaging_configured` pattern. Credentials are
server-side only and never reach the browser.

### Key layout

```
uploads/{owner}/{video_id}/source{suffix}
uploads/{owner}/{video_id}/pose.json
uploads/{owner}/{video_id}/thumb.jpg
```

`owner` is the authenticated user's id, or the literal `anon` for unauthenticated demo
uploads. `videos.storage_key` stores the **prefix** `uploads/{owner}/{video_id}` (not a single
object), which is what `delete_prefix` consumes.

Anonymous uploads are written to R2 so the demo flow behaves identically to the signed-in one,
but nothing references them after the response. An R2 lifecycle rule expiring
`uploads/anon/` after 7 days keeps that from accumulating; it is bucket configuration, not
code.

## Write path

Both `/api/analyze` and `/api/analyze/pose` follow the same order. `process_video` (OpenCV)
and the detector's camera-view estimation both need a **real filesystem path**, so temporary
files remain — they are just no longer the system of record.

1. **Put the source video to R2 first.** The video is the one artifact that cannot be
   recomputed, and the upload is fast relative to analysis. If this put fails, return 503 and
   let the client retry — burning CPU on an analysis whose video cannot be kept is worse than
   failing fast.
2. Write the same bytes to a `tempfile` and run the existing analysis unchanged; pose JSON
   goes to a temp path too.
3. Put `pose.json` and `thumb.jpg` to R2, then delete the temp files. A failure here is logged
   and swallowed — never discard a completed (expensive) analysis over a storage hiccup, the
   same policy `persist_analysis` already uses.
4. Attach `video_url` (presigned, 1 hour) to the response so playback works immediately after
   upload.

`config.UPLOAD_DIR`, `config.UPLOAD_POSE_DIR`, and `config.ensure_runtime_dirs()` are removed,
along with their call in `main.py` and in `analysis.py`.

**`video_url` is attached to the HTTP response only — never to the `result` dict persisted as
JSONB.** Set it after `store.persist_analysis` returns. Storing it would put an
already-expired signature into every history row.

### Thumbnail upload

`/api/analyze` and `/api/analyze/pose` gain an optional `thumbnail: UploadFile | None = File(None)`
form field. The backend validates content type (`image/jpeg` or `image/png`) and enforces a
size cap; a missing thumbnail is not an error (older clients, or a browser where frame capture
failed, still analyze fine).

## Read path

- **New: `GET /api/uploads/{video_id}/url`** → `{video_url, thumbnail_url, expires_in}`.
  Requires authentication. It looks up the `videos` row using the caller's own JWT, so
  **RLS performs the ownership check** — a row belonging to someone else is simply not
  visible, and the handler returns 404. Missing and forbidden are deliberately
  indistinguishable, matching `store.delete_analysis`'s existing behaviour.
- **`GET /api/video-file/{video_id}` is narrowed to library demo clips only.** The
  `library.uploaded_video_path(video_id)` fallback is deleted, and `uploaded_video_path`
  itself goes away with it. The IDOR closes structurally rather than by adding a guard: there
  is no longer any code path from that endpoint to a user upload. Library clips under
  `data/Fitness-AQA/...` are shared demo assets and stay public — that asymmetry is a
  decision, not an oversight.

### Frontend

`VideoPanel.tsx:103` is the single call site; it currently sets `src` synchronously from
`api.videoFileUrl(analysis.video_id)`. It becomes a small hook that resolves the URL by source:

- fresh upload → `analysis.video_url` from the analyze response
- history replay → `GET /api/uploads/{video_id}/url` (the stored result has no URL by design)
- library clip → `api.videoFileUrl(video_id)`, unchanged

While a URL is resolving the panel shows its existing loading treatment; on failure it shows
the analysis without playback rather than blocking the page.

### Thumbnail generation (browser)

One shared utility used by both analysis paths, since the browser holds the `File` in both
cases:

- load the clip into an off-screen `<video>`, seek to **25% of duration** (a frame that is
  usually mid-movement and rarely black), draw to a canvas
- longest edge scaled to 480px, `canvas.toBlob('image/jpeg', 0.8)`
- appended to the same multipart POST as `thumbnail`
- any failure (decode error, seek timeout) resolves to `null` and the upload proceeds without
  a thumbnail

### History cards

`HistoryPage` cards are text-only today. They gain the thumbnail, fetched per row from
`GET /api/uploads/{video_id}/url`, with the existing card layout as the placeholder while
loading and for rows that have no thumbnail (every row predating this change).

## Deletion

`store.delete_analysis` and `store.delete_all_analyses` call `delete_prefix(storage_key)` for
each removed video row. Best-effort and logged: a storage failure must not leave the DB row
undeleted. This closes the orphan noted in `delete_analysis`'s own docstring ("The uploaded
file under `runtime/uploads/` is deliberately left on disk").

`delete_all_analyses` currently deletes `videos` rows without reading them; it must select the
`storage_key`s first so it knows what to reap.

## Error handling summary

| Failure | Behaviour |
| --- | --- |
| Source video put fails | 503, no analysis run |
| pose.json / thumb.jpg put fails | logged, analysis returned normally |
| Presign fails on read | 503 from the URL endpoint; frontend renders analysis without video |
| `delete_prefix` fails | logged, DB deletion still committed |
| R2 unconfigured | `LocalObjectStore` transparently, no error |

## Testing

**Backend (`tests/`, `unittest.TestCase`):**

- `tests/test_storage.py` — `LocalObjectStore` round-trip (put → presigned_url → read →
  delete_prefix); key construction for authenticated and `anon` owners; `R2ObjectStore`
  against a patched fake boto3 client asserting bucket/key/content-type and the presign
  expiry; `get_object_store()` selection by settings.
- Analyze-router tests: thumbnail accepted / rejected / absent; source-put failure yields 503
  and runs no analysis; pose-put failure still returns the analysis; `video_url` present in
  the response and absent from the persisted `result`.
- `GET /api/uploads/{video_id}/url`: owner gets URLs, non-owner and unknown id both get 404,
  unauthenticated gets 401.
- `GET /api/video-file/{id}` returns 404 for an upload id (regression test for the IDOR).
- Deletion tests assert `delete_prefix` is called with the right prefix and that a storage
  failure does not roll back the DB delete.

**Frontend (`frontend/src/test/`, vitest):**

- thumbnail capture utility: produces a blob, respects the size cap, resolves `null` on decode
  failure
- `VideoPanel` URL resolution across the three sources
- `HistoryPage` renders a thumbnail when present and falls back cleanly when absent

CI parity: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95` and
`yarn test:coverage` from `frontend/`.

## Deployment prerequisites

1. Create the R2 bucket and an API token scoped to it.
2. Set `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` in the
   backend environment.
3. Add the lifecycle rule expiring `uploads/anon/` after 7 days.

No bucket CORS configuration is required: a plain `<video src>` playing a cross-origin URL is
not CORS-restricted, and the skeleton overlay draws on a separate canvas above the video
rather than reading its pixels, so no canvas tainting arises.
