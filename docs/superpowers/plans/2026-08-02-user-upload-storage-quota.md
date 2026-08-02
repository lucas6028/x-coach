# Per-user upload storage quota and per-file size cap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refuse an upload that exceeds 100 MB, or that would push a signed-in user past 500 MB of stored uploads, with a localised message telling them what to do about it.

**Architecture:** Both checks live in `_stage_analyze_persist`, the helper `/api/analyze` and `/api/analyze/pose` already share, and both run before `stage_upload` so a refused upload writes no object and spends no CPU. Usage is a new `videos.size_bytes` column summed under the caller's own JWT, so RLS scopes it. Limits are read through the existing admin runtime-override layer.

**Tech Stack:** FastAPI + Starlette `UploadFile`, Supabase (supabase-py, user-JWT mode, RLS), Postgres migration SQL, React 18 + TypeScript + vitest.

**Spec:** `docs/superpowers/specs/2026-08-02-user-upload-storage-quota-design.md`

## Global Constraints

- Per-file cap default **100 MB** (`100 * 1024 * 1024`), override key `max_upload_bytes`, clamped **1 MB .. 2 GB**.
- Per-user quota default **500 MB** (`500 * 1024 * 1024`), override key `user_storage_quota_bytes`, clamped **10 MB .. 100 GB**.
- **Hard gate, never eviction.** Exceeding the quota refuses the new upload. Nothing the user already stored is ever deleted to make room.
- **Anonymous uploads are exempt from the quota** but still subject to the per-file cap.
- **No backfill.** `size_bytes` defaults to `0`; rows predating the column consume no quota.
- A failing usage query is a **503**, never a silent pass.
- Every limit getter is called through `run_in_threadpool` — `_overrides()` can do a synchronous Supabase round trip on a cold cache. `analyze.py` already does this for `allowed_upload_suffixes`.
- Python is **always** `.venv\Scripts\python.exe`. Never bare `python`/`pip`, never `source .venv/bin/activate`.
- Backend tests are **always** scoped to `tests/`.
- ALL frontend commands run with **cwd = `frontend/`**. The Bash and PowerShell tools share one cwd; a stray `cd` elsewhere mass-fails vitest.
- Backend coverage gate is **95%**: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`.
- Any NEW test file must be added to `_DEFAULT_TESTS` in `scripts/run_backend_coverage.py`, or it is measured but never run.
- Tests never touch the network.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `backend/app/settings.py` | the two limit getters, clamped | 1 |
| `db/migrations/20260802000000_video_size_bytes.sql` (new) | `videos.size_bytes` column | 2 |
| `backend/app/services/store.py` | persist the size; sum the caller's usage | 2 |
| `backend/app/services/analysis.py` | artifact writers return bytes actually stored | 3 |
| `backend/app/routers/analyze.py` | the capped read (4), the quota gate + size recording (5) | 4, 5 |
| `tests/test_upload_limits.py` (new) | usage query, cap, quota | 2, 4, 5 |
| `scripts/run_backend_coverage.py` | register the new test file | 2 |
| `frontend/src/api.ts` | parse the 413 into a typed error | 6 |
| `frontend/src/App.tsx` | render it via i18n | 6 |
| `frontend/src/lib/i18n.tsx` | two keys, en + zh-TW | 6 |

---

### Task 1: Settings — the two limit getters

**Files:**
- Modify: `backend/app/settings.py` (constants near `_DEFAULT_UPLOAD_SUFFIXES:199`; getters after `allowed_upload_suffixes:315-327`)
- Test: `tests/test_backend.py` (the settings suite containing `test_rag_kg_defaults_and_overrides:2078`)

**Interfaces:**
- Consumes: `_overrides()`, `_coerce_int(value, default, *, minimum, maximum)` — both already in `settings.py`.
- Produces: `settings.max_upload_bytes() -> int`, `settings.user_storage_quota_bytes() -> int`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_backend.py`, immediately after `test_rag_kg_defaults_and_overrides`. Use the suite's existing `self._no_overrides()` / `self._overrides({...})` context managers:

```python
    def test_upload_limit_defaults_and_overrides(self) -> None:
        with self._no_overrides():
            self.assertEqual(app_settings.max_upload_bytes(), 100 * 1024 * 1024)
            self.assertEqual(app_settings.user_storage_quota_bytes(), 500 * 1024 * 1024)
        with self._overrides({"max_upload_bytes": 5 * 1024 * 1024,
                              "user_storage_quota_bytes": 50 * 1024 * 1024}):
            self.assertEqual(app_settings.max_upload_bytes(), 5 * 1024 * 1024)
            self.assertEqual(app_settings.user_storage_quota_bytes(), 50 * 1024 * 1024)

    def test_upload_limits_fall_back_on_bad_values(self) -> None:
        with self._overrides({"max_upload_bytes": "nonsense",
                              "user_storage_quota_bytes": ""}):
            self.assertEqual(app_settings.max_upload_bytes(), 100 * 1024 * 1024)
            self.assertEqual(app_settings.user_storage_quota_bytes(), 500 * 1024 * 1024)

    def test_upload_limits_clamp_out_of_range_overrides(self) -> None:
        # An out-of-band / direct-DB write must not drive an absurd value downstream: a 0 would
        # reject every upload, and a petabyte would defeat the point of having a limit.
        with self._overrides({"max_upload_bytes": 0, "user_storage_quota_bytes": 0}):
            self.assertEqual(app_settings.max_upload_bytes(), 1 * 1024 * 1024)
            self.assertEqual(app_settings.user_storage_quota_bytes(), 10 * 1024 * 1024)
        with self._overrides({"max_upload_bytes": 10**15, "user_storage_quota_bytes": 10**15}):
            self.assertEqual(app_settings.max_upload_bytes(), 2 * 1024 * 1024 * 1024)
            self.assertEqual(app_settings.user_storage_quota_bytes(), 100 * 1024 * 1024 * 1024)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend.py -k upload_limit -v`
Expected: FAIL — `AttributeError: module 'backend.app.settings' has no attribute 'max_upload_bytes'`

- [ ] **Step 3: Add the constants**

In `backend/app/settings.py`, directly below `_DEFAULT_UPLOAD_SUFFIXES` (line 199):

```python
# Upload limits, in BYTES — the override keys say `_bytes` so no unit guessing is possible.
_DEFAULT_MAX_UPLOAD_BYTES = 100 * 1024 * 1024
_DEFAULT_USER_STORAGE_QUOTA_BYTES = 500 * 1024 * 1024
# Clamp bounds. A floor because a 0 would reject every upload and lock the product; a ceiling
# because a limit an operator can raise without bound is not a limit.
_MIN_MAX_UPLOAD_BYTES = 1 * 1024 * 1024
_MAX_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
_MIN_USER_STORAGE_QUOTA_BYTES = 10 * 1024 * 1024
_MAX_USER_STORAGE_QUOTA_BYTES = 100 * 1024 * 1024 * 1024
```

- [ ] **Step 4: Add the getters**

In `backend/app/settings.py`, directly after `allowed_upload_suffixes()` (which ends at line 327):

```python
def max_upload_bytes() -> int:
    """Largest accepted upload — override ``max_upload_bytes``, else 100 MB (clamped 1 MB..2 GB).

    Bounds what the process READS AND STORES, not what a client can transmit: Starlette has
    already spooled the whole body to a temp file before the handler runs. Rejecting oversized
    requests at the door is a reverse-proxy concern (see the spec's deployment prerequisites).
    """
    return _coerce_int(
        _overrides().get("max_upload_bytes"),
        _DEFAULT_MAX_UPLOAD_BYTES,
        minimum=_MIN_MAX_UPLOAD_BYTES,
        maximum=_MAX_MAX_UPLOAD_BYTES,
    )


def user_storage_quota_bytes() -> int:
    """Total bytes one user's uploads may occupy — override ``user_storage_quota_bytes``, else
    500 MB (clamped 10 MB..100 GB). Anonymous uploads are not counted against any quota."""
    return _coerce_int(
        _overrides().get("user_storage_quota_bytes"),
        _DEFAULT_USER_STORAGE_QUOTA_BYTES,
        minimum=_MIN_USER_STORAGE_QUOTA_BYTES,
        maximum=_MAX_USER_STORAGE_QUOTA_BYTES,
    )
```

- [ ] **Step 5: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend.py -k upload_limit -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/app/settings.py tests/test_backend.py
git commit -m "feat(settings): admin-tunable upload size cap and per-user storage quota"
```

---

### Task 2: The `size_bytes` column, its writer, and the usage query

**Files:**
- Create: `db/migrations/20260802000000_video_size_bytes.sql`
- Create: `tests/test_upload_limits.py`
- Modify: `backend/app/services/store.py` (`persist_analysis:139-189`; new `get_storage_used` beside `get_storage_keys:405-424`)
- Modify: `scripts/run_backend_coverage.py` (`_DEFAULT_TESTS:25-36`)
- Modify: `tests/test_backend.py` (the four direct `persist_analysis(` calls at lines 1495, 1517, 1530, 1547)

**Interfaces:**
- Consumes: `_user_client(token)` from `store.py`.
- Produces: `store.get_storage_used(*, token: str, user_id: str) -> int`; `store.persist_analysis` gains a REQUIRED keyword `size_bytes: int`.

- [ ] **Step 1: Write the migration**

Create `db/migrations/20260802000000_video_size_bytes.sql`:

```sql
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
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_upload_limits.py`. This file grows in Tasks 4 and 5; the fake client mirrors `tests/test_delete_reaping.py`.

```python
"""Per-file upload cap and per-user storage quota.

Offline throughout — the Supabase client is a fake and the object store is never reached.
"""

from __future__ import annotations

import unittest
from unittest import mock

from backend.app.services import store


class _Result:
    def __init__(self, data=None):
        self.data = data if data is not None else []


class _Table:
    def __init__(self, name, responses, log):
        self._name, self._responses, self._log = name, responses, log
        self._op = None

    def select(self, *args, **kwargs):
        self._op = "select"
        self._log.append((self._name, "select", args))
        return self

    def upsert(self, payload, **kwargs):
        self._op = "upsert"
        self._log.append((self._name, "upsert", payload))
        return self

    def insert(self, payload, **kwargs):
        self._op = "insert"
        self._log.append((self._name, "insert", payload))
        return self

    def eq(self, *args, **kwargs):
        return self

    def execute(self):
        return self._responses.get((self._name, self._op), _Result())


class _Client:
    def __init__(self, responses, log):
        self._responses, self._log = responses, log

    def table(self, name):
        return _Table(name, self._responses, self._log)


class GetStorageUsedTests(unittest.TestCase):
    def _used(self, rows):
        client = _Client({("videos", "select"): _Result(data=rows)}, [])
        with mock.patch.object(store, "_user_client", return_value=client):
            return store.get_storage_used(token="t", user_id="u1")

    def test_sums_the_callers_rows(self) -> None:
        self.assertEqual(self._used([{"size_bytes": 10}, {"size_bytes": 32}]), 42)

    def test_no_rows_is_zero_not_an_error(self) -> None:
        self.assertEqual(self._used([]), 0)

    def test_a_null_size_counts_as_zero(self) -> None:
        """Rows predating the column read back as NULL — they must not poison the sum."""
        self.assertEqual(self._used([{"size_bytes": None}, {"size_bytes": 5}]), 5)

    def test_a_row_missing_the_key_counts_as_zero(self) -> None:
        self.assertEqual(self._used([{}, {"size_bytes": 7}]), 7)


class PersistAnalysisRecordsSizeTests(unittest.TestCase):
    def test_writes_size_bytes_onto_the_videos_row(self) -> None:
        log: list[tuple] = []
        client = _Client({("analyses", "insert"): _Result(data=[{"id": "a1"}])}, log)
        with mock.patch.object(store, "_user_client", return_value=client):
            store.persist_analysis(
                token="t",
                user_id="u1",
                video_id="upload_a",
                source="upload",
                storage_key="uploads/u1/upload_a",
                size_bytes=1234,
                result={},
            )
        upserts = [payload for name, op, payload in log if name == "videos" and op == "upsert"]
        self.assertEqual(len(upserts), 1)
        self.assertEqual(upserts[0]["size_bytes"], 1234)
```

- [ ] **Step 3: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_upload_limits.py -v`
Expected: FAIL — `AttributeError: module 'backend.app.services.store' has no attribute 'get_storage_used'`, and a `TypeError` about the unexpected `size_bytes` keyword.

- [ ] **Step 4: Add `size_bytes` to `persist_analysis`**

In `backend/app/services/store.py`, add the parameter to the signature (after `storage_key`):

```python
    storage_key: str,
    size_bytes: int,
```

Add to the docstring, after the `storage_key` paragraph:

```
    ``size_bytes`` is the total the upload's stored artifacts occupy (source + pose JSON +
    thumbnail), and is what the storage quota sums. REQUIRED rather than defaulted: a caller
    that forgets it would silently write a row consuming no quota, which is a hole that fails
    open. It must reflect what was ACTUALLY stored -- ``store_artifacts`` returns 0 for an
    artifact whose put failed, so a partial failure does not bill the user for absent bytes.
```

And add the field to the `videos` upsert dict, after `"storage_key": storage_key,`:

```python
            "size_bytes": size_bytes,
```

- [ ] **Step 5: Add `get_storage_used`**

In `backend/app/services/store.py`, directly after `get_storage_keys` (ends line 424):

```python
def get_storage_used(*, token: str, user_id: str) -> int:
    """Total bytes this user's uploads occupy, for the storage quota.

    Read with the CALLER'S OWN JWT, so the ``videos`` RLS policy is what scopes it — the
    ``user_id`` filter is belt-and-braces, not the security boundary, exactly as everywhere
    else on this path.

    Summed in Python rather than with a PostgREST aggregate: aggregates depend on
    ``db-aggregates-enabled``, which is not ours to guarantee, and the quota itself bounds how
    many rows this can return. A row whose ``size_bytes`` is NULL or absent counts as 0,
    matching the column default for rows that predate it. If the row count ever outgrows this,
    the upgrade is an RPC returning ``coalesce(sum(size_bytes), 0)`` for ``auth.uid()``.
    """
    client = _user_client(token)
    resp = client.table("videos").select("size_bytes").eq("user_id", user_id).execute()
    return sum(int(row.get("size_bytes") or 0) for row in (resp.data or []))
```

- [ ] **Step 6: Register the new test file for coverage**

In `scripts/run_backend_coverage.py`, add to `_DEFAULT_TESTS` after `"tests/test_delete_reaping.py",`:

```python
    "tests/test_upload_limits.py",
```

- [ ] **Step 7: Fix the existing direct `persist_analysis` callers**

`size_bytes` is required, so the four direct calls in `tests/test_backend.py` (lines 1495, 1517, 1530, 1547) now fail. Add `size_bytes=0,` to each call's keyword arguments. These tests assert on the analyses insert, not on sizing, so `0` is the honest value.

- [ ] **Step 8: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_upload_limits.py tests/test_backend.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add db/migrations/20260802000000_video_size_bytes.sql backend/app/services/store.py tests/test_upload_limits.py tests/test_backend.py scripts/run_backend_coverage.py
git commit -m "feat(store): record each upload's stored size and sum a user's usage"
```

---

### Task 3: Artifact writers return the bytes they actually stored

**Files:**
- Modify: `backend/app/services/analysis.py` (`_put_artifact:250-263`, `store_artifacts:266-287`)
- Test: `tests/test_upload_staging.py` (the `store_artifacts` suite, lines 97-143)

**Interfaces:**
- Consumes: nothing new.
- Produces: `analysis._put_artifact(...) -> int` and `analysis.store_artifacts(staged, *, thumbnail=None) -> int`, both still NEVER raising.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_upload_staging.py`, in the class that already exercises `store_artifacts`:

```python
    def test_returns_the_bytes_it_stored(self) -> None:
        self.staged.pose_path.write_text('{"a": 1}', encoding="utf-8")
        pose_len = self.staged.pose_path.stat().st_size
        stored = analysis.store_artifacts(self.staged, thumbnail=b"jpeg-bytes")
        self.assertEqual(stored, pose_len + len(b"jpeg-bytes"))

    def test_returns_zero_when_there_is_nothing_derived_to_store(self) -> None:
        """The no-detector branch writes no pose JSON and may carry no thumbnail."""
        self.assertEqual(analysis.store_artifacts(self.staged, thumbnail=None), 0)

    def test_a_failed_put_contributes_zero_not_its_length(self) -> None:
        """MUTATION CHECK for the quota's honesty: charging a user for bytes that were never
        written bills them for space they do not occupy. Returning len(data) here instead of 0
        must fail this test."""
        self.staged.pose_path.write_text('{"a": 1}', encoding="utf-8")
        with mock.patch.object(storage, "get_object_store", return_value=_FailingStore()):
            stored = analysis.store_artifacts(self.staged, thumbnail=b"jpeg")  # must not raise
        self.assertEqual(stored, 0)

    def test_a_partial_failure_returns_only_what_landed(self) -> None:
        """Thumbnail put fails, pose put succeeds -> only the pose bytes count."""
        self.staged.pose_path.write_text('{"a": 1}', encoding="utf-8")
        pose_len = self.staged.pose_path.stat().st_size
        real_put = self.store.put

        def flaky_put(key, data, *, content_type):
            if key.endswith("thumb.jpg"):
                raise storage.StorageError("nope")
            real_put(key, data, content_type=content_type)

        with mock.patch.object(self.store, "put", side_effect=flaky_put):
            stored = analysis.store_artifacts(self.staged, thumbnail=b"jpeg")  # must not raise
        self.assertEqual(stored, pose_len)
```

If the enclosing test class does not already create `self.staged` with a writable `pose_path`, mirror the setUp used by the existing `store_artifacts` tests in this file.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_upload_staging.py -v`
Expected: FAIL — `AssertionError: None != 18` (the functions currently return `None`).

- [ ] **Step 3: Make `_put_artifact` return its byte count**

In `backend/app/services/analysis.py`, replace `_put_artifact`'s signature, docstring tail and body:

```python
def _put_artifact(staged: StagedUpload, name: str, data: bytes, content_type: str) -> int:
    """Upload one derived artifact, swallowing every failure. Returns the bytes STORED — 0 when
    the put failed. See ``store_artifacts``.

    Returning 0 rather than ``len(data)`` on failure is what keeps the storage quota honest:
    the caller adds this into ``videos.size_bytes``, and counting bytes that were never written
    would charge a user for space they do not occupy.

    ``except Exception`` is deliberate and matches ``store.persist_analysis``'s policy. A
    narrower ``except storage.StorageError`` would NOT hold the contract: ``LocalObjectStore.put``
    does real filesystem IO (``mkdir`` + ``write_bytes``) and raises ``OSError``, not
    ``StorageError`` — so on the dev/CI path a full disk would escape and sink a completed analysis.
    """
    try:
        storage.get_object_store().put(
            f"{staged.prefix}/{name}", data, content_type=content_type
        )
    except Exception:  # noqa: BLE001 — a derived artifact must never sink a completed analysis
        logger.exception("Failed to store %s for %s", name, staged.video_id)
        return 0
    return len(data)
```

- [ ] **Step 4: Make `store_artifacts` return the total**

Replace its signature, add the return contract to the docstring, and accumulate:

```python
def store_artifacts(staged: StagedUpload, *, thumbnail: bytes | None = None) -> int:
    """Best-effort upload of the derived artifacts. NEVER RAISES. Returns bytes ACTUALLY stored.

    Mirrors ``store.persist_analysis``'s policy: a storage hiccup is logged, but it must never
    discard an analysis that already cost a full pipeline run. The caller relies on that literally
    — it does not wrap this call — so every path here has to hold it, including the new return.

    The return value feeds ``videos.size_bytes`` and therefore the storage quota, so it counts
    only what landed: a failed put contributes 0, and an unreadable pose file contributes 0.

    ``pose.json`` is uploaded ONLY when one was actually produced. ``analyze_pose_payload``
    returns the ``analysis_pending`` skeleton without writing any pose JSON for a movement with
    no registered detector — which is most of the movement registry, not an edge case.
    """
    stored = 0
    if staged.pose_path.is_file():
        try:
            pose_bytes = staged.pose_path.read_bytes()
        except OSError:
            # ``is_file()`` and the read are not atomic, and the read raises OSError rather than
            # StorageError — so this needs its own guard, not the put's.
            logger.exception("Failed to read staged pose JSON for %s", staged.video_id)
        else:
            stored += _put_artifact(staged, "pose.json", pose_bytes, "application/json")
    if thumbnail:
        stored += _put_artifact(staged, "thumb.jpg", thumbnail, "image/jpeg")
    return stored
```

- [ ] **Step 5: Fix every test stub of `store_artifacts`**

Task 5 will make the router add this return value to an integer. A stub returning `None` or a `MagicMock` becomes a `TypeError` there, so fix all six now:

- `tests/test_analyze_pose_endpoint.py:46` and `:232` — the lambda currently ends in `self.artifacts.append({...})`, which returns `None`. Change each to append and then return an int, e.g.:
  ```python
  analysis_service.store_artifacts = lambda staged, *, thumbnail=None: (
      self.artifacts.append({"staged": staged, "thumbnail": thumbnail}) or 0
  )
  ```
- `tests/test_analyze_endpoint.py:66` and `:261` — same shape, same fix.
- `tests/test_backend.py:111` and `tests/test_analyze_movement.py:61` — these use `mock.patch.object(analysis, "store_artifacts")`, whose default `MagicMock` return would blow up the addition. Add `return_value=0` to both patch calls.

- [ ] **Step 6: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_upload_staging.py tests/test_analyze_endpoint.py tests/test_analyze_pose_endpoint.py tests/test_analyze_movement.py tests/test_backend.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/analysis.py tests/
git commit -m "feat(analysis): artifact writers report the bytes they actually stored"
```

---

### Task 4: The per-file size cap

**Files:**
- Modify: `backend/app/routers/analyze.py` (`_stage_analyze_persist:103-165`; add `_as_mb` beside `_read_thumbnail`)
- Test: `tests/test_upload_limits.py`

**Interfaces:**
- Consumes: `settings.max_upload_bytes()` (Task 1).
- Produces: a `413` whose `detail` is `{"code": "upload_too_large", "limit_mb": <int>}`; helper `analyze._as_mb(value: int) -> int`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_upload_limits.py`. Add these imports at the top of the file:

```python
import asyncio
import io

from fastapi import HTTPException
from starlette.datastructures import UploadFile

from backend.app.routers import analyze as analyze_router
from backend.app.services import analysis as analysis_service
from backend.app.services import runtime_config
```

Then:

```python
_TINY_POSE = '{"metadata": {"fps": 30, "width": 1, "height": 1, "total_frames": 0}, "frames": []}'


class _StubbedAnalyzePath(unittest.TestCase):
    """Shared harness: the router runs, but staging/analysis/storage are stubs.

    ``analyze_pose`` is invoked directly rather than through FastAPI, so Form/File defaults are
    not resolved by DI — ``max_reps`` and ``thumbnail`` must be passed explicitly.
    """

    def setUp(self) -> None:
        from pathlib import Path

        self._orig = {
            name: getattr(analysis_service, name)
            for name in ("stage_upload", "store_artifacts", "discard_stage", "analyze_pose_payload")
        }
        self.staged = analysis_service.StagedUpload(
            video_id="upload_test",
            prefix="uploads/u1/upload_test",
            video_path=Path("upload_test.mp4"),
            pose_path=Path("pose.json"),
        )
        self.staged_calls: list[int] = []

        def _stage(data, *, suffix=".mp4", owner="anon"):
            self.staged_calls.append(len(data))
            return self.staged

        analysis_service.stage_upload = _stage
        analysis_service.store_artifacts = lambda staged, *, thumbnail=None: 0
        analysis_service.discard_stage = lambda staged: None
        analysis_service.analyze_pose_payload = (
            lambda payload, *, movement, video_id=None, max_reps=-1: {
                "video_id": video_id, "source": "upload", "movement": movement, "detections": [],
            }
        )

        presign = mock.patch.object(
            analyze_router, "_source_url", side_effect=lambda prefix: f"https://signed/{prefix}"
        )
        presign.start()
        self.addCleanup(presign.stop)

        # KEEP OFFLINE: settings getters read the admin overrides, and get_overrides() does a
        # REAL Supabase round trip whenever auth is configured. `{}` is what it returns when
        # auth is unconfigured, i.e. the same path CI takes.
        overrides = mock.patch.object(runtime_config, "get_overrides", return_value={})
        overrides.start()
        self.addCleanup(overrides.stop)

    def tearDown(self) -> None:
        for name, fn in self._orig.items():
            setattr(analysis_service, name, fn)

    def _run(self, data: bytes, user=None):
        return asyncio.run(
            analyze_router.analyze_pose(
                "Squat",
                _TINY_POSE,
                UploadFile(file=io.BytesIO(data), filename="clip.webm"),
                max_reps=None,
                thumbnail=None,
                user=user,
            )
        )


class PerFileSizeCapTests(_StubbedAnalyzePath):
    def test_an_upload_at_exactly_the_limit_is_accepted(self) -> None:
        """The boundary is tested from BOTH sides so an off-by-one cannot slip through."""
        limit = 1 * 1024 * 1024  # the clamped floor, so the test allocates 1 MB, not 100
        with mock.patch.object(runtime_config, "get_overrides",
                               return_value={"max_upload_bytes": limit}):
            self._run(b"x" * limit)
        self.assertEqual(self.staged_calls, [limit])

    def test_one_byte_over_the_limit_is_refused(self) -> None:
        limit = 1 * 1024 * 1024
        with mock.patch.object(runtime_config, "get_overrides",
                               return_value={"max_upload_bytes": limit}):
            with self.assertRaises(HTTPException) as ctx:
                self._run(b"x" * (limit + 1))
        self.assertEqual(ctx.exception.status_code, 413)
        self.assertEqual(ctx.exception.detail["code"], "upload_too_large")
        self.assertEqual(ctx.exception.detail["limit_mb"], 1)

    def test_an_oversized_upload_is_never_staged(self) -> None:
        """The whole point of checking before stage_upload: no object written, no CPU spent."""
        limit = 1 * 1024 * 1024
        with mock.patch.object(runtime_config, "get_overrides",
                               return_value={"max_upload_bytes": limit}):
            with self.assertRaises(HTTPException):
                self._run(b"x" * (limit + 1))
        self.assertEqual(self.staged_calls, [])

    def test_an_empty_upload_is_still_a_400(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            self._run(b"")
        self.assertEqual(ctx.exception.status_code, 400)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_upload_limits.py -k SizeCap -v`
Expected: FAIL — the oversized upload is accepted, so `assertRaises(HTTPException)` fails.

- [ ] **Step 3: Add the MB helper**

In `backend/app/routers/analyze.py`, directly after `MAX_THUMBNAIL_BYTES` (line 29):

```python
def _as_mb(value: int) -> int:
    """Bytes -> whole MB, rounded UP, for a user-facing limit message.

    Rounded up, not down, so a limit is never reported as a number SMALLER than the one actually
    enforced — telling a user their cap is 99 MB when it is 100 MB invites a support question.
    """
    return -(-value // (1024 * 1024))
```

- [ ] **Step 4: Cap the read**

In `_stage_analyze_persist`, replace:

```python
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
```

with:

```python
    # ``max_upload_bytes`` reads the admin overrides, which can do a synchronous Supabase round
    # trip on a cold cache — threadpool it so it never blocks the event loop, exactly as the
    # suffix check above already does.
    max_bytes = await run_in_threadpool(settings.max_upload_bytes)
    # Cap the READ itself, not just a check after it: an unbounded ``read()`` materialises the
    # whole clip as one bytes object before any size check could reject it. Reading one byte
    # past the limit is enough to detect "too large" without ever holding more than that — the
    # same technique ``_read_thumbnail`` uses for the thumbnail part.
    data = await file.read(max_bytes + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={"code": "upload_too_large", "limit_mb": _as_mb(max_bytes)},
        )
```

- [ ] **Step 5: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_upload_limits.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/routers/analyze.py tests/test_upload_limits.py
git commit -m "feat(analyze): cap the size of a single upload at read time"
```

---

### Task 5: The per-user quota gate, and recording the size

**Files:**
- Modify: `backend/app/routers/analyze.py` (`_stage_analyze_persist`)
- Test: `tests/test_upload_limits.py`

**Interfaces:**
- Consumes: `settings.user_storage_quota_bytes()` (Task 1), `store.get_storage_used(token=..., user_id=...)` (Task 2), `analysis.store_artifacts(...) -> int` (Task 3), `store.persist_analysis(..., size_bytes=...)` (Task 2), `analyze._as_mb` (Task 4).
- Produces: a `413` whose `detail` is `{"code": "storage_quota_exceeded", "used_mb": <int>, "limit_mb": <int>}`; a `503` when usage cannot be read.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_upload_limits.py`. Add `from backend.app.services import store as store_service` to the imports if `store` is not already bound in the router's namespace under that name — the router calls `store.get_storage_used`, so patch `analyze_router.store`'s attribute via `mock.patch.object(store, "get_storage_used", ...)`.

```python
class _User:
    """The router only reads ``.id`` and ``.token`` off the current user."""

    id = "u1"
    token = "jwt"


class StorageQuotaTests(_StubbedAnalyzePath):
    def test_an_anonymous_upload_ignores_the_quota(self) -> None:
        """Anonymous uploads have no videos row to count and are expired by the lifecycle rule."""
        with mock.patch.object(runtime_config, "get_overrides",
                               return_value={"user_storage_quota_bytes": 10 * 1024 * 1024}):
            with mock.patch.object(store, "get_storage_used") as used:
                self._run(b"x" * 1024, user=None)
        used.assert_not_called()
        self.assertEqual(self.staged_calls, [1024])

    def test_an_upload_that_fits_is_accepted(self) -> None:
        quota = 10 * 1024 * 1024
        with mock.patch.object(runtime_config, "get_overrides",
                               return_value={"user_storage_quota_bytes": quota}):
            with mock.patch.object(store, "get_storage_used", return_value=quota - 2048):
                with mock.patch.object(store, "persist_analysis", return_value="a1"):
                    self._run(b"x" * 1024, user=_User())
        self.assertEqual(self.staged_calls, [1024])

    def test_an_upload_that_would_exceed_the_quota_is_refused(self) -> None:
        quota = 10 * 1024 * 1024
        with mock.patch.object(runtime_config, "get_overrides",
                               return_value={"user_storage_quota_bytes": quota}):
            with mock.patch.object(store, "get_storage_used", return_value=quota - 512):
                with self.assertRaises(HTTPException) as ctx:
                    self._run(b"x" * 1024, user=_User())
        self.assertEqual(ctx.exception.status_code, 413)
        self.assertEqual(ctx.exception.detail["code"], "storage_quota_exceeded")
        self.assertEqual(ctx.exception.detail["limit_mb"], 10)
        self.assertEqual(self.staged_calls, [], "a refused upload must write no object")

    def test_a_user_exactly_at_the_quota_is_refused_the_next_upload(self) -> None:
        quota = 10 * 1024 * 1024
        with mock.patch.object(runtime_config, "get_overrides",
                               return_value={"user_storage_quota_bytes": quota}):
            with mock.patch.object(store, "get_storage_used", return_value=quota):
                with self.assertRaises(HTTPException) as ctx:
                    self._run(b"x", user=_User())
        self.assertEqual(ctx.exception.status_code, 413)

    def test_a_failing_usage_query_refuses_rather_than_passes(self) -> None:
        """Treating 'cannot determine usage' as 'under quota' would turn a DB hiccup into an
        unbounded write path — the exact thing this feature exists to prevent."""
        with mock.patch.object(store, "get_storage_used", side_effect=RuntimeError("db down")):
            with self.assertRaises(HTTPException) as ctx:
                self._run(b"x" * 1024, user=_User())
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(self.staged_calls, [])


class RecordedSizeTests(_StubbedAnalyzePath):
    def test_persists_source_plus_derived_bytes(self) -> None:
        analysis_service.store_artifacts = lambda staged, *, thumbnail=None: 300
        seen: list[dict] = []
        with mock.patch.object(store, "get_storage_used", return_value=0):
            with mock.patch.object(
                store, "persist_analysis", side_effect=lambda **kw: seen.append(kw) or "a1"
            ):
                self._run(b"x" * 1000, user=_User())
        self.assertEqual(seen[0]["size_bytes"], 1300)

    def test_records_only_what_stored_when_a_derived_put_failed(self) -> None:
        analysis_service.store_artifacts = lambda staged, *, thumbnail=None: 0
        seen: list[dict] = []
        with mock.patch.object(store, "get_storage_used", return_value=0):
            with mock.patch.object(
                store, "persist_analysis", side_effect=lambda **kw: seen.append(kw) or "a1"
            ):
                self._run(b"x" * 1000, user=_User())
        self.assertEqual(seen[0]["size_bytes"], 1000)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_upload_limits.py -k "Quota or RecordedSize" -v`
Expected: FAIL — no quota check exists, and `persist_analysis` is called without `size_bytes`.

- [ ] **Step 3: Add the quota gate**

In `_stage_analyze_persist`, after the size cap and before the `owner = ...` line, insert:

```python
    owner = user.id if user is not None else "anon"
    if user is not None:
        quota = await run_in_threadpool(settings.user_storage_quota_bytes)
        try:
            used = await run_in_threadpool(
                store.get_storage_used, token=user.token, user_id=user.id
            )
        except Exception as exc:  # noqa: BLE001 — see below; this must NOT fail open
            # Treating "cannot determine usage" as "under quota" would turn a database hiccup
            # into an unbounded write path, which is precisely what the quota exists to stop.
            # Refusing is the conservative direction and the caller can retry.
            logger.exception("Failed to read storage usage for %s", user.id)
            raise HTTPException(
                status_code=503, detail="Storage is unavailable; please try again."
            ) from exc
        if used + len(data) > quota:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "storage_quota_exceeded",
                    "used_mb": _as_mb(used),
                    "limit_mb": _as_mb(quota),
                },
            )
```

Delete the now-duplicated `owner = user.id if user is not None else "anon"` line that previously sat above the `try:` for `stage_upload`, keeping its comment attached to the new location:

```python
    # Anonymous demo uploads are still stored, under their own key prefix, so both paths behave
    # identically. A bucket lifecycle rule expires `uploads/anon/` — see the design doc.
```

- [ ] **Step 4: Carry the sizes through to `persist_analysis`**

Replace:

```python
    del data  # bytes are now stored and staged; don't pin the whole video in RAM while queued.
```

with:

```python
    # Captured BEFORE the del: this is the source's contribution to the recorded size, and the
    # quota is checked against it while the derived artifacts do not exist yet. The recorded
    # total therefore includes derived bytes the check did not see, so a user can finish
    # marginally over the limit — bounded by one upload, and the next upload is refused.
    source_size = len(data)
    del data  # bytes are now stored and staged; don't pin the whole video in RAM while queued.
    derived_size = 0
```

Then in the `else:` arm, capture the return:

```python
        derived_size = await run_in_threadpool(
            analysis.store_artifacts, staged, thumbnail=thumb
        )
```

And add the keyword to the `persist_analysis` call, after `storage_key=staged.prefix,`:

```python
                size_bytes=source_size + derived_size,
```

- [ ] **Step 5: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_upload_limits.py tests/test_analyze_endpoint.py tests/test_analyze_pose_endpoint.py -v`
Expected: PASS

- [ ] **Step 6: Run the whole backend suite and the coverage gate**

Run: `.venv\Scripts\python.exe -m pytest tests/`
Run: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
Expected: PASS, coverage ≥ 95%

- [ ] **Step 7: Commit**

```bash
git add backend/app/routers/analyze.py tests/test_upload_limits.py
git commit -m "feat(analyze): refuse an upload that would exceed the user's storage quota"
```

---

### Task 6: Localised 413 in the browser

**Files:**
- Modify: `frontend/src/api.ts` (`analyzeUpload:652`, `analyzePose:672`)
- Modify: `frontend/src/App.tsx` (`runPoseAnalysis:117-145`)
- Modify: `frontend/src/lib/i18n.tsx` (en block near `"upload.hint":188`; zh-TW block near the matching key ~line 858)
- Test: `frontend/src/test/api.test.ts`, `frontend/src/test/App.test.tsx`

**Interfaces:**
- Consumes: the two 413 detail shapes from Tasks 4 and 5.
- Produces: `UploadLimitError` exported from `frontend/src/api.ts`.

**Run every command below with cwd = `frontend/`.**

- [ ] **Step 1: Write the failing tests**

Add to `frontend/src/test/api.test.ts`:

```ts
import { api, UploadLimitError } from "../api";

describe("upload limit errors", () => {
  afterEach(() => vi.restoreAllMocks());

  it("turns a 413 quota body into a typed error", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 413,
      json: async () => ({ detail: { code: "storage_quota_exceeded", used_mb: 480, limit_mb: 500 } }),
    } as Response);
    const err = await api
      .analyzePose("Squat", { metadata: {}, frames: [] } as never, new Blob(["v"]))
      .catch((e) => e);
    expect(err).toBeInstanceOf(UploadLimitError);
    expect(err.code).toBe("storage_quota_exceeded");
    expect(err.usedMb).toBe(480);
    expect(err.limitMb).toBe(500);
  });

  it("turns a 413 file-size body into a typed error with no used figure", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 413,
      json: async () => ({ detail: { code: "upload_too_large", limit_mb: 100 } }),
    } as Response);
    const err = await api
      .analyzePose("Squat", { metadata: {}, frames: [] } as never, new Blob(["v"]))
      .catch((e) => e);
    expect(err).toBeInstanceOf(UploadLimitError);
    expect(err.code).toBe("upload_too_large");
    expect(err.usedMb).toBeNull();
  });

  it("leaves a non-413 failure as a plain Error", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({ detail: "boom" }),
    } as Response);
    const err = await api
      .analyzePose("Squat", { metadata: {}, frames: [] } as never, new Blob(["v"]))
      .catch((e) => e);
    expect(err).not.toBeInstanceOf(UploadLimitError);
    expect(err.message).toBe("boom");
  });
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `yarn vitest run src/test/api.test.ts`
Expected: FAIL — `UploadLimitError` is not exported.

- [ ] **Step 3: Add the typed error and the 413 branch**

In `frontend/src/api.ts`, above the `api` object:

```ts
export type UploadLimitCode = "upload_too_large" | "storage_quota_exceeded";

/**
 * A 413 from an analyze endpoint: the clip is over the per-file cap, or the user is out of
 * storage. Typed rather than a plain Error because the message must be LOCALISED at the call
 * site — this module has no access to the i18n `t()`, and the server's detail is English.
 */
export class UploadLimitError extends Error {
  constructor(
    readonly code: UploadLimitCode,
    readonly limitMb: number,
    readonly usedMb: number | null
  ) {
    super(code);
    this.name = "UploadLimitError";
  }
}

/** Parse an analyze failure body into an UploadLimitError, or null if it is not one. */
function uploadLimitError(status: number, body: unknown): UploadLimitError | null {
  if (status !== 413) return null;
  const detail = (body as { detail?: unknown })?.detail as
    | { code?: string; limit_mb?: number; used_mb?: number }
    | undefined;
  if (detail?.code !== "upload_too_large" && detail?.code !== "storage_quota_exceeded") {
    // A 413 we do not recognise (a proxy's own body, say) falls through to the generic path
    // rather than being mislabelled as a quota problem.
    return null;
  }
  return new UploadLimitError(
    detail.code,
    Number(detail.limit_mb ?? 0),
    detail.used_mb === undefined ? null : Number(detail.used_mb)
  );
}
```

Then in BOTH `analyzeUpload` and `analyzePose`, replace the failure block:

```ts
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      const limit = uploadLimitError(res.status, detail);
      if (limit) throw limit;
      throw new Error((detail as { detail?: string }).detail || `Analyze failed (${res.status})`);
    }
```

- [ ] **Step 4: Add the i18n keys**

In `frontend/src/lib/i18n.tsx`, in the **en** catalogue after `"upload.hint"`:

```ts
  "upload.tooLarge": "That clip is too large — the limit is {limit} MB. Try a shorter recording.",
  "upload.quotaFull": "Your storage is full ({used} MB of {limit} MB). Delete a saved analysis to make room.",
```

And in the **zh-TW** catalogue after its matching `"upload.hint"`:

```ts
  "upload.tooLarge": "這支影片太大了,上限是 {limit} MB。請改用較短的片段。",
  "upload.quotaFull": "儲存空間已滿({used} MB / {limit} MB)。請先刪除一些已存的分析來騰出空間。",
```

- [ ] **Step 5: Write the failing App test**

Add to `frontend/src/test/App.test.tsx`, following that file's existing render helper:

```tsx
  it("shows a localised message when the upload exceeds the storage quota", async () => {
    vi.spyOn(api, "analyzePose").mockRejectedValue(
      new UploadLimitError("storage_quota_exceeded", 500, 480)
    );
    // ...render the app and trigger runPoseAnalysis exactly as the neighbouring upload tests do
    expect(
      await screen.findByText(/Your storage is full \(480 MB of 500 MB\)/)
    ).toBeInTheDocument();
  });
```

- [ ] **Step 6: Map the error in App.tsx**

In `frontend/src/App.tsx`, import `UploadLimitError` alongside `api`, and add above `runPoseAnalysis`:

```tsx
  // The server's 413 detail is English and structured; the message the user reads is neither.
  const errorMessage = useCallback(
    (e: unknown): string => {
      if (e instanceof UploadLimitError) {
        return e.code === "upload_too_large"
          ? t("upload.tooLarge", { limit: e.limitMb })
          : t("upload.quotaFull", { used: e.usedMb ?? 0, limit: e.limitMb });
      }
      return e instanceof Error ? e.message : String(e);
    },
    [t]
  );
```

Then in `runPoseAnalysis`'s catch, replace
`setError(e instanceof Error ? e.message : String(e));`
with `setError(errorMessage(e));`, and add `errorMessage` to the `useCallback` dependency array (replacing nothing else in it).

- [ ] **Step 7: Run the frontend suite and build**

Run: `yarn test`
Run: `yarn build`
Expected: all tests pass; build clean.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/api.ts frontend/src/App.tsx frontend/src/lib/i18n.tsx frontend/src/test/
git commit -m "feat(frontend): show a localised message when an upload is refused for size or quota"
```

---

## Deployment prerequisites

These are not code and are NOT part of any task. Record them in the PR description.

1. Apply `db/migrations/20260802000000_video_size_bytes.sql` to Supabase (SQL Editor, or `psql "$SUPABASE_DB_URL" -f ...`).
2. Set a reverse-proxy request-body limit at or slightly above `max_upload_bytes`. Without it the application cap still protects memory and storage, but oversized bodies are transmitted and spooled to temp disk before being refused.
3. Optionally set `max_upload_bytes` / `user_storage_quota_bytes` overrides in the admin panel. The built-in defaults apply if unset.

## Self-review notes

- **Spec coverage.** Limits table → Task 1. Enforcement point and ordering → Tasks 4, 5. `size_bytes` migration → Task 2. Recording the size → Tasks 2, 3, 5. Usage query → Task 2. Deletion frees space → no code, verified by the existing reap path. Error table → Tasks 4, 5 (413s and the 503). Accepted imprecisions → documented in code comments in Task 5. Frontend → Task 6. Testing → every task. Deployment prerequisites → above.
- **Type consistency.** `store_artifacts` returns `int` (Task 3) and is consumed as `derived_size` (Task 5). `persist_analysis`'s `size_bytes: int` (Task 2) is supplied as `source_size + derived_size` (Task 5). `_as_mb` is defined in Task 4 and used in Task 5. `UploadLimitError`'s `code` / `limitMb` / `usedMb` (Task 6, api.ts) are read under exactly those names in App.tsx.
- **Known ordering hazard.** Task 3 Step 5 fixes six existing test stubs. Skipping it makes Task 5 fail with `TypeError: unsupported operand type(s) for +: 'int' and 'MagicMock'`, which reads like a router bug rather than a stale stub.
