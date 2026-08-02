# Cloudflare R2 Object Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move user-uploaded raw videos, pose JSON, and a new browser-generated thumbnail off the backend's local disk into Cloudflare R2, served to the browser as short-lived presigned URLs.

**Architecture:** A narrow `ObjectStore` interface (`put` / `presigned_url` / `delete_prefix`) with two implementations — `LocalObjectStore` (filesystem, the dev/CI default) and `R2ObjectStore` (boto3 against the S3-compatible R2 endpoint) — selected by whether the `R2_*` settings are present. The analysis pipeline still needs real filesystem paths (OpenCV, camera-view estimation), so uploads are put to the object store first and *staged* to a temp directory for the duration of the analysis. Reads become presigned URLs behind an ownership-checked endpoint, which also closes the existing unauthenticated `/api/video-file` IDOR.

**Tech Stack:** Python 3.11/3.12, FastAPI, boto3 (new), Supabase (postgrest, RLS), React 18 + TypeScript + Vite, vitest.

**Spec:** `docs/superpowers/specs/2026-08-02-r2-object-storage-design.md`

## Global Constraints

- **Python interpreter is always `.venv\Scripts\python.exe`** from the repo root. Never `python`, never `pip`, never `source .venv/bin/activate` (POSIX-only, fails on this machine).
- **Backend tests:** `.venv\Scripts\python.exe -m pytest tests/` — always scope to `tests/`.
- **Backend coverage gate (CI enforces 95%):** `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
- **All frontend commands run with cwd = `frontend/`.** The Bash and PowerShell tools share one cwd; a stray `cd` to the repo root mass-fails vitest.
- **Frontend tests:** `yarn test` (vitest run); coverage `yarn test:coverage`.
- Backend tests are `unittest.TestCase` classes under `tests/`. Frontend tests are vitest files under `frontend/src/test/`.
- Tests must never touch the network. `LocalObjectStore` is the default whenever `R2_*` is unset, and `R2ObjectStore` is only ever exercised against a patched fake boto3 client.
- Object keys are interpolated into filesystem paths by `LocalObjectStore`, so every key passes `_validate_key` first.

## Deviations from the spec (deliberate, decided during planning)

Three refinements. Each is an improvement on what the spec wrote; note them so a reviewer comparing plan to spec does not read them as drift.

1. **The source object is named `source` with no file extension.** The spec wrote `source{suffix}`. Storing the suffix would force the read path to know it (an extra column, or N `head_object` probes per read). R2 returns the `ContentType` set at put time, and `<video src>` plays by content type, not extension — so the extension buys nothing and costs a lookup.
2. **The dev-only `/api/local-object/{key}` endpoint is always registered but returns 404 unless `LocalObjectStore` is live.** The spec said "registered only when `storage_configured` is false". A runtime guard has the identical security property (inert in production), is testable without re-importing the app, and avoids import-time branching on settings.
3. **Thumbnails are accepted as `image/jpeg` only.** The spec allowed PNG too. The frontend produces JPEG via `canvas.toBlob(..., "image/jpeg", 0.8)`, so accepting PNG would mean the `thumb.jpg` key could hold a PNG. One producer, one format.

## Context an implementer needs

- **`api.analyzeUpload` (the `/api/analyze` server-side-MediaPipe path) has no caller in the app** — `grep -rn "analyzeUpload" frontend/src` returns only `api.ts` and test files. `App.tsx` always uses `api.analyzePose`. Both backend paths are still implemented and tested here, but only `/api/analyze/pose` is exercised by real users.
- **`frontend/src/lib/poseExtract.ts` exports `resolveDuration(video, timeoutMs)`.** A `MediaRecorder` WebM has no Duration element, so `video.duration` is `NaN` for every recorded clip. Naively computing `duration * 0.25` for the thumbnail frame would yield `NaN` on the app's primary path. The thumbnail utility MUST route through `resolveDuration`.
- **`frontend/src/lib/poseExtract.ts` uses `/* c8 ignore start */` around `<video>` glue** that jsdom cannot run, and keeps the decision logic in pure exported helpers that are unit-tested. The thumbnail utility follows the same split.
- The backend's `data/runtime/` is gitignored; `data/runtime/objects/` inherits that.

## File Structure

**Created:**
- `backend/app/services/storage.py` — the `ObjectStore` interface, both implementations, key builders, `StorageError`. The only module that knows R2 exists.
- `tests/test_storage.py` — storage unit tests.
- `tests/test_upload_staging.py` — `analysis.stage_upload` / `store_artifacts` / `discard_stage` tests.
- `tests/test_upload_urls.py` — read-path endpoint tests.
- `frontend/src/lib/thumbnail.ts` — browser frame capture.
- `frontend/src/test/lib.thumbnail.test.ts`
- `frontend/src/components/HistoryThumb.tsx` — one history row's thumbnail with fallback.

**Modified:**
- `backend/app/settings.py` — four `R2_*` fields + `storage_configured`.
- `backend/app/config.py` — delete `UPLOAD_DIR`, `UPLOAD_POSE_DIR`, `ensure_runtime_dirs`.
- `backend/app/main.py:83` — drop the `ensure_runtime_dirs` startup hook.
- `backend/app/services/analysis.py` — `StagedUpload` + staging functions; `analyze_video_file` / `analyze_pose_payload` take an explicit `pose_json_path`; `save_upload` deleted.
- `backend/app/routers/analyze.py` — thumbnail form field, staged flow, 503 mapping, `video_url` on the response only.
- `backend/app/routers/videos.py` — new URL endpoints + the dev local-object endpoint; `/api/video-file` narrowed to library clips.
- `backend/app/services/library.py:82-95` — delete `uploaded_video_path`.
- `backend/app/services/store.py` — `storage_key` becomes the R2 prefix; new `get_storage_key` / `get_storage_keys`; deletion reaps objects.
- `frontend/src/api.ts` — `video_url` on `Analysis`, `UploadMedia` type, `uploadMedia` / `uploadMediaBatch`, thumbnail arg on both analyze calls.
- `frontend/src/App.tsx:116-125` — capture and pass the thumbnail.
- `frontend/src/components/VideoPanel.tsx:101-107` — async video src resolution.
- `frontend/src/pages/History.tsx:192-194` — thumbnail in place of the static icon.
- `.env.example`, `backend/README.md`, `requirements.txt`, `requirements-ci.txt`.

---

### Task 1: Object storage service

Delivers a working object store end to end (put → URL → fetch) that nothing uses yet.

**Files:**
- Create: `backend/app/services/storage.py`
- Create: `tests/test_storage.py`
- Modify: `backend/app/settings.py` (add fields after `supabase_service_role_key`, add property after `chat_configured`)
- Modify: `backend/app/routers/videos.py` (add the dev endpoint)
- Modify: `requirements.txt`, `requirements-ci.txt`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `storage.StorageError` (subclass of `RuntimeError`)
  - `storage.ObjectStore` protocol: `put(key: str, data: bytes, *, content_type: str) -> None`, `presigned_url(key: str, *, expires_in: int = DEFAULT_URL_TTL) -> str`, `delete_prefix(prefix: str) -> None`
  - `storage.LocalObjectStore(root: Path | None = None)` with the extra `open_object(key: str) -> tuple[Path, str] | None`
  - `storage.R2ObjectStore(*, account_id, access_key_id, secret_access_key, bucket)`
  - `storage.get_object_store() -> ObjectStore` (lru_cached)
  - `storage.upload_prefix(owner: str, video_id: str) -> str`
  - `storage.video_content_type(suffix: str) -> str`
  - `storage.DEFAULT_URL_TTL: int` (3600), `storage.OBJECTS_DIR: Path`
  - `Settings.r2_account_id / r2_access_key_id / r2_secret_access_key / r2_bucket: str`, `Settings.storage_configured: bool`

- [ ] **Step 1: Add the boto3 dependency**

In `requirements.txt`, add below the `supabase` line:

```
boto3>=1.34,<2   # Cloudflare R2 over the S3 API (backend/app/services/storage.py)
```

In `requirements-ci.txt`, add the same line (the backend suite imports `storage`, and `R2ObjectStore`'s tests patch a fake client — but `get_object_store` must still import cleanly).

Install it:

```
.venv\Scripts\python.exe -m pip install "boto3>=1.34,<2"
```

- [ ] **Step 2: Write the failing storage tests**

Create `tests/test_storage.py`:

```python
"""Tests for the object-storage service (local + R2 implementations).

Fully offline: the local store writes into a tmp dir, and the R2 store is exercised against a
hand-rolled fake client so no credentials, network, or boto3 behaviour is required.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend.app.services import storage


class KeyValidationTests(unittest.TestCase):
    def test_rejects_traversal_and_absolute_keys(self) -> None:
        store = storage.LocalObjectStore(Path(tempfile.mkdtemp()))
        for bad in ("", "/etc/passwd", "uploads/../../etc/passwd", "uploads/\x00/x", "uploads\\..\\x"):
            with self.subTest(key=bad), self.assertRaises(storage.StorageError):
                store.put(bad, b"x", content_type="text/plain")

    def test_rejects_windows_drive_anchored_keys(self) -> None:
        """`Path(root) / "C:/x"` is `WindowsPath("C:/x")` — pathlib drops the root for a
        drive-anchored operand, so an unguarded key of this shape escapes the store entirely."""
        store = storage.LocalObjectStore(Path(tempfile.mkdtemp()))
        for bad in ("C:/Windows/System32/evil", "D:evil", "uploads/C:/x"):
            with self.subTest(key=bad), self.assertRaises(storage.StorageError):
                store.put(bad, b"x", content_type="text/plain")

    def test_accepts_a_normal_upload_key(self) -> None:
        root = Path(tempfile.mkdtemp())
        store = storage.LocalObjectStore(root)
        store.put("uploads/u1/upload_abc/source", b"x", content_type="video/mp4")
        self.assertTrue((root / "uploads/u1/upload_abc/source").is_file())


class UploadPrefixTests(unittest.TestCase):
    def test_builds_owner_scoped_prefix(self) -> None:
        self.assertEqual(storage.upload_prefix("u1", "upload_abc"), "uploads/u1/upload_abc")

    def test_anonymous_owner(self) -> None:
        self.assertEqual(storage.upload_prefix("anon", "upload_abc"), "uploads/anon/upload_abc")


class VideoContentTypeTests(unittest.TestCase):
    def test_known_suffixes(self) -> None:
        self.assertEqual(storage.video_content_type(".mp4"), "video/mp4")
        self.assertEqual(storage.video_content_type(".webm"), "video/webm")
        self.assertEqual(storage.video_content_type(".mov"), "video/quicktime")

    def test_unknown_suffix_falls_back_to_mp4(self) -> None:
        self.assertEqual(storage.video_content_type(".xyz"), "video/mp4")


class LocalObjectStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.store = storage.LocalObjectStore(self.root)

    def test_round_trip_preserves_bytes_and_content_type(self) -> None:
        self.store.put("uploads/u1/v1/source", b"video-bytes", content_type="video/webm")
        found = self.store.open_object("uploads/u1/v1/source")
        self.assertIsNotNone(found)
        path, ctype = found
        self.assertEqual(path.read_bytes(), b"video-bytes")
        self.assertEqual(ctype, "video/webm")

    def test_open_object_returns_none_for_a_missing_key(self) -> None:
        self.assertIsNone(self.store.open_object("uploads/u1/v1/nope"))

    def test_presigned_url_points_at_the_dev_endpoint(self) -> None:
        url = self.store.presigned_url("uploads/u1/v1/source")
        self.assertEqual(url, "/api/local-object/uploads/u1/v1/source")

    def test_delete_prefix_removes_every_object_under_it(self) -> None:
        self.store.put("uploads/u1/v1/source", b"a", content_type="video/mp4")
        self.store.put("uploads/u1/v1/pose.json", b"{}", content_type="application/json")
        self.store.delete_prefix("uploads/u1/v1")
        self.assertFalse((self.root / "uploads/u1/v1").exists())

    def test_delete_prefix_is_a_noop_when_absent(self) -> None:
        self.store.delete_prefix("uploads/u1/gone")  # must not raise

    def test_delete_prefix_does_not_touch_a_sibling_with_a_shared_stem(self) -> None:
        """`uploads/u1/v1` must not reap `uploads/u1/v10`."""
        self.store.put("uploads/u1/v1/source", b"a", content_type="video/mp4")
        self.store.put("uploads/u1/v10/source", b"b", content_type="video/mp4")
        self.store.delete_prefix("uploads/u1/v1")
        self.assertTrue((self.root / "uploads/u1/v10/source").is_file())


class FakeS3Client:
    """Minimal stand-in for the boto3 S3 client surface ``R2ObjectStore`` touches."""

    def __init__(self) -> None:
        self.puts: list[dict] = []
        self.presigns: list[dict] = []
        self.deleted: list[list[dict]] = []
        self.pages: list[dict] = [{"Contents": []}]
        self.raise_on_put = False
        # Per-key delete failures, which the real API reports in the response body rather than
        # by raising. Empty means every delete succeeded.
        self.delete_errors: list[dict] = []

    def put_object(self, **kwargs):
        if self.raise_on_put:
            raise ValueError("boom")
        self.puts.append(kwargs)

    def generate_presigned_url(self, operation, *, Params, ExpiresIn):
        self.presigns.append({"operation": operation, "Params": Params, "ExpiresIn": ExpiresIn})
        return f"https://signed/{Params['Key']}?exp={ExpiresIn}"

    def get_paginator(self, name):
        pages = self.pages

        class _Paginator:
            def paginate(self, **kwargs):
                self.kwargs = kwargs
                return pages

        self._paginator = _Paginator()
        return self._paginator

    def delete_objects(self, *, Bucket, Delete):
        self.deleted.append(Delete["Objects"])
        return {"Deleted": Delete["Objects"], "Errors": self.delete_errors}


class R2ObjectStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeS3Client()
        self.store = storage.R2ObjectStore(
            account_id="acc", access_key_id="k", secret_access_key="s", bucket="b"
        )
        self.store._client = self.fake  # bypass the deferred boto3 build

    def test_put_sends_bucket_key_body_and_content_type(self) -> None:
        self.store.put("uploads/u1/v1/source", b"bytes", content_type="video/mp4")
        self.assertEqual(
            self.fake.puts[0],
            {
                "Bucket": "b",
                "Key": "uploads/u1/v1/source",
                "Body": b"bytes",
                "ContentType": "video/mp4",
            },
        )

    def test_put_wraps_client_failure_in_storage_error(self) -> None:
        self.fake.raise_on_put = True
        with self.assertRaises(storage.StorageError):
            self.store.put("uploads/u1/v1/source", b"bytes", content_type="video/mp4")

    def test_presigned_url_requests_get_object_with_the_ttl(self) -> None:
        url = self.store.presigned_url("uploads/u1/v1/source", expires_in=120)
        self.assertEqual(self.fake.presigns[0]["operation"], "get_object")
        self.assertEqual(self.fake.presigns[0]["ExpiresIn"], 120)
        self.assertIn("uploads/u1/v1/source", url)

    def test_presigned_url_default_ttl(self) -> None:
        self.store.presigned_url("uploads/u1/v1/source")
        self.assertEqual(self.fake.presigns[0]["ExpiresIn"], storage.DEFAULT_URL_TTL)

    def test_delete_prefix_lists_with_a_trailing_slash_and_deletes_every_key(self) -> None:
        """The trailing slash is what stops `.../v1` from also reaping `.../v10`."""
        self.fake.pages = [{"Contents": [{"Key": "uploads/u1/v1/source"}, {"Key": "uploads/u1/v1/pose.json"}]}]
        self.store.delete_prefix("uploads/u1/v1")
        self.assertEqual(self.fake._paginator.kwargs["Prefix"], "uploads/u1/v1/")
        self.assertEqual(
            self.fake.deleted[0],
            [{"Key": "uploads/u1/v1/source"}, {"Key": "uploads/u1/v1/pose.json"}],
        )

    def test_delete_prefix_skips_the_delete_call_when_nothing_matches(self) -> None:
        self.fake.pages = [{"Contents": []}]
        self.store.delete_prefix("uploads/u1/v1")
        self.assertEqual(self.fake.deleted, [])

    def test_delete_prefix_raises_when_a_key_fails_to_delete(self) -> None:
        """`delete_objects` reports per-key failures in the body and raises only for whole-request
        errors, so a discarded return value would report a reap that deleted nothing."""
        self.fake.pages = [{"Contents": [{"Key": "uploads/u1/v1/source"}]}]
        self.fake.delete_errors = [{"Key": "uploads/u1/v1/source", "Code": "AccessDenied"}]
        with self.assertRaises(storage.StorageError):
            self.store.delete_prefix("uploads/u1/v1")


class GetObjectStoreTests(unittest.TestCase):
    def tearDown(self) -> None:
        storage.get_object_store.cache_clear()

    def _settings(self, **kwargs):
        base = {
            "r2_account_id": "",
            "r2_access_key_id": "",
            "r2_secret_access_key": "",
            "r2_bucket": "",
        }
        base.update(kwargs)
        base["storage_configured"] = all(base.values())
        return mock.Mock(**base)

    def test_falls_back_to_local_when_unconfigured(self) -> None:
        storage.get_object_store.cache_clear()
        with mock.patch("backend.app.settings.get_settings", return_value=self._settings()):
            self.assertIsInstance(storage.get_object_store(), storage.LocalObjectStore)

    def test_uses_r2_when_every_setting_is_present(self) -> None:
        storage.get_object_store.cache_clear()
        configured = self._settings(
            r2_account_id="acc", r2_access_key_id="k", r2_secret_access_key="s", r2_bucket="b"
        )
        with mock.patch("backend.app.settings.get_settings", return_value=configured):
            self.assertIsInstance(storage.get_object_store(), storage.R2ObjectStore)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_storage.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'backend.app.services.storage'`

- [ ] **Step 4: Add the R2 settings**

In `backend/app/settings.py`, add these fields to `Settings` immediately after `supabase_service_role_key: str = ""`:

```python
    # Cloudflare R2 object storage for user uploads (raw video, pose JSON, thumbnail), reached
    # over the S3-compatible API. Leave any of these blank and the backend transparently uses the
    # local-filesystem store instead (backend/app/services/storage.py) — which is what CI and
    # offline development run on, so no credentials are needed to work on this codebase.
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = ""
```

And this property immediately after the `chat_configured` property:

```python
    @property
    def storage_configured(self) -> bool:
        """True when R2 is fully configured; otherwise the local filesystem store is used."""
        return bool(
            self.r2_account_id
            and self.r2_access_key_id
            and self.r2_secret_access_key
            and self.r2_bucket
        )
```

- [ ] **Step 5: Write the storage service**

Create `backend/app/services/storage.py`:

```python
"""Object storage for user-uploaded artifacts (raw video, pose JSON, thumbnail).

Two backends behind one narrow interface, so no caller ever learns which one is live:

* ``LocalObjectStore`` — plain files under ``data/runtime/objects/``. The default, and what CI
  and offline development run on: no credentials, no network.
* ``R2ObjectStore`` — Cloudflare R2 over the S3 API (boto3), used only when every ``R2_*``
  setting is present.

Key layout (see docs/superpowers/specs/2026-08-02-r2-object-storage-design.md)::

    uploads/{owner}/{video_id}/source      the raw upload; ContentType carries the format
    uploads/{owner}/{video_id}/pose.json   the pose JSON, when the analysis produced one
    uploads/{owner}/{video_id}/thumb.jpg   one browser-captured frame

``owner`` is the authenticated user's id, or the literal ``anon`` for a demo upload. The source
object deliberately carries NO file extension: R2 replays the ``ContentType`` set at put time and
``<video src>`` plays by content type, so an extension would only force the read path to discover
which suffix was used.
"""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

from backend.app import config

# Where LocalObjectStore keeps its objects. Under data/runtime/, which is gitignored.
OBJECTS_DIR = config.RUNTIME_DIR / "objects"

# How long a presigned playback URL stays valid. One hour: long enough for a coaching session,
# short enough that a leaked URL is not a durable capability.
DEFAULT_URL_TTL = 3600

# Suffixes mimetypes gets wrong or does not know, and the fallback for anything unrecognised.
# Guessing wrong here means a clip that will not play, so the map is explicit rather than clever.
_VIDEO_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}
_DEFAULT_VIDEO_CONTENT_TYPE = "video/mp4"

# A key is interpolated into a filesystem path by LocalObjectStore, so backslashes (a Windows
# separator) and NUL are as dangerous as a traversal segment.
_UNSAFE_KEY_CHARS = frozenset("\\\x00")


class StorageError(RuntimeError):
    """A put / presign / delete against the object store failed, or a key was unsafe."""


def _validate_key(key: str) -> str:
    """Return ``key`` if it is safe to interpolate into a path or an S3 key, else raise."""
    if not key or key.startswith("/") or ".." in key.split("/"):
        raise StorageError(f"Unsafe object key: {key!r}")
    if any(ch in _UNSAFE_KEY_CHARS for ch in key):
        raise StorageError(f"Unsafe object key: {key!r}")
    # A DRIVE-ANCHORED key ("C:/Windows/System32/x", or the drive-relative "D:x") escapes the
    # store entirely: pathlib's `/` DISCARDS the left operand when the right one carries a drive,
    # so `OBJECTS_DIR / key` returns the caller's absolute path rather than anything under the
    # store. `Path("C:/tmp/objects") / "C:/Windows/x"` is `WindowsPath("C:/Windows/x")` — verified,
    # not theoretical. The dev serving endpoint has no auth, so on Windows (this project's primary
    # OS) that is an unauthenticated arbitrary file read. A colon never appears in a legitimate
    # key, and rejecting it covers both forms on every platform — which matters because a POSIX
    # CI box can mint a key that only escapes once a Windows box resolves it.
    if ":" in key:
        raise StorageError(f"Unsafe object key: {key!r}")
    return key


def upload_prefix(owner: str, video_id: str) -> str:
    """The key prefix holding every artifact for one upload. Stored as ``videos.storage_key``."""
    return f"uploads/{owner}/{video_id}"


def video_content_type(suffix: str) -> str:
    """Map an upload's file suffix to the content type R2 should replay it with."""
    return _VIDEO_CONTENT_TYPES.get(suffix.lower(), _DEFAULT_VIDEO_CONTENT_TYPE)


class ObjectStore(Protocol):
    """What the rest of the backend is allowed to ask of object storage."""

    def put(self, key: str, data: bytes, *, content_type: str) -> None: ...

    def presigned_url(self, key: str, *, expires_in: int = DEFAULT_URL_TTL) -> str: ...

    def delete_prefix(self, prefix: str) -> None: ...


class LocalObjectStore:
    """Filesystem-backed store for development and tests."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root if root is not None else OBJECTS_DIR

    def _path(self, key: str) -> Path:
        return self._root / _validate_key(key)

    def put(self, key: str, data: bytes, *, content_type: str) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        # Remember the content type alongside the object. Unlike R2 the filesystem stores no
        # metadata, and the source object has no extension to guess from — so without this the
        # dev serving endpoint could not tell a WebM recording from an MP4 upload.
        dest.with_name(dest.name + ".type").write_text(content_type, encoding="utf-8")

    def presigned_url(self, key: str, *, expires_in: int = DEFAULT_URL_TTL) -> str:
        """A URL for the development serving endpoint. Unsigned, and ``expires_in`` is ignored:
        that endpoint is inert whenever R2 is configured, so there is nothing to sign against."""
        return f"/api/local-object/{quote(_validate_key(key))}"

    def delete_prefix(self, prefix: str) -> None:
        shutil.rmtree(self._path(prefix), ignore_errors=True)

    def open_object(self, key: str) -> tuple[Path, str] | None:
        """Local-only: ``(path, content_type)`` behind ``key``, for the dev serving endpoint."""
        path = self._path(key)
        if not path.is_file():
            return None
        meta = path.with_name(path.name + ".type")
        content_type = (
            meta.read_text(encoding="utf-8").strip() if meta.is_file() else "application/octet-stream"
        )
        return path, content_type


class R2ObjectStore:
    """Cloudflare R2 over the S3 API."""

    def __init__(
        self, *, account_id: str, access_key_id: str, secret_access_key: str, bucket: str
    ) -> None:
        self._bucket = bucket
        self._endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._client = None

    def _s3(self):
        """Build the boto3 client on first use. Deferred so importing this module stays light
        and the unit tests can inject a fake by assigning ``_client``."""
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint,
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._secret_access_key,
                region_name="auto",
                config=Config(signature_version="s3v4"),
            )
        return self._client

    def put(self, key: str, data: bytes, *, content_type: str) -> None:
        key = _validate_key(key)
        try:
            self._s3().put_object(
                Bucket=self._bucket, Key=key, Body=data, ContentType=content_type
            )
        except Exception as exc:  # noqa: BLE001 — botocore raises a wide family; callers see one type
            raise StorageError(f"Failed to store object {key!r}.") from exc

    def presigned_url(self, key: str, *, expires_in: int = DEFAULT_URL_TTL) -> str:
        key = _validate_key(key)
        try:
            return self._s3().generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Failed to sign a URL for {key!r}.") from exc

    def delete_prefix(self, prefix: str) -> None:
        prefix = _validate_key(prefix)
        failures: list[dict] = []
        try:
            client = self._s3()
            paginator = client.get_paginator("list_objects_v2")
            # The TRAILING SLASH is load-bearing: without it, deleting `.../upload_ab` would also
            # reap `.../upload_abc`, because S3 prefixes match on raw string prefix, not path segments.
            for page in paginator.paginate(Bucket=self._bucket, Prefix=f"{prefix}/"):
                keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
                if keys:
                    resp = client.delete_objects(Bucket=self._bucket, Delete={"Objects": keys})
                    failures.extend(resp.get("Errors") or [])
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Failed to delete objects under {prefix!r}.") from exc
        # `delete_objects` reports PER-KEY failures in its response body and raises only for
        # whole-request errors, so discarding the return value would report a successful reap that
        # deleted nothing. Raised OUTSIDE the try so it is not caught and re-wrapped as its own cause.
        if failures:
            raise StorageError(
                f"Failed to delete {len(failures)} object(s) under {prefix!r}; "
                f"first: {failures[0].get('Key')} ({failures[0].get('Code')})"
            )


@lru_cache(maxsize=1)
def get_object_store() -> ObjectStore:
    """The live object store: R2 when fully configured, the local filesystem otherwise.

    Cached, so the boto3 client is built once. Tests that change the settings must call
    ``get_object_store.cache_clear()``.
    """
    from backend.app.settings import get_settings  # deferred: settings imports config, not us

    settings = get_settings()
    if settings.storage_configured:
        return R2ObjectStore(
            account_id=settings.r2_account_id,
            access_key_id=settings.r2_access_key_id,
            secret_access_key=settings.r2_secret_access_key,
            bucket=settings.r2_bucket,
        )
    return LocalObjectStore()
```

- [ ] **Step 6: Run the storage tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_storage.py -v`
Expected: PASS (all tests)

- [ ] **Step 7: Add the development serving endpoint**

In `backend/app/routers/videos.py`, add to the imports:

```python
from backend.app.services import analysis, library, storage
```

and append this endpoint:

```python
@router.get("/local-object/{key:path}")
def get_local_object(key: str) -> FileResponse:
    """DEVELOPMENT ONLY: serve an object out of the local filesystem store.

    Inert in production: when R2 is configured, ``get_object_store()`` returns an
    ``R2ObjectStore`` and this endpoint 404s for every key. It exists so ``LocalObjectStore``
    can hand back a URL the browser can actually fetch, keeping the frontend contract identical
    in both modes. It carries no signature — reaching a key still requires having been given it
    by the ownership-checked ``/api/uploads/{video_id}/url``.
    """
    # Named `store_`: Task 4 adds a `store` service import to this module, and the trailing
    # underscore keeps this local from shadowing it later.
    store_ = storage.get_object_store()
    if not isinstance(store_, storage.LocalObjectStore):
        raise HTTPException(status_code=404, detail="Not found.")
    try:
        found = store_.open_object(key)
    except storage.StorageError as exc:
        raise HTTPException(status_code=404, detail="Not found.") from exc
    if found is None:
        raise HTTPException(status_code=404, detail="Not found.")
    path, content_type = found
    return FileResponse(path, media_type=content_type)
```

- [ ] **Step 8: Write the failing dev-endpoint tests**

Append to `tests/test_storage.py` (before the `if __name__` block):

```python
class LocalObjectEndpointTests(unittest.TestCase):
    """The development serving endpoint: reachable locally, inert once R2 is configured."""

    def tearDown(self) -> None:
        storage.get_object_store.cache_clear()

    def test_serves_a_local_object_with_its_stored_content_type(self) -> None:
        from backend.app.routers import videos as videos_router

        root = Path(tempfile.mkdtemp())
        local = storage.LocalObjectStore(root)
        local.put("uploads/u1/v1/source", b"bytes", content_type="video/webm")
        with mock.patch.object(storage, "get_object_store", return_value=local):
            response = videos_router.get_local_object("uploads/u1/v1/source")
        self.assertEqual(response.media_type, "video/webm")

    def test_404s_for_a_missing_key(self) -> None:
        from fastapi import HTTPException

        from backend.app.routers import videos as videos_router

        local = storage.LocalObjectStore(Path(tempfile.mkdtemp()))
        with mock.patch.object(storage, "get_object_store", return_value=local):
            with self.assertRaises(HTTPException) as ctx:
                videos_router.get_local_object("uploads/u1/v1/nope")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_404s_for_an_unsafe_key(self) -> None:
        from fastapi import HTTPException

        from backend.app.routers import videos as videos_router

        local = storage.LocalObjectStore(Path(tempfile.mkdtemp()))
        with mock.patch.object(storage, "get_object_store", return_value=local):
            with self.assertRaises(HTTPException) as ctx:
                videos_router.get_local_object("uploads/../../etc/passwd")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_is_inert_when_r2_is_live(self) -> None:
        from fastapi import HTTPException

        from backend.app.routers import videos as videos_router

        r2 = storage.R2ObjectStore(account_id="a", access_key_id="k", secret_access_key="s", bucket="b")
        with mock.patch.object(storage, "get_object_store", return_value=r2):
            with self.assertRaises(HTTPException) as ctx:
                videos_router.get_local_object("uploads/u1/v1/source")
        self.assertEqual(ctx.exception.status_code, 404)
```

- [ ] **Step 9: Run the full storage suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_storage.py -v`
Expected: PASS (all tests, including the four endpoint tests)

- [ ] **Step 10: Run the whole backend suite to confirm nothing regressed**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS — no existing test touches `storage`, so this is a no-regression check.

- [ ] **Step 11: Commit**

```bash
git add backend/app/services/storage.py backend/app/settings.py backend/app/routers/videos.py tests/test_storage.py requirements.txt requirements-ci.txt
git commit -m "feat(storage): add an object-store service with local and R2 backends"
```

---

### Task 2: Upload staging in the analysis service

Replaces the disk-as-system-of-record with "put to object storage, stage a temp copy for the pipeline".

**Files:**
- Modify: `backend/app/services/analysis.py` (delete `save_upload` at 194-201; change `analyze_video_file` 66-107 and `analyze_pose_payload` 155-191; add the staging block)
- Modify: `backend/app/config.py:35-38, 60-63` (delete the upload dirs and `ensure_runtime_dirs`)
- Modify: `backend/app/main.py:81-83` (delete the startup hook)
- Create: `tests/test_upload_staging.py`

**Interfaces:**
- Consumes: `storage.get_object_store`, `storage.upload_prefix`, `storage.video_content_type`, `storage.StorageError` (Task 1).
- Produces:
  - `analysis.StagedUpload` — frozen dataclass with fields `video_id: str`, `prefix: str`, `video_path: Path`, `pose_path: Path`
  - `analysis.stage_upload(data: bytes, *, suffix: str, owner: str) -> StagedUpload` — raises `storage.StorageError`
  - `analysis.store_artifacts(staged: StagedUpload, *, thumbnail: bytes | None = None) -> None` — never raises
  - `analysis.discard_stage(staged: StagedUpload) -> None` — never raises
  - `analyze_video_file(source_path, *, video_id, pose_json_path, movement=None, max_reps=-1)`
  - `analyze_pose_payload(payload, *, movement, video_id, pose_json_path, max_reps=-1)`

- [ ] **Step 1: Write the failing staging tests**

Create `tests/test_upload_staging.py`:

```python
"""Tests for upload staging: object-store put first, temp files for the pipeline, then cleanup.

Offline throughout — the object store is a ``LocalObjectStore`` rooted in a tmp dir, or a stub
that raises, so no R2 credentials or network are involved.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend.app.services import analysis, storage


class _FailingStore:
    """An object store whose every put fails, for the fail-fast path."""

    def put(self, key, data, *, content_type):
        raise storage.StorageError("nope")

    def presigned_url(self, key, *, expires_in=storage.DEFAULT_URL_TTL):
        raise storage.StorageError("nope")

    def delete_prefix(self, prefix):
        raise storage.StorageError("nope")


class StageUploadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.store = storage.LocalObjectStore(self.root)
        patcher = mock.patch.object(storage, "get_object_store", return_value=self.store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_puts_the_source_video_under_an_owner_scoped_prefix(self) -> None:
        staged = analysis.stage_upload(b"video-bytes", suffix=".mp4", owner="u1")
        self.addCleanup(analysis.discard_stage, staged)
        self.assertEqual(staged.prefix, f"uploads/u1/{staged.video_id}")
        found = self.store.open_object(f"{staged.prefix}/source")
        self.assertIsNotNone(found)
        self.assertEqual(found[0].read_bytes(), b"video-bytes")

    def test_source_content_type_follows_the_suffix(self) -> None:
        staged = analysis.stage_upload(b"v", suffix=".webm", owner="anon")
        self.addCleanup(analysis.discard_stage, staged)
        self.assertEqual(self.store.open_object(f"{staged.prefix}/source")[1], "video/webm")

    def test_video_id_is_an_upload_slug(self) -> None:
        staged = analysis.stage_upload(b"v", suffix=".mp4", owner="u1")
        self.addCleanup(analysis.discard_stage, staged)
        self.assertTrue(staged.video_id.startswith("upload_"))

    def test_video_ids_are_unique_across_uploads(self) -> None:
        """Two uploads must never collide onto one key prefix — one would overwrite the other."""
        first = analysis.stage_upload(b"v", suffix=".mp4", owner="u1")
        self.addCleanup(analysis.discard_stage, first)
        second = analysis.stage_upload(b"v", suffix=".mp4", owner="u1")
        self.addCleanup(analysis.discard_stage, second)
        self.assertNotEqual(first.video_id, second.video_id)

    def test_stages_a_temp_copy_the_pipeline_can_open(self) -> None:
        staged = analysis.stage_upload(b"video-bytes", suffix=".mp4", owner="u1")
        self.addCleanup(analysis.discard_stage, staged)
        self.assertTrue(staged.video_path.is_file())
        self.assertEqual(staged.video_path.read_bytes(), b"video-bytes")
        self.assertEqual(staged.video_path.suffix, ".mp4")

    def test_pose_path_is_in_the_same_temp_dir_and_not_yet_written(self) -> None:
        staged = analysis.stage_upload(b"v", suffix=".mp4", owner="u1")
        self.addCleanup(analysis.discard_stage, staged)
        self.assertEqual(staged.pose_path.parent, staged.video_path.parent)
        self.assertFalse(staged.pose_path.exists())

    def test_a_failed_source_put_raises_before_any_temp_file_is_written(self) -> None:
        """Fail fast: the video is the one artifact that cannot be recomputed."""
        with mock.patch.object(storage, "get_object_store", return_value=_FailingStore()):
            with self.assertRaises(storage.StorageError):
                analysis.stage_upload(b"v", suffix=".mp4", owner="u1")


class StoreArtifactsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.store = storage.LocalObjectStore(self.root)
        patcher = mock.patch.object(storage, "get_object_store", return_value=self.store)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.staged = analysis.stage_upload(b"v", suffix=".mp4", owner="u1")
        self.addCleanup(analysis.discard_stage, self.staged)

    def test_uploads_pose_json_when_the_analysis_produced_one(self) -> None:
        self.staged.pose_path.write_text(json.dumps({"frames": []}), encoding="utf-8")
        analysis.store_artifacts(self.staged)
        found = self.store.open_object(f"{self.staged.prefix}/pose.json")
        self.assertIsNotNone(found)
        self.assertEqual(found[1], "application/json")

    def test_skips_pose_json_when_the_analysis_never_wrote_one(self) -> None:
        """`analyze_pose_payload` returns the analysis_pending skeleton without writing pose
        JSON for any movement with no registered detector — which is most of the registry."""
        analysis.store_artifacts(self.staged)
        self.assertIsNone(self.store.open_object(f"{self.staged.prefix}/pose.json"))

    def test_uploads_the_thumbnail_as_jpeg(self) -> None:
        analysis.store_artifacts(self.staged, thumbnail=b"jpeg-bytes")
        found = self.store.open_object(f"{self.staged.prefix}/thumb.jpg")
        self.assertIsNotNone(found)
        self.assertEqual(found[0].read_bytes(), b"jpeg-bytes")
        self.assertEqual(found[1], "image/jpeg")

    def test_skips_the_thumbnail_when_absent(self) -> None:
        analysis.store_artifacts(self.staged, thumbnail=None)
        self.assertIsNone(self.store.open_object(f"{self.staged.prefix}/thumb.jpg"))

    def test_never_raises_when_a_derived_put_fails(self) -> None:
        """A storage hiccup must not discard an already-completed (expensive) analysis."""
        self.staged.pose_path.write_text("{}", encoding="utf-8")
        with mock.patch.object(storage, "get_object_store", return_value=_FailingStore()):
            analysis.store_artifacts(self.staged, thumbnail=b"jpeg")  # must not raise

    def test_never_raises_when_the_staged_pose_file_cannot_be_read(self) -> None:
        """`is_file()` and the read are not atomic, and the read raises OSError — which a
        StorageError-only guard would let escape."""
        self.staged.pose_path.write_text("{}", encoding="utf-8")
        with mock.patch.object(Path, "read_bytes", side_effect=OSError("gone")):
            analysis.store_artifacts(self.staged)  # must not raise

    def test_never_raises_when_the_put_hits_the_filesystem(self) -> None:
        """LocalObjectStore.put does mkdir + write_bytes, so on the dev/CI path a disk error
        surfaces as OSError, not StorageError."""
        self.staged.pose_path.write_text("{}", encoding="utf-8")
        with mock.patch.object(
            storage.LocalObjectStore, "put", side_effect=OSError("no space left on device")
        ):
            analysis.store_artifacts(self.staged, thumbnail=b"jpeg")  # must not raise

    def test_never_raises_when_the_object_store_is_unavailable(self) -> None:
        with mock.patch.object(storage, "get_object_store", side_effect=RuntimeError("boom")):
            analysis.store_artifacts(self.staged, thumbnail=b"jpeg")  # must not raise


class DiscardStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = storage.LocalObjectStore(Path(tempfile.mkdtemp()))
        patcher = mock.patch.object(storage, "get_object_store", return_value=self.store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_removes_the_temp_directory(self) -> None:
        staged = analysis.stage_upload(b"v", suffix=".mp4", owner="u1")
        tmp_dir = staged.video_path.parent
        analysis.discard_stage(staged)
        self.assertFalse(tmp_dir.exists())

    def test_is_idempotent(self) -> None:
        staged = analysis.stage_upload(b"v", suffix=".mp4", owner="u1")
        analysis.discard_stage(staged)
        analysis.discard_stage(staged)  # must not raise

    def test_leaves_the_stored_objects_alone(self) -> None:
        """Cleanup is about temp files; the object store is the system of record."""
        staged = analysis.stage_upload(b"v", suffix=".mp4", owner="u1")
        analysis.discard_stage(staged)
        self.assertIsNotNone(self.store.open_object(f"{staged.prefix}/source"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_upload_staging.py -v`
Expected: FAIL — `AttributeError: module 'backend.app.services.analysis' has no attribute 'stage_upload'`

- [ ] **Step 3: Add the staging block to the analysis service**

In `backend/app/services/analysis.py`, add to the imports at the top:

```python
import logging
import shutil
import tempfile
from dataclasses import dataclass

from backend.app.services import storage

logger = logging.getLogger(__name__)
```

(`uuid`, `json`, and `Path` are already imported; keep the existing `from backend.app import config` import.)

Replace `save_upload` (currently lines 194-201) with:

```python
@dataclass(frozen=True)
class StagedUpload:
    """One in-flight upload: stored in the object store, staged on disk for the pipeline.

    ``prefix`` is the object-store key prefix holding every artifact for this upload, and is what
    ``videos.storage_key`` records. ``video_path`` and ``pose_path`` live in a temp directory that
    ``discard_stage`` removes once the analysis is done — they are scratch space, not storage.
    """

    video_id: str
    prefix: str
    video_path: Path
    pose_path: Path


def stage_upload(data: bytes, *, suffix: str, owner: str) -> StagedUpload:
    """Store the source video, then stage a temp copy the pose pipeline can open.

    THE OBJECT-STORE PUT HAPPENS FIRST AND IS ALLOWED TO RAISE. The raw clip is the one artifact
    that cannot be recomputed, and the put is fast next to the analysis, so discovering that
    storage is down before spending any CPU beats finishing an expensive analysis whose video
    cannot be kept. Callers map ``StorageError`` to a 503.

    The temp copy exists because ``process_video`` (OpenCV) and the detector's camera-view
    estimation both need a real filesystem path — bytes in memory are not enough.

    ``owner`` is the authenticated user's id, or ``"anon"`` for a demo upload.
    """
    video_id = f"upload_{uuid.uuid4().hex[:12]}"
    prefix = storage.upload_prefix(owner, video_id)
    storage.get_object_store().put(
        f"{prefix}/source", data, content_type=storage.video_content_type(suffix)
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{video_id}_"))
    video_path = tmp_dir / f"source{suffix}"
    video_path.write_bytes(data)
    return StagedUpload(
        video_id=video_id,
        prefix=prefix,
        video_path=video_path,
        pose_path=tmp_dir / "pose.json",
    )


def _put_artifact(staged: StagedUpload, name: str, data: bytes, content_type: str) -> None:
    """Upload one derived artifact, swallowing every failure. See ``store_artifacts``.

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


def store_artifacts(staged: StagedUpload, *, thumbnail: bytes | None = None) -> None:
    """Best-effort upload of the derived artifacts. NEVER RAISES.

    Mirrors ``store.persist_analysis``'s policy: a storage hiccup is logged, but it must never
    discard an analysis that already cost a full pipeline run. The caller relies on that literally
    — it does not wrap this call — so every path here has to hold it.

    ``pose.json`` is uploaded ONLY when one was actually produced. ``analyze_pose_payload``
    returns the ``analysis_pending`` skeleton without writing any pose JSON for a movement with
    no registered detector — which is most of the movement registry, not an edge case.
    """
    if staged.pose_path.is_file():
        try:
            pose_bytes = staged.pose_path.read_bytes()
        except OSError:
            # ``is_file()`` and the read are not atomic, and the read raises OSError rather than
            # StorageError — so this needs its own guard, not the put's.
            logger.exception("Failed to read staged pose JSON for %s", staged.video_id)
        else:
            _put_artifact(staged, "pose.json", pose_bytes, "application/json")
    if thumbnail:
        _put_artifact(staged, "thumb.jpg", thumbnail, "image/jpeg")


def discard_stage(staged: StagedUpload) -> None:
    """Remove the temp directory behind ``staged``. Idempotent and never raises."""
    shutil.rmtree(staged.video_path.parent, ignore_errors=True)
```

- [ ] **Step 4: Point the analysis functions at an explicit pose path**

In `backend/app/services/analysis.py`, change `analyze_video_file`'s signature and body. Replace lines 66-93 (from `def analyze_video_file(` through the `raise RuntimeError` block) with:

```python
def analyze_video_file(
    source_path: Path,
    *,
    video_id: str | None = None,
    pose_json_path: Path,
    movement: str | None = None,
    max_reps: int | None = -1,
) -> dict[str, Any]:
    """Run the full pipeline on an arbitrary video file (the live-upload flow).

    Extracts pose to ``pose_json_path``, runs rule detection with retrieval enrichment, and
    returns the detector result with a slimmed ``pose`` block attached.

    ``pose_json_path`` is supplied by the caller (``stage_upload`` puts it in the upload's temp
    directory) rather than derived from a runtime dir: pose JSON is uploaded to object storage
    after the analysis, so its on-disk location is scratch space with the same lifetime as the
    request.

    ``max_reps`` follows the ``-1`` sentinel convention: ``-1`` means "caller said nothing" and
    resolves to ``config.DEFAULT_MAX_REPS``; ``None`` means "analyze every repetition" and is
    passed through unchanged.
    """
    # Deferred imports: pull in MediaPipe/OpenCV (process_videos) and the detector only when an
    # upload is actually analyzed, keeping module import (and server startup) lightweight.
    from src.pose.pose_rule_detector import detect_pose_rules_from_json
    from src.pose.process_videos import process_video

    vid = video_id or source_path.stem

    ok = process_video(str(source_path), str(pose_json_path))
    if not ok or not pose_json_path.exists():
        raise RuntimeError(f"Pose extraction failed for {source_path.name}")
```

Leave the rest of the function (the `detect_pose_rules_from_json` call onwards) unchanged.

Then change `analyze_pose_payload`. Replace lines 155-186 (from `def analyze_pose_payload(` through the `pose_json_path.write_text(...)` line) with:

```python
def analyze_pose_payload(
    payload: dict[str, Any],
    *,
    movement: str,
    video_id: str | None = None,
    pose_json_path: Path,
    max_reps: int | None = -1,
) -> dict[str, Any]:
    """Analyze a client-supplied pose JSON payload — no server-side MediaPipe.

    Routes by movement to its registered rule detector. Movements with no detector return a
    skeleton-only 'analysis pending' result AND NEVER WRITE ``pose_json_path`` — which is why
    ``store_artifacts`` uploads pose.json conditionally. The video is still stored by the caller.

    ``max_reps`` follows the same ``-1`` sentinel convention as ``analyze_video_file``.
    """
    vid = video_id or f"upload_{uuid.uuid4().hex[:12]}"
    pose_block = build_pose_block_from_payload(payload)
    if not _has_detector(movement):
        return {
            "video_id": vid,
            "source": "upload",
            "analysis_pending": True,
            "movement": movement,
            "detections": [],
            "retrievals": [],
            "pose": pose_block,
        }
    # Persist the client pose JSON so the detector can estimate camera view from it. Without a
    # path, detect_pose_rules_from_payload treats view as "unknown" and suppresses/downweights the
    # view-dependent squat faults (knees_forward on side view, knees_inward, excessive_forward_lean).
    pose_json_path.write_text(json.dumps(payload), encoding="utf-8")
```

Leave the rest of the function unchanged.

- [ ] **Step 5: Delete the runtime upload directories**

In `backend/app/config.py`, delete lines 35-38:

```python
# Runtime scratch space for uploaded videos and their derived pose JSON (gitignored).
RUNTIME_DIR = DATA_DIR / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
UPLOAD_POSE_DIR = RUNTIME_DIR / "pose_json"
```

and replace with (RUNTIME_DIR is still needed — `storage.OBJECTS_DIR` hangs off it):

```python
# Runtime scratch space (gitignored). The local object store lives under this; uploads
# themselves are no longer kept here — see backend/app/services/storage.py.
RUNTIME_DIR = DATA_DIR / "runtime"
```

Delete `ensure_runtime_dirs` entirely (lines 60-63).

In `backend/app/main.py`, delete the startup hook (lines 81-83):

```python
@app.on_event("startup")
def _startup() -> None:
    config.ensure_runtime_dirs()
```

`config` is still imported by `main.py` for `CORS_ORIGINS` and the `/api/health` store checks, so leave the import.

**No startup hook replaces this, by design.** Nothing needs the object directory to pre-exist: `LocalObjectStore.put` creates its own parents, `delete_prefix` uses `rmtree(..., ignore_errors=True)`, and `open_object` returns `None` for a missing path. Do not re-add a directory-creating hook.

- [ ] **Step 6: Run the staging tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_upload_staging.py -v`
Expected: PASS

- [ ] **Step 7: Find and fix every caller the signature change broke**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: FAILURES in `tests/test_analyze_endpoint.py`, `tests/test_analyze_pose_endpoint.py`, `tests/test_backend.py`, `tests/test_backend_analysis.py`, `tests/test_analyze_pose_service.py` — anything that calls `save_upload`, `analyze_video_file`, `analyze_pose_payload`, or `config.UPLOAD_DIR`.

Do NOT fix the router tests yet — Task 3 rewrites those. For this task, fix only the tests that call the **service** directly, by passing an explicit `pose_json_path=tmp_path / "pose.json"`. For any test that references `config.UPLOAD_DIR` / `config.UPLOAD_POSE_DIR` / `config.ensure_runtime_dirs`, delete that assertion — those names are gone by design.

Run the service-level suites until green:

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_analysis.py tests/test_analyze_pose_service.py tests/test_upload_staging.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add backend/app/services/analysis.py backend/app/config.py backend/app/main.py tests/
git commit -m "feat(storage): stage uploads through object storage instead of a runtime dir"
```

---

### Task 3: Wire the analyze endpoints to object storage

**Files:**
- Modify: `backend/app/routers/analyze.py` (both endpoints)
- Modify: `backend/app/services/store.py:151-160` (`persist_analysis`'s `storage_key`)
- Modify: `tests/test_analyze_endpoint.py`, `tests/test_analyze_pose_endpoint.py`

**Interfaces:**
- Consumes: `analysis.stage_upload / store_artifacts / discard_stage / StagedUpload`, `analysis.analyze_video_file(..., pose_json_path=...)`, `analysis.analyze_pose_payload(..., pose_json_path=...)` (Task 2); `storage.get_object_store`, `storage.StorageError`, `storage.DEFAULT_URL_TTL` (Task 1).
- Produces:
  - `POST /api/analyze` and `POST /api/analyze/pose` both accept an optional `thumbnail` file field and return `video_url: str | None` on the response body.
  - `store.persist_analysis(..., storage_key: str)` — a new REQUIRED keyword argument.
  - `analyze_router.MAX_THUMBNAIL_BYTES: int`

- [ ] **Step 1: Write the failing endpoint tests**

Add to `tests/test_analyze_endpoint.py`. First replace the `setUp` stub of `save_upload` (lines 46-53) with a staging stub:

```python
    def setUp(self) -> None:
        self._orig_stage = analysis_service.stage_upload
        self._orig_artifacts = analysis_service.store_artifacts
        self._orig_discard = analysis_service.discard_stage
        self._orig_analyze = analysis_service.analyze_video_file
        self._orig_semaphore = analyze_router._ANALYSIS_SEMAPHORE

        # No real disk or object-store I/O: hand back a deterministic StagedUpload and record
        # what the router does with it.
        self.staged = analysis_service.StagedUpload(
            video_id="upload_test",
            prefix="uploads/anon/upload_test",
            video_path=Path("upload_test.mp4"),
            pose_path=Path("pose.json"),
        )
        self.artifacts: list[dict] = []
        self.discarded: list[object] = []
        analysis_service.stage_upload = lambda data, *, suffix=".mp4", owner="anon": self.staged
        analysis_service.store_artifacts = lambda staged, *, thumbnail=None: self.artifacts.append(
            {"staged": staged, "thumbnail": thumbnail}
        )
        analysis_service.discard_stage = lambda staged: self.discarded.append(staged)

        # Presigning is a storage concern; stub it so these tests stay offline.
        presign = mock.patch.object(
            analyze_router, "_source_url", side_effect=lambda prefix: f"https://signed/{prefix}"
        )
        presign.start()
        self.addCleanup(presign.stop)

        # KEEP THESE TESTS OFFLINE, as the module docstring promises. ``analyze`` calls
        # ``settings.allowed_upload_suffixes()``, which reads the admin overrides via
        # ``runtime_config.get_overrides()`` -- and that does a REAL Supabase round-trip whenever
        # auth is configured. On any machine with a populated ``.env`` that call measured ~10s,
        # which is long enough to break the concurrency test below on its own (see its docstring).
        # ``{}`` is exactly what ``get_overrides`` returns when auth is unconfigured, so this runs
        # the same code path CI does rather than a bespoke stub, and pins no suffix values.
        overrides = mock.patch.object(runtime_config, "get_overrides", return_value={})
        overrides.start()
        self.addCleanup(overrides.stop)

    def tearDown(self) -> None:
        analysis_service.stage_upload = self._orig_stage
        analysis_service.store_artifacts = self._orig_artifacts
        analysis_service.discard_stage = self._orig_discard
        analysis_service.analyze_video_file = self._orig_analyze
        analyze_router._ANALYSIS_SEMAPHORE = self._orig_semaphore
```

Every existing stub of `analyze_video_file` in this file takes `(path, *, video_id=None, movement=None, max_reps=-1)`; add `pose_json_path=None` to each so the new keyword does not blow up. For example line 104-110 becomes:

```python
        analysis_service.analyze_video_file = (
            lambda path, *, video_id=None, pose_json_path=None, movement=None, max_reps=-1: {
                "video_id": video_id,
                "source": "upload",
                "detections": [],
            }
        )
```

Do the same for `record_thread` (line 123) and `blocking` (line 160).

Then append this new test class to the file:

```python
class AnalyzeStorageTests(unittest.TestCase):
    """The object-storage contract of /api/analyze: fail-fast, artifacts, cleanup, video_url."""

    def setUp(self) -> None:
        self._orig = {
            name: getattr(analysis_service, name)
            for name in ("stage_upload", "store_artifacts", "discard_stage", "analyze_video_file")
        }
        overrides = mock.patch.object(runtime_config, "get_overrides", return_value={})
        overrides.start()
        self.addCleanup(overrides.stop)

        self.staged = analysis_service.StagedUpload(
            video_id="upload_test",
            prefix="uploads/anon/upload_test",
            video_path=Path("upload_test.mp4"),
            pose_path=Path("pose.json"),
        )
        self.artifacts: list[dict] = []
        self.discarded: list[object] = []
        analysis_service.stage_upload = lambda data, *, suffix=".mp4", owner="anon": self.staged
        analysis_service.store_artifacts = lambda staged, *, thumbnail=None: self.artifacts.append(
            {"staged": staged, "thumbnail": thumbnail}
        )
        analysis_service.discard_stage = lambda staged: self.discarded.append(staged)
        analysis_service.analyze_video_file = (
            lambda path, *, video_id=None, pose_json_path=None, movement=None, max_reps=-1: {
                "video_id": video_id,
                "source": "upload",
                "detections": [],
            }
        )
        presign = mock.patch.object(
            analyze_router, "_source_url", side_effect=lambda prefix: f"https://signed/{prefix}"
        )
        presign.start()
        self.addCleanup(presign.stop)

    def tearDown(self) -> None:
        for name, value in self._orig.items():
            setattr(analysis_service, name, value)

    def _run(self, **kwargs):
        params = {"movement": "Squat", "max_reps": None, "user": None}
        params.update(kwargs)
        return asyncio.run(analyze_router.analyze(_upload(), **params))

    def test_returns_a_presigned_video_url(self) -> None:
        result = self._run()
        self.assertEqual(result["video_url"], "https://signed/uploads/anon/upload_test")

    def test_a_storage_failure_before_analysis_is_a_503(self) -> None:
        def boom(data, *, suffix=".mp4", owner="anon"):
            raise storage.StorageError("R2 down")

        analysis_service.stage_upload = boom
        ran = []
        analysis_service.analyze_video_file = lambda *a, **k: ran.append(1) or {}
        with self.assertRaises(HTTPException) as ctx:
            self._run()
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ran, [], "no CPU may be spent on a clip whose video could not be stored")

    def test_stores_the_derived_artifacts_after_a_successful_analysis(self) -> None:
        self._run()
        self.assertEqual(len(self.artifacts), 1)
        self.assertIs(self.artifacts[0]["staged"], self.staged)

    def test_forwards_the_thumbnail_bytes(self) -> None:
        thumb = UploadFile(file=io.BytesIO(b"jpeg-bytes"), filename="t.jpg",
                           headers=Headers({"content-type": "image/jpeg"}))
        self._run(thumbnail=thumb)
        self.assertEqual(self.artifacts[0]["thumbnail"], b"jpeg-bytes")

    def test_rejects_a_non_jpeg_thumbnail(self) -> None:
        thumb = UploadFile(file=io.BytesIO(b"png"), filename="t.png",
                           headers=Headers({"content-type": "image/png"}))
        with self.assertRaises(HTTPException) as ctx:
            self._run(thumbnail=thumb)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_an_oversized_thumbnail(self) -> None:
        big = b"x" * (analyze_router.MAX_THUMBNAIL_BYTES + 1)
        thumb = UploadFile(file=io.BytesIO(big), filename="t.jpg",
                           headers=Headers({"content-type": "image/jpeg"}))
        with self.assertRaises(HTTPException) as ctx:
            self._run(thumbnail=thumb)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_discards_the_stage_even_when_the_analysis_fails(self) -> None:
        def boom(*args, **kwargs):
            raise RuntimeError("Pose extraction failed")

        analysis_service.analyze_video_file = boom
        with self.assertRaises(HTTPException):
            self._run()
        self.assertEqual(self.discarded, [self.staged])
        self.assertEqual(self.artifacts, [], "a failed analysis stores no derived artifacts")

    def test_discards_the_stage_after_a_successful_analysis(self) -> None:
        self._run()
        self.assertEqual(self.discarded, [self.staged])

    def test_video_url_is_not_written_into_the_persisted_result(self) -> None:
        """A presigned URL in the history row would be expired the moment it is replayed."""
        persisted: list[dict] = []

        def fake_persist(**kwargs):
            # Snapshot the document AS PERSISTED — the router mutates the same dict afterwards.
            persisted.append(dict(kwargs["result"]))
            return "analysis-1"

        user = mock.Mock(id="u1", token="tok")
        with mock.patch.object(store, "persist_analysis", side_effect=fake_persist):
            result = self._run(user=user)
        self.assertNotIn("video_url", persisted[0])
        self.assertIn("video_url", result)

    def test_persists_the_storage_prefix_as_the_storage_key(self) -> None:
        seen: list[dict] = []
        user = mock.Mock(id="u1", token="tok")
        with mock.patch.object(store, "persist_analysis", side_effect=lambda **kw: seen.append(kw) or "id"):
            self._run(user=user)
        self.assertEqual(seen[0]["storage_key"], "uploads/anon/upload_test")
```

Add these imports at the top of the file:

```python
from starlette.datastructures import Headers, UploadFile

from backend.app.services import storage
from backend.app.services import store
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_analyze_endpoint.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'MAX_THUMBNAIL_BYTES'` and `_source_url`

- [ ] **Step 3: Rewrite the analyze router**

In `backend/app/routers/analyze.py`, add to the imports:

```python
from backend.app.services import analysis, storage, store
```

Add these module-level helpers after `_ANALYSIS_SEMAPHORE`:

```python
# A thumbnail is one downscaled JPEG frame (the browser caps its longest edge at 480px), so
# anything approaching this is not a thumbnail. Bounds what an upload can push into storage.
MAX_THUMBNAIL_BYTES = 512 * 1024


def _source_url(prefix: str) -> str | None:
    """A short-lived playback URL for the upload's source object, or None if signing failed.

    Never raises: the analysis has already been produced by the time this is called, so a
    signing problem degrades playback rather than discarding a completed result.
    """
    try:
        return storage.get_object_store().presigned_url(f"{prefix}/source")
    except storage.StorageError:
        logger.exception("Failed to sign a playback URL for %s", prefix)
        return None


async def _read_thumbnail(thumbnail: UploadFile | None) -> bytes | None:
    """Validate and read the optional browser-captured frame.

    A missing thumbnail is NOT an error — older clients and browsers where frame capture failed
    must still be able to analyze. Only a wrong type or an implausible size is rejected.
    """
    if thumbnail is None:
        return None
    content_type = (thumbnail.content_type or "").split(";")[0].strip().lower()
    if content_type != "image/jpeg":
        raise HTTPException(status_code=400, detail="Thumbnail must be image/jpeg.")
    data = await thumbnail.read()
    if not data:
        return None
    if len(data) > MAX_THUMBNAIL_BYTES:
        raise HTTPException(status_code=400, detail="Thumbnail is too large.")
    return data
```

Now replace the body of `analyze` from `data = await file.read()` (line 129) to the end of the function with:

```python
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    thumb = await _read_thumbnail(thumbnail)

    # Anonymous demo uploads are still stored, under their own key prefix, so both paths behave
    # identically. A bucket lifecycle rule expires `uploads/anon/` — see the design doc.
    owner = user.id if user is not None else "anon"
    try:
        staged = await run_in_threadpool(
            analysis.stage_upload, data, suffix=suffix, owner=owner
        )
    except storage.StorageError as exc:
        logger.exception("Failed to store upload (owner=%s)", owner)
        raise HTTPException(
            status_code=503, detail="Storage is unavailable; please try again."
        ) from exc
    del data  # bytes are now stored and staged; don't pin the whole video in RAM while queued.

    try:
        async with _ANALYSIS_SEMAPHORE:
            result = await run_in_threadpool(
                analysis.analyze_video_file,
                staged.video_path,
                video_id=staged.video_id,
                pose_json_path=staged.pose_path,
                movement=canonical_movement,
                max_reps=resolved_max_reps,
            )
        # Only a SUCCESSFUL analysis has derived artifacts worth keeping.
        await run_in_threadpool(analysis.store_artifacts, staged, thumbnail=thumb)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await run_in_threadpool(analysis.discard_stage, staged)

    if user is not None:
        try:
            result["analysis_id"] = await run_in_threadpool(
                store.persist_analysis,
                token=user.token,
                user_id=user.id,
                video_id=staged.video_id,
                source="upload",
                result=result,
                storage_key=staged.prefix,
                filename=file.filename,
            )
        except Exception:  # noqa: BLE001 — never lose a completed analysis to a storage error
            logger.exception(
                "Failed to persist analysis (user=%s video=%s)", user.id, staged.video_id
            )
            result["analysis_id"] = None

    # AFTER the persist, deliberately: `result` is stored verbatim as JSONB, and a presigned URL
    # written into the history row would already be expired by the time anyone replayed it. The
    # replay path re-signs through GET /api/uploads/{video_id}/url instead.
    result["video_url"] = await run_in_threadpool(_source_url, staged.prefix)
    return result
```

Add the `thumbnail` parameter to the signature:

```python
@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    movement: str = Form(config.DEFAULT_ANALYSIS_MOVEMENT),
    max_reps: int | None = Form(None),
    thumbnail: UploadFile | None = File(None),
    user: CurrentUser | None = Depends(get_optional_user),
) -> dict:
```

Apply the identical treatment to `analyze_pose`: add `thumbnail: UploadFile | None = File(None)` to its signature, and replace its body from `data = await file.read()` (line 193) to the end with the same block, substituting the analysis call:

```python
            result = await run_in_threadpool(
                analysis.analyze_pose_payload,
                payload,
                movement=movement,
                video_id=staged.video_id,
                pose_json_path=staged.pose_path,
                max_reps=resolved_max_reps,
            )
```

and the log message `"Failed to persist pose analysis (user=%s video=%s)"`.

- [ ] **Step 4: Add the storage_key argument to persist_analysis**

In `backend/app/services/store.py`, change `persist_analysis`'s signature (line 134-142) to add `storage_key: str` after `source: str`, and replace the docstring and the `videos` upsert:

```python
def persist_analysis(
    *,
    token: str,
    user_id: str,
    video_id: str,
    source: str,
    storage_key: str,
    result: dict[str, Any],
    filename: str | None = None,
) -> str:
    """Upsert the video row and insert the analysis; return the new analysis id.

    ``storage_key`` is the object-store key PREFIX holding this upload's artifacts
    (``uploads/{owner}/{video_id}``), not a single object — the read path signs
    ``{storage_key}/source`` and ``{storage_key}/thumb.jpg`` off it, and deletion reaps
    everything under it. ``result`` is stored verbatim as JSONB so history replay is
    self-contained; note that the caller attaches the presigned ``video_url`` only AFTER this
    returns, so no expired URL is ever persisted.
    """
    client = _user_client(token)

    client.table("videos").upsert(
        {
            "user_id": user_id,
            "video_id": video_id,
            "filename": filename,
            "storage_key": storage_key,
            "status": "done",
        },
        on_conflict="user_id,video_id",
    ).execute()
```

- [ ] **Step 5: Run the endpoint tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_analyze_endpoint.py -v`
Expected: PASS

- [ ] **Step 6: Apply the same test updates to the pose endpoint suite**

Mirror the Step 1 changes in `tests/test_analyze_pose_endpoint.py`: replace any `save_upload` stub with a `stage_upload` / `store_artifacts` / `discard_stage` trio returning the same `StagedUpload`, add `pose_json_path=None` to every `analyze_pose_payload` stub signature, patch `analyze_router._source_url`, and add a test asserting the pending-skeleton path still stores artifacts:

```python
    def test_stores_artifacts_even_for_the_analysis_pending_skeleton(self) -> None:
        """A movement with no detector still has a source video and a thumbnail worth keeping."""
        analysis_service.analyze_pose_payload = (
            lambda payload, *, movement, video_id=None, pose_json_path=None, max_reps=-1: {
                "video_id": video_id,
                "source": "upload",
                "analysis_pending": True,
                "detections": [],
            }
        )
        asyncio.run(
            analyze_router.analyze_pose(
                movement="High Knee",
                pose=json.dumps({"frames": []}),
                file=_upload(),
                max_reps=None,
                user=None,
            )
        )
        self.assertEqual(len(self.artifacts), 1)
```

Run: `.venv\Scripts\python.exe -m pytest tests/test_analyze_pose_endpoint.py -v`
Expected: PASS

- [ ] **Step 7: Fix every remaining caller of persist_analysis**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: FAILURES in `tests/test_backend.py` / `tests/test_store_movement.py` — `persist_analysis() missing 1 required keyword-only argument: 'storage_key'`.

Add `storage_key="uploads/u1/upload_test"` to each call, and update any assertion that expected `"storage_key": f"runtime/uploads/{video_id}"` to expect the passed-in prefix.

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: remaining failures only in `tests/test_backend.py` around `/api/video-file` and `library.uploaded_video_path` — those are Task 4's.

- [ ] **Step 8: Commit**

```bash
git add backend/app/routers/analyze.py backend/app/services/store.py tests/
git commit -m "feat(storage): store uploads and thumbnails from the analyze endpoints"
```

---

### Task 4: Ownership-checked read path (closes the IDOR)

**Files:**
- Modify: `backend/app/routers/videos.py`
- Modify: `backend/app/services/library.py:82-95` (delete `uploaded_video_path`)
- Modify: `backend/app/services/store.py` (add the two lookups)
- Create: `tests/test_upload_urls.py`
- Modify: `tests/test_backend.py` (the `/api/video-file` expectations)

**Interfaces:**
- Consumes: `storage.get_object_store`, `storage.StorageError`, `storage.DEFAULT_URL_TTL` (Task 1); `store.persist_analysis`'s prefix convention (Task 3).
- Produces:
  - `store.get_storage_key(*, token: str, video_id: str) -> str | None`
  - `store.get_storage_keys(*, token: str, video_ids: list[str]) -> dict[str, str]`
  - `GET /api/uploads/{video_id}/url` → `{"video_url": str, "thumbnail_url": str, "expires_in": int}`
  - `POST /api/uploads/urls` body `{"video_ids": [str]}` → `{"items": {video_id: {"video_url": str, "thumbnail_url": str}}, "expires_in": int}`
  - `videos_router.MAX_URL_BATCH: int` (200)

- [ ] **Step 1: Write the failing read-path tests**

Create `tests/test_upload_urls.py`:

```python
"""Tests for the ownership-checked upload URL endpoints, and for the IDOR they close.

`GET /api/video-file/{video_id}` used to fall back to any user's upload with no auth at all.
It now serves library demo clips only; uploads go through the endpoints below, which resolve
`videos.storage_key` with the CALLER'S OWN JWT so Postgres RLS performs the ownership check.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from backend.app.routers import videos as videos_router
from backend.app.services import storage, store


class _User:
    id = "u1"
    token = "tok"


class UploadUrlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.local = storage.LocalObjectStore(Path(tempfile.mkdtemp()))
        patcher = mock.patch.object(storage, "get_object_store", return_value=self.local)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_owner_gets_signed_source_and_thumbnail_urls(self) -> None:
        with mock.patch.object(store, "get_storage_key", return_value="uploads/u1/upload_a"):
            body = videos_router.get_upload_urls("upload_a", user=_User())
        self.assertEqual(body["video_url"], "/api/local-object/uploads/u1/upload_a/source")
        self.assertEqual(body["thumbnail_url"], "/api/local-object/uploads/u1/upload_a/thumb.jpg")
        self.assertEqual(body["expires_in"], storage.DEFAULT_URL_TTL)

    def test_a_row_the_caller_does_not_own_is_a_404(self) -> None:
        """RLS hides someone else's row, so 'not yours' and 'does not exist' are one answer."""
        with mock.patch.object(store, "get_storage_key", return_value=None):
            with self.assertRaises(HTTPException) as ctx:
                videos_router.get_upload_urls("upload_a", user=_User())
        self.assertEqual(ctx.exception.status_code, 404)

    def test_a_signing_failure_is_a_503(self) -> None:
        class _Failing:
            def presigned_url(self, key, *, expires_in=storage.DEFAULT_URL_TTL):
                raise storage.StorageError("down")

        with mock.patch.object(storage, "get_object_store", return_value=_Failing()):
            with mock.patch.object(store, "get_storage_key", return_value="uploads/u1/upload_a"):
                with self.assertRaises(HTTPException) as ctx:
                    videos_router.get_upload_urls("upload_a", user=_User())
        self.assertEqual(ctx.exception.status_code, 503)


class UploadUrlBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.local = storage.LocalObjectStore(Path(tempfile.mkdtemp()))
        patcher = mock.patch.object(storage, "get_object_store", return_value=self.local)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_returns_one_entry_per_owned_id(self) -> None:
        keys = {"upload_a": "uploads/u1/upload_a", "upload_b": "uploads/u1/upload_b"}
        with mock.patch.object(store, "get_storage_keys", return_value=keys):
            body = videos_router.get_upload_urls_batch(
                videos_router.UploadUrlBatch(video_ids=["upload_a", "upload_b"]), user=_User()
            )
        self.assertEqual(set(body["items"]), {"upload_a", "upload_b"})
        self.assertEqual(
            body["items"]["upload_a"]["video_url"], "/api/local-object/uploads/u1/upload_a/source"
        )

    def test_ids_the_caller_does_not_own_are_simply_absent(self) -> None:
        with mock.patch.object(store, "get_storage_keys", return_value={}):
            body = videos_router.get_upload_urls_batch(
                videos_router.UploadUrlBatch(video_ids=["someone_elses"]), user=_User()
            )
        self.assertEqual(body["items"], {})

    def test_rejects_an_oversized_batch(self) -> None:
        ids = [f"upload_{i}" for i in range(videos_router.MAX_URL_BATCH + 1)]
        with self.assertRaises(HTTPException) as ctx:
            videos_router.get_upload_urls_batch(
                videos_router.UploadUrlBatch(video_ids=ids), user=_User()
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_an_empty_batch_short_circuits_without_a_db_call(self) -> None:
        with mock.patch.object(store, "get_storage_keys") as lookup:
            body = videos_router.get_upload_urls_batch(
                videos_router.UploadUrlBatch(video_ids=[]), user=_User()
            )
        lookup.assert_not_called()
        self.assertEqual(body["items"], {})


class VideoFileIsLibraryOnlyTests(unittest.TestCase):
    """The regression test for the closed IDOR."""

    def test_an_upload_id_is_a_404(self) -> None:
        from backend.app.services import library

        with mock.patch.object(library, "video_path", return_value=None):
            with self.assertRaises(HTTPException) as ctx:
                videos_router.get_video_file("upload_someoneelse")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_uploaded_video_path_no_longer_exists(self) -> None:
        """The lookup itself is gone, so no future caller can re-open the hole."""
        from backend.app.services import library

        self.assertFalse(hasattr(library, "uploaded_video_path"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_upload_urls.py -v`
Expected: FAIL — `AttributeError: module 'backend.app.routers.videos' has no attribute 'get_upload_urls'`

- [ ] **Step 3: Add the storage-key lookups to the store**

Append to `backend/app/services/store.py`:

```python
def get_storage_key(*, token: str, video_id: str) -> str | None:
    """The object-store key prefix for one of the caller's uploads, or ``None``.

    Read with the CALLER'S OWN JWT, so the ``videos`` RLS policy performs the ownership check:
    another user's row is simply not visible, which is why the endpoint answers 404 for both
    "does not exist" and "not yours". A patchable seam — the unit tests replace ``_user_client``.
    """
    client = _user_client(token)
    resp = (
        client.table("videos")
        .select("storage_key")
        .eq("video_id", video_id)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    return rows[0].get("storage_key") if rows else None


def get_storage_keys(*, token: str, video_ids: list[str]) -> dict[str, str]:
    """``{video_id: storage_key}`` for whichever of ``video_ids`` the caller owns.

    One round trip for a whole history page instead of one per row. Ids the caller does not own
    are absent from the result (RLS filters them), never an error.
    """
    if not video_ids:
        return {}
    client = _user_client(token)
    resp = (
        client.table("videos")
        .select("video_id, storage_key")
        .in_("video_id", video_ids)
        .execute()
    )
    return {
        row["video_id"]: row["storage_key"]
        for row in (resp.data or [])
        if row.get("video_id") and row.get("storage_key")
    }
```

- [ ] **Step 4: Add the read endpoints and narrow /api/video-file**

Rewrite `backend/app/routers/videos.py`. The full file:

```python
"""Library listing, precomputed analysis, pose overlay, and video URL endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.app.auth import CurrentUser, get_current_user
from backend.app.services import analysis, library, storage, store

router = APIRouter(prefix="/api", tags=["videos"])

# How many uploads one batch URL request may cover. A history page is 50 rows by default; the
# cap keeps a crafted request from asking the DB for an unbounded `in_` list.
MAX_URL_BATCH = 200


class UploadUrlBatch(BaseModel):
    video_ids: list[str]


def _upload_urls(prefix: str) -> dict[str, str]:
    """Signed URLs for one upload's playable artifacts.

    The thumbnail URL is signed unconditionally — a clip uploaded before thumbnails existed, or
    one whose capture failed, simply 404s when the browser fetches it, and the UI falls back.
    Probing for existence first would cost a round trip per row to save an occasional 404.
    """
    obj = storage.get_object_store()
    return {
        "video_url": obj.presigned_url(f"{prefix}/source"),
        "thumbnail_url": obj.presigned_url(f"{prefix}/thumb.jpg"),
    }


@router.get("/videos")
def list_videos(limit: int = 50, offset: int = 0, fault: str | None = None) -> dict:
    """List precomputed library clips (clips containing faults first)."""
    return library.list_videos(limit=limit, offset=offset, fault=fault)


@router.get("/analysis/{video_id}")
def get_analysis(video_id: str) -> dict:
    """Return the precomputed analysis for a library video (retrieval enriched on demand)."""
    try:
        return library.load_analysis(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/pose/{video_id}")
def get_pose(video_id: str) -> dict:
    """Return the slim 33-landmark overlay block for a library video."""
    pose_path = library.pose_json_path(video_id)
    if pose_path is None:
        raise HTTPException(status_code=404, detail=f"No pose data for '{video_id}'.")
    return analysis.build_pose_block(pose_path)


@router.get("/video-file/{video_id}")
def get_video_file(video_id: str) -> FileResponse:
    """Stream a LIBRARY demo clip's mp4 (public, shared assets). Supports HTTP Range seeking.

    Uploads are deliberately NOT reachable here. This endpoint has no auth dependency, and its
    former fallback to ``library.uploaded_video_path`` therefore handed any caller who knew a
    ``video_id`` any user's upload. Uploads now go through ``/api/uploads/{video_id}/url``, which
    resolves the key as the caller so RLS enforces ownership. The fallback is gone rather than
    guarded, so there is no code path from here to a user's clip to re-open by accident.
    """
    path = library.video_path(video_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail=f"No video file for '{video_id}'.")
    return FileResponse(path, media_type="video/mp4")


@router.get("/uploads/{video_id}/url")
def get_upload_urls(video_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
    """Short-lived playback + thumbnail URLs for one of the CALLER'S uploads.

    404 covers both "no such upload" and "not yours": the storage key is read with the caller's
    own JWT, so RLS makes the two indistinguishable — the same shape ``delete_analysis`` uses.
    """
    prefix = store.get_storage_key(token=user.token, video_id=video_id)
    if prefix is None:
        raise HTTPException(status_code=404, detail=f"No upload '{video_id}'.")
    try:
        urls = _upload_urls(prefix)
    except storage.StorageError as exc:
        raise HTTPException(status_code=503, detail="Storage is unavailable.") from exc
    return {**urls, "expires_in": storage.DEFAULT_URL_TTL}


@router.post("/uploads/urls")
def get_upload_urls_batch(
    body: UploadUrlBatch, user: CurrentUser = Depends(get_current_user)
) -> dict:
    """The same URLs for many uploads at once, for a history page.

    One request and one DB round trip for a whole page, rather than N of each. Ids the caller
    does not own are absent from ``items`` rather than an error — a partial answer is the honest
    one when RLS has filtered the rest.
    """
    if len(body.video_ids) > MAX_URL_BATCH:
        raise HTTPException(
            status_code=400, detail=f"At most {MAX_URL_BATCH} video ids per request."
        )
    if not body.video_ids:
        return {"items": {}, "expires_in": storage.DEFAULT_URL_TTL}
    prefixes = store.get_storage_keys(token=user.token, video_ids=body.video_ids)
    try:
        items = {video_id: _upload_urls(prefix) for video_id, prefix in prefixes.items()}
    except storage.StorageError as exc:
        raise HTTPException(status_code=503, detail="Storage is unavailable.") from exc
    return {"items": items, "expires_in": storage.DEFAULT_URL_TTL}


@router.get("/local-object/{key:path}")
def get_local_object(key: str) -> FileResponse:
    """DEVELOPMENT ONLY: serve an object out of the local filesystem store.

    Inert in production: when R2 is configured, ``get_object_store()`` returns an
    ``R2ObjectStore`` and this endpoint 404s for every key. It exists so ``LocalObjectStore``
    can hand back a URL the browser can actually fetch, keeping the frontend contract identical
    in both modes. It carries no signature — reaching a key still requires having been given it
    by the ownership-checked ``/api/uploads/{video_id}/url``.
    """
    store_ = storage.get_object_store()
    if not isinstance(store_, storage.LocalObjectStore):
        raise HTTPException(status_code=404, detail="Not found.")
    try:
        found = store_.open_object(key)
    except storage.StorageError as exc:
        raise HTTPException(status_code=404, detail="Not found.") from exc
    if found is None:
        raise HTTPException(status_code=404, detail="Not found.")
    path, content_type = found
    return FileResponse(path, media_type=content_type)
```

- [ ] **Step 5: Delete the uploaded-video lookup**

In `backend/app/services/library.py`, delete `uploaded_video_path` entirely (lines 82-95). If `settings` becomes an unused import after this, remove it too — run `.venv\Scripts\python.exe -c "import backend.app.services.library"` to confirm the module still imports.

- [ ] **Step 6: Run the read-path tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_upload_urls.py -v`
Expected: PASS

- [ ] **Step 7: Fix the existing video-file tests**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: failures in `tests/test_backend.py` for tests asserting that an uploaded file is served by `/api/video-file`.

Update each: an upload id must now 404 there. Keep every library-clip assertion as it is.

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add backend/app/routers/videos.py backend/app/services/library.py backend/app/services/store.py tests/
git commit -m "feat(storage): serve uploads via ownership-checked presigned URLs

Closes the unauthenticated /api/video-file IDOR by removing its upload
fallback rather than guarding it."
```

---

### Task 5: Reap stored objects on deletion

**Files:**
- Modify: `backend/app/services/store.py` (`delete_all_analyses` 201-213, `delete_analysis` 216-266)
- Modify: `tests/test_backend.py` (deletion tests) or create `tests/test_delete_reaping.py`

**Interfaces:**
- Consumes: `storage.get_object_store`, `storage.StorageError` (Task 1); `store.get_storage_key` (Task 4).
- Produces: no new public names — `delete_analysis` and `delete_all_analyses` keep their signatures and return types.

- [ ] **Step 1: Write the failing reaping tests**

Create `tests/test_delete_reaping.py`:

```python
"""Deleting an analysis must also reap its stored objects.

Before object storage, `delete_analysis`'s own docstring admitted the uploaded file was
"deliberately left on disk". With a `delete_prefix` to call, that orphan is closed.
"""

from __future__ import annotations

import unittest
from unittest import mock

from backend.app.services import storage, store


class _Result:
    def __init__(self, data=None, count=None):
        self.data = data if data is not None else []
        self.count = count


class _Table:
    """Records the operation order so a test can assert reads happen before deletes."""

    def __init__(self, log, name, responses):
        self._log, self._name, self._responses = log, name, responses
        self._op = None

    def select(self, *args, **kwargs):
        self._op = "select"
        self._log.append((self._name, "select"))
        return self

    def delete(self):
        self._op = "delete"
        self._log.append((self._name, "delete"))
        return self

    def eq(self, *args, **kwargs):
        return self

    def in_(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def execute(self):
        return self._responses.get((self._name, self._op), _Result())


class _Client:
    def __init__(self, log, responses):
        self._log, self._responses = log, responses

    def table(self, name):
        return _Table(self._log, name, self._responses)


class _RecordingStore:
    def __init__(self, fail=False):
        self.deleted: list[str] = []
        self._fail = fail

    def delete_prefix(self, prefix):
        if self._fail:
            raise storage.StorageError("down")
        self.deleted.append(prefix)


class DeleteAnalysisReapsObjectsTests(unittest.TestCase):
    def _run(self, obj_store, *, siblings=0):
        log: list[tuple[str, str]] = []
        responses = {
            ("analyses", "select"): _Result(data=[{"video_id": "upload_a"}], count=siblings),
            ("analyses", "delete"): _Result(data=[{"id": "a1"}]),
            ("videos", "select"): _Result(data=[{"storage_key": "uploads/u1/upload_a"}]),
        }
        client = _Client(log, responses)
        with mock.patch.object(store, "_user_client", return_value=client):
            with mock.patch.object(storage, "get_object_store", return_value=obj_store):
                ok = store.delete_analysis(token="t", analysis_id="a1", user_id="u1")
        return ok, log

    def test_reaps_the_prefix_when_it_was_the_last_analysis(self) -> None:
        obj = _RecordingStore()
        ok, _ = self._run(obj, siblings=0)
        self.assertTrue(ok)
        self.assertEqual(obj.deleted, ["uploads/u1/upload_a"])

    def test_keeps_the_objects_when_a_sibling_analysis_remains(self) -> None:
        """Re-analysing one clip inserts a second `analyses` row against the same video."""
        obj = _RecordingStore()
        self._run(obj, siblings=1)
        self.assertEqual(obj.deleted, [])

    def test_a_storage_failure_does_not_undo_the_db_delete(self) -> None:
        obj = _RecordingStore(fail=True)
        ok, _ = self._run(obj, siblings=0)
        self.assertTrue(ok, "the row is gone; a failed reap leaves an orphan, not a stuck record")


class DeleteAllAnalysesReapsObjectsTests(unittest.TestCase):
    def test_reads_every_storage_key_before_deleting_anything(self) -> None:
        """Selecting after the bulk delete would find no rows and silently reap nothing."""
        log: list[tuple[str, str]] = []
        responses = {
            ("analyses", "delete"): _Result(data=[{"id": "a1"}, {"id": "a2"}]),
            ("videos", "select"): _Result(
                data=[
                    {"storage_key": "uploads/u1/upload_a"},
                    {"storage_key": "uploads/u1/upload_b"},
                ]
            ),
        }
        obj = _RecordingStore()
        client = _Client(log, responses)
        with mock.patch.object(store, "_user_client", return_value=client):
            with mock.patch.object(storage, "get_object_store", return_value=obj):
                deleted = store.delete_all_analyses(token="t", user_id="u1")
        self.assertEqual(deleted, 2)
        self.assertEqual(sorted(obj.deleted), ["uploads/u1/upload_a", "uploads/u1/upload_b"])
        first_delete = next(i for i, (_, op) in enumerate(log) if op == "delete")
        videos_select = log.index(("videos", "select"))
        self.assertLess(videos_select, first_delete)

    def test_a_storage_failure_does_not_stop_the_db_delete(self) -> None:
        log: list[tuple[str, str]] = []
        responses = {
            ("analyses", "delete"): _Result(data=[{"id": "a1"}]),
            ("videos", "select"): _Result(data=[{"storage_key": "uploads/u1/upload_a"}]),
        }
        client = _Client(log, responses)
        with mock.patch.object(store, "_user_client", return_value=client):
            with mock.patch.object(storage, "get_object_store", return_value=_RecordingStore(fail=True)):
                deleted = store.delete_all_analyses(token="t", user_id="u1")
        self.assertEqual(deleted, 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_delete_reaping.py -v`
Expected: FAIL — `obj.deleted` is empty; nothing calls `delete_prefix` yet.

- [ ] **Step 3: Add the reaping**

In `backend/app/services/store.py`, add to the imports at the top:

```python
import logging

from backend.app.services import storage

logger = logging.getLogger(__name__)
```

Add this private helper above `delete_all_analyses`:

```python
def _reap_objects(prefixes: list[str]) -> None:
    """Delete every stored artifact under each prefix. Best-effort: logged, never raised.

    A storage failure must not roll back a DB deletion the user already asked for — an orphaned
    object is a cost, a record that refuses to delete is a bug.
    """
    if not prefixes:
        return
    obj_store = storage.get_object_store()
    for prefix in prefixes:
        try:
            obj_store.delete_prefix(prefix)
        except storage.StorageError:
            logger.exception("Failed to delete stored objects under %s", prefix)
```

Replace `delete_all_analyses` (lines 201-213) with:

```python
def delete_all_analyses(*, token: str, user_id: str) -> int:
    """Delete every analysis (and source video row + stored objects) owned by the caller;
    return how many analyses were removed.

    RLS already scopes writes to ``auth.uid() = user_id``, but we also filter by ``user_id``
    explicitly: PostgREST refuses an unfiltered bulk delete, and the predicate is a second guard.
    """
    client = _user_client(token)
    # READ THE STORAGE KEYS FIRST. PostgREST returns nothing useful from a bulk delete, so a
    # select issued afterwards would find no rows and silently reap nothing — a failure mode that
    # passes a mocked test. The order is the correctness property here.
    videos = client.table("videos").select("storage_key").eq("user_id", user_id).execute()
    prefixes = [row["storage_key"] for row in (videos.data or []) if row.get("storage_key")]

    resp = client.table("analyses").delete().eq("user_id", user_id).execute()
    # Drop the (now orphaned) source video rows and chat threads too, so a "clear" leaves no residue.
    client.table("videos").delete().eq("user_id", user_id).execute()
    client.table("conversations").delete().eq("user_id", user_id).execute()
    _reap_objects(prefixes)
    return len(resp.data or [])
```

In `delete_analysis`, replace the closing block (lines 253-266) with:

```python
    siblings = (
        client.table("analyses")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("video_id", video_id)
        .limit(1)
        .execute()
    )
    if (siblings.count or 0) == 0:
        # Read the key before dropping the row that holds it.
        videos = (
            client.table("videos")
            .select("storage_key")
            .eq("user_id", user_id)
            .eq("video_id", video_id)
            .limit(1)
            .execute()
        )
        rows = videos.data or []
        prefix = rows[0].get("storage_key") if rows else None
        client.table("videos").delete().eq("user_id", user_id).eq("video_id", video_id).execute()
        client.table("conversations").delete().eq("user_id", user_id).eq(
            "video_id", video_id
        ).execute()
        _reap_objects([prefix] if prefix else [])
    return True
```

Also update `delete_analysis`'s docstring — delete the paragraph beginning "The uploaded file under ``runtime/uploads/`` is deliberately left on disk" and replace it with:

```
    The upload's stored objects are reaped along with the video row, but only on that same
    last-analysis condition: a sibling analysis still needs the clip to replay.
```

- [ ] **Step 4: Verify the new import introduced no cycle**

`store.py` importing `storage` is the first `services/ → services/` edge in this codebase, and `auth.py` already imports `store`. The chain should be `auth → store → storage → settings → config`, which is acyclic — but check it rather than reasoning about it:

Run: `.venv\Scripts\python.exe -c "import backend.app.main"`
Expected: imports clean, no output. An `ImportError: cannot import name ... (most likely due to a circular import)` means the `storage` import in `store.py` must move inside `_reap_objects` instead.

- [ ] **Step 5: Run the reaping tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_delete_reaping.py -v`
Expected: PASS

- [ ] **Step 6: Register the new suites with the coverage gate, then run it**

`scripts/run_backend_coverage.py` measures all of `backend/app` but runs only a HARDCODED list of test files (`_DEFAULT_TESTS`, line 25). Four new suites are not in it, so the gate would measure `storage.py` and the amended `store.py` while running none of their tests — reporting a coverage drop that is really a missing entry. Add all four:

```python
_DEFAULT_TESTS = [
    "tests/test_backend.py",
    "tests/test_analyze_pose_endpoint.py",
    "tests/test_chat_endpoint.py",
    "tests/test_backend_line_auth.py",
    "tests/test_backend_line_webhook.py",
    "tests/test_backend_admin_line.py",
    "tests/test_storage.py",
    "tests/test_upload_staging.py",
    "tests/test_upload_urls.py",
    "tests/test_delete_reaping.py",
]
```

(Keep any entries already present that are not shown here — add, don't replace.)

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS

Run: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
Expected: PASS at ≥95%. If `storage.py`'s `R2ObjectStore._s3` (the boto3 build) is the only uncovered branch, add a test that patches `boto3.client` and asserts the endpoint URL and `region_name="auto"` are passed:

```python
class R2ClientBuildTests(unittest.TestCase):
    def test_builds_an_s3v4_client_against_the_account_endpoint(self) -> None:
        store = storage.R2ObjectStore(
            account_id="acc", access_key_id="k", secret_access_key="s", bucket="b"
        )
        with mock.patch("boto3.client") as factory:
            store._s3()
        kwargs = factory.call_args.kwargs
        self.assertEqual(kwargs["endpoint_url"], "https://acc.r2.cloudflarestorage.com")
        self.assertEqual(kwargs["region_name"], "auto")
```

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/store.py tests/test_delete_reaping.py
git commit -m "feat(storage): reap stored objects when an analysis is deleted"
```

---

### Task 6: Browser thumbnail capture and API wiring

**Files:**
- Create: `frontend/src/lib/thumbnail.ts`
- Create: `frontend/src/test/lib.thumbnail.test.ts`
- Modify: `frontend/src/api.ts` (types at 89-105, `analyzeUpload` 616-632, `analyzePose` 634-650, `videoFileUrl` 507)
- Modify: `frontend/src/App.tsx:116-125`
- Modify: `frontend/src/test/api.test.ts`, `frontend/src/test/api.pose.test.ts`

**All commands in this task run with cwd = `frontend/`.**

**Interfaces:**
- Consumes: `resolveDuration` from `frontend/src/lib/poseExtract.ts` (existing).
- Produces:
  - `thumbnail.THUMBNAIL_MAX_EDGE: number` (480)
  - `thumbnail.thumbnailSize(width: number, height: number): {width: number; height: number}` — pure, testable
  - `thumbnail.thumbnailTime(duration: number): number` — pure, testable
  - `thumbnail.captureThumbnail(video: Blob): Promise<Blob | null>` — impure, coverage-excluded
  - `api.Analysis.video_url?: string | null`
  - `api.UploadMedia` interface `{video_url: string; thumbnail_url: string; expires_in: number}`
  - `api.analyzeUpload(file, movement, thumbnail?: Blob | null)`
  - `api.analyzePose(movement, pose, video, thumbnail?: Blob | null)`

- [ ] **Step 1: Write the failing thumbnail tests**

Create `frontend/src/test/lib.thumbnail.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { THUMBNAIL_MAX_EDGE, thumbnailSize, thumbnailTime } from "../lib/thumbnail";

describe("thumbnailSize", () => {
  it("leaves a small frame alone", () => {
    expect(thumbnailSize(320, 240)).toEqual({ width: 320, height: 240 });
  });

  it("scales a landscape frame down by its longest edge", () => {
    expect(thumbnailSize(1920, 1080)).toEqual({ width: 480, height: 270 });
  });

  it("scales a portrait frame down by its longest edge", () => {
    expect(thumbnailSize(1080, 1920)).toEqual({ width: 270, height: 480 });
  });

  it("never returns a zero dimension for an extreme aspect ratio", () => {
    const { width, height } = thumbnailSize(4000, 1);
    expect(width).toBe(THUMBNAIL_MAX_EDGE);
    expect(height).toBeGreaterThanOrEqual(1);
  });
});

describe("thumbnailTime", () => {
  it("picks a frame a quarter of the way in", () => {
    expect(thumbnailTime(8)).toBe(2);
  });

  it("falls back to the first frame when the length is unusable", () => {
    // A MediaRecorder clip whose duration never resolved. 0 is a real frame; NaN is not a time.
    expect(thumbnailTime(Number.NaN)).toBe(0);
    expect(thumbnailTime(0)).toBe(0);
    expect(thumbnailTime(Infinity)).toBe(0);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `yarn test src/test/lib.thumbnail.test.ts`
Expected: FAIL — `Failed to resolve import "../lib/thumbnail"`

- [ ] **Step 3: Write the thumbnail utility**

Create `frontend/src/lib/thumbnail.ts`:

```ts
// One frame of an upload, captured in the browser and sent alongside it so the history page has
// something to show. The pure sizing/timing decisions are exported and unit-tested; the <video>
// and canvas glue below them cannot run under jsdom and is coverage-excluded, matching the split
// in lib/poseExtract.ts.
import { resolveDuration } from "./poseExtract";

/** Longest edge of the stored thumbnail. A history card renders it at ~40px; 480 covers a
 *  retina card and any future larger use without approaching the backend's 512KB cap. */
export const THUMBNAIL_MAX_EDGE = 480;

/** Where in the clip to grab the frame: a quarter in is usually mid-movement, and past the
 *  black or motion-blurred frames a clip tends to open on. */
const THUMBNAIL_POSITION = 0.25;

const CAPTURE_TIMEOUT_MS = 5000;
const JPEG_QUALITY = 0.8;

/** How close to the requested timestamp counts as "the seek landed". Browsers snap to the
 *  nearest keyframe, so an exact match is not something this may depend on. */
const SEEK_TOLERANCE_S = 0.05;

/** Downscale a frame to fit THUMBNAIL_MAX_EDGE, preserving aspect. Never returns 0 in either
 *  dimension — a 0-width canvas throws on drawImage. */
export function thumbnailSize(width: number, height: number): { width: number; height: number } {
  const scale = Math.min(1, THUMBNAIL_MAX_EDGE / Math.max(width, height));
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

/** The timestamp to seek to. A recorded MediaRecorder clip can report NaN or Infinity for its
 *  length (no Duration element in a live-muxed WebM — see poseExtract.ts); seeking to NaN is a
 *  no-op that would hang the capture, so fall back to the opening frame. */
export function thumbnailTime(duration: number): number {
  if (!Number.isFinite(duration) || duration <= 0) return 0;
  return duration * THUMBNAIL_POSITION;
}

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("thumbnail capture timed out")), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      }
    );
  });
}

/* c8 ignore start — <video>/canvas decode glue, unrunnable under jsdom */
/**
 * Grab one frame of `video` as a JPEG blob.
 *
 * Resolves to `null` on ANY failure. A thumbnail is a nicety; a decode problem must never block
 * an analysis, so every error path here is a silent degradation rather than a thrown one.
 */
export async function captureThumbnail(video: Blob): Promise<Blob | null> {
  const url = URL.createObjectURL(video);
  const el = document.createElement("video");
  el.muted = true;
  el.playsInline = true;
  try {
    const loaded = new Promise<void>((resolve, reject) => {
      el.onloadedmetadata = () => resolve();
      el.onerror = () => reject(new Error("could not decode the clip"));
    });
    el.src = url;
    await withTimeout(loaded, CAPTURE_TIMEOUT_MS);

    // A recorded clip's duration is not known until probed — reuse the same recovery the pose
    // extractor needs, so both paths behave the same on a live recording.
    const duration = await resolveDuration(el, CAPTURE_TIMEOUT_MS).catch(() => Number.NaN);
    const target = thumbnailTime(duration);

    const seeked = new Promise<void>((resolve, reject) => {
      // GUARDED ON POSITION, not just on the event firing. `resolveDuration` rewinds to
      // currentTime = 0 as its last act (both on success and on timeout), and that write can emit
      // a `seeked` that lands after this handler is attached but before our own seek takes
      // effect. Resolving on it would capture the clip's OPENING frame — usually black — which is
      // exactly the frame this whole 25% offset exists to avoid. The recorded-clip path is where
      // resolveDuration does its probe-seek dance, so this is the app's live path, not a corner.
      el.onseeked = () => {
        if (Math.abs(el.currentTime - target) < SEEK_TOLERANCE_S) resolve();
      };
      el.onerror = () => reject(new Error("could not seek the clip"));
    });
    if (Math.abs(el.currentTime - target) >= SEEK_TOLERANCE_S) {
      el.currentTime = target;
      await withTimeout(seeked, CAPTURE_TIMEOUT_MS);
    }
    // else: already at the target (the unusable-duration fallback leaves us at 0). Assigning
    // currentTime the value it already holds fires no `seeked`, so awaiting one would only time out.

    if (!el.videoWidth || !el.videoHeight) return null;
    const { width, height } = thumbnailSize(el.videoWidth, el.videoHeight);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(el, 0, 0, width, height);
    return await new Promise<Blob | null>((resolve) =>
      canvas.toBlob((blob) => resolve(blob), "image/jpeg", JPEG_QUALITY)
    );
  } catch {
    return null;
  } finally {
    URL.revokeObjectURL(url);
    el.removeAttribute("src");
    el.load();
  }
}
/* c8 ignore stop */
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `yarn test src/test/lib.thumbnail.test.ts`
Expected: PASS

- [ ] **Step 5: Wire the thumbnail through the API client**

In `frontend/src/api.ts`, add to the `Analysis` interface (after `analysis_id`):

```ts
  /** A short-lived presigned URL for the source clip, attached to the analyze RESPONSE only —
   *  never stored in the history row, where it would already be expired on replay. Replays
   *  re-sign through `api.uploadMedia`. */
  video_url?: string | null;
```

Add after the `HistoryPage` interface:

```ts
// Short-lived URLs for one upload's stored artifacts, from GET /api/uploads/{id}/url.
export interface UploadMedia {
  video_url: string;
  thumbnail_url: string;
  expires_in: number;
}
```

Change `analyzeUpload` and `analyzePose` to take and forward the thumbnail:

```ts
  // No in-app caller today — App.tsx always goes through `analyzePose`. The `thumbnail`
  // parameter exists so reviving this path does not silently lose thumbnails.
  async analyzeUpload(file: File, movement: string, thumbnail?: Blob | null): Promise<Analysis> {
    const form = new FormData();
    form.append("file", file);
    // Which detector runs. The backend rejects an unregistered value with 400 before it spends
    // a MediaPipe pass, and echoes the canonical spelling back as `movement` on the result.
    form.append("movement", movement);
    // Optional by design: a browser where frame capture failed must still be able to analyze.
    if (thumbnail) form.append("thumbnail", thumbnail, "thumb.jpg");
    const res = await fetch("/api/analyze", {
      method: "POST",
      body: form,
      headers: await authHeader(),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error((detail as { detail?: string }).detail || `Analyze failed (${res.status})`);
    }
    return (await res.json()) as Analysis;
  },

  async analyzePose(
    movement: string,
    pose: PoseJson,
    video: Blob,
    thumbnail?: Blob | null
  ): Promise<Analysis> {
    const form = new FormData();
    form.append("movement", movement);
    form.append("pose", JSON.stringify(pose));
    const ext = video.type.includes("mp4") ? "mp4" : "webm";
    form.append("file", video, `capture.${ext}`);
    if (thumbnail) form.append("thumbnail", thumbnail, "thumb.jpg");
    const res = await fetch("/api/analyze/pose", {
      method: "POST",
      body: form,
      headers: await authHeader(),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error((detail as { detail?: string }).detail || `Analyze failed (${res.status})`);
    }
    return (await res.json()) as Analysis;
  },
```

Add the two URL helpers next to `videoFileUrl` (line 507):

```ts
  // Library demo clips only — uploads are not reachable here (they need an ownership check).
  videoFileUrl: (videoId: string) => `/api/video-file/${videoId}`,

  // Short-lived URLs for ONE of the caller's uploads (requires a session).
  uploadMedia: (videoId: string) =>
    getJSON<UploadMedia>(`/api/uploads/${encodeURIComponent(videoId)}/url`),

  // The same URLs for a whole history page in one round trip. Ids the caller does not own are
  // absent from `items` rather than an error.
  async uploadMediaBatch(
    videoIds: string[]
  ): Promise<Record<string, { video_url: string; thumbnail_url: string }>> {
    if (videoIds.length === 0) return {};
    const res = await fetch("/api/uploads/urls", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(await authHeader()) },
      body: JSON.stringify({ video_ids: videoIds }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for /api/uploads/urls`);
    const body = (await res.json()) as {
      items: Record<string, { video_url: string; thumbnail_url: string }>;
    };
    return body.items;
  },
```

- [ ] **Step 6: Capture the thumbnail on the app's upload path**

In `frontend/src/App.tsx`, add the import:

```ts
import { captureThumbnail } from "./lib/thumbnail";
```

and in `runPoseAnalysis`, capture alongside the pose extraction (replacing lines 122-125):

```ts
      const pose = await extractPoseFromBlob(blob, tier);
      // Captured from the same blob the browser just decoded for MediaPipe, so it costs one
      // extra seek. Resolves to null on any failure — a missing thumbnail never blocks analysis.
      const thumbnail = await captureThumbnail(blob);
      // The user's selected movement, not a hardcoded "Squat". `analyzePose` has taken a movement
      // since the client-capture path landed; this is the caller that finally supplies a real one.
      const data = await api.analyzePose(canonicalMovement, pose, blob, thumbnail);
```

- [ ] **Step 7: Write the failing API-client tests**

Add to `frontend/src/test/api.test.ts`:

```ts
describe("api.analyzeUpload thumbnail", () => {
  afterEach(() => vi.restoreAllMocks());

  it("appends the thumbnail when one was captured", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ video_id: "v" }), { status: 200 }));
    const file = new File(["v"], "squat.mp4", { type: "video/mp4" });
    await api.analyzeUpload(file, "Squat", new Blob(["jpeg"], { type: "image/jpeg" }));
    const form = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(form.get("thumbnail")).toBeInstanceOf(Blob);
  });

  it("omits the field when capture failed", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ video_id: "v" }), { status: 200 }));
    const file = new File(["v"], "squat.mp4", { type: "video/mp4" });
    await api.analyzeUpload(file, "Squat", null);
    const form = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(form.get("thumbnail")).toBeNull();
  });
});

describe("api.uploadMedia", () => {
  afterEach(() => vi.restoreAllMocks());

  it("fetches the ownership-checked URL endpoint", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({ video_url: "u", thumbnail_url: "t", expires_in: 3600 }),
        { status: 200 }
      )
    );
    const media = await api.uploadMedia("upload_a");
    expect(fetchMock.mock.calls[0][0]).toBe("/api/uploads/upload_a/url");
    expect(media.video_url).toBe("u");
  });
});

describe("api.uploadMediaBatch", () => {
  afterEach(() => vi.restoreAllMocks());

  it("posts every id and unwraps items", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({ items: { a: { video_url: "u", thumbnail_url: "t" } }, expires_in: 3600 }),
        { status: 200 }
      )
    );
    const items = await api.uploadMediaBatch(["a", "b"]);
    expect(JSON.parse(fetchMock.mock.calls[0][1]?.body as string)).toEqual({
      video_ids: ["a", "b"],
    });
    expect(items.a.video_url).toBe("u");
  });

  it("short-circuits an empty list without a request", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    expect(await api.uploadMediaBatch([])).toEqual({});
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
```

Add the equivalent thumbnail test to `frontend/src/test/api.pose.test.ts`:

```ts
describe("api.analyzePose thumbnail", () => {
  it("appends the thumbnail when one was captured", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ video_id: "v" }), { status: 200 }));
    await api.analyzePose("Squat", pose, new Blob(["v"], { type: "video/webm" }),
      new Blob(["jpeg"], { type: "image/jpeg" }));
    const form = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(form.get("thumbnail")).toBeInstanceOf(Blob);
  });
});
```

- [ ] **Step 8: Run the frontend suite**

Run: `yarn test`
Expected: PASS. `src/test/App.movement.test.tsx` may need `captureThumbnail` mocked — if it fails on a jsdom `<video>` error, add at the top of that file:

```ts
vi.mock("../lib/thumbnail", () => ({ captureThumbnail: () => Promise.resolve(null) }));
```

- [ ] **Step 9: Commit**

```bash
git add frontend/src/lib/thumbnail.ts frontend/src/api.ts frontend/src/App.tsx frontend/src/test/
git commit -m "feat(frontend): capture an upload thumbnail in the browser and post it"
```

---

### Task 7: Resolve the video source asynchronously

**Files:**
- Modify: `frontend/src/components/VideoPanel.tsx` (imports, add the hook, line 101-107)
- Modify: `frontend/src/test/components.VideoPanel.test.tsx`

**All commands in this task run with cwd = `frontend/`.**

**Interfaces:**
- Consumes: `api.uploadMedia`, `api.videoFileUrl`, `Analysis.video_url` (Task 6).
- Produces: no exported names — the hook is internal to `VideoPanel.tsx`.

- [ ] **Step 1: Write the failing VideoPanel tests**

The file already has `makeVideoRef()` and renders through `renderWithProviders` with `mockAnalysis` from `./fixtures`. Add a small local helper that reuses both, then the four cases. Append to `frontend/src/test/components.VideoPanel.test.tsx`:

```tsx
import { waitFor } from "@testing-library/react";
import { api, type Analysis } from "../api";

function renderPanel(analysis: Analysis) {
  return renderWithProviders(
    <VideoPanel
      analysis={analysis}
      videoRef={makeVideoRef()}
      onTimeUpdate={vi.fn()}
      onActiveFault={vi.fn()}
      onSeek={vi.fn()}
    />
  );
}

describe("VideoPanel video source", () => {
  afterEach(() => vi.restoreAllMocks());

  it("uses the local file endpoint for a library clip", () => {
    const media = vi.spyOn(api, "uploadMedia");
    renderPanel({ ...mockAnalysis, source: "library", video_id: "vid_001" });
    expect(document.querySelector("video")?.getAttribute("src")).toBe("/api/video-file/vid_001");
    expect(media).not.toHaveBeenCalled();
  });

  it("uses the presigned URL that came back with a fresh upload", () => {
    const media = vi.spyOn(api, "uploadMedia");
    renderPanel({
      ...mockAnalysis,
      source: "upload",
      video_id: "upload_a",
      video_url: "https://signed/source",
    });
    expect(document.querySelector("video")?.getAttribute("src")).toBe("https://signed/source");
    expect(media).not.toHaveBeenCalled();
  });

  it("re-signs on a history replay, where the stored result carries no URL", async () => {
    vi.spyOn(api, "uploadMedia").mockResolvedValue({
      video_url: "https://signed/replayed",
      thumbnail_url: "https://signed/thumb",
      expires_in: 3600,
    });
    renderPanel({ ...mockAnalysis, source: "upload", video_id: "upload_a" });
    await waitFor(() =>
      expect(document.querySelector("video")?.getAttribute("src")).toBe("https://signed/replayed")
    );
  });

  it("renders the analysis without playback when signing fails", async () => {
    vi.spyOn(api, "uploadMedia").mockRejectedValue(new Error("503"));
    renderPanel({ ...mockAnalysis, source: "upload", video_id: "upload_a" });
    await waitFor(() => expect(document.querySelector("video")).not.toBeNull());
    expect(document.querySelector("video")?.getAttribute("src")).toBeNull();
  });
});
```

Add `afterEach` to the existing `vitest` import at the top of the file if it is not already there.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `yarn test src/test/components.VideoPanel.test.tsx`
Expected: FAIL — the `src` is `/api/video-file/upload_a` for every case, because the component still builds it synchronously.

- [ ] **Step 3: Add the resolution hook**

In `frontend/src/components/VideoPanel.tsx`, add this above the `VideoPanel` component:

```tsx
/**
 * Where this analysis's video actually lives.
 *
 * Three sources, resolved in order of what is already known:
 *  - a library demo clip is a public file the backend streams directly;
 *  - a fresh upload's presigned URL rides along on the analyze response;
 *  - a history replay has neither, because storing a presigned URL in the row would mean
 *    replaying an expired one — so it re-signs through the ownership-checked endpoint.
 *
 * `null` while resolving and after a failure: the panel renders the analysis without playback
 * rather than blocking the page on storage.
 */
function useVideoSrc(analysis: Analysis): string | null {
  const { source, video_id: videoId, video_url: videoUrl } = analysis;
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    if (source === "library") {
      setSrc(api.videoFileUrl(videoId));
      return;
    }
    if (videoUrl) {
      setSrc(videoUrl);
      return;
    }
    let cancelled = false;
    setSrc(null);
    api
      .uploadMedia(videoId)
      .then((media) => {
        if (!cancelled) setSrc(media.video_url);
      })
      .catch(() => {
        if (!cancelled) setSrc(null);
      });
    return () => {
      cancelled = true;
    };
  }, [source, videoId, videoUrl]);

  return src;
}
```

Inside the component, call it below the existing `const { width, height } = analysis.metadata;`:

```tsx
  const videoSrc = useVideoSrc(analysis);
```

and change the `<video>` element (lines 101-107) to:

```tsx
          <video
            ref={videoRef}
            // Omitted entirely while unresolved: an empty `src` makes the browser re-request the
            // page URL as media and log a decode error.
            {...(videoSrc ? { src: videoSrc } : {})}
            className="absolute inset-0 w-full h-full object-contain"
            playsInline
            onClick={togglePlay}
          />
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `yarn test src/test/components.VideoPanel.test.tsx`
Expected: PASS

- [ ] **Step 5: Run the whole frontend suite**

Run: `yarn test`
Expected: PASS. Any suite that renders an upload analysis without mocking `api.uploadMedia` will now make a call — mock it there, returning a resolved `UploadMedia`.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/VideoPanel.tsx frontend/src/test/
git commit -m "feat(frontend): resolve upload playback through presigned URLs"
```

---

### Task 8: Thumbnails on the history page

**Files:**
- Create: `frontend/src/components/HistoryThumb.tsx`
- Modify: `frontend/src/pages/History.tsx` (load 35-46, row 192-194)
- Modify: `frontend/src/test/pages.History.test.tsx`

**All commands in this task run with cwd = `frontend/`.**

**Interfaces:**
- Consumes: `api.uploadMediaBatch` (Task 6).
- Produces: `HistoryThumb({ src }: { src?: string })` — a default-exported component.

- [ ] **Step 1: Write the failing history tests**

The file already has an `item(over)` row factory and a `renderHistory()` helper — use both. Append to `frontend/src/test/pages.History.test.tsx`:

```tsx
describe("History thumbnails", () => {
  afterEach(() => vi.restoreAllMocks());

  it("fetches every row's URLs in one batch request", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 2,
      items: [item({ id: "1", video_id: "upload_a" }), item({ id: "2", video_id: "upload_b" })],
    });
    const batch = vi.spyOn(api, "uploadMediaBatch").mockResolvedValue({});
    renderHistory();
    await waitFor(() => expect(batch).toHaveBeenCalledWith(["upload_a", "upload_b"]));
    expect(batch).toHaveBeenCalledTimes(1);
  });

  it("renders the thumbnail when one is available", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 1,
      items: [item({ id: "1", video_id: "upload_a" })],
    });
    vi.spyOn(api, "uploadMediaBatch").mockResolvedValue({
      upload_a: { video_url: "v", thumbnail_url: "https://signed/thumb" },
    });
    const { container } = renderHistory();
    await waitFor(() =>
      expect(container.querySelector("img")?.getAttribute("src")).toBe("https://signed/thumb")
    );
  });

  it("keeps the icon placeholder for a row with no thumbnail", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 1,
      items: [item({ id: "1", video_id: "upload_a" })],
    });
    vi.spyOn(api, "uploadMediaBatch").mockResolvedValue({});
    const { container } = renderHistory();
    await waitFor(() => expect(container.querySelectorAll("li").length).toBeGreaterThan(0));
    expect(container.querySelector("img")).toBeNull();
  });

  it("still renders the list when the URL batch fails", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 1,
      items: [item({ id: "1", video_id: "upload_a" })],
    });
    vi.spyOn(api, "uploadMediaBatch").mockRejectedValue(new Error("503"));
    const { container } = renderHistory();
    await waitFor(() => expect(container.querySelectorAll("li").length).toBeGreaterThan(0));
    expect(screen.queryByText(/errorTitle/)).toBeNull();
  });
});
```

Note: the existing suites in this file do NOT stub `api.uploadMediaBatch`, so once `History` calls it every one of them will hit the real function. Add a `beforeEach` at the top of the file stubbing it to `{}` so the pre-existing tests keep passing.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `yarn test src/test/pages.History.test.tsx`
Expected: FAIL — `api.uploadMediaBatch` is never called.

- [ ] **Step 3: Write the thumbnail component**

Create `frontend/src/components/HistoryThumb.tsx`:

```tsx
import { useState } from "react";
import { PersonSimpleRun } from "@phosphor-icons/react";

/**
 * A history row's leading tile: the upload's captured frame, falling back to the movement icon.
 *
 * The fallback covers three cases with one branch — a row from before thumbnails existed, a
 * capture that failed in the browser, and a signed URL that 404s because the object is not
 * there. Probing for existence before rendering would cost a request per row to avoid an
 * occasional broken image, so the `onError` handler carries it instead.
 */
export default function HistoryThumb({ src }: { src?: string }) {
  const [failed, setFailed] = useState(false);

  if (!src || failed) {
    return (
      <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
        <PersonSimpleRun size={22} weight="duotone" />
      </span>
    );
  }
  return (
    <img
      src={src}
      alt=""
      loading="lazy"
      onError={() => setFailed(true)}
      className="h-10 w-10 shrink-0 rounded-lg object-cover bg-content/5"
    />
  );
}
```

- [ ] **Step 4: Load the URLs and render them**

In `frontend/src/pages/History.tsx`, add the import:

```tsx
import HistoryThumb from "../components/HistoryThumb";
```

Add state beside the others (after `deleteError`):

```tsx
  // Thumbnail URLs, keyed by video_id. Fetched in ONE batch for the whole page rather than per
  // row: 50 rows would otherwise mean 50 presign requests. A failure here is silent — the rows
  // still render, just with their icon placeholders.
  const [thumbs, setThumbs] = useState<Record<string, string>>({});
```

Extend `load` to fetch them after the list arrives:

```tsx
  const load = useCallback(async () => {
    setStatus("loading");
    setError("");
    try {
      const page = await api.listAnalyses();
      setItems(page.items);
      setStatus("ready");
      const ids = page.items.map((it) => it.video_id);
      try {
        const media = await api.uploadMediaBatch(ids);
        setThumbs(
          Object.fromEntries(Object.entries(media).map(([id, m]) => [id, m.thumbnail_url]))
        );
      } catch {
        // Thumbnails are decoration. A storage problem must not turn a readable history page
        // into an error state.
        setThumbs({});
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  }, []);
```

Replace the static icon span (lines 192-194) with:

```tsx
                          <HistoryThumb src={thumbs[it.video_id]} />
```

`PersonSimpleRun` is no longer used directly in `History.tsx` — remove it from the `@phosphor-icons/react` import there.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `yarn test src/test/pages.History.test.tsx`
Expected: PASS

- [ ] **Step 6: Run the whole frontend suite with coverage**

Run: `yarn test:coverage`
Expected: PASS, thresholds met.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/HistoryThumb.tsx frontend/src/pages/History.tsx frontend/src/test/
git commit -m "feat(frontend): show upload thumbnails on the history page"
```

---

### Task 9: Documentation and full CI parity

**Files:**
- Modify: `.env.example`
- Modify: `backend/README.md:102, 113`
- Modify: `.gitignore` (only if `data/runtime/` is not already covered)

**Interfaces:**
- Consumes: everything above.
- Produces: no code.

- [ ] **Step 1: Document the R2 settings**

Append to `.env.example`, following the file's existing comment style:

```
# --- Cloudflare R2 object storage (user uploads: raw video, pose JSON, thumbnail) ------------
# Leave these blank and the backend stores objects on the local filesystem under
# data/runtime/objects/ instead — which is what CI and offline development run on, so you do
# not need an R2 account to work on this codebase.
# Deploy prerequisites: create the bucket, mint an API token scoped to it, and add a lifecycle
# rule expiring `uploads/anon/` after 7 days (anonymous demo uploads are never referenced again).
R2_ACCOUNT_ID=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET=
```

- [ ] **Step 2: Update the backend README**

In `backend/README.md`, change line 102 from:

```
  config.py          repo-root paths + runtime/upload dirs
```

to:

```
  config.py          repo-root paths + the runtime scratch dir
```

Add a `storage.py` line to the `services/` block:

```
    storage.py       object storage for uploads (local filesystem or Cloudflare R2)
```

Replace line 113:

```
Uploaded videos and their derived pose JSON land in `data/runtime/` (gitignored).
```

with:

```
Uploaded videos, their derived pose JSON, and a browser-captured thumbnail are stored in
Cloudflare R2 under `uploads/{owner}/{video_id}/`. With `R2_*` unset the same objects land on
the local filesystem under `data/runtime/objects/` (gitignored), so no credentials are needed
for development or CI. The pipeline still needs real file paths, so each upload is staged into
a temp directory for the duration of its analysis and removed afterwards.

Uploads are read back through `GET /api/uploads/{video_id}/url`, which resolves the storage key
with the caller's own JWT so Postgres RLS enforces ownership. `GET /api/video-file/{video_id}`
serves library demo clips only.
```

- [ ] **Step 3: Confirm the local object dir is gitignored**

Run: `git check-ignore -v data/runtime/objects/x`
Expected: a matching `.gitignore` rule is printed. If nothing is printed, add `data/runtime/` to `.gitignore`.

- [ ] **Step 4: Run the backend suite and coverage gate**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS

Run: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
Expected: PASS at ≥95%

- [ ] **Step 5: Run the frontend suite with coverage**

With cwd = `frontend/`:

Run: `yarn test:coverage`
Expected: PASS, thresholds met

Run: `yarn build`
Expected: builds clean (this catches TypeScript errors vitest does not)

- [ ] **Step 6: Commit**

```bash
git add .env.example backend/README.md .gitignore
git commit -m "docs(storage): document the R2 settings and the new upload storage layout"
```

---

## Verification checklist

Before declaring the branch done, confirm each of these by running the command, not by reading the code:

- [ ] `.venv\Scripts\python.exe -m pytest tests/ -q` passes
- [ ] `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95` passes
- [ ] `yarn test:coverage` passes (cwd `frontend/`)
- [ ] `yarn build` succeeds (cwd `frontend/`)
- [ ] `grep -rn "UPLOAD_DIR\|UPLOAD_POSE_DIR\|ensure_runtime_dirs\|save_upload\|uploaded_video_path" backend/ tests/` returns nothing
- [ ] `grep -rn "runtime/uploads" backend/` returns nothing
- [ ] With `R2_*` unset, a real upload through the running app plays back, and `data/runtime/objects/uploads/` contains `source`, `pose.json`, and `thumb.jpg` for it
