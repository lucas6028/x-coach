"""Per-file upload cap and per-user storage quota.

Offline throughout — the Supabase client is a fake and the object store is never reached.
"""

from __future__ import annotations

import asyncio
import io
import unittest
from unittest import mock

from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from backend.app.auth import CurrentUser, get_optional_user
from backend.app.main import app
from backend.app.routers import analyze as analyze_router
from backend.app.services import analysis as analysis_service
from backend.app.services import runtime_config
from backend.app.services import store


_MISSING = object()


class _Result:
    def __init__(self, data=None, count=_MISSING):
        self.data = data if data is not None else []
        # ``count`` defaults to "the page is complete" so every existing case keeps meaning what it
        # meant before ``get_storage_used`` started asking for an exact count. Pass it explicitly to
        # model a db-max-rows truncation (count > len(data)) or an omitted count (None).
        self.count = len(self.data) if count is _MISSING else count


class _Table:
    def __init__(self, name, responses, log):
        self._name, self._responses, self._log = name, responses, log
        self._op = None

    # Every entry is a uniform 4-tuple ``(table, op, payload_or_args, kwargs)`` so a consumer can
    # unpack the whole log without knowing which op produced a given row.
    def select(self, *args, **kwargs):
        self._op = "select"
        self._log.append((self._name, "select", args, kwargs))
        return self

    def upsert(self, payload, **kwargs):
        self._op = "upsert"
        self._log.append((self._name, "upsert", payload, kwargs))
        return self

    def insert(self, payload, **kwargs):
        self._op = "insert"
        self._log.append((self._name, "insert", payload, kwargs))
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
    def _used(self, rows, *, count=_MISSING):
        client = _Client({("videos", "select"): _Result(data=rows, count=count)}, [])
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

    def test_a_truncated_page_raises_rather_than_undercounting(self) -> None:
        """A PostgREST ``db-max-rows`` truncation must NOT silently return a smaller total.

        Returning the partial sum would be a fail-OPEN in a feature that fails closed everywhere
        else: the router treats a low usage figure as headroom. The raise is turned into the same
        503 a failed usage query already produces.
        """
        with self.assertRaises(RuntimeError):
            self._used([{"size_bytes": 10}, {"size_bytes": 32}], count=500)

    def test_an_omitted_count_is_not_treated_as_truncation(self) -> None:
        """PostgREST may answer with no count at all; that must stay an ordinary read, not a 503."""
        self.assertEqual(self._used([{"size_bytes": 10}, {"size_bytes": 32}], count=None), 42)

    def test_asks_postgrest_for_the_exact_count(self) -> None:
        """The truncation guard is only possible because the select requests an exact count."""
        log: list[tuple] = []
        client = _Client({("videos", "select"): _Result(data=[])}, log)
        with mock.patch.object(store, "_user_client", return_value=client):
            store.get_storage_used(token="t", user_id="u1")
        selects = [kwargs for name, op, _args, kwargs in log if name == "videos" and op == "select"]
        self.assertEqual(selects, [{"count": "exact"}])


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
        upserts = [payload for name, op, payload, _kw in log if name == "videos" and op == "upsert"]
        self.assertEqual(len(upserts), 1)
        self.assertEqual(upserts[0]["size_bytes"], 1234)


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
            lambda payload, *, movement, video_id=None, pose_json_path=None, max_reps=-1: {
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


class AsMbTests(unittest.TestCase):
    def test_a_partial_megabyte_rounds_up(self) -> None:
        """Asserted DIRECTLY, on a non-multiple, because every limit used elsewhere in this file is
        a whole number of MB — where floor and ceiling division agree, so nothing there can tell a
        reverted ``_as_mb`` from the real one. Reporting a 1 MB + 1 byte cap as "1 MB" would name a
        limit SMALLER than the one actually enforced."""
        self.assertEqual(analyze_router._as_mb(1024 * 1024 + 1), 2)

    def test_an_exact_megabyte_does_not_round_up(self) -> None:
        self.assertEqual(analyze_router._as_mb(1024 * 1024), 1)


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

    def test_an_upload_that_exactly_fills_the_remaining_space_is_accepted(self) -> None:
        """ON the boundary, not near it: ``used + len(data) == quota`` exactly. With any slack the
        test survives ``>`` becoming ``>=``, which would wrongly refuse an upload that fits."""
        quota = 10 * 1024 * 1024
        with mock.patch.object(runtime_config, "get_overrides",
                               return_value={"user_storage_quota_bytes": quota}):
            with mock.patch.object(store, "get_storage_used", return_value=quota - 1024):
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


class ResponseBodyShapeTests(unittest.TestCase):
    """The 413 ``detail`` dict has so far only ever been asserted as an in-process
    ``HTTPException.detail`` (see ``PerFileSizeCapTests`` / ``StorageQuotaTests`` above), which
    proves what the *router* raises but not what a real HTTP client actually receives.
    ``frontend/src/api.ts``'s ``uploadLimitError`` parser depends on FastAPI nesting that dict
    under a top-level ``detail`` key -- i.e. a JSON body shaped ``{"detail": {"code": ...,
    "limit_mb": ...}}``. These tests go through a genuine ``TestClient`` request against the real
    app (not a direct router call) to round-trip that assumption through actual ASGI/JSON
    encoding, for both refusal codes.
    """

    def setUp(self) -> None:
        self.client = TestClient(app)
        # KEEP OFFLINE, same reasoning as ``_StubbedAnalyzePath`` above: unconfigured auth is
        # exactly what CI runs, and ``{}`` is what ``get_overrides`` returns in that case.
        overrides = mock.patch.object(runtime_config, "get_overrides", return_value={})
        overrides.start()
        self.addCleanup(overrides.stop)

    def test_upload_too_large_413_body_is_a_nested_detail_dict(self) -> None:
        limit = 1 * 1024 * 1024  # the clamped floor, same as PerFileSizeCapTests above
        with mock.patch.object(
            runtime_config, "get_overrides", return_value={"max_upload_bytes": limit}
        ):
            resp = self.client.post(
                "/api/analyze",
                files={"file": ("clip.mp4", b"x" * (limit + 1), "video/mp4")},
            )
        self.assertEqual(resp.status_code, 413)
        self.assertEqual(resp.json(), {"detail": {"code": "upload_too_large", "limit_mb": 1}})

    def test_storage_quota_exceeded_413_body_is_a_nested_detail_dict(self) -> None:
        quota = 10 * 1024 * 1024  # the clamped floor
        app.dependency_overrides[get_optional_user] = lambda: CurrentUser(id="u1", token="jwt")
        self.addCleanup(app.dependency_overrides.pop, get_optional_user, None)
        with mock.patch.object(
            runtime_config, "get_overrides", return_value={"user_storage_quota_bytes": quota}
        ):
            with mock.patch.object(store, "get_storage_used", return_value=8 * 1024 * 1024):
                resp = self.client.post(
                    "/api/analyze",
                    files={"file": ("clip.mp4", b"x" * (3 * 1024 * 1024), "video/mp4")},
                )
        self.assertEqual(resp.status_code, 413)
        self.assertEqual(
            resp.json(),
            {"detail": {"code": "storage_quota_exceeded", "used_mb": 8, "limit_mb": 10}},
        )
