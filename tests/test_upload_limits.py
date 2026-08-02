"""Per-file upload cap and per-user storage quota.

Offline throughout — the Supabase client is a fake and the object store is never reached.
"""

from __future__ import annotations

import asyncio
import io
import unittest
from unittest import mock

from fastapi import HTTPException
from starlette.datastructures import UploadFile

from backend.app.routers import analyze as analyze_router
from backend.app.services import analysis as analysis_service
from backend.app.services import runtime_config
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
