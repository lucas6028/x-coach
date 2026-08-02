"""/api/analyze/pose: accept client pose JSON + video, run the detector off the event loop."""
from __future__ import annotations

import asyncio
import io
import json
import threading
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from backend.app.routers import analyze as analyze_router
from backend.app.services import analysis as analysis_service
from backend.app.services import runtime_config
from backend.app.services import storage
from backend.app.services import store

_GOOD_POSE = json.dumps({"metadata": {"fps": 30, "width": 1, "height": 1, "total_frames": 0}, "frames": []})


def _upload(filename: str = "clip.webm", data: bytes = b"fake") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


class AnalyzePoseEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_stage = analysis_service.stage_upload
        self._orig_artifacts = analysis_service.store_artifacts
        self._orig_discard = analysis_service.discard_stage
        self._orig_analyze = analysis_service.analyze_pose_payload

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
        analysis_service.store_artifacts = lambda staged, *, thumbnail=None: (
            self.artifacts.append({"staged": staged, "thumbnail": thumbnail}) or 0
        )
        analysis_service.discard_stage = lambda staged: self.discarded.append(staged)
        analysis_service.analyze_pose_payload = (
            lambda payload, *, movement, video_id=None, pose_json_path=None, max_reps=-1: {
                "video_id": video_id, "source": "upload", "movement": movement, "detections": [],
            }
        )

        # Presigning is a storage concern; stub it so these tests stay offline.
        presign = mock.patch.object(
            analyze_router, "_source_url", side_effect=lambda prefix: f"https://signed/{prefix}"
        )
        presign.start()
        self.addCleanup(presign.stop)

        # Reaping a failed upload's objects is a storage concern too: unpatched it resolves the
        # LIVE object store, which is an R2 client (and a real network delete) on any machine
        # whose env carries R2 credentials. See AnalyzePoseStorageTests for what it should do.
        reap = mock.patch.object(store, "_reap_objects")
        reap.start()
        self.addCleanup(reap.stop)

        # KEEP THESE TESTS OFFLINE, as the module docstring promises. ``analyze_pose`` calls
        # ``settings.allowed_upload_suffixes()``, which reads the admin overrides via
        # ``runtime_config.get_overrides()`` -- and that does a REAL Supabase round-trip whenever
        # auth is configured. ``{}`` is exactly what ``get_overrides`` returns when auth is
        # unconfigured, so this runs the same code path CI does rather than a bespoke stub.
        overrides = mock.patch.object(runtime_config, "get_overrides", return_value={})
        overrides.start()
        self.addCleanup(overrides.stop)

    def tearDown(self) -> None:
        analysis_service.stage_upload = self._orig_stage
        analysis_service.store_artifacts = self._orig_artifacts
        analysis_service.discard_stage = self._orig_discard
        analysis_service.analyze_pose_payload = self._orig_analyze

    # These tests invoke ``analyze_pose`` directly (not via FastAPI), so the ``max_reps``/
    # ``thumbnail`` Form/File defaults are not resolved by FastAPI's DI -- pass ``max_reps=None``
    # and ``thumbnail=None`` explicitly, since an unresolved ``Form(...)``/``File(...)`` sentinel
    # would otherwise reach ``_validated_max_reps`` / ``_read_thumbnail`` verbatim.

    def test_happy_path_returns_analysis(self) -> None:
        result = asyncio.run(
            analyze_router.analyze_pose(
                "Squat", _GOOD_POSE, _upload(), max_reps=None, thumbnail=None, user=None
            )
        )
        self.assertEqual(result["video_id"], "upload_test")
        self.assertEqual(result["movement"], "Squat")

    def test_rejects_bad_json(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", "{not json", _upload(), max_reps=None, thumbnail=None, user=None
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_pose_without_frames_list(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", json.dumps({"metadata": {}}), _upload(),
                    max_reps=None, thumbnail=None, user=None,
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_malformed_landmarks(self) -> None:
        bad = json.dumps({"metadata": {}, "frames": [{"landmarks": [{"x": 1}]}]})
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", bad, _upload(), max_reps=None, thumbnail=None, user=None
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_unsupported_suffix(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", _GOOD_POSE, _upload("x.txt"),
                    max_reps=None, thumbnail=None, user=None,
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_empty_file(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", _GOOD_POSE, _upload(data=b""),
                    max_reps=None, thumbnail=None, user=None,
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_runtime_error_maps_to_422(self) -> None:
        def boom(payload, *, movement, video_id=None, pose_json_path=None, max_reps=-1):
            raise RuntimeError("boom")

        analysis_service.analyze_pose_payload = boom
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze_pose(
                    "Squat", _GOOD_POSE, _upload(), max_reps=None, thumbnail=None, user=None
                )
            )
        self.assertEqual(ctx.exception.status_code, 422)

    def test_runs_off_the_event_loop(self) -> None:
        seen: dict[str, threading.Thread] = {}

        def record(payload, *, movement, video_id=None, pose_json_path=None, max_reps=-1):
            seen["t"] = threading.current_thread()
            return {"video_id": video_id, "source": "upload", "detections": []}

        analysis_service.analyze_pose_payload = record
        asyncio.run(
            analyze_router.analyze_pose(
                "Squat", _GOOD_POSE, _upload(), max_reps=None, thumbnail=None, user=None
            )
        )
        self.assertIsNot(seen["t"], threading.main_thread())

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
                thumbnail=None,
                user=None,
            )
        )
        self.assertEqual(len(self.artifacts), 1)


class AnalyzePoseStorageTests(unittest.TestCase):
    """The object-storage contract of /api/analyze/pose: mirrors ``AnalyzeStorageTests`` in
    ``tests/test_analyze_endpoint.py`` -- both endpoints now share ``_stage_analyze_persist``,
    but each endpoint's own coverage still has to exist independently: nothing stops a future
    change to ``analyze_pose``'s call site (e.g. dropping the ``user``/``thumb`` kwargs it passes
    into the shared helper) from breaking this endpoint alone while `/api/analyze`'s tests stay
    green.
    """

    def setUp(self) -> None:
        self._orig = {
            name: getattr(analysis_service, name)
            for name in ("stage_upload", "store_artifacts", "discard_stage", "analyze_pose_payload")
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
        self.stage_calls: list[dict] = []

        def _stage_upload(data, *, suffix=".mp4", owner="anon"):
            self.stage_calls.append({"data": data, "suffix": suffix, "owner": owner})
            return self.staged

        analysis_service.stage_upload = _stage_upload
        analysis_service.store_artifacts = lambda staged, *, thumbnail=None: (
            self.artifacts.append({"staged": staged, "thumbnail": thumbnail}) or 0
        )
        analysis_service.discard_stage = lambda staged: self.discarded.append(staged)
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

        # Stubbed for the whole class, both to keep the reap offline (unpatched it resolves the
        # LIVE object store) and so every test can assert on WHETHER it fired, not only the ones
        # that expect it to.
        reap = mock.patch.object(store, "_reap_objects")
        self.reap = reap.start()
        self.addCleanup(reap.stop)

    def tearDown(self) -> None:
        for name, value in self._orig.items():
            setattr(analysis_service, name, value)

    def _run(self, **kwargs):
        params = {
            "movement": "Squat",
            "pose": _GOOD_POSE,
            "file": _upload(),
            "max_reps": None,
            "thumbnail": None,
            "user": None,
        }
        params.update(kwargs)
        return asyncio.run(analyze_router.analyze_pose(**params))

    def test_returns_a_presigned_video_url(self) -> None:
        result = self._run()
        self.assertEqual(result["video_url"], "https://signed/uploads/anon/upload_test")

    def test_a_storage_failure_before_analysis_is_a_503(self) -> None:
        def boom(data, *, suffix=".mp4", owner="anon"):
            raise storage.StorageError("R2 down")

        analysis_service.stage_upload = boom
        ran = []
        analysis_service.analyze_pose_payload = lambda *a, **k: ran.append(1) or {}
        with self.assertRaises(HTTPException) as ctx:
            self._run()
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ran, [], "no CPU may be spent on a clip whose video could not be stored")

    def test_stores_the_derived_artifacts_after_a_successful_analysis(self) -> None:
        self._run()
        self.assertEqual(len(self.artifacts), 1)
        self.assertIs(self.artifacts[0]["staged"], self.staged)

    def test_forwards_the_thumbnail_bytes(self) -> None:
        """The mirror of ``/api/analyze``'s own test, for the reason this class exists: nothing
        stops a change to ``analyze_pose``'s call site from dropping the ``thumb`` kwarg it
        passes into the shared helper while the other endpoint's tests stay green."""
        thumb = UploadFile(file=io.BytesIO(b"jpeg-bytes"), filename="t.jpg",
                           headers=Headers({"content-type": "image/jpeg"}))
        self._run(thumbnail=thumb)
        self.assertEqual(self.artifacts[0]["thumbnail"], b"jpeg-bytes")

    def test_accepts_the_image_jpg_content_type_alias(self) -> None:
        thumb = UploadFile(file=io.BytesIO(b"jpeg-bytes"), filename="t.jpg",
                           headers=Headers({"content-type": "image/jpg"}))
        self._run(thumbnail=thumb)
        self.assertEqual(self.artifacts[0]["thumbnail"], b"jpeg-bytes")

    def test_drops_a_thumbnail_with_an_unusable_type_instead_of_failing(self) -> None:
        thumb = UploadFile(file=io.BytesIO(b"png"), filename="t.png",
                           headers=Headers({"content-type": "image/png"}))
        result = self._run(thumbnail=thumb)
        self.assertEqual(result["video_id"], "upload_test")
        self.assertIsNone(self.artifacts[0]["thumbnail"], "the bad thumbnail must be dropped")

    def test_drops_an_oversized_thumbnail_instead_of_failing(self) -> None:
        big = b"x" * (analyze_router.MAX_THUMBNAIL_BYTES + 1)
        thumb = UploadFile(file=io.BytesIO(big), filename="t.jpg",
                           headers=Headers({"content-type": "image/jpeg"}))
        result = self._run(thumbnail=thumb)
        self.assertEqual(result["video_id"], "upload_test")
        self.assertIsNone(self.artifacts[0]["thumbnail"], "the oversized thumbnail is dropped")

    def test_reaps_the_stored_objects_when_the_analysis_fails(self) -> None:
        """The source is stored BEFORE the analysis runs and a failed analysis writes no
        ``videos`` row, so nothing else would ever delete it."""
        def boom(*args, **kwargs):
            raise RuntimeError("detector failed")

        analysis_service.analyze_pose_payload = boom
        with self.assertRaises(HTTPException):
            self._run()
        self.reap.assert_called_once_with([self.staged.prefix])

    def test_reaps_the_stored_objects_when_the_analysis_fails_unexpectedly(self) -> None:
        def boom(*args, **kwargs):
            raise ValueError("something nobody predicted")

        analysis_service.analyze_pose_payload = boom
        with self.assertRaises(ValueError):
            self._run()
        self.reap.assert_called_once_with([self.staged.prefix])

    def test_never_reaps_after_a_successful_analysis(self) -> None:
        """Pins WHERE the reap sits: one that also fired on success would delete the clip the
        caller is being handed a live ``video_url`` for."""
        self._run()
        self.reap.assert_not_called()

    def test_never_reaps_when_only_the_history_write_failed(self) -> None:
        """A documented, accepted orphan: the analysis succeeded and the client holds a live
        playback URL, so deleting the source inline would break a working session."""
        user = mock.Mock(id="u1", token="tok")
        with mock.patch.object(store, "persist_analysis", side_effect=RuntimeError("db down")):
            result = self._run(user=user)
        self.assertIsNone(result["analysis_id"])
        self.reap.assert_not_called()

    def test_discards_the_stage_even_when_the_analysis_fails(self) -> None:
        def boom(*args, **kwargs):
            raise RuntimeError("detector failed")

        analysis_service.analyze_pose_payload = boom
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

    def test_stages_under_the_anon_owner_for_an_anonymous_caller(self) -> None:
        self._run()
        self.assertEqual(self.stage_calls[0]["owner"], "anon")

    def test_stages_under_the_authenticated_users_id(self) -> None:
        user = mock.Mock(id="u1", token="tok")
        with mock.patch.object(store, "persist_analysis", return_value="id"):
            self._run(user=user)
        self.assertEqual(self.stage_calls[0]["owner"], "u1")


if __name__ == "__main__":
    unittest.main()
