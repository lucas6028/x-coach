"""Tests for the live-upload ``/api/analyze`` endpoint's P0 concurrency behaviour.

These call ``backend.app.routers.analyze.analyze`` directly as a coroutine and monkeypatch the
pipeline, so the test never imports MediaPipe / OpenCV / torch. They lock in two guarantees of
the P0 fix:

* the blocking pipeline runs **off** the event loop (in a worker thread), so one analysis no
  longer freezes every other request, and
* concurrent analyses are **bounded** by the semaphore, so uploads queue instead of exhausting
  the machine.

Plus the unchanged request contract (suffix / empty-file validation, RuntimeError -> 422, and
the analysis payload passing through untouched).
"""

from __future__ import annotations

import asyncio
import io
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


def _upload(filename: str = "clip.mp4", data: bytes = b"fake-video-bytes") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


# These tests invoke ``analyze`` directly (not via FastAPI), so neither the ``user`` dependency
# nor the ``movement``/``max_reps``/``thumbnail`` Form/File defaults are resolved by FastAPI's DI
# -- pass ``user=None`` to drive the anonymous "demo" path (no persistence), ``movement="Squat"``
# explicitly, ``max_reps=None`` explicitly, and ``thumbnail=None`` explicitly, since an unresolved
# ``Form(...)``/``File(...)`` sentinel would otherwise reach ``_validated_movement`` /
# ``_validated_max_reps`` / ``_read_thumbnail`` verbatim.


class AnalyzeEndpointTests(unittest.TestCase):
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

        # Reaping a failed upload's objects is a storage concern too: unpatched it resolves the
        # LIVE object store, which is an R2 client (and a real network delete) on any machine
        # whose env carries R2 credentials. See AnalyzeStorageTests for what it should do.
        reap = mock.patch.object(store, "_reap_objects")
        reap.start()
        self.addCleanup(reap.stop)

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

    # --- request contract / error mapping ----------------------------------------

    def test_rejects_unsupported_suffix(self) -> None:
        analysis_service.analyze_video_file = lambda *a, **k: {}
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze(
                    _upload("notes.txt"), movement="Squat", max_reps=None, thumbnail=None, user=None
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_empty_file(self) -> None:
        analysis_service.analyze_video_file = lambda *a, **k: {}
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze(
                    _upload("clip.mp4", data=b""), movement="Squat", max_reps=None, thumbnail=None, user=None
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_runtime_error_maps_to_422(self) -> None:
        def boom(*args, **kwargs):
            raise RuntimeError("Pose extraction failed")

        analysis_service.analyze_video_file = boom
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze(_upload(), movement="Squat", max_reps=None, thumbnail=None, user=None)
            )
        self.assertEqual(ctx.exception.status_code, 422)

    def test_returns_analysis_payload_unchanged(self) -> None:
        analysis_service.analyze_video_file = (
            lambda path, *, video_id=None, pose_json_path=None, movement=None, max_reps=-1: {
                "video_id": video_id,
                "source": "upload",
                "detections": [],
            }
        )
        result = asyncio.run(
            analyze_router.analyze(_upload(), movement="Squat", max_reps=None, thumbnail=None, user=None)
        )
        self.assertEqual(result["video_id"], "upload_test")
        self.assertEqual(result["source"], "upload")

    # --- the P0 guarantees --------------------------------------------------------

    def test_pipeline_runs_off_the_event_loop(self) -> None:
        """The blocking pipeline must execute in a worker thread, not the loop thread."""
        seen: dict[str, threading.Thread] = {}

        def record_thread(path, *, video_id=None, pose_json_path=None, movement=None, max_reps=-1):
            seen["thread"] = threading.current_thread()
            return {"video_id": video_id, "source": "upload"}

        analysis_service.analyze_video_file = record_thread
        asyncio.run(
            analyze_router.analyze(_upload(), movement="Squat", max_reps=None, thumbnail=None, user=None)
        )
        self.assertIsNot(seen["thread"], threading.main_thread())

    def test_concurrent_analyses_are_bounded(self) -> None:
        """At most MAX_CONCURRENT_ANALYSES run at once, and the cap is genuinely reached.

        THE BARRIER IS THE PROOF, NOT A WALL CLOCK. Each call parks in ``blocking`` until
        ``limit`` calls are inside SIMULTANEOUSLY. If the semaphore admits fewer, the barrier
        times out and breaks, which the ``broken`` assertion reports; if it admits more, ``peak``
        exceeds ``limit``. Both directions fail, and neither depends on how fast the machine is.

        The previous version polled a 5s deadline for ``peak`` to reach the cap and then let
        everything drain regardless -- so anything slow upstream could exhaust the deadline
        before the first analysis even started, leaving ``peak`` at 0 and the cap never observed.
        That is what made this test look like a "load-dependent flake": the symptom was right,
        but the cause was a real Supabase round-trip in ``allowed_upload_suffixes`` (now stubbed
        in ``setUp``), not machine load. A deadline race would still be fragile on a loaded CI
        box even with the network gone, so the timing design is replaced rather than patched.
        """
        limit = 2
        # A MULTIPLE of `limit`, so every barrier cycle fills exactly and none is left with a
        # partial batch that could never satisfy it. `limit` slots are held while the first batch
        # parks, so the remainder genuinely queue on the semaphore rather than sailing through.
        total = limit * 2
        lock = threading.Lock()
        state = {"active": 0, "peak": 0, "broken": False}
        # Generous enough not to trip on a loaded machine, small enough that a genuine failure
        # reports in seconds instead of adding a multiple of it to every suite run.
        barrier = threading.Barrier(limit, timeout=10)

        def blocking(path, *, video_id=None, pose_json_path=None, movement=None, max_reps=-1):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                # Returns only once `limit` calls are here together. Swallowed rather than raised
                # so the failure surfaces as a readable assertion, not a BrokenBarrierError
                # escaping through the threadpool.
                barrier.wait()
            except threading.BrokenBarrierError:
                state["broken"] = True
            with lock:
                state["active"] -= 1
            return {"video_id": video_id, "source": "upload"}

        analysis_service.analyze_video_file = blocking

        async def drive() -> None:
            # Bind a fresh semaphore (sized to `limit`) to this event loop.
            analyze_router._ANALYSIS_SEMAPHORE = asyncio.Semaphore(limit)
            await asyncio.gather(
                *(
                    analyze_router.analyze(
                        _upload(f"c{i}.mp4"), movement="Squat", max_reps=None, thumbnail=None, user=None
                    )
                    for i in range(total)
                )
            )

        asyncio.run(drive())
        self.assertFalse(
            state["broken"],
            # ASCII only: this string is read off a console on failure, and a Windows terminal
            # mangles non-ASCII punctuation into noise exactly when someone needs to read it.
            f"fewer than {limit} analyses were ever in flight together - the semaphore admitted "
            f"too few, so the concurrency cap this test exists to pin was never exercised",
        )
        # Never more than `limit` ran simultaneously (the barrier already proved not fewer).
        self.assertEqual(state["peak"], limit)


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
        self.stage_calls: list[dict] = []

        def _stage_upload(data, *, suffix=".mp4", owner="anon"):
            self.stage_calls.append({"data": data, "suffix": suffix, "owner": owner})
            return self.staged

        analysis_service.stage_upload = _stage_upload
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
        params = {"movement": "Squat", "max_reps": None, "thumbnail": None, "user": None}
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

    def test_accepts_the_image_jpg_content_type_alias(self) -> None:
        """``image/jpg`` is not the registered type, but encoders emit it. Since an unusable
        thumbnail now degrades rather than failing, accepting the alias can only help."""
        thumb = UploadFile(file=io.BytesIO(b"jpeg-bytes"), filename="t.jpg",
                           headers=Headers({"content-type": "image/jpg"}))
        self._run(thumbnail=thumb)
        self.assertEqual(self.artifacts[0]["thumbnail"], b"jpeg-bytes")

    def test_drops_a_thumbnail_with_an_unusable_type_instead_of_failing(self) -> None:
        """A thumbnail problem must NEVER block an analysis -- the contract the frontend's
        capture helper holds on its own side. Rejecting the whole upload over the one optional
        part would invert it for exactly the client whose capture succeeded."""
        thumb = UploadFile(file=io.BytesIO(b"png"), filename="t.png",
                           headers=Headers({"content-type": "image/png"}))
        result = self._run(thumbnail=thumb)
        self.assertEqual(result["video_id"], "upload_test")
        self.assertIsNone(self.artifacts[0]["thumbnail"], "the bad thumbnail must be dropped")

    def test_drops_an_oversized_thumbnail_instead_of_failing(self) -> None:
        """The cap is still a cap -- exceeding it means 'no thumbnail', not 'no analysis'."""
        big = b"x" * (analyze_router.MAX_THUMBNAIL_BYTES + 1)
        thumb = UploadFile(file=io.BytesIO(big), filename="t.jpg",
                           headers=Headers({"content-type": "image/jpeg"}))
        result = self._run(thumbnail=thumb)
        self.assertEqual(result["video_id"], "upload_test")
        self.assertIsNone(self.artifacts[0]["thumbnail"], "the oversized thumbnail is dropped")

    def test_reaps_the_stored_objects_when_the_analysis_fails(self) -> None:
        """``stage_upload`` stores the source BEFORE the analysis runs, and a failed analysis
        writes no ``videos`` row -- so without this reap the object is invisible to every
        deletion path and lives forever under ``uploads/{user_id}/``, which has no lifecycle rule.
        """
        def boom(*args, **kwargs):
            raise RuntimeError("Pose extraction failed")

        analysis_service.analyze_video_file = boom
        with self.assertRaises(HTTPException):
            self._run()
        self.reap.assert_called_once_with([self.staged.prefix])

    def test_reaps_the_stored_objects_when_the_analysis_fails_unexpectedly(self) -> None:
        """Not just the RuntimeError -> 422 arm: a bug in the pipeline 500s, and orphans just
        as much."""
        def boom(*args, **kwargs):
            raise ValueError("something nobody predicted")

        analysis_service.analyze_video_file = boom
        with self.assertRaises(ValueError):
            self._run()
        self.reap.assert_called_once_with([self.staged.prefix])

    def test_never_reaps_after_a_successful_analysis(self) -> None:
        """The assertion with teeth: a reap that also fired on success would delete the clip the
        caller is being handed a live ``video_url`` for. This pins WHERE the reap sits, which the
        failure-path tests above cannot."""
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

    def test_stages_under_the_anon_owner_for_an_anonymous_caller(self) -> None:
        """A regression here would put every anonymous upload's artifacts under the WRONG
        prefix and be invisible to every test that stubs ``stage_upload`` and ignores its
        ``owner`` kwarg -- this reads the actual value the router passed."""
        self._run()
        self.assertEqual(self.stage_calls[0]["owner"], "anon")

    def test_stages_under_the_authenticated_users_id(self) -> None:
        """The property this protects: an authenticated user's artifacts must land under
        ``uploads/{user_id}/``, not the shared ``uploads/anon/`` prefix -- which a lifecycle
        rule expires after 7 days. A regression here would silently destroy real users' videos."""
        user = mock.Mock(id="u1", token="tok")
        with mock.patch.object(store, "persist_analysis", return_value="id"):
            self._run(user=user)
        self.assertEqual(self.stage_calls[0]["owner"], "u1")


class SourceUrlTests(unittest.TestCase):
    """Direct coverage of ``_source_url``'s degrade-to-None branch (not exercised by
    ``AnalyzeStorageTests``, which patches ``_source_url`` out entirely to stay offline)."""

    def test_returns_none_when_signing_fails(self) -> None:
        with mock.patch.object(analyze_router.storage, "get_object_store") as get_store:
            get_store.return_value.presigned_url.side_effect = storage.StorageError("R2 down")
            self.assertIsNone(analyze_router._source_url("uploads/anon/upload_test"))


if __name__ == "__main__":
    unittest.main()
