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

from fastapi import HTTPException
from starlette.datastructures import UploadFile

from backend.app.routers import analyze as analyze_router
from backend.app.services import analysis as analysis_service


def _upload(filename: str = "clip.mp4", data: bytes = b"fake-video-bytes") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


class AnalyzeEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_save = analysis_service.save_upload
        self._orig_analyze = analysis_service.analyze_video_file
        self._orig_semaphore = analyze_router._ANALYSIS_SEMAPHORE
        # No real disk I/O: hand back a deterministic (video_id, path) pair.
        analysis_service.save_upload = lambda data, suffix=".mp4": (
            "upload_test",
            Path(f"upload_test{suffix}"),
        )

    def tearDown(self) -> None:
        analysis_service.save_upload = self._orig_save
        analysis_service.analyze_video_file = self._orig_analyze
        analyze_router._ANALYSIS_SEMAPHORE = self._orig_semaphore

    # --- request contract / error mapping ----------------------------------------

    def test_rejects_unsupported_suffix(self) -> None:
        analysis_service.analyze_video_file = lambda *a, **k: {}
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(analyze_router.analyze(_upload("notes.txt")))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_empty_file(self) -> None:
        analysis_service.analyze_video_file = lambda *a, **k: {}
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(analyze_router.analyze(_upload("clip.mp4", data=b"")))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_runtime_error_maps_to_422(self) -> None:
        def boom(*args, **kwargs):
            raise RuntimeError("Pose extraction failed")

        analysis_service.analyze_video_file = boom
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(analyze_router.analyze(_upload()))
        self.assertEqual(ctx.exception.status_code, 422)

    def test_returns_analysis_payload_unchanged(self) -> None:
        analysis_service.analyze_video_file = lambda path, *, video_id=None: {
            "video_id": video_id,
            "source": "upload",
            "detections": [],
        }
        result = asyncio.run(analyze_router.analyze(_upload()))
        self.assertEqual(result["video_id"], "upload_test")
        self.assertEqual(result["source"], "upload")

    # --- the P0 guarantees --------------------------------------------------------

    def test_pipeline_runs_off_the_event_loop(self) -> None:
        """The blocking pipeline must execute in a worker thread, not the loop thread."""
        seen: dict[str, threading.Thread] = {}

        def record_thread(path, *, video_id=None):
            seen["thread"] = threading.current_thread()
            return {"video_id": video_id, "source": "upload"}

        analysis_service.analyze_video_file = record_thread
        asyncio.run(analyze_router.analyze(_upload()))
        self.assertIsNot(seen["thread"], threading.main_thread())

    def test_concurrent_analyses_are_bounded(self) -> None:
        """At most MAX_CONCURRENT_ANALYSES run at once; the rest wait on the semaphore."""
        limit = 2
        lock = threading.Lock()
        release = threading.Event()
        state = {"active": 0, "peak": 0}

        def blocking(path, *, video_id=None):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            # Hold the slot until released so concurrent calls genuinely overlap in time.
            release.wait(timeout=5)
            with lock:
                state["active"] -= 1
            return {"video_id": video_id, "source": "upload"}

        analysis_service.analyze_video_file = blocking

        async def drive() -> int:
            # Bind a fresh semaphore (sized to `limit`) to this event loop.
            analyze_router._ANALYSIS_SEMAPHORE = asyncio.Semaphore(limit)
            tasks = [
                asyncio.create_task(analyze_router.analyze(_upload(f"c{i}.mp4")))
                for i in range(limit + 3)
            ]
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 5
            # Wait until the cap is saturated, then let everything drain.
            while state["peak"] < limit and loop.time() < deadline:
                await asyncio.sleep(0.01)
            release.set()
            await asyncio.gather(*tasks)
            return state["peak"]

        try:
            peak = asyncio.run(drive())
        finally:
            release.set()
        # Never more than `limit` ran simultaneously, and the cap was actually reached.
        self.assertEqual(peak, limit)


if __name__ == "__main__":
    unittest.main()
