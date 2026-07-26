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
from starlette.datastructures import UploadFile

from backend.app.routers import analyze as analyze_router
from backend.app.services import analysis as analysis_service
from backend.app.services import runtime_config


def _upload(filename: str = "clip.mp4", data: bytes = b"fake-video-bytes") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


# These tests invoke ``analyze`` directly (not via FastAPI), so neither the ``user`` dependency
# nor the ``movement``/``max_reps`` Form defaults are resolved by FastAPI's DI -- pass
# ``user=None`` to drive the anonymous "demo" path (no persistence), ``movement="Squat"``
# explicitly, and ``max_reps=None`` explicitly, since an unresolved ``Form(...)`` sentinel would
# otherwise reach ``_validated_movement`` / ``_validated_max_reps`` verbatim.


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
        analysis_service.save_upload = self._orig_save
        analysis_service.analyze_video_file = self._orig_analyze
        analyze_router._ANALYSIS_SEMAPHORE = self._orig_semaphore

    # --- request contract / error mapping ----------------------------------------

    def test_rejects_unsupported_suffix(self) -> None:
        analysis_service.analyze_video_file = lambda *a, **k: {}
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze(
                    _upload("notes.txt"), movement="Squat", max_reps=None, user=None
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_empty_file(self) -> None:
        analysis_service.analyze_video_file = lambda *a, **k: {}
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze(
                    _upload("clip.mp4", data=b""), movement="Squat", max_reps=None, user=None
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_runtime_error_maps_to_422(self) -> None:
        def boom(*args, **kwargs):
            raise RuntimeError("Pose extraction failed")

        analysis_service.analyze_video_file = boom
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                analyze_router.analyze(_upload(), movement="Squat", max_reps=None, user=None)
            )
        self.assertEqual(ctx.exception.status_code, 422)

    def test_returns_analysis_payload_unchanged(self) -> None:
        analysis_service.analyze_video_file = (
            lambda path, *, video_id=None, movement=None, max_reps=-1: {
                "video_id": video_id,
                "source": "upload",
                "detections": [],
            }
        )
        result = asyncio.run(
            analyze_router.analyze(_upload(), movement="Squat", max_reps=None, user=None)
        )
        self.assertEqual(result["video_id"], "upload_test")
        self.assertEqual(result["source"], "upload")

    # --- the P0 guarantees --------------------------------------------------------

    def test_pipeline_runs_off_the_event_loop(self) -> None:
        """The blocking pipeline must execute in a worker thread, not the loop thread."""
        seen: dict[str, threading.Thread] = {}

        def record_thread(path, *, video_id=None, movement=None, max_reps=-1):
            seen["thread"] = threading.current_thread()
            return {"video_id": video_id, "source": "upload"}

        analysis_service.analyze_video_file = record_thread
        asyncio.run(
            analyze_router.analyze(_upload(), movement="Squat", max_reps=None, user=None)
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

        def blocking(path, *, video_id=None, movement=None, max_reps=-1):
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
                        _upload(f"c{i}.mp4"), movement="Squat", max_reps=None, user=None
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


if __name__ == "__main__":
    unittest.main()
