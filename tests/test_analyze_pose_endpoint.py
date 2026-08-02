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
from starlette.datastructures import UploadFile

from backend.app.routers import analyze as analyze_router
from backend.app.services import analysis as analysis_service
from backend.app.services import runtime_config

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
        analysis_service.store_artifacts = lambda staged, *, thumbnail=None: self.artifacts.append(
            {"staged": staged, "thumbnail": thumbnail}
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


if __name__ == "__main__":
    unittest.main()
