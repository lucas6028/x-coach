"""/api/analyze/pose: accept client pose JSON + video, run the detector off the event loop."""
from __future__ import annotations

import asyncio
import io
import json
import threading
import unittest
from pathlib import Path

from fastapi import HTTPException
from starlette.datastructures import UploadFile

from backend.app.routers import analyze as analyze_router
from backend.app.services import analysis as analysis_service

_GOOD_POSE = json.dumps({"metadata": {"fps": 30, "width": 1, "height": 1, "total_frames": 0}, "frames": []})


def _upload(filename: str = "clip.webm", data: bytes = b"fake") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


class AnalyzePoseEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_save = analysis_service.save_upload
        self._orig_analyze = analysis_service.analyze_pose_payload
        analysis_service.save_upload = lambda data, suffix=".mp4": ("upload_test", Path(f"upload_test{suffix}"))
        analysis_service.analyze_pose_payload = lambda payload, *, movement, video_id=None: {
            "video_id": video_id, "source": "upload", "movement": movement, "detections": [],
        }

    def tearDown(self) -> None:
        analysis_service.save_upload = self._orig_save
        analysis_service.analyze_pose_payload = self._orig_analyze

    def test_happy_path_returns_analysis(self) -> None:
        result = asyncio.run(analyze_router.analyze_pose("Squat", _GOOD_POSE, _upload(), user=None))
        self.assertEqual(result["video_id"], "upload_test")
        self.assertEqual(result["movement"], "Squat")

    def test_rejects_bad_json(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(analyze_router.analyze_pose("Squat", "{not json", _upload(), user=None))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_pose_without_frames_list(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(analyze_router.analyze_pose("Squat", json.dumps({"metadata": {}}), _upload(), user=None))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_malformed_landmarks(self) -> None:
        bad = json.dumps({"metadata": {}, "frames": [{"landmarks": [{"x": 1}]}]})
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(analyze_router.analyze_pose("Squat", bad, _upload(), user=None))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_unsupported_suffix(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(analyze_router.analyze_pose("Squat", _GOOD_POSE, _upload("x.txt"), user=None))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_empty_file(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(analyze_router.analyze_pose("Squat", _GOOD_POSE, _upload(data=b""), user=None))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_runtime_error_maps_to_422(self) -> None:
        def boom(payload, *, movement, video_id=None):
            raise RuntimeError("boom")

        analysis_service.analyze_pose_payload = boom
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(analyze_router.analyze_pose("Squat", _GOOD_POSE, _upload(), user=None))
        self.assertEqual(ctx.exception.status_code, 422)

    def test_runs_off_the_event_loop(self) -> None:
        seen: dict[str, threading.Thread] = {}

        def record(payload, *, movement, video_id=None):
            seen["t"] = threading.current_thread()
            return {"video_id": video_id, "source": "upload", "detections": []}

        analysis_service.analyze_pose_payload = record
        asyncio.run(analyze_router.analyze_pose("Squat", _GOOD_POSE, _upload(), user=None))
        self.assertIsNot(seen["t"], threading.main_thread())


if __name__ == "__main__":
    unittest.main()
