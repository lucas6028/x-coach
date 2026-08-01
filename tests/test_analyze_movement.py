from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routers.analyze import _validated_movement


def _stub_result(movement: str) -> dict:
    """A response-shaped stub. The route returns the analysis verbatim, so keep the real keys --
    a thin dict would pass today but hide a shape regression if the route ever post-processes."""
    return {
        "video_id": "vid1",
        "movement": movement,
        "metadata": {"fps": 30.0},
        "view": {"view_type": "side", "view_confidence": 0.8},
        "quality": {"total_frames": 10, "valid_frames": 9, "valid_frame_ratio": 0.9},
        "detections": [],
        "retrievals": [],
        "pose": {"fps": 30.0, "width": 640, "height": 480, "frames": []},
        "source": "upload",
    }


class TestAnalyzeMovement(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def _post(self, movement: str | None):
        data = {"movement": movement} if movement is not None else None
        return self.client.post(
            "/api/analyze",
            files={"file": ("clip.mp4", b"not-a-real-video", "video/mp4")},
            data=data,
        )

    def test_rejects_an_unregistered_movement(self) -> None:
        resp = self._post("Cartwheel")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Cartwheel", resp.json()["detail"])

    def test_rejects_before_running_the_pipeline(self) -> None:
        """A bad movement must not cost a full MediaPipe pass -- neither the upload write nor
        the analysis may be reached."""
        with patch("backend.app.services.analysis.save_upload") as save, patch(
            "backend.app.services.analysis.analyze_video_file"
        ) as run:
            self._post("Cartwheel")
        save.assert_not_called()
        run.assert_not_called()

    def test_validated_movement_rejects_an_empty_or_whitespace_string(self) -> None:
        """Unit-level guard check, called directly rather than through HTTP -- an HTTP client
        cannot deliver a literal "" here at all (see the fix report for why); the reachable
        whitespace-only case is covered end-to-end below via ``self._post``."""
        for blank in ("", "   "):
            with self.subTest(movement=repr(blank)):
                with self.assertRaises(HTTPException) as ctx:
                    _validated_movement(blank)
                self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_a_whitespace_only_movement_over_http(self) -> None:
        """The one "blank-ish" value an HTTP client can actually deliver as itself (see the unit
        test above for why a literal "" cannot): a whitespace-only movement must 400 and must not
        reach save_upload or the pipeline, same as any other invalid movement."""
        resp = self._post("   ")
        self.assertEqual(resp.status_code, 400)

        with patch("backend.app.services.analysis.save_upload") as save, patch(
            "backend.app.services.analysis.analyze_video_file"
        ) as run:
            self._post("   ")
        save.assert_not_called()
        run.assert_not_called()

    def test_forwards_a_valid_movement_to_the_pipeline(self) -> None:
        """Pins the canonicalization contract: a non-canonical spelling ("push-up") must
        resolve to the registry's canonical name ("Push-up") before reaching the pipeline, not
        pass the caller's raw string through untouched."""
        with patch(
            "backend.app.services.analysis.save_upload", return_value=("vid1", "/tmp/vid1.mp4")
        ), patch(
            "backend.app.services.analysis.analyze_video_file",
            return_value=_stub_result("Push-up"),
        ) as run:
            resp = self._post("push-up")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(run.call_args.kwargs["movement"], "Push-up")

    def test_defaults_to_squat_when_the_field_is_omitted(self) -> None:
        """Backward compatible: a caller that has not been updated still works."""
        with patch(
            "backend.app.services.analysis.save_upload", return_value=("vid1", "/tmp/vid1.mp4")
        ), patch(
            "backend.app.services.analysis.analyze_video_file",
            return_value=_stub_result("Squat"),
        ) as run:
            resp = self._post(None)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(run.call_args.kwargs["movement"], "Squat")

    def test_accepts_any_registered_movement(self) -> None:
        for movement in ("Squat", "Push-up", "Overhead Press"):
            with self.subTest(movement=movement):
                with patch(
                    "backend.app.services.analysis.save_upload",
                    return_value=("vid1", "/tmp/vid1.mp4"),
                ), patch(
                    "backend.app.services.analysis.analyze_video_file",
                    return_value=_stub_result(movement),
                ):
                    self.assertEqual(self._post(movement).status_code, 200)
