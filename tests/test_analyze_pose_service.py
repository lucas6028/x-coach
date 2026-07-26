"""analyze_pose_payload: route a client pose payload to a detector strategy (no server extraction)."""
from __future__ import annotations

import unittest

from backend.app.services import analysis as svc


class AnalyzePosePayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = dict(svc._ANALYSIS_STRATEGIES)

    def tearDown(self) -> None:
        svc._ANALYSIS_STRATEGIES.clear()
        svc._ANALYSIS_STRATEGIES.update(self._orig)

    def test_squat_routes_to_strategy_and_attaches_pose_block(self) -> None:
        payload = {
            "metadata": {"fps": 30, "width": 100, "height": 200, "total_frames": 1},
            "frames": [{"frame_index": 0, "landmarks": [{"x": 0.1, "y": 0.2, "z": 0.0, "visibility": 0.9}] * 33}],
        }
        received_paths: list = []

        def _stub_strategy(pl, vid, path):
            received_paths.append(path)
            return {"detections": [], "video_id": vid}

        svc._ANALYSIS_STRATEGIES["Squat"] = _stub_strategy
        result = svc.analyze_pose_payload(payload, movement="Squat", video_id="vid1")
        self.assertEqual(result["source"], "upload")
        self.assertEqual(result["video_id"], "vid1")
        self.assertIn("pose", result)
        self.assertEqual(result["pose"]["fps"], 30.0)
        self.assertEqual(len(result["pose"]["frames"]), 1)
        self.assertEqual(len(received_paths), 1)
        self.assertEqual(received_paths[0].name, "vid1.json")

    def test_unknown_movement_returns_coming_soon_without_detector(self) -> None:
        payload = {"metadata": {"fps": 30, "width": 1, "height": 1, "total_frames": 0}, "frames": []}
        result = svc.analyze_pose_payload(payload, movement="Deadlift", video_id="v2")
        self.assertEqual(result["analysis_pending"], True)
        self.assertEqual(result["detections"], [])
        self.assertEqual(result["video_id"], "v2")
        self.assertIn("pose", result)


if __name__ == "__main__":
    unittest.main()
