"""analyze_pose_payload: route a client pose payload to a detector (no server extraction)."""
from __future__ import annotations

import unittest
from unittest import mock

from backend.app.services import analysis as svc
from src.pose.movements import registry


class AnalyzePosePayloadTests(unittest.TestCase):
    @staticmethod
    def _payload(frames: int = 1) -> dict:
        landmarks = [{"x": 0.1, "y": 0.2, "z": 0.0, "visibility": 0.9}] * 33
        return {
            "metadata": {"fps": 30, "width": 100, "height": 200, "total_frames": frames},
            "frames": [{"frame_index": i, "landmarks": landmarks} for i in range(frames)],
        }

    def test_squat_routes_to_the_detector_and_attaches_pose_block(self) -> None:
        with mock.patch("src.pose.pose_rule_detector.detect_pose_rules_from_payload") as detect:
            detect.return_value = {"detections": [], "video_id": "vid1"}
            result = svc.analyze_pose_payload(self._payload(), movement="Squat", video_id="vid1")

        self.assertEqual(result["source"], "upload")
        self.assertEqual(result["video_id"], "vid1")
        self.assertEqual(result["pose"]["fps"], 30.0)
        self.assertEqual(len(result["pose"]["frames"]), 1)
        kwargs = detect.call_args.kwargs
        self.assertEqual(kwargs["movement"], "Squat")
        # The pose JSON path is load-bearing, not incidental: without it the detector forces
        # view_type="unknown", which suppresses knees_forward on side view and downweights
        # knees_inward / excessive_forward_lean. It was a review Critical once already.
        self.assertEqual(kwargs["pose_json_path"].name, "vid1.json")
        self.assertTrue(kwargs["pose_json_path"].exists())

    # THE BUG THIS FILE MISSED. `/api/movements` advertises every registered detector and the studio
    # offers all of them, but this path routed through a hand-maintained dict holding only "Squat".
    # Push-up and Overhead Press therefore fell through to the `analysis_pending` skeleton: no
    # detector, no retrieval, and -- because that branch carries no `quality` key -- the frontend's
    # wasMeasured() read it as UNMEASURED and showed "no frame in this clip could be measured" for a
    # clip that measured perfectly well. The old test monkeypatched the dict, so it could not have
    # caught a movement missing FROM the dict.
    def test_every_advertised_movement_reaches_a_detector(self) -> None:
        advertised = [d.name for d in registry.list_detectors()]
        self.assertIn("Push-up", advertised)  # guards the premise, not just the conclusion
        self.assertIn("Overhead Press", advertised)

        for movement in advertised:
            with self.subTest(movement=movement):
                with mock.patch("src.pose.pose_rule_detector.detect_pose_rules_from_payload") as detect:
                    detect.return_value = {"detections": [], "video_id": "v"}
                    result = svc.analyze_pose_payload(self._payload(), movement=movement, video_id="v")
                detect.assert_called_once()
                self.assertEqual(detect.call_args.kwargs["movement"], movement)
                self.assertNotIn("analysis_pending", result)

    def test_movement_name_is_matched_case_insensitively(self) -> None:
        # The registry keys on a lowercased name, so the dict's exact-match lookup would have
        # rejected a spelling the detector itself accepts.
        with mock.patch("src.pose.pose_rule_detector.detect_pose_rules_from_payload") as detect:
            detect.return_value = {"detections": [], "video_id": "v3"}
            result = svc.analyze_pose_payload(self._payload(), movement="push-up", video_id="v3")
        self.assertNotIn("analysis_pending", result)
        detect.assert_called_once()

    def test_unknown_movement_returns_coming_soon_without_detector(self) -> None:
        payload = {"metadata": {"fps": 30, "width": 1, "height": 1, "total_frames": 0}, "frames": []}
        result = svc.analyze_pose_payload(payload, movement="Deadlift", video_id="v2")
        self.assertEqual(result["analysis_pending"], True)
        self.assertEqual(result["detections"], [])
        self.assertEqual(result["video_id"], "v2")
        self.assertIn("pose", result)


if __name__ == "__main__":
    unittest.main()
