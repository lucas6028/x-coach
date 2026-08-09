"""analyze_pose_payload: route a client pose payload to a detector (no server extraction)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend.app import config
from backend.app.services import analysis as svc
from src.pose.movements import registry


class AnalyzePosePayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _pose_json_path(self) -> Path:
        # Stands in for the caller-supplied staged path (`stage_upload` puts this in the
        # upload's own temp dir in production).
        return Path(self._tmp.name) / "pose.json"

    @staticmethod
    def _payload(frames: int = 1) -> dict:
        landmarks = [{"x": 0.1, "y": 0.2, "z": 0.0, "visibility": 0.9}] * 33
        return {
            "metadata": {"fps": 30, "width": 100, "height": 200, "total_frames": frames},
            "frames": [{"frame_index": i, "landmarks": landmarks} for i in range(frames)],
        }

    def test_squat_routes_to_the_detector_and_attaches_pose_block(self) -> None:
        pose_json_path = self._pose_json_path()
        with mock.patch("src.pose.pose_rule_detector.detect_pose_rules_from_payload") as detect:
            detect.return_value = {"detections": [], "video_id": "vid1"}
            result = svc.analyze_pose_payload(
                self._payload(), movement="Squat", video_id="vid1", pose_json_path=pose_json_path
            )

        self.assertEqual(result["source"], "upload")
        self.assertEqual(result["video_id"], "vid1")
        self.assertEqual(result["pose"]["fps"], 30.0)
        self.assertEqual(len(result["pose"]["frames"]), 1)
        kwargs = detect.call_args.kwargs
        self.assertEqual(kwargs["movement"], "Squat")
        # The pose JSON path is load-bearing, not incidental: without it the detector forces
        # view_type="unknown", which suppresses knees_forward on side view and downweights
        # knees_inward / excessive_forward_lean. It was a review Critical once already.
        self.assertEqual(kwargs["pose_json_path"], pose_json_path)
        self.assertTrue(kwargs["pose_json_path"].exists())

    # THE BUG THIS FILE MISSED. `/api/movements` advertises every registered detector and the studio
    # offers all of them, but this path routed through a hand-maintained dict holding only "Squat".
    # Push-up and Overhead Press therefore fell through to the `analysis_pending` skeleton: no
    # detector, no retrieval, and -- because that branch carries no `quality` key -- the frontend's
    # wasMeasured() read it as UNMEASURED and showed "no frame in this clip could be measured" for a
    # clip that measured perfectly well. The old test monkeypatched the dict, so it could not have
    # caught a movement missing FROM the dict.
    def test_every_advertised_movement_reaches_a_detector(self) -> None:
        detectors = registry.list_detectors()
        advertised = [d.name for d in detectors]
        self.assertIn("Push-up", advertised)  # guards the premise, not just the conclusion
        self.assertIn("Overhead Press", advertised)
        self.assertIn("Row", advertised)  # Task 6's registration, not just the older two

        for movement in advertised:
            with self.subTest(movement=movement):
                with mock.patch("src.pose.pose_rule_detector.detect_pose_rules_from_payload") as detect:
                    detect.return_value = {"detections": [], "video_id": "v"}
                    result = svc.analyze_pose_payload(
                        self._payload(),
                        movement=movement,
                        video_id="v",
                        pose_json_path=self._pose_json_path(),
                    )
                detect.assert_called_once()
                # Passed through unchanged -- e.g. "Row", never silently coerced to the "Squat"
                # fallback `registry.get_detector` applies to a falsy movement.
                self.assertEqual(detect.call_args.kwargs["movement"], movement)
                self.assertNotIn("analysis_pending", result)

        # Row's specific consequence of being registered: the analysis path reaches it (above),
        # and the flag `GET /api/movements` renders as a frontend Beta tag is False, not True or
        # missing. Squat is the only movement this repo has validated against labeled data
        # (movement-rule-detector-design.md §8); Row must not silently inherit that status.
        row_detector = registry.get_detector("Row")
        self.assertEqual(row_detector.name, "Row")
        self.assertFalse(row_detector.validated)

    def test_default_analysis_movement_is_still_squat(self) -> None:
        """Registering a fifth detector (Row, following Lunge/Push-up/Overhead Press) must not
        move what an unspecified /api/analyze request analyzes -- the registry growing is not
        license to change the fallback out from under existing callers."""
        self.assertEqual(config.DEFAULT_ANALYSIS_MOVEMENT, "Squat")

    def test_movement_name_is_matched_case_insensitively(self) -> None:
        # The registry keys on a lowercased name, so the dict's exact-match lookup would have
        # rejected a spelling the detector itself accepts.
        with mock.patch("src.pose.pose_rule_detector.detect_pose_rules_from_payload") as detect:
            detect.return_value = {"detections": [], "video_id": "v3"}
            result = svc.analyze_pose_payload(
                self._payload(),
                movement="push-up",
                video_id="v3",
                pose_json_path=self._pose_json_path(),
            )
        self.assertNotIn("analysis_pending", result)
        detect.assert_called_once()

    def test_unknown_movement_returns_coming_soon_without_detector(self) -> None:
        # THIS EXAMPLE HAS TO BE ROTATED EVERY TIME A DETECTOR IS REGISTERED, and that is the
        # point of the assertion below rather than a nuisance: the test needs a movement the
        # frontend lists (frontend/src/lib/movements.ts) that has NO registered detector, so it
        # necessarily goes stale as the 16-movement programme lands one movement at a time. It
        # has already moved "Deadlift" -> "Row" -> "Band Pull Apart" -> "Bicep Curl" -> "Arm
        # Abduction" -> "Arm VW" -> "Sit-up" -> "Shoulder Bridge" -> "Leg Abduction" -> "Torso
        # Twist"; when Torso Twist is implemented, move it again to any still-unimplemented
        # movement. The `assertNotIn` is what turns that staleness into a loud failure instead
        # of a silently vacuous test.
        self.assertNotIn("Torso Twist", [d.name for d in registry.list_detectors()])
        payload = {"metadata": {"fps": 30, "width": 1, "height": 1, "total_frames": 0}, "frames": []}
        pose_json_path = self._pose_json_path()
        result = svc.analyze_pose_payload(
            payload, movement="Torso Twist", video_id="v2", pose_json_path=pose_json_path
        )
        self.assertEqual(result["analysis_pending"], True)
        self.assertEqual(result["detections"], [])
        self.assertEqual(result["video_id"], "v2")
        self.assertIn("pose", result)
        # The no-detector branch never writes pose JSON — `store_artifacts` relies on this to
        # decide whether to upload pose.json at all.
        self.assertFalse(pose_json_path.exists())


if __name__ == "__main__":
    unittest.main()
