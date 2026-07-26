from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from src.pose.movements import registry
from src.pose.pose_rule_detector import detect_pose_rules_from_payload
from tests.test_pose_rule_detector import frame
from tests.test_run_detector_per_rep import squat_reps


def payload(frames: list[dict]) -> dict:
    return {"metadata": {"fps": 30.0, "width": 640, "height": 480}, "frames": frames}


class RepsPayloadTests(unittest.TestCase):
    def test_reps_block_describes_every_segment(self) -> None:
        result = detect_pose_rules_from_payload(payload(squat_reps(5)), movement="Squat", max_reps=3)
        reps = result["reps"]
        self.assertEqual(reps["detected"], 5)
        self.assertEqual(reps["analyzed"], [1, 3, 5])
        self.assertEqual(reps["max_reps"], 3)
        self.assertIsNone(reps["fallback"])
        self.assertEqual(len(reps["segments"]), 5)
        for segment in reps["segments"]:
            self.assertEqual(
                set(segment),
                {"index", "start_frame", "end_frame", "start_time", "end_time", "analyzed", "partial"},
            )
        self.assertEqual([s["index"] for s in reps["segments"] if s["analyzed"]], [1, 3, 5])

    def test_quality_gains_analyzed_counters_without_changing_the_old_ones(self) -> None:
        frames = squat_reps(5)
        result = detect_pose_rules_from_payload(payload(frames), movement="Squat", max_reps=3)
        quality = result["quality"]
        self.assertEqual(quality["total_frames"], len(frames))
        self.assertEqual(quality["valid_frames"], len(frames))
        self.assertEqual(quality["valid_frame_ratio"], 1.0)
        self.assertLess(quality["analyzed_frames"], quality["total_frames"])
        self.assertGreater(quality["analyzed_frames"], 0)
        self.assertAlmostEqual(
            quality["analyzed_frame_ratio"], quality["analyzed_frames"] / quality["total_frames"], places=4
        )

    def test_frame_metrics_still_has_one_row_per_frame(self) -> None:
        frames = squat_reps(3)
        result = detect_pose_rules_from_payload(payload(frames), movement="Squat")
        self.assertEqual(len(result["frame_metrics"]), len(frames))

    def test_static_clip_reports_the_fallback_and_still_analyzes(self) -> None:
        frames = [frame(left_knee_x=0.48, right_knee_x=0.88, frame_index=i) for i in range(14)]
        result = detect_pose_rules_from_payload(payload(frames), movement="Squat")
        self.assertEqual(result["reps"]["detected"], 0)
        self.assertEqual(result["reps"]["fallback"], "no_reps_detected")
        self.assertEqual(result["reps"]["segments"], [])
        self.assertEqual(result["quality"]["analyzed_frames"], result["quality"]["total_frames"])

    def test_payload_survives_strict_json_dumps(self) -> None:
        """postgrest serialises with allow_nan=False; a NaN here would drop the analysis."""
        result = detect_pose_rules_from_payload(payload(squat_reps(3)), movement="Squat")
        json.dumps(result, allow_nan=False)

    def test_partial_only_clip_still_lists_what_was_found(self) -> None:
        """`fallback` explains why the clip was analyzed whole; it must not also erase the
        evidence that repetitions were there. Same fixture as the run_detector-level test."""
        result = detect_pose_rules_from_payload(
            payload(squat_reps(1, frames_per_rep=90)[:60]), movement="Squat"
        )
        self.assertEqual(result["reps"]["fallback"], "only_partial_reps")
        self.assertGreater(result["reps"]["detected"], 0)
        self.assertTrue(result["reps"]["segments"])
        self.assertTrue(all(s["partial"] for s in result["reps"]["segments"]))
        self.assertFalse(any(s["analyzed"] for s in result["reps"]["segments"]))
        self.assertEqual(result["reps"]["analyzed"], [])
        self.assertEqual(result["quality"]["analyzed_frames"], result["quality"]["total_frames"])

    def test_empty_frame_list_does_not_raise(self) -> None:
        result = detect_pose_rules_from_payload(payload([]), movement="Squat")
        self.assertEqual(result["reps"]["detected"], 0)
        self.assertEqual(result["quality"]["analyzed_frames"], 0)

    def test_segmentation_disabled_detector_still_gets_analyzed_through_the_payload(self) -> None:
        """No REGISTERED detector currently sets `rep_signal=None`, so this path is otherwise
        unreachable through `detect_pose_rules_from_payload`. Patch the registry lookup to force
        it, mirroring `test_segmentation_disabled_detector_falls_back` in
        test_run_detector_per_rep.py one layer up. The point of every fallback is that the clip
        still gets analyzed -- a segmentation failure must never present as 'no faults found'."""
        disabled = replace(registry.get_detector("Squat"), rep_signal=None)
        with patch("src.pose.movements.registry.get_detector", return_value=disabled):
            result = detect_pose_rules_from_payload(payload(squat_reps(3)), movement="Squat")

        reps = result["reps"]
        self.assertEqual(reps["fallback"], "segmentation_disabled")
        self.assertEqual(reps["detected"], 0)
        self.assertEqual(reps["analyzed"], [])
        self.assertEqual(reps["segments"], [])
        self.assertTrue(
            result["detections"], "a segmentation failure must still be analyzed, not read as clean"
        )
        self.assertEqual(result["quality"]["analyzed_frames"], result["quality"]["total_frames"])


if __name__ == "__main__":
    unittest.main()
