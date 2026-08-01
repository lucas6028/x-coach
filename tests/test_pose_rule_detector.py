from __future__ import annotations

import json
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from src.pose import pose_rule_detector
from src.pose.pose_rule_detector import (
    LANDMARK_COUNT,
    FrameMetrics,
    PoseRuleRequest,
    compute_frame_metrics,
    detect_rule_segments,
    json_safe_view_payload,
    raw_frame_metrics,
)
from src.pose.view_estimation import ViewEstimate


def landmark(x: float, y: float, visibility: float = 1.0) -> dict[str, float]:
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility}


def frame(
    *,
    left_knee_x: float = 0.45,
    right_knee_x: float = 0.55,
    hip_y: float = 0.72,
    knee_y: float = 0.68,
    ankle_y: float = 0.90,
    toe_offset: float = 0.12,
    frame_index: int = 0,
) -> dict[str, object]:
    landmarks = [landmark(0.0, 0.0, visibility=0.0) for _ in range(LANDMARK_COUNT)]
    landmarks[11] = landmark(0.35, 0.25)
    landmarks[12] = landmark(0.65, 0.25)
    landmarks[23] = landmark(0.35, hip_y)
    landmarks[24] = landmark(0.65, hip_y)
    landmarks[25] = landmark(left_knee_x, knee_y)
    landmarks[26] = landmark(right_knee_x, knee_y)
    landmarks[27] = landmark(0.30, ankle_y)
    landmarks[28] = landmark(0.70, ankle_y)
    landmarks[29] = landmark(0.30, ankle_y)
    landmarks[30] = landmark(0.70, ankle_y)
    landmarks[31] = landmark(0.30 + toe_offset, ankle_y)
    landmarks[32] = landmark(0.70 + toe_offset, ankle_y)
    return {"frame_index": frame_index, "landmarks": landmarks, "world_landmarks": landmarks}


class PoseRuleDetectorTests(unittest.TestCase):
    def test_knee_width_ratio_detects_inward_collapse(self) -> None:
        metrics = raw_frame_metrics(frame(left_knee_x=0.47, right_knee_x=0.53), fps=30.0)
        self.assertLess(metrics["knee_width_to_ankle_width"], 0.82)

    def test_knee_forward_projection_requires_side_observability(self) -> None:
        frames = [
            frame(left_knee_x=0.48, right_knee_x=0.88, toe_offset=0.12, frame_index=index)
            for index in range(12)
        ]
        metrics = compute_frame_metrics(frames, fps=30.0)
        side_detections = detect_rule_segments(metrics, fps=30.0, view_type="side", view_confidence=0.8)
        rear_detections = detect_rule_segments(metrics, fps=30.0, view_type="rear_oblique", view_confidence=0.8)

        self.assertTrue(any(item.fault_id == "knees_forward" and item.severity > 0 for item in side_detections))
        self.assertTrue(
            any(
                item.fault_id == "knees_forward" and item.observability == "low" and item.severity == 0
                for item in rear_detections
            )
        )

    def test_depth_rule_distinguishes_above_and_below_parallel(self) -> None:
        shallow = compute_frame_metrics([frame(hip_y=0.45, knee_y=0.70, frame_index=index) for index in range(12)], fps=30.0)
        deep = compute_frame_metrics([frame(hip_y=0.92, knee_y=0.70, frame_index=index) for index in range(12)], fps=30.0)

        shallow_detections = detect_rule_segments(shallow, fps=30.0, view_type="rear", view_confidence=0.8)
        deep_detections = detect_rule_segments(deep, fps=30.0, view_type="rear", view_confidence=0.8)

        self.assertTrue(any(item.fault_id == "shallow_depth" for item in shallow_detections))
        self.assertFalse(any(item.fault_id == "shallow_depth" for item in deep_detections))

        # Every real fault authors an explicit primary metric (label + value + threshold) so the
        # frontend/chat surface the breached number without guessing it from evidence key order.
        shallow = next(item for item in shallow_detections if item.fault_id == "shallow_depth")
        for key in ("primary_label", "primary_value", "primary_threshold"):
            self.assertIn(key, shallow.evidence)
        self.assertIsInstance(shallow.evidence["primary_label"], str)
        self.assertIsInstance(shallow.evidence["primary_value"], (int, float))
        self.assertIsInstance(shallow.evidence["primary_threshold"], (int, float))

    def test_detection_carries_citation_metadata(self) -> None:
        frames = [frame(frame_index=i) for i in range(12)]
        metrics = compute_frame_metrics(frames, fps=30.0)
        detections = detect_rule_segments(metrics, fps=30.0, view_type="rear", view_confidence=0.8)
        inward = next(d for d in detections if d.fault_id == "knees_inward")
        assert inward.citation.startswith("Ford KR")
        assert "73%" in inward.citation_support or "abduction" in inward.citation_support.lower()

    def test_persistence_filter_suppresses_single_frame_spike(self) -> None:
        base = FrameMetrics(
            frame_index=0,
            time=0.0,
            phase="bottom",
            valid=True,
            lower_body_visibility=1.0,
            avg_knee_angle=90.0,
            left_knee_angle=90.0,
            right_knee_angle=90.0,
            left_hip_angle=90.0,
            right_hip_angle=90.0,
            left_ankle_angle=90.0,
            right_ankle_angle=90.0,
            hip_minus_knee_y=0.10,
            knee_width_to_ankle_width=1.0,
            knee_forward_ratio=0.0,
            torso_lean_deg=10.0,
            heel_height_delta=0.0,
        )
        metrics = [
            replace(
                base,
                frame_index=index,
                time=index / 30.0,
                knee_width_to_ankle_width=0.50 if index == 5 else 1.0,
            )
            for index in range(12)
        ]
        detections = detect_rule_segments(metrics, fps=30.0, view_type="rear", view_confidence=0.8)
        self.assertFalse(any(item.fault_id == "knees_inward" for item in detections))


class JsonSafeViewPayloadTests(unittest.TestCase):
    """torso_width_ratio_mean (and any other ViewEstimate float) can legitimately be
    NaN -- that's honest "no width evidence" from estimate_view_for_pose, not a bug.
    But dataclasses.asdict() leaves NaN untouched, and the Supabase/postgrest write
    path serializes with allow_nan=False: it raises before any network call, gets
    swallowed by the analyze route's broad except, and silently drops the analysis
    from the user's history. json_safe_view_payload is the boundary fix: it must
    always produce something strict `json.dumps(..., allow_nan=False)` accepts.
    """

    def _view(self, **overrides) -> ViewEstimate:
        base = dict(
            split_name="",
            video_id="vid1",
            view_type="rear_oblique",
            view_confidence=0.21875,
            front_score=0.0,
            rear_score=0.0,
            side_score=0.25,
            oblique_score=0.3125,
            face_visibility_mean=0.0,
            torso_width_ratio_mean=float("nan"),
            orientation_score_mean=0.0,
            z_asymmetry_mean=0.0,
            valid_frame_ratio=1.0,
            valid_frame_count=1,
            total_frames=1,
        )
        base.update(overrides)
        return ViewEstimate(**base)

    def test_non_finite_float_becomes_none(self) -> None:
        payload = json_safe_view_payload(self._view())
        self.assertIsNone(payload["torso_width_ratio_mean"])

    def test_finite_fields_survive_unchanged(self) -> None:
        payload = json_safe_view_payload(self._view())
        self.assertEqual(payload["view_type"], "rear_oblique")
        self.assertEqual(payload["view_confidence"], 0.21875)
        self.assertEqual(payload["valid_frame_count"], 1)

    def test_payload_survives_strict_json_dumps(self) -> None:
        # This is the exact call postgrest/httpx makes when Supabase-persisting an
        # analysis. Before the fix, asdict(view) fed straight to this call raised
        # ValueError: Out of range float values are not JSON compliant: nan.
        payload = json_safe_view_payload(self._view())
        encoded = json.dumps(payload, allow_nan=False)
        self.assertNotIn("NaN", encoded)
        round_tripped = json.loads(encoded)
        self.assertIsNone(round_tripped["torso_width_ratio_mean"])

    def test_infinite_float_also_becomes_none(self) -> None:
        # math.isfinite rejects +/-inf too, not just NaN -- belt and suspenders in
        # case a future aggregation change introduces an infinite default.
        payload = json_safe_view_payload(self._view(z_asymmetry_mean=math.inf))
        self.assertIsNone(payload["z_asymmetry_mean"])
        json.dumps(payload, allow_nan=False)

    def test_all_finite_view_round_trips_with_no_nones(self) -> None:
        payload = json_safe_view_payload(self._view(torso_width_ratio_mean=0.14828))
        encoded = json.dumps(payload, allow_nan=False)
        round_tripped = json.loads(encoded)
        self.assertEqual(round_tripped["torso_width_ratio_mean"], 0.14828)

    def test_end_to_end_degenerate_pose_json_view_survives_strict_json(self) -> None:
        # Mirrors data/runtime/pose_json/vid1.json (all-coincident landmarks ->
        # torso_width_ratio NaN in every frame) at the real call site,
        # detect_pose_rules_from_payload, not just the helper in isolation.
        import json as json_module
        import tempfile
        from pathlib import Path

        from src.pose.pose_rule_detector import detect_pose_rules_from_payload

        landmarks = [{"x": 0.1, "y": 0.2, "z": 0.0, "visibility": 0.9} for _ in range(33)]
        pose_payload = {"metadata": {}, "frames": [{"frame_index": 0, "landmarks": landmarks}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "degenerate.json"
            path.write_text(json_module.dumps(pose_payload), encoding="utf-8")
            result = detect_pose_rules_from_payload(pose_payload, pose_json_path=path)

        self.assertIsNone(result["view"]["torso_width_ratio_mean"])
        # result["view"] is stored verbatim inside the JSONB blob that
        # backend/app/services/store.py:persist_analysis hands to postgrest/httpx
        # (allow_nan=False). Scoped to result["view"], matching the reviewer's
        # traced call site (asdict(view) at the old pose_rule_detector.py:565) --
        # NOT the full result dict, which separately carries NaN in frame_metrics
        # for degenerate/invalid frames (a pre-existing, unrelated gap outside
        # this fix's scope; see task-2-report.md).
        json_module.dumps(result["view"], allow_nan=False)


class MaxRepsCliForwardingTests(unittest.TestCase):
    """main() has two separate detect_pose_rules_from_json call sites (single-file mode and
    batch mode). --max-reps appearing in --help proves the flag parses; it proves nothing about
    whether args.max_reps actually reaches the detector on both paths. This class mocks
    detect_pose_rules_from_json and asserts the forwarded value in each mode, so a future edit
    that drops max_reps from one call site (but not the other) fails loudly here instead of
    silently doing nothing in half of the CLI's modes.
    """

    def _run_main(self, argv: list[str]):
        calls: list[dict] = []

        def fake_detect(*args, **kwargs):
            calls.append(kwargs)
            return {"video_id": "vid", "detections": []}

        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            pose_rule_detector, "detect_pose_rules_from_json", side_effect=fake_detect
        ), mock.patch.object(pose_rule_detector, "write_detection_json"):
            pose_rule_detector.main()
        return calls

    def test_single_file_mode_forwards_max_reps(self) -> None:
        calls = self._run_main(
            [
                "run_pose_rule_detection.py",
                "--pose-json",
                "fake_pose.json",
                "--max-reps",
                "7",
                "--no-retrieval",
            ]
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["max_reps"], 7)

    def test_batch_mode_forwards_max_reps(self) -> None:
        fake_request = PoseRuleRequest(
            split_name="train",
            video_id="vid1",
            pose_json_path=Path("fake_pose.json"),
            output_path=Path("fake_output.json"),
        )
        with mock.patch.object(pose_rule_detector, "build_requests", return_value=[fake_request]):
            calls = self._run_main(
                [
                    "run_pose_rule_detection.py",
                    "--max-reps",
                    "5",
                    "--no-retrieval",
                ]
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["max_reps"], 5)

    def test_default_max_reps_is_three_in_both_modes(self) -> None:
        """--max-reps omitted entirely should still forward the argparse default (3)."""
        single_calls = self._run_main(
            ["run_pose_rule_detection.py", "--pose-json", "fake_pose.json", "--no-retrieval"]
        )
        self.assertEqual(single_calls[0]["max_reps"], 3)

        fake_request = PoseRuleRequest(
            split_name="train",
            video_id="vid1",
            pose_json_path=Path("fake_pose.json"),
            output_path=Path("fake_output.json"),
        )
        with mock.patch.object(pose_rule_detector, "build_requests", return_value=[fake_request]):
            batch_calls = self._run_main(["run_pose_rule_detection.py", "--no-retrieval"])
        self.assertEqual(batch_calls[0]["max_reps"], 3)


if __name__ == "__main__":
    unittest.main()
