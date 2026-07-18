import unittest
from src.pose.movements.base import CoreFrame, RuleContext, MovementDetector, run_detector
from src.pose.movements import registry


class MovementRegistryTests(unittest.TestCase):
    def test_core_frame_metric_accessor_defaults_nan(self) -> None:
        cf = CoreFrame(0, 0.0, "setup", True, 0.9, {"a": 1.0})
        self.assertEqual(cf.m("a"), 1.0)
        import math
        self.assertTrue(math.isnan(cf.m("missing")))

    def test_default_movement_is_squat(self) -> None:
        det = registry.get_detector(None)
        self.assertEqual(det.name, "Squat")

    def test_unknown_movement_raises(self) -> None:
        with self.assertRaises(KeyError):
            registry.get_detector("Nonexistent Movement")

    def test_squat_via_registry_matches_legacy(self) -> None:
        from src.pose.pose_rule_detector import compute_frame_metrics, detect_rule_segments
        from src.pose.movements import registry
        from src.pose.movements.base import run_detector
        from tests.test_pose_rule_detector import frame  # reuse fixture builder

        frames = [frame(frame_index=i) for i in range(14)]

        def comparable(detections):
            return [
                (
                    d.fault_id,
                    round(d.severity, 4),
                    round(d.confidence, 4),
                    d.observability,
                    d.start_frame,
                    d.end_frame,
                    d.peak_frame,
                    d.phase,
                )
                for d in detections
            ]

        for view_type in ["rear", "side", "rear_oblique", "front", "front_oblique"]:
            legacy = detect_rule_segments(
                compute_frame_metrics(frames, 30.0), fps=30.0, view_type=view_type, view_confidence=0.8
            )
            _, new = run_detector(registry.get_detector("Squat"), frames, 30.0, view_type, 0.8)
            self.assertEqual(comparable(legacy), comparable(new), f"mismatch for view_type={view_type}")

    def test_payload_routes_to_named_movement(self) -> None:
        from src.pose.pose_rule_detector import detect_pose_rules_from_payload
        from tests.test_overhead_press import ohp_frame

        frames = [ohp_frame(elbow_angle=140, wrist_y=0.30, frame_index=i) for i in range(12)]
        payload = {"metadata": {"fps": 30.0}, "frames": frames}
        result = detect_pose_rules_from_payload(payload, movement="Overhead Press")
        ids = {d["fault_id"] for d in result["detections"]}
        self.assertIn("ohp_incomplete_lockout", ids)

    def test_payload_unknown_movement_raises(self) -> None:
        from src.pose.pose_rule_detector import detect_pose_rules_from_payload
        from tests.test_overhead_press import ohp_frame

        frames = [ohp_frame(elbow_angle=140, wrist_y=0.30, frame_index=i) for i in range(12)]
        payload = {"metadata": {"fps": 30.0}, "frames": frames}
        with self.assertRaises(KeyError):
            detect_pose_rules_from_payload(payload, movement="No Such Movement")
