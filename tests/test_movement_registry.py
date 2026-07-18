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
        legacy = detect_rule_segments(compute_frame_metrics(frames, 30.0), fps=30.0, view_type="rear", view_confidence=0.8)
        _, new = run_detector(registry.get_detector("Squat"), frames, 30.0, "rear", 0.8)
        self.assertEqual([d.fault_id for d in legacy], [d.fault_id for d in new])
        self.assertEqual([round(d.severity, 4) for d in legacy], [round(d.severity, 4) for d in new])
