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
