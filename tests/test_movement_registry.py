from __future__ import annotations

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

    def test_pushup_resolves_case_insensitively(self) -> None:
        for spelling in ("Push-up", "push-up", "PUSH-UP"):
            with self.subTest(spelling=spelling):
                self.assertEqual(registry.get_detector(spelling).name, "Push-up")

    def test_pushup_registers_all_five_spec_rules(self) -> None:
        """Four firing rules plus the permanently-silent `rule_scapular_winging`, which is
        registered so the spec and the code stay 1:1 (see its docstring)."""
        from src.pose.movements import pushup

        detector = registry.get_detector("Push-up")
        self.assertEqual(
            [rule.__name__ for rule in detector.rules],
            [
                "rule_hip_sag",
                "rule_shallow_depth",
                "rule_elbow_flare",
                "rule_head_drop",
                "rule_scapular_winging",
            ],
        )
        self.assertIs(detector.compute_raw, pushup.pushup_compute_raw)
        self.assertIs(detector.assign_phases, pushup.pushup_assign_phases)

    def test_pushup_metric_keys_match_the_emitted_metrics(self) -> None:
        """TWO-WAY equality, not a spot check. `run_detector` builds each CoreFrame's metrics
        dict FROM `detector.metric_keys`, so a key `pushup_compute_raw` emits but the tuple omits
        is silently dropped and reads back as NaN. For `hand_offset_ratio` that failure is total:
        it is the camera-inversion guard in both `rule_hip_sag` and `rule_head_drop`, and
        `nan > 0.0` is False, so the two rules would go permanently silent with no error."""
        from src.pose.movements.pushup import PUSHUP_METRIC_KEYS, pushup_compute_raw
        from tests.test_pushup import pushup_frame

        raw = pushup_compute_raw([pushup_frame(frame_index=0)], 30.0)
        self.assertTrue(raw[0]["valid"], "fixture frame must be valid for this comparison")
        framework_keys = {"frame_index", "time", "valid", "lower_body_visibility"}
        emitted = set(raw[0]) - framework_keys
        self.assertEqual(set(PUSHUP_METRIC_KEYS), emitted)
        # No duplicates hiding a missing key behind a matching set size.
        self.assertEqual(len(PUSHUP_METRIC_KEYS), len(set(PUSHUP_METRIC_KEYS)))

    def test_low_observability_never_outranks_a_confident_detection(self) -> None:
        """`run_detector`'s sort key is `(observability == "low", -severity, start_frame)` and
        `False < True`, so EVERY non-low detection precedes every low one no matter how severe
        the low one is. That is what lets `rule_elbow_flare` be registered while hedging at
        observability `low` -- it lands last even at severity 0.5 against a severity-0.05 peer.
        Pinned because the same key is duplicated in `pose_rule_detector.detect_rule_segments`
        and both squat and OHP depend on it. Driven through the REAL `run_detector` -- with two
        canned rules rather than a re-typed sort expression, which would only test itself."""
        from src.pose.pose_rule_detector import PoseRuleDetection

        def detection(fault_id: str, severity: float, observability: str) -> PoseRuleDetection:
            return PoseRuleDetection(
                fault_id=fault_id,
                fault_name=fault_id,
                kg_query="",
                retrieval_mode="kg",
                severity=severity,
                confidence=severity,
                observability=observability,
                start_time=0.0,
                end_time=0.2,
                start_frame=0,
                end_frame=5,
                peak_frame=0,
                phase="bottom",
                evidence={},
            )

        # The `low` rule is listed FIRST so a passing result cannot come from list order.
        detector = MovementDetector(
            "Sort Probe",
            (),
            lambda frames, fps: [{"frame_index": 0, "time": 0.0, "valid": True}],
            lambda raw: ["bottom"],
            (
                lambda core, ctx: [detection("low_but_severe", 0.5, "low")],
                lambda core, ctx: [detection("high_but_mild", 0.05, "high")],
            ),
        )
        _, detections = run_detector(detector, [{}], 30.0, "side", 0.9)
        self.assertEqual([d.fault_id for d in detections], ["high_but_mild", "low_but_severe"])

    def test_payload_routes_to_pushup(self) -> None:
        from src.pose.pose_rule_detector import detect_pose_rules_from_payload
        from tests.test_pushup import pushup_frame

        # Shallow reps: the elbows never bend past the spec's 100 deg fire threshold.
        frames = [pushup_frame(elbow_angle=130.0, frame_index=i) for i in range(14)]
        payload = {"metadata": {"fps": 30.0}, "frames": frames}
        result = detect_pose_rules_from_payload(payload, movement="Push-up")
        self.assertIn("pushup_shallow_depth", {d["fault_id"] for d in result["detections"]})
        # Push-up-only metric keys reach the frame_metrics payload, and squat's do not --
        # i.e. the routing really swapped the metric layer, not just the rule list.
        metric_keys = set(result["frame_metrics"][0])
        self.assertIn("hip_offset_ratio", metric_keys)
        self.assertIn("hand_offset_ratio", metric_keys)
        # `avg_knee_angle` is a real entry in squat's METRIC_KEYS -- asserting on a key that
        # does not exist in EITHER movement would pass trivially and prove nothing.
        self.assertNotIn("avg_knee_angle", metric_keys)

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


class TestMovementRegistry(unittest.TestCase):
    def test_lists_all_three_detectors_in_registration_order(self) -> None:
        from src.pose.movements import registry

        names = [d.name for d in registry.list_detectors()]
        self.assertEqual(names, ["Squat", "Overhead Press", "Push-up"])

    def test_only_squat_is_validated(self) -> None:
        """Push-up and Overhead Press rules are literature-derived and never checked against
        ground-truth labels. The UI marks them Beta off this flag."""
        from src.pose.movements import registry

        validated = {d.name: d.validated for d in registry.list_detectors()}
        self.assertEqual(validated, {"Squat": True, "Overhead Press": False, "Push-up": False})

    def test_validated_defaults_to_false(self) -> None:
        """A new detector must fail toward Beta, never silently present as validated."""
        from src.pose.movements.base import MovementDetector

        detector = MovementDetector("Test", (), lambda frames, fps: [], lambda raw: [], ())
        self.assertFalse(detector.validated)

    def test_names_are_the_canonical_spellings(self) -> None:
        """These strings are simultaneously the KG `movement` scope and the frontend's
        movement.<Name> i18n key. A drift breaks both silently."""
        from src.pose.movements import registry

        self.assertEqual(
            {d.name for d in registry.list_detectors()},
            {"Squat", "Push-up", "Overhead Press"},
        )
