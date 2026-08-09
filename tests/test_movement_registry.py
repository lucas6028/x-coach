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

        # "unknown" is in the list because `rule_shallow_depth` and `rule_heel_rise` now branch
        # on it; a duplicated rule set drifts silently on exactly the branch nobody compares.
        for view_type in ["rear", "side", "rear_oblique", "front", "front_oblique", "unknown"]:
            legacy = detect_rule_segments(
                compute_frame_metrics(frames, 30.0), fps=30.0, view_type=view_type, view_confidence=0.8
            )
            new = run_detector(registry.get_detector("Squat"), frames, 30.0, view_type, 0.8).detections
            self.assertEqual(comparable(legacy), comparable(new), f"mismatch for view_type={view_type}")

    def test_pushup_resolves_case_insensitively(self) -> None:
        for spelling in ("Push-up", "push-up", "PUSH-UP"):
            with self.subTest(spelling=spelling):
                self.assertEqual(registry.get_detector(spelling).name, "Push-up")

    def test_lunge_detector_resolves_case_insensitively(self) -> None:
        from src.pose.movements.registry import get_detector

        self.assertEqual(get_detector("Lunge").name, "Lunge")
        self.assertEqual(get_detector("lunge").name, "Lunge")

    def test_lunge_is_not_marked_validated(self) -> None:
        # Thresholds are spec-derived; Phase 2 measures them. Beta until evidence says otherwise.
        from src.pose.movements.registry import get_detector

        self.assertFalse(get_detector("Lunge").validated)

    def test_row_detector_resolves_case_insensitively(self) -> None:
        from src.pose.movements.registry import get_detector

        self.assertEqual(get_detector("Row").name, "Row")
        self.assertEqual(get_detector("row").name, "Row")

    def test_row_is_not_marked_validated(self) -> None:
        # No labeled row repetition exists anywhere in this repository; see row.py's docstring.
        from src.pose.movements.registry import get_detector

        self.assertFalse(get_detector("Row").validated)

    def test_lunge_registers_all_four_spec_rules(self) -> None:
        """All four fire, unlike push-up's permanently-silent `rule_scapular_winging` --
        mirrors `test_pushup_registers_all_five_spec_rules`."""
        from src.pose.movements import lunge

        detector = registry.get_detector("Lunge")
        self.assertEqual(
            [rule.__name__ for rule in detector.rules],
            [
                "rule_knee_past_toes",
                "rule_knee_valgus",
                "rule_insufficient_depth",
                "rule_pelvic_drop",
            ],
        )
        self.assertIs(detector.compute_raw, lunge.lunge_compute_raw)
        self.assertIs(detector.assign_phases, lunge.lunge_assign_phases)

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
        detections = run_detector(detector, [{}], 30.0, "side", 0.9).detections
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

    def test_registered_detectors_declare_their_rep_signal(self) -> None:
        """A detector's rep signal must be one of the metrics it actually emits, or the
        segmenter would read NaN for every frame and silently find zero reps."""
        expected = {
            "Squat": ("avg_knee_angle", "min", "extended"),
            "Push-up": ("min_elbow_angle", "min", "extended"),
            "Overhead Press": ("avg_elbow_angle", "max", "extended"),
            "Lunge": ("min_knee_angle", "min", "extended"),
            # Deadlift is the one movement that starts flexed (bar on the floor), not extended
            # -- see base.py:55 and src/pose/movements/deadlift.py's registration comment.
            "Deadlift": ("hip_angle_deg", "min", "flexed"),
            "Row": ("min_elbow_angle", "min", "extended"),
            # The first movement whose rep signal is FRONTAL rather than sagittal, and the
            # second (after OHP) to peak at its signal's maximum -- hands together -> spread ->
            # together. Assigned by RS-SP1 spec §3.4; verified end-to-end in
            # tests/test_band_pull_apart.py::EndToEndSegmentationTest.
            "Band Pull Apart": ("wrist_spread_shoulder_norm", "max", "extended"),
            # The only detector whose rep signal AVERAGES the two arms rather than taking an
            # extremum, and the choice was measured rather than preferred: left/right elbow
            # angles correlate r=0.992-0.996 across all 8 Fit3D subjects, so the arms are in
            # phase and the mean halves per-arm landmark noise. Verified end-to-end in
            # tests/test_bicep_curl.py::EndToEndSegmentationTest.
            "Bicep Curl": ("avg_elbow_angle", "min", "extended"),
            # The second averaging signal, and the first whose polarity and averaging BOTH come
            # from measurement rather than inference: left/right arm elevation correlates
            # r=0.9896-0.9964 across all 8 Fit3D `side_lateral_raise` subjects. Peaks at its
            # MAXIMUM (arms down -> raised -> down). Verified end-to-end in
            # tests/test_arm_abduction.py::EndToEndSegmentationTest.
            "Arm Abduction": ("avg_arm_elevation_deg", "max", "extended"),
            # The third averaging signal, sharing Arm Abduction's metric and INVERTING its
            # polarity: this movement's effort peak is the W, where arm elevation is at its
            # LOWEST. Measured on REHAB24-6 Ex2's 208 annotated reps -- median start 140.4 deg,
            # median trough 54.7 at position 0.508 of the rep, median end 141.1 -- and the arms
            # correlate r=0.9977 within-rep (min 0.9628), which is what justifies averaging them.
            # Verified end-to-end in tests/test_arm_vw.py::EndToEndSegmentationTest, which
            # asserts WHICH end of the signal `peak` lands on so it cannot pass under `max`.
            "Arm VW": ("avg_arm_elevation_deg", "min", "extended"),
            # The fourth averaging signal, and the FIRST that is not an arm quantity: the trunk's
            # own `angle(shoulder, hip, knee)`. Peaks at its MINIMUM (supine -> curled -> supine),
            # sharing Arm VW's polarity. Chosen because it is JOINT-RELATIVE and therefore
            # invariant under camera roll -- the parent spec's "trunk flexion vs the
            # floor/horizontal" is not recoverable from the image, and EgoExo-Fitness ships its
            # sagittal sit-up frames rolled 90 degrees with no EXIF tag. Verified end-to-end in
            # tests/test_situp.py::EndToEndSegmentationTest, and the roll invariance is pinned
            # separately by tests/test_situp.py::RollInvarianceTest.
            "Sit-up": ("hip_angle_deg", "min", "extended"),
            # Shoulder Bridge reads the SAME signal as Sit-up on the SAME supine body, with the
            # INVERSE polarity: a bridge's effort peak is the hip OPENING to the shoulder-hip-knee
            # straight line, so it peaks at the signal's MAXIMUM, where a sit-up's curl peaks at
            # its minimum. `extended` names the end away from the effort peak -- here the hips
            # FLEXED on the mat, so the framework's word and the anatomy's word point opposite
            # ways and the framework's is what the flag selects. Roll invariance is pinned by
            # tests/test_shoulder_bridge.py::RollInvarianceTest.
            "Shoulder Bridge": ("hip_angle_deg", "max", "extended"),
            # Leg Abduction's signal is the only one in the registry that is a MAX OVER TWO
            # SIDES rather than a mean or a single side, and that is forced: `compute_raw` runs
            # over the whole clip before `segment_reps`, so there is no repetition boundary yet
            # and therefore no way to know which leg is working. The trunk-referenced pair is
            # used rather than the support-limb pair because only the trunk-referenced one is
            # side-independent -- see `leg_abduction._thigh_trunk_angles`. Roll invariance is
            # pinned by tests/test_leg_abduction.py::RollInvarianceTest.
            "Leg Abduction": ("max_thigh_trunk_deg", "max", "extended"),
        }
        for name, (signal, polarity, rep_start) in expected.items():
            with self.subTest(movement=name):
                detector = registry.get_detector(name)
                self.assertEqual(detector.rep_signal, signal)
                self.assertEqual(detector.rep_polarity, polarity)
                self.assertIn(detector.rep_signal, detector.metric_keys)
                # `rep_rectify` exists for movements RS-SP1 does not implement (spec §3.4);
                # all nine registered detectors use the default.
                self.assertFalse(detector.rep_rectify)
                self.assertEqual(detector.rep_start, rep_start)

    def test_multi_rep_clip_is_mis_phased_by_the_legacy_path_and_fixed_by_the_new_one(self) -> None:
        """Pins BOTH sides of the fix.

        The legacy whole-clip path takes one global argmin for the bottom frame, so on a
        three-rep clip everything after the first bottom is labelled `ascent` -- reps 2 and 3
        get no descent at all. The per-rep path must give every rep its own descent.
        """
        from src.pose.pose_rule_detector import compute_frame_metrics
        from tests.test_run_detector_per_rep import squat_reps

        frames = squat_reps(3)

        legacy_phases = [m.phase for m in compute_frame_metrics(frames, fps=30.0)]
        # Rep 3 lives in the final third; under one global argmin it never descends.
        self.assertNotIn("descent", legacy_phases[60:], "fixture is not multi-rep enough")

        result = run_detector(registry.get_detector("Squat"), frames, 30.0, "rear", 0.8)
        self.assertEqual(len(result.reps), 3)
        for rep in result.reps:
            phases = {c.phase for c in result.core[rep.start : rep.end + 1]}
            with self.subTest(rep=rep.index):
                self.assertIn("descent", phases)


class TestMovementRegistry(unittest.TestCase):
    def test_lists_all_detectors_in_registration_order(self) -> None:
        from src.pose.movements import registry

        names = [d.name for d in registry.list_detectors()]
        self.assertEqual(
            names,
            [
                "Squat", "Overhead Press", "Push-up", "Lunge", "Deadlift", "Row",
                "Band Pull Apart", "Bicep Curl", "Arm Abduction", "Arm VW", "Sit-up",
                "Shoulder Bridge", "Leg Abduction",
            ],
        )

    def test_only_squat_is_validated(self) -> None:
        """Push-up, Overhead Press, Lunge, Deadlift and Row rules are literature-derived and never
        checked against ground-truth labels. The UI marks them Beta off this flag."""
        from src.pose.movements import registry

        validated = {d.name: d.validated for d in registry.list_detectors()}
        self.assertEqual(
            validated,
            {
                "Squat": True,
                "Overhead Press": False,
                "Push-up": False,
                "Lunge": False,
                "Deadlift": False,
                "Row": False,
                "Band Pull Apart": False,
                "Bicep Curl": False,
                # Arm Abduction ships Beta even though REHAB24-6 Ex1 IS arm abduction with 178
                # human-labeled reps. Lunge got there first (Ex5, validated in
                # notes/lunge-rule-validation.md); this is the first movement whose labeled data
                # EXISTS while the check has NOT been run. See arm_abduction.py's registration
                # comment for what Ex1 can and cannot decide.
                "Arm Abduction": False,
                # Arm VW is the SECOND movement whose labeled data exists unchecked, and the
                # first whose labeled data matches the variant the app models: REHAB24-6 Ex2 is
                # arm VW, 208 reps (the largest labeled set of any non-squat movement), 94
                # correct / 114 incorrect, and BILATERAL on measurement. See arm_vw.py's
                # registration comment for what Ex2 does and does not decide.
                "Arm VW": False,
                # Sit-up is Beta for a THIRD distinct reason. Deadlift, Row, Band Pull Apart and
                # Bicep Curl are False because no labeled data exists; Arm Abduction and Arm VW
                # because nobody ran the check against data that does. Sit-up is False because the
                # labeled data that exists -- EgoExo-Fitness, 82 human-judged sit-up actions --
                # describes a DIFFERENT VARIANT: its canonical guidance is a full sit-up ("touch
                # your feet with your hands") while the parent spec specifies a curl-up. REHAB24-6
                # has no sit-up and Fit3D has no supine action at all. See situp.py's registration
                # comment.
                "Sit-up": False,
                # Shoulder Bridge is Beta for a FOURTH distinct reason, and the only one that a
                # DOWNLOAD fixes rather than research. The labels exist AND match the variant:
                # EgoExo-Fitness has 77 human-judged Shoulder Bridge actions whose canonical
                # guidance names this detector's endpoint verbatim, and one of its twelve criteria
                # IS this rule ("Progressively raise your body until your knees, hips, and
                # shoulders align in a straight line", faulted on 16/77). What is missing is the
                # PIXELS: `frames_open` downloads in 3 GiB parts, part `.ac` is absent, and only 2
                # of the 77 judged actions fall in a record that decodes. See shoulder_bridge.py's
                # registration comment.
                "Shoulder Bridge": False,
                # Leg Abduction is Beta for a FIFTH reason, and the first that is not a gap in
                # the evidence: the check WAS RUN. REHAB24-6 Ex4 is standing leg abduction --
                # 210 human-labeled repetitions, 120 correct / 90 incorrect, 9 subjects, two
                # orthogonal cameras -- and it is the matching variant, so unlike Sit-up there
                # was no escape hatch and unlike Shoulder Bridge the pixels are all present. The
                # run decided the roster: it silenced `rule_insufficient_abduction_rom` and
                # confirmed `rule_trunk_lean_compensation`'s signal. What it CANNOT establish is
                # a fault-level claim, because REHAB24-6 labels each repetition correct or
                # incorrect and never names which fault occurred. See leg_abduction.py's
                # registration comment and notes/leg-abduction-rule-validation.md.
                "Leg Abduction": False,
            },
        )

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
            {
                "Squat", "Push-up", "Overhead Press", "Lunge", "Deadlift", "Row",
                "Band Pull Apart", "Bicep Curl", "Arm Abduction", "Arm VW", "Sit-up",
                "Shoulder Bridge", "Leg Abduction",
            },
        )
