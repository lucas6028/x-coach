from __future__ import annotations

import math
import unittest

from src.pose.movements import registry
from src.pose.movements.base import REST_PHASE, RepPlan, run_detector
from src.pose.rep_segmentation import RepWindow
from tests.test_pose_rule_detector import frame


def squat_reps(n_reps: int, frames_per_rep: int = 30) -> list[dict]:
    """`n_reps` squats built from the shared frame fixture: the hip rides from standing
    (hip_y 0.45, above the knee) down to a deep bottom (hip_y 0.92) and back, so
    avg_knee_angle traces one excursion per rep."""
    frames: list[dict] = []
    for index in range(n_reps * frames_per_rep):
        theta = 2.0 * math.pi * (index % frames_per_rep) / frames_per_rep
        hip_y = 0.685 - 0.235 * math.cos(theta)  # 0.45 at the top, 0.92 at the bottom
        frames.append(frame(hip_y=hip_y, knee_y=0.70, frame_index=index))
    return frames


class RunDetectorPerRepTests(unittest.TestCase):
    def test_three_reps_are_detected_and_all_analyzed_by_default(self) -> None:
        result = run_detector(registry.get_detector("Squat"), squat_reps(3), 30.0, "rear", 0.8)
        self.assertEqual(len(result.reps), 3)
        self.assertEqual(len(result.analyzed), 3)
        self.assertIsNone(result.fallback)

    def test_every_rep_gets_its_own_descent_bottom_and_ascent(self) -> None:
        """The bug this change fixes: with one global argmin, reps 2 and 3 had no descent."""
        result = run_detector(registry.get_detector("Squat"), squat_reps(3), 30.0, "rear", 0.8)
        for rep in result.reps:
            phases = {c.phase for c in result.core[rep.start : rep.end + 1]}
            with self.subTest(rep=rep.index):
                self.assertIn("descent", phases)
                self.assertIn("bottom", phases)
                self.assertIn("ascent", phases)

    def test_five_reps_are_capped_to_first_middle_last(self) -> None:
        result = run_detector(
            registry.get_detector("Squat"), squat_reps(5), 30.0, "rear", 0.8, max_reps=3
        )
        self.assertEqual(len(result.reps), 5)
        self.assertEqual([r.index for r in result.analyzed], [1, 3, 5])

    def test_max_reps_zero_analyzes_every_rep(self) -> None:
        result = run_detector(
            registry.get_detector("Squat"), squat_reps(5), 30.0, "rear", 0.8, max_reps=0
        )
        self.assertEqual(len(result.analyzed), 5)

    def test_frames_outside_every_rep_are_rest_and_are_never_scored(self) -> None:
        """Walk-in / rack / rest frames must not be scored -- that is the noise this fixes.

        The idle frames are built with a KNEES-FORWARD posture so they WOULD fire a rule if
        they were scored; a neutral idle stretch would make this test pass for the wrong
        reason.
        """
        # hip_y is pinned to the standing value used at the top of `squat_reps` (0.45) rather
        # than left at the fixture's own default (0.72): the default, combined with this much
        # knee x-offset, drags avg_knee_angle -- the SEGMENTATION signal -- down to ~37 deg,
        # which is far enough into the excursion band that the segmenter folds these frames
        # into rep 1 instead of leaving them REST. Pinning hip_y removes that accidental
        # interaction; `rule_knees_forward` keys off `knee_forward_ratio` (a horizontal, ankle
        # relative metric), not `avg_knee_angle`, so the posture still fires the rule if scored.
        idle = [frame(left_knee_x=0.48, right_knee_x=0.88, hip_y=0.45, frame_index=i) for i in range(20)]
        working = squat_reps(2)
        for offset, item in enumerate(working):
            item["frame_index"] = 20 + offset
        result = run_detector(
            registry.get_detector("Squat"), idle + working, 30.0, "side", 0.8, max_reps=0
        )

        self.assertTrue(result.reps, "the working stretch must still segment")
        self.assertTrue(all(c.phase == REST_PHASE for c in result.core[:20]))

        # Detections are reported in frame_index units; rep windows are sequence positions.
        # Convert once, then require every detection to lie inside some ANALYZED rep.
        analyzed_ranges = [
            (result.core[rep.start].frame_index, result.core[rep.end].frame_index)
            for rep in result.analyzed
        ]
        self.assertTrue(analyzed_ranges)
        for detection in result.detections:
            with self.subTest(fault=detection.fault_id):
                self.assertTrue(
                    any(
                        start <= detection.start_frame and detection.end_frame <= end
                        for start, end in analyzed_ranges
                    ),
                    f"{detection.fault_id} spans {detection.start_frame}-{detection.end_frame}, "
                    f"outside every analyzed rep {analyzed_ranges}",
                )
                # Nothing may reach back into the idle stretch (frame_index 0-19).
                self.assertGreaterEqual(detection.start_frame, 20)

    def test_only_partial_reps_are_still_reported_even_though_the_clip_is_analyzed_whole(self) -> None:
        """A clip cut off before the lifter stands back up -- the labeled dataset's shape.

        One slow rep truncated two-thirds of the way through: long enough to clear the
        `2 * min_frames` guard, but the signal never returns above `exit` at the end, so the
        only window found is partial.
        """
        result = run_detector(
            registry.get_detector("Squat"), squat_reps(1, frames_per_rep=90)[:60], 30.0, "rear", 0.8
        )
        self.assertEqual(result.fallback, "only_partial_reps")
        self.assertTrue(result.reps, "partial reps must still be reported")
        self.assertTrue(all(rep.partial for rep in result.reps))
        self.assertEqual(result.analyzed, [])
        self.assertTrue(result.detections, "the clip must still be analyzed whole")

    def test_static_clip_falls_back_to_whole_clip_analysis(self) -> None:
        """A segmentation failure must never present as 'no faults found'."""
        frames = [frame(left_knee_x=0.48, right_knee_x=0.88, frame_index=i) for i in range(14)]
        result = run_detector(registry.get_detector("Squat"), frames, 30.0, "side", 0.8)
        self.assertEqual(result.reps, [])
        self.assertEqual(result.fallback, "no_reps_detected")
        self.assertTrue(result.detections, "fallback must still produce detections")
        self.assertFalse(any(c.phase == REST_PHASE for c in result.core))

    def test_segmentation_disabled_detector_falls_back(self) -> None:
        from dataclasses import replace

        detector = replace(registry.get_detector("Squat"), rep_signal=None)
        result = run_detector(detector, squat_reps(3), 30.0, "rear", 0.8)
        self.assertEqual(result.reps, [])
        self.assertEqual(result.fallback, "segmentation_disabled")
        self.assertTrue(result.detections, "a segmentation failure must still be analyzed, not read as clean")

    def test_assign_phases_returning_too_many_phases_raises(self) -> None:
        """A detector whose `assign_phases` returns one phase too many for a rep's slice must
        raise, not silently resize `phases` and misalign every later frame -- see base.py's
        slice-assignment guard. The message must name the detector AND both lengths, not just
        signal that something went wrong."""
        from dataclasses import replace

        good_detector = registry.get_detector("Squat")
        observed_input_lengths: list[int] = []

        def too_long(rep_frames: list[dict]) -> list[str]:
            observed_input_lengths.append(len(rep_frames))
            return good_detector.assign_phases(rep_frames) + ["ascent"]

        detector = replace(good_detector, assign_phases=too_long)
        with self.assertRaises(ValueError) as ctx:
            run_detector(detector, squat_reps(3), 30.0, "rear", 0.8)
        message = str(ctx.exception)
        self.assertIn(detector.name, message)
        # `too_long` always returns exactly one MORE phase than it was given: the guard's
        # message must report that expected (input) length and the actual (returned) length,
        # not just gesture at "a mismatch happened".
        self.assertEqual(len(observed_input_lengths), 1, "the guard must raise on the FIRST mismatch")
        expected_len = observed_input_lengths[0]
        self.assertIn(str(expected_len), message, "message must name the expected (slice) length")
        self.assertIn(str(expected_len + 1), message, "message must name the actual (returned) length")

    def test_assign_phases_returning_too_few_phases_raises(self) -> None:
        """The mirror-image direction: too SHORT must also raise explicitly here, not merely
        happen to raise IndexError later via the tail of the frame loop. Same message-content
        requirement as the too-long direction."""
        from dataclasses import replace

        good_detector = registry.get_detector("Squat")
        observed_input_lengths: list[int] = []

        def too_short(rep_frames: list[dict]) -> list[str]:
            observed_input_lengths.append(len(rep_frames))
            return good_detector.assign_phases(rep_frames)[:-1]

        detector = replace(good_detector, assign_phases=too_short)
        with self.assertRaises(ValueError) as ctx:
            run_detector(detector, squat_reps(3), 30.0, "rear", 0.8)
        message = str(ctx.exception)
        self.assertIn(detector.name, message)
        self.assertEqual(len(observed_input_lengths), 1, "the guard must raise on the FIRST mismatch")
        expected_len = observed_input_lengths[0]
        self.assertIn(str(expected_len), message, "message must name the expected (slice) length")
        self.assertIn(str(expected_len - 1), message, "message must name the actual (returned) length")

    def test_detections_carry_absolute_frame_indices(self) -> None:
        """Rules run on a SLICE, so a bug here would report REP-RELATIVE indices.

        Bounds-checking alone would not catch that -- rep-relative indices are also in range.
        The discriminating assertion is that each detection's frames fall inside the window of
        the rep it says it came from: a rep-relative index would land in rep 1's range while
        `rep_index` claimed rep 3.
        """
        result = run_detector(
            registry.get_detector("Squat"), squat_reps(3), 30.0, "rear", 0.8, max_reps=0
        )
        self.assertEqual(len(result.reps), 3)
        ranges = {
            rep.index: (result.core[rep.start].frame_index, result.core[rep.end].frame_index)
            for rep in result.reps
        }
        self.assertTrue(any(index > 1 for index in ranges), "need more than one rep to discriminate")

        for detection in result.detections:
            with self.subTest(fault=detection.fault_id):
                self.assertLessEqual(detection.start_frame, detection.peak_frame)
                self.assertLessEqual(detection.peak_frame, detection.end_frame)
                start, end = ranges[detection.rep_index]
                self.assertGreaterEqual(detection.start_frame, start)
                self.assertLessEqual(detection.end_frame, end)

    def test_a_fault_firing_in_several_reps_is_reported_once(self) -> None:
        result = run_detector(
            registry.get_detector("Squat"), squat_reps(3), 30.0, "rear", 0.8, max_reps=0
        )
        fault_ids = [d.fault_id for d in result.detections]
        self.assertEqual(len(fault_ids), len(set(fault_ids)), "each fault must appear once")
        repeated = [d for d in result.detections if d.rep_count > 1]
        self.assertTrue(repeated, "a fault present in every rep must record rep_count > 1")
        for detection in repeated:
            with self.subTest(fault=detection.fault_id):
                self.assertEqual(detection.rep_count, len(detection.occurred_reps))
                self.assertIn(detection.rep_index, detection.occurred_reps)


    def test_a_rep_plan_fallback_forces_analyzed_empty_even_if_the_plan_says_otherwise(self) -> None:
        """Defence in depth for a caller that bypasses the HTTP validator entirely.

        `backend/app/routers/analyze.py`'s `_validate_reps` already rejects a client `reps`
        payload that pairs a `fallback` string with an `analyzed=True` segment -- see its
        `AnalyzePoseRepsValidationTests::test_rejects_an_analyzed_segment_alongside_any_fallback`.
        But `run_detector` is also called directly, by the CLI and by tests, which do not go
        through that validator. A `RepPlan` with `fallback` set already forces whole-clip PHASE
        assignment (`segmented = []` above) -- if `analyzed` were still honoured from the plan,
        rules would score `core[rep.start:rep.end+1]` slices per-rep against phases that were
        never assigned at that granularity, which is exactly the mis-phasing this whole line of
        work exists to eliminate. This test exists so that guard survives even if the HTTP-layer
        validator is later relaxed or removed: `run_detector` must not trust `rep_plan.analyzed`
        when `rep_plan.fallback` is set, regardless of what the caller supplied.
        """
        window = RepWindow(index=1, start=0, end=29, partial=False)
        plan = RepPlan(reps=(window,), analyzed=(window,), fallback="segmentation_disabled")
        result = run_detector(
            registry.get_detector("Squat"), squat_reps(3), 30.0, "rear", 0.8, rep_plan=plan
        )
        self.assertEqual(result.analyzed, [])
        # Scoring took the whole-clip path (the `else` branch, `rule(core, ctx)`), not the
        # per-rep slice path -- the same shape as the existing fallback tests above.
        self.assertTrue(result.detections, "a fallback plan must still be analyzed, not read as clean")


if __name__ == "__main__":
    unittest.main()
