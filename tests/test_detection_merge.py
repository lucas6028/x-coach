from __future__ import annotations

import unittest

from src.pose.movements.base import merge_by_fault
from src.pose.pose_rule_detector import PoseRuleDetection


def detection(fault_id: str, severity: float, rep: int, start_frame: int) -> PoseRuleDetection:
    return PoseRuleDetection(
        fault_id=fault_id,
        fault_name=fault_id.replace("_", " ").title(),
        kg_query=fault_id,
        retrieval_mode="kg",
        severity=severity,
        confidence=severity,
        observability="high",
        start_time=start_frame / 30.0,
        end_time=(start_frame + 10) / 30.0,
        start_frame=start_frame,
        end_frame=start_frame + 10,
        peak_frame=start_frame + 5,
        phase="bottom",
        evidence={"primary_value": severity},
        rep_index=rep,
        occurred_reps=(rep,),
        rep_count=1,
    )


class MergeByFaultTests(unittest.TestCase):
    def test_same_fault_in_two_reps_becomes_one_entry(self) -> None:
        merged = merge_by_fault([detection("knees_inward", 0.4, 1, 0), detection("knees_inward", 0.7, 3, 60)])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].rep_count, 2)
        self.assertEqual(merged[0].occurred_reps, (1, 3))

    def test_representative_is_the_worst_occurrence(self) -> None:
        merged = merge_by_fault([detection("knees_inward", 0.4, 1, 0), detection("knees_inward", 0.7, 3, 60)])
        self.assertEqual(merged[0].severity, 0.7)
        self.assertEqual(merged[0].rep_index, 3)
        self.assertEqual(merged[0].start_frame, 60)
        self.assertEqual(merged[0].evidence["primary_value"], 0.7)

    def test_distinct_faults_are_not_merged(self) -> None:
        merged = merge_by_fault([detection("knees_inward", 0.4, 1, 0), detection("shallow_depth", 0.5, 1, 0)])
        self.assertEqual({d.fault_id for d in merged}, {"knees_inward", "shallow_depth"})

    def test_occurred_reps_are_sorted_and_deduplicated(self) -> None:
        merged = merge_by_fault(
            [detection("knees_inward", 0.4, 3, 60), detection("knees_inward", 0.5, 1, 0), detection("knees_inward", 0.3, 3, 70)]
        )
        self.assertEqual(merged[0].occurred_reps, (1, 3))
        self.assertEqual(merged[0].rep_count, 2)

    def test_equal_severity_ties_break_on_the_earlier_frame(self) -> None:
        merged = merge_by_fault([detection("knees_inward", 0.5, 3, 60), detection("knees_inward", 0.5, 1, 0)])
        self.assertEqual(merged[0].start_frame, 0)

    def test_empty_input(self) -> None:
        self.assertEqual(merge_by_fault([]), [])

    def test_whole_clip_fallback_detections_pass_through(self) -> None:
        """Fallback detections carry no rep, and must survive with rep_count 0."""
        item = detection("knees_inward", 0.5, 0, 0)
        item = item.__class__(**{**item.__dict__, "occurred_reps": (), "rep_count": 0})
        merged = merge_by_fault([item])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].rep_count, 0)
        self.assertEqual(merged[0].occurred_reps, ())


if __name__ == "__main__":
    unittest.main()
