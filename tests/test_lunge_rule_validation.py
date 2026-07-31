import json
import math
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


def _landmark(x: float, y: float, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility}


def _side_view_frame() -> dict:
    """Sideways skeleton: left/right pairs coincide, so the torso reads maximally narrow."""
    lm = [_landmark(0.5, 0.5) for _ in range(33)]
    lm[0] = _landmark(0.5, 0.12)
    lm[11] = lm[12] = _landmark(0.5, 0.30)
    lm[23] = lm[24] = _landmark(0.5, 0.55)
    lm[25] = lm[26] = _landmark(0.5, 0.75)
    lm[27] = lm[28] = _landmark(0.5, 0.92)
    return {"frame_index": 0, "landmarks": lm, "world_landmarks": lm}


def _broad_frontal_frame() -> dict:
    """Facing the camera: wide torso, left landmark to the right of the right landmark."""
    lm = [_landmark(0.5, 0.5) for _ in range(33)]
    lm[0] = _landmark(0.5, 0.10)
    lm[11], lm[12] = _landmark(0.66, 0.28), _landmark(0.34, 0.28)
    lm[23], lm[24] = _landmark(0.62, 0.55), _landmark(0.38, 0.55)
    lm[25], lm[26] = _landmark(0.62, 0.72), _landmark(0.38, 0.72)
    lm[27], lm[28] = _landmark(0.62, 0.92), _landmark(0.38, 0.92)
    return {"frame_index": 0, "landmarks": lm, "world_landmarks": lm}


@dataclass(frozen=True)
class _Frame:
    """Minimal CoreFrame stand-in: the helpers only touch `.valid`, `.phase` and `.m(key)`."""

    phase: str
    valid: bool = True
    values: dict = None

    def m(self, key: str) -> float:
        return float((self.values or {}).get(key, math.nan))


@dataclass(frozen=True)
class _Rep:
    start: int
    end: int


@dataclass(frozen=True)
class _Result:
    core: list
    analyzed: list
    fallback: str | None
    detections: tuple = ()


def _record(person_id: str, correct: bool, fired: bool, score, camera: str = "cam17",
            orientation: str = "front", pass_name: str = "production",
            actionable: bool = True) -> dict:
    return {
        "person_id": person_id,
        "correct": correct,
        "camera": camera,
        "cam17_orientation": orientation,
        pass_name: {
            "fired": {"lunge_knee_valgus": {"severity": 1.0}} if fired else {},
            "scores": {"lunge_knee_valgus": score},
            "cannot_fire": {"lunge_knee_valgus": not actionable},
            "gate_open": {"lunge_knee_valgus": True},
            "actionable": {"lunge_knee_valgus": actionable},
            "lead_side": "left",
        },
    }


class SliceRepTests(unittest.TestCase):
    def test_slices_inclusive_of_both_bounds(self) -> None:
        from src.rehab24.lunge_rule_validation import slice_rep

        frames = [{"frame_index": i} for i in range(100)]
        window = slice_rep(frames, 10, 19)
        self.assertEqual(len(window), 10)
        self.assertEqual(window[0]["frame_index"], 10)
        self.assertEqual(window[-1]["frame_index"], 19)

    def test_clamps_a_last_frame_beyond_the_clip(self) -> None:
        # Labels come from the mocap timeline; a video can be a few frames shorter.
        from src.rehab24.lunge_rule_validation import slice_rep

        frames = [{"frame_index": i} for i in range(20)]
        self.assertEqual(len(slice_rep(frames, 15, 40)), 5)

    def test_returns_empty_when_the_window_starts_past_the_clip(self) -> None:
        from src.rehab24.lunge_rule_validation import slice_rep

        self.assertEqual(slice_rep([{"frame_index": i} for i in range(5)], 90, 99), [])

    def test_raises_on_negative_first_frame(self) -> None:
        from src.rehab24.lunge_rule_validation import slice_rep

        with self.assertRaises(ValueError):
            slice_rep([{"frame_index": i} for i in range(5)], -1, 3)

    def test_raises_when_last_frame_precedes_first_frame(self) -> None:
        from src.rehab24.lunge_rule_validation import slice_rep

        with self.assertRaises(ValueError):
            slice_rep([{"frame_index": i} for i in range(5)], 3, 1)


class ContingencyTests(unittest.TestCase):
    def test_counts_positives_as_incorrect_reps(self) -> None:
        from src.rehab24.lunge_rule_validation import contingency

        # rep 1: incorrect + fired  -> tp
        # rep 2: correct   + fired  -> fp
        # rep 3: correct   + silent -> tn
        # rep 4: incorrect + silent -> fn
        table = contingency(fired=[True, True, False, False],
                            correct=[False, True, True, False])
        self.assertEqual(table, {"tp": 1, "fp": 1, "tn": 1, "fn": 1})

    def test_the_positive_convention_is_not_symmetric_under_a_swap(self) -> None:
        # Asymmetric counts on purpose. The four-cells-at-one case above cannot catch a
        # `positive = is_correct` inversion: that swap permutes the cells without changing any
        # count. Here a swapped implementation would report {"tp":1,"fp":2,...} instead.
        from src.rehab24.lunge_rule_validation import contingency

        table = contingency(fired=[True, True, True], correct=[False, False, True])
        self.assertEqual(table, {"tp": 2, "fp": 1, "tn": 0, "fn": 0})

    def test_rejects_mismatched_lengths(self) -> None:
        from src.rehab24.lunge_rule_validation import contingency

        with self.assertRaises(ValueError):
            contingency(fired=[True], correct=[True, False])

    def test_empty_inputs_yield_an_all_zero_table(self) -> None:
        from src.rehab24.lunge_rule_validation import contingency

        self.assertEqual(contingency(fired=[], correct=[]), {"tp": 0, "fp": 0, "tn": 0, "fn": 0})


class RankAucTests(unittest.TestCase):
    def test_perfect_separation_scores_one(self) -> None:
        from src.rehab24.lunge_rule_validation import rank_auc

        self.assertAlmostEqual(
            rank_auc([1.0, 2.0, 8.0, 9.0], [False, False, True, True]), 1.0, places=6
        )

    def test_inverted_separation_scores_zero(self) -> None:
        from src.rehab24.lunge_rule_validation import rank_auc

        self.assertAlmostEqual(
            rank_auc([8.0, 9.0, 1.0, 2.0], [False, False, True, True]), 0.0, places=6
        )

    def test_ties_score_one_half(self) -> None:
        from src.rehab24.lunge_rule_validation import rank_auc

        self.assertAlmostEqual(rank_auc([5.0, 5.0], [True, False]), 0.5, places=6)

    def test_is_nan_when_one_class_is_empty(self) -> None:
        # A rule the dataset never exercises must report NaN, not a misleading 0.5.
        from src.rehab24.lunge_rule_validation import rank_auc

        self.assertTrue(math.isnan(rank_auc([1.0, 2.0], [True, True])))

    def test_ignores_non_finite_scores(self) -> None:
        from src.rehab24.lunge_rule_validation import rank_auc

        self.assertAlmostEqual(
            rank_auc([float("nan"), 1.0, 9.0], [True, False, True]), 1.0, places=6
        )


class CameraRoutingTests(unittest.TestCase):
    def test_sagittal_rules_read_cam18_and_frontal_rules_read_cam17(self) -> None:
        from src.rehab24.lunge_rule_validation import RULE_CAMERAS

        self.assertEqual(RULE_CAMERAS["lunge_knee_past_toes"], "cam18")
        self.assertEqual(RULE_CAMERAS["lunge_insufficient_depth"], "cam18")
        self.assertEqual(RULE_CAMERAS["lunge_knee_valgus"], "cam17")
        self.assertEqual(RULE_CAMERAS["lunge_pelvic_drop"], "cam17")

    def test_every_registered_lunge_rule_has_a_camera(self) -> None:
        # A rule added later without a routing entry would be silently dropped from the report.
        from src.pose.movements.lunge import LUNGE_DETECTOR
        from src.rehab24.lunge_rule_validation import RULE_CAMERAS

        emitted = {
            "lunge_knee_past_toes", "lunge_knee_valgus",
            "lunge_insufficient_depth", "lunge_pelvic_drop",
        }
        self.assertEqual(len(LUNGE_DETECTOR.rules), len(emitted))
        self.assertEqual(set(RULE_CAMERAS), emitted)


class OracleViewsTests(unittest.TestCase):
    def test_front_maps_to_a_label_production_can_never_emit(self) -> None:
        # Production always calls estimate_view_for_pose(allow_front=False), so "front" never
        # reaches the rules on the production path. The oracle pass deliberately bypasses that
        # gate to ask "would this rule fire if the view label were correct?"
        from src.rehab24.lunge_rule_validation import ORACLE_VIEWS

        self.assertEqual(ORACLE_VIEWS["front"], "front")

    def test_half_profile_maps_to_front_oblique(self) -> None:
        from src.rehab24.lunge_rule_validation import ORACLE_VIEWS

        self.assertEqual(ORACLE_VIEWS["half-profile"], "front_oblique")

    def test_side_maps_to_side(self) -> None:
        from src.rehab24.lunge_rule_validation import ORACLE_VIEWS

        self.assertEqual(ORACLE_VIEWS["side"], "side")

    def test_oracle_view_confidence_is_pinned_at_one(self) -> None:
        from src.rehab24.lunge_rule_validation import ORACLE_VIEW_CONFIDENCE

        self.assertEqual(ORACLE_VIEW_CONFIDENCE, 1.0)


class PerSubjectTests(unittest.TestCase):
    def test_groups_records_by_key_and_applies_value_fn_per_group(self) -> None:
        from src.rehab24.lunge_rule_validation import per_subject

        records = [
            {"person_id": "1", "score": 1.0},
            {"person_id": "1", "score": 3.0},
            {"person_id": "2", "score": 10.0},
        ]
        result = per_subject(
            records,
            key_fn=lambda r: r["person_id"],
            value_fn=lambda group: sum(r["score"] for r in group) / len(group),
        )
        self.assertEqual(result, {"1": 2.0, "2": 10.0})

    def test_does_not_pool_across_subjects(self) -> None:
        # 174 reps from 8 people are not independent: a pooled mean would let one subject's
        # count dominate. Grouping first and only then reducing is what per_subject exists for.
        from src.rehab24.lunge_rule_validation import per_subject

        records = [{"person_id": "1"}] * 100 + [{"person_id": "2"}] * 1
        result = per_subject(records, key_fn=lambda r: r["person_id"], value_fn=len)
        self.assertEqual(result, {"1": 100, "2": 1})

    def test_empty_records_yields_an_empty_dict(self) -> None:
        from src.rehab24.lunge_rule_validation import per_subject

        result = per_subject([], key_fn=lambda r: r["person_id"], value_fn=len)
        self.assertEqual(result, {})


class EstimateViewForWindowTests(unittest.TestCase):
    def test_a_narrow_torso_window_reads_side(self) -> None:
        from src.rehab24.lunge_rule_validation import estimate_view_for_window

        view, confidence = estimate_view_for_window([_side_view_frame() for _ in range(20)])
        self.assertEqual(view, "side")
        self.assertGreater(confidence, 0.2)

    def test_never_emits_front_because_allow_front_is_the_production_default(self) -> None:
        # `estimate_view_for_pose(allow_front=False)` is what production calls, so a frontal
        # window must come back as an oblique label, never "front".
        from src.rehab24.lunge_rule_validation import estimate_view_for_window

        view, _ = estimate_view_for_window([_broad_frontal_frame() for _ in range(20)])
        self.assertNotEqual(view, "front")

    def test_an_empty_window_is_unknown_at_zero_confidence(self) -> None:
        from src.rehab24.lunge_rule_validation import estimate_view_for_window

        self.assertEqual(estimate_view_for_window([]), ("unknown", 0.0))

    def test_matches_the_production_aggregation_on_the_same_frames(self) -> None:
        # Pins the mirror: whatever score_view returns for this window's aggregated signals is
        # exactly what estimate_view_for_window must return, including the NaN-not-zero
        # torso_width_ratio default.
        import numpy as np

        from src.pose.view_estimation import frame_view_signals, mean_finite, score_view
        from src.rehab24.lunge_rule_validation import estimate_view_for_window

        frames = [_side_view_frame() for _ in range(7)] + [_broad_frontal_frame() for _ in range(3)]
        valid = [s for s in (frame_view_signals(f) for f in frames) if s is not None]
        expected = score_view(
            orientation_score=mean_finite([s["orientation_score"] for s in valid], default=0.0),
            face_visibility=mean_finite([s["face_visibility"] for s in valid], default=0.0),
            torso_width_ratio=mean_finite([s["torso_width_ratio"] for s in valid], default=np.nan),
            z_asymmetry_value=mean_finite([s["z_asymmetry"] for s in valid], default=0.0),
            valid_frame_ratio=len(valid) / len(frames),
            allow_front=False,
        )
        self.assertEqual(estimate_view_for_window(frames), (expected[0], expected[1]))


class ScoreSpecTests(unittest.TestCase):
    def test_each_rule_gets_its_own_metric_key_phases_and_sign(self) -> None:
        from src.pose.movements.lunge import LUNGE_ACTIVE_PHASES, PELVIC_DROP_PHASES
        from src.rehab24.lunge_rule_validation import score_spec

        self.assertEqual(
            score_spec("lunge_knee_past_toes", "left"),
            ("left_knee_forward_ratio", frozenset(LUNGE_ACTIVE_PHASES), 1.0),
        )
        self.assertEqual(
            score_spec("lunge_knee_valgus", "right"),
            ("right_knee_medial_offset_ratio", frozenset(LUNGE_ACTIVE_PHASES), 1.0),
        )
        # Depth masks on `bottom` ALONE -- not LUNGE_ACTIVE_PHASES -- because the spec's
        # predicate is the rep's minimum, and descent/ascent transit >100 degrees every rep.
        self.assertEqual(
            score_spec("lunge_insufficient_depth", "left"),
            ("left_knee_angle", frozenset({"bottom"}), 1.0),
        )
        self.assertEqual(
            score_spec("lunge_pelvic_drop", "left"),
            ("pelvis_tilt_signed_deg", frozenset(PELVIC_DROP_PHASES), 1.0),
        )

    def test_pelvic_drop_sign_flips_with_the_lead_side(self) -> None:
        # Contralateral, not absolute: a right-lead lunge drops the LEFT hip, which is a
        # negative `pelvis_tilt_signed_deg`. Reading the magnitude would score an ipsilateral
        # drop -- a different fault -- as Trendelenburg.
        from src.rehab24.lunge_rule_validation import score_spec

        self.assertEqual(score_spec("lunge_pelvic_drop", "right")[2], -1.0)

    def test_every_routed_rule_has_a_score_spec(self) -> None:
        from src.rehab24.lunge_rule_validation import RULE_CAMERAS, score_spec

        for fault_id in RULE_CAMERAS:
            self.assertIsNotNone(score_spec(fault_id, "left"))

    def test_rejects_an_unknown_fault_id(self) -> None:
        from src.rehab24.lunge_rule_validation import score_spec

        with self.assertRaises(ValueError):
            score_spec("lunge_not_a_rule", "left")


class GateOpenTests(unittest.TestCase):
    def test_matches_each_rule_hard_gate(self) -> None:
        # Pins the restatement against the rules themselves: only two of the four gate on view.
        from src.pose.movements.lunge import (
            LUNGE_DETECTOR, rule_knee_past_toes, rule_pelvic_drop,
        )
        from src.pose.pose_rule_detector import SIDE_VIEW_CONF_THRESHOLD
        from src.rehab24.lunge_rule_validation import gate_open

        self.assertIn(rule_knee_past_toes, LUNGE_DETECTOR.rules)
        self.assertIn(rule_pelvic_drop, LUNGE_DETECTOR.rules)
        for view in ("side", "front", "front_oblique", "rear", "rear_oblique", "unknown"):
            self.assertEqual(
                gate_open("lunge_knee_past_toes", view, 1.0), view == "side",
                f"knee_past_toes gate wrong for {view}",
            )
            self.assertEqual(
                gate_open("lunge_pelvic_drop", view, 1.0), view != "side",
                f"pelvic_drop gate wrong for {view}",
            )
            # The other two downgrade observability off-view but never go silent.
            self.assertTrue(gate_open("lunge_knee_valgus", view, 1.0))
            self.assertTrue(gate_open("lunge_insufficient_depth", view, 1.0))
        # The side gate carries squat's shared confidence floor, not a new number.
        self.assertFalse(
            gate_open("lunge_knee_past_toes", "side", SIDE_VIEW_CONF_THRESHOLD - 0.01)
        )
        self.assertTrue(gate_open("lunge_knee_past_toes", "side", SIDE_VIEW_CONF_THRESHOLD))

    def test_rejects_an_unknown_fault_id(self) -> None:
        from src.rehab24.lunge_rule_validation import gate_open

        with self.assertRaises(ValueError):
            gate_open("lunge_not_a_rule", "side", 1.0)


class Angle2dTests(unittest.TestCase):
    @staticmethod
    def _points(overrides):
        from src.pose.geometry import landmarks_to_array

        lm = [_landmark(0.5, 0.5) for _ in range(33)]
        for index, (x, y, z) in overrides.items():
            lm[index] = {"x": x, "y": y, "z": z, "visibility": 0.95}
        return landmarks_to_array(lm)

    def test_a_right_angle_in_the_image_plane_reads_ninety(self) -> None:
        from src.rehab24.lunge_rule_validation import angle_2d

        points = self._points({23: (0.0, 0.0, 0.0), 25: (0.0, 1.0, 0.0), 27: (1.0, 1.0, 0.0)})
        self.assertAlmostEqual(angle_2d(points, 23, 25, 27), 90.0, places=4)

    def test_ignores_the_pseudo_depth_channel(self) -> None:
        # The whole point: geometry.angle_degrees uses z, this must not. Same x/y, wildly
        # different z, identical answer.
        from src.pose.geometry import angle_degrees
        from src.rehab24.lunge_rule_validation import angle_2d

        flat = self._points({23: (0.0, 0.0, 0.0), 25: (0.0, 1.0, 0.0), 27: (1.0, 1.0, 0.0)})
        deep = self._points({23: (0.0, 0.0, 0.9), 25: (0.0, 1.0, -0.4), 27: (1.0, 1.0, 0.7)})
        self.assertAlmostEqual(angle_2d(flat, 23, 25, 27), angle_2d(deep, 23, 25, 27), places=4)
        self.assertNotAlmostEqual(
            angle_degrees(flat, 23, 25, 27), angle_degrees(deep, 23, 25, 27), places=1
        )

    def test_is_nan_when_a_landmark_is_not_visible(self) -> None:
        from src.pose.geometry import landmarks_to_array
        from src.rehab24.lunge_rule_validation import angle_2d

        lm = [_landmark(0.5, 0.5) for _ in range(33)]
        lm[25] = {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.1}
        self.assertTrue(math.isnan(angle_2d(landmarks_to_array(lm), 23, 25, 27)))


class FaultThresholdsTests(unittest.TestCase):
    def test_thresholds_come_from_the_rule_module_not_a_restatement(self) -> None:
        from src.pose.movements.lunge import (
            LUNGE_DEPTH_MILD_DEG, LUNGE_PELVIC_TILT_MILD_DEG, LUNGE_VALGUS_MILD,
        )
        from src.pose.pose_rule_detector import KNEE_FORWARD_MILD
        from src.rehab24.lunge_rule_validation import fault_thresholds

        self.assertEqual(
            fault_thresholds(),
            {
                "lunge_knee_past_toes": KNEE_FORWARD_MILD,
                "lunge_knee_valgus": LUNGE_VALGUS_MILD,
                "lunge_insufficient_depth": LUNGE_DEPTH_MILD_DEG,
                "lunge_pelvic_drop": LUNGE_PELVIC_TILT_MILD_DEG,
            },
        )


class RulesWindowTests(unittest.TestCase):
    def test_returns_only_the_analyzed_slices_when_segmentation_succeeded(self) -> None:
        # Scoring the whole labeled window here would manufacture rows reading "the metric hit
        # 0.35 and the rule stayed silent" on frames the rule never saw.
        from src.rehab24.lunge_rule_validation import rules_window

        core = [_Frame(phase=str(i)) for i in range(10)]
        result = _Result(core=core, analyzed=[_Rep(2, 4)], fallback=None)
        self.assertEqual([f.phase for f in rules_window(result)], ["2", "3", "4"])

    def test_concatenates_multiple_analyzed_reps(self) -> None:
        from src.rehab24.lunge_rule_validation import rules_window

        core = [_Frame(phase=str(i)) for i in range(10)]
        result = _Result(core=core, analyzed=[_Rep(0, 1), _Rep(7, 8)], fallback=None)
        self.assertEqual([f.phase for f in rules_window(result)], ["0", "1", "7", "8"])

    def test_returns_the_whole_clip_on_a_fallback_path(self) -> None:
        from src.rehab24.lunge_rule_validation import rules_window

        core = [_Frame(phase=str(i)) for i in range(4)]
        result = _Result(core=core, analyzed=[], fallback="only_partial_reps")
        self.assertEqual(len(rules_window(result)), 4)


class PhaseFrameCountTests(unittest.TestCase):
    def test_counts_only_valid_in_phase_frames(self) -> None:
        from src.rehab24.lunge_rule_validation import phase_frame_count

        window = [
            _Frame(phase="bottom"),
            _Frame(phase="bottom", valid=False),
            _Frame(phase="descent"),
        ]
        self.assertEqual(phase_frame_count(window, {"bottom"}), 1)
        self.assertEqual(phase_frame_count(window, {"bottom", "descent"}), 2)

    def test_a_short_window_falls_below_min_frames(self) -> None:
        # A 15-frame rep gets ~5 `bottom` frames (the deepest 30%), below the 6-frame
        # min_frames floor at 30 fps -- so rule_insufficient_depth cannot fire at any angle.
        from src.rehab24.lunge_rule_validation import phase_frame_count

        window = [_Frame(phase="bottom") for _ in range(5)]
        self.assertLess(phase_frame_count(window, {"bottom"}), 6)


class MetricExtremeTests(unittest.TestCase):
    def test_takes_the_max_over_valid_in_phase_frames(self) -> None:
        from src.rehab24.lunge_rule_validation import metric_extreme

        window = [
            _Frame(phase="bottom", values={"k": 1.0}),
            _Frame(phase="bottom", values={"k": 9.0}),
            _Frame(phase="setup", values={"k": 99.0}),
            _Frame(phase="bottom", valid=False, values={"k": 50.0}),
        ]
        self.assertEqual(metric_extreme(window, "k", {"bottom"}, 1.0), 9.0)

    def test_a_negative_sign_selects_the_most_negative_value(self) -> None:
        from src.rehab24.lunge_rule_validation import metric_extreme

        window = [_Frame(phase="bottom", values={"k": -4.0}), _Frame(phase="bottom", values={"k": 2.0})]
        self.assertEqual(metric_extreme(window, "k", {"bottom"}, -1.0), 4.0)

    def test_is_nan_when_no_frame_qualifies(self) -> None:
        from src.rehab24.lunge_rule_validation import metric_extreme

        window = [_Frame(phase="setup", values={"k": 1.0})]
        self.assertTrue(math.isnan(metric_extreme(window, "k", {"bottom"}, 1.0)))

    def test_ignores_non_finite_metric_values(self) -> None:
        from src.rehab24.lunge_rule_validation import metric_extreme

        window = [
            _Frame(phase="bottom", values={"k": float("nan")}),
            _Frame(phase="bottom", values={"k": 3.0}),
        ]
        self.assertEqual(metric_extreme(window, "k", {"bottom"}, 1.0), 3.0)


class SpearmanTests(unittest.TestCase):
    def test_perfect_monotone_increase_is_one(self) -> None:
        from src.rehab24.lunge_rule_validation import spearman_rho

        self.assertAlmostEqual(spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, places=6)

    def test_perfect_monotone_decrease_is_minus_one(self) -> None:
        from src.rehab24.lunge_rule_validation import spearman_rho

        self.assertAlmostEqual(spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]), -1.0, places=6)

    def test_is_rank_based_not_linear(self) -> None:
        from src.rehab24.lunge_rule_validation import spearman_rho

        self.assertAlmostEqual(spearman_rho([1, 2, 3, 4], [1, 2, 3, 4000]), 1.0, places=6)

    def test_ignores_pairs_with_a_non_finite_member(self) -> None:
        from src.rehab24.lunge_rule_validation import spearman_rho

        self.assertAlmostEqual(
            spearman_rho([1, 2, 3, float("nan")], [10, 20, 30, 40]), 1.0, places=6
        )

    def test_is_nan_below_three_pairs(self) -> None:
        from src.rehab24.lunge_rule_validation import spearman_rho

        self.assertTrue(math.isnan(spearman_rho([1, 2], [3, 4])))

    def test_is_nan_when_one_variable_is_constant(self) -> None:
        from src.rehab24.lunge_rule_validation import spearman_rho

        self.assertTrue(math.isnan(spearman_rho([1, 1, 1], [1, 2, 3])))


class PercentileOfTests(unittest.TestCase):
    def test_reports_the_fraction_strictly_below(self) -> None:
        from src.rehab24.lunge_rule_validation import percentile_of

        self.assertAlmostEqual(percentile_of(3.0, [1.0, 2.0, 3.0, 4.0]), 50.0, places=6)

    def test_is_nan_with_no_finite_samples(self) -> None:
        from src.rehab24.lunge_rule_validation import percentile_of

        self.assertTrue(math.isnan(percentile_of(1.0, [float("nan")])))


class MedianAndRangeTests(unittest.TestCase):
    def test_reports_median_min_max_and_the_count_that_produced_them(self) -> None:
        # The count is not the number of subjects: a subject whose reps are all one class
        # yields a NaN AUC and drops out of the median's denominator.
        from src.rehab24.lunge_rule_validation import median_and_range

        self.assertEqual(median_and_range([0.4, float("nan"), 0.6, 0.8]), (0.6, 0.4, 0.8, 3))

    def test_all_nan_yields_nans_and_a_zero_count(self) -> None:
        from src.rehab24.lunge_rule_validation import median_and_range

        med, low, high, n = median_and_range([float("nan")])
        self.assertEqual(n, 0)
        self.assertTrue(math.isnan(med) and math.isnan(low) and math.isnan(high))


class IsCorrectTests(unittest.TestCase):
    def test_one_means_correct_and_zero_means_incorrect(self) -> None:
        from src.rehab24.lunge_rule_validation import is_correct

        @dataclass(frozen=True)
        class _Seg:
            correctness: int

        self.assertTrue(is_correct(_Seg(1)))
        self.assertFalse(is_correct(_Seg(0)))


class AssertDatasetShapeTests(unittest.TestCase):
    @staticmethod
    def _segments(reps=174, correct=78, front=88):
        @dataclass(frozen=True)
        class _Seg:
            correctness: int
            cam17_orientation: str
            person_id: str

        out = []
        for i in range(reps):
            out.append(
                _Seg(
                    correctness=1 if i < correct else 0,
                    cam17_orientation="front" if i < front else "half-profile",
                    person_id=str(i % 8),
                )
            )
        return out

    def test_accepts_the_pinned_ex5_shape(self) -> None:
        from src.rehab24.lunge_rule_validation import assert_dataset_shape

        assert_dataset_shape(self._segments())

    def test_stops_when_the_rep_count_moves(self) -> None:
        from src.rehab24.lunge_rule_validation import assert_dataset_shape

        with self.assertRaises(SystemExit):
            assert_dataset_shape(self._segments(reps=173, correct=78, front=88))

    def test_stops_when_the_correctness_polarity_inverts(self) -> None:
        # 96 correct / 78 incorrect is the inversion this guard exists to catch.
        from src.rehab24.lunge_rule_validation import assert_dataset_shape

        with self.assertRaises(SystemExit):
            assert_dataset_shape(self._segments(correct=96))

    def test_stops_when_a_profile_rep_appears(self) -> None:
        from src.rehab24.lunge_rule_validation import assert_dataset_shape

        segments = self._segments()
        with self.assertRaises(SystemExit):
            assert_dataset_shape(list(segments[:-1]) + [type(segments[-1])(0, "profile", "1")])


class LoadPoseFramesTests(unittest.TestCase):
    def test_accepts_zero_origin_contiguous_frames(self) -> None:
        from src.rehab24.lunge_rule_validation import load_pose_frames

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.json"
            path.write_text(json.dumps({"frames": [{"frame_index": i} for i in range(5)]}))
            self.assertEqual(len(load_pose_frames(path)), 5)

    def test_stops_on_a_gap_because_slice_rep_indexes_by_position(self) -> None:
        # A gap means every rep window is silently misaligned and every number is wrong.
        from src.rehab24.lunge_rule_validation import load_pose_frames

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gap.json"
            path.write_text(json.dumps({"frames": [{"frame_index": i} for i in (0, 1, 3)]}))
            with self.assertRaises(SystemExit):
                load_pose_frames(path)


class SubsetStatsTests(unittest.TestCase):
    def test_counts_and_auc_use_the_incorrect_is_positive_convention(self) -> None:
        from src.rehab24.lunge_rule_validation import subset_stats

        records = [
            _record("1", correct=False, fired=True, score=0.9),
            _record("1", correct=True, fired=False, score=0.1),
            _record("2", correct=False, fired=True, score=0.8),
            _record("2", correct=True, fired=False, score=0.2),
        ]
        stats = subset_stats(records, "lunge_knee_valgus", "production")
        self.assertEqual(stats["table"], {"tp": 2, "fp": 0, "tn": 2, "fn": 0})
        self.assertAlmostEqual(stats["sensitivity"], 1.0)
        self.assertAlmostEqual(stats["specificity"], 1.0)
        self.assertAlmostEqual(stats["pooled_auc"], 1.0)
        self.assertAlmostEqual(stats["subject_auc_median"], 1.0)
        self.assertEqual(stats["subject_auc_n"], 2)

    def test_a_single_class_subject_drops_out_of_the_per_subject_median(self) -> None:
        from src.rehab24.lunge_rule_validation import subset_stats

        records = [
            _record("1", correct=False, fired=True, score=0.9),
            _record("1", correct=True, fired=False, score=0.1),
            _record("2", correct=False, fired=True, score=0.5),
        ]
        stats = subset_stats(records, "lunge_knee_valgus", "production")
        self.assertEqual(stats["n_subjects"], 2)
        self.assertEqual(stats["subject_auc_n"], 1)

    def test_a_none_score_is_excluded_from_the_auc_denominator(self) -> None:
        from src.rehab24.lunge_rule_validation import subset_stats

        records = [
            _record("1", correct=False, fired=False, score=None),
            _record("1", correct=False, fired=True, score=0.9),
            _record("1", correct=True, fired=False, score=0.1),
        ]
        stats = subset_stats(records, "lunge_knee_valgus", "production")
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["n_scored"], 2)

    def test_a_structurally_silent_rep_inflates_the_unconditional_specificity(self) -> None:
        # THE defect this conditional table exists to expose: a rep the rule could never fire on
        # counts as a true negative, so the unconditional specificity reads 1.000 while the
        # conditional one reads 0.500 on the reps where the rule actually ran.
        from src.rehab24.lunge_rule_validation import subset_stats

        records = [
            _record("1", correct=True, fired=False, score=0.1, actionable=True),
            _record("1", correct=True, fired=True, score=0.9, actionable=True),
            _record("1", correct=True, fired=False, score=None, actionable=False),
            _record("1", correct=True, fired=False, score=None, actionable=False),
            _record("1", correct=False, fired=True, score=0.9, actionable=True),
        ]
        stats = subset_stats(records, "lunge_knee_valgus", "production")
        self.assertAlmostEqual(stats["specificity"], 3 / 4)
        self.assertEqual(stats["n_actionable"], 3)
        self.assertAlmostEqual(stats["actionable_specificity"], 0.5)

    def test_conditional_and_unconditional_agree_when_nothing_is_silenced(self) -> None:
        from src.rehab24.lunge_rule_validation import subset_stats

        records = [
            _record("1", correct=True, fired=False, score=0.1),
            _record("1", correct=False, fired=True, score=0.9),
        ]
        stats = subset_stats(records, "lunge_knee_valgus", "production")
        self.assertEqual(stats["table"], stats["actionable_table"])
        self.assertEqual(stats["n_actionable"], 2)

    def test_view_gated_reps_are_counted_separately_from_short_windows(self) -> None:
        from src.rehab24.lunge_rule_validation import subset_stats

        records = [_record("1", correct=True, fired=False, score=0.1) for _ in range(2)]
        records[0]["production"]["gate_open"]["lunge_knee_valgus"] = False
        records[0]["production"]["actionable"]["lunge_knee_valgus"] = False
        stats = subset_stats(records, "lunge_knee_valgus", "production")
        self.assertEqual(stats["view_gated"], 1)
        self.assertEqual(stats["cannot_fire"], 0)
        self.assertEqual(stats["n_actionable"], 1)

    def test_matched_n_restricts_both_lead_choices_to_the_reps_both_scored(self) -> None:
        # Rules out "the two columns differ because their denominators differ".
        from src.rehab24.lunge_rule_validation import matched_n_lines

        records = [
            _record("1", correct=False, fired=False, score=0.9),
            _record("1", correct=True, fired=False, score=0.1),
            _record("1", correct=True, fired=False, score=None),
        ]
        for r in records:
            r["production"]["scores_lead_oracle"] = {"lunge_knee_valgus": 0.5}
        line = matched_n_lines(records, "lunge_knee_valgus", "production")[0]
        self.assertIn("MATCHED n=2", line)

    def test_reads_an_alternate_score_key_for_the_lead_oracle_diagnostic(self) -> None:
        # The lead-oracle scores live beside the real ones so the same statistics can be run
        # over "the metric read off the leg the label names" without a second detector run.
        from src.rehab24.lunge_rule_validation import subset_stats

        records = [
            _record("1", correct=False, fired=False, score=0.1),
            _record("1", correct=True, fired=False, score=0.9),
        ]
        for r in records:
            r["production"]["scores_lead_oracle"] = {"lunge_knee_valgus": 1.0 - r["production"]["scores"]["lunge_knee_valgus"]}
        stats = subset_stats(records, "lunge_knee_valgus", "production", score_key="scores_lead_oracle")
        self.assertAlmostEqual(stats["pooled_auc"], 1.0)

    def test_a_missing_score_key_yields_no_scored_reps_rather_than_raising(self) -> None:
        from src.rehab24.lunge_rule_validation import subset_stats

        records = [
            _record("1", correct=False, fired=False, score=0.1),
            _record("1", correct=True, fired=False, score=0.9),
        ]
        stats = subset_stats(records, "lunge_knee_valgus", "production", score_key="absent")
        self.assertEqual(stats["n_scored"], 0)

    def test_does_not_fold_an_inverted_auc(self) -> None:
        # All four metrics are higher-is-worse, so an AUC below 0.5 is a real, reportable
        # inversion (the valgus contamination signature) -- never 1 - AUC.
        from src.rehab24.lunge_rule_validation import subset_stats

        records = [
            _record("1", correct=False, fired=False, score=0.1),
            _record("1", correct=True, fired=False, score=0.9),
        ]
        self.assertAlmostEqual(subset_stats(records, "lunge_knee_valgus", "production")["pooled_auc"], 0.0)


if __name__ == "__main__":
    unittest.main()
