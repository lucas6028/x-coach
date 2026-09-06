"""Unit tests for the REHAB24-6 within-session identity/appearance control.

The permutation p-value is only as trustworthy as the statistic underneath it, so the
cases here are deliberately hand-computable: a four-repetition session whose AUC can be
written down, a null whose entire support is enumerable, and a two-subject hierarchy
where equal subject weighting and equal sample weighting give different answers.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.rehab24 import videomae_appearance_only as appearance
from src.rehab24 import videomae_identity_control as control


def make_oof_row(
    session: str,
    repetition: int,
    camera: str,
    label: int,
    seed: int,
    probability: float,
    person: str = "1",
    exercise: str = "1",
) -> dict:
    return {
        "sample_id": f"Ex{exercise}_{session}_rep{repetition}_{camera}",
        "person_id": person,
        "exercise_id": exercise,
        "video_id": session,
        "repetition_number": str(repetition),
        "camera": camera,
        "label": label,
        "seed": seed,
        "test_subject": person,
        "probability": probability,
    }


class MidrankTests(unittest.TestCase):
    def test_ranks_are_one_based_and_ordered(self):
        np.testing.assert_allclose(control.midranks(np.array([0.3, 0.1, 0.2])), [3.0, 1.0, 2.0])

    def test_ties_take_the_average_rank(self):
        # Ranks 2 and 3 are tied, so both become 2.5 -- what the definitional AUC
        # implies. argsort would hand out 2 and 3 and quietly change the AUC.
        np.testing.assert_allclose(control.midranks(np.array([0.1, 0.5, 0.5, 0.9])), [1.0, 2.5, 2.5, 4.0])

    def test_matches_definitional_auc_on_random_data(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            scores = np.round(rng.random(12), 2)  # rounding forces ties
            labels = rng.integers(0, 2, size=12).astype(float)
            if len(np.unique(labels)) < 2:
                continue
            ranks = control.midranks(scores)
            n_pos = int(labels.sum())
            n_neg = int(len(labels) - n_pos)
            auc = (labels @ ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
            self.assertAlmostEqual(auc, control.pairwise_auc(labels.astype(int), scores), places=12)


class SessionStatisticTests(unittest.TestCase):
    def session(self, labels, ranks):
        return control.Session(
            session="Ex1_PM_000",
            person_id="1",
            exercise_id="1",
            labels=np.asarray(labels, dtype=float),
            avg_rank=np.asarray(ranks, dtype=float),
            repetitions=tuple(f"rep{i}" for i in range(len(labels))),
        )

    def test_perfect_ordering_scores_one(self):
        session = self.session([0, 0, 1, 1], [1, 2, 3, 4])
        self.assertAlmostEqual(float(session.auc(session.labels)), 1.0)

    def test_reversed_ordering_scores_zero(self):
        session = self.session([1, 1, 0, 0], [1, 2, 3, 4])
        self.assertAlmostEqual(float(session.auc(session.labels)), 0.0)

    def test_hand_computed_intermediate_case(self):
        # positives hold ranks 2 and 4 -> (6 - 3) / 4 = 0.75
        session = self.session([0, 1, 0, 1], [1, 2, 3, 4])
        self.assertAlmostEqual(float(session.auc(session.labels)), 0.75)

    def test_auc_is_vectorised_over_label_matrices(self):
        session = self.session([0, 0, 1, 1], [1, 2, 3, 4])
        matrix = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=float)
        np.testing.assert_allclose(session.auc(matrix), [1.0, 0.0])


class HierarchyTests(unittest.TestCase):
    def build(self, rows):
        scores = control.repetition_scores(rows, seeds=(42,))
        return control.build_sessions(scores, seeds=(42,), drop_subject=None)

    def test_cameras_are_averaged_into_one_repetition_score(self):
        rows = [
            make_oof_row("A", 1, "cam17", 1, 42, 0.2),
            make_oof_row("A", 1, "cam18", 1, 42, 0.8),
        ]
        scores = control.repetition_scores(rows, seeds=(42,))
        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(scores["Ex1_A_rep1"]["score_by_seed"][42], 0.5)

    def test_missing_camera_row_raises_rather_than_dropping(self):
        rows = [make_oof_row("A", 1, "cam17", 1, 42, 0.2)]
        with self.assertRaises(ValueError):
            control.repetition_scores(rows, seeds=(42,))

    def test_disagreeing_camera_labels_raise(self):
        rows = [
            make_oof_row("A", 1, "cam17", 1, 42, 0.2),
            make_oof_row("A", 1, "cam18", 0, 42, 0.8),
        ]
        with self.assertRaises(ValueError):
            control.repetition_scores(rows, seeds=(42,))

    def test_single_class_sessions_are_excluded_and_named(self):
        rows = []
        for repetition, label in ((1, 1), (2, 1)):
            for camera in ("cam17", "cam18"):
                rows.append(make_oof_row("A", repetition, camera, label, 42, 0.5 + 0.1 * repetition))
        for repetition, label in ((1, 0), (2, 1)):
            for camera in ("cam17", "cam18"):
                rows.append(make_oof_row("B", repetition, camera, label, 42, 0.2 * repetition))
        sessions, single_class = self.build(rows)
        self.assertEqual([s.session for s in sessions], ["Ex1_B"])
        self.assertEqual(single_class, ["Ex1_A"])

    def test_seed_averaging_happens_on_ranks_not_after_the_auc(self):
        # Seed 42 orders the positives last (AUC 1.0), seed 7 orders them first (0.0).
        # Both routes must give 0.5; the point of the test is that the rank-averaged
        # implementation agrees with the explicit mean of the two seeds' AUCs.
        rows = []
        for repetition, label, p42, p7 in ((1, 0, 0.1, 0.9), (2, 1, 0.9, 0.1)):
            for camera in ("cam17", "cam18"):
                rows.append(make_oof_row("A", repetition, camera, label, 42, p42))
                rows.append(make_oof_row("A", repetition, camera, label, 7, p7))
        scores = control.repetition_scores(rows, seeds=(42, 7))
        sessions, _ = control.build_sessions(scores, seeds=(42, 7), drop_subject=None)
        self.assertAlmostEqual(float(sessions[0].auc(sessions[0].labels)), 0.5)

    def test_subjects_are_weighted_equally_regardless_of_session_count(self):
        # P1 contributes two sessions at 1.0 and 0.0 (mean 0.5); P2 one session at 0.0.
        # Equal subject weighting gives 0.25; pooling sessions would give 1/3.
        def session(name, person, labels, ranks):
            return control.Session(
                session=name,
                person_id=person,
                exercise_id="1",
                labels=np.asarray(labels, dtype=float),
                avg_rank=np.asarray(ranks, dtype=float),
                repetitions=("a", "b"),
            )

        sessions = [
            session("s1", "1", [0, 1], [1, 2]),
            session("s2", "1", [1, 0], [1, 2]),
            session("s3", "2", [1, 0], [1, 2]),
        ]
        statistic = control.observed_statistic(sessions)
        self.assertAlmostEqual(statistic["mean"], 0.25)
        self.assertEqual(statistic["n_subjects"], 2)
        self.assertEqual(statistic["n_subjects_above_chance"], 0)


class PermutationTests(unittest.TestCase):
    def session(self, labels, ranks, person="1"):
        return control.Session(
            session=f"s{person}",
            person_id=person,
            exercise_id="1",
            labels=np.asarray(labels, dtype=float),
            avg_rank=np.asarray(ranks, dtype=float),
            repetitions=tuple(f"r{i}" for i in range(len(labels))),
        )

    def test_null_support_is_exactly_the_enumerable_set(self):
        # 2 positives among 4 repetitions -> C(4,2) = 6 assignments, AUCs in
        # {0, 0.25, 0.5, 0.75, 1}. Nothing outside that set may ever appear.
        sessions = [self.session([0, 0, 1, 1], [1, 2, 3, 4])]
        null = control.permutation_null(sessions, n_permutations=2000, seed=7)
        self.assertTrue(set(np.round(np.unique(null), 6)).issubset({0.0, 0.25, 0.5, 0.75, 1.0}))

    def test_null_is_centred_on_chance(self):
        sessions = [self.session([0, 0, 1, 1], [1, 2, 3, 4])]
        null = control.permutation_null(sessions, n_permutations=20000, seed=7)
        self.assertAlmostEqual(float(null.mean()), 0.5, places=2)

    def test_class_counts_are_preserved_by_every_permutation(self):
        labels = np.array([0, 0, 0, 1], dtype=float)
        sessions = [self.session(labels, [1, 2, 3, 4])]
        rng = np.random.default_rng(3)
        permuted = rng.permuted(np.tile(labels, (500, 1)), axis=1)
        np.testing.assert_array_equal(permuted.sum(axis=1), np.full(500, labels.sum()))

    def test_p_value_never_reports_zero(self):
        sessions = [self.session([0, 0, 1, 1], [1, 2, 3, 4])]
        null = control.permutation_null(sessions, n_permutations=100, seed=1)
        p_value = (1 + int(np.sum(null >= 1.0))) / (100 + 1)
        self.assertGreater(p_value, 0.0)

    def test_permutation_seed_is_reproducible(self):
        sessions = [self.session([0, 1, 0, 1], [1, 2, 3, 4])]
        first = control.permutation_null(sessions, n_permutations=200, seed=20260823)
        second = control.permutation_null(sessions, n_permutations=200, seed=20260823)
        np.testing.assert_array_equal(first, second)


class VerdictTests(unittest.TestCase):
    def statistic(self, mean, above):
        return {"mean": mean, "n_subjects_above_chance": above}

    def test_all_three_conditions_are_required(self):
        self.assertTrue(control.preregistered_verdict(self.statistic(0.60, 8), 0.001)["primary_pass"])
        self.assertFalse(control.preregistered_verdict(self.statistic(0.54, 8), 0.001)["primary_pass"])
        self.assertFalse(control.preregistered_verdict(self.statistic(0.60, 5), 0.001)["primary_pass"])
        self.assertFalse(control.preregistered_verdict(self.statistic(0.60, 8), 0.20)["primary_pass"])

    def test_failure_is_worded_as_undetermined(self):
        reading = control.preregistered_verdict(self.statistic(0.51, 5), 0.4)["reading"]
        self.assertIn("undetermined", reading)
        self.assertNotIn("no difference", reading)


class LabelAuditTests(unittest.TestCase):
    def rows(self):
        rows = []
        for video, labels in (("PM_000", (1, 0)), ("PM_001", (1, 1))):
            for repetition, label in enumerate(labels, start=1):
                for camera in ("cam17", "cam18"):
                    rows.append(
                        {
                            "sample_id": f"Ex1_{video}_rep{repetition}_{camera}",
                            "person_id": "1",
                            "exercise_id": "1",
                            "video_id": video,
                            "repetition_number": str(repetition),
                            "camera": camera,
                            "correctness": str(label),
                            "video_path": f"Ex1/{video}-{camera}.mp4",
                        }
                    )
        return rows

    def test_mixed_label_sessions_are_counted(self):
        table = control.audit_labels(self.rows())
        self.assertEqual(table["sessions"], 2)
        self.assertEqual(table["mixed_label_sessions"], 1)
        self.assertEqual(table["mixed_label_rows"], 4)
        self.assertEqual(table["dual_camera_sessions"], 2)

    def test_pairing_gate_flags_a_missing_view(self):
        rows = [row for row in self.rows() if row["sample_id"] != "Ex1_PM_000_rep1_cam18"]
        report = control.audit_pairing(rows)
        self.assertFalse(report["passed"])
        self.assertEqual(report["repetitions_without_camera_pair"], ["Ex1_PM_000_rep1"])

    def test_pairing_gate_flags_disagreeing_labels(self):
        rows = self.rows()
        rows[0]["correctness"] = "0"
        report = control.audit_pairing(rows)
        self.assertFalse(report["passed"])
        self.assertEqual(report["repetitions_with_disagreeing_labels"], ["Ex1_PM_000_rep1"])


class AppearanceControlTests(unittest.TestCase):
    def test_canonical_index_is_the_midpoint(self):
        self.assertEqual(appearance.canonical_frame_index(100), 50)
        self.assertEqual(appearance.canonical_frame_index(101), 50)

    def test_zero_length_video_fails_closed(self):
        with self.assertRaises(RuntimeError):
            appearance.canonical_frame_index(0)

    def test_source_video_gate_rejects_the_wrong_count(self):
        rows = [{"video_path": "a.mp4", "person_id": "1"}]
        with self.assertRaises(SystemExit):
            appearance.assert_source_video_count(rows)


if __name__ == "__main__":
    unittest.main()
