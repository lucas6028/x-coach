"""Unit tests for the REHAB24-6 within-class position control.

The cases here are deliberately exact rather than statistical. Two constructions do
most of the work:

*A single-boundary session* -- the correct repetitions first, the erroneous ones after
-- is what REHAB24-6 actually looks like in 27 of its 61 recordings, and it is where a
feature that only knows "how far into the recording am I" scores a perfect
within-session AUC. Residualising it away must land on exactly 0.5, not near it.

*A balanced session* -- the same number of repetitions in each class -- makes the
within-class position target exactly uncorrelated with the label, so the ridge
direction is provably a coordinate axis and the projection can be checked by hand
instead of by tolerance.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.rehab24 import videomae_position_control as control
from src.rehab24.videomae_identity_control import observed_statistic
from src.video.videomae_video_classifier import clear_feature_cache


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #


def manifest_rows(
    subjects: int = 3,
    sessions_per_subject: int = 2,
    n_correct: int = 5,
    n_incorrect: int = 5,
) -> list[dict[str, str]]:
    """Manifest-shaped rows: every session is one contiguous correct block, then wrong.

    Both cameras of a repetition are emitted, because everything downstream averages
    them and a fixture with one view would hide a grouping bug rather than expose it.
    """
    rows: list[dict[str, str]] = []
    for subject in range(1, subjects + 1):
        for session in range(sessions_per_subject):
            video_id = f"V{subject}{session}"
            for number in range(1, n_correct + n_incorrect + 1):
                for camera in ("cam17", "cam18"):
                    rows.append(
                        {
                            "sample_id": f"Ex1_{video_id}_rep{number}_{camera}",
                            "exercise_id": "1",
                            "video_id": video_id,
                            "repetition_number": str(number),
                            "person_id": str(subject),
                            "camera": camera,
                            "correctness": "1" if number <= n_correct else "0",
                        }
                    )
    return rows


def feature_matrix(rows, repetitions, columns) -> tuple[list[str], np.ndarray]:
    """Build ``(sample_ids, features)`` from per-repetition column recipes."""
    by_sample = control.repetition_by_sample(repetitions)
    sample_ids = [row["sample_id"] for row in rows]
    features = np.stack(
        [np.asarray([column(by_sample[sample_id]) for column in columns], dtype=np.float64) for sample_id in sample_ids]
    )
    return sample_ids, features


def within_session_auc(repetitions, sample_ids, scores) -> float:
    values = {sample_id: float(score) for sample_id, score in zip(sample_ids, scores)}
    sessions = control.proxy_sessions(repetitions, values, drop_subject=None)
    return observed_statistic(sessions)["mean"]


def ridge_probe(features: np.ndarray, targets: np.ndarray, lam: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """The probe of plan §3.2, in-sample: standardise, ridge, predict."""
    mean, std = control.standardization_stats(features)
    standardized = (features - mean) / std
    beta = control.ridge_gram(standardized, targets).solve(lam)
    return standardized @ beta, beta


def probe_median(repetitions, sample_ids, predictions) -> float | None:
    scored: dict[str, list[float]] = {}
    by_sample = control.repetition_by_sample(repetitions)
    for sample_id, value in zip(sample_ids, predictions):
        scored.setdefault(by_sample[sample_id].repetition, []).append(float(value))
    averaged = {key: float(np.mean(values)) for key, values in scored.items()}
    cells = control.probe_cells(repetitions, averaged, drop_subject=None)
    if not cells:
        return None
    _, median = control.macro_from_cells(cells, control.cell_values(cells))
    return float(median[0])


# --------------------------------------------------------------------------- #
# (e) the within-class position target                                         #
# --------------------------------------------------------------------------- #


class WithinClassPositionTests(unittest.TestCase):
    def rows(self):
        """One hand-written session: three correct repetitions and a single wrong one."""
        rows = []
        for number, correctness in ((1, "1"), (2, "1"), (3, "1"), (4, "0")):
            for camera in ("cam17", "cam18"):
                rows.append(
                    {
                        "sample_id": f"Ex1_PM_000_rep{number}_{camera}",
                        "exercise_id": "1",
                        "video_id": "PM_000",
                        "repetition_number": str(number),
                        "person_id": "1",
                        "camera": camera,
                        "correctness": correctness,
                    }
                )
        return rows

    def test_ranks_within_the_label_class(self):
        positions = {rep.number: rep.position for rep in control.build_repetitions(self.rows())}
        self.assertAlmostEqual(positions[1], 0.0)
        self.assertAlmostEqual(positions[2], 0.5)
        self.assertAlmostEqual(positions[3], 1.0)

    def test_a_class_of_one_gets_the_midpoint(self):
        """0 or 1 would assert "earliest" or "latest"; 0.5 asserts nothing, which is
        the only honest value when a class holds a single repetition."""
        positions = {rep.number: rep.position for rep in control.build_repetitions(self.rows())}
        self.assertAlmostEqual(positions[4], 0.5)

    def test_naive_variant_ranks_across_the_whole_session(self):
        positions = {rep.number: rep.position for rep in control.build_repetitions(self.rows(), naive=True)}
        for number, expected in ((1, 0.0), (2, 1 / 3), (3, 2 / 3), (4, 1.0)):
            self.assertAlmostEqual(positions[number], expected)

    def test_both_cameras_of_a_repetition_share_the_target(self):
        targets = control.sample_targets(control.build_repetitions(self.rows()))
        for number in (1, 2, 3, 4):
            self.assertEqual(
                targets[f"Ex1_PM_000_rep{number}_cam17"], targets[f"Ex1_PM_000_rep{number}_cam18"]
            )

    def test_disagreeing_camera_labels_are_a_grouping_bug(self):
        rows = self.rows()
        rows[1]["correctness"] = "0"
        with self.assertRaises(ValueError):
            control.build_repetitions(rows)


# --------------------------------------------------------------------------- #
# (b) / (c) the intervention                                                   #
# --------------------------------------------------------------------------- #


class ResidualizationTests(unittest.TestCase):
    def setUp(self):
        self.rows = manifest_rows()
        self.repetitions = control.build_repetitions(self.rows)

    def test_a_pure_position_shortcut_collapses_to_chance(self):
        """A feature that is only "how far into the recording" scores a perfect
        within-session AUC on single-boundary recordings, which is exactly the
        confound this control exists to price. After k = 1 it must be gone entirely,
        not merely reduced."""
        sample_ids, features = feature_matrix(
            self.rows,
            self.repetitions,
            [lambda rep: -float(rep.number), lambda rep: 0.0, lambda rep: 0.0],
        )
        self.assertAlmostEqual(within_session_auc(self.repetitions, sample_ids, features[:, 0]), 1.0)

        targets = np.asarray(
            [control.repetition_by_sample(self.repetitions)[sid].position for sid in sample_ids], dtype=float
        )
        transform = control.fit_residual_transform(features, targets, k=1, lam=1.0)
        residualized = transform(features)
        np.testing.assert_allclose(residualized, 0.0, atol=1e-12)
        self.assertAlmostEqual(within_session_auc(self.repetitions, sample_ids, residualized[:, 0]), 0.5)

    def test_a_movement_direction_orthogonal_to_position_survives(self):
        """Column 0 drifts with within-class position, column 1 states the label. The
        classes are the same size, so the two are exactly uncorrelated and the fitted
        direction is provably the position axis -- the movement column comes through
        untouched."""
        sample_ids, features = feature_matrix(
            self.rows,
            self.repetitions,
            [lambda rep: rep.position, lambda rep: float(rep.label), lambda rep: 0.0],
        )
        targets = features[:, 0].copy()
        self.assertAlmostEqual(within_session_auc(self.repetitions, sample_ids, features[:, 1]), 1.0)

        transform = control.fit_residual_transform(features, targets, k=1, lam=1.0)
        np.testing.assert_allclose(transform.directions[0], [1.0, 0.0, 0.0], atol=1e-12)
        residualized = transform(features)
        np.testing.assert_allclose(residualized[:, 0], 0.0, atol=1e-12)
        self.assertAlmostEqual(within_session_auc(self.repetitions, sample_ids, residualized[:, 1]), 1.0)

    def test_the_projection_is_exactly_orthogonal_to_the_fitted_direction(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(120, 6))
        targets = features @ np.array([1.0, -0.5, 0.25, 0.0, 0.0, 0.0]) + 0.05 * rng.normal(size=120)
        transform = control.fit_residual_transform(features, targets, k=2, lam=1.0)
        residualized = transform(features)
        for direction in transform.directions:
            np.testing.assert_allclose(residualized @ direction, 0.0, atol=1e-10)

    def test_zero_directions_leaves_only_the_standardisation(self):
        rng = np.random.default_rng(1)
        features = rng.normal(size=(40, 4))
        transform = control.fit_residual_transform(features, rng.normal(size=40), k=0, lam=1.0)
        mean, std = control.standardization_stats(features)
        np.testing.assert_allclose(transform(features), (features - mean) / std)

    def test_a_single_vector_round_trips_with_its_own_shape(self):
        rng = np.random.default_rng(2)
        features = rng.normal(size=(30, 5))
        transform = control.fit_residual_transform(features, rng.normal(size=30), k=1, lam=1.0)
        np.testing.assert_allclose(transform(features[3]), transform(features)[3])
        self.assertEqual(transform(features[3]).shape, (5,))


# --------------------------------------------------------------------------- #
# (a) the probe                                                                #
# --------------------------------------------------------------------------- #


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.rows = manifest_rows()
        self.repetitions = control.build_repetitions(self.rows)
        self.by_sample = control.repetition_by_sample(self.repetitions)

    def test_probe_recovers_a_position_axis_and_loses_it_after_residualisation(self):
        """§6.3's positive control in miniature: the probe must read the direction
        before the intervention and must not read it after."""
        sample_ids, features = feature_matrix(
            self.rows,
            self.repetitions,
            [lambda rep: rep.position, lambda rep: float(rep.label), lambda rep: 0.0],
        )
        targets = np.asarray([self.by_sample[sid].position for sid in sample_ids], dtype=float)

        before, _ = ridge_probe(features, targets)
        self.assertAlmostEqual(probe_median(self.repetitions, sample_ids, before), 1.0, places=9)

        transform = control.fit_residual_transform(features, targets, k=1, lam=1.0)
        residualized = transform(features)
        after, beta_after = ridge_probe(residualized, targets)
        # Nothing linear is left to fit, so the ridge returns the zero vector and every
        # prediction ties. `probe_cells` drops those cells rather than scoring them 0.0,
        # which is why an emptied cell list -- not a rho of zero -- is the outcome here.
        np.testing.assert_allclose(beta_after, 0.0, atol=1e-10)
        self.assertIsNone(probe_median(self.repetitions, sample_ids, after))

    def test_probe_correlation_collapses_when_the_features_are_noisy(self):
        """The same claim without the exact-orthogonality crutch: a noisy design leaves
        a non-degenerate probe, and its within-class Spearman still collapses."""
        rng = np.random.default_rng(20260906)
        sample_ids, features = feature_matrix(
            self.rows,
            self.repetitions,
            [lambda rep: rep.position, lambda rep: float(rep.label), lambda rep: 0.0],
        )
        features = features + 0.05 * rng.normal(size=features.shape)
        features[:, 2] = rng.normal(size=len(features))
        targets = np.asarray([self.by_sample[sid].position for sid in sample_ids], dtype=float)

        before, _ = ridge_probe(features, targets)
        after, _ = ridge_probe(control.fit_residual_transform(features, targets, k=1, lam=1.0)(features), targets)
        self.assertGreater(probe_median(self.repetitions, sample_ids, before), 0.95)
        self.assertLess(abs(probe_median(self.repetitions, sample_ids, after)), 0.30)

    def test_a_cell_needs_three_repetitions(self):
        predictions = {rep.repetition: float(rep.number) for rep in self.repetitions}
        self.assertTrue(control.probe_cells(self.repetitions, predictions, drop_subject=None, min_reps=5))
        self.assertFalse(control.probe_cells(self.repetitions, predictions, drop_subject=None, min_reps=6))

    def test_constant_predictions_drop_the_cell_instead_of_scoring_zero(self):
        """Scoring a tied cell 0.0 would drag the macro toward chance and make the
        positive control pass for the wrong reason."""
        self.assertIsNone(control.rank_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
        predictions = {rep.repetition: 7.0 for rep in self.repetitions}
        self.assertEqual(control.probe_cells(self.repetitions, predictions, drop_subject=None), [])

    def test_permutation_null_of_the_probe_is_centred_on_chance(self):
        sample_ids, features = feature_matrix(
            self.rows,
            self.repetitions,
            [lambda rep: rep.position, lambda rep: float(rep.label), lambda rep: 0.0],
        )
        targets = np.asarray([self.by_sample[sid].position for sid in sample_ids], dtype=float)
        predictions, _ = ridge_probe(features, targets)
        scored: dict[str, list[float]] = {}
        for sample_id, value in zip(sample_ids, predictions):
            scored.setdefault(self.by_sample[sample_id].repetition, []).append(float(value))
        cells = control.probe_cells(
            self.repetitions, {key: float(np.mean(v)) for key, v in scored.items()}, drop_subject=None
        )
        null_mean, null_median = control.spearman_permutation_null(cells, 200, 20260906)
        self.assertLess(abs(float(null_mean.mean())), 0.1)
        self.assertEqual(null_median.shape, (200,))


# --------------------------------------------------------------------------- #
# (d) the position-balanced statistic                                          #
# --------------------------------------------------------------------------- #


def pair_scores(sessions: dict[str, list[tuple[int, int, float]]], person: str = "1") -> dict[str, dict]:
    """``repetition_scores``-shaped input: session -> [(rep number, label, score)]."""
    scored: dict[str, dict] = {}
    for name, items in sessions.items():
        for number, label, score in items:
            scored[f"Ex1_{name}_rep{number}"] = {
                "session": f"Ex1_{name}",
                "person_id": person,
                "exercise_id": "1",
                "label": label,
                "score_by_seed": {42: score},
            }
    return scored


class PositionBalancedTests(unittest.TestCase):
    def sessions(self):
        # Labels alternate, so both pair halves exist and the balanced statistic is
        # computable. Repetition numbers are distinct, so the position rule never ties.
        return control.build_pair_sessions(
            pair_scores(
                {
                    "A": [(1, 1, 0.9), (2, 0, 0.4), (3, 1, 0.8), (4, 0, 0.1)],
                    "B": [(1, 0, 0.2), (2, 1, 0.7), (3, 0, 0.3), (4, 1, 0.95)],
                }
            ),
            seeds=(42,),
            drop_subject=None,
        )

    def test_the_position_rule_scores_exactly_one_half(self):
        """By construction, not by luck: the rule wins every correct-first pair and
        loses every incorrect-first one, and the two halves carry equal weight."""
        sessions = self.sessions()
        rule = [control.position_rule_concordance(session.positions) for session in sessions]
        self.assertEqual(control.position_balanced_auc(sessions, rule)["mean"], 0.5)

    def test_the_position_rule_is_perfect_and_useless_on_the_two_halves(self):
        sessions = self.sessions()
        rule = [control.position_rule_concordance(session.positions) for session in sessions]
        self.assertEqual(control.pair_subset_auc(sessions, rule, "correct_first")["mean"], 1.0)
        self.assertEqual(control.pair_subset_auc(sessions, rule, "incorrect_first")["mean"], 0.0)

    def test_a_perfect_model_still_scores_one_after_balancing(self):
        sessions = self.sessions()
        model = [session.concordance for session in sessions]
        self.assertEqual(control.position_balanced_auc(sessions, model)["mean"], 1.0)

    def test_single_boundary_sessions_are_not_computable_and_are_named(self):
        """27 of the real 61 recordings have no incorrect-first pair at all; the split
        has to be reported, not silently dropped."""
        sessions = control.build_pair_sessions(
            pair_scores({"A": [(1, 1, 0.9), (2, 1, 0.8), (3, 0, 0.2), (4, 0, 0.1)]}),
            seeds=(42,),
            drop_subject=None,
        )
        self.assertEqual(control.interleaved_mask(sessions), [False])
        self.assertEqual(control.position_balanced_auc(sessions, [s.concordance for s in sessions])["n_sessions"], 0)
        self.assertEqual(control.subset_auc(sessions, [s.concordance for s in sessions], [True])["mean"], 1.0)

    def test_single_class_sessions_are_excluded(self):
        sessions = control.build_pair_sessions(
            pair_scores({"A": [(1, 1, 0.9), (2, 1, 0.8)]}), seeds=(42,), drop_subject=None
        )
        self.assertEqual(sessions, [])


# --------------------------------------------------------------------------- #
# (f) two-sided p and Holm                                                     #
# --------------------------------------------------------------------------- #


class InferenceHelperTests(unittest.TestCase):
    def test_two_sided_p_counts_both_tails(self):
        """An AUC of 0.2 is as much a shortcut as one of 0.8; the identity control was
        nearly fooled by exactly that asymmetry."""
        # Binary fractions, so the ">= " comparison sits nowhere near a rounding edge.
        # Distances from chance: 0.000, 0.125, 0.125, 0.250, 0.250.
        null = np.array([0.5, 0.625, 0.375, 0.75, 0.25])
        # |0.75 - 0.5| = 0.25, matched by two draws -> (1 + 2) / 6. The mirror image
        # 0.25 must give the identical p, which is the whole point of the two tails.
        self.assertAlmostEqual(control.two_sided_p(null, 0.75), 3 / 6)
        self.assertAlmostEqual(control.two_sided_p(null, 0.25), 3 / 6)
        # |0.5625 - 0.5| = 0.0625, matched or beaten by four draws -> (1 + 4) / 6
        self.assertAlmostEqual(control.two_sided_p(null, 0.5625), 5 / 6)
        # Nothing in the null reaches |0.9 - 0.5| -> the floor of (1 + 0) / 6
        self.assertAlmostEqual(control.two_sided_p(null, 0.9), 1 / 6)
        self.assertAlmostEqual(control.two_sided_p(null, 0.1), 1 / 6)

    def test_two_sided_p_never_reaches_zero(self):
        self.assertAlmostEqual(control.two_sided_p(np.full(9, 0.5), 1.0), 0.1)

    def test_holm_is_step_down_and_monotone(self):
        from src.rehab24.videomae_identity_control import holm_correct

        corrected = holm_correct({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(corrected["a"]["holm"], 0.03)
        self.assertAlmostEqual(corrected["c"]["holm"], 0.06)
        self.assertAlmostEqual(corrected["b"]["holm"], 0.06)
        self.assertTrue(corrected["a"]["significant"])
        self.assertFalse(corrected["b"]["significant"])


# --------------------------------------------------------------------------- #
# fold purity, lambda selection and the exploratory port                       #
# --------------------------------------------------------------------------- #


class FoldPurityTests(unittest.TestCase):
    def transform(self, seed: int) -> control.ResidualTransform:
        rng = np.random.default_rng(seed)
        features = rng.normal(size=(60, 5))
        return control.fit_residual_transform(features, rng.normal(size=60), k=1, lam=1.0)

    def record(self, seed: int) -> dict:
        transform = self.transform(seed)
        return {"transform_digest": transform.digest, "beta_digests": transform.direction_digests()}

    def test_distinct_fits_have_distinct_digests(self):
        control.assert_distinct_fold_digests([self.record(seed) for seed in (0, 1, 2)])

    def test_identical_fits_across_folds_fail_the_gate(self):
        """Two folds fitted on different training subjects cannot agree to 1e-8; if
        they do, one fit saw the whole dataset."""
        with self.assertRaises(SystemExit):
            control.assert_distinct_fold_digests([self.record(0), self.record(0)])

    def test_a_globally_fitted_beta_is_caught_even_with_per_fold_standardisers(self):
        """The leak the gate exists for: standardise per fold (so the transform digests
        differ) but fit beta once on everything. A transform-level check would wave that
        through, which is why the assertion is on the direction."""
        shared = self.transform(0).direction_digests()
        records = [
            {"transform_digest": self.transform(seed).digest, "beta_digests": shared} for seed in (0, 1)
        ]
        self.assertNotEqual(records[0]["transform_digest"], records[1]["transform_digest"])
        with self.assertRaises(SystemExit):
            control.assert_distinct_fold_digests(records)

    def test_lambda_selection_prefers_the_larger_lambda_on_a_tie(self):
        rows = manifest_rows(subjects=3)
        repetitions = control.build_repetitions(rows)
        sample_ids, features = feature_matrix(rows, repetitions, [lambda rep: 0.0, lambda rep: 0.0])
        by_sample = control.repetition_by_sample(repetitions)
        targets = np.asarray([by_sample[sid].position for sid in sample_ids], dtype=float)
        subjects = [by_sample[sid].person_id for sid in sample_ids]
        mean, std = control.standardization_stats(features)
        chosen, scores = control.select_lambda(
            (features - mean) / std, targets, subjects, [by_sample[sid] for sid in sample_ids]
        )
        self.assertEqual(chosen, max(control.DEFAULT_LAMBDA_GRID))
        self.assertEqual(set(scores), {f"{value:g}" for value in control.DEFAULT_LAMBDA_GRID})


class PositionRuleTests(unittest.TestCase):
    def test_loso_threshold_rule_on_a_hand_built_segmentation(self):
        """The zero-parameter shortcut: small repetition index -> predict correct."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Segmentation.csv"
            lines = ["video_id;repetition_number;exercise_id;person_id;correctness"]
            for subject in (1, 2, 3):
                for number in range(1, 5):
                    lines.append(f"V{subject};{number};1;{subject};{1 if number <= 2 else 0}")
            # P10 is excluded by default, and its rows are deliberately anti-correlated
            # so a leak would move the number.
            for number in range(1, 5):
                lines.append(f"V10;{number};1;10;{0 if number <= 2 else 1}")
            path.write_text("\n".join(lines), encoding="utf-8")

            report = control.position_rule_loso(path)
            self.assertEqual(report["n_subjects"], 3)
            self.assertAlmostEqual(report["mean"], 1.0)
            self.assertEqual(report["n_subjects_above_chance"], 3)


class ExploratoryOrderTests(unittest.TestCase):
    def scores(self):
        scored = pair_scores({"B": [(1, 1, 0.9), (2, 0, 0.1)]}, person="2")
        scored.update(pair_scores({"A": [(1, 1, 0.8), (2, 0, 0.2)]}, person="1"))
        return scored

    def test_exploratory_order_follows_first_appearance(self):
        """The permutation null draws one shuffle per session from one generator, so
        the iteration order is load-bearing for the null mean and for nothing else."""
        sessions = control.exploratory_sessions(self.scores(), seeds=(42,), drop_subject=None)
        self.assertEqual([session.session for session in sessions], ["Ex1_B", "Ex1_A"])

    def test_sorted_order_gives_the_identical_statistic(self):
        from src.rehab24.videomae_identity_control import build_sessions

        exploratory = control.exploratory_sessions(self.scores(), seeds=(42,), drop_subject=None)
        sorted_sessions, _ = build_sessions(
            {key: {**value, "score_by_seed": value["score_by_seed"]} for key, value in self.scores().items()},
            seeds=(42,),
            drop_subject=None,
        )
        self.assertEqual(
            observed_statistic(exploratory)["mean"], observed_statistic(sorted_sessions)["mean"]
        )


class BalancedAccuracySecondaryTests(unittest.TestCase):
    """``analyze``'s two report-only helpers, which the real run is the only other
    caller of. Both are places where a wrong answer would look plausible."""

    def folds(self, directory: Path, arm: str, values: dict[str, dict[str, float]]) -> None:
        for seed, per_subject in values.items():
            payload = {
                "seed": int(seed),
                "arm": arm,
                "folds": [
                    {
                        "test_subject": subject,
                        # P10's fold is 16 samples in the real data; it is dropped by
                        # the same n_test rule the framing summary used.
                        "n_test": 16 if subject == "10" else 200,
                        "balanced_accuracy": value,
                    }
                    for subject, value in per_subject.items()
                ],
            }
            (directory / f"folds_{arm}_seed{seed}.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_seed_averaged_subject_macro_excludes_p10(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            self.folds(
                directory,
                "k1",
                {
                    "42": {"1": 0.60, "2": 0.80, "10": 0.05},
                    "7": {"1": 0.70, "2": 0.90, "10": 0.05},
                },
            )
            report = control.loso_balanced_accuracy(directory, "k1", (42, 7))
            self.assertEqual(report["n_subjects"], 2)
            self.assertAlmostEqual(report["seed_averaged_by_subject"]["1"], 0.65)
            self.assertAlmostEqual(report["per_seed"]["42"], 0.70)
            self.assertAlmostEqual(report["mean"], 0.75)

    def test_missing_fold_records_report_nothing_rather_than_guessing(self):
        with tempfile.TemporaryDirectory() as name:
            self.assertIsNone(control.loso_balanced_accuracy(Path(name), "k1", (42,)))

    def test_paired_delta_uses_the_shared_subjects_only(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            summary = directory / "framing.json"
            summary.write_text(
                json.dumps(
                    {"arms": {"full_frame_letterbox": {"seed_averaged_by_subject": {"1": 0.60, "2": 0.70}}}}
                ),
                encoding="utf-8",
            )
            arm = {"seed_averaged_by_subject": {"1": 0.55, "2": 0.80, "3": 0.99}}
            delta = control.paired_delta_vs_framing(arm, summary)
            self.assertTrue(delta["available"])
            self.assertEqual(delta["n_subjects"], 2)
            self.assertAlmostEqual(delta["mean_delta"], 0.025)
            self.assertEqual(delta["n_positive"], 1)

    def test_a_missing_baseline_is_reported_not_approximated(self):
        """Inventing a per-subject baseline from the published 0.6612 mean would make a
        paired test out of an unpaired one."""
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            self.assertFalse(
                control.paired_delta_vs_framing({"seed_averaged_by_subject": {}}, directory / "absent.json")["available"]
            )
            empty = directory / "framing.json"
            empty.write_text(json.dumps({"arms": {"full_frame_letterbox": {}}}), encoding="utf-8")
            self.assertFalse(control.paired_delta_vs_framing({"seed_averaged_by_subject": {"1": 0.6}}, empty)["available"])


class DriftProxyTests(unittest.TestCase):
    def test_a_proxy_that_tracks_position_is_flagged_informative(self):
        rows = manifest_rows(subjects=2, sessions_per_subject=1)
        repetitions = control.build_repetitions(rows)
        proxies = {
            "drifting": {row["sample_id"]: -float(row["repetition_number"]) for row in rows},
            "flat": {row["sample_id"]: 1.0 for row in rows},
        }
        report = control.drift_proxy_report(repetitions, proxies, 200, 20260906)
        self.assertTrue(report["controls"]["drifting"]["informative"])
        self.assertAlmostEqual(report["controls"]["drifting"]["subject_macro"]["mean"], 1.0)
        self.assertFalse(report["controls"]["flat"]["informative"])
        self.assertEqual(set(report["holm"]), {"drifting", "flat"})

    def test_a_cached_luminance_file_at_another_stride_is_refused(self):
        """Reporting stride-4 numbers under a stride-8 label is exactly how a method
        changes without anyone noticing."""
        import csv

        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            rows = manifest_rows(subjects=1, sessions_per_subject=1)
            manifest = directory / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            control.write_luminance(
                directory / "luminance_per_sample.csv",
                {row["sample_id"]: (1.0, 3) for row in rows},
                stride=4,
            )
            with self.assertRaises(SystemExit):
                control.run_drift_proxies(manifest, directory, directory, directory, 10, 20260906, stride=8)


# --------------------------------------------------------------------------- #
# (g) run_arm's transform hook                                                 #
# --------------------------------------------------------------------------- #


class RunArmTransformHookTests(unittest.TestCase):
    """The default path has no other test, so byte-identity is checked here.

    ``MIN_VAL_SUBJECT_SAMPLES`` is 100, so the fixture has to give at least one
    non-test subject that many samples or ``pick_val_subject`` refuses the fold.
    """

    subjects = ("1", "2", "3")
    dimensions = 8

    def setUp(self):
        clear_feature_cache()
        self._directory = tempfile.TemporaryDirectory()
        root = Path(self._directory.name)
        self.source = root / "features"
        self.source.mkdir()
        self.materialize_root = root / "materialized"

        # Manifest-shaped so the real position-control factory can be driven against
        # the same fixture. Six sessions of ten repetitions from two cameras is 120
        # samples per subject, over the 100 a validation subject has to have.
        self.rows = manifest_rows(subjects=len(self.subjects), sessions_per_subject=6)
        self.repetitions = control.build_repetitions(self.rows)
        self.manifest = {row["sample_id"]: row for row in self.rows}
        self.by_sample = control.repetition_by_sample(self.repetitions)

        rng = np.random.default_rng(20260906)
        self.subject_samples = {subject: [] for subject in self.subjects}
        self.labels = {}
        for row in self.rows:
            sample_id = row["sample_id"]
            label = int(row["correctness"])
            rep = self.by_sample[sample_id]
            feature = rng.normal(scale=0.3, size=self.dimensions)
            feature[0] += rep.position  # a drift axis for the residualisation to find
            feature[1] += label  # a movement axis for it to leave alone
            np.savez_compressed(
                self.source / f"{sample_id}.npz",
                video_feature=feature.astype(np.float32),
                sample_id=np.asarray(sample_id),
                person_id=np.asarray(row["person_id"]),
                correctness=np.asarray(label),
            )
            self.labels[sample_id] = label
            self.subject_samples[row["person_id"]].append(sample_id)
        self.per_subject = len(self.subject_samples["1"])

    def tearDown(self):
        clear_feature_cache()
        self._directory.cleanup()

    def run_arm(self, **kwargs):
        import torch

        from src.rehab24.loso_cross_validation import FoldConfig
        from src.rehab24.videomae_stage_a import run_arm

        clear_feature_cache()
        return run_arm(
            self.source,
            self.labels,
            self.subject_samples,
            list(self.subjects),
            {subject: self.per_subject for subject in self.subjects},
            {},
            FoldConfig(epochs=1),
            torch.device("cpu"),
            seed=42,
            retain_predictions=True,
            **kwargs,
        )

    def test_an_identity_transform_reproduces_the_untouched_path_exactly(self):
        baseline = self.run_arm()
        transformed = self.run_arm(
            feature_transform_factory=lambda train_ids: (lambda features: features),
            materialize_root=self.materialize_root,
        )
        self.assertEqual(len(baseline), len(transformed))
        for mine, theirs in zip(baseline, transformed):
            self.assertEqual(mine["test_subject"], theirs["test_subject"])
            self.assertEqual(mine["sample_ids"], theirs["sample_ids"])
            self.assertEqual(mine["threshold"], theirs["threshold"])
            np.testing.assert_array_equal(mine["probabilities"], theirs["probabilities"])

    def test_the_factory_only_ever_sees_training_ids(self):
        """The fold-purity gate is enforced by the call site, not by a convention."""
        seen: list[set[str]] = []

        def factory(train_ids):
            seen.append(set(train_ids))
            return lambda features: features

        self.run_arm(feature_transform_factory=factory, materialize_root=self.materialize_root)
        self.assertEqual(len(seen), len(self.subjects))
        for subject, train_ids in zip(self.subjects, seen):
            self.assertTrue(train_ids.isdisjoint(self.subject_samples[subject]))

    def test_the_real_residual_factory_composes_end_to_end(self):
        """The path ``predict`` will take: fit per fold, materialise, train, and leave
        ten mutually distinct directions behind. Nothing here checks the *size* of the
        effect -- only that the pieces compose and the fold-purity gate holds."""
        records: list[dict] = []
        factory = control.residual_transform_factory(
            self.source, self.by_sample, self.manifest, k=1, record=records
        )
        folds = self.run_arm(feature_transform_factory=factory, materialize_root=self.materialize_root)

        self.assertEqual(len(records), len(self.subjects))
        control.assert_distinct_fold_digests(records)
        for record in records:
            self.assertIn(record["lambda"], control.DEFAULT_LAMBDA_GRID)
            self.assertEqual(len(record["beta_digests"]), 1)
        for fold in folds:
            self.assertEqual(len(fold["sample_ids"]), self.per_subject)
            self.assertTrue(np.all(np.isfinite(fold["probabilities"])))
            directory = self.materialize_root / f"fold_P{fold['test_subject']}"
            self.assertEqual(len(list(directory.glob("*.npz"))), len(self.rows))

        # The fitted direction is the drift axis, so the materialised features have no
        # component along it left.
        transform = control.fit_fold_transform(
            self.source, self.subject_samples["2"] + self.subject_samples["3"], self.by_sample, self.manifest, k=1
        )[0]
        residualized = transform(
            control.load_feature_matrix(self.source, self.subject_samples["1"])
        )
        np.testing.assert_allclose(residualized @ transform.directions[0], 0.0, atol=1e-10)

    def test_a_factory_without_a_materialize_root_is_refused(self):
        with self.assertRaises(SystemExit):
            self.run_arm(feature_transform_factory=lambda train_ids: (lambda features: features))

    def test_materialisation_writes_every_sample_and_reuses_a_matching_directory(self):
        from src.rehab24.videomae_stage_a import materialize_transformed_features

        transform = control.fit_residual_transform(
            np.stack(
                [
                    np.load(self.source / f"{sample_id}.npz")["video_feature"]
                    for sample_id in self.subject_samples["1"]
                ]
            ).astype(np.float64),
            np.arange(self.per_subject, dtype=float),
            k=1,
            lam=1.0,
        )
        ids = self.subject_samples["1"]
        destination = self.materialize_root / "fold_P1"
        self.assertEqual(materialize_transformed_features(self.source, ids, transform, destination), len(ids))
        self.assertEqual(len(list(destination.glob("*.npz"))), len(ids))
        signature = json.load((destination / "_materialization.json").open(encoding="utf-8"))
        self.assertEqual(signature["transform_digest"], transform.digest)

        # A second call with the same transform must not rewrite the directory.
        stamps = {path.name: path.stat().st_mtime_ns for path in destination.glob("*.npz")}
        self.assertEqual(materialize_transformed_features(self.source, ids, transform, destination), len(ids))
        self.assertEqual({path.name: path.stat().st_mtime_ns for path in destination.glob("*.npz")}, stamps)

    def test_materialised_features_carry_the_scalar_metadata(self):
        from src.rehab24.videomae_stage_a import materialize_transformed_features

        ids = self.subject_samples["2"][:5]
        destination = self.materialize_root / "meta"
        materialize_transformed_features(self.source, ids, lambda features: features * 2.0, destination)
        with np.load(destination / f"{ids[0]}.npz", allow_pickle=False) as data:
            self.assertEqual(str(data["sample_id"]), ids[0])
            self.assertEqual(str(data["person_id"]), "2")
            np.testing.assert_allclose(
                data["video_feature"],
                np.load(self.source / f"{ids[0]}.npz")["video_feature"] * 2.0,
                rtol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
