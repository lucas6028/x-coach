"""Within-class position control for VideoMAE on REHAB24-6.

Pre-registered in ``notes/rehab24_videomae_position_control_validation_plan.md``.

The identity control left the recording's *time axis* uncontrolled. Inside one
REHAB24-6 session the repetition index alone reaches balanced accuracy 0.7309 with a
single threshold -- higher than the whole framing pipeline's 0.6612 headline -- because
most recordings put the correct repetitions first and the erroneous ones after. Any
pixel-level quantity that drifts down a recording (lighting, camera settle, where the
subject stands, sweat, clothing slip) is therefore a route to a high within-session
AUC that has nothing to do with movement quality.

This module removes that route in the only place it can be removed for *all* 61
sessions: from the features. Per LOSO fold it fits, on the training subjects only, the
ridge direction that best predicts **within-class position** -- how early or late a
repetition sits among the repetitions *of its own label* in its own session -- and
projects that direction out of every sample's feature vector.

Three design points carry the inference.

*Within-class, not raw position.* Raw position and the label are nearly the same thing
inside a session (AUC 0.1761 backwards). A direction that predicts raw position also
predicts the label, so removing it would lower the AUC even for a model that reads only
the movement, and the drop would be uninterpretable. Fixing the class first makes the
target first-order orthogonal to the label: what is left is drift (plan §3.1).

*The intervention is fitted inside the fold and materialised to disk.* The projection
is fitted on the fold's training samples and then written out as ordinary npz feature
files, which ``train_one_fold`` re-reads unchanged. That is what lets the k = 0 arm
reproduce 0.8741 exactly: it is literally the same code path with an empty transform.

*The exploratory numbers are a gate, not a result.* ``replicate-exploratory`` re-derives
the five numbers that motivated the plan from the *existing* identity-control
out-of-fold probabilities. If the port disagrees in the fourth decimal the module has a
bug, and no new run may be interpreted.

A reproducibility fact worth stating once, because it looks like a discrepancy and is
not: the within-session permutation null draws one permutation per session, in the
order the sessions are iterated, from a single seeded generator. The exploratory
scripts iterate sessions in the order they first appear in the OOF CSV; the
identity-control ``build_sessions`` sorts them by name. Both give the identical observed
statistic (0.8741) and the identical p = 1/10001, but the null *mean* differs in the
fourth decimal (0.5002 vs 0.4999). ``replicate-exploratory`` reproduces the exploratory
order on purpose so its gate is exact; ``analyze`` uses the sorted order because it
reuses the identity-control machinery verbatim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.rehab24.dataset import DEFAULT_DATA_ROOT, DEFAULT_PROCESSED_ROOT, load_manifest
from src.rehab24.loso_cross_validation import (
    FoldConfig,
    MIN_VAL_SUBJECT_SAMPLES,
    pick_val_subject,
    subjects_to_samples,
)
from src.rehab24.videomae_identity_control import (
    ALPHA,
    DECISION_AUC,
    DEFAULT_ARM_DIR,
    MIN_SUBJECTS_ABOVE_CHANCE,
    SEEDS,
    SENSITIVITY_SUBJECT,
    Session,
    bootstrap_interval,
    build_sessions,
    exact_wilcoxon_vs,
    holm_correct,
    manifest_index,
    midranks,
    observed_statistic,
    permutation_null,
    read_oof,
    repetition_id,
    repetition_scores,
    session_id,
    subject_macro,
    verify_estimator,
    write_oof,
)
from src.rehab24.videomae_stage_a import load_metadata, run_arm

DEFAULT_OUTPUT_DIR = DEFAULT_PROCESSED_ROOT / "videomae_position_control"
DEFAULT_SEGMENTATION = DEFAULT_DATA_ROOT / "Segmentation.csv"
DEFAULT_BOX_GEOMETRY_DIR = DEFAULT_PROCESSED_ROOT / "box_geometry_features"
DEFAULT_FRAMING_SUMMARY = DEFAULT_PROCESSED_ROOT / "videomae_framing_summary.json"

#: Plan §6.5 freezes the permutation seed before any new OOF is read.
DEFAULT_PERMUTATION_SEED = 20260906
DEFAULT_PERMUTATIONS = 10_000
DEFAULT_BOOTSTRAP = 10_000

#: Plan §3.1 step 2. Fixed grid, selected by inner leave-one-training-subject-out;
#: written down here so a re-run cannot quietly widen it after seeing an outcome.
DEFAULT_LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)

#: A Spearman over two repetitions is either +1 or -1 and carries no information about
#: whether the features track position, so a cell needs at least three.
PROBE_MIN_REPS = 3

#: Plan §7. ``0.80`` comes from the §5 exploratory prior (0.8741 - 0.07); ``0.55`` is
#: this project's practical threshold. Neither is an equivalence bound.
PRIMARY_AUC_FLOOR = 0.80

#: The published headline this control is testing, and the framing arm it has to keep.
#: Used only to express a share and a paired delta, never as a test statistic.
REFERENCE_WITHIN_SESSION_AUC = 0.8741
FRAMING_BASELINE_BA = 0.6612

#: Plan §2. Hard-coded so a grouping bug cannot redefine the expectation it is checked
#: against.
EXPECTED_MIXED_SESSIONS = 61
EXPECTED_INTERLEAVED_SESSIONS = 34

#: Plan §3.3. Every proxy is a zero-parameter, model-free quantity.
LUMINANCE_STRIDE = 4
LUMINANCE_BACKENDS = ("cv2", "ffmpeg")
DEFAULT_LUMINANCE_BACKEND = "ffmpeg"
FFMPEG_WORKERS = 4


# --------------------------------------------------------------------------- #
# naming                                                                       #
# --------------------------------------------------------------------------- #


def arm_name(k: int, naive: bool = False) -> str:
    """``k1`` / ``naive_k1``: one string that keys every artifact of one arm."""
    return f"naive_k{k}" if naive else f"k{k}"


def oof_path(output_dir: Path, arm: str, seed: int) -> Path:
    return output_dir / f"oof_{arm}_seed{seed}.csv"


def folds_path(output_dir: Path, arm: str, seed: int) -> Path:
    return output_dir / f"folds_{arm}_seed{seed}.json"


def features_root(output_dir: Path, arm: str) -> Path:
    return output_dir / f"features_{arm}"


def fold_feature_dir(output_dir: Path, arm: str, test_subject: str) -> Path:
    return features_root(output_dir, arm) / f"fold_P{test_subject}"


# --------------------------------------------------------------------------- #
# within-class position target (plan §3.1 step 1)                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Repetition:
    """One repetition: the unit the position target is defined on.

    ``position`` is the pre-registered target -- the repetition's rank among the
    repetitions of *its own label* in its own session, mapped to [0, 1]. Both camera
    rows of the repetition share it, because correctness and position are properties of
    the repetition, not of the view.
    """

    repetition: str
    session: str
    person_id: str
    exercise_id: str
    label: int
    number: int
    position: float
    sample_ids: tuple[str, ...]


def build_repetitions(rows: Sequence[dict[str, str]], naive: bool = False) -> list[Repetition]:
    """Manifest rows -> repetitions carrying the within-class position target.

    ``naive`` ranks among *all* repetitions of the session regardless of label. The
    plan keeps it as a secondary arm precisely because it is expected to over-remove:
    raw position is collinear with the label, so the direction it defines is partly the
    label direction (plan §3.1, §3.4).

    A class holding a single repetition gets 0.5 -- the midpoint, which carries no
    early/late information -- rather than 0 or 1, which would assert one.
    """
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        session = session_id(row)
        number = int(row["repetition_number"])
        entry = grouped[session].setdefault(
            number,
            {
                "label": int(row["correctness"]),
                "person_id": row["person_id"],
                "exercise_id": row["exercise_id"],
                "sample_ids": [],
            },
        )
        if entry["label"] != int(row["correctness"]):
            raise ValueError(f"{session} rep{number} carries disagreeing labels across camera rows.")
        entry["sample_ids"].append(row["sample_id"])

    repetitions: list[Repetition] = []
    for session in sorted(grouped):
        members = grouped[session]
        classes: dict[int, list[int]] = defaultdict(list)
        for number, entry in members.items():
            classes[-1 if naive else entry["label"]].append(number)

        positions: dict[int, float] = {}
        for numbers in classes.values():
            ordered = sorted(numbers)
            size = len(ordered)
            for rank, number in enumerate(ordered):
                positions[number] = 0.5 if size == 1 else rank / (size - 1)

        for number in sorted(members):
            entry = members[number]
            repetitions.append(
                Repetition(
                    repetition=f"{session}_rep{number}",
                    session=session,
                    person_id=entry["person_id"],
                    exercise_id=entry["exercise_id"],
                    label=entry["label"],
                    number=number,
                    position=positions[number],
                    sample_ids=tuple(sorted(entry["sample_ids"])),
                )
            )
    return repetitions


def sample_targets(repetitions: Sequence[Repetition]) -> dict[str, float]:
    return {sample_id: rep.position for rep in repetitions for sample_id in rep.sample_ids}


def repetition_by_sample(repetitions: Sequence[Repetition]) -> dict[str, Repetition]:
    return {sample_id: rep for rep in repetitions for sample_id in rep.sample_ids}


# --------------------------------------------------------------------------- #
# ridge, standardisation, residualisation (plan §3.1 steps 2-3)                #
# --------------------------------------------------------------------------- #


def standardization_stats(features: np.ndarray, epsilon: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Train-set z-score statistics, using ``compute_feature_normalization``'s rule.

    A near-constant coordinate would otherwise be divided by ~0 and dominate the ridge
    fit; the classifier already guards it the same way, so the residualised features
    and the classifier's own normalisation agree on what "constant" means.
    """
    matrix = np.asarray(features, dtype=np.float64)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    return mean, np.where(std < epsilon, 1.0, std)


@dataclass(frozen=True)
class RidgeGram:
    """Sufficient statistics for a ridge fit, so held-out subjects cost a subtraction.

    Inner leave-one-training-subject-out over 8 subjects x 6 lambdas would otherwise be
    48 passes over an ~1800 x 768 matrix per fold. ``XtX``/``Xty`` are additive over
    rows, so the held-out fit is the full Gram minus that subject's Gram -- exact, not
    an approximation.
    """

    xtx: np.ndarray
    xty: np.ndarray
    xsum: np.ndarray
    ysum: float
    n: int

    def __sub__(self, other: "RidgeGram") -> "RidgeGram":
        return RidgeGram(
            xtx=self.xtx - other.xtx,
            xty=self.xty - other.xty,
            xsum=self.xsum - other.xsum,
            ysum=self.ysum - other.ysum,
            n=self.n - other.n,
        )

    def solve(self, lam: float) -> np.ndarray:
        """beta = (XtX + lam I)^-1 Xt(y - ybar): centred target, no fitted intercept."""
        mean_y = self.ysum / self.n
        rhs = self.xty - mean_y * self.xsum
        gram = self.xtx + lam * np.eye(self.xtx.shape[0])
        return np.linalg.solve(gram, rhs)

    @property
    def mean_y(self) -> float:
        return self.ysum / self.n


def ridge_gram(features: np.ndarray, targets: np.ndarray) -> RidgeGram:
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    return RidgeGram(
        xtx=matrix.T @ matrix,
        xty=matrix.T @ target,
        xsum=matrix.sum(axis=0),
        ysum=float(target.sum()),
        n=int(matrix.shape[0]),
    )


def ridge_direction(features: np.ndarray, targets: np.ndarray, lam: float) -> np.ndarray:
    """Unit-length ridge direction; the scale is irrelevant to a projection."""
    beta = ridge_gram(features, targets).solve(lam)
    norm = float(np.linalg.norm(beta))
    if norm == 0.0:
        raise ValueError("Ridge returned a zero direction; nothing to project out.")
    return beta / norm


@dataclass(frozen=True)
class ResidualTransform:
    """Standardise with the fold's training statistics, then project out ``k`` directions.

    Applied to train, validation and test samples alike. Only the *fit* is restricted to
    the training subjects (plan §3.1 step 3 / gate §6.2); applying a train-fitted
    transform to the test subject is not leakage, it is the intervention.
    """

    mean: np.ndarray
    std: np.ndarray
    directions: np.ndarray
    lam: float

    def __call__(self, features: np.ndarray) -> np.ndarray:
        original = np.asarray(features, dtype=np.float64)
        matrix = np.atleast_2d(original)
        matrix = (matrix - self.mean) / self.std
        for direction in self.directions:
            matrix = matrix - np.outer(matrix @ direction, direction)
        return matrix.reshape(original.shape)

    @property
    def k(self) -> int:
        return int(self.directions.shape[0])

    @property
    def digest(self) -> str:
        """sha256 of the transform, rounded to 1e-8 -- the fold-purity gate's evidence.

        Two folds fitted on different training subjects cannot produce the same digest;
        if they do, the fit saw the whole dataset (plan §6.2).
        """
        payload = b"".join(
            np.round(np.asarray(part, dtype=np.float64), 8).tobytes()
            for part in (self.mean, self.std, self.directions, np.asarray([self.lam]))
        )
        return hashlib.sha256(payload).hexdigest()

    def direction_digests(self) -> list[str]:
        return [
            hashlib.sha256(np.round(direction, 8).tobytes()).hexdigest() for direction in self.directions
        ]


def fit_residual_transform(
    features: np.ndarray,
    targets: np.ndarray,
    k: int,
    lam: float,
) -> ResidualTransform:
    """Sequentially fit and remove ``k`` ridge directions in standardised space."""
    mean, std = standardization_stats(features)
    matrix = (np.asarray(features, dtype=np.float64) - mean) / std
    target = np.asarray(targets, dtype=np.float64)

    directions: list[np.ndarray] = []
    for _ in range(k):
        direction = ridge_direction(matrix, target, lam)
        directions.append(direction)
        matrix = matrix - np.outer(matrix @ direction, direction)

    stacked = np.stack(directions) if directions else np.zeros((0, matrix.shape[1]))
    return ResidualTransform(mean=mean, std=std, directions=stacked, lam=float(lam))


# --------------------------------------------------------------------------- #
# within-class Spearman (plan §3.2)                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProbeCell:
    """One (session, label class) group with enough repetitions to correlate."""

    person_id: str
    session: str
    label: int
    prediction_ranks: np.ndarray
    target_ranks: np.ndarray


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    """Spearman as the Pearson correlation of midranks; ``None`` if either is constant.

    Returning ``None`` rather than 0.0 matters: residualised features can predict a
    near-constant value, and folding those cells in as zeros would drag the statistic
    toward chance and make the §6.3 positive control pass for the wrong reason.
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return None
    return float(a @ b / denominator)


def probe_cells(
    repetitions: Sequence[Repetition],
    predictions: dict[str, float],
    drop_subject: str | None = SENSITIVITY_SUBJECT,
    min_reps: int = PROBE_MIN_REPS,
) -> list[ProbeCell]:
    """Group scored repetitions into (session, class) cells, ranks pre-computed.

    Ranks are stored rather than raw values because the permutation null permutes
    target *positions* inside a cell, which permutes their ranks and leaves the
    prediction ranks untouched -- so every draw is a cheap reshuffle of a fixed vector.
    """
    grouped: dict[tuple[str, int], list[Repetition]] = defaultdict(list)
    for rep in repetitions:
        if drop_subject is not None and rep.person_id == drop_subject:
            continue
        if rep.repetition not in predictions:
            continue
        grouped[(rep.session, rep.label)].append(rep)

    cells: list[ProbeCell] = []
    for (session, label), members in sorted(grouped.items()):
        if len(members) < min_reps:
            continue
        members = sorted(members, key=lambda rep: rep.number)
        prediction_ranks = midranks(np.asarray([predictions[rep.repetition] for rep in members], dtype=float))
        target_ranks = midranks(np.asarray([rep.position for rep in members], dtype=float))
        if np.ptp(prediction_ranks) == 0 or np.ptp(target_ranks) == 0:
            continue
        cells.append(
            ProbeCell(
                person_id=members[0].person_id,
                session=session,
                label=label,
                prediction_ranks=prediction_ranks,
                target_ranks=target_ranks,
            )
        )
    return cells


def macro_from_cells(cells: Sequence[ProbeCell], values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cell -> session mean -> subject mean -> (macro mean, macro median) per column.

    ``values`` is ``(n_cells, n_draws)``, so the observed statistic and every
    permutation replicate travel the identical hierarchy. The two label classes of one
    session are averaged first (plan §3.2), then sessions inside a subject, then the
    nine subjects.
    """
    by_session: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, cell in enumerate(cells):
        by_session[(cell.person_id, cell.session)].append(index)

    by_subject: dict[str, list[np.ndarray]] = defaultdict(list)
    for (person_id, _), indices in by_session.items():
        by_subject[person_id].append(values[indices, :].mean(axis=0))

    subject_values = np.stack(
        [np.stack(rows, axis=0).mean(axis=0) for _, rows in sorted(by_subject.items(), key=lambda kv: int(kv[0]))]
    )
    return subject_values.mean(axis=0), np.median(subject_values, axis=0)


def cell_values(cells: Sequence[ProbeCell]) -> np.ndarray:
    values = []
    for cell in cells:
        correlation = rank_correlation(cell.prediction_ranks, cell.target_ranks)
        values.append(0.0 if correlation is None else correlation)
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)


def spearman_by_subject(cells: Sequence[ProbeCell]) -> dict[str, float]:
    """Per-subject within-class Spearman -- the unit lambda selection is scored on."""
    if not cells:
        return {}
    values = cell_values(cells)
    by_session: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, cell in enumerate(cells):
        by_session[(cell.person_id, cell.session)].append(index)
    by_subject: dict[str, list[float]] = defaultdict(list)
    for (person_id, _), indices in by_session.items():
        by_subject[person_id].append(float(values[indices, 0].mean()))
    return {person: float(np.mean(rows)) for person, rows in sorted(by_subject.items(), key=lambda kv: int(kv[0]))}


def spearman_permutation_null(
    cells: Sequence[ProbeCell],
    n_permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Plan §3.2: permute the within-class positions inside each session and class.

    The already-fitted predictions are re-scored rather than the probe re-fitted: the
    null asks "could this ordering have arisen by chance?", not "could a different
    model have been fitted?", and re-fitting 10,000 ridges would answer neither.
    """
    rng = np.random.default_rng(seed)
    draws = np.empty((len(cells), n_permutations), dtype=np.float64)
    for index, cell in enumerate(cells):
        centred_prediction = cell.prediction_ranks - cell.prediction_ranks.mean()
        permuted = rng.permuted(np.tile(cell.target_ranks, (n_permutations, 1)), axis=1)
        centred_target = permuted - permuted.mean(axis=1, keepdims=True)
        # A permutation cannot change a vector's norm, so the denominator is constant.
        denominator = float(np.linalg.norm(centred_prediction)) * float(np.linalg.norm(cell.target_ranks - cell.target_ranks.mean()))
        draws[index] = (centred_target @ centred_prediction) / denominator
    return macro_from_cells(cells, draws)


# --------------------------------------------------------------------------- #
# pairwise within-session statistics (plan §3.4, ported from the exploration)   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PairSession:
    """A session kept in pair form, because the position-balanced statistic reweights
    *pairs*, which a rank-sum AUC has already collapsed."""

    session: str
    person_id: str
    positions: np.ndarray
    labels: np.ndarray
    concordance: np.ndarray


def concordance_matrix(scores: np.ndarray) -> np.ndarray:
    """``C[i, j] = 1`` if i outranks j, 0.5 on a tie -- the AUC's summand."""
    return (scores[:, None] > scores[None, :]).astype(float) + 0.5 * (scores[:, None] == scores[None, :])


def position_rule_concordance(positions: np.ndarray) -> np.ndarray:
    """The zero-parameter rule "earlier repetition = more likely correct"."""
    return concordance_matrix(-np.asarray(positions, dtype=float))


def build_pair_sessions(
    scores: dict[str, dict],
    seeds: Sequence[int],
    drop_subject: str | None = SENSITIVITY_SUBJECT,
    sort_sessions: bool = True,
) -> list[PairSession]:
    """Mixed-label sessions in pair form.

    ``sort_sessions=False`` keeps the order in which sessions first appear in the OOF
    rows, which is what the exploratory scripts used; see the module docstring for why
    that order is load-bearing for the null and for nothing else.
    """
    order: dict[str, list[str]] = {}
    meta: dict[str, dict] = {}
    for key, entry in scores.items():
        if drop_subject is not None and entry["person_id"] == drop_subject:
            continue
        order.setdefault(entry["session"], []).append(key)
        meta.setdefault(entry["session"], entry)

    names = sorted(order) if sort_sessions else list(order)
    sessions: list[PairSession] = []
    for name in names:
        keys = sorted(order[name], key=lambda key: int(key.rsplit("rep", 1)[1]))
        labels = np.asarray([scores[key]["label"] for key in keys], dtype=float)
        if len(np.unique(labels)) < 2:
            continue
        positions = np.asarray([int(key.rsplit("rep", 1)[1]) for key in keys], dtype=float)
        concordance = np.mean(
            [
                concordance_matrix(np.asarray([scores[key]["score_by_seed"][seed] for key in keys], dtype=float))
                for seed in seeds
            ],
            axis=0,
        )
        sessions.append(
            PairSession(
                session=name,
                person_id=meta[name]["person_id"],
                positions=positions,
                labels=labels,
                concordance=concordance,
            )
        )
    return sessions


def _pair_masks(session: PairSession, max_distance: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(correct-first, incorrect-first) pair masks; rows are positives, columns negatives."""
    delta = session.positions[:, None] - session.positions[None, :]
    keep = np.ones_like(delta) if max_distance is None else (np.abs(delta) <= max_distance).astype(float)
    return keep * (delta < 0), keep * (delta > 0)


def _subject_macro(per_session: Sequence[tuple[str, float]]) -> dict:
    by_subject: dict[str, list[float]] = defaultdict(list)
    for person_id, value in per_session:
        by_subject[person_id].append(value)
    subjects = {person: float(np.mean(rows)) for person, rows in sorted(by_subject.items(), key=lambda kv: int(kv[0]))}
    values = list(subjects.values())
    return {
        "mean": float(np.mean(values)) if values else float("nan"),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "per_subject": subjects,
        "n_subjects": len(subjects),
        "n_sessions": len(per_session),
    }


def pair_subset_auc(
    sessions: Sequence[PairSession],
    concordances: Sequence[np.ndarray],
    direction: str = "all",
    max_distance: int | None = None,
) -> dict:
    """Within-session AUC restricted to one half of each session's pairs.

    ``incorrect_first`` keeps only the (correct, incorrect) pairs where the *incorrect*
    repetition came first -- pairs on which the position rule scores exactly 0.0. A
    model that still orders those correctly is not reading position.
    """
    per_session: list[tuple[str, float]] = []
    pairs = 0.0
    for session, matrix in zip(sessions, concordances):
        correct_first, incorrect_first = _pair_masks(session, max_distance)
        if direction == "all":
            mask = correct_first + incorrect_first
        elif direction == "correct_first":
            mask = correct_first
        elif direction == "incorrect_first":
            mask = incorrect_first
        else:  # pragma: no cover - guarded by the CLI
            raise ValueError(f"Unknown pair direction {direction!r}.")
        denominator = float(session.labels @ mask @ (1 - session.labels))
        if denominator <= 0:
            continue
        numerator = float(session.labels @ (mask * matrix) @ (1 - session.labels))
        per_session.append((session.person_id, numerator / denominator))
        pairs += denominator
    return {**_subject_macro(per_session), "n_pairs": int(round(pairs)), "direction": direction}


def position_balanced_auc(
    sessions: Sequence[PairSession],
    concordances: Sequence[np.ndarray],
    max_distance: int | None = None,
) -> dict:
    """Each session's correct-first and incorrect-first halves weighted 0.5 each.

    The position rule scores exactly 0.5000 on this statistic *by construction* (it
    scores 1 on one half and 0 on the other), so any excess above 0.5 cannot be
    produced by repetition order alone. Only sessions where both halves exist are
    computable, which is why it covers 34 of the 61 sessions and why the plan's primary
    test is an intervention on the features instead.
    """
    per_session: list[tuple[str, float]] = []
    for session, matrix in zip(sessions, concordances):
        correct_first, incorrect_first = _pair_masks(session, max_distance)
        first = float(session.labels @ correct_first @ (1 - session.labels))
        second = float(session.labels @ incorrect_first @ (1 - session.labels))
        if first == 0 or second == 0:
            continue
        value = 0.5 * float(session.labels @ (correct_first * matrix) @ (1 - session.labels)) / first
        value += 0.5 * float(session.labels @ (incorrect_first * matrix) @ (1 - session.labels)) / second
        per_session.append((session.person_id, value))
    return _subject_macro(per_session)


def interleaved_mask(sessions: Sequence[PairSession]) -> list[bool]:
    """True where both pair halves exist, i.e. the labels alternate at least once.

    The complement -- "the first N repetitions are correct and the rest are wrong" --
    is where position and label are collinear and no pair-reweighting can separate
    them. Naming the split is what stops the 34-session number being read as evidence
    about all 61.
    """
    mask = []
    for session in sessions:
        correct_first, incorrect_first = _pair_masks(session)
        mask.append(
            float(session.labels @ correct_first @ (1 - session.labels)) > 0
            and float(session.labels @ incorrect_first @ (1 - session.labels)) > 0
        )
    return mask


def subset_auc(
    sessions: Sequence[PairSession],
    concordances: Sequence[np.ndarray],
    subset: Sequence[bool],
) -> dict:
    """Unrestricted within-session AUC on a named subset of sessions."""
    per_session: list[tuple[str, float]] = []
    for session, matrix, keep in zip(sessions, concordances, subset):
        if not keep:
            continue
        mask = np.ones_like(matrix)
        denominator = float(session.labels @ mask @ (1 - session.labels))
        if denominator <= 0:
            continue
        per_session.append((session.person_id, float(session.labels @ (mask * matrix) @ (1 - session.labels)) / denominator))
    return _subject_macro(per_session)


def exploratory_sessions(
    scores: dict[str, dict],
    seeds: Sequence[int],
    drop_subject: str | None = SENSITIVITY_SUBJECT,
) -> list[Session]:
    """``Session`` records in OOF first-appearance order, repetitions in numeric order.

    ``build_sessions`` sorts sessions by name and repetitions by key string (so
    ``rep10`` precedes ``rep2``). Both orders give the same AUC. Only the permutation
    null, which draws one shuffle per session from one seeded generator, can tell them
    apart -- and reproducing the exploratory null to four decimals requires this one.
    """
    order: dict[str, list[str]] = {}
    meta: dict[str, dict] = {}
    for key, entry in scores.items():
        if drop_subject is not None and entry["person_id"] == drop_subject:
            continue
        order.setdefault(entry["session"], []).append(key)
        meta.setdefault(entry["session"], entry)

    sessions: list[Session] = []
    for name in order:
        keys = sorted(order[name], key=lambda key: int(key.rsplit("rep", 1)[1]))
        labels = np.asarray([scores[key]["label"] for key in keys], dtype=float)
        if len(np.unique(labels)) < 2:
            continue
        ranks = np.stack(
            [midranks(np.asarray([scores[key]["score_by_seed"][seed] for key in keys], dtype=float)) for seed in seeds]
        )
        sessions.append(
            Session(
                session=name,
                person_id=meta[name]["person_id"],
                exercise_id=meta[name]["exercise_id"],
                labels=labels,
                avg_rank=ranks.mean(axis=0),
                repetitions=tuple(keys),
            )
        )
    return sessions


# --------------------------------------------------------------------------- #
# the zero-parameter position rule (ported from the exploration's loso.py)      #
# --------------------------------------------------------------------------- #


def read_segmentation(path: Path) -> list[dict[str, str]]:
    """REHAB24-6 ships ``Segmentation.csv`` semicolon-delimited."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _threshold_balanced_accuracy(labels: Sequence[int], scores: Sequence[float], threshold: float) -> float | None:
    """A small score means "correct": that is the direction the shortcut runs in."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    true_positive = sum(1 for label, score in zip(labels, scores) if score < threshold and label == 1)
    true_negative = sum(1 for label, score in zip(labels, scores) if score >= threshold and label == 0)
    return (true_positive / positives + true_negative / negatives) / 2


def position_rule_loso(
    segmentation_path: Path,
    exclude_subject: str | None = SENSITIVITY_SUBJECT,
) -> dict:
    """One integer, one threshold, leave-one-subject-out.

    Its only job is to show the shortcut is strong enough to outrank the framing
    headline. It is a one-parameter rule against a full pipeline on n = 9 unpaired
    subjects, so it is not a comparison and the plan (§8) says so explicitly.
    """
    rows = [row for row in read_segmentation(segmentation_path) if row["person_id"] != exclude_subject]
    subjects = sorted({row["person_id"] for row in rows}, key=int)

    values: list[float] = []
    per_subject: dict[str, float] = {}
    for subject in subjects:
        train = [row for row in rows if row["person_id"] != subject]
        test = [row for row in rows if row["person_id"] == subject]
        train_labels = [int(row["correctness"]) for row in train]
        train_scores = [float(row["repetition_number"]) for row in train]
        candidates = sorted(set(train_scores))
        threshold = max(candidates, key=lambda value: _threshold_balanced_accuracy(train_labels, train_scores, value))
        score = _threshold_balanced_accuracy(
            [int(row["correctness"]) for row in test],
            [float(row["repetition_number"]) for row in test],
            threshold,
        )
        if score is not None:
            per_subject[subject] = score
            values.append(score)

    return {
        "feature": "repetition_number",
        "mean": float(np.mean(values)),
        "sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "n_subjects": len(values),
        "n_subjects_above_chance": int(sum(value > 0.5 for value in values)),
        "per_subject": per_subject,
    }


# --------------------------------------------------------------------------- #
# feature I/O                                                                  #
# --------------------------------------------------------------------------- #


def feature_index(feature_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(feature_dir.rglob("*.npz")):
        index.setdefault(path.stem, path)
    return index


def load_feature_matrix(feature_dir: Path, sample_ids: Sequence[str]) -> np.ndarray:
    """``(n, d)`` in the order of ``sample_ids``; missing files are fatal, not skipped."""
    index = feature_index(feature_dir)
    missing = [sample_id for sample_id in sample_ids if sample_id not in index]
    if missing:
        raise SystemExit(
            f"{len(missing)} feature files missing from {feature_dir} (first: {missing[:5]}). "
            "Run the extraction (or `predict`) for this arm first."
        )
    rows = []
    for sample_id in sample_ids:
        with np.load(index[sample_id], allow_pickle=False) as data:
            rows.append(np.asarray(data["video_feature"], dtype=np.float64))
    return np.stack(rows)


# --------------------------------------------------------------------------- #
# per-fold transform fitting (plan §3.1, gate §6.2)                            #
# --------------------------------------------------------------------------- #


def select_lambda(
    standardized: np.ndarray,
    targets: np.ndarray,
    subjects: Sequence[str],
    repetitions: Sequence[Repetition],
    grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
) -> tuple[float, dict[str, float | None]]:
    """Inner leave-one-training-subject-out; score = mean within-class Spearman.

    Scored on the same quantity the probe reports, so the chosen lambda is the one that
    best *finds* the direction the intervention then removes -- picking it by mean
    squared error would optimise a scale the projection throws away.

    The standardisation is the fold's, not each inner fit's. It is a hyperparameter
    search over training subjects only, so no test subject is involved either way; using
    one standardiser is what makes the Gram-subtraction shortcut exact.
    Ties resolve to the larger lambda: less variance for the same score.
    """
    by_subject: dict[str, list[int]] = defaultdict(list)
    for index, subject in enumerate(subjects):
        by_subject[subject].append(index)

    total = ridge_gram(standardized, targets)
    grams = {
        subject: ridge_gram(standardized[indices], targets[indices]) for subject, indices in by_subject.items()
    }

    scores: dict[str, float | None] = {}
    best_lambda = float(grid[0])
    best_score = -np.inf
    for lam in sorted(float(value) for value in grid):
        held_out: list[float] = []
        for subject, indices in sorted(by_subject.items(), key=lambda kv: int(kv[0])):
            remaining = total - grams[subject]
            if remaining.n == 0:
                # Fewer than two training subjects: inner LOSO is undefined. Real folds
                # always have eight, but a small fixture must not crash here -- with no
                # scorable inner fold every lambda ties and the tie rule below picks the
                # largest, the most regularised and least variable choice.
                continue
            beta = remaining.solve(lam)
            predictions = standardized[indices] @ beta
            # Both camera rows of a repetition are averaged before scoring, exactly as
            # the reported probe does; a per-sample Spearman would count each rep twice.
            averaged: dict[str, list[float]] = defaultdict(list)
            for index, value in zip(indices, predictions):
                averaged[repetitions[index].repetition].append(float(value))
            scored = {key: float(np.mean(values)) for key, values in averaged.items()}
            cells = probe_cells([repetitions[index] for index in indices], scored, drop_subject=None)
            values = spearman_by_subject(cells)
            if values:
                held_out.append(float(np.mean(list(values.values()))))
        score = float(np.mean(held_out)) if held_out else -np.inf
        # ``None`` rather than -inf: these dicts are written to JSON, and ``-Infinity``
        # is not valid JSON even though Python will happily emit it.
        scores[f"{lam:g}"] = None if not held_out else score
        if score >= best_score:
            best_score = score
            best_lambda = lam
    return best_lambda, scores


def fit_fold_transform(
    feature_dir: Path,
    train_ids: Sequence[str],
    repetitions: dict[str, Repetition],
    manifest: dict[str, dict[str, str]],
    k: int,
    grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
) -> tuple[ResidualTransform, dict]:
    """Fit the fold's residualisation on the training samples and nothing else."""
    ordered = list(train_ids)
    features = load_feature_matrix(feature_dir, ordered)
    mean, std = standardization_stats(features)
    standardized = (features - mean) / std
    targets = np.asarray([repetitions[sample_id].position for sample_id in ordered], dtype=np.float64)
    subjects = [manifest[sample_id]["person_id"] for sample_id in ordered]

    lam, lambda_scores = select_lambda(
        standardized, targets, subjects, [repetitions[sample_id] for sample_id in ordered], grid
    )
    transform = fit_residual_transform(features, targets, k, lam)
    info = {
        "n_train_samples": len(ordered),
        "train_subjects": sorted({subject for subject in subjects}, key=int),
        "lambda": lam,
        "lambda_grid_scores": lambda_scores,
        "k": k,
        "beta_digests": transform.direction_digests(),
        "transform_digest": transform.digest,
    }
    return transform, info


def residual_transform_factory(
    feature_dir: Path,
    repetitions: dict[str, Repetition],
    manifest: dict[str, dict[str, str]],
    k: int,
    grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
    record: list[dict] | None = None,
    cache: dict[str, ResidualTransform] | None = None,
) -> Callable[[list[str]], Callable[[np.ndarray], np.ndarray]]:
    """A ``run_arm`` transform factory: training sample ids in, fitted transform out.

    ``run_arm`` calls it once per fold with that fold's training ids, so the fit is
    structurally incapable of seeing the test subject -- the gate is enforced by the
    call site, not by a convention someone has to remember.

    ``cache`` keys on the training id set so three seeds of the same arm re-use one
    fit; the ridge is deterministic, so the seeds could only differ by float noise.
    """
    store = cache if cache is not None else {}

    def factory(train_ids: list[str]) -> Callable[[np.ndarray], np.ndarray]:
        key = hashlib.sha256("\n".join(sorted(train_ids)).encode("utf-8")).hexdigest()
        if key not in store:
            transform, info = fit_fold_transform(feature_dir, train_ids, repetitions, manifest, k, grid)
            store[key] = transform
            if record is not None:
                record.append({"train_id_digest": key, **info})
        return store[key]

    return factory


def assert_distinct_fold_digests(records: Sequence[dict]) -> None:
    """Gate §6.2: identical ridge directions across folds mean one fit saw everything.

    The comparison is on ``beta_digests`` -- the directions -- and deliberately NOT on
    ``transform_digest``. The transform also carries the fold's standardisation
    statistics, which differ across folds whatever the direction was fitted on, so a
    run whose standardiser was per-fold but whose beta was global would pass a
    transform-level check while being exactly the leak this gate is here to catch.
    """
    digests = [tuple(record["beta_digests"]) for record in records]
    if len(set(digests)) != len(digests):
        raise SystemExit(
            "Fold-purity gate (§6.2) FAILED: two folds produced the identical ridge direction. "
            "Beta was not fitted per fold; the run is invalid."
        )


# --------------------------------------------------------------------------- #
# drift proxies (plan §3.3)                                                    #
# --------------------------------------------------------------------------- #

#: Column indices inside ``box_geometry_features``' 12-dim vector.
BOX_FEATURE_INDEX = {"box_x0": 0, "box_y0": 1, "box_area": 6}


def box_geometry_values(box_dir: Path, sample_ids: Sequence[str]) -> dict[str, dict[str, float]]:
    matrix = load_feature_matrix(box_dir, list(sample_ids))
    return {
        name: {sample_id: float(matrix[row, column]) for row, sample_id in enumerate(sample_ids)}
        for name, column in BOX_FEATURE_INDEX.items()
    }


def cv2_frame_levels(path: Path, stride: int) -> tuple[list[int], list[float], int]:
    """Reference decoder: grey mean of every ``stride``-th frame via OpenCV."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames: list[int] = []
    levels: list[float] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride == 0:
            frames.append(frame_index)
            levels.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
        frame_index += 1
    capture.release()
    return frames, levels, frame_index


FFMPEG_THUMB = (16, 9)


def ffmpeg_frame_levels(path: Path, stride: int) -> tuple[list[int], list[float], int]:
    """Grey mean of every ``stride``-th frame via one ffmpeg subprocess.

    Every frame is decoded (the stride is applied here, not in ffmpeg, so the total
    frame count is exact and matches the cv2 path) and area-averaged to a 16x9 grey
    thumbnail; the mean of that thumbnail equals the full-frame mean up to the 8-bit
    rounding of each cell. ``-fps_mode passthrough`` stops ffmpeg from duplicating or
    dropping frames to hit a nominal rate.
    """
    width, height = FFMPEG_THUMB
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-i", str(path),
        "-vf", f"scale={width}:{height}:flags=area", "-fps_mode", "passthrough",
        "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"ffmpeg failed on {path}: {completed.stderr.decode(errors='replace').strip()}")
    raw = np.frombuffer(completed.stdout, dtype=np.uint8)
    per_frame = width * height
    if raw.size == 0 or raw.size % per_frame != 0:
        raise SystemExit(f"ffmpeg returned {raw.size} bytes for {path}, not a multiple of {per_frame}.")
    means = raw.reshape(-1, per_frame).astype(np.float64).mean(axis=1)
    n_frames = int(means.shape[0])
    frames = list(range(0, n_frames, stride))
    return frames, [float(means[index]) for index in frames], n_frames


def compute_luminance(
    rows: Sequence[dict[str, str]],
    data_root: Path,
    stride: int = LUMINANCE_STRIDE,
    cache_dir: Path | None = None,
    backend: str = DEFAULT_LUMINANCE_BACKEND,
) -> dict[str, tuple[float, int]]:
    """Mean greyscale level over each repetition's frame range, one decode per video.

    Sampling every ``stride``-th frame is a deliberate approximation: lighting drift is
    a slow quantity and a 4x decode saving is the difference between a minute and four.
    The stride is recorded in the CSV's ``.meta.json`` sidecar and in
    ``drift_proxies.json``, and ``run_drift_proxies`` refuses to reuse a cache written
    at a different stride, so it can never be silently changed.

    ``backend`` picks the decoder. ``cv2`` is the reference (BGR -> grey mean over the
    full frame in Python); on this host it runs at ~11 fps on REHAB24-6's 1080p 4:4:4
    streams, i.e. four to five hours for 130 videos, and the first attempt was killed
    before finishing. ``ffmpeg`` decodes the same frames in a subprocess and area-averages
    each one to 16x9 grey before the mean is taken -- the same quantity up to 8-bit
    rounding of the area average -- at ~150 fps, four videos at a time. Both write the
    same cache/CSV schema, and the backend is recorded in the CSV's sidecar.
    """
    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_video[row["video_path"]].append(row)

    if backend not in LUMINANCE_BACKENDS:
        raise SystemExit(f"Unknown luminance backend {backend!r}; choose from {LUMINANCE_BACKENDS}.")
    if backend == "ffmpeg" and shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not on PATH; use --luminance-backend cv2 or install ffmpeg.")

    for video_path in by_video:
        if not (data_root / video_path).exists():
            raise SystemExit(f"Missing video {data_root / video_path}; cannot compute the luminance drift proxy.")

    def decode(video_path: str) -> tuple[list[int], list[float], int]:
        if backend == "ffmpeg":
            return ffmpeg_frame_levels(data_root / video_path, stride)
        return cv2_frame_levels(data_root / video_path, stride)

    def cache_file(video_path: str) -> Path | None:
        return None if cache_dir is None else cache_dir / f"{Path(video_path).stem}.stride{stride}.npz"

    result: dict[str, tuple[float, int]] = {}
    ordered = sorted(by_video.items())
    decoded: dict[str, tuple[list[int], list[float], int]] = {}
    pending = [video_path for video_path, _ in ordered if cache_file(video_path) is None or not cache_file(video_path).exists()]
    if backend == "ffmpeg" and pending:
        # ffmpeg is a separate process per video, so the GIL is not in the way and the
        # 12-core host can decode several 4:4:4 streams at once.
        with ThreadPoolExecutor(max_workers=FFMPEG_WORKERS) as pool:
            for video_path, value in zip(pending, pool.map(decode, pending)):
                decoded[video_path] = value
    for index, (video_path, members) in enumerate(ordered, start=1):
        path = data_root / video_path
        # Per-video cache: a full decode of the 128 videos takes hours on this CPU and the
        # first attempt was killed by the host at ~1h45 with nothing persisted. Each
        # video's per-frame levels are written as soon as it finishes so a re-run resumes
        # instead of restarting; the stride is part of the file name so a cache written
        # at one stride can never be read back under another.
        cache_path = cache_file(video_path)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                frames = [int(value) for value in cached["frames"]]
                levels = [float(value) for value in cached["levels"]]
                frame_index = int(cached["n_frames"])
            print(f"  [{index}/{len(by_video)}] {video_path}: cached ({frame_index} frames, {len(levels)} sampled)", flush=True)
        else:
            frames, levels, frame_index = decoded[video_path] if video_path in decoded else decode(video_path)
            if cache_path is not None:
                np.savez(
                    cache_path,
                    frames=np.asarray(frames, dtype=np.int64),
                    levels=np.asarray(levels, dtype=np.float64),
                    n_frames=np.asarray(frame_index, dtype=np.int64),
                )
            print(f"  [{index}/{len(by_video)}] {video_path}: {frame_index} frames, {len(levels)} sampled", flush=True)

        frame_array = np.asarray(frames, dtype=np.int64)
        level_array = np.asarray(levels, dtype=np.float64)
        for row in members:
            first = int(row["first_frame"])
            last = int(row["last_frame"])
            selected = (frame_array >= first) & (frame_array <= last)
            count = int(selected.sum())
            if count == 0:
                raise SystemExit(f"No sampled frame falls inside {row['sample_id']}'s range [{first}, {last}].")
            result[row["sample_id"]] = (float(level_array[selected].mean()), count)
    return result


def write_luminance(path: Path, values: dict[str, tuple[float, int]], stride: int, backend: str = DEFAULT_LUMINANCE_BACKEND) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "luminance", "n_frames_used"])
        for sample_id in sorted(values):
            level, count = values[sample_id]
            writer.writerow([sample_id, repr(level), count])
    with path.with_suffix(".meta.json").open("w", encoding="utf-8") as handle:
        json.dump({"frame_stride": stride, "n_samples": len(values), "backend": backend}, handle, indent=2, sort_keys=True)


def read_luminance(path: Path) -> dict[str, tuple[float, int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["sample_id"]: (float(row["luminance"]), int(row["n_frames_used"]))
            for row in csv.DictReader(handle)
        }


def proxy_sessions(
    repetitions: Sequence[Repetition],
    values: dict[str, float],
    drop_subject: str | None = SENSITIVITY_SUBJECT,
) -> list[Session]:
    """Mixed-label sessions scored by a zero-parameter proxy instead of the model.

    Cameras are averaged into one repetition value first, exactly as the model
    probabilities are, so the proxy and the model go through the identical statistic.
    """
    grouped: dict[str, list[Repetition]] = defaultdict(list)
    for rep in repetitions:
        if drop_subject is not None and rep.person_id == drop_subject:
            continue
        grouped[rep.session].append(rep)

    sessions: list[Session] = []
    for name in sorted(grouped):
        members = sorted(grouped[name], key=lambda rep: rep.number)
        labels = np.asarray([rep.label for rep in members], dtype=float)
        if len(np.unique(labels)) < 2:
            continue
        scores = np.asarray(
            [float(np.mean([values[sample_id] for sample_id in rep.sample_ids])) for rep in members], dtype=float
        )
        sessions.append(
            Session(
                session=name,
                person_id=members[0].person_id,
                exercise_id=members[0].exercise_id,
                labels=labels,
                avg_rank=midranks(scores),
                repetitions=tuple(rep.repetition for rep in members),
            )
        )
    return sessions


def two_sided_p(null: np.ndarray, observed: float, centre: float = 0.5) -> float:
    """Plan §3.3: a proxy that predicts the label BACKWARDS is just as much a shortcut.

    The identity control learned this the expensive way -- ``position`` scores 0.1761
    one-sided and would have been reported as "below chance, ignore it"."""
    extreme = int(np.sum(np.abs(null - centre) >= abs(observed - centre)))
    return float((1 + extreme) / (len(null) + 1))


def drift_proxy_report(
    repetitions: Sequence[Repetition],
    proxies: dict[str, dict[str, float]],
    n_permutations: int,
    permutation_seed: int,
    informative_margin: float = 0.15,
) -> dict:
    """Every proxy through the §3.1 statistic, two-sided, Holm-corrected across proxies."""
    controls: dict[str, dict] = {}
    raw_p: dict[str, float] = {}
    for name in sorted(proxies):
        sessions = proxy_sessions(repetitions, proxies[name])
        statistic = observed_statistic(sessions)
        null = permutation_null(sessions, n_permutations, permutation_seed)
        p_value = two_sided_p(null, statistic["mean"])
        raw_p[name] = p_value
        controls[name] = {
            "subject_macro": statistic,
            "distance_from_chance": abs(statistic["mean"] - 0.5),
            "permutation": {
                "n_permutations": n_permutations,
                "permutation_seed": permutation_seed,
                "sidedness": "two-sided",
                "p_value": p_value,
                "null_mean": float(null.mean()),
                "null_sd": float(null.std(ddof=1)),
            },
            "informative": bool(abs(statistic["mean"] - 0.5) > informative_margin and p_value < ALPHA),
        }
    return {
        "informative_margin": informative_margin,
        "controls": controls,
        "raw_p_values": raw_p,
        "holm": holm_correct(raw_p) if raw_p else {},
    }


# --------------------------------------------------------------------------- #
# commands                                                                     #
# --------------------------------------------------------------------------- #


def run_replicate_exploratory(
    output_dir: Path,
    identity_dir: Path,
    segmentation_path: Path,
    n_permutations: int,
    permutation_seed: int,
) -> dict:
    """Gate §6.4: re-derive the five motivating numbers from the EXISTING OOF.

    Nothing here trains anything or reads a new probability. If any number moves in the
    fourth decimal, the port is wrong and no new arm may be interpreted.
    """
    oof_rows: list[dict] = []
    for seed in SEEDS:  # module order, not the caller's: the null is order-sensitive.
        path = identity_dir / f"oof_seed{seed}.csv"
        if not path.exists():
            raise SystemExit(f"Missing {path}. The identity-control OOF is the input to this gate.")
        oof_rows.extend(read_oof(path))

    scores = repetition_scores(oof_rows, SEEDS)
    sessions = exploratory_sessions(scores, SEEDS)
    if len(sessions) != EXPECTED_MIXED_SESSIONS:
        raise SystemExit(f"Grouping gate: {len(sessions)} mixed-label sessions, expected {EXPECTED_MIXED_SESSIONS}.")

    statistic = observed_statistic(sessions)
    null = permutation_null(sessions, n_permutations, permutation_seed)

    pair_sessions = build_pair_sessions(scores, SEEDS, sort_sessions=False)
    model = [session.concordance for session in pair_sessions]
    rule = [position_rule_concordance(session.positions) for session in pair_sessions]
    interleaved = interleaved_mask(pair_sessions)

    report = {
        "source": str(identity_dir),
        "seeds": list(SEEDS),
        "session_order": "OOF first-appearance (the exploratory scripts' order)",
        "unrestricted_all_sessions": {
            "model": statistic["mean"],
            "position_rule": subset_auc(pair_sessions, rule, [True] * len(pair_sessions))["mean"],
            "n_sessions": statistic["n_sessions"],
            "n_subjects": statistic["n_subjects"],
            "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)),
            "p_value": float((1 + int(np.sum(null >= statistic["mean"]))) / (n_permutations + 1)),
            "n_permutations": n_permutations,
            "permutation_seed": permutation_seed,
        },
        "position_balanced": {
            **position_balanced_auc(pair_sessions, model),
            "position_rule": position_balanced_auc(pair_sessions, rule)["mean"],
        },
        "incorrect_first_pairs": {
            **pair_subset_auc(pair_sessions, model, "incorrect_first"),
            "position_rule": pair_subset_auc(pair_sessions, rule, "incorrect_first")["mean"],
        },
        "unrestricted_interleaved_sessions": subset_auc(pair_sessions, model, interleaved),
        "unrestricted_single_boundary_sessions": subset_auc(
            pair_sessions, model, [not value for value in interleaved]
        ),
        "session_split": {
            "interleaved": int(sum(interleaved)),
            "single_boundary": int(len(interleaved) - sum(interleaved)),
        },
        "position_rule_loso": position_rule_loso(segmentation_path),
    }

    # The reference null was drawn with 10,000 permutations at seed 20260906. A smoke
    # run with fewer draws is a legitimate use of this command, so its null is reported
    # but not asserted -- an assertion that can only be met by one argument value would
    # be a trap, not a gate.
    reference_null = n_permutations == DEFAULT_PERMUTATIONS and permutation_seed == DEFAULT_PERMUTATION_SEED
    checks = {
        "unrestricted_all_61_is_0.8741": round(report["unrestricted_all_sessions"]["model"], 4) == 0.8741,
        "null_mean_is_0.5002": (
            round(report["unrestricted_all_sessions"]["null_mean"], 4) == 0.5002 if reference_null else True
        ),
        "null_sd_is_0.0211": (
            round(report["unrestricted_all_sessions"]["null_sd"], 4) == 0.0211 if reference_null else True
        ),
        "position_balanced_is_0.8497": round(report["position_balanced"]["mean"], 4) == 0.8497,
        "incorrect_first_is_0.8223": round(report["incorrect_first_pairs"]["mean"], 4) == 0.8223,
        "interleaved_34_unrestricted_is_0.8487": round(report["unrestricted_interleaved_sessions"]["mean"], 4) == 0.8487,
        "position_rule_loso_is_0.7309": round(report["position_rule_loso"]["mean"], 4) == 0.7309,
        "interleaved_sessions_is_34": report["session_split"]["interleaved"] == EXPECTED_INTERLEAVED_SESSIONS,
        "position_rule_is_exactly_0.5_when_balanced": report["position_balanced"]["position_rule"] == 0.5,
    }
    report["checks"] = checks
    report["null_checked_against_reference"] = reference_null
    report["passed"] = all(checks.values())

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "replicate_exploratory.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report


def print_replicate(report: dict) -> None:
    unrestricted = report["unrestricted_all_sessions"]
    print("\n=== §6.4 reproduction gate: the exploratory numbers, from the existing OOF ===")
    print(
        f"  unrestricted, all {unrestricted['n_sessions']} sessions      model {unrestricted['model']:.4f}   "
        f"position rule {unrestricted['position_rule']:.4f}"
    )
    print(
        f"    within-session permutation null   {unrestricted['null_mean']:.4f} +/- {unrestricted['null_sd']:.4f}   "
        f"p = {unrestricted['p_value']:.5f}  ({unrestricted['n_permutations']} perms, seed {unrestricted['permutation_seed']})"
    )
    balanced = report["position_balanced"]
    print(
        f"  position-balanced, {balanced['n_sessions']} sessions        model {balanced['mean']:.4f}   "
        f"position rule {balanced['position_rule']:.4f}  (0.5 by construction)"
    )
    incorrect = report["incorrect_first_pairs"]
    print(
        f"  incorrect-first pairs only            model {incorrect['mean']:.4f}   "
        f"position rule {incorrect['position_rule']:.4f}   ({incorrect['n_pairs']} pairs, {incorrect['n_sessions']} sessions)"
    )
    interleaved = report["unrestricted_interleaved_sessions"]
    print(f"  unrestricted, the same {interleaved['n_sessions']} sessions   model {interleaved['mean']:.4f}")
    boundary = report["unrestricted_single_boundary_sessions"]
    print(
        f"  unrestricted, {boundary['n_sessions']} single-boundary  model {boundary['mean']:.4f}   "
        f"(n_subjects = {boundary['n_subjects']}, NOT comparable to the line above)"
    )
    rule = report["position_rule_loso"]
    print(
        f"  position rule, LOSO subject-macro BA  {rule['mean']:.4f} +/- {rule['sd']:.4f}   "
        f"({rule['n_subjects_above_chance']}/{rule['n_subjects']} > 0.5)"
    )
    for name, ok in report["checks"].items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")


def run_predict(
    arm_dir: Path,
    manifest_path: Path,
    labels_path: Path,
    output_dir: Path,
    ks: Sequence[int],
    naive: bool,
    seeds: Sequence[int],
    device_arg: str | None,
    grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
) -> dict:
    """Run the frozen LOSO recipe on residualised features, one arm per ``k``.

    ``k = 0`` deliberately runs with no transform at all, so the reproduction gate
    (§6.4) exercises the untouched ``run_arm`` path rather than an identity transform
    that merely ought to behave like one.
    """
    import torch

    device = torch.device(
        "cuda" if (device_arg != "cpu" and device_arg is not None and torch.cuda.is_available()) else "cpu"
    )
    config = FoldConfig()

    labels = {key: int(value) for key, value in json.load(labels_path.open()).items()}
    metadata = load_metadata(manifest_path)
    manifest = manifest_index(manifest_path)
    rows = load_manifest(manifest_path)
    repetitions = repetition_by_sample(build_repetitions(rows, naive=naive))
    subject_samples = subjects_to_samples(manifest_path)
    sample_counts = {person: len(ids) for person, ids in subject_samples.items()}
    ordered_subjects = sorted(subject_samples, key=int)

    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"arm_dir": str(arm_dir), "seeds": list(seeds), "naive": naive, "arms": {}}
    all_records: dict[str, list[dict]] = {}
    for k in ks:
        arm = arm_name(k, naive)
        records: list[dict] = []
        cache: dict[str, ResidualTransform] = {}
        factory = (
            None
            if k == 0
            else residual_transform_factory(arm_dir, repetitions, manifest, k, grid, record=records, cache=cache)
        )
        print(f"\n=== arm {arm} | device {device} | seeds {list(seeds)} ===")
        report["arms"][arm] = {"per_seed": {}}
        for seed in seeds:
            folds = run_arm(
                arm_dir,
                labels,
                subject_samples,
                ordered_subjects,
                sample_counts,
                metadata,
                config,
                device,
                seed,
                retain_predictions=True,
                feature_transform_factory=factory,
                materialize_root=features_root(output_dir, arm) if factory is not None else None,
            )
            written = write_oof(oof_path(output_dir, arm, seed), folds, manifest, seed)
            with folds_path(output_dir, arm, seed).open("w", encoding="utf-8") as handle:
                json.dump({"seed": seed, "arm": arm, "config": vars(config), "folds": folds}, handle, indent=2, sort_keys=True)
            big = [fold for fold in folds if fold["n_test"] >= MIN_VAL_SUBJECT_SAMPLES]
            mean_ba = float(np.mean([fold["balanced_accuracy"] for fold in big]))
            report["arms"][arm]["per_seed"][str(seed)] = {"rows": written, "balanced_accuracy_no_p10": mean_ba}
            print(f"  seed {seed}: {written} OOF rows | bal_acc (9 folds, no P10) {mean_ba:.4f}")
        if records:
            assert_distinct_fold_digests(records)
            all_records[arm] = records
            report["arms"][arm]["lambdas"] = {record["train_id_digest"][:8]: record["lambda"] for record in records}

    if all_records:
        path = output_dir / "fold_betas.json"
        existing = json.load(path.open(encoding="utf-8")) if path.exists() else {}
        existing.update(all_records)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(existing, handle, indent=2, sort_keys=True)
        print(f"\nSaved per-fold ridge directions to {path}")

    with (output_dir / "predict_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report


def loso_balanced_accuracy(output_dir: Path, arm: str, seeds: Sequence[int]) -> dict | None:
    """Subject-macro balanced accuracy from the saved fold records, P10 excluded."""
    by_subject: dict[str, list[float]] = defaultdict(list)
    per_seed: dict[str, float] = {}
    for seed in seeds:
        path = folds_path(output_dir, arm, seed)
        if not path.exists():
            return None
        folds = json.load(path.open(encoding="utf-8"))["folds"]
        big = [fold for fold in folds if fold["n_test"] >= MIN_VAL_SUBJECT_SAMPLES]
        per_seed[str(seed)] = float(np.mean([fold["balanced_accuracy"] for fold in big]))
        for fold in big:
            by_subject[fold["test_subject"]].append(fold["balanced_accuracy"])

    averaged = {person: float(np.mean(values)) for person, values in sorted(by_subject.items(), key=lambda kv: int(kv[0]))}
    values = list(averaged.values())
    return {
        "per_seed": per_seed,
        "mean": float(np.mean(list(per_seed.values()))),
        "sd_over_subjects": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "seed_averaged_by_subject": averaged,
        "n_subjects": len(values),
    }


def paired_delta_vs_framing(arm_balanced_accuracy: dict, framing_summary_path: Path) -> dict:
    """Nine paired subject deltas against ``full_frame_letterbox``'s 0.6612.

    Refuses to invent a baseline: if the per-subject numbers are not in the framing
    summary the delta is reported as unavailable rather than approximated from the mean.
    """
    if not framing_summary_path.exists():
        return {"available": False, "reason": f"missing {framing_summary_path}"}
    summary = json.load(framing_summary_path.open(encoding="utf-8"))
    baseline = summary.get("arms", {}).get("full_frame_letterbox", {}).get("seed_averaged_by_subject")
    if not baseline:
        return {"available": False, "reason": "framing summary has no arms.full_frame_letterbox.seed_averaged_by_subject"}

    candidate = arm_balanced_accuracy["seed_averaged_by_subject"]
    shared = sorted(set(baseline) & set(candidate), key=int)
    deltas = [candidate[person] - float(baseline[person]) for person in shared]
    greater = exact_wilcoxon_vs(deltas, 0.0)
    less = exact_wilcoxon_vs([-delta for delta in deltas], 0.0)
    return {
        "available": True,
        "baseline_mean": float(np.mean([float(baseline[person]) for person in shared])),
        "candidate_mean": float(np.mean([candidate[person] for person in shared])),
        "per_subject_delta": {person: delta for person, delta in zip(shared, deltas)},
        "mean_delta": float(np.mean(deltas)),
        "n_subjects": len(shared),
        "n_positive": int(sum(delta > 0 for delta in deltas)),
        "wilcoxon_greater": greater,
        "wilcoxon_less": less,
        "two_sided_p": (
            min(1.0, 2 * min(greater["p_value"], less["p_value"])) if greater and less else None
        ),
        "reference_mean": FRAMING_BASELINE_BA,
    }


def preregistered_verdict(statistic: dict, p_value: float) -> dict:
    """Plan §7, evaluated as written: three conditions and two thresholds."""
    conditions = {
        "mean_auc_at_least_0.80": statistic["mean"] >= PRIMARY_AUC_FLOOR,
        "mean_auc_at_least_0.55": statistic["mean"] >= DECISION_AUC,
        "at_least_6_of_9_subjects_above_0.5": statistic["n_subjects_above_chance"] >= MIN_SUBJECTS_ABOVE_CHANCE,
        "permutation_p_below_0.05": p_value < ALPHA,
    }
    supported = conditions["at_least_6_of_9_subjects_above_0.5"] and conditions["permutation_p_below_0.05"]
    if supported and conditions["mean_auc_at_least_0.80"]:
        row = "row 1: linear drift directions are NOT the source of the within-session signal"
    elif supported and conditions["mean_auc_at_least_0.55"]:
        row = "row 2: part of the signal runs along the drift direction; report the share, do not call it movement"
    else:
        row = "row 3: undetermined -- this design does not establish within-session correctness ordering"
    return {
        "conditions": conditions,
        "row": row,
        "share_along_drift": (
            (REFERENCE_WITHIN_SESSION_AUC - statistic["mean"]) / (REFERENCE_WITHIN_SESSION_AUC - 0.5)
        ),
        "note": "row 1 additionally requires the §6.3 probe gate; this function does not evaluate that",
    }


def analyze(
    oof_rows: Sequence[dict],
    seeds: Sequence[int],
    n_permutations: int,
    permutation_seed: int,
    n_bootstrap: int,
    include_p10: bool = False,
) -> tuple[dict, np.ndarray]:
    """The §3.1 primary statistic plus the §3.4 pair-level secondaries."""
    drop_subject = None if include_p10 else SENSITIVITY_SUBJECT
    scores = repetition_scores(oof_rows, seeds)
    sessions, single_class = build_sessions(scores, seeds, drop_subject=drop_subject)
    if not include_p10 and len(sessions) != EXPECTED_MIXED_SESSIONS:
        raise SystemExit(
            f"Grouping gate: {len(sessions)} mixed-label sessions, expected {EXPECTED_MIXED_SESSIONS}."
        )

    estimator_error = verify_estimator(sessions, scores, seeds)
    if estimator_error > 1e-12:
        raise SystemExit(f"Rank-based AUC disagrees with the definitional AUC by {estimator_error:.3e}.")

    statistic = observed_statistic(sessions)
    null = permutation_null(sessions, n_permutations, permutation_seed)
    p_value = float((1 + int(np.sum(null >= statistic["mean"]))) / (n_permutations + 1))

    primary = {
        **statistic,
        "single_class_sessions_excluded": single_class,
        "estimator_max_abs_error_vs_definition": estimator_error,
        "permutation": {
            "n_permutations": n_permutations,
            "permutation_seed": permutation_seed,
            "sidedness": "one-sided (greater)",
            "p_value": p_value,
            "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)),
            "null_p95": float(np.percentile(null, 95)),
        },
        "bootstrap_95ci_over_subjects": bootstrap_interval(
            list(statistic["per_subject_auc"].values()), n_bootstrap, permutation_seed
        ),
        "verdict": preregistered_verdict(statistic, p_value),
    }

    pair_sessions = build_pair_sessions(scores, seeds, drop_subject=drop_subject)
    model = [session.concordance for session in pair_sessions]
    rule = [position_rule_concordance(session.positions) for session in pair_sessions]
    interleaved = interleaved_mask(pair_sessions)
    secondary = {
        "position_balanced": {
            **position_balanced_auc(pair_sessions, model),
            "position_rule": position_balanced_auc(pair_sessions, rule)["mean"],
        },
        "unrestricted_interleaved_sessions": {
            **subset_auc(pair_sessions, model, interleaved),
            "position_rule": subset_auc(pair_sessions, rule, interleaved)["mean"],
        },
        "unrestricted_single_boundary_sessions": {
            **subset_auc(pair_sessions, model, [not value for value in interleaved]),
            "position_rule": subset_auc(pair_sessions, rule, [not value for value in interleaved])["mean"],
            "note": "a different set of recordings from the interleaved one; not a contrast",
        },
        "incorrect_first_pairs": {
            **pair_subset_auc(pair_sessions, model, "incorrect_first"),
            "position_rule": pair_subset_auc(pair_sessions, rule, "incorrect_first")["mean"],
        },
        "session_split": {
            "interleaved": int(sum(interleaved)),
            "single_boundary": int(len(interleaved) - sum(interleaved)),
        },
    }

    return {
        "seeds": list(seeds),
        "include_p10": include_p10,
        "n_repetitions": len(scores),
        "primary": primary,
        "secondary": secondary,
    }, null


def print_analysis(summary: dict) -> None:
    primary = summary["primary"]
    print("\n=== primary: within-session correctness ordering after residualisation ===")
    for subject, value in primary["per_subject_auc"].items():
        print(f"  P{subject:<3} AUC {value:.4f}")
    print(
        f"  mean {primary['mean']:.4f} +/- {primary['sd']:.4f}  "
        f"range [{primary['range'][0]:.4f}, {primary['range'][1]:.4f}]  "
        f"{primary['n_subjects_above_chance']}/{primary['n_subjects']} subjects > 0.5"
    )
    ci = primary["bootstrap_95ci_over_subjects"]
    print(f"  bootstrap 95% CI over subjects: [{ci[0]:.4f}, {ci[1]:.4f}]  (description, not the test)")
    perm = primary["permutation"]
    print(
        f"  permutation null {perm['null_mean']:.4f} +/- {perm['null_sd']:.4f}  ->  one-sided p = {perm['p_value']:.5f} "
        f"({perm['n_permutations']} perms, seed {perm['permutation_seed']})"
    )
    verdict = primary["verdict"]
    for name, value in verdict["conditions"].items():
        print(f"    [{'PASS' if value else 'FAIL'}] {name}")
    print(f"  pre-registered reading: {verdict['row']}")
    print(f"  share along the drift direction: {verdict['share_along_drift']:.3f}")

    print("\n=== secondary (never a substitute for the primary) ===")
    secondary = summary["secondary"]
    for name in (
        "position_balanced",
        "unrestricted_interleaved_sessions",
        "unrestricted_single_boundary_sessions",
        "incorrect_first_pairs",
    ):
        block = secondary[name]
        print(
            f"  {name:<40} model {block['mean']:.4f}  position rule {block['position_rule']:.4f}  "
            f"({block['n_sessions']} sessions, {block['n_subjects']} subjects)"
        )
    split = secondary["session_split"]
    print(f"  session split: {split['interleaved']} interleaved, {split['single_boundary']} single boundary")
    if "loso_balanced_accuracy" in secondary and secondary["loso_balanced_accuracy"]:
        ba = secondary["loso_balanced_accuracy"]
        print(f"  LOSO subject-macro balanced accuracy: {ba['mean']:.4f} +/- {ba['sd_over_subjects']:.4f}")
    delta = secondary.get("paired_delta_vs_framing")
    if delta and delta.get("available"):
        print(
            f"  paired delta vs framing {delta['baseline_mean']:.4f}: {delta['mean_delta']:+.4f} "
            f"({delta['n_positive']}/{delta['n_subjects']} positive, two-sided p = {delta['two_sided_p']})"
        )
    elif delta:
        print(f"  paired delta vs framing: unavailable ({delta['reason']})")


def run_probe(
    arm_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    k: int,
    naive: bool,
    n_permutations: int,
    permutation_seed: int,
    include_p10: bool,
    grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
) -> dict:
    """Plan §3.2 / gate §6.3: does the feature space linearly encode within-class position?

    Reads features only -- never a probability -- so ``probe --k 0`` runs before any
    classifier has been trained. For ``k >= 1`` it reads the fold directories
    ``predict`` materialised, which is the only way the probe can see exactly the
    features the classifier saw.
    """
    arm = arm_name(k, naive)
    rows = load_manifest(manifest_path)
    manifest = manifest_index(manifest_path)
    repetitions = build_repetitions(rows, naive=naive)
    rep_by_sample = repetition_by_sample(repetitions)
    subject_samples = subjects_to_samples(manifest_path)
    sample_counts = {person: len(ids) for person, ids in subject_samples.items()}
    ordered_subjects = sorted(subject_samples, key=int)

    per_fold: list[dict] = []
    raw_predictions: dict[str, list[float]] = defaultdict(list)
    for test_subject in ordered_subjects:
        val_subject = pick_val_subject(test_subject, ordered_subjects, sample_counts)
        train_ids = [
            sample_id
            for subject in ordered_subjects
            if subject not in {test_subject, val_subject}
            for sample_id in subject_samples[subject]
        ]
        test_ids = list(subject_samples[test_subject])
        fold_dir = arm_dir if k == 0 else fold_feature_dir(output_dir, arm, test_subject)
        if not fold_dir.exists():
            raise SystemExit(
                f"Missing feature directory {fold_dir}. Run "
                f"`predict --k {k}{' --naive' if naive else ''}` before probing this arm."
            )

        train_features = load_feature_matrix(fold_dir, train_ids)
        mean, std = standardization_stats(train_features)
        standardized = (train_features - mean) / std
        targets = np.asarray([rep_by_sample[sample_id].position for sample_id in train_ids], dtype=np.float64)
        subjects = [manifest[sample_id]["person_id"] for sample_id in train_ids]
        lam, lambda_scores = select_lambda(
            standardized, targets, subjects, [rep_by_sample[sample_id] for sample_id in train_ids], grid
        )
        beta = ridge_gram(standardized, targets).solve(lam)
        test_standardized = (load_feature_matrix(fold_dir, test_ids) - mean) / std
        for sample_id, value in zip(test_ids, test_standardized @ beta):
            raw_predictions[rep_by_sample[sample_id].repetition].append(float(value))
        per_fold.append(
            {
                "test_subject": test_subject,
                "val_subject": val_subject,
                "n_train": len(train_ids),
                "n_test": len(test_ids),
                "lambda": lam,
                "lambda_grid_scores": lambda_scores,
                "feature_dir": str(fold_dir),
            }
        )
        print(f"  fold P{test_subject}: lambda {lam:g}, {len(train_ids)} train / {len(test_ids)} test samples")

    predictions = {key: float(np.mean(values)) for key, values in raw_predictions.items()}
    cells = probe_cells(repetitions, predictions, drop_subject=None if include_p10 else SENSITIVITY_SUBJECT)
    if not cells:
        raise SystemExit("No (session, class) cell had enough repetitions to correlate.")

    observed_mean, observed_median = macro_from_cells(cells, cell_values(cells))
    null_mean, null_median = spearman_permutation_null(cells, n_permutations, permutation_seed)
    by_subject = spearman_by_subject(cells)

    report = {
        "arm": arm,
        "k": k,
        "naive": naive,
        "include_p10": include_p10,
        "n_cells": len(cells),
        "n_sessions": len({cell.session for cell in cells}),
        "per_subject": by_subject,
        "subject_macro_mean": float(observed_mean[0]),
        "subject_macro_median": float(observed_median[0]),
        "null": {
            "n_permutations": n_permutations,
            "permutation_seed": permutation_seed,
            "sidedness": "two-sided",
            "mean_of_null_means": float(null_mean.mean()),
            "median_null_95_interval": [float(np.percentile(null_median, 2.5)), float(np.percentile(null_median, 97.5))],
            "mean_null_95_interval": [float(np.percentile(null_mean, 2.5)), float(np.percentile(null_mean, 97.5))],
            # Centre 0, not the null's own mean: the pre-registered formula (plan §3.2)
            # is p = (1 + #|null| >= |observed|) / (N + 1), and a correlation's null
            # hypothesis is zero correlation.
            "p_value_mean": two_sided_p(null_mean, float(observed_mean[0]), centre=0.0),
            "p_value_median": two_sided_p(null_median, float(observed_median[0]), centre=0.0),
        },
        "folds": per_fold,
        "lambda_grid": list(grid),
    }
    low, high = report["null"]["median_null_95_interval"]
    report["median_inside_null_95_interval"] = bool(low <= report["subject_macro_median"] <= high)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"probe_summary_{arm}.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report


def print_probe(report: dict) -> None:
    print(f"\n=== §3.2 probe: features -> within-class position (arm {report['arm']}) ===")
    for subject, value in report["per_subject"].items():
        print(f"  P{subject:<3} within-class Spearman {value:+.4f}")
    null = report["null"]
    print(
        f"  subject-macro mean {report['subject_macro_mean']:+.4f}   median {report['subject_macro_median']:+.4f}   "
        f"({report['n_cells']} cells over {report['n_sessions']} sessions)"
    )
    print(
        f"  null median 95% interval [{null['median_null_95_interval'][0]:+.4f}, {null['median_null_95_interval'][1]:+.4f}]"
        f"   two-sided p (median) = {null['p_value_median']:.5f}   "
        f"({null['n_permutations']} perms, seed {null['permutation_seed']})"
    )
    inside = report["median_inside_null_95_interval"]
    print(f"  observed median inside the null 95% interval: {inside}")


def run_drift_proxies(
    manifest_path: Path,
    output_dir: Path,
    box_dir: Path,
    data_root: Path,
    n_permutations: int,
    permutation_seed: int,
    stride: int = LUMINANCE_STRIDE,
    luminance_backend: str = DEFAULT_LUMINANCE_BACKEND,
) -> dict:
    rows = load_manifest(manifest_path)
    repetitions = build_repetitions(rows)
    sample_ids = [row["sample_id"] for row in rows]

    luminance_path = output_dir / "luminance_per_sample.csv"
    if luminance_path.exists():
        meta_path = luminance_path.with_suffix(".meta.json")
        cached_stride = json.load(meta_path.open(encoding="utf-8"))["frame_stride"] if meta_path.exists() else None
        if cached_stride != stride:
            raise SystemExit(
                f"{luminance_path} was written at frame stride {cached_stride}, but --frame-stride is {stride}. "
                "Re-run with the cached stride, or delete the CSV and its .meta.json to re-decode. "
                "Reporting one stride's numbers under another's label is how a silent method change happens."
            )
        print(f"Reusing {luminance_path} (delete it to force a re-decode).")
        luminance = read_luminance(luminance_path)
        missing = [sample_id for sample_id in sample_ids if sample_id not in luminance]
        if missing:
            raise SystemExit(f"{luminance_path} is missing {len(missing)} samples (first: {missing[:5]}).")
    else:
        print(f"Decoding {len({row['video_path'] for row in rows})} videos at stride {stride} ...")
        luminance = compute_luminance(rows, data_root, stride, cache_dir=output_dir / "luminance_cache", backend=luminance_backend)
        write_luminance(luminance_path, luminance, stride, backend=luminance_backend)

    proxies = box_geometry_values(box_dir, sample_ids)
    proxies["luminance"] = {sample_id: value for sample_id, (value, _) in luminance.items()}

    report = drift_proxy_report(repetitions, proxies, n_permutations, permutation_seed)
    report["luminance_frame_stride"] = stride
    meta_path = luminance_path.with_suffix(".meta.json")
    report["luminance_backend"] = json.load(meta_path.open(encoding="utf-8")).get("backend", "cv2") if meta_path.exists() else luminance_backend
    report["box_feature_index"] = BOX_FEATURE_INDEX

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "drift_proxies.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report


def print_drift_proxies(report: dict) -> None:
    print("\n=== §3.3 drift proxies (zero-parameter, two-sided) ===")
    for name, values in report["controls"].items():
        statistic = values["subject_macro"]
        permutation = values["permutation"]
        holm = report["holm"].get(name, {})
        print(
            f"  {name:<12} within-session AUC {statistic['mean']:.4f}  |AUC-0.5| {values['distance_from_chance']:.4f}  "
            f"({statistic['n_subjects_above_chance']}/{statistic['n_subjects']} > 0.5)"
        )
        print(
            f"      two-sided p {permutation['p_value']:.5f}  Holm {holm.get('holm', float('nan')):.5f}  "
            f"-> informative: {values['informative']}"
        )


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))


def add_arm(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--k", type=int, required=True, help="Number of ridge directions removed.")
    parser.add_argument("--naive", action="store_true", help="Rank by RAW position, not within-class (expected to over-remove).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="REHAB24-6 VideoMAE within-class position control.")
    sub = parser.add_subparsers(dest="command", required=True)

    replicate = sub.add_parser("replicate-exploratory", help="Gate §6.4: re-derive the exploratory numbers.")
    add_common(replicate)
    replicate.add_argument("--identity-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "videomae_identity_control")
    replicate.add_argument("--segmentation", type=Path, default=DEFAULT_SEGMENTATION)
    replicate.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    replicate.add_argument("--permutation-seed", type=int, default=DEFAULT_PERMUTATION_SEED)

    predict = sub.add_parser("predict", help="LOSO on residualised features; writes OOF probabilities.")
    add_common(predict)
    predict.add_argument("--k", type=int, nargs="+", required=True)
    predict.add_argument("--naive", action="store_true")
    predict.add_argument("--arm-dir", type=Path, default=DEFAULT_ARM_DIR)
    predict.add_argument("--labels", type=Path, default=DEFAULT_PROCESSED_ROOT / "labels" / "correctness.json")
    predict.add_argument("--device", type=str, default=None)

    analyse = sub.add_parser("analyze", help="Primary within-session statistic and permutation test.")
    add_common(analyse)
    add_arm(analyse)
    analyse.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    analyse.add_argument("--permutation-seed", type=int, default=DEFAULT_PERMUTATION_SEED)
    analyse.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    analyse.add_argument("--include-p10", action="store_true")
    analyse.add_argument("--framing-summary", type=Path, default=DEFAULT_FRAMING_SUMMARY)

    probe = sub.add_parser("probe", help="Gate §6.3: ridge probe from features to within-class position.")
    add_common(probe)
    add_arm(probe)
    probe.add_argument("--arm-dir", type=Path, default=DEFAULT_ARM_DIR)
    probe.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    probe.add_argument("--permutation-seed", type=int, default=DEFAULT_PERMUTATION_SEED)
    probe.add_argument("--include-p10", action="store_true")

    drift = sub.add_parser("drift-proxies", help="§3.3 bound: box geometry and luminance as zero-parameter scores.")
    add_common(drift)
    drift.add_argument("--box-dir", type=Path, default=DEFAULT_BOX_GEOMETRY_DIR)
    drift.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    drift.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    drift.add_argument("--permutation-seed", type=int, default=DEFAULT_PERMUTATION_SEED)
    drift.add_argument("--frame-stride", type=int, default=LUMINANCE_STRIDE)
    drift.add_argument("--luminance-backend", choices=LUMINANCE_BACKENDS, default=DEFAULT_LUMINANCE_BACKEND)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "replicate-exploratory":
        report = run_replicate_exploratory(
            args.output_dir, args.identity_dir, args.segmentation, args.permutations, args.permutation_seed
        )
        print_replicate(report)
        print(f"\nSaved to {args.output_dir / 'replicate_exploratory.json'}  ->  {'PASS' if report['passed'] else 'FAIL'}")
        if not report["passed"]:
            raise SystemExit("Reproduction gate §6.4 failed; the port has a bug. Do not run new arms.")
        return

    if args.command == "predict":
        run_predict(
            args.arm_dir,
            args.manifest,
            args.labels,
            args.output_dir,
            args.k,
            args.naive,
            args.seeds,
            args.device,
        )
        print(f"\nSaved OOF predictions to {args.output_dir}")
        return

    if args.command == "analyze":
        arm = arm_name(args.k, args.naive)
        oof_rows: list[dict] = []
        for seed in args.seeds:
            path = oof_path(args.output_dir, arm, seed)
            if not path.exists():
                raise SystemExit(
                    f"Missing {path}. Run `predict --k {args.k}{' --naive' if args.naive else ''}` first."
                )
            oof_rows.extend(read_oof(path))
        summary, null = analyze(
            oof_rows, args.seeds, args.permutations, args.permutation_seed, args.bootstrap, args.include_p10
        )
        summary["arm"] = arm
        summary["k"] = args.k
        summary["naive"] = args.naive
        balanced_accuracy = loso_balanced_accuracy(args.output_dir, arm, args.seeds)
        summary["secondary"]["loso_balanced_accuracy"] = balanced_accuracy
        summary["secondary"]["paired_delta_vs_framing"] = (
            paired_delta_vs_framing(balanced_accuracy, args.framing_summary)
            if balanced_accuracy
            else {"available": False, "reason": "fold records missing; re-run `predict`"}
        )
        print_analysis(summary)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / f"within_session_summary_{arm}.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        np.savez_compressed(
            args.output_dir / f"permutation_null_{arm}.npz",
            null=null,
            observed=np.asarray(summary["primary"]["mean"]),
            permutation_seed=np.asarray(args.permutation_seed),
        )
        print(f"\nSaved analysis to {args.output_dir}")
        return

    if args.command == "probe":
        report = run_probe(
            args.arm_dir,
            args.manifest,
            args.output_dir,
            args.k,
            args.naive,
            args.permutations,
            args.permutation_seed,
            args.include_p10,
        )
        print_probe(report)
        print(f"\nSaved to {args.output_dir / ('probe_summary_' + report['arm'] + '.json')}")
        return

    if args.command == "drift-proxies":
        report = run_drift_proxies(
            args.manifest,
            args.output_dir,
            args.box_dir,
            args.data_root,
            args.permutations,
            args.permutation_seed,
            args.frame_stride,
            args.luminance_backend,
        )
        print_drift_proxies(report)
        print(f"\nSaved to {args.output_dir / 'drift_proxies.json'}")
        return


if __name__ == "__main__":
    main()
