"""Within-session identity/appearance control for VideoMAE on REHAB24-6.

Pre-registered in ``notes/rehab24_videomae_identity_appearance_validation_plan.md``.
The framing run localised the signal to the person's pixels but could not say whether
the model reads the *movement* or the *person*: every REHAB24-6 recording holds one
subject in one outfit in one room, and each recording has its own correctness base
rate, so a model that only recognised the person could score 0.66 without ever looking
at a repetition.

This module removes that explanation by conditioning on the recording. Inside one
``session`` (one ``exercise_id + video_id``) identity, body, clothing, background,
camera and exercise are constant by construction, so a within-session ROC-AUC above
chance cannot be produced by any of them.

Three design points carry the inference and are worth stating.

*The unit is the subject, not the sample.* Nine subjects, 61 mixed-label sessions,
1072 repetitions and three seeds are not 1072 x 3 observations. Sessions are averaged
within a subject and seeds are averaged within a session, so ``n = 9`` throughout
(plan §4.1). Cameras are averaged into one repetition score before any AUC is taken,
because correctness is a property of the repetition and cam17/cam18 are two views of
it, never two samples.

*The null is within-session too.* A global label shuffle would destroy the per-session
base rate as well as the repetition ordering, so beating it would prove only that
sessions differ. Permuting labels *inside* each session, preserving its class counts,
leaves exactly one thing to beat: ordering repetitions of the same person in the same
recording (plan §4.3).

*Nothing is re-implemented.* Folds, validation-subject choice, hyperparameters, seeds
and the threshold objective come from ``videomae_stage_a.run_arm`` unchanged; this
module only asks it to retain the probabilities it already computes. That is what lets
``audit`` prove the saved out-of-fold probabilities reproduce the committed framing
folds exactly, rather than merely landing near 0.6612.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.rehab24.dataset import DEFAULT_PROCESSED_ROOT, load_manifest
from src.rehab24.loso_cross_validation import (
    FoldConfig,
    MIN_VAL_SUBJECT_SAMPLES,
    subjects_to_samples,
    summarize,
)
from src.rehab24.videomae_stage_a import load_metadata, run_arm

#: Plan §3.1 / §4.2. Frozen so a re-run cannot quietly add a fourth seed.
SEEDS = (42, 7, 1234)
#: The primary visual arm, corrected pooling, mean clip aggregation (plan header).
DEFAULT_ARM_DIR = DEFAULT_PROCESSED_ROOT / "videomae_framing" / "full_frame_letterbox" / "videomae_mean_pool_fc_norm_mean"
DEFAULT_OUTPUT_DIR = DEFAULT_PROCESSED_ROOT / "videomae_identity_control"
#: Committed framing folds, used by ``audit`` as the reproduction reference.
DEFAULT_FRAMING_PREFIX = DEFAULT_PROCESSED_ROOT / "videomae_framing"
PRIMARY_ARM_NAME = "full_frame_letterbox"

#: P10 has 16 samples; primary inference is P1-P9 and P10 is sensitivity only (§4.1).
SENSITIVITY_SUBJECT = "10"

#: Plan §6.1 output schema, in this order.
OOF_COLUMNS = (
    "sample_id",
    "person_id",
    "exercise_id",
    "video_id",
    "repetition_number",
    "camera",
    "label",
    "seed",
    "test_subject",
    "probability",
)

#: Plan §2.1 / §6.2. Hard-coded because a gate that recomputes its own expectation
#: from the data it is checking cannot fail.
EXPECTED = {
    "primary_rows": 2128,
    "primary_subjects": 9,
    "primary_sessions": 64,
    "primary_mixed_sessions": 61,
    "primary_mixed_rows": 2056,
    "all_rows": 2144,
    "all_sessions": 65,
    "all_mixed_sessions": 62,
    "repetitions": 1072,
    "mixed_sessions_per_exercise": {"1": 12, "2": 12, "3": 8, "4": 12, "5": 8, "6": 9},
}

#: Plan §7. A project decision threshold, not a scientific constant and not an
#: equivalence bound.
DECISION_AUC = 0.55
MIN_SUBJECTS_ABOVE_CHANCE = 6
ALPHA = 0.05
#: §4.3 fixes both the count and the sidedness before any outcome is seen.
DEFAULT_PERMUTATIONS = 10_000
DEFAULT_PERMUTATION_SEED = 20260823
DEFAULT_BOOTSTRAP = 10_000


# --------------------------------------------------------------------------- #
# grouping                                                                     #
# --------------------------------------------------------------------------- #


def session_id(row: dict[str, str]) -> str:
    """One recording: same person, same exercise, same take, both cameras."""
    return f"Ex{row['exercise_id']}_{row['video_id']}"


def repetition_id(row: dict[str, str]) -> str:
    return f"{session_id(row)}_rep{row['repetition_number']}"


def manifest_index(manifest_path: Path) -> dict[str, dict[str, str]]:
    return {row["sample_id"]: row for row in load_manifest(manifest_path)}


def audit_labels(rows: Sequence[dict[str, str]]) -> dict:
    """The §2.1 feasibility table, computed from metadata and labels only.

    Deliberately outcome-free: it reads no probability, so it can be run (and the
    grouping fixed) before the frozen-analysis gate closes.
    """
    sessions: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        sessions[session_id(row)].append(row)

    mixed = {key: value for key, value in sessions.items() if len({r["correctness"] for r in value}) > 1}
    dual_camera = [key for key, value in sessions.items() if {r["camera"] for r in value} == {"cam17", "cam18"}]

    per_exercise: dict[str, int] = defaultdict(int)
    per_subject: dict[str, int] = defaultdict(int)
    for key, value in mixed.items():
        per_exercise[value[0]["exercise_id"]] += 1
        per_subject[value[0]["person_id"]] += 1

    return {
        "rows": len(rows),
        "subjects": len({row["person_id"] for row in rows}),
        "sessions": len(sessions),
        "dual_camera_sessions": len(dual_camera),
        "mixed_label_sessions": len(mixed),
        "mixed_label_rows": sum(len(value) for value in mixed.values()),
        "mixed_sessions_per_exercise": dict(sorted(per_exercise.items())),
        "mixed_sessions_per_subject": dict(sorted(per_subject.items(), key=lambda kv: int(kv[0]))),
        "source_videos": len({row["video_path"] for row in rows}),
    }


def audit_pairing(rows: Sequence[dict[str, str]]) -> dict:
    """Gate §6.2: every repetition is exactly one cam17 row plus one cam18 row, and the
    two agree on the label. A repetition whose views disagreed would make the
    camera-averaged score meaningless and is a grouping bug, not a data quirk."""
    repetitions: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        repetitions[repetition_id(row)].append(row)

    bad_pairs = sorted(
        key for key, value in repetitions.items() if sorted(r["camera"] for r in value) != ["cam17", "cam18"]
    )
    disagreeing = sorted(key for key, value in repetitions.items() if len({r["correctness"] for r in value}) > 1)
    return {
        "repetitions": len(repetitions),
        "repetitions_without_camera_pair": bad_pairs,
        "repetitions_with_disagreeing_labels": disagreeing,
        "passed": not bad_pairs and not disagreeing,
    }


# --------------------------------------------------------------------------- #
# prediction (out-of-fold probabilities)                                       #
# --------------------------------------------------------------------------- #


def oof_path(output_dir: Path, seed: int) -> Path:
    return output_dir / f"oof_seed{seed}.csv"


def folds_path(output_dir: Path, seed: int) -> Path:
    return output_dir / f"folds_seed{seed}.json"


def write_oof(path: Path, folds: Sequence[dict], manifest: dict[str, dict[str, str]], seed: int) -> int:
    """Flatten one seed's LOSO folds into the §6.1 row schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OOF_COLUMNS))
        writer.writeheader()
        for fold in folds:
            for sample_id, probability in zip(fold["sample_ids"], fold["probabilities"]):
                row = manifest[sample_id]
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "person_id": row["person_id"],
                        "exercise_id": row["exercise_id"],
                        "video_id": row["video_id"],
                        "repetition_number": row["repetition_number"],
                        "camera": row["camera"],
                        "label": int(row["correctness"]),
                        "seed": seed,
                        "test_subject": fold["test_subject"],
                        # repr-round-trippable, so audit's exact reproduction check is
                        # not defeated by the CSV itself.
                        "probability": repr(float(probability)),
                    }
                )
                written += 1
    return written


def read_oof(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
        row["seed"] = int(row["seed"])
        row["probability"] = float(row["probability"])
    return rows


def run_predict(
    arm_dir: Path,
    manifest_path: Path,
    labels_path: Path,
    output_dir: Path,
    seeds: Sequence[int],
    device_arg: str | None,
) -> dict:
    """Re-run the frozen LOSO recipe, saving out-of-fold probabilities and nothing else.

    All ten subjects are cycled through the test position, exactly as the framing run
    did. P10 is never eligible as a validation subject and sits in the training set of
    every other fold either way, so P1-P9's probabilities are identical to what a
    9-fold restriction would give -- and the extra fold supplies §4.4's P10 sensitivity
    analysis for free. 2128 primary rows + 16 P10 rows = 2144.
    """
    import torch

    device = torch.device("cuda" if (device_arg != "cpu" and device_arg is not None and torch.cuda.is_available()) else "cpu")
    config = FoldConfig()

    labels = {key: int(value) for key, value in json.load(labels_path.open()).items()}
    metadata = load_metadata(manifest_path)
    manifest = manifest_index(manifest_path)
    subject_samples = subjects_to_samples(manifest_path)
    sample_counts = {person: len(ids) for person, ids in subject_samples.items()}
    ordered_subjects = sorted(subject_samples, key=int)

    print(f"Identity-control OOF on {device} | arm={arm_dir} | seeds {list(seeds)}")
    report: dict = {"arm_dir": str(arm_dir), "seeds": list(seeds), "config": vars(config), "per_seed": {}}
    for seed in seeds:
        print(f"\n--- full_frame_letterbox | seed {seed} ---")
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
        )
        rows = write_oof(oof_path(output_dir, seed), folds, manifest, seed)
        with folds_path(output_dir, seed).open("w", encoding="utf-8") as handle:
            json.dump({"seed": seed, "config": vars(config), "folds": folds}, handle, indent=2, sort_keys=True)
        big = [f for f in folds if f["n_test"] >= MIN_VAL_SUBJECT_SAMPLES]
        mean_ba = float(np.mean([f["balanced_accuracy"] for f in big]))
        report["per_seed"][str(seed)] = {"rows": rows, "balanced_accuracy_no_p10": mean_ba}
        print(f"  wrote {rows} OOF rows | bal_acc (9 folds, no P10) {mean_ba:.4f}")

    path = output_dir / "predict_report.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"\nSaved OOF predictions to {output_dir}")
    return report


# --------------------------------------------------------------------------- #
# audit (gates §6.1 / §6.2)                                                    #
# --------------------------------------------------------------------------- #


def audit_oof(
    oof_rows: Sequence[dict],
    manifest: dict[str, dict[str, str]],
    seeds: Sequence[int],
) -> dict:
    """Gate §6.1: completeness, uniqueness, finiteness, and metadata agreement."""
    primary_ids = {sid for sid, row in manifest.items() if row["person_id"] != SENSITIVITY_SUBJECT}
    problems: list[str] = []
    per_seed: dict[str, dict] = {}

    for seed in seeds:
        seed_rows = [row for row in oof_rows if row["seed"] == seed]
        ids = [row["sample_id"] for row in seed_rows]
        duplicates = sorted({sid for sid in ids if ids.count(sid) > 1}) if len(set(ids)) != len(ids) else []
        missing = sorted(primary_ids - set(ids))
        unknown = sorted(set(ids) - set(manifest))
        non_finite = [row["sample_id"] for row in seed_rows if not np.isfinite(row["probability"])]
        leaked = [
            row["sample_id"]
            for row in seed_rows
            if manifest[row["sample_id"]]["person_id"] != row["test_subject"]
        ]
        mislabelled = [
            row["sample_id"]
            for row in seed_rows
            if int(manifest[row["sample_id"]]["correctness"]) != row["label"]
        ]
        per_seed[str(seed)] = {
            "rows": len(seed_rows),
            "primary_rows": sum(1 for row in seed_rows if row["person_id"] != SENSITIVITY_SUBJECT),
            "duplicate_sample_ids": duplicates,
            "missing_primary_sample_ids": missing,
            "unknown_sample_ids": unknown,
            "non_finite_probabilities": non_finite,
            "rows_scored_outside_own_test_fold": leaked,
            "rows_disagreeing_with_manifest_label": mislabelled,
        }
        if per_seed[str(seed)]["primary_rows"] != EXPECTED["primary_rows"]:
            problems.append(f"seed {seed}: {per_seed[str(seed)]['primary_rows']} primary rows, expected {EXPECTED['primary_rows']}")
        for key in (
            "duplicate_sample_ids",
            "missing_primary_sample_ids",
            "unknown_sample_ids",
            "non_finite_probabilities",
            "rows_scored_outside_own_test_fold",
            "rows_disagreeing_with_manifest_label",
        ):
            if per_seed[str(seed)][key]:
                problems.append(f"seed {seed}: {key} = {per_seed[str(seed)][key][:5]} ...")

    return {"per_seed": per_seed, "problems": problems, "passed": not problems}


def reproduces_framing(
    folds_by_seed: dict[int, list[dict]],
    framing_prefix: Path,
    arm_name: str = PRIMARY_ARM_NAME,
    tolerance: float = 1e-9,
) -> dict:
    """The check that discriminates: per-fold identity with the committed framing run.

    "Close to 0.6612" would survive a drifted validation-subject rule, a different
    threshold objective, or a seed applied at the wrong point. Per-fold equality of
    threshold and balanced accuracy will not.
    """
    result: dict = {"reference": str(framing_prefix), "tolerance": tolerance, "per_seed": {}, "problems": []}
    for seed, folds in folds_by_seed.items():
        path = Path(f"{framing_prefix}_seed{seed}.json")
        if not path.exists():
            result["problems"].append(f"seed {seed}: no framing reference at {path}")
            continue
        reference = json.load(path.open(encoding="utf-8"))["arms"][arm_name]
        deltas = []
        for mine, theirs in zip(folds, reference):
            if mine["test_subject"] != theirs["test_subject"]:
                result["problems"].append(f"seed {seed}: fold order differs")
                break
            deltas.append(
                {
                    "test_subject": mine["test_subject"],
                    "balanced_accuracy_delta": mine["balanced_accuracy"] - theirs["balanced_accuracy"],
                    "threshold_delta": mine["threshold"] - theirs["threshold"],
                }
            )
        worst = max((max(abs(d["balanced_accuracy_delta"]), abs(d["threshold_delta"])) for d in deltas), default=None)
        result["per_seed"][str(seed)] = {"n_folds": len(deltas), "max_abs_delta": worst, "folds": deltas}
        if worst is None or worst > tolerance:
            result["problems"].append(f"seed {seed}: max |delta| vs framing = {worst}")
    result["passed"] = not result["problems"]
    return result


# --------------------------------------------------------------------------- #
# within-session statistic                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Session:
    """One recording, reduced to what the statistic needs.

    ``avg_rank`` is the repetition's midrank within this session, averaged over seeds.
    Storing it is what makes the permutation test cheap AND exact: a within-session
    label permutation never moves a score, so the ranks are invariant and only the
    membership of the positive set changes. Because mean-over-seeds AUC is linear in
    the rank sum (the class counts are fixed), averaging ranks across seeds first is
    identical to averaging the three seeds' AUCs -- not an approximation of it.
    """

    session: str
    person_id: str
    exercise_id: str
    labels: np.ndarray
    avg_rank: np.ndarray
    repetitions: tuple[str, ...]

    @property
    def n_pos(self) -> int:
        return int(self.labels.sum())

    @property
    def n_neg(self) -> int:
        return int(len(self.labels) - self.labels.sum())

    @property
    def offset(self) -> float:
        return self.n_pos * (self.n_pos + 1) / 2.0

    @property
    def denominator(self) -> float:
        return float(self.n_pos * self.n_neg)

    def auc(self, labels: np.ndarray) -> np.ndarray:
        """AUC for one or many label vectors; rows of ``labels`` are permutations."""
        return (labels @ self.avg_rank - self.offset) / self.denominator


def midranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged -- the definition ``roc_auc_score`` uses.

    Camera-averaged probabilities do tie (two views of one repetition can average to
    the same value as another's), and ``argsort`` ranks disagree with the AUC exactly
    on those ties, which would make the observed statistic and the null come from two
    slightly different estimators.
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_values = values[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def repetition_scores(
    oof_rows: Sequence[dict],
    seeds: Sequence[int],
    camera: str | None = None,
) -> dict[str, dict]:
    """Collapse camera rows into one score per repetition per seed (§4.2 step 2).

    ``camera`` restricts to a single view for the §4.4 secondary analysis; the primary
    analysis averages the pair, because correctness belongs to the repetition.
    """
    grouped: dict[str, dict] = {}
    for row in oof_rows:
        if camera is not None and row["camera"] != camera:
            continue
        key = repetition_id(row)
        entry = grouped.setdefault(
            key,
            {
                "session": session_id(row),
                "person_id": row["person_id"],
                "exercise_id": row["exercise_id"],
                "label": row["label"],
                "by_seed": defaultdict(list),
            },
        )
        if entry["label"] != row["label"]:
            raise ValueError(f"Repetition {key} carries disagreeing labels across camera rows.")
        entry["by_seed"][row["seed"]].append(row["probability"])

    expected = 1 if camera is not None else 2
    for key, entry in grouped.items():
        for seed in seeds:
            probabilities = entry["by_seed"].get(seed, [])
            if len(probabilities) != expected:
                raise ValueError(
                    f"Repetition {key} has {len(probabilities)} rows for seed {seed}, expected {expected}. "
                    "Fix the grouping before analysing."
                )
        entry["score_by_seed"] = {seed: float(np.mean(entry["by_seed"][seed])) for seed in seeds}
        del entry["by_seed"]
    return grouped


def build_sessions(
    scores: dict[str, dict],
    seeds: Sequence[int],
    drop_subject: str | None = SENSITIVITY_SUBJECT,
) -> tuple[list[Session], list[str]]:
    """Mixed-label sessions as ``Session`` records, plus the single-class ones dropped.

    §4.2 step 3: an AUC is undefined on a single-class session and the plan forbids
    substituting another metric there, so those sessions are named and excluded rather
    than rescued.
    """
    grouped: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for key, entry in scores.items():
        if drop_subject is not None and entry["person_id"] == drop_subject:
            continue
        grouped[entry["session"]].append((key, entry))

    sessions: list[Session] = []
    single_class: list[str] = []
    for name in sorted(grouped):
        items = sorted(grouped[name], key=lambda kv: kv[0])
        labels = np.asarray([entry["label"] for _, entry in items], dtype=float)
        if len(np.unique(labels)) < 2:
            single_class.append(name)
            continue
        rank_stack = np.stack(
            [midranks(np.asarray([entry["score_by_seed"][seed] for _, entry in items], dtype=float)) for seed in seeds]
        )
        sessions.append(
            Session(
                session=name,
                person_id=items[0][1]["person_id"],
                exercise_id=items[0][1]["exercise_id"],
                labels=labels,
                avg_rank=rank_stack.mean(axis=0),
                repetitions=tuple(key for key, _ in items),
            )
        )
    return sessions, single_class


def subject_macro(sessions: Sequence[Session], labels_per_session: Sequence[np.ndarray]) -> np.ndarray:
    """Session AUC -> subject mean -> equal-weight mean over subjects (§4.2 steps 4-6).

    Accepts label matrices so the observed statistic and all 10,000 permutations run
    through this identical hierarchy; a null computed by a shortcut would not be a null
    for this statistic.
    """
    per_subject: dict[str, list[np.ndarray]] = defaultdict(list)
    for session, labels in zip(sessions, labels_per_session):
        per_subject[session.person_id].append(np.atleast_2d(session.auc(np.atleast_2d(labels))))
    subject_means = [np.mean(np.concatenate(values, axis=0), axis=0) for _, values in sorted(per_subject.items(), key=lambda kv: int(kv[0]))]
    return np.mean(np.stack(subject_means, axis=0), axis=0)


def observed_statistic(sessions: Sequence[Session]) -> dict:
    per_session = {session.session: float(session.auc(session.labels)) for session in sessions}
    per_subject: dict[str, list[float]] = defaultdict(list)
    for session in sessions:
        per_subject[session.person_id].append(per_session[session.session])
    subjects = {person: float(np.mean(values)) for person, values in sorted(per_subject.items(), key=lambda kv: int(kv[0]))}
    values = list(subjects.values())
    return {
        "per_session_auc": per_session,
        "per_subject_auc": subjects,
        "n_subjects": len(subjects),
        "n_sessions": len(sessions),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "range": [float(np.min(values)), float(np.max(values))],
        "n_subjects_above_chance": int(sum(value > 0.5 for value in values)),
    }


def pairwise_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC straight from its definition: P(pos > neg) + 0.5 P(pos == neg).

    Written out rather than imported so the rank machinery is checked against an
    independent implementation. scikit-learn is not a dependency of this repo, and
    checking the fast path against a rearrangement of itself would check nothing.
    """
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / (len(positives) * len(negatives)))


def verify_estimator(sessions: Sequence[Session], scores: dict[str, dict], seeds: Sequence[int]) -> float:
    """Assert the seed-averaged rank statistic IS the seed-averaged ROC-AUC.

    Without this the permutation p-value would rest on an estimator nobody checked --
    and observed and null must come from the identical estimator or the p-value means
    nothing. Returns the largest absolute discrepancy over all sessions.
    """
    worst = 0.0
    for session in sessions:
        labels = np.asarray([scores[key]["label"] for key in session.repetitions], dtype=int)
        reference = float(
            np.mean(
                [
                    pairwise_auc(
                        labels,
                        np.asarray([scores[key]["score_by_seed"][seed] for key in session.repetitions], dtype=float),
                    )
                    for seed in seeds
                ]
            )
        )
        worst = max(worst, abs(reference - float(session.auc(session.labels))))
    return worst


def permutation_null(
    sessions: Sequence[Session],
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """§4.3: shuffle labels inside each session, preserving its class counts.

    One permutation is drawn per session per replicate and applied to both cameras and
    all three seeds at once -- which is automatic here, because cameras were already
    averaged into the repetition score and seeds into ``avg_rank``.
    """
    rng = np.random.default_rng(seed)
    per_session = [rng.permuted(np.tile(session.labels, (n_permutations, 1)), axis=1) for session in sessions]
    return subject_macro(sessions, per_session)


def bootstrap_interval(values: Sequence[float], n_resamples: int, seed: int) -> list[float]:
    """Percentile interval resampling SUBJECTS -- the independent unit (§4.3).

    Reported as an uncertainty description next to the permutation test, never in place
    of it.
    """
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(array), size=(n_resamples, len(array)))
    means = array[draws].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def exact_wilcoxon_vs(values: Sequence[float], reference: float = 0.5) -> dict | None:
    """§4.4 sensitivity test. Exact, because the normal approximation is invalid at n=9."""
    deltas = [float(value) - reference for value in values]
    if not deltas or all(delta == 0 for delta in deltas):
        return None
    try:
        from scipy.stats import wilcoxon
    except ImportError:  # pragma: no cover
        return None
    try:
        stat, p_value = wilcoxon(deltas, method="exact", alternative="greater")
    except TypeError:  # older scipy
        stat, p_value = wilcoxon(deltas, mode="exact", alternative="greater")
    return {"stat": float(stat), "p_value": float(p_value), "alternative": "greater", "method": "exact"}


def holm_correct(p_values: dict[str, float]) -> dict[str, dict[str, float | bool]]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    corrected: dict[str, dict[str, float | bool]] = {}
    running = 0.0
    for index, (name, raw) in enumerate(ordered):
        running = max(running, min(1.0, raw * (total - index)))
        corrected[name] = {"raw": raw, "holm": running, "significant": running < ALPHA}
    return corrected


def preregistered_verdict(statistic: dict, p_value: float) -> dict:
    """Plan §7 row 1, evaluated exactly as written: three conditions, all required."""
    conditions = {
        "mean_auc_at_least_0.55": statistic["mean"] >= DECISION_AUC,
        "at_least_6_of_9_subjects_above_0.5": statistic["n_subjects_above_chance"] >= MIN_SUBJECTS_ABOVE_CHANCE,
        "permutation_p_below_0.05": p_value < ALPHA,
    }
    passed = all(conditions.values())
    return {
        "conditions": conditions,
        "primary_pass": passed,
        "reading": (
            "stable identity/clothing/body/background/session conditions do NOT explain all of the "
            "VideoMAE correctness signal; a repetition-varying visual signal is present"
            if passed
            else "undetermined: this design does not establish within-subject/within-recording correctness ordering"
        ),
    }


# --------------------------------------------------------------------------- #
# analysis driver                                                              #
# --------------------------------------------------------------------------- #


def analyze(
    oof_rows: Sequence[dict],
    seeds: Sequence[int],
    n_permutations: int,
    permutation_seed: int,
    n_bootstrap: int,
) -> tuple[dict, np.ndarray]:
    primary_scores = repetition_scores(oof_rows, seeds)
    sessions, single_class = build_sessions(primary_scores, seeds)
    if len(sessions) != EXPECTED["primary_mixed_sessions"]:
        raise SystemExit(
            f"Grouping gate (§6.2): {len(sessions)} mixed-label sessions, expected "
            f"{EXPECTED['primary_mixed_sessions']}. Fix grouping before analysing."
        )

    estimator_error = verify_estimator(sessions, primary_scores, seeds)
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

    # --- secondary (§4.4). None of these can replace the primary result. -------
    secondary: dict = {}
    for camera in ("cam17", "cam18"):
        camera_sessions, _ = build_sessions(repetition_scores(oof_rows, seeds, camera=camera), seeds)
        secondary[f"by_camera_{camera}"] = observed_statistic(camera_sessions)

    per_exercise: dict[str, dict] = {}
    for exercise in sorted({session.exercise_id for session in sessions}):
        subset = [session for session in sessions if session.exercise_id == exercise]
        per_exercise[exercise] = observed_statistic(subset)
    secondary["by_exercise"] = per_exercise

    with_p10_sessions, with_p10_single = build_sessions(primary_scores, seeds, drop_subject=None)
    secondary["p10_inclusive_sensitivity"] = {
        **observed_statistic(with_p10_sessions),
        "single_class_sessions_excluded": with_p10_single,
    }

    secondary["wilcoxon_vs_chance"] = exact_wilcoxon_vs(list(statistic["per_subject_auc"].values()))

    raw_p = {
        name: value["p_value"]
        for name, value in (
            ("wilcoxon_vs_chance", secondary["wilcoxon_vs_chance"]),
        )
        if value
    }
    secondary["holm"] = holm_correct(raw_p) if raw_p else {}

    summary = {
        "seeds": list(seeds),
        "n_repetitions": len(primary_scores),
        "primary": primary,
        "secondary": secondary,
    }
    return summary, null


def print_analysis(summary: dict) -> None:
    primary = summary["primary"]
    print("\n=== primary: within-session correctness ordering (9 subjects) ===")
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
        f"  within-session permutation null: mean {perm['null_mean']:.4f}, p95 {perm['null_p95']:.4f}  ->  "
        f"one-sided p = {perm['p_value']:.5f}  ({perm['n_permutations']} permutations, seed {perm['permutation_seed']})"
    )
    verdict = primary["verdict"]
    for name, value in verdict["conditions"].items():
        print(f"    [{'PASS' if value else 'FAIL'}] {name}")
    print(f"  pre-registered reading: {verdict['reading']}")

    print("\n=== secondary (never a substitute for the primary) ===")
    for camera in ("cam17", "cam18"):
        stat = summary["secondary"][f"by_camera_{camera}"]
        print(f"  {camera}: mean {stat['mean']:.4f} ({stat['n_subjects_above_chance']}/{stat['n_subjects']} > 0.5)")
    for exercise, stat in summary["secondary"]["by_exercise"].items():
        print(
            f"  Ex{exercise}: mean {stat['mean']:.4f}  ({stat['n_sessions']} sessions, "
            f"{stat['n_subjects_above_chance']}/{stat['n_subjects']} subjects > 0.5)"
        )
    p10 = summary["secondary"]["p10_inclusive_sensitivity"]
    print(f"  P10-inclusive: mean {p10['mean']:.4f} over {p10['n_subjects']} subjects")
    wilcoxon = summary["secondary"]["wilcoxon_vs_chance"]
    if wilcoxon:
        print(f"  exact Wilcoxon vs 0.5 (one-sided): p = {wilcoxon['p_value']:.5f}")


# --------------------------------------------------------------------------- #
# exploratory: zero-parameter duration control (NOT pre-registered)             #
# --------------------------------------------------------------------------- #


def shortcut_controls(
    oof_rows: Sequence[dict],
    manifest: dict[str, dict[str, str]],
    seeds: Sequence[int],
    n_permutations: int,
    permutation_seed: int,
) -> dict:
    """Zero-parameter scores run through the identical within-session statistic.

    EXPLORATORY: added after the primary result and flagged as a plan deviation
    wherever it is reported. Two channels, both of which can order repetitions inside
    one recording without any model reading any movement:

    ``duration``
        A repetition's frame range decides which pixels VideoMAE is shown
        (``sample_clip_starts`` spans exactly that range), so a length difference
        between correct and incorrect repetitions is directly readable. This project
        has already been burned by that shortcut once, on Fitness-AQA squats.

    ``position``
        Where the repetition sits in the recording. This one is NOT covered by the
        plan's §10 limits, which speak only about ordering *within* a repetition. If
        REHAB24-6's protocol records correct repetitions before erroneous ones, then
        position is nearly deterministic of the label inside a session, and anything
        that drifts down a recording -- lighting, camera settle, the subject's
        standing position, encoder state -- becomes a route to a high AUC.

    ``label_runs`` counts alternations between correct and incorrect down each
    session's repetition order: 2 runs means the labels are one contiguous block each,
    which is what a blocked recording protocol looks like.

    These bound the shortcuts. They do not remove them, and a strong ``position``
    result cannot be undone by any analysis of these same recordings.
    """
    from scipy.stats import spearmanr

    scores = repetition_scores(oof_rows, seeds)
    sessions, _ = build_sessions(scores, seeds)
    model_auc = {session.session: float(session.auc(session.labels)) for session in sessions}

    features = {
        "duration": {
            repetition_id(row): float(int(row["last_frame"]) - int(row["first_frame"])) for row in manifest.values()
        },
        "position": {repetition_id(row): float(row["repetition_number"]) for row in manifest.values()},
    }

    report: dict = {
        "note": "EXPLORATORY, added after the primary result; not pre-registered",
        "n_sessions": len(sessions),
        "controls": {},
    }

    for name, values in features.items():
        control_sessions = [
            Session(
                session=session.session,
                person_id=session.person_id,
                exercise_id=session.exercise_id,
                labels=session.labels,
                avg_rank=midranks(np.asarray([values[key] for key in session.repetitions], dtype=float)),
                repetitions=session.repetitions,
            )
            for session in sessions
        ]
        statistic = observed_statistic(control_sessions)
        null = permutation_null(control_sessions, n_permutations, permutation_seed)
        # Two-sided in spirit: a control that predicts the label BACKWARDS is just as
        # much a shortcut as one that predicts it forwards, and `position` does exactly
        # that. Both tails are reported rather than only the pre-registered one.
        p_greater = float((1 + int(np.sum(null >= statistic["mean"]))) / (n_permutations + 1))
        p_less = float((1 + int(np.sum(null <= statistic["mean"]))) / (n_permutations + 1))
        control_auc = {s.session: float(s.auc(s.labels)) for s in control_sessions}
        correlations = [
            float(spearmanr(session.avg_rank, [values[key] for key in session.repetitions]).statistic)
            for session in sessions
        ]
        neutral = [key for key, value in control_auc.items() if 0.35 <= value <= 0.65]
        informative = [key for key in control_auc if key not in neutral]
        report["controls"][name] = {
            "subject_macro": statistic,
            "permutation_p_greater": p_greater,
            "permutation_p_less": p_less,
            "distance_from_chance": abs(statistic["mean"] - 0.5),
            "spearman_model_rank_vs_feature": {
                "mean": float(np.mean(correlations)),
                "median": float(np.median(correlations)),
            },
            "model_auc_where_control_is_neutral": {
                "definition": "sessions whose control-only AUC lies in [0.35, 0.65]",
                "n_sessions": len(neutral),
                "mean_model_auc": float(np.mean([model_auc[key] for key in neutral])) if neutral else None,
            },
            "model_auc_where_control_is_informative": {
                "n_sessions": len(informative),
                "mean_model_auc": float(np.mean([model_auc[key] for key in informative])) if informative else None,
            },
            "per_session_auc": control_auc,
        }

    runs = []
    for session in sessions:
        order = np.argsort(np.asarray([features["position"][key] for key in session.repetitions], dtype=float))
        ordered = session.labels[order]
        runs.append(1 + int(np.sum(ordered[1:] != ordered[:-1])))
    report["label_runs_per_session"] = {
        "definition": "alternations between correct and incorrect down the repetition order; 2 = fully blocked",
        "median": int(np.median(runs)),
        "min": int(np.min(runs)),
        "max": int(np.max(runs)),
        "n_fully_blocked": int(sum(run <= 2 for run in runs)),
        "n_sessions": len(runs),
    }
    return report


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="REHAB24-6 VideoMAE identity/appearance control.")
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="Re-run the frozen LOSO recipe, saving out-of-fold probabilities.")
    add_common(predict)
    predict.add_argument("--arm-dir", type=Path, default=DEFAULT_ARM_DIR)
    predict.add_argument("--labels", type=Path, default=DEFAULT_PROCESSED_ROOT / "labels" / "correctness.json")
    predict.add_argument("--device", type=str, default=None, help="cpu, cuda, or auto.")

    audit = sub.add_parser("audit", help="Gates §6.1/§6.2 plus the §2.1 labels-only feasibility table.")
    add_common(audit)
    audit.add_argument("--labels-only", action="store_true", help="Run the outcome-free checks only (safe pre-freeze).")
    audit.add_argument("--framing-prefix", type=Path, default=DEFAULT_FRAMING_PREFIX)

    analyse = sub.add_parser("analyze", help="Primary within-session statistic and permutation test.")
    add_common(analyse)
    analyse.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    analyse.add_argument("--permutation-seed", type=int, default=DEFAULT_PERMUTATION_SEED)
    analyse.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)

    shortcuts = sub.add_parser(
        "shortcut-controls",
        help="EXPLORATORY (not pre-registered): repetition length and position as zero-parameter scores.",
    )
    add_common(shortcuts)
    shortcuts.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    shortcuts.add_argument("--permutation-seed", type=int, default=DEFAULT_PERMUTATION_SEED)

    extract = sub.add_parser("extract-appearance", help="Label-blind canonical-frame appearance-only arm.")
    add_common(extract)
    extract.add_argument("--variant", choices=("canonical_frame_repeat",), default="canonical_frame_repeat")
    extract.add_argument("--data-root", type=Path, default=None)
    extract.add_argument("--model-name", type=str, default="MCG-NJU/videomae-base-finetuned-kinetics")
    extract.add_argument("--device", type=str, default=None)
    extract.add_argument("--overwrite", action="store_true")

    evaluate = sub.add_parser("evaluate-appearance", help="LOSO on the appearance-only arm, paired against the primary arm.")
    add_common(evaluate)
    evaluate.add_argument("--labels", type=Path, default=DEFAULT_PROCESSED_ROOT / "labels" / "correctness.json")
    evaluate.add_argument("--device", type=str, default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "predict":
        run_predict(args.arm_dir, args.manifest, args.labels, args.output_dir, args.seeds, args.device)
        return

    if args.command == "audit":
        rows = load_manifest(args.manifest)
        primary_rows = [row for row in rows if row["person_id"] != SENSITIVITY_SUBJECT]
        report = {
            "expected": EXPECTED,
            "labels_all": audit_labels(rows),
            "labels_primary": audit_labels(primary_rows),
            "pairing": audit_pairing(rows),
        }
        gate = report["labels_primary"]
        checks = {
            "primary_rows": gate["rows"] == EXPECTED["primary_rows"],
            "primary_sessions": gate["sessions"] == EXPECTED["primary_sessions"],
            "primary_mixed_sessions": gate["mixed_label_sessions"] == EXPECTED["primary_mixed_sessions"],
            "primary_mixed_rows": gate["mixed_label_rows"] == EXPECTED["primary_mixed_rows"],
            "mixed_per_exercise": gate["mixed_sessions_per_exercise"] == EXPECTED["mixed_sessions_per_exercise"],
            "camera_pairing": report["pairing"]["passed"],
        }
        report["labels_only_gate"] = {"checks": checks, "passed": all(checks.values())}

        for name, table in (("all (incl. P10)", report["labels_all"]), ("primary (P1-P9)", report["labels_primary"])):
            print(
                f"{name:<20} rows {table['rows']:>5}  subjects {table['subjects']:>3}  sessions {table['sessions']:>3}  "
                f"dual-camera {table['dual_camera_sessions']:>3}  mixed {table['mixed_label_sessions']:>3}  "
                f"mixed rows {table['mixed_label_rows']:>5}  source videos {table['source_videos']:>4}"
            )
        print(f"pairing: {report['pairing']['repetitions']} repetitions, "
              f"{'all cam17+cam18 with agreeing labels' if report['pairing']['passed'] else 'FAILED'}")
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

        if not args.labels_only:
            oof_rows: list[dict] = []
            folds_by_seed: dict[int, list[dict]] = {}
            for seed in args.seeds:
                path = oof_path(args.output_dir, seed)
                if not path.exists():
                    raise SystemExit(f"Missing {path}. Run `predict` first, or pass --labels-only.")
                oof_rows.extend(read_oof(path))
                folds_by_seed[seed] = json.load(folds_path(args.output_dir, seed).open(encoding="utf-8"))["folds"]
            report["oof"] = audit_oof(oof_rows, manifest_index(args.manifest), args.seeds)
            report["framing_reproduction"] = reproduces_framing(folds_by_seed, args.framing_prefix)
            print(f"\nOOF completeness (§6.1): {'PASS' if report['oof']['passed'] else 'FAIL'}")
            for problem in report["oof"]["problems"]:
                print(f"  {problem}")
            print(f"framing reproduction (§6.1): {'PASS' if report['framing_reproduction']['passed'] else 'FAIL'}")
            for seed, values in report["framing_reproduction"]["per_seed"].items():
                print(f"  seed {seed}: {values['n_folds']} folds, max |delta| = {values['max_abs_delta']:.3e}")
            for problem in report["framing_reproduction"]["problems"]:
                print(f"  {problem}")
            report["passed"] = report["labels_only_gate"]["passed"] and report["oof"]["passed"] and report["framing_reproduction"]["passed"]
        else:
            report["passed"] = report["labels_only_gate"]["passed"]

        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / ("audit_labels_only.json" if args.labels_only else "audit.json")
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"\nSaved audit to {path}  ->  {'PASS' if report['passed'] else 'FAIL'}")
        if not report["passed"]:
            raise SystemExit("Audit gate failed; do not proceed to `analyze`.")
        return

    if args.command == "analyze":
        oof_rows: list[dict] = []
        for seed in args.seeds:
            path = oof_path(args.output_dir, seed)
            if not path.exists():
                raise SystemExit(f"Missing {path}. Run `predict` then `audit` first.")
            oof_rows.extend(read_oof(path))
        summary, null = analyze(oof_rows, args.seeds, args.permutations, args.permutation_seed, args.bootstrap)
        print_analysis(summary)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "within_session_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        np.savez_compressed(
            args.output_dir / "permutation_null.npz",
            null=null,
            observed=np.asarray(summary["primary"]["mean"]),
            permutation_seed=np.asarray(args.permutation_seed),
        )
        print(f"\nSaved analysis to {args.output_dir}")
        return

    if args.command == "shortcut-controls":
        oof_rows = []
        for seed in args.seeds:
            oof_rows.extend(read_oof(oof_path(args.output_dir, seed)))
        report = shortcut_controls(
            oof_rows, manifest_index(args.manifest), args.seeds, args.permutations, args.permutation_seed
        )
        print("\n=== EXPLORATORY zero-parameter shortcut controls (not pre-registered) ===")
        for name, values in report["controls"].items():
            statistic = values["subject_macro"]
            print(
                f"  {name:<9} within-session AUC {statistic['mean']:.4f}  "
                f"({statistic['n_subjects_above_chance']}/{statistic['n_subjects']} subjects > 0.5, "
                f"|AUC-0.5| = {values['distance_from_chance']:.4f})"
            )
            print(
                f"    permutation p: greater {values['permutation_p_greater']:.5f}, "
                f"less {values['permutation_p_less']:.5f}   "
                f"Spearman(model rank, {name}) median {values['spearman_model_rank_vs_feature']['median']:+.3f}"
            )
            neutral = values["model_auc_where_control_is_neutral"]
            informative = values["model_auc_where_control_is_informative"]
            neutral_text = f"{neutral['mean_model_auc']:.4f}" if neutral["mean_model_auc"] is not None else "n/a"
            informative_text = (
                f"{informative['mean_model_auc']:.4f}" if informative["mean_model_auc"] is not None else "n/a"
            )
            print(
                f"    model AUC where this control is neutral ({neutral['n_sessions']} sessions): {neutral_text}   "
                f"informative ({informative['n_sessions']} sessions): {informative_text}"
            )
        runs = report["label_runs_per_session"]
        print(
            f"  label blocking: median {runs['median']} runs per session (min {runs['min']}, max {runs['max']}); "
            f"{runs['n_fully_blocked']}/{runs['n_sessions']} sessions are one contiguous block per class"
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / "shortcut_controls_summary.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"\nSaved to {path}")
        return

    # The appearance arm lives in its own module; imported lazily so `analyze` never
    # needs torch or a decoder.
    from src.rehab24 import videomae_appearance_only

    if args.command == "extract-appearance":
        videomae_appearance_only.run_extract(
            manifest_path=args.manifest,
            data_root=args.data_root,
            model_name=args.model_name,
            device_arg=args.device,
            overwrite=args.overwrite,
        )
        return

    if args.command == "evaluate-appearance":
        videomae_appearance_only.run_evaluate(
            manifest_path=args.manifest,
            labels_path=args.labels,
            output_dir=args.output_dir,
            seeds=args.seeds,
            device_arg=args.device,
        )
        return


if __name__ == "__main__":
    main()
