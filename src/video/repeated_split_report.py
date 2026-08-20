"""Score the repeated-split arms out-of-fold and compare them.

The fixed-split reader in ``stage_b_report`` refuses runs scored on different test
sets, and that refusal is correct there -- the fixed-split numbers still have to be
reported and they depend on it. Repeated splits are the opposite shape by design: each
fold scores a different slice, and a repeat's five slices together cover the corpus
exactly once. So this is a separate aggregator rather than a relaxation of that guard.

How a repeat becomes one score:

* every fold's test rows are pooled into one out-of-fold vector over all 1623 videos,
  refusing any repeat that misses a video or scores one twice;
* each row keeps the decision its OWN fold made, taken from the CSV's
  ``selected_threshold_prediction`` column. The threshold was selected on that fold's
  validation slice, so pooling the decisions keeps "threshold from validation only"
  true fold by fold. Pooling the probabilities and then choosing one threshold would
  select it on what is effectively the whole corpus.

The five repeats are NOT five independent samples: the same videos are in all of them,
and only the partition changes. A bootstrap therefore resamples VIDEOS and applies the
same draw to every repeat, so the interval carries video-sampling variance while the
averaging over repeats removes only split noise. Pooling 25 folds as independent draws
would report an interval roughly sqrt(5) too narrow.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.video.classification_metrics import compute_metrics

REPORTED_METRICS = ("balanced_accuracy", "recall", "specificity", "macro_f1", "f1")


@dataclass
class RepeatedArm:
    """One arm's out-of-fold decisions, one vector per repeat."""

    name: str
    repeats: list[int]
    #: repeat -> 0/1 decision per video, aligned to ``video_ids``
    decisions: dict[int, np.ndarray]
    labels: np.ndarray
    video_ids: list[str]
    metrics: dict[int, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metrics:
            # Decisions are already 0/1, so a 0.5 threshold reproduces them exactly and
            # the metric implementation stays the one every other arm is scored with.
            self.metrics = {
                repeat: compute_metrics(self.decisions[repeat].astype(np.float64), self.labels, threshold=0.5)
                for repeat in self.repeats
            }

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            metric: {
                "mean": float(np.mean([self.metrics[r][metric] for r in self.repeats])),
                "std": float(np.std([self.metrics[r][metric] for r in self.repeats], ddof=1))
                if len(self.repeats) > 1
                else 0.0,
                "min": float(np.min([self.metrics[r][metric] for r in self.repeats])),
                "max": float(np.max([self.metrics[r][metric] for r in self.repeats])),
            }
            for metric in REPORTED_METRICS
        }

    def subset(self, video_ids: set[str]) -> "RepeatedArm":
        """The same arm restricted to a subset of videos, decisions untouched.

        Restricting AFTER scoring is the point: the fold that produced each decision
        trained and chose its threshold on the whole corpus, exactly as it would in
        production. Re-training on the subset would answer a different question.
        """
        keep = np.asarray([video_id in video_ids for video_id in self.video_ids], dtype=bool)
        if not keep.any():
            raise ValueError(f"{self.name}: the requested subset selects no videos.")
        return RepeatedArm(
            name=self.name,
            repeats=list(self.repeats),
            decisions={repeat: values[keep] for repeat, values in self.decisions.items()},
            labels=self.labels[keep],
            video_ids=[video_id for video_id, flag in zip(self.video_ids, keep) if flag],
        )


def read_fold_decisions(path: Path) -> tuple[list[str], list[int], list[int]]:
    """``(video_ids, decisions, labels)`` for one fold's TEST rows."""
    video_ids: list[str] = []
    decisions: list[int] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != "test":
                continue
            video_ids.append(row["video_id"])
            decisions.append(int(row["selected_threshold_prediction"]))
            labels.append(int(row["label"]))
    if not video_ids:
        raise ValueError(f"{path} contains no test rows.")
    return video_ids, decisions, labels


def fold_prediction_path(predictions_dir: Path, label_mode: str, fold_name: str) -> Path:
    return predictions_dir / f"{label_mode}_{fold_name}_predictions.csv"


def load_repeated_arm(
    name: str,
    predictions_dir: Path,
    label_mode: str,
    fold_names: dict[int, list[str]],
) -> RepeatedArm:
    """Pool every fold of every repeat into one out-of-fold vector per repeat."""
    reference_ids: list[str] | None = None
    reference_labels: dict[str, int] = {}
    decisions: dict[int, np.ndarray] = {}

    for repeat in sorted(fold_names):
        pooled: dict[str, int] = {}
        labels: dict[str, int] = {}
        for fold_name in fold_names[repeat]:
            path = fold_prediction_path(predictions_dir, label_mode, fold_name)
            if not path.exists():
                raise FileNotFoundError(f"{name}: no predictions for fold {fold_name} at {path}")
            fold_ids, fold_decisions, fold_labels = read_fold_decisions(path)
            for video_id, decision, label in zip(fold_ids, fold_decisions, fold_labels):
                if video_id in pooled:
                    raise ValueError(
                        f"{name}: repeat {repeat} scores {video_id} in more than one fold. "
                        "An out-of-fold vector must hold exactly one decision per video."
                    )
                pooled[video_id] = decision
                labels[video_id] = label

        if reference_ids is None:
            reference_ids = sorted(pooled)
            reference_labels = labels
        elif sorted(pooled) != reference_ids:
            missing = set(reference_ids) - set(pooled)
            extra = set(pooled) - set(reference_ids)
            raise ValueError(
                f"{name}: repeat {repeat} covers a different video set "
                f"({len(missing)} missing, {len(extra)} unexpected)."
            )
        elif any(labels[video_id] != reference_labels[video_id] for video_id in reference_ids):
            raise ValueError(f"{name}: repeat {repeat} disagrees with repeat 1 about the labels.")

        decisions[repeat] = np.asarray([pooled[video_id] for video_id in reference_ids], dtype=np.int32)

    if reference_ids is None:
        raise ValueError(f"{name}: no repeats were supplied.")

    return RepeatedArm(
        name=name,
        repeats=sorted(decisions),
        decisions=decisions,
        labels=np.asarray([reference_labels[video_id] for video_id in reference_ids], dtype=np.int32),
        video_ids=reference_ids,
    )


def align_arms(baseline: RepeatedArm, candidate: RepeatedArm) -> None:
    if baseline.video_ids != candidate.video_ids:
        raise ValueError(
            f"{baseline.name} and {candidate.name} were scored on different video sets; "
            "a paired comparison needs the same corpus in the same order."
        )
    if baseline.repeats != candidate.repeats:
        raise ValueError(f"{baseline.name} and {candidate.name} do not share the same repeats.")


def paired_bootstrap_delta(
    baseline: RepeatedArm,
    candidate: RepeatedArm,
    metric: str = "balanced_accuracy",
    resamples: int = 2000,
    seed: int = 20260812,
) -> dict[str, float]:
    """Paired bootstrap over videos of the repeat-averaged metric difference.

    One resample draws videos with replacement and scores BOTH arms on exactly that
    draw in EVERY repeat, then averages over repeats. Applying one draw across all
    repeats is what keeps the interval honest: the repeats share their videos, so
    letting each repeat draw independently would treat the same 1623 videos as 8115
    and shrink the interval by about sqrt(5).

    An athlete appearing in several videos would still make this optimistic, and
    nothing in Fitness-AQA can detect that.
    """
    align_arms(baseline, candidate)
    rng = np.random.default_rng(seed)
    n = baseline.labels.size

    def repeat_averaged_delta(index: np.ndarray) -> float:
        drawn_labels = baseline.labels[index]
        deltas = []
        for repeat in baseline.repeats:
            base = compute_metrics(baseline.decisions[repeat][index].astype(np.float64), drawn_labels, 0.5)[metric]
            cand = compute_metrics(candidate.decisions[repeat][index].astype(np.float64), drawn_labels, 0.5)[metric]
            deltas.append(cand - base)
        return float(np.mean(deltas))

    observed = repeat_averaged_delta(np.arange(n))
    draws = [repeat_averaged_delta(rng.integers(0, n, size=n)) for _ in range(resamples)]

    low, high = (float(value) for value in np.percentile(draws, [2.5, 97.5]))
    return {
        "observed_delta": observed,
        "ci_low": low,
        "ci_high": high,
        "half_width": (high - low) / 2.0,
        "n_videos": int(n),
        "n_repeats": len(baseline.repeats),
        "resamples": resamples,
    }


def denominator_gate(
    arm: RepeatedArm,
    fixed_split_value: float,
    tolerance: float,
) -> dict[str, object]:
    """Did pose-only land near its fixed-split value once the resampling changed?

    This is the leakage check, and it only reads one way. Each fold trains on FEWER
    videos than the fixed split's 1136, so more training data cannot explain a higher
    score; a materially higher pose-only says same-athlete videos are landing on both
    sides of the split, and then every downstream delta is inflated too.
    """
    re_derived = arm.summary()["balanced_accuracy"]["mean"]
    difference = re_derived - fixed_split_value
    return {
        "re_derived": re_derived,
        "fixed_split": fixed_split_value,
        "difference": difference,
        "tolerance": tolerance,
        "passed": abs(difference) <= tolerance,
        "verdict": (
            "comparable"
            if abs(difference) <= tolerance
            else ("suspect leakage" if difference > tolerance else "resampling is harder than the fixed split")
        ),
    }


def format_arm_table(arms: list[RepeatedArm]) -> str:
    header = f"{'arm':<28}{'balanced acc':>16}{'recall':>10}{'specificity':>13}{'repeats':>9}"
    lines = [header]
    for arm in arms:
        summary = arm.summary()
        lines.append(
            f"{arm.name:<28}"
            + f"{summary['balanced_accuracy']['mean']:.4f} ± {summary['balanced_accuracy']['std']:.4f}".rjust(16)
            + f"{summary['recall']['mean']:.3f}".rjust(10)
            + f"{summary['specificity']['mean']:.3f}".rjust(13)
            + f"{len(arm.repeats):>9}"
        )
    return "\n".join(lines)
