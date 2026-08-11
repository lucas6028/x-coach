"""Assemble the Stage B evidence table from the arms' prediction CSVs.

One place where every number in `notes/videomae_stage_b_results.md` §2 is computed,
so the retention decision is read off the pre-registered rules rather than assembled
by hand from several runs:

* per-arm test metrics at each seed's own validation-selected threshold, reported as
  a mean over the five seeds with the spread beside it;
* the primary paired delta (late fusion minus normalized pose-only) with a paired
  video-level bootstrap interval;
* the plan's five retention conditions, each evaluated and printed with the number
  that decided it;
* the denominator gate -- whether the re-derived pose-only baseline landed inside the
  published 0.635 +/- 0.010 band. Every delta is meaningless until this passes, so it
  is checked first and reported first.

Guardrails are evaluated on the seed *mean*, never a single seed: the historical
baseline's recall is 0.717 +/- 0.101, so a 0.03 movement in one seed is noise.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.video.classification_metrics import compute_metrics
from src.video.late_fusion import SplitPredictions, align, fuse_run, read_predictions

#: The published normalized pose-only combined balanced accuracy and its band.
DENOMINATOR_TARGET = 0.635
DENOMINATOR_TOLERANCE = 0.010

#: The plan's retention thresholds, quoted rather than re-derived.
MIN_DELTA = 0.02
MAX_GUARDRAIL_DROP = 0.03
REPORTED_METRICS = ("balanced_accuracy", "recall", "specificity", "macro_f1", "f1")


@dataclass
class ArmRuns:
    """One arm's per-seed test predictions, thresholds and metrics."""

    name: str
    seeds: list[int]
    probabilities: list[np.ndarray]
    thresholds: list[float]
    labels: np.ndarray
    video_ids: list[str]
    metrics: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.metrics:
            self.metrics = [
                compute_metrics(probabilities, self.labels, threshold=threshold)
                for probabilities, threshold in zip(self.probabilities, self.thresholds)
            ]

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            metric: {
                "mean": float(np.mean([m[metric] for m in self.metrics])),
                "std": float(np.std([m[metric] for m in self.metrics], ddof=1)) if len(self.metrics) > 1 else 0.0,
                "min": float(np.min([m[metric] for m in self.metrics])),
                "max": float(np.max([m[metric] for m in self.metrics])),
            }
            for metric in REPORTED_METRICS
        }


def prediction_path(predictions_dir: Path, label_mode: str, seed: int) -> Path:
    return predictions_dir / f"{label_mode}_seed{seed}_predictions.csv"


def read_selected_threshold(path: Path) -> float:
    """The validation-selected threshold this run used, as recorded in its CSV.

    It is read rather than recomputed: recomputing on the test probabilities is
    exactly the leak the plan's "threshold only from validation" rule forbids.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        values = {row["selected_threshold"] for row in rows}
    if len(values) != 1:
        raise ValueError(f"{path} records {len(values)} different selected thresholds: {sorted(values)[:5]}")
    return float(values.pop())


def load_single_arm(name: str, predictions_dir: Path, label_mode: str, seeds: list[int]) -> ArmRuns:
    """Read one trained arm's test predictions, keeping each seed's own threshold.

    The threshold rides in the CSV because it was selected on validation during that
    seed's run; recomputing it here would quietly re-select on test.
    """
    probabilities: list[np.ndarray] = []
    thresholds: list[float] = []
    reference: SplitPredictions | None = None

    for seed in seeds:
        path = prediction_path(predictions_dir, label_mode, seed)
        if not path.exists():
            raise FileNotFoundError(f"{name}: no predictions for seed {seed} at {path}")
        test = read_predictions(path)["test"]
        threshold = read_selected_threshold(path)

        if reference is None:
            reference = test
            probabilities.append(np.asarray(test.probabilities, dtype=np.float64))
        else:
            # align() raises if the seeds cover different videos or disagree on a
            # label -- every seed must be scored on exactly the same test set for the
            # seed mean and the paired bootstrap to mean anything.
            _, ordered, _, _ = align(reference, test)
            probabilities.append(ordered)
        thresholds.append(threshold)

    labels = reference.labels
    video_ids = reference.video_ids

    return ArmRuns(
        name=name,
        seeds=list(seeds),
        probabilities=probabilities,
        thresholds=thresholds,
        labels=labels,
        video_ids=video_ids,
    )


def load_late_fusion_arm(
    name: str,
    first_dir: Path,
    second_dir: Path,
    label_mode: str,
    seeds: list[int],
    weight: float = 0.5,
) -> ArmRuns:
    """Fuse two branches seed-for-seed, calibrating and thresholding on validation."""
    probabilities: list[np.ndarray] = []
    thresholds: list[float] = []
    labels: np.ndarray | None = None
    video_ids: list[str] | None = None

    for seed in seeds:
        first = read_predictions(prediction_path(first_dir, label_mode, seed))
        second = read_predictions(prediction_path(second_dir, label_mode, seed))
        fused = fuse_run(first, second, weight=weight)
        if labels is None:
            labels, video_ids = fused["test_labels"], fused["test_video_ids"]
        probabilities.append(np.asarray(fused["test_probabilities"], dtype=np.float64))
        thresholds.append(float(fused["threshold"]))

    return ArmRuns(
        name=name,
        seeds=list(seeds),
        probabilities=probabilities,
        thresholds=thresholds,
        labels=labels,
        video_ids=video_ids,
    )


def denominator_gate(pose_only: ArmRuns) -> dict[str, object]:
    """Did the re-derived pose-only baseline reproduce the published 0.635?"""
    observed = float(np.mean([m["balanced_accuracy"] for m in pose_only.metrics]))
    delta = observed - DENOMINATOR_TARGET
    return {
        "published": DENOMINATOR_TARGET,
        "tolerance": DENOMINATOR_TOLERANCE,
        "re_derived": observed,
        "delta_vs_published": delta,
        "passed": abs(delta) <= DENOMINATOR_TOLERANCE,
    }


def retention_conditions(
    baseline: ArmRuns,
    candidate: ArmRuns,
    bootstrap: dict[str, float],
) -> dict[str, dict[str, object]]:
    """Evaluate the plan's retention thresholds, each with the number that decided it."""
    baseline_summary = baseline.summary()
    candidate_summary = candidate.summary()

    delta = candidate_summary["balanced_accuracy"]["mean"] - baseline_summary["balanced_accuracy"]["mean"]
    per_seed_deltas = [
        candidate.metrics[index]["balanced_accuracy"] - baseline.metrics[index]["balanced_accuracy"]
        for index in range(len(candidate.metrics))
    ]
    positive_seeds = sum(1 for value in per_seed_deltas if value > 0)

    guardrails = {
        metric: candidate_summary[metric]["mean"] - baseline_summary[metric]["mean"]
        for metric in ("recall", "specificity")
    }
    worst_guardrail = min(guardrails.values())

    return {
        "delta_at_least_0.02": {"value": delta, "threshold": MIN_DELTA, "passed": delta >= MIN_DELTA},
        "ci_lower_bound_above_zero": {
            "value": bootstrap["ci_low"],
            "passed": bootstrap["ci_low"] > 0.0,
        },
        "no_guardrail_drop_over_0.03": {
            "value": worst_guardrail,
            "detail": guardrails,
            "threshold": -MAX_GUARDRAIL_DROP,
            "passed": worst_guardrail >= -MAX_GUARDRAIL_DROP,
        },
        "consistent_across_seeds": {
            "value": f"{positive_seeds}/{len(per_seed_deltas)}",
            "per_seed": per_seed_deltas,
            "passed": positive_seeds > len(per_seed_deltas) / 2,
        },
    }


def format_summary_table(arms: list[ArmRuns]) -> str:
    header = f"{'arm':<34}" + "".join(f"{metric:>20}" for metric in REPORTED_METRICS)
    lines = [header, "-" * len(header)]
    for arm in arms:
        summary = arm.summary()
        row = f"{arm.name:<34}"
        for metric in REPORTED_METRICS:
            row += f"{summary[metric]['mean']:>13.3f} +/-{summary[metric]['std']:.3f}"
        lines.append(row)
    return "\n".join(lines)


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=float)
