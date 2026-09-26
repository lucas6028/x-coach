"""Does a silent rule's metric separate the human labels? Action-level, participant-clustered.

The two silent rules this serves -- `jj_incomplete_leg_rom` and `hk_insufficient_knee_lift` -- are
silent because their NUMBER fails, not their metric (see each rule's docstring). EgoExo-Fitness
labels whole ACTIONS, so the question asked here is the one those labels can answer: ranked by
the rule's metric, do the actions humans flagged sit on the fault side of the ones they did not?

Pre-registered in `docs/superpowers/specs/2026-09-26-egoexo-silent-rule-separability-prereg.md`
before the full archive was decoded. The choices that matter, fixed there and implemented here:

- AUC as the Mann-Whitney probability, ties counted half, oriented so 1.0 means "every flagged
  action scores on the fault side of every unflagged one".
- CI by resampling PARTICIPANTS, not actions: one person contributes several actions (EgoExo's
  `original_actor` spans dates; group by name), and resampling actions would treat those as
  independent. Resamples in which a class is empty are skipped and counted.
- Verdict from the CI alone: lower bound above 0.5 -> "separates"; upper bound below 0.5 ->
  "sorted backwards"; otherwise "undetermined". Never "no difference".
- A candidate cut is read off the labels only under leave-one-participant-out, so the cut scored
  on a person never saw that person. It is REPORTED, never applied: the user's decision
  (2026-09-26) is report-only.

Pure functions over plain lists, so the statistics are unit-testable without any pose data.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def auc(scores: Sequence[float], labels: Sequence[int], *, lower_is_fault: bool) -> float:
    """P(flagged action is on the fault side of an unflagged one), ties half. NaN if a class is empty."""
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return math.nan
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p == n:
                wins += 0.5
            elif (p < n) == lower_is_fault:
                wins += 1.0
    return wins / (len(positives) * len(negatives))


@dataclass(frozen=True)
class BootstrapCI:
    low: float
    high: float
    used: int
    skipped: int


def cluster_bootstrap_auc(
    scores: Sequence[float],
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    lower_is_fault: bool,
    n_boot: int = 2000,
    seed: int = 20260926,
    level: float = 0.95,
) -> BootstrapCI:
    """Percentile CI of :func:`auc`, resampling whole groups (participants) with replacement."""
    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        by_group.setdefault(group, []).append(index)
    names = sorted(by_group)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    skipped = 0
    for _ in range(n_boot):
        picked = rng.choice(len(names), size=len(names), replace=True)
        indices = [i for g in picked for i in by_group[names[g]]]
        value = auc([scores[i] for i in indices], [labels[i] for i in indices],
                    lower_is_fault=lower_is_fault)
        if math.isnan(value):
            skipped += 1
            continue
        values.append(value)
    if not values:
        return BootstrapCI(math.nan, math.nan, 0, skipped)
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return BootstrapCI(float(low), float(high), len(values), skipped)


def verdict(ci: BootstrapCI) -> str:
    """The only three words this analysis is allowed to use."""
    if not (_finite(ci.low) and _finite(ci.high)):
        return "undetermined"
    if ci.low > 0.5:
        return "separates"
    if ci.high < 0.5:
        return "sorted backwards"
    return "undetermined"


def fires(score: float, cut: float, *, lower_is_fault: bool) -> bool:
    """Whether a rule reading `score` against `cut` would flag the action."""
    return score < cut if lower_is_fault else score > cut


def youden_cut(scores: Sequence[float], labels: Sequence[int], *, lower_is_fault: bool) -> float:
    """The cut maximising sensitivity + specificity - 1; midpoints between adjacent scores.

    Ties in J break toward the cut that fires LESS, because a false fire is what silenced these
    rules in the first place. NaN if a class is empty.
    """
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = sum(1 for y in labels if y == 0)
    if not n_pos or not n_neg:
        return math.nan
    ordered = sorted(set(scores))
    candidates = [ordered[0] - 1.0, ordered[-1] + 1.0] + [
        (a + b) / 2.0 for a, b in zip(ordered, ordered[1:])
    ]
    best: tuple[float, int, float] | None = None
    for cut in candidates:
        flagged = [fires(s, cut, lower_is_fault=lower_is_fault) for s in scores]
        tp = sum(1 for f, y in zip(flagged, labels) if f and y == 1)
        fp = sum(1 for f, y in zip(flagged, labels) if f and y == 0)
        j = tp / n_pos - fp / n_neg
        key = (j, -sum(flagged), cut if not lower_is_fault else -cut)
        if best is None or key > best:
            best = key
            chosen = cut
    return chosen


@dataclass(frozen=True)
class HeldOut:
    tp: int
    fp: int
    tn: int
    fn: int
    folds_without_a_cut: int

    @property
    def sensitivity(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else math.nan

    @property
    def specificity(self) -> float:
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) else math.nan


def leave_one_group_out(
    scores: Sequence[float],
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    lower_is_fault: bool,
) -> HeldOut:
    """Fit :func:`youden_cut` without each participant, score that participant with it, pool.

    A fold whose training side lacks a class yields no cut; its actions are left unscored and the
    fold is counted, rather than scored against a cut fitted to nothing.
    """
    tp = fp = tn = fn = 0
    no_cut = 0
    for held in sorted(set(groups)):
        train = [i for i, g in enumerate(groups) if g != held]
        test = [i for i, g in enumerate(groups) if g == held]
        cut = youden_cut([scores[i] for i in train], [labels[i] for i in train],
                         lower_is_fault=lower_is_fault)
        if math.isnan(cut):
            no_cut += 1
            continue
        for i in test:
            flagged = fires(scores[i], cut, lower_is_fault=lower_is_fault)
            if flagged and labels[i] == 1:
                tp += 1
            elif flagged:
                fp += 1
            elif labels[i] == 1:
                fn += 1
            else:
                tn += 1
    return HeldOut(tp, fp, tn, fn, no_cut)


def median_or_nan(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if _finite(v)]
    return statistics.median(finite) if finite else math.nan


def rate_below(values: Iterable[float], cut: float) -> float:
    finite = [float(v) for v in values if _finite(v)]
    return sum(1 for v in finite if v < cut) / len(finite) if finite else math.nan


# --- Per-movement action scores, read from each validation runner's --json output -------------

def jumping_jacks_action_scores(
    runner_payload: Mapping, *, spec_cut: float
) -> dict[str, dict[str, float]]:
    """sample_id -> {"score", "spec_cut_rep_rate"} from `run_jumping_jacks_validation.py --json`.

    score: per view, the MEDIAN over scored reps of each rep's widest stance ratio; per action,
    the median over the exo views that scored at least one rep. spec_cut_rep_rate: the share of
    ALL scored reps (views pooled) whose widest stance is below the parent spec's cut -- what the
    rule as written would have done. Actions with no scored rep on any view are omitted.
    """
    out: dict[str, dict[str, float]] = {}
    for action in runner_payload["detail"]["actions"]:
        per_view = [
            median_or_nan(view.get("per_rep_widest", []))
            for view in action["views"].values()
        ]
        pooled = [v for view in action["views"].values() for v in view.get("per_rep_widest", [])]
        score = median_or_nan(per_view)
        if math.isnan(score):
            continue
        out[action["sample_id"]] = {
            "score": score,
            "spec_cut_rep_rate": rate_below(pooled, spec_cut),
        }
    return out


def high_knee_action_scores(
    runner_payload: Mapping, *, gated_views: Sequence[str]
) -> dict[str, dict[str, float]]:
    """sample_id -> {"score", "cited_cut_rate"} from `run_high_knee_validation.py --json`.

    score: the median, over the GATED side cameras, of each camera's median per-rep peak thigh
    elevation. cited_cut_rate: the median over the same cameras of the cited 45-degree cut's
    per-rep fire rate. Actions with no finite score on a gated camera are omitted.
    """
    out: dict[str, dict[str, float]] = {}
    for sample_id, views in runner_payload["per_action"].items():
        rows = [row for name, row in views.items() if name in gated_views]
        score = median_or_nan(row.get("peak_elevation_median", math.nan) for row in rows)
        if math.isnan(score):
            continue
        out[sample_id] = {
            "score": score,
            "cited_cut_rate": median_or_nan(row.get("fire_rate_cited_cut", math.nan) for row in rows),
        }
    return out
