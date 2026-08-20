"""Calibrated late fusion of two prediction branches, plus the paired bootstrap.

Stage B's primary arm (see `notes/videomae_stage_b_results.md` §0.1). Everything here
runs *offline* from the prediction CSVs the classifier already writes, so fusing two
branches costs no retraining and cannot accidentally use a different feature set,
seed or split than the arms it is being compared against.

The three degrees of freedom were pinned before any result existed:

**Platt scaling** to calibrate each branch on the validation split. Isotonic
regression is the usual alternative and is rejected here on sample size: it is a
non-parametric step function and Fitness-AQA's validation split is 243 videos at a
~0.68 positive rate. Platt's own target smoothing (Platt 1999 §2.2) is included --
without it a perfectly separable validation split drives the fit to infinity.

**An unweighted mean of the two calibrated probabilities.** A weight tuned on
validation would be a *fourth* use of the same 243 samples, which already carry
checkpoint selection, threshold selection and the calibration fit. The unweighted
mean is the zero-parameter version; the tuned weight stays available as a diagnostic.

**Seed-to-seed pairing** (pose seed s with VideoMAE seed s), so the fused arm is five
runs rather than a 25-way grid with a best cell in it.

Thresholds come from validation via the same ``find_best_threshold`` the training
runs use -- imported, not reimplemented, so a fused arm and its baseline can never
disagree because of a second copy of balanced accuracy.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.video.classification_metrics import compute_metrics, find_best_threshold

#: Probabilities are clipped before the logit so a saturated branch cannot produce
#: an infinite feature for the calibrator.
PROBABILITY_EPS = 1e-6


@dataclass(frozen=True)
class SplitPredictions:
    video_ids: list[str]
    probabilities: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class PlattCalibration:
    slope: float
    intercept: float

    def apply(self, probabilities: np.ndarray) -> np.ndarray:
        scores = logit(probabilities)
        return 1.0 / (1.0 + np.exp(-(self.slope * scores + self.intercept)))


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
    return np.log(clipped / (1.0 - clipped))


def read_predictions(path: Path) -> dict[str, SplitPredictions]:
    """Load a classifier predictions CSV, grouped by split and ordered as written."""
    by_split: dict[str, tuple[list[str], list[float], list[int]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ids, probabilities, labels = by_split.setdefault(row["split"], ([], [], []))
            ids.append(row["video_id"])
            probabilities.append(float(row["probability"]))
            labels.append(int(row["label"]))

    return {
        split: SplitPredictions(
            video_ids=ids,
            probabilities=np.asarray(probabilities, dtype=np.float64),
            labels=np.asarray(labels, dtype=np.int32),
        )
        for split, (ids, probabilities, labels) in by_split.items()
    }


def align(left: SplitPredictions, right: SplitPredictions) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Align two branches by video id, refusing to fuse mismatched sample sets.

    Two arms trained on different feature dirs iterate the same split file, but a
    missing feature bundle silently shrinks one of them. Fusing by position would
    then pair different videos and still produce a plausible number.
    """
    if sorted(left.video_ids) != sorted(right.video_ids):
        only_left = sorted(set(left.video_ids) - set(right.video_ids))
        only_right = sorted(set(right.video_ids) - set(left.video_ids))
        raise ValueError(
            "The two branches cover different videos and cannot be fused: "
            f"{len(only_left)} only in the first ({only_left[:5]}), "
            f"{len(only_right)} only in the second ({only_right[:5]})."
        )

    # A repeated id would collapse in the position map below and leave one slot of the
    # reordered array unwritten -- i.e. uninitialised memory scored as a probability.
    if len(set(left.video_ids)) != len(left.video_ids):
        seen: set[str] = set()
        duplicates = sorted({vid for vid in left.video_ids if vid in seen or seen.add(vid)})
        raise ValueError(f"Duplicate video ids in a single split cannot be aligned: {duplicates[:5]}")

    order = {video_id: index for index, video_id in enumerate(left.video_ids)}
    right_index = np.asarray([order[video_id] for video_id in right.video_ids])
    right_probabilities = np.empty_like(right.probabilities)
    right_probabilities[right_index] = right.probabilities
    right_labels = np.empty_like(right.labels)
    right_labels[right_index] = right.labels

    if not np.array_equal(left.labels, right_labels):
        raise ValueError("The two branches disagree on labels for the same videos.")

    return left.probabilities, right_probabilities, left.labels, left.video_ids


def fit_platt(probabilities: np.ndarray, labels: np.ndarray, max_iterations: int = 100) -> PlattCalibration:
    """Fit ``sigmoid(a * logit(p) + b)`` on validation, with Platt target smoothing.

    Smoothed targets replace 1/0 with (N+ + 1)/(N+ + 2) and 1/(N- + 2), which keeps a
    separable split from driving the slope to infinity. Solved by Newton/IRLS on two
    parameters, falling back to the last stable step if the Hessian goes singular.
    """
    scores = logit(probabilities)
    labels = np.asarray(labels, dtype=np.float64)
    n_positive = float(labels.sum())
    n_negative = float(labels.size - n_positive)
    high = (n_positive + 1.0) / (n_positive + 2.0)
    low = 1.0 / (n_negative + 2.0)
    targets = np.where(labels > 0.5, high, low)

    design = np.column_stack([scores, np.ones_like(scores)])
    weights = np.zeros(2, dtype=np.float64)
    for _ in range(max_iterations):
        predictions = 1.0 / (1.0 + np.exp(-(design @ weights)))
        gradient = design.T @ (predictions - targets)
        variance = np.clip(predictions * (1.0 - predictions), 1e-12, None)
        hessian = design.T @ (design * variance[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        weights = weights - step
        if np.max(np.abs(step)) < 1e-10:
            break

    return PlattCalibration(slope=float(weights[0]), intercept=float(weights[1]))


def fuse_probabilities(first: np.ndarray, second: np.ndarray, weight: float = 0.5) -> np.ndarray:
    """Weighted mean of two calibrated probabilities; ``weight`` is the first's share."""
    return weight * np.asarray(first, dtype=np.float64) + (1.0 - weight) * np.asarray(second, dtype=np.float64)


def fuse_run(
    first_predictions: dict[str, SplitPredictions],
    second_predictions: dict[str, SplitPredictions],
    objective: str = "balanced_accuracy",
    weight: float = 0.5,
) -> dict[str, object]:
    """Calibrate on val, fuse, pick a threshold on val, evaluate once on test."""
    val_first, val_second, val_labels, _ = align(first_predictions["val"], second_predictions["val"])
    test_first, test_second, test_labels, test_ids = align(first_predictions["test"], second_predictions["test"])

    first_calibration = fit_platt(val_first, val_labels)
    second_calibration = fit_platt(val_second, val_labels)

    val_fused = fuse_probabilities(
        first_calibration.apply(val_first), second_calibration.apply(val_second), weight
    )
    threshold, val_metrics = find_best_threshold(val_fused, val_labels, objective)

    test_fused = fuse_probabilities(
        first_calibration.apply(test_first), second_calibration.apply(test_second), weight
    )

    return {
        "threshold": float(threshold),
        "weight": float(weight),
        "calibration": {
            "first": {"slope": first_calibration.slope, "intercept": first_calibration.intercept},
            "second": {"slope": second_calibration.slope, "intercept": second_calibration.intercept},
        },
        "val_metrics": val_metrics,
        "test_metrics": compute_metrics(test_fused, test_labels, threshold=float(threshold)),
        "test_video_ids": test_ids,
        "test_probabilities": test_fused,
        "test_labels": test_labels,
    }


def paired_bootstrap_delta(
    baseline_probabilities: list[np.ndarray],
    baseline_thresholds: list[float],
    candidate_probabilities: list[np.ndarray],
    candidate_thresholds: list[float],
    labels: np.ndarray,
    metric: str = "balanced_accuracy",
    resamples: int = 2000,
    seed: int = 20260810,
) -> dict[str, float]:
    """Paired bootstrap over VIDEOS of the seed-averaged metric difference.

    One resample draws videos with replacement and scores *both* arms on exactly that
    draw, for every seed, then averages over seeds -- so the interval carries both
    the sampling of the test set and the seed spread, and the pairing removes the
    variance the two arms share.

    The resampling unit is the video because Fitness-AQA has no participant mapping;
    all 1623 labeled ids carry distinct source-video prefixes, so there is no clip
    cluster to respect. An athlete appearing in several test videos would still make
    this interval optimistic, and nothing in the dataset can detect that.
    """
    if len(baseline_probabilities) != len(candidate_probabilities):
        raise ValueError("Both arms must supply the same number of seeds.")
    if not baseline_probabilities:
        raise ValueError("No seeds supplied.")

    rng = np.random.default_rng(seed)
    n = labels.size

    def seed_mean_delta(index: np.ndarray) -> float:
        drawn_labels = labels[index]
        deltas = []
        for base, base_threshold, cand, cand_threshold in zip(
            baseline_probabilities, baseline_thresholds, candidate_probabilities, candidate_thresholds
        ):
            base_metric = compute_metrics(base[index], drawn_labels, threshold=base_threshold)[metric]
            cand_metric = compute_metrics(cand[index], drawn_labels, threshold=cand_threshold)[metric]
            deltas.append(cand_metric - base_metric)
        return float(np.mean(deltas))

    observed = seed_mean_delta(np.arange(n))
    samples = np.asarray([seed_mean_delta(rng.integers(0, n, size=n)) for _ in range(resamples)])

    return {
        "observed_delta": observed,
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "resamples": int(resamples),
        "fraction_positive": float((samples > 0).mean()),
    }
