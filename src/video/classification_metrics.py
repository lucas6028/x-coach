"""Threshold-free and threshold-applied classification metrics, in numpy only.

Split out of ``videomae_video_classifier`` so the offline steps -- late fusion,
calibration, the bootstrap -- score their predictions with the *same* implementation
the training runs use. A second copy of balanced accuracy or of the threshold search
would let a fusion arm and its baseline disagree for a reason that has nothing to do
with the features.

Numpy only, deliberately: the classifier needs torch, these do not, and keeping them
importable without it is what lets the fusion tests run in CI.
"""

from __future__ import annotations

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def compute_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    if probabilities.size == 0:
        return {
            "threshold": float(threshold),
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "tn": 0.0,
            "fn": 0.0,
            "specificity": 0.0,
            "false_positive_rate": 0.0,
            "balanced_accuracy": 0.0,
            "negative_precision": 0.0,
            "negative_recall": 0.0,
            "negative_f1": 0.0,
            "macro_f1": 0.0,
            "youden_j": 0.0,
        }

    preds = (probabilities >= threshold).astype(np.int32)
    labels = labels.astype(np.int32)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy = float((preds == labels).mean())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    balanced_accuracy = (recall + specificity) / 2
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    negative_recall = specificity
    negative_f1 = (
        2 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if negative_precision + negative_recall
        else 0.0
    )
    macro_f1 = (f1 + negative_f1) / 2
    youden_j = recall + specificity - 1

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "balanced_accuracy": float(balanced_accuracy),
        "negative_precision": float(negative_precision),
        "negative_recall": float(negative_recall),
        "negative_f1": float(negative_f1),
        "macro_f1": float(macro_f1),
        "youden_j": float(youden_j),
    }


def find_best_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    objective: str,
) -> tuple[float, dict[str, float]]:
    if probabilities.size == 0:
        return 0.5, compute_metrics(probabilities, labels, threshold=0.5)

    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32),
                probabilities.astype(np.float32),
            ]
        )
    )

    best_threshold = 0.5
    best_metrics = compute_metrics(probabilities, labels, threshold=0.5)
    for threshold in candidate_thresholds.tolist():
        metrics = compute_metrics(probabilities, labels, threshold=float(threshold))
        if metrics[objective] > best_metrics[objective]:
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics
