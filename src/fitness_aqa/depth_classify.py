"""Shallow-squat classification harness shared by every arm.

Deliberately plain: L2-regularised logistic regression on standardised cue features,
threshold picked on val for balanced accuracy, reported on test. A weak model is the
right instrument here -- the question is whether the *features* carry the depth signal,
so anything that could paper over a bad feature space (deep nets, heavy tuning) works
against the measurement. ``fit_mlp`` exists only as a capacity check.

Uncertainty and arm-vs-arm deltas use a **cluster bootstrap over videos**, not frames:
several labelled frames come from the same clip, so frame-level resampling would treat
correlated samples as independent and shrink the intervals dishonestly. The same
resample indices are applied to both arms, making the delta paired.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


@dataclass
class Standardizer:
    """Train-split mean/std, with NaN cells imputed to the train mean (= 0 after scaling)."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std = np.where(std < 1e-9, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.std
        return np.where(np.isfinite(z), z, 0.0)


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> tuple[np.ndarray, float]:
    """L2-regularised logistic regression via L-BFGS. Returns ``(weights, bias)``."""
    n, d = x.shape
    xb = np.hstack([x, np.ones((n, 1))])
    yy = y.astype(np.float64)

    def obj(theta: np.ndarray) -> tuple[float, np.ndarray]:
        z = xb @ theta
        # log(1 + exp(z)) computed stably
        loss = float(np.mean(np.logaddexp(0.0, z) - yy * z) + 0.5 * l2 * np.sum(theta[:-1] ** 2) / n)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = xb.T @ (p - yy) / n
        grad[:-1] += l2 * theta[:-1] / n
        return loss, grad

    res = minimize(obj, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10})
    return res.x[:-1], float(res.x[-1])


def predict_proba(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x @ w + b)))


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    return 0.5 * (float(pred[pos].mean()) + float(1.0 - pred[neg].mean()))


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC (ties averaged). NaN if either class is absent."""
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    # average ranks within tied score groups
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def select_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising balanced accuracy on the given (validation) scores."""
    cands = np.unique(np.concatenate([score, [0.0, 1.0]]))
    best_t, best_ba = 0.5, -1.0
    for t in cands:
        ba = balanced_accuracy(y, (score >= t).astype(np.float64))
        if ba > best_ba:
            best_t, best_ba = float(t), ba
    return best_t


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (score >= threshold).astype(np.float64)
    pos, neg = y == 1, y == 0
    tp = float(((pred == 1) & pos).sum())
    fp = float(((pred == 1) & neg).sum())
    tn = float(((pred == 0) & neg).sum())
    fn = float(((pred == 0) & pos).sum())
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    return {
        "balanced_accuracy": balanced_accuracy(y, pred),
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if precision and recall else float("nan"),
        "auc": roc_auc(y, score),
        "accuracy": float((pred == y).mean()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "threshold": float(threshold),
    }


@dataclass
class ArmResult:
    name: str
    metrics: dict[str, float]
    test_scores: np.ndarray = field(repr=False)
    test_labels: np.ndarray = field(repr=False)
    test_groups: np.ndarray = field(repr=False)
    coef: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))


def run_arm(
    name: str,
    x: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    groups: dict[str, np.ndarray],
    l2: float = 1.0,
) -> ArmResult:
    """Standardise on train, fit, pick the threshold on val, score test."""
    scaler = Standardizer.fit(x["train"])
    xtr = scaler.transform(x["train"])
    w, b = fit_logistic(xtr, y["train"], l2=l2)
    val_scores = predict_proba(scaler.transform(x["val"]), w, b)
    threshold = select_threshold(y["val"], val_scores)
    test_scores = predict_proba(scaler.transform(x["test"]), w, b)
    metrics = binary_metrics(y["test"], test_scores, threshold)
    metrics["val_balanced_accuracy"] = balanced_accuracy(y["val"], (val_scores >= threshold).astype(float))
    metrics["train_balanced_accuracy"] = balanced_accuracy(
        y["train"], (predict_proba(xtr, w, b) >= threshold).astype(float)
    )
    return ArmResult(name=name, metrics=metrics, test_scores=test_scores,
                     test_labels=y["test"], test_groups=groups["test"], coef=w)


def cluster_bootstrap_indices(groups: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    """Resample whole videos with replacement; return per-replicate row indices."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    reps = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        reps.append(np.concatenate([by_group[g] for g in drawn]))
    return reps


def bootstrap_metric(result: ArmResult, reps: list[np.ndarray], metric: str = "balanced_accuracy") -> np.ndarray:
    t = result.metrics["threshold"]
    out = np.empty(len(reps))
    for i, idx in enumerate(reps):
        out[i] = binary_metrics(result.test_labels[idx], result.test_scores[idx], t)[metric]
    return out


def paired_delta(a: ArmResult, b: ArmResult, reps: list[np.ndarray],
                 metric: str = "balanced_accuracy") -> dict[str, float]:
    """Bootstrap distribution of ``a - b`` on the same resampled videos.

    ``p_two_sided`` is the bootstrap proportion of replicates on the wrong side of zero,
    doubled -- a descriptive significance read, not an exact test.
    """
    da = bootstrap_metric(a, reps, metric)
    db = bootstrap_metric(b, reps, metric)
    d = da - db
    d = d[np.isfinite(d)]
    point = a.metrics[metric] - b.metrics[metric]
    frac_le0 = float(np.mean(d <= 0.0))
    return {
        "delta": float(point),
        "ci_low": float(np.percentile(d, 2.5)),
        "ci_high": float(np.percentile(d, 97.5)),
        "p_two_sided": float(min(1.0, 2.0 * min(frac_le0, 1.0 - frac_le0))),
        "frac_positive": float(np.mean(d > 0.0)),
    }


def fit_mlp(x: dict[str, np.ndarray], y: dict[str, np.ndarray], seed: int,
            hidden: int = 32, epochs: int = 300, lr: float = 1e-2,
            weight_decay: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Capacity check: tiny MLP on the same standardised features. Returns (val, test) scores."""
    import torch  # local: the linear path must not need torch

    torch.manual_seed(seed)
    scaler = Standardizer.fit(x["train"])
    tensors = {k: torch.tensor(scaler.transform(v), dtype=torch.float32) for k, v in x.items()}
    ytr = torch.tensor(y["train"], dtype=torch.float32).unsqueeze(1)
    model = torch.nn.Sequential(
        torch.nn.Linear(tensors["train"].shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Dropout(0.2), torch.nn.Linear(hidden, 1),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = lossf(model(tensors["train"]), ytr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        val = torch.sigmoid(model(tensors["val"])).squeeze(1).numpy()
        test = torch.sigmoid(model(tensors["test"])).squeeze(1).numpy()
    return val, test
