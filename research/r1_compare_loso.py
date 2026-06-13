"""Paired LOSO comparison for R1 (temporal smoothing) — brief §4 discipline.

Loads two ``loso_cross_validation`` JSON outputs (a baseline and a variant) and
reports, on the matched per-subject folds:

- per-fold delta (variant - base), so we see "every fold improves" vs "one fold
  carries the mean" (brief §4.2);
- mean +/- std including and excluding P5 (the data-ceiling subject that dilutes
  any real monocular gain, brief §4.5), and excluding the under-powered P10;
- a Wilcoxon signed-rank p-value across the matched folds (brief §4.2), when
  scipy is available.

Usage:
    python research/r1_compare_loso.py BASE.json VARIANT.json
    python research/r1_compare_loso.py BASE.json V1.json V2.json ...   # sweep
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def fold_map(path: Path) -> dict[str, dict]:
    data = json.load(path.open())
    return {f["test_subject"]: f for f in data["folds"]}


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def subset_mean(deltas: dict[str, float], exclude: set[str]) -> tuple[float, float, int]:
    vals = [v for s, v in deltas.items() if s not in exclude]
    m, sd = mean_std(vals)
    return m, sd, len(vals)


def compare(base_path: Path, var_path: Path) -> None:
    base = fold_map(base_path)
    var = fold_map(var_path)
    shared = [s for s in base if s in var]
    shared.sort(key=int)

    print(f"\n=== {var_path.name}  vs  {base_path.name} ===")
    print(f"{'fold':<6}{'base':>8}{'variant':>9}{'delta':>9}{'n_test':>8}")
    deltas: dict[str, float] = {}
    for s in shared:
        b = base[s]["balanced_accuracy"]
        v = var[s]["balanced_accuracy"]
        d = v - b
        deltas[s] = d
        flag = "  <-P5" if s == "5" else ("  (P10 small)" if base[s]["n_test"] < 100 else "")
        print(f"P{s:<5}{b:>8.3f}{v:>9.3f}{d:>+9.3f}{base[s]['n_test']:>8}{flag}")

    # Brief: report 9-fold (drop tiny P10), plus excl-P5, plus per-fold sign tally.
    big = {s: d for s, d in deltas.items() if base[s]["n_test"] >= 100}
    bm, bsd, bn = subset_mean(big, exclude=set())
    pm, psd, pn = subset_mean(big, exclude={"5"})
    base_big = mean_std([base[s]["balanced_accuracy"] for s in big])
    var_big = mean_std([var[s]["balanced_accuracy"] for s in big])

    pos = sum(1 for d in big.values() if d > 0)
    neg = sum(1 for d in big.values() if d < 0)
    print(
        f"\n  base    (>=100, n={bn}): {base_big[0]:.3f} +/- {base_big[1]:.3f}"
        f"\n  variant (>=100, n={bn}): {var_big[0]:.3f} +/- {var_big[1]:.3f}"
    )
    print(f"  mean delta incl P5 (n={bn}): {bm:+.4f} +/- {bsd:.4f}   folds +/-: {pos}/{neg}")
    print(f"  mean delta excl P5 (n={pn}): {pm:+.4f} +/- {psd:.4f}")

    try:
        from scipy.stats import wilcoxon

        vals = list(big.values())
        if any(v != 0 for v in vals):
            stat, p = wilcoxon(vals)
            print(f"  Wilcoxon signed-rank (incl P5, n={bn}): p={p:.3f}")
    except Exception as exc:  # pragma: no cover
        print(f"  (Wilcoxon skipped: {exc})")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    base = Path(sys.argv[1])
    for var in sys.argv[2:]:
        compare(base, Path(var))


if __name__ == "__main__":
    main()
