"""EgoExo-Fitness E3 — inter-annotator agreement (the human-performance ceiling).

The interpretable-action-judgement labels are crowd/expert annotations with a variable
number of annotators per action (1..5, mean ~1.67). Before training E1/E2 we want to know
how reliably *humans* agree, because that bounds the accuracy a model can meaningfully reach:
if annotators disagree on "keep your back straight" 25% of the time, ~0.75 is the realistic
ceiling for that criterion, and a model scoring there is at the noise floor, not failing.

We report, on the subset of multi-annotator items:

  * **TKV** (binary per technical keypoint, the E1 target): Krippendorff's alpha (nominal),
    plus raw pairwise agreement. Alpha is chance-corrected, which matters because faults are
    only ~17% prevalent — raw agreement is inflated by the easy "pass" majority. Per-criterion
    alpha shows *which* criteria are reliably judged (and thus learnable).
  * **Quality score** (1..5 ordinal, the E2 target): Krippendorff's alpha (ordinal), plus
    pairwise exact / within-1 agreement.

Krippendorff's alpha is implemented locally (numpy only) to keep the repo dependency-light;
it handles the variable annotator counts that Fleiss' kappa / classic ICC do not.

Run from the repo root::

    python scripts/egoexo/compute_agreement.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from src.egoexo.dataset import (
    DEFAULT_ANNOTATION_ROOT,
    DEFAULT_PROCESSED_ROOT,
    build_record_info,
    load_json,
    parse_judged_actions,
)


# --------------------------------------------------------------------------- core metric


def krippendorff_alpha(units: list[list], level: str = "nominal") -> float:
    """Krippendorff's alpha over ``units`` (each a list of ratings for one item).

    Units with fewer than two ratings carry no agreement information and are ignored.
    ``level`` is ``"nominal"`` (categorical) or ``"ordinal"`` (rank-ordered). Returns NaN
    when there is no pairable data and 1.0 when every pairable rating agrees.
    """
    units = [list(u) for u in units if len(u) >= 2]
    if not units:
        return float("nan")

    values = sorted({v for u in units for v in u})
    index = {v: i for i, v in enumerate(values)}
    size = len(values)

    coincidence = np.zeros((size, size), dtype=float)
    for unit in units:
        m = len(unit)
        counts = np.zeros(size)
        for v in unit:
            counts[index[v]] += 1.0
        for c in range(size):
            if counts[c] == 0:
                continue
            for k in range(size):
                if counts[k] == 0:
                    continue
                if c == k:
                    coincidence[c, k] += counts[c] * (counts[c] - 1.0) / (m - 1.0)
                else:
                    coincidence[c, k] += counts[c] * counts[k] / (m - 1.0)

    marginal = coincidence.sum(axis=1)
    n = marginal.sum()
    if n < 2:
        return float("nan")

    def delta(ci: int, ck: int) -> float:
        if level == "nominal":
            return 0.0 if ci == ck else 1.0
        if level == "ordinal":
            lo, hi = (ci, ck) if ci <= ck else (ck, ci)
            span = marginal[lo : hi + 1].sum() - (marginal[ci] + marginal[ck]) / 2.0
            return float(span * span)
        raise ValueError(f"Unsupported level: {level!r}")

    observed = 0.0
    expected = 0.0
    for c in range(size):
        for k in range(c + 1, size):
            d = delta(c, k)
            observed += coincidence[c, k] * d
            expected += marginal[c] * marginal[k] * d
    if expected == 0:
        return float("nan") if observed > 0 else 1.0
    return float(1.0 - (n - 1.0) * observed / expected)


def pairwise_agreement(units: list[list], tol: float = 0.0) -> float:
    """Fraction of within-item annotator pairs whose ratings differ by at most ``tol``."""
    agree = 0
    total = 0
    for unit in units:
        if len(unit) < 2:
            continue
        for a, b in combinations(unit, 2):
            total += 1
            if abs(a - b) <= tol:
                agree += 1
    return agree / total if total else float("nan")


def mean_abs_pairwise_diff(units: list[list]) -> float:
    diffs = [abs(a - b) for unit in units if len(unit) >= 2 for a, b in combinations(unit, 2)]
    return float(np.mean(diffs)) if diffs else float("nan")


# --------------------------------------------------------------------------- assembly


def collect_units(actions):
    """Build per-item rating lists from parsed judged actions.

    Returns ``(score_units, tkv_units_by_criterion, annotator_counts)`` where each
    score unit is a list of 1..5 scores and each TKV unit (per criterion) is a list of
    0/1 pass/fail votes reconstructed from the aggregated vote counts.
    """
    score_units: list[list[int]] = []
    tkv_by_criterion: dict[str, list[list[int]]] = {}
    annotator_counts: Counter[int] = Counter()

    for action in actions:
        annotator_counts[action.num_annotators] += 1
        if action.scores:
            score_units.append(list(action.scores))
        for text, votes in action.criteria.items():
            # 1 = pass (True), 0 = fault (False); order is irrelevant for the metrics.
            ratings = [1] * votes["n_true"] + [0] * votes["n_false"]
            tkv_by_criterion.setdefault(text, []).append(ratings)

    return score_units, tkv_by_criterion, annotator_counts


def compute_report(actions) -> dict:
    score_units, tkv_by_criterion, annotator_counts = collect_units(actions)

    all_tkv_units = [u for units in tkv_by_criterion.values() for u in units]
    multi = lambda units: [u for u in units if len(u) >= 2]

    per_criterion = {}
    for text, units in tkv_by_criterion.items():
        m = multi(units)
        per_criterion[text] = {
            "n_units": len(units),
            "n_units_multi": len(m),
            "alpha_nominal": krippendorff_alpha(m, "nominal") if m else float("nan"),
            "pairwise_agreement": pairwise_agreement(m) if m else float("nan"),
        }

    # Reliability summary -> the learnable label space for E1. Criteria need enough
    # multi-annotator support to estimate alpha; we bucket those by the usual thresholds.
    eligible = {t: m for t, m in per_criterion.items() if m["n_units_multi"] >= 5}

    def bucket(a: float) -> str | None:
        if a != a:  # NaN
            return None
        if a < 0:
            return "worse_than_chance"
        if a < 0.2:
            return "poor"
        if a < 0.4:
            return "weak"
        if a < 0.667:
            return "moderate"
        return "acceptable"

    bucket_counts = Counter(bucket(m["alpha_nominal"]) for m in eligible.values())
    reliable = sorted(
        (t for t, m in eligible.items() if m["alpha_nominal"] >= 0.4),
        key=lambda t: -eligible[t]["alpha_nominal"],
    )
    reliability_summary = {
        "min_multi_units": 5,
        "reliable_threshold": 0.4,
        "n_eligible": len(eligible),
        "buckets": {k: bucket_counts.get(k, 0)
                    for k in ("worse_than_chance", "poor", "weak", "moderate", "acceptable")},
        "n_reliable": len(reliable),
        "reliable_criteria": reliable,
    }

    total_items = len(actions)
    multi_items = sum(c for k, c in annotator_counts.items() if k >= 2)
    return {
        "coverage": {
            "total_judged_actions": total_items,
            "multi_annotator_actions": multi_items,
            "multi_annotator_fraction": round(multi_items / total_items, 3) if total_items else 0.0,
            "annotators_per_action": dict(sorted(annotator_counts.items())),
        },
        "reliability_summary": reliability_summary,
        "score": {
            "n_units_multi": len(multi(score_units)),
            "alpha_ordinal": krippendorff_alpha(score_units, "ordinal"),
            "pairwise_exact_agreement": pairwise_agreement(score_units, tol=0),
            "pairwise_within_1": pairwise_agreement(score_units, tol=1),
            "mean_abs_pairwise_diff": mean_abs_pairwise_diff(score_units),
        },
        "tkv_overall": {
            "n_units_multi": len(multi(all_tkv_units)),
            "alpha_nominal": krippendorff_alpha(all_tkv_units, "nominal"),
            "pairwise_agreement": pairwise_agreement(multi(all_tkv_units)),
        },
        "tkv_per_criterion": per_criterion,
    }


def _fmt(x) -> str:
    return "  nan" if x != x else f"{x:.3f}"  # x!=x detects NaN


def print_report(report: dict) -> None:
    cov = report["coverage"]
    print(f"\nCoverage: {cov['total_judged_actions']} judged actions | "
          f"{cov['multi_annotator_actions']} multi-annotator "
          f"({cov['multi_annotator_fraction']:.0%}) inform agreement")
    print("Annotators per action:", cov["annotators_per_action"])

    s = report["score"]
    print(f"\nQuality score (E2 ceiling) over {s['n_units_multi']} multi-annotator items:")
    print(f"  Krippendorff alpha (ordinal) : {_fmt(s['alpha_ordinal'])}")
    print(f"  pairwise exact agreement     : {_fmt(s['pairwise_exact_agreement'])}")
    print(f"  pairwise within-1            : {_fmt(s['pairwise_within_1'])}")
    print(f"  mean abs pairwise diff       : {_fmt(s['mean_abs_pairwise_diff'])}")

    t = report["tkv_overall"]
    print(f"\nTKV (E1 ceiling) over {t['n_units_multi']} multi-annotator criterion-units:")
    print(f"  Krippendorff alpha (nominal) : {_fmt(t['alpha_nominal'])}")
    print(f"  raw pairwise agreement       : {_fmt(t['pairwise_agreement'])}  (inflated by ~83% pass)")

    rs = report["reliability_summary"]
    print(f"\nReliability of the {rs['n_eligible']} criteria (>=5 multi-annotator units):")
    print("  buckets:", rs["buckets"])
    print(f"  reliable (alpha>={rs['reliable_threshold']}): {rs['n_reliable']} criteria "
          f"-> the learnable label space for E1")

    crit = report["tkv_per_criterion"]
    ranked = sorted(
        (c for c in crit.items() if c[1]["n_units_multi"] >= 5),
        key=lambda kv: (kv[1]["alpha_nominal"] if kv[1]["alpha_nominal"] == kv[1]["alpha_nominal"] else -9),
    )
    print(f"\nLeast-reliable criteria (alpha, agreement, n_multi) — bottom 10 of "
          f"{len(ranked)} with >=5 multi-annotator units:")
    for text, m in ranked[:10]:
        print(f"  a={_fmt(m['alpha_nominal'])}  agr={_fmt(m['pairwise_agreement'])}  "
              f"n={m['n_units_multi']:3d}  {text[:60]}")


# --------------------------------------------------------------------------- entrypoint


def compute_agreement_main() -> None:
    parser = argparse.ArgumentParser(description="EgoExo-Fitness E3: inter-annotator agreement "
                                                 "(human ceiling) for TKV and quality score.")
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    meta = load_json(args.annotation_root / "meta_records.json")
    iaj = load_json(args.annotation_root / "interpretable_action_judgement.json")
    actions = parse_judged_actions(iaj, build_record_info(meta))

    report = compute_report(actions)

    output = args.output or args.processed_root / "agreement_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print_report(report)
    print(f"\nFull per-criterion report -> {output}")


if __name__ == "__main__":
    compute_agreement_main()
