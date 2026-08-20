"""Paired LOSO across REHAB24-6 framing arms, with the plan's seed arithmetic.

Stage A's runner cannot do this job for two reasons. It resolves arms as names under
one ``--feature-parent``, and the framing layout puts every arm's features under the
SAME leaf name (``videomae_mean_pool_fc_norm_mean``) inside a per-variant directory --
so arms are given here as explicit ``name=path`` pairs. And Stage A reports one seed;
plan §7.2 asks for three, combined in a specific order.

That order is the part worth stating, because the obvious implementation is wrong.
The independent unit is the held-out SUBJECT, not the sample and not the seed. So each
subject's balanced accuracy is averaged over seeds 42/7/1234 FIRST, and only then are
the nine paired deltas formed and tested. Running ``paired_delta`` once per seed and
averaging the three results is a different statistic; treating 9 subjects x 3 seeds as
27 observations is pseudo-replication and would roughly halve every p-value in the
report by pretending re-running the same data is new evidence.

With n=9 the test is underpowered by construction, which is why ``verdict`` never
returns "no difference": a non-significant result is reported as undetermined, with the
observed range printed next to it so the reader can see how wide the uncertainty is.

Everything else -- folds, validation-subject selection, hyperparameters, threshold
objective -- is Stage A's, reused rather than reimplemented, so the arms differ in
framing and in nothing else.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("REHAB24-6 framing evaluation requires `torch`.") from exc

#: Pre-registered in plan §7.2. Fixed here so no run can quietly add a fourth seed and
#: report the mean of the four that came out best.
DEFAULT_SEEDS = (42, 7, 1234)
#: Plan §8: a practical-effect band, NOT a significance threshold and not the output of
#: a power analysis. It says how big a delta has to be to be worth acting on.
PRACTICAL_EFFECT = 0.02


def parse_arm(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Arm {spec!r} must be given as name=path.")
    name, _, path = spec.partition("=")
    return name.strip(), Path(path.strip())


def parse_pair(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Comparison {spec!r} must be given as candidate:baseline.")
    candidate, _, baseline = spec.partition(":")
    return candidate.strip(), baseline.strip()


def seed_averaged_accuracy(
    folds_by_seed: dict[int, list[dict]],
    key: str = "balanced_accuracy",
    drop_p10: bool = True,
) -> dict[str, float]:
    """Per-subject balanced accuracy, averaged over seeds -- step 2 of §7.2.

    Averaging here rather than after the delta is what keeps the seed from entering the
    test as if it were an independent observation.
    """
    per_subject: dict[str, list[float]] = defaultdict(list)
    for folds in folds_by_seed.values():
        for fold in folds:
            if drop_p10 and fold["n_test"] < MIN_VAL_SUBJECT_SAMPLES:
                continue
            per_subject[fold["test_subject"]].append(float(fold[key]))
    return {subject: float(np.mean(values)) for subject, values in sorted(per_subject.items(), key=lambda kv: int(kv[0]))}


def seed_averaged_strata(
    folds_by_seed: dict[int, list[dict]],
    key: str,
    drop_p10: bool = True,
) -> dict[str, dict[str, float]]:
    """``{stratum: {subject: seed-averaged balanced accuracy}}`` for camera/exercise.

    A cell is kept only when it is present in EVERY seed. ``stratified_metrics`` drops
    a stratum whose fold has fewer than ``MIN_STRATUM_SAMPLES`` of it or only one
    class, and that can differ between seeds -- so without this requirement one
    subject's value would be an average over three seeds and another's over one, and
    the two would sit in the same column looking alike. Dropped cells surface in the
    summary's ``*_excluded`` record rather than vanishing.
    """
    n_seeds = len(folds_by_seed)
    per_stratum: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for folds in folds_by_seed.values():
        for fold in folds:
            if drop_p10 and fold["n_test"] < MIN_VAL_SUBJECT_SAMPLES:
                continue
            for stratum, metrics in fold[key].items():
                per_stratum[stratum][fold["test_subject"]].append(float(metrics["balanced_accuracy"]))
    return {
        stratum: complete
        for stratum, subjects in sorted(per_stratum.items())
        if (
            complete := {
                subject: float(np.mean(values))
                for subject, values in sorted(subjects.items(), key=lambda kv: int(kv[0]))
                if len(values) == n_seeds
            }
        )
    }


def exact_wilcoxon(deltas: Sequence[float]) -> dict | None:
    """Exact paired Wilcoxon, as §7.2 requires -- the normal approximation is not valid
    at n=9. Returns ``None`` when scipy is absent or every delta is zero."""
    values = [float(delta) for delta in deltas]
    if not values or all(delta == 0 for delta in values):
        return None
    try:
        from scipy.stats import wilcoxon
    except ImportError:  # pragma: no cover - scipy is a hard dep of the LOSO runners
        return None
    try:
        stat, p_value = wilcoxon(values, method="exact")
        method = "exact"
    except TypeError:  # older scipy spells it `mode`
        stat, p_value = wilcoxon(values, mode="exact")
        method = "exact"
    return {"stat": float(stat), "p_value": float(p_value), "method": method}


def practical_reading(mean_delta: float, majority_positive: bool, band: float = PRACTICAL_EFFECT) -> str:
    """Plan §8's table, which is keyed on effect SIZE and direction -- never on p.

    §8 row 1 is "Δ >= +0.02 且多數 subjects 同向", with no significance condition
    attached. Folding p into this label would let a p just over 0.05 silently overrule
    a pre-registered practical rule, and the correction §10 forbids is exactly the one
    that would then be tempting: rewriting the plan to match the code. The p-value is
    reported next to this, as its own axis, by ``significance_reading``.
    """
    if abs(mean_delta) < band:
        # Explicitly NOT equivalence: a small point estimate with a wide interval is a
        # small point estimate, and §8 says so.
        return "practically_small_point_estimate"
    consistent = majority_positive if mean_delta > 0 else not majority_positive
    if not consistent:
        return "practical_size_but_direction_inconsistent"
    return "practical_gain" if mean_delta > 0 else "practical_loss"


def significance_reading(p_value: float | None) -> str:
    """The other axis. "undetermined" rather than "no difference" is not hedging: at
    n=9 this design cannot separate a null from an effect too small for it to see."""
    if p_value is None:
        return "not_testable"
    return "significant" if p_value < 0.05 else "undetermined"


def paired_comparison(candidate: dict[str, float], baseline: dict[str, float]) -> dict:
    """The nine paired subject deltas and their exact test -- steps 3 and 4 of §7.2."""
    subjects = sorted(set(candidate) & set(baseline), key=int)
    missing = sorted(set(candidate) ^ set(baseline), key=int)
    if missing:
        raise ValueError(f"Arms were not evaluated on the same subjects; unmatched: {missing}")

    deltas = [candidate[subject] - baseline[subject] for subject in subjects]
    stats = exact_wilcoxon(deltas)
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    majority_positive = sum(delta > 0 for delta in deltas) > len(deltas) / 2
    return {
        "n_subjects": len(subjects),
        "subjects": {
            subject: {"baseline": baseline[subject], "candidate": candidate[subject], "delta": delta}
            for subject, delta in zip(subjects, deltas)
        },
        "baseline_mean": float(np.mean([baseline[s] for s in subjects])) if subjects else 0.0,
        "candidate_mean": float(np.mean([candidate[s] for s in subjects])) if subjects else 0.0,
        "delta": summarize(deltas),
        "delta_range": [float(min(deltas)), float(max(deltas))] if deltas else [0.0, 0.0],
        "n_positive": int(sum(delta > 0 for delta in deltas)),
        "majority_positive": majority_positive,
        "wilcoxon": stats,
        # Two axes, reported side by side, never collapsed into one label (§7.2 + §8).
        "practical_reading": practical_reading(mean_delta, majority_positive),
        "significance": significance_reading(stats["p_value"] if stats else None),
    }


def holm_correct(p_values: dict[str, float]) -> dict[str, dict[str, float | bool]]:
    """Holm-Bonferroni over the SECONDARY comparisons only (§7.3).

    Raw and corrected values are both kept: the plan requires the correction to be
    visible rather than applied silently to a number a reader might quote as raw.
    """
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    corrected: dict[str, dict[str, float | bool]] = {}
    running = 0.0
    for index, (name, raw) in enumerate(ordered):
        running = max(running, min(1.0, raw * (total - index)))
        corrected[name] = {"raw": raw, "holm": running, "significant": running < 0.05}
    return corrected


def arm_provenance(feature_dir: Path) -> dict[str, str]:
    """The provenance stamp one arm's features carry, recorded into the summary.

    ``videomae_materialize`` forwards the raw stamp, so a materialized letterbox dir
    still says ``variant: full_frame_letterbox``. Writing it into the report is what
    makes a swapped ``--arm`` mapping visible after the fact: without it the summary
    names arms only by the label whoever ran the command typed.
    """
    path = next(feature_dir.rglob("*.npz"), None)
    if path is None:
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {key[len("provenance_") :]: str(data[key]) for key in data.files if key.startswith("provenance_")}


def assert_arms_are_distinct(arm_dirs: dict[str, Path]) -> None:
    """Two arms pointed at one directory yields all-zero deltas and no test at all,
    which prints exactly like a null result rather than like the mistake it is."""
    seen: dict[Path, str] = {}
    for arm, path in arm_dirs.items():
        resolved = path.resolve()
        if resolved in seen:
            raise SystemExit(f"Arms `{seen[resolved]}` and `{arm}` point at the same feature dir {path}.")
        seen[resolved] = arm


def run_all_arms(
    arm_dirs: dict[str, Path],
    seeds: Sequence[int],
    labels: dict[str, int],
    subject_samples: dict[str, list[str]],
    ordered_subjects: list[str],
    sample_counts: dict[str, int],
    metadata: dict[str, dict[str, str]],
    config: FoldConfig,
    device: torch.device,
) -> dict[str, dict[int, list[dict]]]:
    results: dict[str, dict[int, list[dict]]] = {}
    for arm, feature_dir in arm_dirs.items():
        if not feature_dir.exists():
            raise SystemExit(f"Missing feature dir for arm `{arm}`: {feature_dir}. Run videomae_materialize first.")
        results[arm] = {}
        for seed in seeds:
            print(f"\n--- arm: {arm} | seed {seed} ---")
            folds = run_arm(
                feature_dir, labels, subject_samples, ordered_subjects, sample_counts, metadata, config, device, seed
            )
            results[arm][seed] = folds
            big = [f for f in folds if f["n_test"] >= MIN_VAL_SUBJECT_SAMPLES]
            print(f"  bal_acc (9 folds, no P10): {np.mean([f['balanced_accuracy'] for f in big]):.3f}")
    return results


def build_summary(
    results: dict[str, dict[int, list[dict]]],
    primary: tuple[str, str],
    secondary: Sequence[tuple[str, str]],
    seeds: Sequence[int],
) -> dict:
    accuracy = {arm: seed_averaged_accuracy(folds_by_seed) for arm, folds_by_seed in results.items()}
    # §7.1: the P10-inclusive 10-fold run is a SENSITIVITY analysis, reported next to
    # the primary rather than instead of it. P10 contributes 16 samples, on which a
    # balanced accuracy is barely defined.
    accuracy_with_p10 = {arm: seed_averaged_accuracy(folds, drop_p10=False) for arm, folds in results.items()}
    supplementary = {
        key: {arm: seed_averaged_accuracy(folds, key=key) for arm, folds in results.items()}
        # §7.1 asks for these alongside the primary endpoint; they do not replace it,
        # and accuracy in particular is left out because class balance moves it.
        for key in ("macro_f1", "recall", "specificity")
    }

    arms = {
        arm: {
            "seed_averaged_by_subject": subjects,
            "balanced_accuracy_no_p10": summarize(list(subjects.values())),
            "balanced_accuracy_with_p10_sensitivity": summarize(list(accuracy_with_p10[arm].values())),
            "above_chance": float(np.mean(list(subjects.values())) - 0.5),
            **{f"{key}_no_p10": summarize(list(values[arm].values())) for key, values in supplementary.items()},
            "per_seed_balanced_accuracy": {
                str(seed): summarize(
                    [f["balanced_accuracy"] for f in folds if f["n_test"] >= MIN_VAL_SUBJECT_SAMPLES]
                )
                for seed, folds in results[arm].items()
            },
        }
        for arm, subjects in accuracy.items()
    }

    def compare(candidate: str, baseline: str) -> dict:
        comparison = paired_comparison(accuracy[candidate], accuracy[baseline])
        comparison["with_p10_sensitivity"] = paired_comparison(
            accuracy_with_p10[candidate], accuracy_with_p10[baseline]
        )
        for key in ("by_camera", "by_exercise"):
            cand_strata = seed_averaged_strata(results[candidate], key)
            base_strata = seed_averaged_strata(results[baseline], key)
            shared = sorted(set(cand_strata) & set(base_strata))
            comparison[key] = {
                stratum: paired_comparison(cand_strata[stratum], base_strata[stratum])
                for stratum in shared
                if set(cand_strata[stratum]) == set(base_strata[stratum])
            }
            # §7.4 wants all six exercises reported. A stratum vanishes when a fold has
            # fewer than MIN_STRATUM_SAMPLES of it or only one class, and an absent row
            # reads as "not applicable" when it means "excluded" -- so name it.
            comparison[f"{key}_excluded"] = {
                stratum: sorted(set(cand_strata.get(stratum, {})) ^ set(base_strata.get(stratum, {})), key=int)
                for stratum in sorted(set(cand_strata) | set(base_strata))
                if stratum not in comparison[key]
            }
        return comparison

    primary_result = compare(*primary)
    secondary_results = {f"{cand}-{base}": compare(cand, base) for cand, base in secondary}

    raw_p = {
        name: result["wilcoxon"]["p_value"]
        for name, result in secondary_results.items()
        if result.get("wilcoxon")
    }

    return {
        "seeds": list(seeds),
        "practical_effect_band": PRACTICAL_EFFECT,
        "arms": arms,
        "primary": {"comparison": f"{primary[0]}-{primary[1]}", **primary_result},
        "secondary": secondary_results,
        "secondary_holm": holm_correct(raw_p) if raw_p else {},
    }


def print_comparison(name: str, result: dict, indent: str = "  ") -> None:
    stats = result.get("wilcoxon")
    p_text = f", exact p={stats['p_value']:.4f}" if stats else ", p=n/a"
    print(
        f"{indent}{name}: {result['baseline_mean']:.4f} -> {result['candidate_mean']:.4f}  "
        f"d={result['delta']['mean']:+.4f} +/- {result['delta']['std']:.4f}  "
        f"[{result['delta_range'][0]:+.3f}, {result['delta_range'][1]:+.3f}]  "
        f"({result['n_positive']}/{result['n_subjects']} subjects positive{p_text})"
    )
    # ASCII "sec." rather than the section sign: this console is cp950 and mangles it.
    print(
        f"{indent}  practical (plan sec.8): {result['practical_reading']}"
        f"   significance (sec.7.2): {result['significance']}"
    )


def print_summary(summary: dict) -> None:
    print(f"\n=== framing arms (seeds {summary['seeds']}, 9 subjects, P10 excluded) ===")
    for arm, values in summary["arms"].items():
        accuracy = values["balanced_accuracy_no_p10"]
        sensitivity = values["balanced_accuracy_with_p10_sensitivity"]
        print(
            f"  {arm:<26} bal_acc {accuracy['mean']:.4f} +/- {accuracy['std']:.4f}   "
            f"above chance {values['above_chance']:+.4f}   "
            f"macro-F1 {values['macro_f1_no_p10']['mean']:.4f}   "
            f"recall {values['recall_no_p10']['mean']:.4f}   spec {values['specificity_no_p10']['mean']:.4f}   "
            f"[10-fold sensitivity {sensitivity['mean']:.4f}]"
        )

    print("\n=== primary (pre-registered, one test) ===")
    print_comparison(summary["primary"]["comparison"], summary["primary"])
    print_comparison("  ^ 10-fold sensitivity (P10 included)", summary["primary"]["with_p10_sensitivity"])
    for key in ("by_camera", "by_exercise"):
        print(f"\n  --- primary {key} (mechanism check, not a substitute endpoint) ---")
        for stratum, result in summary["primary"].get(key, {}).items():
            print_comparison(stratum, result, indent="    ")
        for stratum, subjects in summary["primary"].get(f"{key}_excluded", {}).items():
            print(f"    {stratum}: EXCLUDED (unmatched subjects {subjects}; stratum too small or single-class)")

    if summary["secondary"]:
        print("\n=== secondary / exploratory (Holm-corrected) ===")
        for name, result in summary["secondary"].items():
            print_comparison(name, result)
        for name, values in summary["secondary_holm"].items():
            print(f"    {name}: raw p={values['raw']:.4f}, Holm p={values['holm']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired LOSO across REHAB24-6 VideoMAE framing arms.")
    parser.add_argument("--arm", action="append", required=True, help="name=path/to/feature_dir, repeatable.")
    parser.add_argument("--primary", required=True, help="candidate:baseline, the one pre-registered test.")
    parser.add_argument("--secondary", nargs="*", default=[], help="candidate:baseline pairs, Holm-corrected.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--labels", type=Path, default=DEFAULT_PROCESSED_ROOT / "labels" / "correctness.json")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or auto.")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_PROCESSED_ROOT / "videomae_framing")
    args = parser.parse_args()

    arm_dirs = dict(parse_arm(spec) for spec in args.arm)
    assert_arms_are_distinct(arm_dirs)
    primary = parse_pair(args.primary)
    secondary = [parse_pair(spec) for spec in args.secondary]
    for candidate, baseline in [primary, *secondary]:
        for name in (candidate, baseline):
            if name not in arm_dirs:
                raise SystemExit(f"Comparison references unknown arm `{name}`; declared arms: {sorted(arm_dirs)}")

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    config = FoldConfig()  # identical to Stage A and to the committed LOSO baselines

    labels = {key: int(value) for key, value in json.load(args.labels.open()).items()}
    metadata = load_metadata(args.manifest)
    subject_samples = subjects_to_samples(args.manifest)
    sample_counts = {person: len(ids) for person, ids in subject_samples.items()}
    ordered_subjects = sorted(subject_samples, key=int)

    print(f"Framing LOSO on {device} | arms {sorted(arm_dirs)} | seeds {args.seeds}")
    results = run_all_arms(
        arm_dirs, args.seeds, labels, subject_samples, ordered_subjects, sample_counts, metadata, config, device
    )

    for seed in args.seeds:
        payload = {
            "seed": seed,
            "config": vars(config),
            "arms": {arm: folds_by_seed[seed] for arm, folds_by_seed in results.items()},
        }
        path = Path(f"{args.output_prefix}_seed{seed}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"Saved per-seed folds to {path}")

    summary = build_summary(results, primary, secondary, args.seeds)
    summary["arm_dirs"] = {arm: str(path) for arm, path in arm_dirs.items()}
    summary["arm_provenance"] = {arm: arm_provenance(path) for arm, path in arm_dirs.items()}
    summary["config"] = vars(config)

    print("\n=== arm provenance (what each label's features actually are) ===")
    for arm, stamp in summary["arm_provenance"].items():
        print(f"  {arm:<26} variant={stamp.get('variant', 'full_frame (unstamped)')} "
              f"pooling={stamp.get('token_pooling')} agg={stamp.get('clip_aggregation')} "
              f"transformers={stamp.get('transformers_version')}")
    print_summary(summary)

    summary_path = Path(f"{args.output_prefix}_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"\nSaved framing summary to {summary_path}")


if __name__ == "__main__":
    main()
