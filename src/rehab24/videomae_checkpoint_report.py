"""REHAB24-6 checkpoint comparison report: SSv2-finetuned vs Kinetics-finetuned VideoMAE.

Plan: ``notes/rehab24_videomae_ssv2_checkpoint_validation_plan.md``. This module reads
the summaries that the existing framing / identity-control / position-control commands
write for each checkpoint and evaluates the plan's rule tables exactly as written. It
trains nothing and computes no new statistic beyond paired subject deltas, their exact
Wilcoxon test and a subject bootstrap; every number it reports traces to a summary file.

Sections, each independently ``available`` so the report can be run while the later
arms are still extracting:

* ``primary``  -- paired LOSO balanced-accuracy delta, ssv2 - kin, ``full_frame_letterbox``
* ``s1``       -- paired within-session AUC delta on the same arm
* ``s2``       -- position probe rho and the k = 16 residualization share per checkpoint
* ``s3``       -- the SSv2 ``background_only`` bound, cross-subject and within-session
* ``s4``       -- camera / exercise strata copied from the framing report
* ``s5``       -- P10-inclusive sensitivity for the primary and S1
* ``reading``  -- which row of the secondary reading table the pattern hits
* ``gates``    -- G4 reproduction checks on the Kinetics side

Statistical helpers are imported from the identity-control module, never re-implemented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from src.rehab24.dataset import DEFAULT_PROCESSED_ROOT
from src.rehab24.videomae_identity_control import (
    ALPHA,
    bootstrap_interval,
    exact_wilcoxon_vs,
    holm_correct,
    preregistered_verdict,
)

PRACTICAL_BAND = 0.02
MIN_SUBJECTS_POSITIVE = 6
BACKGROUND_BA_THRESHOLD = 0.55
BACKGROUND_AUC_MARGIN = 0.15
DEFAULT_BOOTSTRAP = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260908

# G4 targets, to 4 decimals, from the framing, identity and position notes.
KIN_LETTERBOX_BA = 0.6612
KIN_WITHIN_SESSION_AUC = 0.8741
KIN_PROBE_MEDIAN = 0.4324

DEFAULT_PATHS = {
    "framing_summary": DEFAULT_PROCESSED_ROOT / "videomae_checkpoint_summary.json",
    "kin_within": DEFAULT_PROCESSED_ROOT / "videomae_identity_control" / "within_session_summary.json",
    "ssv2_within": DEFAULT_PROCESSED_ROOT / "videomae_identity_control_ssv2" / "within_session_summary.json",
    "ssv2_background_within": DEFAULT_PROCESSED_ROOT
    / "videomae_identity_control_ssv2_background_only"
    / "within_session_summary.json",
    "kin_probe_k0": DEFAULT_PROCESSED_ROOT / "videomae_position_control" / "probe_summary_k0.json",
    "kin_probe_k16": DEFAULT_PROCESSED_ROOT / "videomae_position_control" / "probe_summary_k16.json",
    "kin_within_k16": DEFAULT_PROCESSED_ROOT / "videomae_position_control" / "within_session_summary_k16.json",
    "ssv2_probe_k0": DEFAULT_PROCESSED_ROOT / "videomae_position_control_ssv2" / "probe_summary_k0.json",
    "ssv2_probe_k16": DEFAULT_PROCESSED_ROOT / "videomae_position_control_ssv2" / "probe_summary_k16.json",
    "ssv2_within_k0": DEFAULT_PROCESSED_ROOT / "videomae_position_control_ssv2" / "within_session_summary_k0.json",
    "ssv2_within_k16": DEFAULT_PROCESSED_ROOT / "videomae_position_control_ssv2" / "within_session_summary_k16.json",
}
DEFAULT_OUTPUT = DEFAULT_PROCESSED_ROOT / "videomae_checkpoint_report.json"


# --------------------------------------------------------------------------- #
# small pure helpers                                                           #
# --------------------------------------------------------------------------- #


def load_json(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    return json.load(Path(path).open(encoding="utf-8"))


def unavailable(reason: str) -> dict:
    return {"available": False, "reason": reason}


def paired_subject_delta(
    candidate: dict[str, float],
    baseline: dict[str, float],
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict:
    """Nine paired deltas, exact two-sided Wilcoxon, subject bootstrap.

    Subjects are matched by id; a subject present on one side only is dropped and
    listed, never imputed. Two-sided p is ``2 * min(greater, less)`` from the exact
    one-sided tests, capped at 1, the same construction the position control used.
    """
    shared = sorted(set(candidate) & set(baseline), key=int)
    dropped = sorted((set(candidate) | set(baseline)) - set(shared), key=int)
    deltas = [float(candidate[s]) - float(baseline[s]) for s in shared]
    greater = exact_wilcoxon_vs(deltas, 0.0)
    less = exact_wilcoxon_vs([-d for d in deltas], 0.0)
    two_sided = min(1.0, 2 * min(greater["p_value"], less["p_value"])) if greater and less else None
    return {
        "n_subjects": len(shared),
        "dropped_subjects": dropped,
        "baseline_mean": float(np.mean([float(baseline[s]) for s in shared])) if shared else None,
        "candidate_mean": float(np.mean([float(candidate[s]) for s in shared])) if shared else None,
        "per_subject_delta": {s: d for s, d in zip(shared, deltas)},
        "mean_delta": float(np.mean(deltas)) if deltas else None,
        "sd_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
        "n_positive": int(sum(d > 0 for d in deltas)),
        "wilcoxon_greater": greater,
        "wilcoxon_less": less,
        "two_sided_p": two_sided,
        "bootstrap_95ci_over_subjects": bootstrap_interval(deltas, n_bootstrap, bootstrap_seed) if deltas else None,
    }


def primary_rule(delta: dict) -> dict:
    """Plan 'Primary comparison' rule table, three rows, evaluated as written."""
    mean = delta["mean_delta"]
    p = delta["two_sided_p"]
    if mean is None or p is None:
        return {"row": "undetermined", "reason": "no deltas or all-zero deltas"}
    adopt = mean >= PRACTICAL_BAND and delta["n_positive"] >= MIN_SUBJECTS_POSITIVE and p < ALPHA
    worse = mean <= -PRACTICAL_BAND and p < ALPHA
    conditions = {
        "mean_delta_at_least_+0.02": mean >= PRACTICAL_BAND,
        "at_least_6_of_9_subjects_positive": delta["n_positive"] >= MIN_SUBJECTS_POSITIVE,
        "two_sided_p_below_0.05": p < ALPHA,
        "mean_delta_at_most_-0.02": mean <= -PRACTICAL_BAND,
    }
    if adopt:
        row, reading = "ssv2_adopted", "SSv2 scores higher cross-subject; SSv2 becomes the REHAB24-6 VideoMAE checkpoint"
    elif worse:
        row, reading = "ssv2_worse", "SSv2 scores lower cross-subject; Kinetics remains"
    else:
        row, reading = "undetermined", "undetermined at n = 9; Kinetics remains the quoted checkpoint; no equivalence claim"
    return {"row": row, "reading": reading, "conditions": conditions}


def per_subject_balanced_accuracy(framing_summary: dict, arm: str) -> dict[str, float] | None:
    arm_block = framing_summary.get("arms", {}).get(arm)
    if not arm_block or "seed_averaged_by_subject" not in arm_block:
        return None
    return {str(k): float(v) for k, v in arm_block["seed_averaged_by_subject"].items()}


def per_subject_auc(within_summary: dict, block: str = "primary") -> dict[str, float] | None:
    section = within_summary.get(block) if block == "primary" else within_summary.get("secondary", {}).get(block)
    if not section or "per_subject_auc" not in section:
        return None
    return {str(k): float(v) for k, v in section["per_subject_auc"].items()}


def position_share(auc_k0: float, auc_k16: float) -> float:
    """(AUC_k0 - AUC_k16) / (AUC_k0 - 0.5); the position-control definition."""
    return (auc_k0 - auc_k16) / (auc_k0 - 0.5)


def probe_summary(probe: dict) -> dict:
    values = [float(v) for v in probe["per_subject"].values()]
    return {
        "median": float(np.median(values)),
        "mean": float(probe.get("subject_macro_mean", np.mean(values))),
        "n_subjects": len(values),
        "n_positive": int(sum(v > 0 for v in values)),
        "null_median_95": probe.get("null", {}).get("median_null_95_interval"),
        "p_value_median": probe.get("null", {}).get("p_value_median"),
        "p_value_mean": probe.get("null", {}).get("p_value_mean"),
        "median_inside_null_95_interval": probe.get("median_inside_null_95_interval"),
    }


def matches_to_4dp(value: float | None, target: float) -> bool | None:
    return None if value is None else round(float(value), 4) == round(target, 4)


# --------------------------------------------------------------------------- #
# sections                                                                     #
# --------------------------------------------------------------------------- #


def primary_section(framing: dict | None, n_bootstrap: int, seed: int) -> dict:
    if framing is None:
        return unavailable("framing summary missing")
    ssv2, kin = per_subject_balanced_accuracy(framing, "ssv2"), per_subject_balanced_accuracy(framing, "kin")
    if ssv2 is None or kin is None:
        return unavailable("framing summary lacks arms.ssv2 / arms.kin seed_averaged_by_subject")
    delta = paired_subject_delta(ssv2, kin, n_bootstrap, seed)
    return {
        "available": True,
        "comparison": "ssv2 - kin, full_frame_letterbox, LOSO balanced accuracy, P1-P9",
        "framing_report_primary": framing.get("primary", {}).get("comparison"),
        **delta,
        "rule": primary_rule(delta),
    }


def s1_section(kin: dict | None, ssv2: dict | None, n_bootstrap: int, seed: int) -> dict:
    if kin is None or ssv2 is None:
        return unavailable("a within-session summary is missing")
    kin_auc, ssv2_auc = per_subject_auc(kin), per_subject_auc(ssv2)
    if kin_auc is None or ssv2_auc is None:
        return unavailable("within-session summary lacks primary.per_subject_auc")
    delta = paired_subject_delta(ssv2_auc, kin_auc, n_bootstrap, seed)
    arms = {}
    for name, summary in (("kin", kin), ("ssv2", ssv2)):
        stat = summary["primary"]
        arms[name] = {
            "mean": stat["mean"],
            "sd": stat.get("sd"),
            "n_subjects_above_chance": stat["n_subjects_above_chance"],
            "permutation_p": stat["permutation"]["p_value"],
            "identity_rule": preregistered_verdict(stat, stat["permutation"]["p_value"]),
        }
    return {"available": True, "comparison": "ssv2 - kin within-session AUC", **delta, "arms": arms}


def s2_section(files: dict[str, dict | None]) -> dict:
    out: dict = {"available": True, "checkpoints": {}}
    for name in ("kin", "ssv2"):
        probe0, probe16 = files.get(f"{name}_probe_k0"), files.get(f"{name}_probe_k16")
        within0 = files.get("kin_within") if name == "kin" else files.get("ssv2_within_k0")
        within16 = files.get(f"{name}_within_k16")
        block: dict = {}
        block["probe_k0"] = probe_summary(probe0) if probe0 else unavailable("probe k0 missing")
        block["probe_k16"] = probe_summary(probe16) if probe16 else unavailable("probe k16 missing")
        if within0 and within16:
            auc0, auc16 = within0["primary"]["mean"], within16["primary"]["mean"]
            block["auc_k0"], block["auc_k16"] = auc0, auc16
            block["share"] = position_share(auc0, auc16)
            block["share_is_floor"] = (
                not probe16.get("median_inside_null_95_interval", False) if probe16 else None
            )
        else:
            block["share"] = None
            block["share_reason"] = "within-session summary at k0 or k16 missing"
        out["checkpoints"][name] = block
    kin, ssv2 = out["checkpoints"]["kin"], out["checkpoints"]["ssv2"]
    if kin["probe_k0"].get("available", True) and ssv2["probe_k0"].get("available", True):
        out["probe_median_ssv2_minus_kin"] = ssv2["probe_k0"]["median"] - kin["probe_k0"]["median"]
        out["ssv2_probe_below_kin"] = ssv2["probe_k0"]["median"] < kin["probe_k0"]["median"]
    if kin.get("share") is not None and ssv2.get("share") is not None:
        out["share_ssv2_minus_kin"] = ssv2["share"] - kin["share"]
        out["ssv2_share_at_or_below_kin"] = ssv2["share"] <= kin["share"]
    out["note"] = "cross-checkpoint probe and share differences are directions, not tests; no p-value is attached"
    if not (ssv2["probe_k0"].get("available", True) and ssv2.get("share") is not None):
        out["available"] = False
        out["reason"] = "SSv2 probe or share not yet computable"
    return out


def s3_section(framing: dict | None, within: dict | None, n_bootstrap: int, seed: int) -> dict:
    out: dict = {"available": True}
    if framing is not None and per_subject_balanced_accuracy(framing, "ssv2_background_only"):
        ba = per_subject_balanced_accuracy(framing, "ssv2_background_only")
        values = [ba[s] for s in sorted(ba, key=int)]
        greater = exact_wilcoxon_vs(values, 0.5)
        mean = float(np.mean(values))
        out["balanced_accuracy"] = {
            "mean": mean,
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "per_subject": ba,
            "n_subjects_above_chance": int(sum(v > 0.5 for v in values)),
            "wilcoxon_vs_chance_greater": greater,
            "bootstrap_95ci_over_subjects": bootstrap_interval(values, n_bootstrap, seed),
            "kin_reference": 0.5074,
            "informative": bool(mean >= BACKGROUND_BA_THRESHOLD and greater and greater["p_value"] < ALPHA),
        }
    else:
        out["balanced_accuracy"] = unavailable("framing summary lacks arms.ssv2_background_only")
    if within is not None and "primary" in within:
        stat = within["primary"]
        p = stat["permutation"]["p_value"]
        out["within_session"] = {
            "mean": stat["mean"],
            "sd": stat.get("sd"),
            "n_subjects_above_chance": stat["n_subjects_above_chance"],
            "permutation_p": p,
            "bootstrap_95ci_over_subjects": stat.get("bootstrap_95ci_over_subjects"),
            "kin_reference": 0.5357,
            "informative": bool(stat["mean"] - 0.5 > BACKGROUND_AUC_MARGIN and p < ALPHA),
        }
    else:
        out["within_session"] = unavailable("background_only within-session summary missing")
    ba_ok = out["balanced_accuracy"].get("available", True)
    ws_ok = out["within_session"].get("available", True)
    out["informative"] = bool(
        (ba_ok and out["balanced_accuracy"]["informative"]) or (ws_ok and out["within_session"]["informative"])
    )
    if not (ba_ok and ws_ok):
        out["available"] = False
        out["reason"] = "one or both background_only readouts missing; `informative` covers what is present"
    return out


def s4_section(framing: dict | None) -> dict:
    if framing is None or "primary" not in framing:
        return unavailable("framing summary missing")
    primary = framing["primary"]
    strata = {}
    for kind in ("by_camera", "by_exercise"):
        for name, block in primary.get(kind, {}).items():
            strata[f"{kind}:{name}"] = {
                "baseline_mean": block.get("baseline_mean"),
                "candidate_mean": block.get("candidate_mean"),
                "mean_delta": block.get("delta", {}).get("mean"),
                "n_positive": block.get("n_positive"),
                "n_subjects": block.get("n_subjects"),
                "wilcoxon_p": (block.get("wilcoxon") or {}).get("p_value"),
            }
    return {"available": True, "reported_only": True, "strata": strata}


def s5_section(framing: dict | None, kin_within: dict | None, ssv2_within: dict | None, n_bootstrap: int, seed: int) -> dict:
    out: dict = {"available": True}
    if framing is not None:
        arms = framing.get("arms", {})
        with_p10 = {
            name: arms.get(name, {}).get("balanced_accuracy_with_p10_sensitivity", {}).get("mean") for name in ("kin", "ssv2")
        }
        out["balanced_accuracy_with_p10"] = {
            **with_p10,
            "mean_delta": (with_p10["ssv2"] - with_p10["kin"]) if None not in with_p10.values() else None,
            "note": "means only; the framing summary carries no per-subject P10-inclusive table",
        }
    else:
        out["balanced_accuracy_with_p10"] = unavailable("framing summary missing")
    if kin_within and ssv2_within:
        kin_auc = per_subject_auc(kin_within, "p10_inclusive_sensitivity")
        ssv2_auc = per_subject_auc(ssv2_within, "p10_inclusive_sensitivity")
        if kin_auc and ssv2_auc:
            out["within_session_with_p10"] = paired_subject_delta(ssv2_auc, kin_auc, n_bootstrap, seed)
        else:
            out["within_session_with_p10"] = unavailable("no p10_inclusive_sensitivity block")
    else:
        out["within_session_with_p10"] = unavailable("a within-session summary is missing")
    return out


def reading_section(primary: dict, s1: dict, s2: dict, s3: dict) -> dict:
    """The plan's 'Reading the secondaries' table, one boolean per row."""
    s1_up = bool(s1.get("available") and s1["mean_delta"] > 0 and s1["two_sided_p"] < ALPHA)
    s1_down = bool(s1.get("available") and s1["mean_delta"] < 0 and s1["two_sided_p"] < ALPHA)
    s1_undetermined = bool(s1.get("available") and not (s1_up or s1_down))
    probe_below = s2.get("ssv2_probe_below_kin")
    share_ok = s2.get("ssv2_share_at_or_below_kin")
    s3_informative = bool(s3.get("informative"))
    primary_undetermined = primary.get("rule", {}).get("row") == "undetermined"
    h2_pattern = bool(s1_up and probe_below and share_ok and not s3_informative)
    h3_pattern = bool(s1_up and (probe_below is False or share_ok is False or s3_informative))
    h1_pattern = bool(s1_undetermined and primary_undetermined and not s3_informative)
    return {
        "inputs": {
            "s1_delta_auc_positive_p<0.05": s1_up,
            "s1_delta_auc_negative_p<0.05": s1_down,
            "s1_undetermined": s1_undetermined,
            "ssv2_probe_below_kin": probe_below,
            "ssv2_share_at_or_below_kin": share_ok,
            "s3_informative": s3_informative,
            "primary_undetermined": primary_undetermined,
        },
        "rows": {
            "H2_motion_centric": h2_pattern,
            "H3_drift_reading": h3_pattern,
            "H1_checkpoint_general": h1_pattern,
            "S3_checkpoint_specific_non_person_path": s3_informative,
            "S1_ssv2_ranks_worse_within_recording": s1_down,
        },
        "fusion_gate_closed_for_ssv2": bool(h3_pattern or s3_informative),
        "note": "H2 and H3 are separated by the probe direction, the share and the background bound, per the plan",
    }


def gates_section(framing: dict | None, kin_within: dict | None, kin_probe: dict | None) -> dict:
    kin_ba = None
    if framing is not None:
        kin_ba = framing.get("arms", {}).get("kin", {}).get("balanced_accuracy_no_p10", {}).get("mean")
    kin_auc = kin_within["primary"]["mean"] if kin_within else None
    kin_probe_median = float(np.median(list(kin_probe["per_subject"].values()))) if kin_probe else None
    checks = {
        "kin_letterbox_ba_is_0.6612": matches_to_4dp(kin_ba, KIN_LETTERBOX_BA),
        "kin_within_session_auc_is_0.8741": matches_to_4dp(kin_auc, KIN_WITHIN_SESSION_AUC),
        "kin_probe_median_is_0.4324": matches_to_4dp(kin_probe_median, KIN_PROBE_MEDIAN),
    }
    return {
        "G4_reproduction": {
            "values": {"kin_ba": kin_ba, "kin_auc": kin_auc, "kin_probe_median": kin_probe_median},
            "checks": checks,
            "pass": all(v is True for v in checks.values()),
            "all_present": all(v is not None for v in checks.values()),
        }
    }


def build_report(paths: dict[str, Path], n_bootstrap: int, bootstrap_seed: int) -> dict:
    files = {key: load_json(path) for key, path in paths.items()}
    framing = files["framing_summary"]
    primary = primary_section(framing, n_bootstrap, bootstrap_seed)
    s1 = s1_section(files["kin_within"], files["ssv2_within"], n_bootstrap, bootstrap_seed)
    s2 = s2_section(files)
    s3 = s3_section(framing, files["ssv2_background_within"], n_bootstrap, bootstrap_seed)
    s4 = s4_section(framing)
    s5 = s5_section(framing, files["kin_within"], files["ssv2_within"], n_bootstrap, bootstrap_seed)
    raw_p = {}
    if s1.get("available") and s1["two_sided_p"] is not None:
        raw_p["s1_delta_auc_two_sided"] = s1["two_sided_p"]
    ba = s3.get("balanced_accuracy", {})
    if ba.get("available", True) and ba.get("wilcoxon_vs_chance_greater"):
        raw_p["s3_background_ba_vs_chance"] = ba["wilcoxon_vs_chance_greater"]["p_value"]
    ws = s3.get("within_session", {})
    if ws.get("available", True) and "permutation_p" in ws:
        raw_p["s3_background_within_session_permutation"] = ws["permutation_p"]
    return {
        "plan": "notes/rehab24_videomae_ssv2_checkpoint_validation_plan.md",
        "inputs": {key: str(path) for key, path in paths.items()},
        "inputs_present": {key: files[key] is not None for key in paths},
        "bootstrap": {"n_resamples": n_bootstrap, "seed": bootstrap_seed},
        "gates": gates_section(framing, files["kin_within"], files["kin_probe_k0"]),
        "primary": primary,
        "s1_within_session": s1,
        "s2_position": s2,
        "s3_background_only": s3,
        "s4_strata": s4,
        "s5_p10_sensitivity": s5,
        "secondary_holm": holm_correct(raw_p) if raw_p else {},
        "reading": reading_section(primary, s1, s2, s3),
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def print_report(report: dict) -> None:
    gate = report["gates"]["G4_reproduction"]
    print(f"G4 reproduction: pass={gate['pass']} all_present={gate['all_present']} {gate['values']}")
    primary = report["primary"]
    if primary.get("available"):
        print(
            f"PRIMARY dBA ssv2-kin = {primary['mean_delta']:+.4f} +/- {primary['sd_delta']:.4f} "
            f"({primary['n_positive']}/{primary['n_subjects']} positive), two-sided p = {primary['two_sided_p']:.4f}, "
            f"95% CI {primary['bootstrap_95ci_over_subjects']} -> {primary['rule']['row']}"
        )
    else:
        print(f"PRIMARY unavailable: {primary['reason']}")
    s1 = report["s1_within_session"]
    if s1.get("available"):
        print(
            f"S1 dAUC ssv2-kin = {s1['mean_delta']:+.4f} ({s1['n_positive']}/{s1['n_subjects']} positive), "
            f"two-sided p = {s1['two_sided_p']:.4f}, 95% CI {s1['bootstrap_95ci_over_subjects']}"
        )
    else:
        print(f"S1 unavailable: {s1['reason']}")
    s2 = report["s2_position"]
    for name, block in s2["checkpoints"].items():
        probe = block["probe_k0"]
        median = probe.get("median")
        share = block.get("share")
        print(
            f"S2 {name}: probe median {median if median is None else round(median, 4)}, "
            f"share {share if share is None else round(share, 4)}"
        )
    s3 = report["s3_background_only"]
    print(f"S3 background_only informative={s3['informative']} (available={s3['available']})")
    print("reading rows:", {k: v for k, v in report["reading"]["rows"].items() if v})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for key, default in DEFAULT_PATHS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=Path, default=default)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = {key: getattr(args, key) for key in DEFAULT_PATHS}
    report = build_report(paths, args.bootstrap, args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report)
    print(f"wrote {args.output}")
