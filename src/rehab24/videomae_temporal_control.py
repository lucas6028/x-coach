"""REHAB24-6 VideoMAE temporal-order control (pre-registered).

Plan: ``notes/rehab24_videomae_temporal_shuffle_validation_plan.md``. The plan asks how
much of the within-session ranking (0.8741 on ``full_frame_letterbox``) survives when
the 16 frames of every clip are reordered before they reach the frozen backbone. Every
carrier of recording position the earlier controls chased is static within a clip, so
signal that dies under reordering cannot come from any of them.

Five arms, all extracted with ``videomae_features.py --temporal <arm>`` on the same
pixels and the same clip starts as the stored baseline:

* ``frame_shuffle`` (primary): uniform random order per clip.
* ``frame_reverse``: reversed; local continuity intact, direction destroyed.
* ``tubelet_shuffle``: the 8 consecutive frame pairs in random order, pairs intact.
* ``rep_static_frame``: the clip's 9th frame repeated 16 times.
* ``frame_identity``: natural order through the same code path (reproduction gate G3).

This module holds the gates (G1 frame pairing, G2 permutation validity, G4 OOF
integrity), the LOSO prediction driver, the within-session analysis (reused from the
identity control, unchanged) and the paired nine-subject inference the plan registers.
Nothing statistical is rewritten: ``repetition_scores``, ``build_sessions``,
``observed_statistic``, ``permutation_null``, ``bootstrap_interval`` and
``exact_wilcoxon_vs`` are the identity control's; the position-balanced secondary is the
position control's.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from src.rehab24.dataset import DEFAULT_PROCESSED_ROOT
from src.rehab24.loso_cross_validation import FoldConfig, MIN_VAL_SUBJECT_SAMPLES, subjects_to_samples
from src.rehab24.videomae_features import STATIC_FRAME_INDEX, TEMPORAL_ARMS, TUBELET_SIZE, temporal_permutation
from src.rehab24.videomae_identity_control import (
    ALPHA,
    DECISION_AUC,
    SEEDS,
    analyze as within_session_analysis,
    audit_oof,
    bootstrap_interval,
    build_sessions,
    exact_wilcoxon_vs,
    manifest_index,
    observed_statistic,
    print_analysis,
    read_oof,
    repetition_scores,
    write_oof,
)
from src.rehab24.videomae_identity_control import DEFAULT_OUTPUT_DIR as BASELINE_OOF_DIR
from src.rehab24.videomae_identity_control import oof_path as baseline_oof_path
from src.rehab24.videomae_position_control import (
    DEFAULT_FRAMING_SUMMARY,
    build_pair_sessions,
    folds_path,
    interleaved_mask,
    loso_balanced_accuracy,
    oof_path,
    paired_delta_vs_framing,
    position_balanced_auc,
    position_rule_concordance,
    subset_auc,
)
from src.rehab24.videomae_stage_a import load_metadata, run_arm

# --------------------------------------------------------------------------- #
# registered constants                                                         #
# --------------------------------------------------------------------------- #

BASELINE_NAME = "full_frame_letterbox"
PRIMARY_ARM = "frame_shuffle"
FEATURE_DIR_NAME = "videomae_mean_pool_fc_norm_mean"

DEFAULT_OUTPUT_DIR = DEFAULT_PROCESSED_ROOT / "videomae_temporal_control"
DEFAULT_RAW_PARENT = DEFAULT_PROCESSED_ROOT / "videomae_raw_temporal"
DEFAULT_FEATURE_PARENT = DEFAULT_PROCESSED_ROOT / "videomae_temporal"
DEFAULT_BASELINE_RAW_DIR = DEFAULT_PROCESSED_ROOT / "videomae_raw_full_frame_letterbox"

#: Plan Primary comparison: permutation and bootstrap seed.
DEFAULT_PERMUTATION_SEED = 20260911
DEFAULT_PERMUTATIONS = 10_000
DEFAULT_BOOTSTRAP = 10_000

#: The stored baseline the arms are paired against (identity-control note) and the
#: framing headline (framing note). G3 requires both to four decimals.
REFERENCE_WITHIN_SESSION_AUC = 0.8741
FRAMING_BASELINE_BA = 0.6612
REPRODUCTION_TOLERANCE = 5e-5

#: Plan gates G1/G2.
EXPECTED_BUNDLES = 2144
CLIP_LENGTH = 16
EXPECTED_MEAN_DISPLACEMENT = (CLIP_LENGTH**2 - 1) / (3 * CLIP_LENGTH)  # 5.3125
DISPLACEMENT_TOLERANCE = 0.15
RESEED_SAMPLE = 32

#: Plan reading table.
ORDER_SHARE_BOUNDARY = 0.10
REVERSE_DROP_BOUNDARY = 0.05
MIN_SUBJECTS_DROP = 7
MIN_SUBJECTS_STATIC_GE_SHUFFLE = 6


def temporal_feature_dir(feature_parent: Path, arm: str) -> Path:
    return feature_parent / arm / FEATURE_DIR_NAME


def summary_path(output_dir: Path, arm: str) -> Path:
    return output_dir / f"within_session_summary_{arm}.json"


def null_path(output_dir: Path, arm: str) -> Path:
    return output_dir / f"permutation_null_{arm}.npz"


# --------------------------------------------------------------------------- #
# G1 / G2: raw-bundle gates                                                    #
# --------------------------------------------------------------------------- #

PAIRING_KEYS = ("clip_starts", "first_frame", "last_frame", "total_frames")
#: Provenance keys a temporal arm may legitimately differ from the baseline in.
TEMPORAL_PROVENANCE_KEYS = ("temporal_transform", "temporal_seed_tag")
FEATURE_KEY = "clip_features_mean_pool_fc_norm"


def read_provenance(data) -> dict[str, str]:
    return {key[len("provenance_"):]: str(data[key]) for key in data.files if key.startswith("provenance_")}


def bundle_paths(raw_dir: Path) -> dict[str, Path]:
    """Relative POSIX path -> bundle, so two raw dirs can be walked in lockstep."""
    return {path.relative_to(raw_dir).as_posix(): path for path in sorted(raw_dir.rglob("*.npz"))}


def relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(second))
    return float(np.linalg.norm(first - second) / denominator) if denominator > 0 else float("inf")


def frame_pairing_gate(arm_raw_dir: Path, baseline_raw_dir: Path) -> dict:
    """G1: every temporal bundle samples exactly the baseline's frames, from the same
    pixels, model and library.

    Three checks. (1) The four sampling keys are equal bundle by bundle. (2) Every
    provenance key except the temporal ones is equal, so an arm extracted with the
    wrong ``--variant`` or from another transformers version cannot pass (those keys
    are computed before any pixel transform, so check 1 alone would let it through).
    (3) The ``mean_pool_fc_norm`` features actually moved for a reordered arm (mean
    relative L2 > 0); a reorder that never reached the pixels would otherwise return
    four null deltas and read as "undetermined". For ``frame_identity`` the distance is
    reported, not gated: G3 owns that reading and has its registered fallback.
    """
    arm = bundle_paths(arm_raw_dir)
    base = bundle_paths(baseline_raw_dir)
    missing = sorted(set(base) - set(arm))
    extra = sorted(set(arm) - set(base))
    mismatched: list[str] = []
    provenance_mismatch: list[str] = []
    arm_temporal: set[str] = set()
    distances: list[float] = []
    for rel in sorted(set(arm) & set(base)):
        with np.load(arm[rel], allow_pickle=False) as a, np.load(base[rel], allow_pickle=False) as b:
            for key in PAIRING_KEYS:
                if key not in a.files or key not in b.files or not np.array_equal(a[key], b[key]):
                    mismatched.append(f"{rel}:{key}")
                    break
            prov_a, prov_b = read_provenance(a), read_provenance(b)
            arm_temporal.add(prov_a.get("temporal_transform", "<unstamped>"))
            shared_keys = {k for k in prov_a if k not in TEMPORAL_PROVENANCE_KEYS} | set(prov_b)
            for key in sorted(shared_keys):
                if prov_a.get(key) != prov_b.get(key):
                    provenance_mismatch.append(f"{rel}:{key}={prov_a.get(key)!r} vs baseline {prov_b.get(key)!r}")
            if FEATURE_KEY in a.files and FEATURE_KEY in b.files and a[FEATURE_KEY].shape == b[FEATURE_KEY].shape:
                distances.append(relative_l2(a[FEATURE_KEY].astype(np.float64), b[FEATURE_KEY].astype(np.float64)))
    mean_distance = float(np.mean(distances)) if distances else None
    temporal = arm_temporal.pop() if len(arm_temporal) == 1 else None
    features_moved = (
        True
        if temporal == "frame_identity"
        else (mean_distance is not None and mean_distance > 0)
    )
    return {
        "arm_bundles": len(arm),
        "baseline_bundles": len(base),
        "expected_bundles": EXPECTED_BUNDLES,
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
        "provenance_mismatch": provenance_mismatch,
        "temporal_transform": temporal,
        "feature_relative_l2": {
            "mean": mean_distance,
            "max": float(np.max(distances)) if distances else None,
            "n": len(distances),
            "gated": temporal != "frame_identity",
        },
        "passed": (
            len(arm) == EXPECTED_BUNDLES
            and not missing
            and not extra
            and not mismatched
            and not provenance_mismatch
            and temporal is not None
            and features_moved
        ),
    }


def permutation_rule_ok(arm: str, permutation: np.ndarray) -> bool:
    """The per-arm rule from plan gate G2, one clip at a time."""
    n = len(permutation)
    natural = np.arange(n)
    if arm == "frame_identity":
        return bool(np.array_equal(permutation, natural))
    if arm == "frame_reverse":
        return bool(np.array_equal(permutation, natural[::-1]))
    if arm == "rep_static_frame":
        return bool(np.all(permutation == min(STATIC_FRAME_INDEX, n - 1)))
    if not np.array_equal(np.sort(permutation), natural):
        return False
    if arm == "frame_shuffle":
        return True
    if arm == "tubelet_shuffle":
        heads = permutation[::TUBELET_SIZE]
        return bool(np.all(heads % TUBELET_SIZE == 0)) and bool(
            np.array_equal(permutation[1::TUBELET_SIZE], heads + 1)
        )
    raise ValueError(f"unknown temporal arm {arm!r}")


def mean_displacement(permutations: np.ndarray) -> float:
    """Mean |i - pi(i)| over every clip; 5.3125 in expectation for a uniform 16-permutation."""
    index = np.arange(permutations.shape[-1])
    return float(np.mean(np.abs(permutations - index)))


def permutation_gate(arm_raw_dir: Path, arm: str) -> dict:
    """G2: the stored permutations obey the arm's rule and re-derive from the seed."""
    paths = bundle_paths(arm_raw_dir)
    invalid: list[str] = []
    identity_draws = 0
    stacks: list[np.ndarray] = []
    clips: list[tuple[str, int, np.ndarray]] = []
    for rel, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            if "frame_permutations" not in data.files:
                invalid.append(f"{rel}:missing frame_permutations")
                continue
            permutations = data["frame_permutations"].astype(np.int64)
            sample_id = str(data["sample_id"])
            starts = data["clip_starts"].astype(int)
        if permutations.shape[1:] != (CLIP_LENGTH,) or permutations.shape[0] != len(starts):
            invalid.append(f"{rel}:shape {permutations.shape}")
            continue
        for start, permutation in zip(starts, permutations):
            if not permutation_rule_ok(arm, permutation):
                invalid.append(f"{rel}:clip{int(start)}")
            if arm == "frame_shuffle" and np.array_equal(permutation, np.arange(CLIP_LENGTH)):
                identity_draws += 1
            clips.append((sample_id, int(start), permutation))
        stacks.append(permutations)

    n_clips = len(clips)
    displacement = mean_displacement(np.concatenate(stacks)) if stacks else None
    displacement_ok = (
        abs(displacement - EXPECTED_MEAN_DISPLACEMENT) <= DISPLACEMENT_TOLERANCE
        if arm == "frame_shuffle" and displacement is not None
        else True
    )

    # Re-derive a spread of clips from the seeding rule; must match bit for bit.
    step = max(1, n_clips // RESEED_SAMPLE)
    reseed_mismatch: list[str] = []
    checked = 0
    for sample_id, start, permutation in clips[::step][:RESEED_SAMPLE]:
        checked += 1
        expected = temporal_permutation(arm, CLIP_LENGTH, sample_id, start)
        if not np.array_equal(expected, permutation):
            reseed_mismatch.append(f"{sample_id}:clip{start}")

    return {
        "arm": arm,
        "bundles": len(paths),
        "clips": n_clips,
        "invalid": invalid,
        "identity_draws": identity_draws,
        "mean_abs_displacement": displacement,
        "expected_mean_abs_displacement": EXPECTED_MEAN_DISPLACEMENT,
        "displacement_tolerance": DISPLACEMENT_TOLERANCE,
        "displacement_gated": arm == "frame_shuffle",
        "reseed_checked": checked,
        "reseed_mismatch": reseed_mismatch,
        "passed": len(paths) == EXPECTED_BUNDLES and not invalid and displacement_ok and not reseed_mismatch and checked > 0,
    }


# --------------------------------------------------------------------------- #
# predict: the frozen LOSO recipe on one temporal arm                          #
# --------------------------------------------------------------------------- #


def assert_feature_dir_is_arm(feature_dir: Path, arm: str) -> int:
    """The materialised dir must carry exactly one temporal stamp, equal to ``arm``.

    Gates run on the raw dir; the classifier reads the materialised one; only the path
    convention links them. The stamp is carried through materialisation, so it is the
    one thing that ties an OOF file to the arm its gates were run on.
    """
    stamps: set[str] = set()
    count = 0
    for path in feature_dir.rglob("*.npz"):
        count += 1
        with np.load(path, allow_pickle=False) as data:
            stamps.add(str(data["provenance_temporal_transform"]) if "provenance_temporal_transform" in data.files else "<unstamped>")
    if count == 0:
        raise SystemExit(f"No materialised bundles under {feature_dir}. Run `materialize --arm {arm}` first.")
    if stamps != {arm}:
        raise SystemExit(f"{feature_dir} carries temporal stamps {sorted(stamps)}, expected exactly {{{arm!r}}}.")
    return count


def run_predict(
    arm: str,
    feature_dir: Path,
    manifest_path: Path,
    labels_path: Path,
    output_dir: Path,
    seeds: Sequence[int],
    device_arg: str | None,
) -> dict:
    """``videomae_stage_a.run_arm`` unchanged, saving out-of-fold probabilities per arm."""
    import torch

    device = torch.device(
        "cuda" if (device_arg != "cpu" and device_arg is not None and torch.cuda.is_available()) else "cpu"
    )
    n_bundles = assert_feature_dir_is_arm(feature_dir, arm)
    config = FoldConfig()
    labels = {key: int(value) for key, value in json.load(labels_path.open()).items()}
    metadata = load_metadata(manifest_path)
    manifest = manifest_index(manifest_path)
    subject_samples = subjects_to_samples(manifest_path)
    sample_counts = {person: len(ids) for person, ids in subject_samples.items()}
    ordered_subjects = sorted(subject_samples, key=int)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Temporal-control OOF on {device} | arm={arm} | features={feature_dir} | seeds {list(seeds)}")
    report: dict = {
        "arm": arm,
        "feature_dir": str(feature_dir),
        "feature_bundles": n_bundles,
        "seeds": list(seeds),
        "config": vars(config),
        "per_seed": {},
    }
    for seed in seeds:
        print(f"\n--- {arm} | seed {seed} ---")
        folds = run_arm(
            feature_dir,
            labels,
            subject_samples,
            ordered_subjects,
            sample_counts,
            metadata,
            config,
            device,
            seed,
            retain_predictions=True,
        )
        rows = write_oof(oof_path(output_dir, arm, seed), folds, manifest, seed)
        with folds_path(output_dir, arm, seed).open("w", encoding="utf-8") as handle:
            json.dump({"seed": seed, "arm": arm, "config": vars(config), "folds": folds}, handle, indent=2, sort_keys=True)
        big = [fold for fold in folds if fold["n_test"] >= MIN_VAL_SUBJECT_SAMPLES]
        mean_ba = float(np.mean([fold["balanced_accuracy"] for fold in big]))
        report["per_seed"][str(seed)] = {"rows": rows, "balanced_accuracy_no_p10": mean_ba}
        print(f"  wrote {rows} OOF rows | bal_acc (9 folds, no P10) {mean_ba:.4f}")

    with (output_dir / f"predict_report_{arm}.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report


def load_arm_oof(output_dir: Path, arm: str, seeds: Sequence[int]) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        path = oof_path(output_dir, arm, seed)
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run `predict --arm {arm}` first.")
        rows.extend(read_oof(path))
    return rows


def load_baseline_oof(baseline_oof_dir: Path, seeds: Sequence[int]) -> list[dict]:
    """The identity control's stored OOF (``oof_seed<seed>.csv``): the 0.8741 arm."""
    rows: list[dict] = []
    for seed in seeds:
        path = baseline_oof_path(baseline_oof_dir, seed)
        if not path.exists():
            raise SystemExit(f"Missing baseline OOF {path}.")
        rows.extend(read_oof(path))
    return rows


# --------------------------------------------------------------------------- #
# analyze: within-session statistic per arm (identity control, unchanged)      #
# --------------------------------------------------------------------------- #


def absolute_verdict(statistic: dict, p_value: float) -> dict:
    """Is the arm itself above chance? (plan reading table, row 4 trigger).

    The registered trigger is exactly two conditions: mean > 0.55 and permutation
    p < 0.05. The subject count is reported alongside but does not enter the verdict;
    the identity control's 6/9 rule is that plan's, not this one's.
    """
    conditions = {
        "mean_auc_above_0.55": statistic["mean"] > DECISION_AUC,
        "permutation_p_below_0.05": p_value < ALPHA,
        "subjects_above_0.5_descriptive": f"{statistic['n_subjects_above_chance']}/{statistic['n_subjects']}",
    }
    above = conditions["mean_auc_above_0.55"] and conditions["permutation_p_below_0.05"]
    return {
        "conditions": conditions,
        "above_chance": above,
        # `reading` is what the identity control's printer shows; the registered
        # reading-table row lives in `paired`, not here.
        "reading": (
            "arm ranks correct above incorrect within recordings; the paired delta decides the reading-table row"
            if above
            else "arm is not above chance within recordings (reading table row 4 trigger)"
        ),
    }


def position_secondaries(oof_rows: Sequence[dict], seeds: Sequence[int]) -> dict:
    """Position-balanced AUC on the interleaved sessions (position control, reused)."""
    scores = repetition_scores(oof_rows, seeds)
    pair_sessions = build_pair_sessions(scores, seeds)
    model = [session.concordance for session in pair_sessions]
    rule = [position_rule_concordance(session.positions) for session in pair_sessions]
    interleaved = interleaved_mask(pair_sessions)
    return {
        "position_balanced": {
            **position_balanced_auc(pair_sessions, model),
            "position_rule": position_balanced_auc(pair_sessions, rule)["mean"],
        },
        "unrestricted_interleaved_sessions": {
            **subset_auc(pair_sessions, model, interleaved),
            "position_rule": subset_auc(pair_sessions, rule, interleaved)["mean"],
        },
        "session_split": {
            "interleaved": int(sum(interleaved)),
            "single_boundary": int(len(interleaved) - sum(interleaved)),
        },
    }


def reproduction_check(statistic_mean: float, balanced_accuracy: dict | None) -> dict:
    """G3: ``frame_identity`` must return the stored baseline's two headline numbers."""
    auc_delta = statistic_mean - REFERENCE_WITHIN_SESSION_AUC
    ba_mean = balanced_accuracy["mean"] if balanced_accuracy else None
    ba_delta = (ba_mean - FRAMING_BASELINE_BA) if ba_mean is not None else None
    auc_ok = abs(auc_delta) < REPRODUCTION_TOLERANCE
    # A missing folds file is "unavailable", never a failed reproduction: the plan's
    # fallback row (re-base on frame_identity) is for a venv mismatch, not a lost file.
    ba_ok = (abs(ba_delta) < REPRODUCTION_TOLERANCE) if ba_delta is not None else None
    return {
        "within_session_auc": statistic_mean,
        "reference_within_session_auc": REFERENCE_WITHIN_SESSION_AUC,
        "auc_delta": auc_delta,
        "loso_balanced_accuracy": ba_mean,
        "reference_loso_balanced_accuracy": FRAMING_BASELINE_BA,
        "ba_delta": ba_delta,
        "tolerance": REPRODUCTION_TOLERANCE,
        "auc_reproduced": auc_ok,
        "ba_reproduced": ba_ok,
        "passed": (auc_ok and ba_ok) if ba_ok is not None else None,
        "reason": None if ba_ok is not None else "LOSO folds unavailable; run predict with all seeds",
        "on_failure": "reading-table last row: pair every delta against frame_identity and report both baselines",
    }


def analyze_arm(
    arm: str,
    output_dir: Path,
    manifest_path: Path,
    seeds: Sequence[int],
    n_permutations: int,
    permutation_seed: int,
    n_bootstrap: int,
    framing_summary_path: Path,
) -> tuple[dict, np.ndarray]:
    oof_rows = load_arm_oof(output_dir, arm, seeds)
    audit = audit_oof(oof_rows, manifest_index(manifest_path), seeds)  # G4
    if not audit["passed"]:
        raise SystemExit(f"G4 OOF integrity failed for {arm}: {audit['problems']}")

    summary, null = within_session_analysis(oof_rows, seeds, n_permutations, permutation_seed, n_bootstrap)
    summary["arm"] = arm
    summary["gates"] = {"oof_integrity": audit}
    summary["primary"]["verdict"] = absolute_verdict(summary["primary"], summary["primary"]["permutation"]["p_value"])
    summary["secondary"].update(position_secondaries(oof_rows, seeds))

    balanced_accuracy = loso_balanced_accuracy(output_dir, arm, seeds)
    summary["secondary"]["loso_balanced_accuracy"] = balanced_accuracy
    summary["secondary"]["loso_ba_vs_framing"] = (
        paired_delta_vs_framing(balanced_accuracy, framing_summary_path) if balanced_accuracy else None
    )
    if arm == "frame_identity":
        summary["reproduction"] = reproduction_check(summary["primary"]["mean"], balanced_accuracy)
    return summary, null


# --------------------------------------------------------------------------- #
# paired: the registered nine-subject inference                                #
# --------------------------------------------------------------------------- #


def per_subject_auc(oof_rows: Sequence[dict], seeds: Sequence[int]) -> dict[str, float]:
    sessions, _ = build_sessions(repetition_scores(oof_rows, seeds), seeds)
    return observed_statistic(sessions)["per_subject_auc"]


def paired_comparison(
    baseline: dict[str, float],
    candidate: dict[str, float],
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Baseline minus candidate per subject; positive is a drop. Two-sided exact Wilcoxon."""
    shared = sorted(set(baseline) & set(candidate), key=int)
    base = np.asarray([baseline[person] for person in shared], dtype=float)
    cand = np.asarray([candidate[person] for person in shared], dtype=float)
    deltas = base - cand
    greater = exact_wilcoxon_vs(list(deltas), 0.0)
    less = exact_wilcoxon_vs(list(-deltas), 0.0)
    two_sided = min(1.0, 2 * min(greater["p_value"], less["p_value"])) if greater and less else None

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(shared), size=(n_bootstrap, len(shared)))
    delta_means = deltas[draws].mean(axis=1)
    base_means = base[draws].mean(axis=1)
    shares = np.where(base_means > 0.5, delta_means / np.maximum(base_means - 0.5, 1e-12), np.nan)
    baseline_mean = float(base.mean())
    return {
        "subjects": shared,
        "n_subjects": len(shared),
        "baseline_mean": baseline_mean,
        "candidate_mean": float(cand.mean()),
        "per_subject_baseline": {person: float(value) for person, value in zip(shared, base)},
        "per_subject_candidate": {person: float(value) for person, value in zip(shared, cand)},
        "per_subject_delta": {person: float(value) for person, value in zip(shared, deltas)},
        "mean_delta": float(deltas.mean()),
        "sd_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        "n_drop": int(np.sum(deltas > 0)),
        "wilcoxon_greater": greater,
        "wilcoxon_less": less,
        "two_sided_p": two_sided,
        "bootstrap_95ci_delta": [float(np.percentile(delta_means, 2.5)), float(np.percentile(delta_means, 97.5))],
        "share_of_above_chance_signal": float(deltas.mean() / (baseline_mean - 0.5)) if baseline_mean > 0.5 else None,
        "bootstrap_95ci_share": [
            float(np.nanpercentile(shares, 2.5)),
            float(np.nanpercentile(shares, 97.5)),
        ]
        if np.isfinite(shares).any()
        else None,
    }


def reading_table_row(primary_paired: dict, primary_absolute: dict | None) -> dict:
    """Plan 'Reading the result', evaluated as written for ``frame_shuffle``."""
    drop = primary_paired["mean_delta"]
    n_drop = primary_paired["n_drop"]
    p = primary_paired["two_sided_p"]
    paired_significant = p is not None and p < ALPHA and n_drop >= MIN_SUBJECTS_DROP
    above_chance = bool(primary_absolute["above_chance"]) if primary_absolute else None
    conditions = {
        "drop": drop,
        "n_drop_of_9": n_drop,
        "two_sided_p": p,
        "paired_significant_and_at_least_7_drop": paired_significant,
        "shuffled_arm_above_chance": above_chance,
    }
    if above_chance is None:
        row = "absolute verdict unavailable: run `analyze --arm frame_shuffle` first"
    elif not above_chance:
        row = (
            "row 4: the within-session signal is order-dependent almost entirely; OOD degradation of the "
            "backbone cannot be separated from it -- read tubelet_shuffle and frame_reverse"
        )
    elif paired_significant and drop >= ORDER_SHARE_BOUNDARY:
        row = (
            "row 1: at least this share of the within-session signal needs frame order; no static carrier "
            "can produce it; reported as an upper bound on order use"
        )
    elif paired_significant and 0 < drop < ORDER_SHARE_BOUNDARY:
        row = "row 2: order is used but most of the ranking is order-free; report the share, do not call it dynamics"
    else:
        row = "row 3: undetermined -- not 'order does not matter', not 'the model ignores time'"
    return {"conditions": conditions, "row": row}


def static_vs_shuffle(static: dict[str, float], shuffle: dict[str, float]) -> dict:
    shared = sorted(set(static) & set(shuffle), key=int)
    n_ge = int(sum(static[person] >= shuffle[person] for person in shared))
    return {
        "n_subjects": len(shared),
        "n_static_ge_shuffle": n_ge,
        "row_5_triggered": n_ge >= MIN_SUBJECTS_STATIC_GE_SHUFFLE,
        "reading": "row 5: a single frame ranks as well as an unordered set; the signal is a static pose cue"
        if n_ge >= MIN_SUBJECTS_STATIC_GE_SHUFFLE
        else "row 5 not triggered",
    }


def holm_fixed_family(p_values: dict[str, float | None]) -> dict[str, dict]:
    """Holm over a family whose size is the number of arms COMPARED, not the number
    with a p-value. An unavailable p (all deltas zero, arm not analysed) stays in the
    table as unavailable and keeps the denominator; dropping it would make the
    survivors more significant than the registered family allows."""
    family = len(p_values)
    available = sorted(((name, p) for name, p in p_values.items() if p is not None), key=lambda item: item[1])
    corrected: dict[str, dict] = {name: {"raw": None, "holm": None, "significant": None, "unavailable": True} for name in p_values}
    running = 0.0
    for index, (name, raw) in enumerate(available):
        running = max(running, min(1.0, raw * (family - index)))
        corrected[name] = {"raw": raw, "holm": running, "significant": running < ALPHA, "unavailable": False}
    return {"family_size": family, "per_arm": corrected}


def paired_report(
    output_dir: Path,
    arms: Sequence[str],
    baseline_rows: Sequence[dict],
    baseline_label: str,
    seeds: Sequence[int],
    n_bootstrap: int,
    seed: int,
) -> dict:
    # Every compared arm must have been through `analyze` (G4 OOF integrity, the
    # 61-session gate and the estimator check live there); a truncated OOF would
    # otherwise yield a clean-looking delta over fewer subjects.
    summaries: dict[str, dict] = {}
    for arm in arms:
        path = summary_path(output_dir, arm)
        if not path.exists():
            raise SystemExit(f"Missing {path}: run `analyze --arm {arm}` (G4) before `paired`.")
        summaries[arm] = json.load(path.open(encoding="utf-8"))

    baseline = per_subject_auc(baseline_rows, seeds)
    per_arm: dict[str, dict[str, float]] = {}
    comparisons: dict[str, dict] = {}
    for arm in arms:
        per_arm[arm] = per_subject_auc(load_arm_oof(output_dir, arm, seeds), seeds)
        comparisons[arm] = paired_comparison(baseline, per_arm[arm], n_bootstrap, seed)

    # Registered Holm family (plan, Secondary): LOSO BA of every temporal arm vs the
    # framing headline, corrected across the arms compared.
    loso_ba_p = {
        arm: (summaries[arm]["secondary"].get("loso_ba_vs_framing") or {}).get("two_sided_p")
        for arm in arms
    }
    # Exploratory: the secondary arms' within-session paired p-values. The primary
    # arm is excluded so the registered primary p is never Holm-multiplied.
    within_secondary_p = {arm: comparisons[arm]["two_sided_p"] for arm in arms if arm != PRIMARY_ARM}
    report: dict = {
        "baseline": baseline_label,
        "baseline_per_subject": baseline,
        "seeds": list(seeds),
        "bootstrap": {"n_resamples": n_bootstrap, "seed": seed},
        "comparisons": comparisons,
        "loso_ba_vs_framing": {arm: summaries[arm]["secondary"].get("loso_ba_vs_framing") for arm in arms},
        "holm_loso_ba_vs_framing": holm_fixed_family(loso_ba_p),
        "holm_within_session_secondary_arms": {
            "note": "exploratory family; the primary arm's paired p is read raw and is not in it",
            **holm_fixed_family(within_secondary_p),
        },
    }
    if PRIMARY_ARM in comparisons:
        report["reading_table"] = reading_table_row(comparisons[PRIMARY_ARM], summaries[PRIMARY_ARM]["primary"]["verdict"])
    if "rep_static_frame" in per_arm and PRIMARY_ARM in per_arm:
        report["static_vs_shuffle"] = static_vs_shuffle(per_arm["rep_static_frame"], per_arm[PRIMARY_ARM])
    if "frame_reverse" in comparisons:
        reverse = comparisons["frame_reverse"]
        report["reverse_row_6"] = {
            "triggered": reverse["mean_delta"] >= REVERSE_DROP_BOUNDARY
            and reverse["two_sided_p"] is not None
            and reverse["two_sided_p"] < ALPHA,
            "reading": "row 6: the backbone reads direction; unexpected under the prior",
        }
    return report


def print_paired(report: dict) -> None:
    print(f"\n=== paired within-session AUC vs {report['baseline']} (n = subjects) ===")
    for arm, comparison in report["comparisons"].items():
        holm = report["holm_within_session_secondary_arms"]["per_arm"].get(arm, {})
        share = comparison["share_of_above_chance_signal"]
        share_text = f"{share:+.3f}" if share is not None else "n/a"
        p_text = f"{comparison['two_sided_p']:.4f}" if comparison["two_sided_p"] is not None else "n/a"
        print(
            f"  {arm:<17} AUC {comparison['candidate_mean']:.4f}  delta {comparison['mean_delta']:+.4f} "
            f"± {comparison['sd_delta']:.4f}  drop {comparison['n_drop']}/{comparison['n_subjects']}  "
            f"CI {comparison['bootstrap_95ci_delta'][0]:+.4f}..{comparison['bootstrap_95ci_delta'][1]:+.4f}  "
            f"p {p_text}  Holm {'raw (primary)' if arm == PRIMARY_ARM else (f'{holm['holm']:.4f}' if holm.get('holm') is not None else 'n/a')}  share {share_text}"
        )
    print("\n  LOSO balanced accuracy vs 0.6612, Holm across the arms compared (registered family):")
    for arm, entry in report["holm_loso_ba_vs_framing"]["per_arm"].items():
        vs = report["loso_ba_vs_framing"].get(arm) or {}
        if entry["unavailable"] or not vs.get("available"):
            print(f"    {arm:<17} unavailable")
        else:
            print(f"    {arm:<17} delta {vs['mean_delta']:+.4f} ({vs['n_positive']}/{vs['n_subjects']} up)  raw p {entry['raw']:.4f}  Holm {entry['holm']:.4f}")
    if "reading_table" in report:
        print(f"\n  {report['reading_table']['row']}")
    if "static_vs_shuffle" in report:
        print(f"  {report['static_vs_shuffle']['reading']} ({report['static_vs_shuffle']['n_static_ge_shuffle']}/9)")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="REHAB24-6 VideoMAE temporal-order control (pre-registered).")
    sub = parser.add_subparsers(dest="command", required=True)

    gates = sub.add_parser("gates", help="G1 frame pairing and G2 permutation validity on one arm's raw bundles.")
    add_common(gates)
    gates.add_argument("--arm", choices=TEMPORAL_ARMS, required=True)
    gates.add_argument("--raw-parent", type=Path, default=DEFAULT_RAW_PARENT)
    gates.add_argument("--baseline-raw-dir", type=Path, default=DEFAULT_BASELINE_RAW_DIR)

    materialize = sub.add_parser("materialize", help="Raw bundles -> the single LOSO-ready feature dir the plan uses.")
    add_common(materialize)
    materialize.add_argument("--arm", choices=TEMPORAL_ARMS, required=True)
    materialize.add_argument("--raw-parent", type=Path, default=DEFAULT_RAW_PARENT)
    materialize.add_argument("--feature-parent", type=Path, default=DEFAULT_FEATURE_PARENT)

    predict = sub.add_parser("predict", help="Frozen LOSO recipe on one arm, saving out-of-fold probabilities.")
    add_common(predict)
    predict.add_argument("--arm", choices=TEMPORAL_ARMS, required=True)
    predict.add_argument("--feature-parent", type=Path, default=DEFAULT_FEATURE_PARENT)
    predict.add_argument("--labels", type=Path, default=DEFAULT_PROCESSED_ROOT / "labels" / "correctness.json")
    predict.add_argument("--device", type=str, default=None, help="cpu, cuda, or auto.")

    analyse = sub.add_parser("analyze", help="Within-session statistic, permutation null, secondaries, G3/G4.")
    add_common(analyse)
    analyse.add_argument("--arm", choices=TEMPORAL_ARMS, required=True)
    analyse.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    analyse.add_argument("--permutation-seed", type=int, default=DEFAULT_PERMUTATION_SEED)
    analyse.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    analyse.add_argument("--framing-summary", type=Path, default=DEFAULT_FRAMING_SUMMARY)

    paired = sub.add_parser("paired", help="Registered paired nine-subject inference across arms.")
    add_common(paired)
    paired.add_argument("--arms", nargs="+", choices=TEMPORAL_ARMS, default=[a for a in TEMPORAL_ARMS if a != "frame_identity"])
    paired.add_argument(
        "--baseline",
        default=BASELINE_NAME,
        help=f"`{BASELINE_NAME}` (the identity control's stored OOF) or a temporal arm such as frame_identity.",
    )
    paired.add_argument("--baseline-oof-dir", type=Path, default=BASELINE_OOF_DIR)
    paired.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    paired.add_argument("--seed", type=int, default=DEFAULT_PERMUTATION_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "gates":
        raw_dir = args.raw_parent / args.arm
        pairing = frame_pairing_gate(raw_dir, args.baseline_raw_dir)
        permutations = permutation_gate(raw_dir, args.arm)
        report = {"arm": args.arm, "frame_pairing": pairing, "permutations": permutations, "passed": pairing["passed"] and permutations["passed"]}
        path = args.output_dir / f"gates_{args.arm}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        l2 = pairing["feature_relative_l2"]
        print(
            f"G1 frame pairing: {'PASS' if pairing['passed'] else 'FAIL'} "
            f"({pairing['arm_bundles']} bundles, {len(pairing['mismatched'])} mismatched, "
            f"{len(pairing['missing'])} missing, {len(pairing['extra'])} extra; "
            f"feature rel. L2 vs baseline mean {l2['mean']!s} max {l2['max']!s})"
        )
        disp = permutations["mean_abs_displacement"]
        print(
            f"G2 permutations: {'PASS' if permutations['passed'] else 'FAIL'} "
            f"({permutations['clips']} clips, {len(permutations['invalid'])} invalid, "
            f"mean |i - pi(i)| {disp if disp is None else round(disp, 4)} vs {EXPECTED_MEAN_DISPLACEMENT}, "
            f"identity draws {permutations['identity_draws']}, reseed mismatches "
            f"{len(permutations['reseed_mismatch'])}/{permutations['reseed_checked']})"
        )
        print(f"Saved {path}  ->  {'PASS' if report['passed'] else 'FAIL'}")
        if not report["passed"]:
            raise SystemExit("Gate failed; do not materialize or predict this arm.")
        return

    if args.command == "materialize":
        from src.rehab24.videomae_materialize import materialize_all

        counts = materialize_all(
            raw_dir=args.raw_parent / args.arm,
            output_parent=args.feature_parent / args.arm,
            token_poolings=("mean_pool_fc_norm",),
            aggregations=("mean",),
        )
        print(f"Materialized {counts} under {args.feature_parent / args.arm}")
        return

    if args.command == "predict":
        run_predict(
            args.arm,
            temporal_feature_dir(args.feature_parent, args.arm),
            args.manifest,
            args.labels,
            args.output_dir,
            args.seeds,
            args.device,
        )
        return

    if args.command == "analyze":
        summary, null = analyze_arm(
            args.arm,
            args.output_dir,
            args.manifest,
            args.seeds,
            args.permutations,
            args.permutation_seed,
            args.bootstrap,
            args.framing_summary,
        )
        print_analysis(summary)
        pb = summary["secondary"]["position_balanced"]
        print(f"\nposition-balanced (34 interleaved): {pb['mean']:.4f} (rule {pb['position_rule']:.4f})")
        ba = summary["secondary"]["loso_balanced_accuracy"]
        if ba:
            vs = summary["secondary"]["loso_ba_vs_framing"]
            if vs and vs.get("available"):
                p_text = f"{vs['two_sided_p']:.4f}" if vs["two_sided_p"] is not None else "n/a (all deltas zero)"
                delta = f"delta {vs['mean_delta']:+.4f} ({vs['n_positive']}/{vs['n_subjects']} up, p {p_text})"
            else:
                delta = "no framing baseline"
            print(f"LOSO balanced accuracy: {ba['mean']:.4f} ± {ba['sd_over_subjects']:.4f} vs {FRAMING_BASELINE_BA}: {delta}")
        if "reproduction" in summary:
            rep = summary["reproduction"]
            status = "UNAVAILABLE" if rep["passed"] is None else ("PASS" if rep["passed"] else "FAIL")
            print(
                f"G3 reproduction: {status} "
                f"(AUC {rep['within_session_auc']:.4f} vs {REFERENCE_WITHIN_SESSION_AUC}, "
                f"BA {rep['loso_balanced_accuracy']} vs {FRAMING_BASELINE_BA})"
            )
        with summary_path(args.output_dir, args.arm).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        np.savez_compressed(
            null_path(args.output_dir, args.arm),
            null=null,
            observed=np.asarray(summary["primary"]["mean"]),
            permutation_seed=np.asarray(args.permutation_seed),
        )
        print(f"\nSaved analysis for {args.arm} to {args.output_dir}")
        return

    if args.command == "paired":
        if args.baseline == BASELINE_NAME:
            baseline_rows = load_baseline_oof(args.baseline_oof_dir, args.seeds)
        elif args.baseline in TEMPORAL_ARMS:
            baseline_rows = load_arm_oof(args.output_dir, args.baseline, args.seeds)
        else:
            raise SystemExit(f"--baseline must be `{BASELINE_NAME}` or one of {TEMPORAL_ARMS}")
        report = paired_report(args.output_dir, args.arms, baseline_rows, args.baseline, args.seeds, args.bootstrap, args.seed)
        print_paired(report)
        path = args.output_dir / "paired_deltas.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"\nSaved {path}")
        return


if __name__ == "__main__":
    main()
