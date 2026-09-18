"""Evaluation half of the frozen VideoMAE-vs-V-JEPA-2 comparison on REHAB24-6.

Pre-registered in ``notes/rehab24_videomae_vjepa2_validation_plan.md``. This module
consumes feature bundles produced by the extraction half (built in parallel) and is
responsible for: freezing the fold manifest before any feature exists (``init``),
turning per-clip raw bundles into the ``video_feature`` shape the existing LOSO
machinery already reads (``materialize``), running the frozen MLP readout plus an
independent L2-logistic control and the pre-registered null/sensitivity checks
(``evaluate``), and producing the paired-subject statistics and gates (``report``).

Every classifier fit reuses ``loso_cross_validation.train_one_fold`` and
``videomae_video_classifier`` unchanged -- this module never re-implements a fold
loop, a threshold search or a normalization step. What is new here is: the frozen
fold lists (arm-independent, built once from the manifest), the raw-to-materialized
conversion for the new bundle contract, the torch-only L2 logistic control (no
sklearn is available), and the paired-subject inference machinery (sign-flip test,
bootstrap CI, Holm family, gates) that the plan's readout section specifies.

Three subjects matter throughout: "arm" (a feature/checkpoint configuration: vm16,
vj64, vj16), "fold" (a held-out test subject plus its cyclic validation subject) and
"seed" (a readout initialization). A cell in most of this module's dictionaries is
keyed by some subset of those three.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from src.rehab24 import loso_cross_validation as loso
from src.rehab24.dataset import CAMERAS, DEFAULT_PROCESSED_ROOT, load_manifest
from src.rehab24.videomae_identity_control import (
    bootstrap_interval,
    build_sessions,
    holm_correct,
    observed_statistic,
    pairwise_auc,
)
from src.rehab24.videomae_stage_a import (
    MIN_STRATUM_SAMPLES,
    load_metadata,
    ordered_test_ids,
    permute_labels_within_subject,
)
from src.video.classification_metrics import compute_metrics, find_best_threshold, sigmoid
from src.video.videomae_materialize import read_provenance
from src.video.videomae_video_classifier import (
    build_samples,
    clear_feature_cache,
    compute_feature_normalization,
    load_video_feature,
)

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The video-backbone comparison requires `torch`.") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]

#: The three registered arms and their expected pooled-feature dimension (plan
#: "Models and comparisons"). vj16 shares vj64's checkpoint but a shorter input.
ARM_DIMS: dict[str, int] = {"vm16": 768, "vj64": 1024, "vj16": 1024}
ARMS: tuple[str, ...] = tuple(ARM_DIMS)

#: Model-identity gate expectations, read off the raw bundles' provenance keys.
ARM_MODEL_CONFIG: dict[str, dict[str, str]] = {
    "vm16": {"model_name": "MCG-NJU/videomae-base-finetuned-kinetics", "clip_length": "16", "resolution": "224"},
    "vj64": {"model_name": "facebook/vjepa2-vitl-fpc64-256", "clip_length": "64", "resolution": "256"},
    "vj16": {"model_name": "facebook/vjepa2-vitl-fpc64-256", "clip_length": "16", "resolution": "256"},
}

#: Provenance keys every raw bundle must carry and agree on within one arm.
PROVENANCE_KEYS: tuple[str, ...] = (
    "provenance_model_name",
    "provenance_revision",
    "provenance_arm",
    "provenance_clip_length",
    "provenance_frame_stride",
    "provenance_num_clips",
    "provenance_resolution",
    "provenance_pooling",
    "provenance_output_field",
    "provenance_sampler",
    "provenance_variant",
    "provenance_dtype",
    "provenance_transformers_version",
    "provenance_torch_version",
    "provenance_code_fingerprint",
)

#: Metadata carried from the raw bundle into every materialized bundle.
CARRIED_KEYS: tuple[str, ...] = ("sample_id", "video_id", "exercise_id", "person_id", "camera", "correctness")

SEEDS: tuple[int, ...] = (42, 7, 1234)
NULL_SEEDS: tuple[int, ...] = (101, 202, 303)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260918
PRACTICAL_MARGIN = 0.02
ALPHA = 0.05
NULL_AUDIT_THRESHOLD = 0.55

#: Primary reporting is P1-P9 (plan "Dataset, exclusions and sampling"); P10 has only
#: 16 samples and is never an eligible validation subject.
SENSITIVITY_SUBJECT = "10"
PRIMARY_SUBJECTS: tuple[str, ...] = tuple(str(i) for i in range(1, 10))

#: The one primary contrast, and the four-test secondary family (plan "Secondary
#: analyses and controls"). Holm applies to exactly this family, never a subset.
PRIMARY_CONTRAST = ("vj64", "vm16")
SECONDARY_FAMILY = (
    ("vj16_minus_vm16_ba", "ba", "vj16", "vm16"),
    ("vj64_minus_vj16_ba", "ba", "vj64", "vj16"),
    ("vj64_minus_vm16_auc", "within_session_auc", "vj64", "vm16"),
    ("vj64_minus_vm16_logreg_ba", "logistic_ba", "vj64", "vm16"),
)

OOF_COLUMNS = (
    "sample_id",
    "repetition_id",
    "session_id",
    "person_id",
    "exercise_id",
    "camera",
    "seed",
    "label",
    "probability",
    "threshold",
    "test_subject",
    "val_subject",
)


# --------------------------------------------------------------------------- #
# paths                                                                        #
# --------------------------------------------------------------------------- #


def run_config_path(run_dir: Path) -> Path:
    return run_dir / "run_config.json"


def manifest_audit_path(run_dir: Path) -> Path:
    return run_dir / "manifest_audit.json"


def folds_path(run_dir: Path, no_p10: bool = False) -> Path:
    return run_dir / ("folds_no_p10.json" if no_p10 else "folds.json")


def raw_arm_dir(run_dir: Path, arm: str) -> Path:
    return run_dir / "raw" / arm


def features_arm_dir(run_dir: Path, arm: str) -> Path:
    return run_dir / "features" / arm


def materialize_audit_path(run_dir: Path, arm: str) -> Path:
    return features_arm_dir(run_dir, arm) / "materialize_audit.json"


def oof_dir(run_dir: Path, name: str) -> Path:
    return run_dir / "oof" / name


def oof_seed_path(run_dir: Path, name: str, seed: int) -> Path:
    return oof_dir(run_dir, name) / f"seed{seed}.csv"


def gates_path(run_dir: Path) -> Path:
    return run_dir / "gates.json"


def summary_path(run_dir: Path) -> Path:
    return run_dir / "summary.json"


# --------------------------------------------------------------------------- #
# small utilities                                                              #
# --------------------------------------------------------------------------- #


def repetition_key(row: dict[str, str]) -> str:
    """Repetition key = video_id + repetition_number (plan "Terminology")."""
    return f"{row['video_id']}_rep{row['repetition_number']}"


def session_key(row: dict[str, str]) -> str:
    """Session key = exercise_id + video_id (plan "Terminology")."""
    return f"{row['exercise_id']}_{row['video_id']}"


def hash_ids(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def git_commit_hash(repo_root: Path = REPO_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:  # pragma: no cover - best-effort provenance only
        return None


def resolve_device(device_arg: str | None) -> "torch.device":
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# init: manifest audit + frozen folds                                         #
# --------------------------------------------------------------------------- #


def manifest_audit(manifest_path: Path, labels_path: Path) -> dict:
    """Row/rep/session counts and the two hard data-quality invariants.

    Binary labels and complete camera pairs are asserted unconditionally: any
    manifest that violates them cannot support the plan's LOSO/camera-averaging
    design regardless of dataset size. The 2144/1072/65 REHAB24-6 counts are only
    ever *reported*, never asserted, so this same function runs over the small
    synthetic manifests the tests use.
    """
    rows = load_manifest(manifest_path)
    labels = {key: int(value) for key, value in json.load(labels_path.open(encoding="utf-8")).items()}

    label_values = {row["correctness"] for row in rows}
    if not label_values <= {"0", "1"}:
        raise ValueError(f"Manifest correctness column is not exactly binary: {sorted(label_values)}")
    if not set(labels.values()) <= {0, 1}:
        raise ValueError("Labels file is not exactly binary.")

    reps: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        reps[repetition_key(row)].add(row["camera"])
    bad_pairs = sorted(key for key, cams in reps.items() if cams != set(CAMERAS))
    if bad_pairs:
        raise ValueError(f"{len(bad_pairs)} repetitions do not have exactly the cameras {CAMERAS}: {bad_pairs[:5]}")

    sessions = {session_key(row) for row in rows}
    by_subject: dict[str, int] = defaultdict(int)
    by_exercise: dict[str, int] = defaultdict(int)
    by_camera: dict[str, int] = defaultdict(int)
    by_label: dict[str, int] = defaultdict(int)
    by_subject_exercise_camera_label: dict[str, int] = defaultdict(int)
    for row in rows:
        by_subject[row["person_id"]] += 1
        by_exercise[row["exercise_id"]] += 1
        by_camera[row["camera"]] += 1
        by_label[row["correctness"]] += 1
        key = f"{row['person_id']}|{row['exercise_id']}|{row['camera']}|{row['correctness']}"
        by_subject_exercise_camera_label[key] += 1

    counts = {
        "rows": len(rows),
        "repetitions": len(reps),
        "sessions": len(sessions),
        "by_subject": dict(sorted(by_subject.items(), key=lambda kv: int(kv[0]))),
        "by_exercise": dict(sorted(by_exercise.items())),
        "by_camera": dict(sorted(by_camera.items())),
        "by_label": dict(sorted(by_label.items())),
        "by_subject_exercise_camera_label": dict(sorted(by_subject_exercise_camera_label.items())),
    }
    expected = {"rows": 2144, "repetitions": 1072, "sessions": 65}
    counts["matches_expected_rehab24_6"] = {
        key: counts[key] == value for key, value in expected.items() if key in counts
    }
    counts["expected_rehab24_6"] = expected
    return counts


def build_fold(
    test_subject: str,
    ordered_subjects: list[str],
    subject_samples: dict[str, list[str]],
    sample_counts: dict[str, int],
    drop_p10_from_train: bool,
) -> dict:
    val_subject = loso.pick_val_subject(test_subject, ordered_subjects, sample_counts)
    test_ids = sorted(subject_samples[test_subject])
    val_ids = sorted(subject_samples[val_subject])
    train_subjects = [s for s in ordered_subjects if s not in {test_subject, val_subject}]
    if drop_p10_from_train:
        train_subjects = [s for s in train_subjects if s != SENSITIVITY_SUBJECT]
    train_ids = sorted(sid for s in train_subjects for sid in subject_samples[s])
    return {
        "test_subject": test_subject,
        "val_subject": val_subject,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }


def build_all_folds(manifest_path: Path, drop_p10_from_train: bool = False) -> list[dict]:
    subject_samples = loso.subjects_to_samples(manifest_path)
    sample_counts = {person: len(ids) for person, ids in subject_samples.items()}
    ordered_subjects = sorted(subject_samples, key=int)
    return [
        build_fold(subject, ordered_subjects, subject_samples, sample_counts, drop_p10_from_train)
        for subject in ordered_subjects
    ]


def folds_payload(folds: list[dict]) -> dict:
    return {
        "folds": [
            {
                **fold,
                "hashes": {
                    "train": hash_ids(fold["train_ids"]),
                    "val": hash_ids(fold["val_ids"]),
                    "test": hash_ids(fold["test_ids"]),
                },
            }
            for fold in folds
        ]
    }


def load_folds(path: Path) -> list[dict]:
    return json.load(path.open(encoding="utf-8"))["folds"]


def run_init(run_dir: Path, manifest_path: Path, labels_path: Path, arms: Sequence[str]) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)

    audit = manifest_audit(manifest_path, labels_path)
    with manifest_audit_path(run_dir).open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)

    folds = build_all_folds(manifest_path, drop_p10_from_train=False)
    folds_no_p10 = build_all_folds(manifest_path, drop_p10_from_train=True)
    with folds_path(run_dir).open("w", encoding="utf-8") as handle:
        json.dump(folds_payload(folds), handle, indent=2, sort_keys=True)
    with folds_path(run_dir, no_p10=True).open("w", encoding="utf-8") as handle:
        json.dump(folds_payload(folds_no_p10), handle, indent=2, sort_keys=True)

    config = {
        "arms": list(arms),
        "seeds": list(SEEDS),
        "null_seeds": list(NULL_SEEDS),
        "fold_config": asdict(loso.FoldConfig()),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "practical_margin": PRACTICAL_MARGIN,
        "git_commit": git_commit_hash(),
        "manifest_path": str(manifest_path),
        "manifest_md5": file_md5(manifest_path),
    }
    with run_config_path(run_dir).open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    return {"manifest_audit": audit, "folds": folds, "folds_no_p10": folds_no_p10, "config": config}


# --------------------------------------------------------------------------- #
# materialize: raw per-clip bundles -> video_feature bundles                  #
# --------------------------------------------------------------------------- #


def index_raw_bundles(arm_dir: Path) -> dict[str, list[Path]]:
    """Stem -> every path with that stem, so duplicates are visible rather than silently shadowed."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(arm_dir.rglob("*.npz")):
        grouped[path.stem].append(path)
    return dict(grouped)


def materialize(run_dir: Path, arm: str, manifest_path: Path) -> dict:
    """Turn ``raw/<arm>/<split>/<id>.npz`` bundles into ``features/<arm>/<split>/<id>.npz``.

    Refuses (raises ``SystemExit`` and still writes the audit describing why) on any
    missing/duplicate sample id, non-finite vector, dimension mismatch against
    ``ARM_DIMS[arm]``, or a provenance key that disagrees across bundles of the same
    arm -- each is a reason the arm cannot be scored against its declared identity.
    """
    if arm not in ARM_DIMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}.")

    manifest_rows = {row["sample_id"]: row for row in load_manifest(manifest_path)}
    expected_ids = set(manifest_rows)
    arm_raw_dir = raw_arm_dir(run_dir, arm)
    if not arm_raw_dir.exists():
        raise SystemExit(f"No raw bundles for arm {arm!r} at {arm_raw_dir}.")

    stems = index_raw_bundles(arm_raw_dir)
    duplicates = {stem: [str(p) for p in paths] for stem, paths in stems.items() if len(paths) > 1}
    found_ids = set(stems)
    missing = sorted(expected_ids - found_ids)
    unexpected = sorted(found_ids - expected_ids)

    expected_dim = ARM_DIMS[arm]
    non_finite: list[str] = []
    dim_mismatches: dict[str, int] = {}
    provenance_values: dict[str, set[str]] = defaultdict(set)
    materialized_records: list[tuple[str, str, np.ndarray, dict[str, np.ndarray]]] = []

    for sample_id in sorted(expected_ids & found_ids):
        path = stems[sample_id][0]
        with np.load(path, allow_pickle=False) as data:
            clip_features = np.asarray(data["clip_features"], dtype=np.float32)
            if not np.isfinite(clip_features).all():
                non_finite.append(sample_id)
            video_feature = clip_features.mean(axis=0).astype(np.float32)
            if video_feature.shape[0] != expected_dim:
                dim_mismatches[sample_id] = int(video_feature.shape[0])
            provenance = read_provenance(data)
            for key in PROVENANCE_KEYS:
                if key in data.files:
                    provenance_values[key].add(str(data[key]))
            carried = {key: data[key] for key in CARRIED_KEYS if key in data.files}
        materialized_records.append((sample_id, path.parent.name, video_feature, carried))

    provenance_disagreements = {key: sorted(values) for key, values in provenance_values.items() if len(values) > 1}
    missing_provenance = sorted(set(PROVENANCE_KEYS) - set(provenance_values))

    # A genuine measurement of the array shape, independent of ``expected_dim`` (the
    # arm's declared config): ``gate_model_identity`` reads this, not the config echo,
    # so a bundle that consistently carries the wrong dimension cannot pass by
    # comparing a constant against itself (review defect #2).
    measured_dims = Counter(int(video_feature.shape[0]) for _, _, video_feature, _ in materialized_records)
    measured_dim = measured_dims.most_common(1)[0][0] if measured_dims else None

    problems: list[str] = []
    if duplicates:
        problems.append(f"duplicate sample ids: {sorted(duplicates)[:5]}")
    if missing:
        problems.append(f"missing sample ids: {missing[:5]} (+{max(0, len(missing) - 5)} more)")
    if unexpected:
        problems.append(f"unexpected sample ids not in manifest: {unexpected[:5]}")
    if non_finite:
        problems.append(f"non-finite feature vectors: {non_finite[:5]}")
    if dim_mismatches:
        problems.append(f"dim mismatches (expected {expected_dim}): {dict(list(dim_mismatches.items())[:5])}")
    if provenance_disagreements:
        problems.append(f"provenance disagreements: {provenance_disagreements}")
    if missing_provenance:
        problems.append(f"provenance keys missing from every bundle: {missing_provenance}")

    audit = {
        "arm": arm,
        "expected_dim": expected_dim,
        "measured_dim": measured_dim,
        "measured_dims_observed": {str(dim): count for dim, count in sorted(measured_dims.items())},
        "n_expected": len(expected_ids),
        "n_found": len(found_ids),
        "missing_sample_ids": missing,
        "unexpected_sample_ids": unexpected,
        "duplicate_sample_ids": duplicates,
        "non_finite_sample_ids": non_finite,
        "dim_mismatches": dim_mismatches,
        "provenance": {key: sorted(values)[0] for key, values in provenance_values.items() if len(values) == 1},
        "provenance_disagreements": provenance_disagreements,
        "problems": problems,
        "passed": not problems,
        "written": 0,
    }
    audit_path = materialize_audit_path(run_dir, arm)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)

    if problems:
        raise SystemExit(f"materialize refused for arm {arm!r}: {problems}")

    output_dir = features_arm_dir(run_dir, arm)
    for sample_id, split, video_feature, carried in materialized_records:
        out_path = output_dir / split / f"{sample_id}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, video_feature=video_feature, **carried)

    audit["written"] = len(materialized_records)
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    # Materialize may follow a prior read of the same directory (e.g. an earlier failed
    # attempt, or a test's own setup); a stale cache would silently serve old bytes.
    clear_feature_cache()
    return audit


# --------------------------------------------------------------------------- #
# evaluate: MLP readout, logistic control, nulls, no-P10 sensitivity          #
# --------------------------------------------------------------------------- #


def build_oof_rows(
    fold: dict,
    sample_ids: list[str],
    probabilities: np.ndarray,
    labels_arr: np.ndarray,
    threshold: float,
    seed: int,
    manifest_rows: dict[str, dict],
) -> list[dict]:
    rows = []
    for sample_id, probability, label in zip(sample_ids, probabilities, labels_arr):
        row = manifest_rows[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "repetition_id": repetition_key(row),
                "session_id": session_key(row),
                "person_id": row["person_id"],
                "exercise_id": row["exercise_id"],
                "camera": row["camera"],
                "seed": seed,
                "label": int(label),
                "probability": repr(float(probability)),
                "threshold": repr(float(threshold)),
                "test_subject": fold["test_subject"],
                "val_subject": fold["val_subject"],
            }
        )
    return rows


def write_oof_csv(path: Path, rows: list[dict]) -> int:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OOF_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_oof_csv(path: Path) -> list[dict]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
        row["seed"] = int(row["seed"])
        row["probability"] = float(row["probability"])
        row["threshold"] = float(row["threshold"])
    return rows


def run_mlp_fold(
    feature_dir: Path,
    fold: dict,
    labels: dict[str, int],
    config: "loso.FoldConfig",
    device: "torch.device",
    seed: int,
) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    threshold, test_prob, test_labels = loso.train_one_fold(
        feature_dir, fold["train_ids"], fold["val_ids"], fold["test_ids"], labels, config, device, seed
    )
    sample_ids = ordered_test_ids(feature_dir, fold["test_ids"], labels)
    if len(sample_ids) != len(test_prob):
        raise SystemExit(
            f"Prediction/id misalignment for test_subject {fold['test_subject']}: "
            f"{len(test_prob)} predictions vs {len(sample_ids)} ids."
        )
    return sample_ids, test_prob, test_labels, threshold


def run_mlp_evaluation(
    feature_dir: Path,
    folds: list[dict],
    labels: dict[str, int],
    manifest_rows: dict[str, dict],
    config: "loso.FoldConfig",
    device: "torch.device",
    seed: int,
    output_path: Path,
) -> dict:
    """Every fold, one seed, through the frozen MLP readout; writes one OOF CSV."""
    all_rows: list[dict] = []
    fold_records = []
    for fold in folds:
        sample_ids, probabilities, labels_arr, threshold = run_mlp_fold(feature_dir, fold, labels, config, device, seed)
        all_rows.extend(build_oof_rows(fold, sample_ids, probabilities, labels_arr, threshold, seed, manifest_rows))
        metrics = compute_metrics(probabilities, labels_arr, threshold=threshold)
        fold_records.append(
            {
                "test_subject": fold["test_subject"],
                "val_subject": fold["val_subject"],
                "n_test": int(len(labels_arr)),
                "threshold": float(threshold),
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
        )
    written = write_oof_csv(output_path, all_rows)
    return {"seed": seed, "rows": written, "folds": fold_records}


def mlp_parameter_count(feature_dim: int, config: "loso.FoldConfig") -> int:
    from src.video.videomae_video_classifier import VideoFeatureClassifier

    model = VideoFeatureClassifier(feature_dim=feature_dim, hidden_dim=config.hidden_dim, dropout=config.dropout)
    return int(sum(p.numel() for p in model.parameters()))


def train_logistic_fold(
    feature_dir: Path,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    labels: dict[str, int],
    device: "torch.device",
    max_iter: int = 500,
    tolerance: float = 1e-6,
) -> tuple[float, np.ndarray, np.ndarray, list[str], dict]:
    """L2 logistic regression with training-only class weights, no sklearn.

    Objective: ``C * sum_i w_i * BCE_i + 0.5 * ||weight||^2`` (bias excluded from the
    penalty), ``C=1``, full-batch LBFGS with a strong-Wolfe line search -- plain
    gradient-descent LBFGS diverges on unregularized logistic loss without it.
    """
    train_samples = build_samples(feature_dir, train_ids, labels)
    val_samples = build_samples(feature_dir, val_ids, labels)
    test_samples = build_samples(feature_dir, test_ids, labels)
    if not train_samples or not val_samples or not test_samples:
        raise SystemExit("A fold is missing feature files for the logistic control.")

    normalization = compute_feature_normalization(train_samples)

    def feature_matrix(samples) -> np.ndarray:
        return np.stack([normalization.apply(load_video_feature(s.feature_path)) for s in samples]).astype(np.float64)

    x_train = feature_matrix(train_samples)
    y_train = np.array([s.label for s in train_samples], dtype=np.float64)
    x_val = feature_matrix(val_samples)
    y_val = np.array([s.label for s in val_samples], dtype=np.float64)
    x_test = feature_matrix(test_samples)
    y_test = np.array([s.label for s in test_samples], dtype=np.float64)

    n_total = len(y_train)
    n_pos = max(float(y_train.sum()), 1.0)
    n_neg = max(n_total - float(y_train.sum()), 1.0)
    weight_pos = n_total / (2.0 * n_pos)
    weight_neg = n_total / (2.0 * n_neg)
    sample_weights = np.where(y_train == 1, weight_pos, weight_neg)

    x_train_t = torch.tensor(x_train, dtype=torch.float64, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float64, device=device)
    w_t = torch.tensor(sample_weights, dtype=torch.float64, device=device)

    dim = x_train.shape[1]
    weight = torch.zeros(dim, dtype=torch.float64, device=device, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=tolerance,
        tolerance_change=tolerance * 1e-3,
        line_search_fn="strong_wolfe",
    )

    def closure() -> "torch.Tensor":
        optimizer.zero_grad()
        logits = x_train_t @ weight + bias
        bce = nn.functional.binary_cross_entropy_with_logits(logits, y_train_t, reduction="none")
        loss = torch.sum(w_t * bce) + 0.5 * torch.sum(weight * weight)
        loss.backward()
        return loss

    final_loss = optimizer.step(closure)
    grad_norm = float(torch.sqrt(weight.grad.pow(2).sum() + bias.grad.pow(2).sum()).item())
    converged = grad_norm < 1e-2

    with torch.no_grad():
        val_logits = (torch.tensor(x_val, dtype=torch.float64, device=device) @ weight + bias).cpu().numpy()
        test_logits = (torch.tensor(x_test, dtype=torch.float64, device=device) @ weight + bias).cpu().numpy()
    val_prob = sigmoid(val_logits)
    test_prob = sigmoid(test_logits)
    threshold, _ = find_best_threshold(val_prob, y_val, objective="balanced_accuracy")

    fit_log = {
        "converged": bool(converged),
        "final_grad_norm": grad_norm,
        "final_loss": float(final_loss.item()) if final_loss is not None else None,
        "n_train": n_total,
        "n_pos": float(y_train.sum()),
        "n_neg": n_total - float(y_train.sum()),
    }
    test_sample_ids = [s.video_id for s in test_samples]
    return threshold, test_prob, y_test, test_sample_ids, fit_log


def run_logistic_evaluation(
    feature_dir: Path,
    folds: list[dict],
    labels: dict[str, int],
    manifest_rows: dict[str, dict],
    device: "torch.device",
    output_path: Path,
    fit_log_path: Path,
) -> dict:
    all_rows: list[dict] = []
    fit_logs: dict[str, dict] = {}
    for fold in folds:
        threshold, probabilities, labels_arr, sample_ids, fit_log = train_logistic_fold(
            feature_dir, fold["train_ids"], fold["val_ids"], fold["test_ids"], labels, device
        )
        if len(sample_ids) != len(probabilities):
            raise SystemExit(f"Logistic prediction/id misalignment for test_subject {fold['test_subject']}.")
        all_rows.extend(build_oof_rows(fold, sample_ids, probabilities, labels_arr, threshold, 0, manifest_rows))
        fit_logs[fold["test_subject"]] = fit_log
    written = write_oof_csv(output_path, all_rows)
    fit_log_path.parent.mkdir(parents=True, exist_ok=True)
    with fit_log_path.open("w", encoding="utf-8") as handle:
        json.dump(fit_logs, handle, indent=2, sort_keys=True)
    return {"rows": written, "fit_logs": fit_logs}


def run_evaluate(
    run_dir: Path,
    arm: str,
    manifest_path: Path,
    labels_path: Path,
    config: "loso.FoldConfig",
    device: "torch.device",
    seeds: Sequence[int] = SEEDS,
    run_nulls: bool = False,
    null_seeds: Sequence[int] = NULL_SEEDS,
    run_no_p10_training: bool = False,
) -> dict:
    """Frozen MLP readout + logistic control, plus opt-in null/no-P10 sensitivity runs."""
    feature_dir = features_arm_dir(run_dir, arm)
    if not feature_dir.exists():
        raise SystemExit(f"No materialized features for arm {arm!r} at {feature_dir}. Run `materialize` first.")

    labels = {key: int(value) for key, value in json.load(labels_path.open(encoding="utf-8")).items()}
    manifest_rows = {row["sample_id"]: row for row in load_manifest(manifest_path)}
    folds = load_folds(folds_path(run_dir))

    report: dict = {"arm": arm, "mlp": [], "logistic": None, "nulls": [], "no_p10_training": []}

    feature_dim = None
    with np.load(next(feature_dir.rglob("*.npz")), allow_pickle=False) as data:
        feature_dim = int(data["video_feature"].shape[0])
    report["mlp_parameter_count"] = mlp_parameter_count(feature_dim, config)
    report["feature_dim"] = feature_dim

    for seed in seeds:
        result = run_mlp_evaluation(
            feature_dir, folds, labels, manifest_rows, config, device, seed, oof_seed_path(run_dir, arm, seed)
        )
        report["mlp"].append(result)

    logistic_result = run_logistic_evaluation(
        feature_dir,
        folds,
        labels,
        manifest_rows,
        device,
        oof_seed_path(run_dir, f"{arm}_logreg", 0),
        oof_dir(run_dir, f"{arm}_logreg") / "fit_log.json",
    )
    report["logistic"] = logistic_result

    if run_nulls:
        metadata = load_metadata(manifest_path)
        subject_samples = loso.subjects_to_samples(manifest_path)
        for null_seed in null_seeds:
            permuted_labels = permute_labels_within_subject(labels, subject_samples, metadata, null_seed)
            result = run_mlp_evaluation(
                feature_dir,
                folds,
                permuted_labels,
                manifest_rows,
                config,
                device,
                null_seed,
                oof_seed_path(run_dir, f"{arm}_null{null_seed}", null_seed),
            )
            report["nulls"].append(result)

    if run_no_p10_training:
        folds_sensitivity = load_folds(folds_path(run_dir, no_p10=True))
        for seed in seeds:
            result = run_mlp_evaluation(
                feature_dir,
                folds_sensitivity,
                labels,
                manifest_rows,
                config,
                device,
                seed,
                oof_seed_path(run_dir, f"{arm}_nop10", seed),
            )
            report["no_p10_training"].append(result)

    clear_feature_cache()
    return report


# --------------------------------------------------------------------------- #
# report: paired-subject statistics                                          #
# --------------------------------------------------------------------------- #


def load_arm_oof(run_dir: Path, name: str) -> list[dict] | None:
    directory = oof_dir(run_dir, name)
    if not directory.exists():
        return None
    rows: list[dict] = []
    found = False
    for path in sorted(directory.glob("seed*.csv")):
        found = True
        rows.extend(read_oof_csv(path))
    return rows if found else None


def group_by_subject_seed(oof_rows: list[dict]) -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in oof_rows:
        grouped[(row["test_subject"], row["seed"])].append(row)
    return grouped


def fold_metrics(rows: list[dict]) -> dict[str, float]:
    thresholds = {row["threshold"] for row in rows}
    if len(thresholds) != 1:
        raise ValueError(f"Inconsistent threshold within one (test_subject, seed) group: {thresholds}")
    threshold = thresholds.pop()
    probabilities = np.array([row["probability"] for row in rows])
    labels = np.array([row["label"] for row in rows])
    return compute_metrics(probabilities, labels, threshold=threshold)


def subject_mean_metric(oof_rows: list[dict], subjects: Sequence[str], metric: str) -> dict[str, float]:
    """Metric per fold -> mean over seeds within subject, for the given subjects only."""
    grouped = group_by_subject_seed(oof_rows)
    per_subject: dict[str, list[float]] = defaultdict(list)
    for (subject, _seed), rows in grouped.items():
        if subject in subjects:
            per_subject[subject].append(fold_metrics(rows)[metric])
    return {subject: float(np.mean(values)) for subject, values in per_subject.items()}


def sign_flip_p_value(deltas: Sequence[float]) -> dict:
    """Exact two-sided paired sign-flip test, enumerating all 2^n sign assignments.

    Both the observed statistic and every candidate are ``mean(signs * deltas)``, so
    the all-positive assignment is bit-identical to the observed value and the
    ``>=`` comparison needs no epsilon for that case; a small tolerance elsewhere
    only guards float noise, not the test's exactness. Zero deltas are unaffected by
    any sign flip, so an all-zero input yields p = 1, as the plan requires.
    """
    deltas_arr = np.asarray(deltas, dtype=float)
    n = len(deltas_arr)
    observed = float(np.mean(deltas_arr))
    observed_abs = abs(observed)
    total = 0
    extreme = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        candidate = float(np.mean(np.asarray(signs) * deltas_arr))
        total += 1
        if abs(candidate) >= observed_abs - 1e-12:
            extreme += 1
    return {
        "observed_mean_delta": observed,
        "n_assignments": total,
        "n_at_least_as_extreme": extreme,
        "p_value": extreme / total,
    }


def paired_deltas(candidate: dict[str, float], baseline: dict[str, float], subjects: Sequence[str]) -> list[float]:
    missing = [s for s in subjects if s not in candidate or s not in baseline]
    if missing:
        raise ValueError(f"Missing subjects for paired contrast: {missing}")
    return [candidate[s] - baseline[s] for s in subjects]


def contrast_summary(candidate: dict[str, float], baseline: dict[str, float], subjects: Sequence[str] = PRIMARY_SUBJECTS) -> dict:
    deltas = paired_deltas(candidate, baseline, subjects)
    mean_delta = float(np.mean(deltas))
    sd_delta = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    n_positive = int(sum(delta > 0 for delta in deltas))
    ci = bootstrap_interval(deltas, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    sign_flip = sign_flip_p_value(deltas)
    return {
        "subjects": list(subjects),
        "candidate": {s: candidate[s] for s in subjects},
        "baseline": {s: baseline[s] for s in subjects},
        "candidate_mean": float(np.mean([candidate[s] for s in subjects])),
        "candidate_sd": float(np.std([candidate[s] for s in subjects], ddof=1)) if len(subjects) > 1 else 0.0,
        "baseline_mean": float(np.mean([baseline[s] for s in subjects])),
        "baseline_sd": float(np.std([baseline[s] for s in subjects], ddof=1)) if len(subjects) > 1 else 0.0,
        "deltas": deltas,
        "mean_delta": mean_delta,
        "sd_delta": sd_delta,
        "n_positive": n_positive,
        "n_subjects": len(subjects),
        "bootstrap_ci_95": ci,
        "sign_flip": sign_flip,
    }


def _camera_averaged_repetition_scores(oof_rows: list[dict], seeds: Sequence[int]) -> dict[str, dict]:
    """Group camera rows into one score per repetition per seed, keyed by ``repetition_id``.

    Reproduces ``videomae_identity_control.repetition_scores`` for this module's OOF
    schema, which already carries ``repetition_id``/``session_id`` at write time
    instead of the raw ``video_id``/``repetition_number`` columns that function reads.
    """
    grouped: dict[str, dict] = {}
    for row in oof_rows:
        key = row["repetition_id"]
        entry = grouped.setdefault(
            key,
            {
                "session": row["session_id"],
                "person_id": row["person_id"],
                "exercise_id": row["exercise_id"],
                "label": row["label"],
                "by_seed": defaultdict(list),
            },
        )
        if entry["label"] != row["label"]:
            raise ValueError(f"Repetition {key} carries disagreeing labels across camera rows.")
        entry["by_seed"][row["seed"]].append(row["probability"])

    for key, entry in grouped.items():
        for seed in seeds:
            probabilities = entry["by_seed"].get(seed, [])
            if len(probabilities) != 2:
                raise ValueError(
                    f"Repetition {key} has {len(probabilities)} rows for seed {seed}, expected 2 (camera pair)."
                )
        entry["score_by_seed"] = {seed: float(np.mean(entry["by_seed"][seed])) for seed in seeds}
        del entry["by_seed"]
    return grouped


def within_session_auc_statistic(oof_rows: list[dict], seeds: Sequence[int], drop_subject: str | None = SENSITIVITY_SUBJECT) -> dict:
    """Within-session ROC-AUC, subject-macro averaged -- reuses the identity-control hierarchy verbatim.

    Returns ``{"status": "pending", ...}`` rather than calling ``observed_statistic`` on
    zero sessions: an empty session list would otherwise reduce to ``np.mean([])``,
    which is a silent nan plus a RuntimeWarning, not a meaningful "AUC".
    """
    scores = _camera_averaged_repetition_scores(oof_rows, seeds)
    sessions, single_class = build_sessions(scores, seeds, drop_subject=drop_subject)
    if not sessions:
        return {"status": "pending", "reason": "no mixed-label sessions available", "single_class_sessions_excluded": single_class}
    statistic = observed_statistic(sessions)
    statistic["single_class_sessions_excluded"] = single_class
    statistic["status"] = "complete"
    return statistic


def pooled_pairwise_auc(oof_rows: list[dict], subjects: Sequence[str]) -> float | None:
    """Pooled ROC-AUC over the given subjects, or ``None`` if it is undefined.

    ``pairwise_auc`` divides by ``len(positives) * len(negatives)``; an empty or
    single-class pool would divide by zero rather than silently return NaN, so this
    guards it explicitly and returns ``None`` -- the same "undefined, not zero"
    convention ``stratum_metrics`` uses.
    """
    rows = [row for row in oof_rows if row["test_subject"] in subjects]
    if not rows:
        return None
    labels = np.array([row["label"] for row in rows], dtype=int)
    if len(np.unique(labels)) < 2:
        return None
    probabilities = np.array([row["probability"] for row in rows], dtype=float)
    return pairwise_auc(labels, probabilities)


def stratum_metrics(rows: list[dict], min_samples: int = MIN_STRATUM_SAMPLES) -> dict | str:
    """Balanced accuracy for one stratum at each row's own fold threshold.

    Strata cut across folds (different thresholds), so a single ``compute_metrics``
    call with one threshold is the wrong tool; predictions are formed per-row against
    that row's own fold threshold and then reduced by hand.
    """
    if len(rows) < min_samples:
        return "undefined"
    labels = np.array([row["label"] for row in rows])
    if len(np.unique(labels)) < 2:
        return "undefined"
    preds = np.array([1 if row["probability"] >= row["threshold"] else 0 for row in rows])
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"n": len(rows), "balanced_accuracy": (recall + specificity) / 2.0}


def strata_by(oof_rows: list[dict], subjects: Sequence[str], key: str) -> dict[str, dict | str]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in oof_rows:
        if row["test_subject"] in subjects:
            buckets[row[key]].append(row)
    return {stratum: stratum_metrics(rows) for stratum, rows in sorted(buckets.items())}


def padding_bucket(mean_padding_fraction: float) -> str:
    if mean_padding_fraction <= 0.0:
        return "0"
    if mean_padding_fraction <= 0.5:
        return "(0, 0.5]"
    return ">0.5"


def padding_strata(run_dir: Path, arm: str, oof_rows: list[dict], subjects: Sequence[str]) -> dict[str, dict | str]:
    """Bucket samples by their raw bundle's mean ``padding_fraction`` across clips."""
    arm_dir = raw_arm_dir(run_dir, arm)
    if not arm_dir.exists():
        return {"pending": "raw bundles not available"}
    stems = index_raw_bundles(arm_dir)
    bucket_of: dict[str, str] = {}
    for sample_id, paths in stems.items():
        with np.load(paths[0], allow_pickle=False) as data:
            if "padding_fraction" not in data.files:
                continue
            bucket_of[sample_id] = padding_bucket(float(np.mean(data["padding_fraction"])))

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in oof_rows:
        if row["test_subject"] in subjects and row["sample_id"] in bucket_of:
            buckets[bucket_of[row["sample_id"]]].append(row)
    return {stratum: stratum_metrics(rows) for stratum, rows in sorted(buckets.items())}


def null_check_summary(run_dir: Path, arm: str, subjects: Sequence[str] = PRIMARY_SUBJECTS) -> dict:
    per_seed: dict[str, dict] = {}
    for null_seed in NULL_SEEDS:
        rows = load_arm_oof(run_dir, f"{arm}_null{null_seed}")
        if rows is None:
            per_seed[str(null_seed)] = {"status": "pending"}
            continue
        by_subject = subject_mean_metric(rows, subjects, "balanced_accuracy")
        missing = [s for s in subjects if s not in by_subject]
        mean_ba = float(np.mean([by_subject[s] for s in subjects if s in by_subject])) if by_subject else None
        per_seed[str(null_seed)] = {
            "mean_balanced_accuracy": mean_ba,
            "missing_subjects": missing,
            "audit_flag": bool(mean_ba is not None and mean_ba > NULL_AUDIT_THRESHOLD),
        }
    return per_seed


def p10_sensitivities(run_dir: Path, arm: str) -> dict:
    result: dict = {"p10_test_fold": {"status": "pending"}, "no_p10_training": {"status": "pending"}}
    primary_rows = load_arm_oof(run_dir, arm)
    if primary_rows is not None:
        p10_rows = [row for row in primary_rows if row["test_subject"] == SENSITIVITY_SUBJECT]
        if p10_rows:
            by_seed = subject_mean_metric(p10_rows, (SENSITIVITY_SUBJECT,), "balanced_accuracy")
            result["p10_test_fold"] = {"balanced_accuracy": by_seed.get(SENSITIVITY_SUBJECT)}

    nop10_rows = load_arm_oof(run_dir, f"{arm}_nop10")
    if nop10_rows is not None:
        by_subject = subject_mean_metric(nop10_rows, PRIMARY_SUBJECTS, "balanced_accuracy")
        if len(by_subject) == len(PRIMARY_SUBJECTS):
            result["no_p10_training"] = {
                "mean_balanced_accuracy": float(np.mean(list(by_subject.values()))),
                "per_subject": by_subject,
            }
    return result


def arm_descriptive_metrics(run_dir: Path, arm: str, oof_rows: list[dict], subjects: Sequence[str] = PRIMARY_SUBJECTS) -> dict:
    ba_by_subject = subject_mean_metric(oof_rows, subjects, "balanced_accuracy")
    f1_by_subject = subject_mean_metric(oof_rows, subjects, "macro_f1")
    recall_by_subject = subject_mean_metric(oof_rows, subjects, "recall")
    specificity_by_subject = subject_mean_metric(oof_rows, subjects, "specificity")
    seeds_present = sorted({row["seed"] for row in oof_rows if row["test_subject"] in subjects})
    return {
        "balanced_accuracy": {
            "per_subject": ba_by_subject,
            "mean": float(np.mean(list(ba_by_subject.values()))) if ba_by_subject else None,
            "sd": float(np.std(list(ba_by_subject.values()), ddof=1)) if len(ba_by_subject) > 1 else 0.0,
        },
        "macro_f1_mean": float(np.mean(list(f1_by_subject.values()))) if f1_by_subject else None,
        "recall_mean": float(np.mean(list(recall_by_subject.values()))) if recall_by_subject else None,
        "specificity_mean": float(np.mean(list(specificity_by_subject.values()))) if specificity_by_subject else None,
        "pooled_roc_auc": pooled_pairwise_auc(oof_rows, subjects) if oof_rows else None,
        "by_exercise": strata_by(oof_rows, subjects, "exercise_id"),
        "by_camera": strata_by(oof_rows, subjects, "camera"),
        "padding_strata": padding_strata(run_dir, arm, oof_rows, subjects),
        "p10_sensitivity": p10_sensitivities(run_dir, arm),
        "null_check": null_check_summary(run_dir, arm, subjects),
        "seeds_present": seeds_present,
    }


def verdict_conditions(
    mean_delta: float,
    n_positive: int,
    n_subjects: int,
    ci: list[float],
    p_value: float,
    gates_pass: bool,
    margin: float = PRACTICAL_MARGIN,
    alpha: float = ALPHA,
) -> dict:
    """Plan "Primary endpoint and inference" outcome table, evaluated exactly as written.

    The plan's fifth outcome row -- "Any required data/model gate fails" -- is
    checked FIRST and short-circuits every significance branch: a failed gate means
    "no valid primary comparison until repaired and documented", not a downgraded
    but still-reportable significant result (review defect #3).
    """
    ci_excludes_zero = ci[0] > 0 or ci[1] < 0
    conditions = {
        "mean_delta_at_least_margin": mean_delta >= margin,
        "at_least_7_of_9_positive": n_positive / n_subjects >= 7 / 9 - 1e-9,
        "ci_excludes_zero": ci_excludes_zero,
        "p_below_alpha": p_value < alpha,
        "gates_pass": gates_pass,
    }
    if not gates_pass:
        verdict = "gate failure: no valid primary comparison"
    else:
        significant = conditions["ci_excludes_zero"] and conditions["p_below_alpha"]
        if all(conditions.values()):
            verdict = "promising improvement"
        elif significant and mean_delta > 0:
            verdict = "significant positive below margin"
        elif significant and mean_delta < 0:
            verdict = "significant negative"
        else:
            verdict = "undetermined"
    return {"conditions": conditions, "verdict": verdict}


def run_report(run_dir: Path) -> dict:
    arm_oof = {arm: load_arm_oof(run_dir, arm) for arm in ARMS}
    available_arms = [arm for arm, rows in arm_oof.items() if rows is not None]

    ba_by_arm: dict[str, dict[str, float]] = {}
    auc_stat_by_arm: dict[str, dict] = {}
    descriptive: dict[str, dict] = {}
    for arm in available_arms:
        rows = arm_oof[arm]
        ba_by_arm[arm] = subject_mean_metric(rows, PRIMARY_SUBJECTS, "balanced_accuracy")
        seeds_present = sorted({row["seed"] for row in rows})
        try:
            auc_stat_by_arm[arm] = within_session_auc_statistic(rows, seeds_present)
        except ValueError as exc:  # camera pairing incomplete for this arm
            auc_stat_by_arm[arm] = {"status": "pending", "reason": str(exc)}
        descriptive[arm] = arm_descriptive_metrics(run_dir, arm, rows)

    logreg_ba_by_arm: dict[str, dict[str, float]] = {}
    for arm in available_arms:
        rows = load_arm_oof(run_dir, f"{arm}_logreg")
        if rows is not None:
            logreg_ba_by_arm[arm] = subject_mean_metric(rows, PRIMARY_SUBJECTS, "balanced_accuracy")

    gates = compute_gates(run_dir)

    primary: dict = {"status": "pending"}
    candidate_arm, baseline_arm = PRIMARY_CONTRAST
    if candidate_arm in ba_by_arm and baseline_arm in ba_by_arm:
        summary = contrast_summary(ba_by_arm[candidate_arm], ba_by_arm[baseline_arm])
        summary["verdict"] = verdict_conditions(
            summary["mean_delta"],
            summary["n_positive"],
            summary["n_subjects"],
            summary["bootstrap_ci_95"],
            summary["sign_flip"]["p_value"],
            gates.get("passed", False),
        )
        summary["status"] = "complete"
        primary = summary

    secondary: dict = {}
    raw_p_values: dict[str, float] = {}
    for name, kind, candidate, baseline in SECONDARY_FAMILY:
        if kind == "ba" and candidate in ba_by_arm and baseline in ba_by_arm:
            summary = contrast_summary(ba_by_arm[candidate], ba_by_arm[baseline])
            secondary[name] = {**summary, "status": "complete"}
            raw_p_values[name] = summary["sign_flip"]["p_value"]
        elif kind == "within_session_auc" and candidate in auc_stat_by_arm and baseline in auc_stat_by_arm:
            cand_stat = auc_stat_by_arm[candidate]
            base_stat = auc_stat_by_arm[baseline]
            if "per_subject_auc" in cand_stat and "per_subject_auc" in base_stat:
                shared = sorted(set(cand_stat["per_subject_auc"]) & set(base_stat["per_subject_auc"]), key=int)
                summary = contrast_summary(cand_stat["per_subject_auc"], base_stat["per_subject_auc"], shared)
                secondary[name] = {**summary, "status": "complete"}
                raw_p_values[name] = summary["sign_flip"]["p_value"]
            else:
                secondary[name] = {"status": "pending"}
        elif kind == "logistic_ba" and candidate in logreg_ba_by_arm and baseline in logreg_ba_by_arm:
            summary = contrast_summary(logreg_ba_by_arm[candidate], logreg_ba_by_arm[baseline])
            secondary[name] = {**summary, "status": "complete"}
            raw_p_values[name] = summary["sign_flip"]["p_value"]
        else:
            secondary[name] = {"status": "pending"}

    if len(raw_p_values) == len(SECONDARY_FAMILY):
        holm = holm_correct(raw_p_values)
        holm_status = "complete"
    else:
        holm = {name: {"raw": p} for name, p in raw_p_values.items()}
        holm_status = "pending (secondary family incomplete: only {}/{} tests available)".format(
            len(raw_p_values), len(SECONDARY_FAMILY)
        )

    summary_payload = {
        "arms_available": available_arms,
        "primary": primary,
        "secondary": secondary,
        "holm": {"status": holm_status, "corrected": holm},
        "descriptive": descriptive,
        "gates": gates,
    }
    with summary_path(run_dir).open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)
    return summary_payload


# --------------------------------------------------------------------------- #
# gates                                                                        #
# --------------------------------------------------------------------------- #


def gate_feature_integrity(run_dir: Path) -> dict:
    per_arm: dict[str, dict] = {}
    for arm in ARMS:
        path = materialize_audit_path(run_dir, arm)
        if not path.exists():
            per_arm[arm] = {"status": "pending"}
            continue
        audit = json.load(path.open(encoding="utf-8"))
        per_arm[arm] = {"status": "pass" if audit.get("passed") else "fail", "evidence": audit}
    statuses = {info["status"] for info in per_arm.values()}
    overall = "fail" if "fail" in statuses else ("pending" if "pending" in statuses else "pass")
    return {"status": overall, "evidence": per_arm}


def gate_fold_purity(run_dir: Path) -> dict:
    problems: list[str] = []
    for label, path in (("folds", folds_path(run_dir)), ("folds_no_p10", folds_path(run_dir, no_p10=True))):
        if not path.exists():
            return {"status": "pending", "evidence": f"{path} missing"}
        for fold in load_folds(path):
            train, val, test = set(fold["train_ids"]), set(fold["val_ids"]), set(fold["test_ids"])
            if train & val:
                problems.append(f"{label}: P{fold['test_subject']} train/val overlap")
            if train & test:
                problems.append(f"{label}: P{fold['test_subject']} train/test overlap")
            if val & test:
                problems.append(f"{label}: P{fold['test_subject']} val/test overlap")
            if fold["val_subject"] == fold["test_subject"]:
                problems.append(f"{label}: P{fold['test_subject']} val equals test")
            if fold["val_subject"] == SENSITIVITY_SUBJECT:
                problems.append(f"{label}: P{fold['test_subject']} chose P10 as validation subject")
    return {"status": "fail" if problems else "pass", "evidence": problems}


def gate_oof_integrity(run_dir: Path, manifest_path: Path, labels_path: Path) -> dict:
    if not manifest_path.exists():
        return {"status": "pending", "evidence": "manifest not available"}
    manifest_rows = {row["sample_id"]: row for row in load_manifest(manifest_path)}
    labels = {key: int(value) for key, value in json.load(labels_path.open(encoding="utf-8")).items()}
    per_arm: dict[str, dict] = {}
    for arm in ARMS:
        rows = load_arm_oof(run_dir, arm)
        if rows is None:
            per_arm[arm] = {"status": "pending"}
            continue
        problems: list[str] = []
        by_seed: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            by_seed[row["seed"]].append(row)
        for seed, seed_rows in by_seed.items():
            ids = [row["sample_id"] for row in seed_rows]
            duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
            if duplicates:
                problems.append(f"seed {seed}: duplicate sample ids {duplicates[:5]}")
            expected = set(manifest_rows)
            missing = sorted(expected - set(ids))
            if missing:
                problems.append(f"seed {seed}: missing sample ids {missing[:5]}")
            for row in seed_rows:
                if row["sample_id"] not in manifest_rows:
                    continue
                if manifest_rows[row["sample_id"]]["person_id"] != row["test_subject"]:
                    problems.append(f"seed {seed}: {row['sample_id']} scored outside its own test fold")
                if labels.get(row["sample_id"]) != row["label"]:
                    problems.append(f"seed {seed}: {row['sample_id']} label disagrees with frozen labels")
            by_rep: dict[str, set[str]] = defaultdict(set)
            for row in seed_rows:
                by_rep[row["repetition_id"]].add(row["camera"])
            incomplete = sorted(key for key, cams in by_rep.items() if cams != set(CAMERAS))
            if incomplete:
                problems.append(f"seed {seed}: repetitions missing a camera {incomplete[:5]}")
        per_arm[arm] = {"status": "fail" if problems else "pass", "evidence": problems}
    statuses = {info["status"] for info in per_arm.values()}
    overall = "fail" if "fail" in statuses else ("pending" if "pending" in statuses else "pass")
    return {"status": overall, "evidence": per_arm}


def gate_model_identity(run_dir: Path) -> dict:
    per_arm: dict[str, dict] = {}
    frame_index_bundles: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        path = materialize_audit_path(run_dir, arm)
        if not path.exists():
            per_arm[arm] = {"status": "pending"}
            continue
        audit = json.load(path.open(encoding="utf-8"))
        provenance = audit.get("provenance", {})
        expected = ARM_MODEL_CONFIG[arm]
        mismatches = {
            key: {"expected": value, "actual": provenance.get(key)}
            for key, value in expected.items()
            if provenance.get(key) != value
        }
        # Compared against the MEASURED dimension (the actual array shape recorded by
        # materialize), not ``expected_dim`` -- that field is the arm's declared config
        # echoed back verbatim, so comparing it to ``ARM_DIMS[arm]`` can never fail
        # (review defect #2).
        if audit.get("measured_dim") != ARM_DIMS[arm]:
            mismatches["dim"] = {"expected": ARM_DIMS[arm], "actual": audit.get("measured_dim")}
        per_arm[arm] = {"status": "fail" if mismatches else "pass", "evidence": mismatches}

        arm_dir = raw_arm_dir(run_dir, arm)
        if arm_dir.exists():
            frame_index_bundles[arm] = {}
            for sample_id, paths in index_raw_bundles(arm_dir).items():
                with np.load(paths[0], allow_pickle=False) as data:
                    if "frame_indices" in data.files:
                        frame_index_bundles[arm][sample_id] = np.asarray(data["frame_indices"])

    matched_frames_status = "pending"
    matched_frames_evidence: dict = {}
    if "vm16" in frame_index_bundles and "vj16" in frame_index_bundles:
        shared = sorted(set(frame_index_bundles["vm16"]) & set(frame_index_bundles["vj16"]))
        mismatched = [
            sample_id
            for sample_id in shared
            if not np.array_equal(frame_index_bundles["vm16"][sample_id], frame_index_bundles["vj16"][sample_id])
        ]
        matched_frames_status = "pass" if shared and not mismatched else ("fail" if mismatched else "pending")
        matched_frames_evidence = {"n_compared": len(shared), "mismatched_sample_ids": mismatched[:5]}

    statuses = {info["status"] for info in per_arm.values()} | {matched_frames_status}
    overall = "fail" if "fail" in statuses else ("pending" if "pending" in statuses else "pass")
    return {
        "status": overall,
        "evidence": {"per_arm": per_arm, "vm16_vj16_frame_indices_match": {"status": matched_frames_status, **matched_frames_evidence}},
    }


def manifest_frame_bounds(manifest_path: Path) -> dict[str, tuple[int, int]]:
    """Decoder-index bounds per sample, from the frozen manifest alone.

    ``first_frame``/``last_frame`` are the dataset's 1-based annotation frame
    numbers; the 0-based decoder index range is ``first_frame - 1 .. last_frame - 1``
    inclusive, with the upper bound clamped up to the lower one for a (degenerate)
    single-frame repetition.
    """
    bounds: dict[str, tuple[int, int]] = {}
    for row in load_manifest(manifest_path):
        first_index = int(row["first_frame"]) - 1
        last_index = max(int(row["last_frame"]) - 1, first_index)
        bounds[row["sample_id"]] = (first_index, last_index)
    return bounds


def gate_sampling(run_dir: Path, manifest_path: Path) -> dict:
    """Every sampled frame index falls inside the REPETITION'S OWN manifest bounds.

    Reads ``first_frame``/``last_frame`` from the frozen manifest rather than the raw
    bundle's own ``first_index``/``last_index`` fields: checking a bundle's indices
    against bounds the same bundle also wrote would pass a buggy extractor that
    writes self-consistent-but-wrong bounds (review defect #1). The bundle's stored
    bounds are then separately asserted to agree with the manifest-derived ones, so a
    silent divergence between what the extractor recorded and the frozen annotation
    is itself a failure, not just an unmeasured index.
    """
    if not manifest_path.exists():
        return {"status": "pending", "evidence": "manifest not available"}
    bounds = manifest_frame_bounds(manifest_path)

    per_arm: dict[str, dict] = {}
    any_checked = False
    for arm in ARMS:
        arm_dir = raw_arm_dir(run_dir, arm)
        if not arm_dir.exists():
            per_arm[arm] = {"status": "pending"}
            continue
        out_of_bounds: list[str] = []
        stored_bounds_disagree: list[str] = []
        for sample_id, paths in index_raw_bundles(arm_dir).items():
            manifest_bound = bounds.get(sample_id)
            if manifest_bound is None:
                continue
            expected_first, expected_last = manifest_bound
            with np.load(paths[0], allow_pickle=False) as data:
                if "frame_indices" not in data.files:
                    continue
                any_checked = True
                indices = np.asarray(data["frame_indices"])
                if int(indices.min()) < expected_first or int(indices.max()) > expected_last:
                    out_of_bounds.append(sample_id)
                if {"first_index", "last_index"} <= set(data.files):
                    stored_first = int(data["first_index"])
                    stored_last = int(data["last_index"])
                    if stored_first != expected_first or stored_last != expected_last:
                        stored_bounds_disagree.append(sample_id)
        per_arm[arm] = {
            "status": "fail" if (out_of_bounds or stored_bounds_disagree) else "pass",
            "evidence": {
                "out_of_manifest_bounds_sample_ids": out_of_bounds[:5],
                "stored_bounds_disagree_with_manifest_sample_ids": stored_bounds_disagree[:5],
            },
        }
    if not any_checked:
        return {"status": "pending", "evidence": per_arm}
    statuses = {info["status"] for info in per_arm.values()}
    overall = "fail" if "fail" in statuses else ("pending" if "pending" in statuses else "pass")
    return {"status": overall, "evidence": per_arm}


def compute_gates(run_dir: Path, manifest_path: Path | None = None, labels_path: Path | None = None) -> dict:
    manifest_path = manifest_path or DEFAULT_PROCESSED_ROOT / "manifest.csv"
    labels_path = labels_path or DEFAULT_PROCESSED_ROOT / "labels" / "correctness.json"
    config = json.load(run_config_path(run_dir).open(encoding="utf-8")) if run_config_path(run_dir).exists() else {}
    if "manifest_path" in config:
        manifest_path = Path(config["manifest_path"])

    gates = {
        "feature_integrity": gate_feature_integrity(run_dir),
        "fold_purity": gate_fold_purity(run_dir),
        "oof_integrity": gate_oof_integrity(run_dir, manifest_path, labels_path) if labels_path.exists() else {"status": "pending"},
        "model_identity": gate_model_identity(run_dir),
        "sampling": gate_sampling(run_dir, manifest_path),
    }
    statuses = {gate["status"] for gate in gates.values()}
    passed = statuses == {"pass"}
    payload = {"gates": gates, "passed": passed}
    with gates_path(run_dir).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--labels", type=Path, default=DEFAULT_PROCESSED_ROOT / "labels" / "correctness.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen VideoMAE-vs-V-JEPA-2 comparison on REHAB24-6.")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="Freeze manifest audit, folds and run config.")
    add_common(init_parser)
    init_parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))

    materialize_parser = sub.add_parser("materialize", help="Raw per-clip bundles -> video_feature bundles.")
    add_common(materialize_parser)
    materialize_parser.add_argument("--arm", required=True, choices=list(ARMS))

    evaluate_parser = sub.add_parser("evaluate", help="MLP readout, logistic control, nulls, no-P10 sensitivity.")
    add_common(evaluate_parser)
    evaluate_parser.add_argument("--arm", required=True, choices=list(ARMS))
    evaluate_parser.add_argument("--device", type=str, default=None)
    evaluate_parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    evaluate_parser.add_argument("--epochs", type=int, default=loso.FoldConfig().epochs)
    evaluate_parser.add_argument("--batch-size", type=int, default=loso.FoldConfig().batch_size)
    evaluate_parser.add_argument("--lr", type=float, default=loso.FoldConfig().lr)
    evaluate_parser.add_argument("--hidden-dim", type=int, default=loso.FoldConfig().hidden_dim)
    evaluate_parser.add_argument("--dropout", type=float, default=loso.FoldConfig().dropout)
    evaluate_parser.add_argument("--weight-decay", type=float, default=loso.FoldConfig().weight_decay)
    evaluate_parser.add_argument("--early-stopping-patience", type=int, default=loso.FoldConfig().early_stopping_patience)
    evaluate_parser.add_argument("--nulls", action="store_true")
    evaluate_parser.add_argument("--no-p10-training", action="store_true", dest="no_p10_training")

    report_parser = sub.add_parser("report", help="Paired-subject statistics, gates and summary.json.")
    report_parser.add_argument("--run-dir", type=Path, required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "init":
        result = run_init(args.run_dir, args.manifest, args.labels, args.arms)
        audit = result["manifest_audit"]
        print(f"manifest: {audit['rows']} rows, {audit['repetitions']} repetitions, {audit['sessions']} sessions")
        print(f"{'test_subject':<14}{'val_subject'}")
        for fold in result["folds"]:
            print(f"P{fold['test_subject']:<13}P{fold['val_subject']}")
        print(f"\nWrote run config, manifest audit and folds to {args.run_dir}")
        return

    if args.command == "materialize":
        audit = materialize(args.run_dir, args.arm, args.manifest)
        print(f"materialized {audit['written']} bundles for arm {args.arm} -> {features_arm_dir(args.run_dir, args.arm)}")
        return

    if args.command == "evaluate":
        device = resolve_device(args.device)
        config = loso.FoldConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping_patience,
        )
        report = run_evaluate(
            args.run_dir,
            args.arm,
            args.manifest,
            args.labels,
            config,
            device,
            seeds=args.seeds,
            run_nulls=args.nulls,
            run_no_p10_training=args.no_p10_training,
        )
        print(f"evaluated arm {args.arm}: {len(report['mlp'])} MLP seed(s), logistic {'ok' if report['logistic'] else 'skipped'}")
        return

    if args.command == "report":
        summary = run_report(args.run_dir)
        print(f"arms available: {summary['arms_available']}")
        print(f"gates passed: {summary['gates']['passed']}")
        if summary["primary"].get("status") == "complete":
            print(
                f"primary vj64-vm16: mean delta {summary['primary']['mean_delta']:+.4f}  "
                f"verdict={summary['primary']['verdict']['verdict']}"
            )
        else:
            print("primary contrast: pending (vm16 and/or vj64 not evaluated yet)")
        print(f"\nWrote {summary_path(args.run_dir)}")
        return


if __name__ == "__main__":
    main()
