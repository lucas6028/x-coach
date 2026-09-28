"""Calibrated late fusion of the NLF pose branch and the VideoMAE branch on REHAB24-6.

Pre-registered in ``notes/rehab24_nlf_videomae_late_fusion_validation_plan.md``. The
plan fixes one fusion rule, one primary contrast (``fused - nlf``), a stop rule and a
position control; this module is that plan and nothing more. Everything a reader could
otherwise choose after seeing a number -- seeds, margin, arm paths, fold and manifest
hashes, the permutation pairing, the weight grid -- is a constant below, so that
committing this file freezes the protocol.

The stages, in the order the plan runs them:

``init``      assert the three hashes, copy the frozen folds, write ``run_config.json``.
``features``  sample-by-sample id diff of both feature directories against the manifest.
``ab-check``  exporting trainer against the unmodified trainer (loaded from the plan
              commit's blob, not from the working tree), bit for bit.
``fit``       refit one branch on the frozen folds; held-out rows go to ``oof/<arm>/``,
              validation rows to ``oof_val/<arm>/``.
``fuse``      per fold and seed: Platt per branch on the validation subject, unweighted
              mean, threshold on the validation subject, score the held-out subject once.
              Then the leak audit, which has a numeric rule and no manual override.
``gates``     the eight blocking gates of the plan's Gates table.
``report``    primary, the Holm family S1-S4, the position control, the diagnostics.

``baseline-check`` compares a rerun of the stock ``loso_cross_validation`` command with
the historical NLF summary. It never blocks; a miss is logged as a deviation.

Validation rows live in a sibling ``oof_val/`` tree on purpose. The comparison module's
OOF gate flags duplicate sample ids and any row whose ``person_id`` differs from
``test_subject``, and every validation row is both.

Nothing here re-implements a fold loop, a threshold search, a calibrator or a paired
statistic. ``train_one_fold``, ``fit_platt``, ``fuse_probabilities``,
``find_best_threshold``, ``contrast_summary``, ``holm_correct`` and the position helpers
are imported and called unchanged. What is new: the per-fold wiring of those pieces under
leave-one-subject-out (``late_fusion.fuse_run`` assumes one global validation split,
which LOSO does not have), the joins that fail loudly where the position helpers skip
silently, and the gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Sequence

import numpy as np

from src.rehab24 import loso_cross_validation as loso
from src.rehab24 import video_backbone_comparison as vbc
from src.rehab24.dataset import CAMERAS, DEFAULT_PROCESSED_ROOT, load_manifest
from src.rehab24.videomae_identity_control import holm_correct, midranks
from src.rehab24.videomae_position_control import (
    DEFAULT_SEGMENTATION,
    EXPECTED_INTERLEAVED_SESSIONS,
    EXPECTED_MIXED_SESSIONS,
    PROBE_MIN_REPS,
    build_pair_sessions,
    build_repetitions,
    interleaved_mask,
    position_balanced_auc,
    position_rule_loso,
    probe_cells,
    rank_correlation,
)
from src.rehab24.videomae_stage_a import ordered_test_ids
from src.video.classification_metrics import find_best_threshold
from src.video.late_fusion import (
    PROBABILITY_EPS,
    PlattCalibration,
    SplitPredictions,
    align,
    fit_platt,
    fuse_probabilities,
)
from src.video.videomae_video_classifier import clear_feature_cache

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The pose-video fusion experiment requires `torch`.") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# frozen protocol (plan "Branches and arms", "Dataset and folds", "Fusion rule")  #
# --------------------------------------------------------------------------- #

PLAN_PATH = "notes/rehab24_nlf_videomae_late_fusion_validation_plan.md"

#: The run whose folds are reused verbatim and whose ``vm16`` features are the video branch.
SOURCE_RUN_DIR = DEFAULT_PROCESSED_ROOT / "video_backbone_comparison" / "20260918_vjepa2_vs_videomae"
DEFAULT_RUN_ROOT = DEFAULT_PROCESSED_ROOT / "pose_video_fusion"

POSE_BRANCH = "nlf"
VIDEO_BRANCH = "vm16"
BRANCHES: tuple[str, ...] = (POSE_BRANCH, VIDEO_BRANCH)
ARM_FEATURE_DIRS: dict[str, Path] = {
    POSE_BRANCH: DEFAULT_PROCESSED_ROOT / "nlf_parametric_3d2d_skeleton_features",
    VIDEO_BRANCH: SOURCE_RUN_DIR / "features" / "vm16",
}
ARM_DIMS: dict[str, int] = {POSE_BRANCH: 2160, VIDEO_BRANCH: 768}

FOLDS_SHA256 = "01ef289c38d680633f67180639d0e566d3bcdaeacd261b2c482d275d6de3a730"
FOLDS_NO_P10_SHA256 = "8a8bc9ebc5b8cb411e8554b9cb4843c418aaf53e06f91fa8fd42ab8fe36e16d9"
MANIFEST_MD5 = "b27e9069b3cc9a2906d57e446630abb1"

SEEDS: tuple[int, ...] = (42, 7, 1234)
PRACTICAL_MARGIN = 0.02
ALPHA = 0.05
MIN_POSITIVE_FRACTION = 7 / 9

#: Fused arms. ``share`` is always the pose branch's weight in the mean.
FUSED_ARM = "fused"
POSE_CALIBRATED_ARM = "nlf_cal"
VIDEO_CALIBRATED_ARM = "vm16_cal"
TUNED_ARM = "fused_tuned"
ARM_SHARES: dict[str, float] = {FUSED_ARM: 0.5, POSE_CALIBRATED_ARM: 1.0, VIDEO_CALIBRATED_ARM: 0.0}

#: Validation-tuned weight diagnostic: the pose share is WEIGHT_GRID[i] = i / 10.
WEIGHT_GRID: tuple[float, ...] = tuple(index / 10 for index in range(11))
EQUAL_SHARE_INDEX = 5

#: Uninformative-branch control: permutation seed -> the model seed it is paired with.
PERMUTATION_PAIRS: tuple[tuple[int, int], ...] = ((101, 42), (202, 7), (303, 1234))
SPLIT_INDEX = {"validation": 0, "held_out": 1}

#: Leak audit (plan "Uninformative-branch control"). It fires when the mean
#: ``fused_perm - nlf_cal`` over the three registered runs exceeds the trigger, and clears
#: when the mean ``fused_perm - fused_const`` over the twenty audit runs is at most the
#: clearing threshold. Both numbers, and the seeds, are frozen here; nothing overrides them.
UNINFORMATIVE_AUDIT_TRIGGER = 0.01
LEAK_AUDIT_CLEAR_THRESHOLD = 0.01
AUDIT_PERMUTATION_SEEDS: tuple[int, ...] = tuple(range(1001, 1021))
CONSTANT_ARM = "fused_const"

POSITION_RULE_BA = 0.7309
#: The seed-42, manifest-order LOSO summary the 0.668 figure comes from (context only).
HISTORICAL_NLF_SUMMARY = DEFAULT_PROCESSED_ROOT / "correctness_loso_nlf_parametric_3d2d.json"
NO_P10_SUFFIX = "_nop10"
REQUIRED_DEVICE = "cpu"

#: The readout recipe, as literal values. ``loso.FoldConfig()`` must equal this; a changed
#: default in the trainer module is then a gate failure, not a silent change of recipe.
FROZEN_FOLD_CONFIG: dict = {
    "epochs": 20,
    "batch_size": 32,
    "lr": 3e-4,
    "hidden_dim": 128,
    "dropout": 0.4,
    "weight_decay": 0.01,
    "early_stopping_patience": 5,
    "normalize_features": True,
    "threshold_objective": "balanced_accuracy",
}

#: The unmodified trainer the A/B check runs against: ``train_one_fold`` as it stood at the
#: plan commit, before the validation export existed. Pinned by commit and by blob hash,
#: and loaded from git, so the working tree cannot stand in for it.
TRAINER_PATH = "src/rehab24/loso_cross_validation.py"
BASELINE_TRAINER_COMMIT = "6a9338e2d621e15706987164b804bf89bf7d2483"
BASELINE_TRAINER_BLOB_SHA256 = "1354f5f44e70a9c359023556b1900284fcf6b01acf2060779a2d1624ce0bc400"

#: The arms of the results note's main table; every other arm is descriptive.
MAIN_TABLE_ARMS: tuple[str, ...] = (POSE_BRANCH, VIDEO_BRANCH, FUSED_ARM, POSE_CALIBRATED_ARM)

#: Files whose committed state is the protocol: the plan, the CLI shim, and every project
#: module that is imported when this one loads (a test derives that list in a fresh
#: interpreter and compares). ``init``, ``ab-check``, ``fit`` and ``fuse`` refuse to run
#: while any of them is modified or untracked.
PROTOCOL_FILES: tuple[str, ...] = (
    PLAN_PATH,
    "scripts/rehab24/run_pose_video_fusion.py",
    "src/__init__.py",
    "src/rehab24/__init__.py",
    "src/rehab24/dataset.py",
    TRAINER_PATH,
    "src/rehab24/pose_video_fusion.py",
    "src/rehab24/video_backbone_comparison.py",
    "src/rehab24/videomae_identity_control.py",
    "src/rehab24/videomae_position_control.py",
    "src/rehab24/videomae_stage_a.py",
    "src/video/__init__.py",
    "src/video/classification_metrics.py",
    "src/video/late_fusion.py",
    "src/video/videomae_materialize.py",
    "src/video/videomae_pooling.py",
    "src/video/videomae_video_classifier.py",
)
#: Arm directories under ``oof/`` that hold fits, as opposed to fused arms.
FIT_ARM_NAMES: tuple[str, ...] = tuple(branch + suffix for branch in BRANCHES for suffix in ("", NO_P10_SUFFIX))

GATE_NAMES: tuple[str, ...] = (
    "frozen_protocol",
    "feature_integrity",
    "inert_trainer_change",
    "validation_export_integrity",
    "fold_purity",
    "degenerate_share_identity",
    "fused_oof_integrity",
    "position_reproduction",
)
#: Gates that must pass before a fused held-out score is written (plan "Reproduce").
PRE_FUSE_GATES: tuple[str, ...] = GATE_NAMES[:4]


class FoldPurityError(ValueError):
    """A row from the wrong subject reached a fit or a score."""


@dataclass(frozen=True)
class Expectations:
    """What the gates compare against. The defaults are the frozen protocol; tests
    substitute the values of their synthetic cohort, which is the only reason this is
    not a set of bare module constants."""

    folds_sha256: str = FOLDS_SHA256
    folds_no_p10_sha256: str = FOLDS_NO_P10_SHA256
    manifest_md5: str = MANIFEST_MD5
    source_run_dir: Path = SOURCE_RUN_DIR
    arm_feature_dirs: dict[str, Path] = field(default_factory=lambda: dict(ARM_FEATURE_DIRS))
    arm_dims: dict[str, int] = field(default_factory=lambda: dict(ARM_DIMS))
    fold_config: dict = field(default_factory=lambda: dict(FROZEN_FOLD_CONFIG))
    baseline_trainer_commit: str = BASELINE_TRAINER_COMMIT
    baseline_trainer_blob_sha256: str = BASELINE_TRAINER_BLOB_SHA256
    audit_trigger: float = UNINFORMATIVE_AUDIT_TRIGGER
    audit_clear_threshold: float = LEAK_AUDIT_CLEAR_THRESHOLD
    mixed_sessions: int = EXPECTED_MIXED_SESSIONS
    interleaved_sessions: int = EXPECTED_INTERLEAVED_SESSIONS
    position_rule_ba: float = POSITION_RULE_BA
    require_clean_source: bool = True


# --------------------------------------------------------------------------- #
# paths and small utilities                                                    #
# --------------------------------------------------------------------------- #


def val_oof_dir(run_dir: Path, name: str) -> Path:
    return run_dir / "oof_val" / name


def val_oof_seed_path(run_dir: Path, name: str, seed: int) -> Path:
    return val_oof_dir(run_dir, name) / f"seed{seed}.csv"


def fit_log_path(run_dir: Path, name: str) -> Path:
    return vbc.oof_dir(run_dir, name) / "fit_log.json"


def calibration_path(run_dir: Path, suffix: str = "") -> Path:
    return run_dir / f"calibration{suffix}.json"


def ab_check_path(run_dir: Path) -> Path:
    return run_dir / "ab_check.json"


def feature_integrity_path(run_dir: Path) -> Path:
    return run_dir / "feature_integrity.json"


def deviations_path(run_dir: Path) -> Path:
    return run_dir / "deviations.json"


def leak_audit_path(run_dir: Path) -> Path:
    return run_dir / "leak_audit.json"


def audit_oof_seed_path(run_dir: Path, name: str, seed: int) -> Path:
    """Audit arms live beside ``oof/``, not inside it: they are not arms of the experiment."""
    return run_dir / "oof_audit" / name / f"seed{seed}.csv"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def protocol_source_state(repo_root: Path = REPO_ROOT, files: Sequence[str] = PROTOCOL_FILES) -> list[str]:
    """Protocol files that are missing, untracked or modified -- empty when all are committed."""
    problems = [f"missing: {name}" for name in files if not (repo_root / name).exists()]
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", *files], cwd=repo_root, capture_output=True, text=True, check=True
        )
    except Exception as exc:  # pragma: no cover - git absent
        return problems + [f"git status failed: {exc}"]
    return problems + [line.strip() for line in result.stdout.splitlines() if line.strip()]


def source_stamp() -> dict:
    """Commit and clean/dirty state of the protocol files, stamped into every fit and A/B cell."""
    return {"git_commit": vbc.git_commit_hash(), "dirty": protocol_source_state()}


def require_committed_protocol(expectations: "Expectations", action: str, run_dir: Path | None = None) -> None:
    """Refuse to ``action`` unless the protocol files are committed and HEAD is the commit ``init`` recorded.

    A stamp only reports afterwards that a fit ran on uncommitted code. By then the fit
    exists and the choice is between keeping it and repeating it. Refusing up front leaves
    no such choice.
    """
    if not expectations.require_clean_source:
        return
    dirty = protocol_source_state()
    if dirty:
        raise SystemExit(
            f"Refusing to {action}: protocol files are not committed: {dirty}. Commit them first. If a file is listed "
            "although only its line endings changed, `git add <file>` refreshes the index entry and clears it."
        )
    if run_dir is not None:
        recorded = read_json(vbc.run_config_path(run_dir)).get("git_commit")
        head = vbc.git_commit_hash()
        if not head or head != recorded:
            raise SystemExit(f"Refusing to {action}: HEAD is {head}, `init` recorded {recorded}. Start a new run id on this commit.")


def append_deviation(run_dir: Path, entry: dict) -> None:
    """Append to ``deviations.json``; an entry with the same ``key`` is replaced, not repeated."""
    path = deviations_path(run_dir)
    entries = read_json(path) if path.exists() else []
    entries = [existing for existing in entries if existing.get("key") != entry.get("key") or entry.get("key") is None]
    entries.append({"logged_utc": utc_now(), **entry})
    write_json(path, entries)


# --------------------------------------------------------------------------- #
# the unmodified trainer, for the A/B check                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainerBaseline:
    """A ``loso_cross_validation`` module loaded from source bytes, with its provenance."""

    module: ModuleType
    commit: str
    blob_sha256: str


def trainer_from_source(source: bytes, commit: str) -> TrainerBaseline:
    """Import trainer source as its own module, separate from the working-tree one.

    The source goes through a real file and ``sys.modules`` because the trainer declares
    a dataclass, and ``dataclasses`` resolves string annotations through
    ``sys.modules[cls.__module__]``. The module name carries the blob hash, so two
    different sources can never alias each other.
    """
    blob_sha256 = hashlib.sha256(source).hexdigest()
    name = f"_loso_cross_validation_baseline_{blob_sha256[:16]}"
    if name in sys.modules:
        return TrainerBaseline(sys.modules[name], commit, blob_sha256)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{name}.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return TrainerBaseline(module, commit, blob_sha256)


def load_baseline_trainer(commit: str = BASELINE_TRAINER_COMMIT, repo_root: Path = REPO_ROOT) -> TrainerBaseline:
    """``train_one_fold`` as committed at ``commit``, read with ``git show`` -- never from disk."""
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{TRAINER_PATH}"], cwd=repo_root, capture_output=True, check=True
        )
    except Exception as exc:
        raise SystemExit(f"Cannot read {TRAINER_PATH} at {commit}: {exc}. The A/B check needs the plan commit in this clone.") from exc
    baseline = trainer_from_source(result.stdout, commit)
    if commit == BASELINE_TRAINER_COMMIT and baseline.blob_sha256 != BASELINE_TRAINER_BLOB_SHA256:
        raise SystemExit(f"Trainer blob at {commit} hashes to {baseline.blob_sha256}, expected {BASELINE_TRAINER_BLOB_SHA256}.")
    return baseline


def load_val_oof(run_dir: Path, name: str) -> list[dict] | None:
    directory = val_oof_dir(run_dir, name)
    if not directory.exists():
        return None
    rows: list[dict] = []
    for path in sorted(directory.glob("seed*.csv")):
        rows.extend(vbc.read_oof_csv(path))
    return rows or None


def rows_by_fold_seed(rows: Sequence[dict]) -> dict[tuple[str, int], list[dict]]:
    return vbc.group_by_subject_seed(list(rows))


def load_run_inputs(run_dir: Path) -> tuple[dict, dict[str, dict], dict[str, int]]:
    config = read_json(vbc.run_config_path(run_dir))
    manifest_rows = {row["sample_id"]: row for row in load_manifest(Path(config["manifest_path"]))}
    labels = {key: int(value) for key, value in read_json(Path(config["labels_path"])).items()}
    return config, manifest_rows, labels


# --------------------------------------------------------------------------- #
# init                                                                         #
# --------------------------------------------------------------------------- #


def labels_disagreeing_with_manifest(manifest_rows: dict[str, dict], labels: dict[str, int]) -> list[str]:
    """The manifest is hash-frozen and the labels file is not, so the labels are tied to it here."""
    return sorted(
        sample_id for sample_id, row in manifest_rows.items() if labels.get(sample_id) != int(row["correctness"])
    ) + sorted(set(labels) - set(manifest_rows))


def run_init(
    run_dir: Path,
    manifest_path: Path,
    labels_path: Path,
    segmentation_path: Path = DEFAULT_SEGMENTATION,
    expectations: Expectations | None = None,
) -> dict:
    """Assert the frozen hashes, copy the folds byte for byte, stamp the run config.

    Refuses to re-stamp a run that already holds a fit: a config written after a fit
    could not show that the protocol preceded it.
    """
    expectations = expectations or Expectations()
    require_committed_protocol(expectations, "init")
    source_run_dir = expectations.source_run_dir
    if any((run_dir / "oof").glob("*/seed*.csv")) or ab_check_path(run_dir).exists():
        raise SystemExit(f"{run_dir} already holds fits or A/B cells; start a new run id instead of re-initialising.")

    sources = {
        "folds.json": (vbc.folds_path(source_run_dir), expectations.folds_sha256),
        "folds_no_p10.json": (vbc.folds_path(source_run_dir, no_p10=True), expectations.folds_no_p10_sha256),
    }
    observed = {name: file_sha256(path) for name, (path, _) in sources.items()}
    observed["manifest_md5"] = vbc.file_md5(manifest_path)
    mismatches = [name for name, (_, expected) in sources.items() if observed[name] != expected]
    if observed["manifest_md5"] != expectations.manifest_md5:
        mismatches.append("manifest_md5")
    if mismatches:
        raise SystemExit(f"Frozen hash mismatch for {mismatches}: observed {observed}.")
    manifest_rows = {row["sample_id"]: row for row in load_manifest(manifest_path)}
    disagreeing = labels_disagreeing_with_manifest(manifest_rows, {k: int(v) for k, v in read_json(labels_path).items()})
    if disagreeing:
        raise SystemExit(f"{len(disagreeing)} labels disagree with the hash-frozen manifest: {disagreeing[:5]}")

    run_dir.mkdir(parents=True, exist_ok=True)
    for name, (path, _) in sources.items():
        shutil.copyfile(path, run_dir / name)

    config = {
        "plan": PLAN_PATH,
        "created_utc": utc_now(),
        "git_commit": vbc.git_commit_hash(),
        "source_dirty": protocol_source_state(),
        "source_run_dir": str(source_run_dir),
        "arm_feature_dirs": {arm: str(path) for arm, path in expectations.arm_feature_dirs.items()},
        "arm_dims": dict(expectations.arm_dims),
        "seeds": list(SEEDS),
        "practical_margin": PRACTICAL_MARGIN,
        "alpha": ALPHA,
        "bootstrap_resamples": vbc.BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": vbc.BOOTSTRAP_SEED,
        "permutation_pairs": [list(pair) for pair in PERMUTATION_PAIRS],
        "weight_grid": list(WEIGHT_GRID),
        "fold_config": dict(expectations.fold_config),
        "baseline_trainer": {
            "commit": expectations.baseline_trainer_commit,
            "blob_sha256": expectations.baseline_trainer_blob_sha256,
            "path": TRAINER_PATH,
        },
        "leak_audit": {
            "trigger": expectations.audit_trigger,
            "clear_threshold": expectations.audit_clear_threshold,
            "permutation_seeds": list(AUDIT_PERMUTATION_SEEDS),
            "model_seeds": [audit_model_seed(seed) for seed in AUDIT_PERMUTATION_SEEDS],
        },
        "required_device": REQUIRED_DEVICE,
        "hashes": observed,
        "manifest_path": str(manifest_path),
        "labels_path": str(labels_path),
        "segmentation_path": str(segmentation_path),
    }
    write_json(vbc.run_config_path(run_dir), config)
    if not deviations_path(run_dir).exists():
        write_json(deviations_path(run_dir), [])
    return config


# --------------------------------------------------------------------------- #
# feature integrity                                                            #
# --------------------------------------------------------------------------- #


def feature_integrity(feature_dir: Path, manifest_rows: dict[str, dict], expected_dim: int) -> dict:
    """Id-by-id diff of one feature directory against the manifest -- never a file count."""
    stems = vbc.index_raw_bundles(feature_dir) if feature_dir.exists() else {}
    expected = set(manifest_rows)
    found = set(stems)
    duplicates = sorted(stem for stem, paths in stems.items() if len(paths) > 1)
    missing_key: list[str] = []
    wrong_dim: dict[str, int] = {}
    non_finite: list[str] = []
    for sample_id in sorted(expected & found):
        with np.load(stems[sample_id][0], allow_pickle=False) as data:
            if "video_feature" not in data.files:
                missing_key.append(sample_id)
                continue
            vector = data["video_feature"]
        if vector.ndim != 1 or int(vector.shape[0]) != expected_dim:
            wrong_dim[sample_id] = int(vector.shape[0]) if vector.ndim else 0
        if not np.all(np.isfinite(vector)):
            non_finite.append(sample_id)

    cameras_by_repetition: dict[str, set[str]] = defaultdict(set)
    for sample_id in expected & found:
        row = manifest_rows[sample_id]
        cameras_by_repetition[f"{vbc.session_key(row)}|{vbc.repetition_key(row)}"].add(row["camera"])
    incomplete_pairs = sorted(key for key, cameras in cameras_by_repetition.items() if cameras != set(CAMERAS))

    audit = {
        "feature_dir": str(feature_dir),
        "expected_ids": len(expected),
        "found_ids": len(found),
        "expected_dim": expected_dim,
        "missing": sorted(expected - found),
        "unexpected": sorted(found - expected),
        "duplicates": duplicates,
        "missing_key": missing_key,
        "wrong_dim": wrong_dim,
        "non_finite": non_finite,
        "incomplete_camera_pairs": incomplete_pairs,
    }
    audit["passed"] = not any(
        audit[key] for key in ("missing", "unexpected", "duplicates", "missing_key", "wrong_dim", "non_finite", "incomplete_camera_pairs")
    )
    return audit


def run_feature_integrity(run_dir: Path) -> dict:
    config, manifest_rows, _ = load_run_inputs(run_dir)
    report = {
        arm: feature_integrity(Path(config["arm_feature_dirs"][arm]), manifest_rows, int(config["arm_dims"][arm]))
        for arm in BRANCHES
    }
    # Truncate the id lists on disk: a wholly missing directory would otherwise write 2,144 ids twice.
    write_json(
        feature_integrity_path(run_dir),
        {arm: {key: (value[:20] if isinstance(value, list) else value) for key, value in audit.items()} for arm, audit in report.items()},
    )
    clear_feature_cache()
    return report


# --------------------------------------------------------------------------- #
# fit: one branch on the frozen folds, with the validation export               #
# --------------------------------------------------------------------------- #


def assert_rows_belong_to(rows: Sequence[dict], subject: str, role: str, fold: dict) -> None:
    strangers = sorted({row["person_id"] for row in rows} - {subject})
    if strangers:
        raise FoldPurityError(
            f"Fold P{fold['test_subject']}: {role} rows must all belong to P{subject}, found subjects {strangers}."
        )


def fit_fold(
    feature_dir: Path,
    fold: dict,
    labels: dict[str, int],
    manifest_rows: dict[str, dict],
    config: "loso.FoldConfig",
    device: "torch.device",
    seed: int,
) -> tuple[list[dict], list[dict], float]:
    """One fold, one seed: (held-out rows, validation rows, threshold), both in the OOF schema."""
    threshold, test_prob, test_labels, validation = loso.train_one_fold(
        feature_dir, fold["train_ids"], fold["val_ids"], fold["test_ids"], labels, config, device, seed, return_validation=True
    )
    test_ids = ordered_test_ids(feature_dir, fold["test_ids"], labels)
    if len(test_ids) != len(test_prob) or len(validation.sample_ids) != len(validation.probabilities):
        raise SystemExit(f"Prediction/id misalignment for test_subject {fold['test_subject']}.")
    held_out = vbc.build_oof_rows(fold, test_ids, test_prob, test_labels, threshold, seed, manifest_rows)
    validation_rows = vbc.build_oof_rows(
        fold, validation.sample_ids, validation.probabilities, validation.labels, threshold, seed, manifest_rows
    )
    assert_rows_belong_to(held_out, fold["test_subject"], "held-out", fold)
    assert_rows_belong_to(validation_rows, fold["val_subject"], "validation", fold)
    return held_out, validation_rows, threshold


def run_fit(
    run_dir: Path,
    arm: str,
    device: "torch.device",
    seeds: Sequence[int] = SEEDS,
    no_p10_training: bool = False,
    expectations: Expectations | None = None,
) -> dict:
    """Refit one branch over every frozen fold; ids are used in the stored (sorted) order.

    The readout configuration is always the one ``init`` froze into ``run_config.json``;
    there is no override, so a fit cannot run under another recipe.
    """
    if arm not in BRANCHES:
        raise ValueError(f"Unknown branch {arm!r}; expected one of {BRANCHES}.")
    require_committed_protocol(expectations or Expectations(), "fit", run_dir)
    run_config, manifest_rows, labels = load_run_inputs(run_dir)
    feature_dir = Path(run_config["arm_feature_dirs"][arm])
    folds = vbc.load_folds(vbc.folds_path(run_dir, no_p10=no_p10_training))
    config = loso.FoldConfig(**run_config["fold_config"])
    name = arm + (NO_P10_SUFFIX if no_p10_training else "")

    log = read_json(fit_log_path(run_dir, name)) if fit_log_path(run_dir, name).exists() else {"seeds": {}}
    for seed in seeds:
        started = utc_now()
        clock = time.perf_counter()
        held_out_rows: list[dict] = []
        validation_rows: list[dict] = []
        for fold in folds:
            held_out, validation, _ = fit_fold(feature_dir, fold, labels, manifest_rows, config, device, seed)
            held_out_rows.extend(held_out)
            validation_rows.extend(validation)
        vbc.write_oof_csv(vbc.oof_seed_path(run_dir, name, seed), held_out_rows)
        vbc.write_oof_csv(val_oof_seed_path(run_dir, name, seed), validation_rows)
        elapsed = time.perf_counter() - clock
        log["seeds"][str(seed)] = {
            "started_utc": started,
            "seconds": elapsed,
            "seconds_per_fit": elapsed / len(folds),
            "held_out_rows": len(held_out_rows),
            "validation_rows": len(validation_rows),
            "source": source_stamp(),
            "device": str(device),
            "fold_config": asdict(config),
        }
        write_json(fit_log_path(run_dir, name), log)
    clear_feature_cache()
    return log


# --------------------------------------------------------------------------- #
# historical-baseline check (never blocks)                                     #
# --------------------------------------------------------------------------- #


def historical_baseline_path(run_dir: Path) -> Path:
    return run_dir / "historical_baseline_check.json"


def historical_baseline_check(run_dir: Path, rerun_summary: Path, stored_summary: Path = HISTORICAL_NLF_SUMMARY) -> dict:
    """Fold balanced accuracy and thresholds of a stock rerun against the stored NLF summary.

    The rerun is the unmodified ``loso_cross_validation`` command on the ``nlf`` features
    at seed 42 in manifest order. The frozen-fold ``nlf`` arm uses sorted sample order, so
    it is *not* what is compared here and is not expected to match the stored file. This
    check never blocks: the 0.668 figure is context, and a miss is logged as a deviation.
    """
    rerun = {fold["test_subject"]: fold for fold in read_json(rerun_summary)["folds"]}
    stored = {fold["test_subject"]: fold for fold in read_json(stored_summary)["folds"]}
    per_fold = {
        subject: {
            "balanced_accuracy": [stored[subject]["balanced_accuracy"], rerun[subject]["balanced_accuracy"]],
            "threshold": [stored[subject]["threshold"], rerun[subject]["threshold"]],
            "identical": stored[subject]["balanced_accuracy"] == rerun[subject]["balanced_accuracy"]
            and stored[subject]["threshold"] == rerun[subject]["threshold"],
        }
        for subject in sorted(set(stored) & set(rerun), key=int)
    }
    report = {
        "rerun_summary": str(rerun_summary),
        "stored_summary": str(stored_summary),
        "folds_only_in_one_file": sorted(set(stored) ^ set(rerun), key=int),
        "per_fold": per_fold,
        "reproduced": bool(per_fold) and set(stored) == set(rerun) and all(cell["identical"] for cell in per_fold.values()),
        "blocking": False,
    }
    write_json(historical_baseline_path(run_dir), report)
    if not report["reproduced"]:
        append_deviation(
            run_dir,
            {
                "key": "historical_nlf_baseline_not_reproduced",
                "item": "stock seed-42 LOSO rerun on the nlf features differs from the stored summary",
                "plan_said": "never blocks; the nlf arm of this experiment is the frozen-fold refit and 0.668 stays context",
                "what_happened": {subject: cell for subject, cell in per_fold.items() if not cell["identical"]},
                "held_out_scores_visible": False,
                "effect": "none on the registered contrasts; the historical figure is labelled as not reproduced in this environment",
            },
        )
    return report


# --------------------------------------------------------------------------- #
# ab-check: the trainer change is inert                                        #
# --------------------------------------------------------------------------- #


def stored_source_rows(source_run_dir: Path, arm: str, seed: int, test_subject: str) -> list[dict] | None:
    path = vbc.oof_seed_path(source_run_dir, arm, seed)
    if not path.exists():
        return None
    return [row for row in vbc.read_oof_csv(path) if row["test_subject"] == test_subject]


def ab_check_fold(
    feature_dir: Path,
    fold: dict,
    labels: dict[str, int],
    config: "loso.FoldConfig",
    device: "torch.device",
    seed: int,
    baseline: TrainerBaseline,
    stored_rows: list[dict] | None = None,
) -> dict:
    """One cell of the A/B: the unmodified trainer against the exporting trainer.

    A is ``baseline.module.train_one_fold`` -- the pre-change source, which has no
    ``return_validation`` parameter at all. B is the working-tree trainer with the export
    switched on. Comparing the working-tree trainer with its own flag off would only show
    that the flag is inert, not that the edit is.
    """
    clock = time.perf_counter()
    plain_threshold, plain_prob, plain_labels = baseline.module.train_one_fold(
        feature_dir, fold["train_ids"], fold["val_ids"], fold["test_ids"], labels, config, device, seed
    )
    seconds_plain = time.perf_counter() - clock
    threshold, test_prob, test_labels, _ = loso.train_one_fold(
        feature_dir, fold["train_ids"], fold["val_ids"], fold["test_ids"], labels, config, device, seed, return_validation=True
    )
    cell = {
        "identical_probabilities": bool(np.array_equal(plain_prob, test_prob)),
        "identical_thresholds": bool(plain_threshold == threshold),
        "identical_labels": bool(np.array_equal(plain_labels, test_labels)),
        "max_abs_difference": (
            float(np.max(np.abs(plain_prob.astype(np.float64) - test_prob.astype(np.float64))))
            if plain_prob.shape == test_prob.shape
            else None
        ),
        "seconds_per_fit": seconds_plain,
        "baseline_trainer": {"commit": baseline.commit, "blob_sha256": baseline.blob_sha256},
        "source": source_stamp(),
        "device": str(device),
    }
    cell["identical"] = cell["identical_probabilities"] and cell["identical_thresholds"] and cell["identical_labels"]

    if stored_rows is not None:
        stored = {row["sample_id"]: row for row in stored_rows}
        test_ids = ordered_test_ids(feature_dir, fold["test_ids"], labels)
        if set(stored) != set(test_ids):
            cell["stored_run"] = {"status": "id mismatch", "bit_identical": False}
        else:
            differences = [abs(stored[sid]["probability"] - float(p)) for sid, p in zip(test_ids, test_prob)]
            cell["stored_run"] = {
                "status": "compared",
                "bit_identical": bool(max(differences) == 0.0 and {row["threshold"] for row in stored_rows} == {float(threshold)}),
                "max_abs_difference": float(max(differences)),
            }
    return cell


def run_ab_check(
    run_dir: Path,
    arms: Sequence[str],
    device: "torch.device",
    seeds: Sequence[int] = SEEDS,
    test_subjects: Sequence[str] | None = None,
    baseline: TrainerBaseline | None = None,
    expectations: Expectations | None = None,
) -> dict:
    """Cells accumulate in ``ab_check.json``; the gate needs every arm x fold x seed cell.

    The stored-run comparison for ``vm16`` never blocks. A mismatch means the environment
    has drifted since the 20260918 run, the refit is the registered arm, and the
    difference is logged to ``deviations.json`` here, not left for someone to remember.
    """
    require_committed_protocol(expectations or Expectations(), "run the A/B check", run_dir)
    run_config, _, labels = load_run_inputs(run_dir)
    baseline = baseline or load_baseline_trainer(run_config["baseline_trainer"]["commit"])
    config = loso.FoldConfig(**run_config["fold_config"])
    folds = vbc.load_folds(vbc.folds_path(run_dir))
    payload = read_json(ab_check_path(run_dir)) if ab_check_path(run_dir).exists() else {"cells": {}}
    for arm in arms:
        feature_dir = Path(run_config["arm_feature_dirs"][arm])
        for seed in seeds:
            for fold in folds:
                if test_subjects is not None and fold["test_subject"] not in test_subjects:
                    continue
                stored = (
                    stored_source_rows(Path(run_config["source_run_dir"]), arm, seed, fold["test_subject"])
                    if arm == VIDEO_BRANCH
                    else None
                )
                key = f"{arm}|P{fold['test_subject']}|seed{seed}"
                cell = ab_check_fold(feature_dir, fold, labels, config, device, seed, baseline, stored)
                payload["cells"][key] = cell
                write_json(ab_check_path(run_dir), payload)
                if "stored_run" in cell and not cell["stored_run"]["bit_identical"]:
                    append_deviation(
                        run_dir,
                        {
                            "key": f"stored_vm16_oof_not_reproduced|{key}",
                            "item": "refit vm16 held-out probabilities differ from the stored 20260918 OOF",
                            "plan_said": "expected bit-identical; non-blocking check",
                            "what_happened": cell["stored_run"],
                            "held_out_scores_visible": False,
                            "effect": "environment drift; the refit is the registered arm",
                        },
                    )
    clear_feature_cache()
    return payload


# --------------------------------------------------------------------------- #
# fuse: the registered rule, per fold and seed                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BranchSplit:
    """Both branches on one split of one fold, aligned by ``sample_id``."""

    sample_ids: list[str]
    labels: np.ndarray
    pose: np.ndarray
    video: np.ndarray
    person_ids: tuple[str, ...]

    def provenance(self) -> dict:
        """Whose rows these are and exactly which ones -- read off the rows, not off the fold."""
        return {"person_ids": list(self.person_ids), "n_rows": len(self.sample_ids), "ids_sha256": vbc.hash_ids(self.sample_ids)}


def split_predictions(rows: Sequence[dict]) -> SplitPredictions:
    return SplitPredictions(
        video_ids=[row["sample_id"] for row in rows],
        probabilities=np.asarray([row["probability"] for row in rows], dtype=np.float64),
        labels=np.asarray([row["label"] for row in rows], dtype=np.int32),
    )


def align_branches(pose_rows: Sequence[dict], video_rows: Sequence[dict], subject: str, role: str, fold: dict) -> BranchSplit:
    """Align by id through ``late_fusion.align`` (which rejects duplicates, mismatched id
    sets and disagreeing labels), after checking every row belongs to ``subject``."""
    if not pose_rows or not video_rows:
        raise ValueError(f"Fold P{fold['test_subject']}: a branch has no {role} rows.")
    assert_rows_belong_to(pose_rows, subject, role, fold)
    assert_rows_belong_to(video_rows, subject, role, fold)
    pose, video, labels, sample_ids = align(split_predictions(pose_rows), split_predictions(video_rows))
    person_ids = tuple(sorted({row["person_id"] for row in (*pose_rows, *video_rows)}, key=int))
    return BranchSplit(sample_ids=list(sample_ids), labels=labels, pose=pose, video=video, person_ids=person_ids)


def select_tuned_share(validation_scores: Sequence[float]) -> int:
    """Index into ``WEIGHT_GRID`` of the validation-best pose share.

    Ties go to the share nearest 0.5, then to the larger pose share. Distance is
    measured on the integer grid index: ``abs(0.4 - 0.5)`` and ``abs(0.6 - 0.5)`` are
    not guaranteed to be the same float, and the tie-break must not depend on that.
    """
    if len(validation_scores) != len(WEIGHT_GRID):
        raise ValueError(f"Expected {len(WEIGHT_GRID)} validation scores, got {len(validation_scores)}.")
    return max(range(len(WEIGHT_GRID)), key=lambda i: (validation_scores[i], -abs(i - EQUAL_SHARE_INDEX), i))


def fuse_fold(
    fold: dict,
    validation_rows: dict[str, Sequence[dict]],
    held_out_rows: dict[str, Sequence[dict]],
    shares: Sequence[float] = (0.5,),
    objective: str = "balanced_accuracy",
    constant_video: bool = False,
) -> dict:
    """The registered rule on one fold and seed.

    Platt per branch on the validation subject, a weighted mean of the two calibrated
    probabilities, the threshold that maximises ``objective`` on the validation
    subject's fused probabilities, and one scoring of the held-out subject. Only
    ``validation_rows`` reach a fit; ``held_out_rows`` are transformed and returned.

    ``constant_video`` is the leak audit's reference arm. The second branch's calibrated
    probability is then the validation subject's positive rate on every row of both
    splits, with no Platt fit for that branch. It is a number read off validation labels
    alone, identical for every row, so it cannot carry information about any repetition;
    what it still does is halve the pose branch's probability scale before the threshold
    is re-selected, which is the effect the audit has to separate from a leak.
    """
    if fold["val_subject"] == fold["test_subject"]:
        raise FoldPurityError(f"Fold P{fold['test_subject']} validates on its own held-out subject.")
    validation = align_branches(validation_rows[POSE_BRANCH], validation_rows[VIDEO_BRANCH], fold["val_subject"], "validation", fold)
    held_out = align_branches(held_out_rows[POSE_BRANCH], held_out_rows[VIDEO_BRANCH], fold["test_subject"], "held-out", fold)

    calibrations = {POSE_BRANCH: fit_platt(validation.pose, validation.labels)}
    constant_rate = None
    if constant_video:
        constant_rate = float(np.mean(validation.labels))
        video_validation = np.full(len(validation.labels), constant_rate, dtype=np.float64)
        video_held_out = np.full(len(held_out.labels), constant_rate, dtype=np.float64)
    else:
        calibrations[VIDEO_BRANCH] = fit_platt(validation.video, validation.labels)
        video_validation = calibrations[VIDEO_BRANCH].apply(validation.video)
        video_held_out = calibrations[VIDEO_BRANCH].apply(held_out.video)
    validation_calibrated = (calibrations[POSE_BRANCH].apply(validation.pose), video_validation)
    held_out_calibrated = (calibrations[POSE_BRANCH].apply(held_out.pose), video_held_out)

    by_share: dict[float, dict] = {}
    for share in shares:
        validation_fused = fuse_probabilities(validation_calibrated[0], validation_calibrated[1], share)
        threshold, validation_metrics = find_best_threshold(validation_fused, validation.labels, objective)
        by_share[share] = {
            "threshold": float(threshold),
            "validation_score": float(validation_metrics[objective]),
            "validation_probabilities": validation_fused,
            "held_out_probabilities": fuse_probabilities(held_out_calibrated[0], held_out_calibrated[1], share),
        }
    return {
        # The arrays handed to `fit_platt` and `find_best_threshold` above are `validation`'s and
        # nothing else, so its provenance is the provenance of every fitted quantity in this cell.
        "fitted_on": validation.provenance(),
        "scored_on": held_out.provenance(),
        "calibrations": calibrations,
        "constant_rate": constant_rate,
        "validation": validation,
        "held_out": held_out,
        "validation_calibrated": validation_calibrated,
        "held_out_calibrated": held_out_calibrated,
        "by_share": by_share,
    }


def audit_model_seed(permutation_seed: int) -> int:
    """Model seed of an audit run: 42, 7, 1234 cyclically from permutation seed 1001."""
    if permutation_seed not in AUDIT_PERMUTATION_SEEDS:
        raise ValueError(f"{permutation_seed} is not a registered audit permutation seed.")
    return SEEDS[(permutation_seed - AUDIT_PERMUTATION_SEEDS[0]) % len(SEEDS)]


def permute_branch_rows(rows: Sequence[dict], permutation_seed: int, test_subject: str, split: str) -> list[dict]:
    """Shuffle one branch's probabilities across repetitions, both camera rows moving together.

    The generator is seeded from (permutation seed, test subject, split), so a cell's
    shuffle does not depend on which other cells were drawn before it.
    """
    by_repetition: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["camera"] in by_repetition[row["repetition_id"]]:
            raise ValueError(f"Repetition {row['repetition_id']} has two {row['camera']} rows.")
        by_repetition[row["repetition_id"]][row["camera"]] = row
    incomplete = sorted(key for key, cameras in by_repetition.items() if set(cameras) != set(CAMERAS))
    if incomplete:
        raise ValueError(f"Repetitions without a complete camera pair cannot be permuted: {incomplete[:5]}")

    repetitions = sorted(by_repetition)
    rng = np.random.default_rng([permutation_seed, int(test_subject), SPLIT_INDEX[split]])
    order = rng.permutation(len(repetitions))
    permuted: list[dict] = []
    for index, repetition in enumerate(repetitions):
        donor = by_repetition[repetitions[order[index]]]
        for camera in CAMERAS:
            permuted.append({**by_repetition[repetition][camera], "probability": donor[camera]["probability"]})
    return permuted


def fused_oof_rows(template_rows: Sequence[dict], sample_ids: Sequence[str], probabilities: np.ndarray, threshold: float) -> list[dict]:
    """Held-out rows of a fused arm: the branch row's identity columns, the fused score."""
    template = {row["sample_id"]: row for row in template_rows}
    return [
        {**template[sample_id], "probability": repr(float(probability)), "threshold": repr(float(threshold))}
        for sample_id, probability in zip(sample_ids, probabilities)
    ]


def calibration_record(fused: dict, seed: int, fold: dict) -> dict:
    """What ``fuse`` may write before any gate or audit has released anything.

    Fitted parameters and provenance only. Nothing here is computed from a held-out
    label: a Brier score of a branch is a score of an arm of the experiment, and it
    belongs to the released report.
    """
    return {
        "test_subject": fold["test_subject"],
        "seed": seed,
        "fitted_on": fused["fitted_on"],
        "scored_on": fused["scored_on"],
        "platt": {
            branch: {"slope": repr(calibration.slope), "intercept": repr(calibration.intercept)}
            for branch, calibration in fused["calibrations"].items()
        },
    }


def permuted_fold(cell: dict, permutation_seed: int) -> tuple[list[dict], dict]:
    """One cell of a permuted-branch run: shuffle ``vm16`` within each split, then the registered rule."""
    fold = cell["fold"]
    permuted = fuse_fold(
        fold,
        {**cell["validation"], VIDEO_BRANCH: permute_branch_rows(cell["validation"][VIDEO_BRANCH], permutation_seed, fold["test_subject"], "validation")},
        {**cell["held_out"], VIDEO_BRANCH: permute_branch_rows(cell["held_out"][VIDEO_BRANCH], permutation_seed, fold["test_subject"], "held_out")},
    )
    result = permuted["by_share"][0.5]
    rows = fused_oof_rows(cell["held_out"][POSE_BRANCH], permuted["held_out"].sample_ids, result["held_out_probabilities"], result["threshold"])
    return rows, permuted


def run_fuse(
    run_dir: Path,
    no_p10_training: bool = False,
    enforce_gates: bool = True,
    expectations: Expectations | None = None,
) -> dict:
    """Write every fused arm's held-out OOF, then run the leak audit. Prints and returns no held-out score."""
    expectations = expectations or Expectations()
    require_committed_protocol(expectations, "fuse", run_dir)
    if enforce_gates:
        gates = compute_gates(run_dir, expectations, write=False)["gates"]
        blocked = [name for name in PRE_FUSE_GATES if gates[name]["status"] != "pass"]
        if blocked:
            raise SystemExit(f"Refusing to fuse: gates not passed: {blocked}. Run `gates` for the evidence.")

    started = utc_now()
    suffix = NO_P10_SUFFIX if no_p10_training else ""
    if not no_p10_training:
        # A fuse replaces the audit it invalidates; neither file may outlive the OOF it describes.
        leak_audit_path(run_dir).unlink(missing_ok=True)
        shutil.rmtree(run_dir / "oof_audit", ignore_errors=True)
    folds = vbc.load_folds(vbc.folds_path(run_dir, no_p10=no_p10_training))
    held_out = {branch: vbc.load_arm_oof(run_dir, branch + suffix) for branch in BRANCHES}
    validation = {branch: load_val_oof(run_dir, branch + suffix) for branch in BRANCHES}
    absent = [branch for branch in BRANCHES if held_out[branch] is None or validation[branch] is None]
    if absent:
        raise SystemExit(f"No fitted rows for {absent}. Run `fit --arm <branch>` first.")
    held_out_cells = {branch: rows_by_fold_seed(held_out[branch]) for branch in BRANCHES}
    validation_cells = {branch: rows_by_fold_seed(validation[branch]) for branch in BRANCHES}

    shares = tuple(sorted(set(WEIGHT_GRID) | set(ARM_SHARES.values())))
    permutation_of = {model_seed: permutation_seed for permutation_seed, model_seed in PERMUTATION_PAIRS}
    arm_rows: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    records: list[dict] = []
    cells: list[dict] = []

    for seed in SEEDS:
        for fold in folds:
            key = (fold["test_subject"], seed)
            missing = [branch for branch in BRANCHES if key not in held_out_cells[branch] or key not in validation_cells[branch]]
            if missing:
                raise SystemExit(f"Fold P{key[0]} seed {seed} has no rows for {missing}.")
            cell = {
                "fold": fold,
                "seed": seed,
                "validation": {branch: validation_cells[branch][key] for branch in BRANCHES},
                "held_out": {branch: held_out_cells[branch][key] for branch in BRANCHES},
            }
            cells.append(cell)
            fused = fuse_fold(fold, cell["validation"], cell["held_out"], shares)
            sample_ids = fused["held_out"].sample_ids
            template = cell["held_out"][POSE_BRANCH]

            for arm, share in ARM_SHARES.items():
                result = fused["by_share"][share]
                arm_rows[arm][seed].extend(fused_oof_rows(template, sample_ids, result["held_out_probabilities"], result["threshold"]))

            tuned_index = select_tuned_share([fused["by_share"][share]["validation_score"] for share in WEIGHT_GRID])
            tuned = fused["by_share"][WEIGHT_GRID[tuned_index]]
            arm_rows[TUNED_ARM][seed].extend(fused_oof_rows(template, sample_ids, tuned["held_out_probabilities"], tuned["threshold"]))

            record = calibration_record(fused, seed, fold)
            record["thresholds"] = {arm: fused["by_share"][share]["threshold"] for arm, share in ARM_SHARES.items()}
            record["tuned_pose_share"] = WEIGHT_GRID[tuned_index]

            if seed in permutation_of and not no_p10_training:
                permutation_seed = permutation_of[seed]
                permuted_rows, permuted = permuted_fold(cell, permutation_seed)
                arm_rows[f"fused_perm{permutation_seed}"][seed].extend(permuted_rows)
                record["permuted_video_slope"] = repr(permuted["calibrations"][VIDEO_BRANCH].slope)
                record["permuted_fitted_on"] = permuted["fitted_on"]
            records.append(record)

    written: dict[str, int] = {}
    for arm, by_seed in arm_rows.items():
        for seed, rows in by_seed.items():
            written[f"{arm + suffix}|seed{seed}"] = vbc.write_oof_csv(vbc.oof_seed_path(run_dir, arm + suffix, seed), rows)

    negative = {
        branch: int(sum(float(record["platt"][branch]["slope"]) <= 0 for record in records)) for branch in BRANCHES
    }
    write_json(
        calibration_path(run_dir, suffix),
        {
            "execution": {"source": source_stamp(), "device": REQUIRED_DEVICE, "started_utc": started},
            "shares": ARM_SHARES,
            "negative_slopes": negative,
            "folds": records,
        },
    )
    result = {"rows_written": written, "negative_slopes": negative, "cells": len(records)}
    if not no_p10_training:
        audit = run_leak_audit(run_dir, cells, expectations)
        result["leak_audit"] = {"fired": audit["trigger"]["fired"], "released": audit["released"]}
    return result


# --------------------------------------------------------------------------- #
# leak audit (plan "Uninformative-branch control"): a numeric rule, no override #
# --------------------------------------------------------------------------- #


def subject_delta(candidate_rows: Sequence[dict], baseline_rows: Sequence[dict]) -> dict:
    """P1-P9 subject-mean balanced-accuracy delta of two arms. Only the delta is kept."""
    subjects = vbc.PRIMARY_SUBJECTS
    return descriptive_delta(
        vbc.subject_mean_metric(list(candidate_rows), subjects, "balanced_accuracy"),
        vbc.subject_mean_metric(list(baseline_rows), subjects, "balanced_accuracy"),
        subjects,
    )


def rows_of_seed(rows: Sequence[dict], seed: int) -> list[dict]:
    return [row for row in rows if row["seed"] == seed]


def uninformative_trigger(run_dir: Path, threshold: float) -> dict:
    """Mean ``fused_perm - nlf_cal`` over the three registered runs, and whether it fires the audit."""
    calibrated = vbc.load_arm_oof(run_dir, POSE_CALIBRATED_ARM)
    runs: dict[str, dict] = {}
    for permutation_seed, model_seed in PERMUTATION_PAIRS:
        permuted = vbc.load_arm_oof(run_dir, f"fused_perm{permutation_seed}")
        if permuted is None or calibrated is None:
            return {"status": "pending", "fired": None}
        runs[str(permutation_seed)] = {
            "model_seed": model_seed,
            "fused_perm_minus_nlf_cal": subject_delta(permuted, rows_of_seed(calibrated, model_seed)),
        }
    mean_delta = float(np.mean([run["fused_perm_minus_nlf_cal"]["mean_delta"] for run in runs.values()]))
    return {
        "status": "complete",
        "runs": runs,
        "mean_fused_perm_minus_nlf_cal": mean_delta,
        "threshold": threshold,
        "fired": bool(mean_delta > threshold),
    }


def read_audit_rows(run_dir: Path, name: str, seed: int) -> list[dict] | None:
    path = audit_oof_seed_path(run_dir, name, seed)
    return vbc.read_oof_csv(path) if path.exists() else None


def audit_statistic(run_dir: Path, clear_threshold: float) -> dict:
    """Mean ``fused_perm - fused_const`` over the twenty audit runs, read back from ``oof_audit/``."""
    calibrated = vbc.load_arm_oof(run_dir, POSE_CALIBRATED_ARM)
    constant = {seed: read_audit_rows(run_dir, CONSTANT_ARM, seed) for seed in SEEDS}
    permuted = {seed: read_audit_rows(run_dir, f"fused_perm{seed}", audit_model_seed(seed)) for seed in AUDIT_PERMUTATION_SEEDS}
    if calibrated is None or any(rows is None for rows in (*constant.values(), *permuted.values())):
        return {"status": "missing", "cleared": False}
    runs = {
        str(seed): {
            "model_seed": audit_model_seed(seed),
            "fused_perm_minus_fused_const": subject_delta(permuted[seed], constant[audit_model_seed(seed)]),
        }
        for seed in AUDIT_PERMUTATION_SEEDS
    }
    statistic = float(np.mean([run["fused_perm_minus_fused_const"]["mean_delta"] for run in runs.values()]))
    return {
        "status": "complete",
        "fused_const_minus_nlf_cal": subject_delta([row for rows in constant.values() for row in rows], calibrated),
        "runs": runs,
        "statistic": statistic,
        "clear_threshold": clear_threshold,
        "cleared": bool(statistic <= clear_threshold),
    }


def audit_input_hashes(run_dir: Path) -> dict[str, str]:
    """Hashes of what the audit was computed from. ``calibration.json`` carries the fuse's start time, so any re-fuse changes it."""
    paths = [calibration_path(run_dir)]
    for name in (*BRANCHES, POSE_CALIBRATED_ARM, *(f"fused_perm{seed}" for seed, _ in PERMUTATION_PAIRS)):
        paths.extend(sorted(vbc.oof_dir(run_dir, name).glob("seed*.csv")))
    for branch in BRANCHES:  # what every fused and audit arm was calibrated on
        paths.extend(sorted(val_oof_dir(run_dir, branch).glob("seed*.csv")))
    paths.extend(sorted((run_dir / "oof_audit").glob("*/seed*.csv")))
    return {path.relative_to(run_dir).as_posix(): file_sha256(path) for path in paths if path.exists()}


def leak_audit_from_disk(run_dir: Path, expectations: Expectations) -> dict:
    """The audit, recomputed from the files on disk. It holds deltas against ``nlf_cal`` and ``fused_const`` only."""
    trigger = uninformative_trigger(run_dir, expectations.audit_trigger)
    audit = None
    if trigger["status"] != "complete":
        released = False
    elif not trigger["fired"]:
        released = True
    else:
        audit = audit_statistic(run_dir, expectations.audit_clear_threshold)
        released = bool(audit["cleared"])
    payload = {"trigger": trigger, "audit": audit, "released": released, "inputs": audit_input_hashes(run_dir)}
    return json.loads(json.dumps(payload))


def run_leak_audit(run_dir: Path, cells: Sequence[dict], expectations: Expectations) -> dict:
    """If the trigger fires, write ``fused_const`` and the twenty audit runs; always write ``leak_audit.json``."""
    if uninformative_trigger(run_dir, expectations.audit_trigger)["fired"]:
        constant_rows: dict[int, list[dict]] = defaultdict(list)
        for cell in cells:
            constant = fuse_fold(cell["fold"], cell["validation"], cell["held_out"], constant_video=True)
            result = constant["by_share"][0.5]
            constant_rows[cell["seed"]].extend(
                fused_oof_rows(cell["held_out"][POSE_BRANCH], constant["held_out"].sample_ids, result["held_out_probabilities"], result["threshold"])
            )
        for seed, rows in constant_rows.items():
            vbc.write_oof_csv(audit_oof_seed_path(run_dir, CONSTANT_ARM, seed), rows)
        for permutation_seed in AUDIT_PERMUTATION_SEEDS:
            model_seed = audit_model_seed(permutation_seed)
            rows = [row for cell in cells if cell["seed"] == model_seed for row in permuted_fold(cell, permutation_seed)[0]]
            vbc.write_oof_csv(audit_oof_seed_path(run_dir, f"fused_perm{permutation_seed}", model_seed), rows)
    payload = leak_audit_from_disk(run_dir, expectations)
    write_json(leak_audit_path(run_dir), payload)
    return payload


def leak_audit_state(run_dir: Path, expectations: Expectations) -> dict:
    """Whether the report may release outcomes, decided from disk and not from the stored file's say-so.

    The stored ``leak_audit.json`` must equal the audit recomputed now. That one comparison
    covers a fuse that ran after the audit (``calibration.json`` changed), an OOF file
    edited since, thresholds that differ from the frozen ones, and a hand-edited record.
    """
    current = leak_audit_from_disk(run_dir, expectations)
    if not leak_audit_path(run_dir).exists():
        return {"released": False, "reason": "leak_audit.json is missing; run `fuse`", "leak_audit": current}
    if read_json(leak_audit_path(run_dir)) != current:
        return {"released": False, "reason": "leak_audit.json is stale or was edited; rerun `fuse`", "leak_audit": current}
    if not current["released"]:
        return {
            "released": False,
            "reason": "the leak audit fired and did not clear; the pipeline is treated as leaking until it is repaired",
            "leak_audit": current,
        }
    return {"released": True, "reason": "trigger did not fire" if not current["trigger"]["fired"] else "audit cleared", "leak_audit": current}


# --------------------------------------------------------------------------- #
# position control                                                             #
# --------------------------------------------------------------------------- #


def join_to_manifest(oof_rows: Sequence[dict], manifest_rows: dict[str, dict]) -> list[dict]:
    """Rebuild every row's repetition and session key from the manifest by ``sample_id``.

    The position helpers key on strings and skip a key they cannot match without saying
    so; two modules also spell the session key differently. Joining on ``sample_id`` and
    raising on the first row that does not resolve, or that disagrees with the manifest,
    is what makes "zero unmatched rows" a checked property.
    """
    joined: list[dict] = []
    for row in oof_rows:
        manifest_row = manifest_rows.get(row["sample_id"])
        if manifest_row is None:
            raise ValueError(f"OOF row {row['sample_id']!r} has no manifest row.")
        if manifest_row["person_id"] != row["person_id"] or int(manifest_row["correctness"]) != row["label"]:
            raise ValueError(f"OOF row {row['sample_id']!r} disagrees with the manifest on person or label.")
        joined.append({**row, "repetition_id": vbc.repetition_key(manifest_row), "session_id": vbc.session_key(manifest_row)})
    return joined


def position_balanced_statistic(
    oof_rows: Sequence[dict],
    manifest_rows: dict[str, dict],
    seeds: Sequence[int],
    expected_mixed: int = EXPECTED_MIXED_SESSIONS,
    expected_interleaved: int = EXPECTED_INTERLEAVED_SESSIONS,
) -> dict:
    """Position-balanced AUC on the interleaved sessions, inputs as the plan locks them."""
    scores = vbc._camera_averaged_repetition_scores(join_to_manifest(oof_rows, manifest_rows), seeds)
    sessions = build_pair_sessions(scores, seeds)  # drops P10; concordances averaged over seeds
    if len(sessions) != expected_mixed:
        raise ValueError(f"{len(sessions)} mixed-label sessions after the join, expected {expected_mixed}.")
    mask = interleaved_mask(sessions)
    interleaved = [session for session, keep in zip(sessions, mask) if keep]
    if len(interleaved) != expected_interleaved:
        raise ValueError(f"{len(interleaved)} interleaved sessions after the join, expected {expected_interleaved}.")
    statistic = position_balanced_auc(interleaved, [session.concordance for session in interleaved], max_distance=None)
    if statistic["n_sessions"] != expected_interleaved:
        raise ValueError(f"Position-balanced AUC covered {statistic['n_sessions']} sessions, expected {expected_interleaved}.")
    return statistic


def within_session_statistic(
    oof_rows: Sequence[dict],
    manifest_rows: dict[str, dict],
    seeds: Sequence[int],
    expected_mixed: int = EXPECTED_MIXED_SESSIONS,
) -> dict:
    """Within-session ROC-AUC over exactly the mixed-label sessions, or an error.

    ``within_session_auc_statistic`` averages whatever sessions it is given. A secondary
    computed on 60 sessions would still look like a number, so the count is asserted here.
    """
    statistic = vbc.within_session_auc_statistic(join_to_manifest(oof_rows, manifest_rows), seeds)
    if statistic.get("status") != "complete":
        raise ValueError(f"Within-session AUC is not computable: {statistic.get('reason')}")
    if statistic["n_sessions"] != expected_mixed:
        raise ValueError(f"Within-session AUC covers {statistic['n_sessions']} sessions, expected {expected_mixed}.")
    return statistic


def manifest_session_counts(manifest_rows: dict[str, dict]) -> dict:
    """Mixed and interleaved session counts from labels alone -- no model score involved."""
    scores: dict[str, dict] = {}
    for row in manifest_rows.values():
        scores.setdefault(
            vbc.repetition_key(row),
            {
                "session": vbc.session_key(row),
                "person_id": row["person_id"],
                "exercise_id": row["exercise_id"],
                "label": int(row["correctness"]),
                "score_by_seed": {0: 0.0},
            },
        )
    sessions = build_pair_sessions(scores, (0,))
    mask = interleaved_mask(sessions)
    per_subject: dict[str, int] = defaultdict(int)
    for session, keep in zip(sessions, mask):
        per_subject[session.person_id] += int(keep)
    return {
        "mixed_sessions": len(sessions),
        "interleaved_sessions": int(sum(mask)),
        "interleaved_by_subject": dict(sorted(per_subject.items(), key=lambda item: int(item[0]))),
    }


def position_correlation_diagnostic(
    oof_rows: Sequence[dict],
    manifest_rows: dict[str, dict],
    seeds: Sequence[int],
    min_reps: int = PROBE_MIN_REPS,
) -> dict:
    """Spearman between an arm's probability and within-class position, per (session, class) cell.

    Probabilities are averaged over the two cameras, then over seeds. Every repetition of
    a primary subject must carry a score: ``probe_cells`` drops an unscored repetition
    silently, which would shrink cells without a trace.
    """
    repetitions = build_repetitions(list(manifest_rows.values()))
    by_sample = {sample_id: repetition for repetition in repetitions for sample_id in repetition.sample_ids}
    by_seed: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in oof_rows:
        repetition = by_sample.get(row["sample_id"])
        if repetition is None:
            raise ValueError(f"OOF row {row['sample_id']!r} does not resolve to a repetition.")
        by_seed[repetition.repetition][row["seed"]].append(row["probability"])

    predictions: dict[str, float] = {}
    for key, per_seed in by_seed.items():
        for seed in seeds:
            if len(per_seed.get(seed, [])) != len(CAMERAS):
                raise ValueError(f"Repetition {key} has {len(per_seed.get(seed, []))} rows for seed {seed}, expected {len(CAMERAS)}.")
        predictions[key] = float(np.mean([np.mean(per_seed[seed]) for seed in seeds]))
    unscored = sorted(
        repetition.repetition
        for repetition in repetitions
        if repetition.person_id != vbc.SENSITIVITY_SUBJECT and repetition.repetition not in predictions
    )
    if unscored:
        raise ValueError(f"{len(unscored)} primary-subject repetitions carry no score: {unscored[:5]}")

    by_subject: dict[str, list[float]] = defaultdict(list)
    for cell in probe_cells(repetitions, predictions, min_reps=min_reps):
        correlation = rank_correlation(cell.prediction_ranks, cell.target_ranks)
        if correlation is not None:
            by_subject[cell.person_id].append(correlation)
    per_subject = {subject: float(np.median(values)) for subject, values in sorted(by_subject.items(), key=lambda item: int(item[0]))}
    return {
        "per_subject_median": per_subject,
        "median_over_subjects": float(np.median(list(per_subject.values()))) if per_subject else None,
        "n_cells": int(sum(len(values) for values in by_subject.values())),
        "min_reps": min_reps,
    }


# --------------------------------------------------------------------------- #
# gates                                                                        #
# --------------------------------------------------------------------------- #


def frozen_config_problems(config: dict, expectations: Expectations) -> list[str]:
    """``run_config.json`` against the module constants: a config edited by hand cannot pass."""
    frozen = {
        "seeds": list(SEEDS),
        "practical_margin": PRACTICAL_MARGIN,
        "alpha": ALPHA,
        "bootstrap_resamples": vbc.BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": vbc.BOOTSTRAP_SEED,
        "permutation_pairs": [list(pair) for pair in PERMUTATION_PAIRS],
        "weight_grid": list(WEIGHT_GRID),
        "required_device": REQUIRED_DEVICE,
        "source_run_dir": str(expectations.source_run_dir),
        "arm_feature_dirs": {arm: str(path) for arm, path in expectations.arm_feature_dirs.items()},
        "arm_dims": dict(expectations.arm_dims),
        "fold_config": dict(expectations.fold_config),
        "baseline_trainer": {
            "commit": expectations.baseline_trainer_commit,
            "blob_sha256": expectations.baseline_trainer_blob_sha256,
            "path": TRAINER_PATH,
        },
        "leak_audit": {
            "trigger": expectations.audit_trigger,
            "clear_threshold": expectations.audit_clear_threshold,
            "permutation_seeds": list(range(1001, 1021)),
            "model_seeds": [42, 7, 1234, 42, 7, 1234, 42, 7, 1234, 42, 7, 1234, 42, 7, 1234, 42, 7, 1234, 42, 7],
        },
    }
    problems = [f"run_config {key} = {config.get(key)!r}, frozen value is {value!r}" for key, value in frozen.items() if config.get(key) != value]
    if asdict(loso.FoldConfig()) != FROZEN_FOLD_CONFIG:
        problems.append(f"loso.FoldConfig() defaults {asdict(loso.FoldConfig())!r} differ from the frozen recipe {FROZEN_FOLD_CONFIG!r}")
    if list(AUDIT_PERMUTATION_SEEDS) != frozen["leak_audit"]["permutation_seeds"]:
        problems.append("AUDIT_PERMUTATION_SEEDS is not 1001-1020")
    return problems


def missing_stamps(run_dir: Path) -> list[str]:
    """Fits and A/B cells that exist without a complete stamp. No stamp means no evidence of what ran."""
    problems: list[str] = []
    for arm in FIT_ARM_NAMES:
        log_path = fit_log_path(run_dir, arm)
        seeds = read_json(log_path)["seeds"] if log_path.exists() else {}
        for path in sorted(vbc.oof_dir(run_dir, arm).glob("seed*.csv")):
            entry = seeds.get(path.stem[len("seed"):], {})
            absent = [key for key in ("source", "device", "fold_config", "started_utc") if key not in entry]
            if absent:
                problems.append(f"fit {arm} {path.stem}: OOF rows exist without a fit-log stamp ({absent} missing)")
    if ab_check_path(run_dir).exists():
        for key, cell in read_json(ab_check_path(run_dir))["cells"].items():
            absent = [name for name in ("source", "device", "baseline_trainer") if name not in cell]
            if absent:
                problems.append(f"ab-check {key}: cell carries no stamp ({absent} missing)")
    return problems


def execution_stamps(run_dir: Path) -> dict[str, dict]:
    """Every stamped execution of the run: each fit (the P10-removed refits included), each A/B cell, each fuse."""
    stamps: dict[str, dict] = {}
    for path in sorted((run_dir / "oof").glob("*/fit_log.json")):
        for seed, entry in read_json(path)["seeds"].items():
            stamps[f"fit {path.parent.name} seed{seed}"] = entry
    if ab_check_path(run_dir).exists():
        for key, cell in read_json(ab_check_path(run_dir))["cells"].items():
            stamps[f"ab-check {key}"] = cell
    for suffix in ("", NO_P10_SUFFIX):
        if calibration_path(run_dir, suffix).exists():
            stamps[f"fuse{suffix}"] = read_json(calibration_path(run_dir, suffix))["execution"]
    return stamps


def gate_frozen_protocol(run_dir: Path, expectations: Expectations) -> dict:
    if not vbc.run_config_path(run_dir).exists():
        return {"status": "pending", "evidence": "run_config.json missing; run `init`"}
    config = read_json(vbc.run_config_path(run_dir))
    problems: list[str] = []
    observed = {
        "folds.json": file_sha256(vbc.folds_path(run_dir)) if vbc.folds_path(run_dir).exists() else None,
        "folds_no_p10.json": file_sha256(vbc.folds_path(run_dir, no_p10=True)) if vbc.folds_path(run_dir, no_p10=True).exists() else None,
        "manifest_md5": vbc.file_md5(Path(config["manifest_path"])) if Path(config["manifest_path"]).exists() else None,
    }
    expected = {
        "folds.json": expectations.folds_sha256,
        "folds_no_p10.json": expectations.folds_no_p10_sha256,
        "manifest_md5": expectations.manifest_md5,
    }
    problems.extend(f"{name}: {observed[name]} != {expected[name]}" for name in expected if observed[name] != expected[name])
    problems.extend(frozen_config_problems(config, expectations))

    if expectations.require_clean_source:
        if not config.get("git_commit"):
            problems.append("run_config carries no git commit")
        if config.get("source_dirty"):
            problems.append(f"protocol files were not committed at init: {config['source_dirty']}")

    problems.extend(missing_stamps(run_dir))
    stamps = execution_stamps(run_dir)
    for name, stamp in stamps.items():
        if stamp.get("device") != REQUIRED_DEVICE:
            problems.append(f"{name}: ran on device {stamp.get('device')!r}, not {REQUIRED_DEVICE!r}")
        if "fold_config" in stamp and stamp["fold_config"] != config.get("fold_config"):
            problems.append(f"{name}: readout configuration differs from run_config")
        if "started_utc" in stamp and stamp["started_utc"] < config["created_utc"]:
            problems.append(f"{name}: started before run_config.json was written")
        if expectations.require_clean_source:
            source = stamp.get("source", {})
            if source.get("dirty"):
                problems.append(f"{name}: ran on uncommitted protocol files {source['dirty']}")
            if source.get("git_commit") != config.get("git_commit"):
                problems.append(f"{name}: ran on commit {source.get('git_commit')}, init recorded {config.get('git_commit')}")
    return {
        "status": "fail" if problems else "pass",
        "evidence": {"problems": problems[:30], "n_problems": len(problems), "hashes": observed, "executions_checked": len(stamps)},
    }


def gate_feature_integrity(run_dir: Path) -> dict:
    if not feature_integrity_path(run_dir).exists():
        return {"status": "pending", "evidence": "feature_integrity.json missing; run `features`"}
    audit = read_json(feature_integrity_path(run_dir))
    status = "pass" if all(audit.get(arm, {}).get("passed") for arm in BRANCHES) else "fail"
    return {"status": status, "evidence": audit}


def gate_inert_trainer_change(run_dir: Path, expectations: Expectations) -> dict:
    """Every arm x fold x seed cell bit-identical against the trainer pinned at the plan commit.

    Whether the cells ran on a clean tree, on the commit ``init`` recorded and on the CPU
    is the frozen-protocol gate's question; this one asks only what was compared with what.
    """
    if not ab_check_path(run_dir).exists():
        return {"status": "pending", "evidence": "ab_check.json missing; run `ab-check`"}
    cells = read_json(ab_check_path(run_dir))["cells"]
    expected = {
        f"{arm}|P{fold['test_subject']}|seed{seed}"
        for arm in BRANCHES
        for seed in SEEDS
        for fold in vbc.load_folds(vbc.folds_path(run_dir))
    }
    present = sorted(expected & set(cells))
    missing = sorted(expected - set(cells))
    different = [key for key in present if not cells[key]["identical"]]
    baselines = sorted({(cells[key]["baseline_trainer"]["commit"], cells[key]["baseline_trainer"]["blob_sha256"]) for key in present})
    wrong_baseline = [
        key
        for key in present
        if cells[key]["baseline_trainer"]["commit"] != expectations.baseline_trainer_commit
        or cells[key]["baseline_trainer"]["blob_sha256"] != expectations.baseline_trainer_blob_sha256
    ]
    stored = {key: cells[key]["stored_run"] for key in present if "stored_run" in cells[key]}
    evidence = {
        "cells_expected": len(expected),
        "cells_missing": missing[:10],
        "cells_not_identical": different,
        "baseline_trainers": [{"commit": commit, "blob_sha256": sha} for commit, sha in baselines],
        "pinned_baseline": {"commit": expectations.baseline_trainer_commit, "blob_sha256": expectations.baseline_trainer_blob_sha256},
        "cells_against_another_baseline": wrong_baseline[:10],
        "stored_20260918_vm16_non_blocking": {
            "compared": len(stored),
            "not_bit_identical": [key for key, value in stored.items() if not value.get("bit_identical")],
        },
    }
    if different or wrong_baseline or len(baselines) > 1:
        return {"status": "fail", "evidence": evidence}
    return {"status": "pending" if missing else "pass", "evidence": evidence}


def gate_validation_export(run_dir: Path) -> dict:
    folds = vbc.load_folds(vbc.folds_path(run_dir))
    objective = loso.FoldConfig().threshold_objective
    problems: list[str] = []
    for branch in BRANCHES:
        validation = load_val_oof(run_dir, branch)
        held_out = vbc.load_arm_oof(run_dir, branch)
        if validation is None or held_out is None:
            return {"status": "pending", "evidence": f"{branch} has not been fitted"}
        validation_cells = rows_by_fold_seed(validation)
        held_out_cells = rows_by_fold_seed(held_out)
        for seed in SEEDS:
            for fold in folds:
                cell = (fold["test_subject"], seed)
                rows = validation_cells.get(cell, [])
                name = f"{branch} P{cell[0]} seed{seed}"
                ids = [row["sample_id"] for row in rows]
                if sorted(ids) != sorted(fold["val_ids"]):
                    problems.append(f"{name}: validation ids are not exactly the fold's validation ids")
                    continue
                if {row["person_id"] for row in rows} != {fold["val_subject"]}:
                    problems.append(f"{name}: validation rows are not all P{fold['val_subject']}")
                stored = {row["threshold"] for row in rows} | {row["threshold"] for row in held_out_cells.get(cell, [])}
                # float32, as the trainer scored them: the search casts its candidates to float32.
                probabilities = np.asarray([row["probability"] for row in rows], dtype=np.float32)
                recomputed, _ = find_best_threshold(probabilities, np.asarray([row["label"] for row in rows]), objective)
                if stored != {float(recomputed)}:
                    problems.append(f"{name}: threshold from exported validation probabilities {recomputed} != stored {sorted(stored)}")
    return {"status": "fail" if problems else "pass", "evidence": problems[:20]}


def gate_fold_purity(run_dir: Path) -> dict:
    """What each calibration was actually fitted on, against ``folds.json`` as it sits on disk.

    ``fuse`` records, per fold and seed, the ``person_id`` values and the id hash of the
    rows it handed to ``fit_platt`` and ``find_best_threshold``, read off those rows. This
    gate compares that record with the fold file, which ``fuse`` never writes.
    """
    structure = vbc.gate_fold_purity(run_dir)
    if structure["status"] != "pass":
        return structure
    if not calibration_path(run_dir).exists():
        return {"status": "pending", "evidence": "calibration.json missing; run `fuse`"}
    folds = {fold["test_subject"]: fold for fold in vbc.load_folds(vbc.folds_path(run_dir))}
    records = read_json(calibration_path(run_dir))["folds"]
    problems: list[str] = []
    seen = {(record["test_subject"], record["seed"]) for record in records}
    expected = {(subject, seed) for subject in folds for seed in SEEDS}
    if seen != expected or len(records) != len(expected):
        problems.append(f"calibration.json covers {len(records)} cells, expected {len(expected)} distinct fold x seed cells")
    for record in records:
        fold = folds.get(record["test_subject"])
        name = f"P{record['test_subject']} seed{record['seed']}"
        if fold is None:
            problems.append(f"{name}: no such fold in folds.json")
            continue
        fitted = [record["fitted_on"]] + ([record["permuted_fitted_on"]] if "permuted_fitted_on" in record else [])
        for provenance in fitted:
            if provenance["person_ids"] != [fold["val_subject"]]:
                problems.append(f"{name}: calibrated on subjects {provenance['person_ids']}, folds.json says P{fold['val_subject']}")
            if fold["test_subject"] in provenance["person_ids"]:
                problems.append(f"{name}: the held-out subject reached a fit")
            if provenance["ids_sha256"] != vbc.hash_ids(fold["val_ids"]):
                problems.append(f"{name}: calibrated on a row set that is not the validation ids of the fold")
        if record["scored_on"]["person_ids"] != [fold["test_subject"]] or record["scored_on"]["ids_sha256"] != vbc.hash_ids(fold["test_ids"]):
            problems.append(f"{name}: scored rows are not exactly the held-out ids of the fold")
    return {"status": "fail" if problems else "pass", "evidence": {"problems": problems[:20], "cells_checked": len(records)}}


def ranking_evidence(raw: np.ndarray, calibrated: np.ndarray) -> dict:
    """How a calibrated ranking relates to the raw one, for a positive-slope Platt map.

    Two harmless things break strict rank equality, and both are counted, not failed.
    ``late_fusion.logit`` clips its input to [eps, 1 - eps], so raw probabilities outside
    that interval land on one value. And float64 cannot separate two sigmoid outputs that
    differ below its resolution, so two distinct interior probabilities can come out
    equal where the calibrated curve is flat. Neither reorders anything. An inversion -- a
    pair the calibration orders the other way round -- does, and so does a raw tie that
    comes apart; neither is ever excused.
    """
    raw = np.asarray(raw, dtype=np.float64)
    calibrated = np.asarray(calibrated, dtype=np.float64)
    interior = (raw > PROBABILITY_EPS) & (raw < 1.0 - PROBABILITY_EPS)
    raw_less = raw[:, None] < raw[None, :]
    inside = interior[:, None] & interior[None, :]
    return {
        "clipped_rows": int(np.sum(~interior)),
        "interior_saturation_ties": int(np.sum(raw_less & inside & (calibrated[:, None] == calibrated[None, :]))),
        "split_ties": int(np.sum(np.triu(raw[:, None] == raw[None, :], k=1) & (calibrated[:, None] != calibrated[None, :]))),
        "inversions": int(np.sum(raw_less & (calibrated[:, None] > calibrated[None, :]))),
        "interior_ranking_equal": bool(np.array_equal(midranks(raw[interior]), midranks(calibrated[interior]))),
    }


def gate_degenerate_share(run_dir: Path) -> dict:
    """Shares 1.0 and 0.0 return a calibrated branch bit for bit; a positive-slope Platt map reorders nothing."""
    if not calibration_path(run_dir).exists():
        return {"status": "pending", "evidence": "calibration.json missing; run `fuse`"}
    records = {(record["test_subject"], record["seed"]): record for record in read_json(calibration_path(run_dir))["folds"]}
    held_out = {branch: rows_by_fold_seed(vbc.load_arm_oof(run_dir, branch) or []) for branch in BRANCHES}
    validation = {branch: rows_by_fold_seed(load_val_oof(run_dir, branch) or []) for branch in BRANCHES}
    stored = {
        POSE_BRANCH: rows_by_fold_seed(vbc.load_arm_oof(run_dir, POSE_CALIBRATED_ARM) or []),
        VIDEO_BRANCH: rows_by_fold_seed(vbc.load_arm_oof(run_dir, VIDEO_CALIBRATED_ARM) or []),
    }
    problems: list[str] = []
    clipped: dict[str, int] = {}
    saturation_ties: dict[str, int] = {}
    skipped: list[str] = []
    rankings_checked = 0
    for cell, record in records.items():
        name = f"P{cell[0]} seed{cell[1]}"
        calibrated: dict[str, np.ndarray] = {}
        # The pose branch's row order is the order `fuse` calibrated both branches in; keeping it
        # here means the recomputation walks the same array positions as the run it checks.
        order = [row["sample_id"] for row in held_out[POSE_BRANCH].get(cell, [])]
        for branch in BRANCHES:
            calibration = PlattCalibration(float(record["platt"][branch]["slope"]), float(record["platt"][branch]["intercept"]))
            by_id = {row["sample_id"]: row for row in held_out[branch].get(cell, [])}
            if set(by_id) != set(order):
                problems.append(f"{name}: {branch} held-out ids differ from the pose branch's")
                continue
            rows = [by_id[sample_id] for sample_id in order]
            raw = np.asarray([row["probability"] for row in rows], dtype=np.float64)
            calibrated[branch] = calibration.apply(raw)
            stored_rows = {row["sample_id"]: row["probability"] for row in stored[branch].get(cell, [])}
            if [stored_rows.get(row["sample_id"]) for row in rows] != calibrated[branch].tolist():
                problems.append(f"{name}: stored {branch} degenerate arm is not the calibrated branch bit for bit")
            if calibration.slope <= 0:
                # The plan requires the ranking only where the slope is positive; a cell left out is named.
                skipped.append(f"{branch}|{name}|slope={record['platt'][branch]['slope']}")
                continue
            for split, split_rows in (("held_out", rows), ("validation", validation[branch].get(cell, []))):
                values = np.asarray([row["probability"] for row in split_rows], dtype=np.float64)
                evidence = ranking_evidence(values, calibration.apply(values))
                rankings_checked += 1
                clipped[f"{branch}|{name}|{split}"] = evidence["clipped_rows"]
                saturation_ties[f"{branch}|{name}|{split}"] = evidence["interior_saturation_ties"]
                if evidence["inversions"]:
                    problems.append(f"{name}: {branch} {split} calibration inverts {evidence['inversions']} pairs")
                if evidence["split_ties"]:
                    problems.append(f"{name}: {branch} {split} calibration separates {evidence['split_ties']} raw ties")
        if len(calibrated) == len(BRANCHES):
            if not np.array_equal(fuse_probabilities(calibrated[POSE_BRANCH], calibrated[VIDEO_BRANCH], 1.0), calibrated[POSE_BRANCH]):
                problems.append(f"{name}: share 1.0 is not the calibrated pose branch")
            if not np.array_equal(fuse_probabilities(calibrated[POSE_BRANCH], calibrated[VIDEO_BRANCH], 0.0), calibrated[VIDEO_BRANCH]):
                problems.append(f"{name}: share 0.0 is not the calibrated video branch")
    return {
        "status": "fail" if problems else "pass",
        "evidence": {
            "problems": problems[:20],
            "rankings_checked": rankings_checked,
            "cells_skipped_nonpositive_slope": skipped,
            "probability_eps": PROBABILITY_EPS,
            "clipped_rows_total": int(sum(clipped.values())),
            "clipped_rows_by_arm_fold_seed_split": {key: count for key, count in sorted(clipped.items()) if count},
            "interior_saturation_ties_total": int(sum(saturation_ties.values())),
            "interior_saturation_ties_by_arm_fold_seed_split": {key: count for key, count in sorted(saturation_ties.items()) if count},
        },
    }


def oof_integrity_problems(rows: Sequence[dict], manifest_rows: dict[str, dict], labels: dict[str, int], seeds: Sequence[int]) -> list[str]:
    problems: list[str] = []
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row)
    if sorted(by_seed) != sorted(seeds):
        problems.append(f"seeds present {sorted(by_seed)} != expected {sorted(seeds)}")
    for seed, seed_rows in by_seed.items():
        ids = [row["sample_id"] for row in seed_rows]
        if len(ids) != len(set(ids)):
            problems.append(f"seed {seed}: duplicate sample ids")
        if set(ids) != set(manifest_rows):
            problems.append(f"seed {seed}: {len(set(manifest_rows) - set(ids))} missing and {len(set(ids) - set(manifest_rows))} unexpected ids")
        for row in seed_rows:
            manifest_row = manifest_rows.get(row["sample_id"])
            if manifest_row is None:
                continue
            if manifest_row["person_id"] != row["test_subject"] or row["person_id"] != row["test_subject"]:
                problems.append(f"seed {seed}: {row['sample_id']} scored outside its own held-out fold")
            if labels.get(row["sample_id"]) != row["label"]:
                problems.append(f"seed {seed}: {row['sample_id']} label disagrees with the frozen labels")
    return problems


def gate_fused_oof_integrity(run_dir: Path, manifest_rows: dict[str, dict], labels: dict[str, int]) -> dict:
    per_arm: dict[str, object] = {}
    for arm in (FUSED_ARM, POSE_CALIBRATED_ARM, VIDEO_CALIBRATED_ARM, TUNED_ARM, *BRANCHES):
        rows = vbc.load_arm_oof(run_dir, arm)
        if rows is None:
            return {"status": "pending", "evidence": f"{arm} has no OOF rows yet"}
        per_arm[arm] = oof_integrity_problems(rows, manifest_rows, labels, SEEDS)[:10]
    for permutation_seed, model_seed in PERMUTATION_PAIRS:
        rows = vbc.load_arm_oof(run_dir, f"fused_perm{permutation_seed}")
        if rows is None:
            return {"status": "pending", "evidence": f"fused_perm{permutation_seed} has no OOF rows yet"}
        per_arm[f"fused_perm{permutation_seed}"] = oof_integrity_problems(rows, manifest_rows, labels, (model_seed,))[:10]
    return {"status": "fail" if any(per_arm.values()) else "pass", "evidence": per_arm}


def gate_position_reproduction(run_dir: Path, manifest_rows: dict[str, dict], segmentation_path: Path, expectations: Expectations) -> dict:
    """Counts only: this gate reads no arm's AUC, so it opens no outcome."""
    counts = manifest_session_counts(manifest_rows)
    rule = position_rule_loso(segmentation_path)
    evidence: dict = {"manifest": counts, "position_rule_loso": {"mean": rule["mean"], "sd": rule["sd"]}, "arms": {}}
    problems: list[str] = []
    if round(rule["mean"], 4) != expectations.position_rule_ba:
        problems.append(f"position rule {rule['mean']:.4f} != {expectations.position_rule_ba}")
    if counts["mixed_sessions"] != expectations.mixed_sessions or counts["interleaved_sessions"] != expectations.interleaved_sessions:
        problems.append(f"session counts {counts['mixed_sessions']}/{counts['interleaved_sessions']} differ from the expected split")

    pending = False
    for arm in (POSE_BRANCH, VIDEO_BRANCH, FUSED_ARM):
        rows = vbc.load_arm_oof(run_dir, arm)
        if rows is None:
            pending = True
            continue
        try:
            statistic = position_balanced_statistic(rows, manifest_rows, SEEDS, expectations.mixed_sessions, expectations.interleaved_sessions)
            evidence["arms"][arm] = {"n_sessions": statistic["n_sessions"], "n_subjects": statistic["n_subjects"], "unmatched_rows": 0}
        except ValueError as exc:
            problems.append(f"{arm}: {exc}")
    evidence["problems"] = problems
    if problems:
        return {"status": "fail", "evidence": evidence}
    return {"status": "pending" if pending else "pass", "evidence": evidence}


def compute_gates(run_dir: Path, expectations: Expectations | None = None, write: bool = True) -> dict:
    expectations = expectations or Expectations()
    if not vbc.run_config_path(run_dir).exists():
        raise SystemExit(f"{run_dir} has no run_config.json. Run `init` first.")
    config, manifest_rows, labels = load_run_inputs(run_dir)
    gates = {
        "frozen_protocol": gate_frozen_protocol(run_dir, expectations),
        "feature_integrity": gate_feature_integrity(run_dir),
        "inert_trainer_change": gate_inert_trainer_change(run_dir, expectations),
        "validation_export_integrity": gate_validation_export(run_dir),
        "fold_purity": gate_fold_purity(run_dir),
        "degenerate_share_identity": gate_degenerate_share(run_dir),
        "fused_oof_integrity": gate_fused_oof_integrity(run_dir, manifest_rows, labels),
        "position_reproduction": gate_position_reproduction(run_dir, manifest_rows, Path(config["segmentation_path"]), expectations),
    }
    payload = {"gates": gates, "passed": all(gates[name]["status"] == "pass" for name in GATE_NAMES)}
    if write:
        write_json(vbc.gates_path(run_dir), payload)
    return payload


# --------------------------------------------------------------------------- #
# report                                                                       #
# --------------------------------------------------------------------------- #


def fusion_verdict(
    primary: dict,
    gates_pass: bool,
    fused_mean: float,
    video_mean: float,
    pose_calibrated_mean: float,
    margin: float = PRACTICAL_MARGIN,
    alpha: float = ALPHA,
) -> dict:
    """The plan's interpretation table; the first matching row applies.

    Row 0 is checked before any significance branch: a failed gate means there is no
    valid comparison, not a downgraded one. The two mean comparisons in row 1 are
    point-estimate guards, not tests.
    """
    ci = primary["bootstrap_ci_95"]
    p_value = primary["sign_flip"]["p_value"]
    delta = primary["mean_delta"]
    conditions = {
        "p_below_alpha": p_value < alpha,
        "delta_positive": delta > 0,
        "delta_at_least_margin": delta >= margin,
        "at_least_7_of_9_positive": primary["n_positive"] / primary["n_subjects"] >= MIN_POSITIVE_FRACTION - 1e-9,
        "ci_excludes_zero": ci[0] > 0 or ci[1] < 0,
        "fused_mean_at_least_vm16_mean": fused_mean >= video_mean,
        "fused_mean_at_least_nlf_cal_mean": fused_mean >= pose_calibrated_mean,
    }
    failed: list[str] = []
    if not gates_pass:
        row, reading = 0, "gate failure: no valid comparison"
    elif conditions["p_below_alpha"] and conditions["delta_positive"]:
        failed = [name for name, met in conditions.items() if not met]
        if failed:
            row, reading = 2, "detectable positive difference that does not meet the claim conditions; no gain claimed"
        else:
            row, reading = 1, "gain claimed: the fusion beats NLF and is not below either single-branch arm under this rule"
    elif conditions["p_below_alpha"] and delta < 0:
        row, reading = 3, "adding the video branch degrades the pose classifier under this rule"
    else:
        row, reading = 4, "undetermined: no claim that the branches are redundant"
    return {
        "row": row,
        "reading": reading,
        "conditions": conditions,
        "failed_conditions": failed,
        "stop_rule_applies": row != 1 and row != 0,
    }


def position_reading(verdict_row: int, s4: dict, holm_s4: dict) -> str:
    if verdict_row != 1:
        return "S4 is reported but changes nothing"
    if s4["mean_delta"] > 0 and holm_s4["significant"]:
        return "the gain is present where repetition order carries no information"
    return "the gain is not separated from recording position; it may not be described as movement information added by video"


def descriptive_delta(candidate: dict[str, float], baseline: dict[str, float], subjects: Sequence[str]) -> dict:
    """Subject deltas with no test attached -- for contrasts the plan reports but never tests."""
    deltas = vbc.paired_deltas(candidate, baseline, subjects)
    return {
        "per_subject": dict(zip(subjects, deltas)),
        "mean_delta": float(np.mean(deltas)),
        "sd_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        "n_positive": int(sum(delta > 0 for delta in deltas)),
    }


def uninformative_branch_control(run_dir: Path, leak_audit: dict) -> dict:
    """The released form of the control: the audit record plus ``fused_perm - nlf``.

    ``fused_perm - nlf`` is measured against an arm of the experiment, so it is written
    only here, once outcomes are released. While they are withheld, ``leak_audit.json``
    and the summary carry deltas against ``nlf_cal`` and ``fused_const`` and nothing else.
    """
    pose = vbc.load_arm_oof(run_dir, POSE_BRANCH)
    runs = {}
    for permutation_seed, model_seed in PERMUTATION_PAIRS:
        permuted = vbc.load_arm_oof(run_dir, f"fused_perm{permutation_seed}")
        runs[str(permutation_seed)] = {
            **leak_audit["trigger"]["runs"][str(permutation_seed)],
            "fused_perm_minus_nlf": subject_delta(permuted, rows_of_seed(pose, model_seed)),
        }
    return {**leak_audit, "trigger": {**leak_audit["trigger"], "runs": runs}}


def brier_by_subject(rows: Sequence[dict], subjects: Sequence[str]) -> dict[str, float]:
    """Brier score per (held-out subject, seed) cell, averaged over seeds within subject."""
    cells: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["test_subject"] in subjects:
            cells[row["test_subject"]][row["seed"]].append((row["probability"] - row["label"]) ** 2)
    return {subject: float(np.mean([np.mean(values) for values in cells[subject].values()])) for subject in subjects}


def brier_summary(run_dir: Path, subjects: Sequence[str] = vbc.PRIMARY_SUBJECTS) -> dict:
    """Held-out Brier score of each branch before and after Platt scaling, from the OOF rows.

    Computed here, in the released report, and nowhere earlier: ``fuse`` runs before the
    gates and the leak audit have released anything, and these are scores of ``nlf`` and
    ``vm16`` on held-out labels.
    """
    arms = {POSE_BRANCH: {"raw": POSE_BRANCH, "calibrated": POSE_CALIBRATED_ARM}, VIDEO_BRANCH: {"raw": VIDEO_BRANCH, "calibrated": VIDEO_CALIBRATED_ARM}}
    summary: dict = {}
    for branch, kinds in arms.items():
        for kind, arm in kinds.items():
            per_subject = brier_by_subject(vbc.load_arm_oof(run_dir, arm) or [], subjects)
            summary.setdefault(branch, {})[kind] = {"mean": float(np.mean(list(per_subject.values()))), "per_subject": per_subject}
    return summary


def arm_row(per_subject: dict[str, float]) -> dict:
    values = list(per_subject.values())
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "per_subject": per_subject}


def withheld_summary(run_dir: Path, gates: dict, reason: str, extra: dict | None = None) -> dict:
    """Row 0 on disk: gates and the reason, no primary, no secondary, no score of any arm of the experiment."""
    summary = {
        "gates": gates,
        "verdict": {"row": 0, "reading": f"no valid comparison: {reason}", "stop_rule_applies": False},
        "outcomes": f"withheld: {reason}",
        **(extra or {}),
    }
    write_json(vbc.summary_path(run_dir), summary)
    return summary


def run_report(run_dir: Path, expectations: Expectations | None = None) -> dict:
    """Gates, then the leak audit, then -- only if both allow it -- the outcomes.

    Two conditions keep every primary, secondary and verdict out of ``summary.json``: a
    gate that has not passed, and a leak audit that is missing, stale, or fired without
    clearing. Both are the plan's row 0. Nothing on the command line overrides either.
    """
    expectations = expectations or Expectations()
    gates = compute_gates(run_dir, expectations)
    if not gates["passed"]:
        # Held-out fused scores stay unopened until every gate that precedes them passes.
        return withheld_summary(run_dir, gates, "not every gate has passed")
    audit_state = leak_audit_state(run_dir, expectations)
    if not audit_state["released"]:
        return withheld_summary(run_dir, gates, audit_state["reason"], {"leak_audit": audit_state["leak_audit"]})

    config, manifest_rows, _ = load_run_inputs(run_dir)
    subjects = vbc.PRIMARY_SUBJECTS
    descriptive_arms = (VIDEO_CALIBRATED_ARM, TUNED_ARM, *(f"fused_perm{permutation_seed}" for permutation_seed, _ in PERMUTATION_PAIRS))
    arms = {arm: vbc.load_arm_oof(run_dir, arm) for arm in (*MAIN_TABLE_ARMS, *descriptive_arms)}
    ba = {arm: vbc.subject_mean_metric(rows, subjects, "balanced_accuracy") for arm, rows in arms.items()}
    mean_ba = {arm: float(np.mean([values[subject] for subject in subjects])) for arm, values in ba.items()}
    auc = {
        arm: within_session_statistic(arms[arm], manifest_rows, SEEDS, expectations.mixed_sessions)
        for arm in (POSE_BRANCH, VIDEO_BRANCH, FUSED_ARM)
    }
    balanced = {
        arm: position_balanced_statistic(arms[arm], manifest_rows, SEEDS, expectations.mixed_sessions, expectations.interleaved_sessions)
        for arm in (POSE_BRANCH, VIDEO_BRANCH, FUSED_ARM)
    }

    primary = vbc.contrast_summary(ba[FUSED_ARM], ba[POSE_BRANCH], subjects)
    verdict = fusion_verdict(primary, gates["passed"], mean_ba[FUSED_ARM], mean_ba[VIDEO_BRANCH], mean_ba[POSE_CALIBRATED_ARM])
    verdict["leak_audit"] = audit_state["reason"]

    secondary = {
        "S1_fused_minus_vm16_ba": vbc.contrast_summary(ba[FUSED_ARM], ba[VIDEO_BRANCH], subjects),
        "S2_fused_minus_nlf_cal_ba": vbc.contrast_summary(ba[FUSED_ARM], ba[POSE_CALIBRATED_ARM], subjects),
        "S3_fused_minus_nlf_within_session_auc": vbc.contrast_summary(
            auc[FUSED_ARM]["per_subject_auc"], auc[POSE_BRANCH]["per_subject_auc"], subjects
        ),
        "S4_fused_minus_nlf_position_balanced_auc": vbc.contrast_summary(
            balanced[FUSED_ARM]["per_subject"], balanced[POSE_BRANCH]["per_subject"], subjects
        ),
    }
    holm = holm_correct({name: contrast["sign_flip"]["p_value"] for name, contrast in secondary.items()})
    s4 = "S4_fused_minus_nlf_position_balanced_auc"

    rule = position_rule_loso(Path(config["segmentation_path"]))
    calibrated_for = {POSE_BRANCH: POSE_CALIBRATED_ARM, VIDEO_BRANCH: VIDEO_CALIBRATED_ARM, FUSED_ARM: FUSED_ARM}
    calibration = read_json(calibration_path(run_dir))
    tuned_shares = [record["tuned_pose_share"] for record in calibration["folds"] if record["test_subject"] in subjects]

    summary = {
        "gates": gates,
        "verdict": verdict,
        "primary": primary,
        "secondary": secondary,
        "holm": holm,
        "main_table": {
            **{arm: arm_row(ba[arm]) for arm in MAIN_TABLE_ARMS},
            "position_rule_reference": {"mean": rule["mean"], "sd": rule["sd"], "per_subject": rule["per_subject"]},
            "fused_below_position_rule": bool(mean_ba[FUSED_ARM] < rule["mean"]),
        },
        "position_control": {
            "position_balanced_auc": balanced,
            "reading": position_reading(verdict["row"], secondary[s4], holm[s4]),
            "probability_vs_position": {
                arm: position_correlation_diagnostic(arms[calibrated_for[arm]], manifest_rows, SEEDS) for arm in calibrated_for
            },
        },
        "descriptive": {
            "nlf_cal_minus_nlf_ba": descriptive_delta(ba[POSE_CALIBRATED_ARM], ba[POSE_BRANCH], subjects),
            "other_arms_ba": {arm: arm_row(ba[arm]) for arm in descriptive_arms},
            "within_session_auc": {arm: {"mean": stat["mean"], "sd": stat["sd"], "per_subject": stat["per_subject_auc"]} for arm, stat in auc.items()},
            "vm16_minus_nlf_within_session_auc": descriptive_delta(auc[VIDEO_BRANCH]["per_subject_auc"], auc[POSE_BRANCH]["per_subject_auc"], subjects),
            "vm16_minus_nlf_position_balanced_auc": descriptive_delta(balanced[VIDEO_BRANCH]["per_subject"], balanced[POSE_BRANCH]["per_subject"], subjects),
            "arms": {arm: vbc.arm_descriptive_metrics(run_dir, arm, arms[arm], subjects) for arm in (*MAIN_TABLE_ARMS, VIDEO_CALIBRATED_ARM, TUNED_ARM)},
            "platt": {"negative_slopes": calibration["negative_slopes"], "brier_held_out": brier_summary(run_dir)},
            "validation_tuned_weight": {
                "fused_tuned_minus_nlf_ba": descriptive_delta(ba[TUNED_ARM], ba[POSE_BRANCH], subjects),
                "pose_share_counts": {str(share): tuned_shares.count(share) for share in WEIGHT_GRID},
            },
            "uninformative_branch_control": uninformative_branch_control(run_dir, audit_state["leak_audit"]),
        },
    }
    write_json(vbc.summary_path(run_dir), summary)
    return summary


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True, help="Directory name under --run-root.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrated late fusion of NLF pose and VideoMAE on REHAB24-6.")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="Assert the frozen hashes, copy the folds, write run_config.json.")
    add_common(init_parser)
    init_parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    init_parser.add_argument("--labels", type=Path, default=DEFAULT_PROCESSED_ROOT / "labels" / "correctness.json")
    init_parser.add_argument("--segmentation", type=Path, default=DEFAULT_SEGMENTATION)

    add_common(sub.add_parser("features", help="Id-by-id integrity of both feature directories."))

    ab_parser = sub.add_parser("ab-check", help="Exporting trainer against the unmodified trainer, bit for bit.")
    add_common(ab_parser)
    ab_parser.add_argument("--arm", choices=list(BRANCHES), nargs="+", default=list(BRANCHES))
    ab_parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), choices=list(SEEDS))
    ab_parser.add_argument("--test-subjects", nargs="+", default=None, help="Restrict to these folds (smoke runs).")

    fit_parser = sub.add_parser("fit", help="Refit one branch on the frozen folds with the validation export.")
    add_common(fit_parser)
    fit_parser.add_argument("--arm", required=True, choices=list(BRANCHES))
    fit_parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), choices=list(SEEDS))
    fit_parser.add_argument("--no-p10-training", action="store_true", help="P10-removed sensitivity (folds_no_p10.json).")

    fuse_parser = sub.add_parser("fuse", help="Apply the registered fusion rule; writes OOF, prints no held-out score.")
    add_common(fuse_parser)
    fuse_parser.add_argument("--no-p10-training", action="store_true")

    baseline_parser = sub.add_parser("baseline-check", help="Stock seed-42 LOSO rerun vs the historical NLF summary (never blocks).")
    add_common(baseline_parser)
    baseline_parser.add_argument("--rerun-summary", type=Path, required=True, help="--summary-output of the stock loso_cross_validation rerun.")
    baseline_parser.add_argument("--stored-summary", type=Path, default=HISTORICAL_NLF_SUMMARY)

    add_common(sub.add_parser("gates", help="The eight blocking gates -> gates.json."))
    add_common(sub.add_parser("report", help="Primary, Holm family, position control, diagnostics -> summary.json."))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = args.run_root / args.run_id
    device = torch.device(REQUIRED_DEVICE)  # the plan runs on the local CPU; there is no device option to get wrong

    if args.command == "init":
        config = run_init(run_dir, args.manifest, args.labels, args.segmentation)
        print(f"hashes match: {config['hashes']}")
        print(f"git commit {config['git_commit']}; uncommitted protocol files: {config['source_dirty'] or 'none'}")
        print(f"Wrote run config and frozen folds to {run_dir}")
        return

    if args.command == "features":
        for arm, audit in run_feature_integrity(run_dir).items():
            print(f"{arm}: {audit['found_ids']}/{audit['expected_ids']} ids, dim {audit['expected_dim']}, passed={audit['passed']}")
        return

    if args.command == "ab-check":
        payload = run_ab_check(run_dir, args.arm, device, args.seeds, args.test_subjects)
        for key, cell in sorted(payload["cells"].items()):
            stored = cell.get("stored_run", {}).get("bit_identical")
            print(f"{key}: identical={cell['identical']} stored_20260918={stored} {cell['seconds_per_fit']:.1f}s/fit")
        return

    if args.command == "fit":
        log = run_fit(run_dir, args.arm, device, args.seeds, args.no_p10_training)
        for seed, entry in log["seeds"].items():
            print(f"{args.arm} seed {seed}: {entry['held_out_rows']} held-out rows, {entry['validation_rows']} validation rows, {entry['seconds_per_fit']:.1f}s/fit")
        return

    if args.command == "fuse":
        result = run_fuse(run_dir, args.no_p10_training)
        print(f"fused {result['cells']} fold x seed cells; negative Platt slopes: {result['negative_slopes']}")
        if "leak_audit" in result:
            print(f"leak audit: trigger fired={result['leak_audit']['fired']}, outcomes releasable={result['leak_audit']['released']} -> {leak_audit_path(run_dir)}")
        print("No held-out score is printed here; run `gates`, then `report`.")
        return

    if args.command == "baseline-check":
        report = historical_baseline_check(run_dir, args.rerun_summary, args.stored_summary)
        differing = [subject for subject, cell in report["per_fold"].items() if not cell["identical"]]
        print(f"historical nlf baseline reproduced: {report['reproduced']} (folds differing: {differing or 'none'}); never blocks")
        return

    if args.command == "gates":
        payload = compute_gates(run_dir)
        for name in GATE_NAMES:
            print(f"{name:<30}{payload['gates'][name]['status']}")
        print(f"passed: {payload['passed']}")
        return

    if args.command == "report":
        summary = run_report(run_dir)
        print(f"Wrote {vbc.summary_path(run_dir)}")
        print(f"gates passed: {summary['gates']['passed']}")
        if "primary" not in summary:
            print(f"outcomes {summary['outcomes']}")
            return
        print(f"verdict row {summary['verdict']['row']}: {summary['verdict']['reading']}")
        primary = summary["primary"]
        print(
            f"primary fused-nlf: mean delta {primary['mean_delta']:+.4f}, {primary['n_positive']}/{primary['n_subjects']} positive, "
            f"CI {primary['bootstrap_ci_95']}, p={primary['sign_flip']['p_value']:.4f}"
        )
        return

if __name__ == "__main__":
    main()
