"""The appearance-only negative control for the REHAB24-6 identity plan (§3.2, §5).

``canonical_frame_repeat`` shows VideoMAE one still frame per source camera video,
repeated to fill a 16-frame clip, and gives every repetition of that video the *same*
embedding. Identity, body, clothing, background, camera, lighting and exercise all
survive intact; repetition-specific motion and pose do not. Whatever LOSO accuracy
this arm reaches is what stable appearance plus each recording's correctness base rate
can buy on their own.

Three properties are load-bearing and each is enforced, not merely intended.

*Label-blind by construction.* The canonical index is ``total_frames // 2`` of the
SOURCE video. It never touches the manifest's repetition boundaries or labels, so the
frame cannot have been chosen to favour or disfavour any class.

*Bit-identical within a video.* If the shared embedding were not literally shared, the
arm would leak exactly the repetition-level variation it exists to remove.
``verify_bit_identical`` hashes the stored arrays and requires one unique digest per
source video, so the property is checked on what reached disk.

*Fail closed.* A decode failure raises. Falling back to a repetition-specific clip or
to the untouched full frame would put full-strength features into a control arm's
directory under the control's name -- the failure mode that already cost this study
half an arm once (see ``squat_video_variants``).

The framing is hard-coded to ``full_frame_letterbox``, matching the primary arm, and is
deliberately NOT exposed as a flag: a negative control with a framing switch invites a
second run and a choice of the better number.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.rehab24.dataset import DEFAULT_DATA_ROOT, DEFAULT_PROCESSED_ROOT, load_manifest, resolve_data_path
from src.rehab24.loso_cross_validation import FoldConfig, MIN_VAL_SUBJECT_SAMPLES, subjects_to_samples
from src.rehab24.videomae_features import save_feature
from src.rehab24.videomae_materialize import CARRIED_KEYS, materialize_all
from src.rehab24.videomae_stage_a import load_metadata, run_arm
from src.video.squat_video_variants import apply_variant
from src.video.videomae_pooling import LEGACY_FIRST_TOKEN, MEAN_POOL_FC_NORM, build_provenance

VARIANT = "canonical_frame_repeat"
BASE_FRAMING = "full_frame_letterbox"
CLIP_LENGTH = 16

RAW_DIR = DEFAULT_PROCESSED_ROOT / f"videomae_raw_{VARIANT}"
FEATURE_PARENT = DEFAULT_PROCESSED_ROOT / "videomae_framing" / VARIANT
FEATURE_DIR = FEATURE_PARENT / "videomae_mean_pool_fc_norm_mean"
PRIMARY_FEATURE_DIR = DEFAULT_PROCESSED_ROOT / "videomae_framing" / BASE_FRAMING / "videomae_mean_pool_fc_norm_mean"

#: Plan §6.3, from the frozen manifest: 128 source camera videos for P1-P9, 130 with P10.
EXPECTED_SOURCE_VIDEOS = {"primary": 128, "all": 130}
SENSITIVITY_SUBJECT = "10"


def canonical_frame_index(total_frames: int) -> int:
    """Temporal midpoint of the source video -- the whole label-blind rule (§6.3)."""
    if total_frames <= 0:
        raise RuntimeError(f"Source video reports {total_frames} frames; cannot pick a canonical frame.")
    return total_frames // 2


def read_canonical_clip(video_path: Path) -> tuple[list[np.ndarray], int]:
    """One letterboxed still, repeated ``CLIP_LENGTH`` times, plus the frame index used.

    Raises on any failure. The index is returned rather than recomputed by the caller
    so the number stamped into the bundle is provably the number that was decoded.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {video_path}.")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        index = canonical_frame_index(total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            # No seek-backwards retry: a fallback index would make the rule depend on
            # decoder behaviour rather than on the video, and §6.3 says fail closed.
            raise RuntimeError(f"Decode of canonical frame {index} failed for {video_path}.")
    finally:
        cap.release()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames = [rgb.copy() for _ in range(CLIP_LENGTH)]
    return apply_variant(frames, BASE_FRAMING, None), index


def group_rows_by_video(rows: Sequence[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["video_path"]].append(row)
    return dict(sorted(grouped.items()))


def assert_source_video_count(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    """Gate §6.3, checked BEFORE any extraction so a mis-grouped run costs nothing."""
    counts = {
        "all": len({row["video_path"] for row in rows}),
        "primary": len({row["video_path"] for row in rows if row["person_id"] != SENSITIVITY_SUBJECT}),
    }
    if counts != EXPECTED_SOURCE_VIDEOS:
        raise SystemExit(f"Source-video gate failed: {counts}, expected {EXPECTED_SOURCE_VIDEOS}.")
    return counts


def verify_bit_identical(raw_dir: Path, rows: Sequence[dict[str, str]]) -> dict:
    """Gate §6.3: every repetition of one source video carries the identical array."""
    by_video = group_rows_by_video(rows)
    digests: dict[str, set[str]] = {}
    for video_path, video_rows in by_video.items():
        seen: set[str] = set()
        for row in video_rows:
            path = raw_dir / row["split"] / f"{row['sample_id']}.npz"
            with np.load(path, allow_pickle=False) as data:
                seen.add(hashlib.sha256(data[f"clip_features_{MEAN_POOL_FC_NORM}"].tobytes()).hexdigest())
        digests[video_path] = seen

    offenders = sorted(video for video, seen in digests.items() if len(seen) != 1)
    return {
        "source_videos": len(digests),
        "videos_with_non_identical_embeddings": offenders,
        "unique_embeddings": len({next(iter(seen)) for seen in digests.values() if len(seen) == 1}),
        "passed": not offenders,
    }


def run_extract(
    manifest_path: Path,
    data_root: Path | None,
    model_name: str,
    device_arg: str | None,
    overwrite: bool,
) -> dict:
    import transformers
    from transformers import VideoMAEImageProcessor

    from src.video.videomae_backbone import encode_clip, load_backbone, resolve_device

    data_root = data_root or DEFAULT_DATA_ROOT
    rows = load_manifest(manifest_path)
    counts = assert_source_video_count(rows)
    device = resolve_device(device_arg)

    print(f"Loading VideoMAE `{model_name}` on {device} for the appearance-only control...")
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    backbone, fc_weight, fc_bias, fc_eps = load_backbone(model_name, device)

    provenance = build_provenance(
        model_name=model_name,
        clip_length=CLIP_LENGTH,
        frame_stride=0,  # a still frame has no stride; 0 records that rather than implying 1
        num_clips=1,
        transformers_version=transformers.__version__,
        variant=VARIANT,
    )
    provenance.update(
        {
            "base_framing": BASE_FRAMING,
            "fill_strategy": "letterbox_114",
            "canonical_frame_rule": "source_total_frames // 2",
            "embedding_scope": "one per source camera video, shared by all its repetitions",
        }
    )

    written = 0
    skipped = 0
    by_video = group_rows_by_video(rows)
    for position, (video_path, video_rows) in enumerate(by_video.items(), start=1):
        pending = [
            row
            for row in video_rows
            if overwrite or not (RAW_DIR / row["split"] / f"{row['sample_id']}.npz").exists()
        ]
        skipped += len(video_rows) - len(pending)
        if not pending:
            continue

        frames, frame_index = read_canonical_clip(resolve_data_path(data_root, video_path))
        legacy, corrected = encode_clip(
            backbone=backbone,
            processor=processor,
            frames=frames,
            device=device,
            fc_norm_weight=fc_weight,
            fc_norm_bias=fc_bias,
            fc_norm_eps=fc_eps,
        )
        bundle = {
            f"clip_features_{LEGACY_FIRST_TOKEN}": legacy[None, :],
            f"clip_features_{MEAN_POOL_FC_NORM}": corrected[None, :],
            "clip_starts": np.asarray([frame_index], dtype=np.int32),
        }
        for row in pending:
            save_feature(
                RAW_DIR / row["split"] / f"{row['sample_id']}.npz",
                row,
                {
                    **bundle,
                    "first_frame": np.asarray(int(row["first_frame"]), dtype=np.int32),
                    "last_frame": np.asarray(int(row["last_frame"]), dtype=np.int32),
                },
                provenance,
            )
            written += 1
        print(f"[{position}/{len(by_video)}] {video_path}: frame {frame_index}, {len(pending)} repetitions (total {written})")

    identity = verify_bit_identical(RAW_DIR, rows)
    if not identity["passed"]:
        raise SystemExit(f"Bit-identity gate failed for {identity['videos_with_non_identical_embeddings'][:5]}")
    print(
        f"\nWrote {written} bundles ({skipped} already present) under {RAW_DIR}; "
        f"{identity['unique_embeddings']} unique embeddings across {identity['source_videos']} source videos."
    )

    materialize_all(
        raw_dir=RAW_DIR,
        output_parent=FEATURE_PARENT,
        token_poolings=(MEAN_POOL_FC_NORM,),
        aggregations=("mean",),
        carried_keys=CARRIED_KEYS,
    )
    report = {"variant": VARIANT, "source_videos": counts, "written": written, "bit_identity": identity}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with (RAW_DIR / "extraction_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report


def run_evaluate(
    manifest_path: Path,
    labels_path: Path,
    output_dir: Path,
    seeds: Sequence[int],
    device_arg: str | None,
) -> dict:
    """LOSO on the appearance-only arm, paired subject-by-subject against the primary.

    The comparison machinery is the framing report's, reused rather than rewritten, so
    the subject-then-seed averaging order and the exact Wilcoxon are the same ones the
    framing result already went through.
    """
    import torch

    from src.rehab24.videomae_framing_report import paired_comparison, seed_averaged_accuracy
    from src.rehab24.videomae_identity_control import folds_path

    if not FEATURE_DIR.exists():
        raise SystemExit(f"Missing {FEATURE_DIR}. Run `extract-appearance` first.")

    device = torch.device("cuda" if (device_arg != "cpu" and device_arg is not None and torch.cuda.is_available()) else "cpu")
    config = FoldConfig()
    labels = {key: int(value) for key, value in json.load(labels_path.open()).items()}
    metadata = load_metadata(manifest_path)
    subject_samples = subjects_to_samples(manifest_path)
    sample_counts = {person: len(ids) for person, ids in subject_samples.items()}
    ordered_subjects = sorted(subject_samples, key=int)

    appearance: dict[int, list[dict]] = {}
    for seed in seeds:
        print(f"\n--- {VARIANT} | seed {seed} ---")
        folds = run_arm(
            FEATURE_DIR, labels, subject_samples, ordered_subjects, sample_counts, metadata, config, device, seed
        )
        appearance[seed] = folds
        big = [f for f in folds if f["n_test"] >= MIN_VAL_SUBJECT_SAMPLES]
        print(f"  bal_acc (9 folds, no P10): {np.mean([f['balanced_accuracy'] for f in big]):.4f}")

    # The baseline is pinned to the identity control's own output dir, NOT to
    # ``output_dir``. Reading it from a caller-supplied directory would let
    # `--output-dir <some other arm>` pair the appearance arm against that arm's
    # folds while still labelling the result `paired_vs_full_frame_letterbox`.
    baseline_dir = DEFAULT_PROCESSED_ROOT / "videomae_identity_control"
    baseline: dict[int, list[dict]] = {}
    for seed in seeds:
        path = folds_path(baseline_dir, seed)
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run `predict` first so the pairing uses the audited primary folds.")
        payload = json.load(path.open(encoding="utf-8"))
        baseline[seed] = payload["folds"]

    summary: dict = {"seeds": list(seeds), "arms": {}, "paired_vs_full_frame_letterbox": {}}
    for metric in ("balanced_accuracy", "macro_f1", "recall", "specificity"):
        candidate = seed_averaged_accuracy(appearance, key=metric)
        reference = seed_averaged_accuracy(baseline, key=metric)
        summary["arms"][metric] = {VARIANT: candidate, BASE_FRAMING: reference}
        summary["paired_vs_full_frame_letterbox"][metric] = paired_comparison(candidate, reference)

    summary["p10_inclusive_sensitivity"] = paired_comparison(
        seed_averaged_accuracy(appearance, drop_p10=False),
        seed_averaged_accuracy(baseline, drop_p10=False),
    )
    # Plan §7 row 6: an appearance-only arm at or above the decision threshold means a
    # shortcut may coexist with any repetition-level signal, and fusion waits.
    summary["shortcut_gate_triggered"] = bool(
        np.mean(list(summary["arms"]["balanced_accuracy"][VARIANT].values())) >= 0.55
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "appearance_only_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    comparison = summary["paired_vs_full_frame_letterbox"]["balanced_accuracy"]
    print(f"\n=== appearance-only ({VARIANT}) vs {BASE_FRAMING}, 9 subjects ===")
    print(
        f"  {BASE_FRAMING}: {comparison['baseline_mean']:.4f}   {VARIANT}: {comparison['candidate_mean']:.4f}   "
        f"delta {comparison['delta']['mean']:+.4f} +/- {comparison['delta']['std']:.4f} "
        f"({comparison['n_positive']}/{comparison['n_subjects']} subjects positive)"
    )
    if comparison.get("wilcoxon"):
        print(f"  exact Wilcoxon p = {comparison['wilcoxon']['p_value']:.5f}  ({comparison['significance']})")
    print(f"  §7 shortcut gate (appearance-only BA >= 0.55): {'TRIGGERED' if summary['shortcut_gate_triggered'] else 'not triggered'}")
    print(f"\nSaved to {path}")
    return summary
