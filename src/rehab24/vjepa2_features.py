"""Extract REHAB24-6 repetition-level V-JEPA 2 features (arms ``vj64`` and ``vj16``).

Companion to ``src.rehab24.videomae_features``'s ``--sampler bounded`` mode: both
extractors call the same :mod:`src.rehab24.clip_sampling` helpers, so ``vm16``
(VideoMAE, 16 frames) and ``vj16`` (V-JEPA 2, the exact ``vm16`` frame indices)
draw byte-identical ``frame_indices`` by construction, and ``vj64`` (V-JEPA 2,
64 frames) uses the identical sampling machinery with a longer clip.

Geometry deliberately does NOT reuse ``VJEPA2VideoProcessor``: that processor
resizes the shortest edge to 292 then centre-crops 256, which would cut into the
padding ``full_frame_letterbox`` adds (and, on the narrower camera, into the
subject). Instead every decoded frame is letterboxed to square, then resized to
256x256 with bilinear interpolation -- the same interpolation
``VideoMAEImageProcessor`` uses for this project's VideoMAE checkpoint (verified
directly against its pretrained config: ``resample=2``, PIL bilinear), so neither
model's arm gets an interpolation-quality edge over the other.

Run ``scripts/rehab24/extract_vjepa2_features.py`` to drive this from the CLI.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.rehab24.clip_sampling import (
    assert_resume_provenance_matches,
    build_bounded_clips,
    decode_frames_at_indices,
    sha256_of_files,
)
from src.rehab24.dataset import DEFAULT_DATA_ROOT, DEFAULT_PROCESSED_ROOT, load_manifest, resolve_data_path
from src.rehab24.videomae_features import group_rows_by_video, save_feature
from src.video.squat_video_variants import letterbox_to_square
from src.video.vjepa2_backbone import (
    DEFAULT_MODEL_NAME,
    DEFAULT_REVISION,
    RESOLUTION,
    encode_clip,
    load_backbone,
)
from src.video.videomae_backbone import resolve_device

import torch
import transformers

#: This extractor's two registered arms (plan, Models and comparisons table).
#: ``vj64`` is the primary challenger at the checkpoint's native clip length;
#: ``vj16`` is the mandatory temporal-budget control at ``vm16``'s clip length.
ARM_VJ64 = "vj64"
ARM_VJ16 = "vj16"
ARMS = (ARM_VJ64, ARM_VJ16)
ARM_CLIP_LENGTH = {ARM_VJ64: 64, ARM_VJ16: 16}
#: Shared with `vm16`: same stride and clip count as the historical VideoMAE
#: extraction, so `vj16`'s starts land on exactly `vm16`'s starts (plan, Sampling).
FRAME_STRIDE = 2
NUM_CLIPS = 4
#: Both arms write this fixed geometry; see module docstring.
VARIANT = "full_frame_letterbox"
SAMPLER = "bounded"
POOLING = "mean_pool_tokens"
OUTPUT_FIELD = "clip_features"

_MODULE_DIR = Path(__file__).resolve().parent
#: Hashed (fixed order) into `provenance_code_fingerprint`; an edit to any of these
#: three files changes the fingerprint of every bundle written afterwards.
FINGERPRINT_FILES = (
    _MODULE_DIR / "clip_sampling.py",
    _MODULE_DIR.parent / "video" / "vjepa2_backbone.py",
    Path(__file__).resolve(),
)

#: Rep-length strata a pilot sample is drawn from (plan, Kaggle execution step 3).
LENGTH_STRATA = ("short", "long")


def letterbox_and_resize(frame: np.ndarray, resolution: int = RESOLUTION) -> np.ndarray:
    """Letterbox an RGB frame to square, then resize to ``resolution`` x ``resolution``.

    See the module docstring for why this, and not the HF video processor's own
    resize/crop, is used.
    """
    squared = letterbox_to_square(frame)
    return cv2.resize(squared, (resolution, resolution), interpolation=cv2.INTER_LINEAR)


def sample_and_decode(
    cap: cv2.VideoCapture,
    total_frames: int,
    first_frame: int,
    last_frame: int,
    clip_length: int,
    frame_stride: int,
    num_clips: int,
    sample_id: str = "",
    video_path: str = "",
) -> tuple[list[dict[str, object]], dict[int, np.ndarray], int, int]:
    """This repetition's bounded clips, and a decoded+letterboxed+resized frame cache.

    Decodes each needed index once (``decode_frames_at_indices`` takes the union
    across all of the repetition's clips), so a 4x64-frame sample never holds 256
    raw 1080p frames -- only the small, already-256x256 cache does. ``sample_id``/
    ``video_path`` are optional and used only to name the repetition in a
    decode-failure message.
    """
    clips, first_index, last_index = build_bounded_clips(
        first_frame, last_frame, total_frames, clip_length, frame_stride, num_clips
    )
    needed_indices = {int(index) for clip in clips for index in clip["frame_indices"].tolist()}
    decoded = decode_frames_at_indices(
        cap, needed_indices, total_frames, context=f"sample={sample_id!r} video={video_path!r}"
    )
    cache = {index: letterbox_and_resize(frame) for index, frame in decoded.items()}
    return clips, cache, first_index, last_index


def encode_clips(
    model: "torch.nn.Module",
    device: "torch.device",
    clips: Sequence[dict[str, object]],
    cache: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    """Encode every clip's cached frames into the clip-indexed parts of the raw bundle."""
    features: list[np.ndarray] = []
    clip_starts: list[int] = []
    frame_index_rows: list[np.ndarray] = []
    unique_counts: list[int] = []
    padding_fractions: list[float] = []
    for clip in clips:
        indices = clip["frame_indices"]
        frames = [cache[int(index)] for index in indices]
        features.append(encode_clip(model, frames, device))
        clip_starts.append(int(clip["start"]))
        frame_index_rows.append(indices)
        unique_counts.append(int(clip["unique_frame_count"]))
        padding_fractions.append(float(clip["padding_fraction"]))
    return {
        "clip_features": np.stack(features, axis=0),
        "clip_starts": np.asarray(clip_starts, dtype=np.int64),
        "frame_indices": np.stack(frame_index_rows, axis=0),
        "unique_frame_count": np.asarray(unique_counts, dtype=np.int64),
        "padding_fraction": np.asarray(padding_fractions, dtype=np.float32),
    }


def extract_repetition_features(
    model: "torch.nn.Module",
    device: "torch.device",
    cap: cv2.VideoCapture,
    total_frames: int,
    first_frame: int,
    last_frame: int,
    clip_length: int,
    frame_stride: int,
    num_clips: int,
    sample_id: str = "",
    video_path: str = "",
) -> dict[str, np.ndarray]:
    """One repetition's full ``vj64``/``vj16`` bundle.

    ``sample_and_decode`` + ``encode_clips`` combined, for callers that do not
    need the per-stage timing the CLI records separately.
    """
    clips, cache, first_index, last_index = sample_and_decode(
        cap, total_frames, first_frame, last_frame, clip_length, frame_stride, num_clips, sample_id, video_path
    )
    bundle = encode_clips(model, device, clips, cache)
    bundle["first_index"] = np.asarray(first_index, dtype=np.int64)
    bundle["last_index"] = np.asarray(last_index, dtype=np.int64)
    bundle["total_frames"] = np.asarray(total_frames, dtype=np.int64)
    return bundle


def build_provenance(
    arm: str,
    device: "torch.device",
    model_name: str = DEFAULT_MODEL_NAME,
    revision: str = DEFAULT_REVISION,
) -> dict[str, str]:
    """Provenance for one ``vj64``/``vj16`` run -- the raw bundle contract."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}.")
    return {
        "model_name": model_name,
        "revision": revision,
        "arm": arm,
        "clip_length": str(ARM_CLIP_LENGTH[arm]),
        "frame_stride": str(FRAME_STRIDE),
        "num_clips": str(NUM_CLIPS),
        "resolution": str(RESOLUTION),
        "pooling": POOLING,
        "output_field": OUTPUT_FIELD,
        "sampler": SAMPLER,
        "variant": VARIANT,
        "dtype": "float32",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": str(device),
        "code_fingerprint": sha256_of_files(FINGERPRINT_FILES),
    }


def select_pilot_samples(
    rows: Sequence[dict[str, str]], n: int = 24, seed: int = 20260918
) -> list[dict[str, str]]:
    """Deterministically pick ``n`` samples covering both cameras, all exercises,
    and both the shortest-quartile and longest-quartile repetitions.

    ``n`` must divide evenly across ``(exercise, camera, length-stratum)`` cells;
    at the default ``n=24`` with REHAB24-6's 6 exercises x 2 cameras x 2 strata,
    that is exactly one sample per cell. Raises if a cell has no eligible
    candidate rather than silently shrinking the pilot below its declared
    coverage (plan, Kaggle execution step 3).
    """
    if not rows:
        raise ValueError("Cannot select a pilot from an empty row set.")
    exercises = sorted({row["exercise_id"] for row in rows})
    cameras = sorted({row["camera"] for row in rows})
    cells = [(exercise, camera, stratum) for exercise in exercises for camera in cameras for stratum in LENGTH_STRATA]
    if n % len(cells) != 0:
        raise ValueError(
            f"n={n} does not divide evenly across {len(cells)} (exercise, camera, length-stratum) cells "
            f"({len(exercises)} exercises x {len(cameras)} cameras x {len(LENGTH_STRATA)} strata)."
        )
    per_cell = n // len(cells)

    lengths = [int(row["last_frame"]) - int(row["first_frame"]) + 1 for row in rows]
    q1, q3 = np.percentile(lengths, 25), np.percentile(lengths, 75)

    by_cell: dict[tuple[str, str, str], list[dict[str, str]]] = {cell: [] for cell in cells}
    for row in rows:
        length = int(row["last_frame"]) - int(row["first_frame"]) + 1
        stratum = "short" if length <= q1 else "long" if length >= q3 else None
        if stratum is None:
            continue
        cell = (row["exercise_id"], row["camera"], stratum)
        if cell in by_cell:
            by_cell[cell].append(row)

    rng = np.random.default_rng(seed)
    selected: list[dict[str, str]] = []
    empty_cells: list[tuple[str, str, str]] = []
    for cell in cells:
        # Sorted by sample_id first so `rng.choice`'s draw is reproducible
        # regardless of the input rows' original order.
        candidates = sorted(by_cell[cell], key=lambda row: row["sample_id"])
        if len(candidates) < per_cell:
            empty_cells.append(cell)
            continue
        chosen = rng.choice(len(candidates), size=per_cell, replace=False)
        selected.extend(candidates[int(index)] for index in sorted(chosen.tolist()))

    if empty_cells:
        raise ValueError(f"No eligible pilot candidates for cells: {empty_cells}.")
    return selected


def check_vm16_against_history(
    run_dir: Path, historical_dir: Path, sample_ids: Sequence[str]
) -> dict[str, object]:
    """Reproducibility gate: compare bounded-sampler ``vm16`` features against the
    historical (unbounded-at-video-end) extraction, restricted to repetitions
    where clamping never fired.

    With ``frame_stride >= 1`` the raw sampled indices ``start + stride*k`` are
    strictly increasing, hence pairwise distinct; a repeated index in the stored
    ``frame_indices`` (``padding_fraction > 0``) can therefore only come from
    clamping to the repetition/video bound. So ``padding_fraction == 0`` on every
    clip of a repetition means clamping never fired, which means this sampler drew
    the exact indices the historical (unbounded) sampler would have -- the only
    repetitions where "old vs new" is a fair reproduction check rather than a
    comparison between two different samplings.

    Historical bundles (``data/REHAB24-6/processed/videomae_raw_full_frame_letterbox``)
    are read-only inputs; nothing here writes to them. Reports max relative L2 and
    min cosine with no hard pass/fail threshold -- the plan registers no tolerance,
    and cross-device/cross-library noise of ~1e-3 relative L2 is already expected
    and documented (``scripts/rehab24/README.md``).
    """
    compared: list[str] = []
    skipped_padded: list[str] = []
    skipped_missing_history: list[str] = []
    relative_l2s: list[float] = []
    cosines: list[float] = []

    for sample_id in sample_ids:
        new_path = next(run_dir.glob(f"raw/vm16/*/{sample_id}.npz"), None)
        if new_path is None:
            raise FileNotFoundError(f"No vm16 bundle for {sample_id!r} under {run_dir}.")
        with np.load(new_path, allow_pickle=False) as data:
            if not np.allclose(data["padding_fraction"], 0.0):
                skipped_padded.append(sample_id)
                continue
            new_features = np.asarray(data["clip_features"], dtype=np.float64)

        split = new_path.parent.name
        historical_path = historical_dir / split / f"{sample_id}.npz"
        if not historical_path.exists():
            skipped_missing_history.append(sample_id)
            continue
        with np.load(historical_path, allow_pickle=False) as data:
            old_features = np.asarray(data["clip_features_mean_pool_fc_norm"], dtype=np.float64)

        new_flat = new_features.reshape(-1)
        old_flat = old_features.reshape(-1)
        old_norm = np.linalg.norm(old_flat)
        new_norm = np.linalg.norm(new_flat)
        relative_l2s.append(float(np.linalg.norm(new_flat - old_flat) / max(old_norm, 1e-12)))
        cosines.append(float(np.dot(new_flat, old_flat) / max(new_norm * old_norm, 1e-12)))
        compared.append(sample_id)

    return {
        "compared": compared,
        "skipped_padded": skipped_padded,
        "skipped_missing_history": skipped_missing_history,
        "max_relative_l2": max(relative_l2s) if relative_l2s else None,
        "min_cosine": min(cosines) if cosines else None,
    }


def _read_sample_ids_file(path: Path, known_ids: set[str]) -> list[str]:
    """Explicit pilot id list. An unknown id raises rather than silently narrowing
    the requested subset (plan: a loader must never choose a different subset per
    arm)."""
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    unknown = [sample_id for sample_id in ids if sample_id not in known_ids]
    if unknown:
        raise SystemExit(f"--sample-ids-file names ids not in the manifest: {unknown}")
    return ids


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract REHAB24-6 repetition-level V-JEPA 2 features (vj64 / vj16).")
    parser.add_argument("--run-dir", type=Path, required=True, help="Bundles land at <run-dir>/raw/<arm>/<split>/.")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto.")
    parser.add_argument("--num-chunks", type=int, default=1, help="Split the work into N round-robin chunks.")
    parser.add_argument("--chunk-index", type=int, default=0, help="Which chunk to process (0-based).")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N (post-filter) rows (smoke test).")
    parser.add_argument(
        "--sample-ids-file",
        type=Path,
        default=None,
        help="Explicit newline-separated sample_id list (e.g. the pilot). Unknown ids raise.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timing-json", type=Path, default=None, help="Per-sample timing/memory log, plus medians/p95.")
    parser.add_argument(
        "--save-examples",
        type=Path,
        default=None,
        help="Dump the first preprocessed clip's first frame of each processed sample as a PNG here (geometry check).",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    clip_length = ARM_CLIP_LENGTH[args.arm]
    arm_dir = args.run_dir / "raw" / args.arm

    provenance = build_provenance(args.arm, device)
    assert_resume_provenance_matches(arm_dir, provenance)

    rows = load_manifest(args.manifest)
    if args.sample_ids_file is not None:
        known_ids = {row["sample_id"] for row in rows}
        wanted = _read_sample_ids_file(args.sample_ids_file, known_ids)
        by_id = {row["sample_id"]: row for row in rows}
        rows = [by_id[sample_id] for sample_id in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.num_chunks > 1:
        rows = [row for index, row in enumerate(rows) if index % args.num_chunks == args.chunk_index]
        print(f"Chunk {args.chunk_index + 1}/{args.num_chunks}: {len(rows)} manifest rows")

    print(f"Loading V-JEPA 2 model `{provenance['model_name']}`@`{provenance['revision']}` on {device}...")
    model = load_backbone(device=device)

    if args.save_examples is not None:
        args.save_examples.mkdir(parents=True, exist_ok=True)

    timing_rows: list[dict[str, object]] = []
    warmed_up = False
    written = 0
    skipped = 0

    for video_index, (video_path, video_rows) in enumerate(group_rows_by_video(rows), start=1):
        pending = [
            row
            for row in video_rows
            if args.overwrite or not (arm_dir / row["split"] / f"{row['sample_id']}.npz").exists()
        ]
        skipped += len(video_rows) - len(pending)
        if not pending:
            continue

        cap = cv2.VideoCapture(str(resolve_data_path(args.data_root, video_path)))
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for row in pending:
                if not warmed_up:
                    # One throwaway forward pass to pay CUDA kernel warm-up cost
                    # outside of every measured timing (plan, Kaggle execution step 4).
                    dummy_frames = [np.zeros((RESOLUTION, RESOLUTION, 3), dtype=np.uint8)] * clip_length
                    encode_clip(model, dummy_frames, device)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats(device)
                    warmed_up = True

                end_to_end_start = time.perf_counter()
                decode_start = time.perf_counter()
                clips, cache, first_index, last_index = sample_and_decode(
                    cap,
                    total_frames,
                    int(row["first_frame"]),
                    int(row["last_frame"]),
                    clip_length,
                    FRAME_STRIDE,
                    NUM_CLIPS,
                    sample_id=row["sample_id"],
                    video_path=video_path,
                )
                decode_seconds = time.perf_counter() - decode_start

                if device.type == "cuda":
                    torch.cuda.synchronize()
                encode_start = time.perf_counter()
                bundle = encode_clips(model, device, clips, cache)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                encode_seconds = time.perf_counter() - encode_start

                bundle["first_index"] = np.asarray(first_index, dtype=np.int64)
                bundle["last_index"] = np.asarray(last_index, dtype=np.int64)
                bundle["total_frames"] = np.asarray(total_frames, dtype=np.int64)
                save_feature(arm_dir / row["split"] / f"{row['sample_id']}.npz", row, bundle, provenance)
                end_to_end_seconds = time.perf_counter() - end_to_end_start
                written += 1

                if args.save_examples is not None:
                    first_clip_first_index = int(clips[0]["frame_indices"][0])
                    example = cache[first_clip_first_index]
                    cv2.imwrite(str(args.save_examples / f"{row['sample_id']}.png"), cv2.cvtColor(example, cv2.COLOR_RGB2BGR))

                if args.timing_json is not None:
                    peak_mib = (
                        float(torch.cuda.max_memory_allocated(device)) / (1024 * 1024) if device.type == "cuda" else None
                    )
                    timing_rows.append(
                        {
                            "sample_id": row["sample_id"],
                            "decode_seconds": decode_seconds,
                            "encoder_seconds": encode_seconds,
                            "end_to_end_seconds": end_to_end_seconds,
                            "peak_allocated_mib": peak_mib,
                            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        }
                    )
        finally:
            cap.release()
        print(f"[{video_index}] {video_path}: wrote {len(pending)} repetitions (total {written})")

    if args.timing_json is not None:
        decode_seconds_all = [row["decode_seconds"] for row in timing_rows]
        encoder_seconds_all = [row["encoder_seconds"] for row in timing_rows]
        end_to_end_all = [row["end_to_end_seconds"] for row in timing_rows]
        summary = {
            "samples": timing_rows,
            "decode_seconds_median": _percentile(decode_seconds_all, 50),
            "decode_seconds_p95": _percentile(decode_seconds_all, 95),
            "encoder_seconds_median": _percentile(encoder_seconds_all, 50),
            "encoder_seconds_p95": _percentile(encoder_seconds_all, 95),
            "end_to_end_seconds_median": _percentile(end_to_end_all, 50),
            "end_to_end_seconds_p95": _percentile(end_to_end_all, 95),
        }
        args.timing_json.parent.mkdir(parents=True, exist_ok=True)
        import json

        args.timing_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {written} `{args.arm}` V-JEPA 2 feature bundles ({skipped} already present) under {arm_dir}")


if __name__ == "__main__":
    main()
