"""Extract REHAB24-6 repetition-level VideoMAE features.

Emits BOTH token-pooling modes (see ``src.video.videomae_pooling``) from a single
forward pass, and stores the per-clip stacks rather than a pre-aggregated vector.

``--variant`` selects which pixels the model is shown, for the framing arms of
``notes/rehab24_videomae_framing_validation_plan.md``. The transform is applied in
memory between decode and the processor -- deliberately NOT by writing variant mp4s
and re-reading them, which would make only the variant arms pay an extra lossy
generation and confound any drop with codec degradation (plan §4.2).

Both properties are deliberate. Computing ``legacy_first_token`` and
``mean_pool_fc_norm`` from the same ``last_hidden_state`` makes the two arms share
frames, weights, clip sampling and transformers version *by construction*, so a
paired LOSO delta measures the pooling fix and nothing else. Storing the clip stack
keeps clip aggregation (max vs mean) an offline decision, so the two pooling axes
never move together in one measured delta. The stacks are cheap: 2144 samples x 2
modes x 4 clips x 768 float32 is ~52 MB.

Run ``videomae_materialize`` afterwards to derive the LOSO-ready feature dirs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.rehab24.dataset import DEFAULT_DATA_ROOT, DEFAULT_PROCESSED_ROOT, load_manifest, resolve_data_path
from src.rehab24.videomae_boxes import BOX_SOURCE, box_for_video, load_index
from src.video.squat_video_variants import apply_variant
from src.video.variant_geometry import BOX_VARIANTS, DEFAULT_MARGIN, Box
from src.video.videomae_backbone import (  # noqa: F401 - assert_fc_norm_pretrained is re-exported
    assert_fc_norm_pretrained,
    encode_clip,
    load_backbone,
    resolve_device,
)
from src.video.videomae_pooling import LEGACY_FIRST_TOKEN, MEAN_POOL_FC_NORM, build_provenance

import torch
import transformers
from transformers import VideoMAEImageProcessor

#: The framing arms this extractor can produce. ``full_frame`` is the untouched source
#: -- it is a choice here rather than a transform, which is why it is absent from
#: ``variant_geometry.VARIANTS``. ``person_crop_centercrop`` and ``reencoded`` are
#: deliberately not offered: the plan's REHAB24-6 design does not include them.
FRAMING_VARIANTS = ("full_frame", "full_frame_letterbox", "person_crop", "background_only")


def transform_frames(frames: list[np.ndarray], variant: str, box: Box | None) -> list[np.ndarray]:
    """Apply one framing arm's pixel transform to already-decoded RGB frames.

    Two properties are load-bearing:

    *Fail closed.* ``apply_variant`` treats a null box as "leave the video untouched",
    which for a box arm would write full-frame features into a control arm's directory
    under that arm's name. That degradation is acceptable in the Fitness-AQA builder,
    where some videos genuinely have no detected person; here every box comes from
    mocap and a missing one is a bug, so it raises.

    *Colour order does not matter.* Frames arrive RGB (``read_clip_frames`` converts),
    while the Fitness-AQA builder feeds the same helpers BGR. The grey fill 114 is
    channel-symmetric and the surroundings fill is per-channel, so both callers get the
    same geometry; fixing the application point here keeps the later box arms consistent
    with this one.
    """
    if variant == "full_frame":
        return frames
    if variant in BOX_VARIANTS and box is None:
        raise RuntimeError(f"Variant {variant!r} requires a person box, but none was resolved.")
    return apply_variant(frames, variant, box)


def sample_clip_starts(first_frame: int, last_frame: int, clip_length: int, frame_stride: int, num_clips: int) -> list[int]:
    start = max(first_frame - 1, 0)
    stop = max(last_frame, start + 1)
    effective_length = 1 + frame_stride * (clip_length - 1)
    max_start = max(stop - effective_length, start)
    if num_clips <= 1:
        return [(start + max_start) // 2]
    return np.linspace(start, max_start, num=num_clips, dtype=int).tolist()


def read_clip_frames(
    cap: cv2.VideoCapture,
    start_frame: int,
    clip_length: int,
    frame_stride: int,
    total_frames: int,
) -> list[np.ndarray]:
    # Seek once, then decode forward. A per-frame cap.set(POS_FRAMES) forces the
    # decoder back to the nearest keyframe on every call, which dominates runtime
    # on long H.264 clips (~64 keyframe re-seeks per rep). Reading sequentially and
    # grab()-ing the strided frames we skip keeps the decode linear.
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(start_frame, max(total_frames - 1, 0)))
    frames: list[np.ndarray] = []
    last_frame: np.ndarray | None = None
    for offset in range(clip_length):
        if offset > 0:
            for _ in range(frame_stride - 1):
                cap.grab()
        ok, frame = cap.read()
        if not ok:
            if last_frame is None:
                continue
            frame = last_frame.copy()
        last_frame = frame
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    while frames and len(frames) < clip_length:
        frames.append(frames[-1].copy())
    return frames


def extract_repetition_features(
    backbone: torch.nn.Module,
    processor: VideoMAEImageProcessor,
    cap: cv2.VideoCapture,
    total_frames: int,
    first_frame: int,
    last_frame: int,
    clip_length: int,
    frame_stride: int,
    num_clips: int,
    device: torch.device,
    fc_norm_weight: np.ndarray,
    fc_norm_bias: np.ndarray,
    fc_norm_eps: float,
    variant: str = "full_frame",
    box: Box | None = None,
) -> dict[str, np.ndarray]:
    """Both token-pooling modes for one repetition, from one forward pass per clip.

    ``variant`` changes only the pixels. Clip starts are computed from the repetition's
    frame range before any transform, so every arm samples the identical frames and the
    LOSO deltas stay paired frame-for-frame (plan §4.3).
    """
    clip_starts = sample_clip_starts(first_frame, last_frame, clip_length, frame_stride, num_clips)
    legacy_clips: list[np.ndarray] = []
    corrected_clips: list[np.ndarray] = []
    used_starts: list[int] = []

    for start_frame in clip_starts:
        frames = read_clip_frames(cap, start_frame, clip_length, frame_stride, total_frames)
        if not frames:
            continue
        frames = transform_frames(frames, variant, box)
        legacy, corrected = encode_clip(
            backbone=backbone,
            processor=processor,
            frames=frames,
            device=device,
            fc_norm_weight=fc_norm_weight,
            fc_norm_bias=fc_norm_bias,
            fc_norm_eps=fc_norm_eps,
        )
        legacy_clips.append(legacy)
        corrected_clips.append(corrected)
        used_starts.append(int(start_frame))

    if not legacy_clips:
        raise RuntimeError(f"No VideoMAE features could be extracted for frames {first_frame}-{last_frame}")

    return {
        f"clip_features_{LEGACY_FIRST_TOKEN}": np.stack(legacy_clips, axis=0),
        f"clip_features_{MEAN_POOL_FC_NORM}": np.stack(corrected_clips, axis=0),
        "clip_starts": np.asarray(used_starts, dtype=np.int32),
        "first_frame": np.asarray(first_frame, dtype=np.int32),
        "last_frame": np.asarray(last_frame, dtype=np.int32),
        "total_frames": np.asarray(total_frames, dtype=np.int32),
    }


def save_feature(path: Path, row: dict[str, str], bundle: dict[str, np.ndarray], provenance: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_id=np.asarray(row["sample_id"]),
        video_id=np.asarray(row["video_id"]),
        exercise_id=np.asarray(row["exercise_id"]),
        person_id=np.asarray(row["person_id"]),
        camera=np.asarray(row["camera"]),
        correctness=np.asarray(int(row["correctness"]), dtype=np.int64),
        **bundle,
        **{f"provenance_{key}": np.asarray(value) for key, value in provenance.items()},
    )


def group_rows_by_video(rows: Sequence[dict[str, str]]) -> list[tuple[str, list[dict[str, str]]]]:
    """Group manifest rows by source video, ordered by start frame within each video.

    REHAB24-6 has ~16 repetitions per video, so opening one capture per row would
    re-open and re-parse each file 16 times. Extraction here is decode-bound, not
    GPU-bound, so this grouping is the cheapest available speedup.
    """
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["video_path"]].append(row)
    return [
        (video_path, sorted(video_rows, key=lambda r: int(r["first_frame"])))
        for video_path, video_rows in sorted(grouped.items())
    ]


def assert_output_dir_matches_variant(output_dir: Path, variant: str) -> None:
    """Refuse to add bundles of one variant to a directory holding another.

    Plan §5.1: different variants must not share a raw dir. Nothing downstream would
    notice -- the audit's ``single_provenance`` check runs per directory, so a mixed
    dir fails only if the mixture reaches it, and the resume path (skip whatever
    exists) means a re-run with the wrong ``--variant`` writes the remainder in the
    wrong framing and stops there. Checked against what is already on disk rather than
    against the directory's name, because the name is a convention and the stamp is
    evidence.
    """
    existing = next(output_dir.rglob("*.npz"), None) if output_dir.exists() else None
    if existing is None:
        return
    with np.load(existing, allow_pickle=False) as data:
        # Bundles predating --variant carry no stamp; they are all full_frame.
        found = str(data["provenance_variant"]) if "provenance_variant" in data.files else "full_frame"
    if found != variant:
        raise SystemExit(
            f"{output_dir} already holds `{found}` features ({existing.name}), but --variant is `{variant}`. "
            "Write each variant to its own --output-dir."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract REHAB24-6 repetition-level VideoMAE features (both pooling modes).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT / "videomae_raw",
        help="Raw per-clip bundles. Deliberately NOT the legacy `videomae_features` dir, "
        "so a stale cache can never be mistaken for a corrected re-extraction.",
    )
    parser.add_argument(
        "--variant",
        choices=FRAMING_VARIANTS,
        default="full_frame",
        help="Which pixels to show the model. Each variant needs its own --output-dir.",
    )
    parser.add_argument(
        "--box-index",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT / "videomae_boxes.json",
        help="Fixed per-video mocap boxes, required by the box variants. Build with build_videomae_boxes.py.",
    )
    parser.add_argument("--model-name", type=str, default="MCG-NJU/videomae-base-finetuned-kinetics")
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--num-clips", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N manifest rows (smoke test).")
    parser.add_argument("--num-chunks", type=int, default=1, help="Split the work into N round-robin chunks.")
    parser.add_argument("--chunk-index", type=int, default=0, help="Which chunk to process (0-based).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    assert_output_dir_matches_variant(args.output_dir, args.variant)

    box_index = None
    if args.variant in BOX_VARIANTS:
        if not args.box_index.exists():
            raise SystemExit(
                f"Variant `{args.variant}` needs the fixed box index at {args.box_index}, which does not exist. "
                "Run scripts/rehab24/build_videomae_boxes.py first."
            )
        box_index = load_index(args.box_index)

    print(f"Loading VideoMAE model `{args.model_name}` on {device}...")
    processor = VideoMAEImageProcessor.from_pretrained(args.model_name)
    backbone, fc_weight, fc_bias, fc_eps = load_backbone(args.model_name, device)
    print(f"fc_norm loaded from checkpoint (weight mean={fc_weight.mean():.4f}, bias mean={fc_bias.mean():.4f})")

    provenance = build_provenance(
        model_name=args.model_name,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        num_clips=args.num_clips,
        transformers_version=transformers.__version__,
        variant=args.variant,
    )
    if args.variant in BOX_VARIANTS:
        provenance["box_source"] = BOX_SOURCE
        provenance["box_margin"] = str(box_index.get("margin", DEFAULT_MARGIN))
        provenance["fill_strategy"] = "horizontal_interpolation" if args.variant == "background_only" else "letterbox_114"
    elif args.variant == "full_frame_letterbox":
        provenance["fill_strategy"] = "letterbox_114"

    rows = load_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.num_chunks > 1:
        rows = [row for index, row in enumerate(rows) if index % args.num_chunks == args.chunk_index]
        print(f"Chunk {args.chunk_index + 1}/{args.num_chunks}: {len(rows)} manifest rows")

    written = 0
    skipped = 0
    for video_index, (video_path, video_rows) in enumerate(group_rows_by_video(rows), start=1):
        pending = [
            row
            for row in video_rows
            if args.overwrite or not (args.output_dir / row["split"] / f"{row['sample_id']}.npz").exists()
        ]
        skipped += len(video_rows) - len(pending)
        if not pending:
            continue

        # Resolved before the capture opens, so a missing box aborts the run rather
        # than leaving a partially-written arm behind.
        box = box_for_video(box_index, video_path) if box_index is not None else None

        cap = cv2.VideoCapture(str(resolve_data_path(args.data_root, video_path)))
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for row in pending:
                bundle = extract_repetition_features(
                    backbone=backbone,
                    processor=processor,
                    cap=cap,
                    total_frames=total_frames,
                    first_frame=int(row["first_frame"]),
                    last_frame=int(row["last_frame"]),
                    clip_length=args.clip_length,
                    frame_stride=args.frame_stride,
                    num_clips=args.num_clips,
                    device=device,
                    fc_norm_weight=fc_weight,
                    fc_norm_bias=fc_bias,
                    fc_norm_eps=fc_eps,
                    variant=args.variant,
                    box=box,
                )
                save_feature(args.output_dir / row["split"] / f"{row['sample_id']}.npz", row, bundle, provenance)
                written += 1
        finally:
            cap.release()
        print(f"[{video_index}] {video_path}: wrote {len(pending)} repetitions (total {written})")

    print(
        f"Wrote {written} `{args.variant}` VideoMAE feature bundles "
        f"({skipped} already present) under {args.output_dir}"
    )


if __name__ == "__main__":
    main()
