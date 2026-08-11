"""Extract Fitness-AQA squat video-level VideoMAE features (Stage B).

This is the second of the two extractors named in
``notes/videomae_dataset_validation_plan.md``. Until now it carried the same defect
Stage A fixed on the REHAB24-6 side: it read ``last_hidden_state[:, 0, :]`` and called
it a CLS token (VideoMAE has none -- the encoder emits 8 tubelets x 196 patches with
no prepended summary token, so index 0 was the top-left patch of the first tubelet),
then aggregated clips with ``max``. The features behind the archived Fitness-AQA
VideoMAE-only baseline (combined balanced accuracy ~0.555) are therefore
``legacy_first_token`` features, not clip representations.

Like the REHAB24-6 extractor it emits BOTH token-pooling modes from a single forward
pass and stores the per-clip stacks rather than a pre-aggregated vector, so that
(a) the legacy and corrected arms share frames, weights, clip sampling and library
version by construction, and (b) clip aggregation stays an offline axis that can
never move together with token pooling inside one measured delta.

``--variant`` records which pixels the features came from -- ``full_frame`` for the
main arms, ``person_crop`` / ``background_only`` for the plan's shortcut controls --
and is stamped into provenance so two variants can never be silently mixed in one
feature dir.

Run ``src.video.videomae_materialize`` afterwards to derive the classifier-ready dirs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from src.video.squat_dataset import SPLIT_NAMES, SQUAT_LABELED_ROOT, load_json_list
from src.video.squat_video_variants import Box, apply_variant
from src.video.videomae_backbone import encode_clip, load_backbone, resolve_device
from src.video.videomae_pooling import LEGACY_FIRST_TOKEN, MEAN_POOL_FC_NORM, build_provenance

import transformers  # noqa: E402 - after videomae_backbone so its import guard reports first
from transformers import VideoMAEImageProcessor  # noqa: E402

VARIANTS = ("full_frame", "person_crop", "background_only", "reencoded")


@dataclass(frozen=True)
class ClipRequest:
    video_id: str
    split: str
    video_path: Path


def find_video_path(video_root: Path, video_id: str) -> Path | None:
    direct = video_root / f"{video_id}.mp4"
    if direct.exists():
        return direct
    matches = sorted(video_root.rglob(f"{video_id}.mp4"))
    if matches:
        return matches[0]
    return None


def build_requests(video_root: Path, split_dir: Path, split_names: Sequence[str]) -> list[ClipRequest]:
    requests: list[ClipRequest] = []
    missing: list[str] = []

    for split_name in split_names:
        for video_id in load_json_list(split_dir / f"{split_name}_keys.json"):
            video_path = find_video_path(video_root, video_id)
            if video_path is None:
                missing.append(f"{split_name}/{video_id}")
                continue
            requests.append(ClipRequest(video_id=video_id, split=split_name, video_path=video_path))

    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        print(f"Warning: {len(missing)} videos were not found: {preview}{suffix}")

    return requests


def read_total_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def sample_clip_starts(total_frames: int, clip_length: int, frame_stride: int, num_clips: int) -> list[int]:
    """Evenly spaced clip starts across the whole video -- the historical sampling.

    Unchanged from the pre-fix extractor so that the re-extracted legacy arm can be
    checked against the archived 0.555 baseline.
    """
    if total_frames <= 0:
        return [0]

    effective_length = 1 + frame_stride * (clip_length - 1)
    max_start = max(total_frames - effective_length, 0)
    if num_clips <= 1:
        return [max_start // 2 if max_start else 0]
    return np.linspace(0, max_start, num=num_clips, dtype=int).tolist()


def read_clip_frames(
    cap: cv2.VideoCapture,
    start_frame: int,
    clip_length: int,
    frame_stride: int,
    total_frames: int,
) -> list[np.ndarray]:
    """Read ``clip_length`` frames from ``start_frame`` with ``frame_stride`` spacing.

    Seeks once, then decodes forward: a per-frame ``cap.set(POS_FRAMES)`` sends the
    decoder back to the nearest keyframe on every call, which dominates runtime. The
    frames selected are the same ones the old per-frame-seek version selected.
    """
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


def load_variant_boxes(manifest_path: Path) -> dict[str, Box | None]:
    """Read ``video_id -> box`` from a ``build_video_variants`` manifest.

    The controls are a deterministic function of (source video, one box), so the box
    is the only thing that has to travel. Applying it here rather than shipping
    re-encoded videos removes an entire lossy generation that the untouched
    ``full_frame`` arm would not have paid -- which is the difference between a
    control that can be read directly and one confounded with codec loss.
    """
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    boxes: dict[str, Box | None] = {}
    unrecorded: list[str] = []
    for row in manifest["rows"]:
        video_id = str(row["video_id"])
        if "box" not in row:
            # A row that never recorded a box is NOT the same as a row whose box is
            # null. Null means "no person was visible", a real and rare state that
            # legitimately leaves the video untouched. A missing key means the
            # builder skipped this video (its output already existed) and wrote a
            # stub -- and mapping that to None silently extracted an UNTRANSFORMED
            # video into a control arm. That happened: 51% of one control and 26% of
            # the other were full-frame before this check existed.
            unrecorded.append(video_id)
            continue
        box = row["box"]
        boxes[video_id] = Box(*box) if box else None

    if unrecorded:
        raise SystemExit(
            f"{len(unrecorded)} rows in {manifest_path} record no box "
            f"({unrecorded[:5]}). Rebuild the manifest with --boxes-only; a control arm "
            "that silently mixes in untransformed videos measures nothing."
        )
    return boxes


def extract_video_features(
    backbone,
    processor: VideoMAEImageProcessor,
    video_path: Path,
    clip_length: int,
    frame_stride: int,
    num_clips: int,
    device,
    fc_norm_weight: np.ndarray,
    fc_norm_bias: np.ndarray,
    fc_norm_eps: float,
    variant: str = "full_frame",
    box: Box | None = None,
) -> dict[str, np.ndarray]:
    """Both token-pooling modes for one video, from one forward pass per clip."""
    total_frames = read_total_frames(video_path)
    clip_starts = sample_clip_starts(total_frames, clip_length, frame_stride, num_clips)

    legacy_clips: list[np.ndarray] = []
    corrected_clips: list[np.ndarray] = []
    used_starts: list[int] = []

    cap = cv2.VideoCapture(str(video_path))
    try:
        for start_frame in clip_starts:
            frames = read_clip_frames(cap, start_frame, clip_length, frame_stride, total_frames)
            if not frames:
                continue
            if variant != "full_frame":
                # Box ops are channel-agnostic (crop, edge-blend, grey pad), so applying
                # them to the RGB frames the reader produced matches what the BGR video
                # builder writes.
                frames = apply_variant(frames, variant, box)
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
    finally:
        cap.release()

    if not legacy_clips:
        raise RuntimeError(f"No features could be extracted from {video_path}.")

    return {
        f"clip_features_{LEGACY_FIRST_TOKEN}": np.stack(legacy_clips, axis=0),
        f"clip_features_{MEAN_POOL_FC_NORM}": np.stack(corrected_clips, axis=0),
        "clip_starts": np.asarray(used_starts, dtype=np.int32),
        "total_frames": np.asarray(total_frames, dtype=np.int32),
    }


def save_feature_bundle(
    output_path: Path,
    request: ClipRequest,
    bundle: dict[str, np.ndarray],
    provenance: dict[str, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        video_id=np.asarray(request.video_id),
        split=np.asarray(request.split),
        video_path=np.asarray(str(request.video_path)),
        **bundle,
        **{f"provenance_{key}": np.asarray(value) for key, value in provenance.items()},
    )


def select_chunk(requests: Sequence[ClipRequest], num_chunks: int, chunk_index: int) -> list[ClipRequest]:
    """Round-robin slice of the work, so N Kaggle kernels can split one extraction."""
    if num_chunks <= 1:
        return list(requests)
    if not 0 <= chunk_index < num_chunks:
        raise ValueError(f"--chunk-index must be in [0, {num_chunks}), got {chunk_index}.")
    return [request for index, request in enumerate(requests) if index % num_chunks == chunk_index]


def iter_requests(requests: Sequence[ClipRequest], limit: int | None) -> Iterable[ClipRequest]:
    if limit is None:
        yield from requests
        return
    yield from requests[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Fitness-AQA squat VideoMAE features (both pooling modes, per-clip stacks)."
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=SQUAT_LABELED_ROOT / "videos",
        help="Root directory containing .mp4 videos for the chosen --variant.",
    )
    parser.add_argument("--split-dir", type=Path, default=SQUAT_LABELED_ROOT / "Splits")
    parser.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, default=list(SPLIT_NAMES))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SQUAT_LABELED_ROOT / "videomae_raw",
        help="Raw per-clip bundles. Deliberately NOT the legacy `videomae_features` dir, "
        "so a stale cache can never be mistaken for a corrected re-extraction.",
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default="full_frame",
        help="Which pixels to feed the model; stamped into provenance.",
    )
    parser.add_argument(
        "--variant-manifest",
        type=Path,
        default=None,
        help="build_video_variants manifest supplying one box per video. Required for "
        "every variant except full_frame; the transform is applied in memory so the "
        "controls decode the same source file as the main arm.",
    )
    parser.add_argument("--model-name", type=str, default="MCG-NJU/videomae-base-finetuned-kinetics")
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--num-clips", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N videos (smoke test).")
    parser.add_argument("--num-chunks", type=int, default=1, help="Split the work into N round-robin chunks.")
    parser.add_argument("--chunk-index", type=int, default=0, help="Which chunk to process (0-based).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto.")
    args = parser.parse_args()

    requests = build_requests(args.video_root, args.split_dir, args.splits)
    if not requests:
        raise SystemExit("No videos were found to process.")

    boxes: dict[str, Box | None] = {}
    if args.variant != "full_frame":
        if args.variant_manifest is None:
            raise SystemExit(f"--variant {args.variant} needs --variant-manifest to supply its boxes.")
        boxes = load_variant_boxes(args.variant_manifest)
        uncovered = [request.video_id for request in requests if request.video_id not in boxes]
        if uncovered:
            raise SystemExit(
                f"{len(uncovered)} videos are missing from {args.variant_manifest} "
                f"({uncovered[:5]}). A control arm covering fewer videos than the main arm "
                "is not a paired comparison."
            )
    requests = select_chunk(requests, args.num_chunks, args.chunk_index)
    if args.num_chunks > 1:
        print(f"Chunk {args.chunk_index + 1}/{args.num_chunks}: {len(requests)} videos")

    device = resolve_device(args.device)
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

    work = list(iter_requests(requests, args.limit))
    written = 0
    skipped = 0
    for index, request in enumerate(work, start=1):
        output_path = args.output_dir / request.split / f"{request.video_id}.npz"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        bundle = extract_video_features(
            backbone=backbone,
            processor=processor,
            video_path=request.video_path,
            clip_length=args.clip_length,
            frame_stride=args.frame_stride,
            num_clips=args.num_clips,
            device=device,
            fc_norm_weight=fc_weight,
            fc_norm_bias=fc_bias,
            fc_norm_eps=fc_eps,
            variant=args.variant,
            box=boxes.get(request.video_id),
        )
        save_feature_bundle(output_path, request, bundle, provenance)
        written += 1
        if written % 25 == 0 or index == len(work):
            print(f"[{index}/{len(work)}] wrote {written} bundles (skipped {skipped})")

    print(f"Wrote {written} VideoMAE feature bundles ({skipped} already present) under {args.output_dir}")


if __name__ == "__main__":
    main()
