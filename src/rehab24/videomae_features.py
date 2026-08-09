"""Extract REHAB24-6 repetition-level VideoMAE features.

Emits BOTH token-pooling modes (see ``src.video.videomae_pooling``) from a single
forward pass, and stores the per-clip stacks rather than a pre-aggregated vector.

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
from src.video.videomae_pooling import LEGACY_FIRST_TOKEN, MEAN_POOL_FC_NORM, build_provenance, layer_norm

try:
    import torch
    import transformers
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
except ImportError as exc:  # pragma: no cover - imported at runtime for Colab/GPU workflows
    raise SystemExit(
        "REHAB24-6 VideoMAE extraction requires `torch` and `transformers`.\n"
        "Install them with: pip install torch transformers accelerate timm"
    ) from exc


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


def assert_fc_norm_pretrained(weight: np.ndarray, bias: np.ndarray, model_name: str) -> None:
    """Fail loudly if ``fc_norm`` is still at LayerNorm default init.

    ``from_pretrained`` warns about missing keys but does not raise, so a checkpoint
    without ``fc_norm`` yields weight=1/bias=0 -- an identity-ish normalization that
    would look exactly like "corrected pooling doesn't help" in the Stage A delta.
    """
    if np.allclose(weight, 1.0) and np.allclose(bias, 0.0):
        raise SystemExit(
            f"fc_norm of `{model_name}` is at LayerNorm default init (weight=1, bias=0), "
            "so it was NOT loaded from the checkpoint. Refusing to extract features that "
            "would silently misrepresent the classification path."
        )


def load_backbone(model_name: str, device: torch.device) -> tuple[torch.nn.Module, np.ndarray, np.ndarray, float]:
    """Load the VideoMAE backbone plus the *pretrained* ``fc_norm`` parameters.

    ``fc_norm`` exists only on ``VideoMAEForVideoClassification``. Loading
    ``VideoMAEModel`` instead would leave no way to reproduce the classification
    representation, because ``use_mean_pooling=True`` checkpoints set the backbone's
    own ``layernorm`` to ``None``.
    """
    model = VideoMAEForVideoClassification.from_pretrained(model_name)
    if model.fc_norm is None:
        raise SystemExit(
            f"`{model_name}` has use_mean_pooling=False, so it has no fc_norm and the "
            "mean_pool_fc_norm mode is undefined for it. Use a mean-pooling checkpoint."
        )

    weight = model.fc_norm.weight.detach().cpu().numpy().astype(np.float32)
    bias = model.fc_norm.bias.detach().cpu().numpy().astype(np.float32)

    assert_fc_norm_pretrained(weight, bias, model_name)

    backbone = model.videomae.to(device)
    backbone.eval()
    return backbone, weight, bias, float(model.config.layer_norm_eps)


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
) -> dict[str, np.ndarray]:
    """Both token-pooling modes for one repetition, from one forward pass per clip."""
    clip_starts = sample_clip_starts(first_frame, last_frame, clip_length, frame_stride, num_clips)
    legacy_clips: list[np.ndarray] = []
    corrected_clips: list[np.ndarray] = []
    used_starts: list[int] = []

    for start_frame in clip_starts:
        frames = read_clip_frames(cap, start_frame, clip_length, frame_stride, total_frames)
        if not frames:
            continue
        inputs = processor(frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            hidden = backbone(pixel_values=pixel_values).last_hidden_state

        # Index 0 is the first patch token, not a CLS token -- reproduced only so the
        # paired comparison can isolate the pooling fix.
        legacy_clips.append(hidden[:, 0, :].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False))
        mean_pooled = hidden.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        corrected_clips.append(layer_norm(mean_pooled, fc_norm_weight, fc_norm_bias, fc_norm_eps))
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

    device = torch.device("cpu") if args.device == "cpu" else torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    )

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
                )
                save_feature(args.output_dir / row["split"] / f"{row['sample_id']}.npz", row, bundle, provenance)
                written += 1
        finally:
            cap.release()
        print(f"[{video_index}] {video_path}: wrote {len(pending)} repetitions (total {written})")

    print(f"Wrote {written} VideoMAE feature bundles ({skipped} already present) under {args.output_dir}")


if __name__ == "__main__":
    main()
