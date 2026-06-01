from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from src.rehab24.dataset import DEFAULT_DATA_ROOT, DEFAULT_PROCESSED_ROOT, load_manifest, resolve_data_path

try:
    import torch
    from transformers import VideoMAEImageProcessor, VideoMAEModel
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
    frames: list[np.ndarray] = []
    last_frame: np.ndarray | None = None
    for offset in range(clip_length):
        frame_index = min(start_frame + offset * frame_stride, max(total_frames - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
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
    model: VideoMAEModel,
    processor: VideoMAEImageProcessor,
    video_path: Path,
    first_frame: int,
    last_frame: int,
    clip_length: int,
    frame_stride: int,
    num_clips: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_starts = sample_clip_starts(first_frame, last_frame, clip_length, frame_stride, num_clips)
    clip_features: list[np.ndarray] = []

    try:
        for start_frame in clip_starts:
            frames = read_clip_frames(cap, start_frame, clip_length, frame_stride, total_frames)
            if not frames:
                continue
            inputs = processor(frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
            clip_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().numpy()
            clip_features.append(clip_embedding.astype(np.float32, copy=False))
    finally:
        cap.release()

    if not clip_features:
        raise RuntimeError(f"No VideoMAE features could be extracted from {video_path}")

    stacked = np.stack(clip_features, axis=0)
    return {
        "video_feature": stacked.max(axis=0).astype(np.float32, copy=False),
        "clip_features": stacked.astype(np.float32, copy=False),
        "clip_starts": np.asarray(clip_starts, dtype=np.int32),
        "first_frame": np.asarray(first_frame, dtype=np.int32),
        "last_frame": np.asarray(last_frame, dtype=np.int32),
        "total_frames": np.asarray(total_frames, dtype=np.int32),
    }


def save_feature(path: Path, row: dict[str, str], bundle: dict[str, np.ndarray]) -> None:
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
    )


def iter_rows(rows: Sequence[dict[str, str]], limit: int | None) -> Iterable[dict[str, str]]:
    if limit is None:
        yield from rows
        return
    yield from rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract REHAB24-6 repetition-level VideoMAE features.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROCESSED_ROOT / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_ROOT / "videomae_features")
    parser.add_argument("--model-name", type=str, default="MCG-NJU/videomae-base-finetuned-kinetics")
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--num-clips", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto.")
    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading VideoMAE model `{args.model_name}` on {device}...")
    processor = VideoMAEImageProcessor.from_pretrained(args.model_name)
    model = VideoMAEModel.from_pretrained(args.model_name).to(device)
    model.eval()

    rows = load_manifest(args.manifest)
    written = 0
    for index, row in enumerate(iter_rows(rows, args.limit), start=1):
        output_path = args.output_dir / row["split"] / f"{row['sample_id']}.npz"
        if output_path.exists() and not args.overwrite:
            continue
        bundle = extract_repetition_features(
            model=model,
            processor=processor,
            video_path=resolve_data_path(args.data_root, row["video_path"]),
            first_frame=int(row["first_frame"]),
            last_frame=int(row["last_frame"]),
            clip_length=args.clip_length,
            frame_stride=args.frame_stride,
            num_clips=args.num_clips,
            device=device,
        )
        save_feature(output_path, row, bundle)
        written += 1
        if index % 50 == 0:
            print(f"Processed {index} manifest rows...")

    print(f"Wrote {written} VideoMAE feature files under {args.output_dir}")


if __name__ == "__main__":
    main()

