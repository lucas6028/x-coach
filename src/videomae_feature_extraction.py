from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
    from transformers import VideoMAEImageProcessor, VideoMAEModel
except ImportError as exc:  # pragma: no cover - imported at runtime on user machines
    raise SystemExit(
        "VideoMAE feature extraction requires `torch` and `transformers`.\n"
        "Install them with something like:\n"
        "  pip install torch transformers accelerate timm\n"
        "Then rerun this command."
    ) from exc


@dataclass(frozen=True)
class ClipRequest:
    video_id: str
    video_path: Path


def load_json_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}.")
    return [str(item) for item in data]


def find_video_path(video_root: Path, video_id: str) -> Path | None:
    direct = video_root / f"{video_id}.mp4"
    if direct.exists():
        return direct
    matches = list(video_root.rglob(f"{video_id}.mp4"))
    if matches:
        return matches[0]
    return None


def build_requests(video_root: Path, video_ids: Sequence[str]) -> list[ClipRequest]:
    requests: list[ClipRequest] = []
    missing: list[str] = []

    for video_id in video_ids:
        video_path = find_video_path(video_root, video_id)
        if video_path is None:
            missing.append(video_id)
            continue
        requests.append(ClipRequest(video_id=video_id, video_path=video_path))

    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        print(f"Warning: {len(missing)} videos were not found: {preview}{suffix}")

    return requests


def read_total_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return total_frames


def sample_clip_starts(total_frames: int, clip_length: int, frame_stride: int, num_clips: int) -> list[int]:
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
    frames: list[np.ndarray] = []
    last_frame: np.ndarray | None = None

    for offset in range(clip_length):
        frame_index = start_frame + offset * frame_stride
        if total_frames > 0:
            frame_index = min(frame_index, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            if last_frame is None:
                continue
            frame = last_frame.copy()

        last_frame = frame
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not frames and last_frame is not None:
        frames = [cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)] * clip_length

    while frames and len(frames) < clip_length:
        frames.append(frames[-1].copy())

    return frames


def extract_video_features(
    model: VideoMAEModel,
    processor: VideoMAEImageProcessor,
    video_path: Path,
    clip_length: int,
    frame_stride: int,
    num_clips: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    total_frames = read_total_frames(video_path)
    clip_starts = sample_clip_starts(total_frames, clip_length, frame_stride, num_clips)

    clip_features: list[np.ndarray] = []
    actual_starts: list[int] = []

    cap = cv2.VideoCapture(str(video_path))
    try:
        for start_frame in clip_starts:
            frames = read_clip_frames(
                cap=cap,
                start_frame=start_frame,
                clip_length=clip_length,
                frame_stride=frame_stride,
                total_frames=total_frames,
            )
            if not frames:
                continue

            inputs = processor(frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)

            # The first token is the CLS token for VideoMAE backbones.
            clip_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().numpy()
            clip_features.append(clip_embedding.astype(np.float32, copy=False))
            actual_starts.append(start_frame)
    finally:
        cap.release()

    if not clip_features:
        raise RuntimeError(f"No features could be extracted from {video_path}.")

    stacked = np.stack(clip_features, axis=0)
    video_feature = stacked.mean(axis=0)

    return {
        "video_feature": video_feature.astype(np.float32, copy=False),
        "clip_features": stacked.astype(np.float32, copy=False),
        "clip_starts": np.asarray(actual_starts, dtype=np.int32),
        "total_frames": np.asarray([total_frames], dtype=np.int32),
    }


def save_feature_bundle(output_path: Path, video_id: str, video_path: Path, bundle: dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        video_id=np.asarray(video_id),
        video_path=np.asarray(str(video_path)),
        **bundle,
    )


def iter_requests(requests: Sequence[ClipRequest], limit: int | None) -> Iterable[ClipRequest]:
    if limit is None:
        yield from requests
        return
    for request in requests[:limit]:
        yield request


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract VideoMAE spatiotemporal features from video clips.")
    parser.add_argument(
        "--video-root",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "videos",
        help="Root directory containing .mp4 videos.",
    )
    parser.add_argument(
        "--video-ids",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "Splits" / "train_keys.json",
        help="JSON list of video IDs to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "videomae_features" / "train",
        help="Directory to write feature bundles.",
    )
    parser.add_argument("--model-name", type=str, default="MCG-NJU/videomae-base-finetuned-kinetics")
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--num-clips", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto.")
    args = parser.parse_args()

    if args.video_ids is not None:
        video_ids = load_json_list(args.video_ids)
    else:
        video_ids = sorted({path.stem for path in args.video_root.rglob("*.mp4")})

    requests = build_requests(args.video_root, video_ids)
    if not requests:
        raise SystemExit("No videos were found to process.")

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading VideoMAE model `{args.model_name}` on {device}...")
    processor = VideoMAEImageProcessor.from_pretrained(args.model_name)
    model = VideoMAEModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    print(f"Processing {len(requests)} videos...")
    for index, request in enumerate(iter_requests(requests, args.limit), start=1):
        output_path = args.output_dir / f"{request.video_id}.npz"
        if output_path.exists():
            print(f"[{index}] Skipping {request.video_id}, already exists.")
            continue

        print(f"[{index}] Extracting {request.video_id} from {request.video_path.name}...")
        bundle = extract_video_features(
            model=model,
            processor=processor,
            video_path=request.video_path,
            clip_length=args.clip_length,
            frame_stride=args.frame_stride,
            num_clips=args.num_clips,
            device=device,
        )
        save_feature_bundle(output_path, request.video_id, request.video_path, bundle)

    print(f"Done. Feature bundles saved under {args.output_dir}")


if __name__ == "__main__":
    main()
