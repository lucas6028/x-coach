from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class PoseRequest:
    video_id: str
    video_path: Path
    json_path: Path
    annotated_video_path: Path | None


def load_json_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}.")
    return [str(item) for item in data]


def parse_split_names(value: str) -> list[str]:
    split_names = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(split_names) - set(SPLIT_NAMES))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported splits: {', '.join(invalid)}")
    return split_names


def find_video_path(video_dir: Path, video_id: str) -> Path | None:
    direct = video_dir / f"{video_id}.mp4"
    if direct.exists():
        return direct
    matches = list(video_dir.rglob(f"{video_id}.mp4"))
    if matches:
        return matches[0]
    return None


def build_labeled_requests(
    video_dir: Path,
    split_dir: Path,
    output_dir: Path,
    split_names: list[str],
    write_video: bool,
) -> list[PoseRequest]:
    requests: list[PoseRequest] = []
    missing: list[str] = []

    for split_name in split_names:
        video_ids = load_json_list(split_dir / f"{split_name}_keys.json")
        split_output_dir = output_dir / split_name
        for video_id in video_ids:
            video_path = find_video_path(video_dir, video_id)
            if video_path is None:
                missing.append(f"{split_name}/{video_id}")
                continue
            annotated_video_path = (
                split_output_dir / f"{video_id}_annotated.mp4"
                if write_video
                else None
            )
            requests.append(
                PoseRequest(
                    video_id=video_id,
                    video_path=video_path,
                    json_path=split_output_dir / f"{video_id}.json",
                    annotated_video_path=annotated_video_path,
                )
            )

    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        print(f"Warning: {len(missing)} split videos were not found: {preview}{suffix}")

    return requests


def build_unlabeled_requests(video_dir: Path, output_dir: Path, write_video: bool) -> list[PoseRequest]:
    requests: list[PoseRequest] = []
    for video_path in sorted(video_dir.rglob("*.mp4")):
        video_id = video_path.stem
        annotated_video_path = output_dir / f"{video_id}_annotated.mp4" if write_video else None
        requests.append(
            PoseRequest(
                video_id=video_id,
                video_path=video_path,
                json_path=output_dir / f"{video_id}.json",
                annotated_video_path=annotated_video_path,
            )
        )
    return requests


def iter_requests(requests: list[PoseRequest], limit: int | None):
    if limit is None:
        yield from requests
        return
    for request in requests[:limit]:
        yield request


def run_request(script_path: Path, request: PoseRequest) -> None:
    request.json_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script_path),
        "--input",
        str(request.video_path),
        "--output_json",
        str(request.json_path),
    ]
    if request.annotated_video_path is not None:
        request.annotated_video_path.parent.mkdir(parents=True, exist_ok=True)
        command.extend(["--output_video", str(request.annotated_video_path)])
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch process squat videos with MediaPipe Pose.")
    parser.add_argument(
        "--dataset",
        choices=("labeled", "unlabeled"),
        default="labeled",
        help="Dataset layout to process. The labeled layout follows train/val/test split files.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Directory containing .mp4 videos. Defaults depend on --dataset.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "Splits",
        help="Directory containing train_keys.json, val_keys.json, and test_keys.json.",
    )
    parser.add_argument("--splits", type=parse_split_names, default=list(SPLIT_NAMES))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for pose JSON outputs. Defaults depend on --dataset.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process.")
    parser.add_argument("--no-video", action="store_true", help="Do not generate annotated videos.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing pose JSON files.")
    args = parser.parse_args()

    if args.dataset == "labeled":
        video_dir = args.video_dir or REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "videos"
        output_dir = args.output_dir or REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "pose_json"
        requests = build_labeled_requests(
            video_dir=video_dir,
            split_dir=args.split_dir,
            output_dir=output_dir,
            split_names=args.splits,
            write_video=not args.no_video,
        )
    else:
        video_dir = args.video_dir or REPO_ROOT / "data" / "Squat" / "Unlabeled_Dataset" / "videos"
        output_dir = args.output_dir or REPO_ROOT / "data" / "Squat" / "Unlabeled_Dataset" / "processed_poses"
        requests = build_unlabeled_requests(video_dir=video_dir, output_dir=output_dir, write_video=not args.no_video)

    if not requests:
        raise SystemExit("No videos were found to process.")

    script_path = REPO_ROOT / "src" / "process_videos.py"
    print(f"Found {len(requests)} videos to process from {video_dir}.")
    print(f"Writing pose outputs under {output_dir}.")

    processed = 0
    skipped = 0
    failed = 0
    for index, request in enumerate(iter_requests(requests, args.limit), start=1):
        if request.json_path.exists() and not args.overwrite:
            print(f"[{index}] Skipping {request.video_id}, already processed.")
            skipped += 1
            continue

        print(f"[{index}] Processing {request.video_id} from {request.video_path.name}...")
        try:
            run_request(script_path=script_path, request=request)
            processed += 1
        except subprocess.CalledProcessError as exc:
            failed += 1
            print(f"Error processing {request.video_id}: {exc}")

    print(f"Done. processed={processed} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
