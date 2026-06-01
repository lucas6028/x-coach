from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "REHAB24-6"
DEFAULT_PROCESSED_ROOT = DEFAULT_DATA_ROOT / "processed"

CAMERAS = ("cam17", "cam18")
SPLIT_NAMES = ("train", "val", "test")
TRAIN_PERSON_IDS = frozenset({"1", "2", "3", "4", "5", "7", "10"})
VAL_PERSON_IDS = frozenset({"6"})
TEST_PERSON_IDS = frozenset({"8", "9"})

EXERCISE_NAMES = {
    "1": "arm abduction",
    "2": "arm VW",
    "3": "table push-ups",
    "4": "leg abduction",
    "5": "leg lunge",
    "6": "squats",
}

CAMERA18_ORIENTATION = {
    "front": "side",
    "half-profile": "half-profile",
    "profile": "front",
    "side": "front",
}

MANIFEST_FIELDS = [
    "sample_id",
    "split",
    "video_id",
    "repetition_number",
    "exercise_id",
    "exercise_name",
    "person_id",
    "first_frame",
    "last_frame",
    "camera",
    "camera_orientation",
    "cam17_orientation",
    "correctness",
    "mocap_erroneous",
    "exercise_subtype",
    "lights_on",
    "extra_person_in_camera",
    "skeleton_3d_path",
    "skeleton_2d_path",
    "video_path",
]


@dataclass(frozen=True)
class Segment:
    video_id: str
    repetition_number: str
    exercise_id: str
    person_id: str
    first_frame: int
    last_frame: int
    cam17_orientation: str
    mocap_erroneous: str
    exercise_subtype: str
    lights_on: str
    extra_person_in_cam17: str
    extra_person_in_cam18: str
    correctness: int


def read_segmentation(path: Path) -> list[Segment]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f, delimiter=";")
        return [segment_from_row(row) for row in rows]


def segment_from_row(row: dict[str, str]) -> Segment:
    return Segment(
        video_id=row["video_id"],
        repetition_number=row["repetition_number"],
        exercise_id=row["exercise_id"],
        person_id=row["person_id"],
        first_frame=int(row["first_frame"]),
        last_frame=int(row["last_frame"]),
        cam17_orientation=row["cam17_orientation"],
        mocap_erroneous=row["mocap_erroneous"],
        exercise_subtype=row["exercise_subtype"],
        lights_on=row["lights_on"],
        extra_person_in_cam17=row["extra_person_in_cam17"],
        extra_person_in_cam18=row["extra_person_in_cam18"],
        correctness=int(row["correctness"]),
    )


def split_for_person(person_id: str) -> str:
    if person_id in TRAIN_PERSON_IDS:
        return "train"
    if person_id in VAL_PERSON_IDS:
        return "val"
    if person_id in TEST_PERSON_IDS:
        return "test"
    raise ValueError(f"Unsupported REHAB24-6 person_id for fixed split: {person_id}")


def sample_id(segment: Segment, camera: str) -> str:
    return f"Ex{segment.exercise_id}_{segment.video_id}_rep{segment.repetition_number}_{camera}"


def camera_orientation(segment: Segment, camera: str) -> str:
    if camera == "cam17":
        return segment.cam17_orientation
    if camera == "cam18":
        return CAMERA18_ORIENTATION.get(segment.cam17_orientation, "unknown")
    raise ValueError(f"Unsupported camera: {camera}")


def relative_paths(segment: Segment, camera: str) -> tuple[str, str, str]:
    exercise_dir = f"Ex{segment.exercise_id}"
    if camera == "cam17":
        skeleton_2d = f"{exercise_dir}/{segment.video_id}-c17-30fps.npy"
        video = f"{exercise_dir}/{segment.video_id}-Camera17-30fps.mp4"
    elif camera == "cam18":
        skeleton_2d = f"{exercise_dir}/{segment.video_id}-c18-30fps.npy"
        video = f"{exercise_dir}/{segment.video_id}-Camera18-30fps-transposed.mp4"
    else:
        raise ValueError(f"Unsupported camera: {camera}")
    skeleton_3d = f"{exercise_dir}/{segment.video_id}-30fps.npy"
    return skeleton_3d, skeleton_2d, video


def build_manifest_rows(segments: Sequence[Segment], cameras: Sequence[str] = CAMERAS) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for segment in segments:
        for camera in cameras:
            skeleton_3d, skeleton_2d, video = relative_paths(segment, camera)
            rows.append(
                {
                    "sample_id": sample_id(segment, camera),
                    "split": split_for_person(segment.person_id),
                    "video_id": segment.video_id,
                    "repetition_number": segment.repetition_number,
                    "exercise_id": segment.exercise_id,
                    "exercise_name": EXERCISE_NAMES.get(segment.exercise_id, ""),
                    "person_id": segment.person_id,
                    "first_frame": str(segment.first_frame),
                    "last_frame": str(segment.last_frame),
                    "camera": camera,
                    "camera_orientation": camera_orientation(segment, camera),
                    "cam17_orientation": segment.cam17_orientation,
                    "correctness": str(segment.correctness),
                    "mocap_erroneous": segment.mocap_erroneous,
                    "exercise_subtype": segment.exercise_subtype,
                    "lights_on": segment.lights_on,
                    "extra_person_in_camera": (
                        segment.extra_person_in_cam17 if camera == "cam17" else segment.extra_person_in_cam18
                    ),
                    "skeleton_3d_path": skeleton_3d,
                    "skeleton_2d_path": skeleton_2d,
                    "video_path": video,
                }
            )
    return rows


def write_manifest(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_splits_and_labels(processed_root: Path, rows: Sequence[dict[str, str]]) -> None:
    split_dir = processed_root / "splits"
    label_dir = processed_root / "labels"
    split_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    labels = {row["sample_id"]: int(row["correctness"]) for row in rows}
    with (label_dir / "correctness.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, sort_keys=True)

    for split_name in SPLIT_NAMES:
        split_ids = [row["sample_id"] for row in rows if row["split"] == split_name]
        with (split_dir / f"{split_name}_keys.json").open("w", encoding="utf-8") as f:
            json.dump(split_ids, f, indent=2)


def resolve_data_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return data_root / path


def validate_manifest_paths(data_root: Path, rows: Iterable[dict[str, str]]) -> list[str]:
    missing: list[str] = []
    for row in rows:
        for field in ("skeleton_3d_path", "skeleton_2d_path", "video_path"):
            path = resolve_data_path(data_root, row[field])
            if not path.exists():
                missing.append(f"{row['sample_id']}:{field}:{path}")
    return missing


def build_manifest_main() -> None:
    parser = argparse.ArgumentParser(description="Build a repetition-level manifest for REHAB24-6.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--segmentation", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--cameras", type=str, default="cam17,cam18")
    parser.add_argument("--skip-path-validation", action="store_true")
    args = parser.parse_args()

    segmentation_path = args.segmentation or args.data_root / "Segmentation.csv"
    manifest_output = args.manifest_output or args.processed_root / "manifest.csv"
    cameras = [item.strip() for item in args.cameras.split(",") if item.strip()]
    invalid_cameras = sorted(set(cameras) - set(CAMERAS))
    if invalid_cameras:
        raise SystemExit(f"Unsupported cameras: {', '.join(invalid_cameras)}")

    rows = build_manifest_rows(read_segmentation(segmentation_path), cameras=cameras)
    if not args.skip_path_validation:
        missing = validate_manifest_paths(args.data_root, rows)
        if missing:
            preview = "\n".join(missing[:10])
            suffix = "" if len(missing) <= 10 else f"\n... +{len(missing) - 10} more"
            raise SystemExit(f"Manifest references missing files:\n{preview}{suffix}")

    write_manifest(manifest_output, rows)
    write_splits_and_labels(args.processed_root, rows)
    print(f"Wrote {len(rows)} REHAB24-6 samples to {manifest_output}")
    for split_name in SPLIT_NAMES:
        count = sum(row["split"] == split_name for row in rows)
        print(f"{split_name}: {count}")


if __name__ == "__main__":
    build_manifest_main()

