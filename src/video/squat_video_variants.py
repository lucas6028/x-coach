"""Build the person-crop and background-only video variants for the Stage B controls.

``notes/videomae_dataset_validation_plan.md`` asks whether a corrected VideoMAE
representation carries movement quality or merely scene and viewpoint. The control is
to re-extract features from videos where one of the two is removed:

``person_crop``
    Crop to a square box around the athlete. Background is mostly gone; if accuracy
    survives, the signal is on the body. Square rather than tight because the
    VideoMAE processor resizes the shortest edge to 224 and then centre-crops 224 --
    a tall, narrow crop would have its own feet cut off, which is exactly the
    preprocessing asymmetry Stage A had to log as a caveat on REHAB24-6 cam18.

``background_only``
    Paint the person's box over with a per-pixel temporal median of the video, i.e.
    an estimate of the scene without the athlete. If accuracy survives *this*, the
    classifier was reading the gym, not the squat. Median inpainting rather than a
    grey rectangle because a filled rectangle still draws the athlete's position and
    size in the frame -- a leak that would let the control pass for the wrong reason.

Both variants use ONE box per video (the union over frames) rather than a per-frame
box: a per-frame box makes the crop pan with the athlete, which injects motion that
is not in the original and would differ between the two variants.

Frame count and fps are preserved exactly, so ``sample_clip_starts`` picks the same
clip starts in every variant and the arms stay paired frame-for-frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VARIANTS = ("person_crop", "background_only")
DEFAULT_VISIBILITY = 0.5
DEFAULT_MARGIN = 0.15
#: Frames sampled to estimate the background. Odd so the median never interpolates.
BACKGROUND_SAMPLES = 31


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


def person_box_from_pose(
    pose: dict,
    visibility_threshold: float = DEFAULT_VISIBILITY,
) -> Box | None:
    """Union over frames of the visible landmarks' bounding box, in pixels.

    Returns ``None`` when no landmark anywhere in the video clears the visibility
    threshold -- the caller decides what a person-less video means for its variant.
    """
    metadata = pose.get("metadata", {})
    width = int(metadata.get("width", 0))
    height = int(metadata.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Pose metadata is missing frame width/height.")

    xs: list[float] = []
    ys: list[float] = []
    for frame in pose.get("frames", []):
        for landmark in frame.get("landmarks") or []:
            if float(landmark.get("visibility", 0.0)) < visibility_threshold:
                continue
            xs.append(float(landmark["x"]) * width)
            ys.append(float(landmark["y"]) * height)

    if not xs:
        return None

    x0 = max(int(np.floor(min(xs))), 0)
    y0 = max(int(np.floor(min(ys))), 0)
    x1 = min(int(np.ceil(max(xs))), width)
    y1 = min(int(np.ceil(max(ys))), height)
    if x1 <= x0 or y1 <= y0:
        return None
    return Box(x0, y0, x1, y1)


def square_crop_box(box: Box, width: int, height: int, margin: float = DEFAULT_MARGIN) -> Box:
    """Smallest square containing ``box`` plus ``margin``, clamped inside the frame.

    MediaPipe's landmarks stop at the wrists/ankles, so the margin also buys back the
    hands and feet that a landmark-tight box would clip.
    """
    side = max(box.x1 - box.x0, box.y1 - box.y0) * (1.0 + margin)
    side = int(min(side, width, height))
    side = max(side, 1)

    centre_x = (box.x0 + box.x1) / 2.0
    centre_y = (box.y0 + box.y1) / 2.0
    x0 = int(round(centre_x - side / 2.0))
    y0 = int(round(centre_y - side / 2.0))
    x0 = min(max(x0, 0), width - side)
    y0 = min(max(y0, 0), height - side)
    return Box(x0, y0, x0 + side, y0 + side)


def read_all_frames(video_path: Path, target_frames: int | None = None) -> list[np.ndarray]:
    """Decode a whole video, padded/truncated to ``target_frames`` when given.

    ``CAP_PROP_FRAME_COUNT`` is a container header value and can disagree with what
    actually decodes. The extractor derives clip starts from the *variant's* frame
    count, so a variant that decodes one frame short of its source would sample
    slightly different clips and quietly break the pairing. Padding with the last
    frame keeps every variant on the source's frame grid.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be decoded from {video_path}.")

    if target_frames is not None and target_frames > 0:
        if len(frames) > target_frames:
            frames = frames[:target_frames]
        while len(frames) < target_frames:
            frames.append(frames[-1].copy())

    return frames


def estimate_background(frames: list[np.ndarray], samples: int = BACKGROUND_SAMPLES) -> np.ndarray:
    """Per-pixel temporal median over evenly spaced frames.

    On a static camera this is the empty scene. On a handheld one it is a smear --
    still person-free, which is what the control needs, but it is why the
    background-only arm is read as "scene cues survive?" and not as a clean image.
    """
    indices = np.unique(np.linspace(0, len(frames) - 1, num=min(samples, len(frames)), dtype=int))
    stack = np.stack([frames[index] for index in indices], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def apply_variant(frames: list[np.ndarray], variant: str, box: Box | None) -> list[np.ndarray]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}.")
    if box is None:
        # No person was ever visible. Both variants degrade to the untouched video;
        # the manifest records these so they can be reported, not silently counted.
        return frames

    if variant == "person_crop":
        return [frame[box.y0 : box.y1, box.x0 : box.x1].copy() for frame in frames]

    background = estimate_background(frames)
    masked: list[np.ndarray] = []
    for frame in frames:
        painted = frame.copy()
        painted[box.y0 : box.y1, box.x0 : box.x1] = background[box.y0 : box.y1, box.x0 : box.x1]
        masked.append(painted)
    return masked


def write_video(frames: list[np.ndarray], output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a writer for {output_path}.")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def build_variant_video(
    video_path: Path,
    pose_path: Path,
    output_path: Path,
    variant: str,
    visibility_threshold: float = DEFAULT_VISIBILITY,
    margin: float = DEFAULT_MARGIN,
) -> dict:
    """Write one variant video; return its manifest row."""
    with pose_path.open("r", encoding="utf-8") as f:
        pose = json.load(f)

    metadata = pose.get("metadata", {})
    width = int(metadata.get("width", 0))
    height = int(metadata.get("height", 0))
    fps = float(metadata.get("fps", 30.0)) or 30.0
    total_frames = int(metadata.get("total_frames", 0))

    landmark_box = person_box_from_pose(pose, visibility_threshold)
    box = landmark_box
    if landmark_box is not None and variant == "person_crop":
        box = square_crop_box(landmark_box, width, height, margin)

    frames = read_all_frames(video_path, target_frames=total_frames)
    written_frames = apply_variant(frames, variant, box)
    write_video(written_frames, output_path, fps)

    return {
        "video_id": video_path.stem,
        "variant": variant,
        "pose_detected": landmark_box is not None,
        "box": box.as_tuple() if box is not None else None,
        "source_frames": total_frames,
        "written_frames": len(written_frames),
        "fps": fps,
        "frame_size": [int(written_frames[0].shape[1]), int(written_frames[0].shape[0])],
    }
