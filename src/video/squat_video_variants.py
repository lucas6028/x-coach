"""Build the person-crop and background-only video variants for the Stage B controls.

``notes/videomae_dataset_validation_plan.md`` asks whether a corrected VideoMAE
representation carries movement quality or merely scene and viewpoint. The control is
to re-extract features from videos where one of the two is removed:

``person_crop``
    Crop tight to the athlete, then letterbox to square on a neutral grey. Background
    is mostly gone; if accuracy survives, the signal is on the body.

    The letterbox is not cosmetic. The VideoMAE processor resizes the shortest edge
    to 224 and centre-crops 224, so a tall narrow crop would have its feet cut off --
    the preprocessing asymmetry Stage A had to log as a caveat on REHAB24-6 cam18.
    Padding to square makes that centre crop a no-op. *Expanding* the crop to square
    instead was tried first and defeats the control: a standing athlete's box is tall
    and narrow, so squaring it kept a median 77% of the original frame area (p90
    100%, measured over 300 train videos) -- it would have removed almost no
    background. The landmark box itself is a median 25% of the frame.

``background_only``
    Paint the person's box over by interpolating horizontally between the columns
    just outside it, so the scene keeps its colours and lighting and the athlete
    keeps nothing. If accuracy survives *this*, the classifier was reading the gym,
    not the squat.

    A per-pixel temporal median was tried first and is wrong here: the athlete
    occupies the middle of that box in essentially every frame of a squat, so the
    median *is* the athlete and the "background" video shows a recognisable smeared
    person (verified on 33048_1). The true background behind a stationary lifter is
    never observed by a fixed camera, so no fill can recover it -- the honest choice
    is a fill that provably carries no body structure. What survives is the box's
    position and size, a rectangle-shaped leak that is stated in the results rather
    than engineered away.

``person_crop_centercrop``
    The same crop as ``person_crop``, from the same expanded box, but WITHOUT the
    letterbox -- the processor is left to resize the shortest edge to 224 and centre
    crop, which on a tall narrow crop keeps only the middle band of the athlete.
    Background removed, body truncated: the cell that
    ``notes/videomae_person_crop_validation_plan.md`` needs to separate "removing the
    scene" (F1) from "finally seeing the whole person" (F2). It must differ from
    ``person_crop`` in the letterbox and nothing else, which is why it takes the
    expanded box rather than the raw landmark box.

``full_frame_letterbox``
    The whole frame padded to square on the same neutral grey, nothing cropped. Scene
    kept, body complete -- the other new cell of the 2x2. It needs no box at all, and
    it is a deliberate no-op on the 768 of 1623 Fitness-AQA squat videos that are
    already square (47.3%, measured from the person_crop manifest): those videos'
    ``full_frame`` features were never centre-cropped, so there is nothing for this
    arm to restore. The expected identical count against ``full_frame`` is therefore
    768, not zero, and the F2 contrast is carried by the 855 non-square videos.

``reencoded``
    The identity variant: the same frames, through the same decode/encode path, with
    no box applied. Both controls pay one extra lossy generation that the untouched
    ``full_frame`` arm does not, so a control that *drops* is confounded with codec
    degradation -- and that is the direction that would decide the retention
    question. This arm prices that generation. It is only worth extracting if a
    control actually drops.

Both box variants use ONE box per video (the union over frames) rather than a per-frame
box: a per-frame box makes the crop pan with the athlete, which injects motion that
is not in the original and would differ between the two variants.

Frame count and fps are preserved exactly, so ``sample_clip_starts`` picks the same
clip starts in every variant and the arms stay paired frame-for-frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.video.variant_geometry import (  # re-exported: this stays the pixel-side entry point
    BOX_VARIANTS,
    CROP_VARIANTS,
    DEFAULT_MARGIN,
    DEFAULT_VISIBILITY,
    LETTERBOX_FILL,
    VARIANTS,
    Box,
    expand_box,
)

__all__ = [
    "BOX_VARIANTS",
    "CROP_VARIANTS",
    "DEFAULT_MARGIN",
    "DEFAULT_VISIBILITY",
    "LETTERBOX_FILL",
    "VARIANTS",
    "Box",
    "apply_variant",
    "build_variant_video",
    "describe_variant",
    "expand_box",
    "fill_box_from_surroundings",
    "letterbox_to_square",
    "person_box_from_pose",
    "read_all_frames",
    "verify_variant_video",
    "write_video",
]


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


def letterbox_to_square(frame: np.ndarray, fill: int = LETTERBOX_FILL) -> np.ndarray:
    """Centre ``frame`` on a square grey canvas without changing its aspect ratio."""
    height, width = frame.shape[:2]
    side = max(height, width)
    if side == height == width:
        return frame
    canvas = np.full((side, side, frame.shape[2]), fill, dtype=frame.dtype)
    top = (side - height) // 2
    left = (side - width) // 2
    canvas[top : top + height, left : left + width] = frame
    return canvas


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


def fill_box_from_surroundings(frame: np.ndarray, box: Box) -> np.ndarray:
    """Replace the box with a horizontal blend of the columns flanking it.

    Per row, the fill ramps from the pixel immediately left of the box to the one
    immediately right, so a wall, a rack upright or a floor line continues across at
    roughly the right colour while nothing body-shaped can survive: the output inside
    the box is a rank-1 function of two columns of the *surrounding* scene.

    Falls back to whichever side exists when the box touches a frame edge, and to a
    vertical blend when it spans the full width. A box covering the entire frame
    leaves nothing to sample and is filled with the frame's mean.
    """
    height, width = frame.shape[:2]
    filled = frame.copy()
    box_width = box.x1 - box.x0
    box_height = box.y1 - box.y0
    if box_width <= 0 or box_height <= 0:
        return filled

    left = frame[box.y0 : box.y1, box.x0 - 1] if box.x0 > 0 else None
    right = frame[box.y0 : box.y1, box.x1] if box.x1 < width else None

    if left is not None or right is not None:
        if left is None:
            left = right
        if right is None:
            right = left
        weights = np.linspace(0.0, 1.0, num=box_width, dtype=np.float32).reshape(1, box_width, 1)
        blended = left[:, None, :].astype(np.float32) * (1.0 - weights) + right[:, None, :].astype(np.float32) * weights
        filled[box.y0 : box.y1, box.x0 : box.x1] = blended.round().astype(np.uint8)
        return filled

    top = frame[box.y0 - 1, box.x0 : box.x1] if box.y0 > 0 else None
    bottom = frame[box.y1, box.x0 : box.x1] if box.y1 < height else None
    if top is not None or bottom is not None:
        if top is None:
            top = bottom
        if bottom is None:
            bottom = top
        weights = np.linspace(0.0, 1.0, num=box_height, dtype=np.float32).reshape(box_height, 1, 1)
        blended = top[None, :, :].astype(np.float32) * (1.0 - weights) + bottom[None, :, :].astype(np.float32) * weights
        filled[box.y0 : box.y1, box.x0 : box.x1] = blended.round().astype(np.uint8)
        return filled

    filled[box.y0 : box.y1, box.x0 : box.x1] = frame.mean(axis=(0, 1)).round().astype(np.uint8)
    return filled


def apply_variant(frames: list[np.ndarray], variant: str, box: Box | None) -> list[np.ndarray]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}.")
    if variant == "reencoded":
        return frames
    if variant == "full_frame_letterbox":
        # Answered BEFORE the box check, not after. This arm is never given a box, so
        # a `box is None -> return frames` fallthrough would hand back the untouched
        # video for EVERY video and produce an arm byte-identical to full_frame --
        # silently, which is the failure mode that already cost this study 51% of one
        # control arm.
        return [letterbox_to_square(frame) for frame in frames]
    if box is None:
        # No person was ever visible. The box variants degrade to the untouched
        # video; the manifest records these so they are reported, not silently counted.
        return frames

    if variant in CROP_VARIANTS:
        cropped = [frame[box.y0 : box.y1, box.x0 : box.x1] for frame in frames]
        if variant == "person_crop_centercrop":
            # No letterbox: the processor's shortest-edge resize plus centre crop then
            # keeps only the middle band of a tall athlete. That truncation IS this
            # arm's manipulation.
            return cropped
        return [letterbox_to_square(frame) for frame in cropped]

    # Filled per frame rather than from one static plate, so lighting changes and
    # camera drift stay consistent with the untouched part of the scene.
    return [fill_box_from_surroundings(frame, box) for frame in frames]


def write_video(frames: list[np.ndarray], output_path: Path, fps: float) -> None:
    """Encode ``frames`` to ``output_path``, atomically.

    Written to a sibling ``.partial.mp4`` and renamed only once the writer has closed.
    A 1600-video build is interrupted often enough that this matters: the resume path
    skips any output that already exists, so a half-written file from a killed run
    would be silently accepted as a finished variant and would extract features from
    a truncated video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    partial_path = output_path.with_suffix(".partial.mp4")
    writer = cv2.VideoWriter(str(partial_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a writer for {partial_path}.")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    partial_path.replace(output_path)


def verify_variant_video(output_path: Path, expected_frames: int) -> int | None:
    """Return the output's frame count when it disagrees with the source, else None.

    Runs over the whole tree after a build. Six of the first 837 person-crop videos
    were 0-frame stubs ("moov atom not found") left by interrupted runs before writes
    became atomic, and the resume path skips whatever already exists -- so without
    this check they would have reached the extractor as finished variants.
    """
    if not output_path.exists():
        return 0
    cap = cv2.VideoCapture(str(output_path))
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return None if frames == expected_frames else frames


def describe_variant(
    video_path: Path,
    pose_path: Path,
    variant: str,
    visibility_threshold: float = DEFAULT_VISIBILITY,
    margin: float = DEFAULT_MARGIN,
) -> dict:
    """The manifest row for one video WITHOUT encoding anything.

    The extractor applies the box in memory, so the box is the only output anything
    downstream consumes; encoding a video is now purely for eyeballing. Splitting
    this out is also what lets every row carry a box even when the video file is
    already on disk -- the omission that fed untransformed videos into a control arm.
    """
    with pose_path.open("r", encoding="utf-8") as f:
        pose = json.load(f)

    metadata = pose.get("metadata", {})
    width = int(metadata.get("width", 0))
    height = int(metadata.get("height", 0))
    landmark_box = person_box_from_pose(pose, visibility_threshold)

    box = landmark_box
    if landmark_box is not None and variant in CROP_VARIANTS:
        box = expand_box(landmark_box, width, height, margin)

    return {
        "video_id": video_path.stem,
        "variant": variant,
        "pose_detected": landmark_box is not None,
        "box": box.as_tuple() if box is not None else None,
        "source_frames": int(metadata.get("total_frames", 0)),
        "fps": float(metadata.get("fps", 30.0)) or 30.0,
        "frame_size": [width, height],
    }


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
    if landmark_box is not None and variant in CROP_VARIANTS:
        box = expand_box(landmark_box, width, height, margin)

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
