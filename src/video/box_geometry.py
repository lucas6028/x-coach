"""The zero-parameter "where is the person and how big" descriptor.

Twelve numbers per sample, no pixels: the person's bounding box, the frame it sits
in, and how long the clip is. It exists as a control -- anything a video model scores
above this is what the pixels actually bought.

On Fitness-AQA squats it turned out not to be a formality. Scored on the same repeated
splits as everything else it reaches 0.6120 balanced accuracy against ``full_frame``'s
0.6706, i.e. 65.6% of the above-chance signal, and the entire gym scene adds only
+0.0119 (CI [-0.0122, +0.0357]) on top of it. See
``notes/videomae_b1_repeated_splits_results.md``.

The definition was originally written ad hoc and survived only as .npz files, which is
why it now lives here: ``tests/test_box_geometry.py`` regenerates the archived
Fitness-AQA features from pose JSON and requires a bit-for-bit match, so the definition
cannot drift away from the numbers already reported.

Two details that matter for reuse:

* The box is the RAW landmark box -- the union over frames of landmarks clearing the
  visibility threshold -- NOT the 15%-expanded crop box. It is the same rectangle
  ``background_only`` paints over, which is what makes the two arms comparable.
* Four of the twelve (frame width, height, aspect, length) are pure recording format
  and carry almost nothing on their own: that subset alone scores 0.5080, i.e. chance.
  They are kept so the box terms are interpretable relative to the frame, not because
  they contribute.
"""

from __future__ import annotations

import numpy as np

from src.video.variant_geometry import Box

#: Order is fixed and load-bearing -- the archived feature files use it.
FEATURE_NAMES = (
    "x0_norm",
    "y0_norm",
    "x1_norm",
    "y1_norm",
    "width_norm",
    "height_norm",
    "area_norm",
    "box_aspect",
    "frame_width",
    "frame_height",
    "frame_aspect",
    "n_frames",
)
FEATURE_DIM = len(FEATURE_NAMES)


def box_geometry_feature(box: Box, frame_width: int, frame_height: int, n_frames: int) -> np.ndarray:
    """The 12-number descriptor for one sample.

    ``box_aspect`` is width/height in PIXELS, not in normalised units, so it is the
    athlete's real shape rather than the shape distorted by a non-square frame.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"Frame size must be positive, got {frame_width}x{frame_height}.")
    box_w = box.x1 - box.x0
    box_h = box.y1 - box.y0
    if box_w <= 0 or box_h <= 0:
        raise ValueError(f"Box must have positive extent, got {box.as_tuple()}.")

    width_norm = box_w / frame_width
    height_norm = box_h / frame_height
    return np.asarray(
        [
            box.x0 / frame_width,
            box.y0 / frame_height,
            box.x1 / frame_width,
            box.y1 / frame_height,
            width_norm,
            height_norm,
            width_norm * height_norm,
            box_w / box_h,
            frame_width,
            frame_height,
            frame_width / frame_height,
            n_frames,
        ],
        dtype=np.float32,
    )


def box_from_points(
    xs: np.ndarray,
    ys: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> Box | None:
    """The clamped integer bounding box of a point cloud, or ``None`` if empty.

    Mirrors ``person_box_from_pose``'s rounding exactly -- floor the minimum, ceil the
    maximum, clamp to the frame -- so a box derived from a dataset's own 2D skeleton is
    the same object as one derived from MediaPipe landmarks.
    """
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = max(int(np.floor(float(xs.min()))), 0)
    y0 = max(int(np.floor(float(ys.min()))), 0)
    x1 = min(int(np.ceil(float(xs.max()))), frame_width)
    y1 = min(int(np.ceil(float(ys.max()))), frame_height)
    if x1 <= x0 or y1 <= y0:
        return None
    return Box(x0, y0, x1, y1)
