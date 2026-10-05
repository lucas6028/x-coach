"""PyAV decoder -- ``Decoder`` for ``loop.run_one``.

The Phase 0 decode path (``data/models/nlf/phase0_3_clip.py``): decode sequentially with frame
threading, keep frames by STREAM TIME at a fixed rate (never by frame count or seeking -- recorded
WebM has no Duration element and OpenCV reports a garbage frame count for it), apply the
container's display rotation the way a browser does (PyAV does not), and downscale so the short
side is at most 720 px (measured: 100% detection at ~120 ms/frame on a phone WebM and a rotated
4K iPhone MOV).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import av
import cv2
import numpy as np

from src.nlf_worker.timeline import SampleClock, rotation_k


def fit_short_side(img: np.ndarray, short_side: int) -> np.ndarray:
    """Downscale ``img`` (INTER_AREA) so its short side is ``short_side``; a frame already at or
    below that is returned unchanged (upscaling adds no detail for the detector)."""
    h, w = img.shape[:2]
    short = min(h, w)
    if short <= short_side:
        return img
    scale = short_side / short
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


class PyAvDecoder:
    def __init__(self, short_side: int = 720) -> None:
        self.short_side = short_side

    def iterate(self, path: str | Path, sample_fps: float) -> Iterator[tuple[float, np.ndarray]]:
        """Yields ``(stream_time_s, upright RGB uint8 frame)`` at ``sample_fps``."""
        clock = SampleClock(sample_fps)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                t = frame.time
                if t is None or not clock.keep(t):
                    continue
                img = frame.to_ndarray(format="rgb24")
                k = rotation_k(getattr(frame, "rotation", 0))
                if k:
                    img = np.ascontiguousarray(np.rot90(img, k=k))
                yield float(t), np.ascontiguousarray(fit_short_side(img, self.short_side))
