"""Frame timing: the display-rotation rule, the fixed-rate sampling clock, and frame-index to
seconds conversion. Never uses frame counts -- only stream time -- because recorded WebM has
no reliable Duration/frame-count and OpenCV reports garbage counts for it.
"""
from __future__ import annotations


def rotation_k(rotation_degrees: float | None) -> int:
    """``np.rot90`` k for a container-reported display rotation, in degrees.

    PyAV reports the rotation as a counter-clockwise angle (an iPhone portrait clip reports
    -90). ``np.rot90(img, k=rotation_k(rotation))`` makes the frame upright the way a browser
    would display it.
    """
    return int(round((rotation_degrees or 0) / 90)) % 4


class SampleClock:
    """Decides which decoded frames to keep, at a fixed rate, by stream time.

    Keeps the first frame whose stream time is at or after each tick (0, 1/fps, 2/fps, ...).
    Ticks accumulate rather than reset to the nearest multiple, matching the reference Phase 0
    sampler: a frame far past a tick still only advances the clock by one step.
    """

    def __init__(self, fps: float = 10.0) -> None:
        if fps <= 0:
            raise ValueError(f"SampleClock: fps must be positive, got {fps!r}")
        self.fps = float(fps)
        self._next_t = 0.0

    def keep(self, t: float) -> bool:
        """Whether the frame at stream time ``t`` (seconds) should be kept.

        On a keep, every tick at or before ``t`` is consumed (not just one step), so a burst
        of frames after a gap in the stream still yields one kept frame per 1/fps tick rather
        than one kept frame per burst member.
        """
        if t + 1e-6 < self._next_t:
            return False
        while self._next_t <= t + 1e-6:
            self._next_t += 1.0 / self.fps
        return True


def frame_to_seconds(frame_index: int, pose_fps: float | None) -> float | None:
    """Convert an analysis frame index to seconds, using that analysis's own fps.

    Returns ``None`` when ``pose_fps`` is missing or not positive -- the analysis frame index
    can't be mapped to a time without it, and callers must not assume a fixed rate (in-app
    recordings use a virtual 30 Hz timeline; the legacy path stored the video's native fps).
    """
    if pose_fps is None or pose_fps <= 0:
        return None
    return frame_index / pose_fps
