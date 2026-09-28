"""Typing-only seams between the pure logic in this package and the heavy runtime adapters
(torch, PyAV, pyrender, the Supabase client, boto3) that a later task provides under
``src/nlf_worker/runtime/``. Nothing here imports any of them; ``loop.run_one`` is written
entirely against these protocols, so tests exercise it with plain fakes.
"""
from __future__ import annotations

from typing import Any, Iterable, Protocol

import numpy as np


class Db(Protocol):
    """The worker's own view of the job queue, backed by the ``nlf_*`` Supabase RPCs."""

    def claim(self) -> dict[str, Any] | None:
        """Claim the oldest queued job, or ``None`` if there isn't one.

        On a hit, returns a dict with (at least): ``job_id``, ``owner_id``, ``video_id``,
        ``attempts`` (already incremented), ``storage_key``, ``analysis_id``, ``movement``,
        ``pose_fps``, ``rep_segments``, ``detections``.
        """
        ...

    def complete(self, job_id: str, result_key: str, meta: dict[str, Any]) -> bool:
        """Mark the job done. Returns ``False`` if it's gone (the user deleted the video mid
        job, or it was no longer claimed) -- the caller must then delete what it uploaded."""
        ...

    def fail(self, job_id: str, error: str) -> None: ...

    def heartbeat(self) -> None: ...


class ObjectStore(Protocol):
    """The R2 surface the worker needs, mirroring ``backend/app/services/storage.ObjectStore``."""

    def put(self, key: str, data: bytes, *, content_type: str) -> None: ...

    def delete_prefix(self, prefix: str) -> None: ...


class Decoder(Protocol):
    """Decodes a local video file into already-upright, already-sampled RGB frames."""

    def iterate(self, path: str, sample_fps: float) -> Iterable[tuple[float, np.ndarray]]:
        """Yields ``(stream_time_s, rgb_frame)`` in stream order, at ``sample_fps``, with any
        container display-rotation already applied."""
        ...


class Model(Protocol):
    """The NLF body model, run in batches for throughput."""

    def infer(self, frames: list[np.ndarray]) -> list[dict[str, Any]]:
        """One result dict per input frame, each with ``joints3d`` ((24, 3) or ``None``),
        ``joint_uncertainties`` ((24,) or ``None``) and ``vertices3d`` ((6890, 3) or ``None``)
        -- all ``None`` together when nobody was detected in that frame."""
        ...


class Renderer(Protocol):
    """Renders one turntable strip for a single key frame's mesh."""

    def render_strip(
        self,
        vertices_mm: np.ndarray | None,
        joints_mm: np.ndarray,
        basis: np.ndarray,
        framing: dict[str, float] | None = None,
    ) -> bytes:
        """WebP-encoded bytes for a strip of tiles around the body, angle 0 = the camera's own
        view. ``vertices_mm``/``joints_mm`` are raw camera-coordinate mm for this sample (not
        yet pelvis-relative or rotated); ``basis`` is the ``geometry.body_basis`` matrix for
        this clip, for the renderer to apply itself (see ``geometry.to_body_frame``).

        ``framing`` is the job-wide ``{cy, mag}`` view (``loop.shared_framing``) so every key
        frame of a job is drawn at the same scale; one strip alone cannot know the others'
        extents. ``None`` frames this mesh on its own."""
        ...


class Fetcher(Protocol):
    """Downloads the source video the worker needs to process."""

    def download(self, key: str, dest_dir: str) -> str:
        """Downloads the object at ``key`` (e.g. ``f"{storage_key}/source"``) into ``dest_dir``
        and returns the local file path."""
        ...
