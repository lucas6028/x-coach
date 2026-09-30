"""The pure half of ``scripts/nlf_worker/run_worker.py``: argument parsing, the poll loop, and
the offline ``--dry-run`` stand-ins for the network adapters (a one-job in-memory ``Db``, a
local-directory ``ObjectStore``, a ``Fetcher`` that copies a local clip).

Nothing here imports torch/PyAV/pyrender/boto3/supabase or anything under ``runtime/``; the
heavy adapters are wired together in ``src/nlf_worker/runtime/app.py``.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from src.nlf_worker import config, geometry

# The dry-run's synthetic analysis uses the browser path's fixed virtual timeline
# (``CANONICAL_FPS = 30`` in frontend/src/lib/repSpans.ts), so --segment seconds map to frames
# the same way a real in-app analysis's reps.segments do.
DRY_RUN_POSE_FPS = 30.0

# After these outcomes there may be more queued work, so the loop polls again at once.
POLL_AGAIN_NOW = frozenset({"done", "failed", "orphaned"})


# ---------------------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_worker.py",
        description=(
            "NLF 3D-view worker. Default: poll Supabase for jobs every NLF_POLL_SECONDS. "
            "--once: run one poll and print the outcome. --dry-run CLIP: process a local clip "
            "end to end with no network and write the results under --out."
        ),
    )
    parser.add_argument(
        "--env",
        default=config.DEFAULT_PATH,
        help="worker.env path, relative to the repo root (default: %(default)s)",
    )
    parser.add_argument("--once", action="store_true", help="run a single poll, print its outcome, exit")
    parser.add_argument("--dry-run", metavar="CLIP", help="process CLIP offline (no Supabase, no R2)")
    parser.add_argument("--out", metavar="DIR", help="dry-run output directory (required with --dry-run)")
    parser.add_argument("--movement", default="squat", help="dry-run movement (default: %(default)s)")
    parser.add_argument(
        "--segment",
        nargs=2,
        type=float,
        action="append",
        default=[],
        metavar=("START_S", "END_S"),
        help="dry-run rep segment in seconds; repeat for several reps",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        if not args.out:
            parser.error("--dry-run requires --out DIR")
        try:
            segments_to_rep_segments(args.segment)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.out or args.segment:
        parser.error("--out and --segment only apply to --dry-run")
    return args


# ---------------------------------------------------------------------------------------
# the dry-run's synthetic job and stand-in adapters
# ---------------------------------------------------------------------------------------
def segments_to_rep_segments(
    segments: list[tuple[float, float]] | list[list[float]],
    pose_fps: float = DRY_RUN_POSE_FPS,
) -> list[dict[str, int]]:
    """``[(start_s, end_s), ...]`` -> ``[{index, start_frame, end_frame}, ...]`` at ``pose_fps``,
    the shape of an analysis's ``reps.segments`` as ``nlf_claim_job`` returns them."""
    out = []
    for index, (start_s, end_s) in enumerate(segments):
        if not (math.isfinite(start_s) and math.isfinite(end_s)) or start_s < 0 or end_s <= start_s:
            raise ValueError(f"--segment {start_s} {end_s}: need 0 <= START_S < END_S")
        out.append(
            {
                "index": index,
                "start_frame": int(round(start_s * pose_fps)),
                "end_frame": int(round(end_s * pose_fps)),
            }
        )
    return out


def dry_run_job(
    clip: str | Path, movement: str, segments: list[tuple[float, float]] | list[list[float]]
) -> dict[str, Any]:
    """The one job the dry-run's ``Db`` hands out, shaped like an ``nlf_claim_job`` row."""
    stem = Path(clip).stem
    return {
        "job_id": "dryrun",
        "owner_id": "dryrun",
        "video_id": stem,
        "attempts": 1,
        "storage_key": f"dryrun/{stem}",
        "analysis_id": None,
        "movement": movement,
        "pose_fps": DRY_RUN_POSE_FPS,
        "rep_segments": segments_to_rep_segments(segments),
        "detections": [],
    }


class DryRunDb:
    """In-memory ``Db``: hands out ``job`` once, then reports an empty queue."""

    def __init__(self, job: dict[str, Any]) -> None:
        self._job: dict[str, Any] | None = job
        self.heartbeats = 0
        self.completed: list[tuple[str, str, dict[str, Any]]] = []
        self.failed: list[tuple[str, str]] = []

    def heartbeat(self) -> None:
        self.heartbeats += 1

    def claim(self) -> dict[str, Any] | None:
        job, self._job = self._job, None
        return job

    def complete(self, job_id: str, result_key: str, meta: dict[str, Any]) -> bool:
        self.completed.append((job_id, result_key, meta))
        return True

    def fail(self, job_id: str, error: str) -> None:
        self.failed.append((job_id, error))


class LocalDirStore:
    """``ObjectStore`` over a local directory: object ``key`` lives at ``root/key``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.content_types: dict[str, str] = {}

    def path_for(self, key: str) -> Path:
        parts = key.split("/")
        if not key or key.startswith("/") or ".." in parts or any(c in key for c in "\\:\x00"):
            raise ValueError(f"unsafe object key: {key!r}")
        path = (self.root / key).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"unsafe object key: {key!r}")
        return path

    def put(self, key: str, data: bytes, *, content_type: str) -> None:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.content_types[key] = content_type

    def get(self, key: str) -> bytes:
        return self.path_for(key).read_bytes()

    def delete_prefix(self, prefix: str) -> None:
        prefix = prefix.rstrip("/")
        path = self.path_for(prefix)
        if path.is_dir():
            shutil.rmtree(path)
        for key in [k for k in self.content_types if k.startswith(prefix + "/")]:
            del self.content_types[key]


class CopyFetcher:
    """``Fetcher`` for a local clip. Always hands back a fresh COPY: ``run_one`` deletes whatever
    path the fetcher returns, and that must never be the caller's original file."""

    def __init__(self, clip: str | Path) -> None:
        self.clip = Path(clip)
        self.keys: list[str] = []

    def download(self, key: str, dest_dir: str) -> str:
        self.keys.append(key)
        os.makedirs(dest_dir, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="nlf_dryrun_", suffix=self.clip.suffix, dir=dest_dir)
        os.close(fd)
        shutil.copyfile(self.clip, path)
        return path


# ---------------------------------------------------------------------------------------
# thin wrappers: a job log for the real worker, phase timers for the dry-run report
# ---------------------------------------------------------------------------------------
class LoggingDb:
    """Wraps a ``Db`` and logs each claim/complete/fail (job ids and errors only -- never
    credentials or the job's analysis content)."""

    def __init__(self, inner: Any, log: Callable[[str], None] = print) -> None:
        self._inner = inner
        self._log = log

    def heartbeat(self) -> None:
        self._inner.heartbeat()

    def claim(self) -> dict[str, Any] | None:
        job = self._inner.claim()
        if job is not None:
            self._log(f"claimed job {job.get('job_id')} (attempt {job.get('attempts')}, {job.get('movement')})")
        return job

    def complete(self, job_id: str, result_key: str, meta: dict[str, Any]) -> bool:
        ok = self._inner.complete(job_id, result_key, meta)
        n = len(meta.get("key_frames", []))
        self._log(f"job {job_id}: " + (f"done, {n} key frame(s)" if ok else "gone before completion"))
        return ok

    def fail(self, job_id: str, error: str) -> None:
        self._log(f"job {job_id} failed: {error}")
        self._inner.fail(job_id, error)


class TimedDecoder:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.seconds = 0.0
        self.frames = 0

    def iterate(self, path: str, sample_fps: float) -> Iterator[tuple[float, np.ndarray]]:
        """Passes samples straight through (never buffers them), timing only the decoder's own
        work between yields -- not the inference the consumer runs in between."""
        it = iter(self._inner.iterate(path, sample_fps))
        while True:
            t0 = time.perf_counter()
            try:
                sample = next(it)
            except StopIteration:
                self.seconds += time.perf_counter() - t0
                return
            self.seconds += time.perf_counter() - t0
            self.frames += 1
            yield sample
            del sample


class TimedModel:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.seconds = 0.0
        self.frames = 0
        self.detected = 0

    def infer(self, frames: list[np.ndarray]) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        out = self._inner.infer(frames)
        self.seconds += time.perf_counter() - t0
        self.frames += len(frames)
        self.detected += sum(r.get("joints3d") is not None for r in out)
        return out


class TimedRenderer:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.seconds = 0.0
        self.strips = 0

    def render_strip(
        self, vertices_mm: Any, joints_mm: np.ndarray, basis: np.ndarray, framing: Any = None
    ) -> bytes:
        t0 = time.perf_counter()
        data = self._inner.render_strip(vertices_mm, joints_mm, basis, framing=framing)
        self.seconds += time.perf_counter() - t0
        self.strips += 1
        return data


# ---------------------------------------------------------------------------------------
# key-frame readout for the dry-run report
# ---------------------------------------------------------------------------------------
def describe_key_frames(pose3d: dict[str, Any], meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Per key frame: time, kind, the 3D knee angle there, and the hip joint-centre height minus
    the knee joint-centre height in the body frame (cm; negative = hip below knee). Read back
    from the uploaded ``pose3d`` payload, so it checks what was actually written."""
    times = np.asarray(pose3d["sample_times_s"], dtype=float)
    joints = np.stack(
        [
            np.asarray(j, dtype=float) if j is not None else np.full((24, 3), np.nan)
            for j in pose3d["joints3d"]
        ]
    )
    knee = np.asarray(geometry.knee_angle(joints), dtype=float)
    basis = geometry.body_basis(np.asarray(pose3d["up_axis"], dtype=float))
    out = []
    for i, kf in enumerate(meta["key_frames"]):
        idx = int(np.argmin(np.abs(times - kf["t_s"])))
        body = geometry.to_body_frame(joints[idx], joints[idx][geometry.PELVIS], basis)
        hip_y = (body[geometry.L_HIP, 1] + body[geometry.R_HIP, 1]) / 2.0
        knee_y = (body[geometry.L_KNEE, 1] + body[geometry.R_KNEE, 1]) / 2.0
        out.append(
            {
                "index": i,
                "t_s": float(kf["t_s"]),
                "kind": kf["kind"],
                "rep_index": kf.get("rep_index"),
                "knee_deg": float(knee[idx]),
                "hip_minus_knee_cm": float((hip_y - knee_y) * 100.0),
            }
        )
    return out


# ---------------------------------------------------------------------------------------
# the poll loop
# ---------------------------------------------------------------------------------------
def serve(
    run_once: Callable[[], str],
    poll_seconds: float,
    *,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
    max_polls: int | None = None,
) -> None:
    """Call ``run_once`` forever (or ``max_polls`` times). After a job ends ('done', 'failed',
    'orphaned') poll again at once; otherwise sleep ``poll_seconds``.

    Any exception from a poll (a network blip, an expired session the Db couldn't renew) is
    logged and waited out like an idle poll, never raised: ``Db.heartbeat``/``Db.claim`` run
    outside ``run_one``'s own error handling. ``KeyboardInterrupt`` is not an ``Exception`` and
    propagates, so Ctrl+C stops the loop. Consecutive idle polls are logged once.
    """
    polls = 0
    last = None
    while max_polls is None or polls < max_polls:
        polls += 1
        try:
            outcome = run_once()
        except Exception as exc:  # noqa: BLE001 -- the loop must outlive any single poll
            outcome = "error"
            log(f"poll error: {type(exc).__name__}: {exc}")
        else:
            if outcome != "idle" or last != "idle":
                log(f"poll: {outcome}" + (f" (next poll in {poll_seconds:g}s)" if outcome == "idle" else ""))
        last = outcome
        if outcome in POLL_AGAIN_NOW:
            continue
        sleep(poll_seconds)
