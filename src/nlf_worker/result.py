"""Result-object naming and serialisation: the ``nlf.v1.a{attempts}`` prefix, the turntable
strip keys under it, and the ``pose3d.json.gz`` / job-``meta`` payloads, per the fixed contract
in docs/superpowers/plans/2026-09-27-nlf-home-gpu-worker.md.
"""
from __future__ import annotations

import gzip
import json
import math
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def result_prefix(storage_key: str, attempts: int) -> str:
    """Where this attempt's objects live, e.g. ``uploads/{owner}/{video}/nlf.v1.a3``."""
    return f"{storage_key}/nlf.v1.a{attempts}"


def strip_key(prefix: str, index: int) -> str:
    """The turntable strip object key for the ``index``-th key frame (zero-padded, 2 digits)."""
    return f"{prefix}/turntable/k{index:02d}.webp"


def _clean(value: Any) -> Any:
    """Recursively convert numpy arrays/scalars to plain JSON-safe Python, NaN -> None."""
    if value is None:
        return None
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) else f
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def build_pose3d(
    *,
    model_sha256: str,
    torch_version: str,
    sample_fps: float,
    sample_times_s: list[float],
    joints3d: list[np.ndarray | None],
    joint_uncertainties: list[np.ndarray | None],
    up_axis: np.ndarray,
) -> dict:
    """The ``pose3d.json.gz`` payload (before gzip/JSON encoding). ``joints3d`` /
    ``joint_uncertainties`` are one entry per sample: a (24, 3) / (24,) array, or ``None`` for
    an undetected sample. NaN values (there should be none inside a detected sample) become
    ``null`` too, since JSON has no NaN literal.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "model_sha256": model_sha256,
        "torch": torch_version,
        "sample_fps": float(sample_fps),
        "sample_times_s": _clean(sample_times_s),
        "joints3d": [_clean(j) for j in joints3d],
        "joint_uncertainties": [_clean(u) for u in joint_uncertainties],
        "up_axis": _clean(up_axis),
    }


def encode_pose3d(pose3d: dict) -> bytes:
    """Gzip-compressed JSON bytes for a ``build_pose3d`` payload."""
    return gzip.compress(json.dumps(pose3d).encode("utf-8"))


def decode_pose3d(data: bytes) -> dict:
    """Inverse of ``encode_pose3d`` (used by tests to round-trip)."""
    return json.loads(gzip.decompress(data).decode("utf-8"))


def build_meta(
    *,
    analysis_id: str,
    angles: int,
    tile_px: int,
    prefix: str,
    key_frames: list[dict],
) -> dict:
    """The ``nlf_jobs.meta`` payload passed to ``nlf_complete_job``.

    ``key_frames`` is the output of ``keyframes.select_key_frames`` (each has ``t_s``,
    ``kind``, ``fault_ids``, ``rep_index``, ``sample_index``); this adds each one's
    ``strip_key`` and drops the sample-index bookkeeping field, which is internal to the
    worker only.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "angles": angles,
        "tile_px": tile_px,
        "prefix": prefix,
        "key_frames": [
            {
                "t_s": kf["t_s"],
                "kind": kf["kind"],
                "fault_ids": list(kf["fault_ids"]),
                "rep_index": kf.get("rep_index"),
                "strip_key": strip_key(prefix, i),
            }
            for i, kf in enumerate(key_frames)
        ],
    }
