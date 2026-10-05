"""One pass of the worker loop: heartbeat, claim, fetch, decode, infer, pick key frames,
render, upload, complete -- entirely over the ``interfaces`` protocols, so it can be tested
with plain fakes and never imports torch/PyAV/pyrender/boto3/supabase.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

from src.nlf_worker import geometry, result
from src.nlf_worker.interfaces import Db, Decoder, Fetcher, Model, ObjectStore, Renderer
from src.nlf_worker.keyframes import select_key_frames

ANGLES = 24
TILE_PX = 384


def _keep(value: Any) -> np.ndarray | None:
    """A detached float32 copy of one per-frame model output (never a view that could keep a
    whole batch's buffer alive), or ``None`` for an undetected frame."""
    return None if value is None else np.array(value, dtype=np.float32)


def _decode_and_infer(
    decoder: Decoder, model: Model, path: str, sample_fps: float, batch: int
) -> tuple[list[float], list, list, list]:
    """Streams decode -> infer: each batch goes to the model as soon as it fills and its RGB
    frames are dropped right after, so at most ``batch`` frames are ever held and peak memory
    does not grow with clip length x frame size. Only the small per-frame outputs are kept."""
    times: list[float] = []
    joints: list[np.ndarray | None] = []
    uncertainties: list[np.ndarray | None] = []
    vertices: list[np.ndarray | None] = []
    pending: list[np.ndarray] = []

    def flush() -> None:
        outputs = model.infer(pending)
        if len(outputs) != len(pending):
            raise RuntimeError(f"model returned {len(outputs)} results for {len(pending)} frames")
        pending.clear()
        for r in outputs:
            joints.append(_keep(r.get("joints3d")))
            uncertainties.append(_keep(r.get("joint_uncertainties")))
            vertices.append(_keep(r.get("vertices3d")))

    for t, img in decoder.iterate(path, sample_fps):
        times.append(float(t))
        pending.append(img)
        del img  # the batch list is now the only reference
        if len(pending) >= batch:
            flush()
    if pending:
        flush()
    return times, joints, uncertainties, vertices


def shared_framing(
    meshes: list[tuple[np.ndarray | None, np.ndarray]], basis: np.ndarray
) -> dict[str, float] | None:
    """One orthographic framing (``cy``, ``mag``) for every key frame of a job, from the union of
    their body-frame extents, so the body is drawn at the same scale in every strip. ``meshes``
    is ``(vertices_mm, joints_mm)`` per key frame in raw camera mm; entries without a mesh are
    ignored. ``None`` when there is nothing to frame."""
    bodies = [
        geometry.to_body_frame(v, j[geometry.PELVIS], basis) for v, j in meshes if v is not None
    ]
    return geometry.turntable_framing(bodies) if bodies else None


def run_one(
    db: Db,
    store: ObjectStore,
    fetcher: Fetcher,
    decoder: Decoder,
    model: Model,
    renderer: Renderer,
    *,
    sample_fps: float = 10.0,
    batch: int = 8,
    tmp_dir: str | None = None,
    model_sha256: str,
    torch_version: str,
) -> str:
    """Runs one job to completion (or failure). Returns 'idle' | 'done' | 'orphaned' | 'failed'.

    Claims at most one job. A missing detection everywhere, or any exception past that point,
    fails the job and best-effort deletes anything already uploaded for this attempt. The
    locally downloaded source file is always removed, on every path.
    """
    db.heartbeat()

    job = db.claim()
    if job is None:
        return "idle"

    prefix = result.result_prefix(job["storage_key"], job["attempts"])
    dest_dir = tmp_dir or tempfile.gettempdir()
    local_path: str | None = None

    try:
        local_path = fetcher.download(f"{job['storage_key']}/source", dest_dir)

        times, joints3d_raw, uncertainties, vertices3d = _decode_and_infer(
            decoder, model, local_path, sample_fps, batch
        )
        if not times:
            raise RuntimeError("no frames decoded from the source video")
        sample_times = np.array(times, dtype=float)

        if not any(j is not None for j in joints3d_raw):
            raise RuntimeError("no frame had a detection")

        joints_arr = np.stack(
            [j if j is not None else np.full((24, 3), np.nan) for j in joints3d_raw]
        )
        knee_angles = geometry.knee_angle(joints_arr)
        up_axis = geometry.estimate_up_axis(joints_arr, knee_angles)
        basis = geometry.body_basis(up_axis)

        key_frames = select_key_frames(
            job.get("movement"),
            sample_times,
            knee_angles,
            job.get("rep_segments"),
            job.get("detections"),
            job.get("pose_fps"),
        )

        framing = shared_framing(
            [(vertices3d[kf["sample_index"]], joints_arr[kf["sample_index"]]) for kf in key_frames],
            basis,
        )
        for i, kf in enumerate(key_frames):
            idx = kf["sample_index"]
            # Raw camera-coordinate mm, matching the Renderer protocol -- the renderer does the
            # pelvis-relative/rotate/metres conversion itself (see geometry.to_body_frame), the
            # same way phase0_6_turntable.py does it inline. Doing that conversion here too would
            # apply it twice. Only the job-wide framing (a view centre and scale) is decided here.
            strip_bytes = renderer.render_strip(vertices3d[idx], joints_arr[idx], basis, framing=framing)
            store.put(result.strip_key(prefix, i), strip_bytes, content_type="image/webp")

        pose3d = result.build_pose3d(
            model_sha256=model_sha256,
            torch_version=torch_version,
            sample_fps=sample_fps,
            sample_times_s=list(sample_times),
            joints3d=joints3d_raw,
            joint_uncertainties=uncertainties,
            up_axis=up_axis,
        )
        pose3d_key = f"{prefix}/pose3d.json.gz"
        store.put(pose3d_key, result.encode_pose3d(pose3d), content_type="application/json")

        meta = result.build_meta(
            analysis_id=job.get("analysis_id"),
            angles=ANGLES,
            tile_px=TILE_PX,
            prefix=prefix,
            key_frames=key_frames,
        )

        ok = db.complete(job["job_id"], pose3d_key, meta)
        if not ok:
            store.delete_prefix(prefix)
            return "orphaned"
        return "done"

    except Exception as exc:  # noqa: BLE001 -- any failure here fails the job, never crashes the loop
        try:
            store.delete_prefix(prefix)
        except Exception:
            pass
        db.fail(job["job_id"], str(exc)[:500])
        return "failed"

    finally:
        if local_path is not None:
            try:
                os.remove(local_path)
            except OSError:
                pass
