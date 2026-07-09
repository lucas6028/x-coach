"""Live-upload analysis endpoint."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.app import config
from backend.app.auth import CurrentUser, get_optional_user
from backend.app.services import analysis, object_store, store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analyze"])

_ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _clip_duration_s(result: dict) -> float:
    """Playback duration of the analyzed clip, derived from the pose overlay metadata.

    The pose block already carries ``fps`` and one entry per frame, so this needs no extra probe of
    the video. Returns ``0.0`` when the block is absent or fps is unknown (can't judge — allow it).
    """
    pose = result.get("pose") or {}
    fps = float(pose.get("fps") or 0.0)
    frames = pose.get("frames") or []
    return len(frames) / fps if fps > 0 else 0.0


def _enforce_quota(token: str, user_id: str, incoming_bytes: int) -> None:
    """Reject the upload with 507 if it would push the user past their storage quota.

    Runs before the (expensive) pipeline so an over-quota user doesn't burn compute. A failure of
    the usage lookup itself is non-fatal (fail-open): the quota is a demo guardrail, not a security
    boundary, and the codebase already treats persistence as best-effort.
    """
    try:
        usage = store.get_usage(token=token)
    except Exception:  # noqa: BLE001 — a usage-check hiccup must not block a legitimate upload
        logger.exception("Usage lookup failed (user=%s); allowing upload", user_id)
        return
    over_count = usage["count"] >= config.USER_VIDEO_QUOTA_COUNT
    over_bytes = usage["bytes"] + incoming_bytes > config.USER_STORAGE_QUOTA_BYTES
    if over_count or over_bytes:
        mb = config.USER_STORAGE_QUOTA_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=507,
            detail=(
                f"Storage quota reached ({config.USER_VIDEO_QUOTA_COUNT} videos / {mb} MB). "
                "Delete older analyses to free space."
            ),
        )

# Bound how many uploads run the (blocking, CPU/RAM-heavy) pipeline at once. Excess requests
# await a slot here instead of spawning unbounded worker threads. This is a single-process
# stop-gap; durable queueing and backpressure move to Celery + Redis later.
_ANALYSIS_SEMAPHORE = asyncio.Semaphore(config.MAX_CONCURRENT_ANALYSES)


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    user: CurrentUser | None = Depends(get_optional_user),
) -> dict:
    """Accept a squat video, extract pose, detect faults, and return the full analysis.

    The pose pipeline is synchronous and CPU-bound, so the disk write and the analysis run in a
    worker thread via ``run_in_threadpool`` — this keeps the event loop responsive instead of
    freezing every other request for the duration of one analysis. A concurrency semaphore caps
    how many analyses run at once so concurrent uploads queue rather than exhaust the machine.

    Auth is optional: an anonymous caller gets the analysis but nothing is saved (public demo);
    an authenticated caller also has the result persisted to their history and gets an
    ``analysis_id`` back. Persistence is best-effort — a DB hiccup is logged but never discards
    a completed (and expensive) analysis.

    The response contract is otherwise unchanged: it matches the library analysis shape
    (metadata/view/quality/detections/retrievals/pose) so the frontend renders uploads and
    library clips identically.
    """
    suffix = Path(file.filename or "").suffix.lower() or ".mp4"
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    size = len(data)
    if size > config.MAX_UPLOAD_BYTES:
        del data
        mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File too large (max {mb} MB).")

    # Per-user storage quota — signed-in only. An anonymous demo upload isn't persisted, so it
    # consumes no quota; the check runs before the pipeline so an over-quota user burns no compute.
    if user is not None:
        await run_in_threadpool(_enforce_quota, user.token, user.id, size)

    video_id, saved_path = await run_in_threadpool(analysis.save_upload, data, suffix=suffix)
    del data  # bytes are now on disk; don't pin the whole video in RAM while queued for a slot.
    try:
        async with _ANALYSIS_SEMAPHORE:
            result = await run_in_threadpool(
                analysis.analyze_video_file, saved_path, video_id=video_id
            )
    except RuntimeError as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Duration backstop: a low-bitrate clip can slip past the byte cap while still being far too
    # long. Enforced from the analysis metadata (no extra probe) before anything is stored.
    if _clip_duration_s(result) > config.MAX_UPLOAD_DURATION_S:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422, detail=f"Video too long (max {config.MAX_UPLOAD_DURATION_S}s)."
        )

    if user is not None:
        # Push the durable copy to object storage (when configured), then drop the local temp so
        # prod doesn't rely on the ephemeral container disk. A storage failure is non-fatal: keep the
        # local copy and persist without an R2 key rather than lose a completed, expensive analysis.
        storage_key: str | None = None
        try:
            if object_store.is_configured():
                storage_key = await run_in_threadpool(
                    object_store.upload_video_file, video_id, saved_path, suffix
                )
                saved_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 — object-storage hiccup must not discard the analysis
            logger.exception("R2 upload failed (user=%s video=%s)", user.id, video_id)
        try:
            result["analysis_id"] = await run_in_threadpool(
                store.persist_analysis,
                token=user.token,
                user_id=user.id,
                video_id=video_id,
                source="upload",
                result=result,
                filename=file.filename,
                size_bytes=size,
                storage_key=storage_key,
            )
        except Exception:  # noqa: BLE001 — never lose a completed analysis to a storage error
            logger.exception("Failed to persist analysis (user=%s video=%s)", user.id, video_id)
            result["analysis_id"] = None
    return result
