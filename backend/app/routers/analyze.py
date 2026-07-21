"""Live-upload analysis endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.app import config, settings
from backend.app.auth import CurrentUser, get_optional_user
from backend.app.services import analysis, store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analyze"])

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
    # ``allowed_upload_suffixes`` reads the admin overrides, which can do a synchronous Supabase
    # round-trip on a cold cache — run it in a threadpool so it never blocks the event loop.
    allowed = await run_in_threadpool(settings.allowed_upload_suffixes)
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    video_id, saved_path = await run_in_threadpool(analysis.save_upload, data, suffix=suffix)
    del data  # bytes are now on disk; don't pin the whole video in RAM while queued for a slot.
    try:
        async with _ANALYSIS_SEMAPHORE:
            result = await run_in_threadpool(
                analysis.analyze_video_file, saved_path, video_id=video_id
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if user is not None:
        try:
            result["analysis_id"] = await run_in_threadpool(
                store.persist_analysis,
                token=user.token,
                user_id=user.id,
                video_id=video_id,
                source="upload",
                result=result,
                filename=file.filename,
            )
        except Exception:  # noqa: BLE001 — never lose a completed analysis to a storage error
            logger.exception("Failed to persist analysis (user=%s video=%s)", user.id, video_id)
            result["analysis_id"] = None
    return result


@router.post("/analyze/pose")
async def analyze_pose(
    movement: str = Form(...),
    pose: str = Form(...),
    file: UploadFile = File(...),
    user: CurrentUser | None = Depends(get_optional_user),
) -> dict:
    """Analyze a client-extracted pose JSON (no server-side MediaPipe).

    The browser ran MediaPipe on the recorded/uploaded clip and posts the resulting pose JSON
    alongside the raw video (still stored for replay/overlay). Routing/persistence mirror
    ``/api/analyze`` so uploads and library clips still render identically.
    """
    suffix = Path(file.filename or "").suffix.lower() or ".mp4"
    allowed = await run_in_threadpool(settings.allowed_upload_suffixes)
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'.")

    try:
        payload = json.loads(pose)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Malformed pose JSON.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise HTTPException(status_code=400, detail="Pose JSON must have a 'frames' list.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    video_id, _saved_path = await run_in_threadpool(analysis.save_upload, data, suffix=suffix)
    del data
    try:
        async with _ANALYSIS_SEMAPHORE:
            result = await run_in_threadpool(
                analysis.analyze_pose_payload, payload, movement=movement, video_id=video_id
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if user is not None:
        try:
            result["analysis_id"] = await run_in_threadpool(
                store.persist_analysis,
                token=user.token,
                user_id=user.id,
                video_id=video_id,
                source="upload",
                result=result,
                filename=file.filename,
            )
        except Exception:  # noqa: BLE001 — never lose a completed analysis to a storage error
            logger.exception("Failed to persist pose analysis (user=%s video=%s)", user.id, video_id)
            result["analysis_id"] = None
    return result
