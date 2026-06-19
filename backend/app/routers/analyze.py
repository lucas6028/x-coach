"""Live-upload analysis endpoint."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.app import config
from backend.app.services import analysis

router = APIRouter(prefix="/api", tags=["analyze"])

_ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Bound how many uploads run the (blocking, CPU/RAM-heavy) pipeline at once. Excess requests
# await a slot here instead of spawning unbounded worker threads. This is a single-process
# stop-gap; durable queueing and backpressure move to Celery + Redis later.
_ANALYSIS_SEMAPHORE = asyncio.Semaphore(config.MAX_CONCURRENT_ANALYSES)


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    """Accept a squat video, extract pose, detect faults, and return the full analysis.

    The pose pipeline is synchronous and CPU-bound, so the disk write and the analysis run in a
    worker thread via ``run_in_threadpool`` — this keeps the event loop responsive instead of
    freezing every other request for the duration of one analysis. A concurrency semaphore caps
    how many analyses run at once so concurrent uploads queue rather than exhaust the machine.

    The response contract is unchanged: it matches the library analysis shape (metadata/view/
    quality/detections/retrievals/pose) so the frontend renders uploads and library clips
    identically.
    """
    suffix = Path(file.filename or "").suffix.lower() or ".mp4"
    if suffix not in _ALLOWED_SUFFIXES:
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
    return result
