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


def _validated_movement(movement: str) -> str:
    """Resolve a requested movement to its canonical name, or 400.

    Rejecting HERE -- before save_upload and before pose extraction -- means a bad request
    costs no compute. The registry lookup is case-insensitive (get_detector lowercases its
    key), so the canonical spelling is what comes back.

    An explicit empty/whitespace-only ``movement`` is rejected here rather than left to
    ``registry.get_detector``: that function's ``(movement or "Squat")`` fallback exists for
    callers that legitimately pass ``None`` (the library path), not for a client that sent
    ``movement=""`` on purpose. Silently mapping an explicit empty string to Squat would mask
    exactly the kind of bad request this endpoint is meant to catch.
    """
    from src.pose.movements import registry

    if not movement or not movement.strip():
        known = ", ".join(d.name for d in registry.list_detectors())
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported movement '{movement}'. Analyzable movements: {known}.",
        )

    try:
        return registry.get_detector(movement).name
    except KeyError:
        known = ", ".join(d.name for d in registry.list_detectors())
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported movement '{movement}'. Analyzable movements: {known}.",
        ) from None


def _validated_max_reps(max_reps: int | None) -> int | None:
    """Resolve an optional client-supplied rep cap. 0 means 'every rep'.

    Bounded so a client cannot ask for an unbounded amount of per-rep work on the shared
    analysis semaphore.
    """
    if max_reps is None:
        return config.DEFAULT_MAX_REPS
    if max_reps < 0 or max_reps > 20:
        raise HTTPException(status_code=400, detail="max_reps must be between 0 and 20.")
    return None if max_reps == 0 else max_reps


def _validate_pose_landmarks(payload: dict) -> None:
    """Reject structurally malformed landmark entries before they reach the pose pipeline.

    Landmark COUNT is not validated here — a ``landmarks`` list with fewer than 33
    well-formed entries is legitimate (the detector treats it as "no pose"). Only
    structural malformation (non-dict frames/landmarks, missing/non-numeric fields)
    is rejected. ``world_landmarks`` is intentionally not checked: the detector never
    reads it.
    """
    for frame in payload.get("frames", []):
        if not isinstance(frame, dict):
            raise HTTPException(status_code=400, detail="Malformed pose frame.")
        lms = frame.get("landmarks")
        if lms is None:
            continue
        if not isinstance(lms, list):
            raise HTTPException(status_code=400, detail="Malformed pose landmarks.")
        for lm in lms:
            if not isinstance(lm, dict) or not all(
                isinstance(lm.get(k), (int, float)) for k in ("x", "y", "z", "visibility")
            ):
                raise HTTPException(status_code=400, detail="Malformed pose landmarks.")


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    movement: str = Form(config.DEFAULT_ANALYSIS_MOVEMENT),
    max_reps: int | None = Form(None),
    user: CurrentUser | None = Depends(get_optional_user),
) -> dict:
    """Accept a video of a supported movement, extract pose, detect faults, and return the analysis.

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

    canonical_movement = await run_in_threadpool(_validated_movement, movement)
    resolved_max_reps = _validated_max_reps(max_reps)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    video_id, saved_path = await run_in_threadpool(analysis.save_upload, data, suffix=suffix)
    del data  # bytes are now on disk; don't pin the whole video in RAM while queued for a slot.
    try:
        async with _ANALYSIS_SEMAPHORE:
            result = await run_in_threadpool(
                analysis.analyze_video_file,
                saved_path,
                video_id=video_id,
                movement=canonical_movement,
                max_reps=resolved_max_reps,
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
    max_reps: int | None = Form(None),
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

    resolved_max_reps = _validated_max_reps(max_reps)

    try:
        payload = json.loads(pose)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Malformed pose JSON.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise HTTPException(status_code=400, detail="Pose JSON must have a 'frames' list.")
    _validate_pose_landmarks(payload)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    video_id, _saved_path = await run_in_threadpool(analysis.save_upload, data, suffix=suffix)
    del data
    try:
        async with _ANALYSIS_SEMAPHORE:
            result = await run_in_threadpool(
                analysis.analyze_pose_payload,
                payload,
                movement=movement,
                video_id=video_id,
                max_reps=resolved_max_reps,
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
